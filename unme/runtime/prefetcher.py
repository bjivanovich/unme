"""MemoryPrefetcher: async NVMe -> RAM -> VRAM memory pipeline for UNME Fase 3E.

Responsibility
==============
Before each LLM forward pass reaches layer L_inj, the prefetcher must have
loaded the top-K relevant canonical memory vectors into the MemoryHookManager
VRAM buffer.

If retrieval + transfer does not complete in time (or is not triggered),
the buffer remains empty and the hook returns the hidden state unchanged.
This is the zero-stall guarantee: memory injection NEVER blocks token generation.

Architecture
============
                                                    LLM forward pass
  ┌─────────────┐  query_text   ┌──────────────┐       layer 0
  │   Caller    │ ─────────────>│  Prefetcher  │          ...
  │  (e.g. LLM  │               │  (background │       layer L_inj  <── hook fires
  │   wrapper)  │               │   thread)    │          ...
  └─────────────┘               └──────┬───────┘       layer N-1
                                       │
                        ┌──────────────┼───────────────┐
                        │              │               │
                   [1] GAR query  [2] NVMe read   [3] tensor to device
                   (CPU, RAM)     (RAM buffer)    (RAM or VRAM)
                        │              │               │
                        └──────────────┴───────────────┘
                                       │
                        MemoryHookManager.set_memory_buffer(tensor)

Thread model
============
- One background daemon thread per Prefetcher instance.
- Thread sleeps on a threading.Event, waking when trigger() is called.
- Each trigger() call cancels any in-flight work and starts fresh.
- PrefetchResult is stored atomically via threading.Lock.
- Zero-stall: hook reads buffer without waiting; if prefetch is not done,
  the previous buffer (or empty buffer) is used.

VRAM budget
===========
top_k=5, d_c=768, float16: 5 * 768 * 2 = 7,680 bytes ≈ 7.5 KB.
Negligible relative to the 256 MB VRAM ceiling for UNME.

Usage
=====
    prefetcher = MemoryPrefetcher(
        retriever=gar,            # GlobalAssociativeRetrieval instance
        encoder=enc,              # ICanonicalEncoder instance
        hook_manager=manager,     # MemoryHookManager instance
        namespace_id=ns_id,
        top_k=5,
        device="cuda",
    )
    prefetcher.start()

    # Before each forward pass — non-blocking:
    prefetcher.trigger(query_text="What is the Kalman filter gain?")

    # LLM runs; hook fires at L_inj; buffer may or may not be ready.
    output = model.generate(...)

    # Optionally wait for prefetch to complete (diagnostic / test use only):
    result = prefetcher.wait(timeout=0.5)

    prefetcher.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PrefetchResult
# ---------------------------------------------------------------------------

@dataclass
class PrefetchResult:
    """Outcome of a single prefetch operation.

    Attributes
    ----------
    success         : True if at least one memory was retrieved and loaded.
    n_retrieved     : Number of memory vectors placed in the hook buffer.
    best_similarity : Cosine similarity of the top-1 retrieved memory.
    latency_ms      : Wall-clock time from trigger() to buffer load, ms.
    query_text      : The query string that triggered this prefetch.
    error           : Exception message if the prefetch failed.
    """

    success: bool = False
    n_retrieved: int = 0
    best_similarity: float = 0.0
    latency_ms: float = 0.0
    query_text: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# MemoryPrefetcher
# ---------------------------------------------------------------------------

class MemoryPrefetcher:
    """Async background prefetcher: NVMe/RAM -> VRAM memory buffer.

    Parameters
    ----------
    retriever       : GlobalAssociativeRetrieval instance (or any object with
                      a retrieve() method returning list[GARResult]).
    encoder         : ICanonicalEncoder instance for query encoding.
    hook_manager    : MemoryHookManager whose buffer this prefetcher fills.
    namespace_id    : Current namespace identifier for local/global retrieval.
    top_k           : Maximum number of memory vectors to prefetch (default 5).
    device          : Target device for the loaded tensor ('cpu' or 'cuda').
                      Use 'cpu' when VRAM is fully occupied by the LLM.
    similarity_threshold : Minimum similarity for a retrieved memory to be
                           loaded into the buffer.  Below this, the buffer
                           is cleared (zero-stall, no injection).
    """

    def __init__(
        self,
        retriever: Any,                 # GlobalAssociativeRetrieval
        encoder: Any,                   # ICanonicalEncoder
        hook_manager: Any,              # MemoryHookManager
        namespace_id: int,
        top_k: int = 5,
        device: str = "cpu",
        similarity_threshold: float = 0.74,
    ) -> None:
        self._retriever = retriever
        self._encoder = encoder
        self._hook_manager = hook_manager
        self._namespace_id = namespace_id
        self._top_k = top_k
        self._device = torch.device(device)
        self._threshold = similarity_threshold

        # Threading primitives
        self._trigger_event   = threading.Event()
        self._stop_event      = threading.Event()
        self._result_lock     = threading.Lock()
        self._pending_query: Optional[str] = None
        self._last_result: Optional[PrefetchResult] = None
        self._done_event      = threading.Event()   # set when a prefetch finishes

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Starts the background prefetch thread.  Idempotent."""
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="unme-prefetcher",
            daemon=True,          # dies automatically when main thread exits
        )
        self._thread.start()
        self._running = True

    def stop(self, timeout: float = 2.0) -> None:
        """Signals the worker thread to stop and waits for it to exit."""
        if not self._running:
            return
        self._stop_event.set()
        self._trigger_event.set()   # unblock a waiting worker
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._running = False

    def __enter__(self) -> "MemoryPrefetcher":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Trigger API (called from main thread, non-blocking)
    # ------------------------------------------------------------------

    def trigger(self, query_text: str) -> None:
        """Schedules a prefetch for query_text.  Returns immediately.

        If a prefetch is already in-flight, the new query supersedes it.
        The hook buffer is NOT cleared here — the previous buffer remains
        available until the new prefetch completes (or the threshold fails).
        """
        with self._result_lock:
            self._pending_query = query_text
        self._done_event.clear()
        self._trigger_event.set()

    def trigger_and_wait(
        self, query_text: str, timeout: float = 1.0
    ) -> PrefetchResult:
        """Triggers a prefetch and blocks until it completes or times out.

        Primarily for testing and diagnostic use.  Do NOT call from within
        a LLM forward pass (violates the zero-stall contract).
        """
        self.trigger(query_text)
        self._done_event.wait(timeout=timeout)
        return self.last_result or PrefetchResult(
            success=False, query_text=query_text, error="timeout"
        )

    def clear(self) -> None:
        """Immediately clears the hook buffer (zero-stall for next pass)."""
        self._hook_manager.clear_memory_buffer()

    # ------------------------------------------------------------------
    # Result access (thread-safe)
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> Optional[PrefetchResult]:
        """The result of the most recently completed prefetch, or None."""
        with self._result_lock:
            return self._last_result

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Main loop of the background prefetch thread."""
        while not self._stop_event.is_set():
            # Sleep until triggered or stopped
            self._trigger_event.wait()
            self._trigger_event.clear()

            if self._stop_event.is_set():
                break

            # Drain the pending query (take the latest one)
            with self._result_lock:
                query_text = self._pending_query
                self._pending_query = None

            if query_text is None:
                continue

            result = self._do_prefetch(query_text)

            with self._result_lock:
                self._last_result = result
            self._done_event.set()

    def _do_prefetch(self, query_text: str) -> PrefetchResult:
        """Performs retrieval + RAM-to-device transfer.  Runs in worker thread."""
        t0 = time.perf_counter()
        result = PrefetchResult(query_text=query_text)

        try:
            # Step 1: Encode query to canonical vector (CPU, RAM)
            q_vec = self._encoder.encode_query(query_text)   # float16, (d_c,)

            # Step 2: Retrieve top-K from GAR (CPU, RAM)
            gar_results = self._retriever.retrieve(
                current_namespace_id=self._namespace_id,
                query_vector=q_vec,
                top_k_local=self._top_k,
                top_k_global=self._top_k,
                threshold_local=self._threshold,
                threshold_global=self._threshold + 0.06,   # stricter cross-namespace
            )

            if not gar_results:
                # No memories above threshold — clear buffer (zero-stall)
                self._hook_manager.clear_memory_buffer()
                result.success = False
                result.n_retrieved = 0
                result.latency_ms = (time.perf_counter() - t0) * 1000
                return result

            # Step 3: Collect value vectors from results
            # GARResult has: canonical_value (np.ndarray float16, d_c)
            # or vector (raw float16) depending on retrieval mode.
            vectors = []
            best_sim = 0.0
            for r in gar_results[: self._top_k]:
                vec = self._extract_vector(r)
                if vec is not None:
                    vectors.append(vec)
                    sim = float(getattr(r, "final_score", 0.0))
                    if sim > best_sim:
                        best_sim = sim

            if not vectors:
                self._hook_manager.clear_memory_buffer()
                result.success = False
                result.latency_ms = (time.perf_counter() - t0) * 1000
                return result

            # Step 4: Stack into tensor and move to target device (RAM or VRAM)
            V = np.stack(vectors, axis=0).astype(np.float16)   # (K, d_c)
            tensor = torch.from_numpy(V).to(dtype=torch.float16, device=self._device)

            # Step 5: Load into hook buffer (thread-safe: GIL protects assignment)
            self._hook_manager.set_memory_buffer(tensor, best_similarity=best_sim)

            result.success = True
            result.n_retrieved = len(vectors)
            result.best_similarity = best_sim
            result.latency_ms = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            result.error = str(exc)
            result.success = False
            # Safety: clear buffer so hook does not use stale data
            try:
                self._hook_manager.clear_memory_buffer()
            except Exception:
                pass

        return result

    def _extract_vector(self, gar_result: Any) -> Optional[np.ndarray]:
        """Extracts the canonical value vector from a GARResult object.

        Handles both canonical mode (canonical_value) and raw vector mode (vector).
        Returns float16 np.ndarray of shape (d_c,) or None if unavailable.
        """
        # Canonical mode: GARResult.canonical_value is an np.ndarray
        v = getattr(gar_result, "canonical_value", None)
        if v is not None and isinstance(v, np.ndarray):
            return v.astype(np.float16)

        # Raw vector mode: GARResult.vector
        v = getattr(gar_result, "vector", None)
        if v is not None and isinstance(v, np.ndarray):
            return v.astype(np.float16)

        return None
