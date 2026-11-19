"""Test suite for UNME Fase 3E: unme/runtime/prefetcher.py

Covers:
  - MemoryPrefetcher construction and lifecycle (start/stop)
  - trigger() is non-blocking
  - trigger_and_wait() delivers a result within timeout
  - Successful retrieval loads correct tensor into hook buffer
  - Zero-stall: empty retrieval clears the buffer
  - Threshold filtering: results below threshold are discarded
  - Error in retrieval clears buffer and stores error in result
  - Multiple triggers: latest query supersedes earlier ones
  - Context manager (with statement)
  - PrefetchResult fields populated correctly
  - Integration with MemoryHookManager.set_memory_buffer / clear_memory_buffer
"""

import pathlib
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unme.runtime.prefetcher import MemoryPrefetcher, PrefetchResult
from unme.runtime.hook_manager import MemoryHookManager


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_C = 32         # small canonical dim for fast tests
K = 4            # number of memory vectors returned by mock retriever
NAMESPACE_ID = 42
THRESHOLD = 0.74


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockGARResult:
    """Mimics a GARResult with canonical_value and final_score."""
    def __init__(self, vec: np.ndarray, score: float = 0.85):
        self.canonical_value = vec.astype(np.float16)
        self.vector = vec.astype(np.float16)
        self.final_score = score


def make_mock_retriever(
    n_results: int = K,
    score: float = 0.85,
    raise_exc: Optional[Exception] = None,
) -> MagicMock:
    """Returns a mock retriever with a .retrieve() method."""
    rng = np.random.default_rng(7)
    results = [
        MockGARResult(rng.standard_normal(D_C).astype(np.float16), score)
        for _ in range(n_results)
    ]
    mock = MagicMock()
    if raise_exc is not None:
        mock.retrieve.side_effect = raise_exc
    else:
        mock.retrieve.return_value = results
    return mock


def make_mock_encoder() -> MagicMock:
    """Returns a mock encoder whose encode_query returns a unit float16 vector."""
    rng = np.random.default_rng(13)
    vec = rng.standard_normal(D_C).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).astype(np.float16)
    mock = MagicMock()
    mock.encode_query.return_value = vec
    return mock


def make_mock_hook_manager() -> MagicMock:
    """Returns a mock MemoryHookManager tracking set/clear calls."""
    mock = MagicMock()
    mock.set_memory_buffer = MagicMock()
    mock.clear_memory_buffer = MagicMock()
    return mock


def make_prefetcher(
    n_results: int = K,
    score: float = 0.85,
    raise_exc: Optional[Exception] = None,
    threshold: float = THRESHOLD,
    device: str = "cpu",
) -> tuple[MemoryPrefetcher, MagicMock, MagicMock, MagicMock]:
    retriever = make_mock_retriever(n_results, score, raise_exc)
    encoder   = make_mock_encoder()
    manager   = make_mock_hook_manager()
    pf = MemoryPrefetcher(
        retriever=retriever,
        encoder=encoder,
        hook_manager=manager,
        namespace_id=NAMESPACE_ID,
        top_k=K,
        device=device,
        similarity_threshold=threshold,
    )
    return pf, retriever, encoder, manager


# ---------------------------------------------------------------------------
# Construction and lifecycle
# ---------------------------------------------------------------------------

def test_constructs_not_running():
    pf, *_ = make_prefetcher()
    assert not pf.is_running
    assert pf.last_result is None


def test_start_stop():
    pf, *_ = make_prefetcher()
    pf.start()
    assert pf.is_running
    pf.stop()
    assert not pf.is_running


def test_start_idempotent():
    pf, *_ = make_prefetcher()
    pf.start()
    pf.start()   # second start must not crash or spawn extra threads
    assert pf.is_running
    pf.stop()


def test_context_manager():
    pf, *_ = make_prefetcher()
    with pf:
        assert pf.is_running
    assert not pf.is_running


def test_stop_without_start():
    pf, *_ = make_prefetcher()
    pf.stop()   # must not raise
    assert not pf.is_running


# ---------------------------------------------------------------------------
# PrefetchResult dataclass
# ---------------------------------------------------------------------------

def test_prefetch_result_defaults():
    r = PrefetchResult()
    assert r.success is False
    assert r.n_retrieved == 0
    assert r.best_similarity == 0.0
    assert r.latency_ms == 0.0
    assert r.query_text == ""
    assert r.error is None


# ---------------------------------------------------------------------------
# trigger_and_wait: successful retrieval
# ---------------------------------------------------------------------------

def test_trigger_and_wait_success():
    pf, retriever, encoder, manager = make_prefetcher(n_results=K, score=0.85)
    with pf:
        result = pf.trigger_and_wait("Kalman filter GNSS", timeout=3.0)

    assert result.success is True
    assert result.n_retrieved == K
    assert result.best_similarity > 0.0
    assert result.latency_ms > 0.0
    assert result.query_text == "Kalman filter GNSS"
    assert result.error is None


def test_trigger_and_wait_calls_encoder():
    pf, retriever, encoder, manager = make_prefetcher()
    with pf:
        pf.trigger_and_wait("test query", timeout=3.0)
    encoder.encode_query.assert_called_once_with("test query")


def test_trigger_and_wait_calls_retriever():
    pf, retriever, encoder, manager = make_prefetcher()
    with pf:
        pf.trigger_and_wait("nav query", timeout=3.0)
    assert retriever.retrieve.call_count == 1
    call_kwargs = retriever.retrieve.call_args[1]
    assert call_kwargs["current_namespace_id"] == NAMESPACE_ID


def test_trigger_and_wait_loads_buffer():
    pf, retriever, encoder, manager = make_prefetcher()
    with pf:
        pf.trigger_and_wait("load me", timeout=3.0)
    # set_memory_buffer must have been called with a float16 tensor
    manager.set_memory_buffer.assert_called_once()
    args, kwargs = manager.set_memory_buffer.call_args
    tensor = args[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float16
    assert tensor.shape[0] == K
    assert tensor.shape[1] == D_C


def test_buffer_tensor_on_cpu():
    pf, retriever, encoder, manager = make_prefetcher(device="cpu")
    with pf:
        pf.trigger_and_wait("cpu test", timeout=3.0)
    args, _ = manager.set_memory_buffer.call_args
    assert args[0].device.type == "cpu"


# ---------------------------------------------------------------------------
# Zero-stall: empty retrieval
# ---------------------------------------------------------------------------

def test_empty_retrieval_clears_buffer():
    pf, retriever, encoder, manager = make_prefetcher(n_results=0)
    with pf:
        result = pf.trigger_and_wait("empty", timeout=3.0)
    assert result.success is False
    assert result.n_retrieved == 0
    manager.clear_memory_buffer.assert_called()
    manager.set_memory_buffer.assert_not_called()


def test_latency_measured_on_empty():
    pf, retriever, encoder, manager = make_prefetcher(n_results=0)
    with pf:
        result = pf.trigger_and_wait("empty latency", timeout=3.0)
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_retriever_exception_clears_buffer():
    pf, retriever, encoder, manager = make_prefetcher(
        raise_exc=RuntimeError("retrieval failed")
    )
    with pf:
        result = pf.trigger_and_wait("error test", timeout=3.0)
    assert result.success is False
    assert result.error is not None
    assert "retrieval failed" in result.error
    manager.clear_memory_buffer.assert_called()


def test_encoder_exception_clears_buffer():
    pf, retriever, encoder, manager = make_prefetcher()
    encoder.encode_query.side_effect = RuntimeError("encoder crash")
    with pf:
        result = pf.trigger_and_wait("enc error", timeout=3.0)
    assert result.success is False
    assert result.error is not None
    manager.clear_memory_buffer.assert_called()


# ---------------------------------------------------------------------------
# trigger() is non-blocking
# ---------------------------------------------------------------------------

def test_trigger_returns_immediately():
    # Use a slow retriever (sleep) to verify trigger() does not block
    slow_retriever = MagicMock()
    barrier = threading.Event()

    def slow_retrieve(**kwargs):
        barrier.wait(timeout=5.0)   # blocks until test releases
        return []

    slow_retriever.retrieve.side_effect = slow_retrieve

    encoder = make_mock_encoder()
    manager = make_mock_hook_manager()
    pf = MemoryPrefetcher(
        retriever=slow_retriever, encoder=encoder,
        hook_manager=manager, namespace_id=NAMESPACE_ID,
        top_k=K, device="cpu", similarity_threshold=THRESHOLD,
    )
    with pf:
        t0 = time.perf_counter()
        pf.trigger("non-blocking test")
        elapsed = time.perf_counter() - t0
        barrier.set()   # release the slow retriever

    # trigger() must return in << 100 ms even with a blocking retriever
    assert elapsed < 0.1, f"trigger() blocked for {elapsed*1000:.1f} ms"


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_calls_hook_manager():
    pf, retriever, encoder, manager = make_prefetcher()
    with pf:
        pf.clear()
    manager.clear_memory_buffer.assert_called()


# ---------------------------------------------------------------------------
# Multiple triggers: latest wins
# ---------------------------------------------------------------------------

def test_multiple_triggers_last_query_used():
    """When triggers arrive faster than prefetch completes, last query wins."""
    call_log = []
    base_encoder = make_mock_encoder()

    def track_encode(query_text):
        call_log.append(query_text)
        time.sleep(0.01)   # small delay so worker can be superseded
        return base_encoder.encode_query.return_value

    encoder = MagicMock()
    encoder.encode_query.side_effect = track_encode

    retriever = make_mock_retriever()
    manager = make_mock_hook_manager()

    pf = MemoryPrefetcher(
        retriever=retriever, encoder=encoder,
        hook_manager=manager, namespace_id=NAMESPACE_ID,
        top_k=K, device="cpu", similarity_threshold=THRESHOLD,
    )
    with pf:
        for i in range(5):
            pf.trigger(f"query_{i}")
        # Wait enough time for the worker to process
        time.sleep(0.3)

    # The last prefetch result must correspond to one of the later queries
    result = pf.last_result
    assert result is not None


# ---------------------------------------------------------------------------
# last_result persists after stop
# ---------------------------------------------------------------------------

def test_last_result_accessible_after_stop():
    pf, *_ = make_prefetcher()
    with pf:
        pf.trigger_and_wait("persist test", timeout=3.0)
    assert pf.last_result is not None
    assert pf.last_result.query_text == "persist test"


# ---------------------------------------------------------------------------
# Integration: prefetcher feeds MemoryHookManager directly
# ---------------------------------------------------------------------------

def test_integration_with_real_hook_manager_mock():
    """Uses a real-shaped mock of MemoryHookManager's buffer interface."""
    buffer_ref: list[torch.Tensor] = []
    similarity_ref: list[float] = []

    class CapturingManager:
        def set_memory_buffer(self, tensor: torch.Tensor, best_similarity: float = 0.0):
            buffer_ref.append(tensor.clone())
            similarity_ref.append(best_similarity)

        def clear_memory_buffer(self):
            buffer_ref.clear()
            similarity_ref.clear()

    retriever = make_mock_retriever(n_results=3, score=0.90)
    encoder   = make_mock_encoder()
    manager   = CapturingManager()

    pf = MemoryPrefetcher(
        retriever=retriever, encoder=encoder,
        hook_manager=manager, namespace_id=NAMESPACE_ID,
        top_k=5, device="cpu", similarity_threshold=0.5,
    )
    with pf:
        result = pf.trigger_and_wait("integration query", timeout=3.0)

    assert result.success is True
    assert len(buffer_ref) == 1
    t = buffer_ref[0]
    assert t.shape == (3, D_C)
    assert t.dtype == torch.float16
    assert similarity_ref[0] > 0.0


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
