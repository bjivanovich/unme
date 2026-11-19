# Universal Neural Memory Engine (UNME)

Universal Neural Memory Engine (UNME) is a persistent, associative, and trainable neural memory infrastructure decoupled from any specific LLM, model architecture, or runtime.

UNME treats language models as interchangeable consumers of memory. Models and runtimes (PyTorch, llama.cpp, vLLM) may change, quantize, or upgrade, but the underlying persistent associative memory survives indefinitely on NVMe with strict cryptographic isolation and zero-stall retrieval guarantees.

---

## Key Architectural Principles

1. **Decoupled Memory Lifecycle:**
   - Raw memory records reside on NVMe storage in binary Universal Memory Record (UMR V2) format.
   - Semantic indices reside in a Canonical Memory Representation (CMR) space of 768 dimensions (`intfloat/multilingual-e5-base`), normalized in L2.
   - Base LLM weights remain completely frozen; memory injection is handled by a lightweight neural adapter (< 15 MB).

2. **Hardware Budget & Zero-Stall Guarantee:**
   - **VRAM ceiling during inference:** Strictly < 256 MB (12.58 MB for Qwen/Gyro-Coder adapter).
   - **Zero-Stall:** Memory retrieval never blocks token generation. If prefetching is incomplete when the forward pass reaches the injection layer, the learned gate remains closed ($\alpha = 0$) and generation proceeds without latency penalty.

3. **Cryptographic Multi-Project Isolation:**
   - Per-namespace encryption at rest using ChaCha20-Poly1305.
   - HKDF-V3 nonce generation deriving 96-bit nonces from a 128-bit epoch token and record counter, eliminating nonce-reuse risks upon backup restore.

4. **Global Associative Retrieval (GAR):**
   - Cross-project associative search without namespace silos.
   - Composite scoring balancing semantic similarity, local preference, recency decay, and access frequency.

---

## Repository Structure

```
d:/memory/
  unme/
    core/             # NVMe storage engine (append-only UMR, CRC32, mmap, Direct I/O)
      constants.py
      header.py       # 64-byte binary header definition
      record.py       # UMR record serialization/deserialization
      storage.py      # SegmentWriter, SegmentReader, MultiSegmentReader
      cml.py          # Contextual Memory Link (CML) storage and graph records
      lifecycle.py    # MemoryLifecycleManager (tombstones, independent derivations)
    crypto/           # Cryptographic isolation layer
      namespace.py    # NamespaceManager, NonceEpochManager (HKDF-V3)
    index/            # Vector indexing and cross-namespace retrieval
      vector_index.py # PartitionedVectorIndex
      gar.py          # GlobalAssociativeRetrieval (canonical + raw modes)
    canonical/        # Canonical Memory Representation (CMR)
      encoder.py      # ICanonicalEncoder, MultilingualE5BaseEncoder (768-dim)
      store.py        # NVMeCanonicalStore, CanonicalMemoryRecord, CanonicalIndexPartition
    adapter/          # Architecture-specific neural injection adapter
      projector.py    # CanonicalProjector (W_q, W_o matrices)
      cross_attention.py # MemoryCrossAttention (stateless single-head)
      gating.py       # MemoryGate (scalar/vector sigmoid gate, init alpha ~ 0.018)
      adapter.py      # IModelAdapter, QwenAdapter, make_adapter factory
    calibration/      # Forward and gradient activation capture (.umrc files)
      record.py       # CalibrationRecord, GradientCalibrationRecord (64-byte UMRC header)
      store.py        # CalibrationStore (append-only NVMe writer/scanner)
      capture.py      # ForwardCaptureSession, GradientCaptureSession
    training/         # Offline adapter training
      surrogate_trainer.py # SurrogateTrainer (trains W_q, W_o, alpha without LLM in VRAM)
    runtime/          # Real-time memory injection
      hook_manager.py # MemoryHookManager (PyTorch forward hook, zero-stall)
      prefetcher.py   # MemoryPrefetcher (background async NVMe -> RAM -> VRAM)
  tests/              # Comprehensive test suites (125 tests, 100% pass)
    test_storage_robustness.py
    test_isolation_and_indexing.py
    test_storage_aead_recovery.py
    test_gar_and_cml_lifecycle.py
    test_durability_and_rotation.py
    test_multiprocess_concurrency.py
    test_canonical_retrieval.py
    test_adapter.py
    test_calibration.py
    test_surrogate_trainer.py
    test_prefetcher.py
  benchmarks/         # NVMe cold vs warm latency and cryptographic overhead benchmarks
  scratch/            # Helper scripts and validation utilities
```

---

## Implemented Phases

| Phase | Component | Status | Verified Features |
|---|---|:---:|---|
| **Phase 1** | Storage Core | Done | 64-byte UMR header, append-only NVMe, mmap & Direct I/O, dual CRC32 |
| **Phase 2** | Cryptographic Isolation | Done | ChaCha20-Poly1305, HKDF-V3 128-bit epoch tokens, CML, GAR |
| **Phase 3A** | Canonical Memory (CMR) | Done | 768-dim `multilingual-e5-base`, NVMeCanonicalStore, calibrated GAR |
| **Phase 3B** | Neural Adapter | Done | QwenAdapter (12.58 MB), W_q, W_o, gate, MemoryHookManager |
| **Phase 3C** | Activation Calibration | Done | UMRC 64-byte format, ForwardCaptureSession, GradientCaptureSession |
| **Phase 3D** | Surrogate Trainer | Done | Offline training with LLM out of VRAM, loss = -<g, dh> + lambda ||dh||^2 |
| **Phase 3E** | Async Runtime Prefetcher | Done | Daemon prefetch thread, zero-stall memory buffer, automatic clear |

---

## Running Test Suites

All tests can be executed via the unified test runner:

```bash
python scratch/run_tests.py
```

Or individually via pytest:

```bash
pytest tests/test_canonical_retrieval.py -v
pytest tests/test_adapter.py -v
pytest tests/test_calibration.py -v
pytest tests/test_surrogate_trainer.py -v
pytest tests/test_prefetcher.py -v
```

Current test status: **125 tests passing (0 failures).**
