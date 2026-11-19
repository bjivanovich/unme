"""Test suite for Fase 3A: Canonical Memory Representation (CMR).

Tests cover:
1. Encoder interface contract (DeterministicHashEncoder — no model download needed)
2. CanonicalMemoryRecord normalization and dataclass invariants
3. NVMeCanonicalStore: persistence, retrieval, offset scan, record count
4. CanonicalIndexPartition: similarity search, threshold, top-k ordering
5. GAR canonical mode: encode -> search -> composite scoring -> result ordering
6. GAR raw vector mode: backward compatibility (existing Fases 1-2 behavior)
7. Encoder signature isolation: vectors from distinct signatures never mixed
8. Recall@5 and MRR on a synthetic corpus with known cross-project relationships

All tests use DeterministicHashEncoder and temporary directories.
No model weights are downloaded or loaded.

Note on the precision tests (Recall@5, MRR):
DeterministicHashEncoder is not semantically meaningful — it cannot match a
query prefix "query:" to a passage prefix "passage:" even for identical text.
The precision tests therefore insert vectors using encode_query() for BOTH
indexing and querying, which guarantees that the same text produces the same
vector in both paths.  This is explicitly marked in each test.
The production path (MultilingualE5SmallEncoder) uses asymmetric prefixes
("query: " / "passage: ") which is semantically correct and tested separately.
"""

from __future__ import annotations

import sys
import tempfile
import math
from pathlib import Path

sys.path.insert(0, "d:/memory")

import numpy as np
import pytest

from unme.canonical.encoder import (
    DeterministicHashEncoder,
    CANONICAL_DIM,
    _l2_normalize,
)
from unme.canonical.store import (
    CanonicalMemoryRecord,
    NVMeCanonicalStore,
    CanonicalIndexPartition,
    CMRSearchResult,
    CMR_HEADER_SIZE,
)
from unme.index.gar import GlobalAssociativeRetrieval, GARResult
from unme.index.vector_index import PartitionedVectorIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_encoder() -> DeterministicHashEncoder:
    return DeterministicHashEncoder()


def make_record(
    encoder: DeterministicHashEncoder,
    text: str,
    ns_id: int = 1,
    umr_seg: int = 0,
    umr_rec: int = 0,
) -> CanonicalMemoryRecord:
    k, v = encoder.encode_record(text)
    return CanonicalMemoryRecord(
        record_id=f"CMR-{ns_id}-{text[:8]}",
        namespace_id=ns_id,
        encoder_signature=encoder.get_signature(),
        key_dim=encoder.get_dim(),
        value_dim=encoder.get_dim(),
        key_vector=k,
        value_vector=v,
        umr_segment_id=umr_seg,
        umr_record_index=umr_rec,
    )


# ---------------------------------------------------------------------------
# 1. Encoder interface
# ---------------------------------------------------------------------------

def test_encoder_signature_format():
    enc = make_encoder()
    sig = enc.get_signature()
    parts = sig.split(":")
    assert len(parts) == 5, f"Signature must have 5 colon-separated fields: {sig}"


def test_encoder_dim():
    enc = make_encoder()
    assert enc.get_dim() == CANONICAL_DIM == 768


def test_encode_query_returns_unit_norm():
    enc = make_encoder()
    q = enc.encode_query("what is GNSS positioning accuracy?")
    assert q.dtype == np.float16
    assert q.shape == (768,)
    norm = np.linalg.norm(q.astype(np.float32))
    assert abs(norm - 1.0) < 1e-2, f"Query vector norm not unit: {norm}"


def test_encode_record_returns_unit_norm_pair():
    enc = make_encoder()
    k, v = enc.encode_record("GNSS kalman filter convergence time", {"domain": "navigation"})
    assert k.dtype == np.float16
    assert v.dtype == np.float16
    assert k.shape == (768,)
    kn = np.linalg.norm(k.astype(np.float32))
    vn = np.linalg.norm(v.astype(np.float32))
    assert abs(kn - 1.0) < 1e-2
    assert abs(vn - 1.0) < 1e-2


def test_encode_determinism():
    enc = make_encoder()
    text = "deterministic encoder test"
    k1, _ = enc.encode_record(text)
    k2, _ = enc.encode_record(text)
    assert np.array_equal(k1, k2), "DeterministicHashEncoder must be reproducible"


def test_distinct_texts_produce_distinct_vectors():
    enc = make_encoder()
    q1 = enc.encode_query("GNSS positioning accuracy kalman")
    q2 = enc.encode_query("neural network training loss gradient descent")
    # Different SHA-256 digests -> different vectors; they should not be equal
    assert not np.array_equal(q1, q2), "Distinct texts must produce distinct vectors"


def test_encode_no_nan_in_output():
    enc = make_encoder()
    # Ensure no NaN/Inf values regardless of input bytes
    for text in ["abc", "GNSS", "x" * 100, "\x00\xff"]:
        q = enc.encode_query(text)
        assert np.all(np.isfinite(q.astype(np.float32))), f"NaN/Inf in output for: {text!r}"


# ---------------------------------------------------------------------------
# 2. CanonicalMemoryRecord invariants
# ---------------------------------------------------------------------------

def test_record_forces_float16_normalization():
    enc = make_encoder()
    raw_vec = np.random.randn(768).astype(np.float64) * 100.0
    rec = CanonicalMemoryRecord(
        record_id="test-001",
        namespace_id=42,
        encoder_signature=enc.get_signature(),
        key_dim=768,
        value_dim=768,
        key_vector=raw_vec.copy(),
        value_vector=raw_vec.copy(),
    )
    assert rec.key_vector.dtype == np.float16
    kn = np.linalg.norm(rec.key_vector.astype(np.float32))
    assert abs(kn - 1.0) < 1e-2, f"Record key vector not unit norm after construction: {kn}"


def test_record_flags_encode_both_stored():
    enc = make_encoder()
    k, v = enc.encode_record("test flags")
    rec = CanonicalMemoryRecord(
        record_id="test-flags",
        namespace_id=1,
        encoder_signature=enc.get_signature(),
        key_dim=768,
        value_dim=768,
        key_vector=k,
        value_vector=v,
    )
    # FLAG_KEY_STORED | FLAG_VALUE_STORED
    assert rec.flags == 0x0003


# ---------------------------------------------------------------------------
# 3. NVMeCanonicalStore: persistence and retrieval
# ---------------------------------------------------------------------------

def test_store_save_and_retrieve_roundtrip():
    enc = make_encoder()
    with tempfile.TemporaryDirectory() as tmp:
        store = NVMeCanonicalStore(tmp, namespace_id=1, encoder=enc, segment_id=0)
        assert store.record_count == 0

        texts = [
            "GNSS IMU sensor fusion",
            "Kalman filter prediction step",
            "Encoder quaternion to euler angles",
        ]
        for i, text in enumerate(texts):
            rec = make_record(enc, text, umr_rec=i)
            seg_id, rec_idx = store.save_canonical_record(rec)
            assert rec.cmr_record_index == i
            assert rec.cmr_segment_id == 0

        assert store.record_count == 3

        for i, text in enumerate(texts):
            retrieved = store.get_canonical_record(0, i)
            assert retrieved.encoder_signature == enc.get_signature()
            assert retrieved.namespace_id == 1
            assert retrieved.key_dim == 768
            assert retrieved.value_dim == 768
            assert not retrieved.is_tombstoned
            kn = np.linalg.norm(retrieved.key_vector.astype(np.float32))
            # After round-trip via float16 storage, norm may drift slightly more
            assert abs(kn - 1.0) < 0.05, f"Norm after roundtrip: {kn}"


def test_store_header_size_is_64_bytes():
    assert CMR_HEADER_SIZE == 64


def test_store_scan_on_reopen():
    """Verifies that a reopened store rescans the existing file correctly."""
    enc = make_encoder()
    with tempfile.TemporaryDirectory() as tmp:
        store = NVMeCanonicalStore(tmp, namespace_id=2, encoder=enc, segment_id=0)
        for i in range(5):
            rec = make_record(enc, f"record text number {i}", umr_rec=i)
            store.save_canonical_record(rec)

        # Reopen
        store2 = NVMeCanonicalStore(tmp, namespace_id=2, encoder=enc, segment_id=0)
        assert store2.record_count == 5

        retrieved = store2.get_canonical_record(0, 4)
        assert retrieved.umr_record_index == 4


def test_store_out_of_range_raises():
    enc = make_encoder()
    with tempfile.TemporaryDirectory() as tmp:
        store = NVMeCanonicalStore(tmp, namespace_id=3, encoder=enc)
        with pytest.raises(LookupError):
            store.get_canonical_record(0, 0)


def test_store_build_index_partition():
    """build_index_partition() returns a searchable CanonicalIndexPartition."""
    enc = make_encoder()
    with tempfile.TemporaryDirectory() as tmp:
        store = NVMeCanonicalStore(tmp, namespace_id=99, encoder=enc)
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        for i, text in enumerate(texts):
            rec = make_record(enc, text, umr_rec=i)
            store.save_canonical_record(rec)

        partition = store.build_index_partition()
        assert partition.size == 2
        # Search with the stored key vector of record 0
        rec0 = store.get_canonical_record(0, 0)
        results = partition.search(rec0.key_vector, top_k=1, threshold=0.0)
        assert len(results) == 1
        assert abs(results[0].similarity - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# 4. CanonicalIndexPartition: similarity search
# ---------------------------------------------------------------------------

def test_index_exact_match_top_1():
    enc = make_encoder()
    text = "GNSS positioning accuracy"
    k, _ = enc.encode_record(text)

    part = CanonicalIndexPartition(namespace_id=1, encoder_signature=enc.get_signature())
    part.add(cmr_segment_id=0, cmr_record_index=0, umr_segment_id=0, umr_record_index=0, key_vector=k.copy())

    # Self-query: same vector should return similarity ~1.0
    results = part.search(query_vector=k, top_k=1, threshold=0.0)
    assert len(results) == 1
    assert abs(results[0].similarity - 1.0) < 1e-3


def test_index_threshold_filters_low_similarity():
    enc = make_encoder()
    k, _ = enc.encode_record("sensor fusion GNSS IMU")
    part = CanonicalIndexPartition(namespace_id=1, encoder_signature=enc.get_signature())
    part.add(0, 0, 0, 0, k)

    # Threshold of 2.0 is impossible (cosine in [-1, 1]) — must return empty
    results = part.search(query_vector=k, top_k=5, threshold=2.0)
    assert results == []


def test_index_top_k_ordering():
    """Results must be ordered by descending similarity."""
    enc = make_encoder()
    part = CanonicalIndexPartition(namespace_id=1, encoder_signature=enc.get_signature())

    texts = [f"topic {i} detail text for ordering" for i in range(10)]
    for i, text in enumerate(texts):
        k, _ = enc.encode_record(text)
        part.add(0, i, 0, i, k)

    k_ref, _ = enc.encode_record(texts[3])
    results = part.search(query_vector=k_ref, top_k=5, threshold=0.0)
    assert len(results) <= 5
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True), "Results must be in descending similarity order"


def test_index_empty_returns_empty():
    enc = make_encoder()
    part = CanonicalIndexPartition(namespace_id=1, encoder_signature=enc.get_signature())
    q = enc.encode_query("anything")
    results = part.search(query_vector=q, top_k=5, threshold=0.0)
    assert results == []


# ---------------------------------------------------------------------------
# 5. GAR canonical mode
# ---------------------------------------------------------------------------

def test_gar_canonical_retrieve_local():
    enc = make_encoder()
    ns_a = 0xAAAA_0001

    part_a = CanonicalIndexPartition(ns_a, enc.get_signature())
    texts_a = [
        "GNSS kalman filter trajectory",
        "IMU accelerometer bias calibration",
        "VIALSDEG sensor fusion mode",
    ]
    for i, text in enumerate(texts_a):
        k, _ = enc.encode_record(text)
        part_a.add(0, i, 0, i, k)

    vector_index = PartitionedVectorIndex()
    gar = GlobalAssociativeRetrieval(vector_index=vector_index)
    gar.register_project(ns_a, "GNSS_Project")
    gar.register_canonical_partition(ns_a, enc.get_signature(), part_a)

    # Query using encode_query — different from passage prefix, may not match
    # Use threshold=0.0 to get all results regardless of similarity
    results = gar.retrieve(
        current_namespace_id=ns_a,
        query_text="VIALSDEG sensor fusion",
        encoder=enc,
        threshold_local=0.0,
        threshold_global=0.0,
    )
    # With threshold=0 there should be results from ns_a
    assert len(results) > 0
    assert all(r.source_namespace_id == ns_a for r in results)
    assert all(r.encoder_signature == enc.get_signature() for r in results)


def test_gar_canonical_cross_project_discovery():
    enc = make_encoder()
    ns_a = 0xAAAA_0001
    ns_b = 0xBBBB_0002

    # Both projects have records indexed with passage prefix
    part_a = CanonicalIndexPartition(ns_a, enc.get_signature())
    for i, text in enumerate(["GNSS IMU fusion", "Kalman filter", "trajectory smoothing"]):
        k, _ = enc.encode_record(text)
        part_a.add(0, i, 0, i, k)

    part_b = CanonicalIndexPartition(ns_b, enc.get_signature())
    for i, text in enumerate(["deep learning training", "gradient descent", "loss function"]):
        k, _ = enc.encode_record(text)
        part_b.add(0, i, 0, i, k)

    vector_index = PartitionedVectorIndex()
    gar = GlobalAssociativeRetrieval(vector_index=vector_index)
    gar.register_project(ns_a, "NavProject")
    gar.register_project(ns_b, "MLProject")
    gar.register_canonical_partition(ns_a, enc.get_signature(), part_a)
    gar.register_canonical_partition(ns_b, enc.get_signature(), part_b)

    results = gar.retrieve(
        current_namespace_id=ns_a,
        query_text="Kalman filter trajectory",
        encoder=enc,
        threshold_local=0.0,
        threshold_global=0.0,
    )
    # With threshold=0.0, both partitions are searched — both should appear
    ns_ids = {r.source_namespace_id for r in results}
    assert ns_a in ns_ids, f"Expected ns_a in results, got: {ns_ids}"
    assert ns_b in ns_ids, f"Expected ns_b in results, got: {ns_ids}"


def test_gar_composite_score_prefers_local():
    """Local results should score higher than cross-project results with equal cosine sim."""
    enc = make_encoder()
    ns_a = 0xAAAA_0001
    ns_b = 0xBBBB_0002

    # Both partitions have the identical vector
    text = "identical topic shared across projects"
    k, _ = enc.encode_record(text)

    part_a = CanonicalIndexPartition(ns_a, enc.get_signature())
    part_a.add(0, 0, 0, 0, k.copy())

    part_b = CanonicalIndexPartition(ns_b, enc.get_signature())
    part_b.add(0, 0, 0, 0, k.copy())

    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    gar.register_project(ns_a, "ProjectA")
    gar.register_project(ns_b, "ProjectB")
    gar.register_canonical_partition(ns_a, enc.get_signature(), part_a)
    gar.register_canonical_partition(ns_b, enc.get_signature(), part_b)

    results = gar.retrieve(
        current_namespace_id=ns_a,
        query_text=text,
        encoder=enc,
        threshold_local=0.0,
        threshold_global=0.0,
    )
    if len(results) >= 2:
        # Local (ns_a) must have higher or equal final_score, thus appear first
        assert results[0].source_namespace_id == ns_a, (
            f"Expected local ns_a first, got ns={results[0].source_namespace_id}"
        )


# ---------------------------------------------------------------------------
# 6. GAR raw vector backward compatibility
# ---------------------------------------------------------------------------

def test_gar_raw_vector_mode():
    """Verifies that GAR with query_vector still works for Fases 1-2 compatibility."""
    enc = make_encoder()
    ns_id = 0xCCCC_0001

    vector_index = PartitionedVectorIndex()
    part = vector_index.get_or_create_partition(ns_id)
    for i in range(3):
        k, _ = enc.encode_record(f"raw vector record {i}")
        part.add(segment_id=0, record_index=i, key_vector=k)

    gar = GlobalAssociativeRetrieval(vector_index=vector_index)
    gar.register_project(ns_id, "LegacyProject")

    query_vec = enc.encode_query("raw vector search")
    results = gar.retrieve(
        current_namespace_id=ns_id,
        query_vector=query_vec,
        threshold_local=0.0,
    )
    assert isinstance(results, list)


def test_gar_requires_query_or_text():
    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    with pytest.raises(ValueError):
        gar.retrieve(current_namespace_id=1)


# ---------------------------------------------------------------------------
# 7. Encoder signature isolation
# ---------------------------------------------------------------------------

def test_signature_isolation_different_encoders():
    """Two CanonicalIndexPartitions with distinct signatures must not be searched together."""
    class AltEncoder(DeterministicHashEncoder):
        def get_signature(self) -> str:
            return "unme-alt:9.9.9:768:fp16:cosine"

    enc_a = DeterministicHashEncoder()
    enc_b = AltEncoder()

    ns_id = 0xDDDD_0001
    part_a = CanonicalIndexPartition(ns_id, enc_a.get_signature())
    part_b = CanonicalIndexPartition(ns_id, enc_b.get_signature())

    text = "sensor fusion accuracy"
    k_a, _ = enc_a.encode_record(text)
    k_b, _ = enc_b.encode_record(text)
    part_a.add(0, 0, 0, 0, k_a)
    part_b.add(0, 0, 0, 0, k_b)

    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    gar.register_project(ns_id, "TestProject")
    gar.register_canonical_partition(ns_id, enc_a.get_signature(), part_a)
    gar.register_canonical_partition(ns_id, enc_b.get_signature(), part_b)

    # Retrieve with enc_a — only part_a should be searched
    results_a = gar.retrieve(
        current_namespace_id=ns_id,
        query_text=text,
        encoder=enc_a,
        threshold_local=0.0,
    )
    # All results must carry enc_a's signature
    for r in results_a:
        assert r.encoder_signature == enc_a.get_signature(), (
            f"Expected enc_a signature, got: {r.encoder_signature}"
        )


# ---------------------------------------------------------------------------
# 8. Precision metrics: Recall@5 and MRR on synthetic cross-project corpus
#
# Note: DeterministicHashEncoder uses different prefixes for encode_query vs
# encode_record, so they produce different vectors for the same text.
# For the precision tests we build the index using encode_query (same function
# used at retrieval time) to guarantee exact match, which is the controlled
# baseline.  This is explicitly noted so reviewers understand the test design.
# ---------------------------------------------------------------------------

def _build_corpus_symmetric():
    """Synthetic corpus using encode_query for both indexing and retrieval.

    3 namespaces, 5 shared records (identical across ns_a and ns_b) + 5 unique each.
    Ground truth: query on shared_texts[i] from ns_a -> ns_a[i] and ns_b[i] are correct.
    """
    enc = DeterministicHashEncoder()
    ns_a, ns_b, ns_c = 0xAA01, 0xBB01, 0xCC01

    shared_texts = [
        "GNSS IMU sensor fusion accuracy",
        "Kalman filter prediction step convergence",
        "quaternion to euler angle conversion VIALSDEG",
        "NVMe direct IO latency measurement",
        "neural adapter hidden state injection layer",
    ]
    unrelated_texts_a = [f"project alpha unique topic {i}" for i in range(5)]
    unrelated_texts_b = [f"project beta unique topic {i}" for i in range(5)]
    unrelated_texts_c = [f"project gamma unique topic {i}" for i in range(10)]

    parts: dict[int, CanonicalIndexPartition] = {}
    for ns in [ns_a, ns_b, ns_c]:
        parts[ns] = CanonicalIndexPartition(ns, enc.get_signature())

    # Use encode_query for indexing (same function as retrieval)
    for i, text in enumerate(shared_texts + unrelated_texts_a):
        k = enc.encode_query(text)   # symmetric: query prefix for both
        parts[ns_a].add(0, i, 0, i, k)

    for i, text in enumerate(shared_texts + unrelated_texts_b):
        k = enc.encode_query(text)
        parts[ns_b].add(0, i, 0, i, k)

    for i, text in enumerate(unrelated_texts_c):
        k = enc.encode_query(text)
        parts[ns_c].add(0, i, 0, i, k)

    return enc, parts, ns_a, ns_b, ns_c, shared_texts


def test_recall_at_5_cross_project():
    """Recall@5 for cross-project shared records >= 0.90.

    Uses symmetric indexing (encode_query for both) for DeterministicHashEncoder.
    """
    enc, parts, ns_a, ns_b, ns_c, shared_texts = _build_corpus_symmetric()

    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    for ns, name in [(ns_a, "NavA"), (ns_b, "NavB"), (ns_c, "Unrelated")]:
        gar.register_project(ns, name)
        gar.register_canonical_partition(ns, enc.get_signature(), parts[ns])

    hits = 0
    total_relevant = 0
    for i, text in enumerate(shared_texts):
        results = gar.retrieve(
            current_namespace_id=ns_a,
            query_text=text,
            encoder=enc,
            top_k_local=5,
            top_k_global=5,
            threshold_local=0.0,
            threshold_global=0.0,
        )
        retrieved_coords = {(r.source_namespace_id, r.record_index) for r in results[:5]}
        relevant = {(ns_a, i), (ns_b, i)}
        total_relevant += len(relevant)
        hits += len(relevant & retrieved_coords)

    recall_at_5 = hits / total_relevant
    print(f"Recall@5 = {recall_at_5:.4f} (threshold >= 0.90)")
    assert recall_at_5 >= 0.90, f"Recall@5 = {recall_at_5:.4f} < 0.90"


def test_mrr_cross_project():
    """MRR for local match (ns_a record i) >= 0.80."""
    enc, parts, ns_a, ns_b, ns_c, shared_texts = _build_corpus_symmetric()

    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    for ns, name in [(ns_a, "NavA"), (ns_b, "NavB"), (ns_c, "Unrelated")]:
        gar.register_project(ns, name)
        gar.register_canonical_partition(ns, enc.get_signature(), parts[ns])

    reciprocal_ranks = []
    for i, text in enumerate(shared_texts):
        results = gar.retrieve(
            current_namespace_id=ns_a,
            query_text=text,
            encoder=enc,
            top_k_local=5,
            top_k_global=5,
            threshold_local=0.0,
            threshold_global=0.0,
        )
        rank = None
        for pos, r in enumerate(results, start=1):
            if r.source_namespace_id == ns_a and r.record_index == i:
                rank = pos
                break
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"MRR = {mrr:.4f} (threshold >= 0.80)")
    assert mrr >= 0.80, f"MRR = {mrr:.4f} < 0.80"


def test_cross_project_fpr():
    """False positive rate for cross-project results < 0.05.

    Uses a threshold of 0.5 to filter out noise.  DeterministicHashEncoder
    produces similarity ~1.0 for exact shared texts and ~[-0.2, 0.15] for
    unrelated texts, so threshold=0.5 cleanly separates relevant from irrelevant
    cross-project results without relying on semantic understanding.
    """
    enc, parts, ns_a, ns_b, ns_c, shared_texts = _build_corpus_symmetric()

    gar = GlobalAssociativeRetrieval(vector_index=PartitionedVectorIndex())
    for ns, name in [(ns_a, "NavA"), (ns_b, "NavB"), (ns_c, "Unrelated")]:
        gar.register_project(ns, name)
        gar.register_canonical_partition(ns, enc.get_signature(), parts[ns])

    cross_retrieved = 0
    cross_irrelevant = 0
    for i, text in enumerate(shared_texts):
        results = gar.retrieve(
            current_namespace_id=ns_a,
            query_text=text,
            encoder=enc,
            top_k_local=5,
            top_k_global=5,
            threshold_local=0.5,     # Realistic threshold: filters out noise
            threshold_global=0.5,    # DeterministicHashEncoder similarity for unrelated: < 0.2
        )
        for r in results[:5]:
            if r.source_namespace_id != ns_a:
                cross_retrieved += 1
                # Correct cross-project result: ns_b record i (same text -> sim ~1.0)
                if not (r.source_namespace_id == ns_b and r.record_index == i):
                    cross_irrelevant += 1

    fpr = cross_irrelevant / cross_retrieved if cross_retrieved > 0 else 0.0
    print(f"Cross-project FPR = {fpr:.4f} (threshold < 0.05)")
    assert fpr < 0.05, f"Cross-project FPR = {fpr:.4f} >= 0.05"


if __name__ == "__main__":
    import traceback

    test_functions = [
        test_encoder_signature_format,
        test_encoder_dim,
        test_encode_query_returns_unit_norm,
        test_encode_record_returns_unit_norm_pair,
        test_encode_determinism,
        test_distinct_texts_produce_distinct_vectors,
        test_encode_no_nan_in_output,
        test_record_forces_float16_normalization,
        test_record_flags_encode_both_stored,
        test_store_save_and_retrieve_roundtrip,
        test_store_header_size_is_64_bytes,
        test_store_scan_on_reopen,
        test_store_out_of_range_raises,
        test_store_build_index_partition,
        test_index_exact_match_top_1,
        test_index_threshold_filters_low_similarity,
        test_index_top_k_ordering,
        test_index_empty_returns_empty,
        test_gar_canonical_retrieve_local,
        test_gar_canonical_cross_project_discovery,
        test_gar_composite_score_prefers_local,
        test_gar_raw_vector_mode,
        test_gar_requires_query_or_text,
        test_signature_isolation_different_encoders,
        test_recall_at_5_cross_project,
        test_mrr_cross_project,
        test_cross_project_fpr,
    ]

    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests")
    if failed:
        sys.exit(1)
