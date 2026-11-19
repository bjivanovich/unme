"""Multiprocess concurrency test suite for UNME storage engine.
Verifies file-level locking with multiple independent worker processes appending
simultaneously to the same segment without offset collisions or duplicate record indices.
"""

from __future__ import annotations
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader
from unme.crypto.namespace import NamespaceManager


def _worker_append(segment_dir: str, worker_id: int, records_per_worker: int, dim: int = 128) -> list[int]:
    """Worker process appending records concurrently using file locking."""
    mgr = NamespaceManager(master_secret=b"MULTIPROCESS_CONCURRENCY_TEST_SECRET")
    ns_id = 999
    key = mgr.derive_namespace_key("Concurrent_Project")

    writer = SegmentWriter(segment_dir, segment_id=1, namespace_mgr=mgr, namespace_key=key)
    assigned_indices = []

    for i in range(records_per_worker):
        k = np.random.randn(dim).astype(np.float16)
        v = np.random.randn(dim).astype(np.float16)
        rec = UMRRecord(
            record_id=f"W{worker_id}-REC-{i:04d}",
            namespace_id=ns_id,
            canonical_text=f"Concurrent memory entry from worker {worker_id} iteration {i}",
            key_vector=k,
            value_vector=v,
        )
        _, rec_idx = writer.append(rec, encrypt=True)
        assigned_indices.append(rec_idx)

    writer.close()
    return assigned_indices


def test_multiprocess_concurrent_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        num_workers = 4
        records_per_worker = 50
        total_expected = num_workers * records_per_worker

        with mp.Pool(processes=num_workers) as pool:
            tasks = [
                pool.apply_async(_worker_append, (tmpdir, w_id, records_per_worker))
                for w_id in range(num_workers)
            ]
            results = [t.get() for t in tasks]

        all_indices = []
        for worker_indices in results:
            all_indices.extend(worker_indices)

        # 1. Verify total count
        assert len(all_indices) == total_expected, f"Expected {total_expected} records, got {len(all_indices)}"

        # 2. Verify strict uniqueness (NO duplicate record_index!)
        unique_indices = set(all_indices)
        assert len(unique_indices) == total_expected, f"Duplicate record_index detected! Unique: {len(unique_indices)} vs Expected: {total_expected}"
        assert sorted(all_indices) == list(range(total_expected)), "Indices are not a contiguous sequence 0..N-1!"

        # 3. Verify physical integrity of every record
        mgr = NamespaceManager(master_secret=b"MULTIPROCESS_CONCURRENCY_TEST_SECRET")
        key = mgr.derive_namespace_key("Concurrent_Project")
        reader = SegmentReader(tmpdir, segment_id=1, namespace_key=key)
        assert reader.record_count == total_expected

        for idx in range(total_expected):
            record = reader.get_record(idx, verify_checksum=True)
            assert "Concurrent memory entry" in record.canonical_text

        reader.close()
        print(f"- Multiprocess concurrent appends ({num_workers} workers, {total_expected} records): PASSED (0 collisions, 0 duplicates)")


if __name__ == "__main__":
    mp.freeze_support()
    print("Running UNME Multiprocess Concurrency Tests...")
    test_multiprocess_concurrent_appends()
    print("ALL MULTIPROCESS CONCURRENCY TESTS PASSED.")
