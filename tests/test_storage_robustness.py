"""Robustness and crash recovery test suite for UNME storage core.
Tests data corruption detection, partial write truncation, and recovery.
"""

from __future__ import annotations
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.record import UMRRecord
from unme.core.storage import (
    SegmentWriter,
    SegmentReader,
    SegmentRecovery,
    StorageCorruptionError,
)


def test_payload_integrity_and_tampering():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir, segment_id=1)
        k = np.random.randn(1536).astype(np.float16)
        v = np.random.randn(1536).astype(np.float16)
        rec = UMRRecord(
            namespace_id=101,
            canonical_text="Untampered memory record",
            key_vector=k,
            value_vector=v,
        )
        seg_id, idx = writer.append(rec)
        writer.close()

        # 1. Normal read with verification
        reader = SegmentReader(tmpdir, segment_id=1)
        rec_read = reader.get_record(idx, verify_checksum=True)
        assert rec_read.canonical_text == "Untampered memory record"
        reader.close()

        # 2. Simulate silent disk corruption by flipping a byte in .umrt
        tensor_file = Path(tmpdir) / "segment_000001.umrt"
        with open(tensor_file, "r+b") as f:
            f.seek(10)  # Inside the key vector
            orig_byte = f.read(1)
            corrupted_byte = bytes([orig_byte[0] ^ 0xFF])
            f.seek(10)
            f.write(corrupted_byte)

        # 3. Reading with verification must raise StorageCorruptionError
        reader_corrupted = SegmentReader(tmpdir, segment_id=1)
        corruption_caught = False
        try:
            _ = reader_corrupted.get_record(idx, verify_checksum=True)
        except StorageCorruptionError:
            corruption_caught = True
        finally:
            reader_corrupted.close()

        assert corruption_caught, "Failed to catch silent data corruption in tensors/payload!"
        print("- Data corruption detection test: PASSED")


def test_crash_recovery_partial_writes():
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir, segment_id=2)
        dim = 512
        for i in range(5):
            k = np.random.randn(dim).astype(np.float16)
            v = np.random.randn(dim).astype(np.float16)
            rec = UMRRecord(
                namespace_id=202,
                canonical_text=f"Record {i}",
                key_vector=k,
                value_vector=v,
            )
            writer.append(rec)
        writer.close()

        # Append partial garbage bytes to simulate sudden power outage / crash mid-write
        header_file = Path(tmpdir) / "segment_000002.umrh"
        tensor_file = Path(tmpdir) / "segment_000002.umrt"

        # Simulate incomplete 6th record in .umrt (written halfway)
        with open(tensor_file, "ab") as f:
            f.write(b"\xAA" * 123)

        # Incomplete 6th header in .umrh (written 20 bytes out of 64)
        with open(header_file, "ab") as f:
            f.write(b"\xBB" * 20)

        # Run recovery
        recovered_count = SegmentRecovery.recover_segment(tmpdir, segment_id=2)
        assert recovered_count == 5, f"Expected 5 valid records, got {recovered_count}"

        # Verify files were truncated back to clean boundary
        assert header_file.stat().st_size == 5 * 64, "Header file was not truncated to valid boundary!"

        # Opening writer again should resume cleanly from index 5
        writer2 = SegmentWriter(tmpdir, segment_id=2)
        assert writer2.record_count == 5
        k_new = np.random.randn(dim).astype(np.float16)
        v_new = np.random.randn(dim).astype(np.float16)
        rec_new = UMRRecord(
            namespace_id=202,
            canonical_text="Record 5 after recovery",
            key_vector=k_new,
            value_vector=v_new,
        )
        _, new_idx = writer2.append(rec_new)
        assert new_idx == 5
        writer2.close()

        reader = SegmentReader(tmpdir, segment_id=2)
        assert reader.record_count == 6
        rec5 = reader.get_record(5, verify_checksum=True)
        assert rec5.canonical_text == "Record 5 after recovery"
        reader.close()

        print("- Crash recovery and rollback test: PASSED")


if __name__ == "__main__":
    print("Running UNME Storage Robustness & Recovery Tests...")
    test_payload_integrity_and_tampering()
    test_crash_recovery_partial_writes()
    print("ALL ROBUSTNESS TESTS PASSED.")
