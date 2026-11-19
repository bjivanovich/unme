"""Verification test suite for authentic AEAD storage, canonical AAD binding, and strict checksum recovery.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.constants import FLAG_ENCRYPTED
from unme.core.record import UMRRecord
from unme.core.storage import (
    SegmentWriter,
    SegmentReader,
    SegmentRecovery,
    StorageCorruptionError,
)
from unme.crypto.namespace import NamespaceManager


def test_aead_at_rest_and_aad_binding():
    mgr = NamespaceManager(master_secret=b"MASTER_KEY_AEAD_TEST_SECRET_012345")
    key_alpha = mgr.derive_namespace_key("Project_Alpha")
    key_beta = mgr.derive_namespace_key("Project_Beta")
    ns_alpha = mgr.derive_namespace_id("Project_Alpha")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write encrypted record
        writer = SegmentWriter(tmpdir, segment_id=1, namespace_mgr=mgr, namespace_key=key_alpha)
        k = np.random.randn(1536).astype(np.float16)
        v = np.random.randn(1536).astype(np.float16)
        secret_fact = "Classified internal architecture decision for Project Alpha."
        rec = UMRRecord(
            namespace_id=ns_alpha,
            canonical_text=secret_fact,
            structured_data={"confidential": True},
            key_vector=k,
            value_vector=v,
        )
        seg_id, idx = writer.append(rec, encrypt=True)
        writer.close()

        # 2. Inspect physical file on disk to ensure it is NOT plaintext
        tensor_file = Path(tmpdir) / "segment_000001.umrt"
        with open(tensor_file, "rb") as f:
            disk_bytes = f.read()
            assert secret_fact.encode("utf-8") not in disk_bytes, "PLAINTEXT LEAK! Payload was not encrypted on disk!"

        # 3. Read back with correct key
        reader_auth = SegmentReader(tmpdir, segment_id=1, namespace_key=key_alpha)
        rec_read = reader_auth.get_record(idx, verify_checksum=True)
        assert rec_read.canonical_text == secret_fact
        reader_auth.close()

        # 4. Read back with unauthorized key (Project Beta key) MUST fail authentication
        reader_unauth = SegmentReader(tmpdir, segment_id=1, namespace_key=key_beta)
        unauth_failed = False
        try:
            _ = reader_unauth.get_record(idx, verify_checksum=True)
        except StorageCorruptionError:
            unauth_failed = True
        finally:
            reader_unauth.close()

        assert unauth_failed, "Project Beta was able to decrypt Project Alpha's storage file!"
        print("- Storage AEAD encryption and key isolation: PASSED")


def test_strict_recovery_with_silent_corruption():
    mgr = NamespaceManager(master_secret=b"MASTER_KEY_AEAD_TEST_SECRET_012345")
    key = mgr.derive_namespace_key("Project_Alpha")

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SegmentWriter(tmpdir, segment_id=5, namespace_mgr=mgr, namespace_key=key)
        dim = 512
        for i in range(4):
            k = np.random.randn(dim).astype(np.float16)
            v = np.random.randn(dim).astype(np.float16)
            rec = UMRRecord(
                namespace_id=123,
                canonical_text=f"Factual record {i}",
                key_vector=k,
                value_vector=v,
            )
            writer.append(rec, encrypt=True)
        writer.close()

        # Simulate silent corruption of 4th record: flip byte inside value vector
        header_file = Path(tmpdir) / "segment_000005.umrh"
        tensor_file = Path(tmpdir) / "segment_000005.umrt"

        # Corrupt the 4th record by writing garbage in its tensor bytes
        with open(header_file, "rb") as hf:
            hf.seek(3 * 64)  # 4th header
            from unme.core.header import UMRHeader
            h4 = UMRHeader.unpack(hf.read(64))

        with open(tensor_file, "r+b") as tf:
            tf.seek(h4.offset_v + 4)
            tf.write(b"\xFF\xFF")

        # Run strict recovery: 4th record's data CRC will fail!
        # Recovery MUST truncate and keep ONLY 3 records.
        recovered_count = SegmentRecovery.recover_segment(tmpdir, segment_id=5)
        assert recovered_count == 3, f"Expected 3 verified records after corruption rollback, got {recovered_count}"
        assert header_file.stat().st_size == 3 * 64

        reader = SegmentReader(tmpdir, segment_id=5, namespace_key=key)
        assert reader.record_count == 3
        reader.close()
        print("- Strict recovery with CRC checksum verification: PASSED")


if __name__ == "__main__":
    print("Running UNME AEAD & Strict Recovery Tests...")
    test_aead_at_rest_and_aad_binding()
    test_strict_recovery_with_silent_corruption()
    print("ALL STORAGE HARDENING TESTS PASSED.")
