"""Test suite for UNME durability, segment rotation, encrypted CML event logs,
and NonceEpochManager recovery uniqueness.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.cml import CMLRecord, CMLStorage, CMLEvaluationStatus
from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader, MultiSegmentReader
from unme.crypto.namespace import NamespaceManager, NonceEpochManager


def test_nonce_epoch_manager_uniqueness_on_restart():
    with tempfile.TemporaryDirectory() as tmpdir:
        # First session
        mgr1 = NonceEpochManager(tmpdir, segment_id=1)
        uuid1 = mgr1.segment_uuid
        epoch1 = mgr1.write_epoch
        nonce_s1_r0 = mgr1.generate_nonce(0)
        nonce_s1_r1 = mgr1.generate_nonce(1)
        assert nonce_s1_r0 != nonce_s1_r1

        # Second session (simulating process restart on same segment)
        mgr2 = NonceEpochManager(tmpdir, segment_id=1)
        uuid2 = mgr2.segment_uuid
        epoch2 = mgr2.write_epoch
        assert uuid1 == uuid2, "Segment UUID must be persistent across restarts!"
        assert epoch2 == epoch1 + 1, f"Epoch must increment monotonically on restart! ({epoch2} vs {epoch1})"

        nonce_s2_r0 = mgr2.generate_nonce(0)
        # Even for the exact same record_index 0, nonces across restarts MUST be strictly distinct!
        assert nonce_s2_r0 != nonce_s1_r0, "CRITICAL: Nonce reuse detected across process restart!"
        print("- NonceEpochManager restart uniqueness: PASSED")


def test_encrypted_cml_storage_and_tamper_resistance():
    mgr = NamespaceManager(master_secret=b"MASTER_KEY_CML_ENC_0123456789")
    ns_id = mgr.derive_namespace_id("VIALSDE")
    key = mgr.derive_namespace_key("VIALSDE")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save CML records with encryption
        cml_store = CMLStorage(tmpdir, namespace_id=ns_id, namespace_key=key, namespace_mgr=mgr)
        cml = CMLRecord(
            source_memory_id="MEM-ALPHA-01",
            source_namespace_id=101,
            target_namespace_id=ns_id,
            relevance_reason="Confidential trajectory correction data",
            intended_usage="Kalman filter antecedent",
        )
        cml_store.save_cml(cml)

        # Transition status
        cml.transition_to(CMLEvaluationStatus.EVALUATING, reason="Testing in simulator")
        cml_store.save_cml(cml)

        # 2. Inspect physical file on disk to verify it is NOT plaintext JSON
        enc_file = cml_store.enc_file
        with open(enc_file, "rb") as f:
            raw_bytes = f.read()
            assert b"Confidential trajectory correction" not in raw_bytes, "PLAINTEXT LEAK in CML storage!"

        # 3. Reload storage and verify events replay correctly
        reloaded_store = CMLStorage(tmpdir, namespace_id=ns_id, namespace_key=key, namespace_mgr=mgr)
        loaded_cml = reloaded_store.get_cml(cml.link_id)
        assert loaded_cml is not None
        assert loaded_cml.evaluation_status == CMLEvaluationStatus.EVALUATING.value
        assert len(loaded_cml.history) == 1
        assert loaded_cml.history[0]["to_status"] == CMLEvaluationStatus.EVALUATING.value

        # 4. Attempt reload with wrong key (tamper/unauthorized)
        wrong_key = mgr.derive_namespace_key("Other_Project")
        tampered_store = CMLStorage(tmpdir, namespace_id=ns_id, namespace_key=wrong_key, namespace_mgr=mgr)
        # Should not deserialize unauthenticated events
        assert len(tampered_store.list_links()) == 0

        print("- Encrypted CML event log persistence & AEAD protection: PASSED")


def test_automated_segment_rotation_and_multireader():
    mgr = NamespaceManager(master_secret=b"MASTER_KEY_ROTATION_TEST_998877")
    ns_id = mgr.derive_namespace_id("Rotation_Project")
    key = mgr.derive_namespace_key("Rotation_Project")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Configure writer to rotate every 5 records
        writer = SegmentWriter(
            tmpdir,
            segment_id=1,
            namespace_mgr=mgr,
            namespace_key=key,
            max_records_per_segment=5,
        )

        dim = 256
        created_records = []
        for i in range(12):
            k = np.random.randn(dim).astype(np.float16)
            v = np.random.randn(dim).astype(np.float16)
            rec = UMRRecord(
                record_id=f"REC-{i:03d}",
                namespace_id=ns_id,
                canonical_text=f"Rotation record #{i}",
                key_vector=k,
                value_vector=v,
            )
            seg_id, rec_idx = writer.append(rec, encrypt=True)
            created_records.append((seg_id, rec_idx, f"REC-{i:03d}"))
        writer.close()

        # Verify segments were rotated: 12 records / 5 per segment = 3 segments (000001, 000002, 000003)
        segment_files = list(Path(tmpdir).glob("segment_*.umrh"))
        assert len(segment_files) == 3, f"Expected 3 segment files, found {len(segment_files)}"

        # Verify MultiSegmentReader can access all 12 records seamlessly
        multi_reader = MultiSegmentReader(tmpdir, namespace_key=key)
        assert multi_reader.total_records == 12
        assert multi_reader.segment_ids == [1, 2, 3]

        for seg_id, rec_idx, expected_id in created_records:
            read_rec = multi_reader.get_record(seg_id, rec_idx, verify_checksum=True)
            assert read_rec.record_id == expected_id

        multi_reader.close()
        print("- Automated segment rotation and MultiSegmentReader: PASSED")


if __name__ == "__main__":
    print("Running UNME Durability & Rotation Tests...")
    test_nonce_epoch_manager_uniqueness_on_restart()
    test_encrypted_cml_storage_and_tamper_resistance()
    test_automated_segment_rotation_and_multireader()
    print("ALL DURABILITY & ROTATION TESTS PASSED.")
