"""End-to-end test suite for Global Associative Retrieval (GAR), Contextual Memory Links (CML),
and independent derived knowledge lifecycles.
Verifies the complete scenario: VIALSDE discovering Project A memory, creating CML,
deriving local memory, and surviving origin memory tombstone.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unme.core.cml import CMLRecord, CMLStorage, CMLEvaluationStatus, CMLRelationshipType
from unme.core.lifecycle import MemoryLifecycleManager
from unme.core.record import UMRRecord
from unme.core.storage import SegmentWriter, SegmentReader
from unme.crypto.namespace import NamespaceManager
from unme.index.vector_index import PartitionedVectorIndex
from unme.index.gar import GlobalAssociativeRetrieval


def test_gar_cross_project_and_cml_lifecycle():
    mgr = NamespaceManager(master_secret=b"MASTER_KEY_GAR_LIFECYCLE_TEST_12345")
    ns_vialsde = mgr.derive_namespace_id("VIALSDE")
    ns_project_a = mgr.derive_namespace_id("Project_A_Navigation")

    dim = 512
    v_index = PartitionedVectorIndex(dim=dim)
    lifecycle_mgr = MemoryLifecycleManager(data_dir=tempfile.gettempdir())

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        dir_vialsde = dir_path / "vialsde"
        dir_project_a = dir_path / "project_a"

        # 1. Populate Project A with original memory: R-001 (GNSS/IMU trajectory analysis)
        key_a = mgr.derive_namespace_key("Project_A_Navigation")
        writer_a = SegmentWriter(dir_project_a, segment_id=1, namespace_mgr=mgr, namespace_key=key_a)

        gnss_imu_vec = np.random.randn(dim).astype(np.float32)
        gnss_imu_vec /= np.linalg.norm(gnss_imu_vec)

        rec_a = UMRRecord(
            record_id="MEM-00128",
            namespace_id=ns_project_a,
            canonical_text="Propuesta de utilizacion de GNSS e IMU para detectar desviaciones de trayectoria.",
            structured_data={"domain": "navigation", "hardware": ["GNSS", "IMU"]},
            provenance={"author": "team_navigation", "version": 1},
            key_vector=gnss_imu_vec.astype(np.float16),
            value_vector=gnss_imu_vec.astype(np.float16),
        )
        seg_id_a, idx_a = writer_a.append(rec_a, encrypt=True)
        writer_a.close()

        v_index.add_record(namespace_id=ns_project_a, segment_id=seg_id_a, record_index=idx_a, key_vector=gnss_imu_vec)

        # 2. Populate VIALSDE with its own local memory: R-010 (Pickup distance estimation)
        key_v = mgr.derive_namespace_key("VIALSDE")
        writer_v = SegmentWriter(dir_vialsde, segment_id=1, namespace_mgr=mgr, namespace_key=key_v)

        vialsde_local_vec = np.random.randn(dim).astype(np.float32)
        vialsde_local_vec /= np.linalg.norm(vialsde_local_vec)

        rec_v_init = UMRRecord(
            record_id="MEM-00010",
            namespace_id=ns_vialsde,
            canonical_text="Discrepancias entre distancia de recogida estimada y real en ruta VIALSDE.",
            structured_data={"domain": "fleet_routing"},
            key_vector=vialsde_local_vec.astype(np.float16),
            value_vector=vialsde_local_vec.astype(np.float16),
        )
        seg_id_v, idx_v = writer_v.append(rec_v_init, encrypt=True)
        writer_v.close()

        v_index.add_record(namespace_id=ns_vialsde, segment_id=seg_id_v, record_index=idx_v, key_vector=vialsde_local_vec)

        # 3. Setup Readers and CML Storage
        reader_a = SegmentReader(dir_project_a, segment_id=1, namespace_key=key_a)
        reader_v = SegmentReader(dir_vialsde, segment_id=1, namespace_key=key_v)
        cml_storage_vialsde = CMLStorage(dir_vialsde, namespace_id=ns_vialsde, namespace_key=key_v, namespace_mgr=mgr)

        gar = GlobalAssociativeRetrieval(vector_index=v_index)
        gar.register_project(ns_vialsde, "VIALSDE", cml_storage=cml_storage_vialsde, reader=reader_v)
        gar.register_project(ns_project_a, "Project_A_Navigation", reader=reader_a)

        # 4. Global Associative Query from VIALSDE session:
        # VIALSDE issues a query about trajectory discrepancy (correlated with gnss_imu_vec)
        query_vec = gnss_imu_vec + (np.random.randn(dim).astype(np.float32) * 0.005)
        query_vec /= np.linalg.norm(query_vec)

        retrieved_results = gar.retrieve(
            current_namespace_id=ns_vialsde,
            query_vector=query_vec,
            threshold_local=0.65,
            threshold_global=0.75,
        )

        assert len(retrieved_results) > 0, "GAR failed to discover relevant memory!"

        # Verify discovery of cross-project memory from Project A
        match_from_a = next((r for r in retrieved_results if r.source_namespace_id == ns_project_a), None)
        assert match_from_a is not None, "Failed to retrieve antecedent from Project A!"
        assert match_from_a.source_record_id == "MEM-00128"
        assert match_from_a.source_project_name == "Project_A_Navigation"
        assert match_from_a.is_local is False, "Cross-project memory incorrectly flagged as local!"
        assert "GNSS" in match_from_a.canonical_text

        # 5. Create Contextual Memory Link (CML) in VIALSDE
        cml = gar.create_contextual_link(
            target_namespace_id=ns_vialsde,
            gar_result=match_from_a,
            relationship_type=CMLRelationshipType.ANTECEDENTE_TECNICO.value,
            relevance_reason="Util para corregir desviaciones de trayectoria en recogida",
            intended_usage="Antecedente tecnico para integracion de filtro Kalman adaptativo",
            applicability_constraints="Requiere calibracion para frecuencia de muestreo de 10 Hz",
        )
        assert cml.evaluation_status == CMLEvaluationStatus.DISCOVERED.value
        assert cml.source_memory_id == "MEM-00128"
        assert cml.target_namespace_id == ns_vialsde

        # 6. Verify Original Memory in Project A was 100% untouched
        rec_a_post_cml = reader_a.get_record(0, verify_checksum=True)
        assert rec_a_post_cml.record_id == "MEM-00128"
        assert rec_a_post_cml.namespace_id == ns_project_a
        assert rec_a_post_cml.canonical_text == rec_a.canonical_text, "Original memory was modified!"

        # 7. VIALSDE develops its own DERIVED memory (R-090) during conversation
        # Writer VIALSDE creates R-090
        writer_v_2 = SegmentWriter(dir_vialsde, segment_id=2, namespace_mgr=mgr, namespace_key=key_v)
        rec_v_derived = UMRRecord(
            record_id="MEM-00090",
            namespace_id=ns_vialsde,
            canonical_text="Implementacion VIALSDE: Filtro Kalman adaptativo con compensacion de latencia GNSS.",
            structured_data={"derived_from_cml": cml.link_id},
            provenance={"author": "vialsde_engineer", "cml_ref": cml.link_id},
            key_vector=query_vec.astype(np.float16),
            value_vector=query_vec.astype(np.float16),
        )
        seg_id_d, idx_derived = writer_v_2.append(rec_v_derived, encrypt=True)
        writer_v_2.close()

        # Update CML status in VIALSDE to ADAPTED
        cml.link_derived_memory("MEM-00090")
        cml_storage_vialsde.save_cml(cml)
        assert cml.evaluation_status == CMLEvaluationStatus.ADAPTED.value
        assert "MEM-00090" in cml.derived_memory_ids

        # 8. Origin Memory MEM-00128 in Project A is TOMBSTONED / DELETED
        lifecycle_mgr.tombstone_record(
            namespace_id=ns_project_a,
            record_id="MEM-00128",
            reason="Project A navigation stack deprecated",
            cml_storages=[cml_storage_vialsde],
        )

        # 9. Verify Post-Tombstone Independence:
        # A) CML in VIALSDE transitioned to is_source_available = False and SOURCE_BECAME_UNAVAILABLE
        reloaded_cml = cml_storage_vialsde.get_cml(cml.link_id)
        assert reloaded_cml is not None
        assert reloaded_cml.is_source_available is False, "CML did not reflect source tombstone!"
        assert reloaded_cml.evaluation_status == CMLEvaluationStatus.SOURCE_BECAME_UNAVAILABLE.value
        assert any(h["to_status"] == CMLEvaluationStatus.SOURCE_BECAME_UNAVAILABLE.value for h in reloaded_cml.history)

        # B) VIALSDE's derived memory MEM-00090 continues to exist, decrypt, and function 100% independently!
        reader_v_2 = SegmentReader(dir_vialsde, segment_id=2, namespace_key=key_v)
        rec_derived_read = reader_v_2.get_record(0, verify_checksum=True)
        assert rec_derived_read.record_id == "MEM-00090"
        assert "Filtro Kalman adaptativo" in rec_derived_read.canonical_text
        reader_v_2.close()

        reader_a.close()
        reader_v.close()
        print("- Global Associative Retrieval, CML lifecycle, and independent derived memory: PASSED")


if __name__ == "__main__":
    print("Running UNME GAR & CML Lifecycle Tests...")
    test_gar_cross_project_and_cml_lifecycle()
    print("ALL GAR & CML LIFECYCLE TESTS PASSED.")
