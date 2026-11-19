"""Global Associative Retrieval (GAR) Engine for UNME — updated for Fase 3A.

Changes from previous version:
- retrieve() now optionally accepts a canonical ICanonicalEncoder.  When provided,
  the query is encoded into the canonical space before searching CanonicalIndexPartitions.
  When no encoder is provided, raw vector search on NamespaceIndexPartitions is used
  (backward-compatible with Fases 1-2 tests).
- Ranking: multidimensional composite score as specified in the Phase 3A contract.
- CanonicalIndexPartition lookup is by (namespace_id, encoder_signature), enabling
  strict separation of vectors from distinct encoder versions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unme.core.cml import CMLRecord, CMLStorage, CMLEvaluationStatus, CMLRelationshipType
from unme.core.storage import SegmentReader, MultiSegmentReader
from unme.index.vector_index import PartitionedVectorIndex, SearchResult


@dataclass
class GARResult:
    """Rich associative retrieval result with source context, provenance, and CML info."""
    source_namespace_id: int
    source_project_name: str
    source_record_id: str
    record_index: int
    segment_id: int
    similarity: float
    final_score: float               # Composite multidimensional score
    is_local: bool
    canonical_text: str
    provenance: dict[str, Any]
    structured_data: dict[str, Any]
    existing_cml: CMLRecord | None = None
    encoder_signature: str = ""      # Non-empty when canonical search was used


class GlobalAssociativeRetrieval:
    """Orchestrator for global cross-project memory discovery.

    Supports two retrieval modes:
    1. Raw vector mode (backward-compatible): caller provides a pre-encoded
       query_vector directly.  Searches NamespaceIndexPartitions.
    2. Canonical mode: caller provides query_text + an ICanonicalEncoder.
       The engine encodes the query and searches CanonicalIndexPartitions,
       enforcing strict encoder_signature separation.
    """

    # Multidimensional ranking weights (Phase 3A specification)
    BETA_COSINE: float   = 1.0
    BETA_RECENCY: float  = 0.05
    BETA_CML: float      = 0.10
    GAMMA_DOMAIN: float  = 0.08

    def __init__(
        self,
        vector_index: PartitionedVectorIndex,
        project_names: dict[int, str] | None = None,
        cml_storages: dict[int, CMLStorage] | None = None,
        readers: dict[int, Any] | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.project_names = project_names or {}
        self.cml_storages = cml_storages or {}
        self.readers = readers or {}
        # Canonical index partitions: (namespace_id, encoder_signature) -> CanonicalIndexPartition
        self._canonical_indices: dict[tuple[int, str], Any] = {}

    def register_project(
        self,
        namespace_id: int,
        project_name: str,
        cml_storage: CMLStorage | None = None,
        reader: Any | None = None,
    ) -> None:
        self.project_names[namespace_id] = project_name
        if cml_storage is not None:
            self.cml_storages[namespace_id] = cml_storage
        if reader is not None:
            self.readers[namespace_id] = reader

    def register_canonical_partition(
        self,
        namespace_id: int,
        encoder_signature: str,
        partition: Any,  # CanonicalIndexPartition
    ) -> None:
        """Registers a CanonicalIndexPartition for a given namespace + encoder signature."""
        self._canonical_indices[(namespace_id, encoder_signature)] = partition

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recency_bonus(self, provenance: dict[str, Any]) -> float:
        """Small recency bonus in [0, 1] based on provenance version field."""
        version = provenance.get("version", 1)
        try:
            return min(float(version) / 100.0, 1.0)
        except (TypeError, ValueError):
            return 0.0

    def _cml_bonus(self, cml_entry: CMLRecord | None) -> float:
        """Priority bonus for memories already adapted in the target project."""
        if cml_entry is None:
            return 0.0
        if cml_entry.evaluation_status == CMLEvaluationStatus.ADAPTED.value:
            return 1.0
        if cml_entry.evaluation_status == CMLEvaluationStatus.REFERENCED.value:
            return 0.5
        return 0.0

    def _domain_distance(self, is_local: bool) -> float:
        """Small penalty for cross-project results to prefer local over distant."""
        return 0.0 if is_local else 1.0

    def _compose_score(
        self,
        similarity: float,
        provenance: dict[str, Any],
        cml_entry: CMLRecord | None,
        is_local: bool,
    ) -> float:
        """Computes the multidimensional ranking score per Phase 3A specification:
        S_final = beta1 * Sim_cos + beta2 * phi_recency + beta3 * I_cml_adapted - gamma * Delta_domain
        """
        return (
            self.BETA_COSINE * similarity
            + self.BETA_RECENCY * self._recency_bonus(provenance)
            + self.BETA_CML * self._cml_bonus(cml_entry)
            - self.GAMMA_DOMAIN * self._domain_distance(is_local)
        )

    def _enrich_match(
        self,
        ns_id: int,
        segment_id: int,
        record_index: int,
        similarity: float,
        is_local: bool,
        encoder_signature: str,
        existing_cmls: dict[str, CMLRecord],
    ) -> GARResult:
        """Fetches record content and builds a GARResult with composite score."""
        reader = self.readers.get(ns_id)
        proj_name = self.project_names.get(ns_id, f"Project_{ns_id:016x}")
        rec_id = f"REC-{ns_id:08x}-{record_index:04d}"
        canonical_text = ""
        provenance: dict[str, Any] = {"namespace_id": ns_id}
        structured_data: dict[str, Any] = {}

        if reader is not None:
            try:
                if isinstance(reader, MultiSegmentReader):
                    record = reader.get_record(segment_id, record_index, verify_checksum=False)
                else:
                    record = reader.get_record(record_index, verify_checksum=False)
                rec_id = record.record_id or rec_id
                canonical_text = record.canonical_text
                provenance = record.provenance
                structured_data = record.structured_data
            except Exception:
                pass

        cml_entry = existing_cmls.get(rec_id)
        final_score = self._compose_score(similarity, provenance, cml_entry, is_local)

        return GARResult(
            source_namespace_id=ns_id,
            source_project_name=proj_name,
            source_record_id=rec_id,
            record_index=record_index,
            segment_id=segment_id,
            similarity=similarity,
            final_score=final_score,
            is_local=is_local,
            canonical_text=canonical_text,
            provenance=provenance,
            structured_data=structured_data,
            existing_cml=cml_entry,
            encoder_signature=encoder_signature,
        )

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        current_namespace_id: int,
        query_vector: np.ndarray | None = None,
        query_text: str | None = None,
        encoder: Any | None = None,   # ICanonicalEncoder
        top_k_local: int = 5,
        top_k_global: int = 5,
        threshold_local: float = 0.74,
        threshold_global: float = 0.80,
        allowed_namespaces: list[int] | None = None,
    ) -> list[GARResult]:
        """Executes global associative retrieval.

        Two modes:
        - Canonical (recommended): pass query_text + encoder.  The encoder encodes
          the query and searches CanonicalIndexPartitions with strict signature separation.
        - Raw vector (backward-compatible): pass query_vector.  Searches
          NamespaceIndexPartitions directly (Fases 1-2 behavior).

        Results are sorted by final_score (descending), with local results preferred
        for equal scores.
        """
        if query_text is not None and encoder is not None:
            return self._retrieve_canonical(
                current_namespace_id=current_namespace_id,
                query_text=query_text,
                encoder=encoder,
                top_k_local=top_k_local,
                top_k_global=top_k_global,
                threshold_local=threshold_local,
                threshold_global=threshold_global,
                allowed_namespaces=allowed_namespaces,
            )
        elif query_vector is not None:
            return self._retrieve_raw(
                current_namespace_id=current_namespace_id,
                query_vector=query_vector,
                top_k_local=top_k_local,
                top_k_global=top_k_global,
                threshold_local=threshold_local,
                threshold_global=threshold_global,
                allowed_namespaces=allowed_namespaces,
            )
        else:
            raise ValueError(
                "retrieve() requires either (query_text + encoder) or query_vector"
            )

    def _retrieve_canonical(
        self,
        current_namespace_id: int,
        query_text: str,
        encoder: Any,
        top_k_local: int,
        top_k_global: int,
        threshold_local: float,
        threshold_global: float,
        allowed_namespaces: list[int] | None,
    ) -> list[GARResult]:
        """Canonical retrieval path using ICanonicalEncoder and CanonicalIndexPartitions."""
        sig = encoder.get_signature()
        query_vec = encoder.encode_query(query_text)   # float16 unit vector

        all_ns = (
            allowed_namespaces
            if allowed_namespaces is not None
            else list(self.project_names.keys())
        )
        if current_namespace_id not in all_ns:
            all_ns.append(current_namespace_id)

        target_cml_storage = self.cml_storages.get(current_namespace_id)
        existing_cmls: dict[str, CMLRecord] = {}
        if target_cml_storage is not None:
            for cml in target_cml_storage.list_links():
                existing_cmls[cml.source_memory_id] = cml

        results: list[GARResult] = []

        for ns_id in all_ns:
            partition = self._canonical_indices.get((ns_id, sig))
            if partition is None:
                continue
            is_local = (ns_id == current_namespace_id)
            threshold = threshold_local if is_local else threshold_global
            top_k = top_k_local if is_local else top_k_global

            raw_matches = partition.search(
                query_vector=query_vec,
                top_k=top_k,
                threshold=threshold,
            )
            for match in raw_matches:
                result = self._enrich_match(
                    ns_id=ns_id,
                    segment_id=match.umr_segment_id,
                    record_index=match.umr_record_index,
                    similarity=match.similarity,
                    is_local=is_local,
                    encoder_signature=sig,
                    existing_cmls=existing_cmls,
                )
                results.append(result)

        results.sort(key=lambda r: (-r.final_score, not r.is_local))
        return results

    def _retrieve_raw(
        self,
        current_namespace_id: int,
        query_vector: np.ndarray,
        top_k_local: int,
        top_k_global: int,
        threshold_local: float,
        threshold_global: float,
        allowed_namespaces: list[int] | None,
    ) -> list[GARResult]:
        """Backward-compatible raw vector retrieval (Fases 1-2 behavior)."""
        all_ns = (
            allowed_namespaces
            if allowed_namespaces is not None
            else list(self.project_names.keys())
        )
        if current_namespace_id not in all_ns:
            all_ns.append(current_namespace_id)

        target_cml_storage = self.cml_storages.get(current_namespace_id)
        existing_cmls: dict[str, CMLRecord] = {}
        if target_cml_storage is not None:
            for cml in target_cml_storage.list_links():
                existing_cmls[cml.source_memory_id] = cml

        results: list[GARResult] = []

        for ns_id in all_ns:
            is_local = (ns_id == current_namespace_id)
            threshold = threshold_local if is_local else threshold_global
            top_k = top_k_local if is_local else top_k_global

            raw_matches = self.vector_index.search(
                namespace_id=ns_id,
                query_vector=query_vector,
                top_k=top_k,
                threshold=threshold,
                use_cache=True,
            )
            for match in raw_matches:
                result = self._enrich_match(
                    ns_id=ns_id,
                    segment_id=match.segment_id,
                    record_index=match.record_index,
                    similarity=match.similarity,
                    is_local=is_local,
                    encoder_signature="",
                    existing_cmls=existing_cmls,
                )
                results.append(result)

        results.sort(key=lambda r: (-r.final_score, not r.is_local))
        return results

    def create_contextual_link(
        self,
        target_namespace_id: int,
        gar_result: GARResult,
        relationship_type: str = CMLRelationshipType.ANTECEDENTE_TECNICO.value,
        relevance_reason: str = "",
        intended_usage: str = "",
        applicability_constraints: str = "",
    ) -> CMLRecord:
        """Creates an independent Contextual Memory Link (CML) in the target project."""
        cml_storage = self.cml_storages.get(target_namespace_id)
        if cml_storage is None:
            raise ValueError(f"No CML storage registered for namespace {target_namespace_id}")

        cml = CMLRecord(
            source_memory_id=gar_result.source_record_id,
            source_namespace_id=gar_result.source_namespace_id,
            target_namespace_id=target_namespace_id,
            relationship_type=relationship_type,
            relevance_reason=relevance_reason,
            intended_usage=intended_usage,
            applicability_constraints=applicability_constraints,
            evaluation_status=CMLEvaluationStatus.DISCOVERED.value,
            is_source_available=True,
        )
        cml_storage.save_cml(cml)
        gar_result.existing_cml = cml
        return cml
