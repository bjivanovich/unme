"""Memory lifecycle management, tombstone handling, and non-blocking deletion for UNME.
Implements the 4 distinct lifecycle operations:
1. DESINDEXAR: Removes vector embeddings from the active search index without altering disk.
2. ARCHIVAR: Marks record as archived (FLAG_ARCHIVED); retained for audit but excluded from active search.
3. ELIMINAR (Tombstone): Logically deletes origin content, creates immutable tombstone, converts CMLs to historical references.
4. PURGAR: Physically zeroes out data or compacts segment files, supporting privacy requests.
"""

from __future__ import annotations
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from unme.core.constants import (
    FLAG_IMMUTABLE,
    FLAG_TOMBSTONE,
    FLAG_ARCHIVED,
    UMR_HEADER_SIZE,
)
from unme.core.cml import CMLStorage, CMLEvaluationStatus
from unme.core.header import UMRHeader
from unme.index.vector_index import PartitionedVectorIndex


@dataclass
class TombstoneRecord:
    """Minimal non-sensitive record indicating a memory previously existed and was deleted."""
    record_id: str
    namespace_id: int
    deletion_timestamp_ms: int
    reason: str = "Ordinary cleanup"
    privacy_purge: bool = False


class MemoryLifecycleManager:
    """Coordinates lifecycle transitions, tombstone propagation to CMLs,
    and non-blocking deletion across independent project namespaces.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self._tombstones: dict[tuple[int, str], TombstoneRecord] = {}

    def desindexar(
        self,
        vector_index: PartitionedVectorIndex,
        namespace_id: int,
        record_index: int,
    ) -> None:
        """Removes the record from the active associative index.
        The underlying physical record remains on disk.
        """
        if namespace_id in vector_index._partitions:
            partition = vector_index._partitions[namespace_id]
            # Mark the entry in the partition as inactive / zeroed
            for i, (seg_id, rec_idx) in enumerate(partition._record_map):
                if rec_idx == record_index:
                    partition._keys_list[i] = np.zeros(partition.dim, dtype=np.float32)
                    partition._keys_matrix = None
                    break
            vector_index.clear_cache(namespace_id)

    def archivar(
        self,
        segment_dir: Path | str,
        segment_id: int,
        record_index: int,
        vector_index: PartitionedVectorIndex | None = None,
        namespace_id: int | None = None,
    ) -> None:
        """Marks the record header as ARCHIVED on disk and desindexes it from active search."""
        header_path = Path(segment_dir) / f"segment_{segment_id:06d}.umrh"
        if not header_path.exists():
            raise FileNotFoundError(f"Segment header file not found: {header_path}")

        offset = record_index * UMR_HEADER_SIZE
        with open(header_path, "r+b") as hf:
            hf.seek(offset)
            h_bytes = hf.read(UMR_HEADER_SIZE)
            header = UMRHeader.unpack(h_bytes)
            header.flags |= FLAG_ARCHIVED
            hf.seek(offset)
            hf.write(header.pack())
            hf.flush()
            os.fsync(hf.fileno())

        if vector_index is not None and namespace_id is not None:
            self.desindexar(vector_index, namespace_id, record_index)

    def eliminar_tombstone(
        self,
        namespace_id: int,
        record_id: str,
        reason: str = "Deleted by user",
        cml_storages: list[CMLStorage] | None = None,
        vector_index: PartitionedVectorIndex | None = None,
        record_index: int | None = None,
    ) -> TombstoneRecord:
        """Creates an immutable tombstone record.
        Transitions any linked CMLs to SOURCE_BECAME_UNAVAILABLE.
        CRITICAL: Never deletes derived memories created in target projects.
        """
        now_ms = int(time.time() * 1000)
        tombstone = TombstoneRecord(
            record_id=record_id,
            namespace_id=namespace_id,
            deletion_timestamp_ms=now_ms,
            reason=reason,
            privacy_purge=False,
        )
        self._tombstones[(namespace_id, record_id)] = tombstone

        # Desindex if index provided
        if vector_index is not None and record_index is not None:
            self.desindexar(vector_index, namespace_id, record_index)

        # Propagate unavailability to connected CMLs
        if cml_storages:
            for storage in cml_storages:
                matched_links = storage.find_by_source(record_id)
                for link in matched_links:
                    link.mark_source_unavailable(reason=f"Tombstone in origin namespace: {reason}")
                    storage.save_cml(link)

        return tombstone

    def tombstone_record(
        self,
        namespace_id: int,
        record_id: str,
        reason: str = "Deleted by user",
        cml_storages: list[CMLStorage] | None = None,
        vector_index: PartitionedVectorIndex | None = None,
        record_index: int | None = None,
    ) -> TombstoneRecord:
        """Alias for eliminar_tombstone."""
        return self.eliminar_tombstone(
            namespace_id=namespace_id,
            record_id=record_id,
            reason=reason,
            cml_storages=cml_storages,
            vector_index=vector_index,
            record_index=record_index,
        )

    def purgar(
        self,
        segment_dir: Path | str,
        segment_id: int,
        record_index: int,
        namespace_id: int,
        record_id: str,
        cml_storages: list[CMLStorage] | None = None,
        vector_index: PartitionedVectorIndex | None = None,
    ) -> None:
        """Physical purge for privacy requests:
        1. Overwrites payload and value tensor on disk with zeros.
        2. Sets FLAG_TOMBSTONE in header.
        3. Converts CMLs to historical references without retaining original content.
        4. Clears vector index and query cache.
        """
        header_path = Path(segment_dir) / f"segment_{segment_id:06d}.umrh"
        tensor_path = Path(segment_dir) / f"segment_{segment_id:06d}.umrt"

        if header_path.exists() and tensor_path.exists():
            offset = record_index * UMR_HEADER_SIZE
            with open(header_path, "r+b") as hf:
                hf.seek(offset)
                header = UMRHeader.unpack(hf.read(UMR_HEADER_SIZE))
                header.flags |= FLAG_TOMBSTONE

                # Zero out data in tensor file
                with open(tensor_path, "r+b") as tf:
                    tf.seek(header.offset_v)
                    # Overwrite value tensor + payload with zeros
                    val_bytes = header.value_dim * 2  # assuming fp16
                    zero_len = val_bytes + header.payload_size
                    tf.write(b"\x00" * zero_len)
                    tf.flush()
                    os.fsync(tf.fileno())

                header.payload_size = 0
                hf.seek(offset)
                hf.write(header.pack())
                hf.flush()
                os.fsync(hf.fileno())

        # Register tombstone as privacy purge
        now_ms = int(time.time() * 1000)
        self._tombstones[(namespace_id, record_id)] = TombstoneRecord(
            record_id=record_id,
            namespace_id=namespace_id,
            deletion_timestamp_ms=now_ms,
            reason="Privacy Purge",
            privacy_purge=True,
        )

        if vector_index is not None:
            self.desindexar(vector_index, namespace_id, record_index)

        if cml_storages:
            for storage in cml_storages:
                matched_links = storage.find_by_source(record_id)
                for link in matched_links:
                    link.mark_source_unavailable(reason="Source purged under privacy request")
                    storage.save_cml(link)

    def is_tombstoned(self, namespace_id: int, record_id: str) -> bool:
        return (namespace_id, record_id) in self._tombstones
