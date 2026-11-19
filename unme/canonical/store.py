"""Canonical Memory Record dataclass and NVMe-backed canonical store for UNME Fase 3A.

Design:
- CanonicalMemoryRecord: the Level-2 representation bridging UMR (L1) and LLM states (L3).
- NVMeCanonicalStore: persists canonical tensors alongside UMR records, integrating with
  the existing SegmentWriter / SegmentReader infrastructure and NamespaceManager.
- Each encoder signature gets its own sub-index partition within the namespace.
  Vectors from distinct signatures are never compared by dot product.
- Storage format per canonical record:
    64-byte header (little-endian):
      magic        4s  "UCMR"
      version      H   1
      flags        H   0x0001 = key_stored | 0x0002 = value_stored
      namespace_id Q   uint64
      segment_id   I   uint32 (of the companion UMR segment)
      record_index I   uint32 (of the companion UMR record)
      key_dim      H   uint16
      value_dim    H   uint16
      sig_len      H   uint16 (encoder_signature UTF-8 byte length)
      threshold    f   float32 (similarity threshold assigned at creation)
      is_tombstone ?   bool as u8
      _pad         29s zero padding to reach 64 bytes
    sig_len bytes  encoder_signature (UTF-8)
    key_dim * 2 bytes   key_vector (float16, row-major)
    value_dim * 2 bytes value_vector (float16, row-major)
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from unme.canonical.encoder import ICanonicalEncoder, _l2_normalize, CANONICAL_DIM

# ---------------------------------------------------------------------------
# CMR header format
# ---------------------------------------------------------------------------
CMR_MAGIC = b"UCMR"
CMR_VERSION: int = 1
CMR_HEADER_FMT = "<4sHHQIIHHHf?29s"   # 64 bytes total
CMR_HEADER_SIZE: int = struct.calcsize(CMR_HEADER_FMT)
assert CMR_HEADER_SIZE == 64, f"CMR header size mismatch: {CMR_HEADER_SIZE}"

FLAG_KEY_STORED: int   = 0x0001
FLAG_VALUE_STORED: int = 0x0002


# ---------------------------------------------------------------------------
# CanonicalMemoryRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class CanonicalMemoryRecord:
    """Level-2 Canonical Memory Representation (CMR).

    Bridges UMR semantic storage (L1) and model-specific hidden states (L3).
    Fields are immutable after creation; rebuild from L1 to migrate encoder.
    """
    record_id: str                       # UMR record UUID (matches UMRRecord.record_id)
    namespace_id: int                    # Cryptographic namespace (uint64)
    encoder_signature: str               # Strict version tag — mixing prohibited
    key_dim: int                         # d_c (e.g. 768)
    value_dim: int                       # d_v (e.g. 768)
    key_vector: np.ndarray               # float16, L2-norm = 1.0, shape (key_dim,)
    value_vector: np.ndarray             # float16, L2-norm = 1.0, shape (value_dim,)
    similarity_threshold: float = 0.65  # Minimum cosine similarity for retrieval
    is_tombstoned: bool = False
    # Storage coordinates (set after persisting; (0, 0) until saved)
    cmr_segment_id: int = 0
    cmr_record_index: int = 0
    # Companion UMR coordinates (the L1 record this CMR was derived from)
    umr_segment_id: int = 0
    umr_record_index: int = 0

    def __post_init__(self) -> None:
        # Enforce dtype and normalization
        self.key_vector = _l2_normalize(self.key_vector)
        self.value_vector = _l2_normalize(self.value_vector)

    @property
    def flags(self) -> int:
        return FLAG_KEY_STORED | FLAG_VALUE_STORED


# ---------------------------------------------------------------------------
# Abstract store contract
# ---------------------------------------------------------------------------

class ICanonicalStore(ABC):
    """Contract for canonical record persistence and retrieval."""

    @abstractmethod
    def save_canonical_record(
        self,
        record: CanonicalMemoryRecord,
    ) -> tuple[int, int]:
        """Persists a CMR record.  Returns (cmr_segment_id, cmr_record_index)."""
        ...

    @abstractmethod
    def get_canonical_record(
        self,
        cmr_segment_id: int,
        cmr_record_index: int,
    ) -> CanonicalMemoryRecord:
        """Retrieves a CMR record by its storage coordinates.
        Raises ValueError on header magic / version mismatch.
        Raises LookupError if coordinates are out of range.
        """
        ...


# ---------------------------------------------------------------------------
# NVMe-backed canonical store
# ---------------------------------------------------------------------------

class NVMeCanonicalStore(ICanonicalStore):
    """Persistent canonical record store backed by NVMe flat files.

    One store instance per (namespace, encoder_signature).  Records are
    appended to a single file <store_dir>/cmr_<sig_hash>_<seg_id>.cmrd.
    The file grows monotonically; records are never modified in place.

    Integration with NamespaceManager:
    - Encryption is intentionally NOT applied at this layer for Fase 3A.
      Canonical vectors are derived representations that can be regenerated
      from L1 at any time; L1 is already encrypted by SegmentWriter.
      Encrypting regenerable derived data would add overhead without security
      benefit.  This decision is documented here and can be revisited.
    - If the security model requires encrypting L2, wrap the payload in
      NamespaceManager.encrypt_data() before writing.

    Thread safety: single-writer, multiple-reader (append-only design).
    Concurrent writers require external locking or separate store instances.
    """

    def __init__(
        self,
        store_dir: str | Path,
        namespace_id: int,
        encoder: ICanonicalEncoder,
        segment_id: int = 0,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._namespace_id = namespace_id
        self._encoder = encoder
        self._segment_id = segment_id
        self._sig = encoder.get_signature()
        self._sig_hash = self._make_sig_hash(self._sig)
        self._data_path = (
            self._store_dir / f"cmr_{self._sig_hash}_{segment_id:06d}.cmrd"
        )
        self._records: list[int] = []   # byte offsets of each record start
        self._scan_existing()

    @staticmethod
    def _make_sig_hash(sig: str) -> str:
        """Derives a short filesystem-safe hash from the encoder signature."""
        import hashlib
        return hashlib.sha256(sig.encode()).hexdigest()[:12]

    def _scan_existing(self) -> None:
        """Scans the data file to build an in-memory offset table."""
        self._records.clear()
        if not self._data_path.exists():
            return
        with open(self._data_path, "rb") as fh:
            while True:
                offset = fh.tell()
                raw_header = fh.read(CMR_HEADER_SIZE)
                if len(raw_header) < CMR_HEADER_SIZE:
                    break
                fields = struct.unpack(CMR_HEADER_FMT, raw_header)
                magic = fields[0]
                key_dim = fields[6]
                val_dim = fields[7]
                sig_len = fields[8]
                if magic != CMR_MAGIC:
                    break   # Corrupted or truncated tail — stop scanning
                # Skip signature + vectors
                skip = sig_len + key_dim * 2 + val_dim * 2
                fh.seek(skip, 1)
                self._records.append(offset)

    def save_canonical_record(
        self,
        record: CanonicalMemoryRecord,
    ) -> tuple[int, int]:
        """Appends a CMR record to the store file.

        Returns (segment_id, record_index) which are the CMR storage coordinates.
        Updates record.cmr_segment_id and record.cmr_record_index in place.
        """
        cmr_record_index = len(self._records)
        sig_bytes = self._sig.encode("utf-8")
        key_bytes = record.key_vector.astype(np.float16).tobytes()
        val_bytes = record.value_vector.astype(np.float16).tobytes()

        header = struct.pack(
            CMR_HEADER_FMT,
            CMR_MAGIC,
            CMR_VERSION,
            record.flags,
            self._namespace_id,
            record.umr_segment_id,
            record.umr_record_index,
            record.key_dim,
            record.value_dim,
            len(sig_bytes),
            record.similarity_threshold,
            record.is_tombstoned,
            b"\x00" * 29,
        )
        assert len(header) == CMR_HEADER_SIZE

        with open(self._data_path, "ab") as fh:
            offset = fh.seek(0, 2)  # Current EOF
            self._records.append(offset)
            fh.write(header + sig_bytes + key_bytes + val_bytes)
            fh.flush()

        record.cmr_segment_id = self._segment_id
        record.cmr_record_index = cmr_record_index
        return self._segment_id, cmr_record_index

    def get_canonical_record(
        self,
        cmr_segment_id: int,
        cmr_record_index: int,
    ) -> CanonicalMemoryRecord:
        """Reads a CMR record by its storage index."""
        if cmr_record_index >= len(self._records):
            raise LookupError(
                f"CMR record index {cmr_record_index} out of range "
                f"(store has {len(self._records)} records)"
            )
        offset = self._records[cmr_record_index]
        with open(self._data_path, "rb") as fh:
            fh.seek(offset)
            raw_header = fh.read(CMR_HEADER_SIZE)
            if len(raw_header) < CMR_HEADER_SIZE:
                raise ValueError(f"Truncated CMR header at offset {offset}")

            fields = struct.unpack(CMR_HEADER_FMT, raw_header)
            magic, version, flags, ns_id, seg_id, rec_idx, key_dim, val_dim, sig_len, threshold, tombstone, _pad = fields

            if magic != CMR_MAGIC:
                raise ValueError(f"CMR magic mismatch at offset {offset}: {magic!r}")
            if version != CMR_VERSION:
                raise ValueError(f"Unsupported CMR version {version} at offset {offset}")

            sig_bytes = fh.read(sig_len)
            sig = sig_bytes.decode("utf-8")

            key_bytes = fh.read(key_dim * 2)
            val_bytes = fh.read(val_dim * 2)

        key_vec = np.frombuffer(key_bytes, dtype=np.float16).copy()
        val_vec = np.frombuffer(val_bytes, dtype=np.float16).copy()

        return CanonicalMemoryRecord(
            record_id=f"CMR-{ns_id:016x}-{cmr_record_index:06d}",
            namespace_id=ns_id,
            encoder_signature=sig,
            key_dim=key_dim,
            value_dim=val_dim,
            key_vector=key_vec,
            value_vector=val_vec,
            similarity_threshold=threshold,
            is_tombstoned=bool(tombstone),
            cmr_segment_id=cmr_segment_id,
            cmr_record_index=cmr_record_index,
            umr_segment_id=seg_id,
            umr_record_index=rec_idx,
        )

    @property
    def record_count(self) -> int:
        return len(self._records)

    def build_index_partition(self) -> "CanonicalIndexPartition":
        """Builds an in-memory cosine index from all stored CMR records.

        Returns a CanonicalIndexPartition ready for fast retrieval.
        """
        part = CanonicalIndexPartition(
            namespace_id=self._namespace_id,
            encoder_signature=self._sig,
            dim=self._encoder.get_dim(),
        )
        for idx in range(len(self._records)):
            rec = self.get_canonical_record(self._segment_id, idx)
            if not rec.is_tombstoned:
                part.add(
                    cmr_segment_id=self._segment_id,
                    cmr_record_index=idx,
                    umr_segment_id=rec.umr_segment_id,
                    umr_record_index=rec.umr_record_index,
                    key_vector=rec.key_vector,
                )
        return part


# ---------------------------------------------------------------------------
# In-memory canonical index partition (per namespace, per encoder_signature)
# ---------------------------------------------------------------------------

@dataclass
class CMRSearchResult:
    """Result of a canonical similarity search."""
    cmr_segment_id: int
    cmr_record_index: int
    umr_segment_id: int
    umr_record_index: int
    similarity: float
    namespace_id: int
    encoder_signature: str


class CanonicalIndexPartition:
    """Fast cosine similarity index for a single (namespace, encoder_signature) pair.

    Uses numpy-based brute-force matrix multiplication (vectorized dot product).
    Suitable for collections < 500K records with the available hardware (Xeon AVX2).
    For larger collections, replace with FAISS when available.
    """

    def __init__(
        self,
        namespace_id: int,
        encoder_signature: str,
        dim: int = CANONICAL_DIM,
    ) -> None:
        self.namespace_id = namespace_id
        self.encoder_signature = encoder_signature
        self.dim = dim
        self._keys: list[np.ndarray] = []             # list of float32 unit vectors
        self._meta: list[tuple[int, int, int, int]] = []  # (cmr_seg, cmr_rec, umr_seg, umr_rec)
        self._matrix: np.ndarray | None = None        # cached (N, dim) float32

    def add(
        self,
        cmr_segment_id: int,
        cmr_record_index: int,
        umr_segment_id: int,
        umr_record_index: int,
        key_vector: np.ndarray,
    ) -> None:
        """Inserts a record into the index."""
        vec = key_vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 1e-8 else np.zeros_like(vec)
        self._keys.append(vec)
        self._meta.append((cmr_segment_id, cmr_record_index, umr_segment_id, umr_record_index))
        self._matrix = None   # Invalidate cache

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None and self._keys:
            self._matrix = np.vstack(self._keys).astype(np.float32)  # (N, dim)
        return self._matrix if self._matrix is not None else np.empty((0, self.dim), dtype=np.float32)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.65,
    ) -> list[CMRSearchResult]:
        """Returns top_k results with cosine similarity >= threshold."""
        matrix = self._ensure_matrix()
        if matrix.shape[0] == 0:
            return []

        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q = q / q_norm

        similarities = matrix @ q   # (N,)
        valid = np.where(similarities >= threshold)[0]
        if len(valid) == 0:
            return []

        valid_sims = similarities[valid]
        k = min(top_k, len(valid))
        if len(valid) > k:
            top_idx = np.argpartition(valid_sims, -k)[-k:]
            sorted_idx = valid[top_idx[np.argsort(-valid_sims[top_idx])]]
        else:
            sorted_idx = valid[np.argsort(-valid_sims)]

        results = []
        for idx in sorted_idx:
            cmr_seg, cmr_rec, umr_seg, umr_rec = self._meta[idx]
            results.append(
                CMRSearchResult(
                    cmr_segment_id=cmr_seg,
                    cmr_record_index=cmr_rec,
                    umr_segment_id=umr_seg,
                    umr_record_index=umr_rec,
                    similarity=float(similarities[idx]),
                    namespace_id=self.namespace_id,
                    encoder_signature=self.encoder_signature,
                )
            )
        return results

    @property
    def size(self) -> int:
        return len(self._keys)
