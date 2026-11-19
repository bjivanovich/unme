"""Append-only persistent storage engine using memory-mapped I/O (mmap).
Implements the Universal Memory Record (UMR) binary layout defined in UNME Technical Contract V2.
Features:
- Defense-in-depth AEAD encryption (ChaCha20-Poly1305 + Canonical AAD + NonceEpochManager).
- Automated segment rotation (configurable by record count and byte size).
- Multiprocess concurrency synchronization with file locking.
- Multi-segment unified reader.
- Strict checksum and crash recovery.
"""

from __future__ import annotations
import mmap
import msvcrt
import os
import struct
import zlib
from pathlib import Path
from typing import Any
import numpy as np

from unme.core.constants import (
    UMR_HEADER_SIZE,
    PAGE_ALIGNMENT_BYTES,
    TENSOR_ALIGNMENT_BYTES,
    FLAG_IMMUTABLE,
    FLAG_ENCRYPTED,
    FLAG_TOMBSTONE,
    FLAG_ARCHIVED,
)
from unme.core.header import UMRHeader, UMRHeaderError
from unme.core.record import UMRRecord
from unme.crypto.namespace import NamespaceManager, NonceEpochManager, build_canonical_aad

CHECKSUM_FOOTER_FMT = "<I"
CHECKSUM_FOOTER_SIZE = struct.calcsize(CHECKSUM_FOOTER_FMT)


def align_offset(current_offset: int, alignment: int = TENSOR_ALIGNMENT_BYTES) -> int:
    """Calculates padding required to align offset to the specified byte boundary."""
    remainder = current_offset % alignment
    if remainder == 0:
        return current_offset
    return current_offset + (alignment - remainder)


class StorageCorruptionError(Exception):
    """Raised when record payload, tensor integrity, or AEAD authentication check fails."""
    pass


class SegmentRecovery:
    """Utility to inspect, verify, and repair crashed or uncommitted segment files."""

    @staticmethod
    def recover_segment(segment_dir: Path | str, segment_id: int) -> int:
        """Inspects .umrh and .umrt files, validates every record's header and data checksum,
        and truncates any uncommitted, corrupted, or partial trailing data to the last known valid record.
        Returns the number of verified valid records.
        """
        segment_dir = Path(segment_dir)
        header_path = segment_dir / f"segment_{segment_id:06d}.umrh"
        tensor_path = segment_dir / f"segment_{segment_id:06d}.umrt"

        if not header_path.exists() or not tensor_path.exists():
            return 0

        header_file_size = os.path.getsize(header_path)
        tensor_file_size = os.path.getsize(tensor_path)

        valid_count = 0
        last_valid_header_offset = 0
        last_valid_tensor_offset = 0

        with open(header_path, "r+b") as hf, open(tensor_path, "r+b") as tf:
            offset = 0
            while offset + UMR_HEADER_SIZE <= header_file_size:
                hf.seek(offset)
                h_bytes = hf.read(UMR_HEADER_SIZE)
                try:
                    header = UMRHeader.unpack(h_bytes)
                except UMRHeaderError:
                    break

                np_dtype = UMRRecord.numpy_dtype(header.dtype)
                key_byte_len = header.key_dim * np_dtype.itemsize
                val_byte_len = header.value_dim * np_dtype.itemsize
                end_of_record = header.offset_v + val_byte_len + header.payload_size + CHECKSUM_FOOTER_SIZE

                if end_of_record > tensor_file_size:
                    break

                # Verify CRC32 footer
                tf.seek(header.offset_k)
                key_bytes = tf.read(key_byte_len)
                tf.seek(header.offset_v)
                val_bytes = tf.read(val_byte_len)
                payload_bytes = tf.read(header.payload_size) if header.payload_size > 0 else b""
                footer_bytes = tf.read(CHECKSUM_FOOTER_SIZE)

                if len(footer_bytes) != CHECKSUM_FOOTER_SIZE:
                    break

                stored_crc = struct.unpack(CHECKSUM_FOOTER_FMT, footer_bytes)[0]
                computed_crc = zlib.crc32(key_bytes + val_bytes + payload_bytes) & 0xFFFFFFFF
                if stored_crc != computed_crc:
                    break

                valid_count += 1
                offset += UMR_HEADER_SIZE
                last_valid_header_offset = offset
                last_valid_tensor_offset = end_of_record

            if header_file_size > last_valid_header_offset:
                hf.seek(last_valid_header_offset)
                hf.truncate()
            if tensor_file_size > last_valid_tensor_offset:
                tf.seek(last_valid_tensor_offset)
                tf.truncate()

        return valid_count


class SegmentWriter:
    """Append-only writer for UNME storage segments.
    Supports authentic AEAD encryption at rest, persistent write epoch nonces,
    multiprocess locking, and automated segment rotation.
    """

    def __init__(
        self,
        segment_dir: Path | str,
        segment_id: int = 1,
        namespace_mgr: NamespaceManager | None = None,
        namespace_key: bytes | None = None,
        max_records_per_segment: int = 10000,
        max_segment_bytes: int = 1024 * 1024 * 1024,  # 1 GB
    ) -> None:
        self.segment_dir = Path(segment_dir)
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.segment_id = segment_id
        self.namespace_mgr = namespace_mgr or NamespaceManager()
        self.namespace_key = namespace_key
        self.max_records_per_segment = max_records_per_segment
        self.max_segment_bytes = max_segment_bytes

        self._init_segment(segment_id)

    def _init_segment(self, segment_id: int) -> None:
        self.segment_id = segment_id
        self.epoch_mgr = NonceEpochManager(self.segment_dir, segment_id)
        self.segment_key = (
            NamespaceManager.derive_segment_key(self.namespace_key, self.segment_id, self.epoch_mgr.segment_uuid)
            if self.namespace_key is not None
            else None
        )

        self.header_path = self.segment_dir / f"segment_{segment_id:06d}.umrh"
        self.tensor_path = self.segment_dir / f"segment_{segment_id:06d}.umrt"

        SegmentRecovery.recover_segment(self.segment_dir, self.segment_id)

        self._header_file = open(self.header_path, "a+b")
        self._tensor_file = open(self.tensor_path, "a+b")
        self.record_count = self._header_file.tell() // UMR_HEADER_SIZE

    def _rotate_if_needed(self) -> None:
        cur_tensor_size = self._tensor_file.tell()
        if self.record_count >= self.max_records_per_segment or cur_tensor_size >= self.max_segment_bytes:
            self.flush()
            self._header_file.close()
            self._tensor_file.close()
            next_seg_id = self.segment_id + 1
            self._init_segment(next_seg_id)

    def append(self, record: UMRRecord, encrypt: bool = False) -> tuple[int, int]:
        """Appends a new UMRRecord. Automatically rotates segment if thresholds are reached.
        Returns: (segment_id, record_index).
        """
        self._rotate_if_needed()

        header_fd = self._header_file.fileno()
        try:
            msvcrt.locking(header_fd, msvcrt.LK_LOCK, 1)
        except OSError:
            pass

        try:
            # Sync record_count with physical disk file in case of concurrent appends
            self._header_file.seek(0, os.SEEK_END)
            self._tensor_file.seek(0, os.SEEK_END)
            self.record_count = self._header_file.tell() // UMR_HEADER_SIZE

            record_idx = self.record_count
            flags = FLAG_IMMUTABLE
            if encrypt or (self.segment_key is not None):
                flags |= FLAG_ENCRYPTED

            raw_payload_bytes = record.serialize_payload()

            if flags & FLAG_ENCRYPTED:
                if self.segment_key is None:
                    raise ValueError("Cannot encrypt record without a valid namespace/segment key.")
                aad = build_canonical_aad(
                    version=record.version if hasattr(record, "version") else 2,
                    flags=flags,
                    segment_id=self.segment_id,
                    record_index=record_idx,
                    namespace_id=record.namespace_id,
                )
                nonce = self.epoch_mgr.generate_nonce(record_idx)
                payload_to_write = NamespaceManager.encrypt_payload(
                    raw_payload_bytes,
                    self.segment_key,
                    aad,
                    nonce,
                )
            else:
                payload_to_write = raw_payload_bytes

            payload_size = len(payload_to_write)

            np_dtype = UMRRecord.numpy_dtype(record.dtype)
            if record.key_vector is None or record.value_vector is None:
                raise ValueError("Record must have both key_vector and value_vector populated.")

            key_data = np.ascontiguousarray(record.key_vector, dtype=np_dtype).tobytes()
            val_data = np.ascontiguousarray(record.value_vector, dtype=np_dtype).tobytes()

            data_checksum = zlib.crc32(key_data + val_data + payload_to_write) & 0xFFFFFFFF
            checksum_bytes = struct.pack(CHECKSUM_FOOTER_FMT, data_checksum)

            cur_tensor_offset = self._tensor_file.tell()
            aligned_k_offset = align_offset(cur_tensor_offset, TENSOR_ALIGNMENT_BYTES)
            if aligned_k_offset > cur_tensor_offset:
                self._tensor_file.write(b"\x00" * (aligned_k_offset - cur_tensor_offset))

            offset_k = self._tensor_file.tell()
            self._tensor_file.write(key_data)

            cur_tensor_offset = self._tensor_file.tell()
            aligned_v_offset = align_offset(cur_tensor_offset, TENSOR_ALIGNMENT_BYTES)
            if aligned_v_offset > cur_tensor_offset:
                self._tensor_file.write(b"\x00" * (aligned_v_offset - cur_tensor_offset))

            offset_v = self._tensor_file.tell()
            self._tensor_file.write(val_data)

            if payload_size > 0:
                self._tensor_file.write(payload_to_write)
            self._tensor_file.write(checksum_bytes)

            header = UMRHeader(
                flags=flags,
                segment_id=self.segment_id,
                record_index=record_idx,
                namespace_id=record.namespace_id,
                offset_k=offset_k,
                offset_v=offset_v,
                key_dim=int(record.key_vector.shape[0]),
                value_dim=int(record.value_vector.shape[0]),
                dtype=record.dtype,
                payload_size=payload_size,
            )
            header_bytes = header.pack()
            self._header_file.write(header_bytes)

            self.record_count += 1
            return (self.segment_id, record_idx)
        finally:
            try:
                msvcrt.locking(header_fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass

    def flush(self) -> None:
        self._header_file.flush()
        self._tensor_file.flush()
        os.fsync(self._header_file.fileno())
        os.fsync(self._tensor_file.fileno())

    def close(self) -> None:
        self.flush()
        self._header_file.close()
        self._tensor_file.close()


class SegmentReader:
    """Memory-mapped reader for a single UNME storage segment."""

    def __init__(
        self,
        segment_dir: Path | str,
        segment_id: int = 1,
        namespace_key: bytes | None = None,
    ) -> None:
        self.segment_dir = Path(segment_dir)
        self.segment_id = segment_id
        self.namespace_key = namespace_key

        # Load segment UUID from epoch meta if available
        meta_path = self.segment_dir / ".meta" / f"segment_{segment_id:06d}.epoch"
        segment_uuid = None
        if meta_path.exists():
            with open(meta_path, "rb") as mf:
                segment_uuid = mf.read(16)

        self.segment_key = (
            NamespaceManager.derive_segment_key(self.namespace_key, self.segment_id, segment_uuid)
            if self.namespace_key is not None
            else None
        )

        self.header_path = self.segment_dir / f"segment_{segment_id:06d}.umrh"
        self.tensor_path = self.segment_dir / f"segment_{segment_id:06d}.umrt"

        if not self.header_path.exists():
            raise FileNotFoundError(f"Segment header file not found: {self.header_path}")
        if not self.tensor_path.exists():
            raise FileNotFoundError(f"Segment tensor file not found: {self.tensor_path}")

        self._header_file = open(self.header_path, "rb")
        self._tensor_file = open(self.tensor_path, "rb")

        self.header_size_total = os.path.getsize(self.header_path)
        self.tensor_size_total = os.path.getsize(self.tensor_path)

        self.record_count = self.header_size_total // UMR_HEADER_SIZE

        self._mmap_header = (
            mmap.mmap(self._header_file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.header_size_total > 0 else None
        )
        self._mmap_tensor = (
            mmap.mmap(self._tensor_file.fileno(), 0, access=mmap.ACCESS_READ)
            if self.tensor_size_total > 0 else None
        )

    def get_header(self, record_index: int) -> UMRHeader:
        if record_index < 0 or record_index >= self.record_count:
            raise IndexError(f"Record index out of range: {record_index} (count: {self.record_count})")

        start = record_index * UMR_HEADER_SIZE
        end = start + UMR_HEADER_SIZE
        assert self._mmap_header is not None
        header_data = self._mmap_header[start:end]
        return UMRHeader.unpack(header_data)

    def get_key_vector(self, header: UMRHeader) -> np.ndarray:
        assert self._mmap_tensor is not None
        np_dtype = UMRRecord.numpy_dtype(header.dtype)
        byte_len = header.key_dim * np_dtype.itemsize
        tensor_bytes = self._mmap_tensor[header.offset_k : header.offset_k + byte_len]
        return np.frombuffer(tensor_bytes, dtype=np_dtype)

    def get_value_vector(self, header: UMRHeader) -> np.ndarray:
        assert self._mmap_tensor is not None
        np_dtype = UMRRecord.numpy_dtype(header.dtype)
        byte_len = header.value_dim * np_dtype.itemsize
        tensor_bytes = self._mmap_tensor[header.offset_v : header.offset_v + byte_len]
        return np.frombuffer(tensor_bytes, dtype=np_dtype)

    def get_payload_and_verify(
        self,
        header: UMRHeader,
        verify_checksum: bool = True,
        override_key: bytes | None = None,
    ) -> dict[str, Any]:
        if self._mmap_tensor is None:
            return {}

        np_dtype = UMRRecord.numpy_dtype(header.dtype)
        val_byte_len = header.value_dim * np_dtype.itemsize
        key_byte_len = header.key_dim * np_dtype.itemsize

        payload_offset = header.offset_v + val_byte_len
        payload_bytes = (
            self._mmap_tensor[payload_offset : payload_offset + header.payload_size]
            if header.payload_size > 0 else b""
        )

        checksum_offset = payload_offset + header.payload_size
        stored_checksum_bytes = self._mmap_tensor[checksum_offset : checksum_offset + CHECKSUM_FOOTER_SIZE]

        if verify_checksum:
            stored_checksum = struct.unpack(CHECKSUM_FOOTER_FMT, stored_checksum_bytes)[0]
            key_bytes = self._mmap_tensor[header.offset_k : header.offset_k + key_byte_len]
            val_bytes = self._mmap_tensor[header.offset_v : header.offset_v + val_byte_len]
            computed_checksum = zlib.crc32(key_bytes + val_bytes + payload_bytes) & 0xFFFFFFFF
            if stored_checksum != computed_checksum:
                raise StorageCorruptionError(
                    f"Record data checksum mismatch at index {header.record_index}: "
                    f"stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
                )

        if not payload_bytes:
            return {}

        if header.flags & FLAG_ENCRYPTED:
            key_to_use = override_key or self.segment_key
            if key_to_use is None:
                raise StorageCorruptionError("Encrypted record cannot be decrypted without namespace_key.")

            aad = build_canonical_aad(
                version=header.version,
                flags=header.flags,
                segment_id=header.segment_id,
                record_index=header.record_index,
                namespace_id=header.namespace_id,
            )
            try:
                decrypted_bytes = NamespaceManager.decrypt_payload(payload_bytes, key_to_use, aad)
            except Exception as e:
                raise StorageCorruptionError(f"AEAD authentication failed (tag mismatch / wrong key / altered AAD): {e}")
            return UMRRecord.deserialize_payload(decrypted_bytes)

        return UMRRecord.deserialize_payload(payload_bytes)

    def get_payload(self, header: UMRHeader) -> dict[str, Any]:
        return self.get_payload_and_verify(header, verify_checksum=False)

    def get_record(self, record_index: int, verify_checksum: bool = True) -> UMRRecord:
        header = self.get_header(record_index)
        key_vec = self.get_key_vector(header)
        val_vec = self.get_value_vector(header)
        payload = self.get_payload_and_verify(header, verify_checksum=verify_checksum)

        return UMRRecord(
            record_id=payload.get("record_id", ""),
            namespace_id=header.namespace_id,
            canonical_text=payload.get("canonical_text", ""),
            structured_data=payload.get("structured_data", {}),
            provenance=payload.get("provenance", {}),
            graph_relations=payload.get("graph_relations", {}),
            key_vector=key_vec,
            value_vector=val_vec,
            dtype=header.dtype,
        )

    def close(self) -> None:
        if self._mmap_header is not None:
            self._mmap_header.close()
        if self._mmap_tensor is not None:
            self._mmap_tensor.close()
        self._header_file.close()
        self._tensor_file.close()


class MultiSegmentReader:
    """Unified reader capable of querying records seamlessly across rotated segments."""

    def __init__(self, segment_dir: Path | str, namespace_key: bytes | None = None) -> None:
        self.segment_dir = Path(segment_dir)
        self.namespace_key = namespace_key
        self._readers: dict[int, SegmentReader] = {}
        self._discover_segments()

    def _discover_segments(self) -> None:
        header_files = sorted(self.segment_dir.glob("segment_*.umrh"))
        for hf in header_files:
            seg_str = hf.stem.replace("segment_", "")
            try:
                seg_id = int(seg_str)
                if seg_id not in self._readers:
                    self._readers[seg_id] = SegmentReader(self.segment_dir, segment_id=seg_id, namespace_key=self.namespace_key)
            except ValueError:
                continue

    def get_record(self, segment_id: int, record_index: int, verify_checksum: bool = True) -> UMRRecord:
        self._discover_segments()
        if segment_id not in self._readers:
            raise KeyError(f"Segment ID {segment_id} not found in {self.segment_dir}")
        return self._readers[segment_id].get_record(record_index, verify_checksum=verify_checksum)

    def get_header(self, segment_id: int, record_index: int) -> UMRHeader:
        self._discover_segments()
        if segment_id not in self._readers:
            raise KeyError(f"Segment ID {segment_id} not found in {self.segment_dir}")
        return self._readers[segment_id].get_header(record_index)

    @property
    def total_records(self) -> int:
        self._discover_segments()
        return sum(r.record_count for r in self._readers.values())

    @property
    def segment_ids(self) -> list[int]:
        self._discover_segments()
        return sorted(self._readers.keys())

    def close(self) -> None:
        for r in self._readers.values():
            r.close()
        self._readers.clear()
