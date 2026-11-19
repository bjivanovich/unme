"""Binary header parser and serializer for the Universal Memory Record (UMR).
Complies with the 64-byte aligned specification defined in UNME Technical Contract V2.
"""

from __future__ import annotations
import struct
import time
import zlib
from dataclasses import dataclass

from unme.core.constants import (
    UMR_MAGIC_BYTES,
    UMR_FORMAT_VERSION,
    UMR_HEADER_SIZE,
    FLAG_IMMUTABLE,
    DTYPE_FLOAT16,
    COMP_NONE,
)

HEADER_STRUCT_FMT = "<4sHHIIQQQQHHBBHII"
HEADER_STRUCT_FMT_NO_CRC = "<4sHHIIQQQQHHBBHI"


class UMRHeaderError(Exception):
    """Raised when header serialization or validation fails."""
    pass


@dataclass
class UMRHeader:
    """In-memory representation of the 64-byte UMR binary header."""
    magic: bytes = UMR_MAGIC_BYTES
    version: int = UMR_FORMAT_VERSION
    flags: int = FLAG_IMMUTABLE
    segment_id: int = 0
    record_index: int = 0
    namespace_id: int = 0
    timestamp_ms: int = 0
    offset_k: int = 0
    offset_v: int = 0
    key_dim: int = 1536
    value_dim: int = 1536
    dtype: int = DTYPE_FLOAT16
    comp_type: int = COMP_NONE
    reserved: int = 0
    payload_size: int = 0
    checksum: int = 0

    def __post_init__(self) -> None:
        if self.timestamp_ms == 0:
            self.timestamp_ms = int(time.time() * 1000)

    def pack(self) -> bytes:
        """Serializes the header into 64 bytes with calculated CRC32 checksum."""
        pre_crc_bytes = struct.pack(
            HEADER_STRUCT_FMT_NO_CRC,
            self.magic,
            self.version,
            self.flags,
            self.segment_id,
            self.record_index,
            self.namespace_id,
            self.timestamp_ms,
            self.offset_k,
            self.offset_v,
            self.key_dim,
            self.value_dim,
            self.dtype,
            self.comp_type,
            self.reserved,
            self.payload_size,
        )
        calculated_crc = zlib.crc32(pre_crc_bytes) & 0xFFFFFFFF
        self.checksum = calculated_crc
        return pre_crc_bytes + struct.pack("<I", calculated_crc)

    @classmethod
    def unpack(cls, data: bytes) -> UMRHeader:
        """Parses a 64-byte buffer into a UMRHeader instance and verifies CRC32."""
        if len(data) != UMR_HEADER_SIZE:
            raise UMRHeaderError(
                f"Expected exactly {UMR_HEADER_SIZE} bytes, received {len(data)} bytes."
            )

        fields = struct.unpack(HEADER_STRUCT_FMT, data)
        magic = fields[0]
        if magic != UMR_MAGIC_BYTES:
            raise UMRHeaderError(
                f"Invalid magic bytes: expected {UMR_MAGIC_BYTES!r}, got {magic!r}."
            )

        version = fields[1]
        if version != UMR_FORMAT_VERSION:
            raise UMRHeaderError(
                f"Unsupported format version: {version} (expected {UMR_FORMAT_VERSION})."
            )

        expected_crc = fields[15]
        calculated_crc = zlib.crc32(data[:60]) & 0xFFFFFFFF
        if expected_crc != calculated_crc:
            raise UMRHeaderError(
                f"Header CRC32 mismatch: stored={expected_crc:#010x}, calculated={calculated_crc:#010x}."
            )

        return cls(
            magic=fields[0],
            version=fields[1],
            flags=fields[2],
            segment_id=fields[3],
            record_index=fields[4],
            namespace_id=fields[5],
            timestamp_ms=fields[6],
            offset_k=fields[7],
            offset_v=fields[8],
            key_dim=fields[9],
            value_dim=fields[10],
            dtype=fields[11],
            comp_type=fields[12],
            reserved=fields[13],
            payload_size=fields[14],
            checksum=fields[15],
        )
