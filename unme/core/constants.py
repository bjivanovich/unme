"""Constants and layout definitions for the Universal Neural Memory Engine (UNME).
"""

from typing import Final

# Binary header constants
UMR_MAGIC_BYTES: Final[bytes] = b"UNME"
UMR_FORMAT_VERSION: Final[int] = 2
UMR_HEADER_SIZE: Final[int] = 64

# Flag bitmasks (uint16)
FLAG_IMMUTABLE: Final[int] = 1 << 0
FLAG_ENCRYPTED: Final[int] = 1 << 1
FLAG_COMPRESSED: Final[int] = 1 << 2
FLAG_TOMBSTONE: Final[int] = 1 << 3
FLAG_ARCHIVED: Final[int] = 1 << 4
FLAG_CML_LINK: Final[int] = 1 << 5

# Canonical AAD layout: version (uint16), flags (uint16), segment_id (uint32), record_index (uint32), namespace_id (uint64)
AAD_STRUCT_FMT: Final[str] = "<HHIIQ"
AAD_SIZE_BYTES: Final[int] = 20

# Supported tensor data types (uint8)
DTYPE_FLOAT32: Final[int] = 0
DTYPE_FLOAT16: Final[int] = 1
DTYPE_BFLOAT16: Final[int] = 2
DTYPE_INT8: Final[int] = 3

# Data type sizes in bytes
DTYPE_ITEM_SIZES: Final[dict[int, int]] = {
    DTYPE_FLOAT32: 4,
    DTYPE_FLOAT16: 2,
    DTYPE_BFLOAT16: 2,
    DTYPE_INT8: 1,
}

# Compression types (uint8)
COMP_NONE: Final[int] = 0
COMP_ZSTD: Final[int] = 1
COMP_LZ4: Final[int] = 2

# Alignment constants
PAGE_ALIGNMENT_BYTES: Final[int] = 4096  # 4 KB NVMe block alignment
TENSOR_ALIGNMENT_BYTES: Final[int] = 64   # 64-byte cache line alignment
