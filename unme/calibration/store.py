"""CalibrationStore: NVMe-backed append-only writer for .umrc files.

Each store instance manages a single .umrc file.  Records are written
sequentially with no internal index (the training loop scans the file
linearly).  Concurrent readers can safely read up to the last completed
record by scanning for valid UMRC magic + CRC.

Thread safety: a single writer with concurrent readers is safe because:
- Python writes to files are atomic at the OS level for sizes < page_size,
  and the header is always written as a single struct.pack call.
- Each record is written as a single bytes object: header + payload.
- A partial write (interrupted mid-record) will produce a CRC mismatch
  that CalibrationRecord.from_bytes() will detect and report.

Design constraints:
- Append-only: no seek, no overwrite, no delete.
- Maximum file size: 4 GB (uint32 offset limit for simple 32-bit indexing).
  In practice, rotate before 2 GB to stay comfortably within mmap bounds.
- No encryption at this layer.  CalibrationStore writes plaintext float16.
  Add encryption at the filesystem or OS level if calibration data is sensitive.
"""

from __future__ import annotations

import os
import pathlib
import struct
import zlib
from typing import Iterator

from unme.calibration.record import (
    CalibrationRecord,
    UMRC_MAGIC,
    UMRC_HEADER_FMT,
    UMRC_HEADER_SIZE,
)


class CalibrationStore:
    """NVMe-backed append-only store for CalibrationRecord objects.

    Usage (writer):
        store = CalibrationStore(path)
        with store:
            store.append(record)

    Usage (reader — scan all records):
        for record in CalibrationStore.scan(path):
            process(record)
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._record_count: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CalibrationStore":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Opens the store file in append-binary mode."""
        if self._fh is not None:
            return
        self._fh = open(self._path, "ab", buffering=0)   # unbuffered append
        # Count existing records so record_count is accurate across sessions
        self._record_count = sum(1 for _ in CalibrationStore.scan(self._path))

    def close(self) -> None:
        """Flushes and closes the store file."""
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            finally:
                self._fh.close()
                self._fh = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, record: CalibrationRecord) -> int:
        """Serializes and appends one record.  Returns the record index."""
        if self._fh is None:
            raise RuntimeError("CalibrationStore is not open.  Use as context manager.")
        raw = record.to_bytes()
        self._fh.write(raw)
        self._record_count += 1
        return self._record_count - 1

    # ------------------------------------------------------------------
    # Read (class method so callers do not need a writer instance)
    # ------------------------------------------------------------------

    @classmethod
    def scan(cls, path: str | pathlib.Path) -> Iterator[CalibrationRecord]:
        """Yields all valid CalibrationRecord objects from a .umrc file.

        Skips any corrupt or partial records (logs the offset to stderr).
        Safe to call on a file being written by a concurrent writer.
        """
        path = pathlib.Path(path)
        if not path.exists():
            return

        with open(path, "rb") as fh:
            file_bytes = fh.read()

        offset = 0
        total = len(file_bytes)
        while offset < total:
            if total - offset < UMRC_HEADER_SIZE:
                break

            magic = file_bytes[offset: offset + 4]
            if magic != UMRC_MAGIC:
                # Not a record boundary — file is corrupt past this point
                break

            # Parse header to determine record payload length
            try:
                (
                    _magic, version, record_type, session_id, record_index,
                    timestamp_ns, d_model, d_c, seq_len_q, seq_len_inj,
                    num_tokens, has_gradient, model_sig_len, loss_val, crc,
                    _pad,
                ) = struct.unpack_from(UMRC_HEADER_FMT, file_bytes, offset)
            except struct.error:
                break

            payload_len = (
                model_sig_len
                + seq_len_q  * d_model * 2    # h_Lq  float16
                + seq_len_inj * d_model * 2   # h_Linj float16
                + num_tokens * 4              # token_ids int32
                + (seq_len_inj * d_model * 2 if has_gradient else 0)  # g_inj
            )
            record_end = offset + UMRC_HEADER_SIZE + payload_len
            if record_end > total:
                # Partial write — stop here, do not yield
                break

            record_bytes = file_bytes[offset: record_end]
            try:
                record = CalibrationRecord.from_bytes(record_bytes)
                yield record
            except (ValueError, AssertionError) as exc:
                import sys
                print(
                    f"CalibrationStore.scan: skipping corrupt record at "
                    f"offset {offset}: {exc}",
                    file=sys.stderr,
                )

            offset = record_end

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def record_count(self) -> int:
        """Number of records written in this session (approximate if opened on existing file)."""
        return self._record_count

    @property
    def file_size_bytes(self) -> int:
        """Current file size in bytes."""
        if self._path.exists():
            return self._path.stat().st_size
        return 0
