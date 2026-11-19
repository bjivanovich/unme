"""CalibrationRecord: binary format for UNME activation captures (.umrc files).

Binary layout of a .umrc file
==============================

  [64-byte header]
  [model_signature  — model_sig_len bytes, UTF-8, no null terminator]
  [h_Lq             — seq_len_q  * d_model * 2 bytes, float16, row-major]
  [h_Linj           — seq_len_inj * d_model * 2 bytes, float16, row-major]
  [token_ids        — num_tokens * 4 bytes, int32]
  [g_inj            — seq_len_inj * d_model * 2 bytes, float16  (GRADIENT only)]

Header format: '<4sHHQIQHHHHI?HfI13s'  (exactly 64 bytes, little-endian)
  +00  magic          4s    b"UMRC"
  +04  version        H     UMRC_VERSION (1)
  +06  record_type    H     RecordType enum: 1=FORWARD_CAPTURE, 2=GRADIENT_CALIBRATION
  +08  session_id     Q     uint64 session identifier (random, unique per capture session)
  +16  record_index   I     uint32 monotonic index within the session
  +20  timestamp_ns   Q     uint64 POSIX nanoseconds
  +28  d_model        H     uint16 LLM hidden dimension (e.g. 4096 for Gyro-Coder)
  +30  d_c            H     uint16 canonical dimension (768)
  +32  seq_len_q      H     uint16 sequence length captured at L_q
  +34  seq_len_inj    H     uint16 sequence length captured at L_inj
  +36  num_tokens     I     uint32 number of input token ids
  +40  has_gradient   ?     bool   True only for GRADIENT_CALIBRATION records
  +41  model_sig_len  H     uint16 length of model_signature string
  +43  loss_val       f     float32 task loss value (0.0 for FORWARD_CAPTURE)
  +47  crc32          I     uint32 CRC32 of everything after the header
  +51  pad            13s   zero-padded

Notes:
- CRC32 covers: model_signature bytes + h_Lq + h_Linj + token_ids (+ g_inj if present).
- The CRC in the header is computed AFTER all payload bytes are assembled.
- Both h_Lq and h_Linj store the FULL sequence (all token positions) so the
  training loop can apply any positional masking strategy it chooses.
- token_ids are stored int32 (not int64) to halve storage; vocabulary sizes
  up to 2^31 are covered (Gyro-Coder vocab: 151,936).
- Gradient g_inj = nabla_{h_Linj} L_task.  Stored only when has_gradient=True.
  Shape matches h_Linj exactly.
"""

from __future__ import annotations

import enum
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

UMRC_MAGIC: bytes = b"UMRC"
UMRC_VERSION: int = 1
UMRC_HEADER_FMT: str = "<4sHHQIQHHHHI?HfI13s"
UMRC_HEADER_SIZE: int = struct.calcsize(UMRC_HEADER_FMT)

assert UMRC_HEADER_SIZE == 64, (
    f"UMRC header must be exactly 64 bytes, got {UMRC_HEADER_SIZE}"
)


class RecordType(enum.IntEnum):
    FORWARD_CAPTURE = 1
    GRADIENT_CALIBRATION = 2


# ---------------------------------------------------------------------------
# Dataclass definitions
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    """Single forward-pass calibration sample.

    h_Lq   : float16 array (seq_len_q,  d_model)  — activations at query layer
    h_Linj : float16 array (seq_len_inj, d_model)  — activations at injection layer
    token_ids      : int32 array (num_tokens,)
    model_signature: str identifying the LLM and injection layers
    session_id     : uint64 unique per capture session
    record_index   : uint32 monotonic within session
    timestamp_ns   : uint64 POSIX nanoseconds at capture time
    """

    h_Lq: np.ndarray            # shape (seq_len_q,  d_model), dtype float16
    h_Linj: np.ndarray          # shape (seq_len_inj, d_model), dtype float16
    token_ids: np.ndarray       # shape (num_tokens,),          dtype int32
    model_signature: str
    session_id: int
    record_index: int
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())

    @property
    def record_type(self) -> RecordType:
        return RecordType.FORWARD_CAPTURE

    @property
    def d_model(self) -> int:
        return self.h_Lq.shape[1]

    @property
    def seq_len_q(self) -> int:
        return self.h_Lq.shape[0]

    @property
    def seq_len_inj(self) -> int:
        return self.h_Linj.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.token_ids.shape[0]

    def _validate_shapes(self) -> None:
        assert self.h_Lq.ndim == 2, "h_Lq must be 2D (seq_len_q, d_model)"
        assert self.h_Linj.ndim == 2, "h_Linj must be 2D (seq_len_inj, d_model)"
        assert self.h_Lq.shape[1] == self.h_Linj.shape[1], (
            "h_Lq and h_Linj must have the same d_model"
        )
        assert self.token_ids.ndim == 1, "token_ids must be 1D"
        assert self.h_Lq.dtype == np.float16, f"h_Lq dtype must be float16, got {self.h_Lq.dtype}"
        assert self.h_Linj.dtype == np.float16, f"h_Linj dtype must be float16, got {self.h_Linj.dtype}"
        assert self.token_ids.dtype == np.int32, f"token_ids dtype must be int32, got {self.token_ids.dtype}"

    def to_bytes(self) -> bytes:
        """Serializes this record to the UMRC binary format."""
        self._validate_shapes()

        sig_bytes = self.model_signature.encode("utf-8")
        h_lq_bytes = self.h_Lq.astype(np.float16).tobytes()
        h_linj_bytes = self.h_Linj.astype(np.float16).tobytes()
        tok_bytes = self.token_ids.astype(np.int32).tobytes()
        grad_bytes = self._gradient_bytes()

        payload = sig_bytes + h_lq_bytes + h_linj_bytes + tok_bytes + grad_bytes
        crc = zlib.crc32(payload) & 0xFFFF_FFFF

        header = struct.pack(
            UMRC_HEADER_FMT,
            UMRC_MAGIC,
            UMRC_VERSION,
            int(self.record_type),
            self.session_id & 0xFFFF_FFFF_FFFF_FFFF,
            self.record_index & 0xFFFF_FFFF,
            self.timestamp_ns & 0xFFFF_FFFF_FFFF_FFFF,
            self.d_model,
            768,                    # d_c — CANONICAL_DIM, constant for UNME v1
            self.seq_len_q,
            self.seq_len_inj,
            self.num_tokens,
            self._has_gradient(),
            len(sig_bytes),
            self._loss_val(),
            crc,
            b"\x00" * 13,          # pad
        )
        return header + payload

    # Overrideable hooks for subclass fields
    def _gradient_bytes(self) -> bytes:
        return b""

    def _has_gradient(self) -> bool:
        return False

    def _loss_val(self) -> float:
        return 0.0

    @classmethod
    def from_bytes(cls, data: bytes) -> "CalibrationRecord":
        """Deserializes a UMRC binary blob.  Returns the appropriate subclass."""
        if len(data) < UMRC_HEADER_SIZE:
            raise ValueError(
                f"Data too short: {len(data)} < {UMRC_HEADER_SIZE}"
            )
        (
            magic, version, record_type, session_id, record_index,
            timestamp_ns, d_model, d_c, seq_len_q, seq_len_inj,
            num_tokens, has_gradient, model_sig_len, loss_val, crc,
            _pad,
        ) = struct.unpack_from(UMRC_HEADER_FMT, data, 0)

        if magic != UMRC_MAGIC:
            raise ValueError(f"Invalid UMRC magic: {magic!r}")
        if version != UMRC_VERSION:
            raise ValueError(f"Unsupported UMRC version: {version}")

        offset = UMRC_HEADER_SIZE
        sig_bytes = data[offset: offset + model_sig_len]
        model_signature = sig_bytes.decode("utf-8")
        offset += model_sig_len

        h_lq_bytes = d_model * seq_len_q * 2         # float16 = 2 bytes
        h_linj_bytes = d_model * seq_len_inj * 2
        tok_bytes = num_tokens * 4                    # int32 = 4 bytes

        h_Lq = np.frombuffer(
            data[offset: offset + h_lq_bytes], dtype=np.float16
        ).reshape(seq_len_q, d_model).copy()
        offset += h_lq_bytes

        h_Linj = np.frombuffer(
            data[offset: offset + h_linj_bytes], dtype=np.float16
        ).reshape(seq_len_inj, d_model).copy()
        offset += h_linj_bytes

        token_ids = np.frombuffer(
            data[offset: offset + tok_bytes], dtype=np.int32
        ).copy()
        offset += tok_bytes

        # Verify CRC over payload (after header)
        payload_start = UMRC_HEADER_SIZE
        payload_so_far = data[payload_start: offset]

        g_inj: Optional[np.ndarray] = None
        if has_gradient:
            g_bytes = d_model * seq_len_inj * 2
            g_inj = np.frombuffer(
                data[offset: offset + g_bytes], dtype=np.float16
            ).reshape(seq_len_inj, d_model).copy()
            offset += g_bytes

        full_payload = data[payload_start: offset]
        actual_crc = zlib.crc32(full_payload) & 0xFFFF_FFFF
        if actual_crc != crc:
            raise ValueError(
                f"UMRC CRC32 mismatch: stored={crc:#010x} computed={actual_crc:#010x}"
            )

        if has_gradient and g_inj is not None:
            return GradientCalibrationRecord(
                h_Lq=h_Lq,
                h_Linj=h_Linj,
                token_ids=token_ids,
                model_signature=model_signature,
                session_id=session_id,
                record_index=record_index,
                timestamp_ns=timestamp_ns,
                g_inj=g_inj,
                loss_val=loss_val,
            )
        return cls(
            h_Lq=h_Lq,
            h_Linj=h_Linj,
            token_ids=token_ids,
            model_signature=model_signature,
            session_id=session_id,
            record_index=record_index,
            timestamp_ns=timestamp_ns,
        )


@dataclass
class GradientCalibrationRecord(CalibrationRecord):
    """CalibrationRecord extended with the task-loss gradient at L_inj.

    g_inj : float16 array (seq_len_inj, d_model)
        Gradient of the task loss with respect to h_Linj:
            g_inj = nabla_{h_Linj} L_task
        Used to compute the surrogate training loss:
            L_surrogate = <g_inj, Delta_h> + lambda * R(Delta_h)
        where Delta_h = adapter_output - h_Linj_original.

    loss_val : float32
        Scalar task loss at the time of capture.
    """

    g_inj: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float16))
    loss_val: float = 0.0

    @property
    def record_type(self) -> RecordType:
        return RecordType.GRADIENT_CALIBRATION

    def _validate_shapes(self) -> None:
        super()._validate_shapes()
        assert self.g_inj.ndim == 2, "g_inj must be 2D (seq_len_inj, d_model)"
        assert self.g_inj.shape == self.h_Linj.shape, (
            f"g_inj shape {self.g_inj.shape} must match h_Linj shape {self.h_Linj.shape}"
        )
        assert self.g_inj.dtype == np.float16, (
            f"g_inj dtype must be float16, got {self.g_inj.dtype}"
        )

    def _gradient_bytes(self) -> bytes:
        return self.g_inj.astype(np.float16).tobytes()

    def _has_gradient(self) -> bool:
        return True

    def _loss_val(self) -> float:
        return float(self.loss_val)
