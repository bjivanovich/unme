"""UNME Calibration module — Fase 3C.

Captures forward-pass activations and optional task-loss gradients from a
frozen LLM for use in adapter training.  All data is persisted to NVMe as
binary .umrc files, independent of any specific model runtime.

Public API:
    CalibrationRecord          — dataclass for a captured forward sample
    GradientCalibrationRecord  — extends CalibrationRecord with gradient data
    CalibrationStore           — NVMe-backed append-only writer
    ForwardCaptureSession      — context manager: hooks LLM, captures h_Lq + h_Linj
    GradientCaptureSession     — context manager: same + g_inj via autograd
"""

from unme.calibration.record import (
    CalibrationRecord,
    GradientCalibrationRecord,
    RecordType,
    UMRC_MAGIC,
    UMRC_VERSION,
    UMRC_HEADER_FMT,
    UMRC_HEADER_SIZE,
)
from unme.calibration.store import CalibrationStore
from unme.calibration.capture import ForwardCaptureSession, GradientCaptureSession

__all__ = [
    "CalibrationRecord",
    "GradientCalibrationRecord",
    "RecordType",
    "UMRC_MAGIC",
    "UMRC_VERSION",
    "UMRC_HEADER_FMT",
    "UMRC_HEADER_SIZE",
    "CalibrationStore",
    "ForwardCaptureSession",
    "GradientCaptureSession",
]
