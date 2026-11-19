"""Record abstractions for the Universal Memory Record (UMR).
Encapsulates Level 1 Semantic content and Level 2 Canonical tensor representations.
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from unme.core.constants import DTYPE_FLOAT16, DTYPE_FLOAT32, DTYPE_INT8


@dataclass
class UMRRecord:
    """Universal Memory Record data structure holding semantic truth and canonical tensors."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    namespace_id: int = 0
    canonical_text: str = ""
    structured_data: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=lambda: {
        "source_type": "USER_DIRECT",
        "confidence_score": 1.0,
        "immutable": True,
        "version": 1,
    })
    graph_relations: dict[str, list[str]] = field(default_factory=lambda: {
        "predecessors": [],
        "successors": [],
        "conflicts_with": [],
    })
    key_vector: np.ndarray | None = None
    value_vector: np.ndarray | None = None
    dtype: int = DTYPE_FLOAT16

    def serialize_payload(self) -> bytes:
        """Serializes Level 1 semantic and provenance metadata to UTF-8 JSON bytes."""
        payload_dict = {
            "record_id": self.record_id,
            "namespace_id": self.namespace_id,
            "canonical_text": self.canonical_text,
            "structured_data": self.structured_data,
            "provenance": self.provenance,
            "graph_relations": self.graph_relations,
        }
        return json.dumps(payload_dict, ensure_ascii=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def deserialize_payload(cls, data: bytes) -> dict[str, Any]:
        """Deserializes metadata payload from UTF-8 JSON bytes."""
        return json.loads(data.decode("utf-8"))

    @staticmethod
    def numpy_dtype(dtype_code: int) -> np.dtype:
        """Maps internal UMR dtype code to corresponding numpy dtype."""
        if dtype_code == DTYPE_FLOAT16:
            return np.dtype(np.float16)
        elif dtype_code == DTYPE_FLOAT32:
            return np.dtype(np.float32)
        elif dtype_code == DTYPE_INT8:
            return np.dtype(np.int8)
        raise ValueError(f"Unsupported dtype code: {dtype_code}")
