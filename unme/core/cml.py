"""Contextual Memory Link (CML) and cross-project memory relationship engine for UNME.
Features:
- Authentic AEAD encryption at rest for relational metadata (links.enc).
- Append-only event transitions: DISCOVERED -> EVALUATING -> REFERENCE -> ADAPTED -> REJECTED -> ARCHIVED -> SOURCE_BECAME_UNAVAILABLE.
- Complete lifecycle decoupling between source memories, session context, and derived memories.
"""

from __future__ import annotations
import json
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from unme.crypto.namespace import NamespaceManager


class CMLEvaluationStatus(str, Enum):
    """Lifecycle evaluation states for a contextual memory link."""
    DISCOVERED = "DISCOVERED"
    EVALUATING = "EVALUATING"
    REFERENCE = "REFERENCE"
    ADAPTED = "ADAPTED"
    REJECTED = "REJECTED"
    ARCHIVED = "ARCHIVED"
    SOURCE_BECAME_UNAVAILABLE = "SOURCE_BECAME_UNAVAILABLE"


class CMLRelationshipType(str, Enum):
    """Categorization of cross-project relationships."""
    ANTECEDENTE_TECNICO = "ANTECEDENTE_TECNICO"
    CONTRASTE = "CONTRASTE"
    ALTERNATIVA_EVALUADA = "ALTERNATIVA_EVALUADA"
    REQUISITO_COMPARTIDO = "REQUISITO_COMPARTIDO"


@dataclass
class CMLRecord:
    """Contextual Memory Link representing an independent relationship between an origin
    memory and a target project without modifying or taking ownership of the origin.
    """
    link_id: str = field(default_factory=lambda: f"CML-{uuid.uuid4().hex[:8].upper()}")
    source_memory_id: str = ""
    source_namespace_id: int = 0
    target_namespace_id: int = 0
    relationship_type: str = CMLRelationshipType.ANTECEDENTE_TECNICO.value
    relevance_reason: str = ""
    intended_usage: str = ""
    applicability_constraints: str = ""
    evaluation_status: str = CMLEvaluationStatus.DISCOVERED.value
    is_source_available: bool = True
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    derived_memory_ids: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition_to(self, new_status: CMLEvaluationStatus | str, reason: str = "") -> None:
        """Transitions evaluation status and records an immutable historical event."""
        target_val = new_status.value if isinstance(new_status, CMLEvaluationStatus) else str(new_status)
        now_ms = int(time.time() * 1000)
        event = {
            "from_status": self.evaluation_status,
            "to_status": target_val,
            "timestamp_ms": now_ms,
            "reason": reason,
        }
        self.history.append(event)
        self.evaluation_status = target_val
        self.updated_at_ms = now_ms

    def mark_source_unavailable(self, reason: str = "Origin memory deleted/tombstoned") -> None:
        """Converts CML into an unresolved historical reference without deleting derived memories."""
        self.is_source_available = False
        self.transition_to(
            CMLEvaluationStatus.SOURCE_BECAME_UNAVAILABLE,
            reason=reason,
        )

    def link_derived_memory(self, derived_memory_id: str) -> None:
        """Associates an independently created local memory that was derived from this context."""
        if derived_memory_id not in self.derived_memory_ids:
            self.derived_memory_ids.append(derived_memory_id)
            self.transition_to(CMLEvaluationStatus.ADAPTED, reason=f"Derived local memory {derived_memory_id} created")

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source_memory_id": self.source_memory_id,
            "source_namespace_id": self.source_namespace_id,
            "target_namespace_id": self.target_namespace_id,
            "relationship_type": self.relationship_type,
            "relevance_reason": self.relevance_reason,
            "intended_usage": self.intended_usage,
            "applicability_constraints": self.applicability_constraints,
            "evaluation_status": self.evaluation_status,
            "is_source_available": self.is_source_available,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "derived_memory_ids": self.derived_memory_ids,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CMLRecord:
        return cls(
            link_id=data.get("link_id", ""),
            source_memory_id=data.get("source_memory_id", ""),
            source_namespace_id=data.get("source_namespace_id", 0),
            target_namespace_id=data.get("target_namespace_id", 0),
            relationship_type=data.get("relationship_type", CMLRelationshipType.ANTECEDENTE_TECNICO.value),
            relevance_reason=data.get("relevance_reason", ""),
            intended_usage=data.get("intended_usage", ""),
            applicability_constraints=data.get("applicability_constraints", ""),
            evaluation_status=data.get("evaluation_status", CMLEvaluationStatus.DISCOVERED.value),
            is_source_available=data.get("is_source_available", True),
            created_at_ms=data.get("created_at_ms", int(time.time() * 1000)),
            updated_at_ms=data.get("updated_at_ms", int(time.time() * 1000)),
            derived_memory_ids=data.get("derived_memory_ids", []),
            history=data.get("history", []),
        )


class CMLStorage:
    """Persistent encrypted storage manager for Contextual Memory Links.
    Protects all relational metadata (project links, relevance reasons, constraints)
    using ChaCha20-Poly1305 AEAD in an append-only event log (links.enc).
    """

    def __init__(
        self,
        storage_dir: Path | str,
        namespace_id: int,
        namespace_key: bytes | None = None,
        namespace_mgr: NamespaceManager | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) / f"cml_ns_{namespace_id:016x}"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.namespace_id = namespace_id
        self.namespace_key = namespace_key
        self.namespace_mgr = namespace_mgr or NamespaceManager()

        self.enc_file = self.storage_dir / "links.enc"
        self._links: dict[str, CMLRecord] = {}
        self._event_count: int = 0
        self._load()

    def _build_event_aad(self, event_seq: int) -> bytes:
        # Canonical binary AAD: version (u16), flags (u16), namespace_id (u64), event_seq (u64)
        return struct.pack("<HHQQ", 2, 1, self.namespace_id, event_seq)

    def _load(self) -> None:
        if not self.enc_file.exists():
            return

        with open(self.enc_file, "rb") as f:
            while True:
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break
                record_len = struct.unpack("<I", length_bytes)[0]
                blob = f.read(record_len)
                if len(blob) < record_len:
                    break  # Truncated write

                if self.namespace_key is not None:
                    aad = self._build_event_aad(self._event_count)
                    try:
                        plaintext_bytes = NamespaceManager.decrypt_payload(blob, self.namespace_key, aad)
                    except Exception:
                        continue  # Skip corrupt/tampered event
                else:
                    plaintext_bytes = blob

                try:
                    event_dict = json.loads(plaintext_bytes.decode("utf-8"))
                    rec = CMLRecord.from_dict(event_dict)
                    self._links[rec.link_id] = rec
                    self._event_count += 1
                except Exception:
                    continue

    def save_cml(self, cml: CMLRecord) -> None:
        """Appends an encrypted CML event record to the immutable relational log."""
        self._links[cml.link_id] = cml
        event_dict = cml.to_dict()
        plaintext_bytes = json.dumps(event_dict, ensure_ascii=True).encode("utf-8")

        if self.namespace_key is not None:
            aad = self._build_event_aad(self._event_count)
            nonce = os.urandom(12)
            blob = NamespaceManager.encrypt_payload(plaintext_bytes, self.namespace_key, aad, nonce)
        else:
            blob = plaintext_bytes

        with open(self.enc_file, "ab") as f:
            f.write(struct.pack("<I", len(blob)))
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())

        self._event_count += 1

    def get_cml(self, link_id: str) -> CMLRecord | None:
        return self._links.get(link_id)

    def find_by_source(self, source_memory_id: str) -> list[CMLRecord]:
        return [l for l in self._links.values() if l.source_memory_id == source_memory_id]

    def list_links(self) -> list[CMLRecord]:
        return list(self._links.values())
