"""Canonical encoder interface and reference implementation for UNME Fase 3A.

Design constraints:
- d_c = 768 (canonical key/value dimensionality).  Higher discriminability than
  384 for long-lived cross-project memory; acceptable search cost with numpy
  brute-force for collections < 500K records.
- Encoder: intfloat/multilingual-e5-base via HuggingFace transformers
  (available; sentence_transformers is NOT installed).
  Model: xlm-roberta base, hidden_size=768, 12 layers, 1.1 GB safetensors.
  HF cache path: models--intfloat--multilingual-e5-base (standard hub layout).
- Output: float16 L2-normalized unit vectors in R^768.
- VRAM: encoder runs entirely on CPU during Fase 3A; zero VRAM during inference.
- Encoder signature format: "<family>:<version>:<dim>:<quant>:<metric>"
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Canonical dimensionality constants
# ---------------------------------------------------------------------------
CANONICAL_DIM: int = 768          # d_c — fixed for the UNME v1 canonical space
CANONICAL_DTYPE = np.float16
L2_EPSILON: float = 1e-8          # Division guard for normalization


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Projects a vector onto the unit L2 sphere.

    Returns a zero vector if the input norm is effectively zero (degenerate).
    Always returns float16 for storage efficiency.
    Handles NaN/Inf values by zeroing them before normalization.
    """
    v = np.array(vec, dtype=np.float32)
    # Replace any NaN or Inf values with 0.0 (can appear in float16 reinterpretation)
    v = np.where(np.isfinite(v), v, 0.0)
    norm = np.linalg.norm(v)
    if norm > L2_EPSILON:
        v = v / norm
    else:
        v = np.zeros_like(v)
    return v.astype(np.float16)


# ---------------------------------------------------------------------------
# Abstract base contract
# ---------------------------------------------------------------------------

class ICanonicalEncoder(ABC):
    """Contract for a canonical memory encoder.

    Implementations must be stateless with respect to the records being encoded;
    they may hold model weights in CPU RAM.  Every implementation must expose a
    stable encoder_signature string that changes whenever the underlying weights,
    tokenizer, or output dimensionality change.  Vectors from two distinct
    signatures MUST NOT be compared by dot product.
    """

    @abstractmethod
    def get_signature(self) -> str:
        """Returns the canonical encoder signature.

        Format: "<family>:<semantic_version>:<dim>:<quant>:<metric>"
        Example: "unme-mE5s:1.0.0:768:fp16:cosine"
        """
        ...

    @abstractmethod
    def get_dim(self) -> int:
        """Returns the output vector dimensionality (d_c)."""
        ...

    @abstractmethod
    def encode_query(self, query_text: str) -> np.ndarray:
        """Encodes a query string into a canonical key vector K_q in R^{d_c}.

        Returns float16 L2-normalized unit vector.
        """
        ...

    @abstractmethod
    def encode_record(
        self,
        canonical_text: str,
        structured_data: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encodes a UMR Level-1 record into the canonical key/value pair.

        Args:
            canonical_text: The natural-language ground-truth text from UMR L1.
            structured_data: Optional structured JSON metadata for enrichment.

        Returns:
            (K_canon, V_canon): both float16 L2-normalized unit vectors in R^{d_c}.
            For this reference implementation K_canon == V_canon (symmetric space).
        """
        ...


# ---------------------------------------------------------------------------
# Reference implementation: intfloat/multilingual-e5-small (768-dim, CPU-only)
# ---------------------------------------------------------------------------

_MODEL_NAME = "intfloat/multilingual-e5-base"
_MODEL_LOCAL_PATH = (
    r"C:\Users\bj_iv\.cache\huggingface\hub"
    r"\models--intfloat--multilingual-e5-base"
    r"\snapshots\d128750597153bb5987e10b1c3493a34e5a4502a"
)
_SIGNATURE = "unme-mE5b:1.0.0:768:fp16:cosine"


class MultilingualE5SmallEncoder(ICanonicalEncoder):
    """Canonical encoder backed by intfloat/multilingual-e5-base.

    Model: xlm-roberta base, hidden_size=768, 12 layers, ~1.1 GB safetensors.
    Supports 100+ languages including Spanish and English.
    Runs entirely on CPU RAM; zero VRAM during Fase 3A inference.
    Uses mean pooling over last hidden states followed by L2-normalization.

    The transformers library is available in this environment.
    sentence_transformers is NOT available; uses AutoModel + AutoTokenizer directly.
    Loads from local HF cache (local_files_only=True) to avoid network calls.

    Lazy initialization: model weights are loaded on first call to encode_query
    or encode_record, not at construction time.
    """

    def __init__(self, device: str = "cpu", max_length: int = 512) -> None:
        self._device = device
        self._max_length = max_length
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for MultilingualE5SmallEncoder"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_LOCAL_PATH, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            _MODEL_LOCAL_PATH, local_files_only=True
        )
        self._model.eval()
        self._model.to(self._device)
        self._torch = torch

    def _mean_pool(self, model_output: Any, attention_mask: Any) -> Any:
        """Applies mean pooling over token embeddings, respecting the attention mask."""
        token_embeddings = model_output.last_hidden_state  # (B, T, H)
        mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        summed = (token_embeddings * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=L2_EPSILON)
        return summed / counts  # (B, H)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encodes a batch of strings, returns float32 numpy array (B, 768)."""
        self._ensure_loaded()
        torch = self._torch
        # multilingual-e5 expects "query: " or "passage: " prefix for best results
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            output = self._model(**encoded)
        pooled = self._mean_pool(output, encoded["attention_mask"])
        return pooled.cpu().float().numpy()  # (B, 768) float32

    def get_signature(self) -> str:
        return _SIGNATURE

    def get_dim(self) -> int:
        return CANONICAL_DIM

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encodes a query with the 'query: ' prefix convention of multilingual-e5."""
        raw = self._encode_texts(["query: " + query_text])  # (1, 768)
        return _l2_normalize(raw[0])  # float16, unit norm

    def encode_record(
        self,
        canonical_text: str,
        structured_data: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encodes a UMR record with the 'passage: ' prefix convention."""
        text = "passage: " + canonical_text
        if structured_data:
            summary_parts = []
            for k, v in list(structured_data.items())[:8]:
                summary_parts.append(f"{k}: {v}")
            if summary_parts:
                text = text + " | " + "; ".join(summary_parts)

        raw = self._encode_texts([text])   # (1, 768)
        vec = _l2_normalize(raw[0])        # float16, unit norm
        return vec, vec.copy()             # K_canon == V_canon (symmetric)


# ---------------------------------------------------------------------------
# Deterministic encoder for unit tests (no model download required)
# ---------------------------------------------------------------------------

_HASH_SIGNATURE = "unme-tfidf-sha256:1.0.0:768:fp16:cosine"


class DeterministicHashEncoder(ICanonicalEncoder):
    """Deterministic pseudo-encoder for unit testing without model weights.

    Produces reproducible 768-dim float16 unit vectors derived from SHA-256
    of the input text.  NOT semantically meaningful; use only in tests.
    Signature is distinct from the production encoder to prevent index mixing.

    Implementation: interprets SHA-256 bytes as uint8, expands to 768 floats
    via tiling, centers around 0 by subtracting 127.5, then L2-normalizes.
    This avoids the float16 NaN/Inf trap that occurs when reinterpreting
    arbitrary bytes directly as float16 (many byte patterns map to NaN).
    """

    def get_signature(self) -> str:
        return _HASH_SIGNATURE

    def get_dim(self) -> int:
        return CANONICAL_DIM

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """Deterministic pseudo-embedding via SHA-256 repetition over uint8."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes of uint8
        # Tile the digest to fill CANONICAL_DIM values
        repetitions = (CANONICAL_DIM // len(digest)) + 1
        raw_bytes = bytearray(digest * repetitions)[:CANONICAL_DIM]
        # Interpret as uint8 -> float32, center around 0
        vec = np.frombuffer(bytes(raw_bytes), dtype=np.uint8).astype(np.float32)
        vec = vec - 127.5   # Center: values in [-127.5, 127.5], all finite
        return _l2_normalize(vec)  # Returns float16 unit vector

    def encode_query(self, query_text: str) -> np.ndarray:
        return self._hash_to_vector("query:" + query_text)

    def encode_record(
        self,
        canonical_text: str,
        structured_data: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        vec = self._hash_to_vector("passage:" + canonical_text)
        return vec, vec.copy()
