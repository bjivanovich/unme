"""Unified UNME neural adapter: IModelAdapter ABC and QwenAdapter implementation.

IModelAdapter is the contractual interface for all architecture-specific adapters.
Each LLM family (Qwen/Gyro-Coder, Gemma, etc.) requires its own implementation
because d_model differs across architectures, but the canonical memory side (d_c=768)
is always the same.

The adapter composes:
  CanonicalProjector  — W_q (hidden -> canonical) and W_o (canonical -> hidden)
  MemoryCrossAttention — stateless attention over retrieved memory vectors
  MemoryGate          — learned scalar/vector gate alpha

Full injection pipeline per forward pass:
  1. h_inj = LLM hidden state at layer L_inj, shape (batch, seq, d_model)
  2. q_c = W_q(h_inj), L2-normalized               shape (batch, seq, d_c)
  3. Retrieve top-K CMR vectors from VRAM buffer    shape (K, d_c)
  4. M = cross_attention(q_c, K_mem=V_mem)          shape (batch, seq, d_c)
  5. delta_h = W_o(M)                               shape (batch, seq, d_model)
  6. h_new = h_inj + alpha * delta_h

Zero-stall: if top-K buffer is empty (memory not ready), step 3 returns empty,
step 4 returns zeros, and step 6 leaves h_inj unchanged.  Alpha is 0 at init.

VRAM budget (fp16, Gyro-Coder d_model=4096):
  W_q:  4096 * 768 * 2  =  6.29 MB
  W_o:  768 * 4096 * 2  =  6.29 MB
  Gate: 1 scalar * 4    =  < 0.001 MB
  Top-K buffer (K=5):   5 * 768 * 2 = 7.5 KB
  Total:                ~12.6 MB  (well within 256 MB ceiling)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from unme.adapter.projector import CanonicalProjector
from unme.adapter.cross_attention import MemoryCrossAttention
from unme.adapter.gating import MemoryGate


# ---------------------------------------------------------------------------
# Abstract contract
# ---------------------------------------------------------------------------

class IModelAdapter(ABC):
    """Contract for all architecture-specific UNME neural adapters.

    Each implementation binds a specific (d_model, d_c, injection_layer) tuple.
    The same canonical memory (d_c=768) works across all implementations.
    """

    @abstractmethod
    def get_model_signature(self) -> str:
        """Returns a stable identifier for this adapter's target architecture.

        Format: "<family>:<d_model>:<d_c>:<layer>"
        Example: "qwen:4096:768:L16"
        """
        ...

    @abstractmethod
    def project_query(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Projects LLM hidden state to canonical query space.

        Args:
            hidden_state: Shape (..., d_model), dtype matches LLM.

        Returns:
            q_c: Shape (..., d_c), float16, L2-normalized unit vector.
        """
        ...

    @abstractmethod
    def inject_memory(
        self,
        hidden_state: torch.Tensor,
        canonical_values: torch.Tensor,
        best_similarity: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        """Computes cross-attention and applies residual gate.

        Args:
            hidden_state: LLM hidden state at L_inj, shape (..., d_model).
            canonical_values: Retrieved CMR value vectors, shape (K, d_c).
                              Empty tensor (K=0) triggers zero-stall.
            best_similarity: Cosine sim of best retrieved memory, for gating.

        Returns:
            (h_new, effective_alpha):
                h_new: Updated hidden state, shape (..., d_model).
                effective_alpha: Actual gate value used (0.0 if zero-stall).
        """
        ...

    @abstractmethod
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Returns the list of parameters that should be trained.

        Only W_q, W_o, and alpha are trained.
        The LLM base weights are always frozen.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementation: Qwen / Gyro-Coder adapter
# ---------------------------------------------------------------------------

class QwenAdapter(nn.Module, IModelAdapter):
    """UNME adapter for Qwen-family models (including Gyro-Coder).

    Gyro-Coder is a pruned Qwen3.8-Flash-Next variant with d_model=4096.
    This adapter injects canonical memory at layer L_inj (default: layer 16
    of a 32-layer model, the midpoint where contextual representations are
    richest without being too close to the output distribution).

    Parameters
    ----------
    d_model : int
        Hidden dimensionality of the target Qwen model (4096 for Gyro-Coder).
    d_c : int
        Canonical dimensionality (768, fixed for UNME v1).
    injection_layer : int
        Index of the transformer layer where the hook is registered (0-indexed).
    gate_mode : str
        'scalar' or 'vector' — see MemoryGate documentation.
    dtype : torch.dtype
        Storage dtype for W_q and W_o.  fp16 saves VRAM; use fp32 for training.
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_c: int = 768,
        injection_layer: int = 16,
        gate_mode: str = "scalar",
        dtype: torch.dtype = torch.float16,
        confidence_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c
        self.injection_layer = injection_layer

        self.projector = CanonicalProjector(d_model=d_model, d_c=d_c, dtype=dtype)
        self.cross_attn = MemoryCrossAttention(d_c=d_c)
        self.gate = MemoryGate(
            d_model=d_model,
            mode=gate_mode,
            confidence_threshold=confidence_threshold,
        )

    def get_model_signature(self) -> str:
        return f"qwen:{self.d_model}:{self.d_c}:L{self.injection_layer}"

    def project_query(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projector.project_query(hidden_state)

    def inject_memory(
        self,
        hidden_state: torch.Tensor,
        canonical_values: torch.Tensor,
        best_similarity: float = 1.0,
    ) -> tuple[torch.Tensor, float]:
        """Runs the full injection pipeline at layer L_inj.

        Zero-stall guarantee: if canonical_values is empty (K=0), returns
        (hidden_state, 0.0) immediately without touching any weights.
        """
        if canonical_values.shape[0] == 0:
            # Zero-stall: memory not ready
            return hidden_state, 0.0

        # Step 1: project hidden state to canonical query
        q_c = self.projector.project_query(hidden_state)   # (..., d_c)

        # Step 2: cross-attend over retrieved memory values
        M = self.cross_attn.attend(q_c, canonical_values)  # (..., d_c)

        # Step 3: project memory context back to LLM hidden space
        delta_h = self.projector.project_output(M)         # (..., d_model)

        # Step 4: apply learned gate
        h_new = self.gate(hidden_state, delta_h, best_similarity)
        effective_alpha = self.gate.effective_alpha(best_similarity)

        return h_new, effective_alpha

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Returns W_q, W_o, and alpha — the only parameters trained by UNME."""
        return list(self.projector.parameters()) + list(self.gate.parameters())

    def freeze_check(self) -> dict[str, bool]:
        """Diagnostic: confirms which parameter groups are frozen vs trainable."""
        return {
            "projector.W_q": self.projector.W_q.weight.requires_grad,
            "projector.W_o": self.projector.W_o.weight.requires_grad,
            "gate.raw_alpha": self.gate.raw_alpha.requires_grad,
        }

    @property
    def vram_bytes(self) -> int:
        """Estimated VRAM footprint in bytes (projector only; gate is negligible)."""
        return self.projector.vram_bytes

    def parameter_count(self) -> dict[str, int]:
        return {
            "W_q": self.projector.W_q.weight.numel(),
            "W_o": self.projector.W_o.weight.numel(),
            "gate": self.gate.parameter_count(),
            "total": self.projector.parameter_count() + self.gate.parameter_count(),
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_adapter(
    architecture: str,
    d_model: int,
    injection_layer: int,
    d_c: int = 768,
    gate_mode: str = "scalar",
    dtype: torch.dtype = torch.float16,
) -> IModelAdapter:
    """Factory function for creating architecture-specific adapters.

    Args:
        architecture: 'qwen' (includes Gyro-Coder), 'gemma' (future).
        d_model: LLM hidden dimensionality.
        injection_layer: Layer index for hook registration.
        d_c: Canonical dimensionality (default 768).
        gate_mode: 'scalar' or 'vector'.
        dtype: Weight storage dtype.

    Raises:
        ValueError: If architecture is not supported.
    """
    if architecture in ("qwen", "gyro", "gyro-coder"):
        return QwenAdapter(
            d_model=d_model,
            d_c=d_c,
            injection_layer=injection_layer,
            gate_mode=gate_mode,
            dtype=dtype,
        )
    raise ValueError(
        f"Unsupported architecture: {architecture!r}. "
        f"Supported: 'qwen', 'gyro', 'gyro-coder'."
    )
