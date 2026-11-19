"""Residual memory gate (alpha) for the UNME neural adapter.

The gate controls how much of the retrieved memory context M is added to
the LLM hidden state at the injection layer L_inj:

    h_new = h_inj + alpha * W_o(M)

Design goals:
  1. alpha starts near 0 so the adapter has no effect before training.
     This guarantees that an untrained adapter does not degrade LLM output.
  2. alpha is learned during adapter training (not fixed).
  3. alpha is bounded in [0, 1] via sigmoid to prevent runaway amplification.
  4. Zero-stall: if memory context M is zero (no memories retrieved), the
     gate contributes zero regardless of alpha's value.

Two gating modes:
  - Scalar gate: single alpha shared across all hidden dimensions.
    Fewer parameters; simpler to train; sufficient for most cases.
  - Vector gate: per-dimension alpha of shape (d_model,).
    More expressive; allows selective boosting of specific hidden dimensions.
    More parameters but still negligible (d_model floats ~ 32 KB at fp32).

The gate also computes a confidence score based on the cosine similarity
between the query and the best retrieved memory.  When confidence is low
(e.g. all retrieved memories have similarity < 0.65), it optionally suppresses
the gate below a configurable floor, preventing injection of irrelevant content.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryGate(nn.Module):
    """Learned residual gate controlling memory injection strength.

    Parameters
    ----------
    d_model : int
        LLM hidden dimensionality (used only for vector gate mode).
    mode : str
        'scalar' (default) or 'vector'.
    init_value : float
        Initial gate value before sigmoid.  Default -4.0 maps to sigmoid(-4) ≈ 0.018,
        meaning the gate starts nearly closed and must be learned open during training.
    confidence_threshold : float
        Minimum cosine similarity of the best retrieved memory to allow injection.
        If best_similarity < confidence_threshold, effective alpha is clamped to 0.
        Set to 0.0 to disable confidence gating.
    dtype : torch.dtype
        Storage dtype for gate parameters.
    """

    def __init__(
        self,
        d_model: int = 4096,
        mode: str = "scalar",
        init_value: float = -4.0,
        confidence_threshold: float = 0.0,
        dtype: torch.dtype = torch.float32,  # Keep gate in fp32 for training stability
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.confidence_threshold = confidence_threshold

        if mode == "scalar":
            # Single learnable pre-activation value; alpha = sigmoid(raw_alpha)
            self.raw_alpha = nn.Parameter(
                torch.tensor(init_value, dtype=dtype)
            )
        elif mode == "vector":
            # Per-dimension gate: alpha shape (d_model,)
            self.raw_alpha = nn.Parameter(
                torch.full((d_model,), init_value, dtype=dtype)
            )
        else:
            raise ValueError(f"Unknown gate mode: {mode!r}. Use 'scalar' or 'vector'.")

    @property
    def alpha(self) -> torch.Tensor:
        """Current gate value in [0, 1] after sigmoid."""
        return torch.sigmoid(self.raw_alpha)

    def forward(
        self,
        hidden_state: torch.Tensor,
        memory_delta: torch.Tensor,
        best_similarity: float = 1.0,
    ) -> torch.Tensor:
        """Applies the memory gate to produce the updated hidden state.

        Args:
            hidden_state:
                LLM hidden state at layer L_inj, shape (..., d_model).
            memory_delta:
                Output projection W_o(M), same shape as hidden_state (..., d_model).
                If memory_delta is a zero tensor (zero-stall), the result equals
                hidden_state unchanged regardless of alpha.
            best_similarity:
                Cosine similarity of the best retrieved memory to the query.
                Used for confidence-based gating.  Default 1.0 (gate fully open).

        Returns:
            h_new: Updated hidden state, shape (..., d_model).

        Formula:
            h_new = h_inj + alpha * memory_delta
        """
        # Confidence check: suppress gate if best memory is below threshold
        if self.confidence_threshold > 0.0 and best_similarity < self.confidence_threshold:
            return hidden_state

        alpha = self.alpha.to(hidden_state.dtype)
        return hidden_state + alpha * memory_delta

    def effective_alpha(self, best_similarity: float = 1.0) -> float:
        """Returns the effective scalar alpha value for logging and diagnostics."""
        if self.confidence_threshold > 0.0 and best_similarity < self.confidence_threshold:
            return 0.0
        a = torch.sigmoid(self.raw_alpha)
        if self.mode == "vector":
            return float(a.mean().item())
        return float(a.item())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
