"""Linear projector matrices for the UNME neural adapter.

Bridges two orthogonal vector spaces:
  - LLM hidden space: R^{d_model}  (e.g. 4096 for Qwen/Gyro-Coder)
  - Canonical memory space: R^{d_c} (768, fixed for UNME v1)

Two projections are needed:
  W_q: R^{d_model} -> R^{d_c}   (query projection: hidden state -> canonical query)
  W_o: R^{d_c}    -> R^{d_model} (output projection: canonical memory -> hidden space)

Parameter budget (fp16, Gyro-Coder d_model=4096, d_c=768):
  W_q: 4096 * 768 * 2 bytes =  6.29 MB
  W_o: 768 * 4096 * 2 bytes =  6.29 MB
  Total:                       12.58 MB  (well within 256 MB VRAM ceiling)

Initialization:
  - Kaiming uniform (W_q, W_o) for stable gradient flow at training start.
  - No bias terms: bias would shift the projection away from the unit sphere
    and complicate L2 normalization of the query in canonical space.

The projector is architecture-specific. Each LLM family (Qwen, Gemma, etc.)
requires its own W_q and W_o with matching d_model.  The canonical side (d_c=768)
is always the same, enabling memory interoperability across architectures.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CanonicalProjector(nn.Module):
    """Bidirectional linear projector between LLM hidden space and canonical space.

    Parameters
    ----------
    d_model : int
        Dimensionality of the LLM hidden state (architecture-specific).
    d_c : int
        Canonical memory dimensionality (768 for UNME v1).
    dtype : torch.dtype
        Storage dtype for weights. fp16 saves VRAM; fp32 more stable for training.
    """

    def __init__(
        self,
        d_model: int,
        d_c: int = 768,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_c = d_c

        # W_q: projects LLM hidden state to canonical query space
        # Shape: (d_c, d_model) so that q_c = W_q @ h  ->  (batch, seq, d_c)
        self.W_q = nn.Linear(d_model, d_c, bias=False, dtype=dtype)

        # W_o: projects canonical memory output back to LLM hidden space
        # Shape: (d_model, d_c) so that delta_h = W_o @ m  ->  (batch, seq, d_model)
        self.W_o = nn.Linear(d_c, d_model, bias=False, dtype=dtype)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialization for both projection matrices."""
        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_o.weight, a=math.sqrt(5))

    def project_query(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Projects a hidden state tensor to canonical query space with L2 normalization.

        Args:
            hidden_state: LLM hidden state, shape (batch, seq_len, d_model) or (d_model,).

        Returns:
            q_c: Canonical query vector, L2-normalized, same batch/seq prefix, last dim d_c.
        """
        q_c = self.W_q(hidden_state.to(self.W_q.weight.dtype))  # (..., d_c)
        # L2 normalize to unit hypersphere (consistent with canonical index)
        norm = q_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return q_c / norm

    def project_output(self, canonical_memory: torch.Tensor) -> torch.Tensor:
        """Projects canonical memory values back to LLM hidden space.

        Args:
            canonical_memory: Retrieved memory tensor, shape (..., d_c).

        Returns:
            delta_h: Hidden space delta, shape (..., d_model).
        """
        return self.W_o(canonical_memory.to(self.W_o.weight.dtype))

    @property
    def vram_bytes(self) -> int:
        """Estimated VRAM footprint of this module in bytes."""
        elem_bytes = 2 if self.W_q.weight.dtype == torch.float16 else 4
        return (self.d_model * self.d_c + self.d_c * self.d_model) * elem_bytes

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
