"""Cross-attention mechanism between LLM queries and UNME canonical memory values.

The LLM hidden state at layer L_inj is projected to a canonical query q_c.
This module computes attention over the top-K retrieved memory value vectors
(V_mem), producing a single memory context vector M that modulates the residual.

Mathematical specification (from Contrato Tecnico Fundacional UNME V2):

  Attention scores:
    A = softmax( (q_c @ K_mem.T) / sqrt(d_c) )   shape: (batch, K)

  Memory context:
    M = A @ V_mem                                  shape: (batch, d_c)

Design decisions:
  - Single-head attention: d_c=768 is small enough that multi-head adds overhead
    without benefit for top-K <= 10 memories.
  - Temperature scaling: 1/sqrt(d_c) keeps softmax gradients from vanishing.
  - K_mem == V_mem (symmetric CMR space from Fase 3A): same float16 unit vectors
    serve as both keys and values.  This simplifies the interface and matches
    the NVMeCanonicalStore design where key_vector == value_vector.
  - No positional encoding: memory slots are unordered; attention score alone
    determines contribution weight.
  - Zero-stall: if no memories are available (K=0), returns a zero tensor
    immediately without blocking the LLM forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class MemoryCrossAttention:
    """Stateless cross-attention between a canonical query and retrieved memory vectors.

    This class is intentionally NOT an nn.Module because it has no learnable
    parameters itself — it uses the projector's W_q output and operates on
    externally retrieved memory tensors.  The learnable components (W_q, W_o,
    alpha) live in CanonicalProjector and MemoryGate respectively.
    """

    def __init__(self, d_c: int = 768) -> None:
        self.d_c = d_c
        self._scale = 1.0 / math.sqrt(d_c)

    def attend(
        self,
        query: torch.Tensor,
        memory_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cross-attention of query over memory_vectors.

        Args:
            query:
                Canonical query vector, shape (d_c,) or (batch, d_c) or (batch, seq, d_c).
                Must be L2-normalized (unit norm) — enforced by CanonicalProjector.
            memory_vectors:
                Retrieved CMR value vectors, shape (K, d_c) where K = number of
                retrieved memories.  Vectors must be L2-normalized (unit norm).
                If K == 0 (no memories available), returns zero tensor immediately.

        Returns:
            M: Memory context tensor, same shape as query (last dim replaced with d_c).
               Zero tensor if K == 0 (zero-stall policy).
        """
        if memory_vectors.shape[0] == 0:
            # Zero-stall: no memories retrieved — return zero delta (alpha will also be 0)
            return torch.zeros_like(query)

        q = query.to(memory_vectors.dtype)   # Match dtype to memory vectors

        # Handle different query shapes
        squeeze_batch = False
        if q.dim() == 1:
            q = q.unsqueeze(0)   # (1, d_c)
            squeeze_batch = True

        squeeze_seq = False
        if q.dim() == 2:
            q = q.unsqueeze(1)   # (batch, 1, d_c)
            squeeze_seq = True

        # q: (batch, seq, d_c), memory_vectors: (K, d_c)
        # Scores: (batch, seq, K)
        # K_mem == V_mem (symmetric space) so we use memory_vectors for both
        scores = torch.einsum("bsd,kd->bsk", q, memory_vectors) * self._scale

        # Softmax over K dimension
        weights = F.softmax(scores, dim=-1)   # (batch, seq, K)

        # Weighted sum of value vectors: (batch, seq, d_c)
        M = torch.einsum("bsk,kd->bsd", weights, memory_vectors)

        if squeeze_seq:
            M = M.squeeze(1)   # (batch, d_c)
        if squeeze_batch:
            M = M.squeeze(0)   # (d_c,)

        return M

    def attend_with_scores(
        self,
        query: torch.Tensor,
        memory_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as attend() but also returns attention weights for inspection/logging.

        Returns:
            (M, weights): memory context and attention distribution over K memories.
        """
        if memory_vectors.shape[0] == 0:
            empty_weights = torch.zeros(0, dtype=query.dtype, device=query.device)
            return torch.zeros_like(query), empty_weights

        q = query.to(memory_vectors.dtype)
        was_1d = q.dim() == 1
        if was_1d:
            q = q.unsqueeze(0).unsqueeze(0)   # (1, 1, d_c)

        scores = torch.einsum("bsd,kd->bsk", q, memory_vectors) * self._scale
        weights = F.softmax(scores, dim=-1)
        M = torch.einsum("bsk,kd->bsd", weights, memory_vectors)

        if was_1d:
            M = M.squeeze(0).squeeze(0)
            weights = weights.squeeze(0).squeeze(0)

        return M, weights
