"""SurrogateTrainer: offline adapter training for UNME Fase 3D.

Training runs ENTIRELY WITHOUT the LLM in VRAM or memory.
Inputs come from pre-recorded .umrc calibration files.
Only three parameter groups are trained: W_q, W_o, and alpha (gate).

Surrogate Loss (from Contrato Tecnico Fundacional UNME V2, corrected sign)
===========================================================================
Given a GradientCalibrationRecord with:
  h_Lq   : hidden state at layer L_q  (seq_len_q,  d_model) float16
  h_Linj : hidden state at layer L_inj (seq_len_inj, d_model) float16
  g_inj  : nabla_{h_Linj} L_task       (seq_len_inj, d_model) float16

And memory value vectors V_mem : (K, d_c) float16

The adapter computes a delta:
  q_c     = normalize(W_q @ h_Lq[query_pos])          (d_c,)
  A       = softmax(q_c @ V_mem.T / sqrt(d_c))        (K,)
  M       = A @ V_mem                                  (d_c,)
  delta_h = alpha * W_o @ M                            (d_model,)

Surrogate loss (minimize in PyTorch — equivalent to maximizing alignment):
  L = -<g_inj_flat, delta_h_broadcast> + lambda_reg * ||delta_h||^2

Interpretation:
  -<g_inj, delta_h> : steer delta_h to align with the gradient direction
                      that reduces the original task loss.
  lambda * ||delta||^2 : regularize; prevents unbounded delta growth.

Training strategy:
  - Adapter parameters in fp32 during training (master weights).
  - Memory vectors V_mem fixed (not trained); detached from autograd graph.
  - Only GradientCalibrationRecord instances contribute (has_gradient=True).
  - ForwardCapture records are silently skipped (no g_inj available).
  - Checkpoint saved as fp32 state_dict; convert to fp16 at inference load time.

VRAM usage during training: 0 MB (runs entirely on CPU).
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from unme.adapter.adapter import IModelAdapter, make_adapter
from unme.calibration.record import (
    CalibrationRecord,
    GradientCalibrationRecord,
    RecordType,
)
from unme.calibration.store import CalibrationStore


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyperparameters and paths for SurrogateTrainer.

    Attributes
    ----------
    d_model         : LLM hidden dimensionality (4096 for Gyro-Coder).
    d_c             : Canonical dimensionality (768, fixed).
    injection_layer : Layer index where the adapter hooks into the LLM.
    architecture    : Adapter family ('qwen', 'gyro-coder').
    gate_mode       : 'scalar' or 'vector'.
    lambda_reg      : L2 regularization weight on delta_h.
    lr              : AdamW learning rate.
    betas           : AdamW momentum coefficients.
    weight_decay    : AdamW weight decay (applied to W_q, W_o; not to alpha).
    grad_clip       : Max gradient norm (None = no clipping).
    query_position  : Token position in h_Lq used as the memory query.
                      -1 = last token (causal LM convention).
    inj_position    : Token position in h_Linj where delta is injected.
                      -1 = last token.  Must match query_position convention.
    checkpoint_dir  : Directory for saving/loading adapter checkpoints.
    log_every       : Print training metrics every N records.
    """

    d_model: int = 4096
    d_c: int = 768
    injection_layer: int = 16
    architecture: str = "qwen"
    gate_mode: str = "scalar"
    lambda_reg: float = 0.01
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    query_position: int = -1
    inj_position: int = -1
    checkpoint_dir: str | pathlib.Path = "d:/memory/checkpoints"
    log_every: int = 100


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    """Aggregated metrics for one training epoch."""

    n_records: int = 0
    n_skipped: int = 0          # CalibrationRecord (no gradient) skipped
    total_loss: float = 0.0
    total_alignment: float = 0.0
    total_reg: float = 0.0
    total_delta_norm: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def mean_loss(self) -> float:
        return self.total_loss / max(self.n_records, 1)

    @property
    def mean_alignment(self) -> float:
        return self.total_alignment / max(self.n_records, 1)

    @property
    def mean_reg(self) -> float:
        return self.total_reg / max(self.n_records, 1)

    @property
    def mean_delta_norm(self) -> float:
        return self.total_delta_norm / max(self.n_records, 1)

    @property
    def records_per_second(self) -> float:
        return self.n_records / max(self.elapsed_seconds, 1e-9)

    def __str__(self) -> str:
        return (
            f"n={self.n_records} skip={self.n_skipped} "
            f"loss={self.mean_loss:.6f} "
            f"align={self.mean_alignment:.6f} "
            f"reg={self.mean_reg:.6f} "
            f"||delta||={self.mean_delta_norm:.4f} "
            f"{self.records_per_second:.1f} rec/s"
        )


# ---------------------------------------------------------------------------
# Surrogate Trainer
# ---------------------------------------------------------------------------

class SurrogateTrainer:
    """Trains UNME adapter parameters (W_q, W_o, alpha) offline.

    No LLM required during training.  All inputs come from .umrc files.

    Parameters
    ----------
    config       : TrainerConfig with all hyperparameters.
    memory_values: np.ndarray of shape (K, d_c) float16 — fixed memory
                   value vectors from the UNME canonical store.  Not trained.
                   Pass an empty array (K=0) for zero-memory sanity tests.
    """

    def __init__(
        self,
        config: TrainerConfig,
        memory_values: np.ndarray,
    ) -> None:
        self.config = config
        self._device = torch.device("cpu")  # always CPU — zero VRAM

        # Build adapter in fp32 for numerically stable training.
        # QwenAdapter stores W_q/W_o as nn.Linear; dtype arg controls weight dtype.
        self._adapter = make_adapter(
            architecture=config.architecture,
            d_model=config.d_model,
            injection_layer=config.injection_layer,
            d_c=config.d_c,
            gate_mode=config.gate_mode,
            dtype=torch.float32,   # fp32 master weights during training
        ).to(self._device)

        # Only W_q, W_o, alpha are trainable; nothing else.
        trainable = self._adapter.trainable_parameters()
        for p in trainable:
            p.requires_grad_(True)

        # Freeze everything that is NOT in trainable_parameters.
        # (MemoryCrossAttention is stateless — no parameters to freeze.)
        trainable_ids = {id(p) for p in trainable}
        for p in self._adapter.parameters():
            if id(p) not in trainable_ids:
                p.requires_grad_(False)

        # AdamW: apply weight_decay only to projection matrices, not to alpha.
        proj_params  = list(self._adapter.projector.parameters())
        gate_params  = list(self._adapter.gate.parameters())
        self._optimizer = optim.AdamW(
            [
                {"params": proj_params, "weight_decay": config.weight_decay},
                {"params": gate_params, "weight_decay": 0.0},
            ],
            lr=config.lr,
            betas=config.betas,
        )

        # Fixed memory values: (K, d_c) float32, detached.
        self._V_mem: torch.Tensor = torch.from_numpy(
            memory_values.astype(np.float32)
        ).to(self._device)   # shape (K, d_c), no grad

        # Training state
        self._global_step: int = 0
        self._checkpoint_dir = pathlib.Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core: single record forward + backward
    # ------------------------------------------------------------------

    def train_on_record(
        self, record: GradientCalibrationRecord
    ) -> dict[str, float]:
        """Runs one forward + backward step on a single calibration record.

        Returns a dict with keys: loss, alignment, regularization, delta_norm.
        """
        cfg = self.config

        # Convert calibration tensors to fp32 on CPU.
        # Records are float16 on disk; upcasting to float32 for stable grad.
        h_Lq = torch.from_numpy(
            record.h_Lq.astype(np.float32)
        ).to(self._device)   # (seq_len_q, d_model)

        h_Linj = torch.from_numpy(
            record.h_Linj.astype(np.float32)
        ).to(self._device)   # (seq_len_inj, d_model)

        g_inj = torch.from_numpy(
            record.g_inj.astype(np.float32)
        ).to(self._device)   # (seq_len_inj, d_model) — detached reference signal

        # Select token positions for query and injection.
        # h_Lq[query_pos] is the token that formulates the memory query.
        q_pos   = cfg.query_position    # -1 = last token
        inj_pos = cfg.inj_position      # -1 = last token

        h_q_tok   = h_Lq[q_pos]        # (d_model,)
        h_inj_tok = h_Linj[inj_pos]    # (d_model,) — reference for delta
        g_tok     = g_inj[inj_pos]     # (d_model,) — gradient signal

        # Early exit: no memory available → delta is zero → loss undefined.
        if self._V_mem.shape[0] == 0:
            return {"loss": 0.0, "alignment": 0.0,
                    "regularization": 0.0, "delta_norm": 0.0}

        # --- Forward pass through adapter ---
        # project_query applies W_q + L2 norm → (d_c,)
        q_c = self._adapter.projector.project_query(h_q_tok.unsqueeze(0)).squeeze(0)

        # Cross-attention: scores = softmax(q_c @ V_mem.T / sqrt(d_c))
        d_c = q_c.shape[-1]
        scores = torch.softmax(
            (self._V_mem @ q_c) / (d_c ** 0.5), dim=0
        )                                           # (K,)
        M = scores @ self._V_mem                   # (d_c,)

        # project_output applies W_o → (d_model,)
        delta_raw = self._adapter.projector.project_output(M.unsqueeze(0)).squeeze(0)

        # Gate: alpha * delta_raw (alpha starts near 0 via sigmoid(-4) ≈ 0.018)
        # We call gate directly with best_similarity=1.0 (confidence always full
        # during offline training — the adapter learns when to open the gate).
        h_new = self._adapter.gate(
            h_inj_tok.unsqueeze(0).unsqueeze(0),
            delta_raw.unsqueeze(0).unsqueeze(0),
            best_similarity=1.0,
        ).squeeze(0).squeeze(0)                     # (d_model,)

        delta_h = h_new - h_inj_tok                 # actual delta injected

        # --- Surrogate loss ---
        # alignment term: maximize <g_inj, delta_h> → minimize negative
        alignment = torch.dot(g_tok, delta_h)
        regularization = cfg.lambda_reg * delta_h.pow(2).sum()
        loss = -alignment + regularization

        # --- Backward ---
        self._optimizer.zero_grad()
        loss.backward()

        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self._adapter.trainable_parameters(), cfg.grad_clip
            )

        self._optimizer.step()
        self._global_step += 1

        return {
            "loss":           loss.item(),
            "alignment":      alignment.item(),
            "regularization": regularization.item(),
            "delta_norm":     delta_h.norm().item(),
        }

    # ------------------------------------------------------------------
    # Epoch-level training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        records: list[CalibrationRecord],
        epoch: int = 0,
    ) -> EpochMetrics:
        """Trains on a list of CalibrationRecord objects (one pass).

        Records that are not GradientCalibrationRecord are silently skipped.
        """
        self._adapter.train()
        metrics = EpochMetrics()
        t0 = time.perf_counter()

        for record in records:
            if not isinstance(record, GradientCalibrationRecord):
                metrics.n_skipped += 1
                continue

            step_metrics = self.train_on_record(record)
            metrics.n_records     += 1
            metrics.total_loss    += step_metrics["loss"]
            metrics.total_alignment   += step_metrics["alignment"]
            metrics.total_reg         += step_metrics["regularization"]
            metrics.total_delta_norm  += step_metrics["delta_norm"]

            if (metrics.n_records % self.config.log_every) == 0:
                print(
                    f"epoch={epoch} step={self._global_step} "
                    f"loss={step_metrics['loss']:.6f} "
                    f"align={step_metrics['alignment']:.6f} "
                    f"||delta||={step_metrics['delta_norm']:.4f}"
                )

        metrics.elapsed_seconds = time.perf_counter() - t0
        self._adapter.eval()
        return metrics

    def train_from_store(
        self,
        umrc_path: str | pathlib.Path,
        epoch: int = 0,
    ) -> EpochMetrics:
        """Streams records from a .umrc file and runs one training epoch.

        Memory-efficient: records are yielded one at a time by CalibrationStore.scan.
        """
        records = list(CalibrationStore.scan(umrc_path))
        return self.train_epoch(records, epoch=epoch)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, name: str = "adapter") -> pathlib.Path:
        """Saves adapter state dict (fp32) to checkpoint_dir/name.pt.

        Returns the path of the saved file.
        """
        path = self._checkpoint_dir / f"{name}.pt"
        state = {
            "global_step":      self._global_step,
            "config": {
                "d_model":          self.config.d_model,
                "d_c":              self.config.d_c,
                "injection_layer":  self.config.injection_layer,
                "architecture":     self.config.architecture,
                "gate_mode":        self.config.gate_mode,
            },
            "adapter_state_dict":    self._adapter.state_dict(),
            "optimizer_state_dict":  self._optimizer.state_dict(),
        }
        torch.save(state, path)
        return path

    def load_checkpoint(self, name: str = "adapter") -> int:
        """Loads adapter + optimizer state from checkpoint.

        Returns the global_step at the time of the checkpoint.
        """
        path = self._checkpoint_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(path, map_location=self._device, weights_only=True)
        self._adapter.load_state_dict(state["adapter_state_dict"])
        self._optimizer.load_state_dict(state["optimizer_state_dict"])
        self._global_step = state["global_step"]
        return self._global_step

    def export_fp16(self, name: str = "adapter_fp16") -> pathlib.Path:
        """Exports adapter weights converted to fp16 for inference.

        Only the adapter state_dict (no optimizer state).
        """
        path = self._checkpoint_dir / f"{name}.pt"
        fp16_state = {}
        for k, v in self._adapter.state_dict().items():
            fp16_state[k] = v.half()   # fp32 → fp16
        torch.save({"adapter_state_dict": fp16_state}, path)
        return path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def adapter(self) -> IModelAdapter:
        return self._adapter

    @property
    def memory_size(self) -> int:
        """Number of memory vectors currently loaded."""
        return self._V_mem.shape[0]

    def effective_alpha(self) -> float:
        """Current gate alpha value (after sigmoid)."""
        return self._adapter.gate.effective_alpha(best_similarity=1.0)
