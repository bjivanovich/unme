"""Activation and gradient capture sessions for UNME Fase 3C.

Both sessions hook a frozen PyTorch model at two layer indices:
  L_q   — the layer whose output serves as the query for the memory projector
  L_inj — the layer where the memory adapter injects its delta

ForwardCaptureSession
    Captures (h_Lq, h_Linj, token_ids) during a single forward pass.
    The LLM weights are NOT modified and require NO gradient.

GradientCaptureSession
    Same as ForwardCaptureSession plus:
    - Detaches h_Linj and sets requires_grad=True (creates a leaf tensor)
    - After loss.backward(), reads h_Linj_leaf.grad as g_inj
    - g_inj = nabla_{h_Linj} L_task  (gradient w.r.t. activation, not weights)

Design constraints
    - Both sessions are context managers.
    - LLM weights remain frozen (no grad) throughout.
    - Hooks are removed unconditionally in __exit__, even on exception.
    - float16 storage: activations are cast to float16 before saving.
    - Zero VRAM budget violation: no new parameters are added.
    - Supports any nn.Module with a `.layers` attribute (Qwen-style)
      or any indexable nn.ModuleList accessible via model_layers.

Usage
------
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with ForwardCaptureSession(
        model=model,
        model_layers=model.model.layers,
        layer_q=14,
        layer_inj=16,
        model_signature="gyro-coder-q4:L14:L16",
        session_id=session_id,
    ) as sess:
        output = model(**inputs)
        record = sess.get_record(
            token_ids=inputs["input_ids"][0].cpu().numpy().astype("int32"),
            record_index=0,
        )

    with GradientCaptureSession(...) as sess:
        output = model(**inputs)
        loss = compute_task_loss(output, targets)
        loss.backward()
        record = sess.get_record(
            token_ids=..., record_index=0, loss_val=loss.item()
        )
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np

from unme.calibration.record import (
    CalibrationRecord,
    GradientCalibrationRecord,
)


class ForwardCaptureSession:
    """Context manager that hooks a frozen LLM and captures h_Lq and h_Linj.

    The model is NOT modified; only PyTorch forward hooks are registered.
    All captured activations are stored as float16 to minimize RAM.

    Args:
        model         : Any PyTorch nn.Module (e.g. the full LLM).
        model_layers  : An indexable sequence of nn.Module layers
                        (e.g. model.model.layers for Qwen-style models).
        layer_q       : Index of the query capture layer L_q.
        layer_inj     : Index of the injection capture layer L_inj.
        model_signature: Stable string identifying this model + layer config.
        session_id    : uint64 identifier for this capture session.
        device        : Device where the model runs ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: Any,
        model_layers: Any,                 # nn.ModuleList or any indexable
        layer_q: int,
        layer_inj: int,
        model_signature: str,
        session_id: int,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._layers = model_layers
        self._layer_q = layer_q
        self._layer_inj = layer_inj
        self._model_signature = model_signature
        self._session_id = session_id
        self._device = device

        self._h_Lq: Optional[np.ndarray] = None
        self._h_Linj: Optional[np.ndarray] = None
        self._hooks: list[Any] = []

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ForwardCaptureSession":
        self._register_hooks()
        return self

    def __exit__(self, *_: object) -> None:
        self._remove_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Registers forward hooks at L_q and L_inj."""
        hook_q = self._layers[self._layer_q].register_forward_hook(
            self._make_capture_hook("q")
        )
        hook_inj = self._layers[self._layer_inj].register_forward_hook(
            self._make_capture_hook("inj")
        )
        self._hooks = [hook_q, hook_inj]

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_capture_hook(self, role: str):
        """Returns a forward hook that captures the layer output tensor."""
        def hook(module: Any, inputs: Any, output: Any) -> None:
            # Qwen-style layers return a tuple; hidden states are output[0]
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            # h shape: (batch, seq_len, d_model)
            # Capture first batch item, all token positions, cast to float16
            captured = h[0].detach().cpu().to(dtype=_torch_float16()).numpy()
            if role == "q":
                self._h_Lq = captured
            else:
                self._h_Linj = captured

        return hook

    # ------------------------------------------------------------------
    # Record construction
    # ------------------------------------------------------------------

    def get_record(
        self,
        token_ids: np.ndarray,
        record_index: int,
    ) -> CalibrationRecord:
        """Returns a CalibrationRecord from the most recent forward pass.

        Must be called AFTER the forward pass has completed.
        """
        if self._h_Lq is None or self._h_Linj is None:
            raise RuntimeError(
                "No activations captured yet.  Run a forward pass inside the session."
            )
        return CalibrationRecord(
            h_Lq=self._h_Lq.astype(np.float16),
            h_Linj=self._h_Linj.astype(np.float16),
            token_ids=np.asarray(token_ids, dtype=np.int32),
            model_signature=self._model_signature,
            session_id=self._session_id,
            record_index=record_index,
            timestamp_ns=time.time_ns(),
        )

    def reset(self) -> None:
        """Clears captured buffers between forward passes."""
        self._h_Lq = None
        self._h_Linj = None


class GradientCaptureSession(ForwardCaptureSession):
    """ForwardCaptureSession extended with gradient capture at L_inj.

    Mechanism
    ---------
    1. The forward hook at L_inj replaces h_Linj with a detached leaf tensor
       that has requires_grad=True.  This breaks the autograd graph at L_inj,
       so backward() computes the gradient of the loss w.r.t. h_Linj without
       backpropagating into the frozen LLM weights.

    2. After loss.backward(), h_Linj_leaf.grad holds g_inj:
           g_inj = nabla_{h_Linj} L_task

    3. get_record() assembles a GradientCalibrationRecord with g_inj.

    Usage requirement
    -----------------
    The model must NOT be wrapped in torch.no_grad() during gradient capture.
    LLM parameters must still have requires_grad=False (frozen weights);
    only the activation leaf is differentiable.

    After calling loss.backward(), call get_record() BEFORE the next forward
    pass (the leaf reference is overwritten on the next hook fire).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._h_Linj_leaf: Any = None   # torch.Tensor with requires_grad=True

    def _register_hooks(self) -> None:
        """Registers forward hooks; L_inj hook creates a differentiable leaf."""
        import torch

        hook_q = self._layers[self._layer_q].register_forward_hook(
            self._make_capture_hook("q")
        )
        hook_inj = self._layers[self._layer_inj].register_forward_hook(
            self._make_gradient_hook()
        )
        self._hooks = [hook_q, hook_inj]

    def _make_gradient_hook(self):
        """Returns a forward hook that injects a differentiable leaf at L_inj."""
        import torch

        def hook(module: Any, inputs: Any, output: Any) -> Any:
            # Extract hidden states from Qwen-style tuple output
            if isinstance(output, tuple):
                h_original = output[0]
                rest = output[1:]
            else:
                h_original = output
                rest = ()

            # Detach from the LLM graph, then re-enable grad as a leaf tensor.
            # This means backward() computes grad w.r.t. h_leaf, not LLM weights.
            h_leaf = h_original.detach().requires_grad_(True)
            self._h_Linj_leaf = h_leaf

            # Capture float16 copy for the CalibrationRecord (shape at this point)
            self._h_Linj = h_leaf[0].detach().cpu().to(dtype=torch.float16).numpy()

            # Return the leaf tensor back into the graph so downstream layers run
            if rest:
                return (h_leaf,) + rest
            return h_leaf

        return hook

    def get_record(
        self,
        token_ids: np.ndarray,
        record_index: int,
        loss_val: float = 0.0,
    ) -> GradientCalibrationRecord:   # type: ignore[override]
        """Returns a GradientCalibrationRecord after forward + backward.

        Must be called AFTER loss.backward() has been called.
        """
        import torch

        if self._h_Lq is None or self._h_Linj is None:
            raise RuntimeError(
                "No activations captured yet.  Run a forward pass inside the session."
            )
        if self._h_Linj_leaf is None:
            raise RuntimeError(
                "GradientCaptureSession: h_Linj leaf not set.  "
                "Ensure loss.backward() was called after the forward pass."
            )

        leaf_grad = self._h_Linj_leaf.grad
        if leaf_grad is None:
            raise RuntimeError(
                "h_Linj.grad is None.  Ensure loss.backward() was called and "
                "the loss depends on the output of layer L_inj."
            )

        g_inj = leaf_grad[0].detach().cpu().to(dtype=torch.float16).numpy()

        return GradientCalibrationRecord(
            h_Lq=self._h_Lq.astype(np.float16),
            h_Linj=self._h_Linj.astype(np.float16),
            token_ids=np.asarray(token_ids, dtype=np.int32),
            model_signature=self._model_signature,
            session_id=self._session_id,
            record_index=record_index,
            timestamp_ns=time.time_ns(),
            g_inj=g_inj.astype(np.float16),
            loss_val=float(loss_val),
        )

    def reset(self) -> None:
        super().reset()
        self._h_Linj_leaf = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _torch_float16():
    """Returns torch.float16 without importing torch at module load time."""
    import torch
    return torch.float16
