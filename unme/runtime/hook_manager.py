"""PyTorch Forward Hook manager for UNME memory injection.

Registers a forward hook on a specific transformer layer of a frozen LLM.
When the hook fires during a forward pass, it:
  1. Reads the top-K canonical memory vectors from the VRAM buffer.
  2. Calls the adapter's inject_memory() pipeline.
  3. Returns the modified hidden state to the LLM graph.

Design constraints:
  - The LLM base is always frozen (requires_grad=False on all base params).
  - The hook fires synchronously within the LLM forward pass.
  - Zero-stall: if the VRAM buffer is empty or not ready, the hook returns
    the hidden state unchanged.
  - The hook handle is stored for clean removal at any time.
  - Multiple hook registrations on the same layer are detected and refused.

VRAM buffer:
  The VRAM buffer holds the top-K canonical value vectors (float16) prefetched
  from NVMe -> RAM -> VRAM by the async prefetcher (Fase 3C).
  For Fase 3B, the buffer is populated manually before each forward pass.
  Shape: (K, d_c) where K <= top_k_max (default 5).
  Empty when no memories are available: shape (0, d_c).

Hook target:
  For transformer models, the hook is registered on the output of a specific
  decoder layer.  The exact module path depends on the architecture:
    Qwen2: model.layers[L]
    Gemma2: model.layers[L]
  The hook receives (module, input, output) where output is a tuple; the hidden
  states are always at output[0] for Qwen/Gemma-family models.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from unme.adapter.adapter import IModelAdapter


class MemoryHookManager:
    """Manages the lifecycle of UNME forward hooks on a frozen LLM.

    Usage:
        adapter = QwenAdapter(d_model=4096, injection_layer=16)
        manager = MemoryHookManager(model, adapter, layer_path="model.layers")
        manager.register()

        # Before each forward pass, populate the memory buffer:
        manager.set_memory_buffer(canonical_values_tensor)   # (K, 768)

        # Run normal LLM forward — hook fires automatically at layer 16
        output = model.generate(...)

        # Clean up:
        manager.remove()

    Parameters
    ----------
    model : nn.Module
        The frozen LLM.  All base parameters must have requires_grad=False.
    adapter : IModelAdapter
        The UNME neural adapter for this architecture.
    layer_path : str
        Attribute path to the list of decoder layers on the model.
        Default "model.layers" (Qwen2 / Gemma2 convention).
    hidden_state_index : int
        Index into the layer output tuple where hidden states live.
        Default 0 (Qwen2 / Gemma2: output[0] = hidden states).
    top_k_max : int
        Maximum number of memory vectors in the VRAM buffer.
    log_alpha : bool
        If True, stores the effective alpha value from the last injection
        for diagnostics (accessible via last_alpha property).
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: IModelAdapter,
        layer_path: str = "model.layers",
        hidden_state_index: int = 0,
        top_k_max: int = 5,
        log_alpha: bool = True,
    ) -> None:
        self._model = model
        self._adapter = adapter
        self._layer_path = layer_path
        self._hidden_state_index = hidden_state_index
        self._top_k_max = top_k_max
        self._log_alpha = log_alpha

        self._hook_handle: Any = None
        self._memory_buffer: torch.Tensor = torch.empty(0)
        self._best_similarity: float = 0.0
        self._last_alpha: float = 0.0
        self._is_registered: bool = False

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def set_memory_buffer(
        self,
        canonical_values: torch.Tensor,
        best_similarity: float = 1.0,
    ) -> None:
        """Loads the top-K canonical value vectors into the VRAM buffer.

        Args:
            canonical_values: Float16 tensor, shape (K, d_c).  K can be 0.
            best_similarity: Cosine similarity of the best retrieved memory.
        """
        if canonical_values.shape[0] > self._top_k_max:
            canonical_values = canonical_values[: self._top_k_max]
        self._memory_buffer = canonical_values
        self._best_similarity = best_similarity

    def clear_memory_buffer(self) -> None:
        """Empties the VRAM buffer (zero-stall for next forward pass)."""
        self._memory_buffer = torch.empty(0)
        self._best_similarity = 0.0

    @property
    def last_alpha(self) -> float:
        """Effective gate value used in the most recent forward pass."""
        return self._last_alpha

    @property
    def is_registered(self) -> bool:
        return self._is_registered

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def _get_target_layer(self) -> nn.Module:
        """Traverses the model's attribute tree to find the injection layer."""
        # Follow the layer_path to get the list of layers
        obj = self._model
        for attr in self._layer_path.split("."):
            obj = getattr(obj, attr)
        # obj is now the list/ModuleList of decoder layers
        layer_idx = self._adapter.injection_layer
        try:
            layer = obj[layer_idx]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Cannot access layer {layer_idx} at path '{self._layer_path}': {exc}"
            )
        return layer

    def _build_hook(self):
        """Builds the forward hook closure."""
        adapter = self._adapter
        manager = self   # Capture self for buffer access

        def _hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            # output is typically a tuple; hidden states at index 0
            if isinstance(output, tuple):
                h = output[manager._hidden_state_index]
            else:
                h = output

            mem = manager._memory_buffer
            if mem.numel() == 0:
                # Zero-stall: return output unchanged
                manager._last_alpha = 0.0
                return output

            # Move memory buffer to same device as hidden state
            mem = mem.to(h.device)

            h_new, alpha = adapter.inject_memory(
                hidden_state=h,
                canonical_values=mem,
                best_similarity=manager._best_similarity,
            )

            if manager._log_alpha:
                manager._last_alpha = alpha

            # Reconstruct output tuple with modified hidden state
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        return _hook

    def register(self) -> None:
        """Registers the forward hook on the target layer.

        Raises:
            RuntimeError: If a hook is already registered (use remove() first).
            ValueError: If the target layer cannot be found.
        """
        if self._is_registered:
            raise RuntimeError(
                "Hook is already registered. Call remove() before re-registering."
            )
        target = self._get_target_layer()
        self._hook_handle = target.register_forward_hook(self._build_hook())
        self._is_registered = True

    def remove(self) -> None:
        """Removes the forward hook from the target layer."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._is_registered = False

    def __enter__(self) -> "MemoryHookManager":
        self.register()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict[str, Any]:
        """Returns a snapshot of the manager's current state for logging."""
        return {
            "is_registered": self._is_registered,
            "layer_path": self._layer_path,
            "injection_layer": self._adapter.injection_layer,
            "adapter_signature": self._adapter.get_model_signature(),
            "buffer_size_K": self._memory_buffer.shape[0] if self._memory_buffer.numel() > 0 else 0,
            "best_similarity": self._best_similarity,
            "last_alpha": self._last_alpha,
        }
