"""Test suite for Fase 3B: Neural Adapter (projector, cross-attention, gate, hook manager).

All tests run without loading a real LLM.  The LLM is simulated with a minimal
nn.Module that exposes the same interface (model.layers[i], tuple output).

Tests cover:
  1. CanonicalProjector: shapes, L2 norm, dtype, VRAM budget
  2. MemoryCrossAttention: zero-stall, attention output shape, weight sum to 1
  3. MemoryGate: initial alpha near 0, sigmoid bounds, zero-stall pass-through
  4. QwenAdapter: full pipeline, parameter count, trainable_parameters()
  5. MemoryHookManager: register/remove, zero-stall, hook fires correctly
  6. Integration: full pipeline with synthetic hidden state and memory vectors
"""

from __future__ import annotations

import sys
sys.path.insert(0, "d:/memory")

import torch
import torch.nn as nn

from unme.adapter.projector import CanonicalProjector
from unme.adapter.cross_attention import MemoryCrossAttention
from unme.adapter.gating import MemoryGate
from unme.adapter.adapter import QwenAdapter, make_adapter, IModelAdapter
from unme.runtime.hook_manager import MemoryHookManager


# ---------------------------------------------------------------------------
# Minimal fake LLM for hook tests
# ---------------------------------------------------------------------------

class FakeDecoderLayer(nn.Module):
    """Simulates a single transformer decoder layer: identity on hidden states."""
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        return (hidden_states,)   # Qwen-style: tuple output, [0] = hidden states


class FakeLLM(nn.Module):
    """Minimal frozen LLM with model.layers attribute."""
    def __init__(self, num_layers: int = 32, d_model: int = 4096) -> None:
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([FakeDecoderLayer() for _ in range(num_layers)])
        })

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = hidden_states
        for layer in self.model["layers"]:
            result = layer(h)
            h = result[0]
        return h


# ---------------------------------------------------------------------------
# 1. CanonicalProjector
# ---------------------------------------------------------------------------

def test_projector_query_shape():
    proj = CanonicalProjector(d_model=512, d_c=768, dtype=torch.float32)
    h = torch.randn(2, 10, 512)   # (batch, seq, d_model)
    q_c = proj.project_query(h)
    assert q_c.shape == (2, 10, 768), f"Expected (2, 10, 768), got {q_c.shape}"


def test_projector_query_unit_norm():
    proj = CanonicalProjector(d_model=512, d_c=768, dtype=torch.float32)
    h = torch.randn(4, 512)   # (batch, d_model)
    q_c = proj.project_query(h)
    norms = q_c.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Query vectors must have unit L2 norm. Got: {norms}"
    )


def test_projector_output_shape():
    proj = CanonicalProjector(d_model=512, d_c=768, dtype=torch.float32)
    M = torch.randn(3, 768)   # (batch, d_c)
    delta = proj.project_output(M)
    assert delta.shape == (3, 512), f"Expected (3, 512), got {delta.shape}"


def test_projector_vram_budget_gyro_coder():
    proj = CanonicalProjector(d_model=4096, d_c=768, dtype=torch.float16)
    vram_mb = proj.vram_bytes / (1024 ** 2)
    # W_q + W_o in fp16: 2 * 4096 * 768 * 2 bytes = 12.58 MB
    assert vram_mb < 20.0, f"Projector VRAM {vram_mb:.2f} MB exceeds 20 MB"
    assert vram_mb > 10.0, f"Projector VRAM {vram_mb:.2f} MB seems too small (check calculation)"
    print(f"  Projector VRAM: {vram_mb:.2f} MB")


def test_projector_no_bias():
    proj = CanonicalProjector(d_model=512, d_c=768)
    assert proj.W_q.bias is None, "W_q must not have bias"
    assert proj.W_o.bias is None, "W_o must not have bias"


# ---------------------------------------------------------------------------
# 2. MemoryCrossAttention
# ---------------------------------------------------------------------------

def test_cross_attn_shape_1d():
    attn = MemoryCrossAttention(d_c=768)
    q = torch.randn(768)         # (d_c,)
    mem = torch.randn(5, 768)    # (K=5, d_c)
    # Normalize
    q = q / q.norm()
    mem = mem / mem.norm(dim=-1, keepdim=True)
    M = attn.attend(q, mem)
    assert M.shape == (768,), f"Expected (768,), got {M.shape}"


def test_cross_attn_shape_batched():
    attn = MemoryCrossAttention(d_c=768)
    q = torch.randn(2, 10, 768)   # (batch, seq, d_c)
    q = q / q.norm(dim=-1, keepdim=True)
    mem = torch.randn(5, 768)
    mem = mem / mem.norm(dim=-1, keepdim=True)
    M = attn.attend(q, mem)
    assert M.shape == (2, 10, 768), f"Expected (2, 10, 768), got {M.shape}"


def test_cross_attn_zero_stall_empty_memory():
    attn = MemoryCrossAttention(d_c=768)
    q = torch.randn(768)
    mem = torch.empty(0, 768)   # K = 0: zero-stall
    M = attn.attend(q, mem)
    assert M.shape == q.shape
    assert torch.all(M == 0), "Zero-stall: M must be zero when K=0"


def test_cross_attn_weights_sum_to_one():
    attn = MemoryCrossAttention(d_c=768)
    q = torch.randn(768)
    q = q / q.norm()
    mem = torch.randn(4, 768)
    mem = mem / mem.norm(dim=-1, keepdim=True)
    _, weights = attn.attend_with_scores(q, mem)
    assert abs(weights.sum().item() - 1.0) < 1e-5, (
        f"Attention weights must sum to 1.0, got {weights.sum().item()}"
    )


def test_cross_attn_self_attention():
    """When query equals one memory vector, that memory gets highest weight."""
    attn = MemoryCrossAttention(d_c=768)
    mem = torch.randn(3, 768)
    mem = mem / mem.norm(dim=-1, keepdim=True)
    # Query = mem[1] exactly
    q = mem[1].clone()
    _, weights = attn.attend_with_scores(q, mem)
    assert weights.argmax().item() == 1, (
        f"Query identical to mem[1] must get highest weight, got argmax={weights.argmax()}"
    )


# ---------------------------------------------------------------------------
# 3. MemoryGate
# ---------------------------------------------------------------------------

def test_gate_initial_alpha_near_zero():
    gate = MemoryGate(d_model=512, mode="scalar", init_value=-4.0)
    alpha = gate.effective_alpha()
    # sigmoid(-4) ≈ 0.018
    assert alpha < 0.05, f"Initial alpha must be near 0, got {alpha}"


def test_gate_alpha_bounded_01():
    gate = MemoryGate(d_model=512, mode="scalar")
    alpha = gate.alpha.item()
    assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0, 1], got {alpha}"


def test_gate_scalar_forward():
    gate = MemoryGate(d_model=512, mode="scalar")
    h = torch.randn(2, 10, 512)
    delta = torch.randn(2, 10, 512)
    h_new = gate(h, delta)
    assert h_new.shape == h.shape


def test_gate_vector_mode():
    gate = MemoryGate(d_model=128, mode="vector")
    h = torch.randn(2, 128)
    delta = torch.randn(2, 128)
    h_new = gate(h, delta)
    assert h_new.shape == h.shape
    assert gate.raw_alpha.shape == (128,)


def test_gate_confidence_threshold_suppresses():
    gate = MemoryGate(d_model=512, confidence_threshold=0.75)
    # Manually open the gate
    with torch.no_grad():
        gate.raw_alpha.fill_(10.0)   # sigmoid(10) ≈ 1.0
    h = torch.randn(2, 512)
    delta = torch.randn(2, 512)
    # Similarity below threshold: gate must return h unchanged
    h_new = gate(h, delta, best_similarity=0.50)
    assert torch.allclose(h_new, h), "Gate must suppress injection when similarity < threshold"


def test_gate_zero_delta_no_change():
    gate = MemoryGate(d_model=512)
    with torch.no_grad():
        gate.raw_alpha.fill_(10.0)   # Fully open gate
    h = torch.randn(4, 512)
    delta = torch.zeros(4, 512)   # Zero memory delta
    h_new = gate(h, delta)
    assert torch.allclose(h_new, h), "Zero delta must leave h unchanged regardless of alpha"


# ---------------------------------------------------------------------------
# 4. QwenAdapter
# ---------------------------------------------------------------------------

def test_adapter_signature_format():
    adapter = QwenAdapter(d_model=4096, d_c=768, injection_layer=16)
    sig = adapter.get_model_signature()
    assert "qwen" in sig and "4096" in sig and "768" in sig and "L16" in sig, (
        f"Unexpected signature: {sig}"
    )


def test_adapter_parameter_count():
    adapter = QwenAdapter(d_model=4096, d_c=768)
    counts = adapter.parameter_count()
    # W_q: 4096*768 = 3,145,728
    # W_o: 768*4096 = 3,145,728
    # gate: 1
    assert counts["W_q"] == 4096 * 768
    assert counts["W_o"] == 768 * 4096
    assert counts["total"] == 2 * 4096 * 768 + 1
    print(f"  Total adapter parameters: {counts['total']:,}")


def test_adapter_inject_memory_pipeline():
    adapter = QwenAdapter(d_model=256, d_c=64, dtype=torch.float32)
    h = torch.randn(1, 10, 256)   # (batch, seq, d_model)
    mem = torch.randn(3, 64)
    mem = mem / mem.norm(dim=-1, keepdim=True)
    h_new, alpha = adapter.inject_memory(h, mem, best_similarity=0.9)
    assert h_new.shape == h.shape, f"Expected {h.shape}, got {h_new.shape}"
    assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0,1], got {alpha}"


def test_adapter_zero_stall_no_memory():
    adapter = QwenAdapter(d_model=256, d_c=64, dtype=torch.float32)
    h = torch.randn(2, 5, 256)
    mem = torch.empty(0, 64)   # K=0: zero-stall
    h_new, alpha = adapter.inject_memory(h, mem)
    assert torch.allclose(h_new, h), "Zero-stall: hidden state must be unchanged"
    assert alpha == 0.0, f"Zero-stall: alpha must be 0.0, got {alpha}"


def test_adapter_trainable_parameters_excludes_nothing_extra():
    adapter = QwenAdapter(d_model=256, d_c=64, dtype=torch.float32)
    trainable = adapter.trainable_parameters()
    # Must include W_q, W_o, and gate
    param_names = {
        id(p): name
        for name, p in adapter.named_parameters()
    }
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in adapter.parameters())
    assert n_trainable == n_total, (
        f"All adapter params should be trainable: {n_trainable} != {n_total}"
    )


def test_make_adapter_factory():
    adapter = make_adapter("qwen", d_model=512, injection_layer=8)
    assert isinstance(adapter, IModelAdapter)


def test_make_adapter_unsupported_raises():
    try:
        make_adapter("bert", d_model=512, injection_layer=8)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 5. MemoryHookManager
# ---------------------------------------------------------------------------

def _make_fake_model(d_model=256, num_layers=4):
    model = FakeLLM(num_layers=num_layers, d_model=d_model)
    # Freeze all base params
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_hook_manager_register_remove():
    model = _make_fake_model()
    adapter = QwenAdapter(d_model=256, d_c=64, injection_layer=2, dtype=torch.float32)
    manager = MemoryHookManager(model, adapter, layer_path="model.layers")

    assert not manager.is_registered
    manager.register()
    assert manager.is_registered
    manager.remove()
    assert not manager.is_registered


def test_hook_manager_double_register_raises():
    model = _make_fake_model()
    adapter = QwenAdapter(d_model=256, d_c=64, injection_layer=2, dtype=torch.float32)
    manager = MemoryHookManager(model, adapter, layer_path="model.layers")
    manager.register()
    try:
        manager.register()
        assert False, "Should raise RuntimeError on double registration"
    except RuntimeError:
        pass
    finally:
        manager.remove()


def test_hook_manager_zero_stall_empty_buffer():
    """With empty buffer, hook fires but returns hidden state unchanged."""
    model = _make_fake_model(d_model=256, num_layers=4)
    adapter = QwenAdapter(d_model=256, d_c=64, injection_layer=2, dtype=torch.float32)
    manager = MemoryHookManager(model, adapter, layer_path="model.layers")

    h_input = torch.randn(1, 5, 256)
    h_original = h_input.clone()

    with manager:
        # Buffer is empty by default
        h_out = model(h_input)

    assert torch.allclose(h_out, h_original), (
        "Empty buffer must leave hidden state unchanged (zero-stall)"
    )
    assert manager.last_alpha == 0.0


def test_hook_manager_fires_with_memory():
    """With a populated buffer, the hook modifies the hidden state."""
    d_model = 256
    d_c = 64
    model = _make_fake_model(d_model=d_model, num_layers=4)
    adapter = QwenAdapter(d_model=d_model, d_c=d_c, injection_layer=2, dtype=torch.float32)

    # Open the gate manually so injection has a visible effect
    with torch.no_grad():
        adapter.gate.raw_alpha.fill_(10.0)   # alpha ~ 1.0

    manager = MemoryHookManager(model, adapter, layer_path="model.layers")
    mem = torch.randn(3, d_c)
    mem = mem / mem.norm(dim=-1, keepdim=True)
    manager.set_memory_buffer(mem, best_similarity=0.9)

    h_input = torch.randn(1, 5, d_model)
    h_original = h_input.clone()

    with manager:
        h_out = model(h_input)

    # With alpha ~ 1.0 and non-zero memory, output must differ from input
    assert not torch.allclose(h_out, h_original, atol=1e-6), (
        "With open gate and non-zero memory, hidden state must be modified"
    )


def test_hook_manager_context_manager():
    model = _make_fake_model()
    adapter = QwenAdapter(d_model=256, d_c=64, injection_layer=1, dtype=torch.float32)
    manager = MemoryHookManager(model, adapter, layer_path="model.layers")
    with manager:
        assert manager.is_registered
    assert not manager.is_registered


def test_hook_manager_diagnostics():
    model = _make_fake_model()
    adapter = QwenAdapter(d_model=256, d_c=64, injection_layer=1, dtype=torch.float32)
    manager = MemoryHookManager(model, adapter, layer_path="model.layers")
    diag = manager.diagnostics()
    assert "injection_layer" in diag
    assert "adapter_signature" in diag
    assert "last_alpha" in diag


# ---------------------------------------------------------------------------
# 6. Integration: full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_integration():
    """End-to-end: canonical vectors -> adapter -> modified hidden state."""
    import numpy as np
    from unme.canonical.encoder import DeterministicHashEncoder
    from unme.canonical.store import CanonicalIndexPartition

    d_model = 256
    d_c = 768
    enc = DeterministicHashEncoder()

    # Build a small canonical index
    part = CanonicalIndexPartition(namespace_id=1, encoder_signature=enc.get_signature(), dim=d_c)
    for i, text in enumerate(["GNSS IMU fusion", "Kalman filter", "neural adapter"]):
        k, _ = enc.encode_record(text)
        part.add(0, i, 0, i, k)

    # Retrieve relevant memories for a query
    q_text = "IMU sensor fusion"
    q_vec = enc.encode_query(q_text)
    results = part.search(q_vec, top_k=3, threshold=0.0)

    # Convert CMR float16 vectors to torch tensor for adapter
    if results:
        mem_np = np.array([
            part._keys[r.cmr_record_index] for r in results
        ], dtype=np.float32)
        mem_tensor = torch.from_numpy(mem_np)   # (K, d_c)
    else:
        mem_tensor = torch.empty(0, d_c)

    # Run through adapter
    adapter = QwenAdapter(d_model=d_model, d_c=d_c, injection_layer=0, dtype=torch.float32)
    with torch.no_grad():
        adapter.gate.raw_alpha.fill_(5.0)   # Partially open gate

    h = torch.randn(1, 8, d_model)
    h_new, alpha = adapter.inject_memory(h, mem_tensor, best_similarity=0.9)

    assert h_new.shape == h.shape
    assert 0.0 < alpha < 1.0, f"Gate should be partially open, got alpha={alpha}"
    print(f"  Integration test: K={mem_tensor.shape[0]}, alpha={alpha:.4f}")


if __name__ == "__main__":
    import traceback

    test_functions = [
        test_projector_query_shape,
        test_projector_query_unit_norm,
        test_projector_output_shape,
        test_projector_vram_budget_gyro_coder,
        test_projector_no_bias,
        test_cross_attn_shape_1d,
        test_cross_attn_shape_batched,
        test_cross_attn_zero_stall_empty_memory,
        test_cross_attn_weights_sum_to_one,
        test_cross_attn_self_attention,
        test_gate_initial_alpha_near_zero,
        test_gate_alpha_bounded_01,
        test_gate_scalar_forward,
        test_gate_vector_mode,
        test_gate_confidence_threshold_suppresses,
        test_gate_zero_delta_no_change,
        test_adapter_signature_format,
        test_adapter_parameter_count,
        test_adapter_inject_memory_pipeline,
        test_adapter_zero_stall_no_memory,
        test_adapter_trainable_parameters_excludes_nothing_extra,
        test_make_adapter_factory,
        test_make_adapter_unsupported_raises,
        test_hook_manager_register_remove,
        test_hook_manager_double_register_raises,
        test_hook_manager_zero_stall_empty_buffer,
        test_hook_manager_fires_with_memory,
        test_hook_manager_context_manager,
        test_hook_manager_diagnostics,
        test_full_pipeline_integration,
    ]

    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests")
    if failed:
        import sys as _sys
        _sys.exit(1)
