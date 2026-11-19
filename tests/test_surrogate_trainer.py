"""Test suite for UNME Fase 3D: unme/training/surrogate_trainer.py

Covers:
  - TrainerConfig defaults and construction
  - SurrogateTrainer construction (adapter in fp32, optimizer, memory)
  - train_on_record: loss decreases, alignment increases over steps
  - Frozen LLM guarantee: only W_q, W_o, alpha receive gradients
  - Gate alpha starts near-zero (sigmoid(-4) ~ 0.018)
  - train_epoch: metrics aggregation, skip non-gradient records
  - train_from_store: reads .umrc file end-to-end
  - Checkpoint save/load: state_dict roundtrip, global_step preserved
  - export_fp16: all weights cast to float16
  - Zero-memory edge case: returns zero loss without crash
  - Surrogate loss formula: L = -<g, delta> + lambda * ||delta||^2
"""

import pathlib
import tempfile
import time

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unme.calibration.record import (
    CalibrationRecord,
    GradientCalibrationRecord,
)
from unme.calibration.store import CalibrationStore
from unme.training.surrogate_trainer import SurrogateTrainer, TrainerConfig, EpochMetrics


# ---------------------------------------------------------------------------
# Constants for tests (small d_model to run fast on CPU)
# ---------------------------------------------------------------------------

D_MODEL = 64
D_C = 32          # small canonical dim for speed; production is 768
SEQ_LEN = 4
NUM_TOKENS = SEQ_LEN
K_MEM = 8         # number of memory vectors


def make_config(tmpdir: str) -> TrainerConfig:
    return TrainerConfig(
        d_model=D_MODEL,
        d_c=D_C,
        injection_layer=2,
        architecture="qwen",
        gate_mode="scalar",
        lambda_reg=0.01,
        lr=1e-3,
        grad_clip=1.0,
        checkpoint_dir=tmpdir,
        log_every=1000,
    )


def make_memory_values(K: int = K_MEM) -> np.ndarray:
    rng = np.random.default_rng(0)
    V = rng.standard_normal((K, D_C)).astype(np.float32)
    # L2-normalize rows (matches canonical store convention)
    norms = np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-8)
    return (V / norms).astype(np.float16)


def make_grad_record(record_index: int = 0, seed: int = 42) -> GradientCalibrationRecord:
    rng = np.random.default_rng(seed)
    h_Lq   = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
    h_Linj = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
    g_inj  = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
    return GradientCalibrationRecord(
        h_Lq=h_Lq,
        h_Linj=h_Linj,
        token_ids=np.arange(NUM_TOKENS, dtype=np.int32),
        model_signature="test-model:L1:L2",
        session_id=0xCAFE,
        record_index=record_index,
        g_inj=g_inj,
        loss_val=1.0,
    )


def make_fwd_record() -> CalibrationRecord:
    """Forward-only record (no gradient) — should be skipped by trainer."""
    rng = np.random.default_rng(99)
    return CalibrationRecord(
        h_Lq=rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16),
        h_Linj=rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16),
        token_ids=np.arange(NUM_TOKENS, dtype=np.int32),
        model_signature="test-model:L1:L2",
        session_id=1,
        record_index=0,
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_trainer_constructs():
    with tempfile.TemporaryDirectory() as td:
        cfg = make_config(td)
        trainer = SurrogateTrainer(cfg, make_memory_values())
        assert trainer.global_step == 0
        assert trainer.memory_size == K_MEM


def test_adapter_in_fp32():
    """Adapter is built in float32 for numerically stable training."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        for p in trainer.adapter.projector.parameters():
            assert p.dtype == torch.float32, f"Expected float32, got {p.dtype}"


def test_only_trainable_params_have_grad():
    """W_q, W_o, alpha require grad; nothing else does."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        trainable_ids = {id(p) for p in trainer.adapter.trainable_parameters()}
        for p in trainer.adapter.parameters():
            if id(p) in trainable_ids:
                assert p.requires_grad, "Trainable param must require grad"
            else:
                assert not p.requires_grad, "Non-trainable param must NOT require grad"


def test_gate_alpha_starts_near_zero():
    """Gate init=-4.0 -> sigmoid(-4) ~ 0.018."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        alpha = trainer.effective_alpha()
        assert 0.0 < alpha < 0.05, f"Expected alpha near 0, got {alpha:.4f}"


def test_checkpoint_dir_created():
    with tempfile.TemporaryDirectory() as td:
        ckpt_dir = pathlib.Path(td) / "sub" / "ckpt"
        cfg = make_config(str(ckpt_dir))
        SurrogateTrainer(cfg, make_memory_values())
        assert ckpt_dir.exists()


# ---------------------------------------------------------------------------
# train_on_record tests
# ---------------------------------------------------------------------------

def test_train_on_record_returns_metrics():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        metrics = trainer.train_on_record(make_grad_record())
        assert set(metrics.keys()) == {"loss", "alignment", "regularization", "delta_norm"}
        assert all(isinstance(v, float) for v in metrics.values())


def test_global_step_increments():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        for i in range(5):
            trainer.train_on_record(make_grad_record(i))
        assert trainer.global_step == 5


def test_loss_decreases_over_steps():
    """Loss should decrease when training repeatedly on the same record."""
    with tempfile.TemporaryDirectory() as td:
        cfg = make_config(td)
        cfg.lr = 1e-2    # higher lr for faster convergence in test
        cfg.lambda_reg = 0.0   # pure alignment loss
        trainer = SurrogateTrainer(cfg, make_memory_values())
        record = make_grad_record(0)

        losses = []
        for _ in range(30):
            m = trainer.train_on_record(record)
            losses.append(m["loss"])

        # Loss at end should be lower than at start (not necessarily monotone due to
        # the stochastic optimizer, but the trend must be downward)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
        )


def test_alignment_increases_over_steps():
    """Alignment <g, delta> should increase (become more positive) over training."""
    with tempfile.TemporaryDirectory() as td:
        cfg = make_config(td)
        cfg.lr = 1e-2
        cfg.lambda_reg = 0.0
        trainer = SurrogateTrainer(cfg, make_memory_values())
        record = make_grad_record(0)

        alignments = []
        for _ in range(30):
            m = trainer.train_on_record(record)
            alignments.append(m["alignment"])

        assert alignments[-1] > alignments[0], (
            f"Alignment did not increase: start={alignments[0]:.4f} end={alignments[-1]:.4f}"
        )


def test_regularization_term_is_non_negative():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        for i in range(5):
            m = trainer.train_on_record(make_grad_record(i))
            assert m["regularization"] >= 0.0


def test_loss_formula_correctness():
    """L = -alignment + lambda * regularization."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        m = trainer.train_on_record(make_grad_record())
        expected = -m["alignment"] + make_config(td).lambda_reg * (m["delta_norm"] ** 2)
        assert abs(m["loss"] - expected) < 1e-3, (
            f"Loss formula mismatch: got {m['loss']:.6f}, expected ~{expected:.6f}"
        )


def test_zero_memory_returns_zero_loss():
    """With K=0 memory vectors, loss must be 0.0 without crashing."""
    with tempfile.TemporaryDirectory() as td:
        empty_mem = np.zeros((0, D_C), dtype=np.float16)
        trainer = SurrogateTrainer(make_config(td), empty_mem)
        m = trainer.train_on_record(make_grad_record())
        assert m["loss"] == 0.0


def test_llm_weights_do_not_accumulate_grad():
    """After train_on_record, only W_q/W_o/alpha should have .grad populated."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        trainer.train_on_record(make_grad_record())

        trainable_ids = {id(p) for p in trainer.adapter.trainable_parameters()}
        for p in trainer.adapter.parameters():
            if id(p) not in trainable_ids:
                assert p.grad is None, (
                    f"Non-trainable param has grad: {p.shape}"
                )


# ---------------------------------------------------------------------------
# train_epoch tests
# ---------------------------------------------------------------------------

def test_epoch_metrics_type():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        result = trainer.train_epoch([make_grad_record(i) for i in range(5)])
        assert isinstance(result, EpochMetrics)


def test_epoch_counts_records():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        records = [make_grad_record(i) for i in range(10)]
        m = trainer.train_epoch(records)
        assert m.n_records == 10
        assert m.n_skipped == 0


def test_epoch_skips_forward_records():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        records = [make_grad_record(0), make_fwd_record(), make_grad_record(1)]
        m = trainer.train_epoch(records)
        assert m.n_records == 2
        assert m.n_skipped == 1


def test_epoch_elapsed_time_positive():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        m = trainer.train_epoch([make_grad_record(0)])
        assert m.elapsed_seconds > 0.0


def test_epoch_mean_loss_property():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        m = trainer.train_epoch([make_grad_record(i) for i in range(4)])
        assert m.mean_loss == m.total_loss / 4


# ---------------------------------------------------------------------------
# train_from_store tests
# ---------------------------------------------------------------------------

def test_train_from_store():
    with tempfile.TemporaryDirectory() as td:
        umrc_path = pathlib.Path(td) / "cal.umrc"
        with CalibrationStore(umrc_path) as store:
            for i in range(6):
                store.append(make_grad_record(i))

        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        m = trainer.train_from_store(umrc_path)
        assert m.n_records == 6


def test_train_from_nonexistent_store():
    """Scanning a nonexistent file yields 0 records — no crash."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        m = trainer.train_from_store(pathlib.Path(td) / "ghost.umrc")
        assert m.n_records == 0


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

def test_save_and_load_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        cfg = make_config(td)
        trainer = SurrogateTrainer(cfg, make_memory_values())

        # Train a few steps so weights differ from init
        for i in range(5):
            trainer.train_on_record(make_grad_record(i))

        step_before = trainer.global_step
        wq_before = trainer.adapter.projector.W_q.weight.clone()

        path = trainer.save_checkpoint("test_ckpt")
        assert path.exists()

        # Build a fresh trainer and load
        trainer2 = SurrogateTrainer(cfg, make_memory_values())
        loaded_step = trainer2.load_checkpoint("test_ckpt")

        assert loaded_step == step_before
        assert trainer2.global_step == step_before
        torch.testing.assert_close(
            trainer2.adapter.projector.W_q.weight, wq_before
        )


def test_checkpoint_file_created():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        path = trainer.save_checkpoint("my_ckpt")
        assert (pathlib.Path(td) / "my_ckpt.pt").exists()


def test_load_nonexistent_checkpoint_raises():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("ghost")


# ---------------------------------------------------------------------------
# export_fp16 tests
# ---------------------------------------------------------------------------

def test_export_fp16_dtype():
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        trainer.train_on_record(make_grad_record())
        path = trainer.export_fp16()

        state = torch.load(path, map_location="cpu", weights_only=True)
        for k, v in state["adapter_state_dict"].items():
            assert v.dtype == torch.float16, (
                f"Key {k}: expected float16, got {v.dtype}"
            )


def test_export_fp16_values_close_to_fp32():
    """fp16 export must be numerically close to the fp32 master weights."""
    with tempfile.TemporaryDirectory() as td:
        trainer = SurrogateTrainer(make_config(td), make_memory_values())
        trainer.train_on_record(make_grad_record())

        path = trainer.export_fp16()
        state_fp16 = torch.load(path, map_location="cpu", weights_only=True)
        state_fp32 = trainer.adapter.state_dict()

        for k in state_fp16["adapter_state_dict"]:
            v16 = state_fp16["adapter_state_dict"][k].float()
            v32 = state_fp32[k]
            # fp16 has ~3 decimal digits of precision; allow generous tolerance
            torch.testing.assert_close(v16, v32, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_full_training_pipeline():
    """End-to-end: capture -> store -> load -> train -> checkpoint -> reload."""
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL) for _ in range(6)])
        def forward(self, x):
            h = x
            for layer in self.layers:
                h = layer(h)
            return h

    from unme.calibration.capture import GradientCaptureSession

    model = TinyModel()
    for p in model.parameters():
        p.requires_grad_(False)

    with tempfile.TemporaryDirectory() as td:
        umrc_path = pathlib.Path(td) / "cal.umrc"

        # Phase 1: capture 10 gradient records
        with CalibrationStore(umrc_path) as store:
            for i in range(10):
                x = torch.randn(1, SEQ_LEN, D_MODEL)
                with GradientCaptureSession(
                    model=model, model_layers=model.layers,
                    layer_q=2, layer_inj=4,
                    model_signature="tiny:L2:L4", session_id=0xABCD,
                ) as sess:
                    out = model(x)
                    out.sum().backward()
                    rec = sess.get_record(
                        np.arange(SEQ_LEN, dtype=np.int32), i, out.sum().item()
                    )
                store.append(rec)

        # Phase 2: train adapter from .umrc
        cfg = TrainerConfig(
            d_model=D_MODEL, d_c=D_C, injection_layer=4,
            architecture="qwen", gate_mode="scalar",
            lr=1e-3, lambda_reg=0.01, checkpoint_dir=td, log_every=10000,
        )
        trainer = SurrogateTrainer(cfg, make_memory_values())
        m = trainer.train_from_store(umrc_path)
        assert m.n_records == 10

        # Phase 3: save checkpoint and reload
        ckpt_path = trainer.save_checkpoint("final")
        step = trainer.global_step

        trainer2 = SurrogateTrainer(cfg, make_memory_values())
        loaded = trainer2.load_checkpoint("final")
        assert loaded == step

        # Phase 4: export fp16
        fp16_path = trainer.export_fp16("final_fp16")
        assert fp16_path.exists()
        print(f"Integration test PASS: {m.n_records} records, "
              f"mean_loss={m.mean_loss:.4f}, global_step={step}")


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
