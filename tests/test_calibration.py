"""Test suite for UNME Fase 3C: unme/calibration/

Covers:
  - CalibrationRecord binary format (header, CRC32, roundtrip)
  - GradientCalibrationRecord (gradient field, record_type)
  - CalibrationStore (append, scan, partial-write resilience)
  - ForwardCaptureSession (hook fires, shapes, float16 cast)
  - GradientCaptureSession (leaf creation, grad capture, record assembly)
  - Integration: session -> store -> scan -> verify
"""

import os
import pathlib
import struct
import tempfile
import time
import zlib

import numpy as np
import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unme.calibration.record import (
    CalibrationRecord,
    GradientCalibrationRecord,
    RecordType,
    UMRC_MAGIC,
    UMRC_VERSION,
    UMRC_HEADER_FMT,
    UMRC_HEADER_SIZE,
)
from unme.calibration.store import CalibrationStore
from unme.calibration.capture import ForwardCaptureSession, GradientCaptureSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64         # small d_model for fast tests (Gyro-Coder uses 4096)
D_C = 768
SEQ_LEN = 8
NUM_TOKENS = SEQ_LEN
SESSION_ID = 0xDEAD_BEEF_CAFE_1234
SIGNATURE = "test-model:L2:L4:v1"


def make_record(has_grad: bool = False) -> CalibrationRecord:
    rng = np.random.default_rng(42)
    h_Lq   = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
    h_Linj = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
    token_ids = np.arange(NUM_TOKENS, dtype=np.int32)
    base_kwargs = dict(
        h_Lq=h_Lq,
        h_Linj=h_Linj,
        token_ids=token_ids,
        model_signature=SIGNATURE,
        session_id=SESSION_ID,
        record_index=0,
        timestamp_ns=time.time_ns(),
    )
    if has_grad:
        g_inj = rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16)
        return GradientCalibrationRecord(**base_kwargs, g_inj=g_inj, loss_val=1.234)
    return CalibrationRecord(**base_kwargs)


class SmallTransformer(nn.Module):
    """Tiny two-layer transformer for hook testing (no LLM download needed)."""

    def __init__(self, d_model: int = D_MODEL, num_layers: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


# ---------------------------------------------------------------------------
# UMRC header tests
# ---------------------------------------------------------------------------

def test_header_size_is_64():
    assert UMRC_HEADER_SIZE == 64


def test_magic_and_version():
    assert UMRC_MAGIC == b"UMRC"
    assert UMRC_VERSION == 1


def test_record_type_enum():
    assert RecordType.FORWARD_CAPTURE == 1
    assert RecordType.GRADIENT_CALIBRATION == 2


# ---------------------------------------------------------------------------
# CalibrationRecord serialization
# ---------------------------------------------------------------------------

def test_forward_record_roundtrip():
    rec = make_record(has_grad=False)
    raw = rec.to_bytes()
    restored = CalibrationRecord.from_bytes(raw)

    assert isinstance(restored, CalibrationRecord)
    assert not isinstance(restored, GradientCalibrationRecord)
    assert restored.model_signature == SIGNATURE
    assert restored.session_id == SESSION_ID
    assert restored.record_index == 0
    np.testing.assert_array_equal(restored.h_Lq,    rec.h_Lq)
    np.testing.assert_array_equal(restored.h_Linj,  rec.h_Linj)
    np.testing.assert_array_equal(restored.token_ids, rec.token_ids)


def test_gradient_record_roundtrip():
    rec = make_record(has_grad=True)
    assert isinstance(rec, GradientCalibrationRecord)

    raw = rec.to_bytes()
    restored = CalibrationRecord.from_bytes(raw)

    assert isinstance(restored, GradientCalibrationRecord)
    assert restored.record_type == RecordType.GRADIENT_CALIBRATION
    np.testing.assert_array_equal(restored.g_inj, rec.g_inj)
    assert abs(restored.loss_val - 1.234) < 1e-3   # float32 precision


def test_record_type_forward():
    assert make_record(has_grad=False).record_type == RecordType.FORWARD_CAPTURE


def test_record_type_gradient():
    assert make_record(has_grad=True).record_type == RecordType.GRADIENT_CALIBRATION


def test_header_starts_with_magic():
    raw = make_record().to_bytes()
    assert raw[:4] == UMRC_MAGIC


def test_crc32_is_correct():
    rec = make_record()
    raw = rec.to_bytes()
    # Header stores CRC at offset +47, covering everything after the header
    (_, _, _, _, _, _, _, _, _, _, _, _, _, _, stored_crc, _) = struct.unpack_from(
        UMRC_HEADER_FMT, raw, 0
    )
    payload = raw[UMRC_HEADER_SIZE:]
    computed = zlib.crc32(payload) & 0xFFFF_FFFF
    assert stored_crc == computed


def test_crc32_corruption_raises():
    rec = make_record()
    raw = bytearray(rec.to_bytes())
    # Flip a byte well inside h_Lq (past header + signature) to guarantee
    # it hits float16 payload bytes, not the UTF-8 signature region.
    sig_len = len(rec.model_signature.encode("utf-8"))
    payload_offset = UMRC_HEADER_SIZE + sig_len + 10  # 10 bytes into h_Lq
    raw[payload_offset] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        CalibrationRecord.from_bytes(bytes(raw))


def test_wrong_magic_raises():
    raw = bytearray(make_record().to_bytes())
    raw[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        CalibrationRecord.from_bytes(bytes(raw))


def test_shape_properties():
    rec = make_record()
    assert rec.d_model    == D_MODEL
    assert rec.seq_len_q  == SEQ_LEN
    assert rec.seq_len_inj == SEQ_LEN
    assert rec.num_tokens == NUM_TOKENS


def test_validate_dtype_enforcement():
    rng = np.random.default_rng(0)
    # h_Lq as float32 — should raise on to_bytes
    rec = CalibrationRecord(
        h_Lq=rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float32),
        h_Linj=rng.standard_normal((SEQ_LEN, D_MODEL)).astype(np.float16),
        token_ids=np.zeros(NUM_TOKENS, dtype=np.int32),
        model_signature=SIGNATURE,
        session_id=0,
        record_index=0,
    )
    with pytest.raises(AssertionError):
        rec.to_bytes()


def test_minimum_data_roundtrip():
    """Single token, single-position sequence — edge case."""
    rng = np.random.default_rng(7)
    rec = CalibrationRecord(
        h_Lq=rng.standard_normal((1, D_MODEL)).astype(np.float16),
        h_Linj=rng.standard_normal((1, D_MODEL)).astype(np.float16),
        token_ids=np.array([42], dtype=np.int32),
        model_signature="min-test",
        session_id=1,
        record_index=0,
    )
    restored = CalibrationRecord.from_bytes(rec.to_bytes())
    assert restored.num_tokens == 1
    assert restored.seq_len_q == 1


# ---------------------------------------------------------------------------
# CalibrationStore tests
# ---------------------------------------------------------------------------

def test_store_append_and_scan():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "test.umrc"
        records = [make_record() for _ in range(5)]
        for i, r in enumerate(records):
            r.record_index = i

        with CalibrationStore(path) as store:
            for r in records:
                store.append(r)
            assert store.record_count == 5

        scanned = list(CalibrationStore.scan(path))
        assert len(scanned) == 5
        for i, s in enumerate(scanned):
            assert s.record_index == i
            np.testing.assert_array_equal(s.h_Lq, records[i].h_Lq)


def test_store_mixed_record_types():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "mixed.umrc"
        r_fwd  = make_record(has_grad=False)
        r_grad = make_record(has_grad=True)
        r_fwd.record_index  = 0
        r_grad.record_index = 1

        with CalibrationStore(path) as store:
            store.append(r_fwd)
            store.append(r_grad)

        scanned = list(CalibrationStore.scan(path))
        assert len(scanned) == 2
        assert scanned[0].record_type == RecordType.FORWARD_CAPTURE
        assert scanned[1].record_type == RecordType.GRADIENT_CALIBRATION
        assert isinstance(scanned[1], GradientCalibrationRecord)


def test_store_scan_empty_file():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "empty.umrc"
        path.touch()
        scanned = list(CalibrationStore.scan(path))
        assert scanned == []


def test_store_scan_nonexistent_file():
    path = pathlib.Path("/nonexistent/path/file.umrc")
    scanned = list(CalibrationStore.scan(path))
    assert scanned == []


def test_store_partial_write_resilience():
    """A truncated record at end of file must not corrupt earlier records."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "partial.umrc"

        good = make_record()
        good.record_index = 0
        with CalibrationStore(path) as store:
            store.append(good)

        # Append garbage (simulates partial write)
        with open(path, "ab") as f:
            f.write(UMRC_MAGIC + b"\x00" * 30)   # incomplete record

        scanned = list(CalibrationStore.scan(path))
        assert len(scanned) == 1  # only the complete record
        assert scanned[0].record_index == 0


def test_store_file_size_grows():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "grow.umrc"
        with CalibrationStore(path) as store:
            assert store.file_size_bytes == 0
            store.append(make_record())
            size1 = store.file_size_bytes
            store.append(make_record())
            size2 = store.file_size_bytes
        assert size2 > size1 > 0


def test_store_persist_across_sessions():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "persist.umrc"
        rec = make_record()
        rec.record_index = 99

        with CalibrationStore(path) as s1:
            s1.append(rec)

        # Open again — record_count should reflect existing records
        with CalibrationStore(path) as s2:
            assert s2.record_count == 1


def test_store_raises_if_not_open():
    import tempfile
    path = pathlib.Path(tempfile.gettempdir()) / "unme_test_not_open.umrc"
    store = CalibrationStore(path)
    with pytest.raises(RuntimeError, match="not open"):
        store.append(make_record())


# ---------------------------------------------------------------------------
# ForwardCaptureSession tests
# ---------------------------------------------------------------------------

def test_forward_session_captures_shapes():
    model = SmallTransformer(d_model=D_MODEL, num_layers=6)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model,
        model_layers=model.layers,
        layer_q=2,
        layer_inj=4,
        model_signature=SIGNATURE,
        session_id=SESSION_ID,
    ) as sess:
        with torch.no_grad():
            _ = model(x)
        token_ids = np.arange(NUM_TOKENS, dtype=np.int32)
        record = sess.get_record(token_ids=token_ids, record_index=0)

    assert record.h_Lq.shape   == (SEQ_LEN, D_MODEL)
    assert record.h_Linj.shape == (SEQ_LEN, D_MODEL)
    assert record.h_Lq.dtype   == np.float16
    assert record.h_Linj.dtype == np.float16


def test_forward_session_signature_stored():
    model = SmallTransformer()
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature="my-sig", session_id=42,
    ) as sess:
        with torch.no_grad():
            model(x)
        record = sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)

    assert record.model_signature == "my-sig"
    assert record.session_id == 42


def test_forward_session_no_grad_on_model():
    """Model weights must not receive gradients during capture."""
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        with torch.no_grad():
            model(x)
        sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)

    for p in model.parameters():
        assert p.grad is None, "LLM weights must not accumulate gradients"


def test_forward_session_raises_before_forward():
    model = SmallTransformer()
    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        with pytest.raises(RuntimeError, match="No activations captured"):
            sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)


def test_forward_session_hooks_removed_after_exit():
    model = SmallTransformer()
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        with torch.no_grad():
            model(x)

    # After exit, internal hook list must be empty
    assert len(sess._hooks) == 0


def test_forward_session_reset():
    model = SmallTransformer()
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        with torch.no_grad():
            model(x)
        sess.reset()
        with pytest.raises(RuntimeError):
            sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)


def test_forward_session_record_is_valid_umrc():
    """The record produced by ForwardCaptureSession must pass to_bytes/from_bytes."""
    model = SmallTransformer()
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with ForwardCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=2, layer_inj=4,
        model_signature=SIGNATURE, session_id=SESSION_ID,
    ) as sess:
        with torch.no_grad():
            model(x)
        record = sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 7)

    raw = record.to_bytes()
    restored = CalibrationRecord.from_bytes(raw)
    assert restored.record_index == 7
    np.testing.assert_array_equal(restored.h_Lq, record.h_Lq)


# ---------------------------------------------------------------------------
# GradientCaptureSession tests
# ---------------------------------------------------------------------------

def test_gradient_session_captures_grad():
    """g_inj must be non-zero and have correct shape after backward()."""
    model = SmallTransformer(d_model=D_MODEL, num_layers=6)
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with GradientCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=2, layer_inj=4,
        model_signature=SIGNATURE, session_id=SESSION_ID,
    ) as sess:
        output = model(x)                       # forward — hook fires here
        loss = output.sum()                     # trivial task loss
        loss.backward()
        record = sess.get_record(
            token_ids=np.arange(SEQ_LEN, dtype=np.int32),
            record_index=0,
            loss_val=loss.item(),
        )

    assert isinstance(record, GradientCalibrationRecord)
    assert record.g_inj.shape  == (SEQ_LEN, D_MODEL)
    assert record.g_inj.dtype  == np.float16
    assert record.h_Linj.shape == (SEQ_LEN, D_MODEL)
    # Gradient must be non-zero (loss.sum() -> grad is all-ones downstream)
    assert np.any(record.g_inj != 0)


def test_gradient_session_loss_val_stored():
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with GradientCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        output = model(x)
        loss = output.mean()
        loss.backward()
        record = sess.get_record(
            token_ids=np.arange(SEQ_LEN, dtype=np.int32),
            record_index=0,
            loss_val=loss.item(),
        )

    assert abs(record.loss_val - loss.item()) < 1e-3


def test_gradient_session_record_type():
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with GradientCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        output = model(x)
        output.sum().backward()
        record = sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)

    assert record.record_type == RecordType.GRADIENT_CALIBRATION


def test_gradient_session_llm_weights_unmodified():
    """LLM weights must NOT accumulate gradients from backward()."""
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with GradientCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        output = model(x)
        output.sum().backward()
        sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)

    for p in model.parameters():
        assert p.grad is None, "LLM weights must not receive gradients"


def test_gradient_session_raises_without_backward():
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with GradientCaptureSession(
        model=model, model_layers=model.layers,
        layer_q=1, layer_inj=3,
        model_signature=SIGNATURE, session_id=0,
    ) as sess:
        _ = model(x)
        # backward() NOT called
        with pytest.raises(RuntimeError, match="grad is None"):
            sess.get_record(np.arange(SEQ_LEN, dtype=np.int32), 0)


def test_gradient_session_roundtrip_via_store():
    """Full pipeline: GradientCaptureSession -> CalibrationStore -> scan."""
    model = SmallTransformer()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, SEQ_LEN, D_MODEL)

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "grad.umrc"

        with GradientCaptureSession(
            model=model, model_layers=model.layers,
            layer_q=1, layer_inj=3,
            model_signature=SIGNATURE, session_id=SESSION_ID,
        ) as sess:
            output = model(x)
            loss = output.sum()
            loss.backward()
            record = sess.get_record(
                np.arange(SEQ_LEN, dtype=np.int32), 0, loss.item()
            )

        with CalibrationStore(path) as store:
            store.append(record)

        scanned = list(CalibrationStore.scan(path))
        assert len(scanned) == 1
        r = scanned[0]
        assert isinstance(r, GradientCalibrationRecord)
        assert r.g_inj.shape == (SEQ_LEN, D_MODEL)
        np.testing.assert_array_equal(r.h_Lq, record.h_Lq)
        np.testing.assert_array_equal(r.g_inj, record.g_inj)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_full_calibration_pipeline():
    """ForwardCaptureSession x5 -> store -> scan -> verify all 5 records."""
    model = SmallTransformer(d_model=D_MODEL, num_layers=6)
    for p in model.parameters():
        p.requires_grad_(False)

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "pipeline.umrc"
        captured = []

        with CalibrationStore(path) as store:
            for i in range(5):
                x = torch.randn(1, SEQ_LEN, D_MODEL)
                with ForwardCaptureSession(
                    model=model, model_layers=model.layers,
                    layer_q=2, layer_inj=4,
                    model_signature=SIGNATURE, session_id=SESSION_ID,
                ) as sess:
                    with torch.no_grad():
                        model(x)
                    rec = sess.get_record(
                        np.arange(SEQ_LEN, dtype=np.int32), record_index=i
                    )
                store.append(rec)
                captured.append(rec)

        scanned = list(CalibrationStore.scan(path))
        assert len(scanned) == 5
        for i, s in enumerate(scanned):
            assert s.record_index == i
            assert s.record_type == RecordType.FORWARD_CAPTURE
            np.testing.assert_array_equal(s.h_Lq, captured[i].h_Lq)


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "--tb=short"])
