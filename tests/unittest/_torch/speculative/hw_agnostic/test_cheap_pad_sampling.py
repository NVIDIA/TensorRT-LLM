# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph padding rows of one-model rejection sampling.

With ``set_cheap_padding_rows(True)`` padding rows get a one-hot logits row and ``_uniform_padding_probs`` restores,
after the filters, the uniform probs zeroed logits give; real rows are untouched. A target may also install
``set_padding_rows_fill_override``, which must equal the stock fills bit for bit and falls back to them on None.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

DUMMY_ROW = 9


@pytest.fixture(autouse=True)
def cheap_rows(monkeypatch):
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", True)
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", None)


def _meta(slot_ids, *, verify_lens=None):
    meta = types.SimpleNamespace(
        batch_slot_ids=torch.tensor(slot_ids, dtype=torch.long),
        dummy_slot_row=DUMMY_ROW,
        is_ragged_verify=verify_lens is not None,
        verify_lens=None,
        total_verify_tokens=None,
    )
    if verify_lens is not None:
        meta.verify_lens = torch.tensor(verify_lens, dtype=torch.int32)
        meta.total_verify_tokens = int(sum(verify_lens))
    return meta


def _zeroed_padding(x, meta, rows_per_request):
    """Reference: padding rows zeroed, as uniform probs come from."""
    mask = SpecWorkerBase._padding_row_mask(
        meta, x.shape[0], 0, len(meta.batch_slot_ids), rows_per_request
    )
    return x.masked_fill(mask.unsqueeze(-1), 0.0)


def test_off_the_rows_are_only_zeroed(monkeypatch):
    interface.set_cheap_padding_rows(False)
    assert interface._CHEAP_PADDING_ROWS is False
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    meta = _meta([0, 1, DUMMY_ROW])
    out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2)
    assert torch.equal(out[:4], logits[:4])
    assert torch.equal(out[4:], torch.zeros(2, 11))
    probs = torch.softmax(torch.randn(6, 11), -1)
    original = probs.clone()
    assert torch.equal(SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2), original)


def test_mask_layouts():
    meta = _meta([0, 1, DUMMY_ROW])
    mask = SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 2)
    assert mask.tolist() == [False, False, False, False, True, True]
    # Rows that do not match the layout, or an unpublished dummy row, are not masked.
    assert SpecWorkerBase._padding_row_mask(meta, 5, 0, 3, 2) is None
    meta.dummy_slot_row = 0
    assert SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 2) is None
    # Ragged verify, and a context prefix.
    meta = _meta([0, 1, DUMMY_ROW], verify_lens=[3, 1, 2])
    mask = SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 4, packed_verify=True)
    assert mask.tolist() == [False] * 4 + [True] * 2
    mask = SpecWorkerBase._padding_row_mask(_meta([5, 0, DUMMY_ROW]), 2, 1, 3, 1)
    assert mask.tolist() == [False, True]


def test_padding_logits_and_uniform_probs():
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    meta = _meta([0, 1, DUMMY_ROW])
    out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2)
    assert torch.equal(out[:4], logits[:4])
    expected = torch.zeros(2, 11)
    expected[:, 0] = interface._PADDING_ONEHOT_LOGIT
    assert torch.equal(out[4:], expected)
    probs = torch.softmax(torch.randn(6, 11), -1)
    original = probs.clone()
    out = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2)
    assert torch.equal(out[:4], original[:4])
    assert torch.equal(out[4:], torch.full((2, 11), 1.0 / 11))


@pytest.mark.parametrize("temperature", [1.0e-5, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_onehot_logits_are_onehot_after_softmax(temperature, dtype):
    logits = torch.randn(2, 200064).to(dtype)
    out = SpecWorkerBase._zero_padding_rows(logits, _meta([0, DUMMY_ROW]), 0, 2, 1)
    probs = torch.softmax(out[1].float() / temperature, -1)
    assert probs[0].item() == 1.0
    assert torch.count_nonzero(probs).item() == 1


def test_cheap_pad_matches_zeroed_logits_probs():
    torch.manual_seed(0)
    logits = torch.randn(9, 37)
    temperatures = torch.tensor([0.7] * 6 + [1.0e-5] * 3)
    meta = _meta([0, 1, DUMMY_ROW])

    off = torch.softmax(_zeroed_padding(logits, meta, 3) / temperatures[:, None], -1)
    masked = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 3)
    probs = torch.softmax(masked / temperatures[:, None], -1)
    on = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 3)
    assert torch.equal(on[:6], off[:6])
    torch.testing.assert_close(on[6:], off[6:], rtol=0, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_real_rows_bitwise_with_flashinfer_top_p():
    """Real rows' filtered probs do not depend on the padding rows' contents."""
    flashinfer_sampling = pytest.importorskip(
        "tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer"
    )
    torch.manual_seed(0)
    vocab = 200064
    logits = torch.randn(14, vocab, device="cuda") * 3
    meta = _meta([0, DUMMY_ROW])
    meta.batch_slot_ids = meta.batch_slot_ids.cuda()
    temperatures = torch.tensor([1.0] * 7 + [1.0e-5] * 7, device="cuda")
    top_ps = torch.tensor([0.95] * 7 + [1.0] * 7, device="cuda")

    off = flashinfer_sampling.compute_probs_from_logits_op(
        _zeroed_padding(logits, meta, 7), temperatures, None, top_ps
    )
    masked = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 2, 7)
    probs = flashinfer_sampling.compute_probs_from_logits_op(masked, temperatures, None, top_ps)
    on = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 2, 7)
    assert torch.equal(on[:7], off[:7])
    torch.testing.assert_close(on[7:], off[7:], rtol=0, atol=1e-9)


class _RecordingFill:
    """An in-place fill written with index ops, recording each call."""

    def __init__(self, decline=False):
        self.calls = []
        self.decline = decline

    def __call__(self, rows, is_padding, value, col0_value):
        self.calls.append((rows.data_ptr(), is_padding.tolist(), value, col0_value))
        if self.decline:
            return None
        idx = is_padding.nonzero().flatten()
        rows.index_fill_(0, idx, value)
        if col0_value is not None:
            rows[idx, 0] = col0_value
        return rows


@pytest.mark.parametrize("cheap", [True, False])
def test_fill_override_matches_the_stock_fills(monkeypatch, cheap):
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", cheap)
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    probs = torch.softmax(torch.randn(6, 11), -1)
    meta = _meta([0, 1, DUMMY_ROW])
    stock = SpecWorkerBase._zero_padding_rows(logits.clone(), meta, 0, 3, 2)
    stock_probs = SpecWorkerBase._uniform_padding_probs(probs.clone(), meta, 0, 3, 2)
    fill = _RecordingFill()
    interface.set_padding_rows_fill_override(fill)
    x = logits.clone()
    out = SpecWorkerBase._zero_padding_rows(x, meta, 0, 3, 2)
    # The fill runs on a copy: the caller's logits are not mutated, as with the stock fill.
    assert out.data_ptr() != x.data_ptr() and torch.equal(x, logits)
    assert torch.equal(out.view(torch.int32), stock.view(torch.int32))
    mask = [False] * 4 + [True] * 2
    assert fill.calls == [
        (out.data_ptr(), mask, 0.0, interface._PADDING_ONEHOT_LOGIT if cheap else None)
    ]
    if cheap:
        out_probs = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2)
        assert torch.equal(out_probs.view(torch.int32), stock_probs.view(torch.int32))
        assert fill.calls[1] == (probs.data_ptr(), mask, 1.0 / 11, None)


def test_a_declined_fill_runs_the_stock_fills(monkeypatch):
    declining = _RecordingFill(decline=True)
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", declining)
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    original = logits.clone()
    meta = _meta([0, 1, DUMMY_ROW])
    out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2)
    # The stock fill is out of place: the input keeps its padding rows.
    assert torch.equal(logits, original)
    expected = original.clone()
    expected[4:] = 0.0
    expected[4:, 0] = interface._PADDING_ONEHOT_LOGIT
    assert torch.equal(out, expected)
    probs = torch.softmax(torch.randn(6, 11), -1)
    out = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2)
    assert torch.equal(out[4:], torch.full((2, 11), 1.0 / 11))
    assert len(declining.calls) == 2


def test_fill_override_skips_rows_without_padding(monkeypatch):
    fill = _RecordingFill()
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", fill)
    # An unpublished dummy row calls no fill.
    meta = _meta([0, 1, DUMMY_ROW])
    meta.dummy_slot_row = 0
    logits = torch.randn(6, 11)
    assert SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2) is logits
    assert fill.calls == []
    # A batch without padding writes no row.
    original = logits.clone()
    out = SpecWorkerBase._zero_padding_rows(logits, _meta([0, 1, 2]), 0, 3, 2)
    assert torch.equal(out, original) and torch.equal(logits, original)
    assert fill.calls[0][1] == [False] * 6
