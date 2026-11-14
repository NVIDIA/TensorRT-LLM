# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the padding-row fill override of one-model rejection sampling.

A target may install ``set_padding_rows_fill_override``: ``_zero_padding_rows`` and
``_uniform_padding_probs`` then hand it the rows and the padding mask, and the
stock masked fills run only when it returns None. Its result must equal the
stock fills bit for bit.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

DUMMY_ROW = 9


@pytest.fixture(autouse=True)
def cheap_rows(monkeypatch):
    """The one-hot logits rows and the uniform restore (set_cheap_padding_rows); tests that need them off say so."""
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", True)


def _meta(slot_ids):
    return types.SimpleNamespace(
        batch_slot_ids=torch.tensor(slot_ids, dtype=torch.long),
        dummy_slot_row=DUMMY_ROW,
        is_ragged_verify=False,
        verify_lens=None,
        total_verify_tokens=None,
    )


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


@pytest.fixture
def fill(monkeypatch):
    recorder = _RecordingFill()
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", recorder)
    return recorder


@pytest.mark.parametrize("cheap", [True, False])
def test_zero_padding_rows_takes_the_override(fill, monkeypatch, cheap):
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", cheap)
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    meta = _meta([0, 1, DUMMY_ROW])
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", None)
    stock = SpecWorkerBase._zero_padding_rows(logits.clone(), meta, 0, 3, 2)
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", fill)
    x = logits.clone()
    out = SpecWorkerBase._zero_padding_rows(x, meta, 0, 3, 2)
    # The fill runs on a copy: the caller's logits are not mutated, as with the stock fill.
    assert out.data_ptr() != x.data_ptr() and torch.equal(x, logits)
    assert torch.equal(out.view(torch.int32), stock.view(torch.int32))
    assert fill.calls == [
        (
            out.data_ptr(),
            [False] * 4 + [True] * 2,
            0.0,
            interface._PADDING_ONEHOT_LOGIT if cheap else None,
        )
    ]


def test_uniform_padding_probs_takes_the_override(fill, monkeypatch):
    probs = torch.softmax(torch.randn(6, 11), -1)
    meta = _meta([0, 1, DUMMY_ROW])
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", None)
    stock = SpecWorkerBase._uniform_padding_probs(probs.clone(), meta, 0, 3, 2)
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", fill)
    out = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2)
    assert torch.equal(out.view(torch.int32), stock.view(torch.int32))
    assert fill.calls == [(probs.data_ptr(), [False] * 4 + [True] * 2, 1.0 / 11, None)]


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


def test_an_unpublished_dummy_row_calls_no_fill(fill):
    meta = _meta([0, 1, DUMMY_ROW])
    meta.dummy_slot_row = 0
    logits = torch.randn(6, 11)
    out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2)
    assert out is logits and fill.calls == []


def test_a_batch_without_padding_writes_no_row(fill):
    logits = torch.randn(6, 11)
    original = logits.clone()
    out = SpecWorkerBase._zero_padding_rows(logits, _meta([0, 1, 2]), 0, 3, 2)
    assert torch.equal(out, original) and torch.equal(logits, original)
    assert fill.calls[0][1] == [False] * 6


def test_setter_installs_and_removes(monkeypatch):
    monkeypatch.setattr(interface, "_PADDING_FILL_OVERRIDE", None)
    fn = _RecordingFill()
    interface.set_padding_rows_fill_override(fn)
    assert interface._PADDING_FILL_OVERRIDE is fn
    interface.set_padding_rows_fill_override(None)
    assert interface._PADDING_FILL_OVERRIDE is None
