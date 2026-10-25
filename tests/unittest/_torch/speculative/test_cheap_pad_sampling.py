# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the CUDA-graph padding rows of one-model rejection sampling.

Without TRTLLM_M3_CHEAP_PAD the padding rows' logits are zeroed, which makes
their probs uniform. With it they get a one-hot logits row (cheap for the radix
top-p filter) and ``_uniform_padding_probs`` restores the uniform probs after
the filters. Real rows must be untouched either way.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

DUMMY_ROW = 9


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


@pytest.fixture(params=[False, True], ids=["off", "cheap_pad"])
def cheap_pad(request, monkeypatch):
    monkeypatch.setattr(interface, "CHEAP_PAD_SAMPLING", request.param)
    return request.param


def test_mask_layouts():
    meta = _meta([0, 1, DUMMY_ROW])
    mask = SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 2)
    assert mask.tolist() == [False, False, False, False, True, True]
    # Rows that do not match the layout, or an unpublished dummy row, are not masked.
    assert SpecWorkerBase._padding_row_mask(meta, 5, 0, 3, 2) is None
    meta.dummy_slot_row = 0
    assert SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 2) is None


def test_mask_ragged_verify():
    meta = _meta([0, 1, DUMMY_ROW], verify_lens=[3, 1, 2])
    mask = SpecWorkerBase._padding_row_mask(meta, 6, 0, 3, 4, packed_verify=True)
    assert mask.tolist() == [False] * 4 + [True] * 2


def test_mask_skips_context_prefix():
    meta = _meta([5, 0, DUMMY_ROW])
    mask = SpecWorkerBase._padding_row_mask(meta, 2, 1, 3, 1)
    assert mask.tolist() == [False, True]


def test_padding_logits(cheap_pad):
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    meta = _meta([0, 1, DUMMY_ROW])
    out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 2)
    assert torch.equal(out[:4], logits[:4])
    expected = torch.zeros(2, 11)
    if cheap_pad:
        expected[:, 0] = interface._PADDING_ONEHOT_LOGIT
    assert torch.equal(out[4:], expected)


@pytest.mark.parametrize("temperature", [1.0e-5, 0.6, 1.0])
def test_onehot_logits_are_onehot_after_softmax(temperature, monkeypatch):
    monkeypatch.setattr(interface, "CHEAP_PAD_SAMPLING", True)
    for dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(2, 200064).to(dtype)
        meta = _meta([0, DUMMY_ROW])
        out = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 2, 1)
        probs = torch.softmax(out[1].float() / temperature, -1)
        assert probs[0].item() == 1.0
        assert torch.count_nonzero(probs).item() == 1


def test_uniform_padding_probs(cheap_pad):
    probs = torch.softmax(torch.randn(6, 11), -1)
    original = probs.clone()
    meta = _meta([0, 1, DUMMY_ROW])
    out = SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 2)
    assert torch.equal(out[:4], original[:4])
    if cheap_pad:
        assert torch.equal(out[4:], torch.full((2, 11), 1.0 / 11))
    else:
        assert torch.equal(out, original)


def test_cheap_pad_matches_zeroed_logits_probs(monkeypatch):
    """One-hot logits + uniform restore give the probs zeroed logits give."""
    torch.manual_seed(0)
    logits = torch.randn(9, 37)
    temperatures = torch.tensor([0.7] * 6 + [1.0e-5] * 3)
    meta = _meta([0, 1, DUMMY_ROW])

    def probs_for(enabled):
        monkeypatch.setattr(interface, "CHEAP_PAD_SAMPLING", enabled)
        masked = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 3, 3)
        probs = torch.softmax(masked / temperatures[:, None], -1)
        return SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 3, 3)

    off, on = probs_for(False), probs_for(True)
    assert torch.equal(on[:6], off[:6])
    torch.testing.assert_close(on[6:], off[6:], rtol=0, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_real_rows_bitwise_with_flashinfer_top_p(monkeypatch):
    """Real rows' filtered probs do not depend on the padding rows' contents."""
    flashinfer_sampling = pytest.importorskip("tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer")
    torch.manual_seed(0)
    vocab = 200064
    logits = torch.randn(14, vocab, device="cuda") * 3
    meta = _meta([0, DUMMY_ROW])
    meta.batch_slot_ids = meta.batch_slot_ids.cuda()
    temperatures = torch.tensor([1.0] * 7 + [1.0e-5] * 7, device="cuda")
    top_ps = torch.tensor([0.95] * 7 + [1.0] * 7, device="cuda")

    def probs_for(enabled):
        monkeypatch.setattr(interface, "CHEAP_PAD_SAMPLING", enabled)
        masked = SpecWorkerBase._zero_padding_rows(logits, meta, 0, 2, 7)
        probs = flashinfer_sampling.compute_probs_from_logits_op(
            masked, temperatures, None, top_ps
        )
        return SpecWorkerBase._uniform_padding_probs(probs, meta, 0, 2, 7)

    off, on = probs_for(False), probs_for(True)
    assert torch.equal(on[:7], off[:7])
    torch.testing.assert_close(on[7:], off[7:], rtol=0, atol=1e-9)
