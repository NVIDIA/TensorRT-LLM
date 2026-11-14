# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The override registries a model installs into the sampler ops and the logits processor.

Each registry must leave the stock call byte for byte in place while nothing is installed, replace it with the
override's result when the override returns one, and fall back to the stock call when the override declines (returns
None). The FlashInfer module is a recording stand-in, so no GPU is needed.
"""

import types
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.modules import logits_processor as lp
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as fi_ops

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """The sampler ops module with a stand-in flashinfer whose calls are recorded, and no override around it."""
    stock_probs = torch.full((2, 8), 0.125)
    stock_tokens = torch.zeros(2, dtype=torch.int32)
    sampling = types.SimpleNamespace(
        softmax=mock.Mock(return_value=stock_probs),
        top_p_renorm_probs=mock.Mock(side_effect=lambda probs, top_p: probs),
        top_k_mask_logits=mock.Mock(side_effect=lambda logits, top_k: logits),
        sampling_from_probs=mock.Mock(return_value=stock_tokens),
    )
    monkeypatch.setattr(
        fi_ops, "flashinfer", types.SimpleNamespace(sampling=sampling), raising=False
    )

    def clear():
        fi_ops.set_softmax_override(None)
        fi_ops.set_top_p_renorm_override(None)
        fi_ops.set_sampling_from_probs_override(None)

    clear()
    yield sampling, stock_probs, stock_tokens
    clear()


def test_softmax_unset_is_stock(fake_flashinfer):
    sampling, stock, _ = fake_flashinfer
    logits, temperature = torch.rand(2, 8), torch.ones(2)
    assert fi_ops.softmax_op(logits, temperature) is stock
    sampling.softmax.assert_called_once_with(logits, temperature, enable_pdl=mock.ANY)
    assert fi_ops.compute_probs_from_logits_op(logits, temperature, None, None) is stock
    assert sampling.softmax.call_count == 2


def test_softmax_override_result_is_used(fake_flashinfer):
    sampling, _, _ = fake_flashinfer
    mine = torch.zeros(2, 8)
    override = mock.Mock(return_value=mine)
    fi_ops.set_softmax_override(override)
    logits, temperature = torch.rand(2, 8), torch.ones(2)
    assert fi_ops.softmax_op(logits, temperature) is mine
    override.assert_called_once_with(logits, temperature)
    assert fi_ops.compute_probs_from_logits_op(logits, temperature, None, None) is mine
    top_p = torch.full((2,), 0.9)
    assert fi_ops.compute_probs_from_logits_op(logits, temperature, None, top_p) is mine
    sampling.top_p_renorm_probs.assert_called_once_with(mine, top_p)
    sampling.softmax.assert_not_called()


def test_softmax_override_declining_falls_back(fake_flashinfer):
    sampling, stock, _ = fake_flashinfer
    override = mock.Mock(return_value=None)
    fi_ops.set_softmax_override(override)
    logits, temperature = torch.rand(2, 8), torch.ones(2)
    assert fi_ops.softmax_op(logits, temperature) is stock
    assert fi_ops.compute_probs_from_logits_op(logits, temperature, None, None) is stock
    assert override.call_count == 2 and sampling.softmax.call_count == 2


def test_top_p_renorm_override(fake_flashinfer):
    sampling, stock, _ = fake_flashinfer
    probs, top_p = torch.rand(2, 8), torch.full((2,), 0.9)
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is probs
    sampling.top_p_renorm_probs.assert_called_once_with(probs, top_p)

    mine = torch.ones(2, 8)
    override = mock.Mock(return_value=mine)
    fi_ops.set_top_p_renorm_override(override)
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is mine
    assert fi_ops.compute_probs_from_logits_op(torch.rand(2, 8), torch.ones(2), None, top_p) is mine
    override.assert_called_with(stock, top_p)
    assert sampling.top_p_renorm_probs.call_count == 1

    override.return_value = None
    assert fi_ops.top_p_renorm_probs_op(probs, top_p) is probs
    assert sampling.top_p_renorm_probs.call_count == 2


def test_sampling_from_probs_override_needs_seed_and_offset_tensors(fake_flashinfer):
    sampling, _, stock = fake_flashinfer
    mine = torch.ones(2, dtype=torch.int32)
    override = mock.Mock(return_value=mine)
    fi_ops.set_sampling_from_probs_override(override)
    probs = torch.rand(2, 8)
    seed, offset = torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)

    assert fi_ops.sampling_from_probs_op(probs, seed=seed, offset=offset) is mine
    override.assert_called_once_with(probs, seed, offset)
    sampling.sampling_from_probs.assert_not_called()

    # An eager generator, integer seeds or a NaN check keep the stock call.
    for kwargs in (
        {"generator": torch.Generator()},
        {"seed": 1, "offset": 2},
        {"seed": seed, "offset": offset, "check_nan": True},
    ):
        assert fi_ops.sampling_from_probs_op(probs, **kwargs) is stock
    assert override.call_count == 1 and sampling.sampling_from_probs.call_count == 3

    override.return_value = None
    assert fi_ops.sampling_from_probs_op(probs, seed=seed, offset=offset) is stock


def test_logits_widen_override():
    lm_head = mock.Mock(return_value=torch.ones(2, 8, dtype=torch.bfloat16))
    processor = lp.LogitsProcessor()
    hidden = torch.rand(2, 4)
    try:
        lp.set_logits_widen_override(None)
        stock = processor(hidden, lm_head, None, return_context_logits=True)
        assert stock.dtype == torch.float32

        mine = torch.zeros(2, 8)
        override = mock.Mock(return_value=mine)
        lp.set_logits_widen_override(override)
        assert processor(hidden, lm_head, None, return_context_logits=True) is mine
        assert override.call_args.args[0].dtype == torch.bfloat16

        override.return_value = None
        declined = processor(hidden, lm_head, None, return_context_logits=True)
        assert declined.dtype == torch.float32 and torch.equal(declined, stock)
    finally:
        lp.set_logits_widen_override(None)
