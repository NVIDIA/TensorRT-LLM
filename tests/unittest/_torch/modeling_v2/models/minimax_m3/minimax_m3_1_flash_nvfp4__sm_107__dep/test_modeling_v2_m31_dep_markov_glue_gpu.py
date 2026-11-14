# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_MARKOV_GLUE on a GPU: each ``markov_glue.MarkovGlue`` method gives the stock line's result bit for bit (the
masked Markov gather + GEMV, the widening add on a strided base, the proposal store over slot rows, including
out-of-vocab anchors, duplicate CUDA-graph dummy slots and non-finite / subnormal values), and each declines inputs
outside its contract."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    markov_glue,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

VOCAB, VPAD, RANK, K = 200064, 200192, 256, 7
GLUE = markov_glue.MarkovGlue()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _weights(gen):
    w1 = (torch.randn(VOCAB, RANK, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
    return w1, w2


def _prev(rows, dtype, gen):
    prev = torch.randint(0, VOCAB, (rows,), generator=gen, device="cuda")
    prev[0] = -1  # chain_speculative_sampling's pad for a non-accepted position
    if rows > 1:
        prev[1] = VOCAB  # an out-of-range id from a row whose residual had no mass
    if rows > 2:
        prev[2] = VOCAB + 5
    return prev.to(dtype)


@pytest.mark.parametrize("rows", [1, 8, 12, 16, 64])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_step_bias_is_bitwise_the_stock_bias(rows, dtype):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    w1, w2 = _weights(gen)
    prev = _prev(rows, dtype, gen)
    got = GLUE.step_bias(prev, w1, w2)
    assert got is not None
    assert torch.equal(_bits(got), _bits(dspark_markov_step_bias(prev, w1, w2)))


def _base(rows, strided, gen):
    """[rows, K, VOCAB] fp32 logits, either contiguous or the vocab slice of a padded buffer, with zeros of both signs,
    infinities and subnormals mixed in."""
    full = torch.randn(rows, K, VPAD if strided else VOCAB, generator=gen, device="cuda") * 4.0
    full[..., 3] = -0.0
    full[..., 5] = float("inf")
    full[..., 7] = float("-inf")
    full[..., 11] = 1e-41
    full[..., 13] = -3e-39
    return full[..., :VOCAB] if strided else full


@pytest.mark.skipif(
    not markov_glue.NATIVE_WIDEN_ADD,
    reason="TRTLLM_M31_DISABLE_NATIVE_WIDEN_ADD=1 keeps the stock add",
)
@pytest.mark.parametrize("rows", [1, 8, 12, 16, 128])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize("bias_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_biased_logits_are_bitwise_the_stock_add(rows, strided, bias_dtype):
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    gen_logits = _base(rows, strided, gen)
    bias = (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 2.0).to(bias_dtype)
    bias[:, 3] = -0.0
    bias[:, 11] = 2e-40
    bias[:, 17] = float("nan")
    for step in (0, 3, K - 1):
        got = GLUE.biased_logits(gen_logits, step, bias)
        assert got is not None and got.is_contiguous() and got.dtype == torch.float32
        want = gen_logits[:, step] + bias.to(gen_logits.dtype)
        assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("rows", [1, 8, 12, 16, 128])
@pytest.mark.parametrize("dummies", [0, 3])
def test_store_probs_is_bitwise_the_stock_index_put(rows, dummies):
    assert markov_glue.scatter_rows_op() is not None, (
        "the family library lacks minimax_m31_scatter_rows"
    )
    gen = torch.Generator(device="cuda").manual_seed(200 + rows)
    capacity = 256
    real = rows - min(dummies, rows - 1)
    slots = torch.randperm(capacity, generator=gen, device="cuda")[:real]
    # The CUDA-graph padding requests repeat one dummy request: one scratch row, identical values.
    slots = torch.cat(
        [slots, torch.full((rows - real,), capacity, dtype=torch.int64, device="cuda")]
    )
    probs = torch.rand(rows, VOCAB, generator=gen, device="cuda")
    if real < rows:
        probs[real:] = probs[real]
    for width in (VOCAB, VPAD):
        base = torch.rand(capacity + 1, K, width, generator=gen, device="cuda")
        for step in (0, K - 1):
            got, want = base.clone(), base.clone()
            assert GLUE.store_probs(got, slots, step, probs)
            want[slots, step, :VOCAB] = probs
            assert torch.equal(_bits(got), _bits(want))


def test_the_methods_decline_outside_their_contracts():
    gen = torch.Generator(device="cuda").manual_seed(5)
    w1, w2 = _weights(gen)
    prev = _prev(8, torch.int64, gen)
    assert GLUE.step_bias(prev.view(8, 1), w1, w2) is None
    assert GLUE.step_bias(prev.float(), w1, w2) is None
    assert GLUE.step_bias(prev, w1.float(), w2) is None
    logits = torch.randn(8, K, VOCAB, device="cuda")
    bias = torch.randn(8, VOCAB, device="cuda").to(torch.bfloat16)
    assert GLUE.biased_logits(logits.to(torch.bfloat16), 0, bias) is None
    assert GLUE.biased_logits(logits, K, bias) is None
    assert GLUE.biased_logits(logits, 0, bias.t().contiguous().t()) is None
    assert GLUE.biased_logits(logits, 0, bias[:4]) is None
    probs = torch.rand(8, VOCAB, device="cuda")
    draft_probs = torch.zeros(9, K, VOCAB, device="cuda")
    slots = torch.arange(8, device="cuda")
    assert not GLUE.store_probs(draft_probs, slots.int(), 0, probs)
    assert not GLUE.store_probs(draft_probs, slots, K, probs)
    assert not GLUE.store_probs(draft_probs[..., :100], slots, 0, probs)
    assert not GLUE.store_probs(draft_probs, slots, 0, probs.t().contiguous().t())
    assert torch.count_nonzero(draft_probs) == 0
