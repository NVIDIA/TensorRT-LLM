# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DEP target's sampler glue kernels on a GPU, each bit for bit against the stock line it replaces.

* DEP_LOGITS_WIDEN: ``logits_widen.widen`` gives ``.float()`` of the lm_head output (the padded head's vocabulary slice
  and a contiguous input; bf16 and fp16; +-0, +-inf, subnormals and the extremes; a NaN stays a NaN).
* DEP_PAD_FILL: ``pad_fill.pad_rows_fill`` leaves every row as the stock padding fills leave it (the one-hot logits
  fill, the zero fill and the uniform probs fill).
* DEP_MARKOV_GLUE: each ``markov_glue.MarkovGlue`` method gives the stock line's result (the masked Markov gather +
  GEMV, the widening add on a strided base, the proposal store over slot rows, including out-of-vocab anchors,
  duplicate CUDA-graph dummy slots and non-finite / subnormal values).
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    logits_widen,
    markov_glue,
    pad_fill,
)
from tensorrt_llm._torch.models.modeling_speculative import dspark_markov_step_bias
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

VOCAB, VPAD, RANK, K = 200064, 200192, 256, 7
# A one-hot padding logit: finite in every logits dtype and large enough that softmax of the row is one-hot.
ONEHOT = 1.0e4
GLUE = markov_glue.MarkovGlue()


def _bits(t):
    return t.contiguous().view(torch.uint8)


# --- DEP_LOGITS_WIDEN ---


def _logits(rows, dtype, padded, gen):
    """[rows, VOCAB] logits in ``dtype``, the vocabulary slice of a [rows, VPAD] buffer when ``padded``, with signed
    zeros, infinities, subnormals and the dtype's extremes mixed in."""
    buf = (torch.randn(rows, VPAD if padded else VOCAB, generator=gen, device="cuda") * 8.0).to(
        dtype
    )
    info = torch.finfo(dtype)
    for col, value in (
        (3, -0.0),
        (5, float("inf")),
        (7, float("-inf")),
        (11, info.tiny / 4),
        (13, -info.tiny / 8),
        (17, info.max),
        (19, -info.max),
        (VOCAB - 1, info.tiny),
    ):
        buf[:, col] = value
    return buf[:, :VOCAB] if padded else buf


@pytest.mark.parametrize("rows", [1, 128])
@pytest.mark.parametrize("padded", [True, False], ids=["padded", "contiguous"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_widen_is_bitwise_float(rows, padded, dtype):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    logits = _logits(rows, dtype, padded, gen)
    got = logits_widen.widen(logits)
    assert got is not None and got.is_contiguous() and got.dtype == torch.float32
    assert torch.equal(_bits(got), _bits(logits.float()))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_a_nan_stays_a_nan(dtype):
    gen = torch.Generator(device="cuda").manual_seed(9)
    logits = _logits(12, dtype, True, gen)
    logits[:, 23] = float("nan")
    logits[4] = float("nan")
    got, want = logits_widen.widen(logits), logits.float()
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    keep = ~torch.isnan(want)
    assert torch.equal(got.view(torch.int32)[keep], want.view(torch.int32)[keep])


# --- DEP_PAD_FILL ---


def _mask(rows, kind, gen):
    if kind == "none":
        return torch.zeros(rows, dtype=torch.bool, device="cuda")
    if kind == "all":
        return torch.ones(rows, dtype=torch.bool, device="cuda")
    return torch.rand(rows, generator=gen, device="cuda") < 0.25


def _stock_logits(x, mask, col0):
    y = x.masked_fill(mask.unsqueeze(-1), 0.0)
    if col0 is not None:
        y[:, 0].masked_fill_(mask, col0)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [384])
@pytest.mark.parametrize("kind", ["random", "all"])
def test_the_fills_are_bitwise_the_stock_fills(dtype, rows, kind):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3.0).to(dtype)
    mask = _mask(rows, kind, gen)
    for col0 in (ONEHOT, None):
        y = x.clone()
        assert pad_fill.pad_rows_fill(y, mask, 0.0, col0) is y
        assert torch.equal(_bits(y), _bits(_stock_logits(x, mask, col0)))
    y = x.clone()
    pad_fill.pad_rows_fill(y, mask, 1.0 / VOCAB)
    assert torch.equal(_bits(y), _bits(x.clone().masked_fill_(mask.unsqueeze(-1), 1.0 / VOCAB)))


# --- DEP_MARKOV_GLUE ---


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


@pytest.mark.parametrize("rows", [1, 64])
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
@pytest.mark.parametrize("rows", [1, 128])
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


@pytest.mark.parametrize("rows", [1, 128])
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
