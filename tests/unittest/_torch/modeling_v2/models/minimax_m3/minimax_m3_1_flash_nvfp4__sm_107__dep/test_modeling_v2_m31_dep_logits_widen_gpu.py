# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_LOGITS_WIDEN on a GPU: ``logits_widen.widen`` gives ``.float()`` of the lm_head output bit for bit (the padded
head's vocabulary slice and a contiguous input; bf16 and fp16; +-0, +-inf, subnormals and the extremes; a NaN stays a
NaN), as a contiguous float32 tensor, and declines inputs outside its contract."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    logits_widen,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

VOCAB, VPAD, HIDDEN = 200064, 200192, 512


def _bits(t):
    return t.contiguous().view(torch.int32)


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


@pytest.mark.parametrize("rows", [1, 56, 64, 84, 96, 112, 128])
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
    assert torch.equal(_bits(got)[keep], _bits(want)[keep])


def test_widen_declines_outside_its_contract():
    x = torch.randn(4, 64, device="cuda")
    assert logits_widen.widen(x) is None
    assert logits_widen.widen(x.to(torch.bfloat16).t()) is None
    assert logits_widen.widen(x.to(torch.bfloat16)[0]) is None
    assert logits_widen.widen(x.to(torch.bfloat16)[:, :0]) is None
