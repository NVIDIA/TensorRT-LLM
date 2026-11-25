# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash block merge's CUDA op (``minimax_m31_dflash_merge_later_keys``, ``ops/dspark/dflash_merge``) against the
built-in eager merge (``modeling_dflash.dflash_merge_future_block_attention``) at the MiniMax-M3.1 DEP8 drafter shapes
(64 / 4 heads, head_dim 128, block 7): within 2 bf16 ULP on |out| >= 1/16 and 4 ULP of the largest output anywhere
(each side rounds its own fp32 result once),
finite; a row without later keys keeps its output bit for bit."""

import math

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import dflash_merge
from tensorrt_llm._torch.models import modeling_dflash


def _op_built() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return hasattr(dflash_merge.family_ops(), "minimax_m31_dflash_merge_later_keys")
    except NoTarget:  # this installation carries no family library; a broken one fails
        return False


pytestmark = pytest.mark.skipif(
    not _op_built(), reason="needs the family's DFlash merge op on a GPU"
)

HQ, HKV, D, BLK = 64, 4, 128, 7
SCALE = D**-0.5


def _ulp(x: torch.Tensor) -> torch.Tensor:
    """A monotonic integer image of bf16 values: neighbours differ by 1."""
    i = x.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def _operands(b, lse_mean, gen):
    q = torch.randn(b, BLK, HQ, D, generator=gen, device="cuda").to(torch.bfloat16)
    k = torch.randn(b, BLK, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
    v = torch.randn(b, BLK, HKV, D, generator=gen, device="cuda").to(torch.bfloat16)
    out = (torch.randn(b * BLK, HQ, D, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    lse = torch.randn(b * BLK, HQ, generator=gen, device="cuda") + lse_mean
    return out, lse, q, k, v


def _eager(out, lse_log2, q, k, v, window_right):
    return modeling_dflash.dflash_merge_future_block_attention(
        out, lse_log2 * math.log(2.0), q, k, v, SCALE, window_right
    )


@pytest.mark.parametrize("b", [1, 11])
@pytest.mark.parametrize("window_right", [4095, 3, -1])
@pytest.mark.parametrize("lse_mean", [12.0, 0.0], ids=["ctx", "near"])
def test_the_cuda_merge_matches_the_eager_merge(b, window_right, lse_mean):
    gen = torch.Generator(device="cuda").manual_seed(1000 * b + window_right)
    out, lse, q, k, v = _operands(b, lse_mean, gen)
    want = _eager(out, lse, q, k, v, window_right)
    got = out.clone()
    assert dflash_merge.merge_later_keys_(got, lse, q, k, v, SCALE, window_right)
    assert torch.isfinite(got).all()
    big = want.float().abs() >= 1.0 / 16
    assert (_ulp(got) - _ulp(want)).abs()[big].max().item() <= 2
    four_ulp = 4.0 * torch.finfo(torch.bfloat16).eps * want.float().abs().max().item()
    assert (got.float() - want.float()).abs().max().item() <= four_ulp


def test_rows_without_later_keys_keep_their_output():
    gen = torch.Generator(device="cuda").manual_seed(3)
    out, lse, q, k, v = _operands(6, 12.0, gen)
    got = out.clone()
    assert dflash_merge.merge_later_keys_(got, lse, q, k, v, SCALE, 0)
    assert torch.equal(got.view(torch.int16), out.view(torch.int16))
    got = out.clone()
    assert dflash_merge.merge_later_keys_(got, lse, q, k, v, SCALE, 4095)
    last = (
        torch.arange(6, device="cuda") * BLK + BLK - 1
    )  # each request's last query has no later key
    assert torch.equal(got[last].view(torch.int16), out[last].view(torch.int16))
