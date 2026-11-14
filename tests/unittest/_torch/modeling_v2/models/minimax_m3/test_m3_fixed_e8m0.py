# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The UE8M0 zero code in the family library's fixed low-M MXFP8 GEMM (``minimax_m3_mxfp8_low_m_gemm``).

An activation block whose largest magnitude is at most 448 * 2^-127 takes the scale byte 0 (2^-127, an fp32
denormal). Its reciprocal must not be the flush-to-zero +inf, which turns the block into NaN or +-448 values: the
block quantizes to zero, exactly like an all-zero block, so the GEMM output is bit for bit the one with that block
zeroed, and finite.
"""

import pytest
import torch
from _m3_mxfp8_operands import mxfp8

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the family library's low-M MXFP8 GEMM runs on SM100 and SM103 only",
)

# (N, K) -> (warps, cluster_k, row_tiles): configurations the op is built for.
_CONFIGS = {(2560, 6144): (4, 4, 2), (6144, 3072): (4, 2, 2)}
# Values whose block maximum is below 448 * 2^-127 (~2.6e-36): a BF16 normal and a BF16 subnormal.
_TINY = (1.0e-37, 2.0e-39)


@pytest.mark.parametrize("shape", sorted(_CONFIGS), ids=lambda s: f"N{s[0]}_K{s[1]}")
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("tiny", _TINY)
def test_an_all_tiny_block_quantizes_to_zero(shape, rows, tiny):
    n, k = shape
    warps, cluster_k, row_tiles = _CONFIGS[shape]
    g = torch.Generator(device="cuda").manual_seed(n + k + rows)
    w, wsf = mxfp8(n, k, g)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    # Blocks 0 and 5 of every row, and the last block of row 0, are all tiny (both signs).
    signs = torch.where(torch.rand(rows, 32, generator=g, device="cuda") < 0.5, -1.0, 1.0)
    tiny_x = x.clone()
    zero_x = x.clone()
    for block in (0, 5):
        tiny_x[:, block * 32 : (block + 1) * 32] = (signs * tiny).to(torch.bfloat16)
        zero_x[:, block * 32 : (block + 1) * 32] = 0
    tiny_x[0, -32:] = tiny
    zero_x[0, -32:] = 0
    assert bool((tiny_x[:, :32] != 0).all()), "the tiny values must be representable in BF16"
    out_tiny = family_ops().minimax_m3_mxfp8_low_m_gemm(
        tiny_x, w, wsf, warps, cluster_k, row_tiles, False
    )
    out_zero = family_ops().minimax_m3_mxfp8_low_m_gemm(
        zero_x, w, wsf, warps, cluster_k, row_tiles, False
    )
    assert bool(torch.isfinite(out_tiny).all())
    assert torch.equal(out_tiny.view(torch.int16), out_zero.view(torch.int16))
