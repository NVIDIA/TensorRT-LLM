# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the shared_expert_fp8 entries (stock's DeepGEMM path): FP8 block-scale GEMMs against the fp32 product over
the dequantized weights.

Weights are requantized from random bf16 with `requantize_block128`; the gate is a relative RMS error of 6%: from the
context chunk on both activations are quantized to FP8 E4M3 per 1x128 block (3 mantissa bits) and the reference keeps
them exact; measured 4.2-4.4%. The decode-sized cell takes stock's small-M GEMMs. The block-128 requantization of
power-of-two-scaled E4M3 values is checked to be exact.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (  # noqa: E501
    deep_gemm_weight_scale,
    requantize_block128,
    shared_expert_fp8_deep_gemm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="DeepGEMM kernels, SM103",
)


@pytest.mark.parametrize(
    "cell",
    shared_expert_fp8_deep_gemm.CELLS,
    ids=[c.why[:44] for c in shared_expert_fp8_deep_gemm.CELLS],
)
def test_deep_gemm_cells(cell, certifying):  # noqa: F811
    """Stock's SM100 path for the cell's rows, same 6% gate."""
    certifying(shared_expert_fp8_deep_gemm)
    s = cell.spec
    gen = gen_for(cell)
    h, i = s["hidden"], s["inter"]
    w1 = torch.randn(i, h, generator=gen, device=DEV) * 0.02
    w3 = torch.randn(i, h, generator=gen, device=DEV) * 0.02
    w2 = torch.randn(h, i, generator=gen, device=DEV) * 0.02
    w13, s13 = requantize_block128(torch.cat([w1, w3], 0))
    w2q, s2 = requantize_block128(w2)
    sf13, sf2 = deep_gemm_weight_scale(s13, *w13.shape), deep_gemm_weight_scale(s2, *w2q.shape)
    x = (torch.randn(s["n"], h, generator=gen, device=DEV) * 2).bfloat16()
    y = shared_expert_fp8_deep_gemm(x, w13, sf13, s13, w2q, sf2, s2, s["limit"])
    ref = shared_expert_fp8_deep_gemm.reference(x, w13, sf13, s13, w2q, sf2, s2, s["limit"])
    assert y.shape == ref.shape == (s["n"], h) and y.dtype == torch.bfloat16
    rel = ((y.float() - ref.float()).pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()
    assert rel < 0.06, rel


def test_requant_keeps_pow2_scaled_e4m3_exactly():
    """A weight that is E4M3 times power-of-two 32x32 scales (the checkpoint's form) requantizes without loss."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    # normal-range E4M3 codes (|q| in [1, 448]) and block exponents within 2**4 of each other: rescaling to the coarser
    # tile's scale keeps every value in E4M3's normal range, where it is exact
    mag = torch.exp2(torch.rand(256, 512, generator=gen, device=DEV) * 8.8)
    sign = torch.where(torch.rand(256, 512, generator=gen, device=DEV) < 0.5, -1.0, 1.0)
    q = (mag * sign).to(torch.float8_e4m3fn).float()
    exps = torch.randint(-6, -2, (8, 16), generator=gen, device=DEV).float()
    w = q * torch.exp2(exps).repeat_interleave(32, 0).repeat_interleave(32, 1)
    wq, sc = requantize_block128(w)
    back = wq.float() * sc.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert tuple(sc.shape) == (2, 4) and torch.equal(back, w)
