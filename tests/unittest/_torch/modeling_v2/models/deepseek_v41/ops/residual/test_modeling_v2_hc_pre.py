# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_pre entry: collapse of the hc residual copies.

Band: fp32 accumulation of 4 products vs float64, then one cast. bf16 outputs differ at most by a
one-ulp flip where the sum sits within an fp32 ulp of a rounding tie: 1 bf16 ulp of the row's
largest value elementwise. fp32 outputs: 4 fp32 ulp.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import (
    hc_pre,  # noqa: E501
)


@pytest.mark.parametrize("cell", hc_pre.CELLS, ids=[c.why[:44] for c in hc_pre.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_pre)
    s = cell.spec
    gen = gen_for(cell)
    x = torch.randn(s["n"], s["hc"], s["d"], generator=gen, device=DEV).to(s["dtype"])
    pre = torch.rand(s["n"], s["hc"], generator=gen, device=DEV)
    got, ref = hc_pre(x, pre), hc_pre.reference(x, pre)
    assert got.dtype == x.dtype and got.shape == (s["n"], s["d"])
    if x.dtype == torch.float32:
        assert_within_ulp(got, ref, element_ulp=4.0, rms_ulp=1.0)
    else:
        # element_ulp is in units of ULP[bf16] = 2**-8, the spacing of values at the top of a binade. A
        # value at the bottom of a binade has spacing 2**-7, and the row maximum that scales the band can
        # sit there, so one bf16 rounding step of that element is up to 2 units. The 4097-row cell (~16M
        # outputs) had 1-2 elements land one step apart (0.0043-0.005 of the row max against 1 unit =
        # 0.0039); 2 units = 2**-7 is the band for exactly one bf16 step, no more. rms_ulp is unchanged.
        assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)


def test_tiling_changes_nothing():
    import tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre as mod

    x = torch.randn(11, 4, 64, device=DEV).bfloat16()
    pre = torch.rand(11, 4, device=DEV)
    whole = hc_pre(x, pre)
    old, mod._TILE = mod._TILE, 3
    try:
        tiled = hc_pre(x, pre)
    finally:
        mod._TILE = old
    assert torch.equal(whole, tiled)


def test_rejections():
    x = torch.zeros(2, 4, 8, device=DEV)
    hc_pre.is_valid(x, torch.zeros(2, 4, device=DEV))
    with pytest.raises(AssertionError):
        hc_pre.is_valid(x, torch.zeros(2, 3, device=DEV))
    with pytest.raises(AssertionError):
        hc_pre.is_valid(x, torch.zeros(2, 4, device=DEV).bfloat16())
