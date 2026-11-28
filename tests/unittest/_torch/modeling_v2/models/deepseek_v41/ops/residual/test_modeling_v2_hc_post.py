# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_post entry: expand a sub-layer output to hc copies and mix the residual in.

Band as for hc_pre: fp32 accumulation vs float64 and a single cast; 1 bf16 ulp (fp32: 4 ulp) of
the row's largest value elementwise.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post import (
    hc_post,  # noqa: E501
)


@pytest.mark.parametrize("cell", hc_post.CELLS, ids=[c.why[:44] for c in hc_post.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_post)
    s = cell.spec
    n, hc, d = s["n"], s["hc"], s["d"]
    gen = gen_for(cell)
    x = torch.randn(n, d, generator=gen, device=DEV).to(s["dtype"])
    res = torch.randn(n, hc, d, generator=gen, device=DEV).to(s["dtype"])
    post = 2 * torch.rand(n, hc, generator=gen, device=DEV)
    comb = torch.softmax(torch.randn(n, hc, hc, generator=gen, device=DEV), dim=-1)
    got, ref = hc_post(x, res, post, comb), hc_post.reference(x, res, post, comb)
    assert got.dtype == x.dtype and got.shape == res.shape
    if x.dtype == torch.float32:
        assert_within_ulp(got, ref, element_ulp=4.0, rms_ulp=1.0)
    else:
        # element_ulp is in units of ULP[bf16] = 2**-8, the spacing of values at the top of a binade. A
        # value at the bottom of a binade has spacing 2**-7, and the row maximum that scales the band can
        # sit there, so one bf16 rounding step of that element is up to 2 units. The 4097-row cell (~16M
        # outputs) had 1-2 elements land one step apart (0.0043-0.005 of the row max against 1 unit =
        # 0.0039); 2 units = 2**-7 is the band for exactly one bf16 step, no more. rms_ulp is unchanged.
        assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)


def test_comb_is_indexed_source_then_destination():
    # copy 0 feeds only copy 1: out[:, 1] = residual[:, 0], everything else zero
    res = torch.randn(3, 2, 8, device=DEV)
    comb = torch.zeros(3, 2, 2, device=DEV)
    comb[:, 0, 1] = 1.0
    out = hc_post(torch.zeros(3, 8, device=DEV), res, torch.zeros(3, 2, device=DEV), comb)
    assert torch.equal(out[:, 1], res[:, 0]) and not out[:, 0].any()


def test_rejections():
    ok = (
        torch.zeros(2, 8, device=DEV),
        torch.zeros(2, 4, 8, device=DEV),
        torch.zeros(2, 4, device=DEV),
        torch.zeros(2, 4, 4, device=DEV),
    )
    hc_post.is_valid(*ok)
    with pytest.raises(AssertionError):
        hc_post.is_valid(ok[0], ok[1], ok[2], torch.zeros(2, 3, 3, device=DEV))
    with pytest.raises(AssertionError):
        hc_post.is_valid(ok[0], ok[1], ok[2].bfloat16(), ok[3])
