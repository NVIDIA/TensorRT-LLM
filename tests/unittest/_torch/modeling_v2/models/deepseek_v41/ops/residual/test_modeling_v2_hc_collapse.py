# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_collapse entry (the engine's one-launch collapse of the hc residual copies).

Band: fp32 accumulation of 4 products vs float64, then one cast to bf16: one bf16 rounding step of the row's largest
element (2 units of ULP[bf16]), 0.2 ulp RMS (the band of the family's `hc_pre`, which computes the same function); and
against `hc_pre` itself the same band, since the two differ at most in the fp32 summation order."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_collapse import (
    hc_collapse,  # noqa: E501
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import (
    hc_pre,  # noqa: E501
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize("cell", hc_collapse.CELLS, ids=[c.why[:44] for c in hc_collapse.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_collapse)
    s = cell.spec
    gen = gen_for(cell)
    x = torch.randn(s["n"], s["hc"], s["d"], generator=gen, device=DEV).bfloat16()
    pre = torch.rand(s["n"], s["hc"], generator=gen, device=DEV)
    got = hc_collapse(x, pre)
    assert got.dtype == x.dtype and got.shape == (s["n"], s["d"])
    assert_within_ulp(got, hc_collapse.reference(x, pre), element_ulp=2.0, rms_ulp=0.2)
    assert_within_ulp(got, hc_pre(x, pre), element_ulp=2.0, rms_ulp=0.2)


def test_cuda_graph():
    x = torch.randn(160, 4, 5120, device=DEV).bfloat16()
    pre = torch.rand(160, 4, device=DEV)
    hc_collapse(x, pre)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = hc_collapse(x, pre)
    for seed in range(2):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        x.copy_(torch.randn(x.shape, generator=gen, device=DEV))
        pre.copy_(torch.rand(pre.shape, generator=gen, device=DEV))
        g.replay()
        assert torch.equal(out, hc_collapse(x, pre))


def test_rejections():
    x = torch.zeros(2, 4, 8, device=DEV).bfloat16()
    hc_collapse.is_valid(x, torch.zeros(2, 4, device=DEV))
    with pytest.raises(AssertionError):
        hc_collapse.is_valid(x, torch.zeros(2, 3, device=DEV))
    with pytest.raises(AssertionError):
        hc_collapse.is_valid(x.transpose(0, 1), torch.zeros(4, 2, device=DEV))
