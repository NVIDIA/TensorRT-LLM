# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_post_mapping entry: the engine's `mhc_post_mapping` against the float64 `hc_post` reference.

Band as for hc_post (bf16 stream): the kernel accumulates the 5 terms in fp32 and rounds once; the
reference is float64 with one rounding. Elementwise 2 bf16 units of the row's largest value (a value at the
bottom of a binade has spacing 2**-7), 0.2 ulp RMS in aggregate.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post import hc_post
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (  # noqa: E501
    hc_post_mapping,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)


def _inputs(n, hc, d, cell):
    gen = gen_for(cell)
    x = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
    res = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    post = 2 * torch.rand(n, hc, generator=gen, device=DEV)
    comb = torch.softmax(torch.randn(n, hc, hc, generator=gen, device=DEV), dim=-1)
    return x, res, post, comb


@pytest.mark.parametrize(
    "cell", hc_post_mapping.CELLS, ids=[c.why[:44] for c in hc_post_mapping.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_post_mapping)
    s = cell.spec
    args = _inputs(s["n"], s["hc"], s["d"], cell)
    got, ref = hc_post_mapping(*args), hc_post_mapping.reference(*args)
    assert got.dtype == torch.bfloat16 and got.shape == args[1].shape
    if s["n"] == 0:
        return
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.2)


def test_agrees_with_the_plain_torch_op():
    args = _inputs(96, 4, 5120, hc_post_mapping.CELLS[2])
    assert_within_ulp(hc_post_mapping(*args), hc_post(*args), element_ulp=2.0, rms_ulp=0.2)


def test_comb_is_indexed_source_then_destination():
    res = torch.randn(3, 4, 64, device=DEV).bfloat16()
    comb = torch.zeros(3, 4, 4, device=DEV)
    comb[:, 0, 1] = 1.0  # copy 0 feeds only copy 1
    out = hc_post_mapping(
        torch.zeros(3, 64, device=DEV).bfloat16(), res, torch.zeros(3, 4, device=DEV), comb
    )
    assert torch.equal(out[:, 1], res[:, 0]) and not out[:, 0].any() and not out[:, 2:].any()


def test_cuda_graph_replay_is_bitwise_the_eager_result():
    args = list(_inputs(18, 4, 5120, hc_post_mapping.CELLS[1]))
    hc_post_mapping(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = hc_post_mapping(*args)
    args[0].copy_(torch.randn_like(args[0].float()).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, hc_post_mapping(*args), rtol=0, atol=0)


def test_rejections():
    ok = _inputs(2, 4, 64, hc_post_mapping.CELLS[0])
    hc_post_mapping.is_valid(*ok)
    with pytest.raises(AssertionError):
        hc_post_mapping.is_valid(ok[0], ok[1], ok[2], torch.zeros(2, 3, 3, device=DEV))
    with pytest.raises(AssertionError):
        hc_post_mapping.is_valid(ok[0], ok[1], ok[2].bfloat16(), ok[3])
    with pytest.raises(AssertionError):
        hc_post_mapping.is_valid(ok[0].float(), ok[1].float(), ok[2], ok[3])
