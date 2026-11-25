# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_post_collapse_norm entry (the critical half of a decode-sized mHC boundary) against the
float64 composition of `hc_post_mapping`, `hc_pre` and an RMSNorm.

Band: the post update is `hc_post_mapping`'s (2 bf16 units, 0.2 ulp RMS). The collapse is rounded to bf16 before
the norm in the kernel and in the reference; an fp32 vs float64 accumulation of 4 products can flip that
rounding (1 ulp) and the output rounding adds one: 2 ulp elementwise, 0.3 ulp RMS.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_collapse_norm import (  # noqa: E501
    _norm,
    hc_post_collapse_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)

EPS = 1e-6


def _inputs(n, d, post, cell):
    gen = gen_for(cell)
    res = torch.randn(n, 4, d, generator=gen, device=DEV).bfloat16()
    pre = torch.rand(n, 4, generator=gen, device=DEV) + 0.05
    w = (1 + 0.1 * torch.randn(d, generator=gen, device=DEV)).bfloat16()
    if not post:
        return None, res, None, None, pre, w, EPS
    x = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
    p = 2 * torch.rand(n, 4, generator=gen, device=DEV)
    c = torch.softmax(torch.randn(n, 4, 4, generator=gen, device=DEV), dim=-1)
    return x, res, p, c, pre, w, EPS


@pytest.mark.parametrize(
    "cell", hc_post_collapse_norm.CELLS, ids=[c.why[:44] for c in hc_post_collapse_norm.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_post_collapse_norm)
    s = cell.spec
    mx, late = s.get("mxfp8", False), s.get("residual_after_wait", False)
    args = _inputs(s["n"], s["d"], s["post"], cell)
    got = hc_post_collapse_norm(*args, mxfp8=mx, residual_after_wait=late)
    ref = hc_post_collapse_norm.reference(*args, mxfp8=mx)
    assert len(got) == len(ref) == (3 if mx else 2)
    assert got[0].dtype == got[1].dtype == torch.bfloat16
    assert_within_ulp(got[0], ref[0], element_ulp=2.0, rms_ulp=0.2)
    # `h` against the reference collapse and norm of the kernel's own `residual_cur` (gated above): over a reference
    # `residual_cur` the post update's own 2 units would leak into `h` at large row counts
    h_ref = _norm(hc_pre.reference(got[0], args[4]), args[5], args[6])
    assert_within_ulp(got[1], h_ref, element_ulp=2.0, rms_ulp=0.3)
    if mx:
        _assert_mxfp8_copy(got)


def _assert_mxfp8_copy(got):
    """The launch's MXFP8 copy is `mxfp8_quantize(h, True)` of its own `h`, bit for bit in the E4M3 rows; the scale
    tiles' padding rows are left unwritten (torch.empty) and no product reads them, so the quantized GEMM over the copy
    equals the one over a separate quantize."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
        csa2_mxfp8_linear,
        mxfp8_row_scales,
        mxfp8_swizzled_scales,
    )

    n = got[1].shape[0]
    want = torch.ops.trtllm.mxfp8_quantize(got[1].contiguous(), True)
    assert torch.equal(got[2][0].view(torch.uint8), want[0].view(torch.uint8))
    gen = torch.Generator(device=DEV).manual_seed(n)
    w = (torch.randn(1792, 5120, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    s = mxfp8_row_scales(
        torch.randint(121, 127, (56, 160), generator=gen, device=DEV, dtype=torch.uint8),
        (1792, 5120),
    )
    alpha = torch.ones(1, device=DEV)
    a = csa2_mxfp8_linear(got[1], w, s, mxfp8_swizzled_scales(s), alpha, 1280, x_mxfp8=got[2])
    b = csa2_mxfp8_linear(got[1], w, s, mxfp8_swizzled_scales(s), alpha, 1280)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("n,post", [(24, True), (64, False)])
def test_mxfp8_leaves_the_rest_unchanged(n, post):
    """`mxfp8` only adds the copy: the residual and `h` are those of the plain launch, bit for bit."""
    args = _inputs(n, 5120, post, hc_post_collapse_norm.CELLS[0])
    plain, got = hc_post_collapse_norm(*args), hc_post_collapse_norm(*args, mxfp8=True)
    assert len(plain) == 2 and len(got) == 3
    assert torch.equal(plain[0], got[0]) and torch.equal(plain[1], got[1])
    _assert_mxfp8_copy(got)


@pytest.mark.parametrize("n", [96, 1280])
def test_residual_after_wait_gives_the_same_bits(n):
    """`residual_after_wait` only moves the residual load behind the dependency wait."""
    args = _inputs(n, 5120, True, hc_post_collapse_norm.CELLS[0])
    early = hc_post_collapse_norm(*args)
    late = hc_post_collapse_norm(*args, residual_after_wait=True)
    assert all(torch.equal(a, b) for a, b in zip(early, late))


def test_cuda_graph_replay_is_bitwise_the_eager_result():
    args = list(_inputs(6, 5120, True, hc_post_collapse_norm.CELLS[0]))
    hc_post_collapse_norm(*args)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_post_collapse_norm(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = hc_post_collapse_norm(*args)
    args[1].copy_(torch.randn_like(args[1].float()).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    for g, e in zip(out, hc_post_collapse_norm(*args)):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


def test_rejections():
    ok = _inputs(6, 5120, True, hc_post_collapse_norm.CELLS[0])
    hc_post_collapse_norm.is_valid(*ok)
    with pytest.raises(AssertionError, match="split boundary's rows"):
        hc_post_collapse_norm.is_valid(*_inputs(4097, 5120, True, hc_post_collapse_norm.CELLS[0]))
    with pytest.raises(AssertionError, match="go together"):
        hc_post_collapse_norm.is_valid(None, *ok[1:])
    with pytest.raises(AssertionError):
        hc_post_collapse_norm.is_valid(ok[0], ok[1].float(), *ok[2:])
