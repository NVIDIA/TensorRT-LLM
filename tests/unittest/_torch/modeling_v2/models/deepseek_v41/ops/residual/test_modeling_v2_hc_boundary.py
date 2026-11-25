# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_boundary entry: the engine's single-pass lagged mHC boundary (post update of the previous
sub-layer, mixes, Sinkhorn, collapse, RMSNorm) against the float64 composition of `hc_post_mapping` and
`hc_pre_mapping`.

Bands: the kernel multiplies the mixer weight as two bf16 halves of a TF32 rounding against the bf16 residual
(the engine's TF32-class product), so the mixes carry the TF32 band of `test_modeling_v2_hc_pre_mapping`
(`post` rtol 1e-4 + atol 1e-3, `comb` rtol 1e-3 + atol 5e-3, `pre` as `post`). `residual_cur` is the post
update, the same function as `hc_post_mapping` (2 bf16 units, 0.2 ulp RMS). `h` is a bf16 collapse and norm of
it (2 ulp elementwise, 0.4 ulp RMS: the mixes are TF32-class and feed `h` only through the *previous* `pre`,
which is an input here, so `h` is as tight as the post-mapping plus the collapse).
"""

import pytest
import torch
from _v41_util import DEV, gen_for

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_boundary import hc_boundary

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)

NORM_EPS, RMS_EPS, HC_EPS, POST_MULT, ITERS, HC = 1e-6, 1e-6, 1e-6, 2.0, 20, 4


def _inputs(n, d, post, cell):
    gen = gen_for(cell)
    res = torch.randn(n, HC, d, generator=gen, device=DEV).bfloat16()
    mix = (2 + HC) * HC
    fn = torch.randn(mix, HC * d, generator=gen, device=DEV) * (2.0 / (HC * d) ** 0.5)
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(mix, generator=gen, device=DEV) * 0.1
    pre_mix = torch.rand(n, HC, generator=gen, device=DEV) + 0.05
    norm_w = (1 + 0.1 * torch.randn(d, generator=gen, device=DEV)).bfloat16()
    x_prev = post_p = comb_p = None
    if post:
        x_prev = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
        post_p = 2 * torch.rand(n, HC, generator=gen, device=DEV)
        comb_p = torch.softmax(torch.randn(n, HC, HC, generator=gen, device=DEV), dim=-1)
    bw = hc_boundary.prepare_weight(fn)
    return (
        x_prev,
        res,
        post_p,
        comb_p,
        pre_mix,
        fn,
        bw,
        scale,
        base,
        norm_w,
        HC,
        ITERS,
        RMS_EPS,
        NORM_EPS,
        HC_EPS,
        POST_MULT,
    )


def _check(got, ref):
    cur, pre, post, comb, h = got
    rcur, rpre, rpost, rcomb, rh = ref
    assert [t.dtype for t in got] == [
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.bfloat16,
    ]
    assert_within_ulp(cur, rcur, element_ulp=2.0, rms_ulp=0.2)
    torch.testing.assert_close(pre, rpre, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(post, rpost, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb, rcomb, rtol=1e-3, atol=5e-3)
    assert_within_ulp(h, rh, element_ulp=3.0, rms_ulp=0.5)


@pytest.mark.parametrize("cell", hc_boundary.CELLS, ids=[c.why[:44] for c in hc_boundary.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_boundary)
    s = cell.spec
    args = _inputs(s["n"], s["d"], s["post"], cell)
    _check(hc_boundary(*args), hc_boundary.reference(*args))


def test_without_post_update_the_residual_is_returned_as_is():
    cell = hc_boundary.CELLS[4]
    args = _inputs(6, 5120, False, cell)
    cur = hc_boundary(*args)[0]
    assert cur.data_ptr() == args[1].data_ptr()


@pytest.mark.parametrize("n", [6, 96])
def test_cuda_graph_replay_is_bitwise_the_eager_result(n):
    cell = hc_boundary.CELLS[0]
    args = list(_inputs(n, 5120, True, cell))
    assert hc_boundary.available(n, 5120, True, ITERS)
    hc_boundary(*args)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_boundary(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = hc_boundary(*args)
    args[1].copy_(torch.randn_like(args[1].float()).bfloat16())
    args[4].copy_(torch.rand_like(args[4]) + 0.05)
    graph.replay()
    torch.cuda.synchronize()
    eager = hc_boundary(*args)
    for g, e in zip(out, eager):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


@pytest.mark.parametrize("n,post", [(24, True), (160, True), (160, False)])
def test_mxfp8_copy_of_the_layer_input(n, post):
    """With `mxfp8` the same launch also writes the layer input as `mxfp8_quantize(h, True)` gives it, bit for bit
    (the dep attention boundary above 16 rows hands it to the quantized q_a | kv GEMM)."""
    cell = hc_boundary.CELLS[0]
    args = _inputs(n, 5120, post, cell)
    plain = hc_boundary(*args)
    got = hc_boundary(*args, mxfp8=True)
    assert len(plain) == 5 and len(got) == 6
    for a, b in zip(plain, got[:5]):
        assert torch.equal(a, b)
    want = torch.ops.trtllm.mxfp8_quantize(got[4].contiguous(), True)
    assert torch.equal(got[5][0].view(torch.uint8), want[0].view(torch.uint8))
    # the scale tiles' padding rows are left unwritten (torch.empty); the GEMM reads only the valid rows' scales, so the
    # quantized GEMM over the boundary's copy equals the one over a separate quantize
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
        csa2_mxfp8_linear,
        mxfp8_row_scales,
        mxfp8_swizzled_scales,
    )

    gen = torch.Generator(device=DEV).manual_seed(n)
    w = (torch.randn(1792, 5120, generator=gen, device=DEV) * 0.5).to(torch.float8_e4m3fn)
    s = mxfp8_row_scales(
        torch.randint(121, 127, (56, 160), generator=gen, device=DEV, dtype=torch.uint8),
        (1792, 5120),
    )
    alpha = torch.ones(1, device=DEV)
    a = csa2_mxfp8_linear(got[4], w, s, mxfp8_swizzled_scales(s), alpha, 1280, x_mxfp8=got[5])
    b = csa2_mxfp8_linear(got[4], w, s, mxfp8_swizzled_scales(s), alpha, 1280)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("n,post", [(24, True), (160, True), (160, False)])
def test_sinkhorn_on_a_side_stream(n, post):
    """`sinkhorn_side`: the boundary without Sinkhorn iterations, then `mhc_sinkhorn` in place on another stream (the
    engine's deferred boundary). After waiting on `done`, every output equals the single-launch boundary's."""
    cell = hc_boundary.CELLS[0]
    args = _inputs(n, 5120, post, cell)
    plain = hc_boundary(*args)
    side = (torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())
    got = hc_boundary(*args, sinkhorn_side=side)
    torch.cuda.current_stream().wait_event(side[2])
    torch.cuda.synchronize()
    for a, b in zip(got[:3] + got[4:], plain[:3] + plain[4:]):
        assert torch.equal(a, b)
    torch.testing.assert_close(got[3], plain[3], rtol=1e-5, atol=1e-6)
    assert hc_boundary.available(n, 5120, post, ITERS, deferred=True)


def test_available_range():
    assert not hc_boundary.available(0, 5120, True, ITERS)
    assert not hc_boundary.available(1024, 5120, True, ITERS)
    assert hc_boundary.available(4096, 5120, False, ITERS)
    assert not hc_boundary.available(4097, 5120, False, ITERS)


def test_rejections():
    cell = hc_boundary.CELLS[0]
    ok = _inputs(6, 5120, True, cell)
    hc_boundary.is_valid(*ok)

    def bad(i, value, match=None):
        a = list(ok)
        a[i] = value
        with pytest.raises(AssertionError, match=match):
            hc_boundary.is_valid(*a)

    bad(1, ok[1].float(), None)
    bad(0, None, "go together")
    bad(2, ok[2].bfloat16(), None)
    bad(4, ok[4].double(), None)
    bad(6, ok[6].float(), "prepare_weight")
    bad(11, 0, "post scale")
    bad(15, 1.0, "post scale")
    big = _inputs(1024, 5120, True, cell)
    with pytest.raises(AssertionError, match="single-pass range"):
        hc_boundary.is_valid(*big)
