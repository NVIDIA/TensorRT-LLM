# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_mega_boundary entry: DeepGEMM's one-kernel lagged mHC boundary (post update of the previous
sub-layer, mixes, Sinkhorn, collapse, RMSNorm) against the float64 composition of `hc_post_mapping` and
`hc_pre_mapping`, and against `hc_boundary`'s split form (the kernels the dep2 target runs below `MIN_ROWS`).

Bands: as `test_modeling_v2_hc_boundary` (the mixer product is TF32-class: `post` rtol 1e-4 + atol 1e-3, `comb`
rtol 1e-3 + atol 5e-3, `pre` as `post`; `residual_cur` is the post update, 2 bf16 units elementwise and 0.2 ulp RMS;
`h` collapses `residual_cur` with the *input* `pre_mix`, 3 ulp elementwise and 0.5 ulp RMS).
"""

import sys

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mega_boundary import (
    MIN_ROWS,
    PREWARM_ROWS,
    hc_mega_boundary,
    hc_mega_boundary_prewarm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (
    hc_pre_mapping,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3)),
    reason="DeepGEMM's mega_mhc on SM103",
)

NORM_EPS, RMS_EPS, HC_EPS, POST_MULT, ITERS, HC = 1e-6, 1e-6, 1e-6, 2.0, 20, 4


def _inputs(n, d, cell):
    gen = gen_for(cell)
    res = torch.randn(n, HC, d, generator=gen, device=DEV).bfloat16()
    mix = (2 + HC) * HC
    fn = torch.randn(mix, HC * d, generator=gen, device=DEV) * (2.0 / (HC * d) ** 0.5)
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(mix, generator=gen, device=DEV) * 0.1
    pre_mix = torch.rand(n, HC, generator=gen, device=DEV) + 0.05
    norm_w = (1 + 0.1 * torch.randn(d, generator=gen, device=DEV)).bfloat16()
    x_prev = torch.randn(n, d, generator=gen, device=DEV).bfloat16()
    post_p = 2 * torch.rand(n, HC, generator=gen, device=DEV)
    comb_p = torch.softmax(torch.randn(n, HC, HC, generator=gen, device=DEV), dim=-1)
    return (
        x_prev,
        res,
        post_p,
        comb_p,
        pre_mix,
        fn,
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
    torch.testing.assert_close(pre, rpre.float(), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(post, rpost.float(), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb, rcomb.float(), rtol=1e-3, atol=5e-3)
    assert_within_ulp(h, rh, element_ulp=3.0, rms_ulp=0.5)


@pytest.mark.parametrize(
    "cell", hc_mega_boundary.CELLS, ids=[c.why[:44] for c in hc_mega_boundary.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_mega_boundary)
    s = cell.spec
    args = _inputs(s["n"], s["d"], cell)
    assert hc_mega_boundary.available(s["n"], s["d"])
    _check(hc_mega_boundary(*args), hc_mega_boundary.reference(*args))


def test_matches_the_split_kernels_the_target_runs_otherwise():
    """The dep2 target switches between this and `hc_post_mapping` + `hc_pre_mapping` by row count: the post
    update is the same function (bitwise), the mixes and `h` agree within the TF32 band."""
    cell = hc_mega_boundary.CELLS[1]
    args = _inputs(cell.spec["n"], cell.spec["d"], cell)
    cur, pre, post, comb, h = hc_mega_boundary(*args)
    x_prev, res, post_p, comb_p, pre_mix, fn, scale, base, norm_w, *rest = args
    cur2 = hc_post_mapping(x_prev, res, post_p, comb_p)
    pre2, post2, comb2, h2 = hc_pre_mapping(cur2, pre_mix, fn, scale, base, norm_w, *rest)
    assert torch.equal(cur, cur2)
    torch.testing.assert_close(pre, pre2, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(post, post2, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb, comb2, rtol=1e-3, atol=5e-3)
    assert_within_ulp(h, h2, element_ulp=3.0, rms_ulp=0.5)


@pytest.mark.parametrize("n", [MIN_ROWS, 1280, 1300])
def test_mxfp8_copy_of_the_layer_input(n):
    """With `mxfp8` the same launch also writes `h` as `mxfp8_quantize(h, True)` gives it, bit for bit, including the
    zeroed padding rows of the last scale tile (the dep attention boundary hands it to the q_a | kv GEMM)."""
    cell = hc_mega_boundary.CELLS[0]
    args = _inputs(n, 5120, cell)
    plain = hc_mega_boundary(*args)
    got = hc_mega_boundary(*args, mxfp8=True)
    assert len(plain) == 5 and len(got) == 6
    for a, b in zip(plain, got[:5]):
        assert torch.equal(a, b)
    want = torch.ops.trtllm.mxfp8_quantize(got[4].contiguous(), True)
    assert torch.equal(got[5][0].view(torch.uint8), want[0].view(torch.uint8))
    assert torch.equal(got[5][1].view(torch.uint8), want[1].view(torch.uint8).reshape(-1))
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


def test_repeats_bitwise_and_replays_under_a_graph():
    cell = hc_mega_boundary.CELLS[0]
    args = list(_inputs(MIN_ROWS, 5120, cell))
    first = hc_mega_boundary(*args)
    again = hc_mega_boundary(*args)
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph
    ):  # creates the default capture stream: not yet warm, so the op is refused there
        pass
    assert hc_mega_boundary.available(MIN_ROWS, 5120)
    hc_mega_boundary(*args)  # eager: also warms the default capture stream
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert hc_mega_boundary.available(MIN_ROWS, 5120)
        out = hc_mega_boundary(*args)
    args[1].copy_(torch.randn_like(args[1].float()).bfloat16())
    args[4].copy_(torch.rand_like(args[4]) + 0.05)
    graph.replay()
    torch.cuda.synchronize()
    eager = hc_mega_boundary(*args)
    for g, e in zip(out, eager):
        assert torch.equal(g, e)


def test_available_range():
    assert not hc_mega_boundary.available(MIN_ROWS - 1, 5120)
    assert hc_mega_boundary.available(MIN_ROWS, 5120)
    assert not hc_mega_boundary.available(MIN_ROWS, 5120 + 512)


def test_rejections():
    cell = hc_mega_boundary.CELLS[0]
    ok = _inputs(MIN_ROWS, 5120, cell)
    hc_mega_boundary.is_valid(*ok)

    def bad(i, value, match=None):
        a = list(ok)
        a[i] = value
        with pytest.raises(AssertionError, match=match):
            hc_mega_boundary.is_valid(*a)

    bad(1, ok[1].float(), None)
    bad(2, ok[2].bfloat16(), None)
    bad(4, ok[4].double(), None)
    bad(10, 0, "post scale")
    bad(14, 1.0, "post scale")
    small = _inputs(MIN_ROWS - 1, 5120, cell)
    with pytest.raises(AssertionError, match="range"):
        hc_mega_boundary.is_valid(*small)


def _num_splits(rows: int, num_sms: int, k_blocks: int) -> int:
    """DeepGEMM's split-K choice for `mega_mhc` (`csrc/apis/mega_mhc.hpp` with the target's patch): the most splits
    up to 64 (at least 16) the SMs hold per 64-row block, then fewer when they fill one wave fewer more evenly."""
    cdiv = lambda a, b: -(-a // b)  # noqa: E731
    m = cdiv(rows, 64)
    splits = cdiv(k_blocks, cdiv(k_blocks, min(64, max(16, num_sms // m))))
    waves = cdiv(m * splits, num_sms)
    fewer = (waves - 1) * num_sms // m
    return (
        fewer
        if fewer > 0 and (waves - 1) * cdiv(k_blocks, fewer) < waves * cdiv(k_blocks, splits)
        else splits
    )


@pytest.mark.parametrize("num_sms", [132, 148, 160, 192])
def test_prewarm_rows_cover_every_split_count(num_sms):
    """Every split-K count (a JIT template parameter) of 1024 .. 2^20 rows is met by some `PREWARM_ROWS` row count."""
    for k_blocks in (5120 // 64, 4 * 5120 // 64):
        warm = {_num_splits(r, num_sms, k_blocks) for r in PREWARM_ROWS}
        every = {_num_splits(r, num_sms, k_blocks) for r in range(MIN_ROWS, 1 << 20, 64)}
        assert every <= warm


def test_prewarm_runs_once(monkeypatch):
    """With fresh prewarm state, the first call launches the kernel once per `PREWARM_ROWS` row count for each
    output form (bf16, then bf16 plus MXFP8) and returns the row counts; the second launches nothing."""
    if not hc_mega_boundary.available(PREWARM_ROWS[-1], 5120):
        pytest.skip("mega_mhc not available here")
    module = sys.modules[hc_mega_boundary_prewarm.__module__]
    real, launches = module.hc_mega_boundary, []

    class Recording:
        def available(self, *args, **kwargs):
            return real.available(*args, **kwargs)

        def __call__(self, x, *args, **kwargs):
            launches.append((x.shape[0], bool(kwargs.get("mxfp8", False))))
            return real(x, *args, **kwargs)

    monkeypatch.setattr(module, "_prewarmed", set())
    monkeypatch.setattr(module, "hc_mega_boundary", Recording())
    assert hc_mega_boundary_prewarm(torch.device(DEV), 5120) == PREWARM_ROWS
    want = [(r, mx) for mx in (False, True) for r in PREWARM_ROWS]
    assert launches == want
    assert hc_mega_boundary_prewarm(torch.device(DEV), 5120) == ()
    assert launches == want
