# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_pre_mapping entry: the engine's lagged mHC boundary (mixes, Sinkhorn, collapse, RMSNorm).

Bands, against the float64 composition of the plain-torch entries (`reference`):

* `pre`, `post`, `comb`: the engine kernel accumulates the K = hc * d mixer product in fp32 on CUDA cores
  (the fallback tactic outside the autotuner) and runs the sigmoid / softmax / 20 Sinkhorn normalizations in
  fp32. Measured max abs error against float64 over the cells: 2e-7 to 5e-7 (a few fp32 ulps of values <= 2,
  the contractive Sinkhorn loop does not amplify it); the gate is atol 2e-6 + rtol 1e-5, i.e. ~4x the
  measurement, and tighter than the plain-torch entry's own (2e-5 / 2e-4, which includes cuBLAS blocking).
  The DeepGEMM TF32 split-K tactic the autotuner may pick rounds every product to 2^-11 (measured max 2.3e-4):
  its band is the engine's own (`post` rtol 1e-4 + atol 1e-3, `comb` rtol 1e-3 + atol 5e-3).
* `h`: the collapse is rounded to bf16 before the norm in the kernel and in the reference; fp32 vs float64
  accumulation of 4 products can flip a bf16 rounding of the collapse (1 bf16 ulp) and the output rounding
  adds one more: 2 ulp of the row's largest value elementwise and 0.2 ulp RMS in aggregate.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (  # noqa: E501
    hc_pre_mapping,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)

NORM_EPS, RMS_EPS, HC_EPS, POST_MULT, ITERS = 1e-6, 1e-6, 1e-6, 2.0, 20


def _inputs(n, hc, d, cell):
    gen = gen_for(cell)
    x = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    mix = (2 + hc) * hc
    fn = torch.randn(mix, hc * d, generator=gen, device=DEV) * (
        2.0 / (hc * d) ** 0.5
    )  # mixes of order 2
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(mix, generator=gen, device=DEV) * 0.1
    pre_mix = torch.rand(n, hc, generator=gen, device=DEV) + 0.05
    norm_w = (1 + 0.1 * torch.randn(d, generator=gen, device=DEV)).bfloat16()
    return (x, pre_mix, fn, scale, base, norm_w, hc, ITERS, RMS_EPS, NORM_EPS, HC_EPS, POST_MULT)


def _check(got, ref, n, hc, mix_tol):
    assert [t.shape for t in got] == [(n, hc), (n, hc), (n, hc, hc), ref[3].shape]
    assert [t.dtype for t in got] == [torch.float32] * 3 + [torch.bfloat16]
    if n == 0:
        return
    for g, r in zip(got[:3], ref[:3]):
        torch.testing.assert_close(g, r, **mix_tol)
    assert_within_ulp(got[3], ref[3], element_ulp=2.0, rms_ulp=0.2)


@pytest.mark.parametrize(
    "cell", hc_pre_mapping.CELLS, ids=[c.why[:44] for c in hc_pre_mapping.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_pre_mapping)
    s = cell.spec
    args = _inputs(s["n"], s["hc"], s["d"], cell)
    got = hc_pre_mapping(*args)
    ref = hc_pre_mapping.reference(*args)
    _check(got, ref, s["n"], s["hc"], dict(atol=2e-6, rtol=1e-5))


def test_matches_the_plain_torch_chain():
    """The fused entry against the three entries it replaces, called as the target called them."""
    from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes import hc_mixes
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre

    cell = hc_pre_mapping.CELLS[1]
    args = _inputs(18, 4, 5120, cell)
    x, pre_mix, fn, scale, base, norm_w = args[:6]
    got = hc_pre_mapping(*args)
    pre, post, comb = hc_mixes(x, fn, scale, base, 4, ITERS, RMS_EPS, HC_EPS)
    h = flashinfer_rmsnorm(hc_pre(x, pre_mix), norm_w, NORM_EPS)
    for g, r in zip(got[:3], (pre, post, comb)):
        torch.testing.assert_close(g, r, atol=1e-4, rtol=1e-3)
    assert_within_ulp(got[3], h, element_ulp=2.0, rms_ulp=0.3)


def test_tf32_splitk_tactic_is_within_the_engines_band():
    from tensorrt_llm._torch.modules.mhc import mhc_cuda

    if mhc_cuda._get_dg_fn() is None:
        pytest.skip("DeepGEMM tf32_hc_prenorm_gemm is not available")
    cell = hc_pre_mapping.CELLS[2]
    args = _inputs(96, 4, 5120, cell)
    x, pre_mix, fn, scale, base, norm_w = args[:6]
    runner = mhc_cuda._get_pre_mapping_runner(4, 5120, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, ITERS)
    post, comb, h, pre = runner(
        [x.view(96, -1), fn, x, scale, base],
        tactic=("dg_splitk", 0, 0, 256),
        norm_weight=norm_w,
        norm_eps=NORM_EPS,
        pre_mix_ext=pre_mix,
    )
    ref = hc_pre_mapping.reference(*args)
    torch.testing.assert_close(post, ref[1], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb.view(96, 4, 4), ref[2], rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(pre, ref[0], rtol=1e-4, atol=1e-3)
    assert_within_ulp(h, ref[3], element_ulp=4.0, rms_ulp=1.0)


@pytest.mark.parametrize("n", [6, 96, 1280])
def test_cuda_graph_replay_is_bitwise_the_eager_result(n):
    """No host sync and static shapes: the captured op, replayed on new data, equals eager on that data."""
    cell = hc_pre_mapping.CELLS[0]
    args = list(_inputs(n, 4, 5120, cell))
    hc_pre_mapping(*args)  # warm up (kernel / tactic selection) outside the capture
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_pre_mapping(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = hc_pre_mapping(*args)
    args[0].copy_(torch.randn_like(args[0].float()).bfloat16())  # new data, same buffers
    args[1].copy_(torch.rand_like(args[1]) + 0.05)
    graph.replay()
    torch.cuda.synchronize()
    eager = hc_pre_mapping(*args)
    for g, e in zip(out, eager):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


def test_rejections():
    cell = hc_pre_mapping.CELLS[0]
    ok = _inputs(6, 4, 64, cell)
    hc_pre_mapping.is_valid(*ok)

    def bad(i, value, match=None):
        a = list(ok)
        a[i] = value
        with pytest.raises(AssertionError, match=match):
            hc_pre_mapping.is_valid(*a)

    bad(0, ok[0].float(), "bf16")
    bad(0, ok[0].transpose(0, 2), None)
    bad(2, ok[2].bfloat16())
    bad(1, ok[1].double())
    bad(5, ok[5].float())
    bad(7, 0, "iteration")
    bad(11, 1.0, "post scale")
