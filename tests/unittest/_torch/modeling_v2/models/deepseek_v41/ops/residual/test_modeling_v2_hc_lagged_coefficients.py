# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_lagged_coefficients entry (one-launch coefficients of an attention-DP split boundary).

Against the float64 `hc_mixes` reference, in the `hc_coefficients` band (`post`/`pre` rtol 1e-4 + atol 1e-3, `comb`
rtol 1e-3 + atol 5e-3). Against the plain paths the targets ran before:

* 65 .. 4096 rows (mma.sync, tcgen05) against the single-pass `hc_boundary` of the same residual (the targets' boundary
  past 64 rows), the same TF32-split weight and bf16 products: the mma.sync kernel is bit for bit its stream kernels
  (197 rows on); its tile kernel (up to 196 rows) and the tcgen05 kernel differ in fp32 summation order only;
* 17 .. 64 rows (small-batch kernel, exact fp32 products): against the two-launch `hc_coefficients` (the targets' split
  coefficients until now) the difference is fp32 summation order only: `pre`/`post` within 2e-6 relative + 1e-6,
  `comb` (20 Sinkhorn iterations) within 1e-5 relative + 1e-6.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_boundary import hc_boundary
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_coefficients import (
    hc_coefficients,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_lagged_coefficients import (
    hc_lagged_coefficients,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)

EPS, ITERS, HC = 1e-6, 20, 4


def _inputs(n, d, cell):
    gen = gen_for(cell)
    res = torch.randn(n, HC, d, generator=gen, device=DEV).bfloat16()
    fn = torch.randn(24, HC * d, generator=gen, device=DEV) * (2.0 / (HC * d) ** 0.5)
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(24, generator=gen, device=DEV) * 0.1
    bw = hc_boundary.prepare_weight(fn)
    state = hc_lagged_coefficients.prepare_state(torch.device(DEV))
    return res, state, fn, bw, scale, base, HC, ITERS, EPS, EPS, 2.0


@pytest.mark.parametrize(
    "cell", hc_lagged_coefficients.CELLS, ids=[c.why[:44] for c in hc_lagged_coefficients.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_lagged_coefficients)
    args = _inputs(cell.spec["n"], cell.spec["d"], cell)
    n = cell.spec["n"]
    assert hc_lagged_coefficients.ready(n, cell.spec["d"], ITERS, args[1], args[2]) == (n > 16)
    got, ref = hc_lagged_coefficients(*args), hc_lagged_coefficients.reference(*args)
    assert [t.shape for t in got] == [(n, 4), (n, 4), (n, 4, 4)] and all(
        t.dtype == torch.float32 for t in got
    )
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[2], ref[2], rtol=1e-3, atol=5e-3)


def _single_pass(res, fn, bw, scale, base):
    n = res.shape[0]
    pre_mix = torch.full((n, HC), 0.25, device=DEV)
    norm = torch.ones(res.shape[2], device=DEV).bfloat16()
    return hc_boundary(
        None, res, None, None, pre_mix, fn, bw, scale, base, norm, HC, ITERS, EPS, EPS, EPS, 2.0
    )[1:4]


@pytest.mark.parametrize("n", [197, 496, 1023])
def test_mma_rows_are_bitwise_the_stream_boundary(n):
    """The mma.sync kernel against `hc_boundary` where it runs its stream + finalize kernels (more than
    `MAX_TOKENS` = 196 rows): the same coefficients bit for bit (at 1023 rows too, before the tcgen05 weight exists)."""
    res, state, fn, bw, scale, base, *rest = _inputs(n, 5120, hc_lagged_coefficients.CELLS[5])
    assert hc_lagged_coefficients.kernel(n, state) == "mma"
    got = hc_lagged_coefficients(res, state, fn, bw, scale, base, *rest)
    for g, e in zip(got, _single_pass(res, fn, bw, scale, base)):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


@pytest.mark.parametrize("n", [65, 96, 150, 196, 512, 1023, 1280])
def test_rows_against_the_single_pass_boundary_are_fp32_order_only(n):
    """Against `hc_boundary` (the targets' boundary past 64 rows): its tile kernel (up to 196 rows) and the tcgen05
    kernel (512 rows on) sum the same TF32-split products in other fp32 orders. Measured: `pre` / `post` 3.1e-6
    relative at most (tile), 1.4e-5 (tcgen05); gated at 5e-5 relative + 1e-6, `comb` (20 Sinkhorn iterations) at 2e-4
    relative + 1e-6."""
    res, state, fn, bw, scale, base, *rest = _inputs(n, 5120, hc_lagged_coefficients.CELLS[5])
    assert hc_lagged_coefficients.ready(n, 5120, ITERS, state, fn)
    assert hc_lagged_coefficients.kernel(n, state) == ("tc" if n >= 512 else "mma")
    got = hc_lagged_coefficients(res, state, fn, bw, scale, base, *rest)
    plain = _single_pass(res, fn, bw, scale, base)
    torch.testing.assert_close(got[0], plain[0], rtol=5e-5, atol=1e-6)
    torch.testing.assert_close(got[1], plain[1], rtol=5e-5, atol=1e-6)
    torch.testing.assert_close(got[2], plain[2], rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("n", [17, 24, 48, 64])
def test_small_rows_against_the_two_launch_path_are_fp32_order_only(n):
    cell = hc_lagged_coefficients.CELLS[0]
    res, state, fn, bw, scale, base, *rest = _inputs(n, 5120, cell)
    got = hc_lagged_coefficients(res, state, fn, bw, scale, base, *rest)
    plain = hc_coefficients(res, fn, scale, base, *rest)
    torch.testing.assert_close(got[0], plain[0], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(got[1], plain[1], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(got[2], plain[2], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n", [48, 96, 1280])
def test_cuda_graph_replay_is_bitwise_the_eager_result(n):
    args = list(_inputs(n, 5120, hc_lagged_coefficients.CELLS[0]))
    assert hc_lagged_coefficients.ready(n, 5120, ITERS, args[1], args[2])
    hc_lagged_coefficients(*args)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_lagged_coefficients(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        assert hc_lagged_coefficients.ready(n, 5120, ITERS, args[1], args[2])
        out = hc_lagged_coefficients(*args)
    args[0].copy_(torch.randn_like(args[0].float()).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    for g, e in zip(out, hc_lagged_coefficients(*args)):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


def test_rejections():
    ok = _inputs(48, 5120, hc_lagged_coefficients.CELLS[0])
    hc_lagged_coefficients.is_valid(*ok)
    with pytest.raises(AssertionError):
        hc_lagged_coefficients.is_valid(ok[0].float(), *ok[1:])
    with pytest.raises(AssertionError, match="split boundary"):
        big = torch.zeros(4097, HC, 5120, device=DEV).bfloat16()
        hc_lagged_coefficients.is_valid(big, *ok[1:])
