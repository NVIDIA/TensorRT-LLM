# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the hc_coefficients entry (the mixes of a boundary alone) against the float64 `hc_mixes` reference, and for
hc_coefficients_fma (the one-launch CUDA-core kernel of up to 8 rows) against it and against hc_coefficients.

Band: as `hc_pre_mapping`'s mixes, including the engine's TF32 split-K tactic the autotuner may pick
(`post`/`pre` rtol 1e-4 + atol 1e-3, `comb` rtol 1e-3 + atol 5e-3); the exact-fp32 tactic is far inside it.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_coefficients import (
    hc_coefficients,
    hc_coefficients_fma,
    hc_fma_fits,
    hc_fma_state,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the mHC kernels are Blackwell (sm_10x) kernels",
)

EPS, ITERS = 1e-6, 20


def _inputs(n, d, cell):
    gen = gen_for(cell)
    res = torch.randn(n, 4, d, generator=gen, device=DEV).bfloat16()
    fn = torch.randn(24, 4 * d, generator=gen, device=DEV) * (2.0 / (4 * d) ** 0.5)
    scale = torch.tensor([0.5, 0.5, 0.5], device=DEV)
    base = torch.randn(24, generator=gen, device=DEV) * 0.1
    return res, fn, scale, base, 4, ITERS, EPS, EPS, 2.0


@pytest.mark.parametrize(
    "cell", hc_coefficients.CELLS, ids=[c.why[:44] for c in hc_coefficients.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(hc_coefficients)
    args = _inputs(cell.spec["n"], cell.spec["d"], cell)
    got, ref = hc_coefficients(*args), hc_coefficients.reference(*args)
    n = cell.spec["n"]
    assert [t.shape for t in got] == [(n, 4), (n, 4), (n, 4, 4)] and all(
        t.dtype == torch.float32 for t in got
    )
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[2], ref[2], rtol=1e-3, atol=5e-3)


def test_cuda_graph_replay_is_bitwise_the_eager_result():
    args = list(_inputs(6, 5120, hc_coefficients.CELLS[0]))
    hc_coefficients(*args)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_coefficients(*args)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = hc_coefficients(*args)
    args[0].copy_(torch.randn_like(args[0].float()).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    for g, e in zip(out, hc_coefficients(*args)):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


def test_rejections():
    ok = _inputs(6, 5120, hc_coefficients.CELLS[0])
    hc_coefficients.is_valid(*ok)
    with pytest.raises(AssertionError):
        hc_coefficients.is_valid(ok[0].float(), *ok[1:])
    with pytest.raises(AssertionError, match="empty"):
        hc_coefficients.is_valid(ok[0][:0], *ok[1:])


def _fma_args(cell):
    res, fn, scale, base, mult, iters, rms_eps, hc_eps, post = _inputs(
        cell.spec["n"], cell.spec["d"], cell
    )
    weight, counter = hc_fma_state(fn, cell.spec["d"], iters)
    return (res, weight, counter, scale, base, mult, iters, rms_eps, hc_eps, post), (
        res,
        fn,
        scale,
        base,
        mult,
        iters,
        rms_eps,
        hc_eps,
        post,
    )


@pytest.mark.parametrize(
    "cell", hc_coefficients_fma.CELLS, ids=[c.why[:44] for c in hc_coefficients_fma.CELLS]
)
def test_fma_cells(cell, certifying):  # noqa: F811
    certifying(hc_coefficients_fma)
    fma_args, plain_args = _fma_args(cell)
    release = cell.spec.get("release", False)
    got = hc_coefficients_fma(*fma_args, release_after_wait=release)
    ref = hc_coefficients_fma.reference(*fma_args)
    n = cell.spec["n"]
    assert [t.shape for t in got] == [(n, 4), (n, 4), (n, 4, 4)] and all(
        t.dtype == torch.float32 for t in got
    )
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(got[2], ref[2], rtol=1e-3, atol=5e-3)
    # the DeepGEMM path of the same boundary: summation order only (the engine measured up to 2.6e-5 relative on the
    # release weights; these random weights and residuals reach 1.7e-4 absolute on a sigmoid output)
    plain = hc_coefficients(*plain_args)
    for g, p in zip(got, plain):
        torch.testing.assert_close(g, p, rtol=1e-3, atol=5e-4)
    # the release point changes no bit
    other = hc_coefficients_fma(*fma_args, release_after_wait=not release)
    for g, o in zip(got, other):
        assert torch.equal(g, o)


def test_fma_cuda_graph_replay_is_bitwise_the_eager_result():
    fma_args, _ = _fma_args(hc_coefficients_fma.CELLS[0])
    args = list(fma_args)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_coefficients_fma(*args, release_after_wait=True)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = hc_coefficients_fma(*args, release_after_wait=True)
    args[0].copy_(torch.randn_like(args[0].float()).bfloat16())
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for g, e in zip(out, hc_coefficients_fma(*args)):
        torch.testing.assert_close(g, e, rtol=0, atol=0)


def test_fma_rejections():
    fma_args, _ = _fma_args(hc_coefficients_fma.CELLS[0])
    hc_coefficients_fma.is_valid(*fma_args)
    assert hc_fma_fits(8, 5120, 4) and not hc_fma_fits(9, 5120, 4) and not hc_fma_fits(6, 5000, 4)
    nine = (torch.zeros(9, 4, 5120, dtype=torch.bfloat16, device=DEV),) + fma_args[1:]
    with pytest.raises(AssertionError, match="1..8 rows"):
        hc_coefficients_fma.is_valid(*nine)
