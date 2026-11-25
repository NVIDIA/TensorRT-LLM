# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the dsv4_router_logits entry: the router's fp32 logits over the bf16 gate.

Band: both paths multiply the same bf16 operands with fp32 accumulation; against the float64 product the error is fp32
summation noise over K = 5120 => rtol 1e-5, atol 1e-4 (logits are O(1..10)).
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_router_logits import (  # noqa: E501
    dsv4_router_logits,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


def _inputs(s, gen):
    x = torch.randn(s["m"], s["hidden"], generator=gen, device=DEV).bfloat16()
    w = (torch.randn(s["experts"], s["hidden"], generator=gen, device=DEV) * 0.02).bfloat16()
    return x, w


@pytest.mark.parametrize(
    "cell", dsv4_router_logits.CELLS, ids=[c.why[:44] for c in dsv4_router_logits.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(dsv4_router_logits)
    x, w = _inputs(cell.spec, gen_for(cell))
    got = dsv4_router_logits(x, w, partials=cell.spec.get("partials", False))
    if cell.spec.get("partials") and cell.spec["m"] >= 12:  # (from SMALL_M_MIN_ROWS on sm_103)
        assert hasattr(got, "parts") and got.parts.shape[1:] == (
            cell.spec["m"],
            cell.spec["experts"],
        )
        got = got.parts.sum(0)
    assert got.dtype == torch.float32 and got.shape == (cell.spec["m"], cell.spec["experts"])
    torch.testing.assert_close(got, dsv4_router_logits.reference(x, w), rtol=1e-5, atol=1e-4)


def test_the_two_paths_pick_the_same_experts():
    """24-row steps route like one 96-row step: the top-6 sets agree row by row."""
    gen = torch.Generator(device=DEV).manual_seed(3)
    x, w = _inputs(dict(m=96, hidden=5120, experts=384), gen)
    with validating(dsv4_router_logits):
        small = torch.cat([dsv4_router_logits(x[i : i + 24], w) for i in range(0, 96, 24)])
        big = dsv4_router_logits(x, w)
    assert torch.equal(small.topk(6, -1)[1].sort(-1)[0], big.topk(6, -1)[1].sort(-1)[0])


@pytest.mark.parametrize("m", [1, 5, 24, 36, 48, 64, 65, 1300, 2048])
def test_bitwise_the_engine_gate(m):
    """Without partials the logits are stock's gate's (`router_gemm_autotuned`) bit for bit, whichever tactic its tuner
    holds for the bucket; each tactic of its runner stays in the fp32 band of the float64 product."""
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        RouterGemmRunner,
        router_gemm_autotuned,
    )

    x, w = _inputs(dict(m=m, hidden=5120, experts=384), torch.Generator(device=DEV).manual_seed(m))
    assert torch.equal(dsv4_router_logits(x, w), router_gemm_autotuned(x, w))
    ref = dsv4_router_logits.reference(x, w)
    runner = RouterGemmRunner()
    for tactic in [-1, 0] if runner.small_m_applies([x, w]) else [-1]:
        torch.testing.assert_close(runner([x, w], tactic=tactic), ref, rtol=1e-5, atol=1e-4)


def test_under_cuda_graph():
    gen = torch.Generator(device=DEV).manual_seed(4)
    x, w = _inputs(dict(m=6, hidden=5120, experts=384), gen)
    dsv4_router_logits(x, w)  # compile outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = dsv4_router_logits(x, w)
    x.copy_(torch.randn(x.shape, generator=gen, device=DEV).bfloat16())
    g.replay()
    torch.testing.assert_close(out, dsv4_router_logits.reference(x, w), rtol=1e-5, atol=1e-4)


def test_rejections():
    x, w = _inputs(dict(m=2, hidden=128, experts=16), torch.Generator(device=DEV).manual_seed(0))
    dsv4_router_logits.is_valid(x, w)
    with pytest.raises(AssertionError):
        dsv4_router_logits.is_valid(x.float(), w)
    with pytest.raises(AssertionError):
        dsv4_router_logits.is_valid(x, w[:, :64])


def test_partials_route_like_their_sum():
    """`dsv4_route_nvfp4_quant` over the K-chunk partials picks the experts, weights and quantized rows it picks over
    the same partials summed in order (the kernel's documented summation order)."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
        _sum_partials,
        dsv4_route_nvfp4_quant,
    )

    gen = torch.Generator(device=DEV).manual_seed(9)
    x, w = _inputs(dict(m=160, hidden=5120, experts=384), gen)
    bias = torch.randn(384, generator=gen, device=DEV) * 0.1
    gs = torch.ones(1, device=DEV)
    parts = dsv4_router_logits(x, w, partials=True)
    assert hasattr(parts, "parts")
    with validating(dsv4_route_nvfp4_quant):
        a = dsv4_route_nvfp4_quant(parts, bias, x, gs, 6, 1.5)
    b = dsv4_route_nvfp4_quant(_sum_partials(parts.parts), bias, x, gs, 6, 1.5)
    assert torch.equal(a[0].sort(-1)[0], b[0].sort(-1)[0])
    torch.testing.assert_close(
        a[1].sort(-1)[0].float(), b[1].sort(-1)[0].float(), rtol=1e-2, atol=1e-3
    )
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])


@pytest.mark.parametrize("m", [11, 12, 24, 48, 64, 65, 160])
def test_partials_follow_the_engine_opt_in(m):
    """With `partials` the op serves exactly the rows stock's DeepSeek-V4.1 gate serves as partials (its MoEs opt in
    with `moe_router_partials_small_m`: `applies(small_m=True)`), bit for bit the engine's `router_logit_partials`;
    other rows are the tuned gate GEMM, bit for bit `router_gemm_autotuned`."""
    from tensorrt_llm._torch.custom_ops import router_logit_partials as rlp
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import router_gemm_autotuned

    x, w = _inputs(
        dict(m=m, hidden=5120, experts=384), torch.Generator(device=DEV).manual_seed(100 + m)
    )
    got = dsv4_router_logits(x, w, partials=True)
    if rlp.applies(x, w, small_m=True):
        assert m >= rlp.SMALL_M_MIN_ROWS and hasattr(got, "parts")
        assert torch.equal(got.parts, rlp.router_logit_partials(x, w).parts)
    else:
        assert m < rlp.SMALL_M_MIN_ROWS and torch.equal(got, router_gemm_autotuned(x, w))
