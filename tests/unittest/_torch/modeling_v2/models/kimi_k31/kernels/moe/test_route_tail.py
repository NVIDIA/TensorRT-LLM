# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's routing-tail kernel (``route_tail.k31_route_topk``).

The kernel's claim is bitwise: the experts, their order and the BF16 weights equal
``noaux_tc_op`` (one expert group, top-16 of 896) followed by the BF16 cast of its FP32
weights, which is what the routed-expert scheduler hands a TRTLLM-Gen backend. So this
compares against the engine op itself, exactly, over the row counts the kernel's grid
changes shape at, the tie fallback, the saturated sigmoid tails and a CUDA-graph replay.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("the routing tail is a CuTe DSL kernel", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (  # noqa: E402
    route_tail,
)

assert torch.cuda.is_available(), "the routing tail requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
    pytest.skip("the routing tail runs on SM100 / SM103 / SM107", allow_module_level=True)

E, K = route_tail.NUM_EXPERTS, route_tail.TOP_K


def _inputs(rows: int, seed: int, logit_scale: float = 1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, E, generator=g, device="cuda", dtype=torch.float32) * logit_scale
    bias = torch.randn(E, generator=g, device="cuda", dtype=torch.float32) * 0.1
    return logits, bias


def _engine(logits, bias, scale):
    """The path the kernel replaces: noaux_tc_op, then the BF16 cast of its weights."""
    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, K, scale)
    return ids, weights.to(torch.bfloat16)


def _assert_bitwise(got, want):
    ids, weights = got
    ref_ids, ref_weights = want
    assert ids.dtype is torch.int32 and weights.dtype is torch.bfloat16
    assert ids.shape == ref_ids.shape == weights.shape
    assert torch.equal(ids, ref_ids), "selected experts (or their order) differ from noaux_tc_op"
    assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16)), (
        "BF16 weights differ from bf16(noaux_tc_op weights)"
    )


# Grid edges: one 8-row CTA; the 64/65 boundary where the attention-DP MoE switches from its fused
# routing to this kernel; 256/257 (the one-cluster kernel's row limit); the last 8-row CTA count
# (1024) and the first 16-row one (1025); the largest launch.
ROWS = (1, 7, 8, 64, 65, 112, 256, 257, 448, 1000, 1024, 1025, 1536, 2048)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("scale", (1.0, 2.5))
def test_bitwise_vs_noaux_tc(rows: int, scale: float) -> None:
    logits, bias = _inputs(rows, seed=rows)
    got = route_tail.k31_route_topk(logits, bias, scale)
    _assert_bitwise(got, _engine(logits, bias, scale))


def test_heavy_ties_take_the_exact_fallback() -> None:
    """Many equal scores overflow a warp's candidate list: the team's exact arg-max rounds run.
    Ties go to the lower expert id, as in noaux_tc_op."""
    rows = 96
    logits, bias = _inputs(rows, seed=7)
    logits[: rows // 2] = 0.25  # whole rows of equal scores (with a zero bias below)
    logits[rows // 2 :, ::3] = 1.0  # a third of the experts tied at the top
    bias.zero_()
    got = route_tail.k31_route_topk(logits, bias, 1.0)
    _assert_bitwise(got, _engine(logits, bias, 1.0))


def test_saturated_sigmoid_tails() -> None:
    """Large logits saturate the tanh-form sigmoid to exactly 0 / 1 in both implementations."""
    logits, bias = _inputs(130, seed=11, logit_scale=40.0)
    got = route_tail.k31_route_topk(logits, bias, 1.0)
    _assert_bitwise(got, _engine(logits, bias, 1.0))


def test_cuda_graph_replay() -> None:
    """No state between launches: a captured launch replays to the same result on new inputs."""
    rows = 448
    logits, bias = _inputs(rows, seed=3)
    route_tail.k31_route_topk(logits, bias, 1.0)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = route_tail.k31_route_topk(logits, bias, 1.0)
    for seed in (4, 5):
        new_logits, new_bias = _inputs(rows, seed=seed)
        logits.copy_(new_logits)
        bias.copy_(new_bias)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(out, _engine(logits, bias, 1.0))


@pytest.mark.parametrize(
    "bad",
    (
        "bf16_logits",
        "width",
        "strided",
        "bias_size",
        "too_many_rows",
    ),
)
def test_rejects_inputs_it_cannot_route(bad: str) -> None:
    logits, bias = _inputs(16, seed=1)
    if bad == "bf16_logits":
        logits = logits.to(torch.bfloat16)
    elif bad == "width":
        logits = torch.randn(16, 512, device="cuda")
    elif bad == "strided":
        logits = torch.randn(16, 2 * E, device="cuda")[:, ::2]
    elif bad == "bias_size":
        bias = bias[:448].contiguous()
    elif bad == "too_many_rows":
        logits = torch.randn(route_tail.WIDE_MAX_M + 1, E, device="cuda")
    with pytest.raises(ValueError):
        route_tail.k31_route_topk(logits, bias, 1.0)
