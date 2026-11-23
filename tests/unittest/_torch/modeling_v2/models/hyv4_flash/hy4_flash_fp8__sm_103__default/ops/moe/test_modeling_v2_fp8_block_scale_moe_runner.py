# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Hy4-Flash target's fp8_block_scale_moe_runner entry.

Weights are drawn directly in e4m3 / fp32 so both sides read bit-identical values.
"""

import zlib
from collections.abc import Callable, Iterator

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    BLOCK,
    ROUTING_DEEPSEEK_V3,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.moe.fp8_block_scale_moe_runner import (  # noqa: E501
    fp8_block_scale_moe_runner as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "fp8_block_scale_moe_runner requires a CUDA device"

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="fp8_block_scale_moe_runner is certified on sm_103"
)

DEV = "cuda"
CELLS = op.CELLS


def _gate(out: torch.Tensor, ref: torch.Tensor) -> None:
    # The reference models every kernel rounding; what is left is fp32 sum order flipping e4m3
    # roundings, which grows with I * top_k (24576 here).
    assert_within_ulp(out, ref, element_ulp=16.0, rms_ulp=2.5)


def _draw_stack(hidden: int, inter: int, experts: int, seed: int) -> tuple[torch.Tensor, ...]:
    """`(w1, w1s, w2, w2s)` in e4m3 / fp32; FC1 scales put the FC1 tails past the clamp of 10."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    w1 = torch.empty(experts, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=DEV)
    w2 = torch.empty(experts, hidden, inter, dtype=torch.float8_e4m3fn, device=DEV)
    for lo in range(0, experts, 16):
        hi = min(lo + 16, experts)
        w1[lo:hi] = torch.randn(hi - lo, 2 * inter, hidden, device=DEV, generator=g)
        w2[lo:hi] = torch.randn(hi - lo, hidden, inter, device=DEV, generator=g)
    w1s = 0.04 + 0.04 * torch.rand(
        experts, 2 * inter // BLOCK, hidden // BLOCK, device=DEV, generator=g
    )
    w2s = 0.008 + 0.008 * torch.rand(
        experts, hidden // BLOCK, inter // BLOCK, device=DEV, generator=g
    )
    return w1, w1s, w2, w2s


@pytest.fixture(scope="module")
def stack() -> Iterator[Callable[..., tuple[torch.Tensor, ...]]]:
    """The expert stack per geometry, drawn once per module (3.6 GB at Hy4's)."""
    built: dict = {}

    def get(hidden: int, inter: int, experts: int) -> tuple[torch.Tensor, ...]:
        key = (hidden, inter, experts)
        if key not in built:
            built[key] = _draw_stack(hidden, inter, experts, seed=hidden + inter + experts)
        return built[key]

    try:
        yield get
    finally:
        built.clear()


def _routing(
    tokens: int, experts: int, top_k: int, scale: float, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-routed top-k: distinct ids per row and bf16 weights summing to `scale`."""
    logits = torch.randn(tokens, experts, generator=g, device=DEV)
    ids = torch.topk(logits, k=top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(logits.gather(1, ids.long()), dim=-1) * scale
    return ids.contiguous(), weights.to(torch.bfloat16).contiguous()


def _router_operands(
    tokens: int, experts: int, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 router logits and an fp32 correction bias around zero."""
    logits = torch.randn(tokens, experts, generator=g, device=DEV).to(torch.bfloat16)
    bias = torch.rand(experts, generator=g, device=DEV) - 0.5
    return logits, bias


def _args(spec: dict, stack: Callable, seed: int) -> dict:
    hidden, inter, experts = spec["hidden"], spec["inter"], spec["num_experts"]
    lo, local = spec["local_expert_offset"], spec["local_num_experts"]
    w1, w1s, w2, w2s = (t[lo : lo + local] for t in stack(hidden, inter, experts))
    g = torch.Generator(device=DEV).manual_seed(seed)
    tokens, top_k = spec["tokens"], spec["top_k"]
    hs = (2.0 * torch.randn(tokens, hidden, device=DEV, generator=g)).to(torch.float8_e4m3fn)
    # [H/128, T]: the layout fp8_quantize_1x128 returns here.
    hs_scale = 0.5 + torch.rand(hidden // BLOCK, tokens, device=DEV, generator=g)
    if spec.get("entry") == "in-kernel-routing":
        routing_logits, routing_bias = _router_operands(tokens, experts, g)
        ids = weights = None
    else:
        routing_logits = routing_bias = None
        ids, weights = _routing(tokens, experts, top_k, spec["routed_scaling_factor"], g)
    return dict(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hs,
        hidden_states_scale=hs_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        num_experts=experts,
        top_k=top_k,
        num_fused_shared_experts=0,
        n_group=spec["n_group"],
        topk_group=spec["topk_group"],
        intermediate_size=inter,
        local_expert_offset=lo,
        local_num_experts=local,
        routed_scaling_factor=spec["routed_scaling_factor"],
        routing_method_type=ROUTING_DEEPSEEK_V3,
        topk_weights=weights,
        topk_ids=ids,
        gemm1_clamp_limit=spec["gemm1_clamp_limit"],
    )


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying, stack):  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    args = _args(cell.spec, stack, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    assert out.shape == (cell.spec["tokens"], cell.spec["hidden"])
    assert out.dtype is torch.bfloat16
    _gate(out, op.reference(**args))
