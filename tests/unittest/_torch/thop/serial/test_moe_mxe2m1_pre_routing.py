# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-routing of the MXFP4 TRTLLM-Gen MoE on another stream.

trtllm::mxe2m1_moe_routing runs the routing kernel of
trtllm::mxe4m3_mxe2m1_block_scale_moe_runner on its own; the op then skips its
routing (run_moe_routed). The result must be bitwise the same as the op routing
itself, also when the routing stream is late.
"""

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops
from tensorrt_llm._torch.moe.fused_moe import RoutingMethodType
from tensorrt_llm._torch.utils import ActType_TrtllmGen

pytestmark = pytest.mark.skipif(
    not (100 <= getSMVersion() < 110),
    reason="The MXFP4 TRTLLM-Gen MoE is sm_100f only. Current SM is %d." % getSMVersion(),
)

HIDDEN = 512
INTERMEDIATE = 256
# Kimi K3 routing: 896 experts, top 16, one group, sigmoid + bias.
K3_ROUTING = dict(
    num_experts=896,
    top_k=16,
    n_group=1,
    topk_group=1,
    routed_scaling_factor=2.5,
    routing_method_type=int(RoutingMethodType.DeepSeekV3),
    logits_dtype=torch.float32,
    has_bias=True,
)
RENORMALIZE_ROUTING = dict(
    num_experts=128,
    top_k=8,
    n_group=None,
    topk_group=None,
    routed_scaling_factor=None,
    routing_method_type=int(RoutingMethodType.Renormalize),
    logits_dtype=torch.bfloat16,
    has_bias=False,
)


@pytest.fixture
def pre_routing(monkeypatch):
    monkeypatch.setattr(ops, "_MXE2M1_PRE_ROUTING_ENABLED", True)
    monkeypatch.setattr(ops, "_MXE2M1_PRE_ROUTING_MAX_TOKENS", 0)
    monkeypatch.setattr(ops, "_MXE2M1_PRE_ROUTING_REC", {})
    monkeypatch.setattr(ops, "_MXE2M1_PRE_ROUTING", {})
    monkeypatch.setattr(
        ops, "MXE2M1_PRE_ROUTING_STATS", {"routed": 0, "fallback": 0, "recorded": 0}
    )
    return ops.MXE2M1_PRE_ROUTING_STATS


def _make_moe(num_tokens: int, routing: dict):
    """Inputs of mxe4m3_mxe2m1_block_scale_moe_runner.

    Weights are random MXFP4 bytes with unit scales: the test compares two runs
    of the same kernels bit for bit, so the weight values need no meaning.
    """
    generator = torch.Generator(device="cuda").manual_seed(num_tokens)
    num_experts = routing["num_experts"]

    def rand_bytes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=generator)

    hidden = torch.randn(num_tokens, HIDDEN, device="cuda", generator=generator)
    hidden_fp8, hidden_scale = torch.ops.tensorrt_llm.quantize_mxe4m3_host(hidden.cpu(), False)
    unit_scale = 127  # e8m0 2**0
    inputs = dict(
        routing_logits=torch.randn(num_tokens, num_experts, device="cuda", generator=generator).to(
            routing["logits_dtype"]
        ),
        routing_bias=(
            torch.randn(num_experts, device="cuda", generator=generator).to(torch.bfloat16)
            if routing["has_bias"]
            else None
        ),
        hidden_states=hidden_fp8.cuda().view(torch.float8_e4m3fn),
        hidden_states_scale=hidden_scale.cuda().view(torch.uint8),
        gemm1_weights=rand_bytes(num_experts, 2 * INTERMEDIATE, HIDDEN // 2),
        gemm1_weights_scale=torch.full(
            (num_experts, 2 * INTERMEDIATE, HIDDEN // 32),
            unit_scale,
            dtype=torch.uint8,
            device="cuda",
        ),
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=rand_bytes(num_experts, HIDDEN, INTERMEDIATE // 2),
        gemm2_weights_scale=torch.full(
            (num_experts, HIDDEN, INTERMEDIATE // 32),
            unit_scale,
            dtype=torch.uint8,
            device="cuda",
        ),
        gemm2_bias=None,
        num_experts=num_experts,
        top_k=routing["top_k"],
        n_group=routing["n_group"],
        topk_group=routing["topk_group"],
        intermediate_size=INTERMEDIATE,
        valid_hidden_size=None,
        valid_intermediate_size=None,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routing["routed_scaling_factor"],
        routing_method_type=routing["routing_method_type"],
        act_type=ActType_TrtllmGen.SwiGlu.value,
    )
    return inputs


def _run(inputs, **overrides) -> torch.Tensor:
    return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(**dict(inputs, **overrides))


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _pre_route_on_side_stream(inputs, routing_logits, delay_cycles: int) -> None:
    prepared = ops.mxe2m1_pre_routing_prepare(
        inputs["gemm1_weights"], inputs["hidden_states"].shape[0]
    )
    assert prepared is not None
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        if delay_cycles:
            torch.cuda._sleep(delay_cycles)
        ops.mxe2m1_pre_routing_launch(prepared, inputs["gemm1_weights"], routing_logits)


@pytest.mark.parametrize("routing", [K3_ROUTING, RENORMALIZE_ROUTING], ids=["k3", "renormalize"])
@pytest.mark.parametrize("num_tokens", [1, 8, 16])
@pytest.mark.parametrize("delay_cycles", [0, 50_000_000], ids=["no_delay", "late_router"])
@torch.no_grad()
def test_pre_routed_moe_matches_own_routing(pre_routing, routing, num_tokens, delay_cycles):
    inputs = _make_moe(num_tokens, routing)
    expected = _run(inputs)
    assert pre_routing["recorded"] == 1

    _pre_route_on_side_stream(inputs, inputs["routing_logits"], delay_cycles)
    actual = _run(inputs)
    torch.cuda.synchronize()

    assert pre_routing["routed"] == 1 and pre_routing["fallback"] == 0
    assert torch.equal(_bits(actual), _bits(expected))


@torch.no_grad()
def test_pre_routing_needs_a_recorded_shape(pre_routing):
    inputs = _make_moe(8, K3_ROUTING)
    assert ops.mxe2m1_pre_routing_prepare(inputs["gemm1_weights"], 8) is None
    _run(inputs)
    assert ops.mxe2m1_pre_routing_prepare(inputs["gemm1_weights"], 8) is not None
    assert ops.mxe2m1_pre_routing_prepare(inputs["gemm1_weights"], 4) is None


@torch.no_grad()
def test_pre_routing_falls_back_when_routing_inputs_change(pre_routing):
    inputs = _make_moe(8, K3_ROUTING)
    _run(inputs)
    _pre_route_on_side_stream(inputs, inputs["routing_logits"], 0)
    other_bias = inputs["routing_bias"].clone()
    expected = _run(inputs, routing_bias=other_bias)
    torch.cuda.synchronize()
    assert pre_routing["routed"] == 0 and pre_routing["fallback"] == 1

    assert torch.equal(_bits(expected), _bits(_run(inputs, routing_bias=other_bias)))


@torch.no_grad()
def test_pre_routing_rejects_other_logits(pre_routing):
    inputs = _make_moe(8, K3_ROUTING)
    _run(inputs)
    _pre_route_on_side_stream(inputs, inputs["routing_logits"], 0)
    with pytest.raises(RuntimeError, match="other routing logits"):
        _run(inputs, routing_logits=inputs["routing_logits"].clone())


@torch.no_grad()
def test_routing_workspace_validation():
    logits = torch.zeros(8, 896, device="cuda")
    with pytest.raises(RuntimeError, match="tile_tokens_dim"):
        torch.ops.trtllm.mxe2m1_moe_routing_workspace(8, 896, 16, 24, logits)
    workspace = torch.ops.trtllm.mxe2m1_moe_routing_workspace(8, 896, 16, 8, logits)
    with pytest.raises(RuntimeError, match="routing workspace"):
        torch.ops.trtllm.mxe2m1_moe_routing(
            logits, None, workspace[:-1], 896, 16, 1, 1, 0, 896, 2.5, 8, 2
        )
