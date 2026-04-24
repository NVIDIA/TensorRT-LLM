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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_moe import (
    deepseek_v4_limited_swiglu,
    deepseek_v4_moe_reference,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_moe import (
    DeepSeekV4MoELowering,
    DeepSeekV4MoELoweringError,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def _expert_mlp(
    hidden_states: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    swiglu_limit: float,
    *,
    apply_swiglu_limit: bool,
) -> torch.Tensor:
    gate = F.linear(hidden_states.to(w1_weight.dtype), w1_weight).float()
    up = F.linear(hidden_states.to(w3_weight.dtype), w3_weight).float()
    if apply_swiglu_limit:
        hidden = F.silu(torch.clamp(gate, max=swiglu_limit)) * torch.clamp(
            up, min=-swiglu_limit, max=swiglu_limit
        )
    else:
        hidden = F.silu(gate) * up
    return F.linear(hidden.to(w2_weight.dtype), w2_weight).to(hidden_states.dtype)


def _manual_from_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: torch.Tensor | None,
    shared_w2_weight: torch.Tensor | None,
    shared_w3_weight: torch.Tensor | None,
    swiglu_limit: float,
) -> torch.Tensor:
    hidden_shape = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_shape[-1])
    output = torch.zeros_like(hidden_flat)

    for expert_idx in range(routed_w1_weight.shape[0]):
        token_idx, route_idx = torch.where(selected_experts == expert_idx)
        if token_idx.numel() == 0:
            continue
        expert_output = _expert_mlp(
            hidden_flat[token_idx],
            routed_w1_weight[expert_idx],
            routed_w2_weight[expert_idx],
            routed_w3_weight[expert_idx],
            swiglu_limit,
            apply_swiglu_limit=True,
        )
        output[token_idx] += (
            expert_output * routing_weights[token_idx, route_idx, None].to(expert_output.dtype)
        ).to(output.dtype)

    if shared_w1_weight is not None:
        assert shared_w2_weight is not None
        assert shared_w3_weight is not None
        output += _expert_mlp(
            hidden_flat,
            shared_w1_weight,
            shared_w2_weight,
            shared_w3_weight,
            swiglu_limit,
            apply_swiglu_limit=False,
        )

    return output.view(hidden_shape)


def _stacked_weights(
    num_experts: int = 4,
    hidden_size: int = 3,
    intermediate_size: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w1 = (
        torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32)
        .reshape(num_experts, intermediate_size, hidden_size)
        .div(37.0)
        .sub(0.5)
        .to(dtype)
    )
    w2 = (
        torch.arange(num_experts * hidden_size * intermediate_size, dtype=torch.float32)
        .reshape(num_experts, hidden_size, intermediate_size)
        .div(29.0)
        .sub(0.25)
        .to(dtype)
    )
    w3 = (
        torch.arange(num_experts * intermediate_size * hidden_size, dtype=torch.float32)
        .reshape(num_experts, intermediate_size, hidden_size)
        .div(31.0)
        .sub(0.75)
        .to(dtype)
    )
    return w1, w2, w3


def test_from_routing_matches_tiny_bf16_reference_with_precomputed_routing() -> None:
    hidden_states = torch.tensor(
        [
            [0.5, -1.0, 1.5],
            [1.25, 0.25, -0.75],
            [-0.5, 0.75, 0.125],
        ],
        dtype=torch.bfloat16,
    )
    selected_experts = torch.tensor([[0, 2], [3, 1], [2, 0]], dtype=torch.int32)
    routing_weights = torch.tensor([[0.75, 0.25], [0.4, 0.6], [0.125, 0.875]], dtype=torch.float32)
    routed_w1, routed_w2, routed_w3 = _stacked_weights()
    shared_w1 = torch.linspace(-0.25, 0.5, 15, dtype=torch.float32).view(5, 3).to(torch.bfloat16)
    shared_w2 = torch.linspace(0.1, 0.8, 15, dtype=torch.float32).view(3, 5).to(torch.bfloat16)
    shared_w3 = torch.linspace(-0.6, 0.3, 15, dtype=torch.float32).view(5, 3).to(torch.bfloat16)

    expected = _manual_from_routing(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        swiglu_limit=10.0,
    )
    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        shared_w1,
        shared_w2,
        shared_w3,
        10.0,
        "moe",
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_limited_swiglu_limit_zero_zeroes_up_branch() -> None:
    gate = torch.tensor([[20.0, -20.0]])
    up = torch.tensor([[15.0, -15.0]])

    actual = deepseek_v4_limited_swiglu(gate, up, swiglu_limit=0.0)

    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_limited_swiglu_limit_ten_clamps_gate_and_up() -> None:
    gate = torch.tensor([[20.0, -20.0]])
    up = torch.tensor([[15.0, -15.0]])

    actual = deepseek_v4_limited_swiglu(gate, up, swiglu_limit=10.0)
    expected = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor([[10.0, -10.0]])

    torch.testing.assert_close(actual, expected)


def test_from_routing_limit_zero_zeroes_routed_output_without_shared_expert() -> None:
    hidden_states = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    selected_experts = torch.tensor([[0], [0]], dtype=torch.int64)
    routing_weights = torch.ones((2, 1), dtype=torch.float32)
    routed_w1 = torch.tensor([[[20.0], [-20.0]]])
    routed_w2 = torch.tensor([[[1.0, 2.0]]])
    routed_w3 = torch.tensor([[[15.0], [-15.0]]])

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        0.0,
        "moe",
    )

    torch.testing.assert_close(actual, torch.zeros_like(hidden_states))


def test_from_routing_limit_ten_clamps_gate_and_up_inside_expert() -> None:
    hidden_states = torch.tensor([[1.0]], dtype=torch.float32)
    selected_experts = torch.tensor([[0]], dtype=torch.int64)
    routing_weights = torch.ones((1, 1), dtype=torch.float32)
    routed_w1 = torch.tensor([[[20.0], [-20.0]]])
    routed_w2 = torch.tensor([[[1.0, 2.0]]])
    routed_w3 = torch.tensor([[[15.0], [-15.0]]])

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        10.0,
        "moe",
    )
    expected_hidden = F.silu(torch.tensor([[10.0, -20.0]])) * torch.tensor([[10.0, -10.0]])
    expected = expected_hidden @ torch.tensor([[1.0], [2.0]])

    torch.testing.assert_close(actual, expected)


def test_canonical_moe_uses_router_surface() -> None:
    hidden_states = torch.tensor([[0.25, -0.5, 1.0], [1.0, 0.5, -0.25]], dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [0.5, -0.25, 0.75],
            [-0.25, 0.5, 0.125],
            [0.75, 0.125, -0.5],
            [0.25, 0.25, 0.25],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([0.0, 0.25, -0.25, 0.125], dtype=torch.float32)
    routed_w1, routed_w2, routed_w3 = _stacked_weights(
        num_experts=4, hidden_size=3, intermediate_size=2, dtype=torch.float32
    )

    expected = deepseek_v4_moe_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        routed_w1_weight=routed_w1,
        routed_w2_weight=routed_w2,
        routed_w3_weight=routed_w3,
        shared_w1_weight=None,
        shared_w2_weight=None,
        shared_w3_weight=None,
        top_k=2,
        route_scale=1.5,
        swiglu_limit=10.0,
        is_hash_layer=False,
    )
    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe.default(
        hidden_states,
        None,
        router_weight,
        router_bias,
        None,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        2,
        1.5,
        10.0,
        False,
        "moe",
    )

    torch.testing.assert_close(actual, expected)


def test_from_routing_fake_returns_meta_output_shape() -> None:
    hidden_states = torch.empty((2, 3, 4), dtype=torch.bfloat16, device="meta")
    selected_experts = torch.empty((6, 2), dtype=torch.int64, device="meta")
    routing_weights = torch.empty((6, 2), dtype=torch.float32, device="meta")
    routed_w1 = torch.empty((4, 5, 4), dtype=torch.bfloat16, device="meta")
    routed_w2 = torch.empty((4, 4, 5), dtype=torch.bfloat16, device="meta")
    routed_w3 = torch.empty((4, 5, 4), dtype=torch.bfloat16, device="meta")

    actual = torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1,
        routed_w2,
        routed_w3,
        None,
        None,
        None,
        10.0,
        "moe",
    )

    assert actual.shape == hidden_states.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == "meta"


class _FromRoutingExportModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        routed_w1, routed_w2, routed_w3 = _stacked_weights(
            num_experts=2, hidden_size=3, intermediate_size=4, dtype=torch.float32
        )
        self.routed_w1 = nn.Parameter(routed_w1, requires_grad=False)
        self.routed_w2 = nn.Parameter(routed_w2, requires_grad=False)
        self.routed_w3 = nn.Parameter(routed_w3, requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default(
            hidden_states,
            selected_experts,
            routing_weights,
            self.routed_w1,
            self.routed_w2,
            self.routed_w3,
            None,
            None,
            None,
            10.0,
            "moe",
        )


def test_from_routing_custom_op_exports() -> None:
    module = _FromRoutingExportModule()
    hidden_states = torch.randn(3, 3, dtype=torch.float32)
    selected_experts = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64)
    routing_weights = torch.full((3, 2), 0.5, dtype=torch.float32)

    exported = torch.export.export(module, (hidden_states, selected_experts, routing_weights))
    targets = [node.target for node in exported.graph_module.graph.nodes]

    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default in targets


class _CanonicalMoEModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router_weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)
        self.router_bias = nn.Parameter(torch.zeros(3), requires_grad=False)
        routed_w1, routed_w2, routed_w3 = _stacked_weights(
            num_experts=3, hidden_size=2, intermediate_size=4, dtype=torch.float32
        )
        self.routed_w1 = nn.Parameter(routed_w1, requires_grad=False)
        self.routed_w2 = nn.Parameter(routed_w2, requires_grad=False)
        self.routed_w3 = nn.Parameter(routed_w3, requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_moe.default(
            hidden_states,
            None,
            self.router_weight,
            self.router_bias,
            None,
            self.routed_w1,
            self.routed_w2,
            self.routed_w3,
            None,
            None,
            None,
            2,
            1.0,
            10.0,
            False,
            "moe",
        )


def _trace_canonical_moe() -> torch.fx.GraphModule:
    return torch.fx.symbolic_trace(_CanonicalMoEModule())


def test_lowering_skeleton_raises_clear_error_for_production_path() -> None:
    gm = _trace_canonical_moe()
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.POST_LOAD_FUSION)

    with pytest.raises(
        DeepSeekV4MoELoweringError,
        match="triton_mxfp4_moe.*FineGrained FP8.*swiglu_limit",
    ):
        transform._apply(gm, None, None, SharedConfig())


def test_lowering_skeleton_can_opt_into_reference_graph_for_tests() -> None:
    module = _CanonicalMoEModule()
    gm = torch.fx.symbolic_trace(module)
    transform = DeepSeekV4MoELowering.from_kwargs(
        stage=Stages.POST_LOAD_FUSION, allow_reference_lowering=True
    )

    lowered, info = transform._apply(gm, None, None, SharedConfig())
    targets = [node.target for node in lowered.graph.nodes]

    assert info.num_matches == 1
    assert torch.ops.auto_deploy.torch_deepseek_v4_router.default in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_moe_from_routing.default in targets
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe) for node in lowered.graph.nodes
    )

    hidden_states = torch.randn(4, 2, dtype=torch.float32)
    torch.testing.assert_close(lowered(hidden_states), module(hidden_states))
