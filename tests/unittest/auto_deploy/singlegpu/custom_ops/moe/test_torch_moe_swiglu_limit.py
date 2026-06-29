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

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import ActivationType


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    x = torch.tensor(
        [
            [2.0, -1.0],
            [-3.0, 0.5],
            [1.5, 2.0],
        ],
        dtype=torch.float32,
    )
    selected_experts = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=torch.int64,
    )
    routing_weights = torch.tensor(
        [
            [0.75, 0.25],
            [0.5, 0.5],
            [0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    w1_weight = [
        torch.tensor([[2.0, -1.0], [0.5, 1.0]], dtype=torch.float32),
        torch.tensor([[-1.0, 1.5], [2.0, 0.25]], dtype=torch.float32),
    ]
    w2_weight = [
        torch.tensor([[1.0, -0.5], [0.25, 1.5]], dtype=torch.float32),
        torch.tensor([[-0.75, 1.25], [1.0, 0.5]], dtype=torch.float32),
    ]
    w3_weight = [
        torch.tensor([[-2.0, 0.5], [1.5, -1.0]], dtype=torch.float32),
        torch.tensor([[1.0, -2.0], [-1.5, 0.75]], dtype=torch.float32),
    ]
    return x, selected_experts, routing_weights, w1_weight, w2_weight, w3_weight


def _reference_torch_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: list[torch.Tensor],
    w2_weight: list[torch.Tensor],
    w3_weight: list[torch.Tensor],
    swiglu_limit: float,
) -> torch.Tensor:
    x_flat = x.view(-1, x.shape[-1])
    output = torch.zeros_like(x_flat)

    for token_idx, token in enumerate(x_flat):
        token_input = token.unsqueeze(0)
        for route_idx, expert_idx_tensor in enumerate(selected_experts[token_idx]):
            expert_idx = int(expert_idx_tensor.item())
            if expert_idx < 0 or expert_idx >= len(w1_weight):
                continue

            gate_out = F.linear(token_input.to(w1_weight[expert_idx].dtype), w1_weight[expert_idx])
            up_out = F.linear(token_input.to(w3_weight[expert_idx].dtype), w3_weight[expert_idx])
            if swiglu_limit > 0.0:
                gate_out = gate_out.clamp(max=swiglu_limit)
                up_out = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)

            expert_out = F.linear(F.silu(gate_out) * up_out, w2_weight[expert_idx])
            output[token_idx] += (
                expert_out.squeeze(0).to(output.dtype) * routing_weights[token_idx, route_idx]
            )

    return output.view_as(x)


def _torch_moe(swiglu_limit: float | None = None) -> torch.Tensor:
    x, selected_experts, routing_weights, w1_weight, w2_weight, w3_weight = _inputs()
    kwargs = {
        "is_gated_mlp": True,
        "act_fn": int(ActivationType.Silu),
    }
    if swiglu_limit is not None:
        kwargs["swiglu_limit"] = swiglu_limit

    return torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight,
        w2_weight,
        w3_weight,
        **kwargs,
    )


def test_torch_moe_swiglu_non_positive_limit_preserves_default() -> None:
    x, selected_experts, routing_weights, w1_weight, w2_weight, w3_weight = _inputs()
    default_output = _torch_moe()

    for swiglu_limit in (0.0, -1.0):
        actual = _torch_moe(swiglu_limit)
        expected = _reference_torch_moe(
            x,
            selected_experts,
            routing_weights,
            w1_weight,
            w2_weight,
            w3_weight,
            swiglu_limit,
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, default_output)


def test_torch_moe_swiglu_positive_limit_matches_reference_and_changes_output() -> None:
    x, selected_experts, routing_weights, w1_weight, w2_weight, w3_weight = _inputs()
    unlimited_output = _torch_moe(0.0)
    limited_output = _torch_moe(1.0)

    expected_limited = _reference_torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight,
        w2_weight,
        w3_weight,
        1.0,
    )

    torch.testing.assert_close(limited_output, expected_limited)
    assert not torch.allclose(limited_output, unlimited_output)
