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
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_router import (
    deepseek_v4_router_reference,
)


def _expected_weights(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    selected_experts: torch.Tensor,
    route_scale: float,
) -> torch.Tensor:
    logits = F.linear(
        hidden_states.reshape(-1, hidden_states.shape[-1]).float(), router_weight.float()
    )
    scores = torch.sqrt(F.softplus(logits))
    weights = scores.gather(1, selected_experts.long())
    return weights / weights.sum(dim=-1, keepdim=True) * route_scale


def test_hash_router_uses_tid2eid_and_normalizes_scaled_weights() -> None:
    hidden_states = torch.tensor(
        [
            [0.5, -1.0, 2.0],
            [1.5, 0.25, -0.75],
            [-1.0, 0.5, 0.25],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([4, 1, 3], dtype=torch.long)
    router_weight = torch.tensor(
        [
            [0.25, -0.5, 0.75],
            [-0.5, 0.5, 0.25],
            [0.75, 0.25, -0.5],
            [0.125, -0.25, 0.5],
        ],
        dtype=torch.float32,
    )
    tid2eid = torch.tensor(
        [
            [0, 1],
            [3, 0],
            [2, 3],
            [1, 2],
            [2, 0],
        ],
        dtype=torch.int32,
    )
    route_scale = 1.5

    selected_experts, routing_weights = deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias=None,
        tid2eid=tid2eid,
        top_k=2,
        route_scale=route_scale,
        is_hash_layer=True,
    )

    expected_experts = tid2eid[input_ids]
    expected_weights = _expected_weights(
        hidden_states, router_weight, expected_experts, route_scale
    )

    torch.testing.assert_close(selected_experts, expected_experts)
    torch.testing.assert_close(routing_weights, expected_weights)
    torch.testing.assert_close(
        routing_weights.sum(dim=-1),
        torch.full((hidden_states.shape[0],), route_scale),
    )


def test_topk_router_bias_affects_selection_only() -> None:
    hidden_states = torch.eye(3, dtype=torch.float32)
    router_weight = torch.tensor(
        [
            [1.2, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 0.5],
            [0.2, 0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([-4.0, 0.0, 4.0, 0.0], dtype=torch.float32)
    route_scale = 1.25

    selected_experts, routing_weights = deepseek_v4_router_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        top_k=2,
        route_scale=route_scale,
        is_hash_layer=False,
    )

    scores = torch.sqrt(F.softplus(F.linear(hidden_states, router_weight)))
    expected_experts = torch.topk(scores + router_bias, 2, dim=-1).indices
    expected_weights = scores.gather(1, expected_experts)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * route_scale
    biased_weights = (scores + router_bias).gather(1, expected_experts)
    biased_weights = biased_weights / biased_weights.sum(dim=-1, keepdim=True) * route_scale

    torch.testing.assert_close(selected_experts, expected_experts)
    torch.testing.assert_close(routing_weights, expected_weights)
    assert not torch.allclose(routing_weights, biased_weights)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_is_finite_for_large_logits(dtype: torch.dtype) -> None:
    hidden_states = torch.tensor(
        [
            [1.0e20, 1.0e20],
            [-1.0e20, -1.0e20],
        ],
        dtype=dtype,
    )
    router_weight = torch.tensor(
        [
            [1.0e20, 1.0e20],
            [1.0e20, 1.0e20],
            [1.0e20, 1.0e20],
        ],
        dtype=dtype,
    )

    selected_experts, routing_weights = deepseek_v4_router_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=None,
        tid2eid=None,
        top_k=2,
        route_scale=1.0,
        is_hash_layer=False,
    )

    assert selected_experts.shape == (2, 2)
    assert routing_weights.dtype == torch.float32
    assert torch.isfinite(routing_weights).all()


def test_router_flattens_batched_hidden_states_and_input_ids() -> None:
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float32)
    input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    router_weight = torch.randn(5, 4, dtype=torch.float32)
    tid2eid = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 0],
            [4, 0, 1],
            [0, 2, 4],
        ],
        dtype=torch.int64,
    )

    selected_experts, routing_weights = deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias=None,
        tid2eid=tid2eid,
        top_k=3,
        route_scale=0.75,
        is_hash_layer=True,
    )

    assert selected_experts.shape == (6, 3)
    assert routing_weights.shape == (6, 3)
    torch.testing.assert_close(selected_experts, tid2eid[input_ids.reshape(-1)])


def test_custom_op_matches_reference_and_has_meta_fake() -> None:
    hidden_states = torch.randn(4, 3, dtype=torch.float32)
    router_weight = torch.randn(6, 3, dtype=torch.float32)
    router_bias = torch.randn(6, dtype=torch.float32)

    expected = deepseek_v4_router_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        top_k=2,
        route_scale=1.0,
        is_hash_layer=False,
    )
    actual = torch.ops.auto_deploy.torch_deepseek_v4_router.default(
        hidden_states,
        None,
        router_weight,
        router_bias,
        None,
        2,
        1.0,
        False,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    meta_hidden = torch.empty((2, 3, 4), device="meta", dtype=torch.bfloat16)
    meta_weight = torch.empty((5, 4), device="meta", dtype=torch.bfloat16)
    meta_selected, meta_weights = torch.ops.auto_deploy.torch_deepseek_v4_router.default(
        meta_hidden,
        None,
        meta_weight,
        None,
        None,
        3,
        1.0,
        False,
    )

    assert meta_selected.shape == (6, 3)
    assert meta_selected.dtype == torch.int64
    assert meta_weights.shape == (6, 3)
    assert meta_weights.dtype == torch.float32
    assert meta_weights.device.type == "meta"
