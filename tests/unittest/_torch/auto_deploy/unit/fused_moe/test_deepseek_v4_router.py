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

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_router
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_router import (
    DEEPSEEK_V4_HASH_ROUTED_LAYER_INDICES,
    DEEPSEEK_V4_NUM_ROUTED_EXPERTS,
    DEEPSEEK_V4_OBSERVED_EXPERTS_PER_RANK,
    DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE,
    DEEPSEEK_V4_ROUTE_SCALE,
    DEEPSEEK_V4_TOP_K,
    _normalize_and_scale,
    deepseek_v4_ep_expert_range,
    deepseek_v4_experts_per_rank,
    deepseek_v4_localize_expert_ids,
    deepseek_v4_router_reference,
    is_deepseek_v4_hash_routed_layer,
)

CUDA_AVAILABLE = torch.cuda.is_available()


def _fail_router_reference_fallback(*args, **kwargs) -> None:
    del args, kwargs
    raise AssertionError("supported Triton router path must not call the torch reference router")


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


def test_observed_router_contract_constants_and_layer_split() -> None:
    assert DEEPSEEK_V4_NUM_ROUTED_EXPERTS == 256
    assert DEEPSEEK_V4_TOP_K == 6
    assert DEEPSEEK_V4_ROUTE_SCALE == 1.5
    assert DEEPSEEK_V4_HASH_ROUTED_LAYER_INDICES == (0, 1, 2)
    assert DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE == 8
    assert DEEPSEEK_V4_OBSERVED_EXPERTS_PER_RANK == 32

    assert [is_deepseek_v4_hash_routed_layer(layer_idx) for layer_idx in range(5)] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert deepseek_v4_experts_per_rank() == DEEPSEEK_V4_OBSERVED_EXPERTS_PER_RANK
    assert deepseek_v4_ep_expert_range(0) == (0, 32)
    assert deepseek_v4_ep_expert_range(7) == (224, 256)


def test_ep_localize_global_router_ids_preserves_shape_and_masks_rank_routes() -> None:
    selected_experts = torch.tensor([[0, 31, 32, 63, 64, 255]], dtype=torch.int64)
    routing_weights = torch.arange(1, 7, dtype=torch.float32).view(1, 6)

    metadata = deepseek_v4_localize_expert_ids(
        selected_experts,
        routing_weights,
        moe_ep_size=DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE,
        moe_ep_rank=1,
        num_global_experts=DEEPSEEK_V4_NUM_ROUTED_EXPERTS,
    )

    expected_mask = torch.tensor([[False, False, True, True, False, False]])
    expected_local = torch.tensor([[0, 0, 0, 31, 0, 0]], dtype=torch.int64)
    expected_weights = torch.tensor([[0.0, 0.0, 3.0, 4.0, 0.0, 0.0]])

    assert metadata.local_expert_start == 32
    assert metadata.local_expert_end == 64
    assert metadata.experts_per_rank == 32
    assert metadata.selected_experts_global is selected_experts
    assert metadata.routing_weights_global is routing_weights
    assert torch.equal(metadata.rank_mask, expected_mask)
    torch.testing.assert_close(metadata.selected_experts_local, expected_local)
    torch.testing.assert_close(metadata.routing_weights_local, expected_weights)


def test_router_weight_normalization_preserves_zero_and_inf_behavior() -> None:
    weights = torch.tensor(
        [
            [0.0, 0.0],
            [float("inf"), 2.0],
            [float("inf"), float("inf")],
        ],
        dtype=torch.float32,
    )

    actual = _normalize_and_scale(weights, route_scale=1.5)

    expected = torch.tensor([[0.0, 0.0], [1.5, 0.0], [0.75, 0.75]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


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


def test_triton_router_cpu_fallback_matches_reference_for_hash_and_topk() -> None:
    hidden_states = torch.tensor(
        [
            [0.25, -0.5, 1.0],
            [1.25, 0.5, -0.25],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([2, 0], dtype=torch.long)
    router_weight = torch.tensor(
        [
            [0.5, -0.25, 0.75],
            [-0.75, 0.5, 0.125],
            [0.25, 0.75, -0.5],
            [0.625, -0.125, 0.25],
        ],
        dtype=torch.float32,
    )
    router_bias = torch.tensor([0.0, 0.5, -0.25, 0.125], dtype=torch.float32)
    tid2eid = torch.tensor(
        [
            [3, 1],
            [2, 0],
            [1, 2],
        ],
        dtype=torch.int32,
    )

    expected_hash = deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias=None,
        tid2eid=tid2eid,
        top_k=2,
        route_scale=DEEPSEEK_V4_ROUTE_SCALE,
        is_hash_layer=True,
    )
    actual_hash = torch.ops.auto_deploy.triton_deepseek_v4_router.default(
        hidden_states,
        input_ids,
        router_weight,
        None,
        tid2eid,
        2,
        DEEPSEEK_V4_ROUTE_SCALE,
        True,
    )

    expected_topk = deepseek_v4_router_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        top_k=2,
        route_scale=1.25,
        is_hash_layer=False,
    )
    actual_topk = torch.ops.auto_deploy.triton_deepseek_v4_router.default(
        hidden_states,
        None,
        router_weight,
        router_bias,
        None,
        2,
        1.25,
        False,
    )

    torch.testing.assert_close(actual_hash[0], expected_hash[0])
    torch.testing.assert_close(actual_hash[1], expected_hash[1])
    torch.testing.assert_close(actual_topk[0], expected_topk[0])
    torch.testing.assert_close(actual_topk[1], expected_topk[1])


def test_triton_router_fake_shapes_for_hash_topk_and_source_wrapper() -> None:
    meta_hidden = torch.empty((2, 3, 4096), device="meta", dtype=torch.bfloat16)
    meta_input_ids = torch.empty((2, 3), device="meta", dtype=torch.int64)
    meta_weight = torch.empty(
        (DEEPSEEK_V4_NUM_ROUTED_EXPERTS, 4096), device="meta", dtype=torch.bfloat16
    )
    meta_bias = torch.empty((DEEPSEEK_V4_NUM_ROUTED_EXPERTS,), device="meta", dtype=torch.float32)
    meta_tid2eid = torch.empty((129280, DEEPSEEK_V4_TOP_K), device="meta", dtype=torch.int64)

    hash_selected, hash_weights = torch.ops.auto_deploy.triton_deepseek_v4_router_hash.default(
        meta_hidden,
        meta_input_ids,
        meta_weight,
        meta_tid2eid,
        DEEPSEEK_V4_TOP_K,
        DEEPSEEK_V4_ROUTE_SCALE,
    )
    topk_selected, topk_weights = torch.ops.auto_deploy.triton_deepseek_v4_router_topk.default(
        meta_hidden,
        meta_weight,
        meta_bias,
        DEEPSEEK_V4_TOP_K,
        DEEPSEEK_V4_ROUTE_SCALE,
    )
    wrapper_selected, wrapper_weights = torch.ops.auto_deploy.triton_deepseek_v4_router.default(
        meta_hidden,
        meta_input_ids,
        meta_weight,
        None,
        meta_tid2eid,
        DEEPSEEK_V4_TOP_K,
        DEEPSEEK_V4_ROUTE_SCALE,
        True,
    )

    assert hash_selected.shape == (6, DEEPSEEK_V4_TOP_K)
    assert hash_selected.dtype == torch.int64
    assert hash_weights.shape == (6, DEEPSEEK_V4_TOP_K)
    assert hash_weights.dtype == torch.float32
    assert topk_selected.shape == (6, DEEPSEEK_V4_TOP_K)
    assert topk_selected.dtype == torch.int64
    assert topk_weights.shape == (6, DEEPSEEK_V4_TOP_K)
    assert topk_weights.dtype == torch.float32
    assert wrapper_selected.shape == (6, DEEPSEEK_V4_TOP_K)
    assert wrapper_selected.dtype == torch.int64
    assert wrapper_weights.shape == (6, DEEPSEEK_V4_TOP_K)
    assert wrapper_weights.dtype == torch.float32


class _TritonHashRouterExportModule(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        router_weight: torch.Tensor,
        tid2eid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.auto_deploy.triton_deepseek_v4_router.default(
            hidden_states,
            input_ids,
            router_weight,
            None,
            tid2eid,
            2,
            DEEPSEEK_V4_ROUTE_SCALE,
            True,
        )


class _TritonTopKRouterExportModule(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        router_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.auto_deploy.triton_deepseek_v4_router.default(
            hidden_states,
            None,
            router_weight,
            router_bias,
            None,
            2,
            DEEPSEEK_V4_ROUTE_SCALE,
            False,
        )


def test_triton_router_source_wrapper_exports_triton_op_name() -> None:
    hash_graph = torch.fx.symbolic_trace(_TritonHashRouterExportModule())
    topk_graph = torch.fx.symbolic_trace(_TritonTopKRouterExportModule())

    for gm in (hash_graph, topk_graph):
        targets = [node.target for node in gm.graph.nodes]
        assert torch.ops.auto_deploy.triton_deepseek_v4_router.default in targets
        assert torch.ops.auto_deploy.torch_deepseek_v4_router.default not in targets


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
def test_triton_hash_router_cuda_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    hidden_states = torch.tensor(
        [
            [0.5, -1.0, 2.0, 0.25],
            [1.5, 0.25, -0.75, 0.5],
            [-1.0, 0.5, 0.25, -0.5],
            [0.125, -0.625, 0.375, 1.25],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    input_ids = torch.tensor([4, 1, 3, 0], device=device, dtype=torch.int64)
    router_weight = torch.tensor(
        [
            [0.25, -0.5, 0.75, 0.125],
            [-0.5, 0.5, 0.25, -0.25],
            [0.75, 0.25, -0.5, 0.5],
            [0.125, -0.25, 0.5, 0.625],
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    tid2eid = torch.tensor(
        [
            [0, 2, 3],
            [3, 0, 1],
            [2, 3, 0],
            [1, 2, 3],
            [2, 0, 1],
        ],
        device=device,
        dtype=torch.int64,
    )

    expected = deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias=None,
        tid2eid=tid2eid,
        top_k=3,
        route_scale=DEEPSEEK_V4_ROUTE_SCALE,
        is_hash_layer=True,
    )
    assert deepseek_v4_router._deepseek_v4_router_hash_kernel is not None
    assert deepseek_v4_router._can_use_triton_router_base(hidden_states, router_weight, 3)
    monkeypatch.setattr(
        deepseek_v4_router,
        "deepseek_v4_router_reference",
        _fail_router_reference_fallback,
    )
    actual = torch.ops.auto_deploy.triton_deepseek_v4_router_hash.default(
        hidden_states,
        input_ids,
        router_weight,
        tid2eid,
        3,
        DEEPSEEK_V4_ROUTE_SCALE,
    )

    assert actual[0].device.type == "cuda"
    assert actual[1].device.type == "cuda"
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        actual[1].sum(dim=-1),
        torch.full((hidden_states.shape[0],), DEEPSEEK_V4_ROUTE_SCALE, device=device),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
def test_triton_topk_router_cuda_matches_reference_and_bias_selection_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    hidden_states = torch.randn(7, 16, device=device, dtype=torch.bfloat16)
    router_weight = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
    router_bias = torch.linspace(-0.5, 0.75, 16, device=device, dtype=torch.float32)

    expected = deepseek_v4_router_reference(
        hidden_states,
        input_ids=None,
        router_weight=router_weight,
        router_bias=router_bias,
        tid2eid=None,
        top_k=DEEPSEEK_V4_TOP_K,
        route_scale=DEEPSEEK_V4_ROUTE_SCALE,
        is_hash_layer=False,
    )
    assert deepseek_v4_router._deepseek_v4_router_topk_kernel is not None
    assert deepseek_v4_router._can_use_triton_router_base(
        hidden_states,
        router_weight,
        DEEPSEEK_V4_TOP_K,
    )
    monkeypatch.setattr(
        deepseek_v4_router,
        "deepseek_v4_router_reference",
        _fail_router_reference_fallback,
    )
    actual = torch.ops.auto_deploy.triton_deepseek_v4_router_topk.default(
        hidden_states,
        router_weight,
        router_bias,
        DEEPSEEK_V4_TOP_K,
        DEEPSEEK_V4_ROUTE_SCALE,
    )

    assert actual[0].device.type == "cuda"
    assert actual[1].device.type == "cuda"
    scores = torch.sqrt(F.softplus(F.linear(hidden_states.float(), router_weight.float())))
    biased_weights = (scores + router_bias).gather(1, actual[0])
    biased_weights = _normalize_and_scale(biased_weights, DEEPSEEK_V4_ROUTE_SCALE)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], atol=1.0e-5, rtol=1.0e-5)
    assert not torch.allclose(actual[1], biased_weights)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
def test_triton_source_router_cuda_graph_replay_fixed_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5678)
    device = torch.device("cuda")
    hidden_states = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    router_weight = torch.randn(10, 8, device=device, dtype=torch.bfloat16)
    router_bias = torch.linspace(-0.25, 0.5, 10, device=device, dtype=torch.float32)

    monkeypatch.setattr(
        deepseek_v4_router,
        "deepseek_v4_router_reference",
        _fail_router_reference_fallback,
    )
    for _ in range(3):
        torch.ops.auto_deploy.triton_deepseek_v4_router.default(
            hidden_states,
            None,
            router_weight,
            router_bias,
            None,
            3,
            1.25,
            False,
        )
    torch.cuda.synchronize()

    expected = torch.ops.auto_deploy.triton_deepseek_v4_router.default(
        hidden_states,
        None,
        router_weight,
        router_bias,
        None,
        3,
        1.25,
        False,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = torch.ops.auto_deploy.triton_deepseek_v4_router.default(
            hidden_states,
            None,
            router_weight,
            router_bias,
            None,
            3,
            1.25,
            False,
        )

    graph.replay()
    torch.testing.assert_close(replayed[0], expected[0])
    torch.testing.assert_close(replayed[1], expected[1], atol=1.0e-5, rtol=1.0e-5)
