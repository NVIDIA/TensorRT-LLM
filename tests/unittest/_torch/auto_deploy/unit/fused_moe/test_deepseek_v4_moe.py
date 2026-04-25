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

import operator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_moe import (
    deepseek_v4_limited_swiglu,
    deepseek_v4_moe_reference,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_moe import (
    _deepseek_v4_swiglu_torch,
    _interleave_deepseek_v4_gate_up,
    _routing_from_precomputed,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4MoE,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_moe import (
    DeepSeekV4MoELowering,
    DeepSeekV4MoELoweringError,
)
from tensorrt_llm._torch.auto_deploy.transform.library.deepseek_v4_mxfp4 import (
    load_deepseek_v4_mxfp4_experts,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ApplyShardingHints
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op


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


def _run_ir_sharding(
    gm: torch.fx.GraphModule,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.fx.GraphModule, object]:
    transform = ApplyShardingHints.from_kwargs(stage=Stages.SHARDING)
    shared_config = SharedConfig(
        local_rank=rank,
        world_size=world_size,
        dist_config=DistConfig(
            world_size=world_size,
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
        ),
    )
    return transform._apply(gm, None, None, shared_config)


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


def test_deepseek_v4_mxfp4_swiglu_matches_reference_without_gptoss_offset() -> None:
    gate_up_interleaved = torch.tensor([[20.0, 15.0, -20.0, -15.0]], dtype=torch.float32)

    actual = _deepseek_v4_swiglu_torch(gate_up_interleaved, alpha=1.0, limit=10.0)

    expected_gate = torch.tensor([[10.0, -20.0]], dtype=torch.float32)
    expected_up = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    expected = F.silu(expected_gate) * expected_up
    gptoss_style = F.silu(expected_gate) * (expected_up + 1)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, gptoss_style)


def test_deepseek_v4_mxfp4_interleaves_checkpoint_gate_up_order() -> None:
    checkpoint_order = torch.tensor([[30, 31, 32, 10, 11, 12]], dtype=torch.uint8)

    actual = _interleave_deepseek_v4_gate_up(checkpoint_order)

    expected = torch.tensor([[10, 30, 11, 31, 12, 32]], dtype=torch.uint8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prepare_weights_scales_interleaves_deepseek_gate_up_before_swizzle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeSwizzledTensor:
        shape: torch.Size | None = None

    swizzle_calls = []

    def _fake_swizzle_mxfp4(
        weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> tuple[_FakeSwizzledTensor, _FakeSwizzledTensor]:
        swizzle_calls.append((weight.clone(), weight_scale.clone()))
        return _FakeSwizzledTensor(), _FakeSwizzledTensor()

    monkeypatch.setattr(mxfp4_moe, "_swizzle_mxfp4", _fake_swizzle_mxfp4)
    mxfp4_moe._clear_mxfp4_weights_scales_cache()
    hidden_size = 64
    intermediate_size = 32
    gate_up_blocks = (
        torch.arange(2 * intermediate_size, dtype=torch.uint8)
        .view(1, 2 * intermediate_size, 1, 1)
        .expand(1, 2 * intermediate_size, hidden_size // 32, 16)
        .contiguous()
    )
    gate_up_scales = (
        torch.arange(2 * intermediate_size, dtype=torch.uint8)
        .view(1, 2 * intermediate_size, 1)
        .expand(1, 2 * intermediate_size, hidden_size // 32)
        .contiguous()
    )
    down_blocks = torch.zeros((1, hidden_size, intermediate_size // 32, 16), dtype=torch.uint8)
    down_scales = torch.zeros((1, hidden_size, intermediate_size // 32), dtype=torch.uint8)

    mxfp4_moe._prepare_weights_scales(
        hidden_size,
        gate_up_blocks,
        gate_up_scales,
        down_blocks,
        down_scales,
        interleave_gate_up=True,
    )

    gate_up_weight, gate_up_scale = swizzle_calls[0]
    expected_prefix = torch.tensor([32, 0, 33, 1, 34, 2, 35, 3], dtype=torch.uint8)
    torch.testing.assert_close(gate_up_weight[0, 0, :8], expected_prefix, rtol=0, atol=0)
    torch.testing.assert_close(gate_up_scale[0, 0, :8], expected_prefix, rtol=0, atol=0)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_precomputed_routing_accepts_non_power_of_two_topk() -> None:
    selected_experts = torch.tensor(
        [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]],
        dtype=torch.int64,
        device="cuda",
    )
    routing_weights = torch.arange(12, dtype=torch.float32, device="cuda").view(2, 6)

    routing_data, gather_idx, scatter_idx = _routing_from_precomputed(
        selected_experts,
        routing_weights,
        num_experts=256,
    )

    expected_order = torch.argsort(selected_experts.reshape(-1), stable=True).to(torch.int32)
    expected_inverse = torch.argsort(expected_order, stable=True).to(torch.int32)
    expected_hist = torch.zeros(256, dtype=torch.int32, device="cuda")
    expected_hist[:12] = 1

    assert routing_data.n_expts_act == 6
    assert routing_data.n_expts_tot == 256
    torch.testing.assert_close(routing_data.expt_hist, expected_hist)
    torch.testing.assert_close(gather_idx.src_indx, expected_order)
    torch.testing.assert_close(gather_idx.dst_indx, expected_inverse)
    torch.testing.assert_close(scatter_idx.src_indx, expected_inverse)
    torch.testing.assert_close(scatter_idx.dst_indx, expected_order)
    torch.testing.assert_close(
        routing_data.gate_scal,
        routing_weights.reshape(-1)[expected_order.to(torch.int64)],
    )

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_routing_data, captured_gather_idx, captured_scatter_idx = (
            _routing_from_precomputed(
                selected_experts,
                routing_weights,
                num_experts=256,
            )
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_routing_data.expt_hist, expected_hist)
    torch.testing.assert_close(captured_gather_idx.src_indx, expected_order)
    torch.testing.assert_close(captured_gather_idx.dst_indx, expected_inverse)
    torch.testing.assert_close(captured_scatter_idx.src_indx, expected_inverse)
    torch.testing.assert_close(captured_scatter_idx.dst_indx, expected_order)
    torch.testing.assert_close(
        captured_routing_data.gate_scal,
        routing_weights.reshape(-1)[expected_order.to(torch.int64)],
    )


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


def _uint8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.int64)
    return values.reshape(shape).add(offset).remainder(251).to(torch.uint8)


class _MXFP4FromRoutingShardingModule(nn.Module):
    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.ffn = nn.Module()
        self.ffn.gate_up_blocks = nn.Parameter(
            _uint8_pattern((num_experts, 8, 2, 16), 1),
            requires_grad=False,
        )
        self.ffn.gate_up_bias = nn.Parameter(
            torch.arange(num_experts * 8, dtype=torch.float32).reshape(num_experts, 8),
            requires_grad=False,
        )
        self.ffn.gate_up_scales = nn.Parameter(
            _uint8_pattern((num_experts, 8, 2), 11),
            requires_grad=False,
        )
        self.ffn.down_blocks = nn.Parameter(
            _uint8_pattern((num_experts, 4, 2, 16), 23),
            requires_grad=False,
        )
        self.ffn.down_bias = nn.Parameter(
            torch.arange(num_experts * 4, dtype=torch.float32).reshape(num_experts, 4).add(100),
            requires_grad=False,
        )
        self.ffn.down_scales = nn.Parameter(
            _uint8_pattern((num_experts, 4, 2), 37),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default(
            hidden_states,
            selected_experts,
            routing_weights,
            self.ffn.gate_up_blocks,
            self.ffn.gate_up_bias,
            self.ffn.gate_up_scales,
            1.0,
            10.0,
            self.ffn.down_blocks,
            self.ffn.down_bias,
            self.ffn.down_scales,
            "moe",
        )


def _trace_mxfp4_from_routing_sharding_module(num_experts: int = 4) -> torch.fx.GraphModule:
    gm = torch.fx.symbolic_trace(_MXFP4FromRoutingShardingModule(num_experts))
    params_and_buffers = dict(gm.named_parameters())
    params_and_buffers.update(dict(gm.named_buffers()))
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target == "hidden_states":
                node.meta["val"] = torch.empty((3, 4), dtype=torch.bfloat16)
            elif node.target == "selected_experts":
                node.meta["val"] = torch.empty((3, 2), dtype=torch.int64)
            elif node.target == "routing_weights":
                node.meta["val"] = torch.empty((3, 2), dtype=torch.float32)
        elif node.op == "get_attr" and node.target in params_and_buffers:
            node.meta["val"] = params_and_buffers[node.target].detach()
        elif is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        ):
            node.meta["val"] = torch.empty((3, 4), dtype=torch.bfloat16)
    return gm


def _shift_state_value(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.int64).add(19).remainder(251).to(torch.uint8)
    return tensor + 19


def test_deepseek_v4_mxfp4_from_routing_shards_experts_and_localizes_ids() -> None:
    gm = _trace_mxfp4_from_routing_sharding_module()
    full_state = {name: tensor.detach().clone() for name, tensor in gm.named_parameters()}

    transformed, info = _run_ir_sharding(gm, rank=1, world_size=2)

    assert info.num_matches == 1
    for name, full_tensor in full_state.items():
        local_tensor = transformed.get_parameter(name)
        assert local_tensor.shape[0] == 2
        torch.testing.assert_close(local_tensor, full_tensor[2:4], rtol=0, atol=0)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert selected_local.target == operator.mul
    assert routing_local.target == operator.mul
    assert selected_local.args[1] is routing_local.args[1]
    assert selected_local.args[0].target == operator.sub
    assert selected_local.args[0].args[1] == 2
    assert selected_local.args[1].target == torch.logical_and
    assert any(
        node.op == "call_function"
        and node.args
        and node.args[0] is moe_node
        and "all_reduce" in str(node.target)
        for node in transformed.graph.nodes
    )

    shifted_full_state = {name: _shift_state_value(tensor) for name, tensor in full_state.items()}
    load_result = transformed.load_state_dict(shifted_full_state, strict=False)

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    for name, full_tensor in shifted_full_state.items():
        torch.testing.assert_close(
            transformed.get_parameter(name),
            full_tensor[2:4],
            rtol=0,
            atol=0,
        )


def test_deepseek_v4_mxfp4_from_routing_ep8_localizes_rank7_global_ids() -> None:
    gm = _trace_mxfp4_from_routing_sharding_module(num_experts=16)
    full_state = {name: tensor.detach().clone() for name, tensor in gm.named_parameters()}

    transformed, info = _run_ir_sharding(gm, rank=7, world_size=8)

    assert info.num_matches == 1
    for name, full_tensor in full_state.items():
        local_tensor = transformed.get_parameter(name)
        assert local_tensor.shape[0] == 2
        torch.testing.assert_close(local_tensor, full_tensor[14:16], rtol=0, atol=0)

    moe_node = next(
        node
        for node in transformed.graph.nodes
        if is_op(
            node,
            torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing,
        )
    )
    selected_local = moe_node.args[1]
    routing_local = moe_node.args[2]
    assert selected_local.target == operator.mul
    assert routing_local.target == operator.mul
    assert selected_local.args[1] is routing_local.args[1]
    local_ids = selected_local.args[0]
    assert local_ids.target == operator.sub
    assert local_ids.args[1] == 14
    rank_mask = selected_local.args[1]
    ge_lower, lt_upper = rank_mask.args
    assert rank_mask.target == torch.logical_and
    assert ge_lower.target == torch.ge
    assert ge_lower.args[1] == 14
    assert lt_upper.target == torch.lt
    assert lt_upper.args[1] == 16


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


def test_lowering_bridge_rejects_non_layered_graph_with_clear_error() -> None:
    gm = _trace_canonical_moe()
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.POST_LOAD_FUSION)

    with pytest.raises(
        DeepSeekV4MoELoweringError,
        match="requires routed w1 weights.*stack of per-expert get_attr",
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


class _LayeredMoEBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.ffn = DeepseekV4MoE(config, layer_idx=0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.ffn(hidden_states, input_ids)


class _LayeredMoEModel(nn.Module):
    def __init__(self, hidden_size: int = 64, moe_intermediate_size: int = 32) -> None:
        super().__init__()
        config = DeepseekV4Config(
            vocab_size=16,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            num_hash_layers=0,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
            swiglu_limit=10.0,
        )
        self.layers = nn.ModuleList([_LayeredMoEBlock(config)])

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden_states, input_ids)


def _lower_layered_moe_model(
    hidden_size: int = 64,
    moe_intermediate_size: int = 32,
) -> tuple[torch.fx.GraphModule, object]:
    model = _LayeredMoEModel(hidden_size, moe_intermediate_size)
    gm = torch_export_to_gm(
        model,
        args=(torch.randn(1, 2, hidden_size), torch.ones(1, 2, dtype=torch.long)),
    )
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.PATTERN_MATCHER)
    return transform._apply(gm, None, None, SharedConfig())


def _shared_fp8_linear_nodes(gm: torch.fx.GraphModule) -> dict[str, torch.fx.Node]:
    nodes = {}
    for node in gm.graph.nodes:
        if not is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear):
            continue
        [weight] = extract_op_args(node, "weight_quantized")
        if isinstance(weight, torch.fx.Node) and weight.op == "get_attr":
            nodes[str(weight.target)] = node
    return nodes


def _fp8_pattern(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod().item()), dtype=torch.float32)
    values = values.add(offset).remainder(29).sub(14).div(16).reshape(shape)
    return values.to(torch.float8_e4m3fn)


def test_lowering_bridge_shared_fp8_linears_have_tp_hints_and_all_reduce() -> None:
    lowered, info = _lower_layered_moe_model()

    assert info.num_matches == 1
    fp8_nodes = _shared_fp8_linear_nodes(lowered)
    expected_modes = {
        "layers.0.ffn.shared_experts.w1.weight": "colwise",
        "layers.0.ffn.shared_experts.w2.weight": "rowwise",
        "layers.0.ffn.shared_experts.w3.weight": "colwise",
    }
    actual_modes = {
        weight_name: extract_op_args(node, "tp_mode")[0]
        for weight_name, node in fp8_nodes.items()
        if weight_name in expected_modes
    }

    assert actual_modes == expected_modes

    shared_w2_node = fp8_nodes["layers.0.ffn.shared_experts.w2.weight"]
    shared_all_reduces = [
        node
        for node in lowered.graph.nodes
        if is_op(node, torch.ops.auto_deploy.all_reduce) and node.args[0] is shared_w2_node
    ]
    assert len(shared_all_reduces) == 1
    assert extract_op_args(shared_all_reduces[0], "layer_type") == ["moe"]


def test_lowering_bridge_shared_fp8_linears_shard_weights_and_scales() -> None:
    lowered, _ = _lower_layered_moe_model(hidden_size=256, moe_intermediate_size=256)

    transformed, info = _run_ir_sharding(lowered, rank=1, world_size=2)

    assert info.num_matches >= 4
    shared_w1 = transformed.get_submodule("layers.0.ffn.shared_experts.w1")
    shared_w2 = transformed.get_submodule("layers.0.ffn.shared_experts.w2")
    shared_w3 = transformed.get_submodule("layers.0.ffn.shared_experts.w3")
    assert shared_w1.weight.shape == (128, 256)
    assert shared_w2.weight.shape == (256, 128)
    assert shared_w3.weight.shape == (128, 256)
    assert shared_w1.weight_scale_inv.shape == (1, 2)
    assert shared_w2.weight_scale_inv.shape == (2, 1)
    assert shared_w3.weight_scale_inv.shape == (1, 2)
    all_reduce_nodes = [
        node
        for node in transformed.graph.nodes
        if node.op == "call_function" and "all_reduce" in str(node.target)
    ]
    assert all_reduce_nodes
    assert all(node.kwargs == {} for node in all_reduce_nodes)
    assert all(len(node.args) == 2 for node in all_reduce_nodes)

    full_state = {
        "layers.0.ffn.shared_experts.w1.weight": _fp8_pattern((256, 256), 1),
        "layers.0.ffn.shared_experts.w1.weight_scale_inv": torch.arange(
            4, dtype=torch.float32
        ).reshape(2, 2),
        "layers.0.ffn.shared_experts.w2.weight": _fp8_pattern((256, 256), 3),
        "layers.0.ffn.shared_experts.w2.weight_scale_inv": torch.arange(4, dtype=torch.float32)
        .reshape(2, 2)
        .add(10),
        "layers.0.ffn.shared_experts.w3.weight": _fp8_pattern((256, 256), 5),
        "layers.0.ffn.shared_experts.w3.weight_scale_inv": torch.arange(4, dtype=torch.float32)
        .reshape(2, 2)
        .add(20),
    }
    load_result = transformed.load_state_dict(full_state, strict=False)

    assert load_result.unexpected_keys == []
    torch.testing.assert_close(
        shared_w1.weight.float(),
        full_state["layers.0.ffn.shared_experts.w1.weight"][128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w2.weight.float(),
        full_state["layers.0.ffn.shared_experts.w2.weight"][:, 128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w3.weight.float(),
        full_state["layers.0.ffn.shared_experts.w3.weight"][128:256].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w1.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w1.weight_scale_inv"][1:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w2.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w2.weight_scale_inv"][:, 1:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        shared_w3.weight_scale_inv,
        full_state["layers.0.ffn.shared_experts.w3.weight_scale_inv"][1:2],
        rtol=0,
        atol=0,
    )


def _pack_fp4(logical_values: torch.Tensor) -> torch.Tensor:
    low = logical_values[..., 0::2] & 0x0F
    high = (logical_values[..., 1::2] & 0x0F) << 4
    return (low | high).contiguous().view(torch.int8)


def _logical_fp4(rows: int, cols: int, offset: int) -> torch.Tensor:
    return (torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols) + offset) & 0x0F


def _packed_checkpoint_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    hidden_size = 64
    intermediate_size = 32
    for expert_idx in range(2):
        prefix = f"layers.0.ffn.experts.{expert_idx}"
        state[f"{prefix}.w1.weight"] = _pack_fp4(
            _logical_fp4(intermediate_size, hidden_size, 1 + expert_idx)
        )
        state[f"{prefix}.w2.weight"] = _pack_fp4(
            _logical_fp4(hidden_size, intermediate_size, 5 + expert_idx)
        )
        state[f"{prefix}.w3.weight"] = _pack_fp4(
            _logical_fp4(intermediate_size, hidden_size, 9 + expert_idx)
        )
        state[f"{prefix}.w1.scale"] = torch.full(
            (intermediate_size, hidden_size // 32), 17 + expert_idx, dtype=torch.uint8
        )
        state[f"{prefix}.w2.scale"] = torch.full(
            (hidden_size, intermediate_size // 32), 29 + expert_idx, dtype=torch.uint8
        )
        state[f"{prefix}.w3.scale"] = torch.full(
            (intermediate_size, hidden_size // 32), 43 + expert_idx, dtype=torch.uint8
        )

    for proj, shape in (("w1", (32, 64)), ("w2", (64, 32)), ("w3", (32, 64))):
        state[f"layers.0.ffn.shared_experts.{proj}.weight"] = torch.zeros(
            shape, dtype=torch.float8_e4m3fn
        )
        state[f"layers.0.ffn.shared_experts.{proj}.scale"] = torch.ones((1, 1), dtype=torch.float32)
    return state


def test_lowering_bridge_loads_packed_mxfp4_checkpoint_without_dense_shape_mismatch() -> None:
    model = _LayeredMoEModel()
    gm = torch_export_to_gm(
        model,
        args=(torch.randn(1, 2, 64), torch.ones(1, 2, dtype=torch.long)),
    )
    transform = DeepSeekV4MoELowering.from_kwargs(stage=Stages.PATTERN_MATCHER)

    lowered, info = transform._apply(gm, None, None, SharedConfig())
    targets = [node.target for node in lowered.graph.nodes]

    assert info.num_matches == 1
    assert torch.ops.auto_deploy.torch_deepseek_v4_router.default in targets
    assert torch.ops.auto_deploy.triton_deepseek_v4_mxfp4_moe_from_routing.default in targets
    assert torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default in targets
    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_deepseek_v4_moe) for node in lowered.graph.nodes
    )

    state_dict = _packed_checkpoint_state()
    load_result = lowered.load_state_dict(state_dict, strict=False)
    expected_layout = load_deepseek_v4_mxfp4_experts(
        _packed_checkpoint_state(),
        layer_idx=0,
        hidden_size=64,
        intermediate_size=32,
        num_experts=2,
    )

    assert load_result.unexpected_keys == []
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_gate_up_blocks"),
        expected_layout.gate_up_blocks,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        lowered.get_parameter("layers.0.ffn.mxfp4_down_scales"),
        expected_layout.down_scales,
        rtol=0,
        atol=0,
    )
    assert (
        lowered.get_parameter("layers.0.ffn.shared_experts.w1.weight").dtype == torch.float8_e4m3fn
    )
    torch.testing.assert_close(
        lowered.get_buffer("layers.0.ffn.shared_experts.w1.weight_scale_inv"),
        torch.ones((1, 1), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
