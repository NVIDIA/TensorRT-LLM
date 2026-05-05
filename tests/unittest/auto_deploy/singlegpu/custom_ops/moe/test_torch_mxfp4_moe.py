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
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    PackedMxfp4ExpertsCheckpointLayout,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
from tensorrt_llm._torch.auto_deploy.transform.library.mxfp4_moe import (
    InsertMXFP4MLP,
    MXFP4MLPConfig,
)

_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_E8M0_EXPONENT_BIAS = 127
_MXFP4_BLOCK_SIZE = 32


def _scale_from_byte(scale_byte: int) -> float:
    return float(2.0 ** (scale_byte - _E8M0_EXPONENT_BIAS))


def _make_mxfp4_weight(
    shape: tuple[int, ...],
    *,
    scale_byte: int = 127,
    code_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cols = shape[-1]
    assert cols % _MXFP4_BLOCK_SIZE == 0
    codes = (torch.arange(torch.tensor(shape).prod().item()) + code_offset) % len(_E2M1_VALUES)
    codes = codes.reshape(shape).to(torch.uint8)
    dense = _E2M1_VALUES[codes.long()] * _scale_from_byte(scale_byte)

    block_codes = codes.reshape(*shape[:-1], cols // _MXFP4_BLOCK_SIZE, _MXFP4_BLOCK_SIZE)
    low = block_codes[..., 0::2]
    high = block_codes[..., 1::2]
    blocks = (low | (high << 4)).to(torch.uint8)
    scales = torch.full(blocks.shape[:-1], scale_byte, dtype=torch.uint8)
    return dense, blocks, scales


def _make_mxfp4_checkpoint_weight(
    shape: tuple[int, int],
    *,
    scale_byte: int = 127,
    code_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dense, blocks, scales = _make_mxfp4_weight(
        shape,
        scale_byte=scale_byte,
        code_offset=code_offset,
    )
    return dense, blocks.reshape(shape[0], shape[1] // 2), scales


def _deepseek_layout() -> PackedMxfp4ExpertsCheckpointLayout:
    return PackedMxfp4ExpertsCheckpointLayout(
        expert_key_pattern=(
            r"layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
            r"(?P<projection>w[123])\.(?P<tensor_kind>weight|scale)"
        ),
        runtime_gate_up_order=("w3", "w1"),
        runtime_down_projection="w2",
        expert_block_size=32,
    )


def _deepseek_expert_key(layer: int, expert: int, projection: str, tensor_kind: str) -> str:
    return f"layers.{layer}.ffn.experts.{expert}.{projection}.{tensor_kind}"


def _deepseek_packed_params_from_layout(
    num_experts: int,
    *,
    layer: int = 0,
    hidden_size: int = 32,
    intermediate_size: int = 32,
) -> tuple[
    object,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    state = {}
    w1_dense = []
    w2_dense = []
    w3_dense = []

    for expert in range(num_experts):
        w1, w1_packed, w1_scales = _make_mxfp4_checkpoint_weight(
            (intermediate_size, hidden_size),
            scale_byte=127,
            code_offset=3 + 17 * expert,
        )
        w2, w2_packed, w2_scales = _make_mxfp4_checkpoint_weight(
            (hidden_size, intermediate_size),
            scale_byte=127,
            code_offset=7 + 19 * expert,
        )
        w3, w3_packed, w3_scales = _make_mxfp4_checkpoint_weight(
            (intermediate_size, hidden_size),
            scale_byte=127,
            code_offset=11 + 23 * expert,
        )
        w1_dense.append(w1)
        w2_dense.append(w2)
        w3_dense.append(w3)
        for projection, weight, scales in (
            ("w1", w1_packed, w1_scales),
            ("w2", w2_packed, w2_scales),
            ("w3", w3_packed, w3_scales),
        ):
            state[_deepseek_expert_key(layer, expert, projection, "weight")] = weight
            state[_deepseek_expert_key(layer, expert, projection, "scale")] = scales

    packed = _deepseek_layout().pack_experts(
        state,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    return packed, torch.stack(w1_dense), torch.stack(w2_dense), torch.stack(w3_dense)


def _inputs(num_experts: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = 32
    x = torch.linspace(-0.25, 0.25, steps=5 * hidden_size, dtype=torch.float32).reshape(
        5, hidden_size
    )
    x[:, :4] = torch.tensor(
        [
            [2.0, 0.0, 1.0, -1.0],
            [0.0, 2.0, 1.0, 0.0],
            [-1.0, 0.0, 3.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, -1.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )

    router_weight = torch.zeros((num_experts, hidden_size), dtype=torch.float32)
    for expert_idx in range(num_experts):
        router_weight[expert_idx, expert_idx % 4] = 1.0 + 0.25 * expert_idx
    router_bias = torch.linspace(0.3, -0.2, steps=num_experts, dtype=torch.float32)
    return x, router_weight, router_bias


def _dense_reference(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    alpha: float,
    limit: float,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    *,
    expert_start: int = 0,
) -> torch.Tensor:
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    router_logits = F.linear(x, router_weight, router_bias)
    router_top_value, selected_experts = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = torch.softmax(router_top_value, dim=-1, dtype=torch.float32)

    output = torch.zeros((x.shape[0], hidden_size), dtype=torch.float32)
    for local_expert_idx in range(gate_up_weight.shape[0]):
        global_expert_idx = expert_start + local_expert_idx
        token_idx, route_idx = torch.where(selected_experts == global_expert_idx)
        if token_idx.numel() == 0:
            continue

        gate_up = F.linear(
            x[token_idx], gate_up_weight[local_expert_idx], gate_up_bias[local_expert_idx]
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        inter = (gate * torch.sigmoid(gate * alpha)) * (up + 1.0)
        expert_out = F.linear(inter, down_weight[local_expert_idx], down_bias[local_expert_idx])
        output.index_add_(0, token_idx, expert_out * routing_weights[token_idx, route_idx, None])
    return output.reshape(*leading_shape, hidden_size)


def _dense_deepseek_routing_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    *,
    alpha: float,
    limit: float,
    expert_start: int = 0,
) -> torch.Tensor:
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    selected_experts = selected_experts.reshape(x.shape[0], -1)
    routing_weights = routing_weights.reshape_as(selected_experts).to(torch.float32)

    output = torch.zeros((x.shape[0], hidden_size), dtype=torch.float32)
    for local_expert_idx in range(w1_weight.shape[0]):
        global_expert_idx = expert_start + local_expert_idx
        token_idx, route_idx = torch.where(selected_experts == global_expert_idx)
        if token_idx.numel() == 0:
            continue

        gate = F.linear(x[token_idx], w1_weight[local_expert_idx])
        up = F.linear(x[token_idx], w3_weight[local_expert_idx])
        if limit > 0.0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        inter = (gate * torch.sigmoid(gate * alpha)) * up
        expert_out = F.linear(inter, w2_weight[local_expert_idx])
        output.index_add_(0, token_idx, expert_out * routing_weights[token_idx, route_idx, None])
    return output.reshape(*leading_shape, hidden_size)


def _mxfp4_params(
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = 32
    intermediate_size = 32
    gate_up_weight, gate_up_blocks, gate_up_scales = _make_mxfp4_weight(
        (num_experts, 2 * intermediate_size, hidden_size),
        scale_byte=127,
        code_offset=1,
    )
    down_weight, down_blocks, down_scales = _make_mxfp4_weight(
        (num_experts, hidden_size, intermediate_size),
        scale_byte=128,
        code_offset=5,
    )
    return gate_up_weight, gate_up_blocks, gate_up_scales, down_weight, down_blocks, down_scales


def _expert_slice(num_experts: int, ep_size: int, ep_rank: int) -> tuple[int, int]:
    base = num_experts // ep_size
    lo = base * ep_rank
    hi = num_experts if ep_rank == ep_size - 1 else base * (ep_rank + 1)
    return lo, hi


def test_torch_mxfp4_moe_matches_dense_reference_with_bias_and_swiglu_limit() -> None:
    num_experts = 4
    top_k = 2
    alpha = 1.7
    limit = 0.75
    x, router_weight, router_bias = _inputs(num_experts)
    selected_experts = torch.topk(F.linear(x, router_weight, router_bias), top_k, dim=-1).indices
    assert selected_experts.unique().numel() < selected_experts.numel()
    gate_up_weight, gate_up_blocks, gate_up_scales, down_weight, down_blocks, down_scales = (
        _mxfp4_params(num_experts)
    )
    gate_up_bias = torch.linspace(-0.2, 0.3, steps=num_experts * 64, dtype=torch.float32).reshape(
        num_experts, 64
    )
    down_bias = torch.linspace(0.15, -0.1, steps=num_experts * 32, dtype=torch.float32).reshape(
        num_experts, 32
    )

    actual = torch.ops.auto_deploy.torch_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        top_k,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
    )
    expected = _dense_reference(
        x,
        router_weight,
        router_bias,
        top_k,
        gate_up_weight,
        gate_up_bias,
        alpha,
        limit,
        down_weight,
        down_bias,
    )
    unlimited = torch.ops.auto_deploy.torch_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        top_k,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        10.0,
        down_blocks,
        down_bias,
        down_scales,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(actual, unlimited)


def test_torch_mxfp4_moe_ep_partitions_experts_and_sums_to_full_output() -> None:
    num_experts = 5
    ep_size = 3
    top_k = 3
    alpha = 1.2
    limit = 1.25
    x, router_weight, router_bias = _inputs(num_experts)
    gate_up_weight, gate_up_blocks, gate_up_scales, down_weight, down_blocks, down_scales = (
        _mxfp4_params(num_experts)
    )
    gate_up_bias = torch.linspace(0.2, -0.25, steps=num_experts * 64, dtype=torch.float32).reshape(
        num_experts, 64
    )
    down_bias = torch.linspace(-0.1, 0.2, steps=num_experts * 32, dtype=torch.float32).reshape(
        num_experts, 32
    )

    full = torch.ops.auto_deploy.torch_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        top_k,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
    )
    partial_sum = torch.zeros_like(full)
    expected_sum = torch.zeros_like(full)
    for ep_rank in range(ep_size):
        lo, hi = _expert_slice(num_experts, ep_size, ep_rank)
        partial_sum += torch.ops.auto_deploy.torch_mxfp4_moe_ep(
            x,
            router_weight,
            router_bias,
            top_k,
            gate_up_blocks[lo:hi],
            gate_up_bias[lo:hi],
            gate_up_scales[lo:hi],
            alpha,
            limit,
            down_blocks[lo:hi],
            down_bias[lo:hi],
            down_scales[lo:hi],
            ep_size,
            ep_rank,
        )
        expected_sum += _dense_reference(
            x,
            router_weight,
            router_bias,
            top_k,
            gate_up_weight[lo:hi],
            gate_up_bias[lo:hi],
            alpha,
            limit,
            down_weight[lo:hi],
            down_bias[lo:hi],
            expert_start=lo,
        )

    torch.testing.assert_close(partial_sum, full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(partial_sum, expected_sum, rtol=1e-5, atol=1e-5)


def test_torch_mxfp4_moe_from_routing_matches_deepseek_layout_reference() -> None:
    num_experts = 3
    hidden_size = 32
    intermediate_size = 32
    alpha = 1.0
    limit = 0.75
    x = torch.linspace(-0.3, 0.35, steps=4 * hidden_size, dtype=torch.float32).reshape(
        4, hidden_size
    )
    selected_experts = torch.tensor(
        [[0, 0], [2, 1], [1, 2], [2, 2]],
        dtype=torch.int64,
    )
    routing_weights = torch.tensor(
        [[0.2, 0.35], [0.6, 0.1], [0.25, 0.45], [0.4, 0.15]],
        dtype=torch.float32,
    )
    packed, w1_weight, w2_weight, w3_weight = _deepseek_packed_params_from_layout(num_experts)
    gate_up_bias = torch.zeros((num_experts, 2 * intermediate_size), dtype=torch.float32)
    down_bias = torch.zeros((num_experts, hidden_size), dtype=torch.float32)

    actual = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
        x,
        selected_experts,
        routing_weights,
        packed.gate_up_blocks,
        gate_up_bias,
        packed.gate_up_scales,
        alpha,
        limit,
        packed.down_blocks,
        down_bias,
        packed.down_scales,
        "up_gate",
        "deepseek",
    )
    expected = _dense_deepseek_routing_reference(
        x,
        selected_experts,
        routing_weights,
        w1_weight,
        w2_weight,
        w3_weight,
        alpha=alpha,
        limit=limit,
    )

    assert expected.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_torch_mxfp4_moe_from_routing_ep_partitions_deepseek_layout_experts() -> None:
    num_experts = 5
    ep_size = 3
    hidden_size = 32
    intermediate_size = 32
    alpha = 1.0
    limit = 1.0
    x = torch.linspace(-0.4, 0.4, steps=4 * hidden_size, dtype=torch.float32).reshape(
        4, hidden_size
    )
    selected_experts = torch.tensor(
        [[0, 4, 2], [3, 1, 4], [2, 2, 0], [1, 3, 4]],
        dtype=torch.int64,
    )
    routing_weights = torch.tensor(
        [[0.2, 0.5, 0.1], [0.7, 0.15, 0.05], [0.25, 0.2, 0.1], [0.3, 0.25, 0.35]],
        dtype=torch.float32,
    )
    packed, _, _, _ = _deepseek_packed_params_from_layout(num_experts)
    gate_up_bias = torch.zeros((num_experts, 2 * intermediate_size), dtype=torch.float32)
    down_bias = torch.zeros((num_experts, hidden_size), dtype=torch.float32)

    full = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
        x,
        selected_experts,
        routing_weights,
        packed.gate_up_blocks,
        gate_up_bias,
        packed.gate_up_scales,
        alpha,
        limit,
        packed.down_blocks,
        down_bias,
        packed.down_scales,
        "up_gate",
        "deepseek",
    )
    partial_sum = torch.zeros_like(full)
    for ep_rank in range(ep_size):
        lo, hi = _expert_slice(num_experts, ep_size, ep_rank)
        partial_sum += torch.ops.auto_deploy.torch_mxfp4_moe_from_routing_ep(
            x,
            selected_experts,
            routing_weights,
            packed.gate_up_blocks[lo:hi],
            gate_up_bias[lo:hi],
            packed.gate_up_scales[lo:hi],
            alpha,
            limit,
            packed.down_blocks[lo:hi],
            down_bias[lo:hi],
            packed.down_scales[lo:hi],
            "up_gate",
            "deepseek",
            expert_start=lo,
        )

    torch.testing.assert_close(partial_sum, full, rtol=1e-5, atol=1e-5)


def test_mxfp4_transform_backend_selector_prefers_torch_for_checkpoint_layout() -> None:
    config = MXFP4MLPConfig(stage=Stages.PATTERN_MATCHER)
    transform = InsertMXFP4MLP(config)

    assert transform._resolve_backend({"quant_method": "mxfp4"}, None) == "triton"
    assert transform._resolve_backend({"expert_quant_method": "mxfp4"}, object()) == "torch"
    assert (
        InsertMXFP4MLP(
            MXFP4MLPConfig(stage=Stages.PATTERN_MATCHER, mxfp4_backend="triton")
        )._resolve_backend({"expert_quant_method": "mxfp4"}, object())
        == "triton"
    )
