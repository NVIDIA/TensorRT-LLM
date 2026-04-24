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

"""DeepSeek V4 MoE source ops.

The production DeepSeek V4 path lowers routed experts to packed MXFP4 kernels and
the shared expert to FineGrained FP8 linears. This module keeps the exported
source surface small and testable: the eager implementation is a dense PyTorch
reference for tiny synthetic tests, while production lowering is handled by the
transform library.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .deepseek_v4_router import deepseek_v4_router_reference

__all__ = [
    "deepseek_v4_limited_swiglu",
    "deepseek_v4_moe_from_routing_reference",
    "deepseek_v4_moe_reference",
    "torch_deepseek_v4_moe",
    "torch_deepseek_v4_moe_from_routing",
]


def _num_tokens(hidden_states: torch.Tensor) -> int:
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have shape [T, H] or [B, S, H]")
    return hidden_states.numel() // hidden_states.shape[-1]


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _validate_device(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _validate_routed_weights(
    hidden_size: int,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
) -> tuple[int, int]:
    _validate_rank("routed_w1_weight", routed_w1_weight, 3)
    _validate_rank("routed_w2_weight", routed_w2_weight, 3)
    _validate_rank("routed_w3_weight", routed_w3_weight, 3)

    num_experts, intermediate_size, w1_hidden = routed_w1_weight.shape
    if w1_hidden != hidden_size:
        raise ValueError(
            f"routed_w1_weight hidden dimension must be {hidden_size}, got {w1_hidden}"
        )
    if routed_w3_weight.shape != routed_w1_weight.shape:
        raise ValueError(
            "routed_w3_weight must match routed_w1_weight shape "
            f"{tuple(routed_w1_weight.shape)}, got {tuple(routed_w3_weight.shape)}"
        )
    expected_w2_shape = (num_experts, hidden_size, intermediate_size)
    if routed_w2_weight.shape != expected_w2_shape:
        raise ValueError(
            "routed_w2_weight must have shape "
            f"{expected_w2_shape}, got {tuple(routed_w2_weight.shape)}"
        )
    return num_experts, intermediate_size


def _shared_weights_are_present(
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
) -> bool:
    present = (
        shared_w1_weight is not None,
        shared_w2_weight is not None,
        shared_w3_weight is not None,
    )
    if any(present) and not all(present):
        raise ValueError("shared_w1_weight, shared_w2_weight, and shared_w3_weight must all be set")
    return all(present)


def _validate_shared_weights(
    hidden_size: int,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
) -> bool:
    if not _shared_weights_are_present(shared_w1_weight, shared_w2_weight, shared_w3_weight):
        return False

    assert shared_w1_weight is not None
    assert shared_w2_weight is not None
    assert shared_w3_weight is not None
    _validate_rank("shared_w1_weight", shared_w1_weight, 2)
    _validate_rank("shared_w2_weight", shared_w2_weight, 2)
    _validate_rank("shared_w3_weight", shared_w3_weight, 2)

    intermediate_size, w1_hidden = shared_w1_weight.shape
    if w1_hidden != hidden_size:
        raise ValueError(
            f"shared_w1_weight hidden dimension must be {hidden_size}, got {w1_hidden}"
        )
    if shared_w3_weight.shape != shared_w1_weight.shape:
        raise ValueError(
            "shared_w3_weight must match shared_w1_weight shape "
            f"{tuple(shared_w1_weight.shape)}, got {tuple(shared_w3_weight.shape)}"
        )
    expected_w2_shape = (hidden_size, intermediate_size)
    if shared_w2_weight.shape != expected_w2_shape:
        raise ValueError(
            f"shared_w2_weight must have shape {expected_w2_shape}, "
            f"got {tuple(shared_w2_weight.shape)}"
        )
    return True


def _validate_from_routing_inputs(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    swiglu_limit: float,
) -> tuple[int, int, bool]:
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")

    _validate_rank("selected_experts", selected_experts, 2)
    _validate_rank("routing_weights", routing_weights, 2)
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_experts must be int32 or int64, got {selected_experts.dtype}")
    if not routing_weights.is_floating_point():
        raise TypeError(f"routing_weights must be floating point, got {routing_weights.dtype}")

    num_tokens = _num_tokens(hidden_states)
    hidden_size = hidden_states.shape[-1]
    if selected_experts.shape[0] != num_tokens:
        raise ValueError(
            f"selected_experts token dimension must be {num_tokens}, got {selected_experts.shape[0]}"
        )
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "routing_weights must match selected_experts shape "
            f"{tuple(selected_experts.shape)}, got {tuple(routing_weights.shape)}"
        )

    device = hidden_states.device
    _validate_device("selected_experts", selected_experts, device)
    _validate_device("routing_weights", routing_weights, device)
    for name, tensor in (
        ("routed_w1_weight", routed_w1_weight),
        ("routed_w2_weight", routed_w2_weight),
        ("routed_w3_weight", routed_w3_weight),
    ):
        _validate_device(name, tensor, device)
    for name, tensor in (
        ("shared_w1_weight", shared_w1_weight),
        ("shared_w2_weight", shared_w2_weight),
        ("shared_w3_weight", shared_w3_weight),
    ):
        if tensor is not None:
            _validate_device(name, tensor, device)

    num_experts, _ = _validate_routed_weights(
        hidden_size, routed_w1_weight, routed_w2_weight, routed_w3_weight
    )
    has_shared_expert = _validate_shared_weights(
        hidden_size, shared_w1_weight, shared_w2_weight, shared_w3_weight
    )
    return num_tokens, num_experts, has_shared_expert


def deepseek_v4_limited_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Apply DeepSeek V4's routed-expert SwiGLU limit.

    The clamp is unconditional, so ``swiglu_limit=0`` produces a zero ``up``
    branch and therefore a zero gated hidden state.
    """
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)
    return F.silu(gate) * up


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
        activated = deepseek_v4_limited_swiglu(gate, up, swiglu_limit)
    else:
        activated = F.silu(gate) * up
    return F.linear(activated.to(w2_weight.dtype), w2_weight).to(hidden_states.dtype)


def deepseek_v4_moe_from_routing_reference(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    swiglu_limit: float,
) -> torch.Tensor:
    """Dense PyTorch reference for DeepSeek V4 MoE with precomputed routing.

    Args:
        hidden_states: Hidden states with shape ``[T, H]`` or ``[B, S, H]``.
        selected_experts: Expert ids with shape ``[T, top_k]``.
        routing_weights: Per-route weights with shape ``[T, top_k]``.
        routed_w1_weight: Routed gate weights with shape ``[E, I, H]``.
        routed_w2_weight: Routed down weights with shape ``[E, H, I]``.
        routed_w3_weight: Routed up weights with shape ``[E, I, H]``.
        shared_w1_weight: Optional shared gate weight with shape ``[I, H]``.
        shared_w2_weight: Optional shared down weight with shape ``[H, I]``.
        shared_w3_weight: Optional shared up weight with shape ``[I, H]``.
        swiglu_limit: Non-negative routed-expert clamp limit.

    Returns:
        MoE output with the same shape and dtype as ``hidden_states``.
    """
    _, num_experts, has_shared_expert = _validate_from_routing_inputs(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        swiglu_limit,
    )

    hidden_shape = hidden_states.shape
    hidden_size = hidden_shape[-1]
    hidden_flat = hidden_states.reshape(-1, hidden_size)
    output = torch.zeros_like(hidden_flat)

    valid_mask = (selected_experts >= 0) & (selected_experts < num_experts)
    selected_fixed = torch.where(
        valid_mask, selected_experts, torch.full_like(selected_experts, num_experts)
    )
    expert_mask = F.one_hot(selected_fixed.long(), num_classes=num_experts + 1)
    expert_mask = expert_mask[..., :num_experts].permute(2, 1, 0)

    for expert_idx in range(num_experts):
        route_idx, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        expert_input = hidden_flat[None, token_idx].reshape(-1, hidden_size)
        expert_output = _expert_mlp(
            expert_input,
            routed_w1_weight[expert_idx],
            routed_w2_weight[expert_idx],
            routed_w3_weight[expert_idx],
            swiglu_limit,
            apply_swiglu_limit=True,
        )
        scaled = expert_output * routing_weights[token_idx, route_idx, None].to(expert_output.dtype)
        output.index_add_(0, token_idx, scaled.to(output.dtype))

    if has_shared_expert:
        assert shared_w1_weight is not None
        assert shared_w2_weight is not None
        assert shared_w3_weight is not None
        shared_output = _expert_mlp(
            hidden_flat,
            shared_w1_weight,
            shared_w2_weight,
            shared_w3_weight,
            swiglu_limit,
            apply_swiglu_limit=False,
        )
        output = output + shared_output.to(output.dtype)

    return output.view(hidden_shape)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_moe_from_routing", mutates_args=())
def torch_deepseek_v4_moe_from_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    swiglu_limit: float,
    layer_type: str = "moe",
) -> torch.Tensor:
    """Reference source op for DeepSeek V4 MoE with precomputed routing.

    ``layer_type`` is graph metadata for sharding/lowering transforms and does
    not affect the eager reference result.
    """
    del layer_type
    return deepseek_v4_moe_from_routing_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        swiglu_limit,
    )


@torch_deepseek_v4_moe_from_routing.register_fake
def _torch_deepseek_v4_moe_from_routing_fake(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    swiglu_limit: float,
    layer_type: str = "moe",
) -> torch.Tensor:
    del (
        selected_experts,
        routing_weights,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        swiglu_limit,
        layer_type,
    )
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have shape [T, H] or [B, S, H]")
    return hidden_states.new_empty(hidden_states.shape).contiguous()


def deepseek_v4_moe_reference(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    is_hash_layer: bool,
) -> torch.Tensor:
    """Dense PyTorch reference for router-integrated DeepSeek V4 MoE."""
    selected_experts, routing_weights = deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias,
        tid2eid,
        top_k,
        route_scale,
        is_hash_layer,
    )
    return deepseek_v4_moe_from_routing_reference(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        swiglu_limit,
    )


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_moe", mutates_args=())
def torch_deepseek_v4_moe(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    is_hash_layer: bool,
    layer_type: str = "moe",
) -> torch.Tensor:
    """Router-integrated canonical DeepSeek V4 MoE source op."""
    del layer_type
    return deepseek_v4_moe_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias,
        tid2eid,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        top_k,
        route_scale,
        swiglu_limit,
        is_hash_layer,
    )


@torch_deepseek_v4_moe.register_fake
def _torch_deepseek_v4_moe_fake(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    routed_w1_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w3_weight: torch.Tensor,
    shared_w1_weight: Optional[torch.Tensor],
    shared_w2_weight: Optional[torch.Tensor],
    shared_w3_weight: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    swiglu_limit: float,
    is_hash_layer: bool,
    layer_type: str = "moe",
) -> torch.Tensor:
    del (
        input_ids,
        router_weight,
        router_bias,
        tid2eid,
        routed_w1_weight,
        routed_w2_weight,
        routed_w3_weight,
        shared_w1_weight,
        shared_w2_weight,
        shared_w3_weight,
        top_k,
        route_scale,
        swiglu_limit,
        is_hash_layer,
        layer_type,
    )
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have shape [T, H] or [B, S, H]")
    return hidden_states.new_empty(hidden_states.shape).contiguous()
