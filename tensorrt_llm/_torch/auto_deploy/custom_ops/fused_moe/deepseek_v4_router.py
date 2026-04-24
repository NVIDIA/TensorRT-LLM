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

from typing import Optional

import torch
import torch.nn.functional as F


def _num_tokens(hidden_states: torch.Tensor) -> int:
    if hidden_states.dim() < 2:
        raise ValueError("hidden_states must have shape [T, H] or [B, S, H]")
    return hidden_states.numel() // hidden_states.shape[-1]


def _normalize_and_scale(weights: torch.Tensor, route_scale: float) -> torch.Tensor:
    finite_weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    has_inf = torch.isinf(weights).any(dim=-1, keepdim=True)
    inf_mask = torch.isinf(weights).to(weights.dtype)
    inf_count = inf_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    inf_normalized = inf_mask / inf_count

    row_max = finite_weights.amax(dim=-1, keepdim=True)
    tiny = torch.finfo(weights.dtype).tiny
    scaled_weights = finite_weights / row_max.clamp_min(tiny)
    finite_normalized = scaled_weights / scaled_weights.sum(dim=-1, keepdim=True).clamp_min(tiny)
    normalized = torch.where(has_inf, inf_normalized, finite_normalized)
    return normalized * route_scale


def deepseek_v4_router_reference(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    is_hash_layer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference DeepSeek V4 sqrtsoftplus router.

    Args:
        hidden_states: Hidden states with shape ``[T, H]`` or ``[B, S, H]``.
        input_ids: Token ids with shape ``[T]`` or ``[B, S]``. Required for hash layers.
        router_weight: Router projection weight with shape ``[E, H]``.
        router_bias: Optional expert bias with shape ``[E]``. It affects top-k selection only.
        tid2eid: Hash routing table with shape ``[vocab, top_k]``. Required for hash layers.
        top_k: Number of experts selected per token.
        route_scale: Post-normalization routing scale.
        is_hash_layer: Whether to use ``tid2eid[input_ids]`` instead of score top-k.

    Returns:
        ``(selected_experts, routing_weights)`` with shapes ``[T, top_k]``.
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    logits = F.linear(hidden_flat.float(), router_weight.float())
    scores = torch.sqrt(F.softplus(logits))

    if is_hash_layer:
        if input_ids is None:
            raise ValueError("input_ids is required for DeepSeek V4 hash-routed layers")
        if tid2eid is None:
            raise ValueError("tid2eid is required for DeepSeek V4 hash-routed layers")
        selected_experts = tid2eid[input_ids.reshape(-1).long()]
        if selected_experts.shape[-1] != top_k:
            selected_experts = selected_experts[..., :top_k]
    else:
        selection_scores = scores
        if router_bias is not None:
            selection_scores = selection_scores + router_bias.float()
        selected_experts = torch.topk(selection_scores, top_k, dim=-1).indices

    weights = scores.gather(1, selected_experts.long())
    routing_weights = _normalize_and_scale(weights, route_scale)
    return selected_experts, routing_weights


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_router", mutates_args=())
def torch_deepseek_v4_router(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    is_hash_layer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return deepseek_v4_router_reference(
        hidden_states,
        input_ids,
        router_weight,
        router_bias,
        tid2eid,
        top_k,
        route_scale,
        is_hash_layer,
    )


@torch_deepseek_v4_router.register_fake
def _torch_deepseek_v4_router_fake(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    is_hash_layer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input_ids, router_weight, router_bias, route_scale

    num_tokens = _num_tokens(hidden_states)
    selected_dtype = tid2eid.dtype if is_hash_layer and tid2eid is not None else torch.int64
    selected_experts = hidden_states.new_empty(
        (num_tokens, top_k), dtype=selected_dtype, device=hidden_states.device
    )
    routing_weights = hidden_states.new_empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
    return selected_experts, routing_weights
