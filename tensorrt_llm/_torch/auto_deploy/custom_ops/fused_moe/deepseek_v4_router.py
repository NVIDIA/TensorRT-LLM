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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

DEEPSEEK_V4_NUM_ROUTED_EXPERTS = 256
DEEPSEEK_V4_TOP_K = 6
DEEPSEEK_V4_ROUTE_SCALE = 1.5
DEEPSEEK_V4_HASH_ROUTED_LAYER_INDICES = (0, 1, 2)
DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE = 8
DEEPSEEK_V4_OBSERVED_EXPERTS_PER_RANK = 32
_TRITON_MAX_ROUTER_EXPERTS = 4096
_FLOAT32_TINY = torch.finfo(torch.float32).tiny


if triton is not None and tl is not None:

    @triton.jit
    def _tl_deepseek_v4_sqrt_softplus(logits):
        softplus = tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))
        return tl.sqrt(softplus)

    @triton.jit
    def _tl_deepseek_v4_normalize_and_scale(weights, route_scale, tiny: tl.constexpr):
        is_finite = (weights == weights) & (weights != float("inf")) & (weights != -float("inf"))
        is_pos_inf = weights == float("inf")
        finite_weights = tl.where(is_finite, weights, 0.0)

        inf_count = tl.sum(tl.where(is_pos_inf, 1.0, 0.0), axis=0)
        inf_normalized = tl.where(is_pos_inf, 1.0 / tl.maximum(inf_count, 1.0), 0.0)

        row_max = tl.max(finite_weights, axis=0)
        scaled_weights = finite_weights / tl.maximum(row_max, tiny)
        scaled_sum = tl.sum(scaled_weights, axis=0)
        finite_normalized = scaled_weights / tl.maximum(scaled_sum, tiny)
        normalized = tl.where(inf_count > 0.0, inf_normalized, finite_normalized)
        return normalized * route_scale

    @triton.jit
    def _deepseek_v4_router_hash_kernel(
        logits_ptr,
        input_ids_ptr,
        tid2eid_ptr,
        selected_ptr,
        weights_ptr,
        num_experts: tl.constexpr,
        tid2eid_width: tl.constexpr,
        route_scale,
        BLOCK_K: tl.constexpr,
        TOP_K: tl.constexpr,
        TINY: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOP_K

        input_id = tl.load(input_ids_ptr + token_id).to(tl.int64)
        selected = tl.load(
            tid2eid_ptr + input_id * tid2eid_width + offs_k,
            mask=mask_k,
            other=0,
        ).to(tl.int64)
        valid_expert = mask_k & (selected >= 0) & (selected < num_experts)
        logits = tl.load(
            logits_ptr + token_id * num_experts + selected,
            mask=valid_expert,
            other=-float("inf"),
        ).to(tl.float32)
        scores = _tl_deepseek_v4_sqrt_softplus(logits)
        routing_weights = _tl_deepseek_v4_normalize_and_scale(scores, route_scale, TINY)

        tl.store(selected_ptr + token_id * TOP_K + offs_k, selected, mask=mask_k)
        tl.store(weights_ptr + token_id * TOP_K + offs_k, routing_weights, mask=mask_k)

    @triton.jit
    def _deepseek_v4_router_topk_kernel(
        logits_ptr,
        bias_ptr,
        selected_ptr,
        weights_ptr,
        num_experts: tl.constexpr,
        route_scale,
        HAS_BIAS: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TOP_K: tl.constexpr,
        TINY: tl.constexpr,
    ):
        token_id = tl.program_id(0)
        offs_e = tl.arange(0, BLOCK_E)
        mask_e = offs_e < num_experts

        logits = tl.load(
            logits_ptr + token_id * num_experts + offs_e,
            mask=mask_e,
            other=-float("inf"),
        ).to(tl.float32)
        scores = _tl_deepseek_v4_sqrt_softplus(logits)
        selection_scores = scores
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
            selection_scores = selection_scores + bias
        selection_scores = tl.where(mask_e, selection_scores, -float("inf"))

        offs_k = tl.arange(0, BLOCK_K)
        topk_scores = tl.zeros([BLOCK_K], dtype=tl.float32)
        topk_idxs = tl.zeros([BLOCK_K], dtype=tl.int64)

        for k_i in tl.static_range(TOP_K):
            max_val = tl.max(selection_scores, axis=0)
            is_max = selection_scores == max_val
            candidate = tl.where(is_max, offs_e, BLOCK_E)
            max_idx = tl.min(candidate, axis=0)
            score_val = tl.max(tl.where(offs_e == max_idx, scores, 0.0), axis=0)

            slot_mask = offs_k == k_i
            topk_scores = tl.where(slot_mask, score_val, topk_scores)
            topk_idxs = tl.where(slot_mask, max_idx.to(tl.int64), topk_idxs)
            selection_scores = tl.where(offs_e == max_idx, -float("inf"), selection_scores)

        routing_weights = _tl_deepseek_v4_normalize_and_scale(topk_scores, route_scale, TINY)
        mask_k = offs_k < TOP_K
        tl.store(selected_ptr + token_id * TOP_K + offs_k, topk_idxs, mask=mask_k)
        tl.store(weights_ptr + token_id * TOP_K + offs_k, routing_weights, mask=mask_k)

else:
    _deepseek_v4_router_hash_kernel = None
    _deepseek_v4_router_topk_kernel = None


@dataclass(frozen=True)
class DeepSeekV4ExpertParallelRoutingMetadata:
    """Explicit global-to-EP-local routing contract for DeepSeek V4 MoE.

    ``torch_deepseek_v4_router`` returns global expert ids in ``selected_experts``.
    For expert parallel execution, the MXFP4 weight tensors are sliced along the
    expert dimension and route ids must be converted to the local coordinate
    system for the rank-owned slice. Nonlocal routes keep the original
    ``[T, top_k]`` shape, are assigned local id 0, and have routing weight 0.

    Attributes:
        selected_experts_global: Global expert ids with shape ``[T, top_k]``.
        routing_weights_global: Router weights with shape ``[T, top_k]``.
        selected_experts_local: EP-local expert ids with shape ``[T, top_k]``.
        routing_weights_local: Weights with nonlocal routes zeroed.
        rank_mask: Boolean mask selecting routes owned by ``moe_ep_rank``.
        local_expert_start: Inclusive global expert id for this rank.
        local_expert_end: Exclusive global expert id for this rank.
        experts_per_rank: Number of global routed experts assigned to each rank.
        moe_ep_size: Expert-parallel world size.
        moe_ep_rank: Expert-parallel rank.
        num_global_experts: Total global routed expert count.
    """

    selected_experts_global: torch.Tensor
    routing_weights_global: torch.Tensor
    selected_experts_local: torch.Tensor
    routing_weights_local: torch.Tensor
    rank_mask: torch.Tensor
    local_expert_start: int
    local_expert_end: int
    experts_per_rank: int
    moe_ep_size: int
    moe_ep_rank: int
    num_global_experts: int


def is_deepseek_v4_hash_routed_layer(layer_idx: int) -> bool:
    """Return whether ``layer_idx`` uses the observed DeepSeek V4 hash router."""
    return layer_idx in DEEPSEEK_V4_HASH_ROUTED_LAYER_INDICES


def deepseek_v4_experts_per_rank(
    *,
    moe_ep_size: int = DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE,
    num_global_experts: int = DEEPSEEK_V4_NUM_ROUTED_EXPERTS,
) -> int:
    """Return the contiguous expert count per EP rank.

    The inspected DeepSeek V4 graph has 256 routed experts and uses 8-way EP in
    the MXFP4 path, so each rank owns 32 experts.
    """
    if moe_ep_size <= 0:
        raise ValueError(f"moe_ep_size must be positive, got {moe_ep_size}")
    if num_global_experts <= 0:
        raise ValueError(f"num_global_experts must be positive, got {num_global_experts}")
    if num_global_experts % moe_ep_size != 0:
        raise ValueError(
            "DeepSeek V4 EP-local routing requires an even expert split, got "
            f"{num_global_experts} experts across {moe_ep_size} ranks"
        )
    return num_global_experts // moe_ep_size


def deepseek_v4_ep_expert_range(
    moe_ep_rank: int,
    *,
    moe_ep_size: int = DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE,
    num_global_experts: int = DEEPSEEK_V4_NUM_ROUTED_EXPERTS,
) -> tuple[int, int]:
    """Return the global expert id interval owned by one EP rank."""
    if moe_ep_rank < 0 or moe_ep_rank >= moe_ep_size:
        raise ValueError(f"moe_ep_rank must be in [0, {moe_ep_size}), got {moe_ep_rank}")

    experts_per_rank = deepseek_v4_experts_per_rank(
        moe_ep_size=moe_ep_size,
        num_global_experts=num_global_experts,
    )
    local_expert_start = experts_per_rank * moe_ep_rank
    return local_expert_start, local_expert_start + experts_per_rank


def deepseek_v4_localize_expert_ids(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    moe_ep_size: int = DEEPSEEK_V4_OBSERVED_MOE_EP_SIZE,
    moe_ep_rank: int = 0,
    num_global_experts: int = DEEPSEEK_V4_NUM_ROUTED_EXPERTS,
) -> DeepSeekV4ExpertParallelRoutingMetadata:
    """Convert global router ids to rank-local ids and an explicit rank mask.

    Args:
        selected_experts: Global expert ids with shape ``[T, top_k]``.
        routing_weights: Router weights with shape ``[T, top_k]``.
        moe_ep_size: Expert-parallel world size.
        moe_ep_rank: Expert-parallel rank.
        num_global_experts: Total global routed expert count.

    Returns:
        Metadata containing global ids, local ids, rank mask, and masked weights.
    """
    if selected_experts.dim() != 2:
        raise ValueError(f"selected_experts must have rank 2, got rank {selected_experts.dim()}")
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "routing_weights must match selected_experts shape "
            f"{tuple(selected_experts.shape)}, got {tuple(routing_weights.shape)}"
        )
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"selected_experts must be int32 or int64, got {selected_experts.dtype}")
    if not routing_weights.is_floating_point():
        raise TypeError(f"routing_weights must be floating point, got {routing_weights.dtype}")

    local_expert_start, local_expert_end = deepseek_v4_ep_expert_range(
        moe_ep_rank,
        moe_ep_size=moe_ep_size,
        num_global_experts=num_global_experts,
    )
    experts_per_rank = local_expert_end - local_expert_start
    rank_mask = (selected_experts >= local_expert_start) & (selected_experts < local_expert_end)
    selected_experts_local = (selected_experts - local_expert_start) * rank_mask.to(
        selected_experts.dtype
    )
    routing_weights_local = routing_weights * rank_mask.to(routing_weights.dtype)

    return DeepSeekV4ExpertParallelRoutingMetadata(
        selected_experts_global=selected_experts,
        routing_weights_global=routing_weights,
        selected_experts_local=selected_experts_local,
        routing_weights_local=routing_weights_local,
        rank_mask=rank_mask,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
        experts_per_rank=experts_per_rank,
        moe_ep_size=moe_ep_size,
        moe_ep_rank=moe_ep_rank,
        num_global_experts=num_global_experts,
    )


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


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    device = tensors[0].device
    return device.type == "cuda" and all(tensor.device == device for tensor in tensors)


def _router_shape_supported(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
) -> bool:
    if hidden_states.dim() < 2 or router_weight.dim() != 2:
        return False
    num_experts, hidden_dim = router_weight.shape
    return (
        hidden_states.shape[-1] == hidden_dim
        and 0 < top_k <= num_experts
        and num_experts <= _TRITON_MAX_ROUTER_EXPERTS
    )


def _can_use_triton_router_base(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int,
) -> bool:
    return (
        _deepseek_v4_router_topk_kernel is not None
        and _router_shape_supported(hidden_states, router_weight, top_k)
        and _same_cuda_device(hidden_states, router_weight)
    )


def _router_logits(hidden_states: torch.Tensor, router_weight: torch.Tensor) -> torch.Tensor:
    hidden_dim = hidden_states.shape[-1]
    hidden_flat = hidden_states.reshape(-1, hidden_dim)
    return F.linear(hidden_flat.float(), router_weight.float()).contiguous()


def _deepseek_v4_router_hash_impl(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    tid2eid: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids is None:
        raise ValueError("input_ids is required for DeepSeek V4 hash-routed layers")
    if tid2eid is None:
        raise ValueError("tid2eid is required for DeepSeek V4 hash-routed layers")

    can_use_triton = (
        _can_use_triton_router_base(hidden_states, router_weight, top_k)
        and _same_cuda_device(hidden_states, router_weight, input_ids, tid2eid)
        and input_ids.numel() == _num_tokens(hidden_states)
        and tid2eid.dim() == 2
        and top_k <= tid2eid.shape[1]
        and tid2eid.dtype in (torch.int32, torch.int64)
    )
    if not can_use_triton:
        return deepseek_v4_router_reference(
            hidden_states,
            input_ids,
            router_weight,
            router_bias=None,
            tid2eid=tid2eid,
            top_k=top_k,
            route_scale=route_scale,
            is_hash_layer=True,
        )

    logits = _router_logits(hidden_states, router_weight)
    num_tokens, num_experts = logits.shape
    selected_experts = torch.empty(
        (num_tokens, top_k), dtype=tid2eid.dtype, device=hidden_states.device
    )
    routing_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
    input_ids_flat = input_ids.reshape(-1).contiguous()
    tid2eid_contiguous = tid2eid.contiguous()

    with torch.cuda.device(hidden_states.device):
        _deepseek_v4_router_hash_kernel[(num_tokens,)](
            logits,
            input_ids_flat,
            tid2eid_contiguous,
            selected_experts,
            routing_weights,
            num_experts,
            tid2eid_contiguous.shape[1],
            float(route_scale),
            BLOCK_K=_next_power_of_2(top_k),
            TOP_K=top_k,
            TINY=_FLOAT32_TINY,
        )
    return selected_experts, routing_weights


def _deepseek_v4_router_topk_impl(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    can_use_triton = _can_use_triton_router_base(hidden_states, router_weight, top_k) and (
        router_bias is None or _same_cuda_device(hidden_states, router_weight, router_bias)
    )
    if not can_use_triton:
        return deepseek_v4_router_reference(
            hidden_states,
            input_ids=None,
            router_weight=router_weight,
            router_bias=router_bias,
            tid2eid=None,
            top_k=top_k,
            route_scale=route_scale,
            is_hash_layer=False,
        )

    logits = _router_logits(hidden_states, router_weight)
    num_tokens, num_experts = logits.shape
    selected_experts = torch.empty(
        (num_tokens, top_k), dtype=torch.int64, device=hidden_states.device
    )
    routing_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
    bias = router_bias.contiguous() if router_bias is not None else logits

    with torch.cuda.device(hidden_states.device):
        _deepseek_v4_router_topk_kernel[(num_tokens,)](
            logits,
            bias,
            selected_experts,
            routing_weights,
            num_experts,
            float(route_scale),
            HAS_BIAS=router_bias is not None,
            BLOCK_E=_next_power_of_2(num_experts),
            BLOCK_K=_next_power_of_2(top_k),
            TOP_K=top_k,
            TINY=_FLOAT32_TINY,
        )
    return selected_experts, routing_weights


def _router_fake_outputs(
    hidden_states: torch.Tensor,
    top_k: int,
    selected_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = _num_tokens(hidden_states)
    selected_experts = hidden_states.new_empty(
        (num_tokens, top_k), dtype=selected_dtype, device=hidden_states.device
    )
    routing_weights = hidden_states.new_empty(
        (num_tokens, top_k), dtype=torch.float32, device=hidden_states.device
    )
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

    selected_dtype = tid2eid.dtype if is_hash_layer and tid2eid is not None else torch.int64
    return _router_fake_outputs(hidden_states, top_k, selected_dtype)


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_router_hash", mutates_args=())
def triton_deepseek_v4_router_hash(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    router_weight: torch.Tensor,
    tid2eid: torch.Tensor,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4 hash router using Triton when supported, reference otherwise."""
    return _deepseek_v4_router_hash_impl(
        hidden_states,
        input_ids,
        router_weight,
        tid2eid,
        top_k,
        route_scale,
    )


@triton_deepseek_v4_router_hash.register_fake
def _triton_deepseek_v4_router_hash_fake(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    router_weight: torch.Tensor,
    tid2eid: torch.Tensor,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input_ids, router_weight, route_scale
    return _router_fake_outputs(hidden_states, top_k, tid2eid.dtype)


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_router_topk", mutates_args=())
def triton_deepseek_v4_router_topk(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek V4 top-k router using Triton when supported, reference otherwise."""
    return _deepseek_v4_router_topk_impl(
        hidden_states,
        router_weight,
        router_bias,
        top_k,
        route_scale,
    )


@triton_deepseek_v4_router_topk.register_fake
def _triton_deepseek_v4_router_topk_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del router_weight, router_bias, route_scale
    return _router_fake_outputs(hidden_states, top_k, torch.int64)


@torch.library.custom_op("auto_deploy::triton_deepseek_v4_router", mutates_args=())
def triton_deepseek_v4_router(
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    router_weight: torch.Tensor,
    router_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    top_k: int,
    route_scale: float,
    is_hash_layer: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-op-compatible DeepSeek V4 router with guarded Triton paths."""
    if is_hash_layer:
        return _deepseek_v4_router_hash_impl(
            hidden_states,
            input_ids,
            router_weight,
            tid2eid,
            top_k,
            route_scale,
        )
    return _deepseek_v4_router_topk_impl(
        hidden_states,
        router_weight,
        router_bias,
        top_k,
        route_scale,
    )


@triton_deepseek_v4_router.register_fake
def _triton_deepseek_v4_router_fake(
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
    selected_dtype = tid2eid.dtype if is_hash_layer and tid2eid is not None else torch.int64
    return _router_fake_outputs(hidden_states, top_k, selected_dtype)
