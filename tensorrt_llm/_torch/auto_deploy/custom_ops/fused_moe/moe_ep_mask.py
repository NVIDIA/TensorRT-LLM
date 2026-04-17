# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Fused EP-sharding mask for MoE top-k routing output.

Replaces the 5 elementwise aten ops the sharding transform inserts between
`noaux_tc_op` (or topk fallback) and the fused MoE op:

    sub       = selected_experts - (ep_rank * experts_per_rank)
    floordiv  = selected_experts // experts_per_rank
    rank_mask = floordiv == ep_rank      # or `>=` for the last rank
    bool_cast = rank_mask.to(weights.dtype)
    masked    = top_k_weights * bool_cast

with a single Triton kernel that writes both `local_indices` and
`masked_weights` in one pass. All inputs/outputs are small `[M, top_k]`
tensors, so the win is eliminating per-op launch / host-dispatch cost, not
arithmetic work.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _moe_ep_mask_kernel(
    indices_ptr,  # [M, TOP_K] int
    weights_ptr,  # [M, TOP_K] bf16 or fp16 or fp32
    local_indices_ptr,  # [M, TOP_K] int
    masked_weights_ptr,  # [M, TOP_K] bf16 or fp16 or fp32
    M,
    ep_rank,
    experts_per_rank,
    lower_bound,  # ep_rank * experts_per_rank (precomputed)
    TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_LAST_RANK: tl.constexpr,
):
    """One program covers BLOCK_M rows; each row has TOP_K entries.

    For each (row, k):
        local_idx = idx - lower_bound
        group = idx // experts_per_rank
        owned = (group >= ep_rank) if IS_LAST_RANK else (group == ep_rank)
        masked = weight if owned else 0
    """
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    offs_k = tl.arange(0, TOP_K)
    # 2D indexing: [BLOCK_M, TOP_K]
    ptrs = offs_m[:, None] * TOP_K + offs_k[None, :]
    load_mask = m_mask[:, None]

    idx = tl.load(indices_ptr + ptrs, mask=load_mask, other=0)
    w = tl.load(weights_ptr + ptrs, mask=load_mask, other=0.0)

    local_idx = idx - lower_bound
    group = idx // experts_per_rank
    if IS_LAST_RANK:
        owned = group >= ep_rank
    else:
        owned = group == ep_rank

    # Mask weights by multiplying with the bool (promoted through where to keep dtype).
    masked = tl.where(owned, w, tl.zeros_like(w))

    tl.store(local_indices_ptr + ptrs, local_idx, mask=load_mask)
    tl.store(masked_weights_ptr + ptrs, masked, mask=load_mask)


def _moe_ep_mask_impl(
    selected_experts: Tensor,
    top_k_weights: Tensor,
    ep_rank: int,
    ep_size: int,
    experts_per_rank: int,
) -> Tuple[Tensor, Tensor]:
    assert selected_experts.shape == top_k_weights.shape
    assert selected_experts.is_contiguous() and top_k_weights.is_contiguous()

    shape = selected_experts.shape
    top_k = shape[-1]
    M = selected_experts.numel() // top_k

    # Always allocate contiguous outputs (memory_format=contiguous_format explicitly)
    # so downstream consumers (e.g. fp8_block_scale_moe_runner which requires int32
    # contiguous selected_experts and contiguous weights) can skip defensive
    # .contiguous() calls. This is a load-bearing guarantee — do not relax it.
    local_indices = torch.empty(
        shape,
        dtype=selected_experts.dtype,
        device=selected_experts.device,
        memory_format=torch.contiguous_format,
    )
    masked_weights = torch.empty(
        shape,
        dtype=top_k_weights.dtype,
        device=top_k_weights.device,
        memory_format=torch.contiguous_format,
    )

    BLOCK_M = 128 if M >= 128 else triton.next_power_of_2(max(1, M))
    grid = (triton.cdiv(M, BLOCK_M),)
    _moe_ep_mask_kernel[grid](
        selected_experts,
        top_k_weights,
        local_indices,
        masked_weights,
        M,
        ep_rank,
        experts_per_rank,
        ep_rank * experts_per_rank,
        TOP_K=top_k,
        BLOCK_M=BLOCK_M,
        IS_LAST_RANK=(ep_rank == ep_size - 1),
    )
    return local_indices, masked_weights


@torch.library.custom_op("auto_deploy::moe_ep_mask", mutates_args=())
def moe_ep_mask(
    selected_experts: Tensor,
    top_k_weights: Tensor,
    ep_rank: int,
    ep_size: int,
    experts_per_rank: int,
) -> Tuple[Tensor, Tensor]:
    """Fused EP-sharding mask for top-k routing outputs.

    Args:
        selected_experts: [..., top_k] integer expert indices in GLOBAL coords.
        top_k_weights:    [..., top_k] per-token routing weights.
        ep_rank:          Expert-parallel rank (0..ep_size-1).
        ep_size:          Expert-parallel size.
        experts_per_rank: Number of experts owned by each rank.

    Returns:
        local_indices:   [..., top_k] indices in LOCAL coords (global - lower_bound).
        masked_weights:  [..., top_k] weights with remote-rank entries zeroed.
    """
    return _moe_ep_mask_impl(selected_experts, top_k_weights, ep_rank, ep_size, experts_per_rank)


@moe_ep_mask.register_fake
def _moe_ep_mask_fake(
    selected_experts: Tensor,
    top_k_weights: Tensor,
    ep_rank: int,
    ep_size: int,
    experts_per_rank: int,
) -> Tuple[Tensor, Tensor]:
    return torch.empty_like(selected_experts), torch.empty_like(top_k_weights)
