# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher entry for the ULL route-tail operation."""

import torch

from .k3_route_tail import TILE, TOP_K, sort_buffer_sizes
from .k3_route_tail import k3_route_tail as _route_tail


@torch.library.custom_op("trtllm::k3_route_tail", mutates_args=())
def k3_route_tail(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int = TILE,
    num_tokens: int = -1,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _route_tail(
        logits,
        e_score_correction_bias,
        routed_scaling_factor,
        local_expert_offset,
        local_num_experts,
        tile_tokens_dim,
        num_tokens,
    )


@k3_route_tail.register_fake
def _(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int = TILE,
    num_tokens: int = -1,
):
    rows = logits.shape[0] if logits.dim() == 2 else logits.shape[1]
    m = rows if num_tokens < 0 else num_tokens
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num_experts, tile_tokens_dim)

    def i32(*shape):
        return logits.new_empty(shape, dtype=torch.int32)

    return (
        i32(m, TOP_K),
        logits.new_empty((m, TOP_K), dtype=torch.float32),
        i32(max_tiles),
        i32(max_tiles),
        i32(m, TOP_K),
        i32(max_permuted),
        i32(1),
        i32(1),
    )
