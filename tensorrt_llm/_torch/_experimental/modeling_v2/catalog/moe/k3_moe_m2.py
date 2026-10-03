# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts of two decode tokens with every expert on the rank (moe TP16 x EP1, intermediate <= 256):
one weight-stream CuTe DSL kernel over a caller-owned :class:`K3MoeM2State`, returning this rank's routed partial or
pushing it into a latent exchange for ``trtllm::k3_latent_reduce``."""

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import K3MoeM2Layer, K3MoeM2State

__all__ = ["K3MoeM2Layer", "K3MoeM2State", "k3_moe_m2", "k3_moe_m2_push"]


def k3_moe_m2(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM2Layer,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial ``[2, 3584]`` bf16 of ``layer``'s experts for two tokens. Advances the state's
    workspace by one call: the calls on one state run in one stream order."""
    return layer(x_fp8, x_sf, topk_ids, topk_weights, local_expert_offset, out)


def k3_moe_m2_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM2Layer,
    exchange,
) -> None:
    """The same partials stored into this rank's slot of both tokens' rows of every rank's ``exchange`` (a TP
    group's ``K3LatentExchange``) instead of returned; one ``trtllm::k3_latent_reduce`` of the two tokens on that
    exchange must follow before the next push. Advances the state's workspace by one call."""
    layer.push(
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        local_expert_offset,
        exchange.mc,
        exchange.flags,
        exchange.rank,
    )
