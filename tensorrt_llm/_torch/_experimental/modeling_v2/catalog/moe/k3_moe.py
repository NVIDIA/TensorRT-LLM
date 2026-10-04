# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts at decode size: ``trtllm::k3_moe``, this rank's routed partial from the persistent CuTe
DSL kernel (FC1 + SiTU + FC2 with the routing-weighted combine over the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in
place), on caller-owned state: a :class:`K3MoeState` (up to 8 tokens) or :class:`K3MoeWideState` (up to 64) and one
:class:`K3MoeLayer` per MoE layer. Its inputs are the outputs of ``moe/k3_route_quant`` or ``moe/k3_moe_front``. The
push form stores the partial into every rank's ``K3LatentExchange`` for ``comm/k3_latent_reduce`` instead."""

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.latent_op import K3LatentExchange

# The state types (they launch nothing per call); importing the op module registers trtllm::k3_moe. is_supported reads
# metadata only.
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import (
    K3MoeHeadWorkspace,
    K3MoeLayer,
    K3MoeState,
    K3MoeWideState,
    is_supported,
)

__all__ = [
    "K3MoeHeadWorkspace",
    "K3MoeLayer",
    "K3MoeState",
    "K3MoeWideState",
    "is_supported",
    "k3_moe",
    "k3_moe_push",
]


def k3_moe(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeLayer,
    head: Optional[K3MoeHeadWorkspace] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """This rank's routed partial bf16 ``[M, 3584]`` for the routing and MXFP8 latent of ``k3_route_quant`` or
    ``k3_moe_front`` (``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its UE8M0 scales ``[M, 112]``, ``topk_ids``
    int32 and ``topk_weights`` bf16 ``[M, 16]``), over ``layer``'s experts (global ids
    ``[local_expert_offset, local_expert_offset + num_local)``); M <= 8 on a K3MoeState, <= 64 on a K3MoeWideState.

    ``head``: the TP group's K3MoeHeadWorkspace, for and only for a ``head_flags`` state's layers: the call then
    acquires the front's outputs through the workspace's ready words and advances its epoch, so the front call before
    it must have published them (``moe/k3_moe_front`` with ``publish=True``). ``out``: bf16, contiguous, at
    least ``[M, 3584]``; the call writes its first M rows and returns an empty ``[0, 3584]`` tensor. Writes the state's
    slab (left armed) and partial rows, and the layer's counters (left zero).

    Two calls on one layer, in either form, must not run back to back: a call claims its first tile from the layer's
    counters before its grid-dependency wait. Between the two, some kernel must wait for its predecessor before it
    triggers its dependents (``griddepcontrol.wait``, then ``launch_dependents``), or the stream is synchronized."""
    state = layer.state
    if (head is not None) != state.head_flags:
        raise ValueError(
            "k3_moe: head is given for, and only for, the layers of a head_flags K3MoeState"
        )
    return torch.ops.trtllm.k3_moe(
        x_fp8, x_sf, topk_ids, topk_weights, *layer.weights, state.c, state.cs, state.part, layer.counters,
        local_expert_offset, state.num_local, state.num_ctas, state.m_max, state.use_pdl,
        None if head is None else head.ready, None if head is None else head.flags, out,
    )  # fmt: skip


def k3_moe_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeLayer,
    exchange: K3LatentExchange,
    slot: Optional[int] = None,
    head: Optional[K3MoeHeadWorkspace] = None,
) -> None:
    """The push form of :func:`k3_moe` for ``M <= 8`` on a K3MoeState: the same partial, stored into slot ``slot``
    (default ``exchange.rank``) of every rank's ``exchange`` (a TP group's ``K3LatentExchange``) instead of returned.
    One ``comm/k3_latent_reduce`` of the M tokens on that exchange must follow before the next push, on every rank in
    the same order. ``head`` as in :func:`k3_moe`. Writes the state's slab (left armed) and partial rows, the layer's
    counters (left zero), and every rank's exchange. As for :func:`k3_moe`, two calls on one layer must not run back to
    back."""
    state = layer.state
    if (head is not None) != state.head_flags:
        raise ValueError(
            "k3_moe_push: head is given for, and only for, the layers of a head_flags K3MoeState"
        )
    torch.ops.trtllm.k3_moe(
        x_fp8, x_sf, topk_ids, topk_weights, *layer.weights, state.c, state.cs, state.part, layer.counters,
        local_expert_offset, state.num_local, state.num_ctas, state.m_max, state.use_pdl,
        None if head is None else head.ready, None if head is None else head.flags, None,
        exchange.uc, exchange.mc, exchange.flags, exchange.rank if slot is None else slot,
    )  # fmt: skip
