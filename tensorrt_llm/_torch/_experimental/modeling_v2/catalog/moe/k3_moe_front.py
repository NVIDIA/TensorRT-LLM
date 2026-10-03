# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MoE front at decode size in one kernel: the sharded MoE head GEMV, its all-gather over a caller-owned
:class:`K3MoeHeadWorkspace`, the top-16 routing, the MXFP8 latent, and the shared experts' gate_up + SiTU."""

from typing import Tuple

import torch

# Importing front_op registers trtllm::k3_moe_front. front_weight packs the front's one weight at load time;
# weight_supported reads metadata only.
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.front_op import (
    front_weight,
    weight_supported,
)
from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.op import K3MoeHeadWorkspace

__all__ = ["K3MoeHeadWorkspace", "front_weight", "k3_moe_front", "weight_supported"]


def k3_moe_front(
    x: torch.Tensor,
    w_front: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    shared_cols: int,
    gate_cap: float,
    linear_cap: float,
    workspace: K3MoeHeadWorkspace,
    publish: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(topk_ids, topk_weights, quantized, scales, shared)`` for the MoE input ``x`` (bf16 ``[M <= 8, 7168]``,
    the same on every rank): the top-16 routing of the gathered router logits with ``e_score_correction_bias`` and the
    MXFP8 latent with its UE8M0 scales, as ``trtllm::k3_route_quant`` returns them for the gathered head, and the shared
    experts' activation (bf16 ``[M, shared_cols]``). ``w_front`` from :func:`front_weight`. Advances ``workspace`` by
    one call: every rank of the group makes the same front calls on it in the same order.

    ``publish``: also release the workspace's per-token ready words, which the next ``moe/k3_moe`` call on a head_flags
    state (``head=workspace``) acquires; every publishing call is followed by exactly one such call."""
    return torch.ops.trtllm.k3_moe_front(
        x,
        w_front,
        e_score_correction_bias,
        routed_scaling_factor,
        shared_cols,
        gate_cap,
        linear_cap,
        workspace.uc,
        workspace.mc,
        workspace.flags,
        workspace.rank,
        workspace.world_size,
        ag_ready=workspace.ready if publish else None,
    )
