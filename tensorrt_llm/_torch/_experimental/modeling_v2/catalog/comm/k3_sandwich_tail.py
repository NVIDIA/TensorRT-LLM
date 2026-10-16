# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's pre-attention sandwich: the row-parallel MoE tail (the latent up projection of the normed latent slice and
the shared experts' down projection), its TP all-reduce and the next layer's residual update (attention-residual
selection + RMSNorm) in one kernel, over a caller-owned :class:`K3SandwichWorkspace`."""

from typing import Optional, Tuple

import torch

# Importing the op module registers trtllm::k3_sandwich_*; the state type is the one the ops' buffers come from.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich.op import K3SandwichWorkspace

__all__ = ["K3SandwichWorkspace", "k3_sandwich_tail"]


def k3_sandwich_tail(
    latent: torch.Tensor,
    act: torch.Tensor,
    tail_weight: torch.Tensor,
    lo: int,
    lat_eps: float,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: K3SandwichWorkspace,
    tap: Optional[torch.Tensor] = None,
    tap_updated: bool = False,
    updated_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = prefix + allreduce([rmsnorm(latent)[:, lo:lo+224] | act] @
    tail_weight^T)`` over ``workspace``'s TP group (the sum alone without ``prefix``), ``normed`` =
    RMSNorm(attn_res(block_residual..., updated)). ``tap``: also store the pre-norm attention-residual mixture there
    (``updated`` with ``tap_updated``). ``updated_out``: store ``updated`` there; it is then the returned ``updated``.
    Advances ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""
    normed, updated = torch.ops.trtllm.k3_sandwich_tail(
        latent,
        act,
        tail_weight,
        lo,
        lat_eps,
        prefix,
        block_residual,
        res_weight,
        rms_weight,
        output_rms_weight,
        rms_eps,
        output_rms_eps,
        workspace.uc,
        workspace.mc,
        workspace.flags,
        workspace.rank,
        tap=tap,
        tap_updated=tap_updated,
        updated_out=updated_out,
    )
    return normed, (updated if updated_out is None else updated_out)
