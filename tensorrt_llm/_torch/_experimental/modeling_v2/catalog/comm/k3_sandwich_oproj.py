# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's post-attention sandwich: the row-parallel attention output projection, its TP all-reduce and the
residual update (attention-residual selection + RMSNorm) in one kernel, over a caller-owned
:class:`K3SandwichWorkspace`."""

from typing import Optional, Tuple

import torch

# Importing the op module registers trtllm::k3_sandwich_*; the state type is the one the ops' buffers come from.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich.op import K3SandwichWorkspace

__all__ = ["K3SandwichWorkspace", "k3_sandwich_oproj"]


def k3_sandwich_oproj(
    core: torch.Tensor,
    o_weight: torch.Tensor,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
    workspace: K3SandwichWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = prefix + allreduce(core @ o_weight^T)`` over ``workspace``'s TP group
    (the sum alone without ``prefix``), ``normed`` = RMSNorm(attn_res(block_residual..., updated)). Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""
    normed, updated = torch.ops.trtllm.k3_sandwich_oproj(
        core,
        o_weight,
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
    )
    return normed, updated
