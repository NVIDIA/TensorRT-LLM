# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A row-parallel projection, its TP all-reduce, the residual add and an RMSNorm in one kernel (Kimi K3's drafter
layers: the attention output projection, or with ``swiglu`` the MLP's SiLU-and-mul and down projection), over a
caller-owned :class:`K3SandwichWorkspace`."""

from typing import Tuple

import torch

# Importing the op module registers trtllm::k3_sandwich_*; the state type is the one the ops' buffers come from.
from tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich.op import K3SandwichWorkspace

__all__ = ["K3SandwichWorkspace", "k3_sandwich_plain"]


def k3_sandwich_plain(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    workspace: K3SandwichWorkspace,
    swiglu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(normed, updated)``: ``updated = residual + allreduce(x @ weight^T)`` over ``workspace``'s TP group
    (with ``swiglu``, of ``silu_and_mul(x) @ weight^T``), ``normed = RMSNorm(updated) * norm_weight``. Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""
    normed, updated = torch.ops.trtllm.k3_sandwich_plain(
        x,
        weight,
        residual,
        norm_weight,
        eps,
        workspace.uc,
        workspace.mc,
        workspace.flags,
        workspace.rank,
        swiglu=swiglu,
    )
    return normed, updated
