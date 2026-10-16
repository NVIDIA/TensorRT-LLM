# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL all-reduce over a caller-owned :class:`MnnvlWorkspace`: the sum over the TP group, or with a residual the
sum + residual add + RMSNorm; one-shot up to ``one_shot_max_bytes``, two-shot above."""

from typing import Optional, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm.functional import AllReduceFusionOp

from .mnnvl_workspace import MnnvlWorkspace

__all__ = ["MnnvlWorkspace", "mnnvl_fusion_allreduce", "required_buffer_bytes"]


def required_buffer_bytes(
    num_tokens: int, hidden: int, world_size: int, dtype: torch.dtype, one_shot_max_bytes: int
) -> int:
    """Bytes of one Lamport buffer a call of ``num_tokens`` rows of ``hidden`` needs: ``num_tokens * hidden * world *
    element size`` when that is at most ``one_shot_max_bytes`` (one-shot), else two stages of ``num_tokens`` rounded up
    to a multiple of ``world`` (two-shot)."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    one_shot = num_tokens * hidden * world_size * itemsize
    if one_shot <= one_shot_max_bytes:
        return one_shot
    return 2 * -(-num_tokens // world_size) * world_size * hidden * itemsize


def mnnvl_fusion_allreduce(
    input: torch.Tensor,
    workspace: MnnvlWorkspace,
    one_shot_max_bytes: int,
    residual: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """The sum of ``input`` over ``workspace``'s TP group (a new tensor of ``input``'s shape), or with ``residual``,
    ``norm_weight`` and ``eps`` the pair ``(RMSNorm(sum + residual) * norm_weight, sum + residual)``. Sent one-shot
    when ``num_tokens * hidden * world * element size <= one_shot_max_bytes``, else two-shot; the workspace's buffers
    must hold the call (:func:`required_buffer_bytes`). Advances ``workspace`` by one call: every rank of the group
    makes the same MNNVL calls on it in the same order."""
    fused = residual is not None
    if fused != (norm_weight is not None) or fused != (eps is not None):
        raise ValueError("mnnvl_fusion_allreduce: residual, norm_weight and eps go together")
    hidden = input.shape[-1]
    num_tokens = input.numel() // hidden
    need = required_buffer_bytes(
        num_tokens, hidden, workspace.world_size, input.dtype, one_shot_max_bytes
    )
    if need > workspace.buffer_bytes:
        raise ValueError(
            f"mnnvl_fusion_allreduce: the call needs {need} bytes per Lamport buffer, the workspace has "
            f"{workspace.buffer_bytes}"
        )
    two_shot = (
        num_tokens * hidden * workspace.world_size * input.element_size() > one_shot_max_bytes
    )
    if two_shot and workspace.buffer_bytes % 32:
        # The two-shot broadcast stage starts at buffer_bytes / 2 and is accessed in 16-byte vectors.
        raise ValueError(
            "mnnvl_fusion_allreduce: a two-shot call needs the workspace's buffer_bytes to be a multiple of 32, "
            f"not {workspace.buffer_bytes}"
        )
    fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM if fused else AllReduceFusionOp.NONE
    outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
        input,
        norm_weight,
        residual,
        eps,
        workspace.comm_buffer(input.dtype),
        workspace.buffer_flags,
        fused,
        None,
        int(fusion_op),
        one_shot_max_bytes,
        True,
    )
    return (outputs[0], outputs[1]) if fused else outputs[0]
