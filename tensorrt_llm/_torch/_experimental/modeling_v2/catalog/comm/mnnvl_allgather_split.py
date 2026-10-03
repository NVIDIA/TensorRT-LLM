# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot all-gather over a caller-owned :class:`MnnvlWorkspace` of fp32 rows whose leading columns travel, and are
gathered, as bf16 (Kimi K3's sharded MoE head: the latent columns in bf16, the router logits in fp32)."""

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .mnnvl_workspace import MnnvlWorkspace

__all__ = ["MnnvlWorkspace", "mnnvl_allgather_split", "required_buffer_bytes"]


def required_buffer_bytes(
    num_tokens: int, bf16_columns: int, fp32_columns: int, world_size: int
) -> int:
    """Bytes of one Lamport buffer a call occupies: every rank's rows, the bf16 columns at 2 bytes and the fp32 columns
    at 4."""
    return num_tokens * world_size * (bf16_columns * 2 + fp32_columns * 4)


def mnnvl_allgather_split(
    input: torch.Tensor,
    bf16_columns: int,
    workspace: MnnvlWorkspace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather ``input`` (fp32 ``[num_tokens, columns]``, this rank's slice) from every rank of ``workspace``'s TP group.
    Returns ``(bf16_out, fp32_out)``: ``bf16_out[t, r * bf16_columns + j] = bf16(input_r[t, j])`` and ``fp32_out[t, r
    * fp32_columns + j] = input_r[t, bf16_columns + j]``, ``fp32_columns = columns - bf16_columns``, in rank order.
    Takes one turn of the workspace's Lamport rotation, as an all-reduce on it does: every rank of the group makes the
    same MNNVL calls on it in the same order."""
    num_tokens, columns = input.shape
    need = required_buffer_bytes(
        num_tokens, bf16_columns, columns - bf16_columns, workspace.world_size
    )
    if need > workspace.buffer_bytes:
        raise ValueError(
            f"mnnvl_allgather_split: the call needs {need} bytes per Lamport buffer, the workspace has "
            f"{workspace.buffer_bytes}"
        )
    bf16_out, fp32_out = torch.ops.trtllm.mnnvl_allgather_split(
        input,
        bf16_columns,
        workspace.comm_buffer(torch.bfloat16),
        workspace.buffer_flags,
    )
    return bf16_out, fp32_out
