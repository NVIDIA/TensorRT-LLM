# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL Lamport workspaces of the target's hops.

The hops (``mnnvl_hop``, ``mnnvl_moe_finalize_hop``) run on the MNNVL all-reduce workspace of the
engine's ``AllReduce`` (``get_or_scale_allreduce_mnnvl_workspace``): three Lamport buffers whose
empty value is -0.0, and nine uint32 flags (current buffer, dirty buffer, bytes per buffer, dirty
stages, bytes to clear per stage, arrival counter). The hops read, clear and rotate them as the
built-in MNNVL all-reduce does, so both can follow each other on one workspace.

The workspace is replaced by a larger one when an all-reduce needs more than it holds, and a CUDA
graph keeps the workspace it launches on alive only when it was fetched during that graph's
capture. So the target fetches it for every hop it issues, eagerly and under capture, and never
keeps it across forwards.
"""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
from tensorrt_llm.mapping import Mapping

# Lamport buffers of an MNNVL all-reduce workspace.
NUM_LAMPORT_BUFFERS = 3


def hop_buffer_bytes(num_tokens: int, hidden_size: int, tp_size: int) -> int:
    """Bytes of one Lamport buffer that a hop of ``[num_tokens, hidden_size]`` BF16 rows uses."""
    return num_tokens * hidden_size * tp_size * torch.bfloat16.itemsize


def hop_supported(num_tokens: int, hidden_size: int, tp_size: int) -> bool:
    """Whether the hops serve ``[num_tokens, hidden_size]`` BF16 rows over ``tp_size`` ranks.

    Asked of the current device: 2, 4, 8 or 16 ranks; one-shot (up to 1 MiB of Lamport data per
    rank, 1.5 MiB on 16 ranks); one row per cluster of 2 to 8 CTAs of at most 128 threads, eight
    elements per thread, covering the row exactly. Hidden 6144 on 16 ranks: 1 to 8 rows. Another
    shape runs on the built-in ``AllReduce``.
    """
    return bool(torch.ops.trtllm.glm_mnnvl_hop_supported(num_tokens, hidden_size, tp_size))


def _all_reduce_workspace(mapping: Mapping, num_tokens: int, hidden_size: int) -> dict:
    return get_or_scale_allreduce_mnnvl_workspace(
        mapping,
        torch.bfloat16,
        buffer_size_bytes=hop_buffer_bytes(num_tokens, hidden_size, mapping.tp_size),
    )


def hop_workspace(
    mapping: Mapping, num_tokens: int, hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The workspace operands of a hop of ``[num_tokens, hidden_size]`` BF16 rows.

    Returns ``(comm_buffer, buffer_flags)``: the ``[3, N]`` BF16 view of this rank's three Lamport
    buffers of the MNNVL all-reduce workspace of ``mapping``, and the workspace's nine uint32
    flags. The workspace grows first when its buffers hold less than the hop needs; it must not
    grow while a CUDA graph is captured, so every shape runs once eagerly before its capture.
    """
    workspace = _all_reduce_workspace(mapping, num_tokens, hidden_size)
    comm_buffer = workspace["uc_buffer"].view(torch.bfloat16).view(NUM_LAMPORT_BUFFERS, -1)
    return comm_buffer, workspace["buffer_flags"]


def publish_target(
    mapping: Mapping, num_tokens: int, hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Where the GEMM that produces a hop's input stores this rank's rows itself.

    Returns ``(lamport, flags)``: the multicast BF16 view (1-D) of the three Lamport buffers of the
    workspace ``hop_workspace`` returns for the same shape, and its flags viewed as int32. The GEMM
    (the o_proj Lamport publish) reads the current buffer index and the bytes per buffer from the
    flags after its grid dependency wait, releases its dependents right after that wait, and stores
    its BF16 output, -0.0 written as +0.0, in whole 16-byte vectors at ``[token][tp_rank][column]``
    of the current buffer. The hop that follows it on the stream then runs with
    ``shard_published=True`` on that workspace.
    """
    workspace = _all_reduce_workspace(mapping, num_tokens, hidden_size)
    elements = NUM_LAMPORT_BUFFERS * workspace["buffer_size_bytes"] // torch.bfloat16.itemsize
    lamport = workspace["handle"].get_mc_buffer((elements,), torch.bfloat16, 0)
    return lamport, workspace["buffer_flags"].view(torch.int32)
