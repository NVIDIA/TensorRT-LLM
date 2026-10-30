# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MNNVL all-gather of BF16 rows widened to FP32, via the trtllm glm_mnnvl_allgather op."""

from __future__ import annotations

from typing import Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._mnnvl_workspace import NUM_LAMPORT_BUFFERS


class GlmMnnvlAllgather(OpWrapper):
    """Every rank's BF16 rows gathered along the last dimension in rank order, as FP32.

    ``out[..., r * shard:(r + 1) * shard]`` is rank ``r``'s ``input`` widened to FP32, the same on
    every rank: the all-gather, concatenation and ``.float()`` of the vocabulary-parallel LM head's
    logits, in one kernel on a private ``_mnnvl_workspace.LamportExchangeWorkspace``. Each rank
    stores its shard once through the multicast address and copies the gathered rows out of its
    local buffer as they land. Bit-exact, except that a pair of BF16 values whose 32 bits are all
    set (NaNs no conversion produces) arrives with its lowest bit cleared, still NaNs.

    The kernel waits for its producer before it reads ``input`` and releases its PDL dependents
    right after that wait: a dependent reads the output only after its own grid dependency wait.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The c1 verify step at TP16: 8 rows of the LM head's vocabulary shard.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="verify-step logits: 8 rows of this rank's 9,680-entry vocabulary shard "
            "(154,880 over 16 ranks)",
            spec=dict(rows=8, shard=9680),
        ),
    )

    note = """
    Every rank calls the gather in the same order with the same shape. The gathered rows of one
    call must fit one Lamport buffer of the workspace (8 rows of the 154,880-entry vocabulary are
    2.5 MB of BF16); the op refuses more. The workspace is the gather's alone: the kernel empties
    the buffer of its previous call with its own threads.
    """

    def __call__(
        self,
        input: torch.Tensor,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.glm_mnnvl_allgather(input, comm_buffer, buffer_flags, group_size)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        comm_buffer: torch.Tensor | None = None,
        buffer_flags: torch.Tensor | None = None,
        group_size: int | None = None,
    ) -> torch.Tensor:
        """The rank-ordered concatenation along the last dimension, widened to FP32.

        A collective: ``per_rank_inputs`` holds every rank's ``input`` in rank order, in place of
        this rank's. The other arguments are ignored.
        """
        return torch.cat([rows.float() for rows in per_rank_inputs], dim=-1)

    def is_valid(
        self,
        input: torch.Tensor,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        group_size: int,
    ) -> None:
        # The all-reduce workspace marks empty words with -0.0 instead of 0xFFFFFFFF: on it the
        # gather takes stale words for rows that have not arrived.
        assert comm_buffer.dtype == torch.int32 and comm_buffer.dim() == 1, (
            "comm_buffer must be the comm_buffer of a LamportExchangeWorkspace"
        )
        # Flags of another workspace make the kernel poll and fill buffers whose state they do
        # not describe: stale words are gathered, or a buffer still in use is emptied.
        bytes_per_buffer = int(buffer_flags.view(torch.int32)[2].item())
        workspace_bytes = comm_buffer.numel() * comm_buffer.element_size()
        assert NUM_LAMPORT_BUFFERS * bytes_per_buffer == workspace_bytes, (
            "buffer_flags must be the flags of comm_buffer's workspace"
        )


mnnvl_allgather = GlmMnnvlAllgather()
