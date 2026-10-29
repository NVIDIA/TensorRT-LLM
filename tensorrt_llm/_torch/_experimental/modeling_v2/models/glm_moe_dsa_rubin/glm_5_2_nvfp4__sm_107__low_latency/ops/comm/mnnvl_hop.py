# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot MNNVL all-reduce with the residual add and RMSNorm, via the trtllm glm_mnnvl_hop op."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._mnnvl_workspace import NUM_LAMPORT_BUFFERS


class GlmMnnvlHop(OpWrapper):
    """All-reduce of BF16 rows over the ranks plus ``residual``, and the RMSNorm of that sum.

    ``residual_out = sum of input over the ranks + residual``, ``norm_out = RMSNorm(residual_out)``,
    in one kernel on the MNNVL all-reduce workspace (``_mnnvl_workspace.hop_workspace``): every rank
    stores its rows through the multicast address into its slot of the current Lamport buffer of
    every rank and polls all slots in one loop. The sum is in FP32 in rank order (from zero per
    chunk of 8 ranks, the chunks added in order) and rounded to BF16; the residual is added in
    BF16 and the RMSNorm normalizes that BF16 sum, ``(x * rsqrt(mean(x^2) + epsilon)) * gamma``.
    Every rank gets the same bits. Returns ``(norm_out, residual_out)``, both fresh.

    The kernel releases its PDL dependents right after its grid dependency wait (at its start when
    the shard is published): a dependent reads ``norm_out`` and ``residual_out`` only after its
    own grid dependency wait.

    With ``shard_published`` the GEMM that produced ``input`` (o_proj, through
    ``_mnnvl_workspace.publish_target``) has already stored this rank's rows into the workspace;
    ``input`` gives the shape only and is not read. That GEMM must be the hop's predecessor on the
    stream and release its dependents only after its own grid dependency wait.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The c1 decode step at TP16: 8 rows on a verify step and draft step 0, 1 row on draft steps
    # 1-6; every rank holds the full hidden size 6144. The world size is the test's.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="post-attention hop of the 78 verify layers and draft step 0: 8 rows, the shard "
            "published by o_proj",
            spec=dict(rows=8, hidden=6144, shard_published=True, zero_residual=False),
        ),
        Cell(
            why="post-MLP hop of the dense layers 0-2: 8 rows, the shard in input",
            spec=dict(rows=8, hidden=6144, shard_published=False, zero_residual=False),
        ),
        Cell(
            why="draft step 0's eh_proj hop: input_layernorm with a zero residual, the sum as "
            "residual_out",
            spec=dict(rows=8, hidden=6144, shard_published=False, zero_residual=True),
        ),
        Cell(
            why="post-attention hop of draft steps 1-6: one row, the shard published by o_proj",
            spec=dict(rows=1, hidden=6144, shard_published=True, zero_residual=False),
        ),
    )

    note = """
    Every rank of the group calls the hop in the same order, with the same shape and the same
    `shard_published`; a disagreement hangs the group. `_mnnvl_workspace.hop_supported` states the
    shapes the op takes (hidden 6144 on 16 ranks: 1 to 8 rows); it refuses others, which the target
    runs on the built-in `AllReduce`. A -0.0 among the summed elements arrives as +0.0, because
    -0.0 marks an empty Lamport slot.
    """

    def __call__(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        shard_published: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_out, residual_out = torch.ops.trtllm.glm_mnnvl_hop(
            input, residual, gamma, epsilon, comm_buffer, buffer_flags, shard_published
        )
        return norm_out, residual_out

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor | None = None,
        buffer_flags: torch.Tensor | None = None,
        shard_published: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm_out, residual_out)`` in FP32, each rounded once.

        A collective: ``per_rank_inputs`` holds every rank's ``input`` in rank order, in place of
        this rank's. ``residual`` is the calling rank's. ``comm_buffer``, ``buffer_flags`` and
        ``shard_published`` decide how the rows travel, not what is computed, and are ignored.
        """
        total = torch.zeros_like(residual, dtype=torch.float32)
        for rows in per_rank_inputs:
            total = total + rows.float()
        summed = total + residual.float()
        rms = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + epsilon)
        norm = summed * rms * gamma.float()
        return norm.to(residual.dtype), summed.to(residual.dtype)

    def is_valid(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
        shard_published: bool,
    ) -> None:
        # The kernel reads the slots as BF16 and takes -0.0 as the empty value: a workspace with
        # another empty value (the 0xFFFFFFFF words of an exchange workspace) hands it stale words
        # as rows.
        assert comm_buffer.dtype == torch.bfloat16, (
            "comm_buffer must be the BF16 view of the MNNVL all-reduce workspace (hop_workspace)"
        )
        # Flags of another workspace make the kernel poll, clear and rotate buffers whose state
        # they do not describe: stale slots are summed, or a buffer still in use is emptied.
        bytes_per_buffer = int(buffer_flags.view(torch.int32)[2].item())
        workspace_bytes = comm_buffer.numel() * comm_buffer.element_size()
        assert NUM_LAMPORT_BUFFERS * bytes_per_buffer == workspace_bytes, (
            "buffer_flags must be the flags of comm_buffer's workspace"
        )


mnnvl_hop = GlmMnnvlHop()
