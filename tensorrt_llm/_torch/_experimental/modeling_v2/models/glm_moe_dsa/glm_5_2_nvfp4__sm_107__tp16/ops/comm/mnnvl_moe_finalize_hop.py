# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE finalize fused into the post-MoE MNNVL hop, via trtllm glm_mnnvl_moe_finalize_hop."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._mnnvl_workspace import NUM_LAMPORT_BUFFERS


class GlmMnnvlMoeFinalizeHop(OpWrapper):
    """``mnnvl_hop`` of the finalized MoE rows: the routed sum plus ``addend``, all-reduced.

    On each rank, row ``t`` of the finalized rows is ``addend[t] + sum over k of
    expert_weights[t, k] * gemm2_output[expanded_idx_to_permuted_idx[t, k]]``, over the slots whose
    index is not -1: the routed sum in FP32 with an FMA in ``k`` order, rounded to BF16, then the
    addend added in FP32 and rounded once more, as the TRTLLM-Gen finalize with the shared
    experts' output as addend computes it. The rows are then all-reduced with the residual add and
    RMSNorm exactly as ``mnnvl_hop`` does (without a published shard), on the same workspace.
    Returns ``(norm_out, residual_out)``, both fresh, the same bits on every rank.

    The kernel reads ``expert_weights`` and ``expanded_idx_to_permuted_idx`` before its grid
    dependency wait, so the kernel that wrote them must have completed when this one launches: the
    MoE's routing kernel, when this hop follows FC2 on the stream and FC2 releases its dependents
    only after its own grid dependency wait; or a constant written outside graph capture. The
    other operands are read after the wait. The PDL dependents launch right after that wait and
    read the outputs only after their own.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The c1 decode step at TP16, hidden 6144, every expert local (moe_ep 1).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="post-MoE hop of the 75 MoE layers of a verify step: 8 rows, top-8 FC2 rows, the "
            "shared experts' output as addend",
            spec=dict(rows=8, hidden=6144, routing="top8"),
        ),
        Cell(
            why="post-MoE hop of an MTP draft step: one row, its routed row in slot 0 with "
            "weight 1 and slots 1-7 empty (-1), the shared experts' output as addend",
            spec=dict(rows=1, hidden=6144, routing="routed_row"),
        ),
    )

    note = """
    The shapes are those of `mnnvl_hop` (`_mnnvl_workspace.hop_supported`), with exactly 8 routing
    slots per token; the same group discipline applies. FC2 rows may be wider than the hidden size
    (`gemm2_output.shape[1]` is the row pitch); the first `hidden` columns are read.
    """

    def __call__(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        addend: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_out, residual_out = torch.ops.trtllm.glm_mnnvl_moe_finalize_hop(
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            addend,
            residual,
            gamma,
            epsilon,
            comm_buffer,
            buffer_flags,
        )
        return norm_out, residual_out

    def reference(
        self,
        per_rank_inputs: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor | None = None,
        buffer_flags: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm_out, residual_out)`` in FP32, each rounded once.

        A collective: ``per_rank_inputs`` holds every rank's ``(gemm2_output, expert_weights,
        expanded_idx_to_permuted_idx, addend)`` in rank order, in place of this rank's.
        ``residual`` is the calling rank's. ``comm_buffer`` and ``buffer_flags`` are ignored.
        """
        hidden = residual.shape[-1]
        total = torch.zeros_like(residual, dtype=torch.float32)
        for gemm2_output, expert_weights, permuted_idx, addend in per_rank_inputs:
            taken = permuted_idx >= 0
            rows = gemm2_output[permuted_idx.clamp(min=0).long(), :hidden].float()
            weights = torch.where(taken, expert_weights.float(), 0.0)
            total = total + (weights.unsqueeze(-1) * rows).sum(dim=1) + addend.float()
        summed = total + residual.float()
        rms = torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + epsilon)
        norm = summed * rms * gamma.float()
        return norm.to(residual.dtype), summed.to(residual.dtype)

    def is_valid(
        self,
        gemm2_output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        addend: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float,
        comm_buffer: torch.Tensor,
        buffer_flags: torch.Tensor,
    ) -> None:
        # An index outside [-1, rows) reads memory past the FC2 rows as an expert row.
        index = expanded_idx_to_permuted_idx
        in_range = (index >= -1) & (index < gemm2_output.shape[0])
        assert bool(in_range.all()), (
            "expanded_idx_to_permuted_idx must index gemm2_output's rows, or be -1"
        )
        # As for mnnvl_hop: a workspace with another empty value, or flags of another workspace,
        # make the kernel sum stale slots or empty a buffer still in use.
        assert comm_buffer.dtype == torch.bfloat16, (
            "comm_buffer must be the BF16 view of the MNNVL all-reduce workspace (hop_workspace)"
        )
        bytes_per_buffer = int(buffer_flags.view(torch.int32)[2].item())
        workspace_bytes = comm_buffer.numel() * comm_buffer.element_size()
        assert NUM_LAMPORT_BUFFERS * bytes_per_buffer == workspace_bytes, (
            "buffer_flags must be the flags of comm_buffer's workspace"
        )


mnnvl_moe_finalize_hop = GlmMnnvlMoeFinalizeHop()
