# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multicast all-reduce + residual + RMSNorm (+ MXFP8 copy) of the decode layer boundary, 4 ranks."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper as OpWrapper

from ....ops.comm.reference import _E4M3_MAX as _E4M3_MAX
from ....ops.comm.reference import McAllreduceNormReference
from ....ops.comm.reference import boundary as boundary
from ....ops.comm.reference import mxfp8_rows as mxfp8_rows
from ....ops.comm.reference import sf_written as sf_written
from ....ops.comm.reference import swizzle_sf as swizzle_sf


class M3McAllreduceNorm(McAllreduceNormReference):
    """One TP group's boundary: ``updated = bf16(residual + sum_r bf16(x_r + x2_r))``, ``normed = RMSNorm(updated)``.

    Every rank pushes its partial through the multicast mapping ``ws_mc_ptr`` into every rank's exchange, then polls
    its own view ``ws_uc`` (re-arming the words it read) and rotates the per-CTA call parities in ``flags``; the
    workspace is the ``McAllReduceWorkspace`` bucket of ``maxt`` tokens. All ranks must make the same calls in the
    same stream order. ``prepushed`` 2 means a producer already pushed ``x`` (and launched this grid after its
    pushes); 3 that two producers pushed ``x`` into exchange set A and ``x2`` into set B. Returns
    ``[normed, updated]`` or, with ``quant``, also the MXFP8 copy of ``normed`` (E4M3 rows, swizzled UE8M0).
    """

    ARCHS = frozenset({Arch.SM_107})

    # The tp4 decode boundaries (hidden 6144, 4 ranks, eps 1e-6). Buckets 8 / 16 / 48 are separate workspaces; the
    # 8-token bucket runs epilogue option 7 at 1, 2, 7, 8 tokens and 15 at 3..6, the others 15.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode token at the attention boundary, x only, bucket 8 option 7",
            spec=dict(m=1, maxt=8, opts=7, x2=False, quant=False, prepushed=0),
        ),
        Cell(
            why="MoE boundary at 4 tokens: shared + routed summed in-kernel, MXFP8 copy for the next QKV, option 15",
            spec=dict(m=4, maxt=8, opts=15, x2=True, quant=True, prepushed=0),
        ),
        Cell(
            why="a full 8-token bucket with the second addend",
            spec=dict(m=8, maxt=8, opts=7, x2=True, quant=False, prepushed=0),
        ),
        Cell(
            why="attention boundary whose rows the o_proj GEMV already pushed (prepushed 2)",
            spec=dict(m=8, maxt=8, opts=7, x2=False, quant=False, prepushed=2),
        ),
        Cell(
            why="routed engine pushed set A, shared down projection set B (prepushed 3), with the MXFP8 copy",
            spec=dict(m=6, maxt=8, opts=15, x2=True, quant=True, prepushed=3),
        ),
        Cell(
            why="12 tokens pushed by the o_proj GEMV into the 16-token bucket",
            spec=dict(m=12, maxt=16, opts=15, x2=False, quant=False, prepushed=2),
        ),
        Cell(
            why="the 9..13-token MoE kernel's sum into the 16-token bucket, MXFP8 copy",
            spec=dict(m=16, maxt=16, opts=15, x2=False, quant=True, prepushed=0),
        ),
        Cell(
            why="the 48-token bucket at a partial fill",
            spec=dict(m=30, maxt=48, opts=15, x2=False, quant=False, prepushed=0),
        ),
        Cell(
            why="the 48-token bucket full, second addend and MXFP8 copy",
            spec=dict(m=48, maxt=48, opts=15, x2=True, quant=True, prepushed=0),
        ),
        # Every other row count of the decode boundaries (modeling.MC_ALLREDUCE): the bucket and epilogue options the
        # target picks for it (ops/m3_mc_allreduce), the second addend and the MXFP8 copy alternating.
        *(
            Cell(
                why=f"{m} tokens: a boundary row count between the ones above",
                spec=dict(
                    m=m,
                    maxt=8 if m <= 8 else 16 if m <= 16 else 48,
                    opts=(7, 7, 15, 15, 15, 15, 7, 7)[m - 1] if m <= 8 else 15,
                    x2=m % 2 == 0,
                    quant=m % 3 == 0,
                    prepushed=0,
                ),
            )
            for m in range(1, 49)
            if m not in (1, 4, 8, 16, 30, 48)
        ),
        # Every other row count pushed by the o_proj (1..16) or the 1..8-token MoE engine (with the MXFP8 copy):
        # modeling.ATTN_PUSH / modeling.MOE_PUSH.
        *(
            Cell(
                why=f"{m} tokens prepushed: a pushed row count between the ones above",
                spec=dict(
                    m=m,
                    maxt=8 if m <= 8 else 16,
                    opts=(7, 7, 15, 15, 15, 15, 7, 7)[m - 1] if m <= 8 else 15,
                    x2=False,
                    quant=m <= 8,
                    prepushed=2,
                ),
            )
            for m in range(1, 17)
            if m not in (8, 12)
        ),
        # The Eagle3 drafter boundaries (modeling.DRAFT_MC_AR): option 16 (the row kernel's drafter form) on 23 / 31
        # at every draft-pass row count 1..8 (pass 0: 4 x requests, later passes: requests), x only.
        *(
            Cell(
                why=f"Eagle3 drafter boundary at {m} rows (opts {23 if m in (1, 2, 7, 8) else 31})",
                spec=dict(
                    m=m,
                    maxt=8,
                    opts=23 if m in (1, 2, 7, 8) else 31,
                    x2=False,
                    quant=False,
                    prepushed=0,
                    drafter=True,
                ),
            )
            for m in range(1, 9)
        ),
        # The same boundaries on the 8-CTA cluster kernel (opts bit 128, modeling.DRAFT_MC_AR_CLUSTER, launched without
        # the carveout: 23 | 128), alone and fed by the drafter projection's push (prepushed 2, DRAFT_AR_PUSH).
        *(
            Cell(
                why=f"Eagle3 drafter boundary on the cluster kernel at {m} rows, prepushed {pre}",
                spec=dict(
                    m=m,
                    maxt=8,
                    opts=23 | 128,
                    x2=False,
                    quant=False,
                    prepushed=pre,
                    drafter=True,
                ),
            )
            for pre in (0, 2)
            for m in range(1, 9)
        ),
    )

    note = """
    The epilogue option bits (1 early residual / weight load, 2 st.async cluster exchange, 4 early parity rotation,
    8 max carveout) are bit-identical by construction; only the four (bucket, option) pairs the target selects are
    built. `prepushed` 1 (pushed, grid wait kept) is accepted by the op but no caller passes it. `pdl` changes only
    launch overlap and is driven off in the cells.

    `normed` is gated to one BF16 rounding step: the kernel's RMS statistic is a warp / cluster tree of FP32 sums
    and its `rsqrtf` is approximate, so a row whose scaled value lands next to a BF16 midpoint can round either
    way. `updated` and the MXFP8 copy of the op's own `normed` are bit-exact.
    """

    def __call__(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        prepushed: int,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.minimax_m3_sm107_tp4_mc_allreduce_norm(
            x,
            x2,
            residual,
            norm_w,
            eps,
            ws_uc,
            ws_mc_ptr,
            flags,
            rank,
            world,
            cluster,
            pdl,
            None,
            None,
            maxt,
            quant,
            prepushed,
            opts,
            cap_out,
            cap_w,
            cap_eps,
        )

    def is_valid(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor],
        residual: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        ws_uc: torch.Tensor,
        ws_mc_ptr: int,
        flags: torch.Tensor,
        rank: int,
        world: int,
        cluster: int,
        pdl: bool,
        maxt: int,
        quant: bool,
        prepushed: int,
        opts: int,
        cap_out: Optional[torch.Tensor] = None,
        cap_w: Optional[torch.Tensor] = None,
        cap_eps: float = 0.0,
    ) -> None:
        h = residual.shape[-1]
        # The op checks norm_w's element count, not its layout: a strided view is read as packed memory.
        assert norm_w.is_contiguous(), (
            "norm_w must be contiguous; the kernel reads it as packed memory"
        )
        # The workspace is addressed as [set][parity][maxt][world][H / 2] words and never bounds-checked: a smaller
        # buffer (another bucket's, or one built for fewer ranks) is polled and re-armed out of range.
        need = 4 * maxt * world * (h // 2)
        assert ws_uc.is_contiguous() and ws_uc.numel() >= need, (
            f"ws_uc holds {ws_uc.numel()} words, the {maxt}-token bucket of {world} ranks needs {need}"
        )
        # Flags are per (token, cluster CTA) of the bucket; the op checks the count, not that they belong to the
        # workspace. Flags of another bucket hold another call history, and the parity they select is not the one the
        # peers push into: the poll then waits on the other half forever or reads last call's rows.
        assert flags.numel() == maxt * cluster, (
            f"flags has {flags.numel()} counters; the {maxt}-token bucket's are exactly {maxt * cluster}"
        )


m3_mc_allreduce_norm = M3McAllreduceNorm()
