# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 all-reduce fused with the residual add and RMSNorm that consume it."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._lamport import QUANT_TWOSHOT_ABOVE, TWOSHOT_ABOVE, LamportWorkspace, check_rows, tail


class CuteDslAllreduceAddRmsnorm(OpWrapper):
    """`residual += sum over ranks of partial`; `out = rmsnorm(residual) * weight`.

    `partial`, `residual` and `out` are contiguous bf16 `[rows, hidden]`,
    `weight` bf16 `[hidden]`, `rows` up to `workspace.max_rows`. `residual` is
    updated in place, `out` is written and returned. The sum is fp32 in rank
    order on every rank; the residual add is fp32 and rounds once to bf16, and
    the norm reads that bf16 residual. `eps` is the workspace's, fixed at build.

    Up to `TWOSHOT_ABOVE` (512) rows the kernel is one-shot: every rank pulls every
    peer's whole partial. Above it, two-shot: each rank pushes its partial rows to
    the rows' owner, the owner reduces them, and the reduced rows are pushed back
    to every rank. The two-shot gather moves
    the reduced rows as bf16, so the residual add there rounds twice (sum,
    then sum + residual). The choice is a function of the row count alone and
    is made here, identically on every rank.

    `x8 = (q8 [rows, hidden] uint8, sf)`, two-shot rows only: the same launch also
    writes the normalized rows as an MXF8 operand -- per 32 elements a power-of-two
    scale and e4m3 values, the activation quant's exact bytes of `out` -- for the GEMM
    that reads them next (the next layer's qkva).
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every trunk layer opens its MLP and closes into the next layer with this
    # entry, and the MTP layer opens its MLP with it: at the step's row count,
    # decode or prefill. The cells cover both kernels and the boundary between.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row, one-shot", spec=dict(rows=1)),
        Cell(why="the full decode batch, 64 rows, one-shot", spec=dict(rows=64)),
        Cell(why="the last one-shot row count, 512", spec=dict(rows=512)),
        Cell(why="the first two-shot row count, 513: a ragged last row block", spec=dict(rows=513)),
        Cell(why="a two-shot context chunk of 3073 rows", spec=dict(rows=3073)),
        Cell(why="the token budget, 8192 rows, two-shot", spec=dict(rows=8192)),
        Cell(
            why="two-shot with the MXF8 copy of the normalized rows, 3073 rows",
            spec=dict(rows=3073, x8=True),
        ),
        Cell(
            why="two-shot with the MXF8 copy at the token budget",
            spec=dict(rows=8192, x8=True),
        ),
    )

    note = """
    The target builds one workspace with `max_rows` = its token budget (8192)
    and calls this entry at every row count up to it; there is no stock
    fallback. All ranks must call with the same row count: a rank that
    disagrees waits on rows its peers never publish, and hangs rather than
    raising.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
        x8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if x8 is not None:
            q8, sf = x8
            workspace.fused_twoshot_x8(
                partial, residual, weight, out, q8, sf, residual, *tail(workspace, fused=True)
            )
            return out
        rows = partial.shape[0]
        compiled = workspace.fused_twoshot if rows > TWOSHOT_ABOVE else workspace.fused
        if workspace.fused_wide is not None and QUANT_TWOSHOT_ABOVE < rows <= TWOSHOT_ABOVE:
            # A `wide` workspace (MTP3 / MTP5 engines): the same one-shot with the wide AR levers
            # (`_lamport.wide_ar_levers`), same bytes. Every rank has the same workspace kind.
            compiled = workspace.fused_wide
        compiled(partial, residual, weight, out, residual, *tail(workspace, fused=True))
        return out

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
        x8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(norm, new_residual)`, both bf16; `residual` is not modified.

        `partials` is the group's partials in rank order (the collective
        exception to mirroring `__call__`). `x8`'s bytes are the activation quant of the
        norm (`self_attn_prefill_act_quant_mxf8`'s rule), which the caller derives from it.
        The sum is `cute_dsl_allreduce`'s reference before its rounding: fp32 in rank order. `new_residual` is
        `bf16(sum + residual)`, rounded once. The norm is computed in fp64 from
        that bf16 residual with `workspace.eps` and rounded once. `out` is
        ignored.
        """
        acc = torch.zeros(residual.shape, dtype=torch.float32, device=residual.device)
        for p in partials:
            acc += p.float()
        new_residual = (acc + residual.float()).to(torch.bfloat16)
        x = new_residual.double()
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + workspace.eps)
        norm = (x * inv_rms * weight.double()).to(torch.bfloat16)
        return norm, new_residual

    def is_valid(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
        x8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        rows = int(partial.shape[0])
        check_rows(rows, workspace)
        if x8 is not None:
            assert workspace.fused_twoshot_x8 is not None, "the group did not build the MXF8 copy"
            assert rows > TWOSHOT_ABOVE, (
                f"the MXF8 copy is a two-shot epilogue: rows above {TWOSHOT_ABOVE}, got {rows}"
            )
            q8, sf = x8
            n = int(partial.shape[1])
            assert (
                q8.dtype == torch.uint8 and q8.is_contiguous() and tuple(q8.shape) == (rows, n)
            ), f"x8's values must be a contiguous uint8 [{rows}, {n}]"
            need = (rows + 127) // 128 * 128 * (n // 32)
            assert sf.dtype == torch.uint8 and sf.numel() >= need and sf.data_ptr() % 32 == 0, (
                f"x8's scales must hold {need} bytes, 32-byte aligned"
            )


cute_dsl_allreduce_add_rmsnorm = CuteDslAllreduceAddRmsnorm()
