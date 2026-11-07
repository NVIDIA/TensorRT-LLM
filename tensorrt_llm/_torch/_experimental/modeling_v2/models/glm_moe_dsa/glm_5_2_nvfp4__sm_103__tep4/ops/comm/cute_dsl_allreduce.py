# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 all-reduce of a bf16 `[rows, hidden]` partial: one-shot Lamport over symmetric memory."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._lamport import LamportWorkspace, check_rows, tail


class CuteDslAllreduce(OpWrapper):
    """`out = sum over ranks of partial`, written to `out` and returned.

    `partial` and `out` are contiguous bf16 `[rows, hidden]` with `rows` up to
    `workspace.max_rows`. Every rank of the group calls it at the same point
    of its stream with the same row count. The kernel accumulates the four
    partials in fp32 in rank order on every rank and rounds once, so every
    rank holds the bit-identical result. `workspace` is `build_workspace`'s,
    shared with the two fused entries; the launch reads and advances its
    device slot counter.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls it twice: the MTP layer's eh projection partials, and
    # the MTP layer's closing sum of routed and shared expert outputs. Both run
    # at the step's row count, decode or prefill. One compile serves every row
    # count, so the cells span the range rather than a list of compiles.
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=dict(rows=1)),
        Cell(why="the full decode batch, 64 rows", spec=dict(rows=64)),
        Cell(why="the token budget, 8192 rows: every peer buffer slot full", spec=dict(rows=8192)),
    )

    note = """
    The target builds one workspace with `max_rows` = its token budget (8192)
    and calls this entry at every row count up to it; there is no stock
    fallback. All ranks must call with the same row count: a rank that
    disagrees waits on rows its peers never publish, and hangs rather than
    raising.
    """

    def __call__(
        self, partial: torch.Tensor, out: torch.Tensor, workspace: LamportWorkspace
    ) -> torch.Tensor:
        workspace.pure(partial, out, *tail(workspace, fused=False))
        return out

    def reference(
        self, partials: Sequence[torch.Tensor], out: torch.Tensor, workspace: LamportWorkspace
    ) -> torch.Tensor:
        """The fp32 sum of every rank's partial, in rank order, rounded once to bf16.

        The collective exception to mirroring `__call__`: `partials` holds the
        group's partials in rank order rather than this rank's. `out` and
        `workspace` are ignored.
        """
        acc = torch.zeros(partials[0].shape, dtype=torch.float32, device=partials[0].device)
        for p in partials:
            acc += p.float()
        return acc.to(torch.bfloat16)

    def is_valid(
        self, partial: torch.Tensor, out: torch.Tensor, workspace: LamportWorkspace
    ) -> None:
        check_rows(int(partial.shape[0]), workspace)


cute_dsl_allreduce = CuteDslAllreduce()
