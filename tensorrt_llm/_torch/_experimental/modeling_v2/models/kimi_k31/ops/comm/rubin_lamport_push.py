# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin pushed MNNVL all-reduce: the push and finish kernels.

A producer stores its bf16 partial into the Lamport slots of the next all-reduce on an MNNVL
workspace (``rubin_lamport_push``, or a GEMM's push epilogue) and ``rubin_lamport_finish``
completes the call, bit for bit like the C++ MNNVL all-reduce. The kernels are
``kernels/comm/rubin_lamport_ar.py``; ``__call__`` calls their launchers directly. The runtime
builder of the push workspace (``rubin_lamport_push_workspace``) is
``ops/comm/rubin_push_allreduce.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_comm_common import (
    HIDDEN,
    LAMPORT_NOTE,
    WORLD,
    check_rows,
    lamport_allreduce_reference,
    reduce_chunk,
    sanitize_bf16,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.comm.rubin_lamport_ar import run_lamport_finish as _launch_finish
    from ...kernels.comm.rubin_lamport_ar import run_lamport_push as _launch_push

if TYPE_CHECKING:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace


def _cells(one_shot_rows, two_shot_rows, **extra) -> list[Cell]:
    cells = [
        Cell(
            why=f"{rows} rows x {HIDDEN}, {WORLD} ranks, one-shot transport ({site})",
            spec=dict(world=WORLD, rows=rows, hidden=HIDDEN, one_shot=True, **extra),
        )
        for rows, site in one_shot_rows
    ]
    cells += [
        Cell(
            why=f"{rows} rows x {HIDDEN}, {WORLD} ranks, two-shot transport ({site})",
            spec=dict(world=WORLD, rows=rows, hidden=HIDDEN, one_shot=False, **extra),
        )
        for rows, site in two_shot_rows
    ]
    return cells


_ONE_SHOT_ROWS = ((1, "one decode row"), (8, "TEP8 b1 k7 o_proj"))
_TWO_SHOT_ROWS = ((96, "helix8 b16 k5 o_proj"), (8, "TEP8 b1 k7, forced two-shot"))


class RubinLamportPush(OpWrapper):
    """Push this rank's bf16 ``x`` [m, hidden] into the Lamport slots of the next call on ``ws``
    (a ``LamportWorkspace`` of the kernel module): one-shot (every rank's buffer, multicast) or
    two-shot (the owner ``t % world`` of each row). Returns None; the finish reads the slots."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = tuple(_cells(_ONE_SHOT_ROWS, _TWO_SHOT_ROWS))
    note = LAMPORT_NOTE

    def __call__(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> None:
        _launch_push(ws, x, one_shot, use_pdl)

    def reference(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> torch.Tensor:
        """The payload of this rank's slots: ``x`` with ``-0`` sent as ``+0``. A finish of the
        call is :func:`lamport_allreduce_reference` of every rank's payload."""
        return sanitize_bf16(x)

    def is_valid(
        self, ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool
    ) -> None:
        check_rows("x", ws, x)


class RubinLamportFinish(OpWrapper):
    """Complete the call every rank pushed into ``out`` (contiguous bf16 [m, hidden]); returns
    ``out``. ``one_shot``: the producers' transport; ``cpp_one_shot``: the C++ kernel whose
    result to reproduce (None: the transport's); ``ag``: the call carries all-gather rows (one-
    shot), and ``out`` is ``bf16(y + bf16(sum))``; ``wait_first``: the workers wait for the
    producer before they poll."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = (
        *_cells(_ONE_SHOT_ROWS, _TWO_SHOT_ROWS, cpp_one_shot=None),
        Cell(
            why="8 rows, two-shot transport reproducing the C++ one-shot kernel",
            spec=dict(world=WORLD, rows=8, hidden=HIDDEN, one_shot=False, cpp_one_shot=True),
        ),
        Cell(
            why="96 rows, one-shot transport reproducing the C++ two-shot kernel",
            spec=dict(world=WORLD, rows=96, hidden=HIDDEN, one_shot=True, cpp_one_shot=False),
        ),
        Cell(
            why="8 rows, one-shot, wait_first: the workers poll after the producer completes",
            spec=dict(
                world=WORLD,
                rows=8,
                hidden=HIDDEN,
                one_shot=True,
                cpp_one_shot=None,
                wait_first=True,
            ),
        ),
    )
    note = LAMPORT_NOTE

    def __call__(
        self,
        ws: LamportWorkspace,
        out: torch.Tensor,
        one_shot: bool,
        use_pdl: bool,
        wait_first: bool,
        cpp_one_shot: Optional[bool],
        ag: bool,
    ) -> torch.Tensor:
        return _launch_finish(
            ws,
            out,
            one_shot,
            use_pdl=use_pdl,
            wait_first=wait_first,
            cpp_one_shot=cpp_one_shot,
            ag=ag,
        )

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        one_shot: bool,
        cpp_one_shot: Optional[bool],
        ag_rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """What every rank gets back. The collective exception to mirroring ``__call__``: it
        takes every rank's pushed partial in rank order (and the all-gather rows with ``ag``)."""
        cpp = one_shot if cpp_one_shot is None else cpp_one_shot
        chunk = reduce_chunk(len(inputs_in_rank_order), cpp)
        return lamport_allreduce_reference(inputs_in_rank_order, chunk, ag_rows)

    def is_valid(
        self,
        ws: LamportWorkspace,
        out: torch.Tensor,
        one_shot: bool,
        use_pdl: bool,
        wait_first: bool,
        cpp_one_shot: Optional[bool],
        ag: bool,
    ) -> None:
        check_rows("out", ws, out)


rubin_lamport_push = RubinLamportPush()
rubin_lamport_finish = RubinLamportFinish()
