# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin mix after attention (A_mid) whose addend is an MNNVL all-reduce.

The attention block's o_proj all-reduce is pushed into a Lamport workspace
(``ops/comm/rubin_push_allreduce.py``); A_mid reads the result there instead of a tensor and
completes the pushed call itself (rank-order sums, rotation state, re-arm of the previous
buffer), so no finish kernel runs in front of it. The tensor-parallel
targets use it (TEP8, helix8); attention DP has no such all-reduce. The kernel is
``kernels/norm/rubin_attn_res.py``; ``__call__`` calls its launcher directly.

References: the addend is a collective, so each reference takes every rank's pushed partial
in rank order (``partials_in_rank_order``) and the ranks per partial sum (``chunk``) instead of
the workspace addresses (the documented exception to mirroring ``__call__``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_norm_common import (
    LAMPORT_NOTE,
    TEP8_ROWS,
    VERIFY96_ROWS,
    check_attn_res_inputs,
    check_finish,
    lamport_reference,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res import k3_attn_res_lamport_add_rmsnorm as _launch_add


#: Ranks the op tests drive (one VR200 tray).
WORLD = 4


def _finish_cells(mx: bool = False) -> list[Cell]:
    """Cells where the mix finishes a call every rank pushed (4 ranks)."""
    cells = []
    for rows, two_shot, site in (
        (TEP8_ROWS, False, "TEP8 b1 k7, one-shot"),
        (VERIFY96_ROWS, True, "helix8 b16 k5, two-shot"),
    ):
        for n in (5, 9):
            cells.append(
                Cell(
                    why=f"{rows} rows ({site}), N={n}: the mix finishes the pushed call",
                    spec=dict(
                        world=WORLD,
                        rows=rows,
                        candidates=n,
                        two_shot=two_shot,
                        early_poll=False,
                        early_trigger=True,
                    ),
                )
            )
        cells.append(
            Cell(
                why=f"{rows} rows ({site}), N=9: early_poll",
                spec=dict(
                    world=WORLD,
                    rows=rows,
                    candidates=9,
                    two_shot=two_shot,
                    early_poll=True,
                    early_trigger=True,
                ),
            )
        )
    if not mx:
        cells += [
            Cell(
                why="8 rows, N=5, one-shot, round-to-nearest add (the MoE tail's update)",
                spec=dict(
                    world=WORLD,
                    rows=8,
                    candidates=5,
                    two_shot=False,
                    early_poll=False,
                    early_trigger=True,
                    add_rn=True,
                ),
            ),
        ]
    return cells


class RubinAttnResLamportAddRmsnorm(OpWrapper):
    """``(updated, output)`` as :class:`RubinAttnResAddRmsnorm`, with the addend the call every
    rank pushed, which this kernel finishes over ``finish_world`` ranks (``finish_chunk`` ranks per
    partial sum; one-shot or ``finish_two_shot`` transport; ``mc_base``: the multicast base;
    ``rank``: this rank). ``add_rn``: a round-to-nearest add instead of the fast-math one."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = tuple(_finish_cells())
    note = LAMPORT_NOTE

    def __call__(
        self,
        prefix: torch.Tensor,
        uc_local: int,
        flags: int,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        finish_world: int,
        finish_chunk: int,
        finish_two_shot: bool,
        mc_base: int,
        rank: int,
        add_rn: bool,
        early_poll: bool,
        finish_ag: bool,
        *,
        early_trigger: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch_add(
            prefix,
            uc_local,
            flags,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            finish_world=finish_world,
            finish_chunk=finish_chunk,
            finish_two_shot=finish_two_shot,
            mc_base=mc_base,
            rank=rank,
            add_rn=add_rn,
            early_poll=early_poll,
            finish_ag=finish_ag,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        prefix: torch.Tensor,
        partials_in_rank_order: Sequence[torch.Tensor],
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        chunk: int,
        add_rn: bool,
        ag_rows: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``chunk``: ``finish_chunk``; ``ag_rows``: the all-gather rows of a ``finish_ag``
        call."""
        return lamport_reference(
            prefix,
            partials_in_rank_order,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            chunk,
            add_rn,
            ag_rows,
        )

    def is_valid(
        self,
        prefix: torch.Tensor,
        uc_local: int,
        flags: int,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        finish_world: int,
        finish_chunk: int,
        finish_two_shot: bool,
        mc_base: int,
        rank: int,
        add_rn: bool,
        early_poll: bool,
        finish_ag: bool,
        *,
        early_trigger: bool,
    ) -> None:
        check_attn_res_inputs(prefix, (), block, res_weight, rms_weight, out_weight)
        assert uc_local and flags, "the workspace addresses must be set"
        check_finish(finish_world, finish_chunk, rank)


rubin_attn_res_lamport_add_rmsnorm = RubinAttnResLamportAddRmsnorm()
