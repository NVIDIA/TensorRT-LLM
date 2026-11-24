# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin mix after attention (A_mid) whose addend is an MNNVL all-reduce.

The attention block's o_proj all-reduce is pushed into a Lamport workspace
(``ops/comm/rubin_push_allreduce.py``); A_mid reads the result there instead of a tensor, and
with a finish it also completes the pushed call itself (rank-order sums, rotation state,
re-arm of the previous buffer), so no finish kernel runs in front of it. The tensor-parallel
targets use it (TEP8, helix8); attention DP has no such all-reduce. The ``_mx`` entry also
stores the MXFP8 form of its output, as ``rubin_attn_res_moe_add_rmsnorm_mx`` does: a KDA layer's
mix that finishes the column-parallel MoE tail's call (helix8 ``moe_cp_tail``) hands it to the
QKVG GEMM. The kernel is ``kernels/norm/rubin_attn_res.py``; ``__call__`` calls its launcher
directly. Two keywords leave the bits unchanged: ``updated``, a caller buffer the kernel stores the
updated prefix in (helix8 ``mix_bank_in_mix``: a block boundary's bank row), and ``replace``, the
finished all-reduce as the updated prefix instead of ``prefix + addend`` (helix8
``boundary_ar_push``: the mix after a block boundary's attention, whose prefix sum starts over at
the attention output; ``prefix`` is then not read), the bits of the C++ all-reduce followed by
``rubin_attn_res_rmsnorm``.

References: the addend is a collective, so each reference takes every rank's pushed partial
in rank order (``partials_in_rank_order``) and the ranks per partial sum (``chunk``) instead of
the workspace addresses (the documented exception to mirroring ``__call__``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_norm_common import (
    LAMPORT_NOTE,
    TEP8_ROWS,
    VERIFY96_ROWS,
    check_attn_res_inputs,
    check_finish,
    check_updated_out,
    lamport_reference,
    mxfp8_exact_reference,
    mxfp8_reference,
    single_cell_specs,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res import k3_attn_res_lamport_add_rmsnorm as _launch_add
    from ...kernels.norm.rubin_attn_res import k3_attn_res_lamport_add_rmsnorm_mx as _launch_add_mx


#: Ranks the op tests drive (one VR200 tray).
WORLD = 4
#: helix8's MoE TP group: two trays.
TAIL_WORLD = 8


def _single_cells(**extra) -> list[Cell]:
    """Cells that read the result of a finished call from a one-rank stand-in workspace."""
    return [Cell(why=why, spec=spec) for why, spec in single_cell_specs(**extra)]


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
                why="8 rows, N=5, one-shot, early poll, entry trigger (the MoE front's head)",
                spec=dict(
                    world=WORLD,
                    rows=8,
                    candidates=5,
                    two_shot=False,
                    early_poll=True,
                    early_trigger=True,
                    entry_trigger=True,
                ),
            ),
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


def _replace_cells() -> list[Cell]:
    """Cells where the finished all-reduce replaces the prefix (a block boundary's mix after its
    attention, helix8 ``boundary_ar_push``): the k5 verify rows of b2 / b4 / b9 / b11 on both
    transports, with the bank sizes of the boundaries (2 to 8 candidates), at one tray's 4 ranks
    and helix8's 8 (two trays)."""
    cells = []
    for world, rows, n, two_shot in (
        (WORLD, 12, 2, False),
        (WORLD, 24, 5, False),
        (WORLD, 54, 8, True),
        (WORLD, 66, 3, True),
        (TAIL_WORLD, 12, 2, False),
        (TAIL_WORLD, 24, 4, True),
        (TAIL_WORLD, 54, 8, True),
        (TAIL_WORLD, 66, 6, True),
    ):
        cells.append(
            Cell(
                why=(
                    f"{rows} rows over {world} ranks, N={n}, "
                    f"{'two' if two_shot else 'one'}-shot: the all-reduce replaces the prefix"
                ),
                spec=dict(
                    world=world,
                    rows=rows,
                    candidates=n,
                    two_shot=two_shot,
                    early_poll=True,
                    early_trigger=True,
                    replace=True,
                    updated_out=rows in (24, 66),
                ),
            )
        )
    return cells


def _tail_mx_cells() -> list[Cell]:
    """Cells where the mix finishes the column-parallel MoE tail's call (one-shot, all-gather
    rows, the round-to-nearest add) and stores the MXFP8 form of one 128-row tile (R128c4
    atoms): helix8's k5 verify rows (6 per request) in both of its quantization rules (the
    shared low-M / mid-M GEMM's, amax floored; ``mx_exact``: the catalog quantizer's), at one
    tray's 4 ranks and helix8's 8 (two trays)."""
    cells = []
    for world, rows, poll, exact in (
        (WORLD, 24, False, False),
        (WORLD, 24, True, False),
        (WORLD, 54, True, False),
        (WORLD, 54, True, True),
        (WORLD, 72, True, False),
        (WORLD, 72, False, True),
        (WORLD, 36, True, False),
        (WORLD, 66, True, False),
        (TAIL_WORLD, 24, True, False),
        (TAIL_WORLD, 36, True, False),
        (TAIL_WORLD, 48, True, False),
        (TAIL_WORLD, 60, True, False),
        (TAIL_WORLD, 66, True, False),
        (TAIL_WORLD, 72, True, True),
    ):
        cells.append(
            Cell(
                why=(
                    f"{rows} rows over {world} ranks, N=5: the MoE tail's call (all-gather rows), "
                    f"early_poll={int(poll)}, mx_exact={int(exact)}"
                ),
                spec=dict(
                    world=world,
                    rows=rows,
                    candidates=5,
                    two_shot=False,
                    early_poll=poll,
                    early_trigger=True,
                    add_rn=True,
                    finish_ag=True,
                    mx_atom=True,
                    mx_exact=exact,
                    # A block boundary's mix (mix_bank_in_mix) stores its row into a bank row.
                    updated_out=rows in (54, 66),
                ),
            )
        )
    return cells


def _tail_cells() -> list[Cell]:
    """Cells where the BF16 mix finishes the column-parallel MoE tail's call (one-shot, all-gather
    rows, the round-to-nearest add): helix8's k5 verify rows up to the low-M QKVG GEMM's 32, where
    a KDA layer's mix stores no MXFP8 form, and an MLA layer's mix at any tail row up to 72, at one
    tray's 4 ranks and helix8's 8 (two trays)."""
    cells = []
    for world, rows, poll in (
        (WORLD, 6, True),
        (WORLD, 24, False),
        (WORLD, 30, True),
        (WORLD, 36, True),
        (WORLD, 54, True),
        (WORLD, 72, True),
        (TAIL_WORLD, 6, True),
        (TAIL_WORLD, 30, True),
        (TAIL_WORLD, 36, True),
        (TAIL_WORLD, 48, True),
        (TAIL_WORLD, 54, True),
        (TAIL_WORLD, 60, True),
        (TAIL_WORLD, 66, True),
        (TAIL_WORLD, 72, True),
    ):
        cells.append(
            Cell(
                why=(
                    f"{rows} rows over {world} ranks, N=5: the MoE tail's call (all-gather rows), "
                    f"early_poll={int(poll)}"
                ),
                spec=dict(
                    world=world,
                    rows=rows,
                    candidates=5,
                    two_shot=False,
                    early_poll=poll,
                    early_trigger=True,
                    add_rn=True,
                    finish_ag=True,
                    # A block boundary's mix (mix_bank_in_mix) stores its row into a bank row.
                    updated_out=rows in (24, 30),
                ),
            )
        )
    return cells


class RubinAttnResLamportAddRmsnorm(OpWrapper):
    """``(updated, output)`` as :class:`RubinAttnResAddRmsnorm`, with the addend either
    the call finished last (``finish_world == 0``) or the call every rank pushed, which this
    kernel finishes over ``finish_world`` ranks (``finish_chunk`` ranks per partial sum; one-shot
    or ``finish_two_shot`` transport; ``mc_base``: the multicast base; ``rank``: this rank).
    ``add_rn``: a round-to-nearest add instead of the fast-math one."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = (
        *_single_cells(add_rn=False),
        *_finish_cells(),
        *_tail_cells(),
        *_replace_cells(),
    )
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
        entry_trigger: bool,
        finish_ag: bool,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
        replace: bool = False,
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
            entry_trigger=entry_trigger,
            finish_ag=finish_ag,
            early_trigger=early_trigger,
            updated=updated,
            replace=replace,
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
        replace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``chunk``: ``finish_chunk``, or for a call finished earlier its finish's chunk;
        ``ag_rows``: the all-gather rows of a ``finish_ag`` call; ``replace``: the all-reduce
        replaces ``prefix`` (not read)."""
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
            replace,
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
        entry_trigger: bool,
        finish_ag: bool,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
        replace: bool = False,
    ) -> None:
        check_attn_res_inputs(prefix, (), block, res_weight, rms_weight, out_weight)
        assert uc_local and flags, "the workspace addresses must be set"
        check_finish(finish_world, finish_chunk, rank, early_poll, finish_ag)
        check_updated_out(updated, prefix, (block,) if replace else (prefix, block))


class RubinAttnResLamportAddRmsnormMx(OpWrapper):
    """:class:`RubinAttnResLamportAddRmsnorm` plus the MXFP8 form ``(xq, xs)`` of its output:
    returns ``(updated, output, xq, xs)``, xq float8_e4m3fn [M, 7168]; xs uint8 [M, 224], or with
    ``mx_atom`` the mid-M FP8 GEMM's scale atoms ([56 * mx_atom_blocks * 512])."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = (*_finish_cells(mx=True), *_tail_mx_cells())
    note = (
        LAMPORT_NOTE
        + """
    The MXFP8 epilogue is rubin_attn_res_moe_add_rmsnorm_mx's (same kernel): the low-M FP8
    GEMM's own quantization of ``output``, bit for bit; with ``mx_exact`` the catalog
    ``mxfp8_quantize``'s (no amax floor), so that with ``mx_atom`` and one block ``(xq, xs)`` are
    the operands ``rubin_mxfp8_linear`` would make of ``output``. ``mx_atom`` needs
    M <= 128 * mx_atom_blocks (the launcher raises otherwise) and leaves atom rows past M
    unwritten.
    """
    )

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
        entry_trigger: bool,
        finish_ag: bool,
        mx_atom: bool,
        mx_atom_blocks: int,
        mx_exact: bool = False,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
        replace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_add_mx(
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
            entry_trigger=entry_trigger,
            finish_ag=finish_ag,
            mx_atom=mx_atom,
            mx_atom_blocks=mx_atom_blocks,
            early_trigger=early_trigger,
            mx_exact=mx_exact,
            updated=updated,
            replace=replace,
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
        mx_atom: bool = False,
        mx_atom_blocks: int = 1,
        mx_exact: bool = False,
        replace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """:class:`RubinAttnResLamportAddRmsnorm`'s reference plus :func:`mxfp8_reference`
        (``mx_exact``: :func:`mxfp8_exact_reference`) of its output."""
        updated, output = lamport_reference(
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
            replace,
        )
        form = mxfp8_exact_reference if mx_exact else mxfp8_reference
        return (updated, output, *form(output, mx_atom, mx_atom_blocks))

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
        entry_trigger: bool,
        finish_ag: bool,
        mx_atom: bool,
        mx_atom_blocks: int,
        mx_exact: bool = False,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
        replace: bool = False,
    ) -> None:
        check_attn_res_inputs(prefix, (), block, res_weight, rms_weight, out_weight)
        assert uc_local and flags, "the workspace addresses must be set"
        check_finish(finish_world, finish_chunk, rank, early_poll, finish_ag)
        check_updated_out(updated, prefix, (block,) if replace else (prefix, block))
        assert mx_atom_blocks in (1, 2), "one or two 128-row scale blocks"
        # The launcher drops mx_atom_blocks without mx_atom; a caller asking for two blocks of
        # row-major scales would read [M, 224] where it expects atoms.
        assert mx_atom or mx_atom_blocks == 1, "mx_atom_blocks > 1 needs mx_atom"
        assert not mx_atom or prefix.shape[0] <= 128 * mx_atom_blocks, (
            f"mx_atom with {mx_atom_blocks} scale block(s) covers at most {128 * mx_atom_blocks} rows"
        )


rubin_attn_res_lamport_add_rmsnorm = RubinAttnResLamportAddRmsnorm()
rubin_attn_res_lamport_add_rmsnorm_mx = RubinAttnResLamportAddRmsnormMx()
