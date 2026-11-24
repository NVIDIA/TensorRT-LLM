# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin deferred MoE residual update + attention-residual mix + RMSNorm, one kernel.

A MoE layer leaves ``residual + bf16(routed + shared)`` to the next layer's pre-attention mix,
which computes it bit for bit (the ``_mx`` entry also stores the MXFP8 form of its output: the
low-M FP8 GEMM's quantization, or with ``mx_exact`` the catalog ``mxfp8_quantize``'s, which the
dense MXFP8 GEMM of the KDA projection takes). The kernel is ``kernels/norm/rubin_attn_res.py``;
``__call__`` calls its launcher directly. ``updated`` (keyword, optional): a caller buffer the
kernel stores ``prefix`` in, returned as ``prefix`` (helix8 ``mix_bank_in_mix``: a block boundary's
bank row), instead of a new tensor; the same bits.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ._rubin_norm_common import (
    VERIFY96_ROWS,
    attn_res_cells,
    bf16_add,
    check_attn_res_inputs,
    check_updated_out,
    mix_rmsnorm_reference,
    mxfp8_exact_reference,
    mxfp8_reference,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.norm.rubin_attn_res import k3_attn_res_moe_add_rmsnorm as _launch_moe_add
    from ...kernels.norm.rubin_attn_res import k3_attn_res_moe_add_rmsnorm_mx as _launch_moe_add_mx


def moe_residual_reference(
    residual: torch.Tensor, routed: torch.Tensor, shared: torch.Tensor
) -> torch.Tensor:
    """``bf16(residual + bf16(routed + shared))``, both adds rounded to nearest."""
    return bf16_add(residual, bf16_add(routed, shared, ftz=False), ftz=False)


def _mx_cells() -> tuple[Cell, ...]:
    cells = [
        Cell(
            why=f"{rows} rows, N={n}: row-major scales",
            spec=dict(rows=rows, candidates=n, early_trigger=True),
        )
        for rows in (1, 8, VERIFY96_ROWS)
        for n in (2, 9)
    ]
    cells += [
        Cell(
            why="96 rows, N=5, entry trigger (a QKVG GEMM that stages its weights follows)",
            spec=dict(rows=VERIFY96_ROWS, candidates=5, early_trigger=True, entry_trigger=True),
        ),
        Cell(
            why="96 rows, N=6: scales in one 128-row block of the mid-M GEMM's atoms",
            spec=dict(rows=VERIFY96_ROWS, candidates=6, early_trigger=True, mx_atom_blocks=1),
        ),
        Cell(
            why="128 rows, N=2: a full 128-row atom block",
            spec=dict(rows=128, candidates=2, early_trigger=True, mx_atom_blocks=1),
        ),
        Cell(
            why="144 rows, N=6: two 128-row atom blocks (helix8 b24 k5)",
            spec=dict(rows=144, candidates=6, early_trigger=True, mx_atom_blocks=2),
        ),
        Cell(
            why="early trigger off, 8 rows, N=9",
            spec=dict(rows=8, candidates=9, early_trigger=False),
        ),
    ]
    # The TEP8 decode step's KDA input: the catalog quantizer's codes and R128c4 scales of one
    # 128-row tile for the dense MXFP8 QKVG GEMM (b1 / b2 / b4 k7: 8 / 16 / 32 rows).
    cells += [
        Cell(
            why=f"{rows} rows, N={n}: catalog-exact MXFP8, R128c4 scales of one tile",
            spec=dict(rows=rows, candidates=n, early_trigger=True, mx_atom_blocks=1, mx_exact=True),
        )
        for rows, n in ((8, 2), (8, 9), (32, 5))
    ]
    # helix8's KDA input under attn_res_mx_pdl: the k5 verify block of b1..b21 (6 rows per
    # request), up to one 128-row tile, after an early-triggering mix, in the QKVG GEMM's own
    # quantization (the shared low-M / mid-M GEMM's, amax floored) on both sides of its low-M rows.
    cells += [
        Cell(
            why=f"{rows} rows, N={n}: helix8 KDA verify, the low-M GEMM's MXFP8, one tile",
            spec=dict(rows=rows, candidates=n, early_trigger=True, mx_atom_blocks=1),
        )
        for rows, n in ((6, 2), (24, 5), (54, 9), (66, 3), (126, 7))
    ]
    return tuple(cells)


class RubinAttnResMoeAddRmsnorm(OpWrapper):
    """``(prefix, output)``: ``prefix = bf16(residual + bf16(routed + shared))`` (the previous
    MoE layer's residual update) and ``rmsnorm(mix(block, prefix)) * out_weight``. ``residual``,
    ``routed``, ``shared`` are bf16 [M, 7168]; both outputs are new bf16 [M, 7168]."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = attn_res_cells(include_eps=False)
    note = """
    The update has two round-to-nearest adds, as the Triton MoE residual add has; the mix and
    norm are rubin_attn_res_rmsnorm's (same kernel, same rounding). ``early_trigger``: see
    rubin_attn_res_rmsnorm. Inputs are read-only.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch_moe_add(
            residual,
            routed,
            shared,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            early_trigger=early_trigger,
            updated=updated,
        )

    def reference(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two bf16 adds, then the mix and the norm as the kernel rounds them (``updated``,
        where the kernel stores the prefix, is ignored)."""
        prefix = moe_residual_reference(residual, routed, shared)
        output = mix_rmsnorm_reference(
            prefix, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        return prefix, output

    def is_valid(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> None:
        check_attn_res_inputs(residual, (routed, shared), block, res_weight, rms_weight, out_weight)
        check_updated_out(updated, residual, (residual, routed, shared, block))


class RubinAttnResMoeAddRmsnormMx(OpWrapper):
    """:class:`RubinAttnResMoeAddRmsnorm` plus the MXFP8 form ``(xq, xs)`` of its output:
    returns ``(prefix, output, xq, xs)``, xq float8_e4m3fn [M, 7168]; xs uint8 [M, 224], or
    with ``mx_atom`` the mid-M FP8 GEMM's scale atoms ([56 * mx_atom_blocks * 512])."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = _mx_cells()
    note = """
    The codes and scales are the low-M FP8 GEMM's own quantization of ``output``, bit for bit;
    with ``mx_exact`` the catalog ``mxfp8_quantize``'s (no amax floor), so that with ``mx_atom``
    and one block ``(xq, xs)`` are the operands ``rubin_mxfp8_linear`` would make of ``output``.
    ``entry_trigger`` releases the dependent grid at kernel entry and prefers the largest
    shared-memory carve-out; ``mx_atom`` needs M <= 128 * mx_atom_blocks (the launcher raises
    otherwise) and leaves atom rows past M unwritten.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        entry_trigger: bool,
        mx_atom: bool,
        mx_atom_blocks: int,
        mx_exact: bool = False,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _launch_moe_add_mx(
            residual,
            routed,
            shared,
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            entry_trigger=entry_trigger,
            mx_atom=mx_atom,
            mx_atom_blocks=mx_atom_blocks,
            early_trigger=early_trigger,
            mx_exact=mx_exact,
            updated=updated,
        )

    def reference(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        entry_trigger: bool,
        mx_atom: bool,
        mx_atom_blocks: int,
        mx_exact: bool = False,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The non-``_mx`` reference plus :func:`mxfp8_reference` (``mx_exact``:
        :func:`mxfp8_exact_reference`) of its output. The two triggers and ``updated`` do not
        change results."""
        prefix = moe_residual_reference(residual, routed, shared)
        output = mix_rmsnorm_reference(
            prefix, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
        )
        form = mxfp8_exact_reference if mx_exact else mxfp8_reference
        return (prefix, output, *form(output, mx_atom, mx_atom_blocks))

    def is_valid(
        self,
        residual: torch.Tensor,
        routed: torch.Tensor,
        shared: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
        entry_trigger: bool,
        mx_atom: bool,
        mx_atom_blocks: int,
        mx_exact: bool = False,
        *,
        early_trigger: bool,
        updated: Optional[torch.Tensor] = None,
    ) -> None:
        check_attn_res_inputs(residual, (routed, shared), block, res_weight, rms_weight, out_weight)
        check_updated_out(updated, residual, (residual, routed, shared, block))
        # The launcher drops mx_atom_blocks without mx_atom; a caller asking for two blocks
        # of row-major scales would read [M, 224] where it expects atoms.
        assert mx_atom or mx_atom_blocks == 1, "mx_atom_blocks > 1 needs mx_atom"


rubin_attn_res_moe_add_rmsnorm = RubinAttnResMoeAddRmsnorm()
rubin_attn_res_moe_add_rmsnorm_mx = RubinAttnResMoeAddRmsnormMx()
