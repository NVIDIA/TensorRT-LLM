# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An indexer layer's projections folded into the attention's on a wide decode step, each
with the plain projection as its fallback: `qkva_ix_fold` (qkva with the indexer's key /
head-weight rows, `cute_dsl_qkva_ix_fold`) and `q_b_ix_wq_b` (q_b with the indexer's wq_b,
`cute_dsl_q_b_ix_wq_b`).

Their cells are the MTP decode row counts (128 / 192), built only where they match the
plain projections' MTP cells row for row (`TRTLLM_GLM52_WIDE_IX_FOLD`,
`TRTLLM_GLM52_WIDE_QB_IXQB`, an engine with the wide decode cells); the row counts between
them run on the next cell up at their exact rows. Where a fold serves, its attention output
is bitwise the plain projection's MTP cell's, and the indexer's operand comes with it (the
indexer then skips its own projection); elsewhere that operand is None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from . import cute_dsl_q_b_ix_wq_b as _qbix_op
from . import cute_dsl_qkva_ix_fold as _ix_fold_op
from .cute_dsl_q_b_ix_wq_b import cute_dsl_q_b_ix_wq_b
from .cute_dsl_qkva_ix_fold import cute_dsl_qkva_ix_fold
from .projection import DECODE_MAX_ROWS, ProjectionHandle, projection, wide_cell, wide_cells


@dataclass(frozen=True)
class FoldHandle:
    """One fold's cells by row count (`cells`, empty: never folded) and the row counts
    between them on the next one up (`wide`, empty with `wide_bucket` off); `plain` the
    projection it falls back to."""

    cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    plain: ProjectionHandle
    count: Count


def build_qkva(
    cells: Mapping[int, Any], plain: ProjectionHandle, wide_bucket: bool, count: Count
) -> FoldHandle:
    """The qkva fold's handle over its cells (`cute_dsl_qkva_ix_fold.build_all`); re-pins
    them to every row count between them at load (compiles nothing)."""
    return FoldHandle(
        cells=dict(cells),
        wide=wide_cells(cells, _ix_fold_op.at_rows) if wide_bucket else {},
        wide_bucket=wide_bucket,
        plain=plain,
        count=count,
    )


def build_q_b(
    cells: Mapping[int, Any], plain: ProjectionHandle, wide_bucket: bool, count: Count
) -> FoldHandle:
    """The q_b | wq_b fold's handle over its cells (`cute_dsl_q_b_ix_wq_b.build_all`)."""
    return FoldHandle(
        cells=dict(cells),
        wide=wide_cells(cells, _qbix_op.at_rows) if wide_bucket else {},
        wide_bucket=wide_bucket,
        plain=plain,
        count=count,
    )


def _fold_cell(ph, rows: int, w_fold, handle: FoldHandle, route: str):
    """The fold cell for a call, counted `route`: a `Phase.wide` step's rows above 64 of a
    layer with a stacked weight (`w_fold`), where the fold built. None: the plain path."""
    if w_fold is None or not handle.cells or not ph.wide or rows <= DECODE_MAX_ROWS:
        return None
    cell = wide_cell(handle.cells, handle.wide, handle.wide_bucket, rows, handle.count)
    handle.count(route, cell is not None)
    return cell


class QkvaIxFold(OpWrapper):
    """`(qkva, kw)` of `x`: where the fold serves (`w_fold` the layer's stacked qkva + indexer
    weight; bf16 contiguous `x`; counted `ix_fold`), one GEMM writing qkva and the indexer's
    key / head weights `kw` (views of the cell's side buffers, consumed by this layer's
    indexer before the next layer's call). Else the qkva projection (`w`, `w8` its MXF8
    copy, `x8` `x`'s MXF8 operand from the collective before) and `kw` None."""

    def __call__(
        self,
        ph,
        x: torch.Tensor,
        w: torch.Tensor,
        w8,
        x8,
        w_fold: torch.Tensor | None,
        handle: FoldHandle,
    ):
        rows = int(x.shape[0])
        if x.dtype == torch.bfloat16 and x.is_contiguous():
            cell = _fold_cell(ph, rows, w_fold, handle, "ix_fold")
            if cell is not None:
                out = torch.empty([rows, _ix_fold_op.QKVA_N], dtype=x.dtype, device=x.device)
                qkva, key, head_w = cute_dsl_qkva_ix_fold(x, w_fold, out, cell)
                return qkva, (key, head_w)
        return projection(ph, x, w, handle.plain, w8=w8, pre=x8), None

    def reference(
        self,
        ph,
        x: torch.Tensor,
        w: torch.Tensor,
        w8,
        x8,
        w_fold: torch.Tensor | None,
        handle: FoldHandle,
    ):
        """`x @ w` in fp32, rounded once to bf16 (the indexer columns are the indexer
        projection's reference)."""
        return (x.float() @ w.float()).to(x.dtype), None


class QbIxWqB(OpWrapper):
    """`(q, ixq)` of `q_a`: where the fold serves (`w_fold` the layer's stacked q_b + wq_b
    weight, passed only where the step runs the indexer; counted `qb_ixqb`), one GEMM writing
    q and the indexer's q projection `ixq`, two halves of one buffer. Else the q_b projection
    (`w`; `wd8` its decode MXF8 weight, `dec_pre` whether the q_a norm wrote that GEMM's
    operand) and `ixq` None."""

    def __call__(
        self,
        ph,
        q_a: torch.Tensor,
        w: torch.Tensor,
        w_fold: torch.Tensor | None,
        wd8,
        dec_pre: bool,
        handle: FoldHandle,
    ):
        rows = int(q_a.shape[0])
        cell = _fold_cell(ph, rows, w_fold, handle, "qb_ixqb")
        if cell is not None:
            out2 = torch.empty([2, rows, _qbix_op.N], dtype=q_a.dtype, device=q_a.device)
            return cute_dsl_q_b_ix_wq_b(q_a, w_fold, out2, cell)
        return projection(ph, q_a, w, handle.plain, wd8=wd8, dec_pre=dec_pre), None

    def reference(
        self,
        ph,
        q_a: torch.Tensor,
        w: torch.Tensor,
        w_fold: torch.Tensor | None,
        wd8,
        dec_pre: bool,
        handle: FoldHandle,
    ):
        """`q_a @ w` in fp32, rounded once to bf16 (`ixq` is the indexer projection's
        reference)."""
        return (q_a.float() @ w.float()).to(q_a.dtype), None


qkva_ix_fold = QkvaIxFold()
q_b_ix_wq_b = QbIxWqB()
