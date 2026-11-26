# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`indexer_pre_proj`: an indexer layer's key / head-weight projection of the layer input,
`(keys [rows, 128], head_weights [rows, 32] fp32)`, on the split-output GEMM its handle holds
for the call's rows (`cute_dsl_indexer_pre_proj_split`), with the fp32 projection as the
fallback.

The handle (`build`, over the kernels `compile_cells` built at load) holds the decode cells
by row count (1..64), the MTP decode cells (128 / 192) and the runtime-M cell that serves the
row counts between them on a wide decode step and the decode counts without a cell of their
own (`projection.cell_up`), and the prefill GEMM for the context row counts. With
`TRTLLM_GLM52_CTX_IX_NOSPLIT` (default 1) the prefill GEMM's fp32 `[rows, 160]` output is not
split: the two results are its column views, row pitch 160, which the indexer's q side and K
write read in place (`indexer_q_side` picks its pitch-160 launcher by that pitch).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count, build_or_reason, try_build
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from . import cute_dsl_indexer_pre_proj_split as _split_op
from .cute_dsl_indexer_pre_proj_split import cute_dsl_indexer_pre_proj_split
from .projection import (
    DECODE_MAX_ROWS,
    ProjectionHandle,
    cell_up,
    projection,
    wide_cell,
    wide_cells,
)


@dataclass(frozen=True)
class IndexerPreProjHandle:
    """What `indexer_pre_proj` launches. `cells`: the decode split cells by row count;
    `mtp_cells`: the MTP decode cells, taken on a `Phase.wide` step only; `wide`: the row
    counts in 65..192 without an MTP cell of their own, each on the runtime-M cell (None
    where it does not reach; empty with `wide_bucket` off); `ragged`: that runtime-M cell
    (None: not built), which also serves the decode counts without a cell where `cell_up`
    holds; `prefill`: the prefill split GEMM (None: those rows take the fp32 projection);
    `unsplit`: the prefill GEMM's rows come back as column views of one fp32 output;
    `fused`: the fp32 fallback projection's handle; `key_dim` / `heads` the two widths."""

    cells: Mapping[int, Any]
    mtp_cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    ragged: Any
    prefill: Any
    unsplit: bool
    fused: ProjectionHandle
    key_dim: int
    heads: int
    copy: RowCopyHandle
    count: Count


def compile_cells(dev: int, wide: bool, weight_bf16: torch.Tensor) -> SimpleNamespace:
    """The split GEMM's kernels, compiled at load: the decode cells, and with `wide` (an
    engine with the wide decode cells) the MTP cells and the runtime-M cell; the prefill GEMM
    against any indexer layer's bf16 `[160, hidden]` weight (every layer shares it). A cell
    whose build fails is left out (its rows take the next path down, with a warning)."""
    prefill, _ = build_or_reason(
        "indexer prefill split", lambda: _split_op.build_prefill(weight_bf16)
    )
    return SimpleNamespace(
        cells=_split_op.compile_all(dev),
        mtp_cells=_split_op.compile_all(dev, _split_op.MTP_ROWS) if wide else {},
        ragged=(
            try_build("indexer split at runtime M", lambda: _split_op.compile_ragged(dev))
            if wide
            else None
        ),
        prefill=prefill,
        unsplit=(prefill is not None and os.environ.get("TRTLLM_GLM52_CTX_IX_NOSPLIT", "1") == "1"),
    )


def build(
    kernels: SimpleNamespace,
    fused: ProjectionHandle,
    key_dim: int,
    heads: int,
    wide_bucket: bool,
    copy: RowCopyHandle,
    count: Count,
) -> IndexerPreProjHandle:
    """The handle over `compile_cells`' kernels; re-pins the runtime-M cell to the row counts
    between the MTP cells at load (compiles nothing)."""
    rag = kernels.ragged

    def at_rows(cell, rows: int):
        return rag if rag is not None and rows <= rag.max_m else None

    return IndexerPreProjHandle(
        cells=dict(kernels.cells),
        mtp_cells=dict(kernels.mtp_cells),
        wide=wide_cells(kernels.mtp_cells, at_rows) if wide_bucket else {},
        wide_bucket=wide_bucket,
        ragged=rag,
        prefill=kernels.prefill,
        unsplit=kernels.unsplit,
        fused=fused,
        key_dim=key_dim,
        heads=heads,
        copy=copy,
        count=count,
    )


def _cell(ph, x: torch.Tensor, rows: int, handle: IndexerPreProjHandle):
    """The split cell for `rows`: a `Phase.wide` step's MTP cell or the runtime-M cell between
    them (counted `dec_wide`), else the decode cell, else (no cell of its own, `cell_up`) the
    runtime-M cell (counted `draft_cell` / `dec_cell_up`), else above 64 rows the prefill GEMM
    for a contiguous `x`. None: the fp32 projection."""
    h = handle
    if ph.wide and rows > DECODE_MAX_ROWS:
        cell = wide_cell(h.mtp_cells, h.wide, h.wide_bucket, rows, h.count)
        h.count("dec_wide", cell is not None)
    else:
        cell = h.cells.get(rows)
        if cell is None and cell_up(ph, rows):
            rag = h.ragged
            cell = rag if rag is not None and rows <= rag.max_m else None
            h.count("draft_cell" if ph.up == "draft" else "dec_cell_up", cell is not None)
    pre = h.prefill
    if (
        cell is None
        and pre is not None
        and DECODE_MAX_ROWS < rows <= pre.max_rows
        and x.is_contiguous()
    ):
        cell = pre
    return cell


class IndexerPreProj(OpWrapper):
    """`(keys, head_weights)` of the layer input `x` (`w_fused` the layer's fp32-path
    `[hidden, 160]` weight, `w_bf16` its bf16 `[160, hidden]` split-GEMM weight). `kw`: the two
    results where the caller's GEMM already wrote them (`qkva_ix_fold`), returned as they are.
    Else on the split cell for the rows (`_cell`); with `unsplit` the prefill GEMM's rows are
    fp32 column views of one `[rows, 160]` output (counted `ctx_ix_nosplit`). Else the fp32
    projection, its keys copied to bf16 and its head weights made contiguous (the scorers
    check `weights`' stride(1) but not stride(0), then read it densely)."""

    def __call__(
        self,
        ph,
        x: torch.Tensor,
        w_fused: torch.Tensor,
        w_bf16: torch.Tensor,
        kw,
        handle: IndexerPreProjHandle,
    ):
        if kw is not None:
            return kw
        h = handle
        rows = int(x.shape[0])
        cell = _cell(ph, x, rows, h)
        if cell is not None and cell is h.prefill and h.unsplit:
            h.count("ctx_ix_nosplit", True)
            return _split_op.project_unsplit(x, w_bf16, cell)
        kb = torch.empty([rows, h.key_dim], dtype=x.dtype, device=x.device)
        if cell is not None:
            wq = torch.empty([rows, h.heads], dtype=torch.float32, device=x.device)
            cute_dsl_indexer_pre_proj_split(x, w_bf16, kb, wq, cell)
            return kb, wq
        # fp32: the only `weights` dtype the scorers take.
        fused = projection(ph, x.float(), w_fused, h.fused, torch.float32)
        k_raw, wq_raw = torch.split(fused, [h.key_dim, h.heads], -1)
        row_copy(kb, k_raw, h.copy)
        return kb, wq_raw.contiguous()

    def reference(
        self,
        ph,
        x: torch.Tensor,
        w_fused: torch.Tensor,
        w_bf16: torch.Tensor,
        kw,
        handle: IndexerPreProjHandle,
    ):
        """`x @ w_bf16.T` in fp32, split into bf16 keys and fp32 head weights (`ph`, `kw`
        ignored)."""
        full = x.float() @ w_bf16.float().t()
        k, wq = torch.split(full, [handle.key_dim, handle.heads], -1)
        return k.to(x.dtype), wq.contiguous()


indexer_pre_proj = IndexerPreProj()
