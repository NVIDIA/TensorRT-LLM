# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`projection`: one of the target's projections, `x @ w`, on the kernel its handle holds for
the call's rows, with every fallback inside.

A projection's handle (`build`, one per projection name) holds what serves each row count:
the decode MXF8 GEMMs (1..64 rows, and a wide decode step's 65..192), the MXF8 context GEMM,
the bf16 decode cells by row count, the MTP decode cells (128 / 192) and the cells re-pinned
to the row counts between them, the next cell up for the decode counts without a cell of
their own, and the bf16 prefill GEMM; cuBLAS takes what none of them does. The re-pinned
runners are computed here at load (they compile nothing: `cute_dsl_projection.at_rows`), so
the forward only looks them up.

Which row counts map onto which cells depends on the calling Target's `Phase` (`ph`): only
its `wide` and `up` fields are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

from .._build import Count, try_build
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from . import cute_dsl_projection as _projection_op
from . import cute_dsl_projection_mxf8 as _mxf8_op
from . import cute_dsl_projection_mxf8_decode as _dec8_op
from .cute_dsl_projection import cute_dsl_projection
from .cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8
from .cute_dsl_projection_mxf8_decode import cute_dsl_projection_mxf8_decode

#: The decode kernels' widest compiled row count. Above it a projection with no compiled runner
#: takes the prefill GEMM; at or below it, cuBLAS.
DECODE_MAX_ROWS = 64

#: Every projection the target calls, by the name its cells were built under.
NAMES = (
    "qkva",
    "q_b",
    "o_proj",
    "ix_qb",
    "ix_fused",
    "router",
    "dense_gate_up",
    "dense_down",
    "shared_gate_up",
    "shared_down",
    "mtp_eh",
)

#: Projections an MTP3 / MTP5 captured decode step's rows without a cell keep on cuBLAS
#: instead of the next cell up (`cell_up`): there the cell is the slower one. Cold-weight
#: CUDA-graph timing on B300 (cuBLAS vs next cell up, us per call): ix_qb 12 rows 4.95 vs
#: 6.11, 18 rows 5.02 vs 5.42 (MTP 128 cell); dense_down 6 / 12 / 18 rows 8.15 / 8.25 / 8.29
#: vs 9.16 / 9.16 / 9.15. An eager draft step (`Phase.up == "draft"`) keeps its cells.
CELL_UP_KEEPS_CUBLAS = frozenset({"ix_qb", "dense_down"})


@dataclass(frozen=True)
class ProjectionHandle:
    """What `projection` launches for one projection name.

    `cells`: the bf16 decode (and context-sized) cells by row count. `mtp_cells`: the MTP
    decode cells (128 / 192), taken on a `Phase.wide` step only. `wide`: the row counts in
    65..192 without an MTP cell of their own, each on the next MTP cell up re-pinned to its
    exact rows (empty with `wide_bucket` off). `up`: the row counts up to 64 without a cell of
    their own, each on the next cell up (`cell_up`), None where none launches there.
    `prefill`: the bf16 prefill GEMM. `mxf8`: the MXF8 context GEMM; `dec8` / `dec8w`: the
    decode / wide decode MXF8 GEMMs. Any of these may be None or empty: that path is then not
    taken. `copy` serves cuBLAS results into a caller's buffer; `count` the route counters."""

    name: str
    cells: Mapping[int, Any]
    mtp_cells: Mapping[int, Any]
    wide: Mapping[int, Any]
    wide_bucket: bool
    up: Mapping[int, Any]
    keeps_cublas: bool
    prefill: Any
    mxf8: Any
    dec8: Any
    dec8w: Any
    copy: RowCopyHandle
    count: Count


def build_mxf8(name: str, weight: _mxf8_op.QuantizedWeight) -> _mxf8_op.Mxf8ProjectionRunner | None:
    """The MXF8 context GEMM of `name` (any layer's quantized weight), or None with a warning
    (those rows keep the bf16 prefill GEMM)."""
    return try_build(f"MXF8 {name}", lambda: _mxf8_op.build(name, weight))


def build_dec8(name: str, weight: _dec8_op.QuantizedWeight) -> _dec8_op.DecodeMxf8Runner | None:
    """The decode MXF8 GEMM of `name` (1..64 rows), or None with a warning."""
    return try_build(f"decode MXF8 {name}", lambda: _dec8_op.build(name, weight))


def build_dec8_wide(
    name: str, weight: _dec8_op.QuantizedWeight
) -> _dec8_op.DecodeMxf8Runner | None:
    """The wide decode MXF8 GEMM of `name` (65..192 rows), or None with a warning."""
    return try_build(f"wide decode MXF8 {name}", lambda: _dec8_op.build_wide(name, weight))


def wide_cells(
    cells: Mapping[int, Any], at_rows: Callable[[Any, int], Any] | None = None
) -> dict[int, Any]:
    """For every row count in 65..max(`cells`) without a cell of its own, the smallest cell
    above it launched on its exact rows: the cells are compiled on dynamic layouts, so the
    tiles past the last row load zeros and store nothing, and each row is bitwise that cell's
    for it. `at_rows(cell, rows)` re-pins a runner pinned to its own count (None: the cell
    takes any count up to its own)."""
    out = {}
    for rows in range(DECODE_MAX_ROWS + 1, max(cells, default=0)):
        if rows in cells:
            continue
        bucket = min(m for m in cells if m > rows)
        out[rows] = cells[bucket] if at_rows is None else at_rows(cells[bucket], rows)
    return out


def wide_cell(
    cells: Mapping[int, Any], wide: Mapping[int, Any], wide_bucket: bool, rows: int, count: Count
):
    """The MTP decode cell for a `Phase.wide` call's `rows` (above 64): the cell built at
    `rows`, else (`wide_bucket`) the next one up re-pinned at load (`wide_cells`), counted
    `dec_wide_rows`. None: the caller's fallback runs."""
    cell = cells.get(rows)
    if cell is None and wide_bucket:
        cell = wide.get(rows)
        count("dec_wide_rows", cell is not None)
    return cell


def up_cells(
    cells: Mapping[int, Any], mtp_cells: Mapping[int, Any], at_rows: Callable[[Any, int], Any]
) -> dict[int, Any]:
    """For every row count up to 64 without a cell of its own, the cell it runs on where
    `cell_up` holds: the smallest decode cell above it up to 64 that launches at a runtime
    row count, else the smallest such MTP cell (128 / 192); None where none does. A None
    from `at_rows` (a cell compiled for its own M) moves to the next cell up. The
    context-sized cells (1099 rows and up) are never taken: their tiles past the last row
    would each read the weight again. Each row is bitwise the serving cell's for it."""
    out = {}
    for rows in range(1, DECODE_MAX_ROWS + 1):
        if rows in cells:
            continue
        cell = None
        ups = [cells[m] for m in sorted(cells) if rows < m <= DECODE_MAX_ROWS]
        ups += [mtp_cells[m] for m in sorted(mtp_cells) if m > rows]
        for candidate in ups:
            cell = at_rows(candidate, rows)
            if cell is not None:
                break
        out[rows] = cell
    return out


def build(
    name: str,
    *,
    cells: Mapping[int, Any],
    mtp_cells: Mapping[int, Any],
    prefill: Any,
    mxf8: Any,
    dec8: Any,
    dec8w: Any,
    wide_bucket: bool,
    copy: RowCopyHandle,
    count: Count,
) -> ProjectionHandle:
    """`name`'s handle over the runners built at load (`cute_dsl_projection.build_all` /
    `build_prefill_all`, `build_mxf8`, `build_dec8`, `build_dec8_wide`). Re-pins the cells
    to every row count a step can present between them; compiles nothing."""
    at_rows = _projection_op.at_rows
    return ProjectionHandle(
        name=name,
        cells=dict(cells),
        mtp_cells=dict(mtp_cells),
        wide=wide_cells(mtp_cells, at_rows) if wide_bucket else {},
        wide_bucket=wide_bucket,
        up=up_cells(cells, mtp_cells, at_rows),
        keeps_cublas=name in CELL_UP_KEEPS_CUBLAS,
        prefill=prefill,
        mxf8=mxf8,
        dec8=dec8,
        dec8w=dec8w,
        copy=copy,
        count=count,
    )


def cell_up(ph, rows: int) -> bool:
    """Whether a call's `rows` (at most 64) without a decode cell of their own run on the next
    cell up at their exact rows: an eager draft step's (`ph.up == "draft"`), and an MTP3 /
    MTP5 decode step's (`ph.up == "decode"`, `TRTLLM_GLM52_DEC_CELL_UP`). The latter is
    graph-captured: `rows` is fixed per capture shape and the cell is fixed per row count, so
    each graph freezes one choice with no host sync. Rows with a cell never ask."""
    return rows <= DECODE_MAX_ROWS and ph.up is not None


def quant_reads(x: torch.Tensor) -> bool:
    """An activation the MXF8 GEMMs' quant reads: bf16, unit inner stride, 16-byte rows."""
    return (
        x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def dec8_takes(x: torch.Tensor, wd8, handle: ProjectionHandle) -> bool:
    """Whether `projection` sends `x` to the decode MXF8 GEMM: its runner built, `wd8` given,
    a decode row count, and an activation its quant reads."""
    return (
        wd8 is not None
        and handle.dec8 is not None
        and 0 < int(x.shape[0]) <= DECODE_MAX_ROWS
        and quant_reads(x)
    )


def dec8w_takes(ph, x: torch.Tensor, wd8, handle: ProjectionHandle) -> bool:
    """Whether `projection` sends `x` to the wide decode MXF8 GEMM: a `Phase.wide` step's rows
    above 64 (up to 192) of a call whose decode path is the decode MXF8 GEMM (`wd8`: the
    trunk o_proj), its wide runner built, and an activation its quant reads."""
    return (
        ph.wide
        and wd8 is not None
        and handle.dec8w is not None
        and DECODE_MAX_ROWS < int(x.shape[0]) <= _dec8_op.WIDE_ROWS[1]
        and quant_reads(x)
    )


def takes_decode_mxf8(ph, x: torch.Tensor, handle: ProjectionHandle) -> bool:
    """Whether a call that passes its decode MXF8 weight (`wd8`) sends `x` to the decode or the
    wide decode MXF8 GEMM (`dec8_takes` or `dec8w_takes`): where a producer writes that
    GEMM's operand, whether it writes it for these rows."""
    rows = int(x.shape[0])
    if not quant_reads(x):
        return False
    if 0 < rows <= DECODE_MAX_ROWS:
        return handle.dec8 is not None
    return ph.wide and handle.dec8w is not None and DECODE_MAX_ROWS < rows <= _dec8_op.WIDE_ROWS[1]


def mxf8_takes(ph, x: torch.Tensor, w8, wd8, handle: ProjectionHandle) -> bool:
    """Whether `projection` sends `x` to the MXF8 context GEMM: its runner built, `w8` given,
    a row count above the decode cells', and an activation the quant kernel reads. On a
    `Phase.wide` step only where the call's decode path is MXF8 as well (`wd8` and its decode
    runner: the trunk o_proj) and its wide decode GEMM (`dec8w_takes`) did not build, so
    decode rows keep that path's numerics -- the same e4m3 operands, block scales and K order
    (B300: the two GEMMs' outputs are bitwise equal at 64 rows)."""
    return (
        handle.mxf8 is not None
        and w8 is not None
        and int(x.shape[0]) > DECODE_MAX_ROWS
        and (not ph.wide or (wd8 is not None and handle.dec8 is not None))
        and quant_reads(x)
    )


def _bf16_out(out_dtype, out) -> bool:
    return out_dtype in (None, torch.bfloat16) and (
        out is None or (out.dtype == torch.bfloat16 and out.is_contiguous())
    )


class Projection(OpWrapper):
    """`x @ w_colmajor`, on the handle's kernel for the call's rows.

    The kernels read `w_colmajor.t()`, the row-major weight, same memory. `out` is an optional
    caller-owned `[rows, N]` buffer; on the cuBLAS path the result is copied in. `w8` is the
    layer's MXF8 copy of the weight (qkva / o_proj / dense gate|up): above the decode row
    counts the MXF8 context GEMM takes the call instead of the prefill GEMM. `pre` is `x`'s
    MXF8 operand written by its producer: the GEMM then skips its activation quant; a caller
    passes it only where `mxf8_takes` holds. `wd8` is the layer's decode MXF8 weight (q_b /
    o_proj): decode rows take the decode MXF8 GEMM; `dec_pre` says its producer wrote the
    operand (`dec8_takes` or `dec8w_takes` must hold). On a `Phase.wide` step, rows above 64
    keep the numerics of the call's decode path: where that is the decode MXF8 GEMM, the wide
    decode MXF8 GEMM or the context MXF8 GEMM, bitwise the same output; else the MTP decode
    cell at its row count or the next one up, else the bf16 prefill GEMM (or cuBLAS). Rows up
    to 64 without a cell of their own run on the next cell up where `cell_up` holds (but for
    `CELL_UP_KEEPS_CUBLAS` off an eager draft step), else cuBLAS.

    Counted: `dec_proj_mxf8[w][_pre]`, `ctx_proj_mxf8[_pre]`, `dec_wide` / `dec_wide_rows`,
    `draft_cell` / `dec_cell_up`."""

    def __call__(
        self,
        ph,
        x: torch.Tensor,
        w_colmajor: torch.Tensor,
        handle: ProjectionHandle,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        w8=None,
        pre=None,
        wd8=None,
        dec_pre: bool = False,
    ) -> torch.Tensor:
        rows = int(x.shape[0])
        wide8 = dec8w_takes(ph, x, wd8, handle)
        if (wide8 or dec8_takes(x, wd8, handle)) and _bf16_out(out_dtype, out):
            runner = handle.dec8w if wide8 else handle.dec8
            if out is None:
                out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=x.device)
            tag = "dec_proj_mxf8w" if wide8 else "dec_proj_mxf8"
            handle.count(f"{tag}_pre" if dec_pre else tag, True)
            return cute_dsl_projection_mxf8_decode(x, wd8, out, runner, pre=dec_pre)
        assert not dec_pre, f"{handle.name}: a producer's decode operand reached another path"
        if mxf8_takes(ph, x, w8, wd8, handle) and _bf16_out(out_dtype, out):
            if out is None:
                out = torch.empty([rows, handle.mxf8.n], dtype=torch.bfloat16, device=x.device)
            handle.count("ctx_proj_mxf8", True)
            if pre is not None:
                handle.count("ctx_proj_mxf8_pre", True)
            return cute_dsl_projection_mxf8(x, w8, out, handle.mxf8, pre=pre)
        assert pre is None, f"{handle.name}: a producer's MXF8 operand reached a non-MXF8 path"
        runner = handle.cells.get(rows)
        if ph.wide and rows > DECODE_MAX_ROWS:
            runner = handle.mtp_cells.get(rows)
            if runner is None and handle.wide_bucket:
                runner = handle.wide.get(rows)
                handle.count("dec_wide_rows", runner is not None)
            handle.count("dec_wide", runner is not None)
        if runner is None and cell_up(ph, rows) and (ph.up == "draft" or not handle.keeps_cublas):
            runner = handle.up.get(rows)
            handle.count("draft_cell" if ph.up == "draft" else "dec_cell_up", runner is not None)
        prefill = handle.prefill
        if (
            runner is None
            and prefill is not None
            and DECODE_MAX_ROWS < rows <= prefill.max_rows
            and x.dtype == prefill.in_dtype
            and x.is_contiguous()
            and out_dtype in (None, prefill.out_dtype)
            and (out is None or out.dtype == prefill.out_dtype)
        ):
            runner = prefill
        if runner is None:
            res = cublas_mm(x, w_colmajor, None, out_dtype)
            if out is None:
                return res
            return row_copy(out, res, handle.copy)
        if out is None:
            out = torch.empty([rows, runner.n], dtype=runner.out_dtype, device=x.device)
        return cute_dsl_projection(x, w_colmajor.t(), out, runner)

    def reference(
        self,
        ph,
        x: torch.Tensor,
        w_colmajor: torch.Tensor,
        handle: ProjectionHandle,
        out_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
        w8=None,
        pre=None,
        wd8=None,
        dec_pre: bool = False,
    ) -> torch.Tensor:
        """`x @ w_colmajor` in fp32, rounded once to `out_dtype` (default `x`'s). The MXF8
        paths quantize both operands to e4m3 first: their band against this is the
        quantization's. Every argument but `x`, `w_colmajor` and `out_dtype` is ignored."""
        res = x.float() @ w_colmajor.float()
        return res.to(out_dtype or x.dtype)


projection = Projection()


def projection_tail(
    ph, x: torch.Tensor, start: int, w_colmajor: torch.Tensor, handle: ProjectionHandle
) -> torch.Tensor:
    """`projection` of `x[start:]`, a mixed step's generation rows after its context rows.
    Their count is any 1..63, which the decode cells rarely match, and a miss goes to cuBLAS;
    so the smallest cell `b` above the count runs on the trailing `b` rows of `x` (the last
    `b - rows` context rows ride along, their outputs unused) and the result is the view of
    its last rows: no padding copy. Counted `<name>_tail`."""
    total = int(x.shape[0])
    rows = total - start
    cells = handle.cells
    if rows <= DECODE_MAX_ROWS and rows not in cells:
        bucket = min((m for m in cells if m >= rows), default=None)
        if bucket is not None and bucket <= total:
            handle.count(f"{handle.name}_tail", True)
            return projection(ph, x[total - bucket :], w_colmajor, handle)[bucket - rows :]
    return projection(ph, x[start:], w_colmajor, handle)
