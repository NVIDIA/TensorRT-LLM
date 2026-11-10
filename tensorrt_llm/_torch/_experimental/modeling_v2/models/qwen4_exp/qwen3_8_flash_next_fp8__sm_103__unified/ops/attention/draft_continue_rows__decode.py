# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generation-only draft-extend step's continuing rows and their operands, in two launches.

The draft layer's draft-extend step (`p` rows a request) runs its attention core and tail only on
each request's continuing row, `r * p + clamp(num_accepted[r] - 1, 0, p - 1)`:
`draft_continue_select` forms those rows and gathers the QSA row metadata at them;
`draft_rows_gather` gathers the step's 2-D operands at them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Requests a program of `draft_continue_select` covers.
SEQ_BLOCK = 256
#: Columns a program of `draft_rows_gather` copies.
COL_BLOCK = 1024
#: Operands one `draft_rows_gather` launch takes.
MAX_OPERANDS = 6


@triton.jit(do_not_specialize=["num_seqs", "rows_per_request"])
def _continue_select_kernel(
    accepted,
    v0,
    v1,
    v2,
    v3,
    rows_out,
    o0,
    o1,
    o2,
    o3,
    num_seqs,
    rows_per_request,
    BLOCK: tl.constexpr,
):
    seqs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = seqs < num_seqs
    last = tl.load(accepted + seqs, mask=valid, other=1).to(tl.int64) - 1
    last = tl.minimum(tl.maximum(last, 0), rows_per_request - 1)
    rows = seqs.to(tl.int64) * rows_per_request + last
    tl.store(rows_out + seqs, rows, mask=valid)
    tl.store(o0 + seqs, tl.load(v0 + rows, mask=valid), mask=valid)
    tl.store(o1 + seqs, tl.load(v1 + rows, mask=valid), mask=valid)
    tl.store(o2 + seqs, tl.load(v2 + rows, mask=valid), mask=valid)
    tl.store(o3 + seqs, tl.load(v3 + rows, mask=valid), mask=valid)


@triton.jit
def _copy_row(src, dst, s_src, row, idx, width, cols):
    mask = cols < width
    tl.store(dst + row * width + cols, tl.load(src + idx * s_src + cols, mask=mask), mask=mask)


@triton.jit(
    do_not_specialize=[
        "w0", "w1", "w2", "w3", "w4", "w5", "s0", "s1", "s2", "s3", "s4", "s5",
    ],
    do_not_specialize_on_alignment=["x0", "x1", "x2", "x3", "x4", "x5"],
)  # fmt: skip
def _rows_gather_kernel(
    index,
    x0, x1, x2, x3, x4, x5,
    y0, y1, y2, y3, y4, y5,
    w0, w1, w2, w3, w4, w5,
    s0, s1, s2, s3, s4, s5,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    which = tl.program_id(1)
    cols = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    idx = tl.load(index + row).to(tl.int64)
    if which == 0:
        _copy_row(x0, y0, s0, row, idx, w0, cols)
    if COUNT > 1:
        if which == 1:
            _copy_row(x1, y1, s1, row, idx, w1, cols)
    if COUNT > 2:
        if which == 2:
            _copy_row(x2, y2, s2, row, idx, w2, cols)
    if COUNT > 3:
        if which == 3:
            _copy_row(x3, y3, s3, row, idx, w3, cols)
    if COUNT > 4:
        if which == 4:
            _copy_row(x4, y4, s4, row, idx, w4, cols)
    if COUNT > 5:
        if which == 5:
            _copy_row(x5, y5, s5, row, idx, w5, cols)


def _select_spec(seqs: int, p: int) -> dict:
    return dict(seqs=seqs, rows_per_request=p)


class DraftContinueSelect(OpWrapper):
    """`rows, (g0, g1, g2, g3) = draft_continue_select(accepted, num_seqs, p, (v0, v1, v2, v3))`:
    `rows[r] = r * p + clamp(accepted[r] - 1, 0, p - 1)` (int64, `[num_seqs]`) and
    `g_k[r] = v_k[rows[r]]` (fresh, `v_k`'s dtype), for `r < num_seqs`.

    - `accepted`: the worker's accepted counts (`qsa_mtp_num_accepted`, int32, contiguous);
    - `v_k`: 1-D contiguous row vectors of at least `num_seqs * p` entries (the QSA row metadata).

    Bitwise `_draft_continue_rows` followed by four `index_select`s."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="tep2 HT draft-extend step: 192 requests (the largest batch)", spec=_select_spec(192, 3)),
        Cell(why="two programs: 300 requests, 4 rows each", spec=_select_spec(300, 4)),
    )  # fmt: skip

    note = """
    Source: one Triton kernel (`_continue_select_kernel`), grid `ceil(num_seqs / 256)`; the
    request count and rows a request are runtime arguments (one compile per dtype set).
    """

    def __call__(self, accepted, num_seqs: int, rows_per_request: int, vectors):
        dev = accepted.device
        rows = torch.empty([num_seqs], dtype=torch.int64, device=dev)
        outs = tuple(torch.empty([num_seqs], dtype=v.dtype, device=dev) for v in vectors)
        if num_seqs:
            _continue_select_kernel[(triton.cdiv(num_seqs, SEQ_BLOCK),)](
                accepted, *vectors, rows, *outs, num_seqs, rows_per_request, BLOCK=SEQ_BLOCK,
                num_warps=4,
            )  # fmt: skip
        return rows, outs

    def reference(self, accepted, num_seqs: int, rows_per_request: int, vectors):
        last = torch.clamp(accepted[:num_seqs].to(torch.int64) - 1, min=0, max=rows_per_request - 1)
        rows = torch.arange(num_seqs, dtype=torch.int64, device=accepted.device) * rows_per_request
        rows = rows + last
        return rows, tuple(torch.index_select(v, 0, rows) for v in vectors)

    def is_valid(self, accepted, num_seqs: int, rows_per_request: int, vectors) -> None:
        assert len(vectors) == 4 and rows_per_request >= 1
        assert accepted.is_contiguous() and accepted.numel() >= num_seqs
        for v in vectors:
            assert v.dim() == 1 and v.is_contiguous() and v.numel() >= num_seqs * rows_per_request


def _gather_spec(rows: int, widths: tuple) -> dict:
    return dict(rows=rows, widths=widths)


class DraftRowsGather(OpWrapper):
    """`outs = draft_rows_gather(index, operands)`: `outs[k] = operands[k][index]` for up to six
    2-D operands, each with unit column stride (any row stride), as fresh contiguous tensors of
    `[index.numel(), operands[k].shape[1]]`.

    Bitwise `index_select(operands[k], 0, index)`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tep2 HT draft-extend step at 64 requests: index query, QKV, gate, residual, inject",
            spec=_gather_spec(64, (512, 2048, 1536, 10240, 4)),
        ),
        Cell(why="one operand, 7 rows", spec=_gather_spec(7, (300,))),
    )

    note = """
    Source: one Triton kernel (`_rows_gather_kernel`), grid `(rows, operands, ceil(max width /
    1024))`; widths and row strides are runtime arguments, the operand count a constexpr.
    """

    def __call__(self, index, operands):
        count = len(operands)
        assert 1 <= count <= MAX_OPERANDS
        rows = index.numel()
        outs = [torch.empty([rows, t.shape[1]], dtype=t.dtype, device=t.device) for t in operands]
        if rows:
            pad = MAX_OPERANDS - count
            xs = list(operands) + [operands[0]] * pad
            ys = outs + [outs[0]] * pad
            ws = [t.shape[1] for t in operands] + [0] * pad
            ss = [t.stride(0) for t in operands] + [0] * pad
            width = max(t.shape[1] for t in operands)
            _rows_gather_kernel[(rows, count, triton.cdiv(width, COL_BLOCK))](
                index, *xs, *ys, *ws, *ss, COUNT=count, BLOCK=COL_BLOCK, num_warps=4
            )
        return tuple(outs)

    def reference(self, index, operands):
        return tuple(torch.index_select(t, 0, index) for t in operands)

    def is_valid(self, index, operands) -> None:
        assert 1 <= len(operands) <= MAX_OPERANDS and index.dim() == 1
        for t in operands:
            assert t.dim() == 2 and (t.stride(1) == 1 or t.shape[1] == 1)
            assert t.device == index.device


draft_continue_select = DraftContinueSelect()
draft_rows_gather = DraftRowsGather()
