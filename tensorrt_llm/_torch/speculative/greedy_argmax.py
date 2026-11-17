# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-wise argmax of large `[rows, vocab]` logits (greedy sampling) in Triton.

`torch.argmax(logits, dim=-1)` runs one generic reduce kernel that reads fp32 logits of a
248320-entry vocabulary well under the HBM rate (GB300, graph replay, HBM-cold: 76 us at
256 rows, 216 at 1024). Here each row is split into `splits` chunks; one program reduces a
chunk to a (value, index) partial, and a second launch combines a row's partials. With one
split a row the first launch writes the result itself.

The result equals `torch.argmax` element for element: NaN ranks above every number (the
first NaN of a row wins), equal values (including -0.0 and +0.0) resolve to the lowest
index, and an all -inf row returns 0. bf16 / fp16 inputs are compared after an exact
widening to fp32.

Inside a CUDA graph a few rows (`small_rows=True`, at most `SMALL_ROWS`) take the Triton
kernels too, at `_SMALL_SPLITS` chunks a row: torch's generic reduce replays as a memset and a
31-block reduce kernel there (~11 us a call at 1-4 rows of 248320), two short launches here.
Eager callers keep torch.argmax for such inputs (fewer host launches).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Smallest `rows * vocab` routed to the Triton kernels; smaller inputs keep torch.argmax.
MIN_NUMEL = 1 << 22
#: Elements a program loads per iteration (4 warps: 16 a thread, four 16-byte fp32 loads).
_BLOCK = 2048
_WARPS = 4
_INT32_MAX = 2**31 - 1
#: `(min_rows, splits)`, largest first: chunks a row by the row count, so that a launch has
#: ~1000-4000 programs. GB300 sweep over splits x block x warps at 64-1024 rows of 248320.
_SPLITS = ((1024, 1), (512, 2), (256, 8), (128, 16), (0, 32))
#: The table for 16-bit logits (half the bytes a chunk): GB300 sweep over splits x block x
#: warps at 64-1024 rows of 248320 bf16, at this block and warp count.
_SPLITS_16BIT = ((512, 4), (128, 8), (64, 16), (0, 32))
#: Most rows `small_rows` callers route to the Triton kernels below `MIN_NUMEL`.
SMALL_ROWS = 16
#: Chunks a row for `small_rows` launches (GB300 graph replay, 248320 fp32 columns, 1-16 rows:
#: 2.4-5.0 us against 10.6-16.4 us for torch.argmax; 32 and 128 chunks within 0.5 us).
_SMALL_SPLITS = 64


@triton.jit
def _combine(v1, i1, v2, i2):
    """torch's `greater_or_nan` order: NaN first, then larger values, ties to the lower index."""
    n1 = v1 != v1
    n2 = v2 != v2
    lower = i1 < i2
    pick1 = tl.where(n1, (~n2) | lower, (~n2) & ((v1 > v2) | ((v1 == v2) & lower)))
    return tl.where(pick1, v1, v2), tl.where(pick1, i1, i2)


@triton.jit
def _argmax_chunk_kernel(
    x_ptr,
    val_ptr,
    idx_ptr,
    out_ptr,
    stride_row,
    n_cols,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    split = tl.program_id(1)
    base = x_ptr + row.to(tl.int64) * stride_row
    start = split * CHUNK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_cols
    acc_v = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
    acc_i = tl.where(mask, offs, 2147483647)
    # Each lane walks its column positions in increasing index order, so a later element
    # replaces the lane's running pick only when strictly larger, or NaN over a number.
    for k in tl.range(BLOCK, CHUNK, BLOCK):
        o = start + k + tl.arange(0, BLOCK)
        m = o < n_cols
        v = tl.load(base + o, mask=m, other=float("-inf")).to(tl.float32)
        take = (acc_v == acc_v) & ((v != v) | (v > acc_v))
        acc_v = tl.where(take, v, acc_v)
        acc_i = tl.where(take, o, acc_i)
    best_v, best_i = tl.reduce((acc_v, acc_i), 0, _combine)
    if SPLITS == 1:
        tl.store(out_ptr + row, best_i.to(out_ptr.dtype.element_ty))
    else:
        tl.store(val_ptr + row * SPLITS + split, best_v)
        tl.store(idx_ptr + row * SPLITS + split, best_i)


@triton.jit
def _argmax_final_kernel(val_ptr, idx_ptr, out_ptr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    offs = row * SPLITS + tl.arange(0, SPLITS)
    best_v, best_i = tl.reduce((tl.load(val_ptr + offs), tl.load(idx_ptr + offs)), 0, _combine)
    tl.store(out_ptr + row, best_i.to(out_ptr.dtype.element_ty))


def _table(dtype: torch.dtype) -> tuple:
    return _SPLITS if dtype == torch.float32 else _SPLITS_16BIT


def _splits(rows: int, n_cols: int, dtype: torch.dtype = torch.float32) -> int:
    if rows * n_cols < MIN_NUMEL:
        return _SMALL_SPLITS
    return next(splits for min_rows, splits in _table(dtype) if rows >= min_rows)


def _buffers(rows: int, splits: int, out: torch.Tensor):
    """The partial (value, index) buffers of a launch; with one split the result itself."""
    if splits == 1:
        return out, out
    return (
        torch.empty(rows * splits, dtype=torch.float32, device=out.device),
        torch.empty(rows * splits, dtype=torch.int32, device=out.device),
    )


_WARMED: set = set()


def _warm(logits: torch.Tensor, out_dtype: torch.dtype) -> None:
    """Compile every split variant for this input's specialization (dtype, vocab, row stride,
    output dtype) without launching, so that a later row count never compiles mid-serving."""
    key = (logits.dtype, logits.shape[1], logits.stride(0), out_dtype, logits.device)
    if key in _WARMED:
        return
    n_cols = logits.shape[1]
    out = torch.empty(1, dtype=out_dtype, device=logits.device)
    for splits in sorted({s for _, s in _table(logits.dtype)} | {_SMALL_SPLITS}):
        val, idx = _buffers(1, splits, out)
        chunk = triton.cdiv(triton.cdiv(n_cols, splits), _BLOCK) * _BLOCK
        _argmax_chunk_kernel.warmup(
            logits,
            val,
            idx,
            out,
            logits.stride(0),
            n_cols,
            CHUNK=chunk,
            BLOCK=_BLOCK,
            SPLITS=splits,
            num_warps=_WARPS,
            grid=(1,),
        )
        if splits > 1:
            _argmax_final_kernel.warmup(val, idx, out, SPLITS=splits, num_warps=1, grid=(1,))
    _WARMED.add(key)


def eligible(logits: torch.Tensor, small_rows: bool = False) -> bool:
    """Whether `greedy_argmax` takes the Triton kernels for `logits` (else torch.argmax):
    from `MIN_NUMEL` elements on, or with `small_rows` at 1 to `SMALL_ROWS` rows."""
    return (
        logits.is_cuda
        and logits.dim() == 2
        and logits.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and logits.stride(-1) == 1
        and (
            logits.numel() >= MIN_NUMEL
            or (small_rows and 0 < logits.shape[0] <= SMALL_ROWS and logits.shape[1] > 0)
        )
        and logits.shape[-1] < _INT32_MAX
    )


def greedy_argmax(
    logits: torch.Tensor, out_dtype: torch.dtype = torch.int64, small_rows: bool = False
) -> torch.Tensor:
    """`torch.argmax(logits, dim=-1).to(out_dtype)` for a 2-D `[rows, vocab]` tensor: the
    Triton kernels when `eligible(logits, small_rows)`, else torch.argmax. Graph-capturable
    (no host sync); callers pass `small_rows=True` for steps that a CUDA graph replays. The
    first call for an input specialization compiles every split variant (`_warm`); the row
    count then only selects among them."""
    if not eligible(logits, small_rows):
        return torch.argmax(logits, dim=-1).to(out_dtype)
    _warm(logits, out_dtype)
    rows, n_cols = logits.shape
    out = torch.empty(rows, dtype=out_dtype, device=logits.device)
    splits = _splits(rows, n_cols, logits.dtype)
    chunk = triton.cdiv(triton.cdiv(n_cols, splits), _BLOCK) * _BLOCK
    val, idx = _buffers(rows, splits, out)
    _argmax_chunk_kernel[(rows, splits)](
        logits,
        val,
        idx,
        out,
        logits.stride(0),
        n_cols,
        CHUNK=chunk,
        BLOCK=_BLOCK,
        SPLITS=splits,
        num_warps=_WARPS,
    )
    if splits > 1:
        _argmax_final_kernel[(rows,)](val, idx, out, SPLITS=splits, num_warps=1)
    return out
