# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row gather / scatter of byte rows for the GLM-5.2 MTP module's context step.

On a context step the MTP layer's output is read at only a few rows: the last
token of each context request and the generation rows (the draft head gathers
them).  Everything after the layer's KV write is row-local, so the step can run
attention-to-output on those rows alone.  That needs two plain data moves on
rows of arbitrary dtype and width, which this module provides:

* ``gather``:  ``y[i] = x[rows[i]]``  (compact ``y``)
* ``scatter``: ``y[rows[i]] = x[i]``  (rows of ``y`` not named are untouched)

Rows are moved as raw bytes (16-byte vectors), so the result is bitwise the
source row for any dtype.  Operands are 2-D with a contiguous inner dimension:
the row width in bytes and the row pitch in bytes must be multiples of 16 and
the base pointers 16-byte aligned (the FP8 query rows ``[T, 16, 576]`` -> 9216
bytes, BF16 hidden rows 12288 bytes, ...).  One 256-thread CTA per moved row;
each thread copies 16-byte vectors with a 256-vector stride.

Indices are not range-checked on the device (a check would need a host sync);
callers pass host-built in-range plans.

Pattern: glm-rows-select
Device: SM100+ (Blackwell)
Shape: Runtime M rows x W bytes (W % 16 == 0)
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

_THREADS = 256
# One vector = four Int32 = 16 bytes.
_VEC = 4
_ALIGNMENT_BYTES = 16

CompiledKernel = Callable[..., object]
_COMPILED: dict[tuple, CompiledKernel] = {}


@cute.jit
def _load4(ptr, offset):
    values = cute.make_rmem_tensor((_VEC,), cutlass.Int32)
    src = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(src, values)
    return values


@cute.jit
def _store4(ptr, offset, values) -> None:
    dst = cute.make_tensor(ptr + cute.assume(offset, _VEC), cute.make_layout((_VEC,)))
    cute.autovec_copy(values, dst)


@cute.kernel
def _select_kernel(
    x: cute.Tensor,
    rows: cute.Tensor,
    y: cute.Tensor,
    x_row_stride: cutlass.Int32,
    y_row_stride: cutlass.Int32,
    num_vec: cutlass.Int32,
    scatter: cutlass.Constexpr[bool],
) -> None:
    i, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    r = cutlass.Int64(rows[i])
    # gather: source row rows[i] -> destination row i; scatter: the reverse
    src_row = cutlass.Int64(i) if cutlass.const_expr(scatter) else r
    dst_row = r if cutlass.const_expr(scatter) else cutlass.Int64(i)
    src_base = src_row * cutlass.Int64(x_row_stride)
    dst_base = dst_row * cutlass.Int64(y_row_stride)
    for v in cutlass.range(tidx, num_vec, _THREADS, unroll=1):
        col = cutlass.Int64(v) * _VEC
        _store4(y.iterator, dst_base + col, _load4(x.iterator, src_base + col))


@cute.jit
def _launch_select(
    x: cute.Tensor,
    rows: cute.Tensor,
    y: cute.Tensor,
    x_row_stride: cutlass.Int32,
    y_row_stride: cutlass.Int32,
    num_vec: cutlass.Int32,
    stream: cuda.CUstream,
    scatter: cutlass.Constexpr[bool],
) -> None:
    _select_kernel(x, rows, y, x_row_stride, y_row_stride, num_vec, scatter).launch(
        grid=(cute.size(rows, mode=[0]), 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def _fake_rows():
    """[N, W] Int32 with a runtime row pitch (in Int32 elements), 16-byte aligned."""

    return make_fake_tensor(
        cutlass.Int32,
        (cute.sym_int32(), cute.sym_int32(divisibility=_VEC)),
        stride=(cute.sym_int32(divisibility=_VEC), 1),
        assumed_align=_ALIGNMENT_BYTES,
    )


def compile_select(scatter: bool = False) -> CompiledKernel:
    """``fn(x[N,W] i32, rows[M] i32, y[*,W] i32, x_pitch, y_pitch, num_vec, stream)``."""

    key = ("select", bool(scatter))
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            _launch_select,
            _fake_rows(),
            make_fake_compact_tensor(cutlass.Int32, (cute.sym_int32(),), assumed_align=4),
            _fake_rows(),
            cutlass.Int32(_VEC),
            cutlass.Int32(_VEC),
            cutlass.Int32(1),
            make_fake_stream(),
            bool(scatter),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def as_words(t: torch.Tensor) -> torch.Tensor | None:
    """``t`` (2-D, any dtype) reinterpreted as Int32 words ``[rows, bytes // 4]``, or None
    when it is not a row block this kernel moves: CUDA, inner dimension contiguous, row width
    and row pitch multiples of 16 bytes, base 16-byte aligned."""

    if not t.is_cuda or t.ndim != 2 or t.shape[0] < 1 or t.stride(1) != 1:
        return None
    es = t.element_size()
    if (t.shape[1] * es) % _ALIGNMENT_BYTES or (t.stride(0) * es) % _ALIGNMENT_BYTES:
        return None
    if t.data_ptr() % _ALIGNMENT_BYTES:
        return None
    return _words_view(t, t.shape[1] * es // 4, t.stride(0) * es // 4)


def _words_view(t: torch.Tensor, words: int, pitch: int) -> torch.Tensor:
    """An Int32 view of ``t``'s storage: ``[rows, words]`` with a pitch of ``pitch`` words
    (no copy; the 16-byte alignment checked above makes the byte offset a whole word)."""

    return torch.empty(0, dtype=torch.int32, device=t.device).set_(
        t.untyped_storage(),
        t.storage_offset() * t.element_size() // 4,
        (t.shape[0], words),
        (pitch, 1),
    )


def _check(x: torch.Tensor, rows: torch.Tensor, y: torch.Tensor):
    xw, yw = as_words(x), as_words(y)
    if (
        xw is None
        or yw is None
        or xw.shape[1] != yw.shape[1]
        or rows.dtype != torch.int32
        or rows.ndim != 1
        or not rows.is_contiguous()
        or not rows.is_cuda
        or rows.device != x.device
        or y.device != x.device
    ):
        raise ValueError("mtp_rows_select: unsupported operands")
    return xw, yw


def gather(fn: CompiledKernel, x: torch.Tensor, rows: torch.Tensor, y: torch.Tensor) -> None:
    """``y[i] = x[rows[i]]`` for ``i < len(rows)`` (``fn`` from ``compile_select(False)``)."""

    xw, yw = _check(x, rows, y)
    if yw.shape[0] < rows.numel():
        raise ValueError("mtp_rows_select.gather: y has fewer rows than rows")
    if rows.numel():
        fn(
            xw,
            rows,
            yw,
            xw.stride(0),
            yw.stride(0),
            xw.shape[1] // _VEC,
            torch.cuda.current_stream(x.device).cuda_stream,
        )


def scatter(fn: CompiledKernel, x: torch.Tensor, rows: torch.Tensor, y: torch.Tensor) -> None:
    """``y[rows[i]] = x[i]`` for ``i < len(rows)`` (``fn`` from ``compile_select(True)``)."""

    xw, yw = _check(x, rows, y)
    if xw.shape[0] < rows.numel():
        raise ValueError("mtp_rows_select.scatter: x has fewer rows than rows")
    if rows.numel():
        fn(
            xw,
            rows,
            yw,
            xw.stride(0),
            yw.stride(0),
            xw.shape[1] // _VEC,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
