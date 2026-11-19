# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's per-query sliding-window pool rows in one launch (CuTe DSL).

Query row ``r = b * block_size + j`` sits at position ``p = ctx_len[b] + j`` and attends the
positions ``[start, p]`` with ``start = max(0, p - window + 1)``. Entry ``k`` of its list is the
global pool row of position ``start + k`` (``page_tables[b, min(pos // page_size, max_pages - 1)]
* page_size + pos % page_size``) while ``k < p - start + 1`` and ``-1`` past that; the live
length is stored beside it. With a context ring of ``ring_rows`` rows per request, position
``pos`` is read from table row ``pos mod ring_rows``. All index math is 64-bit and each stored
value is truncated to int32 once, as the torch reference does, so every entry is bitwise the
reference's.

One CTA per query row; consecutive threads write consecutive entries.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

_CUTE_INTS = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


class DFlash2WindowRowsKernel:
    """``rows[r, :]`` and ``lengths[r]`` of every query row ``r``, one CTA each."""

    num_threads = 256

    def __init__(
        self, block_size: int, page_size: int, window: int, width: int, ring_rows: int = 0
    ):
        if width < window:
            raise ValueError(f"index width {width} cannot hold a {window}-position window")
        if block_size < 1 or page_size < 1 or window < 1:
            raise ValueError("block_size, page_size and window must be positive")
        if ring_rows < 0:
            raise ValueError(f"ring_rows must be 0 (no ring) or positive, got {ring_rows}")
        self.block_size = block_size
        self.page_size = page_size
        self.window = window
        self.width = width
        self.ring_rows = ring_rows
        self.chunks = (width + self.num_threads - 1) // self.num_threads

    @cute.jit
    def __call__(
        self,
        ctx_len: cute.Tensor,
        page_tables: cute.Tensor,
        rows: cute.Tensor,
        lengths: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(ctx_len, page_tables, rows, lengths).launch(
            grid=[rows.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        ctx_len: cute.Tensor,
        page_tables: cute.Tensor,
        rows: cute.Tensor,
        lengths: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        r, _, _ = cute.arch.block_idx()
        b = r // self.block_size
        j = r - b * self.block_size
        pos_q = Int64(ctx_len[b]) + Int64(j)
        start = pos_q - Int64(self.window - 1)
        if start < Int64(0):
            start = Int64(0)
        length = pos_q - start + Int64(1)
        if tidx == 0:
            lengths[r] = Int32(length)
        max_col = Int64(page_tables.shape[1]) - Int64(1)
        for chunk in cutlass.range_constexpr(self.chunks):
            k = tidx + chunk * self.num_threads
            if k < self.width:
                entry = Int32(-1)
                if Int64(k) < length:
                    pos = start + Int64(k)
                    if cutlass.const_expr(self.ring_rows > 0):
                        pos = pos % Int64(self.ring_rows)
                    col = pos // Int64(self.page_size)
                    if col > max_col:
                        col = max_col
                    page = Int64(page_tables[b, Int32(col)])
                    entry = Int32(page * Int64(self.page_size) + pos % Int64(self.page_size))
                rows[r, k] = entry


@functools.cache
def _compile(
    block_size: int,
    page_size: int,
    window: int,
    width: int,
    ring_rows: int,
    ctx_len_dtype: torch.dtype,
    page_table_dtype: torch.dtype,
):
    batch = cute.sym_int()
    num_rows = cute.sym_int()
    ctx_len_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[ctx_len_dtype], (batch,), stride=(cute.sym_int64(),)
    )
    page_tables_fake = cute.runtime.make_fake_tensor(
        _CUTE_INTS[page_table_dtype],
        (batch, cute.sym_int()),
        stride=(cute.sym_int64(), cute.sym_int64()),
    )
    rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows, width), stride_order=(1, 0)
    )
    lengths_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_rows,), stride_order=(0,)
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = DFlash2WindowRowsKernel(block_size, page_size, window, width, ring_rows)
    return cute.compile(
        kernel,
        ctx_len_fake,
        page_tables_fake,
        rows_fake,
        lengths_fake,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def dflash2_window_rows_into(
    ctx_len: torch.Tensor,
    page_tables: torch.Tensor,
    block_size: int,
    page_size: int,
    window: int,
    rows: torch.Tensor,
    lengths: torch.Tensor,
    ring_rows: int = 0,
) -> None:
    """Fill ``rows`` (int32 ``[B * block_size, width]``) and ``lengths`` (int32 ``[B * block]``).

    ``ctx_len`` is ``[B]`` int32/int64, ``page_tables`` ``[B, max_pages]`` int32/int64 (any
    strides); ``ring_rows`` the rows per request of a context ring (0: a dense table). Compiles
    once per ``(block_size, page_size, window, width, ring_rows, dtypes)`` on the first call,
    which must run outside CUDA-graph capture; later calls only launch.
    """
    if rows.shape[0] == 0:
        return
    compiled = _compile(
        int(block_size),
        int(page_size),
        int(window),
        rows.shape[1],
        int(ring_rows),
        ctx_len.dtype,
        page_tables.dtype,
    )
    compiled(ctx_len, page_tables, rows, lengths)


__all__ = ["DFlash2WindowRowsKernel", "dflash2_window_rows_into"]
