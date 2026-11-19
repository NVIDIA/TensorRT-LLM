# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL zeroing, in place, of the verify logits rows of CUDA-graph padding requests.

A padding request is one whose sequence slot is the dummy slot row; which requests are padding is known on the
device only. Every CTA walks the requests, and for each padding one stores zeros over its share of the request's
rows (16-byte stores, the request's rows split evenly over the CTAs); rows of real requests are never read or
written. A step without padding requests costs one launch whose CTAs read the slot ids and exit.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64

THREADS = 256
#: CTAs of a launch: one wave on a GB300 (148 SMs), each request's rows split over all of them.
CTAS = 148
_IDS = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}


class ZeroPaddingRowsKernel:
    """Zero ``rows[r]`` (all of request ``r``'s rows, flattened: ``[n, rows_per_request * vocab]`` 32-bit
    words) wherever ``slot_ids[r]`` is the dummy slot row."""

    @cute.jit
    def __call__(
        self,
        mRows: cute.Tensor,  # [n, rows_per_request * vocab] int32 (the FP32 logits' bits)
        mSlot: cute.Tensor,  # [n] int32 / int64
        dummy: Int64,
        stream,
    ):
        self.kernel(mRows, mSlot, dummy).launch(
            grid=[CTAS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mRows: cute.Tensor, mSlot: cute.Tensor, dummy: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        n = mRows.shape[0]
        vecs = mRows.shape[1] // 4
        gV = cute.logical_divide(mRows, (None, 4))
        zero = cute.make_rmem_tensor(cute.make_layout((4,), stride=(1,)), Int32)
        for i in cutlass.range_constexpr(4):
            zero[i] = Int32(0)
        first = cta * THREADS + tidx
        for r in cutlass.range(0, n, 1):
            if Int64(mSlot[r]) == dummy:
                for v in cutlass.range(first, vecs, CTAS * THREADS):
                    cute.autovec_copy(zero, gV[r, (None, v)])


def _row_major(dtype, shape, align=None):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
    )


@functools.cache
def _compiled(width: int, slot_dtype: torch.dtype):
    n = cute.sym_int()
    return cute.compile(
        ZeroPaddingRowsKernel(),
        _row_major(Int32, (n, width), 16),
        _row_major(_IDS[slot_dtype], (n,)),
        Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )


def zero_padding_rows_(
    logits: torch.Tensor, slot_ids: torch.Tensor, dummy_slot_row: int, rows_per_request: int
) -> torch.Tensor:
    """Zero, in place, the ``rows_per_request`` rows of every request ``r`` of contiguous FP32
    ``logits [n * rows_per_request, vocab]`` (``vocab`` a multiple of 4, 16-byte aligned) whose
    ``slot_ids[r]`` is ``dummy_slot_row``; returns ``logits``."""
    n = slot_ids.shape[0]
    if n == 0:
        return logits
    width = rows_per_request * logits.shape[1]
    _compiled(width, slot_ids.dtype)(
        logits.view(torch.int32).view(n, width), slot_ids, dummy_slot_row
    )
    return logits


__all__ = ["CTAS", "THREADS", "ZeroPaddingRowsKernel", "zero_padding_rows_"]
