# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for a PDL producer that writes its dependent's input late (tests only).

One 512-thread CTA per row, the shape of the latent all-reduce row kernel: each CTA releases its
programmatic dependents at entry, spins ``ns`` nanoseconds on the global timer, then copies its row
of ``src`` into ``dst``. A dependent that read ``dst`` before its grid wait would see the previous
contents.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass import Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm.rubin_lamport_ar import (  # noqa: E501
    ld_v4,
    st_v4,
)
from tensorrt_llm._torch.kernels.blackwell.utils import griddepcontrol_launch_dependents

THREADS = 512


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class SpinCopyKernel:
    """``rows`` CTAs: release dependents, spin, copy ``row_bytes`` (a multiple of 16, at most
    ``16 * THREADS``) per row."""

    def __init__(self, rows: int, row_bytes: int):
        if row_bytes % 16 or row_bytes > 16 * THREADS:
            raise ValueError(f"row_bytes {row_bytes}")
        self.rows = rows
        self.row_bytes = row_bytes

    @cute.jit
    def __call__(self, src: Int64, dst: Int64, ns: Int64, stream: cuda.CUstream):
        self.kernel(src, dst, ns).launch(
            grid=(self.rows, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, src: Int64, dst: Int64, ns: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        griddepcontrol_launch_dependents()
        if tidx == 0:
            start = _globaltimer()
            now = start
            while now - start < ns:
                now = _globaltimer()
        cute.arch.barrier()
        if tidx < self.row_bytes // 16:
            offset = Int64(row) * Int64(self.row_bytes) + Int64(tidx) * Int64(16)
            v0, v1, v2, v3 = ld_v4(src + offset)
            st_v4(dst + offset, v0, v1, v2, v3)


@functools.cache
def _compiled(rows: int, row_bytes: int):
    zero = Int64(0)
    return cute.compile(
        SpinCopyKernel(rows, row_bytes),
        zero,
        zero,
        zero,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def spin_copy(src: torch.Tensor, dst: torch.Tensor, ns: int) -> None:
    """Copy ``src`` into ``dst`` (contiguous, same shape, rows of at most 8 KiB) on the current
    stream, ``ns`` nanoseconds after the kernel starts; its dependents are released at once."""
    if src.shape != dst.shape or not (src.is_contiguous() and dst.is_contiguous()):
        raise ValueError("src and dst must be contiguous and of one shape")
    rows = src.shape[0]
    row_bytes = src[0].numel() * src.element_size()
    _compiled(rows, row_bytes)(
        Int64(src.data_ptr()),
        Int64(dst.data_ptr()),
        Int64(int(ns)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = ["SpinCopyKernel", "spin_copy", "THREADS"]
