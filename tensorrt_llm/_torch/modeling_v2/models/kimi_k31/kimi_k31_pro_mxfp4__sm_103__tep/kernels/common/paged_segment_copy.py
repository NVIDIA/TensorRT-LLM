# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed byte-segment copies between page pools at raw addresses, in one CuTe DSL launch.

For page ``i < N`` and segment ``s < S`` of the table ``segs [S, 8]`` (int64 columns ``src_base, src_stride,
src_off, dst_base, dst_stride, dst_off, nbytes, q``) the launch copies ``nbytes`` bytes

    src = src_add + src_base + q * q_stride + src_idx[q * idx_q_stride + i] * src_stride + src_off
    dst = dst_add + dst_base +                dst_idx[i] * dst_stride + dst_off

``src_add`` / ``dst_add`` / ``q_stride`` / ``idx_q_stride`` are launch scalars, so one table serves pools whose base
changes per call (a cold host pool or a pinned staging buffer) and gathered rank blocks whose stride depends on the
batch; ``idx_q_stride > 0`` gives each rank block ``q`` its own source positions (``src_idx`` is then ``[Q, N]``). Every
address and ``nbytes`` is a multiple of 16; the bytes move as 128-bit words. Sources and destinations may be device
memory or host memory the GPU can address (pinned / C2C-coherent host pages); they must not overlap.

Grid ``(N, S, Z)``: CTA ``(i, s, z)`` copies bytes ``[z * CHUNK, (z + 1) * CHUNK)`` of segment ``s`` of page ``i``,
``UNROLL`` words in flight per thread.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

THREADS = 256
UNROLL = 4
WORD = 16
CHUNK = THREADS * UNROLL * WORD
SEG_COLS = 8


@dsl_user_op
def _load_v4(address: Int64, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), out, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def _store_v4(address: Int64, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class PagedSegmentCopyKernel:
    """One CTA per (page, segment, CHUNK-byte block); see the module docstring."""

    @cute.jit
    def __call__(
        self,
        segs: cute.Tensor,
        src_idx: cute.Tensor,
        dst_idx: cute.Tensor,
        src_add: Int64,
        dst_add: Int64,
        q_stride: Int64,
        idx_q_stride: Int32,
        z_blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(segs, src_idx, dst_idx, src_add, dst_add, q_stride, idx_q_stride).launch(
            grid=[dst_idx.shape[0], segs.shape[0], z_blocks],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        segs: cute.Tensor,
        src_idx: cute.Tensor,
        dst_idx: cute.Tensor,
        src_add: Int64,
        dst_add: Int64,
        q_stride: Int64,
        idx_q_stride: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i, s, z = cute.arch.block_idx()
        nbytes = segs[s, 6]
        last = nbytes - Int64(WORD)
        begin = Int64(z) * Int64(CHUNK)
        src = (
            src_add
            + segs[s, 0]
            + segs[s, 7] * q_stride
            + Int64(src_idx[Int32(segs[s, 7]) * idx_q_stride + i]) * segs[s, 1]
            + segs[s, 2]
        )
        dst = dst_add + segs[s, 3] + Int64(dst_idx[i]) * segs[s, 4] + segs[s, 5]
        # All loads first (UNROLL words in flight), each at a clamped in-segment offset; only in-range words store.
        words = []
        for k in cutlass.range_constexpr(UNROLL):
            off = begin + Int64(tidx + k * THREADS) * Int64(WORD)
            words.append(_load_v4(src + cutlass.min(off, last)))
        for k in cutlass.range_constexpr(UNROLL):
            off = begin + Int64(tidx + k * THREADS) * Int64(WORD)
            w = words[k]
            if off < nbytes:
                _store_v4(dst + off, w[0], w[1], w[2], w[3])


@functools.cache
def _compile():
    segs = cute.runtime.make_fake_compact_tensor(
        Int64, (cute.sym_int(), SEG_COLS), stride_order=(1, 0)
    )
    src_idx = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(cute.sym_int64(),))
    dst_idx = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(cute.sym_int64(),))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        PagedSegmentCopyKernel(),
        segs,
        src_idx,
        dst_idx,
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(1),
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def paged_segment_copy(
    segs: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    *,
    src_add: int = 0,
    dst_add: int = 0,
    q_stride: int = 0,
    idx_q_stride: int = 0,
    max_segment_bytes: int,
) -> None:
    """Launch on the current torch stream. ``segs`` int64 ``[S, 8]`` contiguous on the device; ``dst_idx`` int32
    ``[N]`` and ``src_idx`` int32 ``[N]`` (or ``[Q * idx_q_stride]`` flat, rank block ``q`` at ``q * idx_q_stride``;
    any stride) on the device; ``max_segment_bytes`` the largest ``nbytes`` of the table
    (sets the grid's z extent). The first call compiles (outside CUDA-graph capture)."""
    n = dst_idx.shape[0]
    if n == 0 or segs.shape[0] == 0:
        return
    if idx_q_stride == 0 and src_idx.shape[0] != n:
        raise ValueError(f"src_idx has {src_idx.shape[0]} pages, dst_idx {n}")
    z_blocks = (int(max_segment_bytes) + CHUNK - 1) // CHUNK
    _compile()(
        segs,
        src_idx,
        dst_idx,
        int(src_add),
        int(dst_add),
        int(q_stride),
        int(idx_q_stride),
        z_blocks,
    )


__all__ = ["CHUNK", "SEG_COLS", "PagedSegmentCopyKernel", "paged_segment_copy"]
