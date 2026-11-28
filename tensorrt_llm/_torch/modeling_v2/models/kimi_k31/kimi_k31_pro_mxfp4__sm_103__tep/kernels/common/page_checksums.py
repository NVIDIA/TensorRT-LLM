# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weighted 64-bit page checksums read straight from the page pools, in one CuTe DSL launch.

A page is the concatenation of its segments as a gather table ``segs [S, 8]`` (the columns of
:mod:`.paged_segment_copy`) lays them out: segment ``s`` of page ``i`` is the ``nbytes`` bytes at
``segs[s, 0] + idx[i] * segs[s, 1] + segs[s, 2]`` and lands at byte ``segs[s, 3] + segs[s, 5]`` of the page. With the
page read as int64 words ``w`` and ``weights`` int64 ``[page_bytes / 8]``, the checksum is

    out[i] = sum_k w[i, k] * weights[k]    (mod 2**64)

which equals ``(words * weights).sum(dim=1)`` over the gathered int64 page bit for bit: the products and the sum wrap
modulo 2**64, so the order of the additions does not matter.

Grid ``(N, S, Z)``: CTA ``(i, s, z)`` multiplies ``UNROLL`` 16-byte words per thread of bytes ``[z * CHUNK, (z + 1) *
CHUNK)`` of segment ``s``, reduces them over the warp with shuffles and adds each warp's partial to ``out[i]`` with one
``red.global.add.u64``. ``out`` must hold zeros when the launch starts; :func:`page_checksums` zeroes it on the launch
stream first. Nothing is allocated.
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

from .match_page_ids import _red_add_u64
from .paged_segment_copy import SEG_COLS

THREADS = 256
UNROLL = 4
WORD = 16
CHUNK = THREADS * UNROLL * WORD
WARP = 32


@dsl_user_op
def _load_v2_u64(address: Int64, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i64()] * 2),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int64(llvm.extractvalue(T.i64(), out, [k], loc=loc, ip=ip)) for k in range(2)]


class PageChecksumsKernel:
    """One CTA per (page, segment, CHUNK-byte block); see the module docstring."""

    @cute.jit
    def __call__(
        self,
        segs: cute.Tensor,
        idx: cute.Tensor,
        weights: Int64,
        out: Int64,
        out_stride: Int64,
        z_blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(segs, idx, weights, out, out_stride).launch(
            grid=[idx.shape[0], segs.shape[0], z_blocks],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        segs: cute.Tensor,
        idx: cute.Tensor,
        weights: Int64,
        out: Int64,
        out_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        i, s, z = cute.arch.block_idx()
        nbytes = segs[s, 6]
        last = nbytes - Int64(WORD)
        begin = Int64(z) * Int64(CHUNK)
        src = segs[s, 0] + Int64(idx[i]) * segs[s, 1] + segs[s, 2]
        wts = weights + segs[s, 3] + segs[s, 5]
        # All loads first (UNROLL words in flight), each at a clamped in-segment offset; only in-range words count.
        words = []
        coefs = []
        for k in cutlass.range_constexpr(UNROLL):
            off = cutlass.min(begin + Int64(tidx + k * THREADS) * Int64(WORD), last)
            words.append(_load_v2_u64(src + off))
            coefs.append(_load_v2_u64(wts + off))
        acc = Int64(0)
        for k in cutlass.range_constexpr(UNROLL):
            off = begin + Int64(tidx + k * THREADS) * Int64(WORD)
            if off < nbytes:
                acc = acc + words[k][0] * coefs[k][0] + words[k][1] * coefs[k][1]
        # Butterfly over the warp on the two 32-bit halves; every lane ends with the warp's sum (mod 2**64).
        for shift in cutlass.range_constexpr(5):
            offset = WARP >> (shift + 1)
            lo = cute.arch.shuffle_sync_bfly(Int32(acc), offset=offset)
            hi = cute.arch.shuffle_sync_bfly(Int32(acc >> Int64(32)), offset=offset)
            acc = acc + ((Int64(hi) << Int64(32)) | (Int64(lo) & Int64(0xFFFFFFFF)))
        if tidx % WARP == 0:
            _red_add_u64(out + Int64(i) * out_stride * Int64(8), acc)


@functools.cache
def _compile():
    segs = cute.runtime.make_fake_compact_tensor(
        Int64, (cute.sym_int(), SEG_COLS), stride_order=(1, 0)
    )
    idx = cute.runtime.make_fake_tensor(Int32, (cute.sym_int(),), stride=(cute.sym_int64(),))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        PageChecksumsKernel(),
        segs,
        idx,
        Int64(0),
        Int64(0),
        Int64(1),
        Int32(1),
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def page_checksums(
    segs: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    out: torch.Tensor,
    *,
    max_segment_bytes: int,
) -> None:
    """Zero ``out`` and launch, both on the current torch stream. ``segs`` int64 ``[S, 8]`` contiguous (a gather
    table: each page's segments land at ``segs[s, 3] + segs[s, 5]``, every address and ``nbytes`` a multiple of 16);
    ``idx`` int32 ``[N]`` page slots (any stride); ``weights`` int64 contiguous, one per 8 bytes of the page; ``out``
    int64 ``[N]`` (any stride, 8-byte aligned) receives the checksums. The first call compiles (outside CUDA-graph
    capture)."""
    n = idx.shape[0]
    if out.shape[0] != n:
        raise ValueError(f"out has {out.shape[0]} entries, idx {n}")
    if n == 0:
        return
    out.zero_()
    if segs.shape[0] == 0:
        return
    z_blocks = (int(max_segment_bytes) + CHUNK - 1) // CHUNK
    _compile()(
        segs,
        idx,
        int(weights.data_ptr()),
        int(out.data_ptr()),
        int(out.stride(0)),
        z_blocks,
    )


__all__ = ["PageChecksumsKernel", "page_checksums"]
