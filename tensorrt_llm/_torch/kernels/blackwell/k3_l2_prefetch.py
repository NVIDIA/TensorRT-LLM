# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch of weight byte ranges (SM100/SM103).

A small kernel that issues ``cp.async.bulk.prefetch.L2`` for up to :data:`MAX_RANGES`
contiguous byte ranges and exits. The prefetches are hints: nothing waits for them and no
result depends on them. The Kimi-K3 decode uses it on a side stream to pull the next layer's
projection weights into L2 while the current layer's latency-bound kernels (recurrence,
all-reduce) leave HBM idle, so the next layer's GEMMs read them from L2 (their weight loads carry
an evict-first hint, as do the MoE expert GEMMs', so prefetched lines with the default policy
survive the expert weight streams in between).

Each thread prefetches whole chunks of :data:`CHUNK_BYTES` (the last chunk of a range may be
shorter); one warp per CTA, as many CTAs as it takes to give every thread about one chunk, at
most ``max_ctas``. A CTA stays resident until its prefetches have been served, and the prefetch
rate grows with the number of SMs that issue them (about 4 TB/s from 64-120 CTAs).

Usage notes:

* Launch it on a side stream behind an event recorded after the last kernel that streams weights
  from HBM, without the PDL attribute: its dependencies then stay full ones and it does not
  compete with them for HBM.
* It triggers its programmatic dependents at once, but a programmatic dependent that also waits
  for another stream loses its early launch: join the side stream where no kernel's head start
  matters (or let a later user of the stream join it).
"""

from __future__ import annotations

import functools
from typing import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

#: Largest number of byte ranges per launch.
MAX_RANGES = 8
#: Bytes per ``cp.async.bulk.prefetch.L2`` instruction.
CHUNK_BYTES = 8 << 10
#: Default upper bound on the CTAs (one warp each) of a launch.
DEFAULT_MAX_CTAS = 64
_THREADS = 32


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "griddepcontrol.launch_dependents;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_prefetch_l2(addr, nbytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class K3L2PrefetchKernel:
    """``ctas`` CTAs of one warp prefetch the first ``num_ranges`` of :data:`MAX_RANGES`
    ``(address, bytes)`` ranges in :data:`CHUNK_BYTES` chunks (addresses 16-byte aligned, sizes
    multiples of 16)."""

    def __init__(self, num_ranges: int, ctas: int):
        if not 1 <= num_ranges <= MAX_RANGES or ctas < 1:
            raise ValueError(f"num_ranges={num_ranges} ctas={ctas}")
        self.num_ranges = num_ranges
        self.ctas = ctas

    @cute.jit
    def __call__(
        self,
        a0: Int64,
        n0: Int64,
        a1: Int64,
        n1: Int64,
        a2: Int64,
        n2: Int64,
        a3: Int64,
        n3: Int64,
        a4: Int64,
        n4: Int64,
        a5: Int64,
        n5: Int64,
        a6: Int64,
        n6: Int64,
        a7: Int64,
        n7: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(a0, n0, a1, n1, a2, n2, a3, n3, a4, n4, a5, n5, a6, n6, a7, n7).launch(
            grid=(self.ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        a0: Int64,
        n0: Int64,
        a1: Int64,
        n1: Int64,
        a2: Int64,
        n2: Int64,
        a3: Int64,
        n3: Int64,
        a4: Int64,
        n4: Int64,
        a5: Int64,
        n5: Int64,
        a6: Int64,
        n6: Int64,
        a7: Int64,
        n7: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        # Nothing reads what this kernel does: a programmatic dependent (e.g. a kernel that joins
        # this stream) need not wait for the prefetches to land.
        _griddepcontrol_launch_dependents()
        thread = bidx * _THREADS + tidx
        total = self.ctas * _THREADS
        ranges = ((a0, n0), (a1, n1), (a2, n2), (a3, n3), (a4, n4), (a5, n5), (a6, n6), (a7, n7))
        for r in cutlass.range_constexpr(self.num_ranges):
            base, nbytes = ranges[r]
            chunks = Int32((nbytes + (CHUNK_BYTES - 1)) // CHUNK_BYTES)
            for c in cutlass.range(thread, chunks, total, unroll=1):
                off = Int64(c) * CHUNK_BYTES
                size = nbytes - off
                if size > CHUNK_BYTES:
                    size = Int64(CHUNK_BYTES)
                _bulk_prefetch_l2(base + off, Int32(size))


@functools.cache
def _compiled(num_ranges: int, ctas: int):
    zero = Int64(0)
    return cute.compile(
        K3L2PrefetchKernel(num_ranges, ctas),
        *([zero] * (2 * MAX_RANGES)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def tensor_range(t: torch.Tensor) -> tuple[int, int]:
    """``(address, bytes)`` spanned by a dense tensor (contiguous up to a permutation of its
    dimensions, e.g. a transposed scale buffer), aligned inwards to 16 bytes."""
    if t.numel() == 0:
        return t.data_ptr(), 0
    if any(s < 0 for s in t.stride()):
        raise ValueError("prefetch ranges need non-negative strides")
    span = (1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))) * t.element_size()
    if span != t.numel() * t.element_size():
        raise ValueError("prefetch ranges must be dense tensors")
    start = t.data_ptr()
    first = (start + 15) // 16 * 16
    return first, max(0, (start + span - first) // 16 * 16)


def launch_ctas(ranges: Sequence[tuple[int, int]], max_ctas: int = DEFAULT_MAX_CTAS) -> int:
    """CTAs that give every thread about one chunk (at least 1, at most ``max_ctas``): the
    prefetch rate scales with the SMs issuing it (each SM's TMA unit handles its own requests)."""
    chunks = sum((n + CHUNK_BYTES - 1) // CHUNK_BYTES for _, n in ranges)
    return max(1, min(max_ctas, (chunks + _THREADS - 1) // _THREADS))


def run_k3_l2_prefetch(
    ranges: Sequence[tuple[int, int]], *, max_ctas: int = DEFAULT_MAX_CTAS
) -> None:
    """Prefetch ``ranges`` (``(address, bytes)``, 16-byte aligned, at most :data:`MAX_RANGES`)
    into L2 on the current stream. Empty ranges are dropped; nothing is launched if none is
    left."""
    ranges = [(a, n) for a, n in ranges if n > 0]
    if not ranges:
        return
    if len(ranges) > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges, got {len(ranges)}")
    for a, n in ranges:
        if a % 16 or n % 16:
            raise ValueError(f"range ({a:#x}, {n}) is not 16-byte aligned")
    args = []
    for a, n in ranges:
        args += [Int64(a), Int64(n)]
    args += [Int64(0)] * (2 * MAX_RANGES - len(args))
    _compiled(len(ranges), launch_ctas(ranges, max_ctas))(
        *args, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )


__all__ = ["CHUNK_BYTES", "MAX_RANGES", "K3L2PrefetchKernel", "run_k3_l2_prefetch", "tensor_range"]
