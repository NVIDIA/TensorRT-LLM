# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Match each page of this rank to the same page in every rank's gathered block, in one CuTe DSL launch.

After the all-gather of a chunk of ``n`` sharded cold pages, rank block ``q`` of the staging buffer holds rank ``q``'s
``n`` cold pages (``cpb`` bytes each) and every cold page ends with a 16-byte trailer at byte ``toff``: the page id
(a 64-bit checksum of the full page, the same on every rank for the same page) and a tag ``MAGIC << 32 | q``.
Thread ``(q, i)`` finds the position ``j`` of this rank's page ``i`` in block ``q``: ``j = i`` when the ids agree
there, otherwise the first ``j`` with the same id. It writes ``source[q * n + i] = j`` (``i`` when none) and adds 1 to
``counters[0]`` when no such ``j`` exists or its tag is wrong, or to ``counters[1]`` when ``j != i``. The counters are
cumulative int64 totals, so the host can read them lazily.
"""

from __future__ import annotations

import functools

import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

from .paged_segment_copy import _load_v4

THREADS = 128


@dsl_user_op
def _red_add_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "red.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class MatchPageIdsKernel:
    """One thread per (rank block q, page i); see the module docstring."""

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        staging: Int64,
        counters: Int64,
        n: Int32,
        world: Int32,
        rank: Int32,
        cpb: Int64,
        toff: Int64,
        magic: Int32,
        stream: cuda.CUstream,
    ):
        total = n * world
        self.kernel(source, staging, counters, n, world, rank, cpb, toff, magic).launch(
            grid=[(total + THREADS - 1) // THREADS, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        staging: Int64,
        counters: Int64,
        n: Int32,
        world: Int32,
        rank: Int32,
        cpb: Int64,
        toff: Int64,
        magic: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        t = bidx * THREADS + tidx
        if t < n * world:
            q = t // n
            i = t - q * n
            block = Int64(n) * cpb
            mine = _load_v4(staging + Int64(rank) * block + Int64(i) * cpb + toff)
            here = _load_v4(staging + Int64(q) * block + Int64(i) * cpb + toff)
            j = Int32(-1)
            if ((here[0] ^ mine[0]) | (here[1] ^ mine[1])) == Int32(0):
                j = i
            k = Int32(0)
            while k < n:
                if j < Int32(0):
                    w = _load_v4(staging + Int64(q) * block + Int64(k) * cpb + toff)
                    if ((w[0] ^ mine[0]) | (w[1] ^ mine[1])) == Int32(0):
                        j = k
                k = k + Int32(1)
            ok = Int32(0)
            if j >= Int32(0):
                tag = _load_v4(staging + Int64(q) * block + Int64(j) * cpb + toff)
                if ((tag[2] ^ q) | (tag[3] ^ magic)) == Int32(0):
                    ok = Int32(1)
            if ok == Int32(1):
                source[t] = j
                if j != i:
                    _red_add_u64(counters + Int64(8), Int64(1))
            else:
                source[t] = i
                _red_add_u64(counters, Int64(1))


@functools.cache
def _compile():
    source = cute.runtime.make_fake_compact_tensor(Int32, (cute.sym_int(),), stride_order=(0,))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        MatchPageIdsKernel(),
        source,
        Int64(0),
        Int64(0),
        Int32(1),
        Int32(1),
        Int32(0),
        Int64(16),
        Int64(0),
        Int32(0),
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


def match_page_ids(
    source: torch.Tensor,
    *,
    staging: int,
    counters: torch.Tensor,
    n: int,
    world: int,
    rank: int,
    cold_page_bytes: int,
    trailer_off: int,
    magic: int,
) -> None:
    """Launch on the current torch stream. ``source`` int32 ``[world * n]`` (contiguous) on the device; ``staging``
    the gathered blocks' device address; ``counters`` int64 ``[2]`` cumulative (no match, reordered) on the device."""
    if n == 0:
        return
    _compile()(
        source,
        int(staging),
        int(counters.data_ptr()),
        int(n),
        int(world),
        int(rank),
        int(cold_page_bytes),
        int(trailer_off),
        int(magic),
    )


__all__ = ["MatchPageIdsKernel", "match_page_ids"]
