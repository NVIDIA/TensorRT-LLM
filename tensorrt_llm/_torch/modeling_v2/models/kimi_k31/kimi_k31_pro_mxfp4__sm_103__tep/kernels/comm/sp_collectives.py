# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sequence-parallel prefill collectives over NVLink peer mappings (CuTe DSL, SM100 / SM103).

A sequence-parallel prefill splits a step's ``M`` tokens over the ``W`` ranks of the TP group in contiguous
blocks of ``rows = M / W`` rows (rank ``r`` owns rows ``[r * rows, (r + 1) * rows)``). Each layer all-gathers
row blocks and reduce-scatters row-parallel partials. Both kernels here push: a rank stores what a peer needs
straight into the peer's receive region through the peer's unicast mapping of one symmetric workspace
(``McastGPUBuffer``; on a multi-node NVLink domain the peers' memory is load / store addressable) with 16-byte
stores (on GB300 the fastest of SM stores, SM loads, multicast stores and copy-engine copies), then publishes
the chunk with a release flag at system scope; consumer CTAs of the receiving rank wait for the chunk's flags and
finish it locally. The receive region ``[W, rows, row_bytes]`` is shared by both kernels:

* reduce-scatter: rank ``s`` stores its partial rows of rank ``d``'s block into slot ``s`` of ``d``'s region and
  adds 1 to ``d``'s counter for that chunk; ``d``'s consumers wait for ``W - 1`` arrivals and sum the ``W``
  partials of each element in FP32 in rank order ``0, 1, ..., W - 1`` (their own partial read from the input),
  round once to BF16 (nearest even) and store the block. No ring, so no BF16 rounding between hops.
* all-gather: rank ``s`` stores its block into slot ``s`` of every peer's region and sets the peer's flag for
  (``s``, chunk) to the call's epoch; consumers copy each published chunk into the output rows (and the own
  block from the input). A copy: bitwise.

Every call (of either kernel) has a host-side epoch, the same on every rank; a rank's consumers publish its
"done" word once the call's region contents are consumed, and every pusher waits at its start for each
destination's "done" word to reach the previous call, so the next call never overwrites unconsumed rows.
Pushers never wait on data; the grid (one CTA per SM) is co-resident, so a waiting consumer cannot block a pusher.
Calls must be made in the same order on every rank of the group with the same shapes.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Threads per CTA; each moves 16-byte vectors.
THREADS = 512
#: Vectors in flight per thread in the copy loops.
UNROLL = 8
#: Chunks (of ``chunk_rows`` rows) a rank's block may have.
MAX_CHUNKS = 1024
#: Largest group the flag layout holds.
MAX_WORLD = 8
#: Flag words (uint32) at the start of the workspace: the reduce-scatter's chunk counters, the done and
#: consumer-count words, and the all-gather's (sender, chunk) epochs.
RS_FLAGS, DONE, COUNT, ITEMQ, AG_FLAGS = (
    0,
    MAX_CHUNKS,
    MAX_CHUNKS + 1,
    MAX_CHUNKS + 2,
    MAX_CHUNKS + 4,
)
#: Bytes reserved for the flags (keeps the receive region 64 KiB aligned).
FLAGS_BYTES = 64 * 1024
#: Column slices of a chunk one reduce-scatter consumer item covers.
RS_SLICES = 1
#: Vectors per thread per round of the reduce-scatter consumers (W loads each).
RS_UNROLL = 2


# ---------------------------------------------------------------------------
# Inline PTX
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _lv(v, *, loc=None, ip=None):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _iv(v, *, loc=None, ip=None):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _v4(text, addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        text,
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    """16 bytes of local memory no kernel writes concurrently."""
    return _v4("ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", addr, loc=loc, ip=ip)


@dsl_user_op
def ld_cg_v4(addr, *, loc=None, ip=None):
    """16 bytes a peer wrote (or a peer's memory), read past L1."""
    return _v4("ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", addr, loc=loc, ip=ip)


@dsl_user_op
def st_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_nc_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_acquire_sys_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_release_sys_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_relaxed_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_relaxed_sys_add(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.relaxed.sys.global.add.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_acq_rel_gpu(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_relaxed_gpu(addr, value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.sys;", "", loc=loc, ip=ip)


@dsl_user_op
def acc_bf16x2(word, acc_lo, acc_hi, *, loc=None, ip=None):
    """``(acc_lo + lo, acc_hi + hi)`` in FP32 (round to nearest) for the BF16 pair ``word``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [
            _iv(word, loc=loc, ip=ip),
            cutlass.Float32(acc_lo).ir_value(loc=loc, ip=ip),
            cutlass.Float32(acc_hi).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b32 l, h;\n.reg .f32 fl, fh;\n"
        "shl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 fl, l;\nmov.b32 fh, h;\n"
        "add.rn.f32 $0, $3, fl;\nadd.rn.f32 $1, $4, fh;\n}",
        "=f,=f,r,f,f",
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _copy_rows(
    src: Int64,
    dst: Int64,
    dst2: Int64,
    nvec: Int32,
    tid: Int32,
    cached: cutlass.Constexpr,
    two: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    unroll: cutlass.Constexpr,
):
    """Copy ``nvec`` 16-byte vectors ``src -> dst`` (and ``-> dst2``), ``unroll`` loads in flight per thread."""
    words = cute.make_rmem_tensor((unroll * 4,), Int32)
    for base in cutlass.range(tid, nvec, threads * unroll, unroll=1):
        for u in cutlass.range_constexpr(unroll):
            v = base + u * threads
            if v < nvec:
                if cutlass.const_expr(cached):
                    w = ld_v4(src + Int64(v) * 16)
                else:
                    w = ld_cg_v4(src + Int64(v) * 16)
                for e in cutlass.range_constexpr(4):
                    words[u * 4 + e] = w[e]
        for u in cutlass.range_constexpr(unroll):
            v = base + u * threads
            if v < nvec:
                st_v4(
                    dst + Int64(v) * 16,
                    words[u * 4],
                    words[u * 4 + 1],
                    words[u * 4 + 2],
                    words[u * 4 + 3],
                )
                if cutlass.const_expr(two):
                    st_v4(
                        dst2 + Int64(v) * 16,
                        words[u * 4],
                        words[u * 4 + 1],
                        words[u * 4 + 2],
                        words[u * 4 + 3],
                    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.jit
def _wait_peers_done(peers: Int64, rank: Int32, epoch: Int32, tid: Int32, world: cutlass.Constexpr):
    """Thread ``t`` waits for peer ``t``'s done word to reach the previous call; then the CTA syncs."""
    if tid < world:
        if tid != rank:
            peer = ld_nc_u64(peers + Int64(tid) * 8)
            done = ld_acquire_sys_u32(peer + Int64(DONE * 4))
            while Int32(done - (epoch - 1)) < 0:
                done = ld_acquire_sys_u32(peer + Int64(DONE * 4))
    cute.arch.barrier()


@cute.jit
def _claim(base: Int64, sitem: cute.Tensor, tid: Int32):
    """The next consumer item of this rank's queue, the same for every thread of the CTA."""
    cute.arch.barrier()
    if tid == 0:
        sitem[0] = atom_add_relaxed_gpu(base + Int64(ITEMQ * 4), Int32(1))
    cute.arch.barrier()
    return sitem[0]


@cute.jit
def _finish_call(base: Int64, epoch: Int32, ctas: Int32, rs_chunks: Int32, tid: Int32):
    """Count the CTA; the last one re-arms the reduce-scatter counters (``rs_chunks`` of them), the queue and the
    count, then publishes the call as done (system-scope release)."""
    cute.arch.barrier()
    if tid == 0:
        old = atom_add_acq_rel_gpu(base + Int64(COUNT * 4), Int32(1))
        if old == ctas - 1:
            for c in cutlass.range(0, rs_chunks, 1, unroll=1):
                st_relaxed_u32(base + Int64(RS_FLAGS * 4) + Int64(c) * 4, Int32(0))
            st_relaxed_u32(base + Int64(ITEMQ * 4), Int32(0))
            st_relaxed_u32(base + Int64(COUNT * 4), Int32(0))
            fence_acq_rel_sys()
            st_release_sys_u32(base + Int64(DONE * 4), epoch)


class SpReduceScatterKernel:
    """Push reduce-scatter of BF16 ``[W * rows, row_bytes / 2]`` partials into this rank's ``[rows, ...]`` sum.

    Grid ``pushers + consumers`` CTAs: pushers copy (chunk, destination) items chunk-major, so every block fills
    front to back; consumers reduce (chunk, column slice) items of this rank's block as the chunks complete.
    """

    def __init__(
        self,
        world: int,
        chunk_rows: int,
        pushers: int,
        consumers: int,
        threads: int = THREADS,
        unroll: int = UNROLL,
        slices: int = RS_SLICES,
    ):
        if not 2 <= world <= MAX_WORLD:
            raise ValueError(f"world {world} outside [2, {MAX_WORLD}]")
        self.W, self.R, self.P, self.Q = world, chunk_rows, pushers, consumers
        self.threads, self.unroll = threads, unroll
        self.slices = slices

    @cute.jit
    def __call__(
        self,
        inp: Int64,
        out: Int64,
        peers: Int64,
        base: Int64,
        rows: Int32,
        row_bytes: Int32,
        rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(inp, out, peers, base, rows, row_bytes, rank, epoch).launch(
            grid=(self.P + self.Q, 1, 1),
            block=(self.threads, 1, 1),
            smem=128,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp: Int64,
        out: Int64,
        peers: Int64,
        base: Int64,
        rows: Int32,
        row_bytes: Int32,
        rank: Int32,
        epoch: Int32,
    ):
        W, R = self.W, self.R
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        sitem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=16
        )
        nchunks = (rows + (R - 1)) // R
        rb = Int64(row_bytes)
        if bidx < self.P:
            _wait_peers_done(peers, rank, epoch, tid, W)
            for pi in cutlass.range(bidx, nchunks * (W - 1), self.P, unroll=1):
                pc = pi // (W - 1)
                pd = (rank + 1 + (pi - pc * (W - 1))) % W
                pr0 = pc * R
                pnr = rows - pr0
                if pnr > R:
                    pnr = R
                ppeer = ld_nc_u64(peers + Int64(pd) * 8)
                psrc = inp + (Int64(pd) * Int64(rows) + Int64(pr0)) * rb
                pdst = ppeer + Int64(FLAGS_BYTES) + (Int64(rank) * Int64(rows) + Int64(pr0)) * rb
                _copy_rows(
                    psrc,
                    pdst,
                    pdst,
                    pnr * (row_bytes // 16),
                    tid,
                    True,
                    False,
                    self.threads,
                    self.unroll,
                )
                cute.arch.barrier()
                if tid == 0:
                    # The CTA's stores (ordered before thread 0 by the barrier) become visible to the
                    # destination before the count: a system-scope fence then a relaxed add.
                    fence_acq_rel_sys()
                    red_relaxed_sys_add(ppeer + Int64(RS_FLAGS * 4) + Int64(pc) * 4, Int32(1))
        # Every CTA (pushers once done pushing) takes (chunk, slice) items of this rank's block in order.
        nvec_row = row_bytes // 16
        total = nchunks * self.slices
        i = _claim(base, sitem, tid)
        while i < total:
            c = i // self.slices
            s = i - c * self.slices
            if tid == 0:
                flag = base + Int64(RS_FLAGS * 4) + Int64(c) * 4
                got = ld_acquire_sys_u32(flag)
                while got < W - 1:
                    got = ld_acquire_sys_u32(flag)
            cute.arch.barrier()
            r0 = c * R
            nr = rows - r0
            if nr > R:
                nr = R
            nvec = nr * nvec_row
            v0 = (nvec * s) // self.slices
            v1 = (nvec * (s + 1)) // self.slices
            own = inp + (Int64(rank) * Int64(rows) + Int64(r0)) * rb
            recv = base + Int64(FLAGS_BYTES) + Int64(r0) * rb
            slot = Int64(rows) * rb
            # RS_UNROLL vectors per thread per round: W x RS_UNROLL loads in flight.
            words = cute.make_rmem_tensor((RS_UNROLL * W * 4,), Int32)
            for vb in cutlass.range(v0 + tid, v1, self.threads * RS_UNROLL, unroll=1):
                for u in cutlass.range_constexpr(RS_UNROLL):
                    v = vb + u * self.threads
                    if v < v1:
                        for r in cutlass.range_constexpr(W):
                            addr = recv + Int64(r) * slot + Int64(v) * 16
                            if r == rank:
                                addr = own + Int64(v) * 16
                            w = ld_cg_v4(addr)
                            for e in cutlass.range_constexpr(4):
                                words[(u * W + r) * 4 + e] = w[e]
                for u in cutlass.range_constexpr(RS_UNROLL):
                    v = vb + u * self.threads
                    if v < v1:
                        acc = [cutlass.Float32(0.0)] * 8
                        for r in cutlass.range_constexpr(W):
                            for e in cutlass.range_constexpr(4):
                                acc[2 * e], acc[2 * e + 1] = acc_bf16x2(
                                    words[(u * W + r) * 4 + e], acc[2 * e], acc[2 * e + 1]
                                )
                        st_v4(
                            out + Int64(r0) * rb + Int64(v) * 16,
                            pack_bf16x2(acc[0], acc[1]),
                            pack_bf16x2(acc[2], acc[3]),
                            pack_bf16x2(acc[4], acc[5]),
                            pack_bf16x2(acc[6], acc[7]),
                        )
            i = _claim(base, sitem, tid)
        _finish_call(base, epoch, Int32(self.P + self.Q), nchunks, tid)


class SpAllGatherKernel:
    """Push all-gather of ``[rows, row_bytes]`` blocks into ``[W * rows, row_bytes]`` (any dtype: a copy).

    Grid ``pushers + consumers``: pushers store this rank's block into every peer's region, (chunk, peer)
    chunk-major; consumers copy this rank's own chunks from the input and each peer's chunk once published.
    """

    def __init__(
        self,
        world: int,
        chunk_rows: int,
        pushers: int,
        consumers: int,
        threads: int = THREADS,
        unroll: int = UNROLL,
        slices: int = RS_SLICES,
    ):
        if not 2 <= world <= MAX_WORLD:
            raise ValueError(f"world {world} outside [2, {MAX_WORLD}]")
        self.W, self.R, self.P, self.Q = world, chunk_rows, pushers, consumers
        self.threads, self.unroll = threads, unroll
        self.slices = slices

    @cute.jit
    def __call__(
        self,
        inp: Int64,
        out: Int64,
        peers: Int64,
        base: Int64,
        rows: Int32,
        row_bytes: Int32,
        rank: Int32,
        epoch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(inp, out, peers, base, rows, row_bytes, rank, epoch).launch(
            grid=(self.P + self.Q, 1, 1),
            block=(self.threads, 1, 1),
            smem=128,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inp: Int64,
        out: Int64,
        peers: Int64,
        base: Int64,
        rows: Int32,
        row_bytes: Int32,
        rank: Int32,
        epoch: Int32,
    ):
        W, R = self.W, self.R
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        sitem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=16
        )
        nchunks = (rows + (R - 1)) // R
        rb = Int64(row_bytes)
        nvec_row = row_bytes // 16
        if bidx < self.P:
            _wait_peers_done(peers, rank, epoch, tid, W)
            for pi in cutlass.range(bidx, nchunks * (W - 1), self.P, unroll=1):
                pc = pi // (W - 1)
                pd = (rank + 1 + (pi - pc * (W - 1))) % W
                pr0 = pc * R
                pnr = rows - pr0
                if pnr > R:
                    pnr = R
                ppeer = ld_nc_u64(peers + Int64(pd) * 8)
                pdst = ppeer + Int64(FLAGS_BYTES) + (Int64(rank) * Int64(rows) + Int64(pr0)) * rb
                _copy_rows(
                    inp + Int64(pr0) * rb,
                    pdst,
                    pdst,
                    pnr * nvec_row,
                    tid,
                    True,
                    False,
                    self.threads,
                    self.unroll,
                )
                cute.arch.barrier()
                if tid == 0:
                    flag = ppeer + Int64((AG_FLAGS + rank * MAX_CHUNKS) * 4) + Int64(pc) * 4
                    fence_acq_rel_sys()
                    st_relaxed_u32(flag, epoch)
        # Every CTA (pushers once done pushing) takes (chunk, source rank) items in order; the own block comes
        # from the input.
        total = nchunks * W
        i = _claim(base, sitem, tid)
        while i < total:
            c = i // W
            s = i - c * W
            r0 = c * R
            nr = rows - r0
            if nr > R:
                nr = R
            src = inp + Int64(r0) * rb
            if s != rank:
                if tid == 0:
                    flag = base + Int64((AG_FLAGS + s * MAX_CHUNKS) * 4) + Int64(c) * 4
                    got = ld_acquire_sys_u32(flag)
                    while Int32(got - epoch) < 0:
                        got = ld_acquire_sys_u32(flag)
                cute.arch.barrier()
                src = base + Int64(FLAGS_BYTES) + (Int64(s) * Int64(rows) + Int64(r0)) * rb
            dst = out + (Int64(s) * Int64(rows) + Int64(r0)) * rb
            _copy_rows(src, dst, dst, nr * nvec_row, tid, False, False, self.threads, self.unroll)
            i = _claim(base, sitem, tid)
        _finish_call(base, epoch, Int32(self.P + self.Q), Int32(0), tid)


@functools.cache
def compiled_reduce_scatter(
    world: int,
    chunk_rows: int,
    pushers: int,
    consumers: int,
    threads: int = THREADS,
    unroll: int = UNROLL,
    slices: int = RS_SLICES,
):
    kernel = SpReduceScatterKernel(world, chunk_rows, pushers, consumers, threads, unroll, slices)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(16),
        Int32(0),
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def compiled_all_gather(
    world: int,
    chunk_rows: int,
    pushers: int,
    consumers: int,
    threads: int = THREADS,
    unroll: int = UNROLL,
    slices: int = RS_SLICES,
):
    kernel = SpAllGatherKernel(world, chunk_rows, pushers, consumers, threads, unroll, slices)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(16),
        Int32(0),
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


__all__ = [
    "FLAGS_BYTES",
    "MAX_CHUNKS",
    "MAX_WORLD",
    "SpAllGatherKernel",
    "SpReduceScatterKernel",
    "compiled_all_gather",
    "compiled_reduce_scatter",
]
