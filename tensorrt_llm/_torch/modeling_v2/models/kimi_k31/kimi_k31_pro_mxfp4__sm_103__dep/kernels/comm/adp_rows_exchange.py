# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 row exchanges of the attention-DP vocab-parallel LM heads over NVLink peer mappings (CuTe DSL, SM100 /
SM103).

The vocab-parallel LM head under attention DP (``kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head``)
moves rows twice per call: every rank's hidden rows to every rank (an all-gather), and each rank's vocab-shard results
of every rank's rows back to that rank (an all-to-all). Both kernels here store 16-byte vectors straight into the peers'
regions of one symmetric workspace (``McastGPUBuffer``) through the peers' unicast mappings and publish with a
system-scope fence followed by a relaxed flag store, as the family's MoE all-to-all does (``moe_a2a``):

* gather: rank ``s`` stores its ``rows`` rows as rows ``s * R ..`` of every rank's gather region ``[W * R, row_bytes]``
  (``R``: the call's padded row count per rank; padding rows keep whatever they held). The grid's last CTA fences,
  sets this rank's gather flag on every rank to the call's epoch and waits for every rank's, so the kernel ends with
  every rank's rows in place.
* return: rank ``p`` stores row ``j * R + r`` of its ``[W * R, col_bytes]`` shard results as bytes
  ``[p * col_bytes, (p + 1) * col_bytes)`` of row ``r`` of rank ``j``'s return region ``[R, W * col_bytes]``: the
  owner's rows of the full-vocab result, in vocab order. The grid's last CTA publishes; every CTA then waits for every
  rank's flag and, with ``widen``, converts this rank's ``rows`` BF16 rows to FP32 into the output.

Copies only: the rows arrive bit for bit. The epoch is a call counter in the workspace that every CTA reads at its
start and the grid's last CTA advances once every rank has published, so CUDA-graph replays need no host state; flags
only grow and are compared as ``flag - epoch >= 0``. A call never overwrites rows a peer still reads: a peer reads
its gather region (the shard GEMM) before it pushes that call's return, which this rank waits for before its next
gather; it reads its return region (the widen, or the merge after the kernel) before its next gather, which every
rank waits for before pushing its next return. Every rank of the group must make the same calls in the same order
with the same ``R``. The return kernel's CTAs all wait for the peers, so its grid is at most one CTA per SM.
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

from .moe_a2a import (
    atom_add_acq_rel_gpu,
    fence_acq_rel_sys,
    ld_acquire_sys_u32,
    ld_cg_v4,
    ld_nc_u64,
    ld_relaxed_u32,
    ld_v4,
    st_relaxed_u32,
    st_v4,
)

#: Largest group the flag layout holds.
MAX_WORLD = 32
#: Flag words (uint32; the hot ones on their own 128-byte lines): the call counter, the gather and return CTA
#: counts, the gather flags and the return flags (one per sender).
EPOCH, GCOUNT, RCOUNT, GFLAGS, RFLAGS = 0, 32, 64, 128, 192
#: Bytes reserved for the flags at the start of the workspace.
FLAGS_BYTES = 4096
#: Threads per CTA of both kernels.
THREADS = 512


@dsl_user_op
def bf16x2_to_f32x2(word, *, loc=None, ip=None):
    """The two BF16 halves of a 32-bit word as FP32 bit patterns ``(low, high)`` (exact)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int32(word).ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\nand.b32 $1, $2, 0xffff0000;",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2))


@cute.jit
def _publish(
    base: Int64,
    peers: Int64,
    rank: Int32,
    epoch: Int32,
    nblk: Int32,
    tid: Int32,
    sitem: cute.Tensor,
    count_word: cutlass.Constexpr,
    flag_word: cutlass.Constexpr,
    world: cutlass.Constexpr,
):
    """Count this CTA as done (its stores ordered before the count); the grid's last CTA fences at system scope and
    sets word ``flag_word + rank`` of every rank to ``epoch``, then re-arms the count. Returns the CTA's ticket
    (``nblk - 1`` for the last)."""
    cute.arch.barrier()
    if tid == 0:
        sitem[0] = atom_add_acq_rel_gpu(base + Int64(count_word * 4), Int32(1))
    cute.arch.barrier()
    ticket = sitem[0]
    if ticket == nblk - 1:
        if tid < world:
            fence_acq_rel_sys()
            peer = ld_nc_u64(peers + Int64(tid) * 8)
            st_relaxed_u32(peer + Int64(flag_word * 4) + Int64(rank) * 4, epoch)
        if tid == 0:
            st_relaxed_u32(base + Int64(count_word * 4), Int32(0))
    return ticket


@cute.jit
def _wait(
    base: Int64, epoch: Int32, tid: Int32, flag_word: cutlass.Constexpr, world: cutlass.Constexpr
):
    """Threads ``0 .. world - 1`` wait until every rank's flag has reached ``epoch``; then the CTA syncs."""
    if tid < world:
        addr = base + Int64(flag_word * 4) + Int64(tid) * 4
        got = ld_acquire_sys_u32(addr)
        while got - epoch < 0:
            got = ld_acquire_sys_u32(addr)
    cute.arch.barrier()


class AdpGatherKernel:
    """All-gather of ``rows`` rows of ``row_bytes`` (padded per rank to ``R``) into every rank's gather region."""

    def __init__(self, world: int, row_bytes: int, threads: int = THREADS):
        if not 2 <= world <= MAX_WORLD or row_bytes % 16:
            raise ValueError(f"world {world} / row bytes {row_bytes} not served")
        self.W, self.RB, self.V, self.threads = world, row_bytes, row_bytes // 16, threads

    @cute.jit
    def __call__(
        self,
        src: Int64,
        peers: Int64,
        base: Int64,
        goff: Int64,
        rows: Int32,
        R: Int32,
        rank: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(src, peers, base, goff, rows, R, rank).launch(
            grid=(ctas, 1, 1), block=(self.threads, 1, 1), smem=128, stream=stream
        )

    @cute.kernel
    def kernel(
        self, src: Int64, peers: Int64, base: Int64, goff: Int64, rows: Int32, R: Int32, rank: Int32
    ):
        W = self.W
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        sitem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=16
        )
        epoch = ld_relaxed_u32(base + Int64(EPOCH * 4)) + 1
        dst = goff + Int64(rank) * Int64(R) * self.RB
        for i in cutlass.range(
            bidx * self.threads + tid, rows * self.V, nblk * self.threads, unroll=1
        ):
            w = ld_v4(src + Int64(i) * 16)
            # Destinations from the next rank on, this rank last: the ranks start on different peers.
            for k in cutlass.range_constexpr(W):
                peer = ld_nc_u64(peers + Int64((rank + 1 + k) % W) * 8)
                st_v4(peer + dst + Int64(i) * 16, w[0], w[1], w[2], w[3])
        ticket = _publish(base, peers, rank, epoch, nblk, tid, sitem, GCOUNT, GFLAGS, W)
        if ticket == nblk - 1:
            _wait(base, epoch, tid, GFLAGS, W)
            if tid == 0:
                st_relaxed_u32(base + Int64(EPOCH * 4), epoch)


class AdpReturnKernel:
    """All-to-all of ``[W * R, col_bytes]`` shard results into the owners' ``[R, W * col_bytes]`` return regions,
    and with ``widen`` the owner's ``rows`` BF16 rows converted to FP32 into ``out``."""

    def __init__(self, world: int, col_bytes: int, widen: bool, threads: int = THREADS):
        if not 2 <= world <= MAX_WORLD or col_bytes % 16:
            raise ValueError(f"world {world} / column bytes {col_bytes} not served")
        self.W, self.CB, self.CV, self.widen, self.threads = (
            world,
            col_bytes,
            col_bytes // 16,
            widen,
            threads,
        )

    @cute.jit
    def __call__(
        self,
        local: Int64,
        peers: Int64,
        base: Int64,
        roff: Int64,
        out: Int64,
        rows: Int32,
        R: Int32,
        rank: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(local, peers, base, roff, out, rows, R, rank).launch(
            grid=(ctas, 1, 1), block=(self.threads, 1, 1), smem=128, stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        local: Int64,
        peers: Int64,
        base: Int64,
        roff: Int64,
        out: Int64,
        rows: Int32,
        R: Int32,
        rank: Int32,
    ):
        W, CB, CV = self.W, self.CB, self.CV
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        sitem = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,)), byte_alignment=16
        )
        epoch = ld_relaxed_u32(base + Int64(EPOCH * 4)) + 1
        gtid = bidx * self.threads + tid
        nthreads = nblk * self.threads
        per_owner = R * CV
        # Item i: owner (rank + 1 + i // per_owner) % W, its row r, vector v: consecutive threads store consecutive
        # vectors of one owner's row; the ranks start on different owners.
        for i in cutlass.range(gtid, W * per_owner, nthreads, unroll=1):
            jj = i // per_owner
            rem = i - jj * per_owner
            r = rem // CV
            v = rem - r * CV
            j = (rank + 1 + jj) % W
            w = ld_v4(local + (Int64(j) * Int64(R) + Int64(r)) * CB + Int64(v) * 16)
            peer = ld_nc_u64(peers + Int64(j) * 8)
            st_v4(
                peer + roff + Int64(r) * (W * CB) + Int64(rank) * CB + Int64(v) * 16,
                w[0],
                w[1],
                w[2],
                w[3],
            )
        ticket = _publish(base, peers, rank, epoch, nblk, tid, sitem, RCOUNT, RFLAGS, W)
        _wait(base, epoch, tid, RFLAGS, W)
        if cutlass.const_expr(self.widen):
            region = base + roff
            for i in cutlass.range(gtid, rows * (W * CV), nthreads, unroll=1):
                w = ld_cg_v4(region + Int64(i) * 16)
                f = [bf16x2_to_f32x2(w[e]) for e in range(4)]
                o = out + Int64(i) * 32
                st_v4(o, f[0][0], f[0][1], f[1][0], f[1][1])
                st_v4(o + 16, f[2][0], f[2][1], f[3][0], f[3][1])
        if ticket == nblk - 1:
            if tid == 0:
                st_relaxed_u32(base + Int64(EPOCH * 4), epoch)


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def compiled_gather(world: int, row_bytes: int, threads: int = THREADS):
    """The gather kernel for ``world`` ranks and rows of ``row_bytes``, compiled once per process."""
    a = Int64(1 << 20)
    return cute.compile(
        AdpGatherKernel(world, row_bytes, threads),
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(1),
        _stream(),
    )


@functools.cache
def compiled_return(world: int, col_bytes: int, widen: bool, threads: int = THREADS):
    """The return kernel for ``world`` ranks, shard rows of ``col_bytes`` and ``widen``, compiled once per process."""
    a = Int64(1 << 20)
    return cute.compile(
        AdpReturnKernel(world, col_bytes, widen, threads),
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(1),
        _stream(),
    )


__all__ = [
    "FLAGS_BYTES",
    "MAX_WORLD",
    "THREADS",
    "AdpGatherKernel",
    "AdpReturnKernel",
    "compiled_gather",
    "compiled_return",
]
