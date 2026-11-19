# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 all-gathers along the last dim in one kernel over the MNNVL Lamport workspace (SM100 / SM103).

Two kernels: :class:`LamportAllGatherKernel` gathers FP32 rows; :class:`LamportVocabGatherKernel`
gathers BF16 rows (the vocab shards of a vocab-parallel LM head) and writes them widened to FP32, the
engine's ``allgather(local, dim=-1).float()`` in one launch. The FP32 kernel is described below; the BF16
one moves 32-bit words (two values) the same way.

``out[t, r * w : (r + 1) * w] = x_r[t, :]`` for every rank ``r`` of the group: each rank multicasts its own row
slices into the current Lamport buffer of every rank (``multimem.st``, at the gathered row's columns), then
polls the other ranks' slices in its local buffer until no 32-bit word holds the sentinel ``0x80000000`` and
writes the gathered rows. One launch replaces the engine's NCCL all-gather for the small exchanges of a decode
step (the DFlash2 drafter's (id, value) candidate pairs: 64 FP32 words per row and rank).

Values are FP32 words; ``-0.0`` (the sentinel's bit pattern) travels as ``+0.0`` on every rank, the only
difference from the NCCL all-gather (the two compare equal). The kernel shares the workspace's rotation
state with the engine MNNVL all-reduce and the family's push / finish kernels (:mod:`.lamport_ar`): it
reads the state after ``griddepcontrol.wait`` (every earlier call on the workspace has completed), re-arms
the previous call's buffer while the rows travel, and its updater CTA writes the next call's state once
every worker CTA has read the current one, like the finish kernel's one-shot transport. Calls that share a
workspace must be in the same order on every rank and ordered by stream.

Each thread owns one 16-byte vector of a gathered row: the vectors of this rank's columns are loaded from
the input, multicast and written; the others are polled. A thread only waits for other ranks' kernels, so
the grid needs no co-residency.
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

from . import lamport_ar as _lam

#: Threads per CTA; one thread owns one 16-byte vector of a gathered row.
THREADS = 128


@dsl_user_op
def sanitize_f32(word, *, loc=None, ip=None):
    """``-0.0`` (``0x80000000``, the Lamport sentinel) as ``+0.0``; every other word unchanged."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.b32 p, $1, 0x80000000;\nselp.b32 $0, 0, $1, p;\n}",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class LamportAllGatherKernel:
    """All-gather of ``x [m, width]`` FP32 (``width`` a multiple of 4) into ``out [m, world * width]``.

    Grid ``m * ceil(world * width / 4 / 128) + 1`` CTAs of 128 threads: worker CTA ``b`` owns gathered row
    ``b // cols`` and its vectors ``(b % cols) * 128 + tid``; the last CTA advances the rotation state.
    """

    def __init__(self, world: int, width: int, use_pdl: bool):
        if width % 4:
            raise ValueError(f"width={width} must be a multiple of 4 FP32 words")
        self.W = world
        self.width = width
        self.own_vecs = width // 4
        self.vecs = world * width // 4
        self.cols = (self.vecs + THREADS - 1) // THREADS
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        x: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, uc_local, mc_base, flags, out, m, rank).launch(
            grid=(m * self.cols + 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = _lam.arrivals_addr(flags)
        row_bytes = self.vecs * 16
        # The input and the rotation state are the previous grids' (the previous call's included).
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_wait()
        if bidx == workers:
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                arrived = _lam.ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _lam.ld_acquire_u32(arrivals)
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * row_bytes, Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            v = by * THREADS + tidx
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            cute.arch.barrier()
            if tidx == 0:
                _lam.red_release_add(arrivals, Int32(1))
            if v < self.vecs:
                off = Int64(tok) * row_bytes + Int64(v) * 16
                dst = out + off
                q = v // self.own_vecs
                if q == rank:
                    src = x + (Int64(tok) * self.width + Int64(v - rank * self.own_vecs) * 4) * 4
                    w = _lam.ld_v4(src)
                    s = [sanitize_f32(w[e]) for e in range(4)]
                    _lam.multimem_st_v4(
                        _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes) + off,
                        s[0],
                        s[1],
                        s[2],
                        s[3],
                    )
                    _lam.st_v4(dst, s[0], s[1], s[2], s[3])
                else:
                    w = _lam.poll_v4(_lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes) + off)
                    _lam.st_v4(dst, w[0], w[1], w[2], w[3])
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                bidx * THREADS + tidx,
                workers * THREADS,
            )


def required_bytes(rows: int, width: int, world: int) -> int:
    """Bytes of one Lamport buffer a call of ``rows`` x ``width`` FP32 words per rank needs."""
    return rows * world * width * 4


@functools.cache
def _compiled(world: int, width: int, use_pdl: bool):
    kernel = LamportAllGatherKernel(world, width, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_lamport_allgather(
    ws: "_lam.LamportWorkspace", x: torch.Tensor, out: torch.Tensor, use_pdl: bool = True
) -> torch.Tensor:
    """Gather the contiguous FP32 ``x [m, width]`` of every rank into ``out [m, world * width]``; returns ``out``."""
    m, width = x.shape
    need = required_bytes(m, width, ws.world)
    if need > ws.bytes_per_buffer:
        raise ValueError(
            f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {width} x {ws.world}"
        )
    _compiled(ws.world, width, use_pdl)(
        Int64(x.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


@dsl_user_op
def bf16x2_lo_f32(word, *, loc=None, ip=None):
    """The FP32 bits of the low BF16 half of ``word`` (exact widening)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $1, 16;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_hi_f32(word, *, loc=None, ip=None):
    """The FP32 bits of the high BF16 half of ``word`` (exact widening)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0xffff0000;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _store_widened_v4(dst: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32):
    """The eight BF16 values of four words, widened to FP32, as 32 bytes at ``dst``."""
    _lam.st_v4(dst, bf16x2_lo_f32(w0), bf16x2_hi_f32(w0), bf16x2_lo_f32(w1), bf16x2_hi_f32(w1))
    _lam.st_v4(
        dst + Int64(16), bf16x2_lo_f32(w2), bf16x2_hi_f32(w2), bf16x2_lo_f32(w3), bf16x2_hi_f32(w3)
    )


@cute.jit
def _poll_from_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32):
    """The 16-byte Lamport vector at ``addr`` polled until it has arrived, starting from an already issued load."""
    a0, a1, a2, a3 = w0, w1, w2, w3
    while _lam.clean4(a0, a1, a2, a3) == 0:
        a0, a1, a2, a3 = _lam.ld_volatile_v4(addr)
    return a0, a1, a2, a3


class LamportVocabGatherKernel:
    """All-gather of BF16 ``x [m, shard]`` (``shard`` a multiple of 8) into FP32 ``out [m, world * shard]``.

    The vocab-parallel LM head's ``allgather(local, dim=-1).float()`` in one launch: the BF16 rows travel
    through the Lamport buffer (two values per 32-bit word, ``world * shard * 2`` bytes per gathered row) and
    every thread widens its 16-byte vectors to 32 bytes of FP32 on the way out. Grid ``m * cols + 1`` CTAs of
    128 threads (``cols = ceil(world * shard / 8 / (128 * vpt))``): worker CTA ``b`` owns gathered row
    ``b // cols`` and its vectors ``(b % cols) * 128 * vpt + j * 128 + tid`` for ``j < vpt``; the last CTA
    advances the rotation state. A thread first multicasts its vectors of this rank's columns, then issues the
    first load of every other vector before it polls any (more loads in flight, fewer spinning threads).

    A word whose bits are the sentinel (low half ``+0.0``, high half ``-0.0``) travels as ``0``: the high
    value arrives as ``+0.0`` on every rank. Every other value, NaN payloads included, is widened unchanged.
    """

    def __init__(self, world: int, shard: int, use_pdl: bool, vpt: int = 1):
        if shard % 8:
            raise ValueError(f"shard={shard} must be a multiple of 8 BF16 values")
        self.W = world
        self.shard = shard
        self.vpt = vpt
        self.own_vecs = shard // 8
        self.vecs = world * shard // 8
        self.chunk = THREADS * vpt
        self.cols = (self.vecs + self.chunk - 1) // self.chunk
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        x: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, uc_local, mc_base, flags, out, m, rank).launch(
            grid=(m * self.cols + 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = _lam.arrivals_addr(flags)
        row_bytes = self.vecs * 16
        # The input and the rotation state are the previous grids' (the previous call's included).
        if cutlass.const_expr(self.use_pdl):
            _lam.griddepcontrol_wait()
        if bidx == workers:
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = _lam.lamport_state(flags)
                arrived = _lam.ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = _lam.ld_acquire_u32(arrivals)
                _lam.st_v4(flags, (cur + 1) % _lam.NUM_BUFFERS, cur, nbytes, Int32(1))
                _lam.st_v4(flags + Int64(16), m * row_bytes, Int32(0), Int32(0), Int32(0))
                _lam.st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            base = by * self.chunk + tidx
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                _lam.griddepcontrol_launch_dependents()
            cute.arch.barrier()
            if tidx == 0:
                _lam.red_release_add(arrivals, Int32(1))
            stage = _lam.stage_addr(uc_local, cur, Int32(0), 1, nbytes) + Int64(tok) * row_bytes
            mstage = _lam.stage_addr(mc_base, cur, Int32(0), 1, nbytes) + Int64(tok) * row_bytes
            # 8 FP32 values (32 bytes) per 16-byte BF16 vector.
            orow = out + Int64(tok) * (self.vecs * 32)
            # This rank's columns first: multicast and write.
            for j in cutlass.range_constexpr(self.vpt):
                v = base + j * THREADS
                if v < self.vecs:
                    if v // self.own_vecs == rank:
                        src = (
                            x + (Int64(tok) * self.own_vecs + Int64(v - rank * self.own_vecs)) * 16
                        )
                        r = _lam.ld_v4(src)
                        w = [sanitize_f32(r[e]) for e in range(4)]
                        _lam.multimem_st_v4(mstage + Int64(v) * 16, w[0], w[1], w[2], w[3])
                        _store_widened_v4(orow + Int64(v) * 32, w[0], w[1], w[2], w[3])
            # Every other vector: the first loads all issued, then polled and written. Past the row's vectors a
            # load reads the next slots of the same Lamport buffer (a served call fills at most half of it) and
            # is never used.
            firsts = []
            for j in cutlass.range_constexpr(self.vpt):
                firsts.append(_lam.ld_volatile_v4(stage + Int64(base + j * THREADS) * 16))
            for j in cutlass.range_constexpr(self.vpt):
                v = base + j * THREADS
                if v < self.vecs:
                    if v // self.own_vecs != rank:
                        f = firsts[j]
                        a0, a1, a2, a3 = _poll_from_v4(
                            stage + Int64(v) * 16, f[0], f[1], f[2], f[3]
                        )
                        _store_widened_v4(orow + Int64(v) * 32, a0, a1, a2, a3)
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            _lam.clear_dirty_stages(
                uc_local,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                bidx * THREADS + tidx,
                workers * THREADS,
            )


def vocab_gather_required_bytes(rows: int, shard: int, world: int) -> int:
    """Bytes of one Lamport buffer a vocab gather of ``rows`` x ``shard`` BF16 values per rank needs."""
    return rows * world * shard * 2


#: Vectors per thread of the vocab gather by gathered rows: one up to this many rows (the extra polls of a
#: multi-vector thread cost more than they hide at one or two rows), four above.
_VPT1_MAX_ROWS = 2


def vocab_gather_vpt(rows: int) -> int:
    """Vectors per thread the vocab gather runs ``rows`` gathered rows with."""
    return 1 if rows <= _VPT1_MAX_ROWS else 4


@functools.cache
def _compiled_vocab(world: int, shard: int, use_pdl: bool, vpt: int):
    kernel = LamportVocabGatherKernel(world, shard, use_pdl, vpt)
    a = Int64(1 << 20)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        Int32(1),
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_lamport_vocab_gather(
    ws: "_lam.LamportWorkspace", x: torch.Tensor, out: torch.Tensor, use_pdl: bool = True
) -> torch.Tensor:
    """Gather the contiguous BF16 ``x [m, shard]`` of every rank into the FP32 ``out [m, world * shard]``
    (widened); returns ``out``."""
    m, shard = x.shape
    need = vocab_gather_required_bytes(m, shard, ws.world)
    if need > ws.bytes_per_buffer:
        raise ValueError(
            f"Lamport buffer of {ws.bytes_per_buffer} B too small for {m} x {shard} x {ws.world} BF16"
        )
    _compiled_vocab(ws.world, shard, use_pdl, vocab_gather_vpt(m))(
        Int64(x.data_ptr()),
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out


__all__ = [
    "LamportAllGatherKernel",
    "LamportVocabGatherKernel",
    "required_bytes",
    "run_lamport_allgather",
    "run_lamport_vocab_gather",
    "sanitize_f32",
    "vocab_gather_required_bytes",
    "vocab_gather_vpt",
]
