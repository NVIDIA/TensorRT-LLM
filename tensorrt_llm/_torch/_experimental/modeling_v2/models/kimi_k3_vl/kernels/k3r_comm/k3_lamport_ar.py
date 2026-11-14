# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MNNVL Lamport all-reduce whose input the producer pushes (SM100 / SM103 / SM107).

The MNNVL all-reduce (``trtllm::mnnvl_fusion_allreduce``, ``mnnvlAllreduceKernels.cu``) starts
by copying every rank's BF16 input row slices into Lamport buffers: in the two-shot kernel each
token row goes to slot ``(t // W, rank)`` of the scatter stage of its owner ``t % W``, in the
one-shot kernel to slot ``(t, rank)`` of every rank's buffer (multicast). This module splits
that copy off: a producer stores its output straight into those slots (the K3 low-M FP8 GEMM's
``ar_push`` epilogue, or :func:`run_lamport_push` for any BF16 tensor), and
:class:`K3LamportFinishKernel` runs the rest of the all-reduce on the same workspace:

* two-shot: the owner of token ``t`` polls its ``W`` slots until no 32-bit word holds the
  sentinel ``0x80000000``, sums them in FP32 in rank order, rounds to BF16, replaces ``-0`` by
  ``+0`` and multicasts the row into the broadcast stage of every rank; every rank then polls the
  broadcast rows it needs and copies them to the output;
* one-shot: every rank polls the ``W`` slots of each token, sums them (chunks of eight ranks
  above eight ranks, like the C++ kernel) and rounds to BF16.

The finish kernel clears the previous call's buffer back to the sentinel and advances the
workspace's rotation state (current / dirty buffer, bytes to clear, CTA arrival counter) exactly
like the C++ kernels, so fused and C++ calls can alternate on one workspace. Values are
sanitized (``-0`` sent as ``+0``) before they enter a Lamport buffer, which the sentinel needs;
the sums use the C++ order and round to nearest even, so the output is bitwise that of the C++
all-reduce of the same input. Either transport can reproduce either C++ kernel's result (see
:class:`K3LamportFinishKernel`).

Ordering (details in :class:`K3LamportFinishKernel`): the producer reads the rotation state
after its own ``griddepcontrol.wait``; the finish kernel's workers read it and poll without
waiting for the producer, and only its updater CTA waits for the producer before it advances
the state. Calls that share a workspace must be ordered by streams (or events) on every rank,
every rank must make the same sequence of calls, and every PDL kernel between two calls must
execute ``griddepcontrol.wait``.

A consumer kernel can also finish a call itself after its ``griddepcontrol.wait``
(``k3_attn_res`` with ``lamport_finish``): :func:`poll_reduce_v4` sums a token's one-shot slots
in the finish kernel's order, :func:`finish_two_shot_v4` reduces and multicasts an owned two-shot
row or polls another rank's, :func:`clear_dirty_stages` re-arms the previous call's buffer and
:func:`arrive_finish` advances the rotation state once every CTA has read it.

All-gather row (``ag``, one-shot transport only): a call of ``m`` rows can carry, after its
``m x W`` slots, ``m`` more rows of the same width that the ranks fill column-wise (each rank
multicasts its own columns of every row; :func:`ag_row_addr`). The finish then returns, per
element, ``bf16(y + bf16(sum of the slots))`` with ``y`` the all-gathered value: the BF16 add of
a column-parallel projection's output and an all-reduced partial (the K3 MoE tail with
latent_up split by output columns). The rotation state then re-arms ``m x (W + 1)`` rows.
"""

from __future__ import annotations

import dataclasses
import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Threads per CTA of the finish and push kernels; one thread owns one 16-byte vector.
THREADS = 128
#: 32-bit word of an empty Lamport slot (``-0.0`` FP32; a BF16 pair ``(+0, -0)``).
SENTINEL = -2147483648  # 0x80000000
#: Number of Lamport buffers in a workspace (rotation period).
NUM_BUFFERS = 3
#: Word of the CTA arrival counter in the workspace's state flags (uint32 [9]: current buffer,
#: dirty buffer, bytes per buffer, dirty stages, bytes to clear per stage x4, arrivals).
_FLAG_ARRIVALS = 8


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers
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


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return cutlass.Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def ld_volatile_v4(addr, *, loc=None, ip=None):
    """16 bytes of a Lamport buffer (or the state flags), read past L1."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_nc_u64(addr, *, loc=None, ip=None):
    """A 64-bit word of a tensor no kernel writes (the peer pointer table)."""
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
def ld_v4(addr, *, loc=None, ip=None):
    """16 bytes of a tensor the previous grid wrote (read after ``griddepcontrol.wait``)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """16-byte store to local memory or to a peer's unicast mapping."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multimem_st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """16-byte store to the address in every rank's buffer (multicast mapping)."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_sentinel_v4(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def clean4(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 if none of the four words is the sentinel (the vector has arrived)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, $1, 0x80000000;\nsetp.ne.and.b32 p, $2, 0x80000000, p;\n"
            "setp.ne.and.b32 p, $3, 0x80000000, p;\nsetp.ne.and.b32 p, $4, 0x80000000, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sanitize_bf16x2(word, *, loc=None, ip=None):
    """Replace each ``-0`` half (0x8000) of a BF16 pair by ``+0``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\n.reg .b32 l, h;\n"
            "and.b32 l, $1, 0xffff;\nsetp.eq.b32 p, l, 0x8000;\nselp.b32 l, 0, l, p;\n"
            "and.b32 h, $1, 0xffff0000;\nsetp.eq.b32 q, h, 0x80000000;\nselp.b32 h, 0, h, q;\n"
            "or.b32 $0, l, h;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def acc_bf16x2(word, acc_lo, acc_hi, *, loc=None, ip=None):
    """``(acc_lo + lo, acc_hi + hi)`` in FP32 (round to nearest) for the BF16 pair ``word``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip), _fv(acc_lo, loc=loc, ip=ip), _fv(acc_hi, loc=loc, ip=ip)],
        "{\n.reg .b32 l, h;\n.reg .f32 fl, fh;\n"
        "shl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 fl, l;\nmov.b32 fh, h;\n"
        "add.rn.f32 $0, $3, fl;\nadd.rn.f32 $1, $4, fh;\n}",
        "=f,=f,r,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), cutlass.Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def add_f32(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(lo, loc=loc, ip=ip), _fv(hi, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ag_add_bf16x2(acc_lo, acc_hi, yword, *, loc=None, ip=None):
    """``bf16(y + bf16(acc))`` for the pair ``(acc_lo, acc_hi)`` of FP32 slot sums and the BF16
    pair ``yword`` (round to nearest even throughout), packed with ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(acc_lo, loc=loc, ip=ip), _fv(acc_hi, loc=loc, ip=ip), _iv(yword, loc=loc, ip=ip)],
            "{\n.reg .b32 s, sl, sh, yl, yh;\n.reg .f32 fsl, fsh, fyl, fyh;\n"
            "cvt.rn.bf16x2.f32 s, $2, $1;\n"
            "shl.b32 sl, s, 16;\nand.b32 sh, s, 0xffff0000;\n"
            "shl.b32 yl, $3, 16;\nand.b32 yh, $3, 0xffff0000;\n"
            "mov.b32 fsl, sl;\nmov.b32 fsh, sh;\nmov.b32 fyl, yl;\nmov.b32 fyh, yh;\n"
            "add.rn.f32 fyl, fyl, fsl;\nadd.rn.f32 fyh, fyh, fsh;\n"
            "cvt.rn.bf16x2.f32 $0, fyh, fyl;\n}",
            "=r,f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_release_add(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_acq_rel_u32(addr, value, *, loc=None, ip=None):
    """``atomicAdd`` (acquire-release, GPU scope); returns the old value."""
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
def atom_add_relaxed_u32(addr, value, *, loc=None, ip=None):
    """``atomicAdd`` (relaxed, GPU scope); returns the old value. The thread stalls only where it
    reads the result."""
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
def ld_acquire_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@cute.jit
def lamport_state(flags_addr: Int64):
    """``(cur, dirty, bytes_per_buffer, dirty_stages, clear0, clear1, clear2, clear3)`` of the
    workspace, read past L1. Valid once the previous call on the workspace has completed (see
    the ordering rules of the kernels)."""
    f = ld_volatile_v4(flags_addr)
    c = ld_volatile_v4(flags_addr + Int64(16))
    return f[0], f[1], f[2], f[3], c[0], c[1], c[2], c[3]


@cute.jit
def stage_addr(base: Int64, index: Int32, stage: Int32, num_stages: int, bytes_per_buffer: Int32):
    """Address of ``stage`` of Lamport buffer ``index`` (``LamportBufferLayout::getStagePtr``)."""
    stage_bytes = Int64(bytes_per_buffer // num_stages)
    return base + Int64(index * num_stages + stage) * stage_bytes


@cute.jit
def ag_row_addr(
    base: Int64, cur: Int32, nbytes: Int32, m: Int32, tok: Int32, world: int, hidden: int
):
    """Address of row ``tok`` of the all-gather rows of a one-shot call of ``m`` rows in buffer
    ``cur`` (they follow the call's ``m x world`` slots); ``base``: a unicast or the multicast
    base of the workspace."""
    return stage_addr(base, cur, Int32(0), 1, nbytes) + Int64(m * world + tok) * (hidden * 2)


@cute.jit
def poll_v4(addr: Int64):
    """The 16-byte Lamport vector at ``addr``, polled until it has arrived."""
    w = ld_volatile_v4(addr)
    while clean4(*w) == 0:
        w = ld_volatile_v4(addr)
    return w


@cute.jit
def clear_stage(addr: Int64, nbytes: Int32, gtid: Int32, nthreads: Int32):
    """Re-arm ``nbytes`` (rounded up to 16) at ``addr`` with the sentinel, grid-strided."""
    nvec = (nbytes + 15) // 16
    for i in cutlass.range(gtid, nvec, nthreads, unroll=1):
        st_sentinel_v4(addr + Int64(i) * 16)


@cute.jit
def push_slot(
    peer_ptrs: Int64,
    mc_base: Int64,
    cur: Int32,
    nbytes: Int32,
    tok: Int32,
    rank: Int32,
    elem: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
):
    """Address where ``rank`` stores its values ``[elem, elem + 8)`` of token ``tok``: slot
    ``(tok // W, rank)`` of the owner's scatter stage (two-shot, through the owner's unicast
    mapping), or slot ``(tok, rank)`` of every rank's buffer (one-shot, multicast address)."""
    row_bytes = hidden * 2
    addr = Int64(0)
    if cutlass.const_expr(one_shot):
        addr = stage_addr(mc_base, cur, Int32(0), 1, nbytes) + Int64(tok * world + rank) * row_bytes
    else:
        owner = ld_nc_u64(peer_ptrs + Int64(tok % world) * 8)
        addr = (
            stage_addr(owner, cur, Int32(0), 2, nbytes)
            + Int64((tok // world) * world + rank) * row_bytes
        )
    return addr + Int64(elem) * 2


@cute.jit
def push_v4(addr: Int64, w0: Int32, w1: Int32, w2: Int32, w3: Int32, one_shot: bool):
    """Store one sanitized 16-byte vector into a Lamport slot."""
    s = [sanitize_bf16x2(w) for w in (w0, w1, w2, w3)]
    if cutlass.const_expr(one_shot):
        multimem_st_v4(addr, *s)
    else:
        st_v4(addr, *s)


def reduce_chunk(world: int, cpp_one_shot: bool) -> int:
    """Ranks per partial sum of the C++ kernel a reduction reproduces (see
    :class:`K3LamportFinishKernel`): chunks of eight ranks for the one-shot kernel above eight
    ranks, up to sixteen ranks in one chunk for the two-shot kernel."""
    return (world if world <= 8 else 8) if cpp_one_shot else min(world, 16)


@cute.jit
def poll_reduce_v4(base: Int64, slot_bytes: int, world: int, chunk: int):
    """Poll the ``world`` 16-byte slots at ``base + r * slot_bytes`` until no word holds the
    sentinel, then sum them in FP32 in rank order, ``chunk`` ranks per partial sum (the order of
    :meth:`K3LamportFinishKernel.reduce_slots`); returns eight FP32 values (four BF16 pairs)."""
    words = cute.make_rmem_tensor((world * 4,), Int32)
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for r in cutlass.range_constexpr(world):
            w = ld_volatile_v4(base + Int64(r * slot_bytes))
            for e in cutlass.range_constexpr(4):
                words[r * 4 + e] = w[e]
            ready = ready & clean4(*w)
    acc = [cutlass.Float32(0.0)] * 8
    for c0 in cutlass.range_constexpr(0, world, chunk):
        part = [cutlass.Float32(0.0)] * 8
        for r in cutlass.range_constexpr(c0, min(c0 + chunk, world)):
            for q in cutlass.range_constexpr(4):
                part[2 * q], part[2 * q + 1] = acc_bf16x2(
                    words[r * 4 + q], part[2 * q], part[2 * q + 1]
                )
        for i in cutlass.range_constexpr(8):
            acc[i] = add_f32(acc[i], part[i])
    return acc


@cute.jit
def clear_dirty_stages(
    uc_local: Int64,
    dirty: Int32,
    nbytes: Int32,
    dstages: Int32,
    c0: Int32,
    c1: Int32,
    c2: Int32,
    c3: Int32,
    gtid: Int32,
    nthreads: Int32,
):
    """Re-arm every stage the previous call on the workspace used (its buffer ``dirty``, ``dstages``
    stages, ``c0..c3`` bytes each, from the rotation state), grid-strided over ``nthreads``."""
    clears = (c0, c1, c2, c3)
    for s in cutlass.range_constexpr(4):
        if s < dstages:
            clear_stage(
                uc_local + Int64(dirty * dstages + s) * Int64(nbytes // dstages),
                clears[s],
                gtid,
                nthreads,
            )


@cute.jit
def arrive_finish(
    flags: Int64,
    total: Int32,
    cur: Int32,
    nbytes: Int32,
    m: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
    extra_rows: int = 0,
):
    """Count the calling CTA (one thread per CTA, after every thread of the CTA has read the
    rotation state) for a call of ``m`` rows that ``total`` CTAs finish; the last CTA to arrive
    writes the next call's state (buffer ``cur`` becomes the dirty one with the stages and bytes
    the transport used) and resets the counter, as :class:`K3LamportFinishKernel`'s updater does.
    The caller has passed ``griddepcontrol.wait``: the producer, which reads the state after its
    own wait, has completed. ``extra_rows``: rows per token past the ``world`` slots (the
    all-gather row) that the next call re-arms too (one-shot transport)."""
    arrivals = flags + Int64(_FLAG_ARRIVALS * 4)
    old = atom_add_acq_rel_u32(arrivals, Int32(1))
    arrive_rotate(flags, old, total, cur, nbytes, m, world, hidden, one_shot, extra_rows)


@cute.jit
def arrive_count(flags: Int64):
    """The counting half of :func:`arrive_finish` (one thread per CTA, after every thread of the
    CTA has read the rotation state and the producer has read it too): a relaxed atomic whose
    result only :func:`arrive_rotate` reads, so the thread can go on until then."""
    return atom_add_relaxed_u32(flags + Int64(_FLAG_ARRIVALS * 4), Int32(1))


@cute.jit
def arrive_rotate(
    flags: Int64,
    old: Int32,
    total: Int32,
    cur: Int32,
    nbytes: Int32,
    m: Int32,
    world: int,
    hidden: int,
    one_shot: bool,
    extra_rows: int = 0,
):
    """The rotation half of :func:`arrive_finish`: the CTA whose count ``old`` was the last of
    ``total`` writes the next call's state and resets the counter (read by the next call, which
    starts after this grid has completed). ``extra_rows`` as in :func:`arrive_finish`."""
    if old == total - 1:
        row_bytes = hidden * 2
        if cutlass.const_expr(one_shot):
            st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(1))
            st_v4(
                flags + Int64(16),
                m * ((world + extra_rows) * row_bytes),
                Int32(0),
                Int32(0),
                Int32(0),
            )
        else:
            st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(2))
            st_v4(
                flags + Int64(16),
                ((m + (world - 1)) // world) * (world * row_bytes),
                m * row_bytes,
                Int32(0),
                Int32(0),
            )
        st_u32(arrivals_addr(flags), Int32(0))


def arrivals_addr(flags):
    """Address of the CTA arrival counter in the state words."""
    return flags + Int64(_FLAG_ARRIVALS * 4)


@cute.jit
def finish_two_shot_v4(
    uc_local: Int64,
    mc_base: Int64,
    cur: Int32,
    nbytes: Int32,
    tok: Int32,
    byte_off: Int32,
    rank: Int32,
    store: Int32,
    world: int,
    chunk: int,
    hidden: int,
):
    """16 bytes at ``byte_off`` of row ``tok`` of the two-shot call in buffer ``cur``, finished by
    the caller: on the row's owner (``tok % world == rank``) the rank-order sum of its scatter
    slots (``chunk`` ranks per partial sum), rounded, ``-0`` replaced by ``+0`` and multicast into
    every rank's broadcast stage (where ``store != 0``); elsewhere the broadcast row polled until it
    arrived. Both give the bits :class:`K3LamportFinishKernel` broadcasts."""
    w0 = Int32(0)
    w1 = Int32(0)
    w2 = Int32(0)
    w3 = Int32(0)
    if tok % world == rank:
        scatter = stage_addr(uc_local, cur, Int32(0), 2, nbytes)
        acc = poll_reduce_v4(
            scatter + Int64((tok // world) * world * hidden) * 2 + Int64(byte_off),
            hidden * 2,
            world,
            chunk,
        )
        w0 = sanitize_bf16x2(pack_bf16x2(acc[0], acc[1]))
        w1 = sanitize_bf16x2(pack_bf16x2(acc[2], acc[3]))
        w2 = sanitize_bf16x2(pack_bf16x2(acc[4], acc[5]))
        w3 = sanitize_bf16x2(pack_bf16x2(acc[6], acc[7]))
        if store != 0:
            bcast_mc = stage_addr(mc_base, cur, Int32(1), 2, nbytes)
            multimem_st_v4(bcast_mc + Int64(tok * hidden) * 2 + Int64(byte_off), w0, w1, w2, w3)
    else:
        src = stage_addr(uc_local, cur, Int32(1), 2, nbytes) + Int64(tok * hidden) * 2
        w = ld_volatile_v4(src + Int64(byte_off))
        while clean4(*w) == 0:
            w = ld_volatile_v4(src + Int64(byte_off))
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        w3 = w[3]
    return w0, w1, w2, w3


@cute.jit
def two_shot_token(v: Int32, m: Int32, rank: Int32, world: int):
    """Row handled by the ``v``-th CTA group of a grid that finishes a two-shot call of ``m``
    rows on ``rank``: the rows this rank owns (``rank``, ``rank + world``, ...) first, then the
    others in order, so the owners' reductions are dispatched before the CTAs that poll for
    them."""
    n_own = (m + (world - 1) - rank) // world
    t = rank + v * world
    if v >= n_own:
        j = v - n_own
        q = j // (world - 1)
        rem = j - q * (world - 1)
        t = q * world + rem
        if rem >= rank:
            t = t + 1
    return t


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class K3LamportPushKernel:
    """Producer half for a BF16 tensor that is already in memory: copy ``x [m, hidden]`` into
    the Lamport slots (the C++ kernels' first phase as a kernel of its own). Grid
    ``(m, ceil(hidden / 8 / 128))``, one 16-byte vector per thread."""

    def __init__(self, world: int, hidden: int, one_shot: bool, use_pdl: bool):
        if hidden % 8:
            raise ValueError(f"hidden={hidden} must be a multiple of 8")
        self.W = world
        self.H = hidden
        self.one_shot = one_shot
        self.use_pdl = use_pdl
        self.vecs = hidden // 8
        self.cols = _ceil_div(self.vecs, THREADS)

    @cute.jit
    def __call__(
        self,
        x: Int64,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(x, peer_ptrs, mc_base, flags, m, rank).launch(
            grid=(m, self.cols, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=self.use_pdl
        )

    @cute.kernel
    def kernel(
        self, x: Int64, peer_ptrs: Int64, mc_base: Int64, flags: Int64, m: Int32, rank: Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tok, by, _ = cute.arch.block_idx()
        v = by * THREADS + tidx
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
        cur, _, nbytes, _, _, _, _, _ = lamport_state(flags)
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        if v < self.vecs:
            w = ld_v4(x + Int64(tok * self.H + v * 8) * 2)
            dst = push_slot(
                peer_ptrs, mc_base, cur, nbytes, tok, rank, v * 8, self.W, self.H, self.one_shot
            )
            push_v4(dst, *w, self.one_shot)


class K3LamportFinishKernel:
    """Rest of an MNNVL all-reduce of ``[m, hidden]`` BF16 rows the producer pushes.

    Grid ``m * ceil(hidden / 8 / 128) + 1`` CTAs of 128 threads. Worker CTA ``b`` owns token
    ``b // cols`` and its 16-byte vectors ``(b % cols) * 128 + tid``; the last CTA advances the
    rotation.

    Ordering. The workers read the rotation state and start polling as soon as they are
    resident, without ``griddepcontrol.wait``: the rows they poll validate themselves (no
    sentinel word), so they react the moment the last rank's data lands instead of after this
    rank's producer has completed. Reading the state early is safe because the producer
    releases this kernel only after its own ``griddepcontrol.wait`` (the low-M GEMM's push
    variant and the push kernel both do): by then every earlier grid has completed (each
    waits for its predecessor before it completes), the previous all-reduce on the workspace
    included, so the state is this call's. Every CTA releases its dependents at its start, so
    they become resident once the producer has passed its wait; consumers that read data
    before their own wait (attn_res reads the prefix sum and snapshot rows, GEMMs their
    weights) read only data older than the producer, which is complete by the same argument.
    The updater CTA alone waits for the producer: the producer reads the state after its own
    wait, so the state may only change once it has completed; the updater then waits until
    every worker has counted itself (having read the state) and writes the next call's state,
    like ``LamportFlags::waitAndUpdate``. Its wait also makes this kernel's completion imply
    the producer's, so dependents that wait for this kernel see everything before it.
    ``wait_first`` makes the workers wait for the producer first (no early polling).

    Numerics. ``cpp_one_shot`` names the C++ kernel whose result the call reproduces (default:
    the transport's own): the one-shot kernel sums chunks of eight ranks above eight ranks and
    stores the rounded sum as is; the two-shot kernel sums up to sixteen ranks in one chunk and
    replaces ``-0`` by ``+0`` (its result passes through a Lamport buffer). Either transport
    reproduces either result. The two-shot transport replaces ``-0`` in a one-shot result too,
    which has none: the inputs carry no ``-0``, and every partial sum of BF16 values in FP32 is
    a multiple of ``2**-133``, the smallest BF16 denormal (FP32 rounds only where its spacing
    exceeds that), so a nonzero sum rounds to a nonzero BF16 value and an exact zero is ``+0``
    under round-to-nearest.
    """

    def __init__(
        self,
        world: int,
        hidden: int,
        one_shot: bool,
        use_pdl: bool,
        wait_first: bool = False,
        cpp_one_shot: bool | None = None,
        ag: bool = False,
    ):
        if hidden % 8:
            raise ValueError(f"hidden={hidden} must be a multiple of 8")
        if cpp_one_shot is None:
            cpp_one_shot = one_shot
        if ag and not one_shot:
            raise ValueError("the all-gather row goes with the one-shot transport")
        # The call carries an all-gather row per token: the output is bf16(y + bf16(sum)).
        self.ag = ag
        self.W = world
        self.H = hidden
        self.one_shot = one_shot
        self.use_pdl = use_pdl
        self.wait_first = wait_first
        self.vecs = hidden // 8
        self.cols = _ceil_div(self.vecs, THREADS)
        # Ranks per partial sum of the reproduced C++ kernel, and whether the one-shot transport
        # sanitizes its output like the two-shot kernel (the two-shot transport always does).
        self.chunk = (world if world <= 8 else 8) if cpp_one_shot else min(world, 16)
        self.sanitize_out = not cpp_one_shot

    @cute.jit
    def __call__(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        out: Int64,
        m: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(uc_local, mc_base, flags, out, m, rank).launch(
            grid=(m * self.cols + 1, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def reduce_slots(self, base: Int64, slot_bytes: int):
        """Poll the ``W`` 16-byte slots at ``base + r * slot_bytes`` until every word has arrived,
        then sum them in the C++ order; returns eight FP32 values (four BF16 pairs)."""
        W = self.W
        words = cute.make_rmem_tensor((W * 4,), Int32)
        ready = Int32(0)
        while ready == 0:
            ready = Int32(1)
            for r in cutlass.range_constexpr(W):
                w = ld_volatile_v4(base + Int64(r * slot_bytes))
                for e in cutlass.range_constexpr(4):
                    words[r * 4 + e] = w[e]
                ready = ready & clean4(*w)
        acc = [cutlass.Float32(0.0)] * 8
        for c0 in cutlass.range_constexpr(0, W, self.chunk):
            part = [cutlass.Float32(0.0)] * 8
            for r in cutlass.range_constexpr(c0, c0 + self.chunk):
                for q in cutlass.range_constexpr(4):
                    part[2 * q], part[2 * q + 1] = acc_bf16x2(
                        words[r * 4 + q], part[2 * q], part[2 * q + 1]
                    )
            for i in cutlass.range_constexpr(8):
                acc[i] = add_f32(acc[i], part[i])
        return acc

    @cute.kernel
    def kernel(
        self, uc_local: Int64, mc_base: Int64, flags: Int64, out: Int64, m: Int32, rank: Int32
    ):
        W, H = self.W, self.H
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        workers = m * self.cols
        arrivals = flags + Int64(_FLAG_ARRIVALS * 4)
        if bidx == workers:
            # Updater: the next call's state, once the producer is done with this call's.
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
                griddepcontrol_wait()
            if tidx == 0:
                cur, _, nbytes, _, _, _, _, _ = lamport_state(flags)
                arrived = ld_acquire_u32(arrivals)
                while arrived < workers:
                    arrived = ld_acquire_u32(arrivals)
                row_bytes = H * 2
                clear0 = m * ((W + (1 if self.ag else 0)) * row_bytes)
                clear1 = Int32(0)
                if cutlass.const_expr(not self.one_shot):
                    clear0 = ((m + (W - 1)) // W) * (W * row_bytes)
                    clear1 = m * row_bytes
                st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(1 if self.one_shot else 2))
                st_v4(flags + Int64(16), clear0, clear1, Int32(0), Int32(0))
                st_u32(arrivals, Int32(0))
        else:
            tok = bidx // self.cols
            by = bidx - tok * self.cols
            v = by * THREADS + tidx
            if cutlass.const_expr(self.use_pdl and self.wait_first):
                griddepcontrol_wait()
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_launch_dependents()
            # Count this CTA once every thread has read the state.
            cute.arch.barrier()
            if tidx == 0:
                red_release_add(arrivals, Int32(1))
            row_bytes = H * 2
            out_v = out + Int64(tok * H + v * 8) * 2
            if cutlass.const_expr(self.one_shot):
                buf = stage_addr(uc_local, cur, Int32(0), 1, nbytes)
                if v < self.vecs:
                    acc = self.reduce_slots(buf + Int64(tok * W * H + v * 8) * 2, row_bytes)
                    if cutlass.const_expr(self.ag):
                        # bf16(y + bf16(sum)): the sum rounds as the C++ kernel stores it (no -0
                        # arises, see Numerics), then the all-gathered value is added.
                        y = poll_v4(
                            ag_row_addr(uc_local, cur, nbytes, m, tok, W, H) + Int64(v * 16)
                        )
                        o = [ag_add_bf16x2(acc[2 * q], acc[2 * q + 1], y[q]) for q in range(4)]
                    else:
                        o = [pack_bf16x2(acc[2 * q], acc[2 * q + 1]) for q in range(4)]
                        if cutlass.const_expr(self.sanitize_out):
                            o = [sanitize_bf16x2(w) for w in o]
                    st_v4(out_v, *o)
            else:
                if v < self.vecs:
                    if tok % W == rank:
                        scatter = stage_addr(uc_local, cur, Int32(0), 2, nbytes)
                        acc = self.reduce_slots(
                            scatter + Int64((tok // W) * W * H + v * 8) * 2, row_bytes
                        )
                        o = [
                            sanitize_bf16x2(pack_bf16x2(acc[2 * q], acc[2 * q + 1]))
                            for q in range(4)
                        ]
                        bcast_mc = stage_addr(mc_base, cur, Int32(1), 2, nbytes)
                        multimem_st_v4(bcast_mc + Int64(tok * H + v * 8) * 2, *o)
            # Re-arm the previous call's buffer (every stage it used) while the rows travel.
            gtid = bidx * THREADS + tidx
            nthreads = workers * THREADS
            clears = (c0, c1, c2, c3)
            for s in cutlass.range_constexpr(4):
                if s < dstages:
                    clear_stage(
                        uc_local + Int64(dirty * dstages + s) * Int64(nbytes // dstages),
                        clears[s],
                        gtid,
                        nthreads,
                    )
            if cutlass.const_expr(not self.one_shot):
                if v < self.vecs:
                    src = (
                        stage_addr(uc_local, cur, Int32(1), 2, nbytes) + Int64(tok * H + v * 8) * 2
                    )
                    w = ld_volatile_v4(src)
                    while clean4(*w) == 0:
                        w = ld_volatile_v4(src)
                    st_v4(out_v, *w)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LamportWorkspace:
    """Device addresses of one MNNVL Lamport workspace (``get_or_scale_allreduce_mnnvl_workspace``).

    ``peer_ptrs``: int64 CUDA tensor with every rank's unicast mapping of the workspace (this
    rank's own at index ``rank``); ``flags``: the uint32 rotation state shared with the C++
    kernels; ``bytes_per_buffer``: size of one of the three Lamport buffers.
    """

    world: int
    rank: int
    uc_local: int
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: torch.Tensor
    bytes_per_buffer: int

    @staticmethod
    def from_mnnvl(workspace: dict, world: int, rank: int) -> "LamportWorkspace":
        handle = workspace["handle"]
        peers = [handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)]
        local = workspace["uc_buffer"].data_ptr()
        if peers[rank] != local:
            raise RuntimeError("MNNVL workspace: rank's unicast mapping is not the local buffer")
        return LamportWorkspace(
            world=world,
            rank=rank,
            uc_local=local,
            mc_base=handle.get_mc_buffer((1,), torch.uint8, 0).data_ptr(),
            peer_ptrs=torch.tensor(peers, dtype=torch.int64, device=workspace["uc_buffer"].device),
            flags=workspace["buffer_flags"],
            bytes_per_buffer=int(workspace["buffer_size_bytes"]),
        )


def one_shot_for(num_tokens: int, hidden: int, world: int, threshold_bytes: int) -> bool:
    """The C++ rule: one-shot when ``tokens * hidden * world * 2 <= threshold``."""
    return num_tokens * hidden * world * 2 <= threshold_bytes


def required_bytes(
    num_tokens: int, hidden: int, world: int, one_shot: bool, ag: bool = False
) -> int:
    """Bytes of one Lamport buffer the call needs (as ``MNNVLAllReduce.get_required_workspace_size``;
    ``ag``: with the all-gather rows, one-shot transport)."""
    if one_shot:
        return num_tokens * hidden * (world + (1 if ag else 0)) * 2
    return 2 * _ceil_div(num_tokens, world) * world * hidden * 2


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _compiled_finish(
    world: int,
    hidden: int,
    one_shot: bool,
    use_pdl: bool,
    wait_first: bool = False,
    cpp_one_shot: bool | None = None,
    ag: bool = False,
):
    kernel = K3LamportFinishKernel(world, hidden, one_shot, use_pdl, wait_first, cpp_one_shot, ag)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, Int32(1), Int32(0), _stream())


@functools.cache
def _compiled_push(world: int, hidden: int, one_shot: bool, use_pdl: bool):
    kernel = K3LamportPushKernel(world, hidden, one_shot, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, Int32(1), Int32(0), _stream())


def _check(
    ws: LamportWorkspace, num_tokens: int, hidden: int, one_shot: bool, ag: bool = False
) -> None:
    if hidden % 8:
        raise ValueError(f"hidden={hidden} must be a multiple of 8")
    need = required_bytes(num_tokens, hidden, ws.world, one_shot, ag)
    if need > ws.bytes_per_buffer:
        raise ValueError(
            f"Lamport buffer of {ws.bytes_per_buffer} B too small for {num_tokens} x {hidden} "
            f"({need} B)"
        )


def run_lamport_push(
    ws: LamportWorkspace, x: torch.Tensor, one_shot: bool, use_pdl: bool = True
) -> None:
    """Push the contiguous BF16 ``x [m, hidden]`` into the Lamport slots of the next call."""
    m, hidden = x.shape
    _check(ws, m, hidden, one_shot)
    _compiled_push(ws.world, hidden, one_shot, use_pdl)(
        Int64(x.data_ptr()),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        _stream(),
    )


def run_lamport_finish(
    ws: LamportWorkspace,
    out: torch.Tensor,
    one_shot: bool,
    use_pdl: bool = True,
    wait_first: bool = False,
    cpp_one_shot: bool | None = None,
    ag: bool = False,
) -> torch.Tensor:
    """Complete the all-reduce whose input every rank pushed for this call into ``out``
    (contiguous BF16 ``[m, hidden]``); returns ``out``. ``one_shot`` is the transport the
    producer used; ``cpp_one_shot`` the C++ kernel whose result to reproduce (default: the
    same; see :class:`K3LamportFinishKernel`). ``ag``: the call carries all-gather rows (see the
    module docstring); ``out`` is then ``bf16(y + bf16(sum))``."""
    m, hidden = out.shape
    _check(ws, m, hidden, one_shot, ag)
    _compiled_finish(ws.world, hidden, one_shot, use_pdl, wait_first, cpp_one_shot, ag)(
        Int64(ws.uc_local),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int64(out.data_ptr()),
        Int32(m),
        Int32(ws.rank),
        _stream(),
    )
    return out


__all__ = [
    "K3LamportFinishKernel",
    "K3LamportPushKernel",
    "LamportWorkspace",
    "ag_add_bf16x2",
    "ag_row_addr",
    "arrive_finish",
    "clear_dirty_stages",
    "finish_two_shot_v4",
    "one_shot_for",
    "poll_reduce_v4",
    "poll_v4",
    "push_slot",
    "push_v4",
    "reduce_chunk",
    "required_bytes",
    "two_shot_token",
    "run_lamport_finish",
    "run_lamport_push",
]
