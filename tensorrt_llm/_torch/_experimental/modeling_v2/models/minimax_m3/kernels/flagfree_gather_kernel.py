# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# The flag-free multicast all-gather of the TP8 decode worker -- CTM (prims/cute) kernel. A copy of
# ops/comm/allgather_cast_kernel.py with its exchange protocol replaced.
# =============================================================================
#
# Every rank r holds shard_r [rows, S] and every rank writes
#   bf16 mode: out[i, r S + j] = float(shard_r[i, j])  (bf16 widened to fp32 by a bit shift)
#   fp32 mode: out[i, r S + j] = shard_r[i, j]         (copied as is)
# i.e. the stock allgather_cast / allgather_f32 (torch.cat(all_gather(shard), -1)[.float()]).
#
# What changes against the stock kernel: the exchange protocol, nothing else (geometry of a vector, the output
# layout, the widening, the epoch word, the launch without programmatic dependent launch).
#   Stock: push the vectors (multimem.st), CTA barrier, raise a per-CTA ready flag with a SYSTEM-scope RELEASE
#   reduction (multimem.red.release.sys), then poll the peers' flags with system-scope ACQUIRE loads. On the TP8
#   decode worker (two trays) a call costs about the same whatever it moves (400 KB per rank for the verify logits,
#   12 KB per rank for the drafter fc output): the cost is the protocol's fixed latency, not the bytes.
#   Here (flag-free, the Lamport form of the multicast all-reduce kernels/m3_mc_allreduce): the data
#   words carry their own readiness. Every word of the workspace that is not this call's data holds EMPTY
#   (0xFFFFFFFF). The push is a relaxed system-scope multicast store (no fence, no flag, as the LL Markov
#   chain posts); the reader polls the 16-byte vectors of every peer with volatile loads until no word is EMPTY, writes
#   them out, and writes EMPTY back over the words it read (re-arm). One one-way NVLink latency instead of a
#   release/acquire round.
#
# Exactness. A word equal to EMPTY is pushed as CANON (0x7FFF7FFF) instead, and this rank's own columns get the same
# replacement, so every rank returns the same tensor. Both are NaN in either reading (two bf16 NaNs, or one fp32 NaN),
# and no other word is changed: the output equals the stock gather's bit for bit except where a shard held that one
# all-ones NaN pattern (its NaN arrives with another payload; GPU arithmetic returns its own canonical NaN for any NaN
# input, so nothing downstream sees the difference). The host module checks this on every rank at set-up against the
# stock kernel (random bit patterns plus every special word, including EMPTY itself).
#
# Workspace (int32 words, behind one multicast mapping): PARITIES = 2 halves of [world][slot] words, slot =
# cap_rows x row_words (row_words = S / 2 in bf16 mode, S in fp32 mode); filled with EMPTY at allocation. flags
# (rank-local int32 [FLAG_WORDS]): [0] calls so far (the epoch), [1] CTAs arrived, [4] debug code, [5] debug detail.
# Call n uses half n & 1.
#
# Why no word is overwritten while a reader still needs it, and why every polled word is EMPTY until this call's push:
# rank q pushes call n + 2 into half (n & 1) of every rank only after q's call n + 1 completed, which needed rank r's
# call n + 1 pushes, which r issues only after its call n completed (the launch waits for all prior work on the stream:
# no programmatic dependent launch, as the stock op launches it), and r's call n re-armed every word of half
# (n & 1) that it read before completing. Calls of other row counts or grids share the epoch; a call touches only the
# words of its own rows, and every one it reads it re-arms, so the rest stay EMPTY. Words of this rank's own slot are
# written by its own multicast store and never read or re-armed (nobody polls them).
#
# Memory ordering: the push is multimem.st.relaxed.sys (no release: only the pushed words are published, each 32-bit
# word single-copy atomic, and each is checked on its own); the poll is ld.volatile (relaxed) 16-byte loads of this
# rank's copy. A poll that sees no EMPTY word for longer than POLL_TIMEOUT_NS (a peer that never pushes) writes the
# debug words and traps instead of hanging, as the stock kernel does.
#
# Geometry: G CTAs of 256 threads; CTA c owns vectors [c N / G, (c + 1) N / G) of every rank's shard (N = rows S / 8
# in bf16 mode, rows S / 4 in fp32 mode); thread t takes vectors lo + t, lo + t + 256, ... For each vector it pushes
# its own, writes its own columns, then polls all peers' copies of that vector together (one pass issues every load
# before any is used) until none holds EMPTY, then writes and re-arms them. Unlike the flag protocol, readiness is per
# word, so a peer that partitions the vectors differently would still be read correctly; every rank uses the same G
# (a function of rows, the shard and the device) anyway.
# =============================================================================
"""Flag-free multicast all-gather (bf16 -> fp32 cast, or fp32 copy) for the TP8 decode worker."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

THREADS = 256
PARITIES = 2
FLAG_WORDS = 8
POLL_TIMEOUT_NS = 30_000_000_000  # 30 s: far past any straggler, short of a hung job
DEBUG_POLL = 3  # the stock kernel uses 2 for its flag poll
VEC_ELEMS = 8  # bf16 per 16-byte vector
VEC_ELEMS_FP32 = 4  # fp32 per 16-byte vector (fp32 mode)
HI_MASK = -65536  # 0xFFFF0000
EMPTY_WORD = -1  # 0xFFFFFFFF: two bf16 NaNs / one fp32 NaN with every payload bit set
CANON_WORD = (
    2147450879  # 0x7FFF7FFF: what a pushed EMPTY word becomes (two bf16 NaNs / one fp32 NaN)
)


def row_words(shard_cols: int, fp32: bool) -> int:
    """Int32 words of one row of a shard of ``shard_cols`` columns."""
    return shard_cols if fp32 else shard_cols // 2


def slot_words(cap_rows: int, shard_cols: int, fp32: bool) -> int:
    """Int32 words of one rank's slot in one half."""
    return cap_rows * row_words(shard_cols, fp32)


def workspace_words(cap_rows: int, shard_cols: int, world: int, fp32: bool) -> int:
    """Int32 words of the multicast workspace: PARITIES halves of world slots."""
    return PARITIES * world * slot_words(cap_rows, shard_cols, fp32)


def vectors(rows: int, shard_cols: int, fp32: bool) -> int:
    """16-byte vectors of one rank's [rows, shard_cols] shard."""
    return rows * shard_cols // (VEC_ELEMS_FP32 if fp32 else VEC_ELEMS)


@dsl_user_op
def _atomic_add_acq_rel(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.global.add.u32, returning the old value (as the stock kernel)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(),
            [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _post(addr_i64, a, b, c, d, *, loc=None, ip=None):
    """multimem.st.relaxed.sys.global.v4.f32: one 16-byte store of {a, b, c, d} into every rank's copy (the bits, as
    stored); relaxed at system scope so the peers' volatile polls observe it, with no release (the LL
    Markov chain posts the same way)."""
    _llvm.inline_asm(
        None,
        [
            addr_i64.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    """%globaltimer (ns)."""
    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _trap(*, loc=None, ip=None):
    """fence.sc.sys (the debug words reach memory), then trap."""
    _llvm.inline_asm(
        None,
        [],
        "fence.sc.sys;\n\ttrap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _store_fp32x8(out, word, got):
    """The 8 bf16 of a 16-byte vector, widened to fp32 bit for bit, as two 16-byte stores at out word `word` (the
    stock kernel's)."""
    lo = []
    for q in range(4):
        w = cutlass.Int32(got[q])
        lo.append(cutlass.Int32(w << cutlass.Int32(16)))
        lo.append(cutlass.Int32(w & cutlass.Int32(HI_MASK)))
    out.store((lo[0], lo[1], lo[2], lo[3]), idx=word, alignment=16)
    out.store((lo[4], lo[5], lo[6], lo[7]), idx=word + cutlass.Int32(4), alignment=16)


def _store_out(out, word, got, fp32: bool):
    """One 16-byte vector at out word `word`: 4 fp32 copied as is (fp32 mode), else 8 bf16 widened (as the
    stock kernel)."""
    if cutlass.const_expr(fp32):
        out.store(
            (
                cutlass.Int32(got[0]),
                cutlass.Int32(got[1]),
                cutlass.Int32(got[2]),
                cutlass.Int32(got[3]),
            ),
            idx=word,
            alignment=16,
        )
    else:
        _store_fp32x8(out, word, got)


def _canon(got, empty, canon):
    """The 4 words of a loaded vector with EMPTY replaced by CANON (every other word unchanged)."""
    return [
        cutlass.Int32(cutlass.select_(cutlass.Int32(got[q]) == empty, canon, cutlass.Int32(got[q])))
        for q in range(4)
    ]


def _has_empty(got, empty):
    """Whether any of the 4 words of a loaded vector is EMPTY."""
    return (
        (cutlass.Int32(got[0]) == empty)
        | (cutlass.Int32(got[1]) == empty)
        | (cutlass.Int32(got[2]) == empty)
        | (cutlass.Int32(got[3]) == empty)
    )


@cute.kernel
def flagfree_gather_kernel(
    shard: cutlass.Array,  # int32 words of this rank's shard [rows, S] (bf16 pairs or fp32)
    out: cutlass.Array,  # int32 bits of fp32 [rows, world * S], out
    buf_uc: cutlass.Array,  # int32 words: this rank's workspace copy
    buf_mc: cutlass.Array,  # int32 words: its multicast mapping
    flags: cutlass.Array,  # int32 [FLAG_WORDS], rank-local
    rank: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shard_cols: cutlass.Constexpr[int],  # S
    world: cutlass.Constexpr[int],
    slot: cutlass.Constexpr[int],  # int32 words per (half, rank) slot
    fp32: cutlass.Constexpr[bool],  # fp32 shard copied as is (else bf16 widened to fp32)
):
    ve = VEC_ELEMS_FP32 if fp32 else VEC_ELEMS
    n_vec = rows * shard_cols // ve
    row_vecs = shard_cols // ve
    out_row = world * shard_cols

    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    s_epoch = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)

    # The shard is the previous kernel's output and the epoch the previous call's: both after the grid wait.
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tx == 0:
        s_epoch.store(flags.load(idx=0, is_volatile=True), idx=0)
    prims.barrier_cta_sync(0)
    epoch = s_epoch.load(idx=0)
    half = (epoch & cutlass.Int32(1)) * cutlass.Int32(world * slot)

    lo = cta * cutlass.Int32(n_vec) // cutlass.Int32(grid)
    hi = (cta + cutlass.Int32(1)) * cutlass.Int32(n_vec) // cutlass.Int32(grid)
    empty = cutlass.Int32(EMPTY_WORD)
    canon = cutlass.Int32(CANON_WORD)

    # Push this CTA's vectors of this rank's shard into slot [half][rank] of every rank (EMPTY words as CANON).
    mine = half + rank * cutlass.Int32(slot)
    v = lo + tx
    while v < hi:
        pw = _canon(shard.load(idx=v * cutlass.Int32(4), vector_size=4, alignment=16), empty, canon)
        _post(buf_mc.data_ptr(mine + v * cutlass.Int32(4)).toint(), pw[0], pw[1], pw[2], pw[3])
        v += cutlass.Int32(THREADS)
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    # This rank's own columns, straight from the input while the peers' pushes are in flight (as the stock kernel),
    # with the same EMPTY -> CANON as the push, so every rank returns the same tensor (the stock contract).
    v = lo + tx
    while v < hi:
        ow = _canon(shard.load(idx=v * cutlass.Int32(4), vector_size=4, alignment=16), empty, canon)
        row = v // cutlass.Int32(row_vecs)
        col = v - row * cutlass.Int32(row_vecs)
        _store_out(
            out,
            row * cutlass.Int32(out_row)
            + rank * cutlass.Int32(shard_cols)
            + col * cutlass.Int32(ve),
            ow,
            fp32,
        )
        v += cutlass.Int32(THREADS)

    # The peers' columns: per vector, every peer's copy polled in one pass (all loads issued before any is used) until
    # no word is EMPTY; then each is loaded again (stable now: only this rank re-arms it), written out and re-armed.
    v = lo + tx
    while v < hi:
        base = v * cutlass.Int32(4)
        start = _globaltimer()
        pending = cutlass.Boolean(True)
        while pending:
            miss = cutlass.Boolean(False)
            for r in cutlass.range_constexpr(world):
                pv = buf_uc.load(
                    idx=half + cutlass.Int32(r * slot) + base,
                    vector_size=4,
                    alignment=16,
                    is_volatile=True,
                )
                miss = miss | (_has_empty(pv, empty) & (cutlass.Int32(r) != rank))
            pending = miss
            if pending:
                if _globaltimer() - start > cutlass.Int64(POLL_TIMEOUT_NS):
                    flags.store(cutlass.Int32(DEBUG_POLL), idx=4)
                    flags.store(v, idx=5)
                    _trap()
        row = v // cutlass.Int32(row_vecs)
        col = v - row * cutlass.Int32(row_vecs)
        for r in cutlass.range_constexpr(world):
            if cutlass.Int32(r) != rank:
                src = half + cutlass.Int32(r * slot) + base
                rv = buf_uc.load(idx=src, vector_size=4, alignment=16, is_volatile=True)
                _store_out(
                    out,
                    row * cutlass.Int32(out_row)
                    + cutlass.Int32(r * shard_cols)
                    + col * cutlass.Int32(ve),
                    rv,
                    fp32,
                )
                buf_uc.store((empty, empty, empty, empty), idx=src, alignment=16)
        v += cutlass.Int32(THREADS)

    # Every CTA read the epoch before arriving; the last one advances it for the next call (as the stock kernel).
    if tx == 0:
        arrived = _atomic_add_acq_rel(flags.data_ptr(1).toint(), cutlass.Int32(1))
        if arrived == cutlass.Int32(grid - 1):
            flags.store(cutlass.Int32(0), idx=1)
            flags.store(epoch + cutlass.Int32(1), idx=0)


@cute.jit
def flagfree_gather(
    shard: cute.Tensor,
    out: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    rank: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shard_cols: cutlass.Constexpr[int],
    world: cutlass.Constexpr[int],
    slot: cutlass.Constexpr[int],
    fp32: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    flagfree_gather_kernel(
        shard,
        out,
        buf_uc,
        buf_mc,
        flags,
        rank,
        grid,
        rows,
        shard_cols,
        world,
        slot,
        fp32,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=False,
    )
