# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# DSpark vanilla-Markov draft chain for a vocab-sharded draft head -- CTM (prims/cute) kernel, BATCH-1 VARIANT WITH
# THE LL (FLAG-IN-PAYLOAD) EXCHANGE (MARKOV_LL)
# A copy of dspark_markov_chain_kernel_b1.py (S4B8) for batch 1 only, with the per-position exchange replaced:
# - post: one 16-byte multicast store per CTA and position, {value bits, tag, index, tag}; no fence and no separate
#   flag word (the tag is in the payload);
# - poll + combine: warp 0 alone polls every entry of the position (16-byte loads) until both tag words of every
#   entry hold this launch's tag, and reduces them with warp shuffles; no CTA barrier after the poll (the other warps
#   go on to the next position's base loads and wait for the markov_w1 row, which only warp 0 can request);
# - the CTA's row reduction before the post: warp 0 waits on a named barrier the other warps only arrive at.
# Everything else is b1's: smem-resident markov_w2, the GEMV FMA order, the bf16 roundings, the fp32-sum option, the
# PDL wait placement, the argmax order (larger value, then lower vocabulary index; NaN never wins). Bitwise the same
# tokens as the b1 kernel: every CTA entry is computed as b1 computes it, and the global reduction folds the same set
# of entries under the same total order (any fold order gives the same winner).
# =============================================================================
#
# For request 0 (batch 1) and block position k = 0 .. K-1 (greedy, temperature 0):
#   bias[v]      = bf16(markov_w2[v] . markov_w1[prev])          v in this rank's vocab shard
#   value[v]     = bf16(base[0, k, v] + bias[v])                  (the reference's bf16 add: fp32 sum, one rounding)
#   token[0, k]  = the first maximum of value over the whole vocabulary (all ranks)
#   prev         = token[0, k]  (prev for k = 0 is the anchor, first_prev[0])
#
# Numerics: dspark_markov_chain_kernel_b1.py's, unchanged (lane l sums the 8 columns 8l .. 8l+7 of a row by FMAs from
# 0, xor-tree lane sums, cvt.rn to bf16, fp32 add, cvt.rn; with fp32_sum the unrounded fp32 sum is compared).
#
# Geometry: b1's (G CTAs of 256 threads, RC rows per CTA, warp w owns RC / 8 rows in blocks of 8, markov_w2 rows in
# shared memory before the grid-dependency wait, markov_w1[prev] one bulk copy per position on an mbarrier phase).
#
# LL exchange. The workspace (dspark_markov_chain.workspace_args) holds the 3 Lamport buffers of the other variants
# (untouched here) followed by LL_PARITIES = 2 LL regions of buf_words words each (zero-filled at allocation):
#   region p = buf_uc[(BUFFERS + p) * buf_words ...], entry e = (k * slots + rank slot) * G + CTA, 4 words at 4 e:
#   {value bits, tag, index, tag}
# Each 8-byte half carries the tag, so a reader that sees the tag in both halves has seen both halves of this write
# (8-byte single-copy granularity of a 16-byte store, the assumption of NCCL's LL protocol).
# Tag: seq, the launch sequence number of this workspace's LL calls (flags[LL_SEQ] holds the last one; this call
# uses seq = that + 1, skipping 0 on wrap; every rank runs the same calls in the same order, so every rank computes
# the same seq). Launch seq uses region seq & 1. The position is in the address, not the tag.
# Why a stale entry is never accepted: an entry is accepted only if both tag words equal this call's seq. Every LL
# call writes every entry it reads (k < K, every rank slot, every CTA) with its own seq, so an entry left by another
# call carries a different seq unless 2^32 LL calls passed since it was written (> 40 h of back-to-back calls of
# only a smaller shape), and the zero fill (tag 0) is never a seq.
# Why no entry is overwritten while a reader still needs it: call n + 1 on rank q uses the other region; call n + 2
# (this region again) on rank q starts after q's call n + 1 completed, which waited for every rank's call n + 1
# entries, which a rank posts only after its call n completed (griddepcontrol.wait before the first post). So no
# rank still reads call n's region when it is next written. (One region would do for K >= 2; two make K = 1 safe and
# keep calls of different shapes apart: the region stride is fixed, the layout inside a region follows the shape.)
# Memory ordering: the post is multimem.st.relaxed.sys (no release, no fence: the payload is the data, nothing else
# is published); the poll is ld.volatile (relaxed, system scope) 16-byte loads of this rank's copy (no acquire: no
# other data is read through the tag). flags: read after griddepcontrol.wait, advanced by the last CTA to arrive
# (atom.acq_rel.gpu), as in b1.
#
# Roles per position: all 8 warps compute their rows; lane 0 of each warp writes the warp's best to shared memory;
# warps 1..7 arrive at named barrier LL_BAR and go on to the next position's base loads and the markov_w1 wait; warp
# 0 waits at LL_BAR, thread 0 reduces the 8 warp bests and posts the CTA entry, warp 0 polls the W x G entries
# (ceil(W G / 32) 16-byte loads per lane, all issued before any is used), folds them per lane and across the warp
# (xor tree), and thread 0 requests the next markov_w1 row (and, in CTA 0, stores the token). The next position's
# readers of s_row_best / s_w1 / s_valid write or read them only after the row's mbarrier phase, which thread 0
# completes after its reads of s_row_best (mbarrier arrive: release; the phase wait: acquire).
# =============================================================================
"""CTM DSpark Markov chain, batch-1 variant with the LL (flag-in-payload) exchange: the draft head's greedy
intra-block Markov bias, the per-position global argmax across the tensor-parallel vocab shards (the draft tokens),
next_new_tokens and the target's KV-length rewind after the draft forward, in one launch."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

THREADS = 256
WARPS = THREADS // 32
MARKOV_RANK = 256  # 32 lanes x 8 columns
ROW_WORDS = MARKOV_RANK // 2  # int32 words (bf16 pairs) per markov row
ROW_BYTES = MARKOV_RANK * 2
BLOCK_ROWS = 8  # rows per warp block: one per epilogue lane
MAX_BATCH = 16
MAX_BLOCK = 16
BUFFERS = 3
FLAG_WORDS = 4
EMPTY_WORD = -(2**31)  # 0x80000000: fp32 -0.0, never a pushed word
INT_MAX = 2**31 - 1
NEG_INF_BITS = -8388608  # 0xFF800000
COPY_ROWS = 32  # rows per weight bulk copy (16 KB)
SMEM_ROW_BUDGET = 400  # 200 KB of markov_w2 rows per CTA
SMEM_BYTES = 227 * 1024
LL_PARITIES = 2  # LL regions after the Lamport buffers, buf_words words each
LL_FLAG_WORDS = 8  # flags words this variant needs: [4] last LL seq, [5] LL CTAs arrived ([0..3]: the other variants)
LL_SEQ = 4
LL_ARRIVED = 5
LL_ENTRY_WORDS = 4  # {value bits, tag, index, tag}
LL_BAR = 1  # named barrier: warp 0 waits for the 8 warp bests (barrier 0 is __syncthreads)


def rows_per_cta(shard: int, grid: int) -> int:
    return shard // grid


def supports(shard: int, grid: int, block: int, batch: int) -> bool:
    """Shapes the kernel runs: whole 8-row blocks for every warp (RC a multiple of 64), the CTA's markov_w2 rows
    within the shared-memory budget, block <= 16, batch <= 16, an even entry count per exchange."""
    if grid <= 0 or shard % grid != 0:
        return False
    rc = rows_per_cta(shard, grid)
    return (
        rc % (WARPS * BLOCK_ROWS) == 0
        and rc <= SMEM_ROW_BUDGET
        and smem_bytes(rc, batch) <= SMEM_BYTES
        and grid % 2 == 0
        and 0 < block <= MAX_BLOCK
        and 0 < batch <= MAX_BATCH
    )


def smem_bytes(rows: int, batch: int) -> int:
    """Shared memory of a CTA: its markov_w2 rows, the markov_w1[prev] rows, the two [b][warp] reductions, the
    barriers, flags and prev (with the allocator's alignment padding)."""
    return rows * ROW_BYTES + batch * ROW_BYTES + 2 * batch * WARPS * 8 + 1024


def buffer_words(block: int, batch: int, slots: int, grid: int) -> int:
    """Int32 words of one Lamport buffer: one (value, index) entry per position, request, rank slot and CTA."""
    return block * batch * slots * grid * 2


@dsl_user_op
def _atomic_add_acq_rel(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.global.add.u32, returning the old value."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    """fma.rn.f32 (fmaf)."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip), c.ir_value(loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    """add.rn.f32: an fp32 add the compiler cannot contract with a neighbouring multiply."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even (cvt.rn, as __float2bfloat16_rn)."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _lo_f32(word):
    """The bf16 in the low half of a word, as fp32 (exact)."""
    return cutlass.Int32(word << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _hi_f32(word):
    """The bf16 in the high half of a word, as fp32 (exact)."""
    return cutlass.Int32(word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


def _better(value, index, best_value, best_index):
    """dsparkMarkovChainKernel's keepBetter order: larger value, or an equal value at a lower index. NaN never
    compares better, so it never wins."""
    return (value > best_value) | ((value == best_value) & (index < best_index))


def _keep_better(best_value, best_index, value, index):
    take = _better(value, index, best_value, best_index)
    return cutlass.Float32(cutlass.select_(take, value, best_value)), cutlass.Int32(
        cutlass.select_(take, index, best_index)
    )


def _warp_best(value, index):
    """Every lane gets the warp's best (value, index) (xor tree: the order is total, so all lanes agree)."""
    for offset in [16, 8, 4, 2, 1]:
        other_value = cute.arch.shuffle_sync_bfly(value, offset=offset, mask=-1, mask_and_clamp=31)
        other_index = cute.arch.shuffle_sync_bfly(index, offset=offset, mask=-1, mask_and_clamp=31)
        value, index = _keep_better(value, index, other_value, other_index)
    return value, index


def _block_dot(s_w2, first_row, lane, w1f):
    """markov_w2[first_row + j] . markov_w1[prev] for the block's 8 rows, in the C++ kernel's order: lane l sums the
    8 columns 8l .. 8l+7 by an FMA chain from 0, then the 32 lane sums are added in pairs at lane distances 16, 8, 4, 2,
    1 (the xor tree). Done as a reduce-scatter: at distances 16, 8 and 4 a lane keeps half of its rows and adds its
    partner's partial of each kept row (the pairs of the full tree, so the same sums: 9 shuffles instead of 40), then
    the full tree at 2 and 1. Returns row (lane >> 2)'s sum."""
    parts = []
    for r in range(BLOCK_ROWS):
        w2v = s_w2.load(idx=(first_row + cutlass.Int32(r)) * cutlass.Int32(ROW_WORDS) + lane * cutlass.Int32(4),
                        vector_size=4, alignment=16)  # fmt: skip
        acc = cutlass.Float32(0.0)
        for q in range(4):
            word = cutlass.Int32(w2v[q])
            acc = _fma_rn(w1f[2 * q], _lo_f32(word), acc)
            acc = _fma_rn(w1f[2 * q + 1], _hi_f32(word), acc)
        parts.append(acc)
    for offset in [16, 8, 4]:
        upper = (lane & cutlass.Int32(offset)) != cutlass.Int32(0)
        half = len(parts) // 2
        kept = []
        for i in range(half):
            send = cutlass.Float32(cutlass.select_(upper, parts[i], parts[half + i]))
            keep = cutlass.Float32(cutlass.select_(upper, parts[half + i], parts[i]))
            kept.append(
                _add_rn(
                    keep,
                    cute.arch.shuffle_sync_bfly(send, offset=offset, mask=-1, mask_and_clamp=31),
                )
            )
        parts = kept
    dot = parts[0]
    for offset in [2, 1]:
        dot = _add_rn(
            dot, cute.arch.shuffle_sync_bfly(dot, offset=offset, mask=-1, mask_and_clamp=31)
        )
    return dot


def _fetch_w1_row(s_w1, s_valid, w1, mbar, b, prev, vocab):
    """Request b's markov_w1[prev] row into shared memory: one 512-byte bulk copy on the position's barrier phase (a
    prev outside the vocabulary copies row 0 and clears s_valid[b]: the readers zero its bias). s_valid[b] is stored
    before the arrive, whose release the readers' phase wait acquires."""
    valid = (prev >= cutlass.Int32(0)) & (prev < vocab)
    row = cutlass.Int32(cutlass.select_(valid, prev, cutlass.Int32(0)))
    s_valid.store(cutlass.Int32(cutlass.select_(valid, cutlass.Int32(1), cutlass.Int32(0))), idx=b)
    prims.mbarrier_arrive_expect_tx(mbar, ROW_BYTES)
    prims.cp_async_bulk_shared_cluster_global(
        s_w1.subview(b * cutlass.Int32(ROW_WORDS)),
        w1.subview(row * cutlass.Int32(ROW_WORDS)),
        mbar,
        ROW_BYTES,
    )


def _reduce_slots(slots_arr, first_word):
    """The best of 8 (value bits, index) pairs at first_word .. in warp order."""
    value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
    index = cutlass.Int32(INT_MAX)
    for w in range(WARPS):
        pair = slots_arr.load(idx=first_word + cutlass.Int32(w * 2), vector_size=2, alignment=8)
        value, index = _keep_better(
            value, index, cutlass.Int32(pair[0]).bitcast(cutlass.Float32), cutlass.Int32(pair[1])
        )
    return value, index


def ll_region_words(block: int, slots: int, grid: int) -> int:
    """Int32 words of one LL region a call uses: one 4-word entry per position, rank slot and CTA (batch 1)."""
    return block * slots * grid * LL_ENTRY_WORDS


def ll_workspace_words(buf_words: int) -> int:
    """Int32 words of the whole workspace with the LL regions: the Lamport buffers, then LL_PARITIES regions."""
    return (BUFFERS + LL_PARITIES) * buf_words


@dsl_user_op
def _post_ll(addr_i64, a, b, c, d, *, loc=None, ip=None):
    """multimem.st.relaxed.sys.global.v4.f32: one 16-byte store of {a, b, c, d} into every rank's copy (the bits, as
    stored); relaxed at system scope so the peers' volatile polls observe it, with no release (nothing else is
    published through it)."""
    _llvm.inline_asm(
        None,
        [addr_i64.ir_value(loc=loc, ip=ip), a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip), d.ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bar_sync(bar_id: int, threads: int, *, loc=None, ip=None):
    """bar.sync on a named barrier: wait until `threads` threads arrived (or synced); memory ordering as bar.sync."""
    _llvm.inline_asm(
        None, [], f"bar.sync {bar_id}, {threads};", "", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bar_arrive(bar_id: int, threads: int, *, loc=None, ip=None):
    """bar.arrive on a named barrier: count this warp's arrival without waiting (the barrier's completion orders the
    arriving threads' prior memory accesses before the waiting threads' later ones, as bar.sync)."""
    _llvm.inline_asm(
        None, [], f"bar.arrive {bar_id}, {threads};", "", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@cute.kernel
def k3_markov_kernel(
    base: cutlass.Array,  # fp32 [K * S] (or its int16 bf16 view with base_bf16): the draft head's shard logits
    first_prev: cutlass.Array,  # int64 [1]: the anchor token
    w1: cutlass.Array,  # int32 words of bf16 markov_w1 [V, 256]
    w2: cutlass.Array,  # int32 words of bf16 markov_w2[shard rows] [S, 256]
    corrected: cutlass.Array,  # unused (kept for the launch signature)
    tokens: cutlass.Array,  # int32 [K], out
    next_new: cutlass.Array,  # int32 [K + 1], out
    accepted: cutlass.Array,  # int32 [*, accepted_stride]: accepted tokens
    num_accepted: cutlass.Array,  # int32 [1]
    accepted_rows: cutlass.Array,  # int32 [1]: the row of accepted of the request
    kv_lens: cutlass.Array,  # int32: the target's KV lengths, rewound here after the draft forward
    rewind: cutlass.Array,  # int32 [rewind_count]: subtracted from kv_lens[rewind_first ...], clamped at 0
    buf_uc: cutlass.Array,  # int32 words: this rank's workspace (Lamport buffers, then the LL regions)
    buf_mc: cutlass.Array,  # int32 words: its multicast mapping
    flags: cutlass.Array,  # int32 [LL_FLAG_WORDS]
    vocab: cutlass.Int32,  # rows of markov_w1: a prev outside [0, vocab) gets a zero bias
    vocab_offset: cutlass.Int32,  # first vocabulary index of this rank's shard
    rank: cutlass.Int32,
    accepted_stride: cutlass.Int32,
    rewind_first: cutlass.Int32,
    rewind_count: cutlass.Int32,  # 0: no rewind
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],  # RC
    shard: cutlass.Constexpr[int],  # S
    block: cutlass.Constexpr[int],  # K
    batch: cutlass.Constexpr[int],  # B: 1 (the op takes this variant only for batch 1)
    slots: cutlass.Constexpr[int],  # rank slots per k: world x push_copies
    push_copies: cutlass.Constexpr[
        int
    ],  # test only: each rank's entry goes into push_copies consecutive slots
    buf_words: cutlass.Constexpr[int],
    base_bf16: cutlass.Constexpr[bool],
    fp32_sum: cutlass.Constexpr[bool],
):
    rows_per_warp = rows // WARPS
    blocks_per_warp = rows_per_warp // BLOCK_ROWS
    entries = slots * grid  # LL entries per position
    vec_per_lane = (entries + 31) // 32  # 16-byte entry loads per lane of warp 0
    ll_base = BUFFERS * buf_words  # the first LL region

    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tx % cutlass.Int32(32)

    s_w2 = cutlass.Array(
        cutlass.Int32, rows * ROW_WORDS, space=cutlass.AddressSpace.smem, alignment=128
    )
    # markov_w1[prev] of the position (one bulk copy, issued by thread 0 once it learns prev).
    s_w1 = cutlass.Array(cutlass.Int32, ROW_WORDS, space=cutlass.AddressSpace.smem, alignment=128)
    s_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    s_mbar_w1 = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # [warp] (value bits, index): the CTA reduction of the rows.
    s_row_best = cutlass.Array(
        cutlass.Int32, WARPS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_flags = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_valid = cutlass.Array(cutlass.Int32, MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16)

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(s_mbar, 1)
            prims.mbarrier_init(s_mbar_w1, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # The CTA's markov_w2 rows: a weight, loaded before the grid dependency.
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(s_mbar, rows * ROW_BYTES)
            for c in cutlass.range_constexpr((rows + COPY_ROWS - 1) // COPY_ROWS):
                n = min(COPY_ROWS, rows - c * COPY_ROWS)
                prims.cp_async_bulk_shared_cluster_global(
                    s_w2.subview(c * COPY_ROWS * ROW_WORDS),
                    w2.subview(
                        cta * cutlass.Int32(rows * ROW_WORDS)
                        + cutlass.Int32(c * COPY_ROWS * ROW_WORDS)
                    ),
                    s_mbar,
                    n * ROW_BYTES,
                )
        # Dependents wait for this whole grid before reading its outputs.
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tx == 0:
        s_flags.store(flags.load(idx=LL_SEQ, is_volatile=True), idx=0)
        _fetch_w1_row(
            s_w1, s_valid, w1, s_mbar_w1, tx, cutlass.Int32(first_prev.load(idx=0)), vocab
        )
    prims.barrier_cta_sync(0)
    # This call's tag: the last LL call's seq + 1, skipping 0 (the zero fill) on wrap; its region: seq & 1.
    seq = s_flags.load(idx=0) + cutlass.Int32(1)
    seq = cutlass.Int32(cutlass.select_(seq == cutlass.Int32(0), cutlass.Int32(2), seq))
    region0 = cutlass.Int32(ll_base) + (seq & cutlass.Int32(1)) * cutlass.Int32(buf_words)

    # next_new_tokens[0] = [accepted[row_0, num_accepted[0] - 1], tokens[0, :]] (-1 wraps, as torch indexing).
    if cta == 0:
        if tx == 0:
            col = num_accepted.load(idx=0) - cutlass.Int32(1)
            col = cutlass.Int32(cutlass.select_(col < cutlass.Int32(0), col + accepted_stride, col))
            row = accepted_rows.load(idx=0)
            next_new.store(accepted.load(idx=row * accepted_stride + col), idx=0)
        # The KV-length rewind after the draft forward (whose kernels, the readers of kv_lens, all finished before
        # this grid's wait): kv_lens[first + b] = max(kv_lens[first + b] - rewind[b], 0).
        if tx < rewind_count:
            left = kv_lens.load(idx=rewind_first + tx) - rewind.load(idx=tx)
            kv_lens.store(cutlass.Int32(cutlass.select_(left < cutlass.Int32(0), cutlass.Int32(0), left)),
                          idx=rewind_first + tx)  # fmt: skip

    # The weights have landed (issued before the wait).
    while not cute.arch.mbarrier_test_wait(s_mbar.data_ptr(), 0):
        pass

    row0 = warp * cutlass.Int32(rows_per_warp)  # first row of the warp within the CTA
    shard_row0 = cta * cutlass.Int32(rows) + row0  # ... within the shard
    has_row = (lane & cutlass.Int32(3)) == cutlass.Int32(0)
    lane_row = shard_row0 + (lane >> cutlass.Int32(2))
    empty = cutlass.Int32(EMPTY_WORD)
    for k in cutlass.range(block, unroll=1):
        # The position's base logits, issued before the W1 wait (b1's order).
        base_row = k * cutlass.Int32(shard)
        # Lane 4j finishes row j of each 8-row block (the reduce-scatter leaves row j's sum on lanes 4j .. 4j+3).
        base_vals = []
        for g in cutlass.range_constexpr(blocks_per_warp):
            if cutlass.const_expr(base_bf16):
                base_vals.append(
                    cutlass.Int32(
                        cutlass.Int32(
                            base.load(idx=base_row + lane_row + cutlass.Int32(g * BLOCK_ROWS))
                        )
                        << cutlass.Int32(16)
                    ).bitcast(cutlass.Float32)
                )
            else:
                base_vals.append(base.load(idx=base_row + lane_row + cutlass.Int32(g * BLOCK_ROWS)))
        # markov_w1[prev] landed (the position's phase of the W1 barrier).
        while not cute.arch.mbarrier_test_wait(s_mbar_w1.data_ptr(), k & cutlass.Int32(1)):
            pass
        valid = s_valid.load(idx=0) != cutlass.Int32(0)
        # markov_w1[prev], columns 8 lane .. 8 lane + 7, as fp32 (zeros for an invalid prev).
        w1v = s_w1.load(idx=lane * 4, vector_size=4, alignment=16)
        w1f = []
        for q in cutlass.range_constexpr(4):
            word = cutlass.Int32(cutlass.select_(valid, cutlass.Int32(w1v[q]), cutlass.Int32(0)))
            w1f.append(_lo_f32(word))
            w1f.append(_hi_f32(word))
        best_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
        best_index = cutlass.Int32(INT_MAX)
        for g in cutlass.range_constexpr(blocks_per_warp):
            dot = _block_dot(s_w2, row0 + cutlass.Int32(g * BLOCK_ROWS), lane, w1f)
            bias = _lo_f32(_pack_bf16x2(dot, dot))
            # fp32_sum (default): the in-tree chain's arithmetic (fp32 logits from LogitsProcessor's .float() plus
            # the widened bf16 bias): compare the unrounded fp32 sum. Without fp32_sum: a bf16 reference rounds that
            # sum once.
            if cutlass.const_expr(fp32_sum):
                value = _add_rn(base_vals[g], bias)
            else:
                value = _lo_f32(
                    _pack_bf16x2(_add_rn(base_vals[g], bias), _add_rn(base_vals[g], bias))
                )
            if has_row:
                my_row = lane_row + cutlass.Int32(g * BLOCK_ROWS)
                best_value, best_index = _keep_better(
                    best_value, best_index, value, vocab_offset + my_row
                )
        best_value, best_index = _warp_best(best_value, best_index)
        if lane == 0:
            s_row_best.store((best_value.bitcast(cutlass.Int32), best_index), idx=warp * cutlass.Int32(2),
                             alignment=8)  # fmt: skip
        if warp == 0:
            # Every warp's best of this position is in shared memory.
            _bar_sync(LL_BAR, THREADS)
            region = region0 + k * cutlass.Int32(entries * LL_ENTRY_WORDS)
            # Thread 0 posts the CTA entry {value bits, tag, index, tag} into every rank's region: one 16-byte
            # multicast store, no fence.
            if tx == 0:
                cta_value, cta_index = _reduce_slots(s_row_best, cutlass.Int32(0))
                bits = cta_value.bitcast(cutlass.Int32)
                # -0.0 is posted as +0.0, as b1 posts it (equal under the order either way).
                bits = cutlass.Int32(cutlass.select_(bits == empty, cutlass.Int32(0), bits))
                for c in cutlass.range_constexpr(push_copies):
                    s_idx = rank * cutlass.Int32(push_copies) + cutlass.Int32(c)
                    word = region + (s_idx * cutlass.Int32(grid) + cta) * cutlass.Int32(
                        LL_ENTRY_WORDS
                    )
                    _post_ll(buf_mc.data_ptr(word).toint(), bits, seq, cta_index, seq)
            # Warp 0 polls all W x G entries of the position in this rank's copy: lane l reads entries l, l + 32, ..
            # (every load of a pass issued before any is used); a pass in which any entry of the lane does not
            # carry this call's tag in both halves is repeated in full. The order is total, so the lane best and
            # the xor tree give b1's winner.
            pending = cutlass.Boolean(True)
            g_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
            g_index = cutlass.Int32(INT_MAX)
            while pending:
                gots = []
                for i in cutlass.range_constexpr(vec_per_lane):
                    e_idx = lane + cutlass.Int32(i * 32)
                    live = e_idx < cutlass.Int32(entries)
                    gots.append((live, buf_uc.load(
                        idx=region + cutlass.Int32(cutlass.select_(live, e_idx, cutlass.Int32(0)))
                        * cutlass.Int32(LL_ENTRY_WORDS),
                        vector_size=4, alignment=16, is_volatile=True)))  # fmt: skip
                pass_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
                pass_index = cutlass.Int32(INT_MAX)
                pass_pending = cutlass.Boolean(False)
                for live, got in gots:
                    vw = cutlass.Int32(got[0])
                    iw = cutlass.Int32(got[2])
                    tagged = (cutlass.Int32(got[1]) == seq) & (cutlass.Int32(got[3]) == seq)
                    stale = (cutlass.Int32(got[1]) != seq) | (cutlass.Int32(got[3]) != seq)
                    pass_pending = pass_pending | (live & stale)
                    take = (
                        live
                        & tagged
                        & _better(vw.bitcast(cutlass.Float32), iw, pass_value, pass_index)
                    )
                    pass_value = cutlass.Float32(
                        cutlass.select_(take, vw.bitcast(cutlass.Float32), pass_value)
                    )
                    pass_index = cutlass.Int32(cutlass.select_(take, iw, pass_index))
                g_value = pass_value
                g_index = pass_index
                pending = pass_pending
            g_value, g_index = _warp_best(g_value, g_index)
            if tx == 0:
                # The next position's markov_w1 row, requested as soon as prev is known: every reader of s_w1 and
                # s_valid for this position arrived at LL_BAR after its reads. The other warps wait for the row's
                # phase before writing s_row_best again, and this thread read s_row_best before the arrive.
                if k + cutlass.Int32(1) < cutlass.Int32(block):
                    _fetch_w1_row(s_w1, s_valid, w1, s_mbar_w1, tx, g_index, vocab)
                if cta == 0:
                    tokens.store(g_index, idx=k)
                    next_new.store(g_index, idx=k + cutlass.Int32(1))
        else:
            _bar_arrive(LL_BAR, THREADS)

    # Every CTA read the flags before arriving; the last one advances the LL seq for the next call.
    if tx == 0:
        arrived = _atomic_add_acq_rel(flags.data_ptr(LL_ARRIVED).toint(), cutlass.Int32(1))
        if arrived == cutlass.Int32(grid - 1):
            flags.store(cutlass.Int32(0), idx=LL_ARRIVED)
            flags.store(seq, idx=LL_SEQ)


@cute.jit
def k3_markov(
    base: cute.Tensor,
    first_prev: cute.Tensor,
    w1: cute.Tensor,
    w2: cute.Tensor,
    corrected: cute.Tensor,
    tokens: cute.Tensor,
    next_new: cute.Tensor,
    accepted: cute.Tensor,
    num_accepted: cute.Tensor,
    accepted_rows: cute.Tensor,
    kv_lens: cute.Tensor,
    rewind: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    vocab: cutlass.Int32,
    vocab_offset: cutlass.Int32,
    rank: cutlass.Int32,
    accepted_stride: cutlass.Int32,
    rewind_first: cutlass.Int32,
    rewind_count: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    shard: cutlass.Constexpr[int],
    block: cutlass.Constexpr[int],
    batch: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[int],
    push_copies: cutlass.Constexpr[int],
    buf_words: cutlass.Constexpr[int],
    base_bf16: cutlass.Constexpr[bool],
    fp32_sum: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    k3_markov_kernel(
        base, first_prev, w1, w2, corrected, tokens, next_new, accepted, num_accepted, accepted_rows, kv_lens,
        rewind, buf_uc, buf_mc, flags, vocab, vocab_offset, rank, accepted_stride, rewind_first, rewind_count,
        grid, rows, shard, block, batch, slots, push_copies, buf_words, base_bf16, fp32_sum,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
