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
# DSpark vanilla-Markov draft chain for a vocab-sharded draft head -- CTM (prims/cute) kernel
# =============================================================================
#
# For every request b and block position k = 0 .. K-1 (greedy, temperature 0):
#   bias[v]      = bf16(markov_w2[v] . markov_w1[prev])          v in this rank's vocab shard
#   corrected[v] = base[b, k, v] + bias[v]                         (fp32)
#   token[b, k]  = the first maximum of corrected over the whole vocabulary (all ranks)
#   prev         = token[b, k]  (prev for k = 0 is the anchor, first_prev[b])
# so the K positions are a chain of dependent global argmaxes across the tensor-parallel ranks.
#
# Numerics are dsparkMarkovChainKernel's (the C++ kernel this replaces): lane l of a warp sums the 8 columns
# 8l .. 8l+7 of a row by FMAs from 0, the 32 lane sums are added by the xor tree 16, 8, 4, 2, 1, the sum is rounded
# to bf16 (cvt.rn), and added to the fp32 base. Argmax order: larger value, then lower vocabulary index; NaN never
# wins. The order is total, so every split of the reduction gives the same token.
#
# Geometry: G CTAs of 256 threads, CTA c owning the RC = S / G rows [c RC, c RC + RC) of the shard (RC a multiple
# of 64). Warp w owns the RC / 8 contiguous rows [w RC / 8, ...), in blocks of 8 (row j's sum ends on lane 4j after
# the reduce-scatter form of the xor tree). The CTA's rows of markov_w2 are bulk-copied into shared memory before the
# grid-dependency wait (a weight) and stay there for all K positions. markov_w1[prev] (512 bytes per request) is one
# bulk copy per CTA and position into shared memory, issued by the thread that reduces request b's global argmax, on
# one phase of an mbarrier per position.
#
# Exchange (one NVLink hop per position): every CTA reduces its rows to one (value, index) entry per request and
# multicast-stores it into slot [k][b][rank][c] of every rank's Lamport buffer; every CTA then polls all
# W x G entries of [k][b] in its own rank's copy (the data is the ready signal: 0x80000000 = empty; values carry
# -0.0 as +0.0 and indices are non-negative, so no entry is the sentinel) and reduces them with the same order.
# Every CTA of every rank computes the same global argmax: no leader, no grid barrier, no publish word.
#
# Buffers: 3 rotating Lamport buffers. Call n uses buffer n % 3 and re-arms buffer (n - 1) % 3 (its readers, all
# CTAs of call n - 1, finished before this call's grid wait returned; its next writer is a remote rank's call n + 2,
# which needs this rank's call n + 1 pushes). flags: [0] buffer of this call, [1] CTAs arrived, [2] words the
# previous call used (to re-arm), [3] unused. The last CTA to arrive advances them.
#
# Roles: warp 0's elected lane issues the weight bulk copy (and the early dependent trigger); thread b < B pushes
# request b's CTA entry, reduces its global argmax and requests the next markov_w1 row; all 8 warps compute and poll
# (the chain inside a position is serial, and the poll is spread over all 256 threads: one load wave). Per position:
# rows → barrier → push → poll → barrier → (thread b) argmax, next row; no barrier between the argmax and the next
# position (its readers wait for the row's phase).
# =============================================================================
"""CTM DSpark Markov chain: the draft head's greedy intra-block Markov bias, the per-position global argmax
across the tensor-parallel vocab shards (the draft tokens), next_new_tokens and the target's KV-length rewind after
the draft forward, in one launch."""

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


@cute.kernel
def k3_markov_kernel(
    base: cutlass.Array,  # fp32 [B * K * S] (or its int16 bf16 view with base_bf16): the draft head's shard logits
    first_prev: cutlass.Array,  # int64 [B]: the anchor token of each request
    w1: cutlass.Array,  # int32 words of bf16 markov_w1 [V, 256]
    w2: cutlass.Array,  # int32 words of bf16 markov_w2[shard rows] [S, 256]
    corrected: cutlass.Array,  # fp32 [B * K * S], out
    tokens: cutlass.Array,  # int32 [B * K], out
    next_new: cutlass.Array,  # int32 [B * (K + 1)], out
    accepted: cutlass.Array,  # int32 [*, accepted_stride]: accepted tokens
    num_accepted: cutlass.Array,  # int32 [B]
    accepted_rows: cutlass.Array,  # int32 [B]: the row of accepted of each request
    kv_lens: cutlass.Array,  # int32: the target's KV lengths, rewound here after the draft forward
    rewind: cutlass.Array,  # int32 [rewind_count]: subtracted from kv_lens[rewind_first ...], clamped at 0
    buf_uc: cutlass.Array,  # int32 words: this rank's Lamport buffers
    buf_mc: cutlass.Array,  # int32 words: their multicast mapping
    flags: cutlass.Array,  # int32 [4]
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
    batch: cutlass.Constexpr[int],  # B
    slots: cutlass.Constexpr[int],  # rank slots per (k, b): world x push_copies
    push_copies: cutlass.Constexpr[
        int
    ],  # test only: each rank's entry goes into push_copies consecutive slots
    buf_words: cutlass.Constexpr[int],
    base_bf16: cutlass.Constexpr[bool],
):
    rows_per_warp = rows // WARPS
    blocks_per_warp = rows_per_warp // BLOCK_ROWS
    entries = slots * grid  # (value, index) entries per (k, b)
    vectors = entries // 2  # 16-byte loads per (k, b)
    vec_per_thread = (vectors + THREADS - 1) // THREADS
    used_words = block * batch * entries * 2

    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tx % cutlass.Int32(32)

    s_w2 = cutlass.Array(
        cutlass.Int32, rows * ROW_WORDS, space=cutlass.AddressSpace.smem, alignment=128
    )
    # [b]: markov_w1[prev] of the position (one bulk copy per request, issued by the thread that learns prev).
    s_w1 = cutlass.Array(
        cutlass.Int32, batch * ROW_WORDS, space=cutlass.AddressSpace.smem, alignment=128
    )
    s_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    s_mbar_w1 = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # [b][warp] (value bits, index): the CTA reduction of the rows, then of the polled entries.
    s_row_best = cutlass.Array(
        cutlass.Int32, batch * WARPS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_poll_best = cutlass.Array(
        cutlass.Int32, batch * WARPS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_flags = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    s_valid = cutlass.Array(cutlass.Int32, MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16)

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(s_mbar, 1)
            prims.mbarrier_init(s_mbar_w1, batch)
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
        s_flags.store(flags.load(idx=0, is_volatile=True), idx=0)
        s_flags.store(flags.load(idx=2, is_volatile=True), idx=2)
    if tx < cutlass.Int32(batch):
        _fetch_w1_row(
            s_w1, s_valid, w1, s_mbar_w1, tx, cutlass.Int32(first_prev.load(idx=tx)), vocab
        )
    prims.barrier_cta_sync(0)
    cur = s_flags.load(idx=0)
    cur_base = cur * cutlass.Int32(buf_words)

    # Re-arm the previous call's buffer (this CTA's 16-byte chunks of the words it used).
    dirty_base = ((cur + cutlass.Int32(2)) % cutlass.Int32(BUFFERS)) * cutlass.Int32(buf_words)
    dirty_words = s_flags.load(idx=2)
    empty = cutlass.Int32(EMPTY_WORD)
    w_clear = (cta * cutlass.Int32(THREADS) + tx) * cutlass.Int32(4)
    while w_clear < dirty_words:
        buf_uc.store((empty, empty, empty, empty), idx=dirty_base + w_clear, alignment=16)
        w_clear += cutlass.Int32(grid * THREADS * 4)

    # next_new_tokens[b] = [accepted[row_b, num_accepted[b] - 1], tokens[b, :]] (-1 wraps, as torch indexing).
    if cta == 0:
        if tx < cutlass.Int32(batch):
            col = num_accepted.load(idx=tx) - cutlass.Int32(1)
            col = cutlass.Int32(cutlass.select_(col < cutlass.Int32(0), col + accepted_stride, col))
            row = accepted_rows.load(idx=tx)
            next_new.store(
                accepted.load(idx=row * accepted_stride + col), idx=tx * cutlass.Int32(block + 1)
            )
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
    for k in cutlass.range(block, unroll=1):
        for b in cutlass.range(batch, unroll=1):
            base_row = (b * cutlass.Int32(block) + k) * cutlass.Int32(shard)
            # Lane 4j finishes row j of each 8-row block (the reduce-scatter leaves row j's sum on lanes 4j .. 4j+3):
            # the warp's base logits, in flight during the wait below.
            has_row = (lane & cutlass.Int32(3)) == cutlass.Int32(0)
            lane_row = shard_row0 + (lane >> cutlass.Int32(2))
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
                    base_vals.append(
                        base.load(idx=base_row + lane_row + cutlass.Int32(g * BLOCK_ROWS))
                    )
            # markov_w1[prev] landed (the position's phase of the W1 barrier).
            while not cute.arch.mbarrier_test_wait(s_mbar_w1.data_ptr(), k & cutlass.Int32(1)):
                pass
            valid = s_valid.load(idx=b) != cutlass.Int32(0)
            # markov_w1[prev], columns 8 lane .. 8 lane + 7, as fp32 (zeros for an invalid prev).
            w1v = s_w1.load(
                idx=b * cutlass.Int32(ROW_WORDS) + lane * 4, vector_size=4, alignment=16
            )
            w1f = []
            for q in cutlass.range_constexpr(4):
                word = cutlass.Int32(
                    cutlass.select_(valid, cutlass.Int32(w1v[q]), cutlass.Int32(0))
                )
                w1f.append(_lo_f32(word))
                w1f.append(_hi_f32(word))
            best_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
            best_index = cutlass.Int32(INT_MAX)
            for g in cutlass.range_constexpr(blocks_per_warp):
                dot = _block_dot(s_w2, row0 + cutlass.Int32(g * BLOCK_ROWS), lane, w1f)
                bias = _lo_f32(_pack_bf16x2(dot, dot))
                value = _add_rn(base_vals[g], bias)
                if has_row:
                    my_row = lane_row + cutlass.Int32(g * BLOCK_ROWS)
                    corrected.store(value, idx=base_row + my_row)
                    best_value, best_index = _keep_better(
                        best_value, best_index, value, vocab_offset + my_row
                    )
            best_value, best_index = _warp_best(best_value, best_index)
            if lane == 0:
                s_row_best.store((best_value.bitcast(cutlass.Int32), best_index),
                                 idx=(b * cutlass.Int32(WARPS) + warp) * cutlass.Int32(2), alignment=8)  # fmt: skip
        prims.barrier_cta_sync(0)
        # Thread b pushes request b's CTA maximum into every rank's buffer.
        if tx < cutlass.Int32(batch):
            cta_value, cta_index = _reduce_slots(s_row_best, tx * cutlass.Int32(WARPS * 2))
            bits = cta_value.bitcast(cutlass.Int32)
            bits = cutlass.Int32(
                cutlass.select_(bits == cutlass.Int32(EMPTY_WORD), cutlass.Int32(0), bits)
            )
            for c in cutlass.range_constexpr(push_copies):
                s_idx = rank * cutlass.Int32(push_copies) + cutlass.Int32(c)
                word = cur_base + (((k * cutlass.Int32(batch) + tx) * cutlass.Int32(slots) + s_idx)
                                   * cutlass.Int32(grid) + cta) * cutlass.Int32(2)  # fmt: skip
                buf_mc.store((bits, cta_index), idx=word, alignment=8)
            # Drain the posted multicast store now (the polls below issue no release that would).
            prims.fence_acq_rel(prims.MemScope.CLUSTER)
        # Poll every rank's entries of [k][b] (this thread's 16-byte vectors, all loads in flight together) until
        # none is empty; the pass that finds them all complete also reduces them.
        for b in cutlass.range(batch, unroll=1):
            region = cur_base + (k * cutlass.Int32(batch) + b) * cutlass.Int32(entries * 2)
            best_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
            best_index = cutlass.Int32(INT_MAX)
            pending = cutlass.Boolean(True)
            while pending:
                pass_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
                pass_index = cutlass.Int32(INT_MAX)
                pass_pending = cutlass.Boolean(False)
                for i in cutlass.range_constexpr(vec_per_thread):
                    v_idx = tx + cutlass.Int32(i * THREADS)
                    live = v_idx < cutlass.Int32(vectors)
                    got = buf_uc.load(
                        idx=region
                        + cutlass.Int32(cutlass.select_(live, v_idx, cutlass.Int32(0)))
                        * cutlass.Int32(4),
                        vector_size=4,
                        alignment=16,
                        is_volatile=True,
                    )
                    for e in cutlass.range_constexpr(2):
                        vw = cutlass.Int32(got[2 * e])
                        iw = cutlass.Int32(got[2 * e + 1])
                        pass_pending = pass_pending | (live & ((vw == empty) | (iw == empty)))
                        take = live & _better(
                            vw.bitcast(cutlass.Float32), iw, pass_value, pass_index
                        )
                        pass_value = cutlass.Float32(
                            cutlass.select_(take, vw.bitcast(cutlass.Float32), pass_value)
                        )
                        pass_index = cutlass.Int32(cutlass.select_(take, iw, pass_index))
                best_value = pass_value
                best_index = pass_index
                pending = pass_pending
            best_value, best_index = _warp_best(best_value, best_index)
            if lane == 0:
                s_poll_best.store((best_value.bitcast(cutlass.Int32), best_index),
                                  idx=(b * cutlass.Int32(WARPS) + warp) * cutlass.Int32(2), alignment=8)  # fmt: skip
        prims.barrier_cta_sync(0)
        if tx < cutlass.Int32(batch):
            g_value, g_index = _reduce_slots(s_poll_best, tx * cutlass.Int32(WARPS * 2))
            # The next position's markov_w1 row, requested as soon as prev is known: every reader of s_w1 and
            # s_valid for this position passed the barrier before the push. No barrier follows: the other threads
            # go on to the next position's base logits and wait for the row's phase. s_poll_best is next written
            # after the next push barrier, which this thread reaches only after these reads.
            if k + cutlass.Int32(1) < cutlass.Int32(block):
                _fetch_w1_row(s_w1, s_valid, w1, s_mbar_w1, tx, g_index, vocab)
            if cta == 0:
                tokens.store(g_index, idx=tx * cutlass.Int32(block) + k)
                next_new.store(g_index, idx=tx * cutlass.Int32(block + 1) + k + cutlass.Int32(1))

    # Every CTA read the flags before arriving; the last one advances them for the next call.
    if tx == 0:
        arrived = _atomic_add_acq_rel(flags.data_ptr(1).toint(), cutlass.Int32(1))
        if arrived == cutlass.Int32(grid - 1):
            flags.store(cutlass.Int32(0), idx=1)
            flags.store(cutlass.Int32(used_words), idx=2)
            flags.store((cur + cutlass.Int32(1)) % cutlass.Int32(BUFFERS), idx=0)


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
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    k3_markov_kernel(
        base, first_prev, w1, w2, corrected, tokens, next_new, accepted, num_accepted, accepted_rows, kv_lens,
        rewind, buf_uc, buf_mc, flags, vocab, vocab_offset, rank, accepted_stride, rewind_first, rewind_count,
        grid, rows, shard, block, batch, slots, push_copies, buf_words, base_bf16,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
