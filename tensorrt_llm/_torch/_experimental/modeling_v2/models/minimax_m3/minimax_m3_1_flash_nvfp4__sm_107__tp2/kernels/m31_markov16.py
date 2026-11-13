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
# MARKOV16: the DSpark vanilla-Markov draft chain of the catalog's one-launch kernel
# (ops/dspark/dspark_markov_chain_kernel.py), laid out for the TP2 16-row step (2 requests, K = 7)
# =============================================================================
#
# The same chain, numerics and exchange as the catalog kernel (see its header): for every request b and position k,
# bias = bf16(markov_w2[v] . markov_w1[prev]) by lane l's 8-column FMA chain from 0 and the xor tree 16, 8, 4, 2, 1;
# value = base + bias (fp32 sum, or bf16-rounded); token = the first maximum over the vocabulary (larger value, then
# lower index; NaN never wins: a total order, so any split of the reduction gives the same token); the per-position
# (value, index) exchange through the same three rotating Lamport buffers, flags and entry layout, so this kernel and
# the catalog's kernels share one workspace and may alternate call by call.
#
# What differs (all bitwise the catalog kernel's tokens):
# - Per-request parallel layout: a warp loads each markov_w2 row's words once per position and runs every request's
#   FMA chain on them (the catalog kernel walks the requests one after another, reloading every row per request and
#   streaming the global-memory blocks once per request). Each (row, request) sum keeps its FMA order and xor tree.
# - The grid is the caller's (``grid``): at TP2 the 101,376-row padded shard splits into 198 CTAs x 512 rows on the
#   SM107's 212 SMs instead of the catalog's 144 x 704 (its CTA cap is 152), so a warp owns 8 blocks of 8 rows instead
#   of 11 and streams 2 of them instead of 5.
# - ``w2_regs``: a warp's streamed blocks (at most REG_BLOCKS_MAX) are loaded into registers once, before the grid
#   dependency (a weight), instead of from global memory at every position.
# - ``base_cols`` < the padded row count (the pad-copy fold): ``base`` is the draft head's unpadded shard logits
#   [B, K, base_stride]; a column at or past ``base_cols`` reads as -inf, the value the padded buffer's tail holds.
# - The exchange is polled for every request in one first pass (the catalog batch-1 variant's poll), then per request.
# =============================================================================
"""MARKOV16: the catalog one-launch DSpark Markov chain kernel with a per-request parallel layout, a caller-chosen grid,
register-resident streamed markov_w2 blocks and an unpadded-logits read (TP2 16-row step)."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain_kernel as _cat,
)

THREADS = _cat.THREADS
WARPS = _cat.WARPS
MARKOV_RANK = _cat.MARKOV_RANK
ROW_WORDS = _cat.ROW_WORDS
ROW_BYTES = _cat.ROW_BYTES
BLOCK_ROWS = _cat.BLOCK_ROWS
BUFFERS = _cat.BUFFERS
FLAG_WORDS = _cat.FLAG_WORDS
EMPTY_WORD = _cat.EMPTY_WORD
INT_MAX = _cat.INT_MAX
NEG_INF_BITS = _cat.NEG_INF_BITS
COPY_ROWS = _cat.COPY_ROWS
SMEM_ROW_BUDGET = _cat.SMEM_ROW_BUDGET
SMEM_BLOCKS = _cat.SMEM_BLOCKS
MAX_ROWS_PER_CTA = _cat.MAX_ROWS_PER_CTA
SMEM_BYTES = _cat.SMEM_BYTES
MAX_BLOCK = _cat.MAX_BLOCK
# Requests per call: the per-request parallel layout keeps every request's FMA chains, w1 row and base logits live at
# once, so it is compiled for the 1- and 2-request steps only.
MAX_BATCH = 2  # capacity: fixed: the TP2 16-row step's requests
# Streamed 8-row blocks per warp held in registers under w2_regs (32 registers each per thread).
REG_BLOCKS_MAX = 2  # capacity: fixed: 64 registers of a 256-thread CTA's budget

_lo_f32 = _cat._lo_f32
_hi_f32 = _cat._hi_f32
_fma_rn = _cat._fma_rn
_add_rn = _cat._add_rn
_pack_bf16x2 = _cat._pack_bf16x2
_better = _cat._better
_warp_best = _cat._warp_best
_fetch_w1_row = _cat._fetch_w1_row
_reduce_slots = _cat._reduce_slots
_atomic_add_acq_rel = _cat._atomic_add_acq_rel
smem_rows = _cat.smem_rows
buffer_words = _cat.buffer_words
rows_per_cta = _cat.rows_per_cta


def smem_bytes(rows: int, batch: int) -> int:
    """Shared memory of a CTA (the catalog kernel's layout: register-resident blocks do not change it)."""
    return _cat.smem_bytes(rows, batch)


def streamed_blocks(rows: int) -> int:
    """8-row blocks per warp past the shared-memory budget (read from global memory, or held in registers)."""
    per_warp = rows // WARPS // BLOCK_ROWS
    return per_warp - smem_rows(rows) // (WARPS * BLOCK_ROWS)


def reg_blocks(rows: int, w2_regs: bool) -> int:
    """Streamed blocks per warp a w2_regs compile holds in registers (all of them, or none past REG_BLOCKS_MAX)."""
    n = streamed_blocks(rows)
    return n if w2_regs and 0 < n <= REG_BLOCKS_MAX else 0


def supports(total: int, grid: int, block: int, batch: int) -> bool:
    """Shapes this kernel runs: the catalog kernel's split rules (whole 8-row blocks per warp, at most
    MAX_ROWS_PER_CTA rows per CTA, the shared-memory budget, an even CTA count) at 1..MAX_BATCH requests."""
    return _cat.supports(total, grid, block, batch) and 0 < batch <= MAX_BATCH


def _dots_of_words(words, lane, w1fs):
    """For every request b (``w1fs[b]``, markov_w1[prev_b] columns 8 lane .. 8 lane + 7 as fp32): the 8 rows' dots in
    the catalog _block_dot's order -- lane l sums its 8 columns by an FMA chain from 0, then the reduce-scatter xor tree
    at 16, 8, 4 and the full tree at 2, 1 -- on the rows' words ``words[r]`` (4 int32 words of bf16 pairs per lane),
    unpacked once for every request. Returns, per request, row (lane >> 2)'s sum."""
    parts = [[] for _ in range(len(w1fs))]
    for r in range(BLOCK_ROWS):
        w2v = words[r]
        vals = []
        for q in range(4):
            word = cutlass.Int32(w2v[q])
            vals.append(_lo_f32(word))
            vals.append(_hi_f32(word))
        for b in range(len(w1fs)):
            w1f = w1fs[b]
            acc = cutlass.Float32(0.0)
            for q in range(4):
                acc = _fma_rn(w1f[2 * q], vals[2 * q], acc)
                acc = _fma_rn(w1f[2 * q + 1], vals[2 * q + 1], acc)
            parts[b].append(acc)
    dots = []
    for b in range(len(w1fs)):
        rows_b = parts[b]
        for offset in [16, 8, 4]:
            upper = (lane & cutlass.Int32(offset)) != cutlass.Int32(0)
            half = len(rows_b) // 2
            kept = []
            for i in range(half):
                send = cutlass.Float32(cutlass.select_(upper, rows_b[i], rows_b[half + i]))
                keep = cutlass.Float32(cutlass.select_(upper, rows_b[half + i], rows_b[i]))
                kept.append(
                    _add_rn(
                        keep,
                        cute.arch.shuffle_sync_bfly(
                            send, offset=offset, mask=-1, mask_and_clamp=31
                        ),
                    )
                )
            rows_b = kept
        dot = rows_b[0]
        for offset in [2, 1]:
            dot = _add_rn(
                dot, cute.arch.shuffle_sync_bfly(dot, offset=offset, mask=-1, mask_and_clamp=31)
            )
        dots.append(dot)
    return dots


def _row_words(src, first_row, lane):
    """The 8 rows' words of this lane (columns 8 lane .. 8 lane + 7) from ``src`` (shared or global memory)."""
    return [
        src.load(
            idx=(first_row + cutlass.Int32(r)) * cutlass.Int32(ROW_WORDS) + lane * cutlass.Int32(4),
            vector_size=4,
            alignment=16,
        )
        for r in range(BLOCK_ROWS)
    ]


@cute.kernel
def m31_markov16_kernel(
    base: cutlass.Array,  # int16 view of bf16 (or fp32) [B * K * base_stride]: the draft head's shard logits
    first_prev: cutlass.Array,  # int64 [B]: the anchor token of each request
    w1: cutlass.Array,  # int32 words of bf16 markov_w1 [V, 256]
    w2: cutlass.Array,  # int32 words of bf16 markov_w2[shard rows], zero-padded [shard, 256]
    corrected: cutlass.Array,  # unused (kept for the launch signature)
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
    rows: cutlass.Constexpr[int],  # RC: rows per CTA
    shard: cutlass.Constexpr[int],  # S: the padded shard (grid x RC)
    base_stride: cutlass.Constexpr[
        int
    ],  # columns per (b, k) row of base: S, or the unpadded shard (fold)
    base_cols: cutlass.Constexpr[
        int
    ],  # columns of base that hold logits; columns past it read as -inf
    block: cutlass.Constexpr[int],  # K
    batch: cutlass.Constexpr[int],  # B
    slots: cutlass.Constexpr[int],  # rank slots per (k, b): world x push_copies
    push_copies: cutlass.Constexpr[
        int
    ],  # test only: each rank's entry goes into push_copies consecutive slots
    buf_words: cutlass.Constexpr[int],
    base_bf16: cutlass.Constexpr[bool],
    fp32_sum: cutlass.Constexpr[bool],
    w2_regs: cutlass.Constexpr[bool],
):
    rows_per_warp = rows // WARPS
    blocks_per_warp = rows_per_warp // BLOCK_ROWS
    # S4STREAM: the warp's first smem_blocks blocks are in shared memory, the rest are read from global memory (or, with
    # w2_regs, held in registers from before the grid dependency).
    smem_blocks = smem_rows(rows) // (WARPS * BLOCK_ROWS)
    streamed = smem_blocks < blocks_per_warp
    in_regs = reg_blocks(rows, w2_regs)
    masked = (
        base_cols < shard
    )  # the fold: base holds base_cols columns per row, the rest of the shard is padding
    entries = slots * grid  # (value, index) entries per (k, b)
    vectors = entries // 2  # 16-byte loads per (k, b)
    vec_per_thread = (vectors + THREADS - 1) // THREADS
    used_words = block * batch * entries * 2

    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tx % cutlass.Int32(32)
    row0 = warp * cutlass.Int32(rows_per_warp)  # first row of the warp within the CTA
    shard_row0 = cta * cutlass.Int32(rows) + row0  # ... within the shard
    has_row = (lane & cutlass.Int32(3)) == cutlass.Int32(0)
    lane_row = shard_row0 + (lane >> cutlass.Int32(2))

    s_w2 = cutlass.Array(
        cutlass.Int32, smem_rows(rows) * ROW_WORDS, space=cutlass.AddressSpace.smem, alignment=128
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
    s_valid = cutlass.Array(
        cutlass.Int32, _cat.MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16
    )

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(s_mbar, 1)
            prims.mbarrier_init(s_mbar_w1, batch)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    # The CTA's markov_w2 rows: a weight, loaded before the grid dependency.
    if warp == 0:
        if prims.elect_sync():
            if cutlass.const_expr(not streamed):
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
            else:
                # S4STREAM: each warp's first smem_blocks blocks (its rows [w RPW, w RPW + smem_blocks 8)) land at
                # [w smem_blocks 8, ...) of shared memory.
                per_warp = smem_blocks * BLOCK_ROWS
                prims.mbarrier_arrive_expect_tx(s_mbar, WARPS * per_warp * ROW_BYTES)
                for w in cutlass.range_constexpr(WARPS):
                    for c in cutlass.range_constexpr((per_warp + COPY_ROWS - 1) // COPY_ROWS):
                        n = min(COPY_ROWS, per_warp - c * COPY_ROWS)
                        prims.cp_async_bulk_shared_cluster_global(
                            s_w2.subview((w * per_warp + c * COPY_ROWS) * ROW_WORDS),
                            w2.subview(
                                cta * cutlass.Int32(rows * ROW_WORDS)
                                + cutlass.Int32((w * rows_per_warp + c * COPY_ROWS) * ROW_WORDS)
                            ),
                            s_mbar,
                            n * ROW_BYTES,
                        )
        # Dependents wait for this whole grid before reading its outputs.
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    # w2_regs: the warp's streamed blocks into registers, once, before the grid dependency (a weight). The same words
    # the streamed dot would read at every position.
    reg_words = []
    for gi in cutlass.range_constexpr(in_regs):
        reg_words.append(
            _row_words(w2, shard_row0 + cutlass.Int32((smem_blocks + gi) * BLOCK_ROWS), lane)
        )

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

    neg_inf = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
    for k in cutlass.range(block, unroll=1):
        # Every request's base logits, issued together before the position's W1 wait. Lane 4j finishes row j of each
        # 8-row block (the reduce-scatter leaves row j's sum on lanes 4j .. 4j+3).
        base_all = []
        for b in cutlass.range_constexpr(batch):
            base_row = (cutlass.Int32(b * block) + k) * cutlass.Int32(base_stride)
            base_vals = []
            for g in cutlass.range_constexpr(blocks_per_warp):
                # (Not `col`: the next_new block above assigns that name inside a dynamic branch.)
                base_col = lane_row + cutlass.Int32(g * BLOCK_ROWS)
                if cutlass.const_expr(masked):
                    live_col = base_col < cutlass.Int32(base_cols)
                    at = base_row + cutlass.Int32(
                        cutlass.select_(live_col, base_col, cutlass.Int32(0))
                    )
                else:
                    at = base_row + base_col
                if cutlass.const_expr(base_bf16):
                    v = cutlass.Int32(
                        cutlass.Int32(base.load(idx=at)) << cutlass.Int32(16)
                    ).bitcast(cutlass.Float32)
                else:
                    v = base.load(idx=at)
                if cutlass.const_expr(masked):
                    # The padded buffer's tail: -inf, never a maximum over a live column.
                    v = cutlass.Float32(cutlass.select_(live_col, v, neg_inf))
                base_vals.append(v)
            base_all.append(base_vals)
        # markov_w1[prev] landed for every request (the position's phase of the W1 barrier).
        while not cute.arch.mbarrier_test_wait(s_mbar_w1.data_ptr(), k & cutlass.Int32(1)):
            pass
        w1fs = []
        for b in cutlass.range_constexpr(batch):
            valid = s_valid.load(idx=b) != cutlass.Int32(0)
            # markov_w1[prev], columns 8 lane .. 8 lane + 7, as fp32 (zeros for an invalid prev).
            w1v = s_w1.load(
                idx=cutlass.Int32(b * ROW_WORDS) + lane * 4, vector_size=4, alignment=16
            )
            w1f = []
            for q in cutlass.range_constexpr(4):
                word = cutlass.Int32(
                    cutlass.select_(valid, cutlass.Int32(w1v[q]), cutlass.Int32(0))
                )
                w1f.append(_lo_f32(word))
                w1f.append(_hi_f32(word))
            w1fs.append(w1f)
        best_values = [neg_inf for _ in range(batch)]
        best_indices = [cutlass.Int32(INT_MAX) for _ in range(batch)]
        for g in cutlass.range_constexpr(blocks_per_warp):
            if cutlass.const_expr(not streamed):
                words = _row_words(s_w2, row0 + cutlass.Int32(g * BLOCK_ROWS), lane)
            elif cutlass.const_expr(g < smem_blocks):
                words = _row_words(
                    s_w2,
                    warp * cutlass.Int32(smem_blocks * BLOCK_ROWS) + cutlass.Int32(g * BLOCK_ROWS),
                    lane,
                )
            elif cutlass.const_expr(in_regs > 0):
                words = reg_words[g - smem_blocks]
            else:  # S4STREAM: a streamed block, its rows read from global memory
                words = _row_words(w2, shard_row0 + cutlass.Int32(g * BLOCK_ROWS), lane)
            dots = _dots_of_words(words, lane, w1fs)
            my_index = vocab_offset + lane_row + cutlass.Int32(g * BLOCK_ROWS)
            for b in cutlass.range_constexpr(batch):
                bias = _lo_f32(_pack_bf16x2(dots[b], dots[b]))
                # fp32_sum (default): the in-tree chain's arithmetic (fp32 logits from LogitsProcessor's .float()
                # plus the widened bf16 bias): compare the unrounded fp32 sum. Without fp32_sum: a bf16 reference
                # rounds that sum once.
                if cutlass.const_expr(fp32_sum):
                    value = _add_rn(base_all[b][g], bias)
                else:
                    value = _lo_f32(
                        _pack_bf16x2(_add_rn(base_all[b][g], bias), _add_rn(base_all[b][g], bias))
                    )
                # The catalog's `if has_row: keep_better(...)`, as a select (only lane 4j holds row j's sum).
                take = has_row & _better(value, my_index, best_values[b], best_indices[b])
                best_values[b] = cutlass.Float32(cutlass.select_(take, value, best_values[b]))
                best_indices[b] = cutlass.Int32(cutlass.select_(take, my_index, best_indices[b]))
        for b in cutlass.range_constexpr(batch):
            best_value, best_index = _warp_best(best_values[b], best_indices[b])
            if lane == 0:
                s_row_best.store((best_value.bitcast(cutlass.Int32), best_index),
                                 idx=cutlass.Int32(b * WARPS * 2) + warp * cutlass.Int32(2), alignment=8)  # fmt: skip
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
        # One first pass loads every request's [k][b] entries together (B load waves in flight at once); a request
        # still incomplete after it falls back to the per-request poll loop. The argmax order is total, so the result
        # is the same whichever pass completes a request (the catalog batch-1 variant's poll).
        firsts = []
        for b in cutlass.range_constexpr(batch):
            region = cur_base + (k * cutlass.Int32(batch) + cutlass.Int32(b)) * cutlass.Int32(
                entries * 2
            )
            gots = []
            for i in cutlass.range_constexpr(vec_per_thread):
                v_idx = tx + cutlass.Int32(i * THREADS)
                live = v_idx < cutlass.Int32(vectors)
                gots.append((live, buf_uc.load(
                    idx=region + cutlass.Int32(cutlass.select_(live, v_idx, cutlass.Int32(0))) * cutlass.Int32(4),
                    vector_size=4, alignment=16, is_volatile=True)))  # fmt: skip
            firsts.append((region, gots))
        for b in cutlass.range_constexpr(batch):
            region, gots = firsts[b]
            best_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
            best_index = cutlass.Int32(INT_MAX)
            pending = cutlass.Boolean(False)
            for live, got in gots:
                for e in cutlass.range_constexpr(2):
                    vw = cutlass.Int32(got[2 * e])
                    iw = cutlass.Int32(got[2 * e + 1])
                    pending = pending | (live & ((vw == empty) | (iw == empty)))
                    take = live & _better(vw.bitcast(cutlass.Float32), iw, best_value, best_index)
                    best_value = cutlass.Float32(
                        cutlass.select_(take, vw.bitcast(cutlass.Float32), best_value)
                    )
                    best_index = cutlass.Int32(cutlass.select_(take, iw, best_index))
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
                                  idx=cutlass.Int32(b * WARPS * 2) + warp * cutlass.Int32(2), alignment=8)  # fmt: skip
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
def m31_markov16(
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
    base_stride: cutlass.Constexpr[int],
    base_cols: cutlass.Constexpr[int],
    block: cutlass.Constexpr[int],
    batch: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[int],
    push_copies: cutlass.Constexpr[int],
    buf_words: cutlass.Constexpr[int],
    base_bf16: cutlass.Constexpr[bool],
    fp32_sum: cutlass.Constexpr[bool],
    w2_regs: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    m31_markov16_kernel(
        base, first_prev, w1, w2, corrected, tokens, next_new, accepted, num_accepted, accepted_rows, kv_lens,
        rewind, buf_uc, buf_mc, flags, vocab, vocab_offset, rank, accepted_stride, rewind_first, rewind_count,
        grid, rows, shard, base_stride, base_cols, block, batch, slots, push_copies, buf_words, base_bf16, fp32_sum,
        w2_regs,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
