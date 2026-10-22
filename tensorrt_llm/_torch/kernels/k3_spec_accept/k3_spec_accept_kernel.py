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
# Speculative-decode acceptance + block-drafter input prep for one decode step -- CTM (prims/cute) kernel
# =============================================================================
#
# For B generation requests (no context requests), K drafts each, greedy strict acceptance (DFlash / DSpark):
#   t[r]            = argmax(logits[r, :])            r < B (K + 1); torch's order: NaN first, then larger, lower index
#   accepted[b, j]  = t[b (K + 1) + j]
#   n[b]            = 1 + #leading j < K with draft[b, j] == t[b (K + 1) + j]    (or the forced count)
#   prev_acc[si[b]] = dummy[b] ? prev_acc[si[b]] : max(n[b] - 1, 0)             (the KDA replay record)
#   rewind[b]       = 1 - n[b];  kv_lens[b] += 1
#   bonus[b]        = accepted[b, max(n[b] - 1, 0)]
#   c = ctx_len[slot[b]];  qpos[b, j] = min(c + n[b], max_ctx) + j  (j < block);  cpos[b, j] = c + j  (j <= K)
#   noise[b, 0, :]  = embed[bonus[b], :];  noise[b, j > 0, :] = mask_row
#   tables[b, i]    = max(off[b, i], 0) // divisor;  counts[b] = #(off[b, i] >= 0)    (the draft pool's block table)
#
# Geometry: G CTAs of 256 threads; CTA c reduces columns [1024 c, 1024 c + 1024) of every logits row (one 16-byte
# load per thread per row) to a (value, index) partial, writes its share of the constant mask rows, and (CTA 0) the
# block table. Each CTA publishes its partials (fence.acq_rel.gpu, then an acq_rel arrival); the last CTA to arrive
# reduces all partials in the same total order, runs the per-request bookkeeping on lanes 0 .. B-1 of warp 0, copies
# the bonus embedding rows, and zeroes the arrival counter for the next call. No CTA waits on another.
#
# Vocabulary-sharded logits (SLOTS > 1): the rank holds bf16 logits of columns [shard_offset, shard_offset + V) of a
# TP column-parallel head. The last CTA's per-row (value, global index) goes to slot [rank] of every rank's Lamport
# buffer through its multicast mapping (fp32 bits, -0.0 sent as +0.0, so no pushed word is the empty word
# 0x80000000); the CTA then polls the SLOTS entries of each row and reduces them in rank order (= index order) with
# the same total order, so every rank gets the argmax of the gathered logits. Call n uses buffer n % 3 (flags[0])
# and first re-arms all MAX_ROWS rows of buffer (n + 2) % 3 (an earlier call may have pushed more rows than this one),
# whose last readers (this rank's call n - 1) are done and whose next writers (the peers' call n + 2) need this rank's
# call n + 1 pushes; only the last CTA pushes, and its push depends on this rank's arrivals alone, so no rank waits on
# a push that waits on it.
# =============================================================================
"""CTM speculative-decode acceptance + block-drafter input prep (``trtllm::k3_spec_accept``)."""

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
COLS_PER_CTA = THREADS * 4  # one fp32x4 per thread per row
MAX_BATCH = 8
MAX_ROWS = 64  # B (K + 1)
INT_MAX = 2**31 - 1
NEG_INF_BITS = -8388608  # 0xFF800000
FORCE_OFF = 0  # natural acceptance
FORCE_INT = 1  # n = base_total
FORCE_FRAC = 2  # n = min(base_total + (pool[idx] < frac), K + 1)
RNG_POOL_MASK = (1 << 16) - 1
BUFFERS = 3
FLAG_WORDS = 4
MAX_SLOTS = 16
BUF_WORDS = (
    MAX_ROWS * MAX_SLOTS * 2
)  # int32 words per Lamport buffer: [row][slot] (value bits, index)
EMPTY_WORD = -(2**31)  # 0x80000000: fp32 -0.0, never a pushed word
RNG_COUNTER_STRIDE = 6007
RNG_SLOT_STRIDE = 1009


def supports_columns(vocab: int, slots: int = 1) -> bool:
    """``vocab``: the columns this rank reduces (the whole vocabulary, or the rank's shard when ``slots`` > 1, one
    exchange slot per rank; an even count, so every row's slots are whole 16-byte vectors)."""
    return vocab % COLS_PER_CTA == 0 and (slots == 1 or (slots % 2 == 0 and slots <= MAX_SLOTS))


def supports(vocab: int, batch: int, block: int, drafts: int, hidden: int, slots: int = 1) -> bool:
    """``vocab`` and ``slots``: see :func:`supports_columns`."""
    return (
        supports_columns(vocab, slots)
        and 0 < batch <= MAX_BATCH
        and batch * (drafts + 1) <= MAX_ROWS
        and 0 < block <= 16
        and drafts + 1 <= 16
        and hidden % 8 == 0
    )


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


def _is_nan(x):
    return x != x


def _torch_better(value, index, best_value, best_index):
    """torch's argmax order (GreaterOrNan): a NaN beats any number, equal values (or two NaNs) go to the lower index,
    else the larger value."""
    v_nan = _is_nan(value)
    b_nan = _is_nan(best_value)
    lower = index < best_index
    numbers = (value > best_value) | ((value == best_value) & lower)
    if_nan = cutlass.Boolean(cutlass.select_(b_nan, lower, cutlass.Boolean(True)))
    if_number = cutlass.Boolean(cutlass.select_(b_nan, cutlass.Boolean(False), numbers))
    return cutlass.Boolean(cutlass.select_(v_nan, if_nan, if_number))


def _keep(best_value, best_index, value, index):
    take = _torch_better(value, index, best_value, best_index)
    return cutlass.Float32(cutlass.select_(take, value, best_value)), cutlass.Int32(
        cutlass.select_(take, index, best_index)
    )


def _warp_best(value, index):
    for offset in [16, 8, 4, 2, 1]:
        other_value = cute.arch.shuffle_sync_bfly(value, offset=offset, mask=-1, mask_and_clamp=31)
        other_index = cute.arch.shuffle_sync_bfly(index, offset=offset, mask=-1, mask_and_clamp=31)
        value, index = _keep(value, index, other_value, other_index)
    return value, index


@cute.kernel
def k3_spec_accept_kernel(
    logits: cutlass.Array,  # fp32 [R * V], R = B (K + 1)
    draft: cutlass.Array,  # int32 [B * K]
    block_off: cutlass.Array,  # int32: the draft pool's block offsets (row b at off_base + b * off_stride)
    block_counts: cutlass.Array,  # int64 [>= B], out
    block_tables: cutlass.Array,  # int32 [>= B, max_blocks], out
    prev_acc: cutlass.Array,  # int32 [state slots], in / out
    state_idx: cutlass.Array,  # int32 [>= B]
    dummy: cutlass.Array,  # uint8 [>= B]
    kv_lens: cutlass.Array,  # int32 [>= B], in / out (+= 1)
    batch_to_slot: cutlass.Array,  # int64 [>= B]
    ctx_len: cutlass.Array,  # int64 [slots]
    embed: cutlass.Array,  # int32 words of the bf16 embedding [V_embed, H]
    mask_row: cutlass.Array,  # int32 words of the bf16 mask embedding [H]
    rng_pool: cutlass.Array,  # fp32 [65536] (FORCE_FRAC)
    rng_counter: cutlass.Array,  # int64 [1], in / out (FORCE_FRAC)
    accepted: cutlass.Array,  # int32 [B * (K + 1)], out
    num_acc: cutlass.Array,  # int32 [B], out
    rewind: cutlass.Array,  # int32 [B], out
    bonus: cutlass.Array,  # int64 [B], out
    qpos: cutlass.Array,  # int64 [B * block], out
    cpos: cutlass.Array,  # int64 [B * (K + 1)], out
    noise: cutlass.Array,  # int32 words of bf16 [B, block, H], out
    partials: cutlass.Array,  # int32 [R * G * 2] scratch
    counter: cutlass.Array,  # int32 [1]: CTAs arrived (0 between calls)
    buf_uc: cutlass.Array,  # sharded: int32 [BUFFERS * BUF_WORDS], this rank's Lamport words
    buf_mc: cutlass.Array,  # sharded: the same words through the multicast mapping (every rank's)
    flags: cutlass.Array,  # sharded: int32 [FLAG_WORDS]; [0] the buffer of this call
    off_base: cutlass.Int32,
    off_stride: cutlass.Int32,
    divisor: cutlass.Int32,
    max_ctx: cutlass.Int64,
    force_total: cutlass.Int32,  # min(int(f) + 1, K + 1)
    force_frac: cutlass.Float32,
    force_mode: cutlass.Int32,  # a runtime value: warmup (off) and the captured step (forced) share one build
    rank: cutlass.Int32,  # sharded: this rank (its slots are rank * push_copies + c)
    shard_offset: cutlass.Int32,  # sharded: the global index of this rank's first column
    grid: cutlass.Constexpr[int],
    vocab: cutlass.Constexpr[int],
    batch: cutlass.Constexpr[int],
    drafts: cutlass.Constexpr[int],  # K
    block: cutlass.Constexpr[int],  # query block width
    hidden: cutlass.Constexpr[int],
    max_blocks: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[
        int
    ],  # 1: the whole vocabulary is here (fp32); > 1: bf16 shards, one slot per rank
    push_copies: cutlass.Constexpr[
        int
    ],  # slots each rank fills (tests emulate a larger group with > 1)
):
    rows = batch * (drafts + 1)
    row_words = hidden // 2  # int32 words per embedding row
    row_vecs = row_words // 4  # 16-byte vectors per embedding row
    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tx % cutlass.Int32(32)

    s_best = cutlass.Array(
        cutlass.Int32, rows * WARPS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_tok = cutlass.Array(cutlass.Int32, MAX_ROWS, space=cutlass.AddressSpace.smem, alignment=16)
    s_bonus = cutlass.Array(cutlass.Int32, MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16)
    # [b]: the first query position and the context length of each request (for the per-element stores).
    s_now = cutlass.Array(cutlass.Int64, MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16)
    s_ctx = cutlass.Array(cutlass.Int64, MAX_BATCH, space=cutlass.AddressSpace.smem, alignment=16)
    s_last = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    # The last CTA's inputs of the per-request bookkeeping, read while the rows reduce: [b] kv_len, state index,
    # prev_acc[state], dummy; the drafts [b][j].
    s_in = cutlass.Array(
        cutlass.Int32, MAX_BATCH * 4, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_draft = cutlass.Array(cutlass.Int32, MAX_ROWS, space=cutlass.AddressSpace.smem, alignment=16)
    # [row]: the (value bits, index) of the row over this rank's columns (the exchange's payload).
    s_row = cutlass.Array(
        cutlass.Int32, MAX_ROWS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )

    # Nothing this kernel reads is a weight: all of it comes after the grid dependency.
    if warp == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    prims.griddepcontrol(prims.GridDepAction.WAIT)

    # Per-row partial argmax over this CTA's 1024 columns.
    col0 = cta * cutlass.Int32(COLS_PER_CTA) + tx * cutlass.Int32(4)
    for r in cutlass.range_constexpr(rows):
        vals = []
        if cutlass.const_expr(slots > 1):
            # bf16 pairs as int32 words; bf16 -> fp32 is exact (the high half of the word).
            lw = logits.load(
                idx=(cutlass.Int32(r * vocab) + col0) // cutlass.Int32(2),
                vector_size=2,
                alignment=8,
            )
            for q in cutlass.range_constexpr(2):
                word = cutlass.Int32(lw[q])
                vals.append(cutlass.Int32(word << cutlass.Int32(16)).bitcast(cutlass.Float32))
                vals.append(cutlass.Int32(word & cutlass.Int32(-65536)).bitcast(cutlass.Float32))
        else:
            v = logits.load(idx=cutlass.Int32(r * vocab) + col0, vector_size=4, alignment=16)
            for q in cutlass.range_constexpr(4):
                vals.append(cutlass.Float32(v[q]))
        best_value = vals[0]
        best_index = col0
        for q in cutlass.range_constexpr(1, 4):
            best_value, best_index = _keep(best_value, best_index, vals[q], col0 + cutlass.Int32(q))
        best_value, best_index = _warp_best(best_value, best_index)
        if lane == 0:
            s_best.store((best_value.bitcast(cutlass.Int32), best_index), idx=cutlass.Int32(r * WARPS * 2) + warp * 2,
                         alignment=8)  # fmt: skip

    # The constant mask rows of the block (j >= 1), this CTA's share of their 16-byte vectors.
    n_mask_vecs = batch * (block - 1) * row_vecs
    v_id = cta * cutlass.Int32(THREADS) + tx
    while v_id < cutlass.Int32(n_mask_vecs):
        b_j = v_id // cutlass.Int32(row_vecs)  # (b, j - 1) row
        within = v_id - b_j * cutlass.Int32(row_vecs)
        b = b_j // cutlass.Int32(block - 1)
        j = b_j - b * cutlass.Int32(block - 1) + cutlass.Int32(1)
        m = mask_row.load(idx=within * cutlass.Int32(4), vector_size=4, alignment=16)
        noise.store((cutlass.Int32(m[0]), cutlass.Int32(m[1]), cutlass.Int32(m[2]), cutlass.Int32(m[3])),
                    idx=((b * cutlass.Int32(block) + j) * cutlass.Int32(row_vecs) + within) * cutlass.Int32(4),
                    alignment=16)  # fmt: skip
        v_id += cutlass.Int32(grid * THREADS)

    # The draft pool's block table (CTA 0): tables[b, i] = max(off, 0) // divisor, counts[b] = #(off >= 0).
    if cta == 0:
        for b in cutlass.range_constexpr(batch):
            present = cutlass.Int32(0)
            i = tx
            while i < cutlass.Int32(max_blocks):
                e = block_off.load(idx=off_base + cutlass.Int32(b) * off_stride + i)
                block_tables.store(cutlass.Int32(cutlass.select_(e > cutlass.Int32(0), e, cutlass.Int32(0))) // divisor,
                                   idx=cutlass.Int32(b * max_blocks) + i)  # fmt: skip
                present += cutlass.Int32(
                    cutlass.select_(e >= cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(0))
                )
                i += cutlass.Int32(THREADS)
            for offset in [16, 8, 4, 2, 1]:
                present += cute.arch.shuffle_sync_bfly(
                    present, offset=offset, mask=-1, mask_and_clamp=31
                )
            if lane == 0:
                s_tok.store(present, idx=cutlass.Int32(b * WARPS) + warp)
        prims.barrier_cta_sync(0)
        if tx < cutlass.Int32(batch):
            total = cutlass.Int32(0)
            for w in cutlass.range_constexpr(WARPS):
                total += s_tok.load(idx=tx * cutlass.Int32(WARPS) + cutlass.Int32(w))
            block_counts.store(cutlass.Int64(total), idx=tx)

    prims.barrier_cta_sync(0)
    # This CTA's partial per row, then its arrival (the partials first: fence.acq_rel.gpu + the acq_rel atom).
    if tx < cutlass.Int32(rows):
        value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
        index = cutlass.Int32(INT_MAX)
        for w in cutlass.range_constexpr(WARPS):
            pair = s_best.load(
                idx=tx * cutlass.Int32(WARPS * 2) + cutlass.Int32(w * 2), vector_size=2, alignment=8
            )
            value, index = _keep(
                value,
                index,
                cutlass.Int32(pair[0]).bitcast(cutlass.Float32),
                cutlass.Int32(pair[1]),
            )
        partials.store((value.bitcast(cutlass.Int32), index), idx=(tx * cutlass.Int32(grid) + cta) * cutlass.Int32(2),
                       alignment=8)  # fmt: skip
    cute.arch.fence_acq_rel_gpu()
    prims.barrier_cta_sync(0)
    if tx == 0:
        arrived = _atomic_add_acq_rel(counter.data_ptr(0).toint(), cutlass.Int32(1))
        s_last.store(
            cutlass.Int32(
                cutlass.select_(
                    arrived == cutlass.Int32(grid - 1), cutlass.Int32(1), cutlass.Int32(0)
                )
            ),
            idx=0,
        )
    prims.barrier_cta_sync(0)
    if s_last.load(idx=0) != cutlass.Int32(0):
        cute.arch.fence_acq_rel_gpu()
        # The bookkeeping's inputs (none depends on the argmax), in flight while the rows reduce: one element per
        # thread for the drafts; lane b of warp 0 for request b's state.
        if tx < cutlass.Int32(batch * drafts):
            s_draft.store(cutlass.Int32(draft.load(idx=tx)), idx=tx)
        if tx < cutlass.Int32(batch):
            si = cutlass.Int32(state_idx.load(idx=tx))
            s_in.store(cutlass.Int32(kv_lens.load(idx=tx)), idx=tx * cutlass.Int32(4))
            s_in.store(si, idx=tx * cutlass.Int32(4) + cutlass.Int32(1))
            s_in.store(
                cutlass.Int32(prev_acc.load(idx=si)), idx=tx * cutlass.Int32(4) + cutlass.Int32(2)
            )
            s_in.store(
                cutlass.Int32(dummy.load(idx=tx)), idx=tx * cutlass.Int32(4) + cutlass.Int32(3)
            )
            s_ctx.store(
                cutlass.Int64(ctx_len.load(idx=cutlass.Int32(batch_to_slot.load(idx=tx)))), idx=tx
            )
        # The target token of each row: warp w reduces rows w, w + 8, ... over the G partials (every partial load of
        # the lane in flight at once).
        r_w = warp
        while r_w < cutlass.Int32(rows):
            pairs = []
            for i in cutlass.range_constexpr((grid + 31) // 32):
                p = lane + cutlass.Int32(i * 32)
                p = cutlass.Int32(cutlass.select_(p < cutlass.Int32(grid), p, cutlass.Int32(0)))
                pairs.append(partials.load(idx=(r_w * cutlass.Int32(grid) + p) * cutlass.Int32(2), vector_size=2,
                                           alignment=8, is_volatile=True))  # fmt: skip
            value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
            index = cutlass.Int32(INT_MAX)
            for i in cutlass.range_constexpr((grid + 31) // 32):
                live = lane + cutlass.Int32(i * 32) < cutlass.Int32(grid)
                v_i, i_i = _keep(
                    value,
                    index,
                    cutlass.Int32(pairs[i][0]).bitcast(cutlass.Float32),
                    cutlass.Int32(pairs[i][1]),
                )
                value = cutlass.Float32(cutlass.select_(live, v_i, value))
                index = cutlass.Int32(cutlass.select_(live, i_i, index))
            value, index = _warp_best(value, index)
            if lane == 0:
                s_tok.store(index, idx=r_w)
                if cutlass.const_expr(slots > 1):
                    s_row.store((value.bitcast(cutlass.Int32), index + shard_offset), idx=r_w * cutlass.Int32(2),
                                alignment=8)  # fmt: skip
            r_w += cutlass.Int32(WARPS)
        prims.barrier_cta_sync(0)
        if cutlass.const_expr(slots > 1):
            # The vocabulary shards' exchange (see the header): re-arm the buffer before last, push, poll, reduce.
            cur = flags.load(idx=0, is_volatile=True)
            cur_base = cur * cutlass.Int32(BUF_WORDS)
            dirty_base = ((cur + cutlass.Int32(2)) % cutlass.Int32(BUFFERS)) * cutlass.Int32(
                BUF_WORDS
            )
            empty = cutlass.Int32(EMPTY_WORD)
            w_clear = tx * cutlass.Int32(4)
            while w_clear < cutlass.Int32(MAX_ROWS * slots * 2):
                buf_uc.store((empty, empty, empty, empty), idx=dirty_base + w_clear, alignment=16)
                w_clear += cutlass.Int32(THREADS * 4)
            if tx < cutlass.Int32(rows):
                mine = s_row.load(idx=tx * cutlass.Int32(2), vector_size=2, alignment=8)
                mine_bits = cutlass.Int32(mine[0])
                mine_bits = cutlass.Int32(
                    cutlass.select_(mine_bits == empty, cutlass.Int32(0), mine_bits)
                )
                for cp in cutlass.range_constexpr(push_copies):
                    slot = rank * cutlass.Int32(push_copies) + cutlass.Int32(cp)
                    push_at = cur_base + (tx * cutlass.Int32(slots) + slot) * cutlass.Int32(2)
                    buf_mc.store((mine_bits, cutlass.Int32(mine[1])), idx=push_at, alignment=8)
                # Drain the posted multicast store now (the polls below issue no release that would).
                prims.fence_acq_rel(prims.MemScope.CLUSTER)
                region = cur_base + tx * cutlass.Int32(slots * 2)
                g_index = cutlass.Int32(INT_MAX)
                pending = cutlass.Boolean(True)
                while pending:
                    p_value = cutlass.Int32(NEG_INF_BITS).bitcast(cutlass.Float32)
                    p_index = cutlass.Int32(INT_MAX)
                    p_pending = cutlass.Boolean(False)
                    got = []
                    for v4 in cutlass.range_constexpr((slots * 2 + 3) // 4):
                        got.append(buf_uc.load(idx=region + cutlass.Int32(v4 * 4), vector_size=4, alignment=16,
                                               is_volatile=True))  # fmt: skip
                    for sl in cutlass.range_constexpr(slots):
                        vw = cutlass.Int32(got[(2 * sl) // 4][(2 * sl) % 4])
                        iw = cutlass.Int32(got[(2 * sl + 1) // 4][(2 * sl + 1) % 4])
                        p_pending = p_pending | (vw == empty) | (iw == empty)
                        p_value, p_index = _keep(p_value, p_index, vw.bitcast(cutlass.Float32), iw)
                    g_index = p_index
                    pending = p_pending
                s_tok.store(g_index, idx=tx)
            prims.barrier_cta_sync(0)
            if tx == 0:
                flags.store((cur + cutlass.Int32(1)) % cutlass.Int32(BUFFERS), idx=0)
        # Per-request bookkeeping on lane b of warp 0.
        if tx < cutlass.Int32(batch):
            b = tx
            base_row = b * cutlass.Int32(drafts + 1)
            n = cutlass.Int32(1)
            run = cutlass.Boolean(True)
            # Volatile: a run of adjacent loads at a dynamic offset is otherwise merged into 16-byte vectors, which
            # the rows of b >= 1 misalign.
            for j in cutlass.range_constexpr(drafts + 1):
                t = s_tok.load(idx=base_row + cutlass.Int32(j), is_volatile=True)
                if cutlass.const_expr(j < drafts):
                    run = run & (
                        s_draft.load(
                            idx=b * cutlass.Int32(drafts) + cutlass.Int32(j), is_volatile=True
                        )
                        == t
                    )
                    n += cutlass.Int32(cutlass.select_(run, cutlass.Int32(1), cutlass.Int32(0)))
            if force_mode == cutlass.Int32(FORCE_INT):
                n = force_total
            if force_mode == cutlass.Int32(FORCE_FRAC):
                step = rng_counter.load(idx=0) + cutlass.Int64(1)
                pool_idx = (
                    step * cutlass.Int64(RNG_COUNTER_STRIDE)
                    + cutlass.Int64(b) * cutlass.Int64(RNG_SLOT_STRIDE)
                ) & cutlass.Int64(RNG_POOL_MASK)
                extra = cutlass.Int32(cutlass.select_(rng_pool.load(idx=cutlass.Int32(pool_idx)) < force_frac,
                                                      cutlass.Int32(1), cutlass.Int32(0)))  # fmt: skip
                n = force_total + extra
                n = cutlass.Int32(
                    cutlass.select_(n > cutlass.Int32(drafts + 1), cutlass.Int32(drafts + 1), n)
                )
            num_acc.store(n, idx=b)
            rewind.store(cutlass.Int32(1) - n, idx=b)
            kv_lens.store(
                cutlass.Int32(s_in.load(idx=b * cutlass.Int32(4))) + cutlass.Int32(1), idx=b
            )
            # The KDA replay record of the accepted drafts.
            si = cutlass.Int32(s_in.load(idx=b * cutlass.Int32(4) + cutlass.Int32(1)))
            keep_prev = cutlass.Int32(
                s_in.load(idx=b * cutlass.Int32(4) + cutlass.Int32(3))
            ) != cutlass.Int32(0)
            drafted = n - cutlass.Int32(1)
            drafted = cutlass.Int32(
                cutlass.select_(drafted < cutlass.Int32(0), cutlass.Int32(0), drafted)
            )
            prev_acc.store(
                cutlass.Int32(
                    cutlass.select_(
                        keep_prev,
                        cutlass.Int32(s_in.load(idx=b * cutlass.Int32(4) + cutlass.Int32(2))),
                        drafted,
                    )
                ),
                idx=si,
            )
            # The drafter's inputs.
            last = n - cutlass.Int32(1)
            last = cutlass.Int32(cutlass.select_(last < cutlass.Int32(0), cutlass.Int32(0), last))
            tok = s_tok.load(idx=base_row + last)
            bonus.store(cutlass.Int64(tok), idx=b)
            s_bonus.store(tok, idx=b)
            c = cutlass.Int64(s_ctx.load(idx=b))
            now = c + cutlass.Int64(n)
            now = cutlass.Int64(cutlass.select_(now > max_ctx, max_ctx, now))
            s_now.store(now, idx=b)
        prims.barrier_cta_sync(0)
        # After the barrier: every lane above has read the counter (the lanes of warp 0 need not run in step).
        if force_mode == cutlass.Int32(FORCE_FRAC):
            if tx == 0:
                rng_counter.store(rng_counter.load(idx=0) + cutlass.Int64(1), idx=0)
        # One element per thread of accepted, qpos and cpos (a thread storing a run of adjacent elements at a dynamic
        # offset gets them merged into wider stores that the odd rows misalign).
        if tx < cutlass.Int32(rows):
            accepted.store(s_tok.load(idx=tx), idx=tx)
            b_c = tx // cutlass.Int32(drafts + 1)
            cpos.store(
                s_ctx.load(idx=b_c) + cutlass.Int64(tx - b_c * cutlass.Int32(drafts + 1)), idx=tx
            )
        if tx < cutlass.Int32(batch * block):
            b_q = tx // cutlass.Int32(block)
            qpos.store(s_now.load(idx=b_q) + cutlass.Int64(tx - b_q * cutlass.Int32(block)), idx=tx)
        # noise[b, 0, :] = embed[bonus[b], :]: every load of the thread in flight, then the stores.
        embed_vecs = []
        for i in cutlass.range_constexpr((batch * row_vecs + THREADS - 1) // THREADS):
            v_id = tx + cutlass.Int32(i * THREADS)
            v_id = cutlass.Int32(
                cutlass.select_(v_id < cutlass.Int32(batch * row_vecs), v_id, cutlass.Int32(0))
            )
            b = v_id // cutlass.Int32(row_vecs)
            within = v_id - b * cutlass.Int32(row_vecs)
            tok = cutlass.Int32(s_bonus.load(idx=b))
            embed_vecs.append(embed.load(idx=(tok * cutlass.Int32(row_vecs) + within) * cutlass.Int32(4), vector_size=4,
                                         alignment=16))  # fmt: skip
        for i in cutlass.range_constexpr((batch * row_vecs + THREADS - 1) // THREADS):
            v_id = tx + cutlass.Int32(i * THREADS)
            if v_id < cutlass.Int32(batch * row_vecs):
                b = v_id // cutlass.Int32(row_vecs)
                within = v_id - b * cutlass.Int32(row_vecs)
                e = embed_vecs[i]
                noise.store((cutlass.Int32(e[0]), cutlass.Int32(e[1]), cutlass.Int32(e[2]), cutlass.Int32(e[3])),
                            idx=((b * cutlass.Int32(block)) * cutlass.Int32(row_vecs) + within) * cutlass.Int32(4),
                            alignment=16)  # fmt: skip
        if tx == 0:
            counter.store(cutlass.Int32(0), idx=0)


@cute.jit
def k3_spec_accept(
    logits: cute.Tensor,
    draft: cute.Tensor,
    block_off: cute.Tensor,
    block_counts: cute.Tensor,
    block_tables: cute.Tensor,
    prev_acc: cute.Tensor,
    state_idx: cute.Tensor,
    dummy: cute.Tensor,
    kv_lens: cute.Tensor,
    batch_to_slot: cute.Tensor,
    ctx_len: cute.Tensor,
    embed: cute.Tensor,
    mask_row: cute.Tensor,
    rng_pool: cute.Tensor,
    rng_counter: cute.Tensor,
    accepted: cute.Tensor,
    num_acc: cute.Tensor,
    rewind: cute.Tensor,
    bonus: cute.Tensor,
    qpos: cute.Tensor,
    cpos: cute.Tensor,
    noise: cute.Tensor,
    partials: cute.Tensor,
    counter: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    off_base: cutlass.Int32,
    off_stride: cutlass.Int32,
    divisor: cutlass.Int32,
    max_ctx: cutlass.Int64,
    force_total: cutlass.Int32,
    force_frac: cutlass.Float32,
    force_mode: cutlass.Int32,
    rank: cutlass.Int32,
    shard_offset: cutlass.Int32,
    grid: cutlass.Constexpr[int],
    vocab: cutlass.Constexpr[int],
    batch: cutlass.Constexpr[int],
    drafts: cutlass.Constexpr[int],
    block: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    max_blocks: cutlass.Constexpr[int],
    slots: cutlass.Constexpr[int],
    push_copies: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    k3_spec_accept_kernel(
        logits, draft, block_off, block_counts, block_tables, prev_acc, state_idx, dummy, kv_lens, batch_to_slot,
        ctx_len, embed, mask_row, rng_pool, rng_counter, accepted, num_acc, rewind, bonus, qpos, cpos, noise, partials,
        counter, buf_uc, buf_mc, flags, off_base, off_stride, divisor, max_ctx, force_total, force_frac, force_mode,
        rank, shard_offset, grid, vocab, batch, drafts, block, hidden, max_blocks, slots, push_copies,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
