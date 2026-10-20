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
# Block drafter context K/V of a decode step -- CTM (prims/cute) kernel
# =============================================================================
#
# For the N <= 64 context tokens of a step (B <= 8 requests of K1 <= 8; token t = request b = t / K1, candidate
# j = t % K1) and every drafter layer l and K/V head h of this rank:
#   kv[t]  = x[t] @ W^T         W = the layers' stacked K / V projection rows [L0 K | L0 V | L1 K | ...] (bf16)
#   K head: k = bf16(kv); k = bf16(k * rsqrt(mean(k^2) + eps)); k = bf16(k * k_norm[l]); k = bf16(NeoX RoPE(k, pos[t]))
#   V head: v = bf16(kv)
#   a masked token (j >= num_acc[b]) is multiplied by 0.0 (signed zeros, as the Python path), and every token's rows go
#   into the paged pool (HND per layer: page, K/V, head, slot, 64; every layer a view of one allocation) at column
#   col = clamp(min(ctx_len[slot_b] + j, counts[row_b] page - 1), 0), page = table[row_b, col / page];
#   then ctx_len[slot_b] = min(ctx_len + num_acc[b], max_ctx) and num_ctx[b] = min(new ctx_len,
#   max(counts[row_b] page - block, 0)).
# (DFlashWorker: precompute_context_kv, the write mask, _store_context_kv_paged, the ctx_len update and num_ctx.)
#
# Rounding points are the Python path's; the GEMM adds in another order than cuBLAS (fp32 tolerance).
#
# Geometry (k3_ctm_gemv_long): one 128-row tile of W per cluster of SPLIT CTAs; rank r streams the k-tiles r,
# r + SPLIT, ... through a RING-stage ring filled before griddepcontrol.wait (the rest prefetched into L2 then), x
# resident after the wait, tcgen05 M128 x N into TMEM (N = 8, 16, 32 or 64 token columns: the smallest that holds the
# step's tokens; rows past them arrive as zeros). The tokens form chunks of 8; pair p = half h x C + chunk c (half h of
# the tile: 64 rows, one head of one layer, K or V; C = N / 8 chunks) is owned by rank p % SPLIT (C <= SPLIT, so a warp
# pair owns at most one chunk; N = 8: half o by rank o). The other ranks push their fp32 partials of a pair into slot
# [p / SPLIT][source] of its owner's mailbox (st.async, completing the owner's barrier p / SPLIT by bytes); the
# owner's two epilogue warps (TMEM lanes 64 h .. 64 h + 63; thread = head row d, the chunk's 8 token values) add the
# SPLIT partials in rank order and finish the head: the per-token sums of squares by a reduce-scatter warp tree and
# the two warps' halves through shared memory, the RoPE partner row d ^ 32 (the other warp, same lane) through shared
# memory, one bf16 store per (token, row): 128 contiguous bytes per token and head.
#
# ctx_len / num_ctx: every CTA reads what it needs of ctx_len (its epilogue threads, after the grid dependency), joins
# its epilogue threads and arrives on a counter (atom.acq_rel.gpu); the last arrival writes ctx_len and num_ctx and
# re-arms the counter to 0 (no CTA waits on it).
#
# Warps: 0 weight TMA (+ early dependent trigger), 1 x TMA after the grid dependency, 2 TMEM allocation + MMA, 3 idle,
# 4-7 epilogue (the token metadata after the grid dependency, while the MMAs run).
# =============================================================================
"""CTM block-drafter context K/V: the stacked K/V projection, k_norm, NeoX RoPE, the write mask and the paged store
of every drafter layer for a decode step's context tokens, and the context-length update, in one launch."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows per tile = the MMA's M
HEAD = 64  # rows of one K or V head (the head dimension)
ROPE_HALF = HEAD // 2
CHUNK = 8  # tokens per owned chunk: the token values an epilogue thread finishes
MAX_TOKENS = 64
MAX_BATCH = 8
CTA_K = 128  # one k-tile: two 64-element halves of the 128-byte swizzle
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
THREADS = 256
EPI_THREADS = 128
OWNER_THREADS = 64
ELEM_BYTES = 2
EVICT_FIRST = 0x12F0000000000000  # createpolicy.fractional.L2::evict_first, fraction 1.0 (sm_100)
SMEM_BUDGET = 220 * 1024
BAR_JOIN = 1  # the 128 epilogue threads (TMEM reads done, ctx_len reads done)
BAR_OWNER = 2  # the owning half's two epilogue warps (+ half when a rank owns a chunk of each half)

# Shared-memory descriptor strides for the 128-byte swizzle, in 16-byte units.
LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES
A_HALF_ELEMS = CTA_M * TMA_K_BOX
STEP = (MMA_K * ELEM_BYTES) >> 4
A_BOX = A_HALF_ELEMS >> 3
STAGE_A = (CTA_M * CTA_K * ELEM_BYTES) >> 4

io_dtype = cutlass.BFloat16


def mma_n(n_tokens: int) -> int:
    """The MMA's token columns: the smallest of 8, 16, 32, 64 that holds the step's tokens."""
    for n in (8, 16, 32, 64):
        if n_tokens <= n:
            return n
    return 0


def owner_slots(n_tokens: int, split: int) -> int:
    """Pairs (half, chunk) a rank may own: 1 when the 2 C pairs land on distinct ranks, else 2 (one of each half)."""
    return 1 if 2 * (mma_n(n_tokens) // CHUNK) <= split else 2


def smem_bytes(split: int, ring: int, my_tiles: int, n_tokens: int = CHUNK) -> int:
    """Shared memory of a CTA: the weight ring, the resident x k-tiles, the mailbox, the exchanges and barriers."""
    slots = owner_slots(n_tokens, split)
    return (
        ring * CTA_M * CTA_K * ELEM_BYTES
        + my_tiles * mma_n(n_tokens) * CTA_K * ELEM_BYTES
        + slots * split * HEAD * CHUNK * 4
        + slots * HEAD * CHUNK * 4
        + slots * 2 * CHUNK * 4
        + 1024
    )


def pick_ring(k_in: int, split: int, n_tokens: int = CHUNK) -> int:
    """The deepest weight ring that fits next to the resident x, the mailbox and the exchanges (0: none)."""
    my_tiles = (k_in // CTA_K) // split
    for ring in range(min(my_tiles, 8), 0, -1):
        if smem_bytes(split, ring, my_tiles, n_tokens) <= SMEM_BUDGET:
            return ring
    return 0


def supports(n_rows: int, k_in: int, split: int, nkv: int, k1: int, n_tokens: int) -> bool:
    """Shapes the kernel runs: whole 128-row tiles of whole heads, whole k-tiles split evenly over a 2-8 CTA cluster,
    a ring that fits, N <= 64 tokens made of B <= 8 whole requests of K1 <= 8, at most one chunk of 8 tokens per
    rank and half (N / 8 <= SPLIT)."""
    k_tiles = k_in // CTA_K
    return (
        nkv > 0
        and n_rows % (2 * nkv * HEAD) == 0
        and n_rows % CTA_M == 0
        and k_in % CTA_K == 0
        and split in (2, 4, 8)
        and k_tiles % split == 0
        and 0 < k1 <= CHUNK
        and 0 < n_tokens <= MAX_TOKENS
        and n_tokens % k1 == 0
        and n_tokens // k1 <= MAX_BATCH
        and mma_n(n_tokens) // CHUNK <= split
        and pick_ring(k_in, split, n_tokens) > 0
    )


def _plus(base, x):
    """``base + x``; ``x`` itself when ``base`` is a Python 0 (the one-chunk build: no op in the IR)."""
    if type(base) is int and base == 0:
        return x
    return base + x


def _slot(barriers, i):
    """Barrier ``i`` of ``barriers`` (the array itself for a Python 0)."""
    if type(i) is int and i == 0:
        return barriers
    return barriers.subview(i)


def _push_chunk(mailbox, mail_full, acc, c, half, chunks, split, slots_owned, rank, slot_row):
    """st.async of this lane's partials of chunk ``c`` of its half (8 token values) into slot [pair / split][rank] of
    the pair's owner, completing the owner's barrier pair / split by 32 bytes."""
    if chunks == 1:
        dest = half
        slot_grp = 0
    else:
        pair = half * cutlass.Int32(chunks) + cutlass.Int32(c)
        dest = pair % cutlass.Int32(split)
        slot_grp = 0 if slots_owned == 1 else pair // cutlass.Int32(split)
    base = _plus(slot_grp * split, rank) * cutlass.Int32(HEAD * CHUNK) + slot_row
    peer_slot = _mapa_u32(mailbox.subview(base).data_ptr(), dest)
    mbar_peer = _mapa_u32(_slot(mail_full, slot_grp).data_ptr(), dest)
    _st_async_v4(
        peer_slot,
        acc[CHUNK * c],
        acc[CHUNK * c + 1],
        acc[CHUNK * c + 2],
        acc[CHUNK * c + 3],
        mbar_peer,
    )
    _st_async_v4(
        peer_slot + cutlass.Int32(16), acc[CHUNK * c + 4], acc[CHUNK * c + 5], acc[CHUNK * c + 6],
        acc[CHUNK * c + 7], mbar_peer,
    )  # fmt: skip


def _own(acc, t, mine):
    """The rank's own partial of token ``t``: ``mine`` when chosen among chunks, else ``acc[t]`` (read in place)."""
    return cutlass.Float32(acc[t]) if mine is None else mine


def _bf16_rn(v):
    """fp32 -> bf16 precision (round to nearest even) in the integer domain, kept as fp32 (no fptrunc/fpext pair the
    compiler could fold into the next multiply)."""
    u = v.bitcast(cutlass.Int32)
    u = u + (((u >> 16) & 1) + 0x7FFF)
    u = (u >> 16) << 16
    return u.bitcast(cutlass.Float32)


@dsl_user_op
def _atomic_add_acq_rel(addr, value, *, loc=None, ip=None):
    """atom.add.acq_rel.gpu.s32 on a global address; returns the old value."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
            "atom.add.acq_rel.gpu.s32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA ``peer``."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. For barriers that other CTAs' st.async complete (their
    complete_tx releases at cluster scope; a CTA-scope acquire does not synchronize with it)."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@dsl_user_op
def _st_async_v4(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 (16 bytes) to a shared::cluster address, completing ``mbar`` (a shared::cluster address)
    by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];", "r,f,f,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


def _warp_token_sums(vals, lane):
    """Per-token sums over the warp's 32 lanes of 8 per-lane values (a reduce-scatter over lane distances 16, 8, 4,
    then the full tree at 2 and 1: 9 shuffles). Lanes 4 i .. 4 i + 3 end with token i's sum."""
    parts = vals
    for offset in [16, 8, 4]:
        upper = (lane & cutlass.Int32(offset)) != cutlass.Int32(0)
        half = len(parts) // 2
        kept = []
        for i in range(half):
            send = cutlass.Float32(cutlass.select_(upper, parts[i], parts[half + i]))
            keep = cutlass.Float32(cutlass.select_(upper, parts[half + i], parts[i]))
            kept.append(
                keep + cute.arch.shuffle_sync_bfly(send, offset=offset, mask=-1, mask_and_clamp=31)
            )
        parts = kept
    total = parts[0]
    for offset in [2, 1]:
        total = total + cute.arch.shuffle_sync_bfly(
            total, offset=offset, mask=-1, mask_and_clamp=31
        )
    return total


@cute.kernel
def k3_ctx_kv_kernel(
    tma_desc_w: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W [n_rows, K] bf16, 5-D, one call per k-tile
    tma_desc_x: cutlass.GridConstant[cuda.TensorMap],  # x [N, K] bf16, box 64 x MMA N
    k_norm: cutlass.Array,  # bf16 [L * 64]
    cos_sin: cutlass.Array,  # fp32 [max_pos * 64]: cos of the 32 pairs, then sin
    cpos: cutlass.Array,  # int64 [N]: RoPE positions
    num_acc: cutlass.Array,  # int32 [B]
    ctx_len: cutlass.Array,  # int64 [slots], updated
    slots: cutlass.Array,  # int64 [B]
    rows: cutlass.Array,  # int64 [B]: rows of the block table
    table: cutlass.Array,  # int32 [table rows * table_stride]
    counts: cutlass.Array,  # int64 [table rows]
    pool: cutlass.Array,  # bf16 elements of the allocation holding every layer's pool
    layer_off: cutlass.Array,  # int64 [L]: element offset of each layer's pool in it
    num_ctx: cutlass.Array,  # int32 [B], out
    counter: cutlass.Array,  # int32 [1], zero between launches
    eps: cutlass.Float32,
    max_ctx: cutlass.Int64,
    page: cutlass.Int64,
    block_size: cutlass.Int64,
    table_stride: cutlass.Int64,
    page_stride: cutlass.Int64,
    kv_stride: cutlass.Int64,
    head_stride: cutlass.Int64,
    n_rows: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    nkv: cutlass.Constexpr[int],
    k1: cutlass.Constexpr[int],
    n_tokens: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
):
    k_tiles = k_in // CTA_K
    my_tiles = k_tiles // split
    grid = (n_rows // CTA_M) * split
    batch = n_tokens // k1
    n_cols = mma_n(n_tokens)
    chunks = n_cols // CHUNK
    slots_owned = owner_slots(n_tokens, split)
    tmem_cols = max(32, n_cols)
    stage_b = (n_cols * CTA_K * ELEM_BYTES) >> 4
    b_half_elems = n_cols * TMA_K_BOX
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cute.arch.block_idx_in_cluster()
    tile = bx // cutlass.Int32(split)
    m_offset = tile * cutlass.Int32(CTA_M)
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()

    smem_a = cutlass.Array(
        io_dtype, ring * CTA_M * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b = cutlass.Array(
        io_dtype, my_tiles * n_cols * CTA_K, space=cutlass.AddressSpace.smem, alignment=1024
    )
    tma_full = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    mma_done = cutlass.Array(cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8)
    act_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(
        cutlass.Int64, slots_owned, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [pair p / split][source rank][head row][token of the chunk] fp32 partials of the pairs this CTA owns (the own
    # rank's slot stays unused).
    mailbox = cutlass.Array(
        cutlass.Float32,
        slots_owned * split * HEAD * CHUNK,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    # Per owning half (one buffer unless a rank owns a chunk of each): [owner warp][token]: the warp's sums of squares;
    # [head row][token]: the RoPE partners.
    s_ss = cutlass.Array(
        cutlass.Float32, slots_owned * 2 * CHUNK, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_rope = cutlass.Array(
        cutlass.Float32, slots_owned * HEAD * CHUNK, space=cutlass.AddressSpace.smem, alignment=16
    )

    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_x)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(tma_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            # Owner ranks: the other ranks' partials of an owned pair arrive by st.async (16-byte stores that
            # complete its barrier's transaction count); expected here, before cluster formation.
            for s in cutlass.range_constexpr(slots_owned):
                prims.mbarrier_init(_slot(mail_full, s), 1)
                prims.mbarrier_arrive_expect_tx(_slot(mail_full, s), (split - 1) * HEAD * CHUNK * 4)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_ptr = cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32)

    if warp_id == 0:
        # =====================================================================
        # Weight TMA: fill the ring and prefetch the rest into L2 before the
        # grid dependency; then refill each stage when its MMAs are done.
        # =====================================================================
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring):
                k = rank + cutlass.Int32(i * split)
                prims.mbarrier_arrive_expect_tx(tma_full.subview(i), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(i * CTA_M * CTA_K),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    tma_full.subview(i),
                    l2_cache_hint=EVICT_FIRST,
                )
            for i in cutlass.range_constexpr(ring, my_tiles):
                k = rank + cutlass.Int32(i * split)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w,
                    [
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ],
                    [],  # tile mode: no im2col offsets
                )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading the pool or ctx_len.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if prims.elect_sync():
            stage = cutlass.Int32(0)
            phase = cutlass.Int32(0)
            for i in cutlass.range(ring, my_tiles, unroll=1):
                while not cute.arch.mbarrier_try_wait(mma_done.subview(stage).data_ptr(), phase):
                    pass
                k = rank + i * cutlass.Int32(split)
                prims.mbarrier_arrive_expect_tx(tma_full.subview(stage), CTA_M * CTA_K * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * cutlass.Int32(CTA_M * CTA_K)),
                    tma_ptr_w,
                    (
                        cutlass.Int32(0),
                        m_offset,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    tma_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                stage = stage + cutlass.Int32(1)
                if stage == cutlass.Int32(ring):
                    stage = cutlass.Int32(0)
                    phase = phase ^ cutlass.Int32(1)
    elif warp_id == 1:
        # =====================================================================
        # Activation TMA: the rank's k-tiles of x, resident, after the wait.
        # =====================================================================
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(act_full, my_tiles * n_cols * CTA_K * ELEM_BYTES)
            for i in cutlass.range_constexpr(my_tiles):
                k = rank + cutlass.Int32(i * split)
                for half in cutlass.range_constexpr(TMA_COPY_ITERS):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(i * n_cols * CTA_K + half * b_half_elems),
                        tma_ptr_x,
                        (
                            k * cutlass.Int32(CTA_K) + cutlass.Int32(half * TMA_K_BOX),
                            cutlass.Int32(0),
                        ),
                        act_full,
                    )
    elif warp_id == 2:
        # =====================================================================
        # MMA: the rank's k-tiles in order, one TMEM accumulator (partial sum).
        # =====================================================================
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=n_cols, m_dim=CTA_M
        )
        desc_a_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_a, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        desc_b_base = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )  # fmt: skip
        while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
            pass
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)
        for i in cutlass.range(my_tiles, unroll=1):
            while not cute.arch.mbarrier_try_wait(tma_full.subview(stage).data_ptr(), phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                desc_a = desc_a_base + (
                    stage * cutlass.Int32(STAGE_A) + cutlass.Int32(box * A_BOX + within * STEP)
                )
                desc_b = desc_b_base + (
                    i * cutlass.Int32(stage_b)
                    + cutlass.Int32(box * (b_half_elems >> 3) + within * STEP)
                )
                accumulate = cutlass.Boolean(True)
                if cutlass.const_expr(kb == 0):
                    accumulate = i > cutlass.Int32(0)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a,
                        desc_b,
                        idesc,
                        accumulate,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(mma_done.subview(stage))
            stage = stage + cutlass.Int32(1)
            if stage == cutlass.Int32(ring):
                stage = cutlass.Int32(0)
                phase = phase ^ cutlass.Int32(1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        # =====================================================================
        # Epilogue: token metadata, TMEM -> registers, push to / reduce at the
        # pair's owner, the head's epilogue and the paged store.
        # =====================================================================
        lane = tx % cutlass.Int32(32)
        w = warp_id - cutlass.Int32(4)  # TMEM lanes 32 w .. 32 w + 31: tile rows 32 w + lane
        half = w // cutlass.Int32(2)
        d = (w % cutlass.Int32(2)) * cutlass.Int32(32) + lane  # row within the head
        # The half's 64-row block of W: (layer, K or V, head).
        q = tile * cutlass.Int32(2) + half
        layer = q // cutlass.Int32(2 * nkv)
        kv = (q % cutlass.Int32(2 * nkv)) // cutlass.Int32(nkv)
        head = q % cutlass.Int32(nkv)
        row_base = (
            cutlass.Int64(layer_off.load(idx=layer)) + cutlass.Int64(kv) * kv_stride + cutlass.Int64(head) * head_stride
            + cutlass.Int64(d)
        )  # fmt: skip
        k_weight = cutlass.BFloat16(k_norm.load(idx=layer * cutlass.Int32(HEAD) + d)).to(
            cutlass.Float32
        )
        # The chunk c of this half that this rank owns (pair half C + c on rank (half C + c) % split), its mailbox
        # slot and its exchange buffer (the half's own when a rank owns a chunk of each half).
        if cutlass.const_expr(chunks == 1):
            c_own = 0  # half h is owned by rank h
        else:
            c_own = (
                rank + cutlass.Int32(2 * split) - half * cutlass.Int32(chunks)
            ) % cutlass.Int32(split)  # >= chunks: this rank owns none of the half's chunks
        if cutlass.const_expr(slots_owned == 1):
            own_slot = buf = 0  # Python zeros: no slot / buffer arithmetic (see _plus)
            owner_bar = BAR_OWNER
        else:
            own_slot = (half * cutlass.Int32(chunks) + c_own) // cutlass.Int32(split)
            buf = half
            owner_bar = cutlass.Int32(BAR_OWNER) + half

        # Token metadata (predecessor outputs: after the grid dependency), while the MMAs run: each owned token's pool
        # element offset, its mask factor and its RoPE cos / sin; per request the slot, length, accepted count and
        # allocation (for the last CTA's update).
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        dst = []
        keep = []
        cos_t = []
        sin_t = []
        req_slot = []
        req_len = []
        req_acc = []
        req_cap = []
        pair = d & cutlass.Int32(ROPE_HALF - 1)
        for b in cutlass.range_constexpr(batch):
            slot = cutlass.Int32(slots.load(idx=b))
            trow = cutlass.Int32(rows.load(idx=b))
            c = cutlass.Int64(ctx_len.load(idx=slot))
            cap = cutlass.Int64(counts.load(idx=trow)) * page
            n_acc = cutlass.Int32(num_acc.load(idx=b))
            req_slot.append(slot)
            req_len.append(c)
            req_acc.append(n_acc)
            req_cap.append(cap)
            if cutlass.const_expr(chunks == 1):
                for j in cutlass.range_constexpr(k1):
                    col = c + cutlass.Int64(j)
                    col = cutlass.Int64(
                        cutlass.select_(col > cap - cutlass.Int64(1), cap - cutlass.Int64(1), col)
                    )
                    col = cutlass.Int64(
                        cutlass.select_(col < cutlass.Int64(0), cutlass.Int64(0), col)
                    )
                    pg = cutlass.Int64(
                        table.load(idx=cutlass.Int64(trow) * table_stride + col // page)
                    )
                    dst.append(pg * page_stride + (col % page) * cutlass.Int64(HEAD) + row_base)
                    keep.append(cutlass.Float32(cutlass.select_(cutlass.Int32(j) < n_acc, cutlass.Float32(1.0),
                                                                cutlass.Float32(0.0))))  # fmt: skip
                    pos = cutlass.Int64(cpos.load(idx=b * k1 + j))
                    cos_t.append(
                        cutlass.Float32(
                            cos_sin.load(idx=pos * cutlass.Int64(HEAD) + cutlass.Int64(pair))
                        )
                    )
                    sin_t.append(
                        cutlass.Float32(
                            cos_sin.load(
                                idx=pos * cutlass.Int64(HEAD)
                                + cutlass.Int64(pair + cutlass.Int32(ROPE_HALF))
                            )
                        )
                    )
        if cutlass.const_expr(chunks > 1):
            # The owned chunk's tokens CHUNK c_own + t (a token past the step's reads token 0's, never stored).
            for t in cutlass.range_constexpr(CHUNK):
                tok = c_own * cutlass.Int32(CHUNK) + cutlass.Int32(t)
                tok = cutlass.Int32(
                    cutlass.select_(tok < cutlass.Int32(n_tokens), tok, cutlass.Int32(0))
                )
                b = tok // cutlass.Int32(k1)
                j = tok - b * cutlass.Int32(k1)
                slot = cutlass.Int32(slots.load(idx=b))
                trow = cutlass.Int32(rows.load(idx=b))
                c = cutlass.Int64(ctx_len.load(idx=slot))
                cap = cutlass.Int64(counts.load(idx=trow)) * page
                n_acc = cutlass.Int32(num_acc.load(idx=b))
                col = c + cutlass.Int64(j)
                col = cutlass.Int64(
                    cutlass.select_(col > cap - cutlass.Int64(1), cap - cutlass.Int64(1), col)
                )
                col = cutlass.Int64(cutlass.select_(col < cutlass.Int64(0), cutlass.Int64(0), col))
                pg = cutlass.Int64(table.load(idx=cutlass.Int64(trow) * table_stride + col // page))
                dst.append(pg * page_stride + (col % page) * cutlass.Int64(HEAD) + row_base)
                keep.append(cutlass.Float32(cutlass.select_(j < n_acc, cutlass.Float32(1.0),
                                                            cutlass.Float32(0.0))))  # fmt: skip
                pos = cutlass.Int64(cpos.load(idx=tok))
                cos_t.append(
                    cutlass.Float32(
                        cos_sin.load(idx=pos * cutlass.Int64(HEAD) + cutlass.Int64(pair))
                    )
                )
                sin_t.append(
                    cutlass.Float32(
                        cos_sin.load(
                            idx=pos * cutlass.Int64(HEAD)
                            + cutlass.Int64(pair + cutlass.Int32(ROPE_HALF))
                        )
                    )
                )

        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), num=n_cols
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        # Every epilogue thread has read TMEM and ctx_len.
        prims.barrier_cta_sync(BAR_JOIN, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols)
        slot_row = d * cutlass.Int32(CHUNK)
        # Push every chunk of this half that another rank owns: two 16-byte st.async per lane into slot
        # [pair / split][rank]; the owner's barrier completes when all (split - 1) x 64 lanes' land. A rank that owns
        # none of the half's chunks only pushes; an owner pushes the others, then finishes its own.
        if cutlass.const_expr(chunks == 1):
            push_only = rank != half
        else:
            push_only = c_own >= cutlass.Int32(chunks)
        if push_only:
            for ch in cutlass.range_constexpr(chunks):
                _push_chunk(
                    mailbox, mail_full, acc, ch, half, chunks, split, slots_owned, rank, slot_row
                )
        else:
            if cutlass.const_expr(chunks > 1):
                for ch in cutlass.range_constexpr(chunks):
                    pushed = half * cutlass.Int32(chunks) + cutlass.Int32(ch)  # the pair index
                    if pushed % cutlass.Int32(split) != rank:
                        _push_chunk(
                            mailbox,
                            mail_full,
                            acc,
                            ch,
                            half,
                            chunks,
                            split,
                            slots_owned,
                            rank,
                            slot_row,
                        )
            # test_wait spin: a warp suspended in try_wait on a barrier completed by remote st.async wakes late.
            while not _test_wait_cluster(_slot(mail_full, own_slot).data_ptr(), 0):
                pass
            slot_base = _plus(own_slot * (split * HEAD * CHUNK), slot_row)
            y = []
            for t in cutlass.range_constexpr(CHUNK):
                # This rank's own partial of token t of its chunk (several chunks: picked by c_own).
                mine = None
                if cutlass.const_expr(chunks > 1):
                    mine = cutlass.Float32(acc[t])
                    for ch in cutlass.range_constexpr(1, chunks):
                        mine = cutlass.Float32(
                            cutlass.select_(
                                c_own == cutlass.Int32(ch),
                                cutlass.Float32(acc[CHUNK * ch + t]),
                                mine,
                            )
                        )
                total = cutlass.Float32(0.0)
                for src in cutlass.range_constexpr(split):
                    part = cutlass.Float32(
                        mailbox.load(idx=cutlass.Int32(src * HEAD * CHUNK + t) + slot_base)
                    )
                    total = total + cutlass.Float32(
                        cutlass.select_(rank == cutlass.Int32(src), _own(acc, t, mine), part)
                    )
                y.append(_bf16_rn(total))
            # Tokens of the chunk that the step has (the rest are the zero rows past N): static for one chunk.
            stored = []
            if cutlass.const_expr(chunks > 1):
                for t in cutlass.range_constexpr(CHUNK):
                    stored.append(
                        c_own * cutlass.Int32(CHUNK) + cutlass.Int32(t) < cutlass.Int32(n_tokens)
                    )
            if kv == cutlass.Int32(0):
                # K head: RMSNorm over the 64 rows of each token, k_norm, NeoX RoPE.
                token_ss = _warp_token_sums([y[t] * y[t] for t in range(CHUNK)], lane)
                ss_base = buf * (2 * CHUNK)
                if (lane & cutlass.Int32(3)) == cutlass.Int32(0):
                    s_ss.store(
                        token_ss,
                        idx=_plus(
                            ss_base,
                            (w % cutlass.Int32(2)) * cutlass.Int32(CHUNK)
                            + (lane >> cutlass.Int32(2)),
                        ),
                    )
                prims.barrier_cta_sync(owner_bar, thread_count=OWNER_THREADS)
                kn = []
                rope_base = buf * (HEAD * CHUNK)
                for t in cutlass.range_constexpr(CHUNK):
                    ss = cutlass.Float32(s_ss.load(idx=_plus(ss_base, t))) + cutlass.Float32(
                        s_ss.load(idx=_plus(ss_base, CHUNK + t))
                    )
                    r = cute.math.rsqrt(ss * cutlass.Float32(1.0 / HEAD) + eps, fastmath=True)
                    kn.append(_bf16_rn(_bf16_rn(y[t] * r) * k_weight))
                    s_rope.store(kn[t], idx=_plus(rope_base, slot_row + cutlass.Int32(t)))
                prims.barrier_cta_sync(owner_bar, thread_count=OWNER_THREADS)
                partner_row = _plus(
                    rope_base, (d ^ cutlass.Int32(ROPE_HALF)) * cutlass.Int32(CHUNK)
                )
                sign = cutlass.Float32(cutlass.select_(d < cutlass.Int32(ROPE_HALF), cutlass.Float32(-1.0),
                                                       cutlass.Float32(1.0)))  # fmt: skip
                if cutlass.const_expr(chunks == 1):
                    for t in cutlass.range_constexpr(n_tokens):
                        partner = cutlass.Float32(s_rope.load(idx=partner_row + cutlass.Int32(t)))
                        out = _bf16_rn(kn[t] * cos_t[t] + sign * partner * sin_t[t])
                        pool.store((out * keep[t]).to(io_dtype), idx=dst[t])
                else:
                    for t in cutlass.range_constexpr(CHUNK):
                        if stored[t]:
                            partner = cutlass.Float32(
                                s_rope.load(idx=partner_row + cutlass.Int32(t))
                            )
                            out = _bf16_rn(kn[t] * cos_t[t] + sign * partner * sin_t[t])
                            pool.store((out * keep[t]).to(io_dtype), idx=dst[t])
            else:
                if cutlass.const_expr(chunks == 1):
                    for t in cutlass.range_constexpr(n_tokens):
                        pool.store((y[t] * keep[t]).to(io_dtype), idx=dst[t])
                else:
                    for t in cutlass.range_constexpr(CHUNK):
                        if stored[t]:
                            pool.store((y[t] * keep[t]).to(io_dtype), idx=dst[t])
        if tx == cutlass.Int32(4 * 32):
            # After this CTA's push or stores (off their critical path); every epilogue thread read ctx_len before the
            # join above. The last CTA of the grid updates the lengths.
            arrived = _atomic_add_acq_rel(counter.data_ptr(0).toint(), cutlass.Int32(1))
            if arrived == cutlass.Int32(grid - 1):
                # From this thread's own metadata (read before its arrival, like every other CTA's).
                for b in cutlass.range_constexpr(batch):
                    grown = req_len[b] + cutlass.Int64(req_acc[b])
                    grown = cutlass.Int64(cutlass.select_(grown > max_ctx, max_ctx, grown))
                    ctx_len.store(grown, idx=req_slot[b])
                    allocated = req_cap[b] - block_size
                    allocated = cutlass.Int64(
                        cutlass.select_(allocated < cutlass.Int64(0), cutlass.Int64(0), allocated)
                    )
                    num_ctx.store(
                        cutlass.Int32(cutlass.select_(grown > allocated, allocated, grown)), idx=b
                    )
                counter.store(cutlass.Int32(0), idx=0)


def _weight_tensor_map(w, n_rows, k_in):
    """W as five TMA dimensions (64-element column chunk, row, 64-element chunk index, 1, 1) so one call per k-tile
    lands both 128-byte-swizzled halves; strides in 16-byte units."""
    return cuda.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[TMA_K_BOX, n_rows, k_in // TMA_K_BOX, 1, 1],
        global_strides=[
            (k_in * ELEM_BYTES) // 16,
            (TMA_K_BOX * ELEM_BYTES) // 16,
            (n_rows * k_in * ELEM_BYTES) // 16,
            (n_rows * k_in * ELEM_BYTES) // 16,
        ],
        box_dims=[TMA_K_BOX, CTA_M, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _activation_tensor_map(x, cols, num_tokens):
    """x [N, cols] (rows dense) as (cols, N) with a box of the MMA's token columns: rows past N arrive as zeros."""
    return cuda.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[cols, num_tokens],
        global_strides=[(cols * ELEM_BYTES) // 16],
        box_dims=[TMA_K_BOX, mma_n(num_tokens)],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_ctx_kv(
    w: cute.Tensor,  # [n_rows, K] bf16, K contiguous
    x: cute.Tensor,  # [N, K] bf16, K contiguous
    k_norm: cute.Tensor,
    cos_sin: cute.Tensor,
    cpos: cute.Tensor,
    num_acc: cute.Tensor,
    ctx_len: cute.Tensor,
    slots: cute.Tensor,
    rows: cute.Tensor,
    table: cute.Tensor,
    counts: cute.Tensor,
    pool: cute.Tensor,
    layer_off: cute.Tensor,
    num_ctx: cute.Tensor,
    counter: cute.Tensor,
    eps: cutlass.Float32,
    max_ctx: cutlass.Int64,
    page: cutlass.Int64,
    block_size: cutlass.Int64,
    table_stride: cutlass.Int64,
    page_stride: cutlass.Int64,
    kv_stride: cutlass.Int64,
    head_stride: cutlass.Int64,
    n_rows: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    nkv: cutlass.Constexpr[int],
    k1: cutlass.Constexpr[int],
    n_tokens: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    tma_desc_w = _weight_tensor_map(w, n_rows, k_in)
    tma_desc_x = _activation_tensor_map(x, k_in, n_tokens)
    k3_ctx_kv_kernel(
        tma_desc_w, tma_desc_x, k_norm, cos_sin, cpos, num_acc, ctx_len, slots, rows, table, counts, pool, layer_off,
        num_ctx, counter, eps, max_ctx, page, block_size, table_stride, page_stride, kv_stride, head_stride,
        n_rows, k_in, split, ring, nkv, k1, n_tokens, trigger_early,
    ).launch(
        grid=((n_rows // CTA_M) * split, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
