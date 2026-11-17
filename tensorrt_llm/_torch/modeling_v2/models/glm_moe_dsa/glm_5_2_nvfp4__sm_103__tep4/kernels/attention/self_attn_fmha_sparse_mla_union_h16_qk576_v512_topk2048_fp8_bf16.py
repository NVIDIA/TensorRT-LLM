# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA decode FMHA over the union of a query-row group's selections, H16, CuTe DSL.

The per-row kernels (``self_attn_fmha_sparse_mla_h16_...`` and its V-reuse ``wide`` variant)
give every query row its own CTA pipeline that gathers that row's 2048 selected latent rows.
At MTP3 / MTP5 the 4 / 6 query rows of a sequence are consecutive positions whose selections
overlap heavily (all of them are subsets of the same few thousand KV positions), so the same
576-byte pool rows are gathered up to 6 times, and the gather is what bounds the kernel: per
SM the TMA gather4 of scattered pool rows delivers ~20 B/clk with every SM pulling, ~34 B/clk
when consecutive rows of a page are gathered together (B300 microbenchmarks).

Here one CTA serves a group of ``rows_per_group`` (RG) consecutive query rows of one sequence:

* Union. The CTA reads the RG rows' selected request-local positions (the indexer's top-k,
  before the pool-row conversion) and builds, in SMEM, one byte per KV position whose bit r
  says "row r selected this position". Positions with any bit set, in increasing position
  order, are the group's union; the CTA keeps its share (``num_chunks`` CTAs of a cluster split
  the union tile by tile) as pool rows (``block_table[req, pos / 32] * stride + pos % 32 +
  layer_offset``) and mask bytes. Above ``MASK_CAP`` positions the byte map does not fit and
  the list is the rows' selections back to back, each entry carrying its own row's bit
  (duplicates are then gathered once per row, as the per-row kernels do).
* QK / PV. A Kv128 tile of union entries is gathered once (TMA gather4; two tile sets of four
  16 KB latent slots, SWIZZLE_128B, and one 8 KB rope slot, SWIZZLE_64B with its own 64-byte
  box; PV reads V from the latent slots as the wide kernel does) and multiplied against all
  NC = 16 RG query columns: S^T = K Q^T with N = NC, O^T += V^T P^T with N = NC.
* Softmax. The online softmax runs per row on its 16 columns, with entry e of the tile masked
  to -inf unless its mask byte has the row's bit (the running max starts at -3e38, so a masked
  entry's exponential is exactly 0 even before the row has seen a selection). A row's running
  max therefore covers exactly its own selections, so its output is softmax(s q.k_r) v_r over
  its selected rows, as the per-row kernels compute, up to fp32 summation order (tiles hold
  union entries in position order, not the row's own order). Maxima are max.f32 (FMNMX), not
  the NaN-propagating max (four instructions each); scores are finite, so the values are the
  same. The softmax warps stage each tile's per-warp partial sums with the new max and
  correction; correction thread j < NC folds them into column j's running sum, so a tile takes
  no second softmax barrier.
* Correction. The correction warps rescale O and normalize the partial in 32-column chunks,
  reading the factors from SMEM per chunk, so their registers hold 64 + 32 values at most: with
  every NC column live at once, 4-row groups (NC = 64) spilled. The warp roles' register split
  (softmax 152, correction 200, gather 80, MMA / load 56) leaves no local memory at 2, 3 or 4
  rows (cuobjdump of the compiled kernels: 0 LDL / STL; the old split 152 / 176 / 80 /
  80 spilled 280 B per thread at 4 rows and 16 B at 2 and 3).
* Split. The ``num_chunks`` CTAs of a cluster hold disjoint union tiles of the same group and
  merge their partials through DSMEM, the base kernel's CGA reducer arithmetic over NC rows.

* Sums (``thread_sums``, per specialization). Each softmax thread keeps its own running sums
  (times each tile's correction, like O) and the CTA reduces them once after the tile loop,
  instead of the per-tile warp reduction staged for the correction warps: no per-tile sum
  shuffles. Same band; another fp32 summation order, so the default keeps the staged sums for
  the plans served before it.

Costs (B300, per-CTA clock64 phase stamps, median CTA, census lengths ~2900 positions;
before / after the register split, chunked correction, max.f32 and -inf
masking): MTP5 at 32 sequences, 3-row groups on 2-chunk clusters, 28.5 -> 24.6 us per CTA
(tile loop 18.3 -> 15.3 us, ~1.4 us per tile); MTP3 at 32, 4-row groups on 4-chunk clusters,
27.2 -> 18.9 us (loop 15.1 -> 9.3 us: the old split spilled the correction warps), 17.8 us with
per-thread sums. What is left of an MTP3 4-row CTA besides the loop: bounds 0.4 us, building
the union 2.8 us, the first tile's gather 1.3 us, the last PV 1.0 us, publishing the partial
1.3 us, the cluster merge 2.8 us (mostly waiting for the peer chunks); a call adds ~5 us of
launch and drain over its slowest CTA.

Validity: the kernel takes row t of a sequence of ``sq`` rows and KV length ``n`` to own the
first ``clamp(n - sq + t + 1, 0, 2048)`` entries of its top-k row (the per-row kernels'
positional prefix), each a position in ``[0, n)``; entries past the prefix, negative entries
and positions outside ``[0, n)`` are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 as _base
from .self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 import (
    ACC_DTYPE,
    BAR_SM0,
    CTA_N,
    EB,
    GRAN,
    H16_LOGICAL_TO_PACKED_STATS,
    IO_DTYPE,
    K_HDIM_STAGES,
    KV_POOL_TOKENS_PER_BLOCK,
    KV_STRIDE,
    LBO_K,
    LBO_V,
    M_TILE,
    N_LAT,
    N_V,
    NUM_GATHER_WARPS,
    OUT_DTYPE,
    PV_M,
    QK_K,
    QK_N,
    SBO_LAT,
    SBO_P,
    SBO_ROPE,
    SBO_V,
    ST_C,
    ST_P,
    ST_S,
    THREADS,
    TOPK,
    V_HDIM_STAGES,
    V_KSTEP,
    W_CORR0,
    W_LOAD_GATHER0,
    W_LOAD_KQ,
    W_MMA_PV,
    W_MMA_QK,
    L,
    R,
    _fmax,
    _fragment_sum4,
    _scale_scores_packed,
    _tma_gather4_cta1,
    _warpgroup_row_max,
    _warpgroup_row_sum,
)

NUM_HEADS: int = 16
NUM_KV_SETS: int = 2
N_KV_SLOTS: int = NUM_KV_SETS * K_HDIM_STAGES
#: Bytes of the K/V ring, which holds the position byte map while the union is built.
#: A tile set: four 16 KB latent slots (128 rows x 128 B, SWIZZLE_128B) and one 8 KB rope slot
#: (128 rows x 64 B, SWIZZLE_64B: the rope's 64 bytes only, not a 128-byte box half out of
#: bounds as in the per-row kernels).
LAT_SLOT_BYTES: int = CTA_N * QK_K
ROPE_SLOT_BYTES: int = CTA_N * R
SET_BYTES: int = N_LAT * LAT_SLOT_BYTES + ROPE_SLOT_BYTES
RING_BYTES: int = NUM_KV_SETS * SET_BYTES
MASK_CAP: int = RING_BYTES
SUPPORTED_ROWS_PER_GROUP: tuple[int, ...] = (2, 3, 4)
MAX_CHUNKS: int = 8
#: Bar id of the correction warpgroup (BAR_SM0 = 1 is the softmax warpgroup's).
BAR_CORR: int = 2
#: Columns the correction warps rescale / normalize per TMEM load (see the module docstring).
CORR_CHUNK: int = 32
#: setmaxregister per warp role: softmax W0-W3, correction W4-W7, MMA / load W8-W11, gather
#: W12-W15. Four warps each, 4 x (152 + 200 + 56 + 80) = 1952 <= 2048 = 512 threads x 128.
REG_SOFTMAX: int = 152
REG_CORRECTION: int = 200
REG_ISSUE: int = 56
REG_GATHER: int = 80
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


def _pow2_at_least(v: int) -> int:
    p = 32
    while p < v:
        p *= 2
    return p


def stage_capacity(rows_per_group: int, num_chunks: int) -> int:
    """Union entries one CTA stages: its share of the largest union (RG x 2048), tile-rounded."""
    tiles = (rows_per_group * TOPK // QK_N + num_chunks - 1) // num_chunks
    return tiles * QK_N


def _col_chunks(nc: int) -> tuple[tuple[int, int], ...]:
    """(offset, width) power-of-two pieces of nc columns, widest first (32x32b repeats)."""
    out, off = [], 0
    for w in (64, 32, 16):
        while nc - off >= w:
            out.append((off, w))
            off += w
    return tuple(out)


@cute.jit
def _tmem_ld_cols(addr, nc: cutlass.Constexpr[int]):
    """32x32b load of nc consecutive fp32 columns (power-of-two pieces); needs a LOAD wait."""
    parts = [
        prims.tcgen05_ld("32x32b", cutlass.inttoptr(addr + off, 6, ACC_DTYPE), num=w)
        for off, w in _col_chunks(nc)
    ]
    return cutlass.Vector.from_elements(
        tuple(parts[k][i] for k, (_, w) in enumerate(_col_chunks(nc)) for i in range(w)),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _tmem_st_cols(addr, values, nc: cutlass.Constexpr[int]):
    for off, w in _col_chunks(nc):
        prims.tcgen05_st(
            "32x32b",
            cutlass.inttoptr(addr + off, 6, ACC_DTYPE),
            cutlass.Vector.from_elements(tuple(values[off + i] for i in range(w)), dtype=ACC_DTYPE),
        )


@cute.jit
def _fragment_max4_f32(values):
    """The base kernel's `_fragment_max4` (per-head max over a thread's 4 KV rows) on max.f32."""
    return cutlass.Vector.from_elements(
        (
            _fmax(_fmax(values[0], values[2]), _fmax(values[8], values[10])),
            _fmax(_fmax(values[1], values[3]), _fmax(values[9], values[11])),
            _fmax(_fmax(values[4], values[6]), _fmax(values[12], values[14])),
            _fmax(_fmax(values[5], values[7]), _fmax(values[13], values[15])),
        ),
        dtype=ACC_DTYPE,
    )


@cute.jit
def _max4_f32(a, b):
    """Elementwise max.f32 of two 4-vectors."""
    return cutlass.Vector.from_elements(tuple(_fmax(a[i], b[i]) for i in range(4)), dtype=ACC_DTYPE)


@cute.jit
def _block_exclusive_scan(value, tidx, scan_smem):
    """Exclusive prefix sum of one Int32 per thread over the CTA, and the total."""
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)
    incl = value
    for off in (1, 2, 4, 8, 16):
        other = prims.shfl_sync(
            thread_mask=0xFFFFFFFF, val=incl, offset=off, mask_and_clamp=0, kind=prims.Shfl.UP
        )
        if lane >= cutlass.Int32(off):
            incl = incl + other
    if lane == cutlass.Int32(31):
        scan_smem[warp] = incl
    prims.barrier_cta_sync(0)
    warp_base = cutlass.Int32(0)
    total = cutlass.Int32(0)
    for w in cutlass.range_constexpr(THREADS // 32):
        wv = scan_smem[w]
        if cutlass.Int32(w) < warp:
            warp_base = warp_base + wv
        total = total + wv
    prims.barrier_cta_sync(0)
    return warp_base + incl - value, total


@cute.kernel
def _fmha_union_kernel(
    tma_ql: cutlass.GridConstant[cuda.TensorMap],
    tma_qr: cutlass.GridConstant[cuda.TensorMap],
    tma_kv: cutlass.GridConstant[cuda.TensorMap],
    tma_kvr: cutlass.GridConstant[cuda.TensorMap],
    topk: cute.Tensor,
    block_table: cute.Tensor,
    req_ids: cute.Tensor,
    o: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    layer_offset: cutlass.Int32,
    stride_factor: cutlass.Int32,
    max_blocks: cutlass.Int32,
    scale_log2: cutlass.Float32,
    rows_per_group: cutlass.Constexpr[int],
    num_chunks: cutlass.Constexpr[int],
    thread_sums: cutlass.Constexpr[bool],
) -> None:
    num_heads: cutlass.Constexpr[int] = NUM_HEADS
    rg: cutlass.Constexpr[int] = rows_per_group
    nc: cutlass.Constexpr[int] = NUM_HEADS * rows_per_group
    tmem_o_off: cutlass.Constexpr[int] = ST_S * nc
    tmem_cols: cutlass.Constexpr[int] = _pow2_at_least((ST_S + N_V) * nc)
    stage_cap: cutlass.Constexpr[int] = stage_capacity(rows_per_group, num_chunks)
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    task_x, _, batch = cute.arch.block_idx()
    if cutlass.const_expr(num_chunks > 1):
        cta = cute.arch.block_idx_in_cluster()
    else:
        cta = cutlass.Int32(0)
    group = task_x // cutlass.Int32(num_chunks)
    num_scale_groups: cutlass.Constexpr[int] = num_heads // 4
    q_repeats: cutlass.Constexpr[int] = num_heads // 8
    scale_alignment: cutlass.Constexpr[int] = 16

    # ---- SMEM ------------------------------------------------------------
    sQ = cutlass.Array(IO_DTYPE, nc * L, space=cutlass.AddressSpace.smem, alignment=1024)
    sQr = cutlass.Array(IO_DTYPE, nc * R, space=cutlass.AddressSpace.smem, alignment=1024)
    sKv = cutlass.Array(IO_DTYPE, RING_BYTES, space=cutlass.AddressSpace.smem, alignment=1024)
    stage_row = cutlass.Array(
        cutlass.Int32, stage_cap, space=cutlass.AddressSpace.smem, alignment=16
    )
    stage_mask = cutlass.Array(
        cutlass.Int8, stage_cap, space=cutlass.AddressSpace.smem, alignment=16
    )
    sP = cutlass.Array(IO_DTYPE, ST_P * QK_N * nc, space=cutlass.AddressSpace.smem, alignment=1024)
    softmax_max = cutlass.Array(
        ACC_DTYPE, ST_S * rg * 4 * num_heads, space=cutlass.AddressSpace.smem, alignment=16
    )
    softmax_sum = cutlass.Array(
        ACC_DTYPE,
        ST_C * rg * 4 * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    compact_stats = cutlass.Array(
        ACC_DTYPE, ST_C * rg * num_heads * 3, space=cutlass.AddressSpace.smem, alignment=16
    )
    # Every chunk's final (max, sum) per output column, [chunk][column][2]: each CTA stores its
    # own into every peer's copy before the cluster barrier, so the merge reads them locally.
    partial_stats = cutlass.Array(
        ACC_DTYPE, num_chunks * nc * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    # Per output column j: 1 / sum (epilogue), then the cluster merge's per-peer coefficients.
    inv_smem = cutlass.Array(ACC_DTYPE, nc, space=cutlass.AddressSpace.smem, alignment=16)
    coef_smem = cutlass.Array(
        ACC_DTYPE, nc * num_chunks, space=cutlass.AddressSpace.smem, alignment=16
    )
    sole_smem = cutlass.Array(cutlass.Int32, nc, space=cutlass.AddressSpace.smem, alignment=16)
    scan_smem = cutlass.Array(
        cutlass.Int32, THREADS // 32, space=cutlass.AddressSpace.smem, alignment=16
    )
    q_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(cutlass.Int64, N_KV_SLOTS, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(cutlass.Int64, N_KV_SLOTS, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    p_empty = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    c_full = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    c_empty = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    o_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    # thread_sums: the softmax warps' final per-warp sums are in `softmax_sum` stage 0.
    l_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # [q_begin, q_end, kv length, block-table row, union tiles of this CTA, first entry]
    bounds = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(q_full, 1)
            for i in cutlass.range_constexpr(N_KV_SLOTS):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_S):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_P):
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
                prims.mbarrier_init(p_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_C):
                # Lanes 0..3 of each softmax warp arrive, each after its own stores.
                prims.mbarrier_init(c_full.data_ptr() + i, 16)
                prims.mbarrier_init(c_empty.data_ptr() + i, 4)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_empty, 4)
            # Lanes 0..3 of each softmax warp arrive, each after its own stores.
            prims.mbarrier_init(l_full, 16)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()

    if warp_idx == W_MMA_PV:
        prims.tcgen05_alloc(tmem_i32, tmem_cols, group="cta_1")
    if warp_idx == W_LOAD_KQ:
        # Only W9 reads the predecessor-owned metadata (PDL wait).
        if prims.elect_sync():
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            qb = cu_q_seqlens[batch] // cutlass.Int32(num_heads)
            bounds[0] = qb
            bounds[1] = cu_q_seqlens[batch + cutlass.Int32(1)] // cutlass.Int32(num_heads)
            bounds[2] = sequence_length[batch]
            bounds[3] = req_ids[qb]
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    if warp_idx == W_MMA_PV:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    q_begin = bounds[0]
    sq = bounds[1] - q_begin
    kv_len = bounds[2]
    req = bounds[3]
    row0 = group * cutlass.Int32(rg)
    nrows = sq - row0
    if nrows > cutlass.Int32(rg):
        nrows = cutlass.Int32(rg)
    if nrows <= cutlass.Int32(0):
        # A group past this sequence's rows: every CTA of the cluster exits here.
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols, group="cta_1")
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        prims.exit()
    packed_q0 = q_begin + row0
    q_bytes = (nc * L + nc * R) * EB
    # W9 - Q of the group's rows (one box of RG rows), issued before the union build.
    if warp_idx == W_LOAD_KQ:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(q_full, q_bytes)
        if prims.elect_sync():
            for c in cutlass.range_constexpr(N_LAT):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sQ.data_ptr() + c * (nc * QK_K),
                    tma_ql.get_ptr(),
                    (c * QK_K, 0, packed_q0),
                    q_full,
                )
            prims.cp_async_bulk_tensor_shared_cta_global(
                sQr,
                tma_qr.get_ptr(),
                (0, 0, packed_q0),
                q_full,
            )

    # =====================================================================
    # Union build (all 512 threads): this CTA's union entries -> stage_row / stage_mask.
    # =====================================================================
    topk_ptr = topk.iterator.raw_ptr()
    bt_ptr = block_table.iterator.raw_ptr()
    bt_row = req * max_blocks
    mask_words = cutlass.Pointer(sKv.data_ptr(), dtype=cutlass.Int32)
    dense = kv_len <= cutlass.Int32(MASK_CAP)
    # Positional valid prefix of row r: clamp(kv_len - sq + row0 + r + 1, 0, 2048).
    valid0 = kv_len - sq + row0 + cutlass.Int32(1)
    union_total = cutlass.Int32(0)
    my_base = cutlass.Int32(0)
    # Thread t owns the positions [t S, t S + S), S a whole number of words; the block-table
    # entries of the segment's first and last page (one or two pages while the segment is at
    # most 32 positions, n <= 16384) are loaded with the selections, in one round trip.
    words = (kv_len + cutlass.Int32(3)) // cutlass.Int32(4)
    seg_words = (words + cutlass.Int32(THREADS - 1)) // cutlass.Int32(THREADS)
    seg_lo = tidx * seg_words * cutlass.Int32(4)
    seg_hi = seg_lo + seg_words * cutlass.Int32(4)
    if seg_hi > kv_len:
        seg_hi = kv_len
    page_lo = seg_lo // cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
    page_hi = page_lo
    if seg_hi > seg_lo:
        page_hi = (seg_hi - cutlass.Int32(1)) // cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
    blk_lo = cutlass.Int32(-1)
    blk_hi = cutlass.Int32(-1)
    if dense:
        # A page past the block table (a length the table does not cover) maps to -1.
        if (seg_hi > seg_lo) and (page_lo < max_blocks):
            blk_lo = (bt_ptr + bt_row + page_lo).load()
        if (seg_hi > seg_lo) and (page_hi < max_blocks):
            blk_hi = (bt_ptr + bt_row + page_hi).load()
        # The group's selected positions, all loads issued before any use: entry k of this
        # thread is row k // (2048 / 512) of the group, column tidx + 512 (k % 4). Rows past
        # the sequence read its last row (in bounds) and are discarded below.
        n_ld: cutlass.Constexpr[int] = rg * TOPK // THREADS
        sel = cutlass.Array(cutlass.Int32, n_ld, space=cutlass.AddressSpace.rmem)
        for k in cutlass.range_constexpr(n_ld):
            r_k: cutlass.Constexpr[int] = (k * THREADS) // TOPK
            j = tidx + cutlass.Int32((k * THREADS) % TOPK)
            rr = cutlass.Int32(r_k)
            if rr >= nrows:
                rr = nrows - cutlass.Int32(1)
            sel[k] = (topk_ptr + (packed_q0 + rr) * cutlass.Int32(TOPK) + j).load()
        for w in cutlass.range(tidx, words, THREADS, unroll=1):
            (mask_words + w).store(cutlass.Int32(0))
        prims.barrier_cta_sync(0)
        for k in cutlass.range_constexpr(n_ld):
            r_k: cutlass.Constexpr[int] = (k * THREADS) // TOPK
            r = cutlass.Int32(r_k)
            j = tidx + cutlass.Int32((k * THREADS) % TOPK)
            valid_r = valid0 + r
            if valid_r > cutlass.Int32(TOPK):
                valid_r = cutlass.Int32(TOPK)
            if (r < nrows) and (j < valid_r):
                pos = sel[k]
                if (pos >= cutlass.Int32(0)) and (pos < kv_len):
                    bit = cutlass.Int32(1) << (r + (pos % cutlass.Int32(4)) * cutlass.Int32(8))
                    prims.inline_ptx_hl(
                        "red.shared::cta.or.b32 [{$r0}], {$r1};",
                        read_only_args=[mask_words + pos // cutlass.Int32(4), bit],
                    )
        prims.barrier_cta_sync(0)
        count = cutlass.Int32(0)
        for w in cutlass.range(tidx * seg_words, tidx * seg_words + seg_words, 1, unroll=1):
            if w < words:
                v = (mask_words + w).load()
                for b in cutlass.range_constexpr(4):
                    if ((v >> cutlass.Int32(8 * b)) & cutlass.Int32(0xFF)) != cutlass.Int32(0):
                        count = count + cutlass.Int32(1)
        my_base, union_total = _block_exclusive_scan(count, tidx, scan_smem)
    else:
        # Concatenated rows: entry u is row r's j-th selection.
        for r in cutlass.range_constexpr(rg):
            valid_r = valid0 + cutlass.Int32(r)
            if valid_r > cutlass.Int32(TOPK):
                valid_r = cutlass.Int32(TOPK)
            if valid_r < cutlass.Int32(0):
                valid_r = cutlass.Int32(0)
            if cutlass.Int32(r) < nrows:
                union_total = union_total + valid_r

    total_tiles = (union_total + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
    tiles_per_cta = (total_tiles + cutlass.Int32(num_chunks - 1)) // cutlass.Int32(num_chunks)
    first_entry = cta * tiles_per_cta * cutlass.Int32(QK_N)
    end_entry = first_entry + tiles_per_cta * cutlass.Int32(QK_N)
    if end_entry > union_total:
        end_entry = union_total
    num_kt = tiles_per_cta
    if first_entry + num_kt * cutlass.Int32(QK_N) > union_total:
        num_kt = (end_entry - first_entry + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
    if num_kt < cutlass.Int32(0):
        num_kt = cutlass.Int32(0)
    # Pad the staging of the last tile (and an empty participant's nothing) with -1 rows.
    for i in cutlass.range(tidx, stage_cap, THREADS, unroll=1):
        stage_row[i] = cutlass.Int32(-1)
        stage_mask[i] = cutlass.Int8(0)
    prims.barrier_cta_sync(0)
    if dense:
        u = my_base
        for w in cutlass.range(
            seg_lo // cutlass.Int32(4), (seg_hi + cutlass.Int32(3)) // cutlass.Int32(4), 1, unroll=1
        ):
            v = (mask_words + w).load()
            if v != cutlass.Int32(0):
                for b in cutlass.range_constexpr(4):
                    byte = (v >> cutlass.Int32(8 * b)) & cutlass.Int32(0xFF)
                    if byte != cutlass.Int32(0):
                        pos = w * cutlass.Int32(4) + cutlass.Int32(b)
                        if (u >= first_entry) and (u < end_entry):
                            page = pos // cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
                            blk = blk_hi
                            if page == page_lo:
                                blk = blk_lo
                            elif page != page_hi:
                                blk = cutlass.Int32(-1)
                                if page < max_blocks:
                                    blk = (bt_ptr + bt_row + page).load()
                            prow = cutlass.Int32(-1)
                            if blk >= cutlass.Int32(0):
                                prow = (
                                    blk * stride_factor
                                    + pos % cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
                                    + layer_offset
                                )
                            stage_row[u - first_entry] = prow
                            stage_mask[u - first_entry] = cutlass.Int8(byte)
                        u = u + cutlass.Int32(1)
    else:
        for u in cutlass.range(first_entry + tidx, end_entry, THREADS, unroll=1):
            r = cutlass.Int32(0)
            j = u
            for rr in cutlass.range_constexpr(rg - 1):
                valid_r = valid0 + r
                if valid_r > cutlass.Int32(TOPK):
                    valid_r = cutlass.Int32(TOPK)
                if valid_r < cutlass.Int32(0):
                    valid_r = cutlass.Int32(0)
                if j >= valid_r:
                    j = j - valid_r
                    r = r + cutlass.Int32(1)
            pos = (topk_ptr + (packed_q0 + r) * cutlass.Int32(TOPK) + j).load()
            prow = cutlass.Int32(-1)
            bits = cutlass.Int32(0)
            page = pos // cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
            if (pos >= cutlass.Int32(0)) and (pos < kv_len) and (page < max_blocks):
                blk = (bt_ptr + bt_row + page).load()
                if blk >= cutlass.Int32(0):
                    prow = (
                        blk * stride_factor
                        + pos % cutlass.Int32(KV_POOL_TOKENS_PER_BLOCK)
                        + layer_offset
                    )
                    bits = cutlass.Int32(1) << r
            stage_row[u - first_entry] = prow
            stage_mask[u - first_entry] = cutlass.Int8(bits)
    # The byte map lived in the K/V ring, which the TMA (async proxy) overwrites next.
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    if tidx < cutlass.Int32(ST_S * rg * num_heads):
        softmax_max[tidx] = ACC_DTYPE(-3.0e38)
    prims.barrier_cta_sync(0)

    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        prims.setmaxregister(REG_GATHER, prims.SetMaxRegisterAction.DECREASE)
    if (warp_idx >= cutlass.Int32(8)) and (warp_idx < cutlass.Int32(12)):
        prims.setmaxregister(REG_ISSUE, prims.SetMaxRegisterAction.DECREASE)

    # =====================================================================
    # W12-W15 - gather4 producers over the staged union entries.
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        producer_rank = warp_idx - cutlass.Int32(W_LOAD_GATHER0)
        routed_rows = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.rmem)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            staged_base = kt * cutlass.Int32(QK_N) + producer_rank * cutlass.Int32(
                CTA_N // NUM_GATHER_WARPS
            )
            for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                values = (stage_row.data_ptr() + staged_base + cutlass.Int32(g * 4)).load(
                    count=4, alignment=16
                )
                for j in cutlass.range_constexpr(4):
                    routed_rows[g * 4 + j] = values[j]
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            set_off = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(SET_BYTES)
            empty_phase = ((kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)) ^ cutlass.Int32(1)
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                slot = set_base + cutlass.Int32(c)
                row_bytes: cutlass.Constexpr[int] = QK_K if c < N_LAT else R
                while not prims.mbarrier_try_wait_parity(kv_empty.data_ptr() + slot, empty_phase):
                    pass
                if producer_rank == cutlass.Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(
                            kv_full.data_ptr() + slot, CTA_N * row_bytes
                        )
                for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                    dst = (
                        sKv.data_ptr()
                        + set_off
                        + cutlass.Int32(c * LAT_SLOT_BYTES)
                        + (
                            producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                            + cutlass.Int32(g * 4)
                        )
                        * cutlass.Int32(row_bytes)
                    )
                    if cutlass.const_expr(c < N_LAT):
                        _tma_gather4_cta1(
                            dst,
                            tma_kv.get_ptr(),
                            kv_full.data_ptr() + slot,
                            cutlass.Int32(c * QK_K),
                            routed_rows[g * 4],
                            routed_rows[g * 4 + 1],
                            routed_rows[g * 4 + 2],
                            routed_rows[g * 4 + 3],
                        )
                    else:
                        _tma_gather4_cta1(
                            dst,
                            tma_kvr.get_ptr(),
                            kv_full.data_ptr() + slot,
                            cutlass.Int32(0),
                            routed_rows[g * 4],
                            routed_rows[g * 4 + 1],
                            routed_rows[g * 4 + 2],
                            routed_rows[g * 4 + 3],
                        )

    # =====================================================================
    # W8 - QK over the tile's set, N = NC; releases the rope slot.
    # =====================================================================
    if warp_idx == W_MMA_QK:
        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE, a_dtype=IO_DTYPE, b_dtype=IO_DTYPE, n_dim=nc, m_dim=M_TILE
        )
        while not prims.mbarrier_try_wait_parity(q_full, 0):
            pass
        dq = _compat48.tcgen05_smem_desc(
            start_address=sQ,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dqr = _compat48.tcgen05_smem_desc(
            start_address=sQr,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        dk_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dkr_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        sst = cutlass.Int32(0)
        sph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            while not prims.mbarrier_try_wait_parity(
                s_empty.data_ptr() + sst, sph ^ cutlass.Int32(1)
            ):
                pass
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            kvph = (kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)
            stmem = cutlass.inttoptr(tmem_raw + sst * cutlass.Int32(nc), 6, cutlass.Float32)
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                slot = set_base + cutlass.Int32(c)
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + slot, kvph):
                    pass
                ko = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(SET_BYTES) + cutlass.Int32(
                    c * LAT_SLOT_BYTES
                )
                if cutlass.const_expr(c < N_LAT):
                    qo: cutlass.Constexpr[int] = c * nc * QK_K * EB
                    for kb in cutlass.range_constexpr(QK_K // GRAN):
                        g: cutlass.Constexpr[int] = kb * GRAN * EB
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F8F6F4,
                                prims.CTAGroup.CTA_1,
                                stmem,
                                dk_root.advance_start_address(ko + g),
                                dq.advance_start_address(qo + g),
                                idesc_qk,
                                (c != 0) or (kb != 0),
                            )
                else:
                    for kb in cutlass.range_constexpr(R // GRAN):
                        g: cutlass.Constexpr[int] = kb * GRAN * EB
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F8F6F4,
                                prims.CTAGroup.CTA_1,
                                stmem,
                                dkr_root.advance_start_address(ko + g),
                                dqr.advance_start_address(g),
                                idesc_qk,
                                True,
                            )
                    if prims.elect_sync():
                        if kt + cutlass.Int32(NUM_KV_SETS) < num_kt:
                            prims.tcgen05_commit(
                                kv_empty.data_ptr() + slot, group=prims.CTAGroup.CTA_1
                            )
            if prims.elect_sync():
                prims.tcgen05_commit(s_full.data_ptr() + sst, group=prims.CTAGroup.CTA_1)
            sst += cutlass.Int32(1)
            if sst == cutlass.Int32(ST_S):
                sst = cutlass.Int32(0)
                sph ^= cutlass.Int32(1)

    # =====================================================================
    # W11 - SwapsAb VP, N = NC; V stripe s is the tile's K latent slot s.
    # =====================================================================
    if warp_idx == W_MMA_PV:
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE,
            a_dtype=IO_DTYPE,
            b_dtype=IO_DTYPE,
            n_dim=nc,
            m_dim=PV_M,
            a_major=1,
            b_major=0,
        )
        dp_root = _compat48.tcgen05_smem_desc(
            start_address=sP,
            leading_byte_offset=16,
            stride_byte_offset=SBO_P,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dv_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_V,
            stride_byte_offset=SBO_V,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        pst = cutlass.Int32(0)
        pph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            while not prims.mbarrier_try_wait_parity(p_full.data_ptr() + pst, pph):
                pass
            oph = kt % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_empty, oph ^ cutlass.Int32(1)):
                pass
            pbase = pst * cutlass.Int32(QK_N * nc)
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            kvph = (kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)
            for s in cutlass.range_constexpr(V_HDIM_STAGES):
                slot = set_base + cutlass.Int32(s)
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + slot, kvph):
                    pass
                vso = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(SET_BYTES) + cutlass.Int32(
                    s * LAT_SLOT_BYTES
                )
                otp = cutlass.inttoptr(tmem_raw + tmem_o_off + s * nc, 6, cutlass.Float32)
                for kb in cutlass.range_constexpr(QK_N // GRAN):
                    po: cutlass.Constexpr[int] = kb * GRAN * EB
                    vo: cutlass.Constexpr[int] = kb * V_KSTEP
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            otp,
                            dv_root.advance_start_address(vso + vo),
                            dp_root.advance_start_address(pbase + po),
                            idesc_pv,
                            (kt > 0) or (kb != 0),
                        )
                if prims.elect_sync():
                    if kt + cutlass.Int32(NUM_KV_SETS) < num_kt:
                        prims.tcgen05_commit(kv_empty.data_ptr() + slot, group=prims.CTAGroup.CTA_1)
            if prims.elect_sync():
                prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)
            if prims.elect_sync():
                if kt + cutlass.Int32(ST_P) < num_kt:
                    prims.tcgen05_commit(p_empty.data_ptr() + pst, group=prims.CTAGroup.CTA_1)
            pst += cutlass.Int32(1)
            if pst == cutlass.Int32(ST_P):
                pst = cutlass.Int32(0)
                pph ^= cutlass.Int32(1)

    # =====================================================================
    # W0-W3 - online softmax, one 16-column H16 block per row of the group.
    # =====================================================================
    if warp_idx < cutlass.Int32(4):
        prims.setmaxregister(REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        col_group = lane % cutlass.Int32(4)
        row_idx = lane // cutlass.Int32(4)
        local_row_idx = row_idx % cutlass.Int32(4)
        base = prims.TmemAddr(tmem_raw)
        row0_t = base.row_id + warp_idx * cutlass.Int32(32)
        addr0 = (row0_t << cutlass.Int32(16)) | base.col_id
        addr1 = ((row0_t + cutlass.Int32(16)) << cutlass.Int32(16)) | base.col_id
        prev_max = cutlass.Array(ACC_DTYPE, rg * num_scale_groups, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(rg * num_scale_groups):
            prev_max[i] = ACC_DTYPE(-3.0e38)
        # thread_sums: this thread's running sums per (row, head slot), rescaled by each tile's
        # correction like O, and reduced over the CTA once after the loop.
        run_l = cutlass.Array(ACC_DTYPE, rg * num_scale_groups, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(rg * num_scale_groups):
            run_l[i] = ACC_DTYPE(0.0)
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)
        # The four KV rows of a tile this thread's 16x256b fragments hold (offsets 0/8/16/24).
        kv_row_base = warp_idx * cutlass.Int32(32) + row_idx

        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            while not prims.mbarrier_try_wait_parity(s_full.data_ptr() + stage, phase):
                pass
            mrow = kt * cutlass.Int32(QK_N) + kv_row_base
            m0 = cutlass.Int32(stage_mask[mrow]) & cutlass.Int32(0xFF)
            m1 = cutlass.Int32(stage_mask[mrow + cutlass.Int32(8)]) & cutlass.Int32(0xFF)
            m2 = cutlass.Int32(stage_mask[mrow + cutlass.Int32(16)]) & cutlass.Int32(0xFF)
            m3 = cutlass.Int32(stage_mask[mrow + cutlass.Int32(24)]) & cutlass.Int32(0xFF)
            all_scores = cutlass.Array(ACC_DTYPE, rg * 16, space=cutlass.AddressSpace.rmem)
            for r in cutlass.range_constexpr(rg):
                frag0 = prims.tcgen05_ld(
                    "16x256b",
                    cutlass.inttoptr(
                        addr0 + stage * cutlass.Int32(nc) + cutlass.Int32(r * 16), 6, ACC_DTYPE
                    ),
                    num=q_repeats,
                )
                frag1 = prims.tcgen05_ld(
                    "16x256b",
                    cutlass.inttoptr(
                        addr1 + stage * cutlass.Int32(nc) + cutlass.Int32(r * 16), 6, ACC_DTYPE
                    ),
                    num=q_repeats,
                )
                for i in cutlass.range_constexpr(8):
                    all_scores[r * 16 + i] = frag0[i]
                    all_scores[r * 16 + 8 + i] = frag1[i]
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            # Fragment value i of a row block sits at KV row offset 8 * (((i%4)//2) + 2*(i//8)).
            for r in cutlass.range_constexpr(rg):
                for i in cutlass.range_constexpr(16):
                    sel: cutlass.Constexpr[int] = ((i % 4) // 2) + 2 * (i // 8)
                    mb = m0
                    if cutlass.const_expr(sel == 1):
                        mb = m1
                    if cutlass.const_expr(sel == 2):
                        mb = m2
                    if cutlass.const_expr(sel == 3):
                        mb = m3
                    if ((mb >> cutlass.Int32(r)) & cutlass.Int32(1)) == cutlass.Int32(0):
                        all_scores[r * 16 + i] = ACC_DTYPE(float("-inf"))
            for r in cutlass.range_constexpr(rg):
                scores_r = cutlass.Vector.from_elements(
                    tuple(all_scores[r * 16 + i] for i in range(16)), dtype=ACC_DTYPE
                )
                thread_max = _fragment_max4_f32(scores_r)
                reduced_max = cutlass.Vector.from_elements(
                    tuple(_warpgroup_row_max(thread_max[i]) for i in range(4)), dtype=ACC_DTYPE
                )
                atomic_value = reduced_max[0]
                if local_row_idx == cutlass.Int32(1):
                    atomic_value = reduced_max[1]
                elif local_row_idx == cutlass.Int32(2):
                    atomic_value = reduced_max[2]
                elif local_row_idx == cutlass.Int32(3):
                    atomic_value = reduced_max[3]
                atomic_value = _fmax(
                    atomic_value,
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=atomic_value,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    ),
                )
                if lane < cutlass.Int32(16):
                    max_head = col_group * cutlass.Int32(4) + local_row_idx
                    max_slot = (
                        (stage * cutlass.Int32(rg) + cutlass.Int32(r))
                        * cutlass.Int32(4 * num_heads)
                        + warp_idx * cutlass.Int32(num_heads)
                        + max_head
                    )
                    softmax_max[max_slot] = atomic_value
            prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                prims.mbarrier_arrive(s_empty.data_ptr() + stage, relaxed=True)
            tile_max = cutlass.Array(
                ACC_DTYPE, rg * num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            max_group = col_group * cutlass.Int32(num_scale_groups)
            for r in cutlass.range_constexpr(rg):
                max_stage = (stage * cutlass.Int32(rg) + cutlass.Int32(r)) * cutlass.Int32(
                    4 * num_heads
                )
                tm = (softmax_max.data_ptr() + max_stage + max_group).load(
                    count=num_scale_groups, alignment=scale_alignment
                )
                for max_warp in cutlass.range_constexpr(1, 4):
                    tm = _max4_f32(
                        tm,
                        (
                            softmax_max.data_ptr() + max_stage + max_warp * num_heads + max_group
                        ).load(count=num_scale_groups, alignment=scale_alignment),
                    )
                for i in cutlass.range_constexpr(num_scale_groups):
                    tile_max[r * num_scale_groups + i] = tm[i]

            while not prims.mbarrier_try_wait_parity(
                c_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass
            while not prims.mbarrier_try_wait_parity(
                p_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass

            stat_base = stage * cutlass.Int32(rg * num_heads * 3)
            head_base = lane * cutlass.Int32(num_scale_groups)
            smem_swizzle = cutlass.Swizzle.from_name("s128b")
            for r in cutlass.range_constexpr(rg):
                pm = cutlass.Vector.from_elements(
                    tuple(prev_max[r * num_scale_groups + i] for i in range(num_scale_groups)),
                    dtype=ACC_DTYPE,
                )
                tm = cutlass.Vector.from_elements(
                    tuple(tile_max[r * num_scale_groups + i] for i in range(num_scale_groups)),
                    dtype=ACC_DTYPE,
                )
                new_max = _max4_f32(pm, tm)
                corr = cute.math.exp2((pm - new_max) * scale_log2, fastmath=True)
                scores_r = cutlass.Vector.from_elements(
                    tuple(all_scores[r * 16 + i] for i in range(16)), dtype=ACC_DTYPE
                )
                scaled = _scale_scores_packed(scores_r, new_max, scale_log2, num_heads)
                # An unselected entry (-inf) contributes exactly 0, even while the row's max is
                # still the -3e38 of "no selection seen yet": -inf * s + bias is -inf.
                probs = cute.math.exp2(scaled, fastmath=True)
                thread_sum = _fragment_sum4(probs)
                if cutlass.const_expr(thread_sums):
                    for i in cutlass.range_constexpr(num_scale_groups):
                        run_l[r * num_scale_groups + i] = (
                            run_l[r * num_scale_groups + i] * corr[i] + thread_sum[i]
                        )
                else:
                    warp_sum = cutlass.Vector.from_elements(
                        tuple(_warpgroup_row_sum(thread_sum[i]) for i in range(4)), dtype=ACC_DTYPE
                    )
                    # This warp's partial sums of the tile, staged: the correction warps fold the
                    # four warps' partials into each column's running sum, so the softmax needs
                    # no second barrier per tile.
                    if lane < cutlass.Int32(4):
                        (
                            softmax_sum.data_ptr()
                            + stage * cutlass.Int32(rg * 4 * num_heads)
                            + cutlass.Int32(r * 4 * num_heads)
                            + warp_idx * cutlass.Int32(num_heads)
                            + lane * cutlass.Int32(num_scale_groups)
                        ).store(warp_sum, alignment=scale_alignment)
                # Warp 0 publishes the new max and the correction.
                if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(4)):
                    rb = stat_base + cutlass.Int32(r * num_heads * 3)
                    (compact_stats.data_ptr() + rb + cutlass.Int32(num_heads) + head_base).store(
                        new_max, alignment=scale_alignment
                    )
                    (
                        compact_stats.data_ptr() + rb + cutlass.Int32(2 * num_heads) + head_base
                    ).store(corr, alignment=scale_alignment)
                for half in cutlass.range_constexpr(2):
                    half_probs = cutlass.Vector.from_elements(
                        tuple(probs[half * 8 + i] for i in range(8)), dtype=ACC_DTYPE
                    )
                    packed_p = half_probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                    address_row = lane % cutlass.Int32(8)
                    address_segment = (lane // cutlass.Int32(8)) % cutlass.Int32(2)
                    row_start = (
                        stage * cutlass.Int32(nc * QK_N)
                        + (cutlass.Int32(r * 16) + address_segment * cutlass.Int32(8) + address_row)
                        * cutlass.Int32(QK_N)
                        + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half)) * cutlass.Int32(16)
                    )
                    prims.stmatrix(
                        (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                        [packed_p[0], packed_p[1]],
                        prims.MMALayout.COL,
                        shape=prims.StoreShape.M16N8,
                    )
                for i in cutlass.range_constexpr(num_scale_groups):
                    prev_max[r * num_scale_groups + i] = new_max[i]
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full.data_ptr() + stage)
            if lane < cutlass.Int32(4):
                prims.mbarrier_arrive(c_full.data_ptr() + stage)
            stage += cutlass.Int32(1)
            if stage == cutlass.Int32(ST_S):
                stage = cutlass.Int32(0)
                phase ^= cutlass.Int32(1)
        if cutlass.const_expr(thread_sums):
            # Each warp's final sums per (row, packed head), [row][warp][16] in the stage-0
            # partial area (unused per tile under thread_sums), then l_full.
            for r in cutlass.range_constexpr(rg):
                warp_l = cutlass.Vector.from_elements(
                    tuple(_warpgroup_row_sum(run_l[r * num_scale_groups + i]) for i in range(4)),
                    dtype=ACC_DTYPE,
                )
                if lane < cutlass.Int32(4):
                    (
                        softmax_sum.data_ptr()
                        + cutlass.Int32(r * 4 * num_heads)
                        + warp_idx * cutlass.Int32(num_heads)
                        + lane * cutlass.Int32(num_scale_groups)
                    ).store(warp_l, alignment=scale_alignment)
            if lane < cutlass.Int32(4):
                prims.mbarrier_arrive(l_full)

    # =====================================================================
    # W4-7 - correction / rescale / epilogue over NC columns.
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_CORR0)) and (warp_idx < cutlass.Int32(W_CORR0 + 4)):
        prims.setmaxregister(REG_CORRECTION, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        corr_warp = warp_idx - cutlass.Int32(W_CORR0)
        out_row = corr_warp * cutlass.Int32(32) + lane
        base = prims.TmemAddr(tmem_raw)
        my_addr = ((base.row_id + corr_warp * cutlass.Int32(32)) << 16) | base.col_id
        cst = cutlass.Int32(0)
        cph = cutlass.Int32(0)
        cthread = corr_warp * cutlass.Int32(32) + lane
        # Thread j < NC keeps column j's running sum (the softmax warps publish per-warp
        # partials and the correction; under thread_sums they keep the sums themselves);
        # (row, head, packed stats slot) of column j.
        jr = cthread // cutlass.Int32(num_heads)
        jh = cthread % cutlass.Int32(num_heads)
        jph = (
            (jh % cutlass.Int32(2))
            + cutlass.Int32(4) * ((jh // cutlass.Int32(2)) % cutlass.Int32(4))
            + cutlass.Int32(2) * (jh // cutlass.Int32(8))
        )
        run_sum = ACC_DTYPE(0.0)
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                pass
            if cutlass.const_expr(not thread_sums):
                if cthread < cutlass.Int32(nc):
                    rbj = cst * cutlass.Int32(rg * num_heads * 3) + jr * cutlass.Int32(
                        num_heads * 3
                    )
                    cj = compact_stats[rbj + cutlass.Int32(2 * num_heads) + jph]
                    pb = (cst * cutlass.Int32(rg) + jr) * cutlass.Int32(4 * num_heads) + jph
                    ts = softmax_sum[pb]
                    for w in cutlass.range_constexpr(1, 4):
                        ts = ts + softmax_sum[pb + cutlass.Int32(w * num_heads)]
                    run_sum = run_sum * cj + ts
            oph = kt % cutlass.Int32(2)
            if kt > cutlass.Int32(0):
                while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                    pass
                # O *= this tile's correction, CORR_CHUNK columns per TMEM load, the factors
                # read from the stage's stats per chunk (so the stage is released after).
                rbs = cst * cutlass.Int32(rg * num_heads * 3)
                for s in cutlass.range_constexpr(N_V):
                    for c0 in cutlass.range_constexpr(0, nc, CORR_CHUNK):
                        cw: cutlass.Constexpr[int] = min(CORR_CHUNK, nc - c0)
                        tile_c = _tmem_ld_cols(my_addr + tmem_o_off + s * nc + c0, cw)
                        factors = []
                        for r in cutlass.range_constexpr(c0 // num_heads, (c0 + cw) // num_heads):
                            packed_corr = (
                                compact_stats.data_ptr()
                                + rbs
                                + cutlass.Int32(r * num_heads * 3 + 2 * num_heads)
                            ).load(count=num_heads, alignment=16)
                            for h in cutlass.range_constexpr(num_heads):
                                ph: cutlass.Constexpr[int] = H16_LOGICAL_TO_PACKED_STATS[h]
                                factors.append(packed_corr[ph])
                        corr_c = cutlass.Vector.from_elements(tuple(factors), dtype=ACC_DTYPE)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        _tmem_st_cols(my_addr + tmem_o_off + s * nc + c0, tile_c * corr_c, cw)
                prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(o_empty, relaxed=True)
            if prims.elect_sync():
                prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)
            cst += cutlass.Int32(1)
            if cst == cutlass.Int32(ST_C):
                cst = cutlass.Int32(0)
                cph ^= cutlass.Int32(1)

        # Final normalization into this CTA's partial (bf16, in the K/V ring: every MMA
        # that read the ring has completed once the last o_full has landed).
        partial_o = sKv.data_ptr()
        po16 = cutlass.Pointer(partial_o, dtype=OUT_DTYPE)
        if num_kt > cutlass.Int32(0):
            final_phase = (num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                pass
            # The last tile's published stats are the CTA's final (sum, max) per column;
            # correction thread j < NC computes column j's reciprocal once.
            last_stage = (num_kt - cutlass.Int32(1)) % cutlass.Int32(ST_C)
            if cthread < cutlass.Int32(nc):
                jr = cthread // cutlass.Int32(num_heads)
                jh = cthread % cutlass.Int32(num_heads)
                # Logical head jh sits at packed slot H16_LOGICAL_TO_PACKED_STATS[jh]:
                # (jh % 2) + 4 ((jh // 2) % 4) + 2 (jh // 8).
                ph = (
                    (jh % cutlass.Int32(2))
                    + cutlass.Int32(4) * ((jh // cutlass.Int32(2)) % cutlass.Int32(4))
                    + cutlass.Int32(2) * (jh // cutlass.Int32(8))
                )
                rb = last_stage * cutlass.Int32(rg * num_heads * 3) + jr * cutlass.Int32(
                    num_heads * 3
                )
                fsum = run_sum
                if cutlass.const_expr(thread_sums):
                    while not prims.mbarrier_try_wait_parity(l_full, 0):
                        pass
                    lb = jr * cutlass.Int32(4 * num_heads) + ph
                    fsum = softmax_sum[lb]
                    for w in cutlass.range_constexpr(1, 4):
                        fsum = fsum + softmax_sum[lb + cutlass.Int32(w * num_heads)]
                fmax = compact_stats[rb + cutlass.Int32(num_heads) + ph]
                inv_smem[cthread] = ACC_DTYPE(1.0) / _fmax(fsum, ACC_DTYPE(1.0e-30))
                _publish_stats(partial_stats, cta, cthread, fmax, fsum, nc, num_chunks)
            prims.barrier_cta_sync(BAR_CORR, thread_count=128)
            for s in cutlass.range_constexpr(N_V):
                out_col = s * PV_M + out_row
                for c0 in cutlass.range_constexpr(0, nc, CORR_CHUNK):
                    cw: cutlass.Constexpr[int] = min(CORR_CHUNK, nc - c0)
                    tile_c = _tmem_ld_cols(my_addr + tmem_o_off + s * nc + c0, cw)
                    inv_c = (inv_smem.data_ptr() + c0).load(count=cw, alignment=16)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    for j in cutlass.range_constexpr(cw):
                        (po16 + cutlass.Int32((c0 + j) * L) + out_col).store(
                            OUT_DTYPE(tile_c[j] * inv_c[j])
                        )
        else:
            # Empty participant: the neutral partial (O = 0, max = -3e38, sum = 0).
            for s in cutlass.range_constexpr(N_V):
                out_col = s * PV_M + out_row
                for j in cutlass.range_constexpr(nc):
                    (po16 + cutlass.Int32(j * L) + out_col).store(OUT_DTYPE(0.0))
            if cthread < cutlass.Int32(nc):
                _publish_stats(
                    partial_stats, cta, cthread, ACC_DTYPE(-3.0e38), ACC_DTYPE(0.0), nc, num_chunks
                )

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(num_chunks == 1):
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols, group="cta_1")
    _reduce_union_output(
        cutlass.Pointer(sKv.data_ptr(), dtype=OUT_DTYPE),
        partial_stats,
        coef_smem,
        sole_smem,
        o,
        tidx,
        packed_q0,
        nrows,
        cta,
        scale_log2,
        nc,
        num_chunks,
    )
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(num_chunks > 1):
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, tmem_cols, group="cta_1")
    if tidx == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    if cutlass.const_expr(num_chunks > 1):
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()


@cute.jit
def _publish_stats(
    partial_stats,
    cta,
    j,
    fmax,
    fsum,
    nc: cutlass.Constexpr[int],
    num_chunks: cutlass.Constexpr[int],
):
    """Column j's (max, sum) of this chunk into slot [cta][j] of every cluster peer's table
    (its own included); the cluster barrier before the merge orders the stores."""
    off = (cta * cutlass.Int32(nc) + j) * cutlass.Int32(2)
    if cutlass.const_expr(num_chunks == 1):
        partial_stats[off] = fmax
        partial_stats[off + cutlass.Int32(1)] = fsum
    else:
        for peer_rank in cutlass.range_constexpr(num_chunks):
            remote = prims.mapa(partial_stats.data_ptr(), cutlass.Int32(peer_rank))
            remote[off] = fmax
            remote[off + cutlass.Int32(1)] = fsum


@cute.jit
def _reduce_union_output(
    po16,
    partial_stats,
    coef_smem,
    sole_smem,
    o: cute.Tensor,
    tidx,
    packed_q0,
    nrows,
    cta,
    scale_log2,
    nc: cutlass.Constexpr[int],
    num_chunks: cutlass.Constexpr[int],
) -> None:
    """Merge the cluster's partials of the group's NC (row, head) outputs and store the
    group's valid rows: output row (packed_q0 * 16 + j) for j < 16 * nrows. With one chunk
    the partial is already normalized and is copied. With several, thread j < NC first turns
    column j's peer stats into per-peer weights sum_k 2^((max_k - max) s) / mass (the base
    kernel's CGA arithmetic, regrouped), and a column with exactly one nonzero partial copies
    that partial through unchanged."""
    if cutlass.const_expr(num_chunks > 1):
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if tidx < cutlass.Int32(nc):
            global_max = ACC_DTYPE(-3.0e38)
            for peer_rank in cutlass.range_constexpr(num_chunks):
                global_max = _fmax(
                    global_max,
                    partial_stats[(cutlass.Int32(peer_rank * nc) + tidx) * cutlass.Int32(2)],
                )
            weights = cutlass.Array(ACC_DTYPE, num_chunks, space=cutlass.AddressSpace.rmem)
            global_mass = ACC_DTYPE(0.0)
            nonzero_peers = cutlass.Int32(0)
            sole_peer = cutlass.Int32(0)
            for peer_rank in cutlass.range_constexpr(num_chunks):
                stat_offset = (cutlass.Int32(peer_rank * nc) + tidx) * cutlass.Int32(2)
                peer_max = partial_stats[stat_offset]
                peer_sum = partial_stats[stat_offset + cutlass.Int32(1)]
                if peer_sum != ACC_DTYPE(0.0):
                    nonzero_peers += cutlass.Int32(1)
                    sole_peer = cutlass.Int32(peer_rank)
                w = peer_sum * cute.math.exp2((peer_max - global_max) * scale_log2, fastmath=True)
                weights[peer_rank] = w
                global_mass += w
            inv_mass = ACC_DTYPE(1.0) / _fmax(global_mass, ACC_DTYPE(1.0e-30))
            for peer_rank in cutlass.range_constexpr(num_chunks):
                coef_smem[tidx * cutlass.Int32(num_chunks) + cutlass.Int32(peer_rank)] = (
                    weights[peer_rank] * inv_mass
                )
            sole = cutlass.Int32(-1)
            if nonzero_peers == cutlass.Int32(1):
                sole = sole_peer
            sole_smem[tidx] = sole
        prims.barrier_cta_sync(0)
    total_vectors: cutlass.Constexpr[int] = nc * L // 8
    valid_vectors = nrows * cutlass.Int32(NUM_HEADS * L // 8)
    first_vector = (cta * cutlass.Int32(total_vectors)) // cutlass.Int32(num_chunks)
    end_vector = ((cta + cutlass.Int32(1)) * cutlass.Int32(total_vectors)) // cutlass.Int32(
        num_chunks
    )
    if end_vector > valid_vectors:
        end_vector = valid_vectors
    vectors_per_thread: cutlass.Constexpr[int] = (total_vectors + num_chunks * THREADS - 1) // (
        num_chunks * THREADS
    )
    o_ptr = o.iterator.raw_ptr()
    for vector_iter in cutlass.range_constexpr(vectors_per_thread):
        flat_vector = first_vector + tidx + cutlass.Int32(vector_iter * THREADS)
        if flat_vector < end_vector:
            j = flat_vector // cutlass.Int32(L // 8)
            col = (flat_vector % cutlass.Int32(L // 8)) * cutlass.Int32(8)
            out_off = (packed_q0 * cutlass.Int32(NUM_HEADS) + j) * cutlass.Int32(L) + col
            if cutlass.const_expr(num_chunks == 1):
                v = (po16 + j * cutlass.Int32(L) + col).load(count=8, alignment=16)
                (o_ptr + out_off).store(v, alignment=16)
            else:
                raws = [
                    (prims.mapa(po16, cutlass.Int32(k)) + j * cutlass.Int32(L) + col).load(
                        count=8, alignment=16
                    )
                    for k in range(num_chunks)
                ]
                merged_v = cutlass.Vector.from_elements(
                    tuple(ACC_DTYPE(0.0) for _ in range(8)), dtype=ACC_DTYPE
                )
                for k in cutlass.range_constexpr(num_chunks):
                    merged_v += (
                        raws[k].to(ACC_DTYPE)
                        * coef_smem[j * cutlass.Int32(num_chunks) + cutlass.Int32(k)]
                    )
                merged = merged_v.to(OUT_DTYPE)
                sole = sole_smem[j]
                for k in cutlass.range_constexpr(num_chunks):
                    if sole == cutlass.Int32(k):
                        merged = raws[k]
                (o_ptr + out_off).store(merged, alignment=16)


@cute.jit
def _launch_fmha_union(
    q_storage: cute.Tensor,
    kv_storage: cute.Tensor,
    topk: cute.Tensor,
    block_table: cute.Tensor,
    req_ids: cute.Tensor,
    output: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    layer_offset: cutlass.Int32,
    stride_factor: cutlass.Int32,
    max_blocks: cutlass.Int32,
    num_groups: cutlass.Int32,
    stream: cuda_driver.CUstream,
    scale_log2: cutlass.Constexpr[float],
    rows_per_group: cutlass.Constexpr[int],
    num_chunks: cutlass.Constexpr[int],
    thread_sums: cutlass.Constexpr[bool],
) -> None:
    num_heads = NUM_HEADS

    def q_tensor_map(offset, inner, box_inner, swizzle):
        return cuda.create_tensor_map_tiled(
            global_address=(q_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, num_heads, q_storage.shape[0]],
            global_strides=[
                KV_STRIDE * EB // 16,
                num_heads * KV_STRIDE * EB // 16,
            ],
            box_dims=[box_inner, num_heads, rows_per_group],
            swizzle=swizzle,
        )

    tma_ql = q_tensor_map(0, L, QK_K, cuda.TensorMapSwizzle.s128b)
    tma_qr = q_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b)
    tma_kv = cuda.create_tensor_map_tiled(
        global_address=kv_storage.iterator.toint(),
        dtype=IO_DTYPE,
        global_dims=[KV_STRIDE, kv_storage.shape[0]],
        global_strides=[KV_STRIDE * EB // 16],
        box_dims=[QK_K, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    # The rope (bytes 512..575 of a row) as its own 64-byte box, SWIZZLE_64B.
    tma_kvr = cuda.create_tensor_map_tiled(
        global_address=(kv_storage.iterator + L).toint(),
        dtype=IO_DTYPE,
        global_dims=[R, kv_storage.shape[0]],
        global_strides=[KV_STRIDE * EB // 16],
        box_dims=[R, 1],
        swizzle=cuda.TensorMapSwizzle.s64b,
    )
    _fmha_union_kernel(
        tma_ql,
        tma_qr,
        tma_kv,
        tma_kvr,
        topk,
        block_table,
        req_ids,
        output,
        cu_q_seqlens,
        sequence_length,
        layer_offset,
        stride_factor,
        max_blocks,
        cutlass.Float32(scale_log2),
        rows_per_group,
        num_chunks,
        thread_sums,
    ).launch(
        grid=(num_chunks * num_groups, 1, cu_q_seqlens.shape[0] - 1),
        block=(THREADS, 1, 1),
        cluster=(num_chunks, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


@dataclass(frozen=True)
class UnionSpecialization:
    """`thread_sums`: each softmax thread keeps its own running sums and the CTA reduces them
    once after the tile loop, instead of a warp reduction staged per tile and folded by the
    correction warps. Same band, another fp32 summation order (so not bitwise the default)."""

    rows_per_group: int
    num_chunks: int
    q_scaling: float = 1.0
    thread_sums: bool = False

    def validate(self) -> None:
        if self.rows_per_group not in SUPPORTED_ROWS_PER_GROUP:
            raise ValueError(f"rows_per_group must be one of {SUPPORTED_ROWS_PER_GROUP}")
        if not 1 <= self.num_chunks <= MAX_CHUNKS:
            raise ValueError(f"num_chunks must be in [1, {MAX_CHUNKS}], got {self.num_chunks}")
        if self.rows_per_group == 4 and self.num_chunks == 1:
            # One CTA would stage the whole 4 x 2048-entry union (40 KB of row / mask arrays):
            # past the SMEM a CTA can launch with next to the K/V ring, Q and P.
            raise ValueError("4-row groups need at least 2 chunks (SMEM)")

    @property
    def scale_log2(self) -> float:
        return _base.MlaFmhaSpecialization(
            all_sparse_indices_valid=False, q_scaling=self.q_scaling
        ).scale_log2


def _fake_tensors() -> tuple[object, ...]:
    make_fake = cute.runtime.make_fake_compact_tensor
    total_q = cute.sym_int64()
    return (
        make_fake(
            cutlass.Uint8, (total_q, NUM_HEADS, KV_STRIDE), stride_order=(2, 1, 0), assumed_align=16
        ),
        make_fake(
            cutlass.Uint8, (cute.sym_int64(), KV_STRIDE), stride_order=(1, 0), assumed_align=16
        ),
        make_fake(cutlass.Int32, (total_q, TOPK), stride_order=(1, 0), assumed_align=16),
        make_fake(
            cutlass.Int32,
            (cute.sym_int64(), cute.sym_int64()),
            stride_order=(1, 0),
            assumed_align=4,
        ),
        make_fake(cutlass.Int32, (total_q,), stride_order=(0,), assumed_align=4),
        make_fake(OUT_DTYPE, (total_q, NUM_HEADS, L), stride_order=(2, 1, 0), assumed_align=16),
        make_fake(cutlass.Int32, (cute.sym_int64(),), stride_order=(0,), assumed_align=4),
        make_fake(cutlass.Int32, (cute.sym_int64(),), stride_order=(0,), assumed_align=4),
    )


def compile_union(spec: UnionSpecialization, device_index: int) -> object:
    spec.validate()
    key = (device_index, spec)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_index)
    compiled = cute.compile(
        _launch_fmha_union,
        *_fake_tensors(),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        spec.scale_log2,
        spec.rows_per_group,
        spec.num_chunks,
        spec.thread_sums,
        options=_base._compile_options(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


@dataclass
class PreparedUnionRunner:
    compiled: object
    args: tuple
    device_index: int
    stream: cuda_driver.CUstream | None
    spec: UnionSpecialization

    def __call__(self) -> None:
        stream = self.stream
        if stream is None:
            stream = _base._current_stream(self.device_index)
        self.compiled(*self.args, stream)


def prepare_union_runner(
    quant_q_buffer: torch.Tensor,
    kv_pool_view: torch.Tensor,
    topk_positions: torch.Tensor,
    block_table: torch.Tensor,
    req_ids: torch.Tensor,
    output: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    sequence_length: torch.Tensor,
    *,
    layer_offset: int,
    stride_factor: int,
    max_sq: int,
    rows_per_group: int,
    num_chunks: int,
    thread_sums: bool = False,
    refresh_stream: bool = False,
) -> PreparedUnionRunner:
    """Validated launcher. `topk_positions` [total_q, 2048] int32 request-local positions,
    `block_table` [requests, blocks] int32, `req_ids` [total_q] int32 (the block-table row of
    each query row; the kernel reads the sequence's first row's), `layer_offset` the pool
    layer's row term (layer * 32), `stride_factor` the pool rows of one block."""
    q_storage = _base._as_uint8_storage(quant_q_buffer, name="quant_q_buffer")
    kv_storage = _base._as_uint8_storage(kv_pool_view, name="kv_pool_view")
    total_q = int(q_storage.shape[0])
    pool_rows = kv_storage.numel() // KV_STRIDE
    batch = int(sequence_length.shape[0])
    if tuple(topk_positions.shape) != (total_q, TOPK) or topk_positions.dtype != torch.int32:
        raise ValueError("topk_positions must be [total_q, 2048] int32")
    if block_table.dtype != torch.int32 or block_table.ndim != 2 or not block_table.is_contiguous():
        raise ValueError("block_table must be a contiguous 2-D int32 tensor")
    if req_ids.dtype != torch.int32 or req_ids.numel() < total_q:
        raise ValueError("req_ids must be int32 with one entry per query row")
    # The grid's z extent is cu_q_seqlens' length - 1 and each z reads sequence_length[z].
    if int(cu_q_seqlens.numel()) != batch + 1:
        raise ValueError(f"cu_q_seqlens must have {batch + 1} entries, got {cu_q_seqlens.numel()}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous (the kernel writes it in place)")
    groups = (int(max_sq) + rows_per_group - 1) // rows_per_group
    spec = UnionSpecialization(
        rows_per_group=int(rows_per_group),
        num_chunks=int(num_chunks),
        thread_sums=bool(thread_sums),
    )
    compiled = compile_union(spec, _base._device_index(quant_q_buffer))
    stream = None if refresh_stream else _base._current_stream(_base._device_index(quant_q_buffer))
    del pool_rows
    return PreparedUnionRunner(
        compiled=compiled,
        args=(
            q_storage,
            kv_storage.reshape(-1, KV_STRIDE),
            topk_positions,
            block_table,
            req_ids,
            output.view(total_q, NUM_HEADS, L),
            cu_q_seqlens,
            sequence_length,
            int(layer_offset),
            int(stride_factor),
            int(block_table.shape[1]),
            int(groups),
        ),
        device_index=_base._device_index(quant_q_buffer),
        stream=stream,
        spec=spec,
    )
