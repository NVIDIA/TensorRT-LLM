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
# Kimi K3 MLA decode attention -- CTM (prims/cute) kernel: R requests of T <= 8 query tokens, 6 heads per rank
# =============================================================================
#
#   o[i, t, h] = softmax_k( scale * q[i, t, h] . kv_i[k] ) @ kv_i[k, :512]      k <= L_i - T + t (bottom-right causal)
#     q    = fused_q [M = R T, 6, 576], request-major (request i's tokens are rows i T .. i T + T - 1); per cluster 48
#            query rows r = 8 h + t, head-major (a head's tokens are adjacent rows)
#     kv_i = request i's pages of the paged latent cache, 64 rows x 576 (512 latent | 64 rope), L_i rows
#     o    = [M, 6, 512] bf16
#
# One cluster of 16 CTAs per (request, group of 6 heads): grid (16 x groups, R), one group at TP16 (a rank holding more
# heads gets one cluster per group, each reading the request's whole cache). A cluster's work does not depend on R; the
# request only selects its page-table row, length, query / output rows and workspace slots. CTA c takes the 128-row KV
# tiles c, c + 16, ... (two pages each).
# Per tile:
#   S = Q K^T    A = Q [64 rows (48 real), 576] from shared memory, B = the tile's K [128 rows, 576] (chunk-major:
#                page p's rows of chunk j at [j][64 p..]), M 64, N 128 (Q read once per chunk), into TMEM (an M = 64
#                accumulator: row 16 w + l in lane 32 w + l)
#   softmax      each row's 128 values over the 4 lanes of a quad (tcgen05.ld 16x256b): masked max, exp2, sum,
#                P = bf16(p) into a 128B-swizzled K-major tile, fenced to the async proxy
#   O = P V      A = P [64, 128], B = V = the tile's latent columns read MN-major (N 256 per MMA), M 64
#   partial      (m, l) and O / l (a convex combination of V rows: fp16 keeps 2^-11) of the tile's 48 rows into the
#                CTA's global workspace slot; a CTA's later tiles (L > 2048) fold into its slot:
#                O / l' = (O / l) (l a / l') + O_k (b / l'), l' = l a + l_k b, a, b = exp2(m - m'), exp2(m_k - m')
# Right after its last softmax a CTA st.async's its (m, l) per row into every CTA's mailbox; warps 1 and 3 turn the
# mailbox into the merge weights exp2(m_s - M) / sum_s exp2(m_s - M) l_s while O is computed and drained. Then a
# cluster barrier (release / acquire), and CTA c merges latent columns [32 c, 32 c + 32) of all rows over the slots
# (flash-decoding combine in fp32, one round of 16-byte loads) and stores o in bf16. The KV pages below L - T are
# TMA'd before griddepcontrol.wait (earlier steps wrote them); Q (3 chunk groups, so S starts on the first) and the
# last T rows' pages after it. O = P V runs in two N = 256 halves, the first drained while the second computes.
# Q is TMA'd as 8 token rows from the request's first token: rows t >= T (the next request's tokens, or zeros past M)
# are computed and written to the workspace slot like the live rows (so the merge's 16-byte loads read 32-byte sectors
# written whole; it weights these rows 0), and never stored.
#
# fuse_vb: the merge is split by (head, 4 tokens) over CTAs 0-11 instead of by columns, and each of those CTAs applies
# v_b to its 4 merged rows: W_vb[h] (128 x 512) TMA'd into the dead KV buffer after the CTA's last P V, o rounded to
# bf16 (the unfused path's attention output) into the dead P buffer as the B operand, y^T = W_vb[h] o^T (M 128, N 8,
# K 512) in TMEM, y [M, heads * 128] bf16 stored (the o_proj input). apply_gate: y is multiplied by the output gate's
# sigmoid (bf16, from the fused projection's gate columns) with the unfused path's rounding, bf16(bf16(y) * s).
#
# no_cluster (R x groups > CLUSTER_WAVE: more 16-CTA clusters than co-reside, which would run a second wave): the grid
# launches without a cluster, CTA c = block x mod 16, and the cluster's mailbox and barriers go through the workspace's
# tail instead. At its last tile a CTA stores its live rows' (m, l) into its slot of an fp32 exchange and arrives on its
# (request, head group)'s first counter; warps 1 and 3 wait for the launch's 16 arrivals and read the slots' (m, l)
# from there. After the drain every CTA arrives on the second counter, which CTAs 0-11 wait for instead of the cluster
# barrier. Counters only grow: a CTA reads its counter after the grid wait and before it arrives, so the launch's 16
# arrivals end at (count & ~15) + 16. The merge sums the same values in the same order, so the outputs do not depend on
# the mode. All 16 R groups CTAs must co-reside (1 per SM), as their waits are spins.
#
# Warps: 0 TMA, 2 TMEM allocation (512 columns) + MMA, 4-7 softmax / partial writer, 1 and 3 merge weights, all 8 in
# the merge.
# =============================================================================
"""CTM kernel: Kimi K3 MLA decode attention over the paged latent cache, split-KV over a 16-CTA cluster."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

HEADS = 6
MAX_TOKENS = 8  # query tokens per request (T)
MAX_REQUESTS = 8  # requests per launch (R): the workspace holds R x groups x 16 slots
ROWS = HEADS * MAX_TOKENS  # 48 query rows
# A softmax thread's rows r and r + 8 lie in one 16-row block, so they are both real or both past ROWS.
assert ROWS % 16 == 0
QK = 576
LATENT = 512
PAGE = 64
TILE = 128  # KV rows per tile = 2 pages
PAGES_PER_TILE = TILE // PAGE
CHUNK = 64  # 128-byte swizzle width in bf16
QK_CHUNKS = QK // CHUNK  # 9
LAT_CHUNKS = LATENT // CHUNK  # 8
CLUSTER = 16
MMA_M = 64
MMA_K = 16
THREADS = 256
EPI_THREADS = 128
TMEM_COLS = 512
ELEM_BYTES = 2
LAT_PER_CTA = LATENT // CLUSTER  # 32 merged columns per CTA
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")

# Shared-memory tiles, bf16 elements.
Q_CHUNK_ELEMS = ROWS * CHUNK  # 3072 (6 KB): Q chunk j = rows 0..47 of columns [64 j, 64 j + 64)
Q_ELEMS = QK_CHUNKS * Q_CHUNK_ELEMS
KV_CHUNK_ELEMS = (
    TILE * CHUNK
)  # 8192 (16 KB): both pages' rows of one 64-column chunk (chunk-major tile)
KV_PAGE_OFF = PAGE * CHUNK  # page p's 64 rows start 8 KB into each chunk block
KV_ELEMS = QK_CHUNKS * KV_CHUNK_ELEMS
P_HALF_ELEMS = MMA_M * CHUNK  # 4096: 64 rows x 64 KV columns
P_ELEMS = PAGES_PER_TILE * P_HALF_ELEMS
Q_OVERREAD_ELEMS = (MMA_M - ROWS) * CHUNK  # the M = 64 MMA reads 16 rows past the last Q chunk

# Descriptor offsets in 16-byte units.
LEADING = 16
SBO = 8 * CHUNK * ELEM_BYTES  # 1024 B between 8-row swizzle atoms
STEP_K = (MMA_K * ELEM_BYTES) >> 4  # K-major: 32 B per K16 step
STEP_MN = (2 * SBO) >> 4  # MN-major: 2 x SBO per K16 step (16 KV rows)
Q_CHUNK_U = (Q_CHUNK_ELEMS * ELEM_BYTES) >> 4
KV_CHUNK_U = (KV_CHUNK_ELEMS * ELEM_BYTES) >> 4
P_HALF_U = (P_HALF_ELEMS * ELEM_BYTES) >> 4

Q_GROUPS = 3  # Q TMA'd as 3 groups of 3 chunks, each on its own barrier
Q_GROUP_CHUNKS = QK_CHUNKS // Q_GROUPS
O_HALVES = LATENT // 256
MERGE_HALF = CLUSTER // 2  # slots per thread of a merge pair
V_DIM = 128  # v_head_dim
VB_CTAS = (
    HEADS * 2
)  # fuse_vb: CTA c < 12 merges head c // 2, tokens 4 (c % 2) .. + 4, and applies v_b to them
VB_TOKENS = MAX_TOKENS // 2
VB_ELEMS = V_DIM * LATENT

io_dtype = cutlass.BFloat16
ws_dtype = cutlass.Float16  # the partials O / l (convex combinations of V rows, |x| <= max |V|)
# Workspace: slot (request x groups + head group) x 16 + CTA. Slot layout, fragment-major: [64 groups of 8 columns][64
# rows (the M = 64 accumulator)][8], so one warp's 16x256b fragment of a group (8 rows x 4 lanes x 2 values) is 128
# contiguous bytes, and a row's 8 columns are 16.
WS_GROUP_ELEMS = MMA_M * 8
WS_SLOT_ELEMS = (LATENT // 8) * WS_GROUP_ELEMS
# 16-CTA clusters of this kernel that co-reside on GB200 (cuOccupancyMaxActiveClusters at its shared memory: the eighth
# GPC holds 12 free SMs); a launch of more clusters takes the no_cluster mode.
CLUSTER_WAVE = 7
# no_cluster's tail of the workspace, after the R x groups x 16 partial slots: an fp32 [slot][row][m, l] exchange, then
# two int32 counters per (request, head group), CTR_STRIDE words apart (a 128-byte line each).
CTR_STRIDE = 32


def ws_sync_elems(groups: int) -> int:
    """fp16 elements of the workspace's no_cluster tail (zeroed once: the counters start at 0)."""
    exchange = MAX_REQUESTS * groups * CLUSTER * ROWS * 2 * 4
    counters = MAX_REQUESTS * groups * 2 * CTR_STRIDE * 4
    return (exchange + counters) // 2


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
def _st_async_v2(dst, a, b, mbar, *, loc=None, ip=None):
    """st.async of two fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 8 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [$0], {$1, $2}, [$3];", "r,f,f,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _red_add_release(addr, value, *, loc=None, ip=None):
    """red.release.gpu.global.add.s32 on a global address (an arrival: no value back)."""
    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ld_acquire(addr, *, loc=None, ip=None):
    """ld.acquire.gpu.global.s32 of a global address."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _try_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.try_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): whether
    phase ``parity`` has completed, acquiring at cluster scope. The barrier is completed by other CTAs' st.async, whose
    complete_tx releases at cluster scope."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@cute.kernel
def k3_mla_attn_kernel(
    tma_q: cutlass.GridConstant[
        cuda.TensorMap
    ],  # fused_q [M, heads, 576]: box 64 cols x 8 tokens x 6 heads x 3 chunks
    tma_kv: cutlass.GridConstant[
        cuda.TensorMap
    ],  # pool as [pages * 64 rows, 576]: box 64 cols x 64 rows x 9 chunks
    tma_vb: cutlass.GridConstant[
        cuda.TensorMap
    ],  # fuse_vb: v_b_proj as [heads * 128 rows, 512]: box 64 x 128 x 8
    page_table: cutlass.Array,  # int32: request i's pages (>= ceil(L_i / 64)) at [i * pt_stride, ...)
    seq_len: cutlass.Array,  # int32 [R]: L_i, request i's KV rows including its T new ones
    ws_o: cutlass.Array,  # fp16 [R, head groups, 16 CTAs, 64 column groups, 64 rows, 8] per-CTA partial O / l
    out: cutlass.Array,  # bf16 [M * heads * 512] (fuse_vb: [M * heads * 128])
    gate: cutlass.Array,  # apply_gate: bf16 [M, gate_ld], sigmoid(g) of head h at columns gate_col0 + 128 h
    tokens: cutlass.Int32,  # T, the query tokens of each request (M = R T)
    pt_stride: cutlass.Int32,  # page-table elements between consecutive requests' rows
    scale_log2: cutlass.Float32,  # softmax scale * log2(e)
    page_offset: cutlass.Int32,  # added to every page-table entry (the layer's slot in an interleaved pool)
    gate_col0: cutlass.Int32,
    gate_ld: cutlass.Int32,
    total_heads: cutlass.Constexpr[int],  # heads of the rank: one cluster per 6
    fuse_vb: cutlass.Constexpr[bool],
    apply_gate: cutlass.Constexpr[bool],
    no_cluster: cutlass.Constexpr[bool],
):
    tx, _, _ = cute.arch.thread_idx()
    warp_id = cute.arch.warp_idx()
    if cutlass.const_expr(not no_cluster):
        cta = cute.arch.block_idx_in_cluster()
    bid, req, _ = cute.arch.block_idx()
    if cutlass.const_expr(no_cluster):
        cta = bid % cutlass.Int32(CLUSTER)
    hg = bid // cutlass.Int32(CLUSTER)  # head group: heads 6 hg .. 6 hg + 5
    ws_slot0 = (req * cutlass.Int32(total_heads // HEADS) + hg) * cutlass.Int32(CLUSTER)
    my_slot = ws_slot0 + cta
    tok0 = req * tokens  # the request's first row of q, out and gate
    pt0 = req * pt_stride  # the request's page-table row
    ptr_q = tma_q.get_ptr()
    ptr_kv = tma_kv.get_ptr()
    ptr_vb = tma_vb.get_ptr()

    # Read before the grid dependency: the host writes the lengths and page table before the step.
    kv_len = seq_len.load(idx=req)
    n_tiles = (kv_len + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
    n_pages = (kv_len + cutlass.Int32(PAGE - 1)) // cutlass.Int32(PAGE)
    my_count = cutlass.Int32(0)
    if cta < n_tiles:
        my_count = (n_tiles - cta + cutlass.Int32(CLUSTER - 1)) // cutlass.Int32(CLUSTER)
    q_rows = tokens * cutlass.Int32(HEADS)  # live rows: 8 h + t with t < T
    # Pages holding a row >= L - T are written this step (the cache append): they load after the wait.
    fresh_page = (kv_len - tokens) // cutlass.Int32(PAGE)
    # Slots (CTAs) that had a tile: min(n_tiles, 16).
    n_valid = cutlass.Int32(
        cutlass.select_(n_tiles < cutlass.Int32(CLUSTER), n_tiles, cutlass.Int32(CLUSTER))
    )

    smem_kv = cutlass.Array(io_dtype, KV_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_q = cutlass.Array(io_dtype, Q_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    # Right after Q so the M = 64 MMA's read past the last Q chunk stays in shared memory.
    smem_p = cutlass.Array(io_dtype, P_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    kv_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    q_full = cutlass.Array(cutlass.Int64, Q_GROUPS, space=cutlass.AddressSpace.smem, alignment=8)
    s_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    p_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    o_full = cutlass.Array(cutlass.Int64, O_HALVES, space=cutlass.AddressSpace.smem, alignment=8)
    o_drained = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    ml_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    vb_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    vb_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Mailbox: [source CTA][row] (m, l) of every CTA with a tile (st.async, completed by bytes); warps 1 and 3 then
    # overwrite m with the merge weight exp2(m_s - M) / L.
    ml_mail = cutlass.Array(
        cutlass.Float32, CLUSTER * ROWS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    if cutlass.const_expr(no_cluster):
        # The workspace's tail: the (m, l) exchange [slot][row][2] and the counters; this CTA's targets in sync_target.
        groups = total_heads // HEADS
        tail = MAX_REQUESTS * groups * CLUSTER * WS_SLOT_ELEMS
        ml_glob = cutlass.Array(
            ws_o.data_ptr(tail), shape=(MAX_REQUESTS * groups * CLUSTER * ROWS * 2,), dtype=cutlass.Float32,
            bounds_check=False, addrspace=cutlass.AddressSpace.gmem.value, alignment=16,
        )  # fmt: skip
        ctrs = cutlass.Array(
            ws_o.data_ptr(tail + MAX_REQUESTS * groups * CLUSTER * ROWS * 4),
            shape=(MAX_REQUESTS * groups * 2 * CTR_STRIDE,), dtype=cutlass.Int32, bounds_check=False,
            addrspace=cutlass.AddressSpace.gmem.value, alignment=16,
        )  # fmt: skip
        ctr_ml = (req * cutlass.Int32(groups) + hg) * cutlass.Int32(2 * CTR_STRIDE)
        ctr_o = ctr_ml + cutlass.Int32(CTR_STRIDE)
        sync_target = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)

    if warp_id == 0:
        prims.prefetch_tensormap(ptr_q)
        prims.prefetch_tensormap(ptr_kv)
        if cutlass.const_expr(fuse_vb):
            prims.prefetch_tensormap(ptr_vb)
        if prims.elect_sync():
            prims.mbarrier_init(kv_full, 1)
            for i in cutlass.range_constexpr(Q_GROUPS):
                prims.mbarrier_init(q_full.subview(i), 1)
            prims.mbarrier_init(s_full, 1)
            prims.mbarrier_init(p_full, EPI_THREADS)
            for i in cutlass.range_constexpr(O_HALVES):
                prims.mbarrier_init(o_full.subview(i), 1)
            prims.mbarrier_init(o_drained, EPI_THREADS)
            prims.mbarrier_init(ml_full, 1)
            prims.mbarrier_init(vb_full, 1)
            prims.mbarrier_init(vb_done, 1)
            if cutlass.const_expr(not no_cluster):
                # Every CTA with a tile sends (m, l) of the M * 6 live rows (8 bytes each).
                prims.mbarrier_arrive_expect_tx(ml_full, n_valid * q_rows * cutlass.Int32(8))
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    if cutlass.const_expr(not no_cluster):
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_base = tmem_ptr_i32.load()

    if warp_id == 0:
        # =====================================================================
        # TMA: each tile's two pages (the pages written earlier before the grid
        # wait), Q after it; the next tile once the previous one's O is done.
        # =====================================================================
        if prims.elect_sync():
            for k in range(my_count):
                tile = cta + k * cutlass.Int32(CLUSTER)
                if k > 0:
                    # The KV buffer is free once the previous tile's P V is done (its last half's commit).
                    while not cute.arch.mbarrier_try_wait(
                        o_full.subview(O_HALVES - 1).data_ptr(),
                        (k - cutlass.Int32(1)) & cutlass.Int32(1),
                    ):
                        pass
                prims.mbarrier_arrive_expect_tx(kv_full, KV_ELEMS * ELEM_BYTES)
                for p in cutlass.range_constexpr(PAGES_PER_TILE):
                    page_idx = tile * cutlass.Int32(PAGES_PER_TILE) + cutlass.Int32(p)
                    page_c = cutlass.Int32(
                        cutlass.select_(page_idx < n_pages, page_idx, n_pages - cutlass.Int32(1))
                    )
                    row0 = (page_table.load(idx=pt0 + page_c) + page_offset) * cutlass.Int32(PAGE)
                    if k == 0:
                        if page_c >= fresh_page:
                            prims.griddepcontrol(prims.GridDepAction.WAIT)
                    # Chunk-major tile: page p's rows of chunk j at [j][64 p .. 64 p + 64), so one MMA covers both
                    # pages.
                    for j in cutlass.range_constexpr(QK_CHUNKS):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_kv.subview(j * KV_CHUNK_ELEMS + p * KV_PAGE_OFF),
                            ptr_kv,
                            (cutlass.Int32(0), row0, cutlass.Int32(j)),
                            kv_full,
                        )
                if k == 0:
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    for g in cutlass.range_constexpr(Q_GROUPS):
                        prims.mbarrier_arrive_expect_tx(
                            q_full.subview(g), Q_GROUP_CHUNKS * Q_CHUNK_ELEMS * ELEM_BYTES
                        )
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            smem_q.subview(g * Q_GROUP_CHUNKS * Q_CHUNK_ELEMS),
                            ptr_q,
                            (
                                cutlass.Int32(0),
                                tok0,
                                hg * cutlass.Int32(HEADS),
                                cutlass.Int32(g * Q_GROUP_CHUNKS),
                            ),
                            q_full.subview(g),
                        )
        # Dependents (v_b / o_proj) wait for this whole grid before reading its output.
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if cutlass.const_expr(no_cluster):
            if tx == cutlass.Int32(0):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                sync_target.store(ctrs.load(idx=ctr_o, is_volatile=True), idx=1)
        if cutlass.const_expr(fuse_vb):
            if cta < cutlass.Int32(VB_CTAS):
                # W_vb[h] into the KV buffer once the CTA's last P V has read it.
                if prims.elect_sync():
                    if my_count > cutlass.Int32(0):
                        while not cute.arch.mbarrier_try_wait(
                            o_full.subview(O_HALVES - 1).data_ptr(),
                            (my_count - cutlass.Int32(1)) & cutlass.Int32(1),
                        ):
                            pass
                    prims.mbarrier_arrive_expect_tx(vb_full, VB_ELEMS * ELEM_BYTES)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_kv,
                        ptr_vb,
                        (
                            cutlass.Int32(0),
                            (hg * cutlass.Int32(HEADS) + cta // cutlass.Int32(2))
                            * cutlass.Int32(V_DIM),
                            cutlass.Int32(0),
                        ),
                        vb_full,
                    )
    elif warp_id == 2:
        # =====================================================================
        # MMA: S = Q K^T (per page N 64), then O = P V (N 256 x 2), per tile.
        # =====================================================================
        idesc_s = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=TILE, m_dim=MMA_M
        )
        idesc_o = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=io_dtype,
            b_dtype=io_dtype,
            n_dim=256,
            m_dim=MMA_M,
            b_major=1,
        )
        swz = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
        desc_q = prims.Tcgen05SmemDesc.build(
            start_address=smem_q, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        desc_k = prims.Tcgen05SmemDesc.build(
            start_address=smem_kv, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        desc_p = prims.Tcgen05SmemDesc.build(
            start_address=smem_p, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        # V read MN-major: N = latent across 64-column chunks KV_CHUNK_ELEMS apart (leading byte offset).
        desc_v = prims.Tcgen05SmemDesc.build(
            start_address=smem_kv,
            leading_byte_offset=KV_CHUNK_ELEMS * ELEM_BYTES,
            stride_byte_offset=SBO,
            layout=swz,
        )
        tmem_s = cutlass.inttoptr(tmem_base, 6, cutlass.Int32)
        for k in range(my_count):
            phase = k & cutlass.Int32(1)
            if k > 0:
                # S overwrites O's first columns: the previous tile's O has been read out.
                while not cute.arch.mbarrier_try_wait(
                    o_drained.data_ptr(), phase ^ cutlass.Int32(1)
                ):
                    pass
            while not cute.arch.mbarrier_try_wait(kv_full.data_ptr(), phase):
                pass
            # S per Q chunk group (each group's barrier completes once; later tiles pass it at once).
            for g in cutlass.range_constexpr(Q_GROUPS):
                while not cute.arch.mbarrier_try_wait(q_full.subview(g).data_ptr(), 0):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for jj in cutlass.range_constexpr(Q_GROUP_CHUNKS):
                    j = g * Q_GROUP_CHUNKS + jj
                    for kk in cutlass.range_constexpr(CHUNK // MMA_K):
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_s,
                                desc_q + (j * Q_CHUNK_U + kk * STEP_K),
                                desc_k + (j * KV_CHUNK_U + kk * STEP_K),
                                idesc_s, not (j == 0 and kk == 0),
                            )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(s_full)
            while not cute.arch.mbarrier_try_wait(p_full.data_ptr(), phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # O in two N = 256 halves, each committed on its own barrier (the first is drained during the second).
            for nh in cutlass.range_constexpr(O_HALVES):
                for kk in cutlass.range_constexpr(TILE // MMA_K):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_base + cutlass.Int32(nh * 256), 6, cutlass.Int32),
                            desc_p + ((kk // (CHUNK // MMA_K)) * P_HALF_U + (kk % (CHUNK // MMA_K)) * STEP_K),
                            desc_v + (nh * 4 * KV_CHUNK_U + kk * STEP_MN),
                            idesc_o, kk != 0,
                        )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_commit(o_full.subview(nh))
    elif warp_id >= 4:
        # =====================================================================
        # Softmax and the partial (m, l, O) -> this CTA's workspace slot.
        #   16x256b: lane l of warp w holds rows 16 w + l / 4 and + 8, columns
        #   8 g + 2 (l % 4) + {0, 1} of every 8-column group g.
        # =====================================================================
        lane = tx % 32
        w = warp_id - 4
        quad = lane % cutlass.Int32(4)
        r0 = w * cutlass.Int32(16) + lane // cutlass.Int32(4)
        r1 = r0 + cutlass.Int32(8)
        # Last visible KV row of each query row (token t = r % 8): L - T + t.
        lim0 = kv_len - tokens + r0 % cutlass.Int32(MAX_TOKENS)
        lim1 = kv_len - tokens + r1 % cutlass.Int32(MAX_TOKENS)
        # The slot takes every real row's partial (rows t >= T too), the mailboxes the live rows' (m, l).
        real0 = r0 < cutlass.Int32(ROWS)
        real1 = r1 < cutlass.Int32(ROWS)
        live0 = real0 & (r0 % cutlass.Int32(MAX_TOKENS) < tokens)
        live1 = real1 & (r1 % cutlass.Int32(MAX_TOKENS) < tokens)
        # The CTA's running (m, l) per row over its tiles (the slot's O is relative to m).
        m_run0 = cutlass.Float32(NEG_INF)
        m_run1 = cutlass.Float32(NEG_INF)
        l_run0 = cutlass.Float32(0.0)
        l_run1 = cutlass.Float32(0.0)
        for k in range(my_count):
            phase = k & cutlass.Int32(1)
            tile = cta + k * cutlass.Int32(CLUSTER)
            kv0 = tile * cutlass.Int32(TILE)
            while not cute.arch.mbarrier_try_wait(s_full.data_ptr(), phase):
                pass
            if cutlass.const_expr(no_cluster):
                count_ml = ctrs.load(idx=ctr_ml, is_volatile=True)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            s = prims.tcgen05_ld(
                "16x256b", cutlass.inttoptr(tmem_base, 6, cutlass.Float32), num=TILE // 8
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            # Masked, scaled scores and their row maxima (the quad's 4 lanes share the two rows).
            m0 = cutlass.Float32(NEG_INF)
            m1 = cutlass.Float32(NEG_INF)
            v0 = []
            v1 = []
            for g in cutlass.range_constexpr(TILE // 8):
                for e in cutlass.range_constexpr(2):
                    col = kv0 + cutlass.Int32(8 * g + e) + cutlass.Int32(2) * quad
                    a = cutlass.Float32(s[4 * g + e]) * scale_log2
                    b = cutlass.Float32(s[4 * g + 2 + e]) * scale_log2
                    a = cutlass.Float32(cutlass.select_(col <= lim0, a, cutlass.Float32(NEG_INF)))
                    b = cutlass.Float32(cutlass.select_(col <= lim1, b, cutlass.Float32(NEG_INF)))
                    v0.append(a)
                    v1.append(b)
                    m0 = cute.arch.fmax(m0, a)
                    m1 = cute.arch.fmax(m1, b)
            for offset in (1, 2):
                m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=offset))
                m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=offset))
            # A row with nothing visible in this tile: m = -inf, p = 0 (the merge weights it by 0).
            base0 = cutlass.Float32(
                cutlass.select_(m0 == cutlass.Float32(NEG_INF), cutlass.Float32(0.0), m0)
            )
            base1 = cutlass.Float32(
                cutlass.select_(m1 == cutlass.Float32(NEG_INF), cutlass.Float32(0.0), m1)
            )
            l0 = cutlass.Float32(0.0)
            l1 = cutlass.Float32(0.0)
            for g in cutlass.range_constexpr(TILE // 8):
                pv0 = []
                pv1 = []
                for e in cutlass.range_constexpr(2):
                    pa = cute.math.exp2(v0[2 * g + e] - base0, fastmath=True)
                    pb = cute.math.exp2(v1[2 * g + e] - base1, fastmath=True)
                    l0 = l0 + pa
                    l1 = l1 + pb
                    pv0.append(pa)
                    pv1.append(pb)
                # P [row][kv] bf16, K-major 128B swizzle: half g // 8 (= page), chunk g % 8, elements 2 quad, +1.
                half = g // 8
                chunk = g % 8
                pair0 = cutlass.Vector.from_elements(
                    (pv0[0].to(io_dtype), pv0[1].to(io_dtype)), io_dtype
                )
                pair1 = cutlass.Vector.from_elements(
                    (pv1[0].to(io_dtype), pv1[1].to(io_dtype)), io_dtype
                )
                off0 = (
                    cutlass.Int32(half * P_HALF_ELEMS)
                    + r0 * cutlass.Int32(CHUNK)
                    + ((cutlass.Int32(chunk) ^ (r0 % cutlass.Int32(8))) * cutlass.Int32(8))
                    + cutlass.Int32(2) * quad
                )
                off1 = (
                    cutlass.Int32(half * P_HALF_ELEMS)
                    + r1 * cutlass.Int32(CHUNK)
                    + ((cutlass.Int32(chunk) ^ (r1 % cutlass.Int32(8))) * cutlass.Int32(8))
                    + cutlass.Int32(2) * quad
                )
                smem_p.store(pair0, idx=off0, vector_size=2, alignment=4)
                smem_p.store(pair1, idx=off1, vector_size=2, alignment=4)
            for offset in (1, 2):
                l0 = l0 + cute.arch.shuffle_sync_bfly(l0, offset=offset)
                l1 = l1 + cute.arch.shuffle_sync_bfly(l1, offset=offset)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.mbarrier_arrive(p_full)
            # Fold scales (k > 0; the slot's m is finite then: an earlier tile of a CTA is fully visible).
            m_new0 = cute.arch.fmax(m_run0, m0)
            m_new1 = cute.arch.fmax(m_run1, m1)
            fa0 = cute.math.exp2(m_run0 - m_new0, fastmath=True)
            fa1 = cute.math.exp2(m_run1 - m_new1, fastmath=True)
            fb0 = cute.math.exp2(m0 - m_new0, fastmath=True)
            fb1 = cute.math.exp2(m1 - m_new1, fastmath=True)
            first = k == cutlass.Int32(0)
            l_new0 = cutlass.Float32(cutlass.select_(first, l0, l_run0 * fa0 + l0 * fb0))
            l_new1 = cutlass.Float32(cutlass.select_(first, l1, l_run1 * fa1 + l1 * fb1))
            # The slot holds O / l (rows with nothing visible: 0): scales of the old slot (ca) and of this tile (cb).
            inv0 = cutlass.Float32(
                cutlass.select_(
                    l_new0 > cutlass.Float32(0.0),
                    cutlass.Float32(1.0) / l_new0,
                    cutlass.Float32(0.0),
                )
            )
            inv1 = cutlass.Float32(
                cutlass.select_(
                    l_new1 > cutlass.Float32(0.0),
                    cutlass.Float32(1.0) / l_new1,
                    cutlass.Float32(0.0),
                )
            )
            ca0 = cutlass.Float32(cutlass.select_(first, cutlass.Float32(0.0), l_run0 * fa0 * inv0))
            ca1 = cutlass.Float32(cutlass.select_(first, cutlass.Float32(0.0), l_run1 * fa1 * inv1))
            cb0 = cutlass.Float32(cutlass.select_(first, inv0, fb0 * inv0))
            cb1 = cutlass.Float32(cutlass.select_(first, inv1, fb1 * inv1))
            l_run0 = l_new0
            l_run1 = l_new1
            m_run0 = cutlass.Float32(cutlass.select_(first, m0, m_new0))
            m_run1 = cutlass.Float32(cutlass.select_(first, m1, m_new1))
            if k == my_count - cutlass.Int32(1):
                if cutlass.const_expr(no_cluster):
                    # The CTA's final (m, l) of its live rows into its slot of the exchange, then its arrival.
                    if quad == cutlass.Int32(0):
                        if live0:
                            ml_glob.store(
                                cutlass.Vector.from_elements((m_run0, l_run0), cutlass.Float32),
                                idx=(my_slot * cutlass.Int32(ROWS) + r0) * cutlass.Int32(2), vector_size=2,
                                alignment=8,
                            )  # fmt: skip
                        if live1:
                            ml_glob.store(
                                cutlass.Vector.from_elements((m_run1, l_run1), cutlass.Float32),
                                idx=(my_slot * cutlass.Int32(ROWS) + r1) * cutlass.Int32(2), vector_size=2,
                                alignment=8,
                            )  # fmt: skip
                    prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                    if tx == cutlass.Int32(4 * 32):
                        _red_add_release(ctrs.data_ptr(ctr_ml).toint(), 1)
                        sync_target.store(
                            (count_ml & cutlass.Int32(-16)) + cutlass.Int32(16), idx=0
                        )
                        prims.mbarrier_arrive(ml_full)
                else:
                    # The CTA's final (m, l) of its live rows into slot `cta` of every CTA's mailbox (this one too).
                    if quad == cutlass.Int32(0):
                        for dst in cutlass.range_constexpr(CLUSTER):
                            mb = _mapa_u32(ml_full.data_ptr(), dst)
                            if live0:
                                _st_async_v2(
                                    _mapa_u32(
                                        ml_mail.data_ptr((cta * cutlass.Int32(ROWS) + r0) * cutlass.Int32(2)), dst
                                    ),
                                    m_run0, l_run0, mb,
                                )  # fmt: skip
                            if live1:
                                _st_async_v2(
                                    _mapa_u32(
                                        ml_mail.data_ptr((cta * cutlass.Int32(ROWS) + r1) * cutlass.Int32(2)), dst
                                    ),
                                    m_run1, l_run1, mb,
                                )  # fmt: skip
            base_o0 = (
                my_slot * cutlass.Int32(WS_SLOT_ELEMS)
                + r0 * cutlass.Int32(8)
                + cutlass.Int32(2) * quad
            )
            base_o1 = (
                my_slot * cutlass.Int32(WS_SLOT_ELEMS)
                + r1 * cutlass.Int32(8)
                + cutlass.Int32(2) * quad
            )
            # O / l -> ws_o[slot, row, :] in fp16 (folded into it for k > 0), 128 columns per load, each N = 256 half
            # as soon as its MMAs are done.
            for part in cutlass.range_constexpr(LATENT // 128):
                if cutlass.const_expr(part % 2 == 0):
                    while not cute.arch.mbarrier_try_wait(
                        o_full.subview(part // 2).data_ptr(), phase
                    ):
                        pass
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                o = prims.tcgen05_ld(
                    "16x256b",
                    cutlass.inttoptr(tmem_base + cutlass.Int32(part * 128), 6, cutlass.Float32),
                    num=16,
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if first:
                    for g in cutlass.range_constexpr(16):
                        col = (part * 16 + g) * WS_GROUP_ELEMS
                        if real0:
                            ws_o.store(
                                cutlass.Vector.from_elements(
                                    ((cutlass.Float32(o[4 * g]) * cb0).to(ws_dtype),
                                     (cutlass.Float32(o[4 * g + 1]) * cb0).to(ws_dtype)),
                                    ws_dtype,
                                ),
                                idx=base_o0 + cutlass.Int32(col), vector_size=2, alignment=4,
                            )  # fmt: skip
                        if real1:
                            ws_o.store(
                                cutlass.Vector.from_elements(
                                    ((cutlass.Float32(o[4 * g + 2]) * cb1).to(ws_dtype),
                                     (cutlass.Float32(o[4 * g + 3]) * cb1).to(ws_dtype)),
                                    ws_dtype,
                                ),
                                idx=base_o1 + cutlass.Int32(col), vector_size=2, alignment=4,
                            )  # fmt: skip
                elif real0:
                    # This thread wrote these slot entries itself (program order): read, rescale, add, write back.
                    # Rows past ROWS are never written, so they are not read either; a thread's two rows are both
                    # real or both past ROWS (ROWS % 16 == 0), so warp 7 skips this whole branch.
                    old0 = []
                    old1 = []
                    for g in cutlass.range_constexpr(16):
                        col = (part * 16 + g) * WS_GROUP_ELEMS
                        old0.append(
                            ws_o.load(idx=base_o0 + cutlass.Int32(col), vector_size=2, alignment=4)
                        )
                        old1.append(
                            ws_o.load(idx=base_o1 + cutlass.Int32(col), vector_size=2, alignment=4)
                        )
                    for g in cutlass.range_constexpr(16):
                        col = (part * 16 + g) * WS_GROUP_ELEMS
                        ws_o.store(
                            cutlass.Vector.from_elements(
                                (
                                    (cutlass.Float32(old0[g][0]) * ca0
                                     + cutlass.Float32(o[4 * g]) * cb0).to(ws_dtype),
                                    (cutlass.Float32(old0[g][1]) * ca0
                                     + cutlass.Float32(o[4 * g + 1]) * cb0).to(ws_dtype),
                                ),
                                ws_dtype,
                            ),
                            idx=base_o0 + cutlass.Int32(col), vector_size=2, alignment=4,
                        )  # fmt: skip
                        ws_o.store(
                            cutlass.Vector.from_elements(
                                (
                                    (cutlass.Float32(old1[g][0]) * ca1
                                     + cutlass.Float32(o[4 * g + 2]) * cb1).to(ws_dtype),
                                    (cutlass.Float32(old1[g][1]) * ca1
                                     + cutlass.Float32(o[4 * g + 3]) * cb1).to(ws_dtype),
                                ),
                                ws_dtype,
                            ),
                            idx=base_o1 + cutlass.Int32(col), vector_size=2, alignment=4,
                        )  # fmt: skip
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(o_drained)
        if cutlass.const_expr(no_cluster):
            if my_count == cutlass.Int32(0):
                if tx == cutlass.Int32(4 * 32):
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    count_ml0 = ctrs.load(idx=ctr_ml, is_volatile=True)
                    _red_add_release(ctrs.data_ptr(ctr_ml).toint(), 1)
                    sync_target.store((count_ml0 & cutlass.Int32(-16)) + cutlass.Int32(16), idx=0)
                    prims.mbarrier_arrive(ml_full)
    else:
        # =====================================================================
        # Warps 1 and 3: merge weights from the (m, l) mailbox while O runs.
        # =====================================================================
        t1 = (warp_id // cutlass.Int32(2)) * cutlass.Int32(32) + tx % cutlass.Int32(32)
        if cutlass.const_expr(no_cluster):
            # This CTA's own arrival; the other CTAs' (m, l) are acquired through the counter.
            while not cute.arch.mbarrier_try_wait(ml_full.data_ptr(), 0):
                pass
            target = sync_target.load(idx=0)
            while _ld_acquire(ctrs.data_ptr(ctr_ml).toint()) - target < cutlass.Int32(0):
                pass
        else:
            # Completed by the cluster's st.async: acquire at cluster scope.
            while not _try_wait_cluster(ml_full.data_ptr(), 0):
                pass
        # A quad of lanes per row (16 rows per pass, 3 passes), lane j of the quad over slots j, j + 4, j + 8, j + 12:
        # M = max_s m_s, weight_s = exp2(m_s - M) l_s / sum_s exp2(m_s - M) l_s (the slots hold O / l), over m_s.
        sub = t1 % cutlass.Int32(4)
        for ps in cutlass.range_constexpr(ROWS // 16):
            row = cutlass.Int32(ps * 16) + t1 // cutlass.Int32(4)
            live = (row < cutlass.Int32(ROWS)) & (row % cutlass.Int32(MAX_TOKENS) < tokens)
            row_c = cutlass.Int32(cutlass.select_(live, row, cutlass.Int32(0)))
            ms = []
            ls = []
            for i in cutlass.range_constexpr(CLUSTER // 4):
                sl = sub + cutlass.Int32(4 * i)
                ok = sl < n_valid
                e = (
                    cutlass.Int32(cutlass.select_(ok, sl, cutlass.Int32(0))) * cutlass.Int32(ROWS)
                    + row_c
                ) * cutlass.Int32(2)
                # Only a live lane's own slot entries are read as m: the m words are overwritten with the
                # weights below by the lane that owns them. A lane past n_valid, or of a dead row, reads (and
                # drops) an l word instead, which nothing writes here.
                e_m = cutlass.Int32(cutlass.select_(ok & live, e, e + cutlass.Int32(1)))
                if cutlass.const_expr(no_cluster):
                    eg = ws_slot0 * cutlass.Int32(ROWS * 2) + e
                    ms.append(
                        cutlass.Float32(
                            cutlass.select_(
                                ok,
                                ml_glob.load(idx=ws_slot0 * cutlass.Int32(ROWS * 2) + e_m),
                                cutlass.Float32(NEG_INF),
                            )
                        )
                    )
                    ls.append(
                        cutlass.Float32(
                            cutlass.select_(
                                ok, ml_glob.load(idx=eg + cutlass.Int32(1)), cutlass.Float32(0.0)
                            )
                        )
                    )
                else:
                    ms.append(
                        cutlass.Float32(
                            cutlass.select_(ok, ml_mail.load(idx=e_m), cutlass.Float32(NEG_INF))
                        )
                    )
                    ls.append(
                        cutlass.Float32(
                            cutlass.select_(
                                ok, ml_mail.load(idx=e + cutlass.Int32(1)), cutlass.Float32(0.0)
                            )
                        )
                    )
            mx = ms[0]
            for i in cutlass.range_constexpr(1, CLUSTER // 4):
                mx = cute.arch.fmax(mx, ms[i])
            for offset in (1, 2):
                mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, offset=offset))
            bs = [cute.math.exp2(ms[i] - mx, fastmath=True) for i in range(CLUSTER // 4)]
            den = bs[0] * ls[0]
            for i in cutlass.range_constexpr(1, CLUSTER // 4):
                den = den + bs[i] * ls[i]
            for offset in (1, 2):
                den = den + cute.arch.shuffle_sync_bfly(den, offset=offset)
            inv = cutlass.Float32(1.0) / den
            if live:
                for i in cutlass.range_constexpr(CLUSTER // 4):
                    sl = sub + cutlass.Int32(4 * i)
                    if sl < n_valid:
                        ml_mail.store(
                            bs[i] * ls[i] * inv,
                            idx=(sl * cutlass.Int32(ROWS) + row) * cutlass.Int32(2),
                        )

    # =========================================================================
    # Every tile's partial is in the workspace (release / acquire over the
    # cluster, all threads; no_cluster: the second counter); CTA c merges
    # latent columns [32 c, 32 c + 32).
    # =========================================================================
    if cutlass.const_expr(no_cluster):
        if cutlass.const_expr(fuse_vb):
            merging = VB_CTAS
        else:
            merging = CLUSTER
        prims.barrier_cta_sync(0)
        if tx == cutlass.Int32(0):
            _red_add_release(ctrs.data_ptr(ctr_o).toint(), 1)
            if cta < cutlass.Int32(merging):
                target_o = (sync_target.load(idx=1) & cutlass.Int32(-16)) + cutlass.Int32(16)
                while _ld_acquire(ctrs.data_ptr(ctr_o).toint()) - target_o < cutlass.Int32(0):
                    pass
        prims.barrier_cta_sync(0)
    else:
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
    if cutlass.const_expr(fuse_vb):
        if cta < cutlass.Int32(VB_CTAS):
            # Rows 8 h + t of tokens t = 4 th + j (j < 4; adjacent rows), all 512 columns: thread -> row j = tx % 4,
            # columns 8 (tx / 4) .. + 8, so a warp reads 8 column groups x 4 adjacent 16-byte rows; the slots in two
            # batches of 8 (16 loads of 16 bytes in flight per batch).
            h_loc = cta // cutlass.Int32(2)
            j_row = tx % cutlass.Int32(VB_TOKENS)
            t_tok = (cta % cutlass.Int32(2)) * cutlass.Int32(VB_TOKENS) + j_row
            live = t_tok < tokens
            # A token t >= T reads its own row too (weight 0); its column of the v_b product is never stored.
            row = h_loc * cutlass.Int32(MAX_TOKENS) + t_tok
            c8 = (tx // cutlass.Int32(VB_TOKENS)) * cutlass.Int32(8)
            acc = [cutlass.Float32(0.0)] * 8
            for half in cutlass.range_constexpr(2):
                vals = []
                wts = []
                for j in cutlass.range_constexpr(MERGE_HALF):
                    sl = cutlass.Int32(half * MERGE_HALF + j)
                    ok = sl < n_valid
                    sl_c = cutlass.Int32(cutlass.select_(ok, sl, cutlass.Int32(0)))
                    base = (
                        (ws_slot0 + sl_c) * cutlass.Int32(WS_SLOT_ELEMS)
                        + (c8 // cutlass.Int32(8)) * cutlass.Int32(WS_GROUP_ELEMS)
                        + row * cutlass.Int32(8)
                    )
                    vals.append(ws_o.load(idx=base, vector_size=8, alignment=16))
                    wts.append(
                        cutlass.Float32(
                            cutlass.select_(
                                ok & live,
                                ml_mail.load(
                                    idx=(sl_c * cutlass.Int32(ROWS) + row) * cutlass.Int32(2)
                                ),
                                cutlass.Float32(0.0),
                            )
                        )
                    )
                for j in cutlass.range_constexpr(MERGE_HALF):
                    for e in cutlass.range_constexpr(8):
                        acc[e] = acc[e] + wts[j] * cutlass.Float32(vals[j][e])
            # bf16 o (the unfused path's attention output) -> the v_b B tile in the dead P buffer (token j, columns
            # c8 .. c8 + 8), K-major 128B swizzle: chunk c8 / 64, 16-byte vector ((c8 % 64) / 8) ^ j.
            smem_p.store(
                cutlass.Vector.from_elements(tuple(v.to(io_dtype) for v in acc), io_dtype),
                idx=(c8 // cutlass.Int32(CHUNK)) * cutlass.Int32(8 * CHUNK) + j_row * cutlass.Int32(CHUNK)
                + (((c8 % cutlass.Int32(CHUNK)) // cutlass.Int32(8)) ^ j_row) * cutlass.Int32(8),
                vector_size=8,
                alignment=16,
            )  # fmt: skip
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)
        if cta < cutlass.Int32(VB_CTAS):
            if warp_id == 2:
                # y^T [128 v, 8 tokens] = W_vb[h] (M 128, K 512 from the KV buffer) x o^T (the B tile), fp32 in TMEM.
                idesc_vb = prims.Tcgen05InstrDesc.build(
                    c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=8, m_dim=128
                )
                swz = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
                desc_w = prims.Tcgen05SmemDesc.build(
                    start_address=smem_kv,
                    leading_byte_offset=LEADING,
                    stride_byte_offset=SBO,
                    layout=swz,
                )
                desc_o = prims.Tcgen05SmemDesc.build(
                    start_address=smem_p,
                    leading_byte_offset=LEADING,
                    stride_byte_offset=SBO,
                    layout=swz,
                )
                while not cute.arch.mbarrier_try_wait(vb_full.data_ptr(), 0):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                for jc in cutlass.range_constexpr(LAT_CHUNKS):
                    for kk in cutlass.range_constexpr(CHUNK // MMA_K):
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1,
                                cutlass.inttoptr(tmem_base, 6, cutlass.Int32),
                                desc_w + (jc * ((V_DIM * CHUNK * ELEM_BYTES) >> 4) + kk * STEP_K),
                                desc_o + (jc * ((8 * CHUNK * ELEM_BYTES) >> 4) + kk * STEP_K),
                                idesc_vb, not (jc == 0 and kk == 0),
                            )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_commit(vb_done)
            elif warp_id >= 4:
                # y[t, h, v]: lane v = 32 (warp - 4) + lane of the accumulator, tokens in its 8 columns.
                v_idx = (warp_id - cutlass.Int32(4)) * cutlass.Int32(32) + tx % cutlass.Int32(32)
                h_glob = hg * cutlass.Int32(HEADS) + cta // cutlass.Int32(2)
                # The gate of this thread's tokens, loaded before the v_b wait so its latency hides under the MMA (a
                # token past T reads the request's first row and is not used).
                gate_pre = []
                if cutlass.const_expr(apply_gate):
                    for j in cutlass.range_constexpr(VB_TOKENS):
                        t_pre = (cta % cutlass.Int32(2)) * cutlass.Int32(VB_TOKENS) + cutlass.Int32(
                            j
                        )
                        t_row = tok0 + cutlass.Int32(
                            cutlass.select_(t_pre < tokens, t_pre, cutlass.Int32(0))
                        )
                        gate_pre.append(
                            cutlass.Float32(
                                gate.load(
                                    idx=t_row * gate_ld
                                    + gate_col0
                                    + h_glob * cutlass.Int32(V_DIM)
                                    + v_idx
                                )
                            )
                        )
                while not cute.arch.mbarrier_try_wait(vb_done.data_ptr(), 0):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                yv = prims.tcgen05_ld(
                    "32x32b", cutlass.inttoptr(tmem_base, 6, cutlass.Float32), num=8
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                for j in cutlass.range_constexpr(VB_TOKENS):
                    t_out = (cta % cutlass.Int32(2)) * cutlass.Int32(VB_TOKENS) + cutlass.Int32(j)
                    if t_out < tokens:
                        y_out = cutlass.Float32(yv[j]).to(io_dtype)
                        if cutlass.const_expr(apply_gate):
                            y_out = (cutlass.Float32(y_out) * gate_pre[j]).to(io_dtype)
                        out.store(
                            y_out,
                            idx=((tok0 + t_out) * cutlass.Int32(total_heads) + h_glob)
                            * cutlass.Int32(V_DIM)
                            + v_idx,
                        )
        # Every TMEM reader of this CTA has waited for its loads.
        prims.barrier_cta_sync(0)
        if warp_id == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), TMEM_COLS)
    else:
        if warp_id == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), TMEM_COLS)
        # 4 groups of 8 columns x 48 rows = 192 items (item = 48 group + row, so consecutive pairs read consecutive
        # 16-byte rows of a group); thread pair (2 p, 2 p + 1) takes items p and p + 128 (< 192), the even thread over
        # slots [0, 8), the odd one over [8, 16): all 16 of a thread's workspace loads (8 fp16 each) are in flight
        # together, then the pair's halves are added by one shuffle. Slots >= n_valid load slot 0 with weight 0, rows
        # of tokens t >= T their own row with weight 0 (never stored).
        hs = tx % cutlass.Int32(2)
        pair = tx // cutlass.Int32(2)
        vals = []
        wts = []
        for i in cutlass.range_constexpr(2):
            item = pair + cutlass.Int32(128 * i)
            item_c = cutlass.Int32(cutlass.select_(item < cutlass.Int32(ROWS * 4), item, pair))
            row = item_c % cutlass.Int32(ROWS)
            live = (row % cutlass.Int32(MAX_TOKENS) < tokens) & (item < cutlass.Int32(ROWS * 4))
            grp = cta * cutlass.Int32(LAT_PER_CTA // 8) + item_c // cutlass.Int32(ROWS)
            for j in cutlass.range_constexpr(MERGE_HALF):
                sl = hs * cutlass.Int32(MERGE_HALF) + cutlass.Int32(j)
                ok = sl < n_valid
                sl_c = cutlass.Int32(cutlass.select_(ok, sl, cutlass.Int32(0)))
                vals.append(
                    ws_o.load(
                        idx=(ws_slot0 + sl_c) * cutlass.Int32(WS_SLOT_ELEMS)
                        + grp * cutlass.Int32(WS_GROUP_ELEMS)
                        + row * cutlass.Int32(8),
                        vector_size=8,
                        alignment=16,
                    )
                )
                wts.append(
                    cutlass.Float32(
                        cutlass.select_(
                            ok & live,
                            ml_mail.load(idx=(sl_c * cutlass.Int32(ROWS) + row) * cutlass.Int32(2)),
                            cutlass.Float32(0.0),
                        )
                    )
                )
        for i in cutlass.range_constexpr(2):
            item = pair + cutlass.Int32(128 * i)
            item_c = cutlass.Int32(cutlass.select_(item < cutlass.Int32(ROWS * 4), item, pair))
            row = item_c % cutlass.Int32(ROWS)
            grp = cta * cutlass.Int32(LAT_PER_CTA // 8) + item_c // cutlass.Int32(ROWS)
            acc = []
            for e in cutlass.range_constexpr(8):
                a = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(MERGE_HALF):
                    a = a + wts[i * MERGE_HALF + j] * cutlass.Float32(vals[i * MERGE_HALF + j][e])
                acc.append(a + cute.arch.shuffle_sync_bfly(a, offset=1))
            out_row = (
                (tok0 + row % cutlass.Int32(MAX_TOKENS)) * cutlass.Int32(total_heads)
                + hg * cutlass.Int32(HEADS)
                + row // cutlass.Int32(MAX_TOKENS)
            )
            if hs == cutlass.Int32(0):
                if (row % cutlass.Int32(MAX_TOKENS) < tokens) & (item < cutlass.Int32(ROWS * 4)):
                    out.store(
                        cutlass.Vector.from_elements(tuple(v.to(io_dtype) for v in acc), io_dtype),
                        idx=out_row * cutlass.Int32(LATENT) + grp * cutlass.Int32(8),
                        vector_size=8,
                        alignment=16,
                    )


def _q_map(q, num_tokens, total_heads):
    """fused_q [M, heads, 576] as (64-column chunk, token, head, chunk index): one call at (token i T, head 6 g, chunk
    3 j) lands the group's [3][6 heads x 8 tokens][64] (row 8 h + t holds token i T + t; tokens >= M zero)."""
    return cuda.create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[CHUNK, num_tokens, total_heads, QK_CHUNKS],
        global_strides=[
            (total_heads * QK * ELEM_BYTES) // 16,
            (QK * ELEM_BYTES) // 16,
            (CHUNK * ELEM_BYTES) // 16,
        ],
        box_dims=[CHUNK, MAX_TOKENS, HEADS, Q_GROUP_CHUNKS],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _vb_map(w_vb, rows):
    """v_b_proj [heads * 128, 512] as (64-column chunk, row, chunk index): one call lands a head's [8][128][64]."""
    return cuda.create_tensor_map_tiled(
        global_address=w_vb.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[CHUNK, rows, LAT_CHUNKS],
        global_strides=[(LATENT * ELEM_BYTES) // 16, (CHUNK * ELEM_BYTES) // 16],
        box_dims=[CHUNK, V_DIM, LAT_CHUNKS],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _kv_map(pool, total_rows, row_stride):
    """The latent pool as [rows, 576] (row stride `row_stride` elements): one call lands a page's 64-column chunk."""
    return cuda.create_tensor_map_tiled(
        global_address=pool.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[CHUNK, total_rows, QK_CHUNKS],
        global_strides=[(row_stride * ELEM_BYTES) // 16, (CHUNK * ELEM_BYTES) // 16],
        box_dims=[CHUNK, PAGE, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_mla_attn(
    q: cute.Tensor,  # [M * heads * 576] bf16, M = R T
    pool: cute.Tensor,  # the latent pool, flat bf16: row i of page p at (p * 64 + i) * row_stride
    page_table: cute.Tensor,  # int32, request i's pages at [i * pt_stride, ...)
    seq_len: cute.Tensor,  # int32 [R]
    ws_o: cute.Tensor,  # fp16 [MAX_REQUESTS * heads / 6 * 16 * WS_SLOT_ELEMS]
    out: cute.Tensor,  # bf16 [M * heads * 512] (fuse_vb: [M * heads * 128])
    w_vb: cute.Tensor,  # fuse_vb: v_b_proj [heads * 128 * 512] bf16 (else any bf16 tensor, unused)
    gate: cute.Tensor,  # apply_gate: bf16 [M * gate_ld] (else any bf16 tensor, unused)
    tokens: cutlass.Int32,  # T <= 8
    num_requests: cutlass.Int32,  # R <= MAX_REQUESTS
    pt_stride: cutlass.Int32,
    scale_log2: cutlass.Float32,
    total_rows: cutlass.Int32,
    page_offset: cutlass.Int32,
    gate_col0: cutlass.Int32,
    gate_ld: cutlass.Int32,
    row_stride: cutlass.Constexpr[int],
    total_heads: cutlass.Constexpr[int],
    fuse_vb: cutlass.Constexpr[bool],
    apply_gate: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
    no_cluster: cutlass.Constexpr[bool] = False,
) -> None:
    tma_q = _q_map(q, tokens * num_requests, total_heads)
    tma_kv = _kv_map(pool, total_rows, row_stride)
    tma_vb = _vb_map(w_vb, total_heads * V_DIM) if cutlass.const_expr(fuse_vb) else tma_kv
    k3_mla_attn_kernel(
        tma_q,
        tma_kv,
        tma_vb,
        page_table,
        seq_len,
        ws_o,
        out,
        gate,
        tokens,
        pt_stride,
        scale_log2,
        page_offset,
        gate_col0,
        gate_ld,
        total_heads,
        fuse_vb,
        apply_gate,
        no_cluster,
    ).launch(  # fmt: skip
        grid=(CLUSTER * (total_heads // HEADS), num_requests, 1),
        block=(THREADS, 1, 1),
        cluster=None if no_cluster else (CLUSTER, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def softmax_scale_log2(
    qk_nope_head_dim: int, qk_rope_head_dim: int, q_scaling: float = 1.0
) -> float:
    return LOG2E / (math.sqrt(qk_nope_head_dim + qk_rope_head_dim) * q_scaling)
