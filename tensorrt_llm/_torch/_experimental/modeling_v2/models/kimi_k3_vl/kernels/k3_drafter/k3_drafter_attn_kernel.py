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
# Kimi K3 DSpark drafter block attention -- CTM (prims/cute) kernel: R <= 8 requests, a block of T <= 8 draft tokens
# each, GQA groups of 6 query heads per KV head, head dim 64
# =============================================================================
#
#   o[t, h] = softmax_k( scale * q[t, h] . K[k] ) @ V[k]      for every k < L = ctx + T (dense: no causal mask; the
#                                                              block attends to itself bidirectionally)
#     q      = qkv[t, 64 h : 64 h + 64] (after the q/k RMSNorm + RoPE), 48 rows per group r = 8 h + t (head-major)
#     K, V   = rows k < ctx: the drafter's paged cache (HND pages of 64 rows x 64, K and V blocks per head);
#              rows ctx + t: the block's own k, v, straight from qkv (they are not appended to the cache)
#     o      = [R T, heads * 64] bf16 (the o_proj input)
# Request r owns rows r T .. r T + T - 1 of qkv, positions and o, length ctx = ctx_len[r] and page-table row r
# (page_table[r * table_stride ...]); t above is the token within its request.
#
# One cluster of 16 CTAs per (KV head, request): grid (16 kv_heads, R). CTA c takes the 128-row tiles c, c + 16, ...
# (two pages each) of its request. One request compiles without any request indexing, so that build is the
# single-sequence kernel; the indexing sits on the step's critical path. Per tile:
#   S = Q K^T   A = Q [64 rows (48 real), 64] K-major, B = the tile's K [128 rows, 64] K-major; M 64, N 128, into TMEM
#   softmax     each row's 128 scores over the 4 lanes of a quad (tcgen05.ld 16x256b), exp2 against the running max,
#               P = bf16(p) into a 128B-swizzled K-major tile, fenced to the async proxy
#   O = P V     A = P [64, 128], B = V [128 rows, 64] read MN-major; M 64, N 64
#   fold        the CTA's running (m, l, O) per row lives in the softmax threads' registers (32 fp32 each)
# Rows >= ctx of a tile are rewritten in shared memory after its TMA lands: the block's k / v from qkv (loaded while
# the TMA is in flight), and V zeros past L (a masked score gives p = 0, and 0 x a stale non-finite V would be NaN).
# Merge: every CTA (with or without tiles) stages its (m, l, O) of the 48 rows and st.async's row r's 64 + 2 floats
# into slot [cta] of CTA r / 3's mailbox, completing that CTA's barrier by bytes (the byte count is fixed, so it is
# armed at init). CTA c then merges its 3 rows over the 16 slots in fp32 (flash-decoding combine) and stores bf16.
# Nothing is read before griddepcontrol.wait: the context lengths, the page table and qkv are this step's.
# norm_rope: qkv holds the raw projection; the kernel applies the per-head q / k RMSNorm and the (NeoX) RoPE itself,
# with fused_qk_norm_rope's arithmetic (a warp per head row, two elements per lane): warps 4-7 build the 48 Q rows in
# shared memory (128 arrivals on q_full instead of the TMA), warp 1 the block's K rows.
#
# Warps: 0 TMA, 1 block rows / tail zeros, 2 TMEM allocation + MMA, 4-7 softmax / fold / push / merge, 3 idle.
# =============================================================================
"""CTM kernel: Kimi K3 DSpark drafter block attention over the paged drafter cache, split-KV over a 16-CTA cluster
per request."""

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

HEADS = 6  # query heads per KV head (GQA group)
MAX_TOKENS = 8
ROWS = HEADS * MAX_TOKENS  # 48 query rows per group
D = 64  # head dim (one 128-byte swizzle row)
PAGE = 64
TILE = 128  # KV rows per tile = 2 pages
PAGES_PER_TILE = TILE // PAGE
CLUSTER = 16
OWN = ROWS // CLUSTER  # 3 output rows merged per CTA
MMA_M = 64
MMA_K = 16
THREADS = 256
EPI_THREADS = 128
TMEM_COLS = 256
TMEM_O = TILE  # O's 64 columns after S's 128
ELEM_BYTES = 2
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")

# Shared-memory tiles, bf16 elements.
Q_ELEMS = MMA_M * D  # the M = 64 MMA reads 16 rows past the 48 of the group
KV_ELEMS = TILE * D  # 16 KB: rows [0, 64) page 0, [64, 128) page 1
PAGE_ELEMS = PAGE * D
P_HALF_ELEMS = MMA_M * 64  # P [64 rows][128 kv] as two 128B-swizzled [64][64] halves
P_ELEMS = 2 * P_HALF_ELEMS

# Descriptor offsets in 16-byte units.
LEADING = 16
SBO = 8 * D * ELEM_BYTES  # 1024 B between 8-row swizzle atoms
STEP_K = (MMA_K * ELEM_BYTES) >> 4  # K-major: 32 B per K16 step
STEP_MN = (2 * SBO) >> 4  # MN-major: 16 rows per K16 step
P_HALF_U = (P_HALF_ELEMS * ELEM_BYTES) >> 4

# Mailbox of the 3 rows a CTA merges: [16 sources][3 rows][64] fp32 and [16][3][2] (m, l).
MAIL_O = CLUSTER * OWN * D
MAIL_ML = CLUSTER * OWN * 2
MAIL_BYTES = (MAIL_O + MAIL_ML) * 4

io_dtype = cutlass.BFloat16


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
def _try_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.try_wait.parity.acquire.cluster on a barrier of this CTA (``mbar``, its shared-memory pointer): as
    ``_test_wait_cluster``, with try_wait's bounded suspend."""
    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


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
def _st_async_v4(dst, a, b, c, d, mbar, *, loc=None, ip=None):
    """st.async of four fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip), cutlass.Float32(c).ir_value(loc=loc, ip=ip),
         cutlass.Float32(d).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,f,f,f,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _approx(op, x, *, loc=None, ip=None):
    """One MUFU op (rsqrt / ex2 / lg2 / sin / cos .approx.f32), as CUDA's rsqrtf / exp2f / __log2f / __sincosf."""
    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)], f"{op}.approx.f32 $0, $1;", "=f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _norm_rope_pair(x0, x1, w0, w1, lane, pos_f, eps, rope_c):
    """fused_qk_norm_rope on one head row held by a warp, lane l with elements (2 l, 2 l + 1): RMSNorm over the 64
    elements (butterfly sum), x * (rrms * w), then NeoX RoPE against the partner lane l ^ 16. Returns fp32."""
    ss = cutlass.Float32(0.0) + x0 * x0
    ss = ss + x1 * x1
    for mask in (16, 8, 4, 2, 1):
        ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=mask)
    r = _approx("rsqrt", ss * cutlass.Float32(1.0 / D) + eps)
    y0 = x0 * (r * w0)
    y1 = x1 * (r * w1)
    p0 = cute.arch.shuffle_sync_bfly(y0, offset=16)
    p1 = cute.arch.shuffle_sync_bfly(y1, offset=16)
    first_half = lane < cutlass.Int32(16)
    p0 = cutlass.Float32(cutlass.select_(first_half, -p0, p0))
    p1 = cutlass.Float32(cutlass.select_(first_half, -p1, p1))
    hd0 = cutlass.Float32((cutlass.Int32(2) * lane) & cutlass.Int32(31))
    hd1 = cutlass.Float32((cutlass.Int32(2) * lane + cutlass.Int32(1)) & cutlass.Int32(31))
    th0 = pos_f * _approx("ex2", hd0 * rope_c)
    th1 = pos_f * _approx("ex2", hd1 * rope_c)
    o0 = y0 * _approx("cos", th0) + p0 * _approx("sin", th0)
    o1 = y1 * _approx("cos", th1) + p1 * _approx("sin", th1)
    return o0, o1


def _plus(base, x):
    """``base + x``; ``x`` itself when ``base`` is the single-request build's Python 0 (no op in the IR, so that build
    is the single-request kernel instruction for instruction)."""
    if type(base) is int and base == 0:
        return x
    return base + x


def _swz(row, chunk):
    """Element index of 16-byte vector `chunk` (< 8) of `row` in a 128B-swizzled [rows][64] bf16 tile."""
    return row * cutlass.Int32(D) + ((chunk ^ (row % cutlass.Int32(8))) * cutlass.Int32(8))


@cute.kernel
def k3_drafter_attn_kernel(
    tma_q: cutlass.GridConstant[
        cuda.TensorMap
    ],  # qkv's q columns as (64 cols, token, head): box 64 x 8 x 6
    tma_kv: cutlass.GridConstant[
        cuda.TensorMap
    ],  # the layer's cache as (64 cols, 64 rows, K/V x head, page)
    qkv: cutlass.Array,  # [R T, (heads + 2 kv_heads) * 64] bf16, after the q/k RMSNorm + RoPE
    page_table: cutlass.Array,  # int32, row r at r * table_stride: >= ceil((ctx_len[r] + T) / 64) pages of request r
    ctx_len: cutlass.Array,  # int32 [R]: cached rows before each request's block
    out: cutlass.Array,  # [R T, heads * 64] bf16
    q_w: cutlass.Array,  # norm_rope: [64] bf16 q_norm weight
    k_w: cutlass.Array,  # norm_rope: [64] bf16 k_norm weight
    positions: cutlass.Array,  # norm_rope: int32 or int64 [R T] RoPE positions of the blocks' tokens
    num_tokens: cutlass.Int32,  # T: tokens per request
    scale_log2: cutlass.Float32,  # softmax scale * log2(e)
    norm_eps: cutlass.Float32,
    rope_base: cutlass.Float32,
    table_stride: cutlass.Int32,  # elements between page-table rows (last: the other parameters keep their offsets)
    total_heads: cutlass.Constexpr[int],  # query heads of the rank: one cluster per 6
    kv_heads: cutlass.Constexpr[int],
    norm_rope: cutlass.Constexpr[bool],
    multi_request: cutlass.Constexpr[bool],  # R > 1 (one request: request 0, no request indexing)
):
    tx, _, _ = cute.arch.thread_idx()
    warp_id = cute.arch.warp_idx()
    cta = cute.arch.block_idx_in_cluster()
    if cutlass.const_expr(multi_request):
        bid, req, _ = cute.arch.block_idx()
        tok0 = req * num_tokens  # the request's first row of qkv, positions and out
        table_row = req * table_stride
    else:
        # One request: no request arithmetic at all (Python zeros, see _plus).
        bid, _, _ = cute.arch.block_idx()
        req = tok0 = table_row = 0
    g_kv = bid // cutlass.Int32(CLUSTER)  # KV head; query heads 6 g .. 6 g + 5
    ptr_q = tma_q.get_ptr()
    ptr_kv = tma_kv.get_ptr()
    qkv_cols = (total_heads + 2 * kv_heads) * D

    smem_q = cutlass.Array(io_dtype, Q_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_k = cutlass.Array(io_dtype, KV_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_v = cutlass.Array(io_dtype, KV_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_p = cutlass.Array(io_dtype, P_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    # This CTA's final (m, l, O) of the 48 rows, staged for the push: O [48][64] fp32, (m, l) [48][2].
    stage_o = cutlass.Array(
        cutlass.Float32, ROWS * D, space=cutlass.AddressSpace.smem, alignment=16
    )
    stage_ml = cutlass.Array(
        cutlass.Float32, ROWS * 2, space=cutlass.AddressSpace.smem, alignment=16
    )
    mail_o = cutlass.Array(cutlass.Float32, MAIL_O, space=cutlass.AddressSpace.smem, alignment=16)
    mail_ml = cutlass.Array(cutlass.Float32, MAIL_ML, space=cutlass.AddressSpace.smem, alignment=16)
    q_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    kv_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # The tile with rows >= ctx lands on its own barrier (used once), so warp 1 waits for exactly that load.
    kv_fix_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    kv_fix = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    s_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    p_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    o_drained = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    if warp_id == 0:
        prims.prefetch_tensormap(ptr_q)
        prims.prefetch_tensormap(ptr_kv)
        if prims.elect_sync():
            prims.mbarrier_init(q_full, EPI_THREADS if norm_rope else 1)
            prims.mbarrier_init(kv_full, 1)
            prims.mbarrier_init(kv_fix_full, 1)
            prims.mbarrier_init(kv_fix, 32)
            prims.mbarrier_init(s_full, 1)
            prims.mbarrier_init(p_full, EPI_THREADS)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_drained, EPI_THREADS)
            # Every CTA of the cluster pushes all 48 rows: the byte count does not depend on the length.
            prims.mbarrier_init(mail_full, 1)
            prims.mbarrier_arrive_expect_tx(mail_full, MAIL_BYTES)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_base = tmem_ptr_i32.load()

    # Everything below reads this step's data (length, pages, qkv): after the grid dependency.
    prims.griddepcontrol(prims.GridDepAction.WAIT)
    ctx = ctx_len.load(idx=req)
    kv_len = ctx + num_tokens
    n_tiles = (kv_len + cutlass.Int32(TILE - 1)) // cutlass.Int32(TILE)
    n_pages = (kv_len + cutlass.Int32(PAGE - 1)) // cutlass.Int32(PAGE)
    my_count = cutlass.Int32(0)
    if cta < n_tiles:
        my_count = (n_tiles - cta + cutlass.Int32(CLUSTER - 1)) // cutlass.Int32(CLUSTER)
    # The one tile of this CTA (if any) that holds rows >= ctx: the block's rows and the zeros past L.
    fix_tile = ctx // cutlass.Int32(TILE)
    fix_k = cutlass.Int32(-1)
    if (fix_tile % cutlass.Int32(CLUSTER)) == cta:
        fix_k = fix_tile // cutlass.Int32(CLUSTER)
    # A block that straddles a tile boundary puts its tail in the next tile (another CTA).
    fix_tile2 = (kv_len - cutlass.Int32(1)) // cutlass.Int32(TILE)
    if fix_tile2 != fix_tile:
        if (fix_tile2 % cutlass.Int32(CLUSTER)) == cta:
            fix_k = fix_tile2 // cutlass.Int32(CLUSTER)

    if warp_id == 0:
        # =====================================================================
        # TMA: Q once; each tile's two pages of K and V (a page past L is not
        # loaded: warp 1 zeroes its rows).
        # =====================================================================
        if prims.elect_sync():
            if cutlass.const_expr(not norm_rope):
                if my_count > cutlass.Int32(0):
                    # The box's 8 token rows start at the request's first; rows >= T are another request's (or
                    # zeros past the last row) and only feed output rows that are not stored.
                    prims.mbarrier_arrive_expect_tx(q_full, ROWS * D * ELEM_BYTES)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_q, ptr_q, (cutlass.Int32(0), cutlass.Int32(tok0), g_kv * cutlass.Int32(HEADS)), q_full,
                    )  # fmt: skip
            for k in range(my_count):
                tile = cta + k * cutlass.Int32(CLUSTER)
                if k > 0:
                    # K / V are free once the previous tile's P V is done.
                    while not cute.arch.mbarrier_try_wait(
                        o_full.data_ptr(), (k - cutlass.Int32(1)) & cutlass.Int32(1)
                    ):
                        pass
                n_here = cutlass.Int32(0)
                for p in cutlass.range_constexpr(PAGES_PER_TILE):
                    if tile * cutlass.Int32(PAGES_PER_TILE) + cutlass.Int32(p) < n_pages:
                        n_here = n_here + cutlass.Int32(1)
                if k == fix_k:
                    prims.mbarrier_arrive_expect_tx(
                        kv_fix_full, n_here * cutlass.Int32(2 * PAGE_ELEMS * ELEM_BYTES)
                    )
                    for p in cutlass.range_constexpr(PAGES_PER_TILE):
                        page_idx = tile * cutlass.Int32(PAGES_PER_TILE) + cutlass.Int32(p)
                        if page_idx < n_pages:
                            page = page_table.load(idx=_plus(table_row, page_idx))
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_k.subview(p * PAGE_ELEMS), ptr_kv,
                                (cutlass.Int32(0), cutlass.Int32(0), g_kv, page), kv_fix_full,
                            )  # fmt: skip
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_v.subview(p * PAGE_ELEMS), ptr_kv,
                                (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(kv_heads) + g_kv, page), kv_fix_full,
                            )  # fmt: skip
                else:
                    prims.mbarrier_arrive_expect_tx(
                        kv_full, n_here * cutlass.Int32(2 * PAGE_ELEMS * ELEM_BYTES)
                    )
                    for p in cutlass.range_constexpr(PAGES_PER_TILE):
                        page_idx = tile * cutlass.Int32(PAGES_PER_TILE) + cutlass.Int32(p)
                        if page_idx < n_pages:
                            page = page_table.load(idx=_plus(table_row, page_idx))
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_k.subview(p * PAGE_ELEMS), ptr_kv,
                                (cutlass.Int32(0), cutlass.Int32(0), g_kv, page), kv_full,
                            )  # fmt: skip
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                smem_v.subview(p * PAGE_ELEMS), ptr_kv,
                                (cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(kv_heads) + g_kv, page), kv_full,
                            )  # fmt: skip
        # The o_proj after this kernel may launch and stream its weights; it waits for this grid before reading o.
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 1:
        # =====================================================================
        # Rows >= ctx of the tile that holds them: the block's k / v from qkv
        # (loaded while the tile's TMA is in flight), zeros for V past L; then
        # release the MMA warp.
        # =====================================================================
        lane = tx % 32
        if fix_k >= cutlass.Int32(0):
            tile = cta + fix_k * cutlass.Int32(CLUSTER)
            row0 = tile * cutlass.Int32(TILE)
            # Lane l: block token l / 8 (+ 4 on the second pass), 16-byte vector l % 8 of its k and v.
            c = lane % cutlass.Int32(8)
            kvecs = []
            vvecs = []
            for j in cutlass.range_constexpr(MAX_TOKENS // 4):
                t = lane // cutlass.Int32(8) + cutlass.Int32(4 * j)
                t_c = _plus(
                    tok0, cutlass.Int32(cutlass.select_(t < num_tokens, t, cutlass.Int32(0)))
                )
                kvecs.append(
                    qkv.load(
                        idx=t_c * cutlass.Int32(qkv_cols)
                        + (cutlass.Int32(total_heads) + g_kv) * cutlass.Int32(D)
                        + c * cutlass.Int32(8),
                        vector_size=8,
                        alignment=16,
                    )
                )
                vvecs.append(
                    qkv.load(
                        idx=t_c * cutlass.Int32(qkv_cols)
                        + (cutlass.Int32(total_heads + kv_heads) + g_kv) * cutlass.Int32(D)
                        + c * cutlass.Int32(8),
                        vector_size=8,
                        alignment=16,
                    )
                )
            if cutlass.const_expr(norm_rope):
                # K rows: token t's raw k (lane: elements 2 lane, 2 lane + 1), normalized and roped like Q.
                kw0 = cutlass.Float32(k_w.load(idx=cutlass.Int32(2) * lane))
                kw1 = cutlass.Float32(k_w.load(idx=cutlass.Int32(2) * lane + cutlass.Int32(1)))
                rope_c = (cutlass.Float32(-2.0) * _approx("lg2", rope_base)) * cutlass.Float32(
                    1.0 / D
                )
                kraw = []
                kpos = []
                for t in cutlass.range_constexpr(MAX_TOKENS):
                    t_c = _plus(
                        tok0,
                        cutlass.Int32(
                            cutlass.select_(
                                cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                            )
                        ),
                    )
                    kraw.append(
                        qkv.load(
                            idx=t_c * cutlass.Int32(qkv_cols)
                            + (cutlass.Int32(total_heads) + g_kv) * cutlass.Int32(D)
                            + cutlass.Int32(2) * lane,
                            vector_size=2,
                            alignment=4,
                        )
                    )
                    kpos.append(cutlass.Int32(positions.load(idx=t_c)).to(cutlass.Float32))
                krot = []
                for t in cutlass.range_constexpr(MAX_TOKENS):
                    o0, o1 = _norm_rope_pair(
                        cutlass.Float32(kraw[t][0]),
                        cutlass.Float32(kraw[t][1]),
                        kw0,
                        kw1,
                        lane,
                        kpos[t],
                        norm_eps,
                        rope_c,
                    )
                    krot.append(
                        cutlass.Vector.from_elements((o0.to(io_dtype), o1.to(io_dtype)), io_dtype)
                    )
            while not cute.arch.mbarrier_try_wait(kv_fix_full.data_ptr(), 0):
                pass
            for j in cutlass.range_constexpr(MAX_TOKENS // 4):
                t = lane // cutlass.Int32(8) + cutlass.Int32(4 * j)
                r = ctx + t - row0  # the block row's row in this tile
                if (t < num_tokens) & (r >= cutlass.Int32(0)) & (r < cutlass.Int32(TILE)):
                    if cutlass.const_expr(not norm_rope):
                        smem_k.store(kvecs[j], idx=_swz(r, c), vector_size=8, alignment=16)
                    smem_v.store(vvecs[j], idx=_swz(r, c), vector_size=8, alignment=16)
            if cutlass.const_expr(norm_rope):
                for t in cutlass.range_constexpr(MAX_TOKENS):
                    r = ctx + cutlass.Int32(t) - row0
                    if (
                        (cutlass.Int32(t) < num_tokens)
                        & (r >= cutlass.Int32(0))
                        & (r < cutlass.Int32(TILE))
                    ):
                        smem_k.store(
                            krot[t],
                            idx=_swz(r, lane // cutlass.Int32(4))
                            + (cutlass.Int32(2) * lane) % cutlass.Int32(8),
                            vector_size=2,
                            alignment=4,
                        )
            # V rows >= L of the tile are zero: a masked score gives p = 0, and 0 x a stale non-finite V is NaN.
            # (K rows >= L only feed masked scores.)
            first_zero = kv_len - row0
            zeros = []
            for e in cutlass.range_constexpr(8):
                zeros.append(cutlass.BFloat16(0.0))
            zero_v = cutlass.Vector.from_elements(tuple(zeros), io_dtype)
            for j in cutlass.range_constexpr(TILE * 8 // 32):
                vi = lane + cutlass.Int32(32 * j)
                r = vi // cutlass.Int32(8)
                if r >= first_zero:
                    smem_v.store(
                        zero_v, idx=_swz(r, vi % cutlass.Int32(8)), vector_size=8, alignment=16
                    )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.mbarrier_arrive(kv_fix)
    elif warp_id == 2:
        # =====================================================================
        # MMA: S = Q K^T (N 128), then O = P V (N 64), per tile.
        # =====================================================================
        idesc_s = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=TILE, m_dim=MMA_M
        )
        idesc_o = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=io_dtype,
            b_dtype=io_dtype,
            n_dim=D,
            m_dim=MMA_M,
            b_major=1,
        )
        swz = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
        desc_q = prims.Tcgen05SmemDesc.build(
            start_address=smem_q, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        desc_k = prims.Tcgen05SmemDesc.build(
            start_address=smem_k, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        desc_p = prims.Tcgen05SmemDesc.build(
            start_address=smem_p, leading_byte_offset=LEADING, stride_byte_offset=SBO, layout=swz
        )
        desc_v = prims.Tcgen05SmemDesc.build(
            start_address=smem_v,
            leading_byte_offset=KV_ELEMS * ELEM_BYTES,
            stride_byte_offset=SBO,
            layout=swz,
        )
        tmem_s = cutlass.inttoptr(tmem_base, 6, cutlass.Int32)
        tmem_o = cutlass.inttoptr(tmem_base + cutlass.Int32(TMEM_O), 6, cutlass.Int32)
        if my_count > cutlass.Int32(0):
            while not cute.arch.mbarrier_try_wait(q_full.data_ptr(), 0):
                pass
        for k in range(my_count):
            phase = k & cutlass.Int32(1)
            if k == fix_k:
                # Warp 1 saw the load land and rewrote the rows >= ctx.
                while not cute.arch.mbarrier_try_wait(kv_fix.data_ptr(), 0):
                    pass
            else:
                # kv_full completes once per tile other than the fix tile.
                seen_fix = cutlass.Int32(
                    cutlass.select_(
                        (fix_k >= cutlass.Int32(0)) & (fix_k < k),
                        cutlass.Int32(1),
                        cutlass.Int32(0),
                    )
                )
                while not cute.arch.mbarrier_try_wait(
                    kv_full.data_ptr(), (k - seen_fix) & cutlass.Int32(1)
                ):
                    pass
            if k > 0:
                # S's columns are free once the softmax warps have read the previous S (they wrote its P).
                while not cute.arch.mbarrier_try_wait(p_full.data_ptr(), phase ^ cutlass.Int32(1)):
                    pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kk in cutlass.range_constexpr(D // MMA_K):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_s,
                        desc_q + kk * STEP_K, desc_k + kk * STEP_K, idesc_s, kk != 0,
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(s_full)
            while not cute.arch.mbarrier_try_wait(p_full.data_ptr(), phase):
                pass
            if k > 0:
                # O's columns are free once the softmax warps have folded the previous O.
                while not cute.arch.mbarrier_try_wait(
                    o_drained.data_ptr(), phase ^ cutlass.Int32(1)
                ):
                    pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kk in cutlass.range_constexpr(TILE // MMA_K):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_o,
                        desc_p + ((kk // (64 // MMA_K)) * P_HALF_U + (kk % (64 // MMA_K)) * STEP_K),
                        desc_v + kk * STEP_MN, idesc_o, kk != 0,
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(o_full)
    elif warp_id >= 4:
        # =====================================================================
        # Softmax, fold, stage, push; then merge this CTA's 3 rows.
        #   16x256b: lane l of warp w holds rows 16 w + l / 4 and + 8, columns
        #   8 g + 2 (l % 4) + {0, 1} of every 8-column group g.
        # =====================================================================
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        quad = lane % cutlass.Int32(4)
        r0 = w * cutlass.Int32(16) + lane // cutlass.Int32(4)
        r1 = r0 + cutlass.Int32(8)
        if cutlass.const_expr(norm_rope):
            if my_count > cutlass.Int32(0):
                # Q rows 8 h + t: warp w takes (h, t) pairs w, w + 4, ...; lane l elements 2 l, 2 l + 1 of the row.
                qw0 = cutlass.Float32(q_w.load(idx=cutlass.Int32(2) * lane))
                qw1 = cutlass.Float32(q_w.load(idx=cutlass.Int32(2) * lane + cutlass.Int32(1)))
                rope_c = (cutlass.Float32(-2.0) * _approx("lg2", rope_base)) * cutlass.Float32(
                    1.0 / D
                )
                qraw = []
                qpos = []
                for i in cutlass.range_constexpr(ROWS // 4):
                    row = w + cutlass.Int32(4 * i)
                    h = row // cutlass.Int32(MAX_TOKENS)
                    t = row % cutlass.Int32(MAX_TOKENS)
                    t_c = _plus(
                        tok0, cutlass.Int32(cutlass.select_(t < num_tokens, t, cutlass.Int32(0)))
                    )
                    qraw.append(
                        qkv.load(
                            idx=t_c * cutlass.Int32(qkv_cols)
                            + (g_kv * cutlass.Int32(HEADS) + h) * cutlass.Int32(D)
                            + cutlass.Int32(2) * lane,
                            vector_size=2,
                            alignment=4,
                        )
                    )
                    qpos.append(cutlass.Int32(positions.load(idx=t_c)).to(cutlass.Float32))
                for i in cutlass.range_constexpr(ROWS // 4):
                    row = w + cutlass.Int32(4 * i)
                    o0, o1 = _norm_rope_pair(
                        cutlass.Float32(qraw[i][0]),
                        cutlass.Float32(qraw[i][1]),
                        qw0,
                        qw1,
                        lane,
                        qpos[i],
                        norm_eps,
                        rope_c,
                    )
                    smem_q.store(
                        cutlass.Vector.from_elements((o0.to(io_dtype), o1.to(io_dtype)), io_dtype),
                        idx=_swz(row, lane // cutlass.Int32(4)) + (cutlass.Int32(2) * lane) % cutlass.Int32(8),
                        vector_size=2, alignment=4,
                    )  # fmt: skip
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.mbarrier_arrive(q_full)
        m_run0 = cutlass.Float32(NEG_INF)
        m_run1 = cutlass.Float32(NEG_INF)
        l_run0 = cutlass.Float32(0.0)
        l_run1 = cutlass.Float32(0.0)
        o_run = [cutlass.Float32(0.0)] * (2 * D // 4)  # rows r0, r1 x 16 columns: [4 g + 2 row + e]
        for k in range(my_count):
            phase = k & cutlass.Int32(1)
            tile = cta + k * cutlass.Int32(CLUSTER)
            kv0 = tile * cutlass.Int32(TILE)
            while not cute.arch.mbarrier_try_wait(s_full.data_ptr(), phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            s = prims.tcgen05_ld(
                "16x256b", cutlass.inttoptr(tmem_base, 6, cutlass.Float32), num=TILE // 8
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            m0 = cutlass.Float32(NEG_INF)
            m1 = cutlass.Float32(NEG_INF)
            v0 = []
            v1 = []
            for g in cutlass.range_constexpr(TILE // 8):
                for e in cutlass.range_constexpr(2):
                    col = kv0 + cutlass.Int32(8 * g + e) + cutlass.Int32(2) * quad
                    a = cutlass.Float32(
                        cutlass.select_(
                            col < kv_len,
                            cutlass.Float32(s[4 * g + e]) * scale_log2,
                            cutlass.Float32(NEG_INF),
                        )
                    )
                    b = cutlass.Float32(
                        cutlass.select_(
                            col < kv_len,
                            cutlass.Float32(s[4 * g + 2 + e]) * scale_log2,
                            cutlass.Float32(NEG_INF),
                        )
                    )
                    v0.append(a)
                    v1.append(b)
                    m0 = cute.arch.fmax(m0, a)
                    m1 = cute.arch.fmax(m1, b)
            for offset in (1, 2):
                m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, offset=offset))
                m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, offset=offset))
            m_new0 = cute.arch.fmax(m_run0, m0)
            m_new1 = cute.arch.fmax(m_run1, m1)
            base0 = cutlass.Float32(
                cutlass.select_(m_new0 == cutlass.Float32(NEG_INF), cutlass.Float32(0.0), m_new0)
            )
            base1 = cutlass.Float32(
                cutlass.select_(m_new1 == cutlass.Float32(NEG_INF), cutlass.Float32(0.0), m_new1)
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
                    + _swz(r0, cutlass.Int32(chunk))
                    + cutlass.Int32(2) * quad
                )
                off1 = (
                    cutlass.Int32(half * P_HALF_ELEMS)
                    + _swz(r1, cutlass.Int32(chunk))
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
            # The running (m, l, O) against the new max: alpha = exp2(m_run - m_new) (0 while nothing was seen).
            alpha0 = cute.math.exp2(m_run0 - base0, fastmath=True)
            alpha1 = cute.math.exp2(m_run1 - base1, fastmath=True)
            l_run0 = l_run0 * alpha0 + l0
            l_run1 = l_run1 * alpha1 + l1
            m_run0 = m_new0
            m_run1 = m_new1
            while not cute.arch.mbarrier_try_wait(o_full.data_ptr(), phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            o = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(tmem_base + cutlass.Int32(TMEM_O), 6, cutlass.Float32),
                num=D // 8,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(o_drained)
            for g in cutlass.range_constexpr(D // 8):
                for e in cutlass.range_constexpr(2):
                    o_run[4 * g + e] = o_run[4 * g + e] * alpha0 + cutlass.Float32(o[4 * g + e])
                    o_run[4 * g + 2 + e] = o_run[4 * g + 2 + e] * alpha1 + cutlass.Float32(
                        o[4 * g + 2 + e]
                    )
        # Stage the 48 rows (rows 48-63 of the M = 64 accumulator are padding).
        for g in cutlass.range_constexpr(D // 8):
            col = cutlass.Int32(8 * g) + cutlass.Int32(2) * quad
            if r0 < cutlass.Int32(ROWS):
                stage_o.store(
                    cutlass.Vector.from_elements((o_run[4 * g], o_run[4 * g + 1]), cutlass.Float32),
                    idx=r0 * cutlass.Int32(D) + col, vector_size=2, alignment=8,
                )  # fmt: skip
            if r1 < cutlass.Int32(ROWS):
                stage_o.store(
                    cutlass.Vector.from_elements((o_run[4 * g + 2], o_run[4 * g + 3]), cutlass.Float32),
                    idx=r1 * cutlass.Int32(D) + col, vector_size=2, alignment=8,
                )  # fmt: skip
        if quad == cutlass.Int32(0):
            if r0 < cutlass.Int32(ROWS):
                stage_ml.store(
                    cutlass.Vector.from_elements((m_run0, l_run0), cutlass.Float32),
                    idx=r0 * cutlass.Int32(2),
                    vector_size=2,
                    alignment=8,
                )
            if r1 < cutlass.Int32(ROWS):
                stage_ml.store(
                    cutlass.Vector.from_elements((m_run1, l_run1), cutlass.Float32),
                    idx=r1 * cutlass.Int32(2),
                    vector_size=2,
                    alignment=8,
                )
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        # Push: 48 rows x 16 vectors of 4 fp32 into slot [cta] of the owner's mailbox (row r -> CTA r / 3, row r % 3).
        for i in cutlass.range_constexpr(ROWS * (D // 4) // EPI_THREADS):
            vi = tid + cutlass.Int32(i * EPI_THREADS)
            row = vi // cutlass.Int32(D // 4)
            c4 = vi % cutlass.Int32(D // 4)
            own = row // cutlass.Int32(OWN)
            vals = stage_o.load(
                idx=row * cutlass.Int32(D) + c4 * cutlass.Int32(4), vector_size=4, alignment=16
            )
            _st_async_v4(
                _mapa_u32(
                    mail_o.data_ptr(
                        (cta * cutlass.Int32(OWN) + row % cutlass.Int32(OWN)) * cutlass.Int32(D)
                        + c4 * cutlass.Int32(4)
                    ),
                    own,
                ),
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                _mapa_u32(mail_full.data_ptr(), own),
            )
        if tid < cutlass.Int32(ROWS):
            own = tid // cutlass.Int32(OWN)
            ml = stage_ml.load(idx=tid * cutlass.Int32(2), vector_size=2, alignment=8)
            _st_async_v2(
                _mapa_u32(
                    mail_ml.data_ptr(
                        (cta * cutlass.Int32(OWN) + tid % cutlass.Int32(OWN)) * cutlass.Int32(2)
                    ),
                    own,
                ),
                ml[0],
                ml[1],
                _mapa_u32(mail_full.data_ptr(), own),
            )
        # =====================================================================
        # Merge rows 3 cta .. 3 cta + 2 over the 16 slots (fp32), store bf16.
        # =====================================================================
        while not _try_wait_cluster(mail_full.data_ptr(), 0):
            pass
        if tid < cutlass.Int32(OWN * D // 2):
            j = tid // cutlass.Int32(D // 2)
            c2 = (tid % cutlass.Int32(D // 2)) * cutlass.Int32(2)
            row = cta * cutlass.Int32(OWN) + j
            ms = []
            for s_ in cutlass.range_constexpr(CLUSTER):
                ms.append(mail_ml.load(idx=(cutlass.Int32(s_ * OWN) + j) * cutlass.Int32(2)))
            mx = ms[0]
            for s_ in cutlass.range_constexpr(1, CLUSTER):
                mx = cute.arch.fmax(mx, ms[s_])
            den = cutlass.Float32(0.0)
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            for s_ in cutlass.range_constexpr(CLUSTER):
                wgt = cutlass.Float32(
                    cutlass.select_(
                        ms[s_] == cutlass.Float32(NEG_INF),
                        cutlass.Float32(0.0),
                        cute.math.exp2(ms[s_] - mx, fastmath=True),
                    )
                )
                den = den + wgt * mail_ml.load(
                    idx=(cutlass.Int32(s_ * OWN) + j) * cutlass.Int32(2) + cutlass.Int32(1)
                )
                ov = mail_o.load(
                    idx=(cutlass.Int32(s_ * OWN) + j) * cutlass.Int32(D) + c2,
                    vector_size=2,
                    alignment=8,
                )
                acc0 = acc0 + wgt * ov[0]
                acc1 = acc1 + wgt * ov[1]
            inv = cutlass.Float32(1.0) / den
            h_loc = row // cutlass.Int32(MAX_TOKENS)
            t_out = row % cutlass.Int32(MAX_TOKENS)
            if t_out < num_tokens:
                out.store(
                    cutlass.Vector.from_elements(
                        ((acc0 * inv).to(io_dtype), (acc1 * inv).to(io_dtype)), io_dtype
                    ),
                    idx=(
                        _plus(tok0, t_out) * cutlass.Int32(total_heads)
                        + g_kv * cutlass.Int32(HEADS)
                        + h_loc
                    )
                    * cutlass.Int32(D)
                    + c2,
                    vector_size=2,
                    alignment=4,
                )
    # Every TMEM reader has waited for its loads; no peer writes this CTA's mailbox after its barrier completed.
    prims.barrier_cta_sync(0)
    if warp_id == 2:
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), TMEM_COLS)


def _q_map(qkv, num_rows, total_heads, kv_heads):
    """qkv's query columns as (64 columns, token, head): one call at (token r T, head 6 g) lands the group's
    [6 heads][8 tokens][64] of request r (row 8 h + t; rows past the last token zero)."""
    qkv_cols = (total_heads + 2 * kv_heads) * D
    return cuda.create_tensor_map_tiled(
        global_address=qkv.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[D, num_rows, total_heads],
        global_strides=[(qkv_cols * ELEM_BYTES) // 16, (D * ELEM_BYTES) // 16],
        box_dims=[D, MAX_TOKENS, HEADS],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _kv_map(cache, total_pages, kv_heads, page_stride):
    """The layer's HND cache [pages, 2, kv_heads, 64, 64] (page stride `page_stride` elements) as (64 columns, 64 rows,
    K / V x head, page): one call lands one head's K or V page."""
    return cuda.create_tensor_map_tiled(
        global_address=cache.iterator.toint(),
        dtype=cutlass.BFloat16,
        global_dims=[D, PAGE, 2 * kv_heads, total_pages],
        global_strides=[
            (D * ELEM_BYTES) // 16,
            (PAGE * D * ELEM_BYTES) // 16,
            (page_stride * ELEM_BYTES) // 16,
        ],
        box_dims=[D, PAGE, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_drafter_attn(
    qkv: cute.Tensor,  # [R T * (heads + 2 kv_heads) * 64] bf16
    cache: cute.Tensor,  # the layer's cache, pointer carrier (first page)
    page_table: cute.Tensor,  # int32: R rows of pages, table_stride apart
    ctx_len: cute.Tensor,  # int32 [R]
    out: cute.Tensor,  # [R T * heads * 64] bf16
    q_w: cute.Tensor,  # norm_rope: [64] bf16
    k_w: cute.Tensor,  # norm_rope: [64] bf16
    positions: cute.Tensor,  # norm_rope: int32 or int64 [R T]
    num_tokens: cutlass.Int32,  # T: tokens per request
    num_requests: cutlass.Int32,  # R
    table_stride: cutlass.Int32,
    scale_log2: cutlass.Float32,
    total_pages: cutlass.Int32,
    norm_eps: cutlass.Float32,
    rope_base: cutlass.Float32,
    total_heads: cutlass.Constexpr[int],
    kv_heads: cutlass.Constexpr[int],
    page_stride: cutlass.Constexpr[int],
    norm_rope: cutlass.Constexpr[bool],
    multi_request: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    tma_q = _q_map(qkv, num_tokens * num_requests, total_heads, kv_heads)
    tma_kv = _kv_map(cache, total_pages, kv_heads, page_stride)
    k3_drafter_attn_kernel(
        tma_q,
        tma_kv,
        qkv,
        page_table,
        ctx_len,
        out,
        q_w,
        k_w,
        positions,
        num_tokens,
        scale_log2,
        norm_eps,
        rope_base,
        table_stride,
        total_heads,
        kv_heads,
        norm_rope,
        multi_request,
    ).launch(  # fmt: skip
        grid=(CLUSTER * kv_heads, num_requests, 1),
        block=(THREADS, 1, 1),
        cluster=(CLUSTER, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def softmax_scale_log2(head_dim: int = D) -> float:
    return LOG2E / math.sqrt(head_dim)
