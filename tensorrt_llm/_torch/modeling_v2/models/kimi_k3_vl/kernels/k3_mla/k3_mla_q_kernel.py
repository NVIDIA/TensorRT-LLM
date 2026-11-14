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
# Kimi K3 MLA decode query path -- CTM (prims/cute) kernel, M <= 64 tokens, any number of heads per rank (6 at TP16)
# =============================================================================
#
#   q_n[t]        = bf16(q_a[t] * rsqrt(mean(q_a[t]^2) + eps) * w_qa)          (q_a_layernorm, 1536)
#   q[t, h]       = bf16(q_n[t] @ W_qb[192 h : 192 h + 192]^T)                  (q_b_proj: 128 nope | 64 pe)
#   fused_q[t, h] = [ bf16(q[t, h, :128] @ W_kb[h]^T) (512) | q[t, h, 128:] (64) ]   (k_b absorb; K3 is NoPE)
#
# The grid's y index is a chunk of N tokens (rows N c .. N c + N - 1, the last chunk possibly shorter; N, the MMA's
# N, is the compile-time `mma_n`: 8, or 32 for the wider steps, whose chunks of 8 would need more co-resident clusters
# than fit): each chunk is the same N-token problem below, on its own rows, the weights read again per chunk (from L2
# after the first). Every token's sums keep the same order at either N, so a 32-token chunk's rows are bit-identical
# to the four 8-token chunks of the same rows. Everything below describes one chunk of 8 tokens; at N 32 every
# per-token step runs for four groups of 8 tokens (token t of group g is row 8 g + t of the chunk).
#
# One cluster of 6 CTAs per head. Rank r holds (TMA'd EVICT_FIRST before griddepcontrol.wait) the head's W_qb
# nope rows (128) and pe rows (64) over k-tiles {2 r, 2 r + 1}, and for r < 4 the k_b rows W_kb[h][128 r:128 r+128].
# After the wait the epilogue warps normalize the q_a rows (the RMS over all 1536 columns) and write this rank's 256
# columns, bf16, into a resident 128B-swizzled B tile (the layout TMA would produce), fence it to the async proxy and
# release the MMA warp (fuse_rmsnorm_qkv_rope's resident-B prolog). cluster_rms: each rank reads only its 256 columns
# and st.async's its per-token partial sums of squares to the 6 ranks, which add them in rank order. The MMA warp
# runs the nope (M 128) and pe (M 64) partial products. Split-K reduce over the cluster: nope rows [32 w, 32 w + 32)
# (TMEM lanes of epilogue warp w) belong to rank w, pe rows [32 j, 32 j + 32) (lanes 0-15 of warps 2 j, 2 j + 1: an
# M = 64 accumulator puts row 16 w + l in lane 32 w + l) to rank 4 + j; non-owners store their fp32 partials into
# slot [rank] of the owner's mailbox through DSMEM and arrive (release, cluster scope); owners add the 6 partials in
# rank order and round once. Every wait on a barrier other CTAs complete acquires at cluster scope.
# pe owners store fused_q[..., 512:]. Each nope owner writes its bf16 32 x 8 slice, as 16-byte chunks, into the k_b B
# tile of ranks 0-3 and arrives on their barrier; those ranks fence the DSMEM writes to the async proxy, run the k_b
# MMA (M 128, N 8, K 128) and store fused_q[t, h, 128 r : 128 r + 128].
# single_hop (8-token chunks only): instead, every rank st.async's its whole nope partial (128 rows x 8 tokens fp32)
# into the mailbox of each k_b rank (completing that rank's barrier by bytes); a k_b rank adds the 6 partials in rank
# order itself (the same fp32 sums, so the same bits), writes its B tile and runs the k_b MMA: one DSMEM hop instead
# of two.
#
# kv_mode (the KV half, warp 1 of CTA j < the chunk's tokens: token t = N c + j): token t's latent row
# ag[t, 1536:2048] RMS-normalized (flashinfer's order, like q_a) and its rope columns ag[t, 2048:2112] (K3 is NoPE:
# copied) form the 576-column cache row, stored (1) into the paged latent pool: the M tokens are R requests of T each
# (request-major), token t is token u = t - i T of request i = t / T, at position p = L_i - T + u (row
# (page_table[i][p / 64] + page_offset) * 64 + p % 64, 64-bit element index; not stored when p < 0), or (2) into
# kv_out [M, 576] (check builds). Page table and lengths are host-filled: read before the grid wait. The attention
# after this kernel reads the new rows after its own grid wait.
#
# Warps: 0 weight TMA (+ early dependent trigger), 1 KV half (kv_mode), 2 TMEM allocation + MMA, 4-7 norm prolog /
# reduce / epilogue, 3 idle.
# =============================================================================
"""CTM kernel for Kimi K3 MLA's decode query path: q_a RMSNorm, q_b projection, k_b absorption -> fused_q."""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

HEADS = 6  # per rank at TP16 (the kernel takes the rank's head count as a compile-time argument)
NOPE = 128
PE = 64
QK = NOPE + PE  # 192 q_b rows per head
LATENT = 512  # kv_lora_rank
Q_LORA = 1536  # q_lora_rank = q_b's K
FUSED = LATENT + PE  # 576
PAGE = 64  # latent cache rows per page
CLUSTER = 6  # CTAs per head
K_TILES = Q_LORA // 128  # 12
MY_TILES = K_TILES // CLUSTER  # 2 q_b k-tiles per rank
KB_RANKS = LATENT // 128  # 4 ranks hold a 128-row k_b tile
MMA_N = 8  # tokens per chunk (the MMA's N), and the group of tokens every per-token step works on
WIDE_CHUNKS = (
    16,
    32,
)  # the other chunk sizes (cluster_rms, two-hop): 64 tokens are 12 clusters in chunks of 32
CTA_K = 128
MMA_K = 16
TMA_K_BOX = 64
TMA_COPY_ITERS = CTA_K // TMA_K_BOX
K_BLOCKS_PER_HALF = TMA_K_BOX // MMA_K
THREADS = 256
EPI_THREADS = 128
TMEM_NOPE = 0  # the nope accumulator's TMEM column; pe and k_b follow at N and 2 N (tmem_cols)
ELEM_BYTES = 2
VEC = 8
EVICT_FIRST = 0x12F0000000000000

LEADING = 16
STRIDE = 8 * TMA_K_BOX * ELEM_BYTES  # 1024 B between 8-row swizzle atoms
STEP = (MMA_K * ELEM_BYTES) >> 4
HALF_128 = (
    128 * TMA_K_BOX * ELEM_BYTES
) >> 4  # one 64-column half of a 128-row tile, 16-byte units
HALF_64 = (64 * TMA_K_BOX * ELEM_BYTES) >> 4
TILE_128 = 2 * HALF_128
TILE_64 = 2 * HALF_64
NOPE_ELEMS = 128 * CTA_K  # per k-tile
PE_ELEMS = 64 * CTA_K

KB_MAIL_BYTES = (
    CLUSTER * 128 * MMA_N * 4
)  # the k_b rank's single-hop mailbox: 6 partials of 128 rows x 8 tokens

io_dtype = cutlass.BFloat16


def tmem_cols(n: int) -> int:
    """TMEM columns of an n-token chunk: the nope, pe and k_b accumulators (n columns each), a power of 2 >= 32."""
    cols = 32
    while cols < 3 * n:
        cols *= 2
    return cols


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
def _st_async_f32(dst, value, mbar, *, loc=None, ip=None):
    """st.async of one fp32 to a shared::cluster address, completing ``mbar`` (shared::cluster) by 4 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Float32(value).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", "r,f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _st_async_v4_b32(dst, w0, w1, w2, w3, mbar, *, loc=None, ip=None):
    """st.async of 16 bytes (four 32-bit words) to a shared::cluster address, completing ``mbar`` by 16 bytes."""
    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w1).ir_value(loc=loc, ip=ip), cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w3).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
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


def _swz(t, k, n=MMA_N):
    """Element index of the 8-column vector holding (row t < n, column k < 128) in an n x 128 bf16 K-major tile,
    128B swizzle (two 64-column halves of 8-row swizzle atoms); add k % 8 for the element itself."""
    half = k // cutlass.Int32(TMA_K_BOX)
    chunk = (k % cutlass.Int32(TMA_K_BOX)) // cutlass.Int32(VEC)
    phase = t if n == MMA_N else t % cutlass.Int32(8)
    return (
        half * cutlass.Int32(n * TMA_K_BOX)
        + t * cutlass.Int32(TMA_K_BOX)
        + (chunk ^ phase) * cutlass.Int32(VEC)
    )


def _plus(base, offset: int):
    """base + offset; a zero offset traces nothing, so 8-token chunks keep exactly their per-token instructions."""
    return base if offset == 0 else base + cutlass.Int32(offset)


@cute.kernel
def k3_mla_q_kernel(
    tma_nope: cutlass.GridConstant[cuda.TensorMap],  # W_qb [1152, 1536], box 128 rows
    tma_pe: cutlass.GridConstant[cuda.TensorMap],  # W_qb [1152, 1536], box 64 rows
    tma_kb: cutlass.GridConstant[cuda.TensorMap],  # W_kb [6 * 512, 128], box 128 rows
    ag: cutlass.Array,  # [M, ag_cols] bf16: q_a in columns [0, 1536)
    w_qa: cutlass.Array,  # [1536] bf16, q_a_layernorm weight
    fused_q: cutlass.Array,  # [M, heads, 576] bf16, out
    w_kv: cutlass.Array,  # [512] bf16, kv_a_layernorm weight (kv_mode)
    kv_pool: cutlass.Array,  # the latent pool, flat bf16: row i of page p at (p * 64 + i) * row_stride (kv_mode 1)
    page_table: cutlass.Array,  # int32, request i's pages at [i * pt_stride, ...) (kv_mode 1)
    seq_len: cutlass.Array,  # int32 [R], request i's KV length including its T tokens of this step (kv_mode 1)
    kv_out: cutlass.Array,  # [M, 576] bf16 (kv_mode 2)
    total_tokens: cutlass.Int32,  # M = R T
    tokens: cutlass.Int32,  # T (kv_mode 1)
    pt_stride: cutlass.Int32,  # page-table elements between consecutive requests' rows (kv_mode 1)
    eps: cutlass.Float32,
    kv_eps: cutlass.Float32,
    page_offset: cutlass.Int32,  # added to every page-table entry (the layer's slot in an interleaved pool)
    ag_cols: cutlass.Constexpr[int],
    total_heads: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    single_hop: cutlass.Constexpr[bool],
    kv_mode: cutlass.Constexpr[int],  # 0 off, 1 into the pool, 2 into kv_out
    row_stride: cutlass.Constexpr[int],
    cluster_rms: cutlass.Constexpr[
        bool
    ],  # the q_a RMS from the cluster's partial sums (each rank reads its 256 columns)
    mma_n: cutlass.Constexpr[
        int
    ],  # tokens per chunk: MMA_N, or one of WIDE_CHUNKS (cluster_rms, two-hop)
):
    tx, _, _ = cute.arch.thread_idx()
    bx, chunk, _ = cute.arch.block_idx()
    warp_id = cute.arch.warp_idx()
    rank = cute.arch.block_idx_in_cluster()
    head = bx // cutlass.Int32(CLUSTER)
    groups = mma_n // MMA_N  # per-token steps run for each group of 8 of the chunk's tokens
    half_n = (
        mma_n * TMA_K_BOX * ELEM_BYTES
    ) >> 4  # one 64-column half of the B tile, 16-byte units
    b_elems = mma_n * CTA_K
    col_pe, col_kb = (
        mma_n,
        2 * mma_n,
    )  # TMEM columns of the pe and k_b accumulators (nope at column 0)
    # This chunk's tokens: rows tok0 .. tok0 + num_tokens - 1 of ag and fused_q.
    tok0 = chunk * cutlass.Int32(mma_n)
    num_tokens = cutlass.Int32(
        cutlass.select_(
            total_tokens - tok0 < cutlass.Int32(mma_n), total_tokens - tok0, cutlass.Int32(mma_n)
        )
    )
    ptr_nope = tma_nope.get_ptr()
    ptr_pe = tma_pe.get_ptr()
    ptr_kb = tma_kb.get_ptr()

    # Same allocation order in every CTA: mapa() addresses the peers' mailbox, B2 tile and barriers.
    smem_wn = cutlass.Array(
        io_dtype, MY_TILES * NOPE_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_wp = cutlass.Array(
        io_dtype, MY_TILES * PE_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_wk = cutlass.Array(io_dtype, NOPE_ELEMS, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_b = cutlass.Array(
        io_dtype, MY_TILES * b_elems, space=cutlass.AddressSpace.smem, alignment=1024
    )
    smem_b2 = cutlass.Array(io_dtype, b_elems, space=cutlass.AddressSpace.smem, alignment=1024)
    stage_n = cutlass.Array(io_dtype, mma_n * 32, space=cutlass.AddressSpace.smem, alignment=16)
    stage_o = cutlass.Array(io_dtype, mma_n * 128, space=cutlass.AddressSpace.smem, alignment=16)
    # [source rank][token][row within the owned 32] fp32 partials (the own rank's slot stays unused).
    mailbox = cutlass.Array(
        cutlass.Float32, CLUSTER * 32 * mma_n, space=cutlass.AddressSpace.smem, alignment=16
    )
    norm_part = cutlass.Array(
        cutlass.Float32, 4 * mma_n, space=cutlass.AddressSpace.smem, alignment=16
    )
    rrms = cutlass.Array(cutlass.Float32, mma_n, space=cutlass.AddressSpace.smem, alignment=16)
    # cluster_rms: [source rank][token] partial sums of squares over the source's 256 q_a columns.
    norm_mail = cutlass.Array(
        cutlass.Float32, CLUSTER * mma_n, space=cutlass.AddressSpace.smem, alignment=16
    )
    norm_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    w_full = cutlass.Array(
        cutlass.Int64, 2 * MY_TILES + 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    b_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    mail_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    qn_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_done = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # single_hop: [source rank][nope row][token] fp32 partials, completed by bytes; the B tile's local readiness.
    # (Not in 32-token builds, which are two-hop: 96 KB.)
    kb_mail = None
    if cutlass.const_expr(mma_n == MMA_N):
        kb_mail = cutlass.Array(
            cutlass.Float32, CLUSTER * 128 * MMA_N, space=cutlass.AddressSpace.smem, alignment=16
        )
    kb_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b2_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)

    if warp_id == 0:
        prims.prefetch_tensormap(ptr_nope)
        prims.prefetch_tensormap(ptr_pe)
        prims.prefetch_tensormap(ptr_kb)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(2 * MY_TILES + 1):
                prims.mbarrier_init(w_full.subview(i), 1)
            prims.mbarrier_init(b_ready, EPI_THREADS)
            prims.mbarrier_init(acc_done, 1)
            # Both two-hop mailboxes complete by bytes (st.async), armed here once.
            prims.mbarrier_init(mail_full, 1)
            prims.mbarrier_init(qn_full, 1)
            prims.mbarrier_init(acc2_done, 1)
            prims.mbarrier_init(kb_full, 1)
            prims.mbarrier_init(b2_ready, EPI_THREADS)
            if cutlass.const_expr(single_hop):
                if rank < cutlass.Int32(KB_RANKS):
                    prims.mbarrier_arrive_expect_tx(kb_full, KB_MAIL_BYTES)
            prims.mbarrier_arrive_expect_tx(mail_full, (CLUSTER - 1) * mma_n * 32 * 4)
            if cutlass.const_expr(cluster_rms):
                prims.mbarrier_init(norm_full, 1)
                prims.mbarrier_arrive_expect_tx(norm_full, CLUSTER * mma_n * 4)
            if rank < cutlass.Int32(KB_RANKS):
                # 4 owners x the chunk's tokens x 32 columns, bf16.
                prims.mbarrier_arrive_expect_tx(qn_full, KB_RANKS * mma_n * 32 * ELEM_BYTES)
    if warp_id == 2:
        prims.tcgen05_alloc(tmem_ptr_i32, tmem_cols(mma_n))
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    # Cluster formation: the peers' shared memory and barriers are addressable from here on.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_base = tmem_ptr_i32.load()

    if warp_id == 0:
        # =====================================================================
        # Weights, ahead of the grid dependency: q_b nope and pe rows over this
        # rank's two k-tiles, and (ranks 0-3) the k_b tile.
        # =====================================================================
        if prims.elect_sync():
            row_nope = head * cutlass.Int32(QK)
            for i in cutlass.range_constexpr(MY_TILES):
                k = rank * cutlass.Int32(MY_TILES) + cutlass.Int32(i)
                prims.mbarrier_arrive_expect_tx(w_full.subview(i), NOPE_ELEMS * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_wn.subview(i * NOPE_ELEMS),
                    ptr_nope,
                    (
                        cutlass.Int32(0),
                        row_nope,
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    w_full.subview(i),
                    l2_cache_hint=EVICT_FIRST,
                )
                prims.mbarrier_arrive_expect_tx(w_full.subview(MY_TILES + i), PE_ELEMS * ELEM_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_wp.subview(i * PE_ELEMS),
                    ptr_pe,
                    (
                        cutlass.Int32(0),
                        row_nope + cutlass.Int32(NOPE),
                        k * cutlass.Int32(TMA_COPY_ITERS),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    w_full.subview(MY_TILES + i),
                    l2_cache_hint=EVICT_FIRST,
                )
            if rank < cutlass.Int32(KB_RANKS):
                prims.mbarrier_arrive_expect_tx(
                    w_full.subview(2 * MY_TILES), NOPE_ELEMS * ELEM_BYTES
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_wk,
                    ptr_kb,
                    (
                        cutlass.Int32(0),
                        head * cutlass.Int32(LATENT) + rank * cutlass.Int32(128),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                    ),
                    w_full.subview(2 * MY_TILES),
                    l2_cache_hint=EVICT_FIRST,
                )
        if cutlass.const_expr(trigger_early):
            # Dependents may launch now; they wait for this whole grid before reading fused_q.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    elif warp_id == 1:
        if cutlass.const_expr(kv_mode != 0):
            # =====================================================================
            # KV half: CTA j < the chunk's tokens, token tok0 + j's cache row (see
            # kv_mode above).
            # =====================================================================
            if bx < num_tokens:
                tok = tok0 + bx
                lane = tx % 32
                # Lane l: latent vectors l and l + 32 (8 columns each) and, for l < 8, rope vector l.
                wk = []
                for j in cutlass.range_constexpr(2):
                    wk.append(
                        w_kv.load(
                            idx=(lane + cutlass.Int32(32 * j)) * cutlass.Int32(VEC),
                            vector_size=VEC,
                            alignment=16,
                        )
                    )
                dst = cutlass.Int64(tok) * cutlass.Int64(FUSED)
                keep = cutlass.Boolean(True)
                if cutlass.const_expr(kv_mode == 1):
                    req = tok // tokens
                    pos = seq_len.load(idx=req) - tokens + (tok - req * tokens)
                    keep = pos >= cutlass.Int32(0)
                    pos_c = cutlass.Int32(cutlass.select_(keep, pos, cutlass.Int32(0)))
                    page = (
                        page_table.load(idx=req * pt_stride + pos_c // cutlass.Int32(PAGE))
                        + page_offset
                    )
                    dst = (
                        cutlass.Int64(page) * cutlass.Int64(PAGE)
                        + cutlass.Int64(pos_c % cutlass.Int32(PAGE))
                    ) * cutlass.Int64(row_stride)
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                src = tok * cutlass.Int32(ag_cols) + cutlass.Int32(Q_LORA)
                xs = []
                for j in cutlass.range_constexpr(2):
                    xs.append(
                        ag.load(
                            idx=src + (lane + cutlass.Int32(32 * j)) * cutlass.Int32(VEC),
                            vector_size=VEC,
                            alignment=16,
                        )
                    )
                pe_v = cutlass.Int32(
                    cutlass.select_(lane < cutlass.Int32(PE // VEC), lane, cutlass.Int32(0))
                )
                pe = ag.load(
                    idx=src + cutlass.Int32(LATENT) + pe_v * cutlass.Int32(VEC),
                    vector_size=VEC,
                    alignment=16,
                )
                ss = cutlass.Float32(0.0)
                for j in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(VEC):
                        xf = cutlass.Float32(xs[j][e])
                        ss = ss + xf * xf
                for offset in (16, 8, 4, 2, 1):
                    ss = ss + cute.arch.shuffle_sync_bfly(ss, offset=offset)
                r = cute.math.rsqrt(ss * cutlass.Float32(1.0 / LATENT) + kv_eps, fastmath=True)
                dst_arr = kv_pool if cutlass.const_expr(kv_mode == 1) else kv_out
                if keep:
                    for j in cutlass.range_constexpr(2):
                        outs = []
                        for e in cutlass.range_constexpr(VEC):
                            # flashinfer's RMSNorm order: (x * rrms) * w in fp32, one bf16 rounding.
                            outs.append(
                                (cutlass.Float32(xs[j][e]) * r * cutlass.Float32(wk[j][e])).to(
                                    io_dtype
                                )
                            )
                        dst_arr.store(
                            cutlass.Vector.from_elements(tuple(outs), io_dtype),
                            idx=dst
                            + cutlass.Int64((lane + cutlass.Int32(32 * j)) * cutlass.Int32(VEC)),
                            vector_size=VEC,
                            alignment=16,
                        )
                    if lane < cutlass.Int32(PE // VEC):
                        dst_arr.store(
                            pe,
                            idx=dst
                            + cutlass.Int64(cutlass.Int32(LATENT) + lane * cutlass.Int32(VEC)),
                            vector_size=VEC,
                            alignment=16,
                        )
    elif warp_id == 2:
        # =====================================================================
        # MMA: q_b partials (nope M 128, pe M 64) once B is normalized; then,
        # on ranks 0-3, the k_b product once q_nope has arrived.
        # =====================================================================
        idesc128 = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=mma_n, m_dim=128
        )
        idesc64 = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=io_dtype, b_dtype=io_dtype, n_dim=mma_n, m_dim=64
        )
        swz = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
        desc_wn = prims.Tcgen05SmemDesc.build(
            start_address=smem_wn,
            leading_byte_offset=LEADING,
            stride_byte_offset=STRIDE,
            layout=swz,
        )
        desc_wp = prims.Tcgen05SmemDesc.build(
            start_address=smem_wp,
            leading_byte_offset=LEADING,
            stride_byte_offset=STRIDE,
            layout=swz,
        )
        desc_wk = prims.Tcgen05SmemDesc.build(
            start_address=smem_wk,
            leading_byte_offset=LEADING,
            stride_byte_offset=STRIDE,
            layout=swz,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            start_address=smem_b, leading_byte_offset=LEADING, stride_byte_offset=STRIDE, layout=swz
        )
        desc_b2 = prims.Tcgen05SmemDesc.build(
            start_address=smem_b2,
            leading_byte_offset=LEADING,
            stride_byte_offset=STRIDE,
            layout=swz,
        )
        tmem_nope = cutlass.inttoptr(tmem_base + cutlass.Int32(TMEM_NOPE), 6, cutlass.Int32)
        tmem_pe = cutlass.inttoptr(tmem_base + cutlass.Int32(col_pe), 6, cutlass.Int32)
        tmem_kb = cutlass.inttoptr(tmem_base + cutlass.Int32(col_kb), 6, cutlass.Int32)
        while not cute.arch.mbarrier_try_wait(b_ready.data_ptr(), 0):
            pass
        for i in cutlass.range_constexpr(MY_TILES):
            while not cute.arch.mbarrier_try_wait(w_full.subview(i).data_ptr(), 0):
                pass
            while not cute.arch.mbarrier_try_wait(w_full.subview(MY_TILES + i).data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                d_b = desc_b + (i * 2 * half_n + box * half_n + within * STEP)
                acc = not (i == 0 and kb == 0)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_nope,
                        desc_wn + (i * TILE_128 + box * HALF_128 + within * STEP), d_b, idesc128, acc,
                    )  # fmt: skip
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_pe,
                        desc_wp + (i * TILE_64 + box * HALF_64 + within * STEP), d_b, idesc64, acc,
                    )  # fmt: skip
        if prims.elect_sync():
            prims.tcgen05_commit(acc_done)
        if rank < cutlass.Int32(KB_RANKS):
            while not cute.arch.mbarrier_try_wait(w_full.subview(2 * MY_TILES).data_ptr(), 0):
                pass
            if cutlass.const_expr(single_hop):
                # This rank's epilogue wrote the B tile (generic stores, each writer fenced to the async proxy).
                while not cute.arch.mbarrier_try_wait(b2_ready.data_ptr(), 0):
                    pass
            else:
                # q_nope arrives from the 4 owners' st.async stores, then feeds the tensor core (async proxy).
                while not _try_wait_cluster(qn_full.data_ptr(), 0):
                    pass
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for kb in cutlass.range_constexpr(TMA_COPY_ITERS * K_BLOCKS_PER_HALF):
                box = kb // K_BLOCKS_PER_HALF
                within = kb % K_BLOCKS_PER_HALF
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16, prims.CTAGroup.CTA_1, tmem_kb,
                        desc_wk + (box * HALF_128 + within * STEP), desc_b2 + (box * half_n + within * STEP),
                        idesc128, kb != 0,
                    )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(acc2_done)
    elif warp_id >= 4:
        tid = tx - cutlass.Int32(EPI_THREADS)
        lane = tx % 32
        w = warp_id - 4
        # =====================================================================
        # Norm prolog: the RMS of every token's q_a row (1536 columns), then this
        # rank's 256 columns normalized into the resident B tile.
        # =====================================================================
        last_row = num_tokens - cutlass.Int32(1)
        if cutlass.const_expr(cluster_rms):
            # Thread tid: token t = tid / 16 of each group, this rank's vectors c and c + 16 (c = tid % 16; columns
            # 256 rank + 8 c). Each rank reads only its own 256 columns; the per-token partial sums of squares go to
            # every rank of the cluster (st.async into slot [rank][t], completing norm_full by bytes), which adds the
            # 6 in rank order: the same total, hence the same rrms, on every rank.
            cr_t_me = tid // cutlass.Int32(16)
            cr_c_me = tid % cutlass.Int32(16)
            cr_wk = []
            for cr_j in cutlass.range_constexpr(2):
                cr_wk.append(
                    w_qa.load(
                        idx=(rank * cutlass.Int32(32) + cr_c_me + cutlass.Int32(16 * cr_j))
                        * cutlass.Int32(VEC),
                        vector_size=VEC,
                        alignment=16,
                    )
                )
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            cr_t = []
            cr_live_t = []
            cr_row_t = []
            for cr_g in cutlass.range_constexpr(groups):
                cr_t.append(_plus(cr_t_me, MMA_N * cr_g))
                cr_live_t.append(cr_t[cr_g] < num_tokens)
                cr_row_t.append(
                    cutlass.Int32(cutlass.select_(cr_live_t[cr_g], cr_t[cr_g], last_row))
                )
            cr_xk = []
            cr_part = []
            for cr_g in cutlass.range_constexpr(groups):
                cr_part_g = cutlass.Float32(0.0)
                for cr_j in cutlass.range_constexpr(2):
                    cr_xv = ag.load(
                        idx=(tok0 + cr_row_t[cr_g]) * cutlass.Int32(ag_cols)
                        + (rank * cutlass.Int32(32) + cr_c_me + cutlass.Int32(16 * cr_j))
                        * cutlass.Int32(VEC),
                        vector_size=VEC,
                        alignment=16,
                    )
                    cr_xk.append(cr_xv)
                    for cr_e in cutlass.range_constexpr(VEC):
                        cr_xf = cutlass.Float32(
                            cutlass.select_(
                                cr_live_t[cr_g], cutlass.Float32(cr_xv[cr_e]), cutlass.Float32(0.0)
                            )
                        )
                        cr_part_g = cr_part_g + cr_xf * cr_xf
                cr_part.append(cr_part_g)
            for cr_g in cutlass.range_constexpr(groups):
                for cr_off in (8, 4, 2, 1):
                    cr_part[cr_g] = cr_part[cr_g] + cute.arch.shuffle_sync_bfly(
                        cr_part[cr_g], offset=cr_off
                    )
            if cr_c_me == cutlass.Int32(0):
                for cr_g in cutlass.range_constexpr(groups):
                    for cr_dst in cutlass.range_constexpr(CLUSTER):
                        _st_async_f32(
                            _mapa_u32(norm_mail.data_ptr(rank * cutlass.Int32(mma_n) + cr_t[cr_g]), cr_dst),
                            cr_part[cr_g], _mapa_u32(norm_full.data_ptr(), cr_dst),
                        )  # fmt: skip
            while not _try_wait_cluster(norm_full.data_ptr(), 0):
                pass
            for cr_g in cutlass.range_constexpr(groups):
                cr_total = norm_mail.load(idx=cr_t[cr_g])
                for cr_src in cutlass.range_constexpr(1, CLUSTER):
                    cr_total = cr_total + norm_mail.load(
                        idx=cutlass.Int32(cr_src * mma_n) + cr_t[cr_g]
                    )
                cr_r = cutlass.Float32(
                    cutlass.select_(
                        cr_live_t[cr_g],
                        cute.math.rsqrt(
                            cr_total * cutlass.Float32(1.0 / Q_LORA) + eps, fastmath=True
                        ),
                        cutlass.Float32(0.0),
                    )
                )
                for cr_j in cutlass.range_constexpr(2):
                    cr_outs = []
                    for cr_e in cutlass.range_constexpr(VEC):
                        # flashinfer's RMSNorm order: (x * rrms) * w in fp32, one bf16 rounding.
                        cr_outs.append(
                            (
                                cutlass.Float32(cr_xk[2 * cr_g + cr_j][cr_e])
                                * cr_r
                                * cutlass.Float32(cr_wk[cr_j][cr_e])
                            ).to(io_dtype)
                        )
                    cr_k_local = (cr_c_me + cutlass.Int32(16 * cr_j)) * cutlass.Int32(VEC)
                    smem_b.store(
                        cutlass.Vector.from_elements(tuple(cr_outs), io_dtype),
                        idx=(cr_k_local // cutlass.Int32(CTA_K)) * cutlass.Int32(b_elems)
                        + _swz(cr_t[cr_g], cr_k_local % cutlass.Int32(CTA_K), mma_n),
                        vector_size=VEC,
                        alignment=16,
                    )
        else:
            # RMS pass: thread tid loads 16-byte vector v = tid + 128 j (j = 1 only for tid < 64) of every token's
            # 1536-column q_a row. This rank's own 32 vectors (v = 32 rank + lane) sit in warp rank % 4, pass rank // 4:
            # that warp normalizes them from registers, so q_a is read once.
            owner = w == rank % cutlass.Int32(4)
            own_j = rank // cutlass.Int32(4)
            # The norm weight of the owner lane's 8 columns is a weight: loaded ahead of the grid dependency.
            wv = w_qa.load(
                idx=(rank * cutlass.Int32(32) + lane) * cutlass.Int32(VEC),
                vector_size=VEC,
                alignment=16,
            )
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            # Rows past M and vectors past the row are loaded clamped and count zero (no loads out of bounds, no
            # values escaping runtime branches).
            sums = [cutlass.Float32(0.0)] * MMA_N
            xs = []
            for t in cutlass.range_constexpr(MMA_N):
                row_t = cutlass.Int32(
                    cutlass.select_(cutlass.Int32(t) < num_tokens, cutlass.Int32(t), last_row)
                )
                for j in cutlass.range_constexpr((Q_LORA // VEC + EPI_THREADS - 1) // EPI_THREADS):
                    v = tid + cutlass.Int32(j * EPI_THREADS)
                    live = (cutlass.Int32(t) < num_tokens) & (v < cutlass.Int32(Q_LORA // VEC))
                    v_c = cutlass.Int32(
                        cutlass.select_(v < cutlass.Int32(Q_LORA // VEC), v, cutlass.Int32(0))
                    )
                    xv = ag.load(
                        idx=(tok0 + row_t) * cutlass.Int32(ag_cols) + v_c * cutlass.Int32(VEC),
                        vector_size=VEC,
                        alignment=16,
                    )
                    xs.append(xv)
                    for e in cutlass.range_constexpr(VEC):
                        xf = cutlass.Float32(
                            cutlass.select_(live, cutlass.Float32(xv[e]), cutlass.Float32(0.0))
                        )
                        sums[t] = sums[t] + xf * xf
            for t in cutlass.range_constexpr(MMA_N):
                for offset in (16, 8, 4, 2, 1):
                    sums[t] = sums[t] + cute.arch.shuffle_sync_bfly(sums[t], offset=offset)
            if lane == 0:
                for t in cutlass.range_constexpr(MMA_N):
                    norm_part.store(sums[t], idx=w * cutlass.Int32(MMA_N) + cutlass.Int32(t))
            prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
            if tid < cutlass.Int32(MMA_N):
                total = norm_part.load(idx=tid)
                for ww in cutlass.range_constexpr(1, 4):
                    total = total + norm_part.load(idx=cutlass.Int32(ww * MMA_N) + tid)
                rrms.store(
                    cute.math.rsqrt(total * cutlass.Float32(1.0 / Q_LORA) + eps, fastmath=True),
                    idx=tid,
                )
            prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
            # The owner warp: 8 tokens x its lane's 8 columns (CTA column 8 lane) into the B tile (rows >= M are zero).
            n_pass = (Q_LORA // VEC + EPI_THREADS - 1) // EPI_THREADS
            if owner:
                for jo in cutlass.range_constexpr(n_pass):
                    if own_j == cutlass.Int32(jo):
                        for t in cutlass.range_constexpr(MMA_N):
                            xv = xs[t * n_pass + jo]
                            r = cutlass.Float32(
                                cutlass.select_(
                                    cutlass.Int32(t) < num_tokens,
                                    rrms.load(idx=cutlass.Int32(t)),
                                    cutlass.Float32(0.0),
                                )
                            )
                            outs = []
                            for e in cutlass.range_constexpr(VEC):
                                # flashinfer's RMSNorm order: (x * rrms) * w in fp32, one bf16 rounding.
                                outs.append(
                                    (cutlass.Float32(xv[e]) * r * cutlass.Float32(wv[e])).to(
                                        io_dtype
                                    )
                                )
                            k_local = lane * cutlass.Int32(VEC)
                            smem_b.store(
                                cutlass.Vector.from_elements(tuple(outs), io_dtype),
                                idx=(k_local // cutlass.Int32(CTA_K)) * cutlass.Int32(b_elems)
                                + _swz(cutlass.Int32(t), k_local % cutlass.Int32(CTA_K)),
                                vector_size=VEC,
                                alignment=16,
                            )
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.mbarrier_arrive(b_ready)

        # =====================================================================
        # q_b partials -> row owners (DSMEM), rank-order sum, bf16.
        # =====================================================================
        while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc_n = prims.tcgen05_ld(
            "32x32b",
            cutlass.inttoptr(tmem_base + cutlass.Int32(TMEM_NOPE), 6, cutlass.Float32),
            num=mma_n,
        )
        acc_p = prims.tcgen05_ld(
            "32x32b",
            cutlass.inttoptr(tmem_base + cutlass.Int32(col_pe), 6, cutlass.Float32),
            num=mma_n,
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        # nope rows 32 w + lane -> rank w; pe rows 16 w + lane (lane < 16) -> rank 4 + w // 2, local row
        # 16 (w % 2) + lane.
        pe_owner = cutlass.Int32(KB_RANKS) + w // cutlass.Int32(2)
        pe_row = (w % cutlass.Int32(2)) * cutlass.Int32(16) + lane
        if cutlass.const_expr(single_hop):
            # Nope row 32 w + lane, 8 tokens -> slot [rank] of every k_b rank's mailbox (this rank's too).
            base_kb = (rank * cutlass.Int32(128) + w * cutlass.Int32(32) + lane) * cutlass.Int32(
                MMA_N
            )
            for dst in cutlass.range_constexpr(KB_RANKS):
                mbar = _mapa_u32(kb_full.data_ptr(), dst)
                for h4 in cutlass.range_constexpr(MMA_N // 4):
                    _st_async_v4(
                        _mapa_u32(kb_mail.data_ptr(base_kb + cutlass.Int32(4 * h4)), dst),
                        acc_n[4 * h4], acc_n[4 * h4 + 1], acc_n[4 * h4 + 2], acc_n[4 * h4 + 3], mbar,
                    )  # fmt: skip
        else:
            if w != rank:
                # st.async into the owner's mailbox [source][token][row]: a warp's 32 lanes fill 128 contiguous bytes
                # per token; the owner's barrier completes when all 5 sources' bytes have landed.
                mb_n = _mapa_u32(mail_full.data_ptr(), w)
                for t in cutlass.range_constexpr(mma_n):
                    _st_async_f32(
                        _mapa_u32(mailbox.data_ptr((rank * cutlass.Int32(mma_n) + cutlass.Int32(t)) * cutlass.Int32(32)
                                                   + lane), w),
                        acc_n[t], mb_n,
                    )  # fmt: skip
        if lane < cutlass.Int32(16):
            if pe_owner != rank:
                mb_p = _mapa_u32(mail_full.data_ptr(), pe_owner)
                for t in cutlass.range_constexpr(mma_n):
                    _st_async_f32(
                        _mapa_u32(mailbox.data_ptr((rank * cutlass.Int32(mma_n) + cutlass.Int32(t)) * cutlass.Int32(32)
                                                   + pe_row), pe_owner),
                        acc_p[t], mb_p,
                    )  # fmt: skip
        if cutlass.const_expr(single_hop):
            if rank < cutlass.Int32(KB_RANKS):
                # Nope row tid: the 6 partials in rank order (the owner's sum of the two-hop path), bf16, into the
                # k_b B tile (token t, column tid), fenced to the async proxy for the MMA warp.
                while not _try_wait_cluster(kb_full.data_ptr(), 0):
                    pass
                parts = []
                for src in cutlass.range_constexpr(CLUSTER):
                    for h4 in cutlass.range_constexpr(MMA_N // 4):
                        parts.append(
                            kb_mail.load(
                                idx=(cutlass.Int32(src * 128) + tid) * cutlass.Int32(MMA_N)
                                + cutlass.Int32(4 * h4),
                                vector_size=4,
                                alignment=16,
                            )
                        )
                for t in cutlass.range_constexpr(MMA_N):
                    total = cutlass.Float32(0.0)
                    for src in cutlass.range_constexpr(CLUSTER):
                        total = total + cutlass.Float32(parts[src * (MMA_N // 4) + t // 4][t % 4])
                    smem_b2.store(
                        total.to(io_dtype),
                        idx=_swz(cutlass.Int32(t), tid) + tid % cutlass.Int32(VEC),
                    )
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.mbarrier_arrive(b2_ready)
        elif cutlass.const_expr(mma_n == MMA_N):
            if w == rank:
                # Rank w < 4 owns nope rows 32 w + lane: sum the 6 partials, round, stage for the k_b broadcast.
                while not _try_wait_cluster(mail_full.data_ptr(), 0):
                    pass
                for t in cutlass.range_constexpr(MMA_N):
                    total = cutlass.Float32(0.0)
                    for src in cutlass.range_constexpr(CLUSTER):
                        part = mailbox.load(idx=cutlass.Int32((src * MMA_N + t) * 32) + lane)
                        total = total + cutlass.Float32(
                            cutlass.select_(
                                rank == cutlass.Int32(src), cutlass.Float32(acc_n[t]), part
                            )
                        )
                    stage_n.store(total.to(io_dtype), idx=cutlass.Int32(t * 32) + lane)
                prims.bar_warp_sync(0xFFFFFFFF)
                # Lane l: token l // 4, 8 of the owner's 32 columns (16 bytes) -> st.async into the k_b B tile of ranks
                # 0-3 (this one too), completing their q_nope barrier.
                ct = lane // cutlass.Int32(4)
                c4 = lane % cutlass.Int32(4)
                stage_w = cutlass.Array(
                    cutlass.inttoptr(stage_n.data_ptr().toint(), 3, cutlass.Int32),
                    shape=MMA_N * 32 // 2,
                )
                words = stage_w.load(
                    idx=(ct * cutlass.Int32(32) + c4 * cutlass.Int32(VEC)) // cutlass.Int32(2),
                    vector_size=4,
                    alignment=16,
                )
                dst_idx = _swz(ct, rank * cutlass.Int32(32) + c4 * cutlass.Int32(VEC))
                for dst in cutlass.range_constexpr(KB_RANKS):
                    _st_async_v4_b32(
                        _mapa_u32(smem_b2.data_ptr(dst_idx), dst), words[0], words[1], words[2], words[3],
                        _mapa_u32(qn_full.data_ptr(), dst),
                    )  # fmt: skip
        else:
            if rank < cutlass.Int32(KB_RANKS):
                # Rank r < 4 owns nope rows 32 r + lane. Its warp r puts its own partials into the mailbox's slot [r];
                # then epilogue warp v sums the 6 partials of tokens 8 v .. 8 v + 7 in rank order, rounds, stages and
                # sends them as above (each group's sums and transfers are an 8-token chunk's).
                if w == rank:
                    for t in cutlass.range_constexpr(mma_n):
                        mailbox.store(
                            cutlass.Float32(acc_n[t]),
                            idx=(rank * cutlass.Int32(mma_n) + cutlass.Int32(t)) * cutlass.Int32(32)
                            + lane,
                        )
                prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
                while not _try_wait_cluster(mail_full.data_ptr(), 0):
                    pass
                if w < cutlass.Int32(groups):
                    for t in cutlass.range_constexpr(MMA_N):
                        g_tok = w * cutlass.Int32(MMA_N) + cutlass.Int32(t)
                        g_total = cutlass.Float32(0.0)
                        for src in cutlass.range_constexpr(CLUSTER):
                            g_total = g_total + cutlass.Float32(
                                mailbox.load(
                                    idx=(cutlass.Int32(src * mma_n) + g_tok) * cutlass.Int32(32)
                                    + lane
                                )
                            )
                        stage_n.store(g_total.to(io_dtype), idx=g_tok * cutlass.Int32(32) + lane)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    g_ct = w * cutlass.Int32(MMA_N) + lane // cutlass.Int32(4)
                    g_c4 = lane % cutlass.Int32(4)
                    g_stage = cutlass.Array(
                        cutlass.inttoptr(stage_n.data_ptr().toint(), 3, cutlass.Int32),
                        shape=mma_n * 32 // 2,
                    )
                    g_words = g_stage.load(
                        idx=(g_ct * cutlass.Int32(32) + g_c4 * cutlass.Int32(VEC))
                        // cutlass.Int32(2),
                        vector_size=4,
                        alignment=16,
                    )
                    g_dst = _swz(g_ct, rank * cutlass.Int32(32) + g_c4 * cutlass.Int32(VEC), mma_n)
                    for dst in cutlass.range_constexpr(KB_RANKS):
                        _st_async_v4_b32(
                            _mapa_u32(smem_b2.data_ptr(g_dst), dst), g_words[0], g_words[1], g_words[2], g_words[3],
                            _mapa_u32(qn_full.data_ptr(), dst),
                        )  # fmt: skip
        if rank >= cutlass.Int32(KB_RANKS):
            if pe_owner == rank:
                if lane < cutlass.Int32(16):
                    # Rank 4 + j owns pe rows 32 j + pe_row: sum, round, store fused_q[t, head, 512 + row].
                    while not _try_wait_cluster(mail_full.data_ptr(), 0):
                        pass
                    p = (rank - cutlass.Int32(KB_RANKS)) * cutlass.Int32(32) + pe_row
                    for t in cutlass.range_constexpr(mma_n):
                        total = cutlass.Float32(0.0)
                        for src in cutlass.range_constexpr(CLUSTER):
                            part = mailbox.load(idx=cutlass.Int32((src * mma_n + t) * 32) + pe_row)
                            total = total + cutlass.Float32(
                                cutlass.select_(
                                    rank == cutlass.Int32(src), cutlass.Float32(acc_p[t]), part
                                )
                            )
                        if cutlass.Int32(t) < num_tokens:
                            fused_q.store(
                                total.to(io_dtype),
                                idx=(tok0 + cutlass.Int32(t)) * cutlass.Int32(total_heads * FUSED)
                                + head * cutlass.Int32(FUSED)
                                + cutlass.Int32(LATENT)
                                + p,
                            )
        # =====================================================================
        # k_b product (ranks 0-3): fused_q[t, head, 128 rank + i].
        # =====================================================================
        if rank < cutlass.Int32(KB_RANKS):
            while not cute.arch.mbarrier_try_wait(acc2_done.data_ptr(), 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_k = prims.tcgen05_ld(
                "32x32b",
                cutlass.inttoptr(tmem_base + cutlass.Int32(col_kb), 6, cutlass.Float32),
                num=mma_n,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            row = w * cutlass.Int32(32) + lane
            for t in cutlass.range_constexpr(mma_n):
                stage_o.store(
                    cutlass.Float32(acc_k[t]).to(io_dtype), idx=cutlass.Int32(t * 128) + row
                )
        # Every TMEM reader has waited for its loads; the staging tile is complete.
        prims.barrier_cta_sync(1, thread_count=EPI_THREADS)
        if warp_id == 4:
            prims.tcgen05_dealloc(cutlass.inttoptr(tmem_base, 6, cutlass.Int32), tmem_cols(mma_n))
        if rank < cutlass.Int32(KB_RANKS):
            o_tok = tid // cutlass.Int32(16)
            oc = tid % cutlass.Int32(16)
            for g in cutlass.range_constexpr(groups):
                o_tok_g = _plus(o_tok, MMA_N * g)
                if o_tok_g < num_tokens:
                    out_vec = stage_o.load(
                        idx=o_tok_g * cutlass.Int32(128) + oc * cutlass.Int32(VEC),
                        vector_size=VEC,
                        alignment=16,
                    )
                    fused_q.store(
                        out_vec,
                        idx=(tok0 + o_tok_g) * cutlass.Int32(total_heads * FUSED)
                        + head * cutlass.Int32(FUSED)
                        + rank * cutlass.Int32(128)
                        + oc * cutlass.Int32(VEC),
                        vector_size=VEC,
                        alignment=16,
                    )


def _weight_map(w, n_rows, k_in, box_rows):
    """W [n_rows, k_in] as five TMA dimensions (64-column chunk, row, chunk index, 1, 1): one call per 128-column
    k-tile lands both 128-byte-swizzled halves of `box_rows` rows."""
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
        box_dims=[TMA_K_BOX, box_rows, TMA_COPY_ITERS, 1, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def k3_mla_q(
    w_qb: cute.Tensor,  # [heads * 192, 1536] bf16
    w_kb: cute.Tensor,  # [heads * 512, 128] bf16 (k_b_proj_trans rows)
    ag: cute.Tensor,  # [M * ag_cols] bf16: q_a in columns [0, 1536) of each row
    w_qa: cute.Tensor,  # [1536] bf16
    fused_q: cute.Tensor,  # [M * heads * 576] bf16
    w_kv: cute.Tensor,  # [512] bf16 (kv_mode)
    kv_pool: cute.Tensor,  # flat bf16 latent pool (kv_mode 1)
    page_table: cute.Tensor,  # int32, request i's pages at [i * pt_stride, ...) (kv_mode 1)
    seq_len: cute.Tensor,  # int32 [R] (kv_mode 1)
    kv_out: cute.Tensor,  # [M * 576] bf16 (kv_mode 2)
    num_tokens: cutlass.Int32,  # M
    tokens: cutlass.Int32,  # T = M / R (kv_mode 1)
    pt_stride: cutlass.Int32,  # (kv_mode 1)
    eps: cutlass.Float32,
    kv_eps: cutlass.Float32,
    page_offset: cutlass.Int32,
    ag_cols: cutlass.Constexpr[int],
    total_heads: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    single_hop: cutlass.Constexpr[bool],
    kv_mode: cutlass.Constexpr[int],
    row_stride: cutlass.Constexpr[int],
    cluster_rms: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    mma_n: cutlass.Constexpr[
        int
    ],  # tokens per chunk: MMA_N, or one of WIDE_CHUNKS with cluster_rms, two-hop
    stream: cuda_driver.CUstream,
) -> None:
    if cutlass.const_expr(
        mma_n != MMA_N and (mma_n not in WIDE_CHUNKS or single_hop or not cluster_rms)
    ):
        raise ValueError(
            f"k3_mla_q: chunks of {mma_n} tokens need one of {WIDE_CHUNKS}, cluster_rms and two-hop"
        )
    tma_nope = _weight_map(w_qb, total_heads * QK, Q_LORA, 128)
    tma_pe = _weight_map(w_qb, total_heads * QK, Q_LORA, 64)
    tma_kb = _weight_map(w_kb, total_heads * LATENT, NOPE, 128)
    k3_mla_q_kernel(
        tma_nope,
        tma_pe,
        tma_kb,
        ag,
        w_qa,
        fused_q,
        w_kv,
        kv_pool,
        page_table,
        seq_len,
        kv_out,
        num_tokens,
        tokens,
        pt_stride,
        eps,
        kv_eps,
        page_offset,
        ag_cols,
        total_heads,
        trigger_early,
        single_hop,
        kv_mode,
        row_stride,
        cluster_rms,
        mma_n,
    ).launch(  # fmt: skip
        grid=(total_heads * CLUSTER, (num_tokens + mma_n - 1) // mma_n, 1),
        block=(THREADS, 1, 1),
        cluster=(CLUSTER, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )
