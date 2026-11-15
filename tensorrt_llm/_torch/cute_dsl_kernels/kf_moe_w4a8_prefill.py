# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Portions of this file are adapted from the CUTLASS CuTe DSL example dense_blockscaled_gemm_persistent.py
# (https://github.com/NVIDIA/cutlass): the persistent block-scaled tcgen05 GEMM structure (pipelines, shared
# storage, TMA and TMEM handling). Those portions are covered by the following copyright notice and BSD-3-Clause
# license; the rest of this file is licensed under the Apache License, Version 2.0.
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Alternative MoE kernel module for prefill-sized steps. The host entry is split (_fc_rows, run, run_unfinalized) so
# that a caller with its own fused finalize can take the FC2 rows before the combine. Used only through the default-off
# TLLM_MOE_KF_PREFILL hook (tensorrt_llm/_torch/moe/fused_moe/kf_moe_w4a8.py).
# GPT-OSS-120B MoE layer (MXFP4 weights x MXFP8 activations, pre-routed top-4 of 128 experts)
# CuTe DSL implementation for Blackwell-family GPUs (tcgen05 block-scaled MMA, kind mxf8f6f4).
#
# Pipeline (all device side, CUDA-graph capturable, no host syncs):
#   routing  : expert-sorted row permutation + per-expert 128-row-aligned segments
#   gather   : permuted MXFP8 activation copy + scale factors in the 128x4 block-scaled layout
#   fc1      : grouped block-scaled GEMM -> bias -> clamped SwiGLU -> MXFP8 requantization
#   fc2      : grouped block-scaled GEMM -> bias -> bf16
#   combine  : fp32 FMA chain over the 4 routing slots -> bf16
import threading

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

# ---------------------------------------------------------------- problem constants
H = 2880          # model hidden / output width
I = 2880          # model intermediate (live)
I_PAD = 2944      # padded intermediate (FC1 rows per half, FC2 K)
H1 = 3072         # hidden_states width
N1 = 5888         # FC1 prepared rows (2 * I_PAD)
H2 = 2944         # FC2 prepared rows
TOPK = 4
SFVEC = 32

# K extents actually consumed (>= 2880 live columns, multiple of 128 for the SF atom)
K1 = 2944
K2 = 2944

# ---------------------------------------------------------------- tuning constants
BLK_M = 128       # CTA tile M (tokens)
BLK_N1 = 256      # FC1 CTA tile N (prepared weight rows)
BLK_N2 = 256      # FC2 CTA tile N
BLK_K = 128       # mainloop K tile
K_LIVE_CHUNKS = 2  # live 32-wide chunks in the final K tile (2880 of 2944)
MMA_K = 32        # tcgen05 block-scaled instruction K (64 on Vera Rubin)
CTA_GROUP = 2     # tcgen05 CTAs per MMA: 256-token MMA tile across a 2-CTA cluster
MMA_M = BLK_M * CTA_GROUP
AB_STAGES1 = 6      # blk_n=256: 34.3KB/stage is the SMEM ceiling
AB_STAGES2 = 5
ACC_STAGES = 2
EPI_N = 32        # epilogue subtile N (= one prepared 32-row block)
GATHER_ROWS_PER_CTA = 4
GATHER_THREADS = 192   # 180 live 16B vectors per row -> one full pass
COMBINE_THREADS = 256
COMBINE_VEC = 8

NT1 = (2 * I + BLK_N1 - 1) // BLK_N1  # prepared rows >= 5760 are all-zero padding
NT2 = (H2 + BLK_N2 - 1) // BLK_N2
KT1 = K1 // BLK_K     # 23
KT2 = K2 // BLK_K     # 23

EPI_WARPS = 8
EPI_GROUPS = EPI_WARPS // 4
MMA_WARP = EPI_WARPS
TMA_WARP = EPI_WARPS + 1
NUM_THREADS = 32 * (EPI_WARPS + 2)


# =================================================================== routing kernels
ROUTE_T = 256
ROUTE_R = 4


@cute.struct
class _HistStore:
    hist: cute.struct.MemRange[cutlass.Int32, 128]


@cute.struct
class _PosStore:
    pos: cute.struct.MemRange[cutlass.Int32, TOPK]


@cute.struct
class _ScanStore:
    tiles: cute.struct.MemRange[cutlass.Int32, 128]
    excl: cute.struct.MemRange[cutlass.Int32, 128]


@cute.kernel
def _route1_kernel(ids_ptr: cute.Pointer, counts_ptr: cute.Pointer, rs_ptr: cute.Pointer,
                   s: cutlass.Int32, m_cap: cutlass.Int32,
                   offset: cutlass.Int32, el: cutlass.Int32):
    """Per-expert histogram in shared memory, plus the per-row slot init."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    st = smem.allocate(_HistStore)
    hp = st.hist.data_ptr()
    sh = cute.make_tensor(hp, cute.make_layout(128))
    if tidx < 128:
        sh[tidx] = cutlass.Int32(0)
    cute.arch.sync_threads()
    mIds = cute.make_tensor(ids_ptr, cute.make_layout(s))
    mRS = cute.make_tensor(rs_ptr, cute.make_layout(m_cap))
    base = bidx * (ROUTE_T * ROUTE_R) + tidx * ROUTE_R
    for j in cutlass.range_constexpr(ROUTE_R):
        i = base + j
        if i < s:
            e = mIds[i] - offset
            if (e >= 0) and (e < el):
                cute.arch.atomic_add(hp + e, cutlass.Int32(1), scope="cta")
        if i < m_cap:
            mRS[i] = cutlass.Int32(-1)
    cute.arch.sync_threads()
    if tidx < el:
        v = sh[tidx]
        if v > 0:
            cute.arch.atomic_add(counts_ptr + tidx, v, scope="gpu")


@cute.kernel
def _route2_kernel(counts_ptr: cute.Pointer, cursor_ptr: cute.Pointer,
                   texp_ptr: cute.Pointer, meta_ptr: cute.Pointer,
                   el: cutlass.Int32, maxt: cutlass.Int32):
    """Single block: tile counts, exclusive scan, per-tile expert id, scatter cursors."""
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    st = smem.allocate(_ScanStore)
    stiles = cute.make_tensor(st.tiles.data_ptr(), cute.make_layout(128))
    sexcl = cute.make_tensor(st.excl.data_ptr(), cute.make_layout(128))
    c = cutlass.Int32(0)
    if tidx < el:
        c = cute.make_tensor(counts_ptr + tidx, cute.make_layout(1))[0]
    nt = (c + (MMA_M - 1)) // MMA_M
    stiles[tidx] = nt
    cute.arch.sync_threads()
    if tidx == 0:
        acc = cutlass.Int32(0)
        for e in cutlass.range(0, el, 1, unroll=1):
            sexcl[e] = acc
            acc = acc + stiles[e]
        cute.make_tensor(meta_ptr, cute.make_layout(4))[0] = acc
    cute.arch.sync_threads()
    if tidx < el:
        stt = sexcl[tidx]
        cute.make_tensor(cursor_ptr + tidx, cute.make_layout(1))[0] = stt * MMA_M
        mT = cute.make_tensor(texp_ptr, cute.make_layout(maxt))
        for j in cutlass.range(0, nt, 1, unroll=1):
            mT[stt + j] = tidx


@cute.kernel
def _scatter_gather_kernel(ids_ptr: cute.Pointer, cursor_ptr: cute.Pointer,
                           rs_ptr: cute.Pointer, hs_ptr: cute.Pointer, hss_ptr: cute.Pointer,
                           a_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
                           t: cutlass.Int32, offset: cutlass.Int32, el: cutlass.Int32):
    """One CTA per token: place its 4 routing slots, then broadcast the token's MXFP8
    row to all 4 expert-sorted destinations.  The row and its scale bytes are read once
    instead of four times, and no permutation index round-trips through global memory."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    st = smem.allocate(_PosStore)
    spos = cute.make_tensor(st.pos.data_ptr(), cute.make_layout(TOPK))
    if tidx < TOPK:
        i = bidx * TOPK + tidx
        p = cutlass.Int32(-1)
        e = cute.make_tensor(ids_ptr + i, cute.make_layout(1))[0] - offset
        if (e >= 0) and (e < el):
            p = cute.arch.atomic_add(cursor_ptr + e, cutlass.Int32(1), scope="gpu")
            cute.make_tensor(rs_ptr + p, cute.make_layout(1))[0] = i
        spos[tidx] = p
    cute.arch.sync_threads()
    # live columns only: FC1 peels its last K tile at I=2880, so a_perm[.., 2880:] is dead
    NV = cutlass.const_expr(I // 16)
    frag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
    for it in cutlass.range_constexpr((NV + GATHER_THREADS - 1) // GATHER_THREADS):
        v = tidx + it * GATHER_THREADS
        if v < NV:
            cute.autovec_copy(
                cute.make_tensor(hs_ptr + (cutlass.Int64(bidx) * (H1 // 4) + v * 4), cute.make_layout(4)), frag)
            for r in cutlass.range_constexpr(TOPK):
                row = spos[r]
                if row >= 0:
                    cute.autovec_copy(frag, cute.make_tensor(
                        a_ptr + (cutlass.Int64(row) * (K1 // 4) + v * 4), cute.make_layout(4)))
    if tidx < cutlass.const_expr(K1 // 128):
        v8 = cute.make_tensor(hss_ptr + (cutlass.Int64(bidx) * (H1 // 128) + tidx), cute.make_layout(1))[0]
        for r in cutlass.range_constexpr(TOPK):
            row = spos[r]
            if row >= 0:
                off = ((cutlass.Int64(row) // 128) * cutlass.const_expr(128 * (K1 // 128)) + tidx * 128
                       + (row % 32) * 4 + ((row % 128) // 32))
                cute.make_tensor(sfa_ptr + off, cute.make_layout(1))[0] = v8


@cute.jit
def _route(ids_ptr, counts_ptr, cursor_ptr, rs_ptr, texp_ptr, meta_ptr,
           hs_ptr, hss_ptr, a_ptr, sfa_ptr,
           s, m_cap, maxt, t, offset, el, stream):
    _route1_kernel(ids_ptr, counts_ptr, rs_ptr, s, m_cap, offset, el).launch(
        grid=[cute.ceil_div(m_cap, ROUTE_T * ROUTE_R), 1, 1], block=[ROUTE_T, 1, 1], stream=stream)
    _route2_kernel(counts_ptr, cursor_ptr, texp_ptr, meta_ptr, el, maxt).launch(
        grid=[1, 1, 1], block=[128, 1, 1], stream=stream)
    _scatter_gather_kernel(ids_ptr, cursor_ptr, rs_ptr, hs_ptr, hss_ptr, a_ptr, sfa_ptr,
                           t, offset, el).launch(
        grid=[t, 1, 1], block=[GATHER_THREADS, 1, 1], stream=stream)


# =================================================================== combine kernel
@cute.kernel
def _combine_kernel(y_ptr: cute.Pointer, mIds: cute.Tensor, mW: cute.Tensor,
                    out_ptr: cute.Pointer, t: cutlass.Int32,
                    offset: cutlass.Int32, el: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    ncol_blk = cutlass.const_expr(H // COMBINE_VEC)
    lin = cutlass.Int64(bidx) * COMBINE_THREADS + tidx
    if lin < cutlass.Int64(t) * ncol_blk:
        tok = lin // ncol_blk
        col = (lin % ncol_blk) * COMBINE_VEC
        acc = cute.make_rmem_tensor(cute.make_layout(COMBINE_VEC), cutlass.Float32)
        yv = cute.make_rmem_tensor(cute.make_layout(COMBINE_VEC), cutlass.BFloat16)
        ov = cute.make_rmem_tensor(cute.make_layout(COMBINE_VEC), cutlass.BFloat16)
        for j in cutlass.range_constexpr(COMBINE_VEC):
            acc[j] = cutlass.Float32(0.0)
        for k in cutlass.range_constexpr(TOPK):
            eid = mIds[tok, k] - offset
            if (eid >= 0) and (eid < el):
                w = mW[tok, k].to(cutlass.Float32)
                cute.autovec_copy(cute.make_tensor(
                    y_ptr + ((tok * TOPK + k) * H + col),
                    cute.make_layout(COMBINE_VEC)), yv)
                for j in cutlass.range_constexpr(COMBINE_VEC):
                    acc[j] = acc[j] + w * yv[j].to(cutlass.Float32)
        for j in cutlass.range_constexpr(COMBINE_VEC):
            ov[j] = acc[j].to(cutlass.BFloat16)
        cute.autovec_copy(ov, cute.make_tensor(out_ptr + (tok * H + col),
                                               cute.make_layout(COMBINE_VEC)))


@cute.jit
def _combine(y_ptr: cute.Pointer, ids_ptr: cute.Pointer, w_ptr: cute.Pointer,
             out_ptr: cute.Pointer, t: cutlass.Int32, offset: cutlass.Int32,
             el: cutlass.Int32, stream):
    mIds = cute.make_tensor(ids_ptr, cute.make_ordered_layout((t, TOPK), order=(1, 0)))
    mW = cute.make_tensor(w_ptr, cute.make_ordered_layout((t, TOPK), order=(1, 0)))
    total = cutlass.Int64(t) * (H // COMBINE_VEC)
    nblk = cute.ceil_div(total, COMBINE_THREADS)
    _combine_kernel(y_ptr, mIds, mW, out_ptr, t, offset, el).launch(
        grid=[nblk, 1, 1], block=[COMBINE_THREADS, 1, 1], stream=stream)


# =================================================================== block-scaled grouped GEMM
class MoeGemm:
    """Persistent warp-specialized tcgen05 block-scaled grouped GEMM.

    FC1 mode: epilogue = bias -> clamped SwiGLU -> MXFP8 requantization (per 32 logical columns).
    FC2 mode: epilogue = bias -> bf16 -> scatter to the (token, slot) row of the combine buffer.
    """

    def __init__(self, is_fc1: bool, blk_n: int, n_tiles: int, k_tiles: int,
                 ab_stages: int):
        self.is_fc1 = is_fc1
        self.blk_n = blk_n
        self.ab_stages = ab_stages
        # staging buffer for coalesced epilogue stores: one padded row per token
        # padded to keep 16B alignment and a conflict-free bank pattern
        # FC2 at N=256 would need a 64KB staging buffer; drain it in 128-column
        # rounds instead so the AB pipeline keeps its stages
        self.n_rounds = 1 if is_fc1 else max(1, blk_n // 128)
        # Int32 words of one output row in the staging buffer
        self.out_rowi32 = (blk_n // 8) if is_fc1 else (blk_n // 2 // self.n_rounds)
        self.out_stride = self.out_rowi32 + 4          # +4 words: conflict-free banks
        self.n_cgrp = blk_n // 64                      # FC1 MXFP8 scale groups per tile
        self.out_words = BLK_M * self.out_stride + ((BLK_M * self.n_cgrp) // 4 if is_fc1 else 0)
        self.out_pass = (BLK_M * self.out_rowi32) // (32 * EPI_WARPS * 4)
        self.n_tiles = n_tiles
        self.k_tiles = k_tiles
        self.acc_stages = 2 if blk_n <= 128 else 1
        self.a_dtype = cutlass.Float8E4M3FN
        self.b_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.TWO
        self.mma_tiler = (MMA_M, blk_n, BLK_K)
        self.mma_tiler_sfb = (BLK_M, cute.round_up(blk_n, 128), BLK_K)
        self.epi_tile = (BLK_M, EPI_N)
        self.epilog_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=32 * EPI_WARPS)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=32 * (EPI_WARPS + 1))
        # smem budget
        self.num_sfa_tmem = (BLK_M // 32) * (BLK_K // MMA_K)
        self.num_sfb_tmem = (cute.round_up(blk_n, 128) // 32) * (BLK_K // MMA_K)
        self.num_acc_tmem = blk_n * self.acc_stages

    def build(self):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, OperandMajorMode.K, OperandMajorMode.K,
            self.sf_dtype, SFVEC, self.cta_group, (MMA_M, self.blk_n))
        self.tiled_mma = tiled_mma
        self.tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, OperandMajorMode.K, OperandMajorMode.K,
            self.sf_dtype, SFVEC, tcgen05.CtaGroup.ONE,
            (BLK_M, cute.round_up(self.blk_n, 128)))
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.ab_stages)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Int8, self.ab_stages)
        self.sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SFVEC, self.ab_stages)
        self.sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SFVEC, self.ab_stages)
        return tiled_mma

    @cute.jit
    def __call__(self, a_ptr, sfa_ptr, b_ptr, sfb_ptr, bias_ptr,
                 o_ptr, osf_ptr, row_ptr, tile_ptr, meta_ptr,
                 ab_ptr, m_cap, el, round_up, num_ctas, stream):
        tiled_mma = self.build()

        nN = cutlass.const_expr(N1 if self.is_fc1 else H2)
        nK = cutlass.const_expr(K1 if self.is_fc1 else K2)
        nKB = cutlass.const_expr(H1 if self.is_fc1 else I_PAD)

        mA = cute.make_tensor(a_ptr, cute.make_ordered_layout(
            (cute.assume(m_cap, MMA_M), nK, 1), order=(1, 0, 2)))
        mB = cute.make_tensor(b_ptr, cute.make_ordered_layout(
            (nN, nKB, el), order=(1, 0, 2)))
        mSFA = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(
            (cute.assume(m_cap, MMA_M), nK, 1), SFVEC))
        mSFB = cute.make_tensor(sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(
            (nN, nKB, el), SFVEC))

        tiled_mma_sfb = self.tiled_mma_sfb
        cluster_mn = cutlass.const_expr((CTA_GROUP, 1))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_mn, tiled_mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_mn, tiled_mma.thr_id)
        # SFB carries the 2-CTA group so its barrier completion lands on the leader
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_mn, tiled_mma.thr_id)
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((CTA_GROUP, 1, 1)), (tiled_mma.thr_id.shape,))
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((CTA_GROUP, 1, 1)), (tiled_mma_sfb.thr_id.shape,))

        a_smem_one = cute.slice_(self.a_smem_layout, (None, None, None, 0))
        b_smem_one = cute.slice_(self.b_smem_layout, (None, None, None, 0))
        sfa_smem_one = cute.slice_(self.sfa_smem_layout, (None, None, None, 0))
        sfb_smem_one = cute.slice_(self.sfb_smem_layout, (None, None, None, 0))

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_one, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_one, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
            internal_type=cutlass.Int8)
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mSFA, sfa_smem_one, self.mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16)
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, mSFB, sfb_smem_one, self.mma_tiler_sfb, tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)

        num_tma_bytes = (cute.size_in_bytes(self.a_dtype, a_smem_one)
                         + cute.size_in_bytes(self.b_dtype, b_smem_one)
                         + cute.size_in_bytes(self.sf_dtype, sfa_smem_one)
                         + cute.size_in_bytes(self.sf_dtype, sfb_smem_one)) * CTA_GROUP

        @cute.struct
        class SharedStorage:
            ab_full: cute.struct.MemRange[cutlass.Int64, self.ab_stages]
            ab_empty: cute.struct.MemRange[cutlass.Int64, self.ab_stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.acc_stages]
            tmem_dealloc: cutlass.Int64
            tmem_hold: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout.outer)], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(self.b_smem_layout.outer)], 1024]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout)], 1024]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout)], 1024]
            sBias: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.blk_n], 128]
            sSlot: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, BLK_M], 128]
            sOut: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.out_words], 128]

        self.shared_storage = SharedStorage

        self.kernel(tiled_mma, tiled_mma_sfb, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
                    tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
                    bias_ptr, o_ptr, osf_ptr, row_ptr, tile_ptr, meta_ptr, ab_ptr,
                    cluster_layout_vmnk, cluster_layout_sfb_vmnk,
                    self.a_smem_layout, self.b_smem_layout,
                    self.sfa_smem_layout, self.sfb_smem_layout,
                    num_tma_bytes, m_cap, el, round_up).launch(
            grid=[num_ctas, 1, 1], block=[NUM_THREADS, 1, 1],
            cluster=[CTA_GROUP, 1, 1], stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma,
               tma_atom_a: cute.CopyAtom, mA: cute.Tensor,
               tma_atom_b: cute.CopyAtom, mB: cute.Tensor,
               tma_atom_sfa: cute.CopyAtom, mSFA: cute.Tensor,
               tma_atom_sfb: cute.CopyAtom, mSFB: cute.Tensor,
               bias_ptr: cute.Pointer, o_ptr: cute.Pointer, osf_ptr: cute.Pointer,
               row_ptr: cute.Pointer, tile_ptr: cute.Pointer, meta_ptr: cute.Pointer,
               ab_ptr: cute.Pointer, cluster_layout_vmnk: cute.Layout,
               cluster_layout_sfb_vmnk: cute.Layout,
               a_smem_layout, b_smem_layout, sfa_smem_layout, sfb_smem_layout,
               num_tma_bytes: cutlass.Constexpr, m_cap: cutlass.Int32,
               el: cutlass.Int32, round_up: cutlass.Int32):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        mma_v = bidx % CTA_GROUP
        is_leader = mma_v == 0
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)
        clu = bidx // CTA_GROUP
        nclu = gdim // CTA_GROUP

        NT = cutlass.const_expr(self.n_tiles)
        KT = cutlass.const_expr(self.k_tiles)

        mMeta = cute.make_tensor(meta_ptr, cute.make_layout(4))
        num_m_tiles = mMeta[0]
        total_work = num_m_tiles * NT
        mTileExp = cute.make_tensor(tile_ptr, cute.make_layout(m_cap // MMA_M))

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full.data_ptr(),
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=num_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * EPI_WARPS * CTA_GROUP),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_hold.ptr, barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0, is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc.ptr)
        pipeline_init_arrive(cluster_shape_mn=(CTA_GROUP, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)

        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB, cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                               (None, None, None))

        thr_mma = tiled_mma.get_slice(mma_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, coord_vmnk[2], a_cta_layout,
                                           cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, coord_vmnk[1], b_cta_layout,
                                           cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_sfa, coord_vmnk[2], a_cta_layout,
                                               cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_sfb, coord_sfb_vmnk[1], sfb_cta_layout,
                                               cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        sfb_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, coord_sfb_vmnk, mcast_mode=1)
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((self.mma_tiler[0], self.mma_tiler[1]))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))

        pipeline_init_wait(cluster_shape_mn=(CTA_GROUP, 1))

        # ------------------------------------------------------------------ TMA warp
        if warp_idx == TMA_WARP:
            ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stages)
            work = cutlass.Int32(clu)
            while work < total_work:
                m_tile = work // NT
                n_tile = work % NT
                expert = mTileExp[m_tile]
                tAgA_s = tAgA[(None, m_tile, None, 0)]
                tBgB_s = tBgB[(None, n_tile, None, expert)]
                tAgSFA_s = tAgSFA[(None, m_tile, None, 0)]
                tBgSFB_s = tBgSFB[(None, n_tile, None, expert)]
                for k_tile in cutlass.range(0, KT, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_prod)
                    bar = ab_pipeline.producer_get_barrier(ab_prod)
                    cute.copy(tma_atom_a, tAgA_s[(None, k_tile)],
                              tAsA[(None, ab_prod.index)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_b, tBgB_s[(None, k_tile)],
                              tBsB[(None, ab_prod.index)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, k_tile)],
                              tAsSFA[(None, ab_prod.index)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfb, tBgSFB_s[(None, k_tile)],
                              tBsSFB[(None, ab_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=sfb_mask)
                    ab_prod.advance()
                work += nclu
            ab_pipeline.producer_tail(ab_prod)

        # ------------------------------------------------------------------ MMA warp
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_acc_tmem, dtype=self.sf_dtype)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, SFVEC,
                cute.slice_(sfa_smem_layout, (None, None, None, 0))))
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_acc_tmem + self.num_sfa_tmem, dtype=self.sf_dtype)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, self.mma_tiler, SFVEC,
                cute.slice_(sfb_smem_layout, (None, None, None, 0))))

            cp_sfa, sSFA_s2t, tSFA_s2t = self._s2t(sSFA, tCtSFA)
            cp_sfb, sSFB_s2t, tSFB_s2t = self._s2t(sSFB, tCtSFB)

            ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stages)
            acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.acc_stages)
            work = cutlass.Int32(clu)
            while work < total_work:
                tCtAcc = tCtAcc_base[(None, None, None, acc_prod.index)]
                if is_leader:
                    acc_pipeline.producer_acquire(acc_prod)
                for k_tile in cutlass.range(0, KT - 1, 1, unroll=1):
                    if is_leader:
                        ab_pipeline.consumer_wait(ab_cons)
                        s2t_crd = (None, None, None, None, ab_cons.index)
                        cute.copy(cp_sfa, sSFA_s2t[s2t_crd], tSFA_s2t)
                        cute.copy(cp_sfb, sSFB_s2t[s2t_crd], tSFB_s2t)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        crd = (None, None, None, ab_cons.index)
                        cute.gemm(tiled_mma, tCtAcc, [tCrA[crd], tCtSFA],
                                  [tCrB[crd], tCtSFB], tCtAcc)
                        ab_pipeline.consumer_release(ab_cons)
                    ab_cons.advance()
                # final K tile: columns >= 2880 are guaranteed-zero padding, so only
                # K_LIVE_CHUNKS of its MMA_K chunks are issued
                if is_leader:
                    ab_pipeline.consumer_wait(ab_cons)
                    s2t_crd = (None, None, None, None, ab_cons.index)
                    cute.copy(cp_sfa, sSFA_s2t[s2t_crd], tSFA_s2t)
                    cute.copy(cp_sfb, sSFB_s2t[s2t_crd], tSFB_s2t)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kk in cutlass.range_constexpr(K_LIVE_CHUNKS):
                        cute.gemm(tiled_mma, tCtAcc,
                                  [tCrA[(None, None, kk, ab_cons.index)],
                                   tCtSFA[(None, None, kk)]],
                                  [tCrB[(None, None, kk, ab_cons.index)],
                                   tCtSFB[(None, None, kk)]], tCtAcc)
                    ab_pipeline.consumer_release(ab_cons)
                ab_cons.advance()
                if is_leader:
                    acc_pipeline.producer_commit(acc_prod)
                acc_prod.advance()
                work += nclu
            acc_pipeline.producer_tail(acc_prod)

        # ------------------------------------------------------------------ epilogue warps
        if warp_idx < MMA_WARP:
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (BLK_M, self.blk_n, BLK_K), utils.LayoutEnum.ROW_MAJOR,
                cutlass.Float32, self.acc_dtype, self.epi_tile, False)
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            epi_tid = tidx % 128
            epi_grp = tidx // 128
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            cId = cute.make_identity_tensor((BLK_M, self.blk_n))
            cId_epi = cute.flat_divide(cId, self.epi_tile)
            tTR_cId = thr_copy_t2r.partition_D(cId_epi)
            nsub_e = cutlass.const_expr(self.blk_n // EPI_N // EPI_GROUPS)
            tTR_rAcc = [
                cute.make_rmem_tensor(tTR_cId[(None, None, None, 0, 0)].shape, self.acc_dtype)
                for _ in range(nsub_e)
            ]

            sBias = storage.sBias.get_tensor(cute.make_layout(self.blk_n))
            self._epilogue(epi_tid, epi_grp, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, acc_pipeline,
                           sBias, storage.sBias.data_ptr(),
                           storage.sOut.data_ptr(), storage.sSlot.data_ptr(),
                           bias_ptr, o_ptr, osf_ptr, row_ptr, ab_ptr,
                           mTileExp, total_work, clu, nclu, mma_v, NT, round_up, m_cap, el)

            tmem.relinquish_alloc_permit()
            self.epilog_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    def _s2t(self, sSF, tSF):
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled = tcgen05.make_s2t_copy(atom, tc)
        thr = tiled.get_slice(0)
        s_part = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(sc))
        t_part = thr.partition_D(tc)
        return tiled, s_part, t_part

    @cute.jit
    def _epilogue(self, tidx, epi_grp, tiled_copy_t2r, tTR_tAcc, tTR_rAcc, acc_pipeline,
                  sBias, sbias_ptr, sout_ptr, sslot_ptr, bias_ptr, o_ptr, osf_ptr, row_ptr, ab_ptr, mTileExp,
                  total_work, clu, nclu, mma_v, NT, round_up, m_cap, el):
        NB = cutlass.const_expr(N1 if self.is_fc1 else H2)
        OSTR = cutlass.const_expr(self.out_stride)
        nsub = cutlass.const_expr(self.blk_n // EPI_N // EPI_GROUPS)
        sSlot = cute.make_tensor(sslot_ptr, cute.make_layout(BLK_M))
        mBias = cute.make_tensor(bias_ptr, cute.make_layout(el * NB))
        mRow = cute.make_tensor(row_ptr, cute.make_layout(m_cap))
        if cutlass.const_expr(self.is_fc1):
            mAB = cute.make_tensor(ab_ptr, cute.make_layout(3 * el))
            sSFb = cute.make_tensor(
                cute.recast_ptr(sout_ptr + (BLK_M * self.out_stride), dtype=cutlass.Uint8),
                cute.make_layout(self.n_cgrp * BLK_M))
            if cutlass.const_expr(self.n_cgrp == 4):
                osf32 = cute.make_tensor(cute.recast_ptr(osf_ptr, dtype=cutlass.Uint32),
                                         cute.make_layout(cutlass.Int64(m_cap) * (I_PAD // 128)))
            else:
                osf16 = cute.make_tensor(cute.recast_ptr(osf_ptr, dtype=cutlass.Uint16),
                                         cute.make_layout(cutlass.Int64(m_cap) * (I_PAD // 64)))

        yout = [cute.make_rmem_tensor(cute.make_layout(32), cutlass.BFloat16)
                for _ in range(0 if self.is_fc1 else nsub)]
        acc_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.acc_stages)
        work = cutlass.Int32(clu)
        while work < total_work:
            m_tile = work // NT
            n_tile = work % NT
            expert = mTileExp[m_tile]
            row = m_tile * MMA_M + mma_v * BLK_M + tidx
            bb = expert * NB
            alpha = cutlass.Float32(0.0)
            beta = cutlass.Float32(0.0)
            limit = cutlass.Float32(0.0)
            nlimit = cutlass.Float32(0.0)
            slot = cutlass.Int32(0)
            nalpha = cutlass.Float32(0.0)
            if cutlass.const_expr(self.is_fc1):
                alpha = mAB[expert]
                beta = mAB[el + expert]
                limit = mAB[2 * el + expert]
                nlimit = cutlass.Float32(0.0) - limit
                nalpha = alpha * cutlass.Float32(-1.4426950408889634)
            else:
                slot = mRow[row]

            self.epilog_barrier.arrive_and_wait()
            if cutlass.const_expr(self.blk_n == 128 * EPI_GROUPS):
                bi = n_tile * self.blk_n + epi_grp * 128 + tidx
                bv = cutlass.Float32(0.0)
                if bi < cutlass.const_expr(NB):
                    bv = mBias[bb + bi]
                sBias[epi_grp * 128 + tidx] = bv
                if cutlass.const_expr(not self.is_fc1):
                    if epi_grp == 0:
                        sSlot[tidx] = slot
            else:
                if epi_grp == 0:
                    sBias[tidx] = mBias[bb + n_tile * self.blk_n + tidx]
                    if cutlass.const_expr(not self.is_fc1):
                        sSlot[tidx] = slot
            self.epilog_barrier.arrive_and_wait()
            acc_pipeline.consumer_wait(acc_cons)
            stage = acc_cons.index
            # drain the whole TMEM accumulator into registers first, then release it:
            # with a single accumulator stage the next tile's MMA only waits for the
            # tcgen05.ld chain, not for the (much longer) epilogue arithmetic.
            for s_ in cutlass.range_constexpr(nsub):
                cute.copy(tiled_copy_t2r,
                          tTR_tAcc[(None, None, None, 0, epi_grp * nsub + s_, stage)],
                          tTR_rAcc[s_])
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_cons)
            if cutlass.const_expr(self.is_fc1):
                acts = [None] * 32
                rBias = cute.make_rmem_tensor(cute.make_layout(EPI_N), cutlass.Float32)
                for pg in cutlass.range_constexpr(nsub // 2):
                    amax = cutlass.Float32(0.0)
                    for half in cutlass.range_constexpr(2):
                        s_ = 2 * pg + half
                        nb = (epi_grp * nsub + s_) * EPI_N
                        rA = tTR_rAcc[s_]
                        # the 32 bias values of a subtile are contiguous and thread
                        # invariant: one vector read beats 32 broadcast LDS
                        cute.autovec_copy(cute.make_tensor(sbias_ptr + nb,
                                                           cute.make_layout(EPI_N)), rBias)
                        for v2 in cutlass.range_constexpr(2):
                            for u in cutlass.range_constexpr(8):
                                lu = v2 * 16 + u
                                j = 16 * half + 2 * u + v2
                                up = rA[lu] + rBias[lu]
                                gt = rA[lu + 8] + rBias[lu + 8]
                                # nvvm.fmax/fmin lower to one FMNMX each; the abs form
                                # (FMNMX.XORSIGN.ABS) folds |a| into the amax reduction,
                                # replacing a 4-instruction NaN-aware select sequence
                                up = cute.arch.fmin(cute.arch.fmax(up, nlimit), limit)
                                gt = cute.arch.fmin(gt, limit)
                                sg = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.arch.exp2(nalpha * gt))
                                a = ((up + beta) * gt) * sg
                                acts[j] = a
                                # sign of amax is irrelevant: only its exponent and
                                # mantissa-nonzero bits feed the UE8M0 scale
                                amax = cute.arch.fmax(amax, a, abs=True)
                    bits_a = amax.bitcast(cutlass.Int32)
                    eb_ocp = ((bits_a >> 23) & 0xFF) - 8
                    eb_ocp = eb_ocp & (~(eb_ocp >> 31))
                    tt = amax / cutlass.Float32(448.0)
                    bits_t = tt.bitcast(cutlass.Int32)
                    eb_ru = ((bits_t >> 23) & 0xFF) + (((bits_t & 0x7FFFFF) + 0x7FFFFF) >> 23)
                    ebias = eb_ocp + (eb_ru - eb_ocp) * round_up
                    inv = ((254 - ebias) << 23).bitcast(cutlass.Float32)
                    cg = epi_grp * cutlass.const_expr(nsub // 2) + pg
                    qf = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float8E4M3FN)
                    for j in cutlass.range_constexpr(32):
                        # cvt to e4m3 saturates to +-448, so no explicit clamp
                        qf[j] = (acts[j] * inv).to(cutlass.Float8E4M3FN)
                    q32 = cute.recast_tensor(qf, cutlass.Int32)
                    cute.autovec_copy(q32, cute.make_tensor(
                        sout_ptr + (tidx * OSTR + cg * 8), cute.make_layout(8)))
                    sSFb[tidx * cutlass.const_expr(self.n_cgrp) + cg] = ebias.to(cutlass.Uint8)
            else:
                rBias = cute.make_rmem_tensor(cute.make_layout(EPI_N), cutlass.Float32)
                for s_ in cutlass.range_constexpr(nsub):
                    sub = epi_grp * nsub + s_
                    nb = sub * EPI_N
                    rA = tTR_rAcc[s_]
                    cute.autovec_copy(cute.make_tensor(sbias_ptr + nb,
                                                       cute.make_layout(EPI_N)), rBias)
                    yb = yout[s_]
                    for i in cutlass.range_constexpr(32):
                        y = rA[i] + rBias[i]
                        yb[4 * (i % 8) + i // 8] = y.to(cutlass.BFloat16)
            acc_cons.advance()
            for rd in cutlass.range_constexpr(self.n_rounds):
              if cutlass.const_expr(self.n_rounds > 1):
                if rd > 0:
                    self.epilog_barrier.arrive_and_wait()
                if epi_grp == rd:
                    for s_ in cutlass.range_constexpr(nsub):
                        y32 = cute.recast_tensor(yout[s_], cutlass.Int32)
                        cute.autovec_copy(y32, cute.make_tensor(
                            sout_ptr + (tidx * OSTR + s_ * 16), cute.make_layout(16)))
              self.epilog_barrier.arrive_and_wait()
              if cutlass.const_expr(self.is_fc1):
                # lane ww owns prepared row rr; the 128x4 swizzle puts that row's scale
                # bytes at byte 4*ww of the tile's block, so consecutive lanes write
                # consecutive words of one contiguous 512B run
                ww = tidx
                rr = 32 * (ww % 4) + (ww // 4)
                if epi_grp == 0:
                    if cutlass.const_expr(self.n_cgrp == 4):
                        # the tile's four 32-col blocks are the four bytes of one word
                        b0 = (sSFb[rr * 4].to(cutlass.Int32) & 0xFF)
                        b1 = (sSFb[rr * 4 + 1].to(cutlass.Int32) & 0xFF)
                        b2 = (sSFb[rr * 4 + 2].to(cutlass.Int32) & 0xFF)
                        b3 = (sSFb[rr * 4 + 3].to(cutlass.Int32) & 0xFF)
                        wv = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                        osf32[(cutlass.Int64(m_tile) * CTA_GROUP + mma_v)
                              * cutlass.const_expr(128 * (I_PAD // 128))
                              + n_tile * 128 + ww] = wv.to(cutlass.Uint32)
                    else:
                        b0 = (sSFb[rr * 2].to(cutlass.Int32) & 0xFF)
                        b1 = (sSFb[rr * 2 + 1].to(cutlass.Int32) & 0xFF)
                        c0 = cutlass.Int32((2 * n_tile) % 4)
                        boff = ((cutlass.Int64(m_tile) * CTA_GROUP + mma_v) * (512 * (I_PAD // 128))
                                + (n_tile // 2) * 512
                                + ww * 4 + c0)
                        osf16[boff // 2] = (b0 | (b1 << 8)).to(cutlass.Uint16)
              # consecutive lanes cover consecutive 16B chunks of one output row
              ROWI = cutlass.const_expr(self.out_rowi32)
              RW = cutlass.const_expr(self.out_rowi32 * self.n_rounds)
              v4 = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
              for pp in cutlass.range_constexpr(self.out_pass):
                lin = cutlass.const_expr(pp * (32 * EPI_WARPS * 4)) + (epi_grp * 128 + tidx) * 4
                r = lin // ROWI
                c = lin % ROWI
                cute.autovec_copy(cute.make_tensor(
                    sout_ptr + (r * OSTR + c), cute.make_layout(4)), v4)
                if cutlass.const_expr(self.is_fc1):
                    cute.autovec_copy(v4, cute.make_tensor(
                        o_ptr + ((cutlass.Int64(m_tile) * MMA_M + mma_v * BLK_M + r) * (I_PAD // 4)
                                 + n_tile * ROWI + c), cute.make_layout(4)))
                else:
                    sl = cute.make_tensor(sslot_ptr + r, cute.make_layout(1))[0]
                    cw = cutlass.const_expr(rd * ROWI) + c
                    if (sl >= 0) and (n_tile * self.blk_n + cw * 2 < H):
                        cute.autovec_copy(v4, cute.make_tensor(
                            o_ptr + (cutlass.Int64(sl) * (H // 2) + n_tile * RW + cw),
                            cute.make_layout(4)))
            work += nclu


# =================================================================== host side
_COMPILED = {}
_COMPILE_LOCK = threading.Lock()
_FC1 = MoeGemm(True, BLK_N1, NT1, KT1, AB_STAGES1)
_FC2 = MoeGemm(False, BLK_N2, NT2, KT2, AB_STAGES2)


def _gp(t, dtype, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _compile(stream):
    global _COMPILED
    # Publish only a complete cache; serialize first-use compilation of the
    # mutable CuTe builder objects as well as cache initialization.
    with _COMPILE_LOCK:
        if not _COMPILED:
            _COMPILED = _compile_all(stream)


def _compile_all(stream):
    compiled = {}
    i32 = cutlass.Int32(0)
    z8 = make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=32)
    z32 = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    zf = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    zbf = make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16)
    za = make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    zb = make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=16)
    zsf = make_ptr(cutlass.Float8E8M0FNU, 0, cute.AddressSpace.gmem, assumed_align=32)
    compiled["route"] = cute.compile(_route, z32, z32, z32, z32, z32, z32,
                                      z32, z32, z32, z32,
                                      i32, i32, i32, i32, i32, i32, stream)
    compiled["fc1"] = cute.compile(_FC1, za, zsf, zb, zsf, zf, z32, z8, z32, z32, z32,
                                    zf, i32, i32, i32, i32, stream)
    compiled["fc2"] = cute.compile(_FC2, za, zsf, zb, zsf, zf, z32, z8, z32, z32, z32,
                                    zf, i32, i32, i32, i32, stream)
    compiled["combine"] = cute.compile(_combine, zbf, z32, zbf, zbf, i32, i32, i32, stream)

    return compiled

def _fc_rows(hidden_states, hidden_states_scale, topk_ids,
             gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
             gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
             local_expert_offset, requant_round_up):
    """Routing, gather, FC1 and FC2 of ``run``: (ybuf, stream). ybuf is [T * TOPK, H] bf16, row t * TOPK + k
    holding token t's k-th routed expert's FC2 output (bias included); rows of experts outside
    [offset, offset + EL) are not written."""
    T = hidden_states.shape[0]
    EL = gemm1_weights.shape[0]
    dev = hidden_states.device
    offset = int(local_expert_offset)
    rup = int(requant_round_up)
    S = T * TOPK
    M_CAP = ((S + MMA_M - 1) // MMA_M) * MMA_M + MMA_M * EL
    MAXT = M_CAP // MMA_M

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compile(stream)

    # ---------------- workspaces
    a_perm = torch.empty((M_CAP, K1), dtype=torch.uint8, device=dev)
    sfa_perm = torch.empty(M_CAP * (K1 // 32), dtype=torch.uint8, device=dev)
    aq = torch.empty((M_CAP, I_PAD), dtype=torch.uint8, device=dev)
    sfq = torch.empty(M_CAP * (I_PAD // 32), dtype=torch.uint8, device=dev)
    ybuf = torch.empty((S, H), dtype=torch.bfloat16, device=dev)
    ab = torch.stack([gemm1_alpha, gemm1_beta, gemm1_clamp_limit]).contiguous()

    # ---------------- routing metadata (device only, no host sync)
    counts = torch.zeros(EL, dtype=torch.int32, device=dev)
    cursor = torch.empty(EL, dtype=torch.int32, device=dev)
    row_slot = torch.empty(M_CAP, dtype=torch.int32, device=dev)
    tile_exp = torch.empty(MAXT, dtype=torch.int32, device=dev)
    meta = torch.zeros(4, dtype=torch.int32, device=dev)

    _COMPILED["route"](
        _gp(topk_ids, cutlass.Int32, 4), _gp(counts, cutlass.Int32, 4),
        _gp(cursor, cutlass.Int32, 4), _gp(row_slot, cutlass.Int32, 4),
        _gp(tile_exp, cutlass.Int32, 4), _gp(meta, cutlass.Int32, 4),
        _gp(hidden_states, cutlass.Int32), _gp(hidden_states_scale, cutlass.Int32, 4),
        _gp(a_perm, cutlass.Int32), _gp(sfa_perm, cutlass.Int32),
        cutlass.Int32(S), cutlass.Int32(M_CAP), cutlass.Int32(MAXT),
        cutlass.Int32(T), cutlass.Int32(offset), cutlass.Int32(EL), stream)

    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    ncta = (nsm // CTA_GROUP) * CTA_GROUP

    _COMPILED["fc1"](
        _gp(a_perm, cutlass.Float8E4M3FN), _gp(sfa_perm, cutlass.Float8E8M0FNU, 32),
        _gp(gemm1_weights, cutlass.Float4E2M1FN),
        _gp(gemm1_weights_scale, cutlass.Float8E8M0FNU, 32),
        _gp(gemm1_bias, cutlass.Float32), _gp(aq, cutlass.Int32),
        _gp(sfq, cutlass.Uint8, 32), _gp(row_slot, cutlass.Int32, 4),
        _gp(tile_exp, cutlass.Int32, 4), _gp(meta, cutlass.Int32, 4),
        _gp(ab, cutlass.Float32), cutlass.Int32(M_CAP), cutlass.Int32(EL),
        cutlass.Int32(rup), cutlass.Int32(ncta), stream)

    _COMPILED["fc2"](
        _gp(aq, cutlass.Float8E4M3FN), _gp(sfq, cutlass.Float8E8M0FNU, 32),
        _gp(gemm2_weights, cutlass.Float4E2M1FN),
        _gp(gemm2_weights_scale, cutlass.Float8E8M0FNU, 32),
        _gp(gemm2_bias, cutlass.Float32), _gp(ybuf, cutlass.Int32),
        _gp(sfq, cutlass.Uint8, 32), _gp(row_slot, cutlass.Int32, 4),
        _gp(tile_exp, cutlass.Int32, 4), _gp(meta, cutlass.Int32, 4),
        _gp(ab, cutlass.Float32), cutlass.Int32(M_CAP), cutlass.Int32(EL),
        cutlass.Int32(rup), cutlass.Int32(ncta), stream)
    return ybuf, stream


@torch.no_grad()
def run(hidden_states, hidden_states_scale, topk_ids, topk_weights,
        gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
        gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
        local_expert_offset, requant_round_up, output):
    T = hidden_states.shape[0]
    EL = gemm1_weights.shape[0]
    offset = int(local_expert_offset)
    ybuf, stream = _fc_rows(hidden_states, hidden_states_scale, topk_ids,
                            gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                            local_expert_offset, requant_round_up)

    _COMPILED["combine"](
        _gp(ybuf, cutlass.BFloat16), _gp(topk_ids, cutlass.Int32, 4),
        _gp(topk_weights, cutlass.BFloat16, 4), _gp(output, cutlass.BFloat16),
        cutlass.Int32(T), cutlass.Int32(offset), cutlass.Int32(EL), stream)


@torch.no_grad()
def run_unfinalized(hidden_states, hidden_states_scale, topk_ids,
                    gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                    gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                    local_expert_offset, requant_round_up):
    """``run`` without the combine: the [T * TOPK, H] bf16 FC2 rows, row t * TOPK + k = token t's k-th routed
    expert (bias included), for a caller that sums them itself."""
    ybuf, _ = _fc_rows(hidden_states, hidden_states_scale, topk_ids,
                       gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                       gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                       local_expert_offset, requant_round_up)
    return ybuf
