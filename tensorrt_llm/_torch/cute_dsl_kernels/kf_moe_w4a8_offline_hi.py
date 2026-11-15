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
# Alternative MoE kernel module for large mixed steps (it uses no programmatic dependent launch).
# Used only through the default-off TLLM_MOE_KF_DECODE hook, by
# tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_offline.
# GPT-OSS-120B MoE layer, MXFP4 weights x MXFP8 activations, pre-routed top-4 of 128 experts.
# CuTe DSL implementation targeting Blackwell-family tcgen05 block-scaled MMA
# (kind mxf8f6f4, E4M3 x E2M1, UE8M0 scales, K = 32 per instruction).
#
# Pipeline (all device side, CUDA-graph capturable, no host reads of device data):
#   1. route_count      : per-expert slot counts
#   2. route_scan       : exclusive scan -> per-expert row base, tile table
#   3. route_gather     : stable-ish scatter of (token, slot) -> permuted row + MXFP8 gather
#   4. fc1              : grouped block-scaled GEMM + bias + clamped SwiGLU + MXFP8 requant
#   5. fc2              : grouped block-scaled GEMM + bias -> bf16 scatter per (token, slot)
#   6. combine          : fp32 FMA chain over the 4 slots -> bf16 output

import math
from typing import Tuple, Type

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
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ------------------------------------------------------------------ problem constants
H = 2880            # model hidden / output width
I_PAD = 2944        # padded intermediate, FC2 K
H1 = 3072           # FC1 K extent of hidden_states
N1 = 5888           # FC1 prepared rows (2 * I_PAD)
H2 = 2944           # FC2 prepared rows
TOPK = 4
SF_VEC = 32

# Live extents actually consumed (everything above is guaranteed-zero padding).
FC1_K = 2944        # 23 k-tiles of 128 (columns 2944..3071 are zero on both operands)
FC2_K = 2944        # 23 k-tiles of 128
MTILE = 128
NTILE = 256
KTILE = 128
CLUSTER_N = 2
FC1_NTILES = N1 // NTILE     # 23 tiles of 256 cover all 5888 prepared rows
FC2_NTILES = 12              # 12*256 = 3072 >= 2944; tail is OOB -> TMA zero fill
YW2 = 2880 // 2              # int32 words per y row (bf16 width 2880 = logical H)
YQ = 2880 // 8               # 360 128-bit quads per y row
YQ_R = (YQ + 127) // 128     # 3 rounds of 128 threads
FC1_KTILES = FC1_K // KTILE  # 23
FC2_KTILES = FC2_K // KTILE  # 23

ACT_K = 2944                 # intermediate activation row width (bytes, e4m3)
ACT_SFC = ACT_K // SF_VEC    # 92 scale bytes per row
AQ = ACT_K // 16             # 184 128-bit quads per activation row
NQ = (AQ + 31) // 32         # 6 rounds of 32 lanes

E4M3 = cutlass.Float8E4M3FN
E2M1 = cutlass.Float4E2M1FN
UE8M0 = cutlass.Float8E8M0FNU


# ------------------------------------------------------------------ small ptx helpers
@dsl_user_op
def _bitcast_f2i(a, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bitcast_i2f(a, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fence_gpu(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        T.i32(), [], "fence.acq_rel.gpu;\n mov.u32 $0, 0;", "=r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ------------------------------------------------------------------ routing kernels
@cute.kernel
def _k_count(ids: cute.Tensor, cnt: cute.Tensor, ebase: cute.Tensor,
             tile_e: cute.Tensor, tile_nv: cute.Tensor, meta: cute.Tensor,
             cursor: cute.Tensor, S: cutlass.Int32,
             off: cutlass.Int32, EL: cutlass.Int32, MAXT: cutlass.Int32,
             nblk: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    lc = smem.allocate_tensor(cutlass.Int32, cute.make_layout(128), byte_alignment=16)
    sc0 = smem.allocate_tensor(cutlass.Int32, cute.make_layout(128), byte_alignment=16)
    sc1 = smem.allocate_tensor(cutlass.Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        lc[tidx] = cutlass.Int32(0)
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(4):
        i = bidx * 1024 + j * 256 + tidx
        if i < S:
            e = ids[i] - off
            if e >= 0:
                if e < EL:
                    cute.arch.atomic_add(lc.iterator + e, cutlass.Int32(1),
                                         scope="cta")
    cute.arch.sync_threads()
    if tidx < 128:
        v = lc[tidx]
        if v > 0:
            cute.arch.atomic_add(cnt.iterator + tidx, v, scope="gpu")
    # The block that finishes last runs the exclusive scan in place of a second
    # kernel launch: one less launch latency per call.
    _fence_gpu()
    cute.arch.sync_threads()
    if tidx == 0:
        lc[0] = cute.arch.atomic_add(meta.iterator + 1, cutlass.Int32(1),
                                     scope="gpu")
    cute.arch.sync_threads()
    last = lc[0]
    cute.arch.sync_threads()
    if last == nblk - 1:
        _fence_gpu()
        ntile = cutlass.Int32(0)
        if tidx < 128:
            c = cutlass.Int32(cutlass.select_(tidx < EL, cnt[tidx],
                                              cutlass.Int32(0)))
            cursor[tidx] = cutlass.Int32(0)
            ntile = (c + (MTILE - 1)) // MTILE
            sc0[tidx] = ntile
        cute.arch.sync_threads()
        # 7-step inclusive scan over the 128 per-expert tile counts; the serial
        # 128-iteration version of this loop dominated the kernel's runtime.
        for d in cutlass.range_constexpr(7):
            dd = 1 << d
            if tidx < 128:
                v = sc0[tidx]
                if tidx >= dd:
                    v += sc0[tidx - dd]
                sc1[tidx] = v
            cute.arch.sync_threads()
            if tidx < 128:
                sc0[tidx] = sc1[tidx]
            cute.arch.sync_threads()
        if tidx < 128:
            pre = sc0[tidx] - ntile
            tot = sc0[127]
            if tidx == 0:
                meta[0] = tot
            if tidx < EL:
                c2 = cutlass.Int32(cutlass.select_(tidx < EL, cnt[tidx],
                                                   cutlass.Int32(0)))
                ebase[tidx] = pre * MTILE
                nt2 = (c2 + (MTILE - 1)) // MTILE
                rem = c2
                for j in cutlass.range(nt2, unroll=1):
                    tile_e[pre + j] = tidx
                    tile_nv[pre + j] = cutlass.min(rem, cutlass.Int32(MTILE))
                    rem -= MTILE
            g = tot + tidx
            while g < MAXT:
                tile_e[g] = cutlass.Int32(-1)
                tile_nv[g] = cutlass.Int32(0)
                g += 128


@cute.kernel
def _k_scan(cnt: cute.Tensor, ebase: cute.Tensor, tile_e: cute.Tensor,
            tile_nv: cute.Tensor, meta: cute.Tensor, cursor: cute.Tensor,
            EL: cutlass.Int32, MAXT: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    s = smem.allocate_tensor(cutlass.Int32, cute.make_layout(128), byte_alignment=16)
    c = cutlass.Int32(cutlass.select_(tidx < EL, cnt[tidx], cutlass.Int32(0)))
    cursor[tidx] = cutlass.Int32(0)
    ntile = (c + (MTILE - 1)) // MTILE
    s[tidx] = ntile
    cute.arch.sync_threads()
    pre = cutlass.Int32(0)
    tot = cutlass.Int32(0)
    for j in cutlass.range(128, unroll=1):
        v = s[j]
        pre += cutlass.Int32(cutlass.select_(j < tidx, v, cutlass.Int32(0)))
        tot += v
    if tidx == 0:
        meta[0] = tot
    if tidx < EL:
        ebase[tidx] = pre * MTILE
        rem = c
        for j in cutlass.range(ntile, unroll=1):
            tile_e[pre + j] = tidx
            tile_nv[pre + j] = cutlass.min(rem, cutlass.Int32(MTILE))
            rem -= MTILE
    # mark the unused tail of the tile table invalid
    g = tot + tidx
    while g < MAXT:
        tile_e[g] = cutlass.Int32(-1)
        tile_nv[g] = cutlass.Int32(0)
        g += 128


@cute.kernel
def _k_gather(ids: cute.Tensor, hs4: cute.Tensor, hsf: cute.Tensor,
              ebase: cute.Tensor, cursor: cute.Tensor, row_src: cute.Tensor,
              aperm4: cute.Tensor, asf: cute.Tensor, yscr: cute.Tensor,
              act: cute.Tensor, actsf: cute.Tensor,
              S: cutlass.Int32, off: cutlass.Int32, EL: cutlass.Int32):
    # 256 threads: 8 warps, warp w gathers ALL TOPK slots of token bidx*8 + w,
    # so each token row of hidden_states is read once instead of TOPK times.
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if bidx == 0:
        for j in cutlass.range_constexpr(6):
            c = tidx + j * 256
            if c < YW2:
                yscr[S * YW2 + c] = cutlass.Int32(0)
    smem = utils.SmemAllocator()
    srow = smem.allocate_tensor(cutlass.Int32, cute.make_layout(32),
                                byte_alignment=16)
    if tidx < 32:
        i = bidx * 32 + tidx
        srow[tidx] = cutlass.Int32(-1)
        if i < S:
            e = ids[i] - off
            if e >= 0:
                if e < EL:
                    p = cute.arch.atomic_add(
                        cursor.iterator + e, cutlass.Int32(1), scope="gpu")
                    r = ebase[e] + p
                    row_src[r] = i
                    srow[tidx] = r
    cute.arch.sync_threads()
    w = tidx // 32
    lane = tidx % 32
    t = bidx * 8 + w
    if t < (S // TOPK):
        # one 128-bit quad per lane per round: 184 quads cover the 2944 B row
        sbase = t * (H1 // 16)         # quad index base in hidden_states
        buf = cute.make_rmem_tensor(cute.make_layout((NQ, 4), stride=(4, 1)),
                                    cutlass.Int32)
        for j in cutlass.range_constexpr(NQ):
            q4 = lane + j * 32
            if q4 < AQ:
                cute.autovec_copy(hs4[(sbase + q4, None)], buf[(j, None)])
        sfv = cute.make_rmem_tensor(cute.make_layout(3), cutlass.Uint8)
        for j in cutlass.range_constexpr(3):
            c = lane + j * 32
            if c < ACT_SFC:
                sfv[j] = hsf[t * (H1 // SF_VEC) + c]
        for q in cutlass.range_constexpr(TOPK):
            row = srow[w * TOPK + q]
            if row >= 0:
                dbase = row * AQ
                for j in cutlass.range_constexpr(NQ):
                    q4 = lane + j * 32
                    if q4 < AQ:
                        cute.autovec_copy(buf[(j, None)],
                                          aperm4[(dbase + q4, None)])
                qrow = row // 128
                rr = row % 128
                obase = qrow * (512 * 23) + (rr % 32) * 16 + (rr // 32) * 4
                for j in cutlass.range_constexpr(3):
                    c = lane + j * 32
                    if c < ACT_SFC:
                        asf[obase + (c // 4) * 512 + (c % 4)] = sfv[j]


@cute.kernel
def _k_combine(y: cute.Tensor, tw: cute.Tensor, ids: cute.Tensor,
               out: cute.Tensor, T_: cutlass.Int32, off: cutlass.Int32,
               EL: cutlass.Int32, S: cutlass.Int32):
    # one CTA of 128 threads per token; 2 bf16 lanes per iteration
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    t = bidx
    if t < T_:
        w0 = cutlass.Float32(tw[t * TOPK + 0].to(cutlass.Float32))
        w1 = cutlass.Float32(tw[t * TOPK + 1].to(cutlass.Float32))
        w2 = cutlass.Float32(tw[t * TOPK + 2].to(cutlass.Float32))
        w3 = cutlass.Float32(tw[t * TOPK + 3].to(cutlass.Float32))
        e0 = cutlass.Uint32(ids[t * TOPK + 0] - off)
        e1 = cutlass.Uint32(ids[t * TOPK + 1] - off)
        e2 = cutlass.Uint32(ids[t * TOPK + 2] - off)
        e3 = cutlass.Uint32(ids[t * TOPK + 3] - off)
        elu = cutlass.Uint32(EL)
        w0 = cutlass.Float32(cutlass.select_(e0 < elu, w0, cutlass.Float32(0.0)))
        w1 = cutlass.Float32(cutlass.select_(e1 < elu, w1, cutlass.Float32(0.0)))
        w2 = cutlass.Float32(cutlass.select_(e2 < elu, w2, cutlass.Float32(0.0)))
        w3 = cutlass.Float32(cutlass.select_(e3 < elu, w3, cutlass.Float32(0.0)))
        s0 = cutlass.Int32(cutlass.select_(e0 < elu, t * TOPK + 0, S))
        s1 = cutlass.Int32(cutlass.select_(e1 < elu, t * TOPK + 1, S))
        s2 = cutlass.Int32(cutlass.select_(e2 < elu, t * TOPK + 2, S))
        s3 = cutlass.Int32(cutlass.select_(e3 < elu, t * TOPK + 3, S))
        y0 = s0 * YW2
        y1 = s1 * YW2
        y2 = s2 * YW2
        y3 = s3 * YW2
        ob = t * (H // 2)
        NFULL = (H // 2) // 128          # 11 full rounds of 128 lanes
        rb = cute.make_rmem_tensor(cute.make_layout(NFULL), cutlass.Int32)
        for j in cutlass.range_constexpr(NFULL):
            c = tidx + j * 128
            rb[j] = _comb4(y[y0 + c], y[y1 + c], y[y2 + c], y[y3 + c],
                           w0, w1, w2, w3)
        for j in cutlass.range_constexpr(NFULL):
            out[ob + tidx + j * 128] = rb[j]
        c = tidx + NFULL * 128
        if c < (H // 2):
            out[ob + c] = _comb4(y[y0 + c], y[y1 + c], y[y2 + c], y[y3 + c],
                                 w0, w1, w2, w3)


@cute.jit
def _comb4(p0, p1, p2, p3, w0, w1, w2, w3) -> cutlass.Int32:
    lo = cutlass.Float32(0.0)
    hi = cutlass.Float32(0.0)
    lo = _fma(w0, _lo(p0), lo)
    hi = _fma(w0, _hi(p0), hi)
    lo = _fma(w1, _lo(p1), lo)
    hi = _fma(w1, _hi(p1), hi)
    lo = _fma(w2, _lo(p2), lo)
    hi = _fma(w2, _hi(p2), hi)
    lo = _fma(w3, _lo(p3), lo)
    hi = _fma(w3, _hi(p3), hi)
    return _pack_bf16(lo, hi)


@dsl_user_op
def _lo(p, *, loc=None, ip=None) -> cutlass.Float32:
    return _bitcast_i2f(cutlass.Int32(p) << 16)


@dsl_user_op
def _hi(p, *, loc=None, ip=None) -> cutlass.Float32:
    return _bitcast_i2f(cutlass.Int32(p) & cutlass.Int32(-65536))


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip),
             cutlass.Float32(c).ir_value(loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _pack_bf16(lo, hi, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
             cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _kslice(t, nk):
    """View a (V, M, K[, stage]) MMA fragment with only the first nk K blocks."""
    sh = t.shape
    st = t.stride
    if cute.rank(sh) == 4:
        lay = cute.make_layout((sh[0], sh[1], nk, sh[3]),
                               stride=(st[0], st[1], st[2], st[3]))
    else:
        lay = cute.make_layout((sh[0], sh[1], nk), stride=(st[0], st[1], st[2]))
    return cute.make_tensor(t.iterator, lay)


# ------------------------------------------------------------------ fused MoE GEMM
class MoeGemm:
    """One grouped block-scaled GEMM (mxf8f6f4: A = MXFP8 e4m3, B = MXFP4 e2m1).

    phase 1: FC1 -> bias + clamped SwiGLU + MXFP8 requantisation into (act, act_sf)
    phase 2: FC2 -> bias + bf16, scattered per (token, slot)
    """

    def __init__(self, phase: int, n_tiles: int, k_tiles: int,
                 n_total: int, bk_full: int, num_sm: int, grid_x: int = 0,
                 live_k: int = 2880):
        self.phase = phase
        self.live_k = live_k
        self.n_tiles = n_tiles
        self.k_tiles = k_tiles
        self.n_total = n_total
        self.bk_full = bk_full
        self.num_sm = num_sm
        self.grid_x = grid_x
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = SF_VEC
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_tiler = (MTILE, NTILE, KTILE)
        self.cluster_n = CLUSTER_N
        self.n_pairs = (n_tiles + self.cluster_n - 1) // self.cluster_n
        self.epi_warps = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * 6
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=160)
        self.num_acc_stage = 1       # one mbarrier pair, two overlapped buffers
        self.num_sfa_tmem_cols = (MTILE // 32) * (KTILE // 32)
        self.num_sfb_tmem_cols = (NTILE // 32) * (KTILE // 32)
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        # buffer 0 at col 0, buffer 1 at col NTILE - num_sf_tmem_cols: they share
        # the top SF-sized slice, which the epilogue drains first and releases early
        self.acc_stage_stride = NTILE - self.num_sf_tmem_cols
        self.num_accumulator_tmem_cols = NTILE + self.acc_stage_stride
        self.num_tmem_alloc_cols = 512
        self.epi_n = 64
        per_stage = (MTILE * KTILE) + (NTILE * KTILE) \
            + (MTILE * KTILE) // 32 + (NTILE * KTILE) // 32
        cap = utils.get_smem_capacity_in_bytes("sm_100")
        reserve = 3072 + 4 * NTILE + 4 * MTILE + (4 * MTILE * 33 if phase == 2 else 0)
        self.num_ab_stage = min(8, (cap - reserve) // per_stage)

    # ---------------------------------------------------------------- host entry
    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,       # permuted activations  [RPAD, ACT_K] e4m3
        sfa_ptr: cute.Pointer,     # swizzled activation scales
        b_ptr: cute.Pointer,       # expert weights [EL, N, Kfull] e2m1 (packed)
        sfb_ptr: cute.Pointer,     # expert weight scales (swizzled)
        bias_ptr: cute.Pointer,    # [EL, N] fp32
        out_ptr: cute.Pointer,     # phase1: act e4m3 [RPAD, ACT_K]; phase2: y bf16
        osf_ptr: cute.Pointer,     # phase1: act scales; phase2: unused (= osf alias)
        aux_ptr: cute.Pointer,     # phase1: alpha [EL]; phase2: row_src
        aux2_ptr: cute.Pointer,    # phase1: beta [EL]
        aux3_ptr: cute.Pointer,    # phase1: clamp limit [EL]
        tile_e_ptr: cute.Pointer,
        tile_nv_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        rpad: cutlass.Int32,
        el: cutlass.Int32,
        round_up: cutlass.Int32,
        nslots: cutlass.Int32,
        out_addr: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            E4M3, E2M1, OperandMajorMode.K, OperandMajorMode.K,
            UE8M0, self.sf_vec_size, self.cta_group, (MTILE, NTILE),
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            E4M3, E2M1, OperandMajorMode.K, OperandMajorMode.K,
            UE8M0, self.sf_vec_size, tcgen05.CtaGroup.ONE, (MTILE, NTILE),
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, self.cluster_n, 1)), (tiled_mma.thr_id.shape,)
        )

        a_layout = cute.make_layout((rpad, ACT_K, 1), stride=(ACT_K, 1, 0))
        a_tensor = cute.make_tensor(a_ptr, a_layout)
        n_total = self.n_total
        bk_full = self.bk_full
        b_layout = cute.make_layout(
            (n_total, bk_full, el), stride=(bk_full, 1, n_total * bk_full)
        )
        b_tensor = cute.make_tensor(b_ptr, b_layout)

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (rpad, ACT_K, 1), self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            (n_total, bk_full, el), self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, E4M3, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Int8, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, self.cluster_n), tiled_mma.thr_id),
            a_tensor, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, self.cluster_n), tiled_mma.thr_id),
            b_tensor, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int8,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, self.cluster_n), tiled_mma.thr_id),
            sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        # a second, non-multicast A/SFA atom: with a flat (m-tile, n-tile) work
        # list the two CTAs of a cluster occasionally straddle an m-tile boundary
        # and then cannot share one A tile
        cluster_layout_one = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        tma_atom_a1, tma_tensor_a1 = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
            a_tensor, a_smem_layout, self.mma_tiler, tiled_mma,
            cluster_layout_one.shape,
        )
        tma_atom_sfa1, tma_tensor_sfa1 = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id),
            sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            cluster_layout_one.shape, internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB((1, self.cluster_n), tiled_mma_sfb.thr_id),
            sfb_tensor, sfb_smem_layout, self.mma_tiler, tiled_mma_sfb,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )

        self.num_tma_load_bytes = (
            cute.size_in_bytes(E4M3, a_smem_layout)
            + cute.size_in_bytes(E2M1, b_smem_layout)
            + cute.size_in_bytes(UE8M0, sfa_smem_layout)
            + cute.size_in_bytes(UE8M0, sfb_smem_layout)
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sBias: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, NTILE], 128]
            sSlot: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, MTILE], 128]
            sEpi: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, MTILE * 33], 128]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    E4M3, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int8, cute.cosize(self.b_smem_layout_staged.outer)],
                self.buffer_align_bytes]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[UE8M0, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[UE8M0, cute.cosize(self.sfb_smem_layout_staged)],
                self.buffer_align_bytes]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, tiled_mma_sfb, self.cluster_layout_vmnk,
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
            tma_atom_a1, tma_tensor_a1, tma_atom_sfa1, tma_tensor_sfa1,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            bias_ptr, out_ptr, osf_ptr, aux_ptr, aux2_ptr, aux3_ptr,
            tile_e_ptr, tile_nv_ptr, meta_ptr,
            round_up, nslots, out_addr,
        ).launch(
            grid=[self.grid_x, self.cluster_n, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, self.cluster_n, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---------------------------------------------------------------- device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        tma_atom_a: cute.CopyAtom, mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB: cute.Tensor,
        tma_atom_a1: cute.CopyAtom, mA1: cute.Tensor,
        tma_atom_sfa1: cute.CopyAtom, mSFA1: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        osf_ptr: cute.Pointer,
        aux_ptr: cute.Pointer,
        aux2_ptr: cute.Pointer,
        aux3_ptr: cute.Pointer,
        tile_e_ptr: cute.Pointer,
        tile_nv_ptr: cute.Pointer,
        meta_ptr: cute.Pointer,
        round_up: cutlass.Int32,
        nslots: cutlass.Int32,
        out_addr: cutlass.Int64,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tidx, _, _ = cute.arch.thread_idx()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)

        mtile_cnt = cute.make_tensor(meta_ptr, cute.make_layout(4))[0]
        # flat (m-tile, n-tile) work list: with an odd n-tile count, pairing the
        # n tiles inside a 2-CTA cluster leaves one dead CTA slot per m-tile
        # (1/24 of FC1's SM time); a flat list only ever wastes the final slot.
        n_total_tiles = mtile_cnt * self.n_tiles
        n_work = (n_total_tiles + self.cluster_n - 1) // self.cluster_n

        tile_e = cute.make_tensor(tile_e_ptr, cute.make_layout(1 << 22))
        tile_nv = cute.make_tensor(tile_nv_ptr, cute.make_layout(1 << 22))

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.cluster_n),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epi_warps[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, self.cluster_n), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sBias = storage.sBias.get_tensor(cute.make_layout(NTILE))
        sSlot = storage.sSlot.get_tensor(cute.make_layout(MTILE))
        sEpi = storage.sEpi.get_tensor(cute.make_layout(MTILE * 33))

        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)),
                             (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)),
                             (None, None, None))
        gSFA = cute.local_tile(mSFA, cute.slice_(self.mma_tiler, (None, 0, None)),
                               (None, None, None))
        gSFB = cute.local_tile(mSFB, cute.slice_(self.mma_tiler, (0, None, None)),
                               (None, None, None))

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)

        one_layout = cute.make_layout(1)
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, blk_coord_vmnk, mcast_mode=2)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, blk_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, one_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa, blk_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, 0, one_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        gA1 = cute.local_tile(mA1, cute.slice_(self.mma_tiler, (None, 0, None)),
                              (None, None, None))
        gSFA1 = cute.local_tile(mSFA1, cute.slice_(self.mma_tiler, (None, 0, None)),
                                (None, None, None))
        tAsA1, tAgA1 = cpasync.tma_partition(
            tma_atom_a1, 0, one_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA1), 0, 3))
        tAsSFA1, tAgSFA1 = cpasync.tma_partition(
            tma_atom_sfa1, 0, one_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gSFA1), 0, 3))
        tAsSFA1 = cute.filter_zeros(tAsSFA1)
        tAgSFA1 = cute.filter_zeros(tAgSFA1)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C((MTILE, NTILE))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 2))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(tCtAcc_fake.stride[0], tCtAcc_fake.stride[1],
                        tCtAcc_fake.stride[2],
                        self.acc_stage_stride * tCtAcc_fake.stride[0][1])))

        pipeline_init_wait(cluster_shape_mn=(1, self.cluster_n))

        # ------------------------------------------------------------- TMA warp
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage)
            w = bidx
            while w < n_work:
                f0 = w * self.cluster_n
                fv = cutlass.min(f0 + cta_rank, n_total_tiles - 1)
                hi = cutlass.min(f0 + (self.cluster_n - 1), n_total_tiles - 1)
                shared_a = (f0 // self.n_tiles) == (hi // self.n_tiles)
                mt = fv // self.n_tiles
                nt = fv % self.n_tiles
                e = tile_e[mt]
                if e >= 0:
                    tBgB_s = tBgB[(None, nt, None, e)]
                    tBgSFB_s = tBgSFB[(None, nt, None, e)]
                    if shared_a:
                        tAgA_s = tAgA[(None, mt, None, 0)]
                        tAgSFA_s = tAgSFA[(None, mt, None, 0)]
                        for k in cutlass.range(self.k_tiles, unroll=1):
                            ab_pipeline.producer_acquire(ab_producer_state)
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cute.copy(tma_atom_a, tAgA_s[(None, k)],
                                      tAsA[(None, ab_producer_state.index)],
                                      tma_bar_ptr=bar, mcast_mask=a_mcast_mask)
                            cute.copy(tma_atom_b, tBgB_s[(None, k)],
                                      tBsB[(None, ab_producer_state.index)], tma_bar_ptr=bar)
                            cute.copy(tma_atom_sfa, tAgSFA_s[(None, k)],
                                      tAsSFA[(None, ab_producer_state.index)],
                                      tma_bar_ptr=bar, mcast_mask=a_mcast_mask)
                            cute.copy(tma_atom_sfb, tBgSFB_s[(None, k)],
                                      tBsSFB[(None, ab_producer_state.index)], tma_bar_ptr=bar)
                            ab_producer_state.advance()
                    else:
                        tAgA1_s = tAgA1[(None, mt, None, 0)]
                        tAgSFA1_s = tAgSFA1[(None, mt, None, 0)]
                        for k in cutlass.range(self.k_tiles, unroll=1):
                            ab_pipeline.producer_acquire(ab_producer_state)
                            bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                            cute.copy(tma_atom_a1, tAgA1_s[(None, k)],
                                      tAsA1[(None, ab_producer_state.index)],
                                      tma_bar_ptr=bar)
                            cute.copy(tma_atom_b, tBgB_s[(None, k)],
                                      tBsB[(None, ab_producer_state.index)], tma_bar_ptr=bar)
                            cute.copy(tma_atom_sfa1, tAgSFA1_s[(None, k)],
                                      tAsSFA1[(None, ab_producer_state.index)],
                                      tma_bar_ptr=bar)
                            cute.copy(tma_atom_sfb, tBgSFB_s[(None, k)],
                                      tBsSFB[(None, ab_producer_state.index)], tma_bar_ptr=bar)
                            ab_producer_state.advance()
                w += gdim
            ab_pipeline.producer_tail(ab_producer_state)

        # ------------------------------------------------------------- MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=UE8M0)
            tCtSFA = cute.make_tensor(
                sfa_tmem_ptr,
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma, self.mma_tiler, self.sf_vec_size,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))))
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=UE8M0)
            tCtSFB = cute.make_tensor(
                sfb_tmem_ptr,
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma, self.mma_tiler, self.sf_vec_size,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))))

            cp_sfa, sSFA_s2t, tSFA_s2t = self._s2t(sSFA, tCtSFA)
            cp_sfb, sSFB_s2t, tSFB_s2t = self._s2t(sSFB, tCtSFB)
            # hidden / intermediate columns >= 2880 are guaranteed zero, so the
            # final K tile only needs its first two K=32 blocks
            nk_tail = (self.live_k - (self.k_tiles - 1) * KTILE) // 32
            tCrA_t = _kslice(tCrA, nk_tail)
            tCrB_t = _kslice(tCrB, nk_tail)
            tCtSFA_t = _kslice(tCtSFA, nk_tail)
            tCtSFB_t = _kslice(tCtSFB, nk_tail)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)
            pp = cutlass.Int32(0)
            w = bidx
            while w < n_work:
                f = w * self.cluster_n + cta_rank
                mt = cutlass.min(f, n_total_tiles - 1) // self.n_tiles
                e = tile_e[mt]
                if e >= 0:
                    tCtAcc = tCtAcc_base[(None, None, None, pp)]
                    pp = cutlass.Int32(1) - pp
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    # n tiles past the live range are a cluster-pairing artefact:
                    # their epilogue stores nothing, so run the pipeline without
                    # spending tensor-core issue slots on them.
                    live_tile = f < n_total_tiles
                    for k in cutlass.range(self.k_tiles, unroll=1):
                        ab_pipeline.consumer_wait(ab_consumer_state)
                        sc = (None, None, None, None, ab_consumer_state.index)
                        if live_tile:
                            cute.copy(cp_sfa, sSFA_s2t[sc], tSFA_s2t)
                            cute.copy(cp_sfb, sSFB_s2t[sc], tSFB_s2t)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k != 0)
                            tc = (None, None, None, ab_consumer_state.index)
                            if k == self.k_tiles - 1:
                                cute.gemm(tiled_mma, tCtAcc,
                                          [tCrA_t[tc], tCtSFA_t],
                                          [tCrB_t[tc], tCtSFB_t], tCtAcc)
                            else:
                                cute.gemm(tiled_mma, tCtAcc,
                                          [tCrA[tc], tCtSFA], [tCrB[tc], tCtSFB],
                                          tCtAcc)
                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                w += gdim
            acc_pipeline.producer_tail(acc_producer_state)

        # ------------------------------------------------------------- epilogue warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            epi_tile = (MTILE, self.epi_n)
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (MTILE, NTILE, KTILE), utils.LayoutEnum.ROW_MAJOR,
                cutlass.Float32, self.acc_dtype, epi_tile, False)
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            cC = cute.make_identity_tensor((MTILE, self.epi_n))
            tTR_cC = thr_copy_t2r.partition_D(cC)
            rAcc = cute.make_rmem_tensor(tTR_cC.shape, self.acc_dtype)

            biasT = cute.make_tensor(bias_ptr, cute.make_layout(1 << 24))
            auxT = cute.make_tensor(aux_ptr, cute.make_layout(1 << 26))
            aux2T = cute.make_tensor(aux2_ptr, cute.make_layout(128))
            aux3T = cute.make_tensor(aux3_ptr, cute.make_layout(128))
            osfT = cute.make_tensor(osf_ptr, cute.make_layout(1 << 28))

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            epp = cutlass.Int32(0)
            w = bidx
            while w < n_work:
                f = w * self.cluster_n + cta_rank
                fv = cutlass.min(f, n_total_tiles - 1)
                mt = fv // self.n_tiles
                nt = fv % self.n_tiles
                e = tile_e[mt]
                if e >= 0:
                    bbase = e * self.n_total + nt * NTILE
                    for bj in cutlass.range_constexpr(NTILE // 128):
                        bo = nt * NTILE + tidx + bj * 128
                        bo = cutlass.Int32(cutlass.select_(
                            bo < self.n_total, bo, cutlass.Int32(0)))
                        sBias[tidx + bj * 128] = biasT[e * self.n_total + bo]
                    alpha = cutlass.Float32(0.0)
                    beta = cutlass.Float32(0.0)
                    limit = cutlass.Float32(0.0)
                    if cutlass.const_expr(self.phase == 1):
                        alpha = auxT[e]
                        beta = aux2T[e]
                        limit = aux3T[e]
                    nv = cutlass.Int32(cutlass.select_(
                        f < n_total_tiles, tile_nv[mt], cutlass.Int32(0)))
                    row = mt * MTILE + tidx
                    if cutlass.const_expr(self.phase == 2):
                        sSlot[tidx] = cutlass.Int32(cutlass.select_(
                            tidx < nv, auxT[row], cutlass.Int32(-1)))
                    self.epilog_sync_barrier.arrive_and_wait()
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_acc_st = tTR_tAcc[(None, None, None, None, None, epp)]
                    rev = epp == 0
                    epp = cutlass.Int32(1) - epp
                    NSUB = NTILE // self.epi_n
                    sfword = cutlass.Int32(0)
                    for si in cutlass.range_constexpr(NSUB):
                        sub = cutlass.Int32(cutlass.select_(
                            rev, cutlass.Int32(NSUB - 1 - si), cutlass.Int32(si)))
                        cute.copy(tiled_copy_t2r,
                                  tTR_acc_st[(None, None, None, 0, sub)], rAcc)
                        if cutlass.const_expr(si == 0):
                            # the two buffers only share this first-drained slice
                            cute.arch.fence_view_async_tmem_load()
                            acc_pipeline.consumer_release(acc_consumer_state)
                            acc_consumer_state.advance()
                        if cutlass.const_expr(self.phase == 1):
                            eb = self._epi_fc1(rAcc, sBias, sub, row, nt, alpha,
                                               beta, limit, round_up, out_addr,
                                               osfT, tidx < nv)
                            sfword = sfword | (eb << (8 * sub))
                        else:
                            self._epi_fc2(rAcc, sBias, sub, nt, tidx,
                                          sSlot, sEpi, out_addr)
                    if cutlass.const_expr(self.phase == 1):
                        if tidx < nv:
                            osfT[(row // 128) * (128 * (ACT_SFC // 4))
                                 + nt * 128 + (row % 128 % 32) * 4
                                 + (row % 128 // 32)] = sfword
                    self.epilog_sync_barrier.arrive_and_wait()
                w += gdim
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)

    # ---------------------------------------------------------------- helpers
    def _s2t(self, sSF: cute.Tensor, tSF: cute.Tensor):
        sc = cute.filter_zeros(sSF)
        tc = cute.filter_zeros(tSF)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), UE8M0)
        tiled = tcgen05.make_s2t_copy(atom, tc)
        thr = tiled.get_slice(0)
        s_part = tcgen05.get_s2t_smem_desc_tensor(tiled, thr.partition_S(sc))
        t_part = thr.partition_D(tc)
        return tiled, s_part, t_part

    @cute.jit
    def _epi_fc1(self, rAcc, sBias, sub, row, nt, alpha, beta, limit,
                 round_up, out_addr, osfT, live) -> cutlass.Int32:
        LOG2E = 1.4426950408889634
        g = nt * (NTILE // 64) + sub            # logical 32-channel group index
        a_scaled = alpha * cutlass.Float32(-LOG2E)
        act = cute.make_rmem_tensor(cute.make_layout(32), cutlass.Float32)
        amax = cutlass.Float32(0.0)
        ebits = cutlass.Int32(-127)
        rscale = cutlass.Float32(0.0)
        # rows past the expert's live count are tile padding: at 160-300 rows per
        # expert roughly a quarter of all epilogue rows, and none of their MUFU /
        # ALU work is ever stored.  Channel groups >= 90 are past the live
        # intermediate width and FC2's K loop stops at 2880, so they are dead too.
        if live and g < (2880 // SF_VEC):
          _add_bias(rAcc, sBias, sub)
          for c in cutlass.range_constexpr(32):
            bq = (c // 16) * 32
            tt = c % 16
            p_up = bq + 16 * (tt % 2) + (tt // 2)
            p_gate = p_up + 8
            up = rAcc[p_up]
            gate = rAcc[p_gate]
            up = cute.arch.fmax(cute.arch.fmin(up, limit), -limit)
            gate = cute.arch.fmin(gate, limit)
            sig = cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.arch.exp2(gate * a_scaled))
            v = (up + beta) * gate * sig
            act[c] = v
            av = cute.arch.fmax(v, -v)
            amax = cute.arch.fmax(amax, av)
          # MXFP8 scale exponent
          yv = amax / cutlass.Float32(448.0)
          ybits = _bitcast_f2i(yv)
          eru = ((ybits >> 23) & 255) - 126
          eru = cutlass.Int32(cutlass.select_(
              (ybits & 8388607) == 0, eru - 1, eru))
          abits = _bitcast_f2i(amax)
          eocp = ((abits >> 23) & 255) - 135
          ebits = cutlass.Int32(cutlass.select_(round_up != 0, eru, eocp))
          ebits = cutlass.Int32(cutlass.select_(
              amax == cutlass.Float32(0.0), cutlass.Int32(-127), ebits))
          ebits = cutlass.max(cutlass.min(ebits, cutlass.Int32(127)),
                              cutlass.Int32(-127))
          rscale = _bitcast_i2f((127 - ebits) << 23)
          base = row * (ACT_K // 4) + g * 8
          if True:
            pk = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Int32)
            for c2 in cutlass.range_constexpr(8):
                pk[c2] = _pack_e4m3x4(act[4 * c2 + 0] * rscale,
                                      act[4 * c2 + 1] * rscale,
                                      act[4 * c2 + 2] * rscale,
                                      act[4 * c2 + 3] * rscale)
            ad = out_addr + base.to(cutlass.Int64) * 4
            _st_v4(ad, pk[0], pk[1], pk[2], pk[3])
            _st_v4(ad + 16, pk[4], pk[5], pk[6], pk[7])
        return (ebits + 127) & 255



    @cute.jit
    def _epi_fc2(self, rAcc, sBias, sub, nt, tidx, sSlot, sEpi, out_addr):
        # Each thread owns one accumulator row = one (token, slot) pair, and its
        # 64 columns are 128 contiguous bytes of that slot's y row, so it stores
        # them directly as eight 16 B stores.  That removes the SMEM transpose
        # and both of its CTA-wide barriers per sub-tile (8 per output tile) and
        # makes tile-padding rows free: they simply store nothing.
        if (nt * NTILE + sub * 64) < 2880:
            # tile-padding rows store nothing, so they do not need the bias add
            # or the bf16 pack either
            if sSlot[tidx] >= 0:
                _add_bias(rAcc, sBias, sub)
                for c2 in cutlass.range_constexpr(32):
                    c = c2 * 2
                    p0 = (c // 32) * 32 + 8 * ((c % 32) % 4) + ((c % 32) // 4)
                    c = c2 * 2 + 1
                    p1 = (c // 32) * 32 + 8 * ((c % 32) % 4) + ((c % 32) // 4)
                    sEpi[tidx * 33 + c2] = _pack_bf16(rAcc[p0], rAcc[p1])
            self.epilog_sync_barrier.arrive_and_wait()
            # 8 threads per row, 16 B each -> 128 contiguous bytes per row
            r0 = tidx // 8
            c16 = tidx % 8
            colbase = (nt * NTILE + sub * 64) // 2 + c16 * 4
            for it in cutlass.range_constexpr(8):
                r = it * 16 + r0
                sl = sSlot[r]
                if sl >= 0:
                    b = r * 33 + c16 * 4
                    _st_v4(out_addr + (sl * YW2 + colbase).to(cutlass.Int64) * 4,
                           sEpi[b], sEpi[b + 1], sEpi[b + 2], sEpi[b + 3])
            self.epilog_sync_barrier.arrive_and_wait()
        else:
            self.epilog_sync_barrier.arrive_and_wait()
            self.epilog_sync_barrier.arrive_and_wait()


@cute.jit
def _add_bias(rAcc, sBias, sub):
    """rAcc[i] += bias[sub*64 + i] with 128-bit shared loads."""
    sbv = cute.make_tensor(sBias.iterator + sub * 64,
                           cute.make_layout((16, 4), stride=(4, 1)))
    rb = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    for q in cutlass.range_constexpr(16):
        cute.autovec_copy(sbv[(q, None)], rb)
        for i in cutlass.range_constexpr(4):
            rAcc[4 * q + i] = rAcc[4 * q + i] + rb[i]


@dsl_user_op
def _st_v4(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        T.i32(),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Int32(c).ir_value(loc=loc, ip=ip),
         cutlass.Int32(d).ir_value(loc=loc, ip=ip)],
        "st.global.v4.b32 [$1], {$2,$3,$4,$5};\n mov.u32 $0, 0;",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(b).ir_value(loc=loc, ip=ip),
             cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(d).ir_value(loc=loc, ip=ip),
             cutlass.Float32(c).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 lo, hi;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 lo, $1, $2;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 hi, $3, $4;\n"
            "  mov.b32 $0, {lo, hi}; }",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# ------------------------------------------------------------------ host launchers
@cute.jit
def _launch_route(ids_ptr, hs_ptr, hsf_ptr, cnt_ptr, ebase_ptr, tile_e_ptr,
                  tile_nv_ptr, meta_ptr, cursor_ptr, row_src_ptr,
                  aperm_ptr, asf_ptr, y_ptr, act_ptr, actsf_ptr,
                  S: cutlass.Int32, off: cutlass.Int32, EL: cutlass.Int32,
                  MAXT: cutlass.Int32, nb_cnt: cutlass.Int32,
                  nb_gat: cutlass.Int32, stream: cuda.CUstream):
    big = cute.make_layout(1 << 26)
    ids = cute.make_tensor(ids_ptr, big)
    cnt = cute.make_tensor(cnt_ptr, cute.make_layout(128))
    ebase = cute.make_tensor(ebase_ptr, cute.make_layout(128))
    cursor = cute.make_tensor(cursor_ptr, cute.make_layout(128))
    tile_e = cute.make_tensor(tile_e_ptr, big)
    tile_nv = cute.make_tensor(tile_nv_ptr, big)
    meta = cute.make_tensor(meta_ptr, cute.make_layout(4))
    row_src = cute.make_tensor(row_src_ptr, big)
    hs4 = cute.make_tensor(hs_ptr, cute.make_layout((1 << 24, 4), stride=(4, 1)))
    hsf = cute.make_tensor(hsf_ptr, cute.make_layout(1 << 26))
    aperm4 = cute.make_tensor(aperm_ptr, cute.make_layout((1 << 24, 4), stride=(4, 1)))
    asf = cute.make_tensor(asf_ptr, cute.make_layout(1 << 26))

    _k_count(ids, cnt, ebase, tile_e, tile_nv, meta, cursor,
             S, off, EL, MAXT, nb_cnt).launch(
        grid=[nb_cnt, 1, 1], block=[256, 1, 1], stream=stream)
    yscr = cute.make_tensor(y_ptr, cute.make_layout(1 << 30))
    act = cute.make_tensor(act_ptr, cute.make_layout(1 << 28))
    actsf = cute.make_tensor(actsf_ptr, cute.make_layout(1 << 26))
    _k_gather(ids, hs4, hsf, ebase, cursor, row_src, aperm4, asf, yscr,
              act, actsf, S, off, EL).launch(
        grid=[nb_gat, 1, 1], block=[256, 1, 1], stream=stream)


@cute.jit
def _launch_combine(y_ptr, tw_ptr, ids_ptr, out_ptr, T_: cutlass.Int32,
                    off: cutlass.Int32, EL: cutlass.Int32, S: cutlass.Int32,
                    nb: cutlass.Int32, stream: cuda.CUstream):
    big = cute.make_layout(1 << 30)
    y = cute.make_tensor(y_ptr, big)
    tw = cute.make_tensor(tw_ptr, cute.make_layout(1 << 26))
    ids = cute.make_tensor(ids_ptr, cute.make_layout(1 << 26))
    out = cute.make_tensor(out_ptr, big)
    _k_combine(y, tw, ids, out, T_, off, EL, S).launch(
        grid=[nb, 1, 1], block=[128, 1, 1], stream=stream)


_CACHE = {}


def _nclusters(num_sm):
    """Resident cluster count for the persistent grid.

    A persistent kernel with a static work split must fit in one wave: if even
    one cluster is left over it runs a whole second wave and nearly doubles the
    makespan.  cuOccupancyMaxActiveClusters accounts for the GPC packing.
    """
    key = ("nclu", num_sm)
    if key not in _CACHE:
        n = num_sm // CLUSTER_N
        try:
            n = min(n, utils.HardwareInfo().get_max_active_clusters(CLUSTER_N))
        except Exception:
            pass
        _CACHE[key] = max(1, n)
    return _CACHE[key]


def _p(dtype, t, align=256):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def _ip(dtype, addr, align=256):
    return make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)


def run(hidden_states, hidden_states_scale, topk_ids, topk_weights,
        gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha,
        gemm1_beta, gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale,
        gemm2_bias, local_expert_offset, requant_round_up, output):
    dev = hidden_states.device
    T = hidden_states.shape[0]
    EL = gemm1_weights.shape[0]
    S = T * TOPK
    MAXT = EL + (S + MTILE - 1) // MTILE
    RPAD = MAXT * MTILE
    off = int(local_expert_offset) if not torch.is_tensor(local_expert_offset) \
        else int(local_expert_offset.item())
    ru = int(requant_round_up) if not torch.is_tensor(requant_round_up) \
        else int(requant_round_up.item())

    num_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    i32 = torch.int32

    small = torch.zeros(4 + 3 * 128, dtype=i32, device=dev)
    sp = small.data_ptr()
    meta_p, cnt_p, ebase_p, cursor_p = sp, sp + 16, sp + 528, sp + 1040
    tile_e = torch.empty(MAXT, dtype=i32, device=dev)
    tile_nv = torch.empty(MAXT, dtype=i32, device=dev)
    row_src = torch.empty(RPAD, dtype=i32, device=dev)
    aperm = torch.empty(RPAD * ACT_K, dtype=torch.uint8, device=dev)
    asf = torch.empty(RPAD * ACT_SFC, dtype=torch.uint8, device=dev)
    act = torch.empty(RPAD * ACT_K, dtype=torch.uint8, device=dev)
    actsf = torch.empty(RPAD * ACT_SFC, dtype=torch.uint8, device=dev)
    yb = torch.empty((S + 1) * 2880, dtype=torch.bfloat16, device=dev)

    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)

    def route_args(nb_cnt, nb_gat):
        return (
            _ip(cutlass.Int32, topk_ids.data_ptr()),
            _ip(cutlass.Int32, hidden_states.data_ptr()),
            _ip(cutlass.Uint8, hidden_states_scale.data_ptr()),
            _ip(cutlass.Int32, cnt_p, 16), _ip(cutlass.Int32, ebase_p, 16),
            _ip(cutlass.Int32, tile_e.data_ptr()),
            _ip(cutlass.Int32, tile_nv.data_ptr()),
            _ip(cutlass.Int32, meta_p, 16), _ip(cutlass.Int32, cursor_p, 16),
            _ip(cutlass.Int32, row_src.data_ptr()),
            _ip(cutlass.Int32, aperm.data_ptr()), _ip(cutlass.Uint8, asf.data_ptr()),
            _ip(cutlass.Int32, yb.data_ptr()),
            _ip(cutlass.Int32, act.data_ptr()), _ip(cutlass.Uint8, actsf.data_ptr()),
            cutlass.Int32(S), cutlass.Int32(off), cutlass.Int32(EL),
            cutlass.Int32(MAXT), cutlass.Int32(nb_cnt), cutlass.Int32(nb_gat), stream)

    def fc1_args():
        return (
            _ip(E4M3, aperm.data_ptr()), _ip(UE8M0, asf.data_ptr()),
            _ip(E2M1, gemm1_weights.data_ptr()),
            _ip(UE8M0, gemm1_weights_scale.data_ptr()),
            _ip(cutlass.Float32, gemm1_bias.data_ptr()),
            _ip(cutlass.Int32, act.data_ptr()), _ip(cutlass.Int32, actsf.data_ptr()),
            _ip(cutlass.Float32, gemm1_alpha.data_ptr()),
            _ip(cutlass.Float32, gemm1_beta.data_ptr()),
            _ip(cutlass.Float32, gemm1_clamp_limit.data_ptr()),
            _ip(cutlass.Int32, tile_e.data_ptr()),
            _ip(cutlass.Int32, tile_nv.data_ptr()),
            _ip(cutlass.Int32, meta_p, 16),
            cutlass.Int32(RPAD), cutlass.Int32(EL),
            cutlass.Int32(ru), cutlass.Int32(S),
            cutlass.Int64(act.data_ptr()), stream)

    def fc2_args():
        return (
            _ip(E4M3, act.data_ptr()), _ip(UE8M0, actsf.data_ptr()),
            _ip(E2M1, gemm2_weights.data_ptr()),
            _ip(UE8M0, gemm2_weights_scale.data_ptr()),
            _ip(cutlass.Float32, gemm2_bias.data_ptr()),
            _ip(cutlass.Int32, yb.data_ptr()), _ip(cutlass.Uint8, actsf.data_ptr()),
            _ip(cutlass.Int32, row_src.data_ptr()),
            _ip(cutlass.Float32, gemm1_beta.data_ptr()),
            _ip(cutlass.Float32, gemm1_clamp_limit.data_ptr()),
            _ip(cutlass.Int32, tile_e.data_ptr()),
            _ip(cutlass.Int32, tile_nv.data_ptr()),
            _ip(cutlass.Int32, meta_p, 16),
            cutlass.Int32(RPAD), cutlass.Int32(EL),
            cutlass.Int32(ru), cutlass.Int32(S),
            cutlass.Int64(yb.data_ptr()), stream)

    def comb_args(nb):
        return (
            _ip(cutlass.Int32, yb.data_ptr()),
            _ip(cutlass.BFloat16, topk_weights.data_ptr()),
            _ip(cutlass.Int32, topk_ids.data_ptr()),
            _ip(cutlass.Int32, output.data_ptr()),
            cutlass.Int32(T), cutlass.Int32(off), cutlass.Int32(EL),
            cutlass.Int32(S), cutlass.Int32(nb), stream)

    nclu = _nclusters(num_sm)
    key = ("v2", num_sm, nclu)
    if key not in _CACHE:
        g1 = MoeGemm(1, FC1_NTILES, FC1_KTILES, N1, H1, num_sm, nclu)
        g2 = MoeGemm(2, FC2_NTILES, FC2_KTILES, H2, I_PAD, num_sm, nclu)
        _CACHE[key] = (
            cute.compile(_launch_route, *route_args(1, 1)),
            cute.compile(g1, *fc1_args()),
            cute.compile(g2, *fc2_args()),
            cute.compile(_launch_combine, *comb_args(1)),
        )
    c_route, c_fc1, c_fc2, c_comb = _CACHE[key]

    c_route(*route_args((S + 255) // 256, (T + 7) // 8))
    c_fc1(*fc1_args())
    c_fc2(*fc2_args())
    c_comb(*comb_args(T))
