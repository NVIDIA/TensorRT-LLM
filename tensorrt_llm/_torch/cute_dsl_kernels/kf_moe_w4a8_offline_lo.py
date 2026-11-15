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
# Alternative MoE kernel module for mid-size mixed steps. The routing chain's first kernel (k_planf / k_count) is
# launched without programmatic dependent launch, because it reads topk_ids without griddepcontrol_wait.
# Used only through the default-off TLLM_MOE_KF_DECODE hook, by
# tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8_offline.
# GPT-OSS-120B MoE layer, MXFP4 weights x MXFP8 activations, pre-routed top-4 of 128 experts.
# CuTe DSL implementation targeting Blackwell-family tcgen05 block-scaled MMA
# (kind::mxf8f6f4, E4M3 x E2M1 with UE8M0 scales).
#
# Pipeline (5 kernels, all written here):
#   1. route_count   : per-expert pair counts (device side, graph safe)
#   2. route_plan    : per-expert tile offsets + tile->expert table (1 CTA)
#   3. route_scatter : stable-ish permutation build  (pair -> row)
#   4. gather        : permuted MXFP8 activations + UE8M0 scales in 128x4 swizzle
#   5. gemm (FC1)    : A = expert weight rows (M=128), B = tokens (N), fused
#                      bias + clamp + SwiGLU + MXFP8 requant epilogue
#   6. gemm (FC2)    : same mainloop, epilogue = bias + bf16
#   7. combine       : fp32 FMA chain over slots -> bf16
import math
from typing import Type

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass import Int32, Float32, Boolean

# ------------------------------------------------------------------ problem constants
H = 2880          # model hidden / output width
I_PAD = 2944      # pad_up(2880,128): FC1 rows per half, FC2 K
H1 = 3072         # FC1 K == hidden_states width
H2 = 2944         # FC2 rows
N1 = 5888         # FC1 prepared rows (2 * I_PAD)
TOPK = 4

FC1_M = 5760      # live FC1 prepared rows (2*2880)
FC1_K = 2944      # K tiles actually streamed (23*128); cols >= 2880 are zero padding
FC2_M = 2944      # 23 M tiles of 128 (rows >= 2880 are zero padding)
FC2_K = 2944

BM = 128          # MMA M (prepared weight rows)
BK = 128          # MMA K tile (fixed by the b4x16_p64 unpack-TMA box size)
TN = 128          # MMA N (tokens per tile)
TNSF = 128        # row stride per token tile: the MXFP8 scale layout has a 128-row atom,
                  # so tiles are always allocated 128 rows even when TN is 64
AB_STAGES = 8
ACC_STAGES = 3
PF_DIST = 8       # K tiles the rolling L2 prefetch of the weight stream runs ahead
PF_MIN_PAIRS = 8192  # below this the prefetch costs more than the L2 policy saves
EPI_CHUNK = 32    # token columns handled per epilogue chunk
EPI_ROW_STRIDE = 132  # fp32 words per token row in the epilogue smem buffer

FC1_MTILES = FC1_M // BM   # 45
FC2_MTILES = FC2_M // BM   # 23
FC1_KTILES = FC1_K // BK   # 23
FC2_KTILES = FC2_K // BK   # 23

# 2-SM (cta_group::2) tiling: one cluster of 2 CTAs owns MMA_M prepared weight rows.
MMA_M = 256                 # weight rows per cluster tile (128 per CTA)
CLUSTER_MNK = (2, 1, 1)
AB_STAGES_DEFAULT = 8
# MMA token-tile width used for a tile holding ceil(rows/32) 32-token groups:
# entry g-1 names the variant (1..4 -> N = 32..128) used for g groups.
NSEL = (1, 2, 3, 4)
FC1_MTILES2 = N1 // MMA_M                      # 23 tiles cover 5888 rows; >= 5760 is padding
FC2_MTILES2 = (FC2_M + MMA_M - 1) // MMA_M     # 12 tiles; TMA zero-fills rows >= 2944

EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
NTHREADS = 192
COUNT_THREADS = 256
PLAN_THREADS = 1024
GATHER_THREADS = 256
COMB_THREADS = 192
COMB_VEC = 4
PAIRS_PER_COUNT_CTA = 64   # tokens per histogram CTA
MAX_COUNT_CTAS = 64
GFILL_STRIDE = 32   # one 128B L2 line per expert counter (faster in testing than shared lines)


# ------------------------------------------------------------------ small ptx helpers
@dsl_user_op
def cvt_e4m3x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    """cvt.rn.satfinite.e4m3x2.f32 : hi -> upper byte, lo -> lower byte."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $1, $2; cvt.u32.u16 $0, t; }",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def f32_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def rcp_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def stg_v4(addr: cutlass.Int64, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(v0).ir_value(loc=loc, ip=ip),
         Int32(v1).ir_value(loc=loc, ip=ip), Int32(v2).ir_value(loc=loc, ip=ip),
         Int32(v3).ir_value(loc=loc, ip=ip)],
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};", "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def st_u16(addr: cutlass.Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.u16 [$0], t; }", "l,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def l2_policy_evict_first(*, loc=None, ip=None) -> cutlass.Int64:
    """L2 cache policy marking a line for early eviction: the weight stream is read
    once per call, so it must not displace the token tile that every M tile re-reads."""
    return cutlass.Int64(llvm.inline_asm(
        T.i64(), [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def l2_policy_evict_last(*, loc=None, ip=None) -> cutlass.Int64:
    """L2 cache policy keeping a line resident: the permuted token tile is re-read by
    every weight-row tile of the same expert."""
    return cutlass.Int64(llvm.inline_asm(
        T.i64(), [], "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def pow2f(e: Int32) -> Float32:
    """2^e for e in [-127,127] (e = -127 lands on the fp32 subnormal 2^-127)."""
    eb = e + 127
    bits = Int32(0)
    if eb <= 0:
        bits = Int32(1) << (e + 149)
    else:
        bits = eb << 23
    return bits_f32(bits)


@cute.jit
def sigmoidf(x: Float32) -> Float32:
    return rcp_f32(Float32(1.0) + cute.arch.exp2(x * Float32(-1.4426950408889634)))


# ------------------------------------------------------------------ routing kernels
# Three small kernels replace the former single-CTA router so the routing cost scales
# with the SM count instead of with T:
#   k_count : per-chunk expert histograms      (RG CTAs, no global atomics)
#   k_plan  : reduce + per-expert tile plan    (1 CTA, 1024 threads)
#   k_sgat  : claim a row per pair and gather  (one warp per pair)
@cute.kernel
def k_count(ids_t: cute.Tensor, pcnt: cute.Tensor,
             Tt: Int32, off: Int32, EL: Int32, CH: Int32):
    """Per-CTA expert histogram over a token range. One 16-byte load covers a token's four
    expert ids, so the pass costs a single vector load per token and no global round trip."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    cnt = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        cnt[tidx] = Int32(0)
    # nothing here reads a prior kernel's output, so the next kernel's prologue may start now
    cute.arch.griddepcontrol_launch_dependents()
    cute.arch.sync_threads()
    lo = bidx * CH
    hi = lo + CH
    if hi > Tt:
        hi = Tt
    r4 = cute.make_rmem_tensor(cute.make_layout(TOPK), Int32)
    t = lo + Int32(tidx)
    while t < hi:
        cute.autovec_copy(
            cute.make_tensor(ids_t.iterator + t * TOPK, cute.make_layout(TOPK)), r4)
        for k in cutlass.range_constexpr(TOPK):
            ee = r4[k] - off
            if (ee >= 0) and (ee < EL):
                cute.arch.atomic_add(cnt.iterator + ee, Int32(1))
        t += COUNT_THREADS
    cute.arch.sync_threads()
    if tidx < 128:
        pcnt[bidx * 128 + tidx] = cnt[tidx]


@cute.kernel
def k_plan(pcnt: cute.Tensor, ebase: cute.Tensor, gfill: cute.Tensor,
           tile_expert: cute.Tensor, tile_nrows: cute.Tensor, meta: cute.Tensor,
           zrow: cute.Tensor, RG: Int32, EL: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    cnt = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    scan = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        cnt[tidx] = Int32(0)
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    cute.arch.sync_threads()
    tot = RG * 128
    # MAX_COUNT_CTAS * 128 / PLAN_THREADS = 8 chunks; issue all of them before consuming any,
    # so the reduction costs one global round trip instead of eight dependent ones
    NCH = (MAX_COUNT_CTAS * 128) // PLAN_THREADS
    pv = cute.make_rmem_tensor(cute.make_layout(NCH), Int32)
    for q in cutlass.range_constexpr(NCH):
        pv[q] = pcnt[tidx + q * PLAN_THREADS]
    for q in cutlass.range_constexpr(NCH):
        i = tidx + q * PLAN_THREADS
        if i < tot:
            v = pv[q]
            if v != 0:
                cute.arch.atomic_add(cnt.iterator + (tidx & 127), v)
    cute.arch.sync_threads()
    nt = Int32(0)
    if tidx < 128:
        nt = (cnt[tidx] + (TN - 1)) // TN
        scan[tidx] = nt
        gfill[tidx * GFILL_STRIDE] = Int32(0)
    cute.arch.sync_threads()
    if tidx < 32:
        # inclusive scan of the 128 per-expert tile counts inside one warp:
        # four shuffle scans chained through the running total
        run = Int32(0)
        for q in cutlass.range_constexpr(4):
            v = scan[q * 32 + tidx]
            for sp in cutlass.range_constexpr(5):
                d = 1 << sp
                o = cute.arch.shuffle_sync_up(v, d)
                if tidx >= d:
                    v = v + o
            scan[q * 32 + tidx] = v + run
            run = run + cute.arch.shuffle_sync(v, 31)
    cute.arch.sync_threads()
    if tidx < 128:
        o = scan[tidx] - nt
        ebase[tidx] = o * TNSF
        if tidx < EL:
            c = cnt[tidx]
            k = Int32(0)
            while k < nt:
                tile_expert[o + k] = tidx
                r = c - k * TN
                if r > TN:
                    r = Int32(TN)
                tile_nrows[o + k] = r
                k += 1
    if tidx == 0:
        meta[0] = scan[127]
        meta[1] = scan[127] * TNSF


@cute.kernel
def k_planf(ids_t: cute.Tensor, ebase: cute.Tensor, gfill: cute.Tensor,
           tile_expert: cute.Tensor, tile_nrows: cute.Tensor, meta: cute.Tensor,
           zrow: cute.Tensor, Tt: Int32, off: Int32, EL: Int32):
    """Small-T routing: count + plan in one CTA: one 16-byte load per token covers its four expert ids,
    so the whole histogram costs ceil(T/1024) vector loads and no global round trip."""
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    cnt = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    scan = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        cnt[tidx] = Int32(0)
    # this kernel only reads call inputs, so the gather may start its 3 KB token loads now
    cute.arch.griddepcontrol_launch_dependents()
    cute.arch.sync_threads()
    r4 = cute.make_rmem_tensor(cute.make_layout(TOPK), Int32)
    r4b = cute.make_rmem_tensor(cute.make_layout(TOPK), Int32)
    t = Int32(tidx)
    while t < Tt:
        # two id vectors in flight: this histogram is load-latency bound, not atomic bound
        t2 = t + PLAN_THREADS
        ok2 = t2 < Tt
        ts = t2
        if not ok2:
            ts = t
        cute.autovec_copy(
            cute.make_tensor(ids_t.iterator + t * TOPK, cute.make_layout(TOPK)), r4)
        cute.autovec_copy(
            cute.make_tensor(ids_t.iterator + ts * TOPK, cute.make_layout(TOPK)), r4b)
        for k in cutlass.range_constexpr(TOPK):
            e = r4[k] - off
            if (e >= 0) and (e < EL):
                cute.arch.atomic_add(cnt.iterator + e, Int32(1))
        if ok2:
            for k in cutlass.range_constexpr(TOPK):
                e = r4b[k] - off
                if (e >= 0) and (e < EL):
                    cute.arch.atomic_add(cnt.iterator + e, Int32(1))
        t = t + (2 * PLAN_THREADS)
    cute.arch.sync_threads()
    nt = Int32(0)
    if tidx < 128:
        nt = (cnt[tidx] + (TN - 1)) // TN
        scan[tidx] = nt
        gfill[tidx * GFILL_STRIDE] = Int32(0)
    cute.arch.sync_threads()
    if tidx < 32:
        # inclusive scan of the 128 per-expert tile counts inside one warp:
        # four shuffle scans chained through the running total
        run = Int32(0)
        for q in cutlass.range_constexpr(4):
            v = scan[q * 32 + tidx]
            for sp in cutlass.range_constexpr(5):
                d = 1 << sp
                o = cute.arch.shuffle_sync_up(v, d)
                if tidx >= d:
                    v = v + o
            scan[q * 32 + tidx] = v + run
            run = run + cute.arch.shuffle_sync(v, 31)
    cute.arch.sync_threads()
    if tidx < 128:
        o = scan[tidx] - nt
        ebase[tidx] = o * TNSF
        if tidx < EL:
            c = cnt[tidx]
            k = Int32(0)
            while k < nt:
                tile_expert[o + k] = tidx
                r = c - k * TN
                if r > TN:
                    r = Int32(TN)
                tile_nrows[o + k] = r
                k += 1
    if tidx == 0:
        meta[0] = scan[127]
        meta[1] = scan[127] * TNSF


@cute.kernel
def k_sgat(topk_ids: cute.Tensor, ebase: cute.Tensor, gfill: cute.Tensor,
           pair2row: cute.Tensor, hs4: cute.Tensor, hssf: cute.Tensor,
           gact4: cute.Tensor, gactsf: cute.Tensor, actqb: cute.Tensor,
           actsf: cute.Tensor, zrow32: cute.Tensor,
           npairs: Int32, off: Int32, EL: Int32):
    """One warp per (token, slot) pair: claim its row, then move 3072 bytes + 96 scales.
    The 3 KB token payload and the per-expert base are issued before the row-claim atomic,
    so the atomic -> ebase -> shuffle chain hides behind the loads instead of gating them.
    Preferred below T = 4096: four times the warps keeps the small grid latency-covered, and
    the duplicated token reads hit L2."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    i = bidx * (GATHER_THREADS // 32) + warp
    if i < npairs:
        e = topk_ids[i] - off
        if (e >= 0) and (e < EL):
            t = i >> 2
            frg = [cute.make_rmem_tensor(cute.make_layout(4), Int32) for _ in range(H1 // 512)]
            for j in cutlass.range_constexpr(H1 // 512):
                cute.autovec_copy(hs4[(t * (H1 // 16) + j * 32 + lane, None)], frg[j])
            sfv = Int32(0)
            if lane < 24:
                sfv = hssf[t * 24 + lane]
            # hidden_states / topk_ids are call inputs, so the payload above is already in
            # flight; only the row claim and the stores need the plan kernel to have retired
            cute.arch.griddepcontrol_wait()
            base = Int32(0)
            slot = Int32(0)
            if lane == 0:
                base = ebase[e]
                slot = cute.arch.atomic_add(
                    gfill.iterator + e * GFILL_STRIDE, Int32(1), scope="gpu")
            # tiles occupy TNSF-row slots but hold at most TN tokens, so a claim past the
            # TN-th row of a tile moves on to the next tile's slot
            prow = base + (slot // TN) * TNSF + (slot % TN)
            p = cute.arch.shuffle_sync(prow, 0)
            if lane == 0:
                pair2row[i] = p
            for j in cutlass.range_constexpr(H1 // 512):
                cute.autovec_copy(frg[j], gact4[(p * (H1 // 16) + j * 32 + lane, None)])
            if lane < 24:
                sb = ((p // 128) * (512 * 24) + lane * 512 + (p % 32) * 16
                      + ((p % 128) // 32) * 4) // 4
                gactsf[sb] = sfv
            # zero the FC2 K-padding of the intermediate (channels 2880..2943)
            if lane < 16:
                cute.recast_ptr(actqb.iterator + (p * I_PAD + H), dtype=Int32)[lane] = Int32(0)
            if lane == 0:
                o = (p // 128) * (512 * 23) + 22 * 512 + (p % 32) * 16 + ((p % 128) // 32) * 4
                st_u16((actsf.iterator + (o + 2)).toint(), Int32(0))
        else:
            cute.arch.griddepcontrol_wait()
            if lane == 0:
                pair2row[i] = Int32(-1)
    else:
        cute.arch.griddepcontrol_wait()
    if bidx == 0:
        # fp32 zero row the combine reads for empty routing slots
        j = Int32(tidx)
        while j < (H // 2):
            zrow32[j] = Int32(0)
            j += GATHER_THREADS
    cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def k_combine(ybuf4: cute.Tensor, p2r4: cute.Tensor, topkw4: cute.Tensor,
              out4: cute.Tensor, Tt: Int32, zrow: Int32):
    """fp32 FMA chain over the 4 routing slots -> bf16. Empty slots read a zero row so all
    loads issue unconditionally (no branches). Each thread owns two 16-byte output groups a
    full block apart, so every load instruction stays one contiguous 512-byte warp access
    while eight 128-bit loads are in flight per thread."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    NVEC = H // 8                     # 360 16-byte groups per token row
    va = [[cute.make_rmem_tensor(cute.make_layout(COMB_VEC), Int32) for _ in range(TOPK)]
          for _ in range(2)]
    vo = cute.make_rmem_tensor(cute.make_layout(COMB_VEC), Int32)
    ri = cute.make_rmem_tensor(cute.make_layout(TOPK), Int32)
    rw = cute.make_rmem_tensor(cute.make_layout(TOPK), cutlass.BFloat16)
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    t = Int32(bidx)
    while t < Tt:
        # one 16-byte index load and one 8-byte weight load per token
        cute.autovec_copy(p2r4[(t, None)], ri)
        cute.autovec_copy(topkw4[(t, None)], rw)
        r0 = ri[0]
        r1 = ri[1]
        r2 = ri[2]
        r3 = ri[3]
        if r0 < 0:
            r0 = zrow
        if r1 < 0:
            r1 = zrow
        if r2 < 0:
            r2 = zrow
        if r3 < 0:
            r3 = zrow
        rr = [r0, r1, r2, r3]
        gs = [Int32(0), Int32(0)]
        oks = [Boolean(False), Boolean(False)]
        for h in cutlass.range_constexpr(2):
            g = Int32(tidx) + h * COMB_THREADS
            ok = g < NVEC
            if not ok:
                g = Int32(NVEC - 1)
            gs[h] = g
            oks[h] = ok
        for h in cutlass.range_constexpr(2):
            for k in cutlass.range_constexpr(TOPK):
                cute.autovec_copy(ybuf4[(rr[k] * NVEC + gs[h], None)], va[h][k])
        w = [Float32(rw[k]) for k in range(TOPK)]
        for h in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(COMB_VEC):
                a0 = fma_rn(w[0], bits_f32((va[h][0][i] & 0xFFFF) << 16), Float32(0.0))
                a1 = fma_rn(w[0], bits_f32(va[h][0][i] & Int32(-65536)), Float32(0.0))
                for q in cutlass.range_constexpr(TOPK - 1):
                    k = q + 1
                    a0 = fma_rn(w[k], bits_f32((va[h][k][i] & 0xFFFF) << 16), a0)
                    a1 = fma_rn(w[k], bits_f32(va[h][k][i] & Int32(-65536)), a1)
                vo[i] = cvt_bf16x2(a1, a0)
            if oks[h]:
                cute.autovec_copy(vo, out4[(t * NVEC + gs[h], None)])
        t += gdim


@dsl_user_op
def cvt_bf16x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )



@dsl_user_op
def fmaxf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fminf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "min.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def fabsf(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;", "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def iclamp(a: Int32, lo: cutlass.Constexpr, hi: cutlass.Constexpr, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [Int32(a).ir_value(loc=loc, ip=ip)],
        "{ .reg .s32 t; max.s32 t, $1, %d; min.s32 $0, t, %d; }" % (lo, hi),
        "=r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


# ------------------------------------------------------------------ main grouped GEMM
# 2-SM (cta_group::2) variant: one cluster of 2 CTAs owns a 256-row weight tile.
# A (weights) is split 128 rows per CTA, B (tokens) is split TN/2 per CTA, so the
# token stream is read once per 256 weight rows instead of once per 128: the L2 read
# per unit of MMA work drops by a third and the MMA issue count halves.
@cute.kernel
def k_gemm(
    tiled_mma: cute.TiledMma,
    tiled_mma_sf: cute.TiledMma,
    mma_v1: cute.TiledMma,
    mma_v2: cute.TiledMma,
    mma_v3: cute.TiledMma,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    bsl_v1: cute.ComposedLayout,
    bsl_v2: cute.ComposedLayout,
    bsl_v3: cute.ComposedLayout,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tb_v1: cute.CopyAtom,
    mB_v1: cute.Tensor,
    tb_v2: cute.CopyAtom,
    mB_v2: cute.Tensor,
    tb_v3: cute.CopyAtom,
    mB_v3: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB: cute.Tensor,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    gbias: cute.Tensor,
    tile_expert: cute.Tensor,
    tile_nrows: cute.Tensor,
    meta: cute.Tensor,
    out_b: cute.Tensor,
    out_sf: cute.Tensor,
    out32: cute.Tensor,
    alpha_t: cute.Tensor,
    beta_t: cute.Tensor,
    limit_t: cute.Tensor,
    round_up: Int32,
    storage_t: cutlass.Constexpr,
    A_STREAM: cutlass.Constexpr,
    PF_ON: cutlass.Constexpr,
    MODE: cutlass.Constexpr,
    MTILES: cutlass.Constexpr,
    KTILES: cutlass.Constexpr,
    BIAS_STRIDE: cutlass.Constexpr,
    MLIVE: cutlass.Constexpr,
    tx_v1: cutlass.Constexpr,
    tx_v2: cutlass.Constexpr,
    tx_v3: cutlass.Constexpr,
    tx_v4: cutlass.Constexpr,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    if warp_idx == TMA_WARP:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tb_v1)
        cpasync.prefetch_descriptor(tb_v2)
        cpasync.prefetch_descriptor(tb_v3)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)

    cta_v = bidx % 2
    is_leader = cta_v == 0
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    bc_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
    bc_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)

    smem = utils.SmemAllocator()
    storage = smem.allocate(storage_t)

    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full.data_ptr(),
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=tx_v4,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full.data_ptr(),
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * EPI_WARPS * 2),
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=32 * (EPI_WARPS + 1)
    )
    epi_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * EPI_WARPS)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=0,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )
    pipeline_init_arrive(cluster_shape_mn=(2, 1), is_relaxed=True)

    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    # Four token-tile widths (N = 32, 64, 96, 128) share one B staging buffer: a tile with
    # r rows only needs ceil(r/32) 32-token groups, so the MMA skips the padded groups
    # instead of issuing tensor-core work on them.
    MMAS = [mma_v1, mma_v2, mma_v3, tiled_mma]
    BSLS = [bsl_v1, bsl_v2, bsl_v3, b_smem_layout_staged]
    TBS = [tb_v1, tb_v2, tb_v3, tma_atom_b]
    MBS = [mB_v1, mB_v2, mB_v3, mB]
    TXS = [tx_v1, tx_v2, tx_v3, tx_v4]
    sBV = [storage.sB.get_tensor(BSLS[j].outer, swizzle=BSLS[j].inner) for j in range(4)]
    sB = sBV[3]
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
    sEpi = storage.sEpi.get_tensor(cute.make_layout(EPI_CHUNK * EPI_ROW_STRIDE))

    mma_tiler = (MMA_M, TN, BK)
    mma_tiler_sf = (BM, TNSF, BK)
    gA = cute.local_tile(mA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gSFA = cute.local_tile(mSFA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gSFB = cute.local_tile(mSFB, cute.slice_(mma_tiler_sf, (0, None, None)), (None, None, None))

    thr_mma = tiled_mma.get_slice(cta_v)
    tCgA = thr_mma.partition_A(gA)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = tiled_mma_sf.get_slice(0).partition_B(gSFB)
    gBV = [cute.local_tile(MBS[j], cute.slice_((MMA_M, 32 * (j + 1), BK), (0, None, None)),
                           (None, None, None)) for j in range(4)]
    tCgBV = [MMAS[j].get_slice(cta_v).partition_B(gBV[j]) for j in range(4)]

    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, bc_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    _tbp = [cpasync.tma_partition(
        TBS[j], bc_vmnk[1], b_cta_layout,
        cute.group_modes(sBV[j], 0, 3), cute.group_modes(tCgBV[j], 0, 3)) for j in range(4)]
    tBsBV = [_tbp[j][0] for j in range(4)]
    tBgBV = [_tbp[j][1] for j in range(4)]
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, bc_vmnk[2], a_cta_layout, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, bc_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3)
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    a_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, bc_vmnk, mcast_mode=2)
    b_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, bc_vmnk, mcast_mode=1)
    sfb_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, bc_sfb_vmnk, mcast_mode=1)

    tCrAV = [MMAS[j].make_fragment_A(sA) for j in range(4)]
    tCrBV = [MMAS[j].make_fragment_B(sBV[j]) for j in range(4)]
    tCrA = tCrAV[3]
    acc_shape = tiled_mma.partition_shape_C((MMA_M, TN))
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))
    tCtAccV_fake = [MMAS[j].make_fragment_C(MMAS[j].partition_shape_C((MMA_M, 32 * (j + 1))))
                    for j in range(4)]

    num_acc_cols = TN * ACC_STAGES
    num_sfa_cols = (BM // 32) * 4
    pipeline_init_wait(cluster_shape_mn=(2, 1))
    cute.arch.griddepcontrol_wait()

    ntiles_n = meta[0]
    total_tiles = ntiles_n * MTILES
    ncl = gdim // 2

    # ------------------------------------------------------------------ TMA warp
    if warp_idx == TMA_WARP:
        ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, AB_STAGES)
        pol_a = l2_policy_evict_first()
        flat = Int32(bidx // 2)
        if cutlass.const_expr(PF_ON):
            if flat < total_tiles:
                e0 = tile_expert[flat // MTILES]
                tAgA_0 = tAgA[(None, flat % MTILES, None, e0)]
                for ktp in cutlass.range_constexpr(PF_DIST // 2):
                    cute.prefetch(tma_atom_a, tAgA_0[(None, 2 * ktp)])
        while flat < total_tiles:
            n_tile = flat // MTILES
            m_tile = flat % MTILES
            e = tile_expert[n_tile]
            ng = (tile_nrows[n_tile] + 31) // 32
            nsub = Int32(NSEL[3])
            for jj in cutlass.range_constexpr(3):
                if ng == jj + 1:
                    nsub = Int32(NSEL[jj])
            tAgA_s = tAgA[(None, m_tile, None, e)]
            tAgSFA_s = tAgSFA[(None, m_tile, None, e)]
            tBgSFB_s = tBgSFB[(None, n_tile, None, 0)]
            tAgA_n = tAgA_s
            if cutlass.const_expr(PF_ON):
                nxt = flat + ncl
                if not (nxt < total_tiles):
                    nxt = flat
                e_n = tile_expert[nxt // MTILES]
                tAgA_n = tAgA[(None, nxt % MTILES, None, e_n)]
            tx_all = Int32(tx_v4)
            for jj in cutlass.range_constexpr(3):
                if nsub == jj + 1:
                    tx_all = Int32(TXS[jj])
            for kt in cutlass.range(KTILES, unroll=1):
                # One arrive covers every byte of the stage across the CTA pair
                # (A + SFA + SFB + B for both CTAs); no expect_tx may follow it.
                ab_pipeline.producer_acquire(ab_prod, expected_tx=tx_all)
                bar = ab_pipeline.producer_get_barrier(ab_prod)
                if cutlass.const_expr(A_STREAM):
                    cute.copy(tma_atom_a, tAgA_s[(None, kt)], tAsA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask, cache_policy=pol_a)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, kt)], tAsSFA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask, cache_policy=pol_a)
                else:
                    cute.copy(tma_atom_a, tAgA_s[(None, kt)], tAsA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, kt)], tAsSFA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, mcast_mask=a_mask)
                cute.copy(tma_atom_sfb, tBgSFB_s[(None, kt)], tBsSFB[(None, ab_prod.index)],
                          tma_bar_ptr=bar, mcast_mask=sfb_mask)
                for jj in cutlass.range_constexpr(4):
                    if nsub == jj + 1:
                        cute.copy(TBS[jj], tBgBV[jj][(None, 0, None, n_tile)][(None, kt)],
                                  tBsBV[jj][(None, ab_prod.index)],
                                  tma_bar_ptr=bar, mcast_mask=b_mask)
                if cutlass.const_expr(PF_ON):
                    if (kt & 1) == 0:
                        pf = kt + PF_DIST
                        if pf < KTILES:
                            cute.prefetch(tma_atom_a, tAgA_s[(None, pf)])
                        else:
                            cute.prefetch(tma_atom_a, tAgA_n[(None, pf - KTILES)])
                ab_prod.advance()
            flat += ncl
        ab_pipeline.producer_tail(ab_prod)

    # ------------------------------------------------------------------ MMA warp
    if warp_idx == MMA_WARP:
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + num_acc_cols, dtype=cutlass.Float8E8M0FNU)
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, mma_tiler, 32, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
            ),
        )
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + num_acc_cols + num_sfa_cols, dtype=cutlass.Float8E8M0FNU
        )
        tCtSFB_cp = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_sf, (BM, TNSF, BK), 32,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, mma_tiler, 32,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
            ),
        )
        tCtSFBV = [cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                MMAS[j], (MMA_M, 32 * (j + 1), BK), 32,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
            ),
        ) for j in range(4)]
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), cutlass.Float8E8M0FNU
        )
        tc_sfa = cute.filter_zeros(sSFA)
        tt_sfa = cute.filter_zeros(tCtSFA)
        tiled_s2t_a = tcgen05.make_s2t_copy(copy_atom_s2t, tt_sfa)
        thr_s2t_a = tiled_s2t_a.get_slice(0)
        s_sfa = tcgen05.get_s2t_smem_desc_tensor(tiled_s2t_a, thr_s2t_a.partition_S(tc_sfa))
        t_sfa = thr_s2t_a.partition_D(tt_sfa)

        tc_sfb = cute.filter_zeros(sSFB)
        tt_sfb = cute.filter_zeros(tCtSFB_cp)
        tiled_s2t_b = tcgen05.make_s2t_copy(copy_atom_s2t, tt_sfb)
        thr_s2t_b = tiled_s2t_b.get_slice(0)
        s_sfb = tcgen05.get_s2t_smem_desc_tensor(tiled_s2t_b, thr_s2t_b.partition_S(tc_sfb))
        t_sfb = thr_s2t_b.partition_D(tt_sfb)

        ab_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, AB_STAGES)
        acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, ACC_STAGES)
        flat = Int32(bidx // 2)
        while flat < total_tiles:
            ng = (tile_nrows[flat // MTILES] + 31) // 32
            nsub = Int32(NSEL[3])
            for jj in cutlass.range_constexpr(3):
                if ng == jj + 1:
                    nsub = Int32(NSEL[jj])
            accp = acc_tmem_ptr + acc_prod.index * TN
            # only the leader CTA of the pair issues tensor-core work, and the token-width
            # choice is per tile: both tests are hoisted out of the K loop so its body is a
            # single straight-line wait / scale-copy / MMA / release sequence
            if is_leader:
                acc_pipeline.producer_acquire(acc_prod)
                for jj in cutlass.range_constexpr(4):
                    if nsub == jj + 1:
                        accj = cute.make_tensor(accp, tCtAccV_fake[jj].layout)
                        for kt in cutlass.range(KTILES, unroll=1):
                            ab_pipeline.consumer_wait(ab_cons)
                            st = (None, None, None, None, ab_cons.index)
                            cute.copy(tiled_s2t_a, s_sfa[st], t_sfa)
                            cute.copy(tiled_s2t_b, s_sfb[st], t_sfb)
                            crd = (None, None, None, ab_cons.index)
                            MMAS[jj].set(tcgen05.Field.ACCUMULATE, kt != 0)
                            cute.gemm(MMAS[jj], accj, [tCrAV[jj][crd], tCtSFA],
                                      [tCrBV[jj][crd], tCtSFBV[jj]], accj)
                            ab_pipeline.consumer_release(ab_cons)
                            ab_cons.advance()
                acc_pipeline.producer_commit(acc_prod)
            acc_prod.advance()
            flat += ncl
        acc_pipeline.producer_tail(acc_prod)

    # ------------------------------------------------------------------ epilogue warps
    if warp_idx < EPI_WARPS:
        tmem.allocate(512)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (BM, TN, BK), utils.LayoutEnum.COL_MAJOR, Float32, Float32,
            (BM, EPI_CHUNK), False,
        )
        tAcc_all = cute.flat_divide(
            tCtAcc_base[((None, None), 0, 0, None)], (BM, EPI_CHUNK)
        )
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_all[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tAcc_base = thr_t2r.partition_S(tAcc_all)
        idC = cute.make_identity_tensor((BM, EPI_CHUNK))
        tTR_idC = thr_t2r.partition_D(idC)
        rAcc = cute.make_rmem_tensor(tTR_idC.shape, Float32)
        blk = tidx // 32
        lane = tidx % 32
        acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, ACC_STAGES)
        flat = Int32(bidx // 2)
        while flat < total_tiles:
            n_tile = flat // MTILES
            m_tile = (flat % MTILES) * 2 + cta_v
            e = tile_expert[n_tile]
            nvr = tile_nrows[n_tile]
            live = m_tile * BM < MLIVE
            prow = m_tile * BM + tidx
            bias = Float32(0.0)
            if live:
                bias = gbias[e * BIAS_STRIDE + prow]
            p0 = n_tile * TNSF
            alpha = Float32(0.0)
            beta = Float32(0.0)
            limit = Float32(0.0)
            if cutlass.const_expr(MODE == 0):
                alpha = alpha_t[e]
                beta = beta_t[e]
                limit = limit_t[e]
            tTR_tAcc = tTR_tAcc_base[(None, None, None, 0, None, acc_cons.index)]
            acc_pipeline.consumer_wait(acc_cons)
            if live:
                for ch in cutlass.range_constexpr(TN // EPI_CHUNK):
                    if ch * EPI_CHUNK < nvr:
                        cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, ch)], rAcc)
                        if cutlass.const_expr(MODE == 0):
                            wpos = tidx
                        else:
                            wpos = blk * 32 + 4 * (lane % 8) + lane // 8
                        wpad = wpos + (wpos >> 5)
                        for j in cutlass.range_constexpr(EPI_CHUNK):
                            sEpi[j * EPI_ROW_STRIDE + wpad] = rAcc[j] + bias
                        if ch == 0:
                            cute.arch.fence_view_async_tmem_load()
                        epi_barrier.arrive_and_wait()
                        tok0 = p0 + ch * EPI_CHUNK
                        if cutlass.const_expr(MODE == 0):
                            token = tidx // 4
                            hq = tidx % 4
                            base_s = token * EPI_ROW_STRIDE
                            vals = [None] * 16
                            amax = Float32(0.0)
                            for i in cutlass.range_constexpr(16):
                                up_row = 33 * hq + 16 * (i % 2) + i // 2
                                up = sEpi[base_s + up_row]
                                gate = sEpi[base_s + up_row + 8]
                                up = fminf(fmaxf(up, -limit), limit)
                                gate = fminf(gate, limit)
                                a = (up + beta) * gate * sigmoidf(alpha * gate)
                                vals[i] = a
                                amax = fmaxf(amax, fabsf(a))
                            amax = fmaxf(amax, cute.arch.shuffle_sync_bfly(amax, 1))
                            ee = Int32(0)
                            if round_up != 0:
                                x = (amax * Float32(1.8446744073709552e19)) / Float32(448.0)
                                bx = f32_bits(x)
                                ee = ((bx >> 23) & 0xFF) - 190
                                if (bx & 0x7FFFFF) == 0:
                                    ee = ee - 1
                            else:
                                ee = ((f32_bits(amax) >> 23) & 0xFF) - 135
                            ee = iclamp(ee, -127, 127)
                            inv = pow2f(-ee)
                            prow_out = tok0 + token
                            obase = prow_out * I_PAD + m_tile * 64 + 16 * hq
                            if ch * EPI_CHUNK + token < nvr:
                                wv = [None] * 4
                                for i in cutlass.range_constexpr(4):
                                    lo = cvt_e4m3x2(vals[4 * i + 1] * inv, vals[4 * i] * inv)
                                    hi = cvt_e4m3x2(vals[4 * i + 3] * inv, vals[4 * i + 2] * inv)
                                    wv[i] = lo | (hi << 16)
                                stg_v4((out_b.iterator + obase).toint(), wv[0], wv[1], wv[2], wv[3])
                                if hq % 2 == 0:
                                    c32 = m_tile * 2 + hq // 2
                                    sfo = (
                                        (prow_out // 128) * (512 * (I_PAD // 128))
                                        + (c32 // 4) * 512
                                        + (prow_out % 32) * 16
                                        + ((prow_out % 128) // 32) * 4
                                        + (c32 % 4)
                                    )
                                    out_sf[sfo] = cutlass.Uint8(ee + 127)
                        else:
                            for tl in cutlass.range_constexpr(EPI_CHUNK // EPI_WARPS):
                                token = blk * (EPI_CHUNK // EPI_WARPS) + tl
                                base_s = token * EPI_ROW_STRIDE
                                orow = (tok0 + token) * (H // 2) + m_tile * (BM // 2)
                                for q in cutlass.range_constexpr(BM // 64):
                                    idx = q * 32 + lane
                                    ipad = 2 * idx + (idx >> 4)
                                    if (m_tile * BM + 2 * idx < H) and (ch * EPI_CHUNK + token < nvr):
                                        out32[orow + idx] = cvt_bf16x2(
                                            sEpi[base_s + ipad + 1], sEpi[base_s + ipad]
                                        )
                        epi_barrier.arrive_and_wait()
            acc_pipeline.consumer_release(acc_cons)
            acc_cons.advance()
            flat += ncl
        tmem.relinquish_alloc_permit()
        tmem.free(acc_tmem_ptr, 512)
    cute.arch.griddepcontrol_launch_dependents()

# ------------------------------------------------------------------ host side
@cute.jit
def moe_jit(
    p_hs32, p_hssf, p_ids, p_topkw, p_w1, p_w1sf, p_b1, p_alpha, p_beta, p_lim,
    p_w2, p_w2sf, p_b2, p_out32,
    p_tile_e, p_tile_nr, p_meta, p_pcnt, p_ebase, p_gfill, p_pair2row,
    p_gact32, p_gactsf, p_gactsf32, p_gactq, p_actqb, p_actsf, p_actqe4, p_ybuf16, p_ybuf32,
    Tt: Int32, EL: Int32, NR: Int32, off: Int32, round_up: Int32,
    RG: Int32, CH: Int32, GG: Int32, CB: Int32,
    num_cta: cutlass.Constexpr,
    a_stream: cutlass.Constexpr,
    pf_on: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    stream,
):
    npairs = Tt * TOPK
    ids_t = cute.make_tensor(p_ids, cute.make_layout(npairs))
    tile_e = cute.make_tensor(p_tile_e, cute.make_layout(NR // TNSF))
    tile_nr = cute.make_tensor(p_tile_nr, cute.make_layout(NR // TNSF))
    meta = cute.make_tensor(p_meta, cute.make_layout(4))
    pcnt = cute.make_tensor(p_pcnt, cute.make_layout(MAX_COUNT_CTAS * 128))
    ebase = cute.make_tensor(p_ebase, cute.make_layout(128))
    gfill = cute.make_tensor(p_gfill, cute.make_layout(128 * GFILL_STRIDE))
    pair2row = cute.make_tensor(p_pair2row, cute.make_layout(npairs))
    hs4 = cute.make_tensor(p_hs32, cute.make_layout((Tt * (H1 // 16), 4), stride=(4, 1)))
    hssf = cute.make_tensor(p_hssf, cute.make_layout(Tt * 24))
    gact4 = cute.make_tensor(p_gact32, cute.make_layout((NR * (H1 // 16), 4), stride=(4, 1)))
    gactsf = cute.make_tensor(p_gactsf32, cute.make_layout(NR * 24))
    actqb = cute.make_tensor(p_actqb, cute.make_layout(NR * I_PAD))
    actsf = cute.make_tensor(p_actsf, cute.make_layout(NR * (I_PAD // 32)))
    ybuf32 = cute.make_tensor(p_ybuf32, cute.make_layout((NR + 1) * (H // 2)))
    ybuf4 = cute.make_tensor(p_ybuf32, cute.make_layout(((NR + 1) * (H // 8), 4), stride=(4, 1)))
    zrow = cute.make_tensor(ybuf32.iterator + NR * (H // 2), cute.make_layout(H // 2))
    out4 = cute.make_tensor(p_out32, cute.make_layout((Tt * (H // 8), 4), stride=(4, 1)))
    p2r4 = cute.make_tensor(p_pair2row, cute.make_layout((Tt, TOPK), stride=(TOPK, 1)))
    topkw4 = cute.make_tensor(p_topkw, cute.make_layout((Tt, TOPK), stride=(TOPK, 1)))
    bias1 = cute.make_tensor(p_b1, cute.make_layout(EL * N1))
    bias2 = cute.make_tensor(p_b2, cute.make_layout(EL * H2))
    alpha_t = cute.make_tensor(p_alpha, cute.make_layout(EL))
    beta_t = cute.make_tensor(p_beta, cute.make_layout(EL))
    lim_t = cute.make_tensor(p_lim, cute.make_layout(EL))

    if cutlass.const_expr(a_stream):
        k_planf(ids_t, ebase, gfill, tile_e, tile_nr, meta, zrow, Tt, off, EL).launch(
            grid=[1, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=False)
    else:
        k_count(ids_t, pcnt, Tt, off, EL, CH).launch(
            grid=[RG, 1, 1], block=[COUNT_THREADS, 1, 1], stream=stream, use_pdl=False)
        k_plan(pcnt, ebase, gfill, tile_e, tile_nr, meta, zrow, RG, EL).launch(
            grid=[1, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=pdl)
    k_sgat(ids_t, ebase, gfill, pair2row, hs4, hssf, gact4, gactsf, actqb, actsf,
           zrow, npairs, off, EL).launch(
        grid=[GG, 1, 1], block=[GATHER_THREADS, 1, 1], stream=stream, use_pdl=pdl)

    mma_tiler = (MMA_M, TN, BK)
    mma_tiler_sf = (BM, TNSF, BK)
    MMAS = [sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.TWO, (MMA_M, 32 * (j + 1)),
    ) for j in range(4)]
    tiled_mma = MMAS[3]
    # scale factors keep the 1-CTA 128-row / 128-token atom
    tiled_mma_sf = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.ONE, (BM, TNSF),
    )
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_MNK), (tiled_mma.thr_id.shape,)
    )
    cluster_layout_sfb_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_MNK), (tiled_mma_sf.thr_id.shape,)
    )
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, cutlass.Int8, AB_STAGES)
    _BSL0 = [sm100_utils.make_smem_layout_b(
        MMAS[j], (MMA_M, 32 * (j + 1), BK), cutlass.Float8E4M3FN, AB_STAGES) for j in range(4)]
    # every width must place pipeline stage s at the same smem offset: the ring's mutual
    # exclusion is per stage index, so a narrow stage with its own (smaller) stride would
    # straddle wide stages the MMA has not released yet.
    _WST = cute.cosize(_BSL0[3].outer) // AB_STAGES
    BSLS = [cute.make_composed_layout(
        _BSL0[j].inner, 0,
        cute.append(cute.slice_(_BSL0[j].outer, (None, None, None, 0)),
                    cute.make_layout(AB_STAGES, stride=_WST))) for j in range(4)]
    b_sl = BSLS[3]
    sfa_sl = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, 32, AB_STAGES)
    sfb_sl = blockscaled_utils.make_smem_layout_sfb(tiled_mma_sf, mma_tiler_sf, 32, AB_STAGES)
    a_one = cute.slice_(a_sl, (None, None, None, 0))
    B_ONE = [cute.slice_(BSLS[j], (None, None, None, 0)) for j in range(4)]
    sfa_one = cute.slice_(sfa_sl, (None, None, None, 0))
    sfb_one = cute.slice_(sfb_sl, (None, None, None, 0))
    nb_a = cute.size_in_bytes(cutlass.Float4E2M1FN, a_one)
    nb_sfa = cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_one)
    nb_sfb = cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_one)
    TXS = [(nb_a + cute.size_in_bytes(cutlass.Float8E4M3FN, B_ONE[j]) + nb_sfa + nb_sfb) * 2
           for j in range(4)]
    NT = NR // TNSF

    @cute.struct
    class SharedStorage:
        ab_full: cute.struct.MemRange[cutlass.Int64, AB_STAGES]
        ab_empty: cute.struct.MemRange[cutlass.Int64, AB_STAGES]
        acc_full: cute.struct.MemRange[cutlass.Int64, ACC_STAGES]
        acc_empty: cute.struct.MemRange[cutlass.Int64, ACC_STAGES]
        tmem_dealloc_mbar: cutlass.Int64
        tmem_holding_buf: cutlass.Int32
        sEpi: cute.struct.Align[
            cute.struct.MemRange[Float32, EPI_CHUNK * EPI_ROW_STRIDE], 1024]
        sA: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int8, cute.cosize(a_sl.outer)], 1024]
        sB: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(b_sl.outer)], 1024]
        sSFA: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_sl)], 1024]
        sSFB: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_sl)], 1024]

    a_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_MNK, tiled_mma.thr_id)
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_MNK, tiled_mma.thr_id)
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(CLUSTER_MNK, tiled_mma.thr_id)

    # ---------------- FC1 ----------------
    a1 = cute.make_tensor(p_w1, cute.make_ordered_layout((N1, H1, EL), order=(1, 0, 2)))
    sfa1 = cute.make_tensor(p_w1sf, blockscaled_utils.tile_atom_to_shape_SF((N1, H1, EL), 32))
    # token tiles are TNSF-row slots: view B as (N_variant, K, tile) so a variant narrower
    # than the slot still starts at the slot base
    B1 = [cute.make_tensor(p_gactq, cute.make_layout(
        (32 * (j + 1), H1, NT), stride=(H1, 1, TNSF * H1))) for j in range(4)]
    b_1 = B1[3]
    sfb1 = cute.make_tensor(p_gactsf, blockscaled_utils.tile_atom_to_shape_SF((NR, H1, 1), 32))
    ta1, tta1 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a1, a_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int8)
    TB1 = [cute.nvgpu.make_tiled_tma_atom_B(
        b_op, B1[j], B_ONE[j], (MMA_M, 32 * (j + 1), BK), MMAS[j], cluster_layout_vmnk.shape)
        for j in range(4)]
    tsa1, ttsa1 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, sfa1, sfa_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    tsb1, ttsb1 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb1, sfb_one, mma_tiler_sf, tiled_mma_sf, cluster_layout_sfb_vmnk.shape,
        internal_type=cutlass.Int16)
    k_gemm(
        tiled_mma, tiled_mma_sf, MMAS[0], MMAS[1], MMAS[2],
        cluster_layout_vmnk, cluster_layout_sfb_vmnk,
        BSLS[0], BSLS[1], BSLS[2],
        ta1, tta1, TB1[3][0], TB1[3][1], TB1[0][0], TB1[0][1], TB1[1][0], TB1[1][1],
        TB1[2][0], TB1[2][1], tsa1, ttsa1, tsb1, ttsb1,
        a_sl, b_sl, sfa_sl, sfb_sl, bias1, tile_e, tile_nr, meta, actqb, actsf, ybuf32,
        alpha_t, beta_t, lim_t, round_up, SharedStorage, a_stream, pf_on, 0,
        FC1_MTILES2, FC1_KTILES, N1, FC1_M, TXS[0], TXS[1], TXS[2], TXS[3],
    ).launch(grid=[num_cta, 1, 1], block=[NTHREADS, 1, 1], cluster=[2, 1, 1],
             stream=stream, use_pdl=pdl)

    # ---------------- FC2 ----------------
    a2 = cute.make_tensor(p_w2, cute.make_ordered_layout((H2, I_PAD, EL), order=(1, 0, 2)))
    sfa2 = cute.make_tensor(p_w2sf, blockscaled_utils.tile_atom_to_shape_SF((H2, I_PAD, EL), 32))
    B2 = [cute.make_tensor(p_actqe4, cute.make_layout(
        (32 * (j + 1), I_PAD, NT), stride=(I_PAD, 1, TNSF * I_PAD))) for j in range(4)]
    b_2 = B2[3]
    sfb2 = cute.make_tensor(p_actsf, blockscaled_utils.tile_atom_to_shape_SF((NR, I_PAD, 1), 32))
    ta2, tta2 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a2, a_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int8)
    TB2 = [cute.nvgpu.make_tiled_tma_atom_B(
        b_op, B2[j], B_ONE[j], (MMA_M, 32 * (j + 1), BK), MMAS[j], cluster_layout_vmnk.shape)
        for j in range(4)]
    tsa2, ttsa2 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, sfa2, sfa_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    tsb2, ttsb2 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb2, sfb_one, mma_tiler_sf, tiled_mma_sf, cluster_layout_sfb_vmnk.shape,
        internal_type=cutlass.Int16)
    k_gemm(
        tiled_mma, tiled_mma_sf, MMAS[0], MMAS[1], MMAS[2],
        cluster_layout_vmnk, cluster_layout_sfb_vmnk,
        BSLS[0], BSLS[1], BSLS[2],
        ta2, tta2, TB2[3][0], TB2[3][1], TB2[0][0], TB2[0][1], TB2[1][0], TB2[1][1],
        TB2[2][0], TB2[2][1], tsa2, ttsa2, tsb2, ttsb2,
        a_sl, b_sl, sfa_sl, sfb_sl, bias2, tile_e, tile_nr, meta, actqb, actsf, ybuf32,
        alpha_t, beta_t, lim_t, round_up, SharedStorage, a_stream, pf_on, 1,
        FC2_MTILES2, FC2_KTILES, H2, H, TXS[0], TXS[1], TXS[2], TXS[3],
    ).launch(grid=[num_cta, 1, 1], block=[NTHREADS, 1, 1], cluster=[2, 1, 1],
             stream=stream, use_pdl=pdl)

    k_combine(ybuf4, p2r4, topkw4, out4, Tt, NR).launch(
        grid=[CB, 1, 1], block=[COMB_THREADS, 1, 1], stream=stream, use_pdl=pdl)


_CACHE = {}


def _ptr(t, dtype, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@torch.no_grad()
def run(hidden_states, hidden_states_scale, topk_ids, topk_weights,
        gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
        gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
        local_expert_offset, requant_round_up, output):
    import cuda.bindings.driver as cuda

    T = hidden_states.shape[0]
    EL = gemm1_weights.shape[0]
    dev = hidden_states.device
    npairs = T * TOPK
    tn = 128
    abst = AB_STAGES_DEFAULT
    accst = 3
    a_stream = 1 if ((npairs + 127) // 128) <= tn else 0
    max_tiles = EL + (npairs + tn - 1) // tn
    NR = max_tiles * TNSF

    i32 = torch.int32
    u8 = torch.uint8
    tile_e = torch.empty(max_tiles, dtype=i32, device=dev)
    tile_nr = torch.empty(max_tiles, dtype=i32, device=dev)
    meta = torch.empty(4, dtype=i32, device=dev)
    pcnt = torch.empty(MAX_COUNT_CTAS * 128, dtype=i32, device=dev)
    ebase = torch.empty(128, dtype=i32, device=dev)
    gfill = torch.empty(128 * GFILL_STRIDE, dtype=i32, device=dev)
    pair2row = torch.empty(npairs, dtype=i32, device=dev)
    gact = torch.empty(NR * H1, dtype=u8, device=dev)
    gact_sf = torch.empty(NR * 96, dtype=u8, device=dev)
    actq = torch.empty(NR * I_PAD, dtype=u8, device=dev)
    act_sf = torch.empty(NR * (I_PAD // 32), dtype=u8, device=dev)
    ybuf = torch.empty((NR + 1) * H, dtype=torch.bfloat16, device=dev)

    # launch shape depends only on T, EL and the device (graph-capture safe)
    rg = min(MAX_COUNT_CTAS, max(1, (T + PAIRS_PER_COUNT_CTA - 1) // PAIRS_PER_COUNT_CTA))
    ch = (T + rg - 1) // rg
    gg = (npairs + (GATHER_THREADS // 32) - 1) // (GATHER_THREADS // 32)
    num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
    num_cta = (num_sms // 2) * 2
    cb = min(T, 10 * num_sms)
    pf_on = 1 if (a_stream and npairs >= PF_MIN_PAIRS) else 0
    pdl = True
    key = (num_cta, a_stream, pf_on, tn, accst, abst)
    compiled = _CACHE.get(key)

    args = (
        _ptr(hidden_states, Int32, 16),
        _ptr(hidden_states_scale, Int32, 16),
        _ptr(topk_ids, Int32, 16),
        _ptr(topk_weights, cutlass.BFloat16, 16),
        _ptr(gemm1_weights, cutlass.Float4E2M1FN, 32),
        _ptr(gemm1_weights_scale, cutlass.Float8E8M0FNU, 32),
        _ptr(gemm1_bias, Float32, 16),
        _ptr(gemm1_alpha, Float32, 16),
        _ptr(gemm1_beta, Float32, 16),
        _ptr(gemm1_clamp_limit, Float32, 16),
        _ptr(gemm2_weights, cutlass.Float4E2M1FN, 32),
        _ptr(gemm2_weights_scale, cutlass.Float8E8M0FNU, 32),
        _ptr(gemm2_bias, Float32, 16),
        _ptr(output, Int32, 16),
        _ptr(tile_e, Int32, 16),
        _ptr(tile_nr, Int32, 16),
        _ptr(meta, Int32, 16),
        _ptr(pcnt, Int32, 16),
        _ptr(ebase, Int32, 16),
        _ptr(gfill, Int32, 16),
        _ptr(pair2row, Int32, 16),
        _ptr(gact, Int32, 16),
        _ptr(gact_sf, cutlass.Uint8, 32),
        _ptr(gact_sf, Int32, 32),
        _ptr(gact, cutlass.Float8E4M3FN, 32),
        _ptr(actq, cutlass.Uint8, 32),
        _ptr(act_sf, cutlass.Uint8, 32),
        _ptr(actq, cutlass.Float8E4M3FN, 32),
        _ptr(ybuf, cutlass.BFloat16, 16),
        _ptr(ybuf, Int32, 16),
        Int32(T), Int32(EL), Int32(NR), Int32(int(local_expert_offset)),
        Int32(int(requant_round_up)),
        Int32(rg), Int32(ch), Int32(gg), Int32(cb),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    if compiled is None:
        g = globals()
        g["TN"], g["AB_STAGES"], g["ACC_STAGES"] = tn, abst, accst
        compiled = cute.compile(moe_jit, *args, num_cta, a_stream, pf_on, pdl, stream)
        _CACHE[key] = compiled
    compiled(*args, stream)
