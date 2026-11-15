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
# Alternative MoE kernel module for decode-sized steps, used only through the default-off TLLM_MOE_KF_DECODE /
# TLLM_MOE_KF_DECODE_DEP hook (tensorrt_llm/_torch/moe/fused_moe/kf_moe_w4a8.py). The per-expert alpha/beta/limit
# pointers are declared 4-byte aligned (scalar loads; expert-parallel rank slices are only 4-byte aligned), and the
# routing chain's first kernel (k_planf / k_count) is launched without programmatic dependent launch, because it reads
# topk_ids without griddepcontrol_wait. One TMA full-barrier arrive per stage for every byte it loads.
# Optional in-kernel Renormalize routing from the bf16 router logits (k_route, run(router_logits=...)), used only with
# the default-off TLLM_MOE_KF_DECODE_ROUTING hook flag.
# Host side: moe_jit's compile-time `combine` switch and run_unfinalized(), which stops before k_combine and returns the
# FC2 rows, routing weights and pair -> row map for a caller that finalizes itself (the hook's
# TLLM_GPTOSS_FUSE_FINALIZE_NORM path), and route(), which launches k_route alone so another MoE kernel can take its
# ids/weights (the hook's prefill route).
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
AB_STAGES = 6
ACC_STAGES = 2
PF_DIST = 8       # K tiles the rolling L2 prefetch of the weight stream runs ahead
PF_MIN_PAIRS = 8192  # below this the prefetch costs more than the L2 policy saves
EPI_CHUNK = 32    # token columns handled per epilogue chunk
EPI_ROW_STRIDE = 132  # fp32 words per token row in the epilogue smem buffer

FC1_MTILES = FC1_M // BM   # 45
FC2_MTILES = FC2_M // BM   # 23
FC1_KTILES = FC1_K // BK   # 23
FC2_KTILES = FC2_K // BK   # 23

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
             Tt: Int32, off: Int32, EL: Int32, CH: Int32, WAIT_IDS: cutlass.Constexpr):
    """Per-CTA expert histogram over a token range. One 16-byte load covers a token's four
    expert ids, so the pass costs a single vector load per token and no global round trip.
    WAIT_IDS: the ids come from k_route (in-kernel routing), so wait for it before reading them."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    cnt = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        cnt[tidx] = Int32(0)
    if cutlass.const_expr(WAIT_IDS):
        cute.arch.griddepcontrol_wait()
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
           zrow: cute.Tensor, Tt: Int32, off: Int32, EL: Int32, WAIT_IDS: cutlass.Constexpr):
    """Small-T routing: count + plan in one CTA: one 16-byte load per token covers its four expert ids,
    so the whole histogram costs ceil(T/1024) vector loads and no global round trip.
    WAIT_IDS: the ids come from k_route (in-kernel routing), so wait for it before reading them; the
    dependents' trigger below then comes after k_route has completed, so k_sgat's early topk_ids
    reads (before its own wait) still see final ids."""
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator()
    cnt = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    scan = smem.allocate_tensor(Int32, cute.make_layout(128), byte_alignment=16)
    if tidx < 128:
        cnt[tidx] = Int32(0)
    if cutlass.const_expr(WAIT_IDS):
        cute.arch.griddepcontrol_wait()
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
@cute.kernel
def k_gemm(
    tiled_mma: cute.TiledMma,
    tiled_mma_sf: cute.TiledMma,
    tiled_mma_n64: cute.TiledMma,
    b64_smem_layout_staged: cute.ComposedLayout,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tiled_mma_b32: cute.TiledMma,
    tma_atom_b32: cute.CopyAtom,
    mB32: cute.Tensor,
    b32_smem_layout_staged: cute.ComposedLayout,
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
    num_tma_bytes: cutlass.Constexpr,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    if warp_idx == TMA_WARP:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)

    smem = utils.SmemAllocator()
    storage = smem.allocate(storage_t)

    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full.data_ptr(),
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=num_tma_bytes,
        defer_sync=True,
    )
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full.data_ptr(),
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * EPI_WARPS),
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
    )
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
    sB32 = storage.sB.get_tensor(b32_smem_layout_staged.outer, swizzle=b32_smem_layout_staged.inner)
    sB64 = storage.sB.get_tensor(b64_smem_layout_staged.outer, swizzle=b64_smem_layout_staged.inner)
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
    sEpi = storage.sEpi.get_tensor(cute.make_layout(EPI_CHUNK * EPI_ROW_STRIDE))

    mma_tiler = (BM, TN, BK)
    mma_tiler_sf = (BM, TNSF, BK)
    gA = cute.local_tile(mA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gB32 = cute.local_tile(mB32, (32, BK), (None, None, None))
    gSFA = cute.local_tile(mSFA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gSFB = cute.local_tile(mSFB, cute.slice_(mma_tiler_sf, (0, None, None)), (None, None, None))

    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB32 = tiled_mma_b32.get_slice(0).partition_B(gB32)
    tCgSFA = thr_mma.partition_A(gSFA)
    tCgSFB = tiled_mma_sf.get_slice(0).partition_B(gSFB)

    cta_layout = cute.make_layout(1)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )
    tBsB32, tBgB32 = cpasync.tma_partition(
        tma_atom_b32, 0, cta_layout, cute.group_modes(sB32, 0, 3), cute.group_modes(tCgB32, 0, 3)
    )
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, 0, cta_layout, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
    )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, 0, cta_layout, cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3)
    )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    tCrA64 = tiled_mma_n64.make_fragment_A(sA)
    tCrB64 = tiled_mma_n64.make_fragment_B(sB64)
    acc_shape = tiled_mma.partition_shape_C((BM, TN))
    tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))
    acc_shape64 = tiled_mma_n64.partition_shape_C((BM, TN // 2))
    tCtAcc64_fake = tiled_mma_n64.make_fragment_C(cute.append(acc_shape64, 2 * ACC_STAGES))

    num_acc_cols = TN * ACC_STAGES
    num_sfa_cols = (BM // 32) * 4
    pipeline_init_wait(cluster_shape_mn=(1, 1))
    # descriptor prefetch, barrier init and TMEM allocation above are independent of the
    # producing kernel; only the routing plan and the gathered tokens are not
    cute.arch.griddepcontrol_wait()

    ntiles_n = meta[0]
    total_tiles = ntiles_n * MTILES

    # ------------------------------------------------------------------ TMA warp
    if warp_idx == TMA_WARP:
        ab_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, AB_STAGES)
        # Weights are read once per call only when each expert fits in a single token
        # tile; above that two or more token tiles re-read them, so keep them cacheable.
        pol_a = l2_policy_evict_first()
        pol_b = l2_policy_evict_last()
        flat = Int32(bidx)
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
            nsub = (tile_nrows[n_tile] + 31) // 32
            tAgA_s = tAgA[(None, m_tile, None, e)]
            tAgSFA_s = tAgSFA[(None, m_tile, None, e)]
            tBgSFB_s = tBgSFB[(None, n_tile, None, 0)]
            tAgA_n = tAgA_s
            if cutlass.const_expr(PF_ON):
                nxt = flat + gdim
                if not (nxt < total_tiles):
                    nxt = flat
                e_n = tile_expert[nxt // MTILES]
                tAgA_n = tAgA[(None, nxt % MTILES, None, e_n)]
            tx_all = Int32(num_tma_bytes) + nsub * (32 * BK)
            for kt in cutlass.range(KTILES, unroll=1):
                # One arrive covers every byte of the stage (A + SFA + SFB + all B slabs);
                # no expect_tx may follow the arrive or the stage could complete early.
                ab_pipeline.producer_acquire(ab_prod, expected_tx=tx_all)
                bar = ab_pipeline.producer_get_barrier(ab_prod)
                if cutlass.const_expr(PF_ON):
                    # enough token tiles in flight that the rolling prefetch beats the
                    # evict_first hint: warm the line, leave the policy at default
                    cute.copy(tma_atom_a, tAgA_s[(None, kt)], tAsA[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, kt)], tAsSFA[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfb, tBgSFB_s[(None, kt)], tBsSFB[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    for sb in cutlass.range_constexpr(TN // 32):
                        if sb < nsub:
                            cute.copy(tma_atom_b32, tBgB32[(None, n_tile * (TNSF // 32) + sb, kt, 0)],
                                      tBsB32[(None, ab_prod.index * (TN // 32) + sb)],
                                      tma_bar_ptr=bar)
                    if (kt & 1) == 0:
                        pf = kt + PF_DIST
                        if pf < KTILES:
                            cute.prefetch(tma_atom_a, tAgA_s[(None, pf)])
                        else:
                            cute.prefetch(tma_atom_a, tAgA_n[(None, pf - KTILES)])
                elif cutlass.const_expr(A_STREAM):
                    # Each expert's weights are read once per call at this shape: mark them
                    # evict_first so they stop displacing the token tile that every weight-row
                    # tile of the same expert re-reads, and pin that token tile with evict_last.
                    cute.copy(tma_atom_a, tAgA_s[(None, kt)], tAsA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, cache_policy=pol_a)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, kt)], tAsSFA[(None, ab_prod.index)],
                              tma_bar_ptr=bar, cache_policy=pol_a)
                    cute.copy(tma_atom_sfb, tBgSFB_s[(None, kt)], tBsSFB[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    for sb in cutlass.range_constexpr(TN // 32):
                        if sb < nsub:
                            cute.copy(tma_atom_b32, tBgB32[(None, n_tile * (TNSF // 32) + sb, kt, 0)],
                                      tBsB32[(None, ab_prod.index * (TN // 32) + sb)],
                                      tma_bar_ptr=bar)
                else:
                    cute.copy(tma_atom_a, tAgA_s[(None, kt)], tAsA[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfa, tAgSFA_s[(None, kt)], tAsSFA[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    cute.copy(tma_atom_sfb, tBgSFB_s[(None, kt)], tBsSFB[(None, ab_prod.index)],
                              tma_bar_ptr=bar)
                    for sb in cutlass.range_constexpr(TN // 32):
                        if sb < nsub:
                            cute.copy(tma_atom_b32, tBgB32[(None, n_tile * (TNSF // 32) + sb, kt, 0)],
                                      tBsB32[(None, ab_prod.index * (TN // 32) + sb)],
                                      tma_bar_ptr=bar)
                ab_prod.advance()
            flat += gdim
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
        # the s2t copy always moves a whole 128-token scale block; the MMA then views the
        # TN-token prefix of it
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
                tiled_mma, mma_tiler, 32, cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
            ),
        )
        tCtAcc64_base = cute.make_tensor(acc_tmem_ptr, tCtAcc64_fake.layout)
        tCtSFB64 = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_n64, (BM, TN // 2, BK), 32,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
            ),
        )
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
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
        flat = Int32(bidx)
        while flat < total_tiles:
            wide = tile_nrows[flat // MTILES] > (TN // 2)
            tCtAcc = tCtAcc_base[(None, None, None, acc_prod.index)]
            tCtAcc64 = tCtAcc64_base[(None, None, None, 2 * acc_prod.index)]
            acc_pipeline.producer_acquire(acc_prod)
            for kt in cutlass.range(KTILES, unroll=1):
                ab_pipeline.consumer_wait(ab_cons)
                st = (None, None, None, None, ab_cons.index)
                cute.copy(tiled_s2t_a, s_sfa[st], t_sfa)
                cute.copy(tiled_s2t_b, s_sfb[st], t_sfb)
                crd = (None, None, None, ab_cons.index)
                if wide:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, kt != 0)
                    cute.gemm(tiled_mma, tCtAcc, [tCrA[crd], tCtSFA], [tCrB[crd], tCtSFB], tCtAcc)
                else:
                    tiled_mma_n64.set(tcgen05.Field.ACCUMULATE, kt != 0)
                    crd64 = (None, None, None, 2 * ab_cons.index)
                    cute.gemm(tiled_mma_n64, tCtAcc64, [tCrA64[crd], tCtSFA],
                              [tCrB64[crd64], tCtSFB64], tCtAcc64)
                ab_pipeline.consumer_release(ab_cons)
                ab_cons.advance()
            acc_pipeline.producer_commit(acc_prod)
            acc_prod.advance()
            flat += gdim
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
        flat = Int32(bidx)
        while flat < total_tiles:
            n_tile = flat // MTILES
            m_tile = flat % MTILES
            e = tile_expert[n_tile]
            nvr = tile_nrows[n_tile]
            prow = m_tile * BM + tidx
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
            for ch in cutlass.range_constexpr(TN // EPI_CHUNK):
                if ch * EPI_CHUNK < nvr:
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, ch)], rAcc)
                    if cutlass.const_expr(MODE == 0):
                        wpos = tidx
                    else:
                        wpos = blk * 32 + 4 * (lane % 8) + lane // 8
                    # one pad word per 32 rows: without it the four threads that share a
                    # token read rows 32 apart and collide 4-way on a single smem bank
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
                            # amax <= 56 (clamped SwiGLU), so the 2^64 pre-scale is exact and
                            # keeps amax/448 normal; without it a tiny amax lands on a
                            # subnormal and frexp's exponent would be wrong.
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
            flat += gdim
        tmem.relinquish_alloc_permit()
        tmem.free(acc_tmem_ptr, 512)
    cute.arch.griddepcontrol_launch_dependents()


# ------------------------------------------------------------------ in-kernel routing (optional)
# Renormalize routing (gpt-oss): top-4 of the 128 bf16 router logits, then softmax over the four selected,
# written as int32 ids and bf16 weights in descending order. Used only when run() gets router_logits
# (TLLM_MOE_KF_DECODE_ROUTING=1 in the engine hook); it replaces the hook's torch topk/softmax/cast kernels.
#
# Selection order is the trtllm-gen routing's: each logit becomes the 32-bit key
# (order-preserving bf16 bits << 16) | (65535 - expert), and the four largest keys win, so equal logits go to
# the lower expert id first (topk::TopKRedType::makeCmpVal in the trtllm-gen routing kernels).
# The weights follow torch.softmax(values.float(), -1).to(bfloat16) operation by operation: x - max, the
# libdevice __nv_expf instruction sequence (non-ftz), sum (e0 + e2) + (e1 + e3) (the 4-lane butterfly of
# softmax_warp_forward), IEEE division, then round-to-nearest-even bf16. All in explicit-rounding PTX so the
# compiler cannot contract or approximate any of it.
ROUTE_THREADS = 256  # one warp per token
NEXP = 128
INT_MIN = -2147483648


@dsl_user_op
def imax_s32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "max.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def warp_max_s32(a: Int32, *, loc=None, ip=None) -> Int32:
    """Max over the full warp in one instruction (integer redux.sync, sm_80+); every lane must take part."""
    return Int32(llvm.inline_asm(T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip)],
        "redux.sync.max.s32 $0, $1, 0xffffffff;", "=r,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def drop_if_eq(k: Int32, m: Int32, *, loc=None, ip=None) -> Int32:
    """k, or INT_MIN when k == m (removes the key just selected)."""
    return Int32(llvm.inline_asm(T.i32(),
        [Int32(k).ir_value(loc=loc, ip=ip), Int32(m).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.eq.s32 p, $1, $2; selp.b32 $0, 0x80000000, $1, p; }", "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def sub_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "sub.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "div.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def expf_libdevice(x: Float32, *, loc=None, ip=None) -> Float32:
    """__nv_expf (libdevice.10.bc, __CUDA_FTZ=0), instruction for instruction: what expf() compiles to in
    nvcc without fast math, i.e. in torch's softmax and in the trtllm-gen routing's calcSoftmax.
    0x3BBB989D = rn(log2e_hi / 252); 0x4B400001 = 12582913; 0x4B40007F = 12583039;
    0x3FB8AA3B / 0x32A57060 = log2e hi / lo."""
    return Float32(llvm.inline_asm(T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 t, j, r, f, s, p; .reg .b32 jb;\n"
        " fma.rn.sat.f32 t, $1, 0f3BBB989D, 0f3F000000;\n"
        " fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
        " sub.rn.f32 r, j, 0f4B40007F;\n"
        " neg.f32 r, r;\n"
        " fma.rn.f32 f, $1, 0f3FB8AA3B, r;\n"
        " fma.rn.f32 f, $1, 0f32A57060, f;\n"
        " mov.b32 jb, j;\n"
        " shl.b32 jb, jb, 23;\n"
        " mov.b32 s, jb;\n"
        " ex2.approx.ftz.f32 p, f;\n"
        " mul.rn.f32 $0, p, s; }",
        "=f,f", has_side_effects=False,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.kernel
def k_route(lg2: cute.Tensor, ids4: cute.Tensor, tw2: cute.Tensor, Tt: Int32):
    """One warp per token: lane l holds logits 4l..4l+3 (one 8-byte load), four rounds of a warp max
    (redux.sync, as the trtllm-gen routing on sm_100+) over the packed keys pick the top 4 in descending order, then lane 0 computes the softmax and stores
    16 bytes of ids and 8 bytes of bf16 weights."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    t = bidx * (ROUTE_THREADS // 32) + warp
    # the plan kernel waits for this whole grid before it reads the ids, so it may launch now
    cute.arch.griddepcontrol_launch_dependents()
    if t < Tt:
        rv = cute.make_rmem_tensor(cute.make_layout(2), Int32)
        cute.autovec_copy(lg2[(t * 32 + lane, None)], rv)
        bb = [rv[0] & 0xFFFF, (rv[0] >> 16) & 0xFFFF, rv[1] & 0xFFFF, (rv[1] >> 16) & 0xFFFF]
        ks = []
        for q in cutlass.range_constexpr(4):
            # signed order of the bf16 value: bits for >= +0, 32767 - bits (negative) for the sign-set half
            sv = bb[q] + (bb[q] >> 15) * (Int32(32767) - bb[q] - bb[q])
            ks.append(sv * 65536 + (Int32(65535 - q) - lane * 4))
        sel = []
        for r in cutlass.range_constexpr(TOPK):
            # warp-uniform branch (one token per warp), so all 32 lanes reach the redux
            m = warp_max_s32(imax_s32(imax_s32(ks[0], ks[1]), imax_s32(ks[2], ks[3])))
            sel.append(m)
            for q in cutlass.range_constexpr(4):
                ks[q] = drop_if_eq(ks[q], m)
        if lane == 0:
            ri = cute.make_rmem_tensor(cute.make_layout(TOPK), Int32)
            vals = []
            for r in cutlass.range_constexpr(TOPK):
                ri[r] = Int32(65535) - (sel[r] & 0xFFFF)
                # high half as 16 unsigned bits (independent of the shift's signedness): h = sv mod 2^16,
                # so bf16 bits = h for h < 2^15, else 32767 - (h - 65536) = 98303 - h
                hb = (sel[r] >> 16) & 0xFFFF
                hn = hb >> 15
                vals.append(bits_f32((hb + hn * (Int32(98303) - hb - hb)) << 16))
            es = [expf_libdevice(sub_rn(vals[r], vals[0])) for r in range(TOPK)]
            ssum = add_rn(add_rn(es[0], es[2]), add_rn(es[1], es[3]))
            ws = [div_rn(es[r], ssum) for r in range(TOPK)]
            rw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
            rw[0] = cvt_bf16x2(ws[1], ws[0])
            rw[1] = cvt_bf16x2(ws[3], ws[2])
            cute.autovec_copy(ri, ids4[(t, None)])
            cute.autovec_copy(rw, tw2[(t, None)])


# ------------------------------------------------------------------ host side
@cute.jit
def moe_jit(
    p_hs32, p_hssf, p_ids, p_topkw, p_w1, p_w1sf, p_b1, p_alpha, p_beta, p_lim,
    p_w2, p_w2sf, p_b2, p_out32,
    p_tile_e, p_tile_nr, p_meta, p_pcnt, p_ebase, p_gfill, p_pair2row,
    p_gact32, p_gactsf, p_gactsf32, p_gactq, p_actqb, p_actsf, p_actqe4, p_ybuf16, p_ybuf32,
    p_logits32, p_topkw32,
    Tt: Int32, EL: Int32, NR: Int32, off: Int32, round_up: Int32,
    RG: Int32, CH: Int32, GG: Int32, CB: Int32, RTG: Int32,
    num_sms: cutlass.Constexpr,
    a_stream: cutlass.Constexpr,
    pf_on: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    route: cutlass.Constexpr,
    combine: cutlass.Constexpr,
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

    # With in-kernel routing, k_route is the chain head (no programmatic launch: it reads router_logits
    # only after the engine's previous kernel has fully completed) and writes the ids/weights scratch that
    # p_ids/p_topkw point at; the count/plan kernel then launches programmatically and waits for it.
    head_pdl = bool(route) and bool(pdl)  # both compile-time constants
    if cutlass.const_expr(route):
        lg2 = cute.make_tensor(p_logits32, cute.make_layout((Tt * 32, 2), stride=(2, 1)))
        ids4 = cute.make_tensor(p_ids, cute.make_layout((Tt, TOPK), stride=(TOPK, 1)))
        tw2 = cute.make_tensor(p_topkw32, cute.make_layout((Tt, 2), stride=(2, 1)))
        k_route(lg2, ids4, tw2, Tt).launch(
            grid=[RTG, 1, 1], block=[ROUTE_THREADS, 1, 1], stream=stream, use_pdl=False)
    if cutlass.const_expr(a_stream):
        # few pairs per expert: one CTA counts and plans in a single pass, saving a launch
        # chain head: no programmatic launch, so it reads topk_ids only after the engine's previous kernel
        # (which may trigger its dependents early) has fully completed; it never calls griddepcontrol_wait
        # (unless the ids come from k_route, which it then waits for)
        k_planf(ids_t, ebase, gfill, tile_e, tile_nr, meta, zrow, Tt, off, EL, route).launch(
            grid=[1, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=head_pdl)
    else:
        # many pairs: the histogram has to be spread over the SMs before the 1-CTA plan
        k_count(ids_t, pcnt, Tt, off, EL, CH, route).launch(  # chain head unless routed (as k_planf)
            grid=[RG, 1, 1], block=[COUNT_THREADS, 1, 1], stream=stream, use_pdl=head_pdl)
        k_plan(pcnt, ebase, gfill, tile_e, tile_nr, meta, zrow, RG, EL).launch(
            grid=[1, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=pdl)
    k_sgat(ids_t, ebase, gfill, pair2row, hs4, hssf, gact4, gactsf, actqb, actsf,
           zrow, npairs, off, EL).launch(
        grid=[GG, 1, 1], block=[GATHER_THREADS, 1, 1], stream=stream, use_pdl=pdl)

    mma_tiler = (BM, TN, BK)
    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.ONE, (BM, TN),
    )
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
    )
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, cutlass.Int8, AB_STAGES)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, cutlass.Float8E4M3FN, AB_STAGES)
    sfa_sl = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, 32, AB_STAGES)
    # scale factors keep the 128-token atom even when the MMA tile is 64 tokens wide
    mma_tiler_sf = (BM, TNSF, BK)
    tiled_mma_sf = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.ONE, (BM, TNSF),
    )
    sfb_sl = blockscaled_utils.make_smem_layout_sfb(tiled_mma_sf, mma_tiler_sf, 32, AB_STAGES)
    tiled_mma_b32 = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.ONE, (BM, 32),
    )
    b32_sl = sm100_utils.make_smem_layout_b(
        tiled_mma_b32, (BM, 32, BK), cutlass.Float8E4M3FN, AB_STAGES * (TN // 32))
    b32_one = cute.slice_(b32_sl, (None, None, None, 0))
    tiled_mma_n64 = sm100_utils.make_blockscaled_trivial_tiled_mma(
        cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
        OperandMajorMode.K, OperandMajorMode.K,
        cutlass.Float8E8M0FNU, 32, tcgen05.CtaGroup.ONE, (BM, TN // 2),
    )
    b64_sl = sm100_utils.make_smem_layout_b(
        tiled_mma_n64, (BM, TN // 2, BK), cutlass.Float8E4M3FN, AB_STAGES * 2)
    a_one = cute.slice_(a_sl, (None, None, None, 0))
    b_one = cute.slice_(b_sl, (None, None, None, 0))
    sfa_one = cute.slice_(sfa_sl, (None, None, None, 0))
    sfb_one = cute.slice_(sfb_sl, (None, None, None, 0))
    num_tma_bytes = (
        cute.size_in_bytes(cutlass.Float4E2M1FN, a_one)
        + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfa_one)
        + cute.size_in_bytes(cutlass.Float8E8M0FNU, sfb_one)
    )

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

    a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
    b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB((1, 1), tiled_mma_sf.thr_id)

    # ---------------- FC1 ----------------
    a1 = cute.make_tensor(p_w1, cute.make_ordered_layout((N1, H1, EL), order=(1, 0, 2)))
    sfa1 = cute.make_tensor(p_w1sf, blockscaled_utils.tile_atom_to_shape_SF((N1, H1, EL), 32))
    b_1 = cute.make_tensor(p_gactq, cute.make_ordered_layout((NR, H1, 1), order=(1, 0, 2)))
    sfb1 = cute.make_tensor(p_gactsf, blockscaled_utils.tile_atom_to_shape_SF((NR, H1, 1), 32))
    ta1, tta1 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a1, a_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int8)
    tb1, ttb1 = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_1, b_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
    tb1s, ttb1s = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_1, b32_one, (BM, 32, BK), tiled_mma_b32, cluster_layout_vmnk.shape)
    tsa1, ttsa1 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, sfa1, sfa_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    tsb1, ttsb1 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb1, sfb_one, mma_tiler_sf, tiled_mma_sf, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    k_gemm(
        tiled_mma, tiled_mma_sf, tiled_mma_n64, b64_sl, ta1, tta1, tb1, ttb1, tiled_mma_b32, tb1s, ttb1s, b32_sl, tsa1, ttsa1, tsb1, ttsb1,
        a_sl, b_sl, sfa_sl, sfb_sl, bias1, tile_e, tile_nr, meta, actqb, actsf, ybuf32,
        alpha_t, beta_t, lim_t, round_up, SharedStorage, a_stream, pf_on, 0, FC1_MTILES, FC1_KTILES,
        N1, num_tma_bytes,
    ).launch(grid=[num_sms, 1, 1], block=[NTHREADS, 1, 1], stream=stream, use_pdl=pdl)

    # ---------------- FC2 ----------------
    a2 = cute.make_tensor(p_w2, cute.make_ordered_layout((H2, I_PAD, EL), order=(1, 0, 2)))
    sfa2 = cute.make_tensor(p_w2sf, blockscaled_utils.tile_atom_to_shape_SF((H2, I_PAD, EL), 32))
    b_2 = cute.make_tensor(p_actqe4, cute.make_ordered_layout((NR, I_PAD, 1), order=(1, 0, 2)))
    sfb2 = cute.make_tensor(p_actsf, blockscaled_utils.tile_atom_to_shape_SF((NR, I_PAD, 1), 32))
    ta2, tta2 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a2, a_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int8)
    tb2, ttb2 = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_2, b_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape)
    tb2s, ttb2s = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_2, b32_one, (BM, 32, BK), tiled_mma_b32, cluster_layout_vmnk.shape)
    tsa2, ttsa2 = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, sfa2, sfa_one, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    tsb2, ttsb2 = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb2, sfb_one, mma_tiler_sf, tiled_mma_sf, cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16)
    k_gemm(
        tiled_mma, tiled_mma_sf, tiled_mma_n64, b64_sl, ta2, tta2, tb2, ttb2, tiled_mma_b32, tb2s, ttb2s, b32_sl, tsa2, ttsa2, tsb2, ttsb2,
        a_sl, b_sl, sfa_sl, sfb_sl, bias2, tile_e, tile_nr, meta, actqb, actsf, ybuf32,
        alpha_t, beta_t, lim_t, round_up, SharedStorage, a_stream, pf_on, 1, FC2_MTILES, FC2_KTILES,
        H2, num_tma_bytes,
    ).launch(grid=[num_sms, 1, 1], block=[NTHREADS, 1, 1], stream=stream, use_pdl=pdl)

    if cutlass.const_expr(combine):
        k_combine(ybuf4, p2r4, topkw4, out4, Tt, NR).launch(
            grid=[CB, 1, 1], block=[COMB_THREADS, 1, 1], stream=stream, use_pdl=pdl)


@cute.jit
def route_jit(p_logits32, p_ids, p_topkw32, Tt: Int32, RTG: Int32, stream):
    """k_route alone (as moe_jit's routed head, without the MoE chain behind it)."""
    lg2 = cute.make_tensor(p_logits32, cute.make_layout((Tt * 32, 2), stride=(2, 1)))
    ids4 = cute.make_tensor(p_ids, cute.make_layout((Tt, TOPK), stride=(TOPK, 1)))
    tw2 = cute.make_tensor(p_topkw32, cute.make_layout((Tt, 2), stride=(2, 1)))
    k_route(lg2, ids4, tw2, Tt).launch(
        grid=[RTG, 1, 1], block=[ROUTE_THREADS, 1, 1], stream=stream, use_pdl=False)


_CACHE = {}
# run() accepts router_logits= (in-kernel routing; read by the engine hook for TLLM_MOE_KF_DECODE_ROUTING)
SUPPORTS_ROUTER_LOGITS = True


def _ptr(t, dtype, align=16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


@torch.no_grad()
def run(hidden_states, hidden_states_scale, topk_ids, topk_weights,
        gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
        gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
        local_expert_offset, requant_round_up, output, router_logits=None):
    """router_logits (optional, contiguous bf16 [T, 128], 16-byte aligned, all 128 experts local): route in-kernel
    (k_route: top-4 + softmax over the four, as the trtllm-gen Renormalize routing). topk_ids / topk_weights are then
    outputs: None (scratch), or contiguous [T, 4] int32 / bf16 tensors that receive the routing (for tests)."""
    _launch(hidden_states, hidden_states_scale, topk_ids, topk_weights,
            gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
            local_expert_offset, requant_round_up, output, router_logits, 1)


@torch.no_grad()
def route(router_logits):
    """(topk_ids [T, 4] int32, topk_weights [T, 4] bf16) from contiguous, 16-byte aligned bf16 [T, 128] logits:
    k_route's top-4 + softmax over the four, exactly as run(router_logits=...) routes."""
    import cuda.bindings.driver as cuda

    T = router_logits.shape[0]
    assert (router_logits.dtype == torch.bfloat16 and tuple(router_logits.shape) == (T, NEXP)
            and router_logits.is_contiguous() and router_logits.data_ptr() % 16 == 0), "route needs bf16 [T, 128]"
    dev = router_logits.device
    topk_ids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
    topk_weights = torch.empty(T, TOPK, dtype=torch.bfloat16, device=dev)
    rtg = (T + (ROUTE_THREADS // 32) - 1) // (ROUTE_THREADS // 32)
    args = (_ptr(router_logits, Int32, 16), _ptr(topk_ids, Int32, 16), _ptr(topk_weights, Int32, 16),
            Int32(T), Int32(rtg))
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    compiled = _CACHE.get("route")
    if compiled is None:
        compiled = cute.compile(route_jit, *args, stream)
        _CACHE["route"] = compiled
    compiled(*args, stream)
    return topk_ids, topk_weights


@torch.no_grad()
def run_unfinalized(hidden_states, hidden_states_scale, topk_ids, topk_weights,
                    gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                    gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                    local_expert_offset, requant_round_up, router_logits=None):
    """``run`` without k_combine: [FC2 rows [NR + 1, H] bf16 (bias included; row pair2row[t, k] is token t's k-th
    expert, the last row and tile padding are unused), routing weights [T, 4] bf16, pair2row [T, 4] int32 (-1 for
    experts outside [offset, offset + EL))], the trtllm-gen unfinalized op's triple. k_combine's fp32 fma chain
    (k = 0..3 from +0, one round to bf16) is the caller's to run."""
    return _launch(hidden_states, hidden_states_scale, topk_ids, topk_weights,
                   gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
                   gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
                   local_expert_offset, requant_round_up, None, router_logits, 0)


def _launch(hidden_states, hidden_states_scale, topk_ids, topk_weights,
            gemm1_weights, gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, gemm2_bias,
            local_expert_offset, requant_round_up, output, router_logits, combine):
    import cuda.bindings.driver as cuda

    T = hidden_states.shape[0]
    EL = gemm1_weights.shape[0]
    dev = hidden_states.device
    npairs = T * TOPK
    route = 1 if router_logits is not None else 0
    if route:
        assert (router_logits.dtype == torch.bfloat16 and tuple(router_logits.shape) == (T, NEXP)
                and router_logits.is_contiguous() and router_logits.data_ptr() % 16 == 0 and EL == NEXP
                and int(local_expert_offset) == 0), "in-kernel routing needs bf16 [T, 128] logits, 128 local experts"
        if topk_ids is None:
            topk_ids = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
        if topk_weights is None:
            topk_weights = torch.empty(T, TOPK, dtype=torch.bfloat16, device=dev)
        assert (topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.bfloat16
                and tuple(topk_ids.shape) == (T, TOPK) and tuple(topk_weights.shape) == (T, TOPK)
                and topk_ids.is_contiguous() and topk_weights.is_contiguous()
                and topk_ids.data_ptr() % 16 == 0 and topk_weights.data_ptr() % 16 == 0)
    # Routing is over 128 global experts, so a local expert holds about npairs/128 rows;
    # when that fits one token tile each expert's weights are streamed exactly once.
    # A local expert holds about npairs/128 rows (routing is over 128 global experts). When
    # that fits a 64-token tile, halving the token tile halves the B staging buffer and buys
    # two more TMA stages (6 -> 8) out of the same shared memory. An expert that overflows
    # just takes a second tile, so this is a pure performance heuristic.
    tn = 64 if npairs <= 6400 else 128
    abst = 8 if tn == 64 else 6
    accst = 4 if tn == 64 else 2
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
    cb = min(T, 10 * num_sms)
    pf_on = 1 if (a_stream and npairs >= PF_MIN_PAIRS) else 0
    pdl = True
    rtg = (T + (ROUTE_THREADS // 32) - 1) // (ROUTE_THREADS // 32)
    key = (num_sms, a_stream, pf_on, tn, accst, route, combine)
    compiled = _CACHE.get(key)

    args = (
        _ptr(hidden_states, Int32, 16),
        _ptr(hidden_states_scale, Int32, 16),
        _ptr(topk_ids, Int32, 16),
        _ptr(topk_weights, cutlass.BFloat16, 16),
        _ptr(gemm1_weights, cutlass.Float4E2M1FN, 32),
        _ptr(gemm1_weights_scale, cutlass.Float8E8M0FNU, 32),
        _ptr(gemm1_bias, Float32, 16),
        _ptr(gemm1_alpha, Float32, 4),  # scalar per-expert loads: DEP-rank slices are only 4-byte aligned
        _ptr(gemm1_beta, Float32, 4),
        _ptr(gemm1_clamp_limit, Float32, 4),
        _ptr(gemm2_weights, cutlass.Float4E2M1FN, 32),
        _ptr(gemm2_weights_scale, cutlass.Float8E8M0FNU, 32),
        _ptr(gemm2_bias, Float32, 16),
        _ptr(output if combine else ybuf, Int32, 16),  # read only by k_combine
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
        _ptr(router_logits if route else topk_ids, Int32, 16),  # read only when route
        _ptr(topk_weights, Int32, 16),
        Int32(T), Int32(EL), Int32(NR), Int32(int(local_expert_offset)),
        Int32(int(requant_round_up)),
        Int32(rg), Int32(ch), Int32(gg), Int32(cb), Int32(rtg),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    if compiled is None:
        g = globals()
        g["TN"], g["AB_STAGES"], g["ACC_STAGES"] = tn, abst, accst
        compiled = cute.compile(moe_jit, *args, num_sms, a_stream, pf_on, pdl, route, combine, stream)
        g["TN"], g["AB_STAGES"], g["ACC_STAGES"] = 128, 6, 2
        _CACHE[key] = compiled
    compiled(*args, stream)
    if not combine:
        return [ybuf.view(NR + 1, H), topk_weights, pair2row.view(T, TOPK)]
