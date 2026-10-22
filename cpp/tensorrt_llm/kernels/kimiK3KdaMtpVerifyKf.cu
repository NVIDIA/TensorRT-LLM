/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Kimi K3 KDA multi-token speculative verify with a BF16 recurrent state, fused with the gated output RMSNorm and the
// 1x128 MXFP8 quantization of its output (trtllm::kda_mtp_decode_fp8, packed token layout).
//
// Origin: NVIDIA Kernel Factory campaigns on the P1b package (KDA MTP verify, BF16 state, MXFP8 output). This is the
// round-2 winner, cuda_cpp candidate ab18acf7 (an extension of round 1's 9a414576, which the previous version of this
// file integrated), with the two layout changes the integration needs: the num_accepted_tokens slot stride and the
// wave-quantized grid of launchNs (the candidate's launch_ns, with these two changes, is the "plus" variant of the
// campaign archive). Validated on Rubin against the campaign package, with the deployed KV cache manager v2 layouts and
// poisoned padding. The code between the "Kernel Factory candidate" markers is the candidate's (KdaParams, KdaMaps,
// Cfg, the kernel), kept as retrieved so that a later candidate with the same interface drops in, apart from an SM10x
// guard and one added instruction: the kernel waits on its programmatic-launch predecessor (griddepcontrol.wait)
// before anything else, so that it can be launched with PDL. Around it: the tensor maps are encoded through
// CUDADriverWrapper (the candidate's encode calls, with their checks), the launch goes through cudaLaunchKernelEx with
// the PDL attribute, the dynamic shared memory limit is set per launch.
//
// Design: CTAs of 256 threads, 2 per SM, each walking a strided list of up to 64 (request, head) items (more items run
// whole waves of such CTAs, see launchNs). The BF16 state S0 [V = 128, K = 128] is TMA-loaded (128B swizzle) one item
// ahead into a double buffer and is the A operand of P = S0 [Kd | Qd]^T on tcgen05 (Kd = k * C, Qd = q * C, C the
// cumulative decay; hi/lo BF16 split of the token side). The token couplings (Gram) are computed in FP32 on CUDA cores
// by the value warps at NS = 3 and by one M = 64 MMA chain at NS = 5, the per-row forward substitution runs on CUDA
// cores (thread = value row), and the committed state (S0 + U Ku^T) * C_a comes from a rank-(a + 1) MMA whose
// accumulator is seeded with S0 by eight identity MMAs; it is rounded to nearest even BF16 in place and TMA-stored.
// Warps 0-3 (q / k conv, norms, MMA operands) and warps 4-7 (gates, decay, v conv, solve, gated RMSNorm + MXFP8
// epilogue) split the work; the latter run one item ahead of the commit epilogue. Token rows, conv history, betas and
// replay caches arrive by TMA (the replay caches of the first item by bulk copies); each CTA holds 256 TMEM columns, so
// two CTAs fill the 512 columns of an SM. The kernel encodes 17 tensor maps per launch on the host (paid at capture
// under CUDA graphs).
//
// Programmatic dependent launch: the kernel waits before reading anything and never triggers early (a CTA counts as
// triggered when it exits), so it is safe after any predecessor and before any dependent.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kimiK3KdaMtpVerifyKf.h"

#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <limits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// ===================================== Kernel Factory candidate (P1b ab18acf7) =====================================

#pragma nv_diag_suppress 177 // the candidate's unused helpers and locals

struct KdaParams
{
    __nv_bfloat16 const *xq, *xk, *xv, *g, *beta, *og;
    long long sxq, sxk, sxv, sg, sbeta, sog; // row strides (elements)
    float const *wq, *wk, *wv, *A_log, *dt_bias, *onorm_w;
    int const *slots, *nacc;
    long long snacc; // num_accepted_tokens slot stride (elements; the hybrid cache keeps one count block per slot)
    __nv_bfloat16 *csq, *csk, *csv;
    long long csq_s, csq_p, csk_s, csk_p, csv_s, csv_p; // slot / position strides (elements)
    __nv_bfloat16 *kc, *vc, *bc;
    float* gc;
    long long kc_s, kc_p, vc_s, vc_p, gc_s, gc_p, bc_s, bc_p;
    __nv_bfloat16 const* state;
    long long sstate0; // recurrent_state slot stride (elements; deployed pool: > H*128*128)
    uint8_t *out, *out_scale;
    int B, H, T, POOL;
};

typedef __nv_bfloat16 bf16;

constexpr int MAXN = 64; // max items per CTA (schedule table)

struct KdaMaps
{
    CUtensorMap state;  // 4D {64, H*128, 2, POOL} (slot stride = recurrent_state.stride(0)) box {64, 128, 2, 1}, SW128
    CUtensorMap x[4];   // x_q, x_k, x_v, onorm_g: 2D {D, T}, box {128, NT}
    CUtensorMap g;      // 2D {D, T}, box {128, NT}
    CUtensorMap csl[3]; // conv caches (load): 3D {D, S, POOL}, box {128, 3, 1}
    CUtensorMap css[3]; // conv caches (store): 3D {D, S, POOL}, box {128, 3 + NS, 1}
    CUtensorMap bnew;   // beta: 2D {H, T} (row stride BS), box {8, NT}
    CUtensorMap brep;   // beta_cache: 3D {H, NS, POOL}, box {8, NS, 1}
    CUtensorMap
        cf[3]; // k_cache, v_cache (bf16), g_cache (f32): 3D {D, NS, POOL}, box {128, NS, 1} (item 0: all positions)
};

template <int NS>
struct Cfg
{
    static constexpr int NT = NS + 1;              // new tokens
    static constexpr int PROD = (NS <= 3) ? 1 : 3; // warp issuing the conv-window store + raw refill (NS=5: warp 3,
                                                   // which also owns the state loads and reaches S2 early)
    static constexpr bool ID = true;      // commit accumulator seeded with S0 by identity MMAs (else FMA epilogue)
    static constexpr bool CG = (NS <= 3); // token Gram in FP32 on CUDA cores by group V (else M=64 MMA chain)
    static constexpr int GE = (2 * NS + 1) * (2 * NS) / 2 + (NS + 1) * (NS + 1) + (NS + 1) * NS / 2; // Gram entries
    static constexpr int GE_PAD = (GE <= 64) ? 64 : 128;
    static constexpr int NROWS = 3 * NS + 2;       // X rows: replay Kd [0,NS), new Kd [NS,2NS+1), Qd [2NS+1,3NS+2)
    static constexpr int NP = (NROWS + 7) / 8 * 8; // MMA1 N (16 / 24)
    static constexpr int XH = NP * 128;            // bytes per (split, k-half) SW128 region (multiple of 1024)
    static constexpr int KUR = (2 * NS + 1 + 7) / 8 * 8; // Ku rows (8 / 16) = Gram MMA N
    static constexpr int KUH = KUR * 128;
    static constexpr int K2 = (3 * NT + 15) / 16 * 16;   // commit MMA K (16 / 32)
    static constexpr int SBO2 = K2 * 16;                 // no-swizzle K-major: 8-row group stride
    static constexpr int OFF_ST = 0;                     // 2 x 32 KB state stages, [half][128 rows][128 B] SW128
    static constexpr int OFF_X = 65536;                  // [half][hi|lo][NP rows][128 B]
    static constexpr int OFF_KU = OFF_X + 4 * XH;        // [half][hi|lo][KUR rows][128 B]
    static constexpr int OFF_A2 = OFF_X;                 // commit operands alias X/Ku (dead after MMA1)
    static constexpr int OFF_B2 = OFF_X + 128 * K2 * 2;
    static constexpr int OFF_RAW = OFF_KU + 4 * KUH;
    static_assert(2 * 128 * K2 * 2 <= 4 * XH + 4 * KUH, "commit operands must fit in the X/Ku region");
    // raw staging (bytes, relative to OFF_RAW)
    static constexpr int R_CQ = 0; // q: [3 history rows (positions a..a+2) | NT token rows]
    static constexpr int R_XQ = R_CQ + 3 * 256;
    static constexpr int R_CK = R_CQ + (3 + NT) * 256;
    static constexpr int R_XK = R_CK + 3 * 256;
    static constexpr int R_CV = R_CK + (3 + NT) * 256;
    static constexpr int R_XV = R_CV + 3 * 256;
    static constexpr int R_OG = R_CV + (3 + NT) * 256;
    static constexpr int R_GR = R_OG + NT * 256;
    static constexpr int R_BN = R_GR + NT * 256; // beta of the new tokens: [NT][8 heads] bf16 (box at h & ~7)
    static constexpr int R_BR = R_BN + 128;      // beta_cache of the replay tokens: [NS][8 heads] bf16
    static constexpr int R_KC = R_BR + 128;      // replay caches (a positions)
    static constexpr int R_VC = R_KC + NS * 256;
    static constexpr int R_GC = R_VC + NS * 256;
    static constexpr int RAW_BYTES = R_GC + NS * 512;
    static constexpr int RAW_FIXED = R_BN + NT * 16 + NS * 16; // bytes always loaded; + a * 1024 for replay caches
    static constexpr int RAW_TOK = 5 * NT * 256 + NT * 16;     // slot-independent part: token rows + new-token betas
    static constexpr int NRAW = (NS <= 3) ? 2 : 1;             // raw staging buffers (double-buffered when smem allows)
    static constexpr int OFF_MISC = OFF_RAW + NRAW * RAW_BYTES;
    // misc (bytes, relative to OFF_MISC)
    static constexpr int M_CA = 0;                 // [2][128] f32 C_a (by item parity: group V runs one item ahead)
    static constexpr int M_GRAM = 1024;            // [NP][16] f32
    static constexpr int M_GN
        = M_GRAM + (NROWS * 64 + 127) / 128 * 128; // [NT][128] f32 cumulative decay C of the new tokens
    static constexpr int M_SOG = M_GN + NT * 512;  // [NT][128] f32 sigmoid(onorm_g)
    static constexpr int M_GP = M_GN;             // [2NP][16] f32 Gram partials (hi rows, lo rows); gates dead after S1
    static_assert(2 * NP * 64 <= NT * 512, "Gram partials must fit in the gate table");
    static constexpr int M_KR = M_SOG + NT * 512; // [NS][128] f32 commit-B values of the replay tokens (group V -> QK)
    static constexpr int M_BETA = M_KR + NS * 512;              // [16] f32: beta of new tokens [0,NT), replay [8,8+NS)
    static constexpr int M_RED = M_BETA + 64;                   // 3 scratch areas of [4][16] f32
    static constexpr int M_SCHED = M_RED + 3 * 256;             // [MAXN] int4 (slot, a, request, head)
    static constexpr int M_BAR = M_SCHED + MAXN * 16;           // st0, st1, m1, m2, raw0, raw1
    static constexpr int M_TMEM = M_BAR + 48;
    static constexpr int M_ID = (M_TMEM + 8 + 127) / 128 * 128; // 16x16 BF16 identity (no-swizzle K-major), 512 B
    static constexpr int SMEM_BYTES = OFF_MISC + M_ID + (ID ? 512 : 0) + 1024;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100) // SM10x

// ------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(cnt));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase)
{
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t}" ::"r"(bar),
        "r"(phase), "r"(0x989680) // suspend-time hint: park the warp until the phase flips instead of re-issuing
        : "memory");
}

__device__ __forceinline__ void mbar_wait_w(uint32_t bar, uint32_t phase)
{
    mbar_wait(bar, phase);
    __syncwarp();
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void tma_load_3d(uint32_t dst, CUtensorMap const* map, int c0, int c1, int c2, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
        "[%5];" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, CUtensorMap const* map, int c0, int c1, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
        "[%4];" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void tma_store_3d(CUtensorMap const* map, uint32_t src, int c0, int c1, int c2)
{
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];" ::"l"(map),
                 "r"(src), "r"(c0), "r"(c1), "r"(c2)
                 : "memory");
}

__device__ __forceinline__ void tma_load_4d(
    uint32_t dst, CUtensorMap const* map, int c0, int c1, int c2, int c3, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, "
        "%5}], [%6];" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void tma_store_4d(CUtensorMap const* map, uint32_t src, int c0, int c1, int c2, int c3)
{
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];" ::"l"(map),
                 "r"(src), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
                 : "memory");
}

__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void prefetch_l2(void const* src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_commit()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait_read0()
{
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

__device__ __forceinline__ void bulk_wait0()
{
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fence_mbar_init()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void tc_fence_before()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tc_fence_after()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void named_bar(int id, int n)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ void named_arrive(int id, int n)
{
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr)
{
    return (uint64_t) ((addr & 0x3FFFF) >> 4) | ((uint64_t) 1 << 16) | ((uint64_t) (1024 >> 4) << 32)
        | ((uint64_t) 1 << 46) | ((uint64_t) 2 << 61);
}

__device__ __forceinline__ uint64_t desc_noswz(uint32_t addr, uint32_t lbo, uint32_t sbo)
{
    return (uint64_t) ((addr & 0x3FFFF) >> 4) | ((uint64_t) (lbo >> 4) << 16) | ((uint64_t) (sbo >> 4) << 32)
        | ((uint64_t) 1 << 46);
}

__host__ __device__ constexpr uint32_t idesc_bf16(int M, int N)
{
    return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t) (N >> 3) << 17) | ((uint32_t) (M >> 4) << 24);
}

__device__ __forceinline__ void mma_bf16(uint32_t dtmem, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(dtmem),
        "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}

__device__ __forceinline__ void mma_bf16_w(uint32_t dtmem, uint64_t ad, uint64_t bd, uint32_t idesc, uint32_t acc)
{
    asm volatile(
        "{\n\t.reg .pred e, p;\n\t"
        "elect.sync _|e, 0xffffffff;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(dtmem),
        "l"(ad), "l"(bd), "r"(idesc), "r"(acc));
}

// D[tmem] (+)= A B^T over K = 128 as 8 K16 steps, A and B SW128 K-major with K halves AH / BH bytes apart. One elect,
// descriptor offsets as immediates (uniform-datapath adds), first step overwrites unless ACC.
#define KSTEP_OFF(H, i) ((((i) >> 2) * (H) + ((i) &3) * 32) >> 4)

template <int AH, int BH, bool ACC>
__device__ __forceinline__ void mma_chain8(uint32_t dtmem, uint64_t ad, uint64_t bd, uint32_t idesc)
{
    asm volatile(
        "{\n\t.reg .pred e, p0, p1;\n\t.reg .b64 a, b;\n\t"
        "elect.sync _|e, 0xffffffff;\n\t"
        "setp.ne.b32 p0, %4, 0;\n\t"
        "setp.eq.b32 p1, %4, %4;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p0;\n\t"
        "add.s64 a, %1, %5;\n\tadd.s64 b, %2, %12;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %6;\n\tadd.s64 b, %2, %13;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %7;\n\tadd.s64 b, %2, %14;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %8;\n\tadd.s64 b, %2, %15;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %9;\n\tadd.s64 b, %2, %16;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %10;\n\tadd.s64 b, %2, %17;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, p1;\n\t"
        "add.s64 a, %1, %11;\n\tadd.s64 b, %2, %18;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [%0], a, b, %3, "
        "p1;\n\t}" ::"r"(dtmem),
        "l"(ad), "l"(bd), "r"(idesc), "r"(ACC ? 1 : 0), "n"(KSTEP_OFF(AH, 1)), "n"(KSTEP_OFF(AH, 2)),
        "n"(KSTEP_OFF(AH, 3)), "n"(KSTEP_OFF(AH, 4)), "n"(KSTEP_OFF(AH, 5)), "n"(KSTEP_OFF(AH, 6)),
        "n"(KSTEP_OFF(AH, 7)), "n"(KSTEP_OFF(BH, 1)), "n"(KSTEP_OFF(BH, 2)), "n"(KSTEP_OFF(BH, 3)),
        "n"(KSTEP_OFF(BH, 4)), "n"(KSTEP_OFF(BH, 5)), "n"(KSTEP_OFF(BH, 6)), "n"(KSTEP_OFF(BH, 7)));
}

// S0 copy: D[:, 16 i + n] = S0[:, 16 i + n] for the 8 K16 slices of S0 (B = one 16x16 identity tile, D column step 16)
__device__ __forceinline__ void mma_identity8(uint32_t dtmem, uint64_t ad, uint64_t bd, uint32_t idesc)
{
    asm volatile(
        "{\n\t.reg .pred e, p0;\n\t.reg .b64 a;\n\t.reg .b32 d;\n\t"
        "elect.sync _|e, 0xffffffff;\n\t"
        "setp.ne.b32 p0, %3, %3;\n\t"
        "@e tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p0;\n\t"
        "add.s64 a, %1, 2;\n\tadd.u32 d, %0, 16;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 4;\n\tadd.u32 d, %0, 32;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 6;\n\tadd.u32 d, %0, 48;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 1024;\n\tadd.u32 d, %0, 64;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 1026;\n\tadd.u32 d, %0, 80;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 1028;\n\tadd.u32 d, %0, 96;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, p0;\n\t"
        "add.s64 a, %1, 1030;\n\tadd.u32 d, %0, 112;\n\t@e tcgen05.mma.cta_group::1.kind::f16 [d], a, %2, %3, "
        "p0;\n\t}" ::"r"(dtmem),
        "l"(ad), "l"(bd), "r"(idesc));
}

__device__ __forceinline__ void mma_commit_w(uint32_t bar)
{
    asm volatile(
        "{\n\t.reg .pred e;\n\t"
        "elect.sync _|e, 0xffffffff;\n\t"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n\t}" ::"r"(bar)
        : "memory");
}

__device__ __forceinline__ void prefetch_tmap(void const* map)
{
    asm volatile("prefetch.tensormap [%0];" ::"l"(map) : "memory");
}

__device__ __forceinline__ void mma_commit(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void tmem_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_ld16(uint32_t taddr, uint32_t* r)
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr));
}

__device__ __forceinline__ void tmem_ld8(uint32_t taddr, uint32_t* r)
{
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(taddr));
}

__device__ __forceinline__ float ex2a(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rcpa(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ float expa(float x)
{
    return ex2a(x * LOG2E);
}

__device__ __forceinline__ float sigm(float x)
{
    return rcpa(1.0f + ex2a(-x * LOG2E));
}

__device__ __forceinline__ float silu(float x)
{
    return x * sigm(x);
}

__device__ __forceinline__ float bf2f(bf16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float bfround(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

// offset of (row, k) inside a [rows][64 bf16] SW128 K-major region (k < 64)
__device__ __forceinline__ uint32_t sw128(int row, int k)
{
    return (uint32_t) ((row >> 3) * 1024 + (row & 7) * 128 + ((((k >> 3) ^ row) & 7) << 4) + (k & 7) * 2);
}

// split x into hi/lo BF16 and store into a [split][khalf][rows] SW128 operand at (row, k)
__device__ __forceinline__ void st_split(
    uint8_t* smem, uint32_t base, uint32_t split_stride, uint32_t half_stride, int row, int k, float x)
{
    const bf16 hi = __float2bfloat16_rn(x);
    const bf16 lo = __float2bfloat16_rn(x - __bfloat162float(hi));
    const uint32_t off = base + (uint32_t) (k >> 6) * half_stride + sw128(row, k & 63);
    *(bf16*) (smem + off) = hi;
    *(bf16*) (smem + off + split_stride) = lo;
}

__device__ __forceinline__ uint64_t f2pk(float lo, float hi)
{
    uint64_t r;
    asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ uint64_t fmul2(uint64_t a, uint64_t b)
{
    uint64_t d;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
    return d;
}

__device__ __forceinline__ uint64_t ffma2(uint64_t a, uint64_t b, uint64_t c)
{
    uint64_t d;
    asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
    return d;
}

// (lo, hi) f32 pair -> packed bf16x2 (RN)
__device__ __forceinline__ uint32_t pk_bf16x2(uint64_t v)
{
    uint32_t r;
    asm("{\n\t.reg .f32 lo, hi;\n\tmov.b64 {lo, hi}, %1;\n\tcvt.rn.bf16x2.f32 %0, hi, lo;\n\t}" : "=r"(r) : "l"(v));
    return r;
}

// bf16x2 word -> (lo, hi) f32 pair
__device__ __forceinline__ uint64_t bf16x2_to_f2(uint32_t w)
{
    return f2pk(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b)
{
    __nv_bfloat162 t = __floats2bfloat162_rn(a, b);
    return *(uint32_t*) &t;
}

__device__ __forceinline__ uint32_t pack2(bf16 a, bf16 b)
{
    return (uint32_t) (*(unsigned short*) &a) | ((uint32_t) (*(unsigned short*) &b) << 16);
}

// In-warp reduce-scatter of P2 (>= 32, power of two) values: afterwards lane holds the warp sums of entries idx + i,
// i < P2 / 32 (halving exchange, P2 - P2/32 shuffles in total)
template <int P2>
__device__ __forceinline__ void warp_reduce_scatter(float (&w)[P2], int lane, int& idx)
{
    idx = 0;
#pragma unroll
    for (int st = 0; st < 5; ++st)
    {
        int const off = 16 >> st;
        int const half = P2 >> (st + 1);
        bool const upper = (lane & off) != 0;
        if (upper)
            idx += half;
#pragma unroll
        for (int i = 0; i < half; ++i)
        {
            float const send = upper ? w[i] : w[i + half];
            float const keep = upper ? w[i + half] : w[i];
            w[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
        }
    }
}

// Sum / max of N per-thread values over 128 threads (4 warps): reduce-scatter inside the warp (halving exchange),
// then one smem exchange [index][warp]. Every thread returns the totals.
template <int N, bool MAX>
__device__ __forceinline__ void reduce128(float (&v)[N], float* scratch, int wq, int lane, int bar_id)
{
    constexpr int P2 = (N <= 1) ? 1 : (N <= 2) ? 2 : (N <= 4) ? 4 : (N <= 8) ? 8 : 16;
    constexpr int L = (P2 == 1) ? 0 : (P2 == 2) ? 1 : (P2 == 4) ? 2 : (P2 == 8) ? 3 : 4;
    float w[P2];
#pragma unroll
    for (int i = 0; i < P2; ++i)
        w[i] = (i < N) ? v[i] : 0.f;
    int idx = 0;
#pragma unroll
    for (int st = 0; st < L; ++st)
    {
        int const off = 16 >> st;
        int const half = P2 >> (st + 1);
        bool const upper = (lane & off) != 0;
        if (upper)
            idx += half;
#pragma unroll
        for (int i = 0; i < half; ++i)
        {
            float const send = upper ? w[i] : w[i + half];
            float const keep = upper ? w[i + half] : w[i];
            float const got = __shfl_xor_sync(0xffffffffu, send, off);
            w[i] = MAX ? fmaxf(keep, got) : keep + got;
        }
    }
    float r = w[0];
#pragma unroll
    for (int off = 16 >> L; off > 0; off >>= 1)
    {
        float const got = __shfl_xor_sync(0xffffffffu, r, off);
        r = MAX ? fmaxf(r, got) : r + got;
    }
    if ((lane & ((32 >> L) - 1)) == 0)
        scratch[idx * 4 + wq] = r;
    named_bar(bar_id, 128);
#pragma unroll
    for (int i = 0; i < N; ++i)
    {
        const float4 t = *(float4 const*) (scratch + i * 4);
        v[i] = MAX ? fmaxf(fmaxf(t.x, t.y), fmaxf(t.z, t.w)) : (t.x + t.y) + (t.z + t.w);
    }
}

// issue the raw loads of one item (lanes of one warp): TMA tiles for token rows / conv history, bulk copies for the
// replay caches; lane 0 arms the barrier first
template <int NS>
__device__ __forceinline__ void issue_raw(
    KdaParams const& p, KdaMaps const& m, uint32_t rbase, uint32_t bar, int4 it, int lane)
{
    using C = Cfg<NS>;
    constexpr int NT = C::NT;
    int const slot = it.x, a = it.y, h = it.w;
    int const row0 = it.z * NT;
    if (lane == 0)
    {
        mbar_arrive_expect_tx(bar, C::RAW_FIXED + a * 1024);
        tma_load_3d(rbase + C::R_CQ, &m.csl[0], h * 128, a, slot, bar);
        tma_load_2d(rbase + C::R_XQ, &m.x[0], h * 128, row0, bar);
        tma_load_3d(rbase + C::R_CK, &m.csl[1], h * 128, a, slot, bar);
        tma_load_2d(rbase + C::R_XK, &m.x[1], h * 128, row0, bar);
        tma_load_3d(rbase + C::R_CV, &m.csl[2], h * 128, a, slot, bar);
        tma_load_2d(rbase + C::R_XV, &m.x[2], h * 128, row0, bar);
        tma_load_2d(rbase + C::R_OG, &m.x[3], h * 128, row0, bar);
        tma_load_2d(rbase + C::R_GR, &m.g, h * 128, row0, bar);
        tma_load_2d(rbase + C::R_BN, &m.bnew, h & ~7, row0, bar);
        tma_load_3d(rbase + C::R_BR, &m.brep, h & ~7, 0, slot, bar);
    }
    __syncwarp();
    if (lane < 3 * a)
    {
        int const which = lane / a, pos = lane - which * a;
        if (which == 0)
            bulk_g2s(rbase + C::R_KC + pos * 256, p.kc + slot * p.kc_s + pos * p.kc_p + h * 128, 256, bar);
        else if (which == 1)
            bulk_g2s(rbase + C::R_VC + pos * 256, p.vc + slot * p.vc_s + pos * p.vc_p + h * 128, 256, bar);
        else
            bulk_g2s(rbase + C::R_GC + pos * 512, p.gc + slot * p.gc_s + pos * p.gc_p + h * 128, 512, bar);
    }
}

// committed conv windows: positions 0..2+NS <- staging rows 1..3+NS (old history a+1, a+2, then the new tokens)
template <int NS>
__device__ __forceinline__ void store_conv_windows(KdaMaps const& m, uint32_t rbase, int4 it)
{
    using C = Cfg<NS>;
    int const slot = it.x, h = it.w;
    tma_store_3d(&m.css[0], rbase + C::R_CQ + 256, h * 128, 0, slot);
    tma_store_3d(&m.css[1], rbase + C::R_CK + 256, h * 128, 0, slot);
    tma_store_3d(&m.css[2], rbase + C::R_CV + 256, h * 128, 0, slot);
    bulk_commit();
}

// ------------------------------------------------------------------ kernel
#endif // SM10x

template <int NS>
__global__ void __launch_bounds__(256, 2) kda_mtp_kernel(const __grid_constant__ KdaMaps maps, const KdaParams p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100) // SM10x
    CUtensorMap const* tmap = &maps.state;
    using C = Cfg<NS>;
    constexpr int NT = C::NT;
    extern __shared__ uint8_t smem_dyn[];
    // 1024-B aligned base; pointer arithmetic on the __shared__ array keeps shared-space (LDS/STS) accesses
    uint8_t* smem = smem_dyn + ((1024u - (smem_u32(smem_dyn) & 1023u)) & 1023u);
    const uint32_t sb = smem_u32(smem);
    int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int const H = p.H;
    int const total = p.B * H;
    int const G = gridDim.x;
    int const cnt = (total - (int) blockIdx.x + G - 1) / G;
    const uint32_t bar_st0 = sb + C::OFF_MISC + C::M_BAR;
    const uint32_t bar_m1 = bar_st0 + 16, bar_m2 = bar_st0 + 24, bar_raw0 = bar_st0 + 32;
    float* misc = (float*) (smem + C::OFF_MISC);
    int4* sched = (int4*) (smem + C::OFF_MISC + C::M_SCHED);
    uint32_t* tmem_slot = (uint32_t*) (smem + C::OFF_MISC + C::M_TMEM);
    float* sbeta = misc + (C::M_BETA >> 2);
    float* ssog = misc + (C::M_SOG >> 2);
    float* sgn = misc + (C::M_GN >> 2);
    float* sca = misc + (C::M_CA >> 2);

    // TRT-LLM: wait for the programmatic-launch predecessor before the TMEM allocation and any global access (a no-op
    // without PDL).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 0)
    {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(tmem_slot)), "r"(256));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
        tc_fence_before();
    }
    else if (warp == 1)
    {
        // item 0 on the latency path (one lane): token rows right away, the state as soon as the slot is known, then
        // the conv window / replay caches once the accepted count is known
        if (lane == 0)
        {
            int const item0 = blockIdx.x;
            int const b0 = item0 / H, h0 = item0 - b0 * H;
            int const s0 = __ldg(p.slots + b0); // longest dependency chain first
            mbar_init(bar_raw0, 1);
            mbar_init(bar_raw0 + 8, 1);
            mbar_init(bar_st0, 1);
            mbar_init(bar_st0 + 8, 1);
            mbar_init(bar_m1, 1);
            mbar_init(bar_m2, 1);
            fence_mbar_init();
            prefetch_tmap(&maps.state);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                prefetch_tmap(&maps.x[i]);
            prefetch_tmap(&maps.g);
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                prefetch_tmap(&maps.csl[i]);
                prefetch_tmap(&maps.css[i]);
                prefetch_tmap(&maps.cf[i]);
            }
            prefetch_tmap(&maps.bnew);
            prefetch_tmap(&maps.brep);
            const uint32_t rb0 = sb + C::OFF_RAW;
            mbar_expect_tx(bar_raw0, C::RAW_TOK);
            tma_load_2d(rb0 + C::R_XQ, &maps.x[0], h0 * 128, b0 * NT, bar_raw0);
            tma_load_2d(rb0 + C::R_XK, &maps.x[1], h0 * 128, b0 * NT, bar_raw0);
            tma_load_2d(rb0 + C::R_XV, &maps.x[2], h0 * 128, b0 * NT, bar_raw0);
            tma_load_2d(rb0 + C::R_OG, &maps.x[3], h0 * 128, b0 * NT, bar_raw0);
            tma_load_2d(rb0 + C::R_GR, &maps.g, h0 * 128, b0 * NT, bar_raw0);
            tma_load_2d(rb0 + C::R_BN, &maps.bnew, h0 & ~7, b0 * NT, bar_raw0);
            int const nacc0 = __ldg(p.nacc + (long long) s0 * p.snacc);
            mbar_arrive_expect_tx(bar_st0, 32768);
            tma_load_4d(sb + C::OFF_ST, &maps.state, 0, h0 * 128, 0, s0, bar_st0);
            // replay caches: all NS positions (no wait on the accepted count)
            mbar_expect_tx(bar_raw0, NS * 1024);
            tma_load_3d(rb0 + C::R_KC, &maps.cf[0], h0 * 128, 0, s0, bar_raw0);
            tma_load_3d(rb0 + C::R_VC, &maps.cf[1], h0 * 128, 0, s0, bar_raw0);
            tma_load_3d(rb0 + C::R_GC, &maps.cf[2], h0 * 128, 0, s0, bar_raw0);
            int const a0 = min(max(nacc0, 0), NS);
            sched[0] = make_int4(s0, a0, b0, h0);
            mbar_arrive_expect_tx(bar_raw0, C::RAW_FIXED - C::RAW_TOK);
            tma_load_3d(rb0 + C::R_CQ, &maps.csl[0], h0 * 128, a0, s0, bar_raw0);
            tma_load_3d(rb0 + C::R_CK, &maps.csl[1], h0 * 128, a0, s0, bar_raw0);
            tma_load_3d(rb0 + C::R_CV, &maps.csl[2], h0 * 128, a0, s0, bar_raw0);
            tma_load_3d(rb0 + C::R_BR, &maps.brep, h0 & ~7, 0, s0, bar_raw0);
        }
    }
    else if (warp == 2)
    {
        // schedule of the remaining items: (slot, accepted count, request, head)
        for (int n = 1 + lane; n < cnt; n += 32)
        {
            int const item = blockIdx.x + n * G;
            int const b = item / H;
            int const s = __ldg(p.slots + b);
            int const a = min(max(__ldg(p.nacc + (long long) s * p.snacc), 0), NS);
            sched[n] = make_int4(s, a, b, item - b * H);
        }
    }
    if (C::ID && tid >= 128)
    {
        // 16x16 identity, element (n, k) at (n>>3)*256 + (n&7)*16 + (k>>3)*128 + (k&7)*2
        int const i = tid - 128;
        uint16_t* idt = (uint16_t*) (smem + C::OFF_MISC + C::M_ID);
        for (int e = i; e < 256; e += 128)
        {
            int const nn = e >> 4, kk = e & 15;
            idt[((nn >> 3) * 256 + (nn & 7) * 16 + (kk >> 3) * 128 + (kk & 7) * 2) >> 1] = (nn == kk) ? 0x3F80 : 0;
        }
        fence_proxy_async();
    }
    __syncthreads();
    tc_fence_after();
    const uint32_t tmem = *(uint32_t volatile*) tmem_slot;

    // prologue: raw of item 1 (double buffering), L2 prefetch of item 1's state
    if (warp == 2 && cnt > 1)
    {
        if (C::NRAW == 2)
            issue_raw<NS>(p, maps, sb + C::OFF_RAW + C::RAW_BYTES, bar_raw0 + 8, sched[1], lane);
        if (lane == 0)
        {
            const int4 i1 = sched[1];
            prefetch_l2(p.state + (long long) i1.x * p.sstate0 + (long long) i1.w * 16384, 32768);
        }
    }

    // per-thread per-item inputs (conv weights, gate bias / decay, beta), loaded one item ahead
    int const cch = tid & 127;
    bool const isqk = warp < 4;
    auto load_item = [&](int4 itx, float4& w0, float4& w1, float& dtb, float& eA, float& bet)
    {
        const long long chx = (long long) itx.w * 128 + cch;
        if (isqk)
        {
            w0 = __ldg((const float4*) p.wq + chx);
            w1 = __ldg((const float4*) p.wk + chx);
        }
        else
        {
            w0 = __ldg((const float4*) p.wv + chx);
            dtb = __ldg(p.dt_bias + chx);
            eA = __ldg(p.A_log + itx.w);
            bet = 0.f;
        }
    };
    float4 cw0 = make_float4(0.f, 0.f, 0.f, 0.f), cw1 = cw0;
    float cdtb = 0.f, ceA = 0.f, cbet = 0.f;
    load_item(sched[0], cw0, cw1, cdtb, ceA, cbet);
    float const ow_reg = __ldg(p.onorm_w + cch);

    for (int n = 0; n < cnt; ++n)
    {
        const int4 it = sched[n];
        int const slot = it.x, a = it.y, h = it.w;
        int const row0 = it.z * NT;
        int const stage = n & 1;
        const uint32_t st_off = C::OFF_ST + stage * 32768;
        int const rb = (C::NRAW == 2) ? (n & 1) : 0;
        const uint32_t rbase = sb + C::OFF_RAW + rb * C::RAW_BYTES;
        const uint32_t bar_raw = bar_raw0 + 8 * rb;
        const uint32_t raw_phase = (C::NRAW == 2) ? ((n >> 1) & 1) : (n & 1);
        bf16 const* sraw = (bf16 const*) (smem + C::OFF_RAW + rb * C::RAW_BYTES);

        float vR[NS], vN[NT]; // group V: values carried into the solve
        float kc0 = 0.f;      // group QK: commit-MMA B value of the first new token (Ku, or k without ID)
        float kn[NT];         // group QK: normalized new-token keys (k cache)
        float gn[NT];         // group V: new-token gates (g cache)

        if (warp < 4)
        {
            // ============ group QK: q/k conv + norms, cumulative decay, MMA operands
            int const c = tid;
            long long const ch = (long long) h * 128 + c;
            const float4 w4q = cw0;
            const float4 w4k = cw1;
            mbar_wait(bar_raw, raw_phase);
            if (tid == C::PROD * 32)
                store_conv_windows<NS>(maps, rbase, it); // committed windows straight from the raw rows
            float rq[3 + NT], rk[3 + NT];
#pragma unroll
            for (int i = 0; i < 3; ++i)
            {
                rq[i] = bf2f(sraw[(C::R_CQ >> 1) + i * 128 + c]);
                rk[i] = bf2f(sraw[(C::R_CK >> 1) + i * 128 + c]);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                rq[3 + j] = bf2f(sraw[(C::R_XQ >> 1) + j * 128 + c]);
                rk[3 + j] = bf2f(sraw[(C::R_XK >> 1) + j * 128 + c]);
            }
            float sq[NT], sk[NT], red[2 * NT];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float cq = w4q.x * rq[j];
                cq = fmaf(w4q.y, rq[j + 1], cq);
                cq = fmaf(w4q.z, rq[j + 2], cq);
                cq = fmaf(w4q.w, rq[j + 3], cq);
                float ck = w4k.x * rk[j];
                ck = fmaf(w4k.y, rk[j + 1], ck);
                ck = fmaf(w4k.z, rk[j + 2], ck);
                ck = fmaf(w4k.w, rk[j + 3], ck);
                sq[j] = silu(cq);
                sk[j] = silu(ck);
                red[j] = sq[j] * sq[j];
                red[NT + j] = sk[j] * sk[j];
            }
            reduce128<2 * NT, false>(red, misc + (C::M_RED >> 2), warp, lane, 4);
            float qn[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                qn[j] = sq[j] * (rsqrtf(fmaxf(red[j], 1e-24f)) * 0.08838834764831845f);
                kn[j] = sk[j] * rsqrtf(fmaxf(red[NT + j], 1e-24f));
            }
            if (C::CG)
            {
                float* ex = (float*) (smem + C::OFF_KU); // [k | q][NT][128] f32 for group V's token Gram
#pragma unroll
                for (int j = 0; j < NT; ++j)
                {
                    ex[j * 128 + c] = kn[j];
                    ex[(NT + j) * 128 + c] = qn[j];
                }
                named_arrive(7, 256);
            }
            named_bar(5, 256); // cumulative decay of the new tokens from group V
            // X rows Kd / Qd and Ku rows of the new tokens, [k-half][hi|lo][rows] SW128 (replay rows: group V)
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float const Cj = sgn[j * 128 + c];
                float const kd = kn[j] * Cj, ku = kn[j] * rcpa(Cj), qd = qn[j] * Cj;
                st_split(smem, C::OFF_X, C::XH, 2 * C::XH, NS + j, c, kd);
                st_split(smem, C::OFF_X, C::XH, 2 * C::XH, 2 * NS + 1 + j, c, qd);
                if (!C::CG)
                    st_split(smem, C::OFF_KU, C::KUH, 2 * C::KUH, NS + j, c, ku);
                if (j == 0)
                    kc0 = C::ID ? ku : kn[0]; // commit-B value of the first new token
            }
            fence_proxy_async();
        }
        else
        {
            // ============ group V: gates, v conv, beta / onorm gates
            int const c = tid - 128;
            long long const ch = (long long) h * 128 + c;
            const float4 w4v = cw0;
            float const dtb = cdtb;
            float const eA = expa(ceA);
            mbar_wait(bar_raw, raw_phase);
            if (c < NT)
                sbeta[c] = sigm(bf2f(sraw[(C::R_BN >> 1) + c * 8 + (h & 7)]));
            else if (c >= 8 && c < 8 + NS)
                sbeta[c] = bf2f(sraw[(C::R_BR >> 1) + (c - 8) * 8 + (h & 7)]);
            float const* sgc = (float const*) ((uint8_t const*) sraw + C::R_GC);
            float er[NS], en[NT];
#pragma unroll
            for (int s = 0; s < NS; ++s)
                er[s] = (s < a) ? expa(sgc[s * 128 + c]) : 1.0f;
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float const graw = bf2f(sraw[(C::R_GR >> 1) + j * 128 + c]);
                gn[j] = -5.0f * sigm((graw + dtb) * eA);
                en[j] = expa(gn[j]);
            }
            // cumulative decay over the chunk (same product order as a sequential scan)
            float Cr[NS];
            float Cc = 1.0f;
#pragma unroll
            for (int s = 0; s < NS; ++s)
            {
                Cc *= er[s];
                Cr[s] = Cc;
            }
            float Ca = 1.f;
            float Cn[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                Cc *= en[j];
                sgn[j * 128 + c] = Cc;
                Cn[j] = Cc;
                if (j == 0)
                    Ca = Cc;
            }
            sca[(n & 1) * 128 + c] = Ca;
            named_arrive(5, 256);
            // the replay X rows alias the previous item's commit operands: wait for its MMA2
            if (n > 0)
                mbar_wait(bar_m2, (n - 1) & 1);
            // replay X / Ku rows and their commit-B values
            float rKd[NS], rKu[NS];
#pragma unroll
            for (int s = 0; s < NS; ++s)
            {
                float kr = 0.f;
                rKd[s] = 0.f;
                rKu[s] = 0.f;
                if (s < a)
                {
                    float const kk = bf2f(sraw[(C::R_KC >> 1) + s * 128 + c]);
                    float const kd = kk * Cr[s], ku = kk * rcpa(Cr[s]);
                    st_split(smem, C::OFF_X, C::XH, 2 * C::XH, s, c, kd);
                    if (!C::CG)
                        st_split(smem, C::OFF_KU, C::KUH, 2 * C::KUH, s, c, ku);
                    kr = C::ID ? ku : ku * Ca;
                    rKd[s] = kd;
                    rKu[s] = ku;
                }
                misc[(C::M_KR >> 2) + s * 128 + c] = kr;
            }
            fence_proxy_async();
            float rv[3 + NT];
#pragma unroll
            for (int i = 0; i < 3; ++i)
                rv[i] = bf2f(sraw[(C::R_CV >> 1) + i * 128 + c]);
#pragma unroll
            for (int j = 0; j < NT; ++j)
                rv[3 + j] = bf2f(sraw[(C::R_XV >> 1) + j * 128 + c]);
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float cv = w4v.x * rv[j];
                cv = fmaf(w4v.y, rv[j + 1], cv);
                cv = fmaf(w4v.z, rv[j + 2], cv);
                cv = fmaf(w4v.w, rv[j + 3], cv);
                vN[j] = silu(cv);
                ssog[j * 128 + c] = sigm(bf2f(sraw[(C::R_OG >> 1) + j * 128 + c]));
            }
#pragma unroll
            for (int s = 0; s < NS; ++s)
                vR[s] = (s < a) ? bf2f(sraw[(C::R_VC >> 1) + s * 128 + c]) : 0.f;
            if (C::CG)
            {
                // ============ token Gram in FP32: G[row][tau] = sum_c X_row[c] Ku_tau[c] (rows: Kd of every token, Qd
                // of the new tokens; tau over earlier tokens), per-channel products reduced over the 128 channels of
                // group V
                named_bar(7, 256); // kn / qn of group QK published
                float const* ex = (float const*) (smem + C::OFF_KU);
                float tKd[2 * NS + 1], tKu[2 * NS + 1], tQd[NT];
#pragma unroll
                for (int s = 0; s < NS; ++s)
                {
                    tKd[s] = rKd[s];
                    tKu[s] = rKu[s];
                }
#pragma unroll
                for (int j = 0; j < NT; ++j)
                {
                    float const knj = ex[j * 128 + c], qnj = ex[(NT + j) * 128 + c];
                    tKd[NS + j] = knj * Cn[j];
                    tKu[NS + j] = knj * rcpa(Cn[j]);
                    tQd[j] = qnj * Cn[j];
                }
                float gv[C::GE_PAD];
                {
                    int e = 0;
#pragma unroll
                    for (int t = 1; t < 2 * NS + 1; ++t)
#pragma unroll
                        for (int tau = 0; tau < t; ++tau)
                            gv[e++] = tKd[t] * tKu[tau];
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int tau = 0; tau <= NS + j; ++tau)
                            gv[e++] = tQd[j] * tKu[tau];
#pragma unroll
                    for (; e < C::GE_PAD; ++e)
                        gv[e] = 0.f;
                }
                int gidx;
                warp_reduce_scatter<C::GE_PAD>(gv, lane, gidx);
                named_bar(1, 128); // every group-V thread has read kn / qn: reuse the region as scratch [entry][warp]
                float* gs = (float*) (smem + C::OFF_KU);
                int const wq = warp - 4;
#pragma unroll
                for (int i = 0; i < C::GE_PAD / 32; ++i)
                    gs[(gidx + i) * 4 + wq] = gv[i];
                named_bar(1, 128);
                if (c < C::GE)
                {
                    int row = 0, col = 0, e = 0;
#pragma unroll
                    for (int t = 1; t < 2 * NS + 1; ++t)
#pragma unroll
                        for (int tau = 0; tau < t; ++tau)
                        {
                            if (e == c)
                            {
                                row = t;
                                col = tau;
                            }
                            ++e;
                        }
#pragma unroll
                    for (int j = 0; j < NT; ++j)
#pragma unroll
                        for (int tau = 0; tau <= NS + j; ++tau)
                        {
                            if (e == c)
                            {
                                row = 2 * NS + 1 + j;
                                col = tau;
                            }
                            ++e;
                        }
                    const float4 t4 = *(float4 const*) (gs + c * 4);
                    misc[(C::M_GRAM >> 2) + row * 16 + col] = (t4.x + t4.y) + (t4.z + t4.w);
                }
            }
        }
        float4 nw0 = cw0, nw1 = cw1;
        float ndtb = cdtb, neA = ceA, nbet = cbet;
        if (n + 1 < cnt)
            load_item(sched[n + 1], nw0, nw1, ndtb, neA, nbet);
        __syncthreads(); // S1: X/Ku written, raw values consumed into registers / smem

        if (warp == 0)
        {
            // ============ Gram (state independent, M=64 over the stacked X_hi | X_lo rows): D[r][c] = X_r . [Ku_hi |
            // Ku_lo]_c
            __syncwarp();
            tc_fence_after();
            const uint64_t dX = desc_sw128(sb + C::OFF_X);
            const uint64_t dK = desc_sw128(sb + C::OFF_KU);
            constexpr uint32_t idg = idesc_bf16(64, 2 * C::KUR);
            if (!C::CG)
                mma_chain8<2 * C::XH, 2 * C::KUH, false>(tmem + 64, dX, dK, idg);
            // ============ MMA1: P = S0 [X_hi; X_lo]^T (M=128, N=2NP)
            mbar_wait_w(bar_st0 + 8 * stage, (n >> 1) & 1);
            tc_fence_after();
            const uint64_t dS = desc_sw128(sb + st_off);
            constexpr uint32_t id1 = idesc_bf16(128, 2 * C::NP);
            mma_chain8<16384, 2 * C::XH, false>(tmem, dS, dX, id1);
            mma_commit_w(bar_m1);
            if (C::ID)
            {
                // S0 copy into the commit accumulator: D'' = S0 I (M=128, N=16 per K step), exact; U Ku^T is added by
                // MMA2
                const uint64_t dI = desc_noswz(sb + C::OFF_MISC + C::M_ID, 128, 256);
                constexpr uint32_t idI = idesc_bf16(128, 16);
                mma_identity8(tmem + 128, dS, dI, idI);
            }
        }
        if (C::PROD == 1 && warp == 1)
        {
            // refill the raw buffer with item n + NRAW once the window store (issued at item start) has read it
            if (lane == 0)
                bulk_wait_read0();
            __syncwarp();
            if (n + C::NRAW < cnt)
                issue_raw<NS>(p, maps, rbase, bar_raw, sched[n + C::NRAW], lane);
        }
        else if (C::PROD == 1 && tid == 96)
        {
            // next state into the other stage (its previous TMA store has finished reading); L2 prefetch of the one
            // after
            if (n + 1 < cnt)
            {
                bulk_wait_read0();
                const int4 itn = sched[n + 1];
                const uint32_t bs = bar_st0 + 8 * (stage ^ 1);
                mbar_arrive_expect_tx(bs, 32768);
                tma_load_4d(sb + C::OFF_ST + (stage ^ 1) * 32768, tmap, 0, itn.w * 128, 0, itn.x, bs);
            }
            if (n + 2 < cnt)
            {
                const int4 i2 = sched[n + 2];
                prefetch_l2(p.state + (long long) i2.x * p.sstate0 + (long long) i2.w * 16384, 32768);
            }
        }
        else if (C::PROD == 3 && warp == 3)
        {
            // producer warp (reaches S2 early): its lane 0 issued the previous state store and this item's window
            // store; once both have read shared memory: next state into the other stage, L2 prefetch of the one after,
            // raw refill
            if (lane == 0)
            {
                bulk_wait_read0();
                if (n + 1 < cnt)
                {
                    const int4 itn = sched[n + 1];
                    const uint32_t bs = bar_st0 + 8 * (stage ^ 1);
                    mbar_arrive_expect_tx(bs, 32768);
                    tma_load_4d(sb + C::OFF_ST + (stage ^ 1) * 32768, tmap, 0, itn.w * 128, 0, itn.x, bs);
                }
                if (n + 2 < cnt)
                {
                    const int4 i2 = sched[n + 2];
                    prefetch_l2(p.state + (long long) i2.x * p.sstate0 + (long long) i2.w * 16384, 32768);
                }
            }
            __syncwarp();
            if (n + C::NRAW < cnt)
                issue_raw<NS>(p, maps, rbase, bar_raw, sched[n + C::NRAW], lane);
        }
        else if (tid == 128)
        {
            // beta_cache was read with plain loads before S1
#pragma unroll
            for (int pp = 0; pp < NS; ++pp)
                p.bc[slot * p.bc_s + pp * p.bc_p + h] = __float2bfloat16_rn(sbeta[pp + 1]);
        }

        float o[NT];
        if (warp < 4)
        {
            // k cache of the NS later tokens
            int const c = tid;
            long long const ch = (long long) h * 128 + c;
#pragma unroll
            for (int pp = 0; pp < NS; ++pp)
                p.kc[slot * p.kc_s + pp * p.kc_p + ch] = __float2bfloat16_rn(kn[pp + 1]);
            // B2 row c (commit MMA B operand, K-major no swizzle): [Kc_hi | Kc_lo | Kc_hi | 0], aliases X: after MMA1
            mbar_wait(bar_m1, n & 1);
            bf16 e[C::K2];
#pragma unroll
            for (int q = 0; q < C::K2; ++q)
                e[q] = __float2bfloat16_rn(0.f);
#pragma unroll
            for (int s = 0; s < NT; ++s)
            {
                float const kv = (s < NS) ? misc[(C::M_KR >> 2) + s * 128 + c] : kc0;
                const bf16 hi = __float2bfloat16_rn(kv);
                const bf16 lo = __float2bfloat16_rn(kv - __bfloat162float(hi));
                e[s] = hi;
                e[NT + s] = lo;
                e[2 * NT + s] = hi;
            }
            uint8_t* rowp = smem + C::OFF_B2 + (c >> 3) * C::SBO2 + (c & 7) * 16;
#pragma unroll
            for (int qc = 0; qc < C::K2 / 8; ++qc)
            {
                uint4 v;
                v.x = pack2(e[qc * 8 + 0], e[qc * 8 + 1]);
                v.y = pack2(e[qc * 8 + 2], e[qc * 8 + 3]);
                v.z = pack2(e[qc * 8 + 4], e[qc * 8 + 5]);
                v.w = pack2(e[qc * 8 + 6], e[qc * 8 + 7]);
                *(uint4*) (rowp + qc * 128) = v;
            }
            fence_proxy_async();
        }
        else
        {
            int const wq = warp - 4;
            int const v = tid - 128;
            {
                // v / g caches of the NS later tokens
                long long const ch = (long long) h * 128 + v;
#pragma unroll
                for (int pp = 0; pp < NS; ++pp)
                {
                    p.vc[slot * p.vc_s + pp * p.vc_p + ch] = __float2bfloat16_rn(vN[pp + 1]);
                    p.gc[slot * p.gc_s + pp * p.gc_p + ch] = gn[pp + 1];
                }
            }
            // ============ solve (thread = value row v)
            mbar_wait_w(bar_m1, n & 1);
            tc_fence_after();
            float Pf[C::NP];
            const uint32_t tl = tmem + ((uint32_t) (wq * 32) << 16);
#pragma unroll
            for (int i = 0; i < C::NP / 8; ++i)
            {
                uint32_t hi8[8], lo8[8];
                tmem_ld8(tl + i * 8, hi8);
                tmem_ld8(tl + C::NP + i * 8, lo8);
                tmem_wait_ld();
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    Pf[i * 8 + e] = __uint_as_float(hi8[e]) + __uint_as_float(lo8[e]);
            }
            float* Gm = misc + (C::M_GRAM >> 2);
            float* Gp = misc + (C::M_GP >> 2);
            if (!C::CG)
            {
                if (wq * 16 < 2 * C::NP)
                {
                    uint32_t G0[32];
                    tmem_ld16(tl + 64, G0);
                    if (2 * C::KUR > 16)
                        tmem_ld16(tl + 80, G0 + 16);
                    tmem_wait_ld();
                    int const r = wq * 16 + lane; // stacked row: hi rows [0, NP), lo rows [NP, 2NP)
                    if (lane < 16 && r < 2 * C::NP)
                    {
#pragma unroll
                        for (int q = 0; q < C::KUR; q += 4)
                        {
                            float4 g4;
                            g4.x = __uint_as_float(G0[q + 0]) + __uint_as_float(G0[C::KUR + q + 0]);
                            g4.y = __uint_as_float(G0[q + 1]) + __uint_as_float(G0[C::KUR + q + 1]);
                            g4.z = __uint_as_float(G0[q + 2]) + __uint_as_float(G0[C::KUR + q + 2]);
                            g4.w = __uint_as_float(G0[q + 3]) + __uint_as_float(G0[C::KUR + q + 3]);
                            *(float4*) (Gp + r * 16 + q) = g4;
                        }
                    }
                }
                named_bar(1, 128);
                for (int i = v; i < C::NROWS * 16; i += 128)
                    Gm[i] = Gp[i] + Gp[C::NP * 16 + i];
                named_bar(1, 128);
            }
            float uR[NS], uN[NT];
#pragma unroll
            for (int s = 0; s < NS; ++s)
            {
                uR[s] = 0.f;
                if (s < a)
                {
                    float w = Pf[s];
#pragma unroll
                    for (int s2 = 0; s2 < s; ++s2)
                        w = fmaf(Gm[s * 16 + s2], uR[s2], w);
                    uR[s] = sbeta[8 + s] * (vR[s] - w);
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                int const nk = NS + j, nq = 2 * NS + 1 + j;
                float w = Pf[nk];
                float oo = Pf[nq];
#pragma unroll
                for (int s2 = 0; s2 < NS; ++s2)
                {
                    if (s2 < a)
                    {
                        w = fmaf(Gm[nk * 16 + s2], uR[s2], w);
                        oo = fmaf(Gm[nq * 16 + s2], uR[s2], oo);
                    }
                }
#pragma unroll
                for (int j2 = 0; j2 < j; ++j2)
                {
                    w = fmaf(Gm[nk * 16 + NS + j2], uN[j2], w);
                    oo = fmaf(Gm[nq * 16 + NS + j2], uN[j2], oo);
                }
                uN[j] = sbeta[j] * (vN[j] - w);
                o[j] = fmaf(Gm[nq * 16 + NS + j], uN[j], oo);
            }
            // A2 row v: [U_hi | U_hi | U_lo | 0] (aliases X: MMA1 is complete)
            {
                bf16 e[C::K2];
#pragma unroll
                for (int q = 0; q < C::K2; ++q)
                    e[q] = __float2bfloat16_rn(0.f);
#pragma unroll
                for (int s = 0; s < NT; ++s)
                {
                    float const uv = (s < NS) ? uR[s] : uN[0];
                    const bf16 hi = __float2bfloat16_rn(uv);
                    const bf16 lo = __float2bfloat16_rn(uv - __bfloat162float(hi));
                    e[s] = hi;
                    e[NT + s] = hi;
                    e[2 * NT + s] = lo;
                }
                uint8_t* rowp = smem + C::OFF_A2 + (v >> 3) * C::SBO2 + (v & 7) * 16;
#pragma unroll
                for (int qc = 0; qc < C::K2 / 8; ++qc)
                {
                    uint4 vv;
                    vv.x = pack2(e[qc * 8 + 0], e[qc * 8 + 1]);
                    vv.y = pack2(e[qc * 8 + 2], e[qc * 8 + 3]);
                    vv.z = pack2(e[qc * 8 + 4], e[qc * 8 + 5]);
                    vv.w = pack2(e[qc * 8 + 6], e[qc * 8 + 7]);
                    *(uint4*) (rowp + qc * 128) = vv;
                }
            }
            fence_proxy_async();
            tc_fence_before();
        }
        __syncthreads(); // S2: commit operands ready, P / Gram consumed

        if (warp < 4)
        {
            // ============ MMA2: D' = U Kc^T  (M=128, N=128, K=K2)
            if (warp == 0)
            {
                __syncwarp();
                tc_fence_after();
                const uint64_t dA = desc_noswz(sb + C::OFF_A2, 128, C::SBO2);
                const uint64_t dB = desc_noswz(sb + C::OFF_B2, 128, C::SBO2);
                constexpr uint32_t id2 = idesc_bf16(128, 128);
#pragma unroll
                for (int ks = 0; ks < C::K2 / 16; ++ks)
                    mma_bf16_w(tmem + 128, dA + ks * 16, dB + ks * 16, id2, (C::ID || ks > 0) ? 1u : 0u);
                mma_commit_w(bar_m2);
            }
            // ============ commit epilogue (group QK, thread = state row): S = S0 * C_a + D' -> BF16 in place
            int const v = tid;
            mbar_wait_w(bar_m2, n & 1);
            tc_fence_after();
            const uint32_t tl = tmem + ((uint32_t) (warp * 32) << 16) + 128;
#pragma unroll
            for (int hb = 0; hb < 4; ++hb)
            {
                uint32_t d[32];
                tmem_ld16(tl + hb * 32, d);
                tmem_ld16(tl + hb * 32 + 16, d + 16);
                tmem_wait_ld();
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4)
                {
                    int const cc = (hb & 1) * 4 + c4; // 16-byte chunk inside the 128-column half
                    int const half = hb >> 1;
                    uint8_t* sp = smem + st_off + half * 16384 + v * 128 + (((cc ^ v) & 7) << 4);
                    const ulonglong2 c0 = *(ulonglong2 const*) (sca + (n & 1) * 128 + hb * 32 + c4 * 8);
                    const ulonglong2 c1 = *(ulonglong2 const*) (sca + (n & 1) * 128 + hb * 32 + c4 * 8 + 4);
                    uint32_t const* dd = d + c4 * 8;
                    uint4 r;
                    if (C::ID)
                    {
                        r.x = pk_bf16x2(fmul2(f2pk(__uint_as_float(dd[0]), __uint_as_float(dd[1])), c0.x));
                        r.y = pk_bf16x2(fmul2(f2pk(__uint_as_float(dd[2]), __uint_as_float(dd[3])), c0.y));
                        r.z = pk_bf16x2(fmul2(f2pk(__uint_as_float(dd[4]), __uint_as_float(dd[5])), c1.x));
                        r.w = pk_bf16x2(fmul2(f2pk(__uint_as_float(dd[6]), __uint_as_float(dd[7])), c1.y));
                    }
                    else
                    {
                        const uint4 s = *(uint4 const*) sp;
                        r.x = pk_bf16x2(
                            ffma2(bf16x2_to_f2(s.x), c0.x, f2pk(__uint_as_float(dd[0]), __uint_as_float(dd[1]))));
                        r.y = pk_bf16x2(
                            ffma2(bf16x2_to_f2(s.y), c0.y, f2pk(__uint_as_float(dd[2]), __uint_as_float(dd[3]))));
                        r.z = pk_bf16x2(
                            ffma2(bf16x2_to_f2(s.z), c1.x, f2pk(__uint_as_float(dd[4]), __uint_as_float(dd[5]))));
                        r.w = pk_bf16x2(
                            ffma2(bf16x2_to_f2(s.w), c1.y, f2pk(__uint_as_float(dd[6]), __uint_as_float(dd[7]))));
                    }
                    *(uint4*) sp = r;
                }
            }
            fence_proxy_async();
            tc_fence_before();
        }
        else
        {
            // ============ output (group V): gated RMSNorm + MXFP8 (1x128, UE8M0, R128c4)
            int const wq = warp - 4;
            int const v = tid - 128;
            float const ow = ow_reg;
            float yb[NT], ss[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                yb[j] = bfround(o[j]);
                ss[j] = yb[j] * yb[j];
            }
            reduce128<NT, false>(ss, misc + (C::M_RED >> 2) + 64, wq, lane, 2);
            float ob[NT], am[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float const rstd = rsqrtf(ss[j] * (1.0f / 128.0f) + 1e-5f);
                ob[j] = bfround(((yb[j] * rstd) * ow) * ssog[j * 128 + v]);
                am[j] = fabsf(ob[j]);
            }
            reduce128<NT, true>(am, misc + (C::M_RED >> 2) + 128, wq, lane, 3);
            int const D = H * 128;
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float const sf = fmaxf(am[j], 1e-10f) * __uint_as_float(0x3B124925u);
                const uint32_t bits = __float_as_uint(sf);
                int e8 = (int) ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
                e8 = min(e8, 254);
                float const inv = __uint_as_float((uint32_t) (254 - e8) << 23);
                const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(ob[j] * inv, __NV_SATFINITE, __NV_E4M3);
                int const r = row0 + j;
                p.out[(long long) r * D + h * 128 + v] = (uint8_t) q;
                if (v == 0)
                {
                    long long const off = ((((long long) (r >> 7) * H + h) * 32 + (r & 31)) * 4 + ((r & 127) >> 5)) * 4;
                    *(uint32_t*) (p.out_scale + off) = (uint32_t) e8 * 0x01010101u;
                }
            }
        }
        // S3 (group A only): committed rows in place; group B's RMSNorm / MXFP8 overlaps the store and the next item
        if (warp < 4)
            named_bar(6, 128);
        if (tid == 96)
        {
            tma_store_4d(tmap, sb + st_off, 0, h * 128, 0, slot);
            bulk_commit();
        }
        cw0 = nw0;
        cw1 = nw1;
        cdtb = ndtb;
        ceA = neA;
        cbet = nbet;
    }

    if (warp == 3 || warp == 1)
    {
        if (lane == 0)
            bulk_wait_read0();
        __syncwarp();
    }
    if (warp == 0)
    {
        tc_fence_after();
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(256));
    }
#endif // SM10x
}

#pragma nv_diag_default 177

// ================================== end of the Kernel Factory candidate ==================================

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kCtasPerSm = 2;
constexpr float kLowerBound = -5.0f;
constexpr float kQScale = 0.08838834764831845f;
constexpr float kOnormEps = 1e-5f;

// The candidate's encode: unit element strides, no interleave, L2 promotion 256 B, no OOB fill (BF16 unless stated).
void encodeMap(CUtensorMap* map, void const* ptr, int rank, cuuint64_t const* dims, cuuint64_t const* strides,
    cuuint32_t const* box, CUtensorMapSwizzle swizzle, CUtensorMapDataType dataType = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16)
{
    cuuint32_t const elementStrides[5] = {1, 1, 1, 1, 1};
    TLLM_CU_CHECK(tensorrt_llm::common::CUDADriverWrapper::getInstance()->cuTensorMapEncodeTiled(map, dataType,
        static_cast<cuuint32_t>(rank), const_cast<void*>(ptr), dims, strides, box, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// The candidate's tensor maps (kda_launch).
KdaMaps makeMaps(KdaParams const& p, int numSpec)
{
    cuuint64_t const dim = static_cast<cuuint64_t>(p.H) * kHeadDim;
    cuuint64_t const window = 3 + numSpec;
    cuuint32_t const newTokens = static_cast<cuuint32_t>(numSpec + 1);
    KdaMaps maps;
    {
        // [POOL][H * 128 rows][2 K-halves of 64] view of the state: rows 256 B apart, K-halves 128 B apart, slots
        // sstate0 elements apart
        cuuint64_t const dims[4] = {64, dim, 2, static_cast<cuuint64_t>(p.POOL)};
        cuuint64_t const strides[3] = {256, 128, static_cast<cuuint64_t>(p.sstate0) * 2};
        cuuint32_t const box[4] = {64, 128, 2, 1};
        encodeMap(&maps.state, p.state, 4, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B);
    }
    __nv_bfloat16 const* const rows[4] = {p.xq, p.xk, p.xv, p.og};
    long long const rowStrides[4] = {p.sxq, p.sxk, p.sxv, p.sog};
    for (int i = 0; i < 4; ++i)
    {
        cuuint64_t const dims[2] = {dim, static_cast<cuuint64_t>(p.T)};
        cuuint64_t const strides[1] = {static_cast<cuuint64_t>(rowStrides[i]) * 2};
        cuuint32_t const box[2] = {128, newTokens};
        encodeMap(&maps.x[i], rows[i], 2, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    {
        cuuint64_t const dims[2] = {dim, static_cast<cuuint64_t>(p.T)};
        cuuint64_t const strides[1] = {static_cast<cuuint64_t>(p.sg) * 2};
        cuuint32_t const box[2] = {128, newTokens};
        encodeMap(&maps.g, p.g, 2, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    __nv_bfloat16 const* const conv[3] = {p.csq, p.csk, p.csv};
    long long const posStrides[3] = {p.csq_p, p.csk_p, p.csv_p};
    long long const slotStrides[3] = {p.csq_s, p.csk_s, p.csv_s};
    for (int i = 0; i < 3; ++i)
    {
        cuuint64_t const dims[3] = {dim, window, static_cast<cuuint64_t>(p.POOL)};
        cuuint64_t const strides[2]
            = {static_cast<cuuint64_t>(posStrides[i]) * 2, static_cast<cuuint64_t>(slotStrides[i]) * 2};
        cuuint32_t const loadBox[3] = {128, 3, 1};
        cuuint32_t const storeBox[3] = {128, static_cast<cuuint32_t>(window), 1};
        encodeMap(&maps.csl[i], conv[i], 3, dims, strides, loadBox, CU_TENSOR_MAP_SWIZZLE_NONE);
        encodeMap(&maps.css[i], conv[i], 3, dims, strides, storeBox, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    {
        // Replay caches k / v (BF16) and g (FP32): [POOL][NS][H * 128] views, boxes of the NS rows of one slot.
        void const* const caches[3] = {p.kc, p.vc, p.gc};
        long long const rowStrides[3] = {p.kc_p, p.vc_p, p.gc_p};
        long long const slotStrides[3] = {p.kc_s, p.vc_s, p.gc_s};
        cuuint64_t const elementBytes[3] = {2, 2, 4};
        for (int i = 0; i < 3; ++i)
        {
            cuuint64_t const dims[3] = {dim, static_cast<cuuint64_t>(numSpec), static_cast<cuuint64_t>(p.POOL)};
            cuuint64_t const strides[2] = {static_cast<cuuint64_t>(rowStrides[i]) * elementBytes[i],
                static_cast<cuuint64_t>(slotStrides[i]) * elementBytes[i]};
            cuuint32_t const box[3] = {128, static_cast<cuuint32_t>(numSpec), 1};
            encodeMap(&maps.cf[i], caches[i], 3, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE,
                i == 2 ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16);
        }
    }
    {
        // beta of the new tokens: [T][H] with row stride sbeta, boxes of 8 heads x (NS + 1) rows.
        cuuint64_t const dims[2] = {static_cast<cuuint64_t>(p.H), static_cast<cuuint64_t>(p.T)};
        cuuint64_t const strides[1] = {static_cast<cuuint64_t>(p.sbeta) * 2};
        cuuint32_t const box[2] = {8, newTokens};
        encodeMap(&maps.bnew, p.beta, 2, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    {
        // beta cache: [POOL][NS][H] view, boxes of 8 heads x the NS rows of one slot.
        cuuint64_t const dims[3]
            = {static_cast<cuuint64_t>(p.H), static_cast<cuuint64_t>(numSpec), static_cast<cuuint64_t>(p.POOL)};
        cuuint64_t const strides[2] = {static_cast<cuuint64_t>(p.bc_p) * 2, static_cast<cuuint64_t>(p.bc_s) * 2};
        cuuint32_t const box[3] = {8, static_cast<cuuint32_t>(numSpec), 1};
        encodeMap(&maps.brep, p.bc, 3, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
    return maps;
}

// One wave of kCtasPerSm CTAs per SM while each CTA's items fit its schedule table (MAXN), else whole waves of
// equally loaded CTAs. (The candidate's launch_ns then took ceil(items / MAXN) CTAs: a nearly empty last wave that made
// B * H just above MAXN * kCtasPerSm * SMs up to 2x slower.) CTA c takes items c, c + grid, ...
template <int NS>
void launchNs(KdaParams const& p, KdaMaps const& maps, bool enablePdl, cudaStream_t stream)
{
    auto kernel = kda_mtp_kernel<NS>;
    int const smem = Cfg<NS>::SMEM_BYTES;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int const total = p.B * p.H;
    int const resident = kCtasPerSm * tensorrt_llm::common::getMultiProcessorCount();
    int const waves = (total + resident * MAXN - 1) / (resident * MAXN);
    int const grid = std::min(waves * resident, total);
    TLLM_CHECK_WITH_INFO((total + grid - 1) / grid <= MAXN, "kimi_k3_kda_mtp_verify_kf: CTA schedule overflow");

    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
    cfg.blockDim = dim3(kThreads, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, maps, p));
}

bool aligned(void const* ptr, uintptr_t bytes)
{
    return reinterpret_cast<uintptr_t>(ptr) % bytes == 0;
}

} // namespace

bool isKimiK3KdaMtpVerifyKfSupported(KimiK3KdaMtpVerifyKfParams const& p, char const** reason)
{
    auto fail = [reason](char const* why)
    {
        if (reason != nullptr)
        {
            *reason = why;
        }
        return false;
    };
    int const sm = tensorrt_llm::common::getSMVersion();
    if (sm < 100 || sm >= 110)
    {
        return fail("requires an SM10x GPU");
    }
    if (p.numSpec != 3 && p.numSpec != 5)
    {
        return fail("numSpec must be 3 or 5");
    }
    if (!p.stateIsBf16)
    {
        return fail("requires a BF16 recurrent state");
    }
    if (p.outScale == nullptr)
    {
        return fail("requires the MXFP8 output");
    }
    if (p.lowerBound != kLowerBound || p.qScale != kQScale || p.onormEps != kOnormEps)
    {
        return fail("requires lower bound -5, q scale 128^-0.5 and output-norm eps 1e-5");
    }
    if (p.numRequests < 1 || p.numHeads < 1 || p.statePoolSize < 1)
    {
        return fail("needs at least one request, one head and one state slot");
    }
    if (p.numAcceptedTokensStride < 0)
    {
        return fail("negative num_accepted_tokens stride");
    }
    int64_t const items = static_cast<int64_t>(p.numRequests) * p.numHeads;
    int64_t const rows = static_cast<int64_t>(p.numRequests) * (p.numSpec + 1);
    if (items > std::numeric_limits<int32_t>::max() / 2 || rows * p.numHeads > std::numeric_limits<int32_t>::max() / 2)
    {
        return fail("too many requests or heads");
    }
    void const* required[] = {p.xQ, p.xK, p.xV, p.g, p.beta, p.onormG, p.wQ, p.wK, p.wV, p.logA, p.dtBias,
        p.onormWeight, p.ssmStateIndices, p.numAcceptedTokens, p.convQ, p.convK, p.convV, p.state, p.kCache, p.gCache,
        p.vCache, p.betaCache, p.out};
    for (void const* ptr : required)
    {
        if (ptr == nullptr)
        {
            return fail("null tensor");
        }
    }
    // Tensor maps, bulk copies and float4 weight loads: bases 16-byte aligned, strides whole 16-byte units (beta and
    // the beta cache are read through tensor maps with 8-head boxes: rows padded to a multiple of 8 heads, as the [f_a
    // | b] projection and the cache manager do).
    void const* rows16[] = {p.xQ, p.xK, p.xV, p.g, p.onormG, p.convQ, p.convK, p.convV, p.kCache, p.gCache, p.vCache,
        p.state, p.wQ, p.wK, p.wV, p.beta, p.betaCache};
    for (void const* ptr : rows16)
    {
        if (!aligned(ptr, 16))
        {
            return fail("tensor base not 16-byte aligned");
        }
    }
    int64_t const bf16Strides[] = {p.xQRowStride, p.xKRowStride, p.xVRowStride, p.gRowStride, p.onormGRowStride,
        p.convQSlotStride, p.convQPosStride, p.convKSlotStride, p.convKPosStride, p.convVSlotStride, p.convVPosStride,
        p.kCacheSlotStride, p.kCacheRowStride, p.vCacheSlotStride, p.vCacheRowStride, p.stateSlotStride,
        p.betaRowStride, p.betaCacheSlotStride, p.betaCacheRowStride};
    for (int64_t s : bf16Strides)
    {
        if (s <= 0 || s % 8 != 0)
        {
            return fail("bf16 row / slot stride not a positive multiple of 8 elements");
        }
    }
    if (p.gCacheSlotStride <= 0 || p.gCacheSlotStride % 4 != 0 || p.gCacheRowStride <= 0 || p.gCacheRowStride % 4 != 0)
    {
        return fail("g_cache stride not a positive multiple of 4 elements");
    }
    if (!aligned(p.outScale, 4))
    {
        return fail("out_scale not 4-byte aligned");
    }
    return true;
}

void invokeKimiK3KdaMtpVerifyKf(KimiK3KdaMtpVerifyKfParams const& params, bool enablePdl, cudaStream_t stream)
{
    char const* reason = "";
    TLLM_CHECK_WITH_INFO(
        isKimiK3KdaMtpVerifyKfSupported(params, &reason), "invokeKimiK3KdaMtpVerifyKf: unsupported (%s)", reason);
    // The candidate's binding (main.cpp): every field from the op's tensors.
    KdaParams p = {};
    p.xq = static_cast<__nv_bfloat16 const*>(params.xQ);
    p.xk = static_cast<__nv_bfloat16 const*>(params.xK);
    p.xv = static_cast<__nv_bfloat16 const*>(params.xV);
    p.g = static_cast<__nv_bfloat16 const*>(params.g);
    p.beta = static_cast<__nv_bfloat16 const*>(params.beta);
    p.og = static_cast<__nv_bfloat16 const*>(params.onormG);
    p.sxq = params.xQRowStride;
    p.sxk = params.xKRowStride;
    p.sxv = params.xVRowStride;
    p.sg = params.gRowStride;
    p.sbeta = params.betaRowStride;
    p.sog = params.onormGRowStride;
    p.wq = params.wQ;
    p.wk = params.wK;
    p.wv = params.wV;
    p.A_log = params.logA;
    p.dt_bias = params.dtBias;
    p.onorm_w = params.onormWeight;
    p.slots = params.ssmStateIndices;
    p.nacc = params.numAcceptedTokens;
    p.snacc = params.numAcceptedTokensStride;
    p.csq = static_cast<__nv_bfloat16*>(params.convQ);
    p.csk = static_cast<__nv_bfloat16*>(params.convK);
    p.csv = static_cast<__nv_bfloat16*>(params.convV);
    p.csq_s = params.convQSlotStride;
    p.csq_p = params.convQPosStride;
    p.csk_s = params.convKSlotStride;
    p.csk_p = params.convKPosStride;
    p.csv_s = params.convVSlotStride;
    p.csv_p = params.convVPosStride;
    p.kc = static_cast<__nv_bfloat16*>(params.kCache);
    p.vc = static_cast<__nv_bfloat16*>(params.vCache);
    p.bc = static_cast<__nv_bfloat16*>(params.betaCache);
    p.gc = params.gCache;
    p.kc_s = params.kCacheSlotStride;
    p.kc_p = params.kCacheRowStride;
    p.vc_s = params.vCacheSlotStride;
    p.vc_p = params.vCacheRowStride;
    p.gc_s = params.gCacheSlotStride;
    p.gc_p = params.gCacheRowStride;
    p.bc_s = params.betaCacheSlotStride;
    p.bc_p = params.betaCacheRowStride;
    p.state = static_cast<__nv_bfloat16 const*>(params.state);
    p.sstate0 = params.stateSlotStride;
    p.out = static_cast<uint8_t*>(params.out);
    p.out_scale = params.outScale;
    p.B = params.numRequests;
    p.H = params.numHeads;
    p.T = params.numRequests * (params.numSpec + 1);
    p.POOL = params.statePoolSize;

    KdaMaps const maps = makeMaps(p, params.numSpec);
    if (params.numSpec == 3)
    {
        launchNs<3>(p, maps, enablePdl, stream);
    }
    else
    {
        launchNs<5>(p, maps, enablePdl, stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
