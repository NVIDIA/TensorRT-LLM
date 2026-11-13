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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/qkvFrontSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <mutex>
#include <utility>
#include <vector>

// The MiniMax-M3.1 sparse-layer front in one launch: a port of the reference front kernel (front_kernel,
// qkv_front_mxfp8_smem_a.cu: boundary on 15 warps while warp 15 stages the weight tile, a Lamport hand-off of the MXFP8
// rows to every GEMV CTA, the tcgen05 block-scaled GEMV with the weights read from shared memory, a DSMEM K-split
// reduce and the producer as its epilogue) with every arithmetic stage replaced by the served kernel it fuses, so that
// at split 8 its outputs are bit for bit those of the 3-kernel chain:
//   1. boundary = minimaxM3Sm107Tp4/mcAllreduceNormTp4.cu's 8-CTA cluster kernel at options 32 (the stock TP4 oneshot
//      reduction order) as the prepushed 2 / 3 consumer: the ranks' BF16 words summed in FP32 in rank order, rounded
//      to BF16, plus the residual (BF16), the RMS statistic in the oneshot tree (24 warp sums as 4 groups of 6),
//      rsqrtf(sum / H + eps), (v * inv) * w, and the MXFP8 copy of the normed row in FlashInfer's CuTe DSL quantize
//      (mxfp8_ue8m0 / mxfp8_inv_scale / mxfp8_e4m3x2 verbatim);
//   2. QKV = kernels/mx_gemv's tcgen05 GEMV at (split 8, ring 8): rank r of the cluster owns the contiguous k-tiles
//      [6 r, 6 r + 6), its 24 M128 x N8 x K32 block-scaled MMAs chain into one accumulator, and the owner of a row adds
//      the 8 FP32 partials in rank order starting from 0.0, then rounds to BF16;
//   3. producer = minimaxM3Sm107Tp4/producerPw4.cu (prewaitStepInputs) per (token, head) warp, verbatim.
// The exchange slots, flags and parities are the cluster kernel's: the MoE engine's push is unchanged.
// Hand-off sentinels: an unwritten activation byte is 0x80 (E4M3 -0, stored as +0: an exactly-zero product's sign
// never reaches the owner's sum, which starts from +0.0) and an unwritten scale byte 0xFF (the quantize clamps at 254).
// Split 4 (the reference geometry: 1536 K per CTA, four accumulators) is exact-class, for the micro-benchmark.

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)                                        \
    && (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)                                      \
        || defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)                                  \
        || defined(__CUDA_ARCH_FEAT_SM107_ALL))
#define M31_QKV_FRONT_ENABLED 1
#else
#define M31_QKV_FRONT_ENABLED 0
#endif

// Every wait of the front is bounded: past M31_QKV_FRONT_SPIN_LIMIT polls it traps (a lost Lamport word or barrier
// must fail the launch, never hang the stream). M31_QKV_FRONT_DIAG (diagnostic builds only) first records the trap's
// site and state into host-mapped memory (g_qkvFrontDiag, 8 words per CTA), which survives the context's fault.
#ifndef M31_QKV_FRONT_SPIN_LIMIT
#define M31_QKV_FRONT_SPIN_LIMIT (1LL << 31)
#endif
#ifdef M31_QKV_FRONT_DIAG
__device__ unsigned long long* g_qkvFrontDiag = nullptr;
#endif

// site: 1 exchange poll, 2 RMS gather, 3 weight TMA, 4 SFA copy, 5 hand-off poll, 6 MMA commit, 7 K-split receive
__device__ __noinline__ void spin_trap(int site, long long a0, unsigned long long a1, unsigned long long a2)
{
#ifdef M31_QKV_FRONT_DIAG
    if (g_qkvFrontDiag != nullptr)
    {
        unsigned long long* const d = g_qkvFrontDiag + (blockIdx.x & 255u) * 8;
        d[1] = threadIdx.x;
        d[2] = static_cast<unsigned long long>(a0);
        d[3] = a1;
        d[4] = a2;
        d[5] = clock64();
        __threadfence_system();
        d[0] = 0xD1A60000ull | static_cast<unsigned>(site);
        __threadfence_system();
    }
#else
    (void) site;
    (void) a0;
    (void) a1;
    (void) a2;
#endif
    __trap();
}

constexpr int kH = kQkvFrontHidden;
constexpr int kHd = 128;
constexpr int kTh = 512;
constexpr int kBndTh = kTh - 32;         // warps 0..14 run the boundary; warp 15 stages the weight tile meanwhile
constexpr int kMaxRows = 16;             // the hand-off buffer's rows (calls take at most kQkvFrontMaxTokens)
constexpr int kMaxT = kQkvFrontMaxTokens;
constexpr int kSlices = 8;               // the exchange's parity flags per row (the cluster kernel's 8 CTAs)
constexpr int kSliceCols = kH / kSlices; // 768
constexpr int kXTile = kMaxRows * 128;   // one k-tile of activation rows, 128-byte swizzled
constexpr int kXAtom = 256;              // one k-tile's activation scales: rows < 16 of a 128x4 atom
constexpr int kXqBuf = kMaxRows * kH;    // one parity of xq
constexpr int kXsfBuf = (kH / 128) * kXAtom + 512;
constexpr int kTmemCols = 128;           // acc 0..31 | SFA 32..79 | SFB 80..127
constexpr int kTSfa = 32;
constexpr int kTSfb = 80;
constexpr int kEmpty = static_cast<int>(0x80000000u);
constexpr int kOneshotWarps = 24; // the stock TP4 oneshot fusion: 4 CTAs x 6 warps per row
constexpr int kRotaryDim = 64;
constexpr int kPageSize = 128;
constexpr int kElems = 4;
static_assert(2LL * kXqBuf == kQkvFrontXqBytes && 2LL * kXsfBuf == kQkvFrontXsfBytes, "workspace layout");

template <int CL>
struct Geo
{
    static constexpr int kKs = kH / CL;                  // K columns per CTA = the boundary columns per CTA
    static constexpr int kTiles = kKs / 128;             // k-tiles per CTA
    static constexpr int kActive = kKs / 8;              // boundary threads per CTA (8 columns each)
    static constexpr int kWarps = kActive / 32;          // boundary warps per CTA
    static constexpr int kSlots = (kMaxT + CL - 1) / CL; // output rows owned per CTA
    static constexpr int kSmW = kTiles * kHd * 128;
    static constexpr int kSmWsfA = (kTiles * 512 + 1023) / 1024 * 1024;
    static constexpr int kRecv = CL * kSlots * kHd * 4; // the K-split receive buffer (reuses sWsf)
    static constexpr int kSmWsf = kSmWsfA > kRecv ? kSmWsfA : (kRecv + 1023) / 1024 * 1024;
    static constexpr int kSmX = kTiles * kXTile;
    static constexpr int kSmXsf = kTiles * kXAtom + 256; // tcgen05.cp reads 512 B from each atom start
    static constexpr int kSmem = kSmW + kSmWsf + kSmX + kSmXsf + 1024;
    static constexpr int kPollQ = (kMaxRows * kTiles * 8 + kBndTh - 1) / kBndTh;
    static_assert(kKs % 128 == 0 && kActive % 32 == 0, "whole k-tiles and boundary warps");
    static_assert(kWarps * CL == kOneshotWarps, "the oneshot tree's 24 warp sums");
    static_assert(kKs % kSliceCols == 0, "a CTA covers whole exchange slices");
    static_assert(kTiles * 4 <= kTSfb - kTSfa, "SFA columns");
    static_assert(kPollQ < 31, "poll mask");
};

// ---------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ int4 ld_volatile_v4(int const* p)
{
    int4 v;
    asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ float bf16_lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

__device__ __forceinline__ float bf16_hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ float bf16_round(float a)
{
    return __bfloat162float(__float2bfloat16_rn(a));
}

__device__ __forceinline__ void bnd_sync()
{
    asm volatile("bar.sync 1, %0;" ::"n"(kBndTh) : "memory");
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, int site)
{
    uint32_t done = 0;
    long long spins = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
        if (!done && ++spins > M31_QKV_FRONT_SPIN_LIMIT)
            spin_trap(site, spins, bar, parity);
    }
}

__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity, int site)
{
    uint32_t done = 0;
    long long spins = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
        if (!done && ++spins > M31_QKV_FRONT_SPIN_LIMIT)
            spin_trap(site, spins, bar, parity);
    }
}

__device__ __forceinline__ void st_async_f32(uint32_t dst, float v, uint32_t bar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst),
                 "r"(__float_as_uint(v)), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ uint32_t mapa(uint32_t addr, int cta)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}

// UMMA shared-memory descriptor (SM100 tcgen05): start, leading / stride byte offsets, version 1, layout type.
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t layout)
{
    return (uint64_t) ((saddr >> 4) & 0x3FFFu) | ((uint64_t) ((lbo >> 4) & 0x3FFFu) << 16)
        | ((uint64_t) ((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46) | ((uint64_t) layout << 61);
}

// ---------------------------------------------------------------- the cluster all-reduce's MXFP8 quantize, verbatim
// (mcAllreduceNormTp4.cu: FlashInfer's CuTe DSL mxfp8 quantize PTX)
__device__ __forceinline__ uint32_t mxfp8_ue8m0(float amax)
{
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(amax), "f"(1.0f / 448.0f));
    uint32_t result;
    asm("{\n"
        " .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
        " .reg .u32 bits, exp_biased, mantissa, bump, r;\n"
        " setp.le.f32 p_zero, %1, 0f00000000;\n"
        " mov.b32 bits, %1;\n"
        " shr.b32 exp_biased, bits, 23;\n"
        " and.b32 exp_biased, exp_biased, 255;\n"
        " and.b32 mantissa, bits, 0x7FFFFF;\n"
        " setp.ne.u32 p_has_mant, mantissa, 0;\n"
        " selp.u32 bump, 1, 0, p_has_mant;\n"
        " setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
        " setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
        " and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
        " @p_tiny_sub mov.u32 bump, 0;\n"
        " add.u32 r, exp_biased, bump;\n"
        " setp.gt.u32 p_ovf, r, 254;\n"
        " selp.u32 r, 254, r, p_ovf;\n"
        " selp.u32 %0, 0, r, p_zero;\n"
        "}\n"
        : "=r"(result)
        : "f"(value));
    return result;
}

__device__ __forceinline__ float mxfp8_inv_scale(uint32_t ue8m0)
{
    float inv;
    asm("{\n"
        " .reg .s32 new_exp;\n"
        " .reg .b32 float_bits;\n"
        " .reg .pred p_zero;\n"
        " setp.eq.u32 p_zero, %1, 0;\n"
        " sub.s32 new_exp, 254, %1;\n"
        " max.s32 new_exp, new_exp, 0;\n"
        " shl.b32 float_bits, new_exp, 23;\n"
        " mov.b32 %0, float_bits;\n"
        " @p_zero mov.b32 %0, 0;\n"
        "}\n"
        : "=f"(inv)
        : "r"(ue8m0));
    return inv;
}

__device__ __forceinline__ uint32_t mxfp8_e4m3x2(float lo, float hi, float inv)
{
    uint16_t pair;
    asm("{\n"
        " .reg .f32 f0, f1;\n"
        " mul.f32 f0, %1, %3;\n"
        " mul.f32 f1, %2, %3;\n"
        " min.f32 f0, f0, 0f43E00000;\n"
        " max.f32 f0, f0, 0fC3E00000;\n"
        " min.f32 f1, f1, 0f43E00000;\n"
        " max.f32 f1, f1, 0fC3E00000;\n"
        " cvt.rn.satfinite.e4m3x2.f32 %0, f1, f0;\n"
        "}\n"
        : "=h"(pair)
        : "f"(lo), "f"(hi), "f"(inv));
    return pair;
}

// ---------------------------------------------------------------- producerPw4.cu's helpers, verbatim
__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint32_t sf_group_mask4()
{
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return 0xFu << (laneId & ~3u);
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1(float (&a)[8])
{
#if M31_QKV_FRONT_ENABLED
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]));
    return val;
#else
    return 0;
#endif
}

__device__ __forceinline__ void store_fp8x4(__nv_fp8_e4m3* out, int64_t off, float const (&e)[kElems])
{
    __nv_fp8x2_e4m3 const low(make_float2(e[0], e[1]));
    __nv_fp8x2_e4m3 const high(make_float2(e[2], e[3]));
    *reinterpret_cast<uint32_t*>(out + off) = static_cast<uint32_t>(low.__x) | (static_cast<uint32_t>(high.__x) << 16);
}

__device__ __forceinline__ uint16_t quantize_fp4x4(float const (&e)[4], float secondLevelScale, uint8_t* scaleOut)
{
    __nv_bfloat16 rounded[4];
    float localMax = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        rounded[i] = __float2bfloat16_rn(e[i]);
        localMax = fmaxf(localMax, fabsf(__bfloat162float(rounded[i])));
    }
    uint32_t const groupMask = sf_group_mask4();
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 1));
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 2));
    float const scaleValue = secondLevelScale * (localMax * rcp_approx_ftz(6.0F));
    __nv_fp8_e4m3 const encodedScale(scaleValue);
    if (scaleOut != nullptr)
        *scaleOut = encodedScale.__x;
    float const narrowedScale = static_cast<float>(encodedScale);
    float const outputScale
        = localMax != 0.0F ? rcp_approx_ftz(narrowedScale * rcp_approx_ftz(secondLevelScale)) : 0.0F;
    float q[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i)
        q[i] = __bfloat162float(rounded[i]) * outputScale;
    return static_cast<uint16_t>(fp32x8_to_e2m1(q));
}

struct alignas(64) FrontArgs
{
    CUtensorMap tmw; // weight [N, 6144] u8, box 128 K x 128 rows, SWIZZLE_128B
    QkvFrontParams p;
    int earlyTrigger;
};

// ---------------------------------------------------------------- the boundary row (warps 0..14 of a cluster < rows)
// The cluster kernel (options 32) for row t over this CTA's kKs columns, then the MXFP8 copy into the hand-off.
template <int CL>
__device__ __forceinline__ void boundary_row(QkvFrontParams const& p, int t, int crank, int call, float* warp_sq,
    float* cluster_wsq, float* s_total, unsigned long long* xbar)
{
    using G = Geo<CL>;
    int const tid = threadIdx.x;
    if (tid == 0)
    {
        mbar_init(smem_u32(xbar), 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); // #1
    int const c0 = crank * G::kKs + tid * 8;
    bool const active = tid < G::kActive;
    int const world = p.world;
    int64_t const half_words = static_cast<int64_t>(p.maxt) * world * (kH / 2);
    int const fl = t * kSlices + c0 / kSliceCols;
    // prepushed >= 2: the MoE pushed its rows and released this grid after its pushes; no grid wait (the cluster
    // kernel's contract: the slots are Lamport-polled, the flags and the residual were written grids earlier).
    int const parity = active ? (*reinterpret_cast<int volatile*>(p.flags + fl) & 1) : 0;
    uint8_t* const xqo = p.xq + (call & 1) * kXqBuf;
    uint8_t* const xsfo = p.xsf + (call & 1) * kXsfBuf;
    uint4 rraw = make_uint4(0u, 0u, 0u, 0u);
    uint4 wraw = make_uint4(0u, 0u, 0u, 0u);
    if (active)
    {
        rraw = *reinterpret_cast<uint4 const*>(p.residual + static_cast<int64_t>(t) * kH + c0);
        wraw = *reinterpret_cast<uint4 const*>(p.normWeight + c0);
    }
    float s[8] = {};
    float sq = 0.f;
    if (active)
    {
        constexpr int kMaxWorld = 8;
        int* const p0 = p.wsUc + parity * half_words + static_cast<int64_t>(t) * world * (kH / 2) + c0 / 2;
        int4 v[kMaxWorld];
        int4 v2[kMaxWorld];
        bool const two = p.prepushed == 3;
        int* const p1 = p0 + 2 * half_words; // set B
        uint32_t pending = (1u << world) - 1u;
        uint32_t pending2 = two ? pending : 0u;
        long long spins = 0;
        while ((pending | pending2) != 0u)
        {
#pragma unroll
            for (int r = 0; r < kMaxWorld; ++r)
            {
                if ((pending >> r) & 1u)
                    v[r] = ld_volatile_v4(p0 + static_cast<int64_t>(r) * (kH / 2));
                if ((pending2 >> r) & 1u)
                    v2[r] = ld_volatile_v4(p1 + static_cast<int64_t>(r) * (kH / 2));
            }
#pragma unroll
            for (int r = 0; r < kMaxWorld; ++r)
            {
                if (((pending >> r) & 1u) && v[r].x != kEmpty && v[r].y != kEmpty && v[r].z != kEmpty
                    && v[r].w != kEmpty)
                    pending &= ~(1u << r);
                if (((pending2 >> r) & 1u) && v2[r].x != kEmpty && v2[r].y != kEmpty && v2[r].z != kEmpty
                    && v2[r].w != kEmpty)
                    pending2 &= ~(1u << r);
            }
            if (++spins > M31_QKV_FRONT_SPIN_LIMIT)
                spin_trap(1, t, (static_cast<unsigned long long>(pending2) << 8) | pending,
                    (static_cast<unsigned long long>(parity) << 32) | static_cast<unsigned>(fl));
        }
#pragma unroll
        for (int r = 0; r < kMaxWorld; ++r)
        {
            if (r < world)
            {
                *reinterpret_cast<int4*>(p0 + static_cast<int64_t>(r) * (kH / 2))
                    = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
                uint32_t ww[4] = {static_cast<uint32_t>(v[r].x), static_cast<uint32_t>(v[r].y),
                    static_cast<uint32_t>(v[r].z), static_cast<uint32_t>(v[r].w)};
                if (two)
                {
                    *reinterpret_cast<int4*>(p1 + static_cast<int64_t>(r) * (kH / 2))
                        = make_int4(kEmpty, kEmpty, kEmpty, kEmpty);
                    uint32_t const w2[4] = {static_cast<uint32_t>(v2[r].x), static_cast<uint32_t>(v2[r].y),
                        static_cast<uint32_t>(v2[r].z), static_cast<uint32_t>(v2[r].w)};
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                    {
                        __nv_bfloat162 const u
                            = __floats2bfloat162_rn(bf16_lo(ww[q]) + bf16_lo(w2[q]), bf16_hi(ww[q]) + bf16_hi(w2[q]));
                        ww[q] = *reinterpret_cast<uint32_t const*>(&u);
                    }
                }
#pragma unroll
                for (int q = 0; q < 4; ++q)
                {
                    s[2 * q] += bf16_lo(ww[q]);
                    s[2 * q + 1] += bf16_hi(ww[q]);
                }
            }
        }
        // Stock rounding points: the rank sum is a BF16 all-reduce output before the residual add.
#pragma unroll
        for (int i = 0; i < 8; ++i)
            s[i] = __bfloat162float(__float2bfloat16_rn(s[i]));
        __nv_bfloat162 const* rr = reinterpret_cast<__nv_bfloat162 const*>(&rraw);
        uint32_t upd[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const rf = __bfloat1622float2(rr[q]);
            __nv_bfloat162 const u = __floats2bfloat162_rn(s[2 * q] + rf.x, s[2 * q + 1] + rf.y);
            upd[q] = *reinterpret_cast<uint32_t const*>(&u);
            float2 const uf = __bfloat1622float2(u);
            s[2 * q] = uf.x;
            s[2 * q + 1] = uf.y;
            // stock oneshot: acc += v * v, one column at a time
            sq += uf.x * uf.x;
            sq += uf.y * uf.y;
        }
        *reinterpret_cast<uint4*>(p.outResidual + static_cast<int64_t>(t) * kH + c0)
            = make_uint4(upd[0], upd[1], upd[2], upd[3]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    if ((tid & 31) == 0 && (tid >> 5) < G::kWarps)
        warp_sq[tid >> 5] = sq;
    bnd_sync();
    // every boundary thread has read its slice's parity: rotate each slice's flag once (read only by later grids)
    if (active && (c0 % kSliceCols) == 0)
        p.flags[fl] = parity ^ 1;
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); // #1: every peer's xbar is initialized
    if (tid == 0)
    {
        // every CTA of the row gets this CTA's kWarps warp sums at [kWarps * crank + w] (the oneshot tree's order)
        uint32_t const b = smem_u32(xbar);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(kOneshotWarps * 4)
                     : "memory");
        for (int c = 0; c < CL; ++c)
        {
            uint32_t const rb = mapa(b, c);
            for (int w = 0; w < G::kWarps; ++w)
                st_async_f32(mapa(smem_u32(&cluster_wsq[G::kWarps * crank + w]), c), warp_sq[w], rb);
        }
    }
    mbar_wait_cluster(smem_u32(xbar), 0, 2);
    if (tid < 32)
    {
        // The stock TP4 oneshot fusion at H 6144 (4 CTAs x 192 threads): per group of 6 warp sums a 32-lane butterfly
        // (lanes >= 6 zero), lane 0's value; the 4 group sums added in order from 0.f.
        int const lane = tid;
        float acc = 0.f;
        for (int g = 0; g < 4; ++g)
        {
            float val = lane < 6 ? cluster_wsq[6 * g + lane] : 0.f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_xor_sync(0xffffffffu, val, off);
            acc += __shfl_sync(0xffffffffu, val, 0);
        }
        if (lane == 0)
            *s_total = acc;
    }
    bnd_sync();
    if (active)
    {
        float const total = *s_total;
        float const inv = rsqrtf(total / static_cast<float>(kH) + p.eps);
        __nv_bfloat162 const* wv = reinterpret_cast<__nv_bfloat162 const*>(&wraw);
        uint32_t o[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 const wf = __bfloat1622float2(wv[q]);
            __nv_bfloat162 const n = __floats2bfloat162_rn(s[2 * q] * inv * wf.x, s[2 * q + 1] * inv * wf.y);
            o[q] = *reinterpret_cast<uint32_t const*>(&n);
            float2 const nf = __bfloat1622float2(n);
            s[2 * q] = nf.x;
            s[2 * q + 1] = nf.y;
        }
        if (p.outNormed != nullptr)
            *reinterpret_cast<uint4*>(p.outNormed + static_cast<int64_t>(t) * kH + c0)
                = make_uint4(o[0], o[1], o[2], o[3]);
    }
    // MXFP8 copy of the normed row (four threads hold one 32-column scale block) into the hand-off: the B-operand
    // layout [48 k-tiles][16 rows][128 B], 16-byte chunks 128B-swizzled (chunk ^ (row & 7)); scales as 128x4 atoms
    // (rows < 16) of 256 B per k-tile.
    float amax = 0.f;
    if (active)
    {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            amax = fmaxf(amax, fabsf(s[e]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (active)
    {
        uint32_t const code = mxfp8_ue8m0(amax);
        float const inv_scale = mxfp8_inv_scale(code);
        uint32_t packed[2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            packed[h] = mxfp8_e4m3x2(s[4 * h], s[4 * h + 1], inv_scale)
                | (mxfp8_e4m3x2(s[4 * h + 2], s[4 * h + 3], inv_scale) << 16);
            packed[h] &= ~(__vcmpeq4(packed[h], 0x80808080u) & 0x80808080u); // -0 is the sentinel: store +0
        }
        int const tile = c0 / 128, ch = (c0 % 128) / 16;
        *reinterpret_cast<uint2*>(xqo + (tile * kMaxRows + t) * 128 + ((ch ^ (t & 7)) * 16) + (c0 % 16))
            = make_uint2(packed[0], packed[1]);
        if ((tid & 3) == 0)
            xsfo[(c0 / 128) * kXAtom + t * 16 + (c0 / 32) % 4] = static_cast<uint8_t>(code);
    }
    bnd_sync();
}

// ---------------------------------------------------------------- the kernel
// Grid (headsQ + 4) * CL CTAs, cluster CL: cluster h computes QKV head h (128 output rows) with K split CL; the CTA
// of cluster rank r reads K columns [r * 6144 / CL, (r + 1) * 6144 / CL). Clusters h < rows also run boundary row h.
template <int CL, int NACC, int ROWS>
__global__ void __launch_bounds__(kTh, 1) qkv_front_kernel(__grid_constant__ FrontArgs const a)
{
#if M31_QKV_FRONT_ENABLED
    using G = Geo<CL>;
    QkvFrontParams const& p = a.p;
    extern __shared__ __align__(16) uint8_t sm_raw[];
    __shared__ float warp_sq[16];
    __shared__ alignas(16) float cluster_wsq[kOneshotWarps];
    __shared__ float s_total;
    __shared__ alignas(8) unsigned long long xbar, wbar, mbar, abar, rbar;
    __shared__ uint32_t tmem_base_s;
    __shared__ int s_call;
    uint8_t* const sm = sm_raw + ((1024 - (smem_u32(sm_raw) & 1023)) & 1023);
    uint8_t* const sW = sm;
    uint8_t* const sWsf = sW + G::kSmW;
    uint8_t* const sX = sWsf + G::kSmWsf;
    uint8_t* const sXsf = sX + G::kSmX;

    int const crank = []
    {
        uint32_t r;
        asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
        return static_cast<int>(r);
    }();
    int const h = blockIdx.x / CL;
    int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int const R = ROWS > 0 ? ROWS : p.numTokens;
    bool const bnd = h < R;
    int const nq = p.headsQ;
    bool const isQ = h < nq;
    bool const isK = h == nq;
    bool const isV = h == nq + 1;
    bool const isIndexQ = h == nq + 2;
    bool const isIndexK = h == nq + 3;

    // Producer inputs no kernel of the step writes (producerPw4 with prewaitStepInputs): norm weight, KV quant scale,
    // position, slot and RoPE row, for the row this warp owns (warps < kSlots: row crank + CL * warp).
    int const ownRow = crank + CL * warp;
    bool const owner = warp < G::kSlots && ownRow < R;
    float w[kElems] = {};
    float quantScale = 0.0F;
    int positionId = 0;
    int slot = 0;
    float4 c = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    float4 s = c;
    bool const rotates = !isV && lane * kElems < kRotaryDim;
    bool const slotted = isK || isV || isIndexK;
    if (owner)
    {
        __nv_bfloat16 const* normWeight
            = isQ ? p.qWeight : (isK ? p.kWeight : (isIndexQ ? p.indexQWeight : p.indexKWeight));
        if (!isV)
        {
            uint2 const wv = *reinterpret_cast<uint2 const*>(normWeight + lane * kElems);
            float2 const w01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&wv.x));
            float2 const w23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&wv.y));
            w[0] = w01.x;
            w[1] = w01.y;
            w[2] = w23.x;
            w[3] = w23.y;
        }
        if (isK || isV)
            quantScale = p.kvQuantScale[isV ? 2 : 1];
        int const coefficient = (lane * kElems) % (kRotaryDim / 2);
        positionId = rotates ? p.positionIds[ownRow] : 0;
        slot = slotted ? p.outCacheLoc[ownRow] : 0;
        if (rotates)
        {
            int64_t const ropeRow = static_cast<int64_t>(positionId) * kRotaryDim;
            c = *reinterpret_cast<float4 const*>(p.rotaryCosSin + ropeRow + coefficient);
            s = *reinterpret_cast<float4 const*>(p.rotaryCosSin + ropeRow + kRotaryDim / 2 + coefficient);
        }
    }

    if (tid == 0)
    {
        mbar_init(smem_u32(&rbar), 1); // the K-split receive barrier: peers' st.async partials land on it
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 15)
    {
        // Warp 15: TMEM, barriers, the weight tile's TMA and its scale atoms into TMEM while warps 0..14 run the
        // boundary (the weights are read by the MMAs straight from shared memory).
        if (bnd)
            asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); // #1
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_s)),
                     "r"(kTmemCols)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncwarp();
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        if (lane == 0)
        {
            mbar_init(smem_u32(&wbar), 1);
            mbar_init(smem_u32(&mbar), 1);
            mbar_init(smem_u32(&abar), 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            uint32_t const tmem = tmem_base_s;
            uint8_t const* const wsrc
                = p.weightScale + static_cast<int64_t>(h) * (kH / 128) * 512 + crank * G::kTiles * 512;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&wbar)),
                         "r"(G::kSmW + G::kTiles * 512)
                         : "memory");
            for (int t = 0; t < G::kTiles; ++t)
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
                    "%3}], [%4];" ::"r"(smem_u32(sW + t * (kHd * 128))),
                    "l"(&a.tmw), "r"(crank * G::kKs + t * 128), "r"(h * kHd), "r"(smem_u32(&wbar))
                    : "memory");
            asm volatile(
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                    smem_u32(sWsf)),
                "l"(wsrc), "r"(G::kTiles * 512), "r"(smem_u32(&wbar))
                : "memory");
            mbar_wait(smem_u32(&wbar), 0, 3);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for (int t = 0; t < G::kTiles; ++t)
                asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem + kTSfa + 4 * t),
                             "l"(sdesc(smem_u32(sWsf + t * 512), 16, 128, 0))
                             : "memory");
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(&abar))
                : "memory");
            mbar_wait(smem_u32(&abar), 0, 4); // sWsf is free once SFA is in TMEM: it becomes the K-split receive buffer
        }
        __syncwarp();
        if (bnd)
            asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); // #1
        else
            asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); // #2: sWsf free, rbar live
    }
    else
    {
        // The call counter (read before any CTA of this call updates its own: every CTA writes only its own slot,
        // after its last read).
        if (tid == 0)
            s_call = p.ctr[blockIdx.x];
        bnd_sync();
        if (bnd)
            boundary_row<CL>(p, h, crank, s_call, warp_sq, cluster_wsq, &s_total, &xbar);
        else
            asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int const call = warp == 15 ? p.ctr[blockIdx.x] : s_call;
    {
        // Re-arm the other parity (its readers, the previous call, have completed), a strided share per CTA.
        uint4* const xr = reinterpret_cast<uint4*>(p.xq + ((call & 1) ^ 1) * kXqBuf);
        uint4* const sr = reinterpret_cast<uint4*>(p.xsf + ((call & 1) ^ 1) * kXsfBuf);
        int const stride = static_cast<int>(gridDim.x) * kTh;
        for (int i = blockIdx.x * kTh + tid; i < kXqBuf / 16; i += stride)
            xr[i] = make_uint4(0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u);
        for (int i = blockIdx.x * kTh + tid; i < kXsfBuf / 16; i += stride)
            sr[i] = make_uint4(~0u, ~0u, ~0u, ~0u);
    }
    if (warp != 15)
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); // #2
    if (a.earlyTrigger)
        asm volatile("griddepcontrol.launch_dependents;");

    // Lamport hand-off: this CTA's k-tiles of every row, polled straight from the boundary's parity buffer into the
    // B tiles and scale atoms (all of a thread's items in flight together); warps 0..14 poll.
    {
        uint8_t const* const xqp = p.xq + (call & 1) * kXqBuf;
        uint8_t const* const xsfp = p.xsf + (call & 1) * kXsfBuf;
        constexpr int kPT = kBndTh;
        constexpr int kQ = G::kPollQ;
        int const nItems = R * G::kTiles * 8, nWords = R * G::kTiles;
        int4 v[kQ];
        uint32_t wsc = 0;
        unsigned pend = 0;
#pragma unroll
        for (int q = 0; q < kQ; ++q)
            if (tid + q * kPT < nItems && warp != 15)
                pend |= 1u << q;
        if (tid < nWords)
            pend |= 1u << kQ;
        long long spins = 0;
        while (pend)
        {
#pragma unroll
            for (int q = 0; q < kQ; ++q)
                if (pend & (1u << q))
                {
                    int const i = tid + q * kPT, r = i / (G::kTiles * 8), ti = (i / 8) % G::kTiles, ck = i % 8;
                    v[q] = ld_volatile_v4(reinterpret_cast<int const*>(
                        xqp + (crank * G::kTiles + ti) * kXTile + r * 128 + ((ck ^ (r & 7)) * 16)));
                }
            if (pend & (1u << kQ))
            {
                int const r = tid / G::kTiles, ti = tid % G::kTiles;
                asm volatile("ld.volatile.global.u32 %0, [%1];"
                             : "=r"(wsc)
                             : "l"(xsfp + (crank * G::kTiles + ti) * kXAtom + r * 16)
                             : "memory");
            }
#pragma unroll
            for (int q = 0; q < kQ; ++q)
                if ((pend & (1u << q))
                    && (__vcmpeq4(v[q].x, 0x80808080u) | __vcmpeq4(v[q].y, 0x80808080u) | __vcmpeq4(v[q].z, 0x80808080u)
                           | __vcmpeq4(v[q].w, 0x80808080u))
                        == 0u)
                {
                    int const i = tid + q * kPT, r = i / (G::kTiles * 8), ti = (i / 8) % G::kTiles, ck = i % 8;
                    *reinterpret_cast<int4*>(sX + ti * kXTile + r * 128 + ((ck ^ (r & 7)) * 16)) = v[q];
                    pend &= ~(1u << q);
                }
            if ((pend & (1u << kQ)) && __vcmpeq4(wsc, 0xFFFFFFFFu) == 0u)
            {
                *reinterpret_cast<uint32_t*>(sXsf + (tid % G::kTiles) * kXAtom + (tid / G::kTiles) * 16) = wsc;
                pend &= ~(1u << kQ);
            }
            if (++spins > M31_QKV_FRONT_SPIN_LIMIT)
                spin_trap(5, call, (static_cast<unsigned long long>(nItems) << 32) | pend,
                    (static_cast<unsigned long long>(static_cast<uint32_t>(v[0].x)) << 32) | wsc);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); // generic smem writes -> tcgen05 reads
        __syncthreads();
    }

    uint32_t const tmem = tmem_base_s;
    // MMA (warp 15, lane 0): M = 128 weight rows (A from shared memory), N = 8 activation rows (B), K = 32 per
    // instruction, UE8M0 scales in TMEM; MMA j of the CTA accumulates into j % NACC (NACC 1: one chain, the GEMV's).
    if (tid == kTh - 32)
    {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint32_t const idesc0 = (1u << 17) | (1u << 23) | (8u << 24); // N 8, UE8M0 block scales, M 128
        for (int i = 0; i < G::kTiles; ++i)
            asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(tmem + kTSfb + 4 * i),
                         "l"(sdesc(smem_u32(sXsf + i * kXAtom), 16, 128, 0))
                         : "memory");
        for (int i = 0; i < G::kTiles; ++i)
        {
            uint64_t const da = sdesc(smem_u32(sW + i * (kHd * 128)), 16, 1024, 2);
            uint64_t const db = sdesc(smem_u32(sX + i * kXTile), 16, 1024, 2);
#pragma unroll
            for (int kb = 0; kb < 4; ++kb)
            {
                uint32_t const idesc = idesc0 | (static_cast<uint32_t>(kb) << 4) | (static_cast<uint32_t>(kb) << 29);
                int const j = i * 4 + kb;
                uint32_t const accum = j >= NACC;
                asm volatile(
                    "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                    " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n}\n" ::"r"(
                        tmem + (j % NACC) * 8),
                    "l"(da + 2 * kb), "l"(db + 2 * kb), "r"(idesc), "r"(accum), "r"(tmem + kTSfa + 4 * i),
                    "r"(tmem + kTSfb + 4 * i)
                    : "memory");
            }
        }
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(&mbar))
            : "memory");
    }
    __syncwarp();
    mbar_wait(smem_u32(&mbar), 0, 6);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    // K-split reduce: row n belongs to CTA n % CL; every CTA st.async's its FP32 partial of that row into the owner's
    // receive buffer [source CTA][slot][128] (sWsf), counted on the owner's rbar.
    if (warp < 4)
    {
        float sum[8];
        uint32_t const ta = tmem + (static_cast<uint32_t>(warp * 32) << 16);
#pragma unroll
        for (int q = 0; q < NACC; ++q)
        {
            uint32_t v[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7])
                : "r"(ta + q * 8)
                : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
            for (int n = 0; n < 8; ++n)
                sum[n] = q == 0 ? __uint_as_float(v[n]) : sum[n] + __uint_as_float(v[n]); // the GEMV's acc0 + acc1 ..
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        asm volatile("bar.arrive 2, 160;" ::: "memory"); // TMEM reads done (warp 15 deallocates)
        int const cnt = R > crank ? (R - crank + CL - 1) / CL : 0;
        if (tid == 0 && cnt > 0)
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&rbar)), "r"(CL * cnt * kHd * 4)
                : "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); // #2: every peer's receive buffer is free
        int const m = warp * 32 + lane;
        uint32_t const rb = smem_u32(sWsf), bb = smem_u32(&rbar);
#pragma unroll
        for (int n = 0; n < 8; ++n)
        {
            if (n < R)
            {
                uint32_t const off = rb + (((crank * G::kSlots + n / CL) * kHd + m) << 2);
                st_async_f32(mapa(off, n % CL), sum[n], mapa(bb, n % CL));
            }
        }
        if (cnt > 0)
            mbar_wait_cluster(bb, 0, 7);
    }
    else if (warp == 15)
    {
        asm volatile("bar.sync 2, 160;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(kTmemCols) : "memory");
    }

    // Producer epilogue (producerPw4.cu, one warp per (token, head)): the owner adds the CL partials in rank order
    // from 0.0 and rounds to BF16 (the GEMV's output), then the head's norm / RoPE / quantize / insert.
    if (owner)
    {
        float const* const rbuf = reinterpret_cast<float const*>(sWsf) + warp * kHd + lane * kElems;
        float4 acc = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
        for (int src = 0; src < CL; ++src)
        {
            float4 const v = *reinterpret_cast<float4 const*>(rbuf + src * G::kSlots * kHd);
            acc.x = acc.x + v.x;
            acc.y = acc.y + v.y;
            acc.z = acc.z + v.z;
            acc.w = acc.w + v.w;
        }
        int const tokenIdx = ownRow;
        int const laneId = lane;
        float cosine[kElems] = {c.x, c.y, c.z, c.w};
        float sine[kElems] = {s.x, s.y, s.z, s.w};
        __nv_bfloat162 const pk[2] = {__floats2bfloat162_rn(acc.x, acc.y), __floats2bfloat162_rn(acc.z, acc.w)};

        float e[kElems];
        float sumSquares = 0.0F;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair)
        {
            float2 const v = __bfloat1622float2(pk[pair]);
            e[2 * pair] = v.x;
            e[2 * pair + 1] = v.y;
            if (!isV)
                sumSquares += v.x * v.x + v.y * v.y;
        }

        if (!isV)
        {
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, mask, 32);
            float const rmsReciprocal = rsqrtf(sumSquares / static_cast<float>(kHd) + p.peps);
#pragma unroll
            for (int i = 0; i < kElems; ++i)
            {
                e[i] *= rmsReciprocal * (1.0F + w[i]);
            }
            constexpr int kPairOffset = (kRotaryDim / 2) / kElems;
#pragma unroll
            for (int i = 0; i < kElems; ++i)
            {
                int const dim = laneId * kElems + i;
                float partner = __shfl_xor_sync(0xffffffff, e[i], kPairOffset);
                if (dim < kRotaryDim)
                {
                    if (dim < kRotaryDim / 2)
                        partner = -partner;
                    // The native kernel's contraction of e * cos + partner * sin; spelled out so this build matches it.
                    e[i] = __fmaf_rn(partner, sine[i], __fmul_rn(e[i], cosine[i]));
                }
            }
        }

        if (isQ)
        {
            store_fp8x4(p.qOut, (static_cast<int64_t>(tokenIdx) * nq + h) * kHd + laneId * kElems, e);
        }
        else if (isIndexQ)
        {
            store_fp8x4(p.indexQOut, static_cast<int64_t>(tokenIdx) * kHd + laneId * kElems, e);
        }
        else if (slot >= 0)
        {
            int const page = slot / kPageSize;
            if (page < p.numPages)
            {
                int const withinPage = slot & (kPageSize - 1);
                if (isIndexK)
                {
                    store_fp8x4(p.indexK,
                        static_cast<int64_t>(page) * p.indexPageStride
                            + static_cast<int64_t>(withinPage) * p.indexTokenStride + laneId * kElems,
                        e);
                }
                else
                {
                    int const head = 0; // one KV head per rank
                    int const plane = isV ? 1 : 0;
                    int const scaleColumn = laneId / 4;
                    int64_t const scaleWithinPage = isV ? static_cast<int64_t>(withinPage / 4) * 32
                            + static_cast<int64_t>(scaleColumn) * 4 + withinPage % 4
                                                        : static_cast<int64_t>(withinPage) * 8 + scaleColumn;
                    int64_t const scaleOffset = static_cast<int64_t>(page) * p.scalePageStride
                        + static_cast<int64_t>(plane) * p.scalePlaneStride
                        + static_cast<int64_t>(head) * p.scaleHeadStride + scaleWithinPage;
                    uint8_t* scaleOut = laneId % 4 == 0 ? p.kvScale + scaleOffset : nullptr;
                    uint16_t const packedFp4 = quantize_fp4x4(e, quantScale, scaleOut);
                    int64_t const dataOffset = static_cast<int64_t>(page) * p.dataPageStride
                        + static_cast<int64_t>(plane) * p.dataPlaneStride
                        + static_cast<int64_t>(head) * p.dataHeadStride
                        + static_cast<int64_t>(withinPage) * p.dataTokenStride + laneId * 2;
                    *reinterpret_cast<uint16_t*>(p.kvData + dataOffset) = packedFp4;
                }
            }
        }
    }
    if (tid == 0)
        p.ctr[blockIdx.x] = call + 1;
    if (!a.earlyTrigger)
        asm volatile("griddepcontrol.launch_dependents;");
#elif defined(__CUDA_ARCH__)
    __trap(); // compiled for an architecture without tcgen05: never a silent no-op
#endif
}

// ---------------------------------------------------------------- host
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, cuuint64_t const*,
    cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
    CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiledFn encodeTiled()
{
    static EncodeTiledFn const fn = []() -> EncodeTiledFn
    {
        void* f = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000, cudaEnableDefault, &q) != cudaSuccess
            || q != cudaDriverEntryPointSuccess)
        {
            cudaGetLastError();
            return nullptr;
        }
        return reinterpret_cast<EncodeTiledFn>(f);
    }();
    return fn;
}

// The weight's tensor map ([N, 6144] u8, box 128 x 128, 128-byte swizzle), cached per (device, weight address).
bool weightMap(void const* w, int64_t rows, CUtensorMap* out)
{
    static std::mutex lock;
    static std::vector<std::pair<std::pair<int, void const*>, CUtensorMap>> cache;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    std::lock_guard<std::mutex> guard(lock);
    for (auto const& e : cache)
    {
        if (e.first.first == dev && e.first.second == w)
        {
            *out = e.second;
            return true;
        }
    }
    EncodeTiledFn const enc = encodeTiled();
    if (enc == nullptr)
        return false;
    CUtensorMap m;
    cuuint64_t const dims[2] = {static_cast<cuuint64_t>(kH), static_cast<cuuint64_t>(rows)};
    cuuint64_t const strides[1] = {static_cast<cuuint64_t>(kH)};
    cuuint32_t const box[2] = {128, 128};
    cuuint32_t const es[2] = {1, 1};
    if (enc(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(w), dims, strides, box, es,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
        != CUDA_SUCCESS)
        return false;
    cache.emplace_back(std::make_pair(dev, w), m);
    *out = m;
    return true;
}

// The launch configuration of `grid` CTAs in clusters of CL (attrs: PDL, cluster).
template <int CL>
void frontConfig(cudaLaunchConfig_t& cfg, cudaLaunchAttribute (&attr)[2], int grid, bool pdl, cudaStream_t stream)
{
    cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(kTh);
    cfg.dynamicSmemBytes = Geo<CL>::kSmem;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CL;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
}

// Per device, once (before any capture uses the kernel: the first, eager, call): the binary check, the shared-memory
// attribute and the number of clusters the device can hold at once. The GEMV clusters wait on the boundary clusters'
// hand-off, so every cluster of the grid must be able to be resident together: a grid with more clusters than that is
// refused (cudaErrorLaunchOutOfResources) instead of risking a co-residency stall.
template <int CL, int NACC, int ROWS>
cudaError_t configureFront(int& maxClusters)
{
    auto kernel = qkv_front_kernel<CL, NACC, ROWS>;
    static std::mutex lock;
    static int clusters[64] = {};
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
        return err;
    std::lock_guard<std::mutex> guard(lock);
    if (dev < 0 || dev >= 64)
        return cudaErrorInvalidDevice;
    if (clusters[dev] == 0)
    {
        cudaFuncAttributes fa{};
        err = cudaFuncGetAttributes(&fa, reinterpret_cast<void const*>(kernel));
        if (err != cudaSuccess)
            return err;
        // the loaded binary must hold sm_10x code: other architectures compile the kernel to a trap
        if (fa.binaryVersion < 100 || fa.binaryVersion >= 110)
            return cudaErrorInvalidDeviceFunction;
        err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Geo<CL>::kSmem);
        if (err != cudaSuccess)
            return err;
        cudaLaunchConfig_t cfg;
        cudaLaunchAttribute attr[2];
        frontConfig<CL>(cfg, attr, kQkvFrontMaxCtas / CL * CL, false, nullptr);
        int n = 0;
        err = cudaOccupancyMaxActiveClusters(&n, kernel, &cfg);
        if (err != cudaSuccess)
            return err;
        clusters[dev] = n > 0 ? n : -1;
    }
    maxClusters = clusters[dev];
    return cudaSuccess;
}

template <int CL, int NACC, int ROWS>
cudaError_t launchFront(FrontArgs const& args, bool pdl, cudaStream_t stream)
{
    auto kernel = qkv_front_kernel<CL, NACC, ROWS>;
    int maxClusters = 0;
    cudaError_t const err = configureFront<CL, NACC, ROWS>(maxClusters);
    if (err != cudaSuccess)
        return err;
    if (maxClusters < args.p.headsQ + 4)
        return cudaErrorLaunchOutOfResources;
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attr[2];
    frontConfig<CL>(cfg, attr, (args.p.headsQ + 4) * CL, pdl, stream);
    return cudaLaunchKernelEx(&cfg, kernel, args);
}

template <int CL, int NACC>
cudaError_t dispatchRows(FrontArgs const& args, bool pdl, cudaStream_t stream)
{
    if (args.p.numTokens == 4)
        return launchFront<CL, NACC, 4>(args, pdl, stream);
    if (args.p.numTokens == 8)
        return launchFront<CL, NACC, 8>(args, pdl, stream);
    return launchFront<CL, NACC, 0>(args, pdl, stream);
}

} // namespace

bool isQkvFrontSplitSupported(int split)
{
    return split == 8 || split == 4;
}

int qkvFrontMaxActiveClusters(int split)
{
    int n = 0;
    cudaError_t const err
        = split == 8 ? configureFront<8, 1, 8>(n) : (split == 4 ? configureFront<4, 4, 8>(n) : cudaErrorInvalidValue);
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return -1;
    }
    return n;
}

cudaError_t invokeQkvFront(QkvFrontParams const& params, int split, bool pdl, bool earlyTrigger, cudaStream_t stream)
{
    if (!isQkvFrontSplitSupported(split) || params.numTokens < 1 || params.numTokens > kQkvFrontMaxTokens
        || params.headsQ < 1 || (params.headsQ + 4) * split > kQkvFrontMaxCtas || params.world < 2 || params.world > 8
        || params.maxt < params.numTokens || (params.prepushed != 2 && params.prepushed != 3) || params.wsUc == nullptr
        || params.flags == nullptr || params.xq == nullptr || params.xsf == nullptr || params.ctr == nullptr
        || params.weight == nullptr || params.weightScale == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    FrontArgs args{};
    if (!weightMap(params.weight, static_cast<int64_t>(params.headsQ + 4) * kHd, &args.tmw))
        return cudaErrorInvalidValue;
    args.p = params;
    args.earlyTrigger = earlyTrigger ? 1 : 0;
    // split 8, one accumulator: the served GEMV's order (bitwise); split 4, four accumulators: the reference geometry
    return split == 8 ? dispatchRows<8, 1>(args, pdl, stream) : dispatchRows<4, 4>(args, pdl, stream);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
