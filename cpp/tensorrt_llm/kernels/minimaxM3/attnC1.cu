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

// MiniMax-M3 attention + o_proj kernel (c1): fused decode attention and o_proj GEMV in one 96-CTA launch.

#include "tensorrt_llm/kernels/minimaxM3/attnC1.h"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mutex>
#include <type_traits>
#include <unordered_map>

// The kernel uses tcgen05/TMEM (including block-scaled MMAs, which family targets lack), cluster DSMEM and E2M1
// conversions: its body is compiled for the arch-specific sm_100a / sm_103a targets only (see the minimax_m3_attn_src
// library in kernels/CMakeLists.txt). Any other target compiles a kernel that traps, so a launch on a build without
// those targets fails loudly instead of returning untouched outputs.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)                                         \
    && (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL))
#define MINIMAX_M3_ATTN_DEVICE_ENABLED 1
#else
#define MINIMAX_M3_ATTN_DEVICE_ENABLED 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kHeadDim = 128;
constexpr int kRotDim = 64;
constexpr int kPage = 128;
constexpr int kHq = 16;
constexpr int kColK = kHq; // packed head slots: [Q x16 | K | V | idxQ | idxK]
constexpr int kColV = kHq + 1;
constexpr int kColIdxK = kHq + 3;
constexpr int kBlocks = 16;               // KV blocks per request == topk == cluster size
constexpr int kComputeWarps = 16;         // QK / softmax / PV and the dequant (one 16-byte chunk per thread)
constexpr int kWarps = kComputeWarps + 4; // plus producer-only warps, so new K/V/index-K never queue behind a Q head
constexpr int kThreads = kWarps * 32;
constexpr int kComputeThreads = kComputeWarps * 32;
constexpr int kOPitch = 132; // floats per partial-O row
constexpr float kNegInf = -__builtin_huge_valf();

using Params = MinimaxM3AttnC1Params;

// Optional per-CTA clock64 probes for development (-DM3FA_PROFILE and a prof tensor): [CTA][64] timestamps.
// -DM3FA_PROFILE_GT records %globaltimer (ns, comparable across SMs and kernels) instead of clock64.
#ifdef M3FA_PROFILE_GT
#define M3FA_PROFILE
#define M3FA_NOW(v) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v))
#else
#define M3FA_NOW(v) v = clock64()
#endif
#ifdef M3FA_PROFILE
#define M3FA_PROF_T(i, thread)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.prof != nullptr && threadIdx.x == (thread))                                                              \
        {                                                                                                              \
            unsigned long long now_;                                                                                   \
            M3FA_NOW(now_);                                                                                            \
            p.prof[blockIdx.x * 64 + (i)] = now_;                                                                      \
        }                                                                                                              \
    } while (0)
#else
#define M3FA_PROF_T(i, thread)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif
#define M3FA_PROF(i) M3FA_PROF_T(i, 0)
// clock64 probes (development): slot i of this CTA, written by every calling thread.
#ifdef M3FA_PROFILE
#define M3FA_CLK(i)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (p.prof != nullptr)                                                                                         \
        {                                                                                                              \
            p.prof[blockIdx.x * 64 + (i)] = clock64();                                                                 \
        }                                                                                                              \
    } while (0)
#else
#define M3FA_CLK(i)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

#ifndef M3FA_MEGA_EARLYCLW
#define M3FA_MEGA_EARLYCLW 1 // pass the cluster barrier (peers' merge mbarriers initialized) early, not before the push
#endif
#ifndef M3FA_V15_EARLYV
#define M3FA_V15_EARLYV 0 // packed path: dequantize V before the PDL wait (switch; off by default)
#endif
#ifndef M3FA_MEGA_TRIGGER
// Megakernel PDL release: 0 o_proj CTAs after their activation (attention CTAs at exit); 1 as 0 with the attention
// CTAs after their KV reads; 2 every CTA right after its griddepcontrol.wait; 3 o_proj CTAs after their epilogue.
#define M3FA_MEGA_TRIGGER 3
#endif
#ifndef M3FA_TRIGGER_AT
#define M3FA_TRIGGER_AT 2 // where the PDL dependents are released: 0 entry, 1 after the wait, 2 after the KV reads
#endif

// Generic-proxy shared-memory writes (fp16 tiles, Q, P) become visible to the tensor core (async proxy).
#define M3FA_FENCE_PROXY_ASYNC() asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory")

__device__ __forceinline__ void cp_async16(void* dst, void const* src)
{
    uint32_t const d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
}

__device__ __forceinline__ float ex2(float x)
{
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float lg2(float x)
{
    float y;
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// One packed E2M1 byte -> half2 (low nibble in the low half).
__device__ __forceinline__ uint32_t e2m1x2_to_h2(uint32_t byte)
{
    uint32_t out;
    asm("{\n .reg .b8 src;\n mov.b32 {src, _, _, _}, %1;\n cvt.rn.f16x2.e2m1x2 %0, src;\n}\n" : "=r"(out) : "r"(byte));
    return out;
}

__device__ __forceinline__ uint32_t hmul2_u(uint32_t a, uint32_t b)
{
    __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint32_t e4m3_to_h2_splat(uint32_t byte)
{
    __half_raw const h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.x) << 16);
}

__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t two_bytes)
{
    __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(two_bytes), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// ---- Exact copies of the C++ producer's helpers (fusedQKNormRopeKernel.cu / quantization.cuh) ----
template <int NUM_THREADS_PER_SF>
__device__ __forceinline__ uint32_t cvt_sf_group_shfl_mask()
{
    constexpr uint32_t groupSize = static_cast<uint32_t>(NUM_THREADS_PER_SF);
    constexpr uint32_t groupMask = 0xFFFFFFFFU >> (32U - groupSize);
    uint32_t laneId = 0;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return groupMask << (laneId & ~(groupSize - 1U));
}

__device__ __forceinline__ uint32_t fp32_vec_to_e2m1(float (&array)[8])
{
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
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]), "f"(array[6]),
        "f"(array[7]));
    return val;
}

__device__ __forceinline__ uint16_t quantizeMinimaxM3Fp4x4(
    float const (&elements)[4], float secondLevelScale, uint8_t* scaleOut)
{
    __nv_bfloat16 rounded[4];
    float localMax = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        rounded[i] = __float2bfloat16_rn(elements[i]);
        localMax = fmaxf(localMax, fabsf(__bfloat162float(rounded[i])));
    }
    uint32_t const groupMask = cvt_sf_group_shfl_mask<4>();
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 1));
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 2));
    float const scaleValue = secondLevelScale * (localMax * rcp_approx_ftz(6.0F));
    __nv_fp8_e4m3 const encodedScale(scaleValue);
    if (scaleOut != nullptr)
    {
        *scaleOut = encodedScale.__x;
    }
    float const narrowedScale = static_cast<float>(encodedScale);
    float const outputScale
        = localMax != 0.0F ? rcp_approx_ftz(narrowedScale * rcp_approx_ftz(secondLevelScale)) : 0.0F;
    float quantValues[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        quantValues[i] = __bfloat162float(rounded[i]) * outputScale;
    }
    return static_cast<uint16_t>(fp32_vec_to_e2m1(quantValues));
}

// RMSNorm (gemma 1+w) + NeoX RoPE over the first 64 dims, exactly as the producer: lane owns 4 dims.
__device__ __forceinline__ void norm_rope(
    float (&e)[4], float const (&w)[4], float eps, float const (&cs)[4], float const (&sn)[4], int lane)
{
    float ss = 0.0F;
    ss += e[0] * e[0] + e[1] * e[1];
    ss += e[2] * e[2] + e[3] * e[3];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        ss = ss + __shfl_xor_sync(0xffffffffu, ss, mask, 32);
    }
    float const rmsReciprocal = rsqrtf(ss / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        e[i] *= rmsReciprocal * (1.0F + w[i]);
    }
    __syncwarp();
    constexpr int kPairOffset = (kRotDim / 2) / 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int const dim = lane * 4 + i;
        float paired = __shfl_xor_sync(0xffffffffu, e[i], kPairOffset);
        if (dim < kRotDim)
        {
            bool const firstHalf = dim < kRotDim / 2;
            if (firstHalf)
            {
                paired = -paired;
            }
            e[i] = e[i] * cs[i] + paired * sn[i];
        }
    }
    __syncwarp();
}

__device__ __forceinline__ void load_w4(float (&w)[4], __nv_bfloat16 const* src, int lane)
{
    uint2 const raw = *reinterpret_cast<uint2 const*>(src + lane * 4);
    float2 const a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw.x));
    float2 const b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw.y));
    w[0] = a.x;
    w[1] = a.y;
    w[2] = b.x;
    w[3] = b.y;
}

__device__ __forceinline__ uint32_t smem_u32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

// QLEN query tokens per request; QG of them per cluster (QLEN / QG clusters per request); BPC KV blocks per CTA
// (cluster of kBlocks / BPC CTAs).
// Byte offset of element (r, k) in a UMMA no-swizzle K-major operand tile of 128 k-elements per row: 8x8 core
// matrices of 16-byte rows, 128 bytes apart along k (LBO) and 2048 bytes apart per 8 rows (SBO). The same formula
// with r = token, k = dim stores V so that it is the MN-major (dims contiguous) A operand of O^T = V^T P^T.
__device__ __forceinline__ int kmaj_off(int r, int k)
{
    return (r >> 3) * 2048 + (k >> 3) * 128 + (r & 7) * 16 + (k & 7) * 2;
}

__device__ __forceinline__ uint64_t umma_desc(uint32_t addr, uint32_t lbo, uint32_t sbo)
{
    return static_cast<uint64_t>((addr >> 4) & 0x3FFFu) | (static_cast<uint64_t>((lbo >> 4) & 0x3FFFu) << 16)
        | (static_cast<uint64_t>((sbo >> 4) & 0x3FFFu) << 32) | (1ull << 46);
}

// kind::f16 instruction descriptor: F32 accumulate, F16 operands, B K-major, M = 128.
__host__ __device__ constexpr uint32_t umma_idesc(int n, int a_mn_major, int b_mn_major)
{
    return (1u << 4) | (static_cast<uint32_t>(a_mn_major) << 15) | (static_cast<uint32_t>(b_mn_major) << 16)
        | (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(128 >> 4) << 24);
}

// 8-row version for the split softmax: the same lane-bit order (16, 8, 4, then 2 and 1 in full) as
// warp_reduce_scatter16, so every row is combined in the same association (fp add is commutative, not associative).
// After it lanes with bits 4..2 = (b4, b3, b2) hold row 4*b4 + 2*b3 + b2.
template <bool kMax>
__device__ __forceinline__ float warp_reduce_scatter8(float (&a)[8], int lane)
{
#pragma unroll
    for (int step = 0; step < 3; ++step)
    {
        int const mask = 16 >> step;
        int const half = 4 >> step;
        bool const upper = (lane & mask) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i)
        {
            float const send = upper ? a[i] : a[i + half];
            float const keep = upper ? a[i + half] : a[i];
            float const recv = __shfl_xor_sync(0xffffffffu, send, mask);
            a[i] = kMax ? fmaxf(keep, recv) : keep + recv;
        }
    }
    float x = a[0];
    float const o2 = __shfl_xor_sync(0xffffffffu, x, 2);
    x = kMax ? fmaxf(x, o2) : x + o2;
    float const o1 = __shfl_xor_sync(0xffffffffu, x, 1);
    return kMax ? fmaxf(x, o1) : x + o1;
}

// Reduce 16 per-lane values over the warp, one value per row: after the reduce-scatter lane l holds row
// 8*b4 + 4*b3 + 2*b2 + b1 (b_k = bit k of l) reduced over all 32 lanes. 16 shuffles instead of 80.
template <bool kMax>
__device__ __forceinline__ float warp_reduce_scatter16(float (&a)[16], int lane)
{
#pragma unroll
    for (int step = 0; step < 4; ++step)
    {
        int const mask = 16 >> step;
        int const half = 8 >> step;
        bool const upper = (lane & mask) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i)
        {
            float const send = upper ? a[i] : a[i + half];
            float const keep = upper ? a[i + half] : a[i];
            float const recv = __shfl_xor_sync(0xffffffffu, send, mask);
            a[i] = kMax ? fmaxf(keep, recv) : keep + recv;
        }
    }
    float const other = __shfl_xor_sync(0xffffffffu, a[0], 1);
    return kMax ? fmaxf(a[0], other) : a[0] + other;
}

__device__ __forceinline__ void umma_f16(
    uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, int accumulate)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::
            "r"(tmem_d),
        "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate)
        : "memory");
}

__device__ __forceinline__ void umma_commit(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_wait_cta(uint32_t bar, uint32_t parity)
{
    uint32_t ready = 0;
#ifdef M3FA_DBG_HANG
    long long spins = 0;
#endif
    while (!ready)
    {
#ifdef M3FA_DBG_HANG
        if (++spins == (1LL << 24))
        {
            printf("M3FA hang: block %d thread %d mbar 0x%x parity %u\n", blockIdx.x, threadIdx.x, bar, parity);
        }
#endif
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(ready)
            : "r"(bar), "r"(parity)
            : "memory");
    }
}

// Spin without suspending (v52's acc_done wait): a suspended try_wait wakes late on a tcgen05.commit arrival.
#ifndef M3FA_EPI_SPIN
#define M3FA_EPI_SPIN 1
#endif
__device__ __forceinline__ void mbar_wait_epi(uint32_t bar, uint32_t parity)
{
    if (!M3FA_EPI_SPIN)
    {
        mbar_wait_cta(bar, parity);
        return;
    }
    uint32_t ready = 0;
    while (!ready)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(ready)
            : "r"(bar), "r"(parity)
            : "memory");
    }
}

__device__ __forceinline__ void tmem_ld8(uint32_t addr, float (&v)[8])
{
    uint32_t r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__ void tmem_ld16(uint32_t addr, float (&v)[16])
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, [%16];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

// Warp-level fp16 MMA (m16n8k16, fp32 accumulate) and its shared-memory fragment loads.
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4])
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t (&r)[2])
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void ldsm_x2_trans(uint32_t addr, uint32_t (&r)[2])
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_f16_16816(float (&d)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Register attention core (warp MMAs, no TMEM) for up to two tokens per cluster; the tcgen05 core above that.
// Off by default: on GB300 it measured ~0.25 us slower than the tcgen05 core at M=1 (GEMV + attention chain).
#ifndef M3FA_REG_CORE
#define M3FA_REG_CORE 0
#endif
template <int QG>
struct RegCore
{
    static constexpr bool kOn = M3FA_REG_CORE != 0 && QG <= 2;
};

template <int QLEN, int QG, int BPC>
struct Smem
{
    static constexpr int kRows = 16 * QG;                   // query rows (token, head) of this cluster = UMMA N
    static constexpr int kTok = BPC * kPage;                // block tokens per CTA
    static constexpr int kTileBytes = kPage * kHeadDim * 2; // one fp16 K or V block tile in UMMA layout
    static constexpr int kK16 = 0;
    static constexpr int kV16 = kK16 + BPC * kTileBytes;
    static constexpr int kQ = kV16 + BPC * kTileBytes; // Q rows, UMMA K-major [row][dim]
    // P^T per block, UMMA MN-major [token][row], over the block's K tile: P is written after the QK MMAs (which
    // read every K tile) have completed.
    static constexpr int kP = kK16;
    // Packed V tiles as copied from the cache: [block][128 rows x 64 data bytes | 1 KB of 4x4-quad scales].
    static constexpr int kRawV = kQ + kRows * kHeadDim * 2;
    static constexpr int kRawVBlock = kPage * 64 + kPage * 8;
    static constexpr int kRowScale = kRawV + BPC * kRawVBlock;
    static constexpr int kRowLse = kRowScale + BPC * kRows * 4;
    // Per-warp softmax partials [block][row][4 warps]: max, then sum.
    static constexpr int kRedM = (kRowLse + BPC * kRows * 4 + 15) / 16 * 16;
    static constexpr int kRedL = kRedM + BPC * kRows * 16;
    // Partials pushed by every CTA of the cluster: [chunk][token][local head][kOPitch] floats, lse at column 128.
    // When they do not fit beside the tiles they overlay the K/V tiles (and P, Q): peers then push only once every
    // CTA of the cluster has completed its PV MMAs (a cluster barrier right before the push).
    // Register core: per-warp row maxima and sums [block][row][warp of the block], two arrays.
    static constexpr int kRedX = kRedL + BPC * kRows * 16;
    static constexpr int kRedXBytes = RegCore<QG>::kOn ? 2 * BPC * kRows * (16 / BPC) * 4 : 0;
    static constexpr int kStatsEnd = kRedX + kRedXBytes;
    // Register core: P (fp16, [row][token] core matrices) at the start of its block's K tile, the scaled-free O
    // partial ([row][kOPitch] fp32) behind it.
    static constexpr int kObufOff = 8192;
    static_assert(!RegCore<QG>::kOn || (kRows * kPage * 2 <= kObufOff && kObufOff + kRows * kOPitch * 4 <= kTileBytes),
        "register-core P and O staging fit in a K tile");
    static constexpr int kRecvBytes = kBlocks * QG * BPC * kOPitch * 4;
    static constexpr bool kRecvAlias = kStatsEnd + kRecvBytes + 32 > 227 * 1024;
    static constexpr int kRecv = kRecvAlias ? 0 : kStatsEnd;
    static constexpr int kMbar = kRecvAlias ? kStatsEnd : kStatsEnd + kRecvBytes; // merge, MMA mbarriers, TMEM addr
    static constexpr int kBytes = kMbar + 32;
    static constexpr int kTmemCols = BPC * 2 * kRows <= 32 ? 32
        : BPC * 2 * kRows <= 64                            ? 64
        : BPC * 2 * kRows <= 128                           ? 128
        : BPC * 2 * kRows <= 256                           ? 256
                                                           : 512;
    static_assert(BPC * 2 * kRows <= 512, "TMEM columns");
    static_assert(!kRecvAlias || kRecvBytes <= kRawV, "aliased partials must fit in the tiles, Q and P");
    static_assert(kRows * kPage * 2 <= kTileBytes, "P^T fits in the K tiles");
    static_assert(kBytes <= 227 * 1024, "shared memory budget");
    static_assert(QLEN % QG == 0 && (BPC == 1 || BPC == 2), "unsupported geometry");
};

// The packed NVFP4 bytes of one 16-byte chunk (32 dims of one block row) and its two block scales.
struct RawChunk
{
    uint4 data;
    uint32_t scales; // byte 0: dims [0, 16), byte 1: dims [16, 32) of the chunk
};

// Chunk q of a block tile: lanes 8j..8j+7 of a warp take 8 consecutive rows of one 32-dim chunk, so every 16-byte
// fp16 store phase writes 8 distinct core-matrix rows (conflict-free in the UMMA layout).
__device__ __forceinline__ int chunk_row(int q)
{
    return (q >> 5) * 8 + (q & 7);
}

__device__ __forceinline__ int chunk_col(int q)
{
    return (q >> 3) & 3;
}

__device__ __forceinline__ RawChunk load_chunk(
    uint8_t const* data_page, uint8_t const* scale_page, int row, int cc, bool is_v)
{
    RawChunk c;
    c.data = *reinterpret_cast<uint4 const*>(data_page + row * 64 + cc * 16);
    if (is_v)
    {
        // V scales: 4x4 token-quad order [row / 4][scale column][row % 4].
        uint8_t const* sc = scale_page + (row >> 2) * 32 + (row & 3);
        c.scales = static_cast<uint32_t>(sc[(cc * 2) * 4]) | (static_cast<uint32_t>(sc[(cc * 2 + 1) * 4]) << 8);
    }
    else
    {
        c.scales = *reinterpret_cast<uint16_t const*>(scale_page + row * 8 + cc * 2);
    }
    return c;
}

__device__ __forceinline__ void store_chunk_fp16(uint8_t* tile, int row, int cc, RawChunk const& c, bool valid)
{
    uint32_t const s0 = e4m3_to_h2_splat(c.scales & 0xFFu);
    uint32_t const s1 = e4m3_to_h2_splat((c.scales >> 8) & 0xFFu);
    uint32_t const wd[4] = {c.data.x, c.data.y, c.data.z, c.data.w};
    uint32_t out[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        uint32_t const b = (wd[i >> 2] >> (8 * (i & 3))) & 0xFFu;
        out[i] = valid ? hmul2_u(e2m1x2_to_h2(b), i < 8 ? s0 : s1) : 0u;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        *reinterpret_cast<uint4*>(tile + kmaj_off(row, cc * 32 + j * 8))
            = make_uint4(out[4 * j], out[4 * j + 1], out[4 * j + 2], out[4 * j + 3]);
    }
}

// ---------------- Fused QKV projection: block-scaled MXFP8 on tcgen05 (N=2560, K=6144) ----------------
// out[M, 2560] = mxfp8(x)[M, 6144] @ W^T, the computation of the CUTLASS MXFP8 GEMM: W is the A operand (128 output
// rows per MMA), the quantized activation the B operand (8 token columns), FP32 accumulation in TMEM. A unit is one
// (128-row tile, K split) pair; the 20 x kQkvSplits units run on the last CTAs of the grid. Before the PDL wait a
// CTA streams its unit's weight tiles and scale atoms through a shared-memory ring into TMEM; after it, it quantizes
// its x slice (mxfp8_quantize math), runs the MMAs and stores its FP32 partial rows.
//
// Hand-off without a flag: the partial buffers are double buffered by call parity and hold a sentinel (an FP32 NaN
// the MMA never produces) at rest, so a consumer polls the partials themselves. The producers of call n reset the
// other parity's buffers after their PDL wait; call n - 1, the last reader of those, is complete by then.
constexpr int kQkvN = 2560;
constexpr int kQkvK = 6144;
constexpr int kQkvKTiles = kQkvK / 128;
constexpr int kQkvRowTiles = kQkvN / 128;
#ifndef M3FA_QKV_SPLITS
#define M3FA_QKV_SPLITS 4
#endif
#ifndef M3FA_QKV_RING
#define M3FA_QKV_RING kQkvMaxTiles // every tile of the unit in flight at once
#endif
#ifndef M3FA_QKV_NACC
#define M3FA_QKV_NACC 1
#endif
#ifndef M3FA_QKV_SCALED_MMA
#define M3FA_QKV_SCALED_MMA 1 // 0: unscaled per-block MMAs with the scales applied in the epilogue (see below)
#endif
#ifndef M3FA_QKV_PIPE
// 1: the weight TMA is issued first at entry; the activation has its own region, copied right after the PDL wait;
// each weight tile's MMAs issue as soon as it lands (v52's order). 0: tiles into TMEM first, activation over the ring.
#define M3FA_QKV_PIPE 1
#endif
#ifndef M3FA_QKV_A_SMEM
#define M3FA_QKV_A_SMEM 1 // scaled MMAs only; 1: they read the weight tiles from shared memory (no copy into TMEM)
#endif
// Default: block-scaled kind::mxf8f6f4 MMAs reading A (the weight tiles) in place from the ring, issued per tile as
// it lands (M3FA_QKV_PIPE), one accumulator; M3FA_QKV_A_SMEM=0 copies each tile into TMEM first. Alternative
// (M3FA_QKV_SCALED_MMA=0): one unscaled kind::f8f6f4 MMA per 32-wide K block into its own accumulator, A read in place
// from shared memory, the epilogue applying both UE8M0 scales. On GB300 an M128 x N8 x K32 MMA occupies the tensor pipe
// ~105 cycles block-scaled and ~60 unscaled, independent of N up to 64; the unscaled form's epilogue (320 accumulator
// columns) costs more than it saves.
constexpr bool kQkvScaled = M3FA_QKV_SCALED_MMA != 0;
constexpr bool kQkvASmem = !kQkvScaled || M3FA_QKV_A_SMEM != 0;
constexpr int kQkvSplits = M3FA_QKV_SPLITS;
constexpr int kQkvUnits = kQkvRowTiles * kQkvSplits;
constexpr int kQkvMaxTiles = (kQkvKTiles + kQkvSplits - 1) / kQkvSplits;
constexpr int kQkvRing = M3FA_QKV_RING;
constexpr int kQkvNacc = M3FA_QKV_NACC;
constexpr int kQkvGrid = 112;        // 7 clusters of 16 (or 14 of 8): one wave with the attention
constexpr int kMaxTok = 8;
constexpr int kQkvWTile = 128 * 128; // E4M3 bytes of one 128 x 128 weight tile
constexpr int kQkvSfAtom = 512;      // one 128-row x 4-block UE8M0 scale atom
constexpr uint32_t kQkvSentinel = 0xFFFFFFFFu;
constexpr int64_t kQkvPartStride = static_cast<int64_t>(kQkvSplits) * kMaxTok * kQkvN; // one parity
// TMEM columns of the one 512-column allocation of a CTA: attention [0, 64), then the GEMM's accumulators, scale
// atoms (A per tile, B per tile) and the A tiles (32 columns of 128 x 128 E4M3 each).
constexpr int kTmemAll = 512;
// Megakernel: the attention CTAs run no projection unit, so the GEMM starts at column 0 of a projection CTA.
constexpr int kTAcc = 0;
constexpr int kTSfa = kTAcc + 16;
constexpr int kTSfb = kTSfa + 4 * kQkvMaxTiles;
constexpr int kTA = (kTSfb + 4 * kQkvMaxTiles + 31) / 32 * 32;
static_assert(!kQkvScaled || kTA + 32 * kQkvMaxTiles <= kTmemAll, "TMEM columns");
static_assert(kQkvScaled || kTAcc + 32 * kQkvMaxTiles <= kTmemAll, "TMEM columns (one accumulator per K block)");
static_assert(!kQkvASmem || kQkvRing == kQkvMaxTiles, "shared-memory A needs every tile resident");
static_assert(kQkvUnits <= kQkvGrid, "at most one unit per CTA");
static_assert(kQkvNacc == 1 || kQkvNacc == 2 || kQkvNacc == 4, "accumulator count");
static_assert(!M3FA_QKV_PIPE || kQkvScaled, "the pipelined projection uses the block-scaled MMAs");
constexpr int kQkvOwnRows = 128 / kQkvSplits; // rows of a tile whose split-K sum a CTA owns
// Megakernel o_proj (v52 m3_mx_gemv for [6144, 2048]: split 2, contiguous k-tiles, one accumulator): unit
// o = CTA - 16 (the last 96 CTAs), row tile o / 2, split o % 2.
constexpr int kOpN = 6144;
constexpr int kOpK = 2048;
constexpr int kOpKTiles = kOpK / 128;
constexpr int kOpSplits = 2;
constexpr int kOpTiles = kOpKTiles / kOpSplits;
constexpr int kOpOwnRows = 128 / kOpSplits;
constexpr int kOpFirstCta = kQkvGrid - kOpN / 128 * kOpSplits; // 16
constexpr int kOpSfBytes = kOpKTiles * 512;                    // one parity of the MXFP8 hand-off's scales
#ifndef M3FA_OP_L2PF
#define M3FA_OP_L2PF                                                                                                   \
    0 // 1: QKV CTAs prefetch their o_proj weight slice into L2 before the PDL wait (competes
      // with the QKV weight stream for HBM behind the MoE)
#endif
// o_proj TMEM columns (after the QKV unit's MMAs completed): accumulator, A / B scale atoms, A (8 tiles x 32)
// c1 (attention + o_proj kernel): the attention holds columns [0, 32) while the o_proj weights stream in, so the
// o_proj accumulator and scale atoms sit above them.
constexpr int kOAcc = 32;
constexpr int kOSfa = 48;
constexpr int kOSfb = kOSfa + 4 * kOpTiles;
constexpr int kOA = 128;
static_assert(kOA + 32 * kOpTiles <= kTmemAll && kOSfb + 4 * kOpTiles <= kOA, "o_proj TMEM columns");
constexpr int kQkvMbStride = kMaxTok + 4; // mailbox row pitch (fp32 words)

__host__ __device__ constexpr int qkv_tile0(int s)
{
    return s * kQkvKTiles / kQkvSplits;
}

__device__ __forceinline__ int qkv_unit()
{
    int const u = kQkvGrid - 1 - static_cast<int>(blockIdx.x);
    return u < kQkvUnits ? u : -1;
}

// Attention layout plus, for the fused projection, the weight ring in front (the attention tiles reuse that space
// once the ring has drained into TMEM); the activation tiles, the pre-wait V copy and the barriers sit behind.
template <int QLEN, int QG, int BPC, bool FQKV>
struct SmemX : Smem<QLEN, QG, BPC>
{
    using B = Smem<QLEN, QG, BPC>;
    static constexpr int kRingW = 0;                              // [ring][128 rows][128 B], 128-byte swizzle
    static constexpr int kRingSf = kRingW + kQkvRing * kQkvWTile; // [ring][512 B]
    static constexpr int kRingEnd = kRingSf + kQkvRing * kQkvSfAtom;
    // The activation tiles overlay the ring: written after the PDL wait, once every weight tile went on into TMEM.
    static constexpr int kXt = kRingW;                     // [tile][8][128 B]
    static constexpr int kXsf = kXt + kQkvMaxTiles * 1024; // [tile][512 B]
    static constexpr int kXsc = (kRingEnd + 127) / 128 * 128;
    // Peers push into recv at any time (an empty CTA right away), so it stays clear of the ring.
    static constexpr int kRecv = FQKV ? kXsc : B::kRecv;
    static constexpr int kRawV = FQKV ? (kRecv + kBlocks * QG * BPC * kOPitch * 4 + 127) / 128 * 128 : B::kRawV;
    // Pipelined projection: the activation tiles and scale atoms in their own region (a projection CTA runs no
    // attention, so it overlays the attention's partial-O and V regions), 1024-byte aligned for the swizzle.
    static constexpr int kXA = FQKV ? (kRecv + 1023) / 1024 * 1024 : 0;
    static constexpr int kXASf = kXA + kQkvMaxTiles * 1024;
    static constexpr int kXAEnd = kXASf + kQkvMaxTiles * kQkvSfAtom;
    static constexpr int kMbarA = FQKV ? kRawV + BPC * B::kRawVBlock : B::kMbar;
    static constexpr int kMbar = FQKV && M3FA_QKV_PIPE && kXAEnd > kMbarA ? (kXAEnd + 127) / 128 * 128 : kMbarA;
    static constexpr int kGBar = kMbar + 32; // full[ring], empty[ring], a_ready, acc_done, tile_done[tiles]
    static constexpr int kRedBar = kGBar + 8 * (2 * kQkvRing + 2 + kQkvMaxTiles); // split-K partials in
    static constexpr int kOBar = kRedBar + 8; // o_proj: full[8], a_ready, acc_done, reduce
    static constexpr int kMailbox = (kOBar + 8 * (kOpTiles + 3) + 15) / 16 * 16; // [split][own row][pitch] fp32
    // o_proj weight ring = ring tiles [0, 8); its activation tiles, scale atoms and mailbox behind it.
    static constexpr int kOXt = kRingW + kOpTiles * kQkvWTile;
    static constexpr int kOXsf = kOXt + kOpTiles * 1024;
    static constexpr int kOMailbox = kOXsf + kOpTiles * kQkvSfAtom;
    static_assert(!FQKV || kOMailbox + kOpSplits * kOpOwnRows * kQkvMbStride * 4 <= kRingW + kQkvRing * kQkvWTile,
        "o_proj activation and mailbox fit behind its ring");
    static constexpr int kBytes = FQKV ? kMailbox + kQkvSplits * kQkvOwnRows * kQkvMbStride * 4 : B::kBytes;
    static constexpr int kTmemCols = FQKV ? kTmemAll : B::kTmemCols;
    static constexpr bool kRecvAlias = !FQKV && B::kRecvAlias;
    // The epilogue reads the weight scale atoms while the attention already writes its tiles.
    static_assert(!FQKV || kQkvScaled || B::kRecv <= kRingSf, "attention regions overlap the scale atoms");
    // The other attention regions overlay the ring: a unit's CTA writes them only once the ring has drained.
    static_assert(kBytes <= 227 * 1024, "shared memory budget");
};

// c1: the packed-input attention plus the o_proj unit blockIdx.x (tile bx / 2, split bx % 2). The o_proj weight ring
// and its barriers sit behind the attention layout; the activation tiles, scale atoms and the pair's split-K mailbox
// overlay the K tiles, free once this CTA has pushed its partials (a peer's mailbox push follows that push: its
// activation needs the merges of this cluster, which need this CTA's partials).
template <int QLEN, int QG, int BPC>
struct SmemC1 : SmemX<QLEN, QG, BPC, false>
{
    using B = Smem<QLEN, QG, BPC>;
    static constexpr int kRingW = (B::kBytes + 1023) / 1024 * 1024;
    static constexpr int kRingSf = kRingW + kOpTiles * kQkvWTile;
    static constexpr int kOBar = kRingSf + kOpTiles * kQkvSfAtom; // full[8], a_ready, acc_done, reduce
    static constexpr int kOXt = B::kK16;
    static constexpr int kOXsf = kOXt + kOpTiles * 1024;
    static constexpr int kOMailbox = kOXsf + kOpTiles * kQkvSfAtom;
    static constexpr int kBytes = kOBar + 8 * (kOpTiles + 3);
    static constexpr int kTmemCols = kTmemAll;
    static_assert(!B::kRecvAlias, "c1: the partials have their own region");
    static_assert(
        kOMailbox + kOpSplits * kOpOwnRows * kQkvMbStride * 4 <= B::kV16, "c1: o_proj overlay in the K tiles");
    static_assert(BPC * 2 * B::kRows <= kOAcc, "c1: attention TMEM columns below the o_proj's");
    static_assert(kBytes <= 227 * 1024, "c1: shared memory budget");
};

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void bulk_g2s(void* dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;\n" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(bar), "l"(0x12F0000000000000ull) // EVICT_FIRST: weights are read once per layer
        : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, void const* map, int c0, int c1, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%2, %3}], [%4], %5;\n" ::"r"(dst),
        "l"(map), "r"(c0), "r"(c1), "r"(bar), "l"(0x12F0000000000000ull)
        : "memory");
}

// K-major operand in the 128-byte swizzle: rows of 128 bytes, 8-row groups 1024 bytes apart.
__device__ __forceinline__ uint64_t umma_desc_sw128(uint32_t addr)
{
    return umma_desc(addr, 16, 1024) | (2ull << 61);
}

__device__ __forceinline__ void tmem_cp_128x256b(uint32_t taddr, uint64_t desc)
{
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n" ::"r"(taddr), "l"(desc));
}

__device__ __forceinline__ void tmem_cp_sf(uint32_t taddr, uint32_t saddr)
{
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n" ::"r"(taddr), "l"(umma_desc(saddr, 16, 128)));
}

// Block-scaled MXFP8 instruction descriptor: E4M3 x E4M3, UE8M0 scales, M = 128, N = 8, scale byte kb.
__host__ __device__ constexpr uint32_t qkv_idesc(int kb)
{
    return (static_cast<uint32_t>(kb) << 4) | ((8u >> 3) << 17) | (1u << 23) | ((128u >> 7) << 27)
        | (static_cast<uint32_t>(kb) << 29);
}

__device__ __forceinline__ void umma_mx(
    uint32_t tmem_d, uint32_t tmem_a, uint64_t bdesc, uint32_t idesc, uint32_t sfa, uint32_t sfb, int accumulate)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
        " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], [%1], %2, %3, [%4], [%5], p;\n}\n" ::
            "r"(tmem_d),
        "r"(tmem_a), "l"(bdesc), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate));
}

__device__ __forceinline__ float pow2_from_ue8m0(uint32_t e)
{
    return __uint_as_float(e == 0 ? 0x00400000u : (e << 23));
}

// Unscaled E4M3 x E4M3 MMA, FP32 accumulator: M = 128, N = 8, K = 32, a fresh accumulator.
constexpr uint32_t kQkvIdescF8 = (1u << 4) | ((8u >> 3) << 17) | ((128u >> 4) << 24);

__device__ __forceinline__ void umma_f8_ss(uint32_t tmem_d, uint64_t adesc, uint64_t bdesc)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 0, 0;\n tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n}\n" ::
            "r"(tmem_d),
        "l"(adesc), "l"(bdesc), "r"(kQkvIdescF8));
}

__device__ __forceinline__ void tmem_ld16_nowait(uint32_t addr, float* v)
{
    uint32_t r[16];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, [%16];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(addr));
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

__device__ __forceinline__ void umma_mx_ss(
    uint32_t tmem_d, uint64_t adesc, uint64_t bdesc, uint32_t idesc, uint32_t sfa, uint32_t sfb, int accumulate)
{
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, %6, 0;\n"
        " tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%4], [%5], p;\n}\n" ::
            "r"(tmem_d),
        "l"(adesc), "l"(bdesc), "r"(idesc), "r"(sfa), "r"(sfb), "r"(accumulate));
}

__device__ __forceinline__ bool elect_one()
{
    uint32_t pred;
    asm volatile("{\n .reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0, 1, 0, P;\n}\n" : "=r"(pred));
    return pred != 0;
}

__device__ __forceinline__ int qkv_parity(Params const& p)
{
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.counters) : "memory");
    return v & 1;
}

// Barriers and TMEM (warp 1), then the first ring stages (warp 0). Warps 0 and 1 of every CTA call it.
template <class SX>
__device__ __forceinline__ void qkv_setup(
    Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int warp, int lane, int u)
{
    constexpr bool kPipe = M3FA_QKV_PIPE != 0;
    if (kPipe && warp == 0 && lane == 0 && u >= 0)
    {
        // First thing at entry: this unit's whole weight slice. The issuing thread initializes the ring's full
        // barriers itself; every waiter on them passes bar.sync 8 below, which this thread reaches after the init.
        asm volatile("prefetch.tensormap [%0];\n" ::"l"(&p.wmap) : "memory");
        int const rt = u / kQkvSplits;
        int const kt0 = qkv_tile0(u % kQkvSplits);
        int const nt = qkv_tile0(u % kQkvSplits + 1) - kt0;
        for (int i = 0; i < kQkvRing; ++i)
        {
            mbar_init(gbar + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        for (int i = 0; i < kQkvRing && i < nt; ++i)
        {
            uint32_t const full = gbar + 8 * i;
            mbar_expect_tx(full, kQkvWTile + kQkvSfAtom);
            if (p.wpre != nullptr)
            {
                // Pre-tiled weight: the tile's 16 KB shared-memory image is contiguous (one bulk copy).
                bulk_g2s(smem + SX::kRingW + i * kQkvWTile,
                    p.wpre + (static_cast<int64_t>(rt) * kQkvKTiles + kt0 + i) * kQkvWTile, kQkvWTile, full);
            }
            else
            {
                tma_load_2d(smem_u32(smem + SX::kRingW + i * kQkvWTile), &p.wmap, (kt0 + i) * 128, rt * 128, full);
            }
            bulk_g2s(smem + SX::kRingSf + i * kQkvSfAtom,
                p.wsf + (static_cast<int64_t>(rt) * kQkvKTiles + kt0 + i) * kQkvSfAtom, kQkvSfAtom, full);
        }
        M3FA_PROF_T(34, 0);
    }
    if (warp == 1)
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                     "r"(kTmemAll)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
        M3FA_PROF_T(35, 32);
        if (lane == 0)
        {
            for (int i = (kPipe && u >= 0) ? kQkvRing : 0; i < 2 * kQkvRing + 2 + kQkvMaxTiles; ++i)
            {
                // PIPE 2: tile_done[0] (unused by the scaled MMAs) is act_ready, one arrival per thread outside warp 1
                mbar_init(gbar + 8 * i, M3FA_QKV_PIPE == 2 && i == 2 * kQkvRing + 2 ? kThreads - 32 : 1);
            }
            // The split-K reduce barrier: this CTA's own arrival carries the peers' partial bytes.
            uint32_t const red = gbar + 8 * (2 * kQkvRing + 2 + kQkvMaxTiles);
            mbar_init(red, 1);
            mbar_expect_tx(red, (kQkvSplits - 1) * kQkvOwnRows * kMaxTok * 4);
            uint32_t const ob = smem_u32(smem + SX::kOBar);
            for (int i = 0; i < kOpTiles + 3; ++i)
            {
                mbar_init(ob + 8 * i, 1);
            }
            mbar_expect_tx(ob + 8 * (kOpTiles + 2), (kOpSplits - 1) * kOpOwnRows * kMaxTok * 4);
            asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        }
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    }
    asm volatile("bar.sync 8, 64;\n" ::: "memory");
    if (!kPipe && warp == 0 && lane == 0 && u >= 0)
    {
        int const rt = u / kQkvSplits;
        int const kt0 = qkv_tile0(u % kQkvSplits);
        int const nt = qkv_tile0(u % kQkvSplits + 1) - kt0;
        for (int i = 0; i < kQkvRing && i < nt; ++i)
        {
            uint32_t const full = gbar + 8 * i;
            mbar_expect_tx(full, kQkvWTile + kQkvSfAtom);
            tma_load_2d(smem_u32(smem + SX::kRingW + i * kQkvWTile), &p.wmap, (kt0 + i) * 128, rt * 128, full);
            bulk_g2s(smem + SX::kRingSf + i * kQkvSfAtom,
                p.wsf + (static_cast<int64_t>(rt) * kQkvKTiles + kt0 + i) * kQkvSfAtom, kQkvSfAtom, full);
        }
        M3FA_PROF_T(34, 0);
    }
}

// The rest of the ring (warp 0, lane 0) and the copies into TMEM (warp 1, lane 0); before the PDL wait.
template <class SX>
__device__ __forceinline__ void qkv_stream_lane(
    Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int warp, int u)
{
    int const kt0 = qkv_tile0(u % kQkvSplits);
    int const nt = qkv_tile0(u % kQkvSplits + 1) - kt0;
    uint32_t const full = gbar;
    uint32_t const a_ready = gbar + 16 * kQkvRing;
    if (warp == 0 || !kQkvScaled)
    {
        return; // every tile was issued in qkv_setup; the unscaled MMAs read them in place
    }
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    uint32_t const tm = *tmem_slot;
    // Fully unrolled with compile-time offsets: tcgen05 operands stay in uniform registers (a per-instruction
    // conversion of computed operands costs ~150 cycles per MMA or copy).
    uint64_t const w0 = umma_desc_sw128(smem_u32(smem + SX::kRingW));
    uint32_t const sf0 = smem_u32(smem + SX::kRingSf);
#pragma unroll
    for (int i = 0; i < kQkvMaxTiles; ++i)
    {
        if (i < nt)
        {
            mbar_wait_cta(full + 8 * i, 0);
            if (i == 0)
            {
                M3FA_PROF_T(30, 32);
            }
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
            if constexpr (!kQkvASmem)
            {
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    tmem_cp_128x256b(tm + kTA + (i * 4 + kb) * 8, w0 + ((i * kQkvWTile + kb * 32) >> 4));
                }
            }
            tmem_cp_sf(tm + kTSfa + i * 4, sf0 + i * kQkvSfAtom);
        }
    }
    M3FA_PROF_T(31, 32);
    umma_commit(a_ready);
}

__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity)
{
    uint32_t ready;
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(ready)
        : "r"(bar), "r"(parity)
        : "memory");
    return ready != 0;
}

// PIPE 2, warp 1's elected thread from before the PDL wait to the last MMA. Tiles that land before the activation
// is ready go on into TMEM (A and scale atom), off the critical path; once the activation is in (act_ready), the
// MMAs issue in tile order: the copied tiles with A in TMEM, every later tile as it lands with A read in place from
// the ring. The MMA sequence (and so the accumulation order) is the same either way.
template <class SX>
__device__ __forceinline__ void qkv_hybrid(Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int u)
{
    int const kt0 = qkv_tile0(u % kQkvSplits);
    int const nt = qkv_tile0(u % kQkvSplits + 1) - kt0;
    uint32_t const act_ready = gbar + 8 * (2 * kQkvRing + 2);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    uint32_t const tm = *tmem_slot;
    uint64_t const w0 = umma_desc_sw128(smem_u32(smem + SX::kRingW));
    uint64_t const b0 = umma_desc_sw128(smem_u32(smem + SX::kXA));
    uint32_t const wsf0 = smem_u32(smem + SX::kRingSf);
    uint32_t const xsf0 = smem_u32(smem + SX::kXASf);
    int copied = 0; // tiles [0, copied) are in TMEM
    bool act = false;
#pragma unroll
    for (int i = 0; i < kQkvMaxTiles; ++i)
    {
        if (i < nt && !act)
        {
            while (!mbar_test(gbar + 8 * i, 0))
            {
                if (mbar_test(act_ready, 0))
                {
                    act = true;
                    break;
                }
            }
            if (!act)
            {
                if (i == 0)
                {
                    M3FA_PROF_T(30, 32);
                }
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    tmem_cp_128x256b(tm + kTA + (i * 4 + kb) * 8, w0 + ((i * kQkvWTile + kb * 32) >> 4));
                }
                tmem_cp_sf(tm + kTSfa + i * 4, wsf0 + i * kQkvSfAtom);
                copied = i + 1;
            }
        }
    }
    while (!act)
    {
        act = mbar_test(act_ready, 0);
    }
    M3FA_PROF_T(21, 32);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#pragma unroll
    for (int i = 0; i < kQkvMaxTiles; ++i)
    {
        if (i < nt)
        {
            bool const in_tmem = i < copied;
            if (!in_tmem)
            {
                mbar_wait_epi(gbar + 8 * i, 0);
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                tmem_cp_sf(tm + kTSfa + i * 4, wsf0 + i * kQkvSfAtom);
            }
            uint32_t const sfb = tm + kTSfb + i * 4;
            tmem_cp_sf(sfb, xsf0 + i * kQkvSfAtom);
#pragma unroll
            for (int kb = 0; kb < 4; ++kb)
            {
                int const j = i * 4 + kb;
                uint64_t const bd = b0 + ((i * 1024 + kb * 32) >> 4);
                if (in_tmem)
                {
                    umma_mx(tm + kTAcc, tm + kTA + j * 8, bd, qkv_idesc(kb), tm + kTSfa + i * 4, sfb, j > 0);
                }
                else
                {
                    umma_mx_ss(tm + kTAcc, w0 + ((i * kQkvWTile + kb * 32) >> 4), bd, qkv_idesc(kb), tm + kTSfa + i * 4,
                        sfb, j > 0);
                }
            }
        }
    }
    M3FA_PROF_T(22, 32);
    M3FA_PROF_T(31, 32);
    umma_commit(gbar + 16 * kQkvRing);     // a_ready (for its waiters)
    umma_commit(gbar + 16 * kQkvRing + 8); // acc_done
}

// The other lanes of warps 0 and 1 hold at __syncwarp rather than at griddepcontrol.wait, which would stall the
// whole warp (lane 0's loop included) until the producer grid completes.
template <class SX>
__device__ __forceinline__ void qkv_stream(
    Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int warp, int lane, int u)
{
    if (M3FA_QKV_PIPE == 2 && u >= 0 && warp == 1)
    {
        if (elect_one())
        {
            qkv_hybrid<SX>(p, smem, gbar, tmem_slot, u);
        }
        __syncwarp();
        return;
    }
    if (u < 0 || warp > 1 || M3FA_QKV_PIPE)
    {
        return; // pipelined: the MMA thread takes each tile as it lands
    }
    if (lane == 0)
    {
        qkv_stream_lane<SX>(p, smem, gbar, tmem_slot, warp, u);
    }
    __syncwarp();
}

// After the PDL wait. Every thread quantizes a piece of the x slice into the B tiles (named barrier 9); warp 1 then
// issues the scale copies and the MMAs.
template <class SX>
__device__ __forceinline__ void qkv_mma(
    Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int tid, int u)
{
    int const warp = tid >> 5;
    int const s = u % kQkvSplits;
    int const kt0 = qkv_tile0(s);
    int const nt = qkv_tile0(s + 1) - kt0;
    {
        // The MXFP8 activation slice (E4M3 rows of the unit's k-tiles in 128-byte-swizzled B tiles, and their scale
        // atoms), copied with plain 16-byte loads over the overlaid ring once its tiles went on into TMEM: every load
        // first, then the stores (one L2 round trip).
        constexpr int kXDst = M3FA_QKV_PIPE ? SX::kXA : SX::kXt;
        constexpr int kXSfDst = M3FA_QKV_PIPE ? SX::kXASf : SX::kXsf;
        if (!M3FA_QKV_PIPE)
        {
            mbar_wait_cta(gbar + 16 * kQkvRing, 0);         // a_ready: the ring's weight tiles are in TMEM
        }
        constexpr int kXItems = kQkvMaxTiles * 8 * kMaxTok; // (tile, 16-byte chunk, token)
        constexpr int kSfItems = kQkvMaxTiles * (kQkvSfAtom / 16);
        constexpr bool kHyb = M3FA_QKV_PIPE == 2;
        constexpr int kCopiers = kHyb ? kThreads - 32 : kThreads; // PIPE 2: warp 1 is issuing the tiles
        if (kHyb && warp == 1)
        {
            return;
        }
        int const ctid = kHyb && tid >= 64 ? tid - 32 : tid;
        constexpr int kPer = (kXItems + kSfItems + kCopiers - 1) / kCopiers;
        uint4 v[kPer];
        int dst[kPer];
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const q = ctid + kCopiers * k;
            dst[k] = -1;
            if (q < kXItems)
            {
                int const t = q % kMaxTok;
                int const c = (q / kMaxTok) % 8;
                int const i = q / (kMaxTok * 8);
                if (i < nt && t < p.num_tokens)
                {
                    v[k] = *reinterpret_cast<uint4 const*>(
                        p.xq + static_cast<int64_t>(t) * p.ldxq + (kt0 + i) * 128 + c * 16);
                    dst[k] = kXDst + i * 1024 + t * 128 + ((c ^ t) * 16);
                }
            }
            else if (q < kXItems + kSfItems)
            {
                int const j = q - kXItems;
                int const i = j / (kQkvSfAtom / 16);
                int const c = j % (kQkvSfAtom / 16);
                if (i < nt)
                {
                    v[k] = *reinterpret_cast<uint4 const*>(
                        p.xsf_act + static_cast<int64_t>(kt0 + i) * kQkvSfAtom + c * 16);
                    dst[k] = kXSfDst + i * kQkvSfAtom + c * 16;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            if (dst[k] >= 0)
            {
                *reinterpret_cast<uint4*>(smem + dst[k]) = v[k];
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if (kHyb)
        {
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(gbar + 8 * (2 * kQkvRing + 2))
                         : "memory");
            return;
        }
        if (warp != 1)
        {
            asm volatile("bar.arrive 9, %0;\n" ::"n"(kThreads) : "memory");
        }
    }
    if (warp == 1)
    {
        M3FA_PROF_T(32, 32);
        asm volatile("bar.sync 9, %0;\n" ::"n"(kThreads) : "memory");
        M3FA_PROF_T(21, 32);
        // Operands computed by the whole warp (warp-uniform, kept in uniform registers); one elected lane issues.
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        uint32_t const tm = *tmem_slot;
        uint64_t const b0 = umma_desc_sw128(smem_u32(smem + (M3FA_QKV_PIPE ? SX::kXA : SX::kXt)));
        uint64_t const a0 = umma_desc_sw128(smem_u32(smem + SX::kRingW));
        if (elect_one())
        {
            M3FA_CLK(41);
            if constexpr (!kQkvScaled)
            {
#pragma unroll
                for (int i = 0; i < kQkvMaxTiles; ++i)
                {
                    if (i < nt)
                    {
                        mbar_wait_cta(gbar + 8 * i, 0); // weight tile i landed
                        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#pragma unroll
                        for (int kb = 0; kb < 4; ++kb)
                        {
                            umma_f8_ss(tm + kTAcc + (i * 4 + kb) * 8, a0 + ((i * kQkvWTile + kb * 32) >> 4),
                                b0 + ((i * 1024 + kb * 32) >> 4));
                        }
                        umma_commit(gbar + 8 * (2 * kQkvRing + 2 + i)); // tile i's blocks are in TMEM
                    }
                }
            }
            else
            {
                uint32_t const xsf0 = smem_u32(smem + (M3FA_QKV_PIPE ? SX::kXASf : SX::kXsf));
                uint32_t const wsf0 = smem_u32(smem + SX::kRingSf);
#pragma unroll
                for (int i = 0; i < kQkvMaxTiles; ++i)
                {
                    if (i < nt)
                    {
                        if constexpr (M3FA_QKV_PIPE != 0)
                        {
                            // Tile i as it lands: its A copy and scale atom into TMEM, then its MMAs (issue order).
                            mbar_wait_cta(gbar + 8 * i, 0);
                            if (i == 0)
                            {
                                M3FA_PROF_T(30, 32);
                            }
                            if (i == nt - 1)
                            {
                                M3FA_CLK(47); // the unit's last weight tile landed (clock64 builds)
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                            if constexpr (!kQkvASmem)
                            {
#pragma unroll
                                for (int kb = 0; kb < 4; ++kb)
                                {
                                    tmem_cp_128x256b(
                                        tm + kTA + (i * 4 + kb) * 8, a0 + ((i * kQkvWTile + kb * 32) >> 4));
                                }
                            }
                            tmem_cp_sf(tm + kTSfa + i * 4, wsf0 + i * kQkvSfAtom);
                        }
                        uint32_t const sfb = tm + kTSfb + i * 4;
                        tmem_cp_sf(sfb, xsf0 + i * kQkvSfAtom);
#pragma unroll
                        for (int kb = 0; kb < 4; ++kb)
                        {
                            int const j = i * 4 + kb;
                            uint64_t const bd = b0 + ((i * 1024 + kb * 32) >> 4);
                            if constexpr (kQkvASmem)
                            {
                                umma_mx_ss(tm + kTAcc + (j % kQkvNacc) * 8, a0 + ((i * kQkvWTile + kb * 32) >> 4), bd,
                                    qkv_idesc(kb), tm + kTSfa + i * 4, sfb, j >= kQkvNacc);
                            }
                            else
                            {
                                umma_mx(tm + kTAcc + (j % kQkvNacc) * 8, tm + kTA + j * 8, bd, qkv_idesc(kb),
                                    tm + kTSfa + i * 4, sfb, j >= kQkvNacc);
                            }
                        }
                    }
                }
            }
            M3FA_CLK(42);
            if (M3FA_QKV_PIPE)
            {
                M3FA_PROF_T(31, 32);
                umma_commit(gbar + 16 * kQkvRing); // a_ready (kept for its waiters: every tile is consumed)
            }
            umma_commit(gbar + 16 * kQkvRing + 8); // acc_done
        }
        M3FA_PROF_T(22, 32);
    }
}

// Warps 16..19 (TMEM lanes 32 (warp % 4) + lane = the unit's rows): partial rows out, the other parity reset.
template <class SX>
__device__ __forceinline__ void qkv_epilogue(
    Params const& p, uint8_t* smem, uint32_t gbar, uint32_t* tmem_slot, int tid, int u, int par, unsigned int calls)
{
    int const warp = tid >> 5;
    int const lane = tid & 31;
    if constexpr (kQkvScaled)
    {
        mbar_wait_epi(gbar + 16 * kQkvRing + 8, 0);
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    }
    M3FA_PROF_T(24, 512);
    if (tid == 512)
    {
        M3FA_CLK(43);
    }
    int const r = (warp & 3) * 32 + lane; // row within the unit's tile
    uint32_t const taddr = *tmem_slot + (((warp & 3) * 32) << 16) + kTAcc;
    float val[kMaxTok];
    if constexpr (kQkvScaled)
    {
        float v[32];
        tmem_ld16(taddr, *reinterpret_cast<float(*)[16]>(v));
        if constexpr (kQkvNacc == 4)
        {
            tmem_ld16(taddr + 16, *reinterpret_cast<float(*)[16]>(v + 16));
        }
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
        {
            val[t] = kQkvNacc == 4 ? (v[t] + v[8 + t]) + (v[16 + t] + v[24 + t])
                : kQkvNacc == 2    ? v[t] + v[8 + t]
                                   : v[t];
        }
    }
    else
    {
        // Sum over K blocks in order: block result x (weight scale x activation scale).
        int const nt = qkv_tile0(u % kQkvSplits + 1) - qkv_tile0(u % kQkvSplits);
        float const* xsc = reinterpret_cast<float const*>(smem + SX::kXsc);
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
        {
            val[t] = 0.0f;
        }
#pragma unroll
        for (int i = 0; i < kQkvMaxTiles; ++i)
        {
            if (i < nt)
            {
                // Tile i's four blocks as soon as its MMAs complete: the epilogue trails the MMAs by one tile.
                mbar_wait_cta(gbar + 8 * (2 * kQkvRing + 2 + i), 0);
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                float d[32];
                tmem_ld16_nowait(taddr + i * 32, d);
                tmem_ld16_nowait(taddr + i * 32 + 16, d + 16);
                uint32_t const wsw = *reinterpret_cast<uint32_t const*>(
                    smem + SX::kRingSf + i * kQkvSfAtom + (r % 32) * 16 + (r / 32) * 4);
                asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                {
                    float const ws = pow2_from_ue8m0((wsw >> (8 * kb)) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < kMaxTok; ++t)
                    {
                        val[t] = fmaf(d[kb * 8 + t], ws * xsc[(i * 4 + kb) * kMaxTok + t], val[t]);
                    }
                }
            }
        }
    }
    // v52's split-K reduce: split s owns rows [32 s, 32 s + 32) of the tile; the other splits push their fp32 partials
    // of those rows into its mailbox (st.async, completing its reduce barrier's bytes); the owner adds the splits in
    // rank order from 0.0 and stores bf16.
    int const s = u % kQkvSplits;
    int const owner = r / kQkvOwnRows;
    int const local = r % kQkvOwnRows;
    uint32_t const red = gbar + 8 * (2 * kQkvRing + 2 + kQkvMaxTiles);
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    // Units are numbered down from the last CTA: split q of this tile is CTA rank cta_rank - (q - s).
    if (owner != s)
    {
        uint32_t const peer = cta_rank + s - owner;
        uint32_t const dst
            = mapa_shared(smem_u32(smem + SX::kMailbox) + ((s * kQkvOwnRows + local) * kQkvMbStride) * 4, peer);
        uint32_t const pbar = mapa_shared(red, peer);
#pragma unroll
        for (int g = 0; g < kMaxTok / 4; ++g)
        {
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(
                    dst + 16 * g),
                "r"(__float_as_uint(val[4 * g])), "r"(__float_as_uint(val[4 * g + 1])),
                "r"(__float_as_uint(val[4 * g + 2])), "r"(__float_as_uint(val[4 * g + 3])), "r"(pbar)
                : "memory");
        }
    }
    else
    {
        M3FA_PROF_T(26, tid - lane);
        // test_wait (no suspend): completions by peers' st.async bytes wake a suspended try_wait late.
        uint32_t ready = 0;
        while (!ready)
        {
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(ready)
                : "r"(red)
                : "memory");
        }
        M3FA_PROF_T(27, tid - lane);
        float const* mb = reinterpret_cast<float const*>(smem + SX::kMailbox);
        int const row = (u / kQkvSplits) * 128 + r;
        // Lamport hand-off: this call's rows go to buffer (calls & 1) with relaxed stores, the consumers poll the
        // values themselves (0xFFFF, a NaN the rounding of a finite sum never produces, marks "not yet"); the other
        // buffer, last read by the previous call (complete: this grid passed its PDL wait), is reset to the marker.
        uint16_t* cur = reinterpret_cast<uint16_t*>(p.qkv_out) + (calls & 1u) * kMaxTok * kQkvN;
        uint16_t* nxt = reinterpret_cast<uint16_t*>(p.qkv_out) + ((calls & 1u) ^ 1u) * kMaxTok * kQkvN;
#pragma unroll
        float qsum[kMaxTok];
#pragma unroll
        for (int g = 0; g < kMaxTok / 4; ++g)
        {
            float4 part[kQkvSplits];
#pragma unroll
            for (int q = 0; q < kQkvSplits; ++q)
            {
                part[q] = *reinterpret_cast<float4 const*>(mb + (q * kQkvOwnRows + local) * kQkvMbStride + 4 * g);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float sum = 0.0F;
#pragma unroll
                for (int q = 0; q < kQkvSplits; ++q)
                {
                    float const pj = j == 0 ? part[q].x : j == 1 ? part[q].y : j == 2 ? part[q].z : part[q].w;
                    sum = sum + (q == s ? val[4 * g + j] : pj);
                }
                qsum[4 * g + j] = sum;
            }
        }
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
        {
            float const sum = qsum[t];
            if (t < p.num_tokens)
            {
                __nv_bfloat16 const b = __float2bfloat16_rn(sum);
                uint16_t bits = *reinterpret_cast<uint16_t const*>(&b);
                // Never the "not yet" marker: a 0xFFFF NaN sum is stored as the NaN 0x7FFF. Only a NaN projection row
                // changes (its K/V cache bytes then differ from v52 + p6 in the NaN's encoding); finite rows are exact.
                bits = bits == 0xFFFFu ? static_cast<uint16_t>(0x7FFFu) : bits;
#ifndef M3FA_QKV_REL
#define M3FA_QKV_REL 0 // 1: the row store is a release (tried: whether it becomes visible to the pollers sooner)
#endif
                if (M3FA_QKV_REL)
                {
                    asm volatile(
                        "st.release.gpu.global.u16 [%0], %1;\n" ::"l"(cur + static_cast<int64_t>(t) * kQkvN + row),
                        "h"(bits)
                        : "memory");
                }
                else
                {
                    asm volatile(
                        "st.relaxed.gpu.global.u16 [%0], %1;\n" ::"l"(cur + static_cast<int64_t>(t) * kQkvN + row),
                        "h"(bits)
                        : "memory");
                }
            }
            asm volatile("st.relaxed.gpu.global.u16 [%0], %1;\n" ::"l"(nxt + static_cast<int64_t>(t) * kQkvN + row),
                         "h"(static_cast<uint16_t>(0xFFFFu))
                         : "memory");
        }
        M3FA_PROF_T(28, tid - lane);
    }
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    M3FA_PROF_T(25, 512);
    if (tid == 512)
    {
        M3FA_CLK(46);
    }
}

// ---------------- Megakernel o_proj (v52 m3_mx_gemv bit for bit: split 2, contiguous k-tiles, nacc 1) ----------------
// The unit's weight tiles and scale atoms into the ring (warp 0, lane 0). From L2 when the CTA prefetched them there.
template <class SX>
__device__ __forceinline__ void op_issue(Params const& p, uint8_t* smem, int ou)
{
    int const rt = ou / kOpSplits;
    int const kt0 = (ou % kOpSplits) * kOpTiles;
    uint32_t const ob = smem_u32(smem + SX::kOBar);
    for (int i = 0; i < kOpTiles; ++i)
    {
        mbar_expect_tx(ob + 8 * i, kQkvWTile + kQkvSfAtom);
        tma_load_2d(smem_u32(smem + SX::kRingW + i * kQkvWTile), &p.wmap_o, (kt0 + i) * 128, rt * 128, ob + 8 * i);
        bulk_g2s(smem + SX::kRingSf + i * kQkvSfAtom,
            p.wsf_o + (static_cast<int64_t>(rt) * kOpKTiles + kt0 + i) * kQkvSfAtom, kQkvSfAtom, ob + 8 * i);
    }
}

// L2 prefetch of the unit's weight tiles and scale atoms (one lane; nothing waits for it).
// The QKV units' share of the window-1 L2 prefetch (the next consumer's weights), issued once their own QKV weight
// stream has landed (acc_done): the attention core reads little from HBM.
__device__ __forceinline__ void window_prefetch_l2(Params const& p, int u)
{
    constexpr long long kChunk = 32768;
    long long const n0 = (p.pf_b0 + kChunk - 1) / kChunk, n1 = (p.pf_b1 + kChunk - 1) / kChunk;
    uint64_t pol = 0;
    if (p.pf_hint)
    {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    }
    for (long long c = u; c < n0 + n1; c += kQkvUnits)
    {
        long long const a = c < n0 ? p.pf_a0 + c * kChunk : p.pf_a1 + (c - n0) * kChunk;
        long long const rem = c < n0 ? p.pf_b0 - c * kChunk : p.pf_b1 - (c - n0) * kChunk;
        unsigned int const n = static_cast<unsigned int>(rem < kChunk ? rem : kChunk) & ~15u;
        if (n == 0)
        {
            continue;
        }
        if (p.pf_hint)
        {
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;\n" ::"l"(a), "r"(n), "l"(pol)
                         : "memory");
        }
        else
        {
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a), "r"(n) : "memory");
        }
    }
}

__device__ __forceinline__ void op_prefetch_l2(Params const& p, int ou)
{
    int const rt = ou / kOpSplits;
    int const kt0 = (ou % kOpSplits) * kOpTiles;
    for (int i = 0; i < kOpTiles; ++i)
    {
        asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];\n" ::"l"(&p.wmap_o),
                     "r"((kt0 + i) * 128), "r"(rt * 128)
                     : "memory");
    }
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(
                     p.wsf_o + (static_cast<int64_t>(rt) * kOpKTiles + kt0) * kQkvSfAtom),
                 "r"(kOpTiles * kQkvSfAtom)
                 : "memory");
}

// The weight tiles into TMEM as they land (warp 1, elected lane, operands warp-uniform).
template <class SX>
__device__ __forceinline__ void op_stream(uint8_t* smem, uint32_t tm)
{
    uint32_t const ob = smem_u32(smem + SX::kOBar);
    uint64_t const w0 = umma_desc_sw128(smem_u32(smem + SX::kRingW));
    uint32_t const sf0 = smem_u32(smem + SX::kRingSf);
#pragma unroll
    for (int i = 0; i < kOpTiles; ++i)
    {
        mbar_wait_cta(ob + 8 * i, 0);
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
        {
            tmem_cp_128x256b(tm + kOA + (i * 4 + kb) * 8, w0 + ((i * kQkvWTile + kb * 32) >> 4));
        }
        tmem_cp_sf(tm + kOSfa + i * 4, sf0 + i * kQkvSfAtom);
    }
    umma_commit(ob + 8 * kOpTiles); // a_ready
}

// The attention's MXFP8 rows of this unit's k-tiles (the heads [kt0, kt0 + 8)), Lamport-polled value by value: 0xFF
// is NaN in both E4M3 and UE8M0, which the merge never writes (see the merge). Every thread of the CTA, one item each.
template <class SX>
__device__ __forceinline__ void op_act(Params const& p, uint8_t* smem, int ou, int tid, unsigned int calls)
{
    int const kt0 = (ou % kOpSplits) * kOpTiles;
    uint8_t const* q8 = p.q8 + (calls & 1u) * kMaxTok * kOpK;
    uint8_t const* sf = p.sf + (calls & 1u) * kOpSfBytes;
    constexpr int kXItems = kOpTiles * 8 * kMaxTok;
    constexpr int kSfItems = kOpTiles * (kQkvSfAtom / 16);
    for (int q = tid; q < kXItems + kSfItems; q += kThreads)
    {
        if (q < kXItems)
        {
            int const t = q % kMaxTok;
            int const c = (q / kMaxTok) % 8;
            int const i = q / (kMaxTok * 8);
            if (t < p.num_tokens)
            {
                uint4 v;
                bool pending;
                do
                {
                    asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "l"(q8 + static_cast<int64_t>(t) * kOpK + (kt0 + i) * 128 + c * 16)
                                 : "memory");
                    uint32_t const w[4] = {v.x, v.y, v.z, v.w};
                    pending = false;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        // a byte equal to 0xFF: (x & 0x7F7F7F7F) + 0x01010101 sets bit 7 of exactly the 0x7F/0xFF bytes
                        uint32_t const hit = (((w[j] & 0x7F7F7F7Fu) + 0x01010101u) & w[j] & 0x80808080u);
                        pending |= hit != 0u;
                    }
                } while (pending);
                *reinterpret_cast<uint4*>(smem + SX::kOXt + i * 1024 + t * 128 + ((c ^ t) * 16)) = v;
            }
        }
        else if (q < kXItems + kSfItems)
        {
            int const j = q - kXItems;
            int const i = j / (kQkvSfAtom / 16);
            int const c = j % (kQkvSfAtom / 16);
            uint8_t const* src = sf + (kt0 + i) * kQkvSfAtom + c * 16;
            uint4 v;
            if (c < p.num_tokens)
            {
                // Row c of the atom: its 4 block scales in bytes [16 c, 16 c + 4).
                do
                {
                    asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "l"(src)
                                 : "memory");
                } while ((((v.x & 0x7F7F7F7Fu) + 0x01010101u) & v.x & 0x80808080u) != 0u);
            }
            else
            {
                v = *reinterpret_cast<uint4 const*>(src);
            }
            *reinterpret_cast<uint4*>(smem + SX::kOXsf + i * kQkvSfAtom + c * 16) = v;
        }
    }
}

// The MMAs (warp 1, elected lane): v52's order (per k-tile: B scale copy, 4 MMAs; one accumulator).
template <class SX>
__device__ __forceinline__ void op_mma(uint8_t* smem, uint32_t tm)
{
    uint32_t const ob = smem_u32(smem + SX::kOBar);
    mbar_wait_cta(ob + 8 * kOpTiles, 0); // a_ready
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    uint64_t const b0 = umma_desc_sw128(smem_u32(smem + SX::kOXt));
    uint32_t const xsf0 = smem_u32(smem + SX::kOXsf);
#pragma unroll
    for (int i = 0; i < kOpTiles; ++i)
    {
        uint32_t const sfb = tm + kOSfb + i * 4;
        tmem_cp_sf(sfb, xsf0 + i * kQkvSfAtom);
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
        {
            int const j = i * 4 + kb;
            umma_mx(tm + kOAcc, tm + kOA + j * 8, b0 + ((i * 1024 + kb * 32) >> 4), qkv_idesc(kb), tm + kOSfa + i * 4,
                sfb, j > 0);
        }
    }
    umma_commit(ob + 8 * (kOpTiles + 1)); // acc_done
}

// Warps 16..19: the pair's split-K reduce (owner rows [64 s, 64 s + 64)) and the bf16 rows of y.
template <class SX>
__device__ __forceinline__ void op_epilogue(Params const& p, uint8_t* smem, uint32_t tm, int tid, int ou)
{
    int const warp = tid >> 5;
    int const lane = tid & 31;
    uint32_t const ob = smem_u32(smem + SX::kOBar);
    // The store's kernel parameters, fetched before the wait: a first touch of their constant-bank line costs a miss.
    __nv_bfloat16* y_out = p.y_out;
    int num_tokens = p.num_tokens;
    int ar_mode = p.ar_mode;
    asm volatile("" : "+l"(y_out), "+r"(num_tokens), "+r"(ar_mode));
#ifndef M3FA_OP_PARPRE
#define M3FA_OP_PARPRE 1 // read the all-reduce slot parities before the accumulator wait (a global load off the tail)
#endif
    // The parity flags are stable here: the all-reduce that flips them last completed before this grid's PDL wait,
    // and the next one reads them only after this kernel's pushes.
    int par_pre[kMaxTok];
    {
        int const row_pre = (ou / kOpSplits) * 128 + ((tid >> 5) & 3) * 32 + (tid & 31);
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
        {
            par_pre[t] = 0;
            if (M3FA_OP_PARPRE && ar_mode != 0 && t < num_tokens)
            {
                asm volatile("ld.volatile.global.s32 %0, [%1];\n"
                             : "=r"(par_pre[t])
                             : "l"(p.ar_flags + t * 8 + row_pre / (kOpN / 8))
                             : "memory");
            }
        }
    }
    mbar_wait_epi(ob + 8 * (kOpTiles + 1), 0);
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    M3FA_PROF_T(51, kComputeThreads);
    int const r = (warp & 3) * 32 + lane;
    float v[16];
    tmem_ld16(tm + (((warp & 3) * 32) << 16) + kOAcc, v);
    int const s = ou % kOpSplits;
    int const owner = r / kOpOwnRows;
    int const local = r % kOpOwnRows;
    uint32_t const red = ob + 8 * (kOpTiles + 2);
    uint32_t cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (owner != s)
    {
        uint32_t const peer = cta_rank + owner - s; // o_proj units count up with the CTA index
        uint32_t const dst
            = mapa_shared(smem_u32(smem + SX::kOMailbox) + ((s * kOpOwnRows + local) * kQkvMbStride) * 4, peer);
        uint32_t const pbar = mapa_shared(red, peer);
#pragma unroll
        for (int g = 0; g < kMaxTok / 4; ++g)
        {
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(
                    dst + 16 * g),
                "r"(__float_as_uint(v[4 * g])), "r"(__float_as_uint(v[4 * g + 1])), "r"(__float_as_uint(v[4 * g + 2])),
                "r"(__float_as_uint(v[4 * g + 3])), "r"(pbar)
                : "memory");
        }
    }
    else
    {
        uint32_t ready = 0;
        while (!ready)
        {
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(ready)
                : "r"(red)
                : "memory");
        }
        M3FA_PROF_T(52, tid - lane);
        float const* mb = reinterpret_cast<float const*>(smem + SX::kOMailbox);
        int const row = (ou / kOpSplits) * 128 + r;
        float sums[kMaxTok];
#pragma unroll
        for (int g = 0; g < kMaxTok / 4; ++g)
        {
            float4 part[kOpSplits];
#pragma unroll
            for (int q = 0; q < kOpSplits; ++q)
            {
                part[q] = *reinterpret_cast<float4 const*>(mb + (q * kOpOwnRows + local) * kQkvMbStride + 4 * g);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float sum = 0.0F;
#pragma unroll
                for (int q = 0; q < kOpSplits; ++q)
                {
                    float const pj = j == 0 ? part[q].x : j == 1 ? part[q].y : j == 2 ? part[q].z : part[q].w;
                    sum = sum + (q == s ? v[4 * g + j] : pj);
                }
                sums[4 * g + j] = sum;
            }
        }
        M3FA_PROF_T(54, tid - lane);
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
        {
            float const sum = sums[t];
#ifndef M3FA_EXP
#define M3FA_EXP 0
#endif
            if (t < num_tokens)
            {
                if (M3FA_EXP == 1)
                {
                    asm volatile(
                        "st.relaxed.gpu.global.u16 [%0], %1;\n" ::"l"(y_out + static_cast<int64_t>(t) * kOpN + row),
                        "h"(__bfloat16_as_ushort(__float2bfloat16_rn(sum)))
                        : "memory");
                }
                else
                {
                    y_out[static_cast<int64_t>(t) * kOpN + row] = __float2bfloat16_rn(sum);
                }
            }
            if (ar_mode != 0)
            {
                // m3_mx_gemv's push into the all-reduce exchange: lane pairs hold row pairs; the even lane writes the
                // bf16x2 word into this rank's slot of the (token, column slice) parity the AR's flags select.
                float const hi = __shfl_xor_sync(0xffffffffu, sum, 1);
                if ((lane & 1) == 0 && t < num_tokens)
                {
                    int par;
                    if (M3FA_OP_PARPRE)
                    {
                        par = par_pre[t];
                    }
                    else
                    {
                        asm volatile("ld.volatile.global.s32 %0, [%1];\n"
                                     : "=r"(par)
                                     : "l"(p.ar_flags + t * 8 + row / (kOpN / 8))
                                     : "memory");
                    }
                    int64_t const off = static_cast<int64_t>(par & 1) * p.ar_maxt * p.ar_world * (kOpN / 2)
                        + (static_cast<int64_t>(t) * p.ar_world + p.ar_rank) * (kOpN / 2) + row / 2;
                    __nv_bfloat162 w2 = __floats2bfloat162_rn(sum, hi);
                    uint32_t word = *reinterpret_cast<uint32_t*>(&w2);
                    // -0.0 halves to +0.0: 0x80000000 marks an empty exchange word (as m3_mx_gemv's _bf16x2_word)
                    word = (word & 0xFFFFu) == 0x8000u ? (word & 0xFFFF0000u) : word;
                    word = (word >> 16) == 0x8000u ? (word & 0x0000FFFFu) : word;
                    asm volatile("st.global.b32 [%0], %1;\n" ::"l"(p.ar_ws + off), "r"(word) : "memory");
                }
            }
        }
        M3FA_PROF_T(53, tid - lane);
    }
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}

// Packed-QKV element group [token][col * 128 + 4 * lane, +4): the split partials, polled until every one is stored,
// summed in split order and rounded to BF16.
__device__ __forceinline__ void qkv_load(Params const& p, int par, int tok, int col, int lane, float (&e)[4])
{
    float const* base = p.part + par * kQkvPartStride + static_cast<int64_t>(tok) * kQkvN + col * kHeadDim + lane * 4;
    uint4 pr[kQkvSplits];
    long long spins = 0;
    bool ready;
    do
    {
        ready = true;
#pragma unroll
        for (int s = 0; s < kQkvSplits; ++s)
        {
            asm volatile("ld.relaxed.gpu.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(pr[s].x), "=r"(pr[s].y), "=r"(pr[s].z), "=r"(pr[s].w)
                         : "l"(base + static_cast<int64_t>(s) * kMaxTok * kQkvN)
                         : "memory");
            ready &= pr[s].x != kQkvSentinel && pr[s].y != kQkvSentinel && pr[s].z != kQkvSentinel
                && pr[s].w != kQkvSentinel;
        }
        if (++spins > (1LL << 26))
        {
            __trap();
        }
    } while (!ready);
    float s0 = __uint_as_float(pr[0].x), s1 = __uint_as_float(pr[0].y), s2 = __uint_as_float(pr[0].z),
          s3 = __uint_as_float(pr[0].w);
#pragma unroll
    for (int s = 1; s < kQkvSplits; ++s)
    {
        s0 += __uint_as_float(pr[s].x);
        s1 += __uint_as_float(pr[s].y);
        s2 += __uint_as_float(pr[s].z);
        s3 += __uint_as_float(pr[s].w);
    }
    e[0] = __bfloat162float(__float2bfloat16_rn(s0));
    e[1] = __bfloat162float(__float2bfloat16_rn(s1));
    e[2] = __bfloat162float(__float2bfloat16_rn(s2));
    e[3] = __bfloat162float(__float2bfloat16_rn(s3));
}

// Every CTA once it has read the call parity and the partials it needs: the last one flips the parity.
__device__ __forceinline__ void qkv_done(Params const& p)
{
    if (atomicAdd(p.counters + 1, 1) == kQkvGrid - 1)
    {
        atomicExch(p.counters + 1, 0);
        atomicXor(p.counters, 1);
    }
}

} // namespace

namespace
{

// FQKV: the kernel also computes the QKV(+index) projection from the layer input (see qkv_mma) and runs on a fixed
// kQkvGrid-CTA grid; the first attn_clusters clusters then run the attention on the projection's output.
#ifndef M3FA_C1_OPISSUE
#define M3FA_C1_OPISSUE                                                                                                \
    4 // c1: o_proj weight TMA issued 0: at entry (before the PDL wait), 1: right after the PDL
      // wait, 2: once the attention's K/V reads landed, 3: after the merge, 4: before the push, 5: after it
#endif
#ifndef M3FA_C1_TRIGGER
#define M3FA_C1_TRIGGER 3 // c1: dependents released 2: at the attention's usual point, 3: after the o_proj pushes
#endif
template <int QLEN, int QG, int BPC, bool MXFP8, bool FQKV, bool OPJ = false>
__global__ void __launch_bounds__(kThreads, 1) m3_fused_decode_attn_kernel(__grid_constant__ Params const p)
{
#if MINIMAX_M3_ATTN_DEVICE_ENABLED
    static_assert(!OPJ || (!FQKV && MXFP8), "c1: packed input, MXFP8 hand-off");
    using S = std::conditional_t<OPJ, SmemC1<QLEN, QG, BPC>, SmemX<QLEN, QG, BPC, FQKV>>;
    constexpr bool kAttnTrigger = !FQKV && !(OPJ && M3FA_C1_TRIGGER == 3);
    constexpr int kRows = S::kRows;
    constexpr int kTok = S::kTok;
    constexpr int kCs = kBlocks / BPC;                  // cluster size
    constexpr int kGroups = QLEN / QG;                  // clusters per request
    constexpr uint32_t kPushBytes = (kHeadDim + 1) * 4; // one partial row: 128 outputs and the LSE
    constexpr int kTasks = kHq * QG + 3 * QLEN;         // producer tasks: 16 Q heads x QG, [K, V, index-K] x QLEN
    constexpr int kMaxT = (kTasks + kWarps - 1) / kWarps;
    constexpr int kChunks = BPC * kPage * 4;            // 16-byte packed chunks per K (or V) tile set
    constexpr int kKPer = (kChunks + kComputeThreads - 1) / kComputeThreads; // K chunks per thread [0, 512)
    constexpr bool kRegCore = RegCore<QG>::kOn;                              // warp-MMA attention core (see RegCore)
    // V tiles (raw copy before the PDL wait, dequantized during QK and the softmax): on the packed path every warp
    // outside the softmax groups takes part (at M=1, warps 4..19), on the fused-projection path the producer warps.
#ifndef M3FA_RS2
#define M3FA_RS2 1 // megakernel at one 16-row chunk: the softmax and push rows split over two warp groups (8 each)
#endif
    // Split softmax: the megakernel and multi-row (spec verify) steps; at one-row decode on the packed path it
    // measured slower (it takes four warps from the V dequantization there).
    constexpr bool kRs2 = M3FA_RS2 && BPC * QG == 1 && !RegCore<QG>::kOn && (FQKV || QLEN > 1);
#ifndef M3FA_MEGA_WARPMMA
#define M3FA_MEGA_WARPMMA 1
#endif
    // Warp-issued MMAs and the early cluster barrier: measured gains in the megakernel and on multi-row / paired-block
    // steps, but not at one-row decode on 16-CTA clusters (the packed path's (1, 1, 1)), where they stay off.
    constexpr bool kTuned = FQKV || QLEN > 1 || BPC > 1;
    constexpr bool kWarpMma = M3FA_MEGA_WARPMMA && kTuned;
    constexpr bool kEarlyClw = M3FA_MEGA_EARLYCLW && kTuned;
    constexpr int kSoftmaxGroups = kRs2 ? 2 : (BPC * QG) < 4 ? (BPC * QG) : 4;
#ifndef M3FA_MEGA_VWIDE
#define M3FA_MEGA_VWIDE                                                                                                \
    1 // megakernel attention CTAs (no projection unit): every warp outside the softmax copies and
      // dequantizes V, as on the packed path (not only the four producer warps)
#endif
    constexpr int kVStart = (kRegCore || (FQKV && !M3FA_MEGA_VWIDE)) ? kComputeThreads : 128 * kSoftmaxGroups;
    constexpr int kVThreads = kThreads - kVStart;    // V chunks: the warps outside the softmax groups
    constexpr int kVPer = (kChunks + kVThreads - 1) / kVThreads;
    constexpr int kPairs = BPC * QG;                 // (block, 16-row chunk) pairs of the softmax and push
    constexpr int kPairsPerGroup = (kPairs + 3) / 4; // per group of four compute warps
    // Named barrier 6: the softmax warps (P written) and the producer warps (V written) before the PV issue.
    constexpr int kPvBarThreads = 128 * kSoftmaxGroups + kVThreads;
    extern __shared__ __align__(128) uint8_t smem[];
    float* row_scale = reinterpret_cast<float*>(smem + S::kRowScale);
    float* row_lse = reinterpret_cast<float*>(smem + S::kRowLse);
    float* red_m = reinterpret_cast<float*>(smem + S::kRedM);
    float* red_l = reinterpret_cast<float*>(smem + S::kRedL);
    float* recv = reinterpret_cast<float*>(smem + S::kRecv);
    uint32_t const mbar = smem_u32(smem + S::kMbar);
    uint32_t const mma_bar = mbar + 8;
    uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + S::kMbar + 16);

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const cr = blockIdx.x % kCs; // cluster rank; owns blocks [cr * BPC, cr * BPC + BPC)
    int const ci = blockIdx.x / kCs;
    int const req = ci / kGroups;
    int const tfirst = (ci % kGroups) * QG;          // this cluster's query tokens [tfirst, tfirst + QG) of the request
    int const tok0 = req * QLEN;
    uint32_t const gbar = smem_u32(smem + S::kGBar); // FQKV: weight ring, A ready, accumulators done
    int const qkv_u = FQKV ? qkv_unit() : -1;        // FQKV: this CTA's projection unit, or -1

    // ---------------- Prologue: nothing here depends on the producer kernel ----------------
#if M3FA_TRIGGER_AT == 0 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (!FQKV)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
    M3FA_PROF(0);
#ifdef M3FA_PROFILE
    if (p.prof != nullptr && tid == 0)
    {
        uint32_t smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        p.prof[blockIdx.x * 64 + 40] = smid;
    }
#endif
    if constexpr (FQKV)
    {
        // The projection weights stream into TMEM while the producer (the all-reduce) still runs.
        if (warp < 2)
        {
            qkv_setup<S>(p, smem, gbar, tmem_slot, warp, lane, qkv_u);
        }
        if (ci >= p.attn_clusters)
        {
            // A projection-only CTA. Every reduce barrier of the cluster is initialized before any peer pushes.
            asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::
                             : "memory");
            int const op_u = static_cast<int>(blockIdx.x) - kOpFirstCta; // this CTA's o_proj unit (>= 0 here)
#ifndef M3FA_OP_EARLY
#define M3FA_OP_EARLY 0 // 1: o_proj-only CTAs stream their weights from launch (0: after the PDL wait)
#endif
            if (M3FA_OP_EARLY && qkv_u < 0 && warp == 0 && lane == 0)
            {
                op_issue<S>(p, smem, op_u); // an o_proj-only CTA: its weights stream from launch on
            }
            if (M3FA_OP_L2PF && qkv_u >= 0 && warp == 0 && lane == 0)
            {
                op_prefetch_l2(p, op_u); // into L2 during the all-reduce, behind this CTA's QKV ring
            }
            qkv_stream<S>(p, smem, gbar, tmem_slot, warp, lane, qkv_u);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (M3FA_MEGA_TRIGGER == 2)
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
#endif
            // This CTA's call counter: read after the PDL wait (the previous call's CTA increments it at its end), and
            // early, so its latency stays off the epilogue's critical path.
            unsigned int const calls = p.flags[32 + blockIdx.x];
            M3FA_PROF(1);
            if (!M3FA_OP_EARLY && qkv_u < 0 && warp == 0 && lane == 0)
            {
                op_issue<S>(p, smem, op_u);
            }
            if (qkv_u >= 0)
            {
                int const par = 0;
                qkv_mma<S>(p, smem, gbar, tmem_slot, tid, qkv_u);
                if (warp >= kComputeWarps)
                {
                    qkv_epilogue<S>(p, smem, gbar, tmem_slot, tid, qkv_u, par, calls);
                }
                if (warp == 0 && lane == 0)
                {
                    // The ring (and the activation over it) is free once the QKV MMAs completed.
                    mbar_wait_cta(gbar + 16 * kQkvRing + 8, 0);
                    op_issue<S>(p, smem, op_u);
                    if (p.pf_b0 > 0 || p.pf_b1 > 0)
                    {
                        window_prefetch_l2(p, qkv_u);
                    }
                }
            }
            __syncthreads(); // the QKV accumulator reads are done: TMEM columns [0, 512) are the o_proj's
            M3FA_PROF(2);
            uint32_t const tm = *tmem_slot;
            if (warp == 1)
            {
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                if (elect_one())
                {
                    op_stream<S>(smem, tm);
                }
                __syncwarp();
            }
            op_act<S>(p, smem, op_u, tid, calls);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            M3FA_PROF_T(36, 0);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
            // Late release: the dependents launch once every unit has its attention input (the rel_v4 rule).
            if (M3FA_MEGA_TRIGGER < 2)
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
#endif
            __syncthreads();
            if (warp == 1)
            {
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                if (elect_one())
                {
                    op_mma<S>(smem, tm);
                    M3FA_PROF_T(50, threadIdx.x);
                }
                __syncwarp();
            }
            if (warp >= kComputeWarps)
            {
                op_epilogue<S>(p, smem, tm, tid, op_u);
            }
            M3FA_PROF_T(37, kComputeThreads);
            __syncthreads();
            M3FA_PROF_T(55, 0);
            if (M3FA_MEGA_TRIGGER == 3)
            {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); // after this CTA's y rows / pushes
            }
            if (warp == 1)
            {
                asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(*tmem_slot), "r"(kTmemAll)
                             : "memory");
                M3FA_PROF_T(56, 32);
            }
            if (tid == 0)
            {
                p.flags[32 + blockIdx.x] = calls + 1u; // a plain store: no load on the exit path
            }
            M3FA_PROF(39);
#ifndef M3FA_FINAL_CLBAR
#define M3FA_FINAL_CLBAR 0
#endif
            if (M3FA_FINAL_CLBAR)
            {
                // No peer may exit while its partials can still be pushed into it (DSMEM lifetime).
                asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::
                                 : "memory");
            }
            M3FA_PROF(38);
            return;
        }
    }
#ifndef M3FA_MEGA_ATTN_WAIT_FIRST
#define M3FA_MEGA_ATTN_WAIT_FIRST 0 // 1: the attention CTAs read nothing (metadata, caches) before their PDL wait
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (FQKV && M3FA_MEGA_ATTN_WAIT_FIRST)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    if (!FQKV && warp == kWarps - 1) // a producer warp: it has nothing else to do before the PDL wait
    {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(tmem_slot)),
                     "r"(S::kTmemCols)
                     : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::: "memory");
    }
    int const seq_len = p.seq_lens[req];
    int page[BPC];
#pragma unroll
    for (int h = 0; h < BPC; ++h)
    {
        int const b = cr * BPC + h;
        page[h] = b < p.bt_cols ? p.block_table[static_cast<int64_t>(req) * p.bt_stride + b] : 0;
    }
    // New-token cache slots, for the tokens up to this cluster's last one.
    int slot[QLEN];
#pragma unroll
    for (int t = 0; t < QLEN; ++t)
    {
        slot[t] = t < tfirst + QG ? p.out_cache_loc[tok0 + t] : -1;
    }
    // This warp's producer tasks and, for every lane that applies RoPE, the task token's cos/sin.
    float cs[kMaxT][4], sn[kMaxT][4];
#pragma unroll
    for (int k = 0; k < kMaxT; ++k)
    {
        int const i = warp + kWarps * k;
        int const t = i < kHq * QG ? tfirst + i / kHq : (i - kHq * QG) / 3;
        if (i < kTasks && lane < 16 && t < QLEN)
        {
            int64_t const row = static_cast<int64_t>(p.position_ids[tok0 + t]) * kRotDim;
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int const coefficient = (lane * 4 + e) & (kRotDim / 2 - 1);
                cs[k][e] = p.rope[row + coefficient];
                sn[k][e] = p.rope[row + kRotDim / 2 + coefficient];
            }
        }
        else
        {
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                cs[k][e] = 1.0F;
                sn[k][e] = 0.0F;
            }
        }
    }
    float qw[4], kw[4], ikw[4];
    load_w4(qw, p.q_w, lane);
    load_w4(kw, p.k_w, lane);
    load_w4(ikw, p.ik_w, lane);
    float const kqs = p.kv_quant_scale[1];
    float const vqs = p.kv_quant_scale[2];
    float const sl2e = (p.sm_scale * 1.4426950409f) * p.k_gs[0];
    float const vgs = p.v_gs[0];
    // Longest attended length among this cluster's tokens, and each block's rows within it.
    int const kv_max = max(seq_len - QLEN + tfirst + QG, 0);
    int nvh[BPC];
#pragma unroll
    for (int h = 0; h < BPC; ++h)
    {
        nvh[h] = min(max(kv_max - (cr * BPC + h) * kPage, 0), kPage);
    }
    bool const any_valid = nvh[0] > 0;
    if (tid == 32)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mbar), "r"(1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(mma_bar), "r"(1));
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(mbar), "r"(kBlocks * QG * BPC * kPushBytes)
            : "memory");
        if constexpr (OPJ)
        {
            uint32_t const ob = smem_u32(smem + S::kOBar);
            for (int i = 0; i < kOpTiles + 3; ++i)
            {
                mbar_init(ob + 8 * i, 1);
            }
            mbar_expect_tx(ob + 8 * (kOpTiles + 2), (kOpSplits - 1) * kOpOwnRows * kMaxTok * 4);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        if constexpr (OPJ)
        {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            if (M3FA_C1_OPISSUE == 0)
            {
                op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
            }
        }
    }
    // This thread's packed K / V chunks, held in registers until they are dequantized after the PDL wait.
    RawChunk kraw[kKPer];
#pragma unroll
    for (int k = 0; k < kKPer; ++k)
    {
        int const q = tid + kComputeThreads * k;
        kraw[k] = RawChunk{make_uint4(0u, 0u, 0u, 0u), 0u};
        int const h = q / (kPage * 4);
        if (q < kChunks && tid < kComputeThreads && nvh[h] > 0)
        {
            kraw[k] = load_chunk(p.kv_data + static_cast<int64_t>(page[h]) * p.d_page,
                p.kv_scale + static_cast<int64_t>(page[h]) * p.s_page, chunk_row(q % (kPage * 4)),
                chunk_col(q % (kPage * 4)), false);
        }
    }
    // V goes through cp.async into shared memory: an asynchronous copy does not hold the producer warps up the way
    // an outstanding register load does, and they need V only after QK has been issued.
    if (tid >= kVStart)
    {
#pragma unroll
        for (int h = 0; h < BPC; ++h)
        {
            if (nvh[h] > 0)
            {
                uint8_t* dst = smem + S::kRawV + h * S::kRawVBlock;
                uint8_t const* data = p.kv_data + static_cast<int64_t>(page[h]) * p.d_page + p.d_plane;
                uint8_t const* scales = p.kv_scale + static_cast<int64_t>(page[h]) * p.s_page + p.s_plane;
                for (int c = tid - kVStart; c < S::kRawVBlock / 16; c += kVThreads)
                {
                    cp_async16(dst + c * 16, c < kPage * 4 ? data + c * 16 : scales + (c - kPage * 4) * 16);
                }
            }
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    if (warp == kWarps - 1)
    {
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    }
    // Every CTA's mbarrier is initialized before any peer pushes into it; the matching wait sits right before the
    // push, so this barrier's latency hides behind the attention (with aliased partials it moves there).
    if constexpr (!S::kRecvAlias)
    {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    }
    int kv_len[QG];
#pragma unroll
    for (int tl = 0; tl < QG; ++tl)
    {
        kv_len[tl] = max(seq_len - QLEN + tfirst + tl + 1, 0);
    }
    int const cta_tok0 = cr * kTok; // request position of this CTA's first block row
    // New token t's row in this CTA's tiles (block * 128 + row), or -1: the CTA owns the block holding the token's
    // position, and the token is attended by this cluster (t <= its last token). Every thread derives the same.
    int new_row[QLEN];
#pragma unroll
    for (int t = 0; t < QLEN; ++t)
    {
        int const pos = seq_len - QLEN + t;
        int const blk = min(pos >> 7, kBlocks - 1);
        int const spage = slot[t] / kPage;
        bool const mine = t < tfirst + QG && pos >= 0 && blk / BPC == cr && slot[t] >= 0 && spage < p.num_pages;
        new_row[t] = mine && spage == page[blk % BPC] ? (blk % BPC) * kPage + (slot[t] & (kPage - 1)) : -1;
    }
    auto is_new_row = [&](int r)
    {
        bool hit = false;
#pragma unroll
        for (int t = 0; t < QLEN; ++t)
        {
            hit |= new_row[t] == r;
        }
        return hit;
    };
    // Cache writes of the new K/V/index-K are issued after this CTA has pushed its partials: only the fp16 tile rows
    // are on the attention's path. kind 0: none, 1: NVFP4 data + scale, 2: FP8 index-K.
    int wr_kind[kMaxT];
    uint8_t* wr_ptr[kMaxT];
    uint8_t* wr_sptr[kMaxT];
    uint32_t wr_val[kMaxT];
#pragma unroll
    for (int k = 0; k < kMaxT; ++k)
    {
        wr_kind[k] = 0;
        wr_ptr[k] = nullptr;
        wr_sptr[k] = nullptr;
        wr_val[k] = 0u;
    }
    uint32_t tmem = 0;
    auto dequant_k = [&]()
    {
        if (tid < kComputeThreads)
        {
#pragma unroll
            for (int k = 0; k < kKPer; ++k)
            {
                int const q = tid + kComputeThreads * k;
                if (q < kChunks)
                {
                    int const h = q / (kPage * 4);
                    int const row = chunk_row(q % (kPage * 4));
                    if (!is_new_row(h * kPage + row))
                    {
                        store_chunk_fp16(
                            smem + S::kK16 + h * S::kTileBytes, row, chunk_col(q % (kPage * 4)), kraw[k], row < nvh[h]);
                    }
                }
            }
        }
    };
    // The producer warps' V tiles (raw tiles from cp.async, dequantized to fp16), then their PV-barrier arrival.
    auto dequant_v = [&]()
    {
#ifdef M3FA_DBG_NO_VDQ
        if constexpr (!kRegCore)
        {
            asm volatile("bar.arrive 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
        }
        return;
#endif
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        asm volatile("bar.sync 4, %0;\n" ::"n"(kVThreads) : "memory");
#pragma unroll
        for (int k = 0; k < kVPer; ++k)
        {
            int const q = tid - kVStart + kVThreads * k;
            if (q < kChunks)
            {
                int const h = q / (kPage * 4);
                int const row = chunk_row(q % (kPage * 4));
                int const cc = chunk_col(q % (kPage * 4));
                if (!is_new_row(h * kPage + row))
                {
                    uint8_t const* rawv = smem + S::kRawV + h * S::kRawVBlock;
                    uint8_t const* sc = rawv + kPage * 64 + (row >> 2) * 32 + (row & 3);
                    RawChunk c;
                    c.data = *reinterpret_cast<uint4 const*>(rawv + row * 64 + cc * 16);
                    c.scales
                        = static_cast<uint32_t>(sc[(cc * 2) * 4]) | (static_cast<uint32_t>(sc[(cc * 2 + 1) * 4]) << 8);
                    store_chunk_fp16(smem + S::kV16 + h * S::kTileBytes, row, cc, c, row < nvh[h]);
                }
            }
        }
        M3FA_FENCE_PROXY_ASYNC(); // V tiles -> tensor core reads (ordered by the barrier before the PV issue)
        if constexpr (!kRegCore)
        {
            asm volatile("bar.arrive 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
        }
        M3FA_PROF_T(16, kComputeThreads);
        M3FA_PROF_T(17, kThreads - 1);
    };
    if constexpr (FQKV)
    {
        qkv_stream<S>(p, smem, gbar, tmem_slot, warp, lane, qkv_u);
    }
    if constexpr (!FQKV)
    {
        if (M3FA_V15_EARLYV && any_valid && tid >= kVStart)
        {
            dequant_v();
        }
        if constexpr (kEarlyClw && !S::kRecvAlias)
        {
            asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
        }
    }
    M3FA_PROF(11);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (FQKV && M3FA_MEGA_TRIGGER == 2)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#if M3FA_TRIGGER_AT == 1
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#endif
    // FQKV (megakernel): this call's QKV rows sit in buffer (calls & 1), polled value by value below.
    unsigned int const mega_calls = (FQKV || OPJ) ? p.flags[32 + blockIdx.x] : 0u;
    if (OPJ && M3FA_C1_OPISSUE == 1 && tid == 32)
    {
        op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
    }
    __nv_bfloat16 const* qkv_src = FQKV ? p.packed + (mega_calls & 1u) * kMaxTok * kQkvN : p.packed;
    M3FA_PROF(1);
    int const par = 0;
    if constexpr (FQKV)
    {
        if (qkv_u >= 0)
        {
            qkv_mma<S>(p, smem, gbar, tmem_slot, tid, qkv_u);
        }
    }

    if constexpr (FQKV)
    {
        if (qkv_u >= 0)
        {
            if (warp >= kComputeWarps)
            {
                qkv_epilogue<S>(p, smem, gbar, tmem_slot, tid, qkv_u, par, mega_calls);
            }
            else
            {
                // The attention regions overlay the weight ring: every weight byte has gone on into TMEM (or, with
                // the MMAs reading the ring, every MMA has completed) first.
                mbar_wait_cta(gbar + 16 * kQkvRing + (kQkvASmem ? 8 : 0), 0);
            }
        }
        if (any_valid)
        {
            dequant_k();
        }
#ifndef M3FA_MEGA_EARLYV
#define M3FA_MEGA_EARLYV 1 // attention CTAs dequantize V here, while the QKV rows are still on their way
#endif
#ifndef M3FA_MEGA_EARLYCLW
#define M3FA_MEGA_EARLYCLW 1 // and pass the cluster barrier (peers' merge mbarriers initialized) here too
#endif
        if (M3FA_MEGA_EARLYV && any_valid && tid >= kVStart)
        {
            dequant_v();
        }
        if constexpr (kEarlyClw && !S::kRecvAlias)
        {
            asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
        }
        M3FA_PROF(14);
    }

    if (any_valid) // CTA-uniform
    {
        // ---------------- Producer ----------------
        // Q rows of this cluster's tokens in every CTA. The new K/V of every token up to this cluster's last one
        // whose position falls in this CTA's blocks go straight into the fp16 tiles; the CTA writes the caches, and
        // the index-K, only for its own tokens.
        uint2 raw[kMaxT];
        float fq[FQKV ? kMaxT : 1][4];
        bool act[kMaxT];
#pragma unroll
        for (int k = 0; k < kMaxT; ++k)
        {
            int const i = warp + kWarps * k;
            raw[k] = make_uint2(0u, 0u);
            act[k] = false;
            if (i < kTasks)
            {
                int t, col;
                bool active = true;
                if (i >= kHq * QG)
                {
                    int const kv = i - kHq * QG;
                    t = kv / 3;
                    int const kind = kv % 3;
                    col = kind == 0 ? kColK : (kind == 1 ? kColV : kColIdxK);
                    int const pos = seq_len - QLEN + t;
                    active = t < tfirst + QG && pos >= 0 && min(pos >> 7, kBlocks - 1) / BPC == cr && slot[t] >= 0
                        && slot[t] / kPage < p.num_pages && (kind != 2 || t >= tfirst);
                }
                else
                {
                    t = tfirst + i / kHq;
                    col = i % kHq;
                }
                act[k] = active;
                if (active)
                {
                    uint2 const* src = reinterpret_cast<uint2 const*>(
                        qkv_src + static_cast<int64_t>(tok0 + t) * p.packed_stride + col * kHeadDim + lane * 4);
                    if constexpr (FQKV)
                    {
                        uint2 v;
                        bool pending;
                        do
                        {
                            asm volatile("ld.relaxed.gpu.global.v2.b32 {%0, %1}, [%2];\n"
                                         : "=r"(v.x), "=r"(v.y)
                                         : "l"(src)
                                         : "memory");
                            pending = (v.x & 0xFFFFu) == 0xFFFFu || (v.x >> 16) == 0xFFFFu || (v.y & 0xFFFFu) == 0xFFFFu
                                || (v.y >> 16) == 0xFFFFu;
                        } while (pending);
                        raw[k] = v;
                    }
                    else
                    {
                        raw[k] = *src;
                    }
                }
            }
        }
        M3FA_PROF(15);
        // While the packed-QKV loads are in flight: dequantize this thread's K chunks to fp16 (exact: e2m1 x e4m3
        // fits fp16). Rows past the longest attended length are zeroed, as the Triton producer masks its loads; the
        // new tokens' rows are written by their producer tasks instead. (Building the tiles before the PDL wait was
        // measured slower: the cache loads issued at entry are still in flight there, so the wait moves later.)
        if constexpr (!FQKV)
        {
            dequant_k();
        }
#pragma unroll
        for (int k = 0; k < kMaxT; ++k)
        {
            int const i = warp + kWarps * k;
            if (i >= kTasks || !act[k])
            {
                continue;
            }
            float e[4];
            {
                float2 const a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw[k].x));
                float2 const b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw[k].y));
                e[0] = a.x;
                e[1] = a.y;
                e[2] = b.x;
                e[3] = b.y;
            }
            if (i < kHq * QG)
            {
                norm_rope(e, qw, p.eps, cs[k], sn[k], lane);
                __nv_fp8x2_e4m3 const lo(make_float2(e[0], e[1]));
                __nv_fp8x2_e4m3 const hi(make_float2(e[2], e[3]));
                uint2 hv;
                hv.x = e4m3x2_to_h2(lo.__x);
                hv.y = e4m3x2_to_h2(hi.__x);
                *reinterpret_cast<uint2*>(smem + S::kQ + kmaj_off(i, lane * 4)) = hv; // row i = token * 16 + head
                continue;
            }
            int const t = (i - kHq * QG) / 3;
            int const kind = (i - kHq * QG) % 3;
            int const spage = slot[t] / kPage;
            int const within = slot[t] & (kPage - 1);
            if (kind == 2)
            {
                norm_rope(e, ikw, p.eps, cs[k], sn[k], lane);
                __nv_fp8x2_e4m3 const lo(make_float2(e[0], e[1]));
                __nv_fp8x2_e4m3 const hi(make_float2(e[2], e[3]));
                wr_kind[k] = 2;
                wr_val[k] = static_cast<uint32_t>(lo.__x) | (static_cast<uint32_t>(hi.__x) << 16);
                wr_ptr[k] = p.idx_k + static_cast<int64_t>(spage) * p.i_page + static_cast<int64_t>(within) * p.i_tok
                    + lane * 4;
                continue;
            }
            bool const isV = kind == 1;
            if (!isV)
            {
                norm_rope(e, kw, p.eps, cs[k], sn[k], lane);
            }
            uint8_t sbyte = 0;
            uint16_t const fp4 = quantizeMinimaxM3Fp4x4(e, isV ? vqs : kqs, &sbyte);
            if (t >= tfirst)
            {
                int const scale_col = lane / 4;
                int const scale_within = isV ? (within / 4) * 32 + scale_col * 4 + within % 4 : within * 8 + scale_col;
                wr_kind[k] = 1;
                wr_val[k] = static_cast<uint32_t>(fp4) | (static_cast<uint32_t>(sbyte) << 16);
                wr_ptr[k] = p.kv_data + static_cast<int64_t>(spage) * p.d_page + (isV ? p.d_plane : 0) + within * 64
                    + lane * 2;
                wr_sptr[k] = p.kv_scale + static_cast<int64_t>(spage) * p.s_page + (isV ? p.s_plane : 0) + scale_within;
            }
            if (new_row[t] >= 0)
            {
                // The attention reads this page: its fp16 tile row is the freshly quantized row, dequantized like
                // the rest of the tile.
                int const h = new_row[t] / kPage;
                int const row = new_row[t] % kPage;
                uint32_t const sc2 = e4m3_to_h2_splat(sbyte);
                uint2 hv;
                hv.x = hmul2_u(e2m1x2_to_h2(fp4 & 0xFFu), sc2);
                hv.y = hmul2_u(e2m1x2_to_h2(fp4 >> 8), sc2);
                *reinterpret_cast<uint2*>(
                    smem + (isV ? S::kV16 : S::kK16) + h * S::kTileBytes + kmaj_off(row, lane * 4))
                    = hv;
            }
        }
        // K tiles, Q rows and patched rows -> tensor core reads. Only threads that wrote fence: the fence also
        // waits for the thread's outstanding loads, and the producer warps still have their V chunks in flight.
        bool wrote = tid < kComputeThreads;
#pragma unroll
        for (int k = 0; k < kMaxT; ++k)
        {
            wrote |= act[k];
        }
        if (wrote)
        {
            M3FA_FENCE_PROXY_ASYNC();
        }
        M3FA_PROF(13);
        __syncthreads();
#if M3FA_TRIGGER_AT == 2 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        // Every K/V byte this CTA reads has arrived (the V chunks sit in registers): release the dependents now so
        // their weight streams do not queue ahead of the KV reads (the megakernel releases late, from its o_proj).
        if constexpr (kAttnTrigger || (FQKV && M3FA_MEGA_TRIGGER == 1))
        {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
#endif
        if (OPJ && M3FA_C1_OPISSUE == 2 && tid == 32)
        {
            op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
        }
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        tmem = *tmem_slot;
        M3FA_PROF(2);
        // ---------------- S^T = K Q^T per block on the tensor core: TMEM lane = token, column = query row ----------
#ifndef M3FA_MEGA_WARPMMA
#define M3FA_MEGA_WARPMMA 1 // attention MMAs issued by warp 0's elected lane with warp-uniform operands (not tid 0)
#endif
        if (!kRegCore && (kWarpMma ? warp == 0 : tid == 0))
        {
            // Operands computed by the whole warp stay in uniform registers; one elected lane issues.
            uint32_t const tm_u = tmem;
            bool const issue = kWarpMma ? elect_one() : true;
            if (issue)
            {
#pragma unroll
                for (int h = 0; h < BPC; ++h)
                {
                    uint32_t const a0 = smem_u32(smem + S::kK16 + h * S::kTileBytes);
                    uint32_t const b0 = smem_u32(smem + S::kQ);
#pragma unroll
                    for (int ks = 0; ks < kHeadDim / 16; ++ks)
                    {
                        umma_f16(tm_u + h * kRows, umma_desc(a0 + ks * 256, 128, 2048),
                            umma_desc(b0 + ks * 256, 128, 2048), umma_idesc(kRows, 0, 0), ks > 0);
                    }
                }
                umma_commit(mma_bar);
            }
            if (kWarpMma)
            {
                __syncwarp();
            }
        }
        if (tid >= kVStart && !(FQKV && M3FA_MEGA_EARLYV) && !(!FQKV && M3FA_V15_EARLYV))
        {
            // The producer warps dequantize V while QK and the softmax run.
            dequant_v();
        }
        if constexpr (kRegCore)
        {
            // ---------------- Register core: warp w of block h = w / kWpb owns 8 * kNt tokens for S and 8 * kNt
            // output dims for O; m16n8k16 fp16 MMAs with fp32 accumulators, the softmax in the S fragments ----------
            constexpr int kWpb = kComputeWarps / BPC; // warps per block
            constexpr int kNt = 16 / kWpb;            // 8-token (S) or 8-dim (O) tiles per warp
            float* redx_m = reinterpret_cast<float*>(smem + S::kRedX);
            float* redx_l = redx_m + BPC * kRows * kWpb;
            if (warp < kComputeWarps)
            {
                int const h = warp / kWpb;
                int const wib = warp % kWpb;
                int const g = lane >> 2;
                int const q4 = lane & 3;
                uint32_t const ktile = smem_u32(smem + S::kK16 + h * S::kTileBytes);
                uint32_t const vtile = smem_u32(smem + S::kV16 + h * S::kTileBytes);
                uint32_t const qrows = smem_u32(smem + S::kQ);
                uint8_t* ptile = smem + S::kK16 + h * S::kTileBytes; // P over this block's K tile, after QK
                auto block_bar = [&]()
                {
                    if (h == 0)
                    {
                        asm volatile("bar.sync 14, %0;\n" ::"n"(kWpb * 32) : "memory");
                    }
                    else
                    {
                        asm volatile("bar.sync 15, %0;\n" ::"n"(kWpb * 32) : "memory");
                    }
                };
                // S = Q K^T: rows (c, g / g + 8), tokens 8 tb + 2 q4 (+1).
                float sv[QG][kNt][4];
#pragma unroll
                for (int c = 0; c < QG; ++c)
                {
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            sv[c][nt][e] = 0.0F;
                        }
                    }
                }
#pragma unroll
                for (int ks = 0; ks < kHeadDim / 16; ++ks)
                {
                    uint32_t bk[kNt][2];
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
                        int const tb = wib * kNt + nt;
                        ldsm_x2(ktile + tb * 2048 + (ks * 2 + ((lane >> 3) & 1)) * 128 + (lane & 7) * 16, bk[nt]);
                    }
#pragma unroll
                    for (int c = 0; c < QG; ++c)
                    {
                        uint32_t aq[4];
                        ldsm_x4(
                            qrows + (c * 2 + ((lane >> 3) & 1)) * 2048 + (ks * 2 + (lane >> 4)) * 128 + (lane & 7) * 16,
                            aq);
#pragma unroll
                        for (int nt = 0; nt < kNt; ++nt)
                        {
                            mma_f16_16816(sv[c][nt], aq, bk[nt]);
                        }
                    }
                }
                M3FA_PROF(7);
                // Row maxima (log2 domain, masked as the producer masks), quad then across the block's warps.
                float mrow[QG][2];
#pragma unroll
                for (int c = 0; c < QG; ++c)
                {
                    float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            int const tok = cta_tok0 + h * kPage + (wib * kNt + nt) * 8 + 2 * q4 + (e & 1);
                            bool const valid = tok < kv_max && tok < kv_len[c];
                            float const v = valid ? sv[c][nt][e] * sl2e : kNegInf;
                            sv[c][nt][e] = v;
                            if (e < 2)
                            {
                                mx0 = fmaxf(mx0, v);
                            }
                            else
                            {
                                mx1 = fmaxf(mx1, v);
                            }
                        }
                    }
                    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
                    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
                    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
                    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
                    if (q4 == 0)
                    {
                        redx_m[(h * kRows + c * 16 + g) * kWpb + wib] = mx0;
                        redx_m[(h * kRows + c * 16 + g + 8) * kWpb + wib] = mx1;
                    }
                }
                block_bar(); // every warp of the block is past QK (P may overwrite the K tile) and wrote its maxima
                M3FA_PROF(8);
#pragma unroll
                for (int c = 0; c < QG; ++c)
                {
#pragma unroll
                    for (int half = 0; half < 2; ++half)
                    {
                        float const* rm = redx_m + (h * kRows + c * 16 + g + 8 * half) * kWpb;
                        float m = kNegInf;
#pragma unroll
                        for (int w4 = 0; w4 < kWpb; w4 += 4)
                        {
                            float4 const v = *reinterpret_cast<float4 const*>(rm + w4);
                            m = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
                        }
                        mrow[c][half] = m;
                    }
                    // P = exp2(s - m) as fp16 into [row][token] core matrices; the fp32 values feed the row sums.
                    float l0 = 0.0F, l1 = 0.0F;
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
                        float a[4];
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            float const m = mrow[c][e >> 1];
                            a[e] = m == kNegInf ? 0.0F : ex2(sv[c][nt][e] - m);
                        }
                        l0 += a[0] + a[1];
                        l1 += a[2] + a[3];
                        int const t0 = (wib * kNt + nt) * 8 + 2 * q4;
                        int const r0 = c * 16 + g;
                        __half2 const p0 = __floats2half2_rn(a[0], a[1]);
                        __half2 const p1 = __floats2half2_rn(a[2], a[3]);
                        *reinterpret_cast<__half2*>(
                            ptile + (r0 >> 3) * 2048 + (t0 >> 3) * 128 + (r0 & 7) * 16 + (t0 & 7) * 2)
                            = p0;
                        *reinterpret_cast<__half2*>(
                            ptile + ((r0 + 8) >> 3) * 2048 + (t0 >> 3) * 128 + (r0 & 7) * 16 + (t0 & 7) * 2)
                            = p1;
                    }
                    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
                    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
                    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
                    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
                    if (q4 == 0)
                    {
                        redx_l[(h * kRows + c * 16 + g) * kWpb + wib] = l0;
                        redx_l[(h * kRows + c * 16 + g + 8) * kWpb + wib] = l1;
                    }
                }
                M3FA_PROF_T(19, 0);
            }
            __syncthreads(); // P, the V tiles (producer warps) and the row sums visible
            M3FA_PROF(9);
            if (warp < kComputeWarps)
            {
                int const h = warp / kWpb;
                int const wib = warp % kWpb;
                int const g = lane >> 2;
                int const q4 = lane & 3;
                uint32_t const vtile = smem_u32(smem + S::kV16 + h * S::kTileBytes);
                uint32_t const ptile = smem_u32(smem + S::kK16 + h * S::kTileBytes);
                float* obuf = reinterpret_cast<float*>(smem + S::kK16 + h * S::kTileBytes + S::kObufOff);
                // O = P V: rows (c, g / g + 8), dims 8 dt + 2 q4 (+1).
                float ov[QG][kNt][4];
#pragma unroll
                for (int c = 0; c < QG; ++c)
                {
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            ov[c][nt][e] = 0.0F;
                        }
                    }
                }
#pragma unroll
                for (int ks = 0; ks < kPage / 16; ++ks)
                {
                    uint32_t bv[kNt][2];
                    int const tok = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
                        int const dt = wib * kNt + nt;
                        ldsm_x2_trans(vtile + (tok >> 3) * 2048 + dt * 128 + (tok & 7) * 16, bv[nt]);
                    }
#pragma unroll
                    for (int c = 0; c < QG; ++c)
                    {
                        uint32_t ap[4];
                        ldsm_x4(
                            ptile + (c * 2 + ((lane >> 3) & 1)) * 2048 + (ks * 2 + (lane >> 4)) * 128 + (lane & 7) * 16,
                            ap);
#pragma unroll
                        for (int nt = 0; nt < kNt; ++nt)
                        {
                            mma_f16_16816(ov[c][nt], ap, bv[nt]);
                        }
                    }
                }
#pragma unroll
                for (int c = 0; c < QG; ++c)
                {
#pragma unroll
                    for (int nt = 0; nt < kNt; ++nt)
                    {
                        int const d = (wib * kNt + nt) * 8 + 2 * q4;
                        int const r0 = c * 16 + g;
                        *reinterpret_cast<float2*>(obuf + r0 * kOPitch + d) = make_float2(ov[c][nt][0], ov[c][nt][1]);
                        *reinterpret_cast<float2*>(obuf + (r0 + 8) * kOPitch + d)
                            = make_float2(ov[c][nt][2], ov[c][nt][3]);
                    }
                }
            }
            else if (tid - kComputeThreads < BPC * kRows)
            {
                // Row statistics of every (block, row): LSE and the PV scale, as the producer derives them.
                int const r = tid - kComputeThreads;
                float const* rm = redx_m + r * kWpb;
                float const* rl = redx_l + r * kWpb;
                float m = kNegInf;
                float l = 0.0F;
#pragma unroll
                for (int w4 = 0; w4 < kWpb; w4 += 4)
                {
                    float4 const vm = *reinterpret_cast<float4 const*>(rm + w4);
                    float4 const vl = *reinterpret_cast<float4 const*>(rl + w4);
                    m = fmaxf(m, fmaxf(fmaxf(vm.x, vm.y), fmaxf(vm.z, vm.w)));
                    l += (vl.x + vl.y) + (vl.z + vl.w);
                }
                float const lse = l > 0.0F ? m + lg2(l) : kNegInf;
                float const scale = lse > kNegInf ? ex2(m - lse) : 0.0F;
                row_scale[r] = scale * vgs;
                row_lse[r] = lse;
            }
            __syncthreads(); // O partials and row statistics visible to the push
        }
        else
        {
            if constexpr (kRs2)
            {
                // ---------------- Split softmax: warps 0..3 rows 0..7, warps 4..7 rows 8..15 (thread = block token)
                // --------
                if (warp < 8)
                {
                    int const rg = warp / 4;
                    int const wq = warp % 4;
                    int const tk = wq * 32 + lane;
                    auto rg_bar = [&]()
                    {
                        if (rg == 0)
                        {
                            asm volatile("bar.sync 10, 128;\n" ::: "memory");
                        }
                        else
                        {
                            asm volatile("bar.sync 11, 128;\n" ::: "memory");
                        }
                    };
                    int const rs8 = ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
                    mbar_wait_cta(mma_bar, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                    M3FA_PROF(7);
                    float v[8];
                    tmem_ld8(tmem + ((wq * 32) << 16) + 8 * rg, v);
                    bool const valid = cta_tok0 + tk < kv_max && cta_tok0 + tk < kv_len[0];
                    float a[8];
#pragma unroll
                    for (int j2 = 0; j2 < 8; ++j2)
                    {
                        v[j2] = valid ? v[j2] * sl2e : kNegInf;
                        a[j2] = v[j2];
                    }
                    float const mw = warp_reduce_scatter8<true>(a, lane);
                    if ((lane & 3) == 0)
                    {
                        red_m[(8 * rg + rs8) * 4 + wq] = mw;
                    }
                    rg_bar();
                    M3FA_PROF(8);
                    uint8_t* pdst = smem + S::kP + (tk >> 3) * 128 + (tk & 7) * 16;
                    float mrow = kNegInf;
                    if (lane < 8)
                    {
                        float4 const mq = *reinterpret_cast<float4 const*>(red_m + (8 * rg + lane) * 4);
                        mrow = fmaxf(fmaxf(mq.x, mq.y), fmaxf(mq.z, mq.w));
                    }
                    float pa[8];
#pragma unroll
                    for (int j2 = 0; j2 < 8; ++j2)
                    {
                        float const m = __shfl_sync(0xffffffffu, mrow, j2);
                        pa[j2] = m == kNegInf ? 0.0F : ex2(v[j2] - m);
                    }
                    if (tid == 0)
                    {
                        M3FA_CLK(41);
                    }
                    uint32_t pw[4];
#pragma unroll
                    for (int j2 = 0; j2 < 4; ++j2)
                    {
                        __half2 const hp = __floats2half2_rn(pa[2 * j2], pa[2 * j2 + 1]);
                        pw[j2] = *reinterpret_cast<uint32_t const*>(&hp);
                    }
                    *reinterpret_cast<uint4*>(pdst + rg * 2048) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
                    M3FA_PROF_T(18, 0);
                    if (tid == 0)
                    {
                        M3FA_CLK(42);
                    }
                    M3FA_FENCE_PROXY_ASYNC(); // P -> tensor core reads
                    if (tid == 0)
                    {
                        M3FA_CLK(43);
                    }
                    if (warp == 0)
                    {
                        asm volatile("bar.sync 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
                        if (tid == 0)
                        {
                            M3FA_CLK(44);
                        }
                    }
                    else
                    {
                        asm volatile("bar.arrive 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
                    }
                    if (kWarpMma ? warp == 0 : tid == 0)
                    {
                        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                        uint32_t const tm_u = tmem;
                        uint32_t const a0 = smem_u32(smem + S::kV16);
                        uint32_t const b0 = smem_u32(smem + S::kP);
                        if (kWarpMma ? elect_one() : true)
                        {
#pragma unroll
                            for (int ks = 0; ks < kPage / 16; ++ks)
                            {
                                umma_f16(tm_u + BPC * kRows, umma_desc(a0 + ks * 4096, 2048, 128),
                                    umma_desc(b0 + ks * 256, 128, 2048), umma_idesc(kRows, 1, 1), ks > 0);
                            }
                            umma_commit(mma_bar);
                        }
                        if (kWarpMma)
                        {
                            __syncwarp();
                        }
                    }
                    M3FA_PROF_T(19, 0);
                    float const lw = warp_reduce_scatter8<false>(pa, lane);
                    if ((lane & 3) == 0)
                    {
                        red_l[(8 * rg + rs8) * 4 + wq] = lw;
                    }
                    rg_bar();
                    M3FA_PROF_T(20, 0);
                    if (wq == 0 && lane < 8)
                    {
                        int const r = 8 * rg + lane;
                        float4 const mq = *reinterpret_cast<float4 const*>(red_m + r * 4);
                        float4 const lq = *reinterpret_cast<float4 const*>(red_l + r * 4);
                        float const m = fmaxf(fmaxf(mq.x, mq.y), fmaxf(mq.z, mq.w));
                        float const l = (lq.x + lq.y) + (lq.z + lq.w);
                        float const lse = l > 0.0F ? m + lg2(l) : kNegInf;
                        float const scale = lse > kNegInf ? ex2(m - lse) : 0.0F;
                        row_scale[r] = scale * vgs;
                        row_lse[r] = lse;
                    }
                }
                __syncthreads(); // row scales and LSEs visible (PV was issued above)
                M3FA_PROF(9);
                if (warp < 8)
                {
                    mbar_wait_cta(mma_bar, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                }
            }
            else
            {
                // ---------------- Softmax: thread = block token; the (block, 16-row chunk) pairs are spread over the
                // four groups of four compute warps (each group covers the 128 TMEM lanes = block tokens)
                // ----------------
                if (warp < kComputeWarps && warp / 4 < kPairs)
                {
                    int const g = warp / 4;
                    int const wq = warp % 4;
                    int const tk = wq * 32 + lane;
                    auto group_bar = [&]()
                    {
                        if (g == 0)
                        {
                            asm volatile("bar.sync 10, 128;\n" ::: "memory");
                        }
                        else if (g == 1)
                        {
                            asm volatile("bar.sync 11, 128;\n" ::: "memory");
                        }
                        else if (g == 2)
                        {
                            asm volatile("bar.sync 12, 128;\n" ::: "memory");
                        }
                        else
                        {
                            asm volatile("bar.sync 13, 128;\n" ::: "memory");
                        }
                    };
                    // Row r(lane) this lane holds after the reduce-scatter: bits 4..1 of the lane.
                    int const rs_row
                        = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
                    mbar_wait_cta(mma_bar, 0);
                    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                    M3FA_PROF(7);
                    float v[kPairsPerGroup][16];
#pragma unroll
                    for (int k = 0; k < kPairsPerGroup; ++k)
                    {
                        int const q = g + 4 * k;
                        if (q < kPairs)
                        {
                            int const h = q % BPC;
                            int const c = q / BPC;
                            tmem_ld16(tmem + ((wq * 32) << 16) + h * kRows + c * 16, v[k]);
                            bool const valid
                                = cta_tok0 + h * kPage + tk < kv_max && cta_tok0 + h * kPage + tk < kv_len[c];
                            float a[16];
#pragma unroll
                            for (int j = 0; j < 16; ++j)
                            {
                                v[k][j] = valid ? v[k][j] * sl2e : kNegInf;
                                a[j] = v[k][j];
                            }
                            float const mw = warp_reduce_scatter16<true>(a, lane);
                            if ((lane & 1) == 0)
                            {
                                red_m[((h * QG + c) * 16 + rs_row) * 4 + wq] = mw;
                            }
                        }
                    }
                    group_bar();
                    M3FA_PROF(8);
                    float pa[kPairsPerGroup][16];
#pragma unroll
                    for (int k = 0; k < kPairsPerGroup; ++k)
                    {
                        int const q = g + 4 * k;
                        if (q < kPairs)
                        {
                            int const h = q % BPC;
                            int const c = q / BPC;
                            uint8_t* pdst = smem + S::kP + h * kRows * kPage * 2 + (tk >> 3) * 128 + (tk & 7) * 16;
                            // Row maxima: lane j (< 16) reduces row j's four warp partials, then broadcasts it.
                            float mrow = kNegInf;
                            if (lane < 16)
                            {
                                float4 const mq
                                    = *reinterpret_cast<float4 const*>(red_m + ((h * QG + c) * 16 + lane) * 4);
                                mrow = fmaxf(fmaxf(mq.x, mq.y), fmaxf(mq.z, mq.w));
                            }
#pragma unroll
                            for (int j = 0; j < 16; ++j)
                            {
                                float const m = __shfl_sync(0xffffffffu, mrow, j);
                                pa[k][j] = m == kNegInf ? 0.0F : ex2(v[k][j] - m);
                            }
                            if (tid == 0)
                            {
                                M3FA_CLK(41);
                            }
                            uint32_t pw[8];
#pragma unroll
                            for (int j = 0; j < 8; ++j)
                            {
                                __half2 const hp = __floats2half2_rn(pa[k][2 * j], pa[k][2 * j + 1]);
                                pw[j] = *reinterpret_cast<uint32_t const*>(&hp);
                            }
                            // P^T as the MN-major B operand: 8 query rows of one token are one 16-byte core-matrix row.
                            *reinterpret_cast<uint4*>(pdst + (2 * c) * 2048) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
                            *reinterpret_cast<uint4*>(pdst + (2 * c + 1) * 2048)
                                = make_uint4(pw[4], pw[5], pw[6], pw[7]);
                            M3FA_PROF_T(18, 0);
                        }
                    }
                    if (tid == 0)
                    {
                        M3FA_CLK(42);
                    }
                    M3FA_FENCE_PROXY_ASYNC(); // P -> tensor core reads
                    if (tid == 0)
                    {
                        M3FA_CLK(43);
                    }
                    // PV starts as soon as P and the V tiles are in: thread 0 waits for the softmax warps and the
                    // producer warps (V); the row sums and statistics below overlap the MMAs.
                    if (warp == 0)
                    {
                        asm volatile("bar.sync 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
                    }
                    else
                    {
                        asm volatile("bar.arrive 6, %0;\n" ::"n"(kPvBarThreads) : "memory");
                    }
                    if (kWarpMma ? warp == 0 : tid == 0)
                    {
                        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                        uint32_t const tm_u = tmem;
                        if (kWarpMma ? elect_one() : true)
                        {
#pragma unroll
                            for (int h = 0; h < BPC; ++h)
                            {
                                uint32_t const a0 = smem_u32(smem + S::kV16 + h * S::kTileBytes);
                                uint32_t const b0 = smem_u32(smem + S::kP + h * kRows * kPage * 2);
#pragma unroll
                                for (int ks = 0; ks < kPage / 16; ++ks)
                                {
                                    umma_f16(tm_u + (BPC + h) * kRows, umma_desc(a0 + ks * 4096, 2048, 128),
                                        umma_desc(b0 + ks * 256, 128, 2048), umma_idesc(kRows, 1, 1), ks > 0);
                                }
                            }
                            umma_commit(mma_bar);
                        }
                        if (kWarpMma)
                        {
                            __syncwarp();
                        }
                    }
                    M3FA_PROF_T(19, 0);
#pragma unroll
                    for (int k = 0; k < kPairsPerGroup; ++k)
                    {
                        int const q = g + 4 * k;
                        if (q < kPairs)
                        {
                            int const h = q % BPC;
                            int const c = q / BPC;
                            float const lw = warp_reduce_scatter16<false>(pa[k], lane);
                            if ((lane & 1) == 0)
                            {
                                red_l[((h * QG + c) * 16 + rs_row) * 4 + wq] = lw;
                            }
                        }
                    }
                    group_bar();
                    M3FA_PROF_T(20, 0);
                    if (wq == 0 && lane < 16)
                    {
#pragma unroll
                        for (int k = 0; k < kPairsPerGroup; ++k)
                        {
                            int const q = g + 4 * k;
                            if (q < kPairs)
                            {
                                int const h = q % BPC;
                                int const c = q / BPC;
                                int const r = (h * QG + c) * 16 + lane;
                                float4 const mq = *reinterpret_cast<float4 const*>(red_m + r * 4);
                                float4 const lq = *reinterpret_cast<float4 const*>(red_l + r * 4);
                                float const m = fmaxf(fmaxf(mq.x, mq.y), fmaxf(mq.z, mq.w));
                                float const l = (lq.x + lq.y) + (lq.z + lq.w);
                                float const lse = l > 0.0F ? m + lg2(l) : kNegInf;
                                float const scale = lse > kNegInf ? ex2(m - lse) : 0.0F;
                                row_scale[h * kRows + c * 16 + lane] = scale * vgs;
                                row_lse[h * kRows + c * 16 + lane] = lse;
                            }
                        }
                    }
                }
                __syncthreads(); // row scales and LSEs visible (PV was issued above)
                M3FA_PROF(9);
                if (warp < kComputeWarps && warp / 4 < kPairs)
                {
                    mbar_wait_cta(mma_bar, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
                }
            }
        }
        M3FA_PROF(3);
    }
    else
    {
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        if (OPJ && M3FA_C1_OPISSUE == 2 && tid == 32)
        {
            op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
        }
#if M3FA_TRIGGER_AT == 2 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        if constexpr (kAttnTrigger || (FQKV && M3FA_MEGA_TRIGGER == 1))
        {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
#endif
        for (int r = tid; r < BPC * kRows; r += kThreads)
        {
            row_lse[r] = kNegInf;
            row_scale[r] = 0.0F;
        }
        __syncthreads();
    }

    // ---------------- Push every partial row to the CTA that merges its head (CTA r merges heads r*BPC..) --------
    if constexpr (S::kRecvAlias)
    {
        // Peers' partials overlay this CTA's tiles: every CTA's PV MMAs have completed.
        asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
    }
    if (!(kEarlyClw && !S::kRecvAlias))
    {
        asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory"); // peers' mbarriers are initialized
    }
    M3FA_PROF(4);
    if (OPJ && M3FA_C1_OPISSUE == 4 && tid == 32)
    {
        op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
    }
    uint32_t const recv_base = smem_u32(recv);
    if (kRs2 && warp < 8)
    {
        // Split push: warp group rg pushes rows 8 rg .. 8 rg + 7 (row j goes to CTA j, which merges head j).
        int const rg = warp / 4;
        int const d = (warp % 4) * 32 + lane;
        float v[8];
        if (any_valid)
        {
            tmem_ld8(tmem + ((warp % 4) * 32 << 16) + BPC * kRows + 8 * rg, v);
        }
        else
        {
#pragma unroll
            for (int j2 = 0; j2 < 8; ++j2)
            {
                v[j2] = 0.0F;
            }
        }
#pragma unroll
        for (int j2 = 0; j2 < 8; ++j2)
        {
            v[j2] *= row_scale[8 * rg + j2];
        }
#pragma unroll
        for (int j2 = 0; j2 < 8; ++j2)
        {
            uint32_t const dst_cta = 8 * rg + j2;
            uint32_t const row_addr = recv_base + static_cast<uint32_t>((cr * kOPitch + d) * 4);
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                             mapa_shared(row_addr, dst_cta)),
                         "r"(__float_as_uint(v[j2])), "r"(mapa_shared(mbar, dst_cta))
                         : "memory");
        }
    }
    else if (!kRs2 && warp < kComputeWarps && warp / 4 < kPairs)
    {
        int const g = warp / 4;
        int const d = (warp % 4) * 32 + lane; // output dim
#pragma unroll
        for (int k = 0; k < kPairsPerGroup; ++k)
        {
            int const q = g + 4 * k;
            if (q < kPairs)
            {
                int const h = q % BPC;
                int const c = q / BPC;
                int const chunk = cr * BPC + h;
                float v[16];
                if (any_valid && kRegCore)
                {
                    float const* obuf
                        = reinterpret_cast<float const*>(smem + S::kK16 + h * S::kTileBytes + S::kObufOff);
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        v[j] = obuf[(c * 16 + j) * kOPitch + d];
                    }
                }
                else if (any_valid)
                {
                    tmem_ld16(tmem + ((warp % 4) * 32 << 16) + (BPC + h) * kRows + c * 16, v);
                }
                else
                {
#pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        v[j] = 0.0F;
                    }
                }
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    v[j] *= row_scale[h * kRows + c * 16 + j];
                }
                // One scalar store per row: a vector store's lanes would address several destination CTAs at once,
                // which measured several times slower.
#pragma unroll
                for (int j = 0; j < 16; ++j)
                {
                    uint32_t const dst_cta = j / BPC;
                    uint32_t const row_addr
                        = recv_base + static_cast<uint32_t>((((chunk * QG + c) * BPC + j % BPC) * kOPitch + d) * 4);
                    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                                     mapa_shared(row_addr, dst_cta)),
                                 "r"(__float_as_uint(v[j])), "r"(mapa_shared(mbar, dst_cta))
                                 : "memory");
                }
            }
        }
    }
    else if (tid >= kComputeThreads)
    {
        // Every partial row's LSE (BPC * kRows of them; more than the producer threads for the widest geometries).
        for (int r = tid - kComputeThreads; r < BPC * kRows; r += kThreads - kComputeThreads)
        {
            int const h = r / kRows;
            int const c = (r % kRows) / 16;
            int const j = r % 16;
            uint32_t const dst_cta = j / BPC;
            uint32_t const row_addr
                = recv_base + static_cast<uint32_t>((((cr * BPC + h) * QG + c) * BPC + j % BPC) * kOPitch * 4);
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(
                             mapa_shared(row_addr + kHeadDim * 4, dst_cta)),
                         "r"(__float_as_uint(row_lse[r])), "r"(mapa_shared(mbar, dst_cta))
                         : "memory");
        }
    }
    M3FA_PROF(10);
    if (OPJ && M3FA_C1_OPISSUE == 5 && tid == 32)
    {
        op_issue<S>(p, smem, static_cast<int>(blockIdx.x));
    }
#pragma unroll
    for (int k = 0; k < kMaxT; ++k)
    {
        if (wr_kind[k] == 1)
        {
            *reinterpret_cast<uint16_t*>(wr_ptr[k]) = static_cast<uint16_t>(wr_val[k] & 0xFFFFu);
            if (lane % 4 == 0)
            {
                *wr_sptr[k] = static_cast<uint8_t>(wr_val[k] >> 16);
            }
        }
        else if (wr_kind[k] == 2)
        {
            *reinterpret_cast<uint32_t*>(wr_ptr[k]) = wr_val[k];
        }
    }
    {
        uint32_t ready = 0;
#ifdef M3FA_DBG_HANG
        long long spins = 0;
#endif
        while (!ready)
        {
#ifdef M3FA_DBG_HANG
            if (++spins == (1LL << 24))
            {
                printf("M3FA hang: block %d thread %d merge recv\n", blockIdx.x, threadIdx.x);
            }
#endif
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
                " selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(ready)
                : "r"(mbar)
                : "memory");
        }
    }
    M3FA_PROF(5);

    // ---------------- Merge heads [cr * BPC, cr * BPC + BPC) (split-K merge kernel's LSE weighting) ----------------
    constexpr int kMergeRows = QG * BPC; // (token, local head) pairs, 16 chunk weights each
    // Wide geometries: chunk weights once per merged row (into the softmax scratch, free by now), then the weighted
    // sums. Narrow ones recompute the weights per output element: no extra barrier on the critical path.
    constexpr bool kRowWeights = kMergeRows >= 4;
    float* wmerge = red_m; // [kMergeRows][kBlocks]; red_m holds BPC * kRows * 4 = kMergeRows * 64 floats
    [[maybe_unused]] auto chunk_weights = [&](int mr, float(&lc)[kBlocks])
    {
        float lmax = kNegInf;
#pragma unroll
        for (int c = 0; c < kBlocks; ++c)
        {
            lc[c] = recv[(c * kMergeRows + mr) * kOPitch + kHeadDim];
            lmax = fmaxf(lmax, lc[c]);
        }
        lmax = lmax == kNegInf ? 0.0F : lmax;
        float denom = 0.0F;
#pragma unroll
        for (int c = 0; c < kBlocks; ++c)
        {
            lc[c] = ex2(lc[c] - lmax); // now the chunk weight; 0 for an empty chunk
            denom += lc[c];
        }
        denom = denom > 0.0F ? denom : 1.0F;
#pragma unroll
        for (int c = 0; c < kBlocks; ++c)
        {
            // -0.0 marks an empty chunk (skipped below, as the merge kernel skips zero weights).
            lc[c] = lc[c] != 0.0F ? __fdividef(lc[c], denom) : -0.0F;
        }
    };
    if constexpr (kRowWeights)
    {
        for (int mr = tid; mr < kMergeRows; mr += kThreads)
        {
            float lc[kBlocks];
            chunk_weights(mr, lc);
#pragma unroll
            for (int c = 0; c < kBlocks; ++c)
            {
                wmerge[mr * kBlocks + c] = lc[c];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int idx = tid; idx < kMergeRows * kHeadDim; idx += kThreads)
    {
        int const mr = idx >> 7;
        int const d = idx & (kHeadDim - 1);
        float wn[kBlocks];
        if constexpr (kRowWeights)
        {
#pragma unroll
            for (int c = 0; c < kBlocks; ++c)
            {
                wn[c] = wmerge[mr * kBlocks + c];
            }
        }
        else
        {
#ifndef M3FA_MEGA_MERGE_LOCAL
#define M3FA_MEGA_MERGE_LOCAL                                                                                          \
    0 // 1 (measured +0.6 us slower): every merging thread derives the 16 chunk weights itself (broadcast shared loads,
      // independent exps): no shuffle chain; the same arithmetic as chunk_weights
#endif
            if constexpr (M3FA_MEGA_MERGE_LOCAL != 0)
            {
                chunk_weights(mr, wn);
            }
            else
            {
                // Narrow geometries: the warp (32 dims of one merged row) derives the 16 chunk weights together, lane c
                // holding chunk c's, and broadcasts them.
                float lc = (lane < kBlocks) ? recv[(lane * kMergeRows + mr) * kOPitch + kHeadDim] : kNegInf;
                float lmax = lc;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffffu, lmax, o));
                }
                lmax = lmax == kNegInf ? 0.0F : lmax;
                float const wc = lane < kBlocks ? ex2(lc - lmax) : 0.0F;
                // The denominator in chunk order from 0.0, as chunk_weights (and the merge kernel) add it: a butterfly
                // reduction rounds differently in the last bit.
                float denom = 0.0F;
#pragma unroll
                for (int c = 0; c < kBlocks; ++c)
                {
                    denom += __shfl_sync(0xffffffffu, wc, c);
                }
                denom = denom > 0.0F ? denom : 1.0F;
                float const wl = wc != 0.0F ? __fdividef(wc, denom) : -0.0F;
#pragma unroll
                for (int c = 0; c < kBlocks; ++c)
                {
                    wn[c] = __shfl_sync(0xffffffffu, wl, c);
                }
            }
        }
        if (tid == 0)
        {
            M3FA_CLK(45);
        }
        float acc = 0.0F;
#pragma unroll
        for (int c = 0; c < kBlocks; ++c)
        {
            float const prod = recv[(c * kMergeRows + mr) * kOPitch + d] * wn[c];
            acc = __float_as_uint(wn[c]) != 0x80000000u ? acc + prod : acc;
        }
        if (tid == 0)
        {
            M3FA_CLK(46);
        }
        int const tl = mr / BPC;
        int const head = cr * BPC + mr % BPC;
        __nv_bfloat16 const ob = __float2bfloat16_rn(acc);
        int const out_row = tok0 + tfirst + tl;
        p.out[static_cast<int64_t>(out_row) * p.out_stride + head * kHeadDim + d] = ob;
        if constexpr (MXFP8)
        {
            float const x = __bfloat162float(ob);
            float amax = fabsf(x);
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
            {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
            }
            float const sfv = amax * rcp_approx_ftz(448.0F);
            int const bits = __float_as_int(sfv);
            int ex = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
            ex = min(ex, 254);
            float scale = __int_as_float((254 - ex) << 23);
            scale = amax != 0.0F ? scale : 0.0F;
            __nv_fp8_e4m3 const q8v(x * scale);
            int64_t const q8off = static_cast<int64_t>(out_row) * (p.num_sf_k_tiles * 128) + head * kHeadDim + d;
            if constexpr (FQKV || OPJ)
            {
                // Megakernel hand-off to the o_proj units (Lamport): this call's parity buffer, the other one reset to
                // 0xFF. 0xFF is never written: a NaN code is canonicalized to 0x7F; the UE8M0 code is at most 254.
                uint8_t const code = q8v.__x == 0xFFu ? static_cast<uint8_t>(0x7Fu) : q8v.__x;
                int64_t const cur = (mega_calls & 1u) * kMaxTok * kOpK;
                int64_t const nxt = ((mega_calls & 1u) ^ 1u) * kMaxTok * kOpK;
                asm volatile(
                    "st.relaxed.gpu.global.u8 [%0], %1;\n" ::"l"(p.q8 + cur + q8off), "h"(static_cast<uint16_t>(code))
                    : "memory");
                asm volatile(
                    "st.relaxed.gpu.global.u8 [%0], %1;\n" ::"l"(p.q8 + nxt + q8off), "h"(static_cast<uint16_t>(0xFFu))
                    : "memory");
            }
            else
            {
                p.q8[q8off] = q8v.__x;
            }
            if (lane == 0)
            {
                int const k_sf = head * (kHeadDim / 32) + (d >> 5);
                int64_t const off = static_cast<int64_t>(out_row / 128) * p.num_sf_k_tiles * 512 + (k_sf / 4) * 512
                    + (out_row % 32) * 16 + ((out_row % 128) / 32) * 4 + k_sf % 4;
                if constexpr (FQKV || OPJ)
                {
                    int64_t const cur = (mega_calls & 1u) * kOpSfBytes;
                    int64_t const nxt = ((mega_calls & 1u) ^ 1u) * kOpSfBytes;
                    asm volatile(
                        "st.relaxed.gpu.global.u8 [%0], %1;\n" ::"l"(p.sf + cur + off), "h"(static_cast<uint16_t>(ex))
                        : "memory");
                    asm volatile("st.relaxed.gpu.global.u8 [%0], %1;\n" ::"l"(p.sf + nxt + off),
                                 "h"(static_cast<uint16_t>(0xFFu))
                                 : "memory");
                }
                else
                {
                    p.sf[off] = static_cast<uint8_t>(ex);
                }
            }
        }
    }
    M3FA_PROF(60);
    if constexpr (OPJ)
    {
        // ---------------- c1: o_proj unit blockIdx.x (v52 m3_mx_gemv bit for bit, mega_v7's phase) ----------------
        if (any_valid)
        {
            asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
        }
        __syncthreads(); // the merge's reads done; the K tiles are free for the activation overlay
        int const op_u = static_cast<int>(blockIdx.x);
        if (M3FA_C1_OPISSUE == 3 && tid == 32)
        {
            op_issue<S>(p, smem, op_u);
        }
        uint32_t const tm = *tmem_slot;
        if (warp == 1)
        {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
            if (elect_one())
            {
                op_stream<S>(smem, tm);
            }
            __syncwarp();
        }
        op_act<S>(p, smem, op_u, tid, mega_calls);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        M3FA_PROF_T(36, 0);
        __syncthreads();
        if (warp == 1)
        {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
            if (elect_one())
            {
                op_mma<S>(smem, tm);
                M3FA_PROF_T(50, threadIdx.x);
            }
            __syncwarp();
        }
        if (warp >= kComputeWarps)
        {
            op_epilogue<S>(p, smem, tm, tid, op_u);
        }
        M3FA_PROF_T(37, kComputeThreads);
        __syncthreads();
#if M3FA_C1_TRIGGER == 3 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); // after this CTA's y rows / pushes
#endif
        if (warp == kWarps - 1)
        {
            asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(tm), "r"(S::kTmemCols)
                         : "memory");
        }
        if (tid == 0)
        {
            p.flags[32 + blockIdx.x] = mega_calls + 1u;
        }
        M3FA_PROF(6);
        return;
    }
    if (any_valid)
    {
        // Every TMEM read completed above (tcgen05.wait::ld); release the columns.
        asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    }
    __syncthreads();
    if (FQKV && tid == 0)
    {
        p.flags[32 + blockIdx.x] = mega_calls + 1u;
    }
    M3FA_PROF(61);
    if (warp == kWarps - 1)
    {
        asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(*tmem_slot), "r"(S::kTmemCols)
                     : "memory");
    }
    M3FA_PROF(6);
#else
    __trap();
#endif // MINIMAX_M3_ATTN_DEVICE_ENABLED
}

template <int QLEN, int QG, int BPC, bool MXFP8, bool FQKV, bool OPJ = false>
cudaError_t launch_impl(Params const& prm, int num_reqs, bool pdl, cudaStream_t stream, int* max_clusters)
{
    auto kernel = m3_fused_decode_attn_kernel<QLEN, QG, BPC, MXFP8, FQKV, OPJ>;
    constexpr int smem = std::conditional_t<OPJ, SmemC1<QLEN, QG, BPC>, SmemX<QLEN, QG, BPC, FQKV>>::kBytes;
    constexpr int kCs = kBlocks / BPC;
    static bool const configured = [&]()
    {
        bool ok = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
        ok = ok
            && (kCs <= 8
                || cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess);
        if (!ok)
        {
            cudaGetLastError();
        }
        return ok;
    }();
    if (!configured)
    {
        return cudaErrorInvalidConfiguration;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(FQKV ? kQkvGrid : kCs * (QLEN / QG) * num_reqs);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kCs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    if (max_clusters != nullptr)
    {
        cfg.numAttrs = 1;
        cudaError_t const err = cudaOccupancyMaxActiveClusters(max_clusters, kernel, &cfg);
        if (err != cudaSuccess)
        {
            cudaGetLastError();
        }
        return err;
    }
    cfg.numAttrs = 2;
    return cudaLaunchKernelEx(&cfg, kernel, prm);
}

// The one geometry the target launches: one request verifying 6 query tokens, 6 clusters of 16 CTAs (96 o_proj
// units).
cudaError_t dispatch(Params const& prm, int qlen, int qg, int bpc, int num_reqs, bool pdl, cudaStream_t stream)
{
    if (qlen == 6 && qg == 1 && bpc == 1)
    {
        return launch_impl<6, 1, 1, true, false, true>(prm, num_reqs, pdl, stream, nullptr);
    }
    return cudaErrorInvalidValue;
}

static_assert(kMaxTok == minimax_m3_attn_c1::kMaxTokens && kOpN == minimax_m3_attn_c1::kOpN
        && kOpK == minimax_m3_attn_c1::kOpK && kOpSfBytes == minimax_m3_attn_c1::kOpSfBytes
        && kOpN / 128 * kOpSplits == minimax_m3_attn_c1::kCtas,
    "header dimensions");

} // namespace

namespace
{
// TMA descriptor of a [n, k] E4M3 weight (128 x 128 boxes, 128-byte swizzle), cached per weight address.
bool weight_map(CUtensorMap& map, void const* w, int k, int n)
{
    static std::mutex mu;
    static std::unordered_map<void const*, CUtensorMap> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(w);
    if (it != cache.end())
    {
        map = it->second;
        return true;
    }
    static PFN_cuTensorMapEncodeTiled_v12000 encode = []()
    {
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult q;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
    }();
    if (encode == nullptr)
    {
        return false;
    }
    cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(n)};
    cuuint64_t strides[1] = {static_cast<cuuint64_t>(k)};
    cuuint32_t box[2] = {128, 128};
    cuuint32_t elem[2] = {1, 1};
    CUresult const r = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(w), dims, strides, box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
    {
        return false;
    }
    cache.emplace(w, map);
    return true;
}
} // namespace

cudaError_t invokeMinimaxM3AttnC1(MinimaxM3AttnC1Params& params, void const* oWeight, int qlen, int qg, int bpc,
    int numReqs, bool pdl, cudaStream_t stream)
{
    if (!weight_map(params.wmap_o, oWeight, kOpK, kOpN))
    {
        return cudaErrorInvalidValue;
    }
    return dispatch(params, qlen, qg, bpc, numReqs, pdl, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
