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

// Kimi K3 noaux_tc routing (top 16 of 896 experts), bit-identical to noaux_tc_op on FP32 logits.
//
// The kernel comes from an automated kernel search against a bit-exact PyTorch reference of noaux_tc_op: two variants
// (the latency schedule and the streaming schedule below) share the selection algorithm. Here they gain PDL and arch
// guards; the outputs were re-validated bit for bit against noaux_tc_op on Rubin.
//
// One warp per token, persistent grid of one CTA per SM. Rows are read with coalesced 16-byte loads (lane l owns
// float4 slots l + 32j, j < 7). Two load schedules:
// - latency (<= 2048 tokens, 24 warps): the first row is loaded into registers before the setup work, the next row
//   while the current one is selected, and the row after that is prefetched into L2;
// - streaming (> 2048 tokens, 32 warps): each row is prefetched into L2 one grid stride ahead and loaded into
//   registers at the top of the row loop, which keeps the kernel within 64 registers for 32 warps per SM.
//
// Phase A (all 896 experts): approximate key ka = 0.5 * tanh.approx(0.5x) + (0.5 + b) with |ka - k| <= eps for the
//   exact key k. The 16th largest of the 32 lane maxima (bitonic split) is a lower bound L of the 16th largest
//   approximate key, so every expert of the exact top 16 has ka >= L - 2 eps. Those candidates (about 21 per token
//   for router-like logits) are gathered into a per-warp shared-memory chunk (ids only, prefix sum by shuffles).
// Phase B (candidates only): exact s = 0.5f * tanhf(0.5f * x) + 0.5f (accurate tanhf, as noaux_tc_op), exact key
//   s + b, rank by (key descending, expert id ascending), then noaux_tc_op's tree sum of the 16 ranked scores (xor
//   butterfly over 16 lanes) and the weight s / S: the correctly rounded float quotient, which equals noaux_tc_op's
//   float(double(s) / (double(S) + 1e-20)) for S >= 2^-13 (no FP64 instruction runs on that path); the FP64
//   expression below 2^-13.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/kimiK3NoAuxTcKf.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kNumExperts = 896;
constexpr int kNumExperts4 = kNumExperts / 4; // float4 slots per row
constexpr int kTopK = 16;
constexpr int kLatencyWarps = 24;
constexpr int kStreamingWarps = 32;
constexpr int64_t kStreamingMinTokens = 2049;

template <int kWarps>
struct __align__(16) KfSmem
{
    float4 biasHalf4[kNumExperts4];               // 0.5 + bias in float4 slot order
    float bias[kNumExperts];                      // the correction bias
    alignas(16) float candKey[kWarps][32];        // exact keys of the current candidate chunk
    alignas(16) int candId[kWarps][32];           // expert ids of the current candidate chunk
    alignas(16) float topKey[kWarps][kTopK];      // running top-16 exact keys (general path)
    alignas(16) float2 topScoreId[kWarps][kTopK]; // ranked (sigmoid score, expert id bits)
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)

constexpr int kExpertsPerLane = kNumExperts / 32;
constexpr uint32_t kRowBytes = kNumExperts * 4;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float tanhApprox(float x)
{
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Packed FP32x2 (SM100+): round to nearest, no flush to zero.
__device__ __forceinline__ float2 mul2(float2 a, float2 b)
{
    float2 d;
    asm("{.reg .b64 ra, rb, rd;\n mov.b64 ra, {%2,%3};\n mov.b64 rb, {%4,%5};\n"
        " mul.rn.f32x2 rd, ra, rb;\n mov.b64 {%0,%1}, rd;}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return d;
}

__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c)
{
    float2 d;
    asm("{.reg .b64 ra, rb, rc, rd;\n mov.b64 ra, {%2,%3};\n mov.b64 rb, {%4,%5};\n"
        " mov.b64 rc, {%6,%7};\n fma.rn.f32x2 rd, ra, rb, rc;\n mov.b64 {%0,%1}, rd;}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
    return d;
}

__device__ __forceinline__ float2 sub2(float2 a, float2 b)
{
    float2 d;
    asm("{.reg .b64 ra, rb, rd;\n mov.b64 ra, {%2,%3};\n mov.b64 rb, {%4,%5};\n"
        " sub.rn.f32x2 rd, ra, rb;\n mov.b64 {%0,%1}, rd;}"
        : "=f"(d.x), "=f"(d.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return d;
}

__device__ __forceinline__ void prefetchRowL2(float const* p)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(kRowBytes) : "memory");
}

// noaux_tc_op's sigmoid: 0.5f * tanhf(0.5f * x) + 0.5f with the accurate (not fast-math) tanhf.
__device__ __forceinline__ float exactSigmoid(float x)
{
    return __fmaf_rn(0.5f, tanhf(__fmul_rn(0.5f, x)), 0.5f);
}

// Monotone float -> uint32 map (total order consistent with float < for non-NaN values).
__device__ __forceinline__ uint32_t orderedKey(float f)
{
    uint32_t const u = __float_as_uint(f);
    return u ^ (static_cast<uint32_t>(static_cast<int32_t>(u) >> 31) | 0x80000000u);
}

__device__ __forceinline__ float unorderedKey(uint32_t o)
{
    uint32_t const u = (o & 0x80000000u) ? (o ^ 0x80000000u) : ~o;
    return __uint_as_float(u);
}

// noaux_tc_op's total order: key descending, then expert id ascending.
__device__ __forceinline__ int beats(float ka, int ia, float kb, int ib)
{
    return (ka > kb) || (ka == kb && ia < ib);
}

// Element i (0..27) of lane l is expert 4 * (l + 32 * (i / 4)) + i % 4.
__device__ __forceinline__ int expertOf(int lane, int i)
{
    return 4 * lane + 128 * (i >> 2) + (i & 3);
}

#endif // SM10x

template <int kWarps, bool kStreaming>
__global__ void __launch_bounds__(kWarps * 32, 1) kimiK3NoAuxTcKfKernel(float const* __restrict__ logits,
    float const* __restrict__ biasGlobal, float* __restrict__ weightsOut, int* __restrict__ idsOut, int numTokens)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
    constexpr int kThreads = kWarps * 32;
    extern __shared__ __align__(128) unsigned char smemRaw[];
    KfSmem<kWarps>& sm = *reinterpret_cast<KfSmem<kWarps>*>(smemRaw);

    // The logits come from the previous kernel (router GEMM).
    cudaGridDependencySynchronize();

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const numWarpsTotal = gridDim.x * kWarps;
    int const globalWarp = warp * gridDim.x + blockIdx.x;
    float4 const* logits4 = reinterpret_cast<float4 const*>(logits);

    // Start reading the first row before the setup work: into registers (latency schedule) or into L2 (streaming).
    float4 xr[kExpertsPerLane / 4];
    if constexpr (kStreaming)
    {
        if (lane == 0 && globalWarp < numTokens)
        {
            prefetchRowL2(logits + static_cast<size_t>(globalWarp) * kNumExperts);
        }
    }
    else if (globalWarp < numTokens)
    {
#pragma unroll
        for (int j = 0; j < kExpertsPerLane / 4; ++j)
        {
            xr[j] = __ldg(logits4 + static_cast<size_t>(globalWarp) * kNumExperts4 + lane + 32 * j);
        }
        if (lane == 0 && globalWarp + numWarpsTotal < numTokens)
        {
            prefetchRowL2(logits + static_cast<size_t>(globalWarp + numWarpsTotal) * kNumExperts);
        }
    }

    float* biasHalf = reinterpret_cast<float*>(sm.biasHalf4);
    for (int i = threadIdx.x; i < kNumExperts; i += kThreads)
    {
        float const b = biasGlobal[i];
        sm.bias[i] = b;
        biasHalf[i] = 0.5f + b;
    }
    __syncthreads();

    // eps from max |b|: tanh.approx relative error 2^-11 (halved by the 0.5 factor) plus the rounding terms,
    // rounded up.
    float biasMax = 0.f;
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i)
    {
        biasMax = fmaxf(biasMax, fabsf(sm.bias[lane * kExpertsPerLane + i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        biasMax = fmaxf(biasMax, __shfl_xor_sync(kFullMask, biasMax, o));
    }
    float const eps = __fmaf_ru(0x1p-21f, __fadd_ru(1.0f, biasMax), 0x1p-12f);
    float const twoEps = 2.0f * eps;
    float2 const half2 = make_float2(0.5f, 0.5f);
    // Bitonic direction of each of the 10 compare steps (sizes 2..16): bit set = keep the maximum.
    uint32_t dirBits = 0;
    {
        int st = 0;
#pragma unroll
        for (int size = 2; size <= 16; size <<= 1)
        {
#pragma unroll
            for (int stride = size >> 1; stride > 0; stride >>= 1, ++st)
            {
                if (((lane & stride) == 0) != ((lane & size) != 0))
                {
                    dirBits |= 1u << st;
                }
            }
        }
    }

    float* candKeyW = sm.candKey[warp];
    int* candIdW = sm.candId[warp];
    float* topKeyW = sm.topKey[warp];
    float2* topScoreIdW = sm.topScoreId[warp];

    for (int row = globalWarp; row < numTokens; row += numWarpsTotal)
    {
        float const* logitsRow = logits + static_cast<size_t>(row) * kNumExperts;
        if constexpr (kStreaming)
        {
            // This row (prefetched into L2 one grid stride ago) into registers, the next one into L2.
#pragma unroll
            for (int j = 0; j < kExpertsPerLane / 4; ++j)
            {
                xr[j] = __ldg(logits4 + static_cast<size_t>(row) * kNumExperts4 + lane + 32 * j);
            }
            if (lane == 0 && row + numWarpsTotal < numTokens)
            {
                prefetchRowL2(logits + static_cast<size_t>(row + numWarpsTotal) * kNumExperts);
            }
        }

        // ---------------- Phase A: approximate keys ----------------
        float2 ka[kExpertsPerLane / 2];
#pragma unroll
        for (int j = 0; j < kExpertsPerLane / 4; ++j)
        {
            float4 const v = xr[j];
            float4 const cc = sm.biasHalf4[lane + 32 * j];
            float2 h0 = mul2(make_float2(v.x, v.y), half2);
            float2 h1 = mul2(make_float2(v.z, v.w), half2);
            h0.x = tanhApprox(h0.x);
            h0.y = tanhApprox(h0.y);
            h1.x = tanhApprox(h1.x);
            h1.y = tanhApprox(h1.y);
            ka[2 * j + 0] = fma2(h0, half2, make_float2(cc.x, cc.y));
            ka[2 * j + 1] = fma2(h1, half2, make_float2(cc.z, cc.w));
        }
        // Latency schedule: the next row into registers, the one after into L2.
        if constexpr (!kStreaming)
        {
            int const nextRow = row + numWarpsTotal;
            if (nextRow < numTokens)
            {
#pragma unroll
                for (int j = 0; j < kExpertsPerLane / 4; ++j)
                {
                    xr[j] = __ldg(logits4 + static_cast<size_t>(nextRow) * kNumExperts4 + lane + 32 * j);
                }
                if (lane == 0 && nextRow + numWarpsTotal < numTokens)
                {
                    prefetchRowL2(logits + static_cast<size_t>(nextRow + numWarpsTotal) * kNumExperts);
                }
            }
        }
        float m = fmaxf(ka[0].x, ka[0].y);
#pragma unroll
        for (int i = 1; i < kExpertsPerLane / 2; ++i)
        {
            m = fmaxf(m, fmaxf(ka[i].x, ka[i].y));
        }

        // L = 16th largest lane maximum: bitonic-sort both 16-lane halves in opposite directions, split (lanes < 16
        // then hold the 16 largest) and take their minimum.
        {
            int st = 0;
#pragma unroll
            for (int size = 2; size <= 16; size <<= 1)
            {
#pragma unroll
                for (int stride = size >> 1; stride > 0; stride >>= 1, ++st)
                {
                    float const p = __shfl_xor_sync(kFullMask, m, stride);
                    m = ((dirBits >> st) & 1u) ? fmaxf(m, p) : fminf(m, p);
                }
            }
        }
        m = fmaxf(m, __shfl_xor_sync(kFullMask, m, 16));
        uint32_t const lo = __reduce_min_sync(kFullMask, lane < kTopK ? orderedKey(m) : 0xffffffffu);
        float const theta = __fsub_rd(unorderedKey(lo), twoEps);

        // Candidate mask: bit (27 - i) set <=> ka[i] >= theta (sign bit of ka - theta clear).
        uint32_t negMask = 0;
        float2 const theta2 = make_float2(theta, theta);
#pragma unroll
        for (int i = 0; i < kExpertsPerLane / 2; ++i)
        {
            float2 const d = sub2(ka[i], theta2);
            negMask = __funnelshift_l(__float_as_uint(d.x), negMask, 1);
            negMask = __funnelshift_l(__float_as_uint(d.y), negMask, 1);
        }
        uint32_t const mask = ~negMask & ((1u << kExpertsPerLane) - 1u);
        int const cnt = __popc(mask);
        int incl = cnt;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int const t = __shfl_up_sync(kFullMask, incl, o);
            if (lane >= o)
            {
                incl += t;
            }
        }
        int const pre = incl - cnt;
        int const numCand = __shfl_sync(kFullMask, incl, 31);
        bool const fast = numCand <= 32;

        // Every output slot holds a valid expert id even when NaN keys (all comparisons false) leave ranks unfilled;
        // for finite logits the selection below overwrites all 16 slots.
        if (lane < kTopK)
        {
            topScoreIdW[lane] = make_float2(0.f, __int_as_float(lane));
        }

        // ---------------- Phase B: exact keys and ranks ----------------
        if (fast)
        {
            // Gather the ids of this lane's candidates into chunk slots [pre, pre + cnt).
            {
                uint32_t mk = mask;
                int pos = pre;
                while (mk)
                {
                    int const lz = __clz(mk);
                    mk &= ~(0x80000000u >> lz);
                    candIdW[pos++] = expertOf(lane, lz - 4);
                }
            }
            __syncwarp();
            bool const valid = lane < numCand;
            float s = 0.f;
            float kf = -INFINITY;
            int idx = 0x7fffffff;
            if (valid)
            {
                idx = candIdW[lane];
                s = exactSigmoid(__ldg(logitsRow + idx));
                kf = __fadd_rn(s, sm.bias[idx]);
            }
            candKeyW[lane] = kf;
            __syncwarp();
            // rank = #{key > kf}: sign bits of kf - key_j.
            int rank = 0;
            float2 const kf2 = make_float2(kf, kf);
            float4 const* candKey4 = reinterpret_cast<float4 const*>(candKeyW);
            for (int j = 0; j < numCand; j += 4)
            {
                float4 const v = candKey4[j >> 2];
                float2 const d0 = sub2(kf2, make_float2(v.x, v.y));
                float2 const d1 = sub2(kf2, make_float2(v.z, v.w));
                rank += static_cast<int>((__float_as_uint(d0.x) >> 31) + (__float_as_uint(d0.y) >> 31)
                    + (__float_as_uint(d1.x) >> 31) + (__float_as_uint(d1.y) >> 31));
            }
            // Exact key ties are resolved by expert id (only when two keys are equal).
            uint32_t const seen = __reduce_or_sync(kFullMask, valid ? (1u << rank) : 0u);
            if (__popc(seen) != numCand)
            {
                for (int j = 0; j < numCand; ++j)
                {
                    rank += (candKeyW[j] == kf) && (candIdW[j] < idx);
                }
            }
            if (valid && rank < kTopK)
            {
                topScoreIdW[rank] = make_float2(s, __int_as_float(idx));
            }
        }
        else
        {
            // General path (many near-ties): chunks of 32 candidates merged into a running top 16 under the
            // (key descending, id ascending) order.
            int nt = 0;
            for (int base = 0; base < numCand; base += 32)
            {
                {
                    uint32_t mk = mask;
                    int pos = pre;
                    while (mk && pos < base + 32)
                    {
                        int const lz = __clz(mk);
                        mk &= ~(0x80000000u >> lz);
                        if (pos >= base)
                        {
                            candIdW[pos - base] = expertOf(lane, lz - 4);
                        }
                        ++pos;
                    }
                }
                __syncwarp();
                int const nv = min(32, numCand - base);
                float s = 0.f;
                float kf = -INFINITY;
                int idx = 0x7fffffff;
                if (lane < nv)
                {
                    idx = candIdW[lane];
                    s = exactSigmoid(__ldg(logitsRow + idx));
                    kf = __fadd_rn(s, sm.bias[idx]);
                }
                candKeyW[lane] = kf;
                float ok = -INFINITY;
                float os = 0.f;
                int oi = 0x7fffffff;
                if (lane < nt)
                {
                    ok = topKeyW[lane];
                    float2 const q = topScoreIdW[lane];
                    os = q.x;
                    oi = __float_as_int(q.y);
                }
                __syncwarp();
                int rank = 0;
                int orank = lane;
                for (int j = 0; j < nv; ++j)
                {
                    float const kj = candKeyW[j];
                    int const ij = candIdW[j];
                    rank += beats(kj, ij, kf, idx);
                    orank += beats(kj, ij, ok, oi);
                }
                for (int j = 0; j < nt; ++j)
                {
                    rank += beats(topKeyW[j], __float_as_int(topScoreIdW[j].y), kf, idx);
                }
                __syncwarp();
                if (lane < nv && rank < kTopK)
                {
                    topKeyW[rank] = kf;
                    topScoreIdW[rank] = make_float2(s, __int_as_float(idx));
                }
                if (lane < nt && orank < kTopK)
                {
                    topKeyW[orank] = ok;
                    topScoreIdW[orank] = make_float2(os, __int_as_float(oi));
                }
                nt = min(kTopK, nt + nv);
                __syncwarp();
            }
        }
        __syncwarp();

        // ---------------- Output: noaux_tc_op's tree sum and weights ----------------
        float2 const me = topScoreIdW[lane & (kTopK - 1)];
        float const r = me.x;
        // xor butterfly over lanes 0..15 (strides 8, 4, 2, 1) == noaux_tc_op's pairwise tree.
        float sum = __fadd_rn(r, __shfl_xor_sync(kFullMask, r, 8));
        sum = __fadd_rn(sum, __shfl_xor_sync(kFullMask, sum, 4));
        sum = __fadd_rn(sum, __shfl_xor_sync(kFullMask, sum, 2));
        sum = __fadd_rn(sum, __shfl_xor_sync(kFullMask, sum, 1));
        if (lane < kTopK)
        {
            float w;
            if (sum >= 0x1p-13f)
            {
                w = __fdiv_rn(r, sum);
            }
            else
            {
                w = static_cast<float>(static_cast<double>(r) / (static_cast<double>(sum) + 1e-20));
            }
            weightsOut[static_cast<size_t>(row) * kTopK + lane] = w;
            idsOut[static_cast<size_t>(row) * kTopK + lane] = __float_as_int(me.y);
        }
        __syncwarp();
    }
    // No explicit PDL trigger: the dependents launch when the grid completes. A trigger would only launch them sooner:
    // their griddepcontrol.wait returns once the grid has completed and its stores are visible either way.
#endif // SM10x
}

} // namespace

void invokeKimiK3NoAuxTcKf(float const* scores, float const* bias, float* topkValues, int32_t* topkIndices,
    int64_t numTokens, cudaStream_t stream)
{
    if (numTokens <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(numTokens <= INT32_MAX, "invokeKimiK3NoAuxTcKf: too many tokens (%ld)", numTokens);
    int const smVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(
        smVersion >= 100 && smVersion < 110, "invokeKimiK3NoAuxTcKf requires an SM10x GPU (got SM%d)", smVersion);
    TLLM_CHECK_WITH_INFO(
        reinterpret_cast<uintptr_t>(scores) % 16 == 0, "invokeKimiK3NoAuxTcKf: scores must be 16-byte aligned");

    bool const streaming = numTokens >= kStreamingMinTokens;
    int const numBlocks
        = static_cast<int>(std::min<int64_t>(tensorrt_llm::common::getMultiProcessorCount(), numTokens));
    cudaLaunchConfig_t config;
    config.gridDim = numBlocks;
    config.blockDim = (streaming ? kStreamingWarps : kLatencyWarps) * 32;
    config.dynamicSmemBytes = streaming ? sizeof(KfSmem<kStreamingWarps>) : sizeof(KfSmem<kLatencyWarps>);
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    auto const kernel
        = streaming ? kimiK3NoAuxTcKfKernel<kStreamingWarps, true> : kimiK3NoAuxTcKfKernel<kLatencyWarps, false>;
    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, kernel, scores, bias, topkValues, topkIndices, static_cast<int>(numTokens)));
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
