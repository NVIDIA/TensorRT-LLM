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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedResidualNormQuantKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <map>
#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kBlock = 128;
// bf16 elements per 16-byte vector.
constexpr int kVec = 8;
// Hidden sizes are multiples of kHiddenStep (one 16-byte vector per thread) up to kMaxHidden.
constexpr int kHiddenStep = kBlock * kVec;
constexpr int kMaxHidden = 8 * kHiddenStep;
// Widest rows of the persistent kernel.
constexpr int kMaxPersistentHidden = 5 * kHiddenStep;

// fp32 -> bf16 (round to nearest even) -> fp32.
__device__ __forceinline__ float roundBf16(float v)
{
    __nv_bfloat162 const p = __floats2bfloat162_rn(0.0F, v);
    return __uint_as_float(*reinterpret_cast<uint32_t const*>(&p));
}

// One block per row; each thread owns kChunks 16-byte vectors of the row.
template <int HIDDEN, ResidualNormMode MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS>
__global__ void __launch_bounds__(kBlock, 8) residualNormQuantKernel(__nv_bfloat16 const* x, __nv_bfloat16 const* y,
    __nv_bfloat16 const* yBias, float const* gate, float const* weight, float const* bias, float const* invOutScale,
    int rowsPerBatch, float eps, __nv_bfloat16* xOut, __nv_fp8_e4m3* normed)
{
    constexpr int kChunks = HIDDEN / kHiddenStep;
    static_assert(HIDDEN % kHiddenStep == 0, "hidden must be a multiple of 1024");
    constexpr bool kRmsNorm = MODE == ResidualNormMode::kRmsNorm;

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const row = blockIdx.x;
    int const batch = row / rowsPerBatch;
    int64_t const base = static_cast<int64_t>(row) * HIDDEN;
    int64_t const batchBase = static_cast<int64_t>(batch) * HIDDEN;

    // Issue the streaming row loads up front.
    uint4 xr[kChunks];
    uint4 yr[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * kVec;
        xr[c] = *reinterpret_cast<uint4 const*>(x + base + off);
        if constexpr (HAS_Y)
        {
            yr[c] = *reinterpret_cast<uint4 const*>(y + base + off);
        }
    }

    uint4 xb[kChunks];
    float sum = 0.0F;
    float sumSq = 0.0F;
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * kVec;
        if constexpr (HAS_Y)
        {
            uint4 br = make_uint4(0, 0, 0, 0);
            float4 gr[2] = {};
            if constexpr (HAS_YBIAS)
            {
                br = *reinterpret_cast<uint4 const*>(yBias + off);
            }
            if constexpr (HAS_GATE)
            {
                gr[0] = *reinterpret_cast<float4 const*>(gate + batchBase + off);
                gr[1] = *reinterpret_cast<float4 const*>(gate + batchBase + off + 4);
            }
            auto const* xh = reinterpret_cast<__nv_bfloat162 const*>(&xr[c]);
            auto const* yh = reinterpret_cast<__nv_bfloat162 const*>(&yr[c]);
            auto const* bh = reinterpret_cast<__nv_bfloat162 const*>(&br);
            auto const* gv = reinterpret_cast<float const*>(gr);
            __nv_bfloat162 o[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 const xx = __bfloat1622float2(xh[i]);
                float2 yy = __bfloat1622float2(yh[i]);
                if constexpr (HAS_YBIAS)
                {
                    float2 const bb = __bfloat1622float2(bh[i]);
                    yy = make_float2(roundBf16(yy.x + bb.x), roundBf16(yy.y + bb.y));
                }
                float2 r;
                // Round the product before the add; no FMA.
                r.x = HAS_GATE ? __fadd_rn(xx.x, __fmul_rn(yy.x, gv[2 * i])) : __fadd_rn(xx.x, yy.x);
                r.y = HAS_GATE ? __fadd_rn(xx.y, __fmul_rn(yy.y, gv[2 * i + 1])) : __fadd_rn(xx.y, yy.y);
                o[i] = __floats2bfloat162_rn(r.x, r.y);
            }
            xb[c] = *reinterpret_cast<uint4 const*>(o);
            *reinterpret_cast<uint4*>(xOut + base + off) = xb[c];
        }
        else
        {
            xb[c] = xr[c];
        }
        if constexpr (MODE != ResidualNormMode::kResidualOnly)
        {
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 const f = __bfloat1622float2(h[i]);
                sum += f.x;
                sumSq += f.x * f.x;
                sum += f.y;
                sumSq += f.y * f.y;
            }
        }
    }

    if constexpr (MODE != ResidualNormMode::kResidualOnly)
    {
        __shared__ float warpSum[kBlock / 32];
        __shared__ float warpSumSq[kBlock / 32];
        __shared__ float stats[2];
#pragma unroll
        for (int o = 16; o > 0; o /= 2)
        {
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
            sumSq += __shfl_xor_sync(0xffffffffu, sumSq, o);
        }
        if (lane == 0)
        {
            warpSum[warp] = sum;
            warpSumSq[warp] = sumSq;
        }
        __syncthreads();
        if (warp == 0)
        {
            float a = lane < kBlock / 32 ? warpSum[lane] : 0.0F;
            float a2 = lane < kBlock / 32 ? warpSumSq[lane] : 0.0F;
#pragma unroll
            for (int o = 16; o > 0; o /= 2)
            {
                a += __shfl_xor_sync(0xffffffffu, a, o);
                a2 += __shfl_xor_sync(0xffffffffu, a2, o);
            }
            if (lane == 0)
            {
                float const invHidden = 1.0F / static_cast<float>(HIDDEN);
                float const mean = kRmsNorm ? 0.0F : a * invHidden;
                // Single-pass variance can cancel slightly below zero.
                float const var = kRmsNorm ? a2 * invHidden : fmaxf(a2 * invHidden - mean * mean, 0.0F);
                stats[0] = mean;
                stats[1] = rsqrtf(var + eps);
            }
        }
        __syncthreads();
        float const mean = stats[0];
        float const rstd = stats[1];
        float const qmul = invOutScale[0];
        constexpr bool kAda = MODE == ResidualNormMode::kAdaLayerNorm;
        int64_t const modBase = kAda ? batchBase : 0;
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            int const off = (c * kBlock + tid) * kVec;
            float4 wr[2];
            float4 br[2] = {};
            wr[0] = *reinterpret_cast<float4 const*>(weight + modBase + off);
            wr[1] = *reinterpret_cast<float4 const*>(weight + modBase + off + 4);
            if constexpr (!kRmsNorm)
            {
                br[0] = *reinterpret_cast<float4 const*>(bias + modBase + off);
                br[1] = *reinterpret_cast<float4 const*>(bias + modBase + off + 4);
            }
            auto const* wv = reinterpret_cast<float const*>(wr);
            auto const* bv = reinterpret_cast<float const*>(br);
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
            __nv_fp8x2_storage_t packed[4];
#pragma unroll
            for (int i = 0; i < kVec / 2; i++)
            {
                float2 const xf = __bfloat1622float2(h[i]);
                float const xv[2] = {xf.x, xf.y};
                float yv[2];
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    if constexpr (kRmsNorm)
                    {
                        // weight * bf16(x * rstd), rounded to bf16 (a bf16 weight times a bf16 value
                        // is exact in fp32), then the saturating static FP8 quantization.
                        yv[j] = roundBf16(wv[2 * i + j] * roundBf16(xv[j] * rstd)) * qmul;
                    }
                    else
                    {
                        float const xn = (xv[j] - mean) * rstd;
                        float const ww = kAda ? 1.0F + wv[2 * i + j] : wv[2 * i + j];
                        // Round to bf16, then the saturating static FP8 quantization.
                        yv[j] = roundBf16(xn * ww + bv[2 * i + j]) * qmul;
                    }
                }
                packed[i] = __nv_cvt_float2_to_fp8x2(make_float2(yv[0], yv[1]), __NV_SATFINITE, __NV_E4M3);
            }
            *reinterpret_cast<uint2*>(normed + base + off) = *reinterpret_cast<uint2 const*>(packed);
        }
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cpAsync16(void* smem, void const* gmem)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
        "l"(gmem)
        : "memory");
}

__device__ __forceinline__ void cpAsyncCommit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cpAsyncWait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
#endif

// Persistent variant of residualNormQuantKernel for the normalizing modes, with the same per-row
// math and thread mapping (bitwise-identical results). Each block owns a contiguous range of rows
// and streams the x / y rows through a STAGES-deep cp.async ring in shared memory; each thread
// reads back only the bytes it copied, so the ring needs no block barrier. The modulation, gate
// and y bias stay in registers across rows, and each row takes one block barrier: every warp
// repeats warp 0's reduction of the warp partials, which are double-buffered by row parity.
template <int HIDDEN, ResidualNormMode MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS, int STAGES>
__global__ void __launch_bounds__(kBlock, 2) residualNormQuantPersistentKernel(__nv_bfloat16 const* x,
    __nv_bfloat16 const* y, __nv_bfloat16 const* yBias, float const* gate, float const* weight, float const* bias,
    float const* invOutScale, int rows, int rowsPerBatch, float eps, __nv_bfloat16* xOut, __nv_fp8_e4m3* normed)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap(); // The launcher selects this kernel only on SM80+.
#else
    static_assert(
        MODE == ResidualNormMode::kLayerNorm || MODE == ResidualNormMode::kAdaLayerNorm, "LayerNorm modes only");
    constexpr int kChunks = HIDDEN / kHiddenStep;
    constexpr int kElts = kChunks * kVec;
    constexpr int kTensors = HAS_Y ? 2 : 1;
    // uint4 vectors per ring stage.
    constexpr int kStage = kTensors * kChunks * kBlock;
    constexpr bool kAda = MODE == ResidualNormMode::kAdaLayerNorm;
    // Whether the modulation or the gate changes with the batch entry.
    constexpr bool kPerBatch = kAda || HAS_GATE;

    extern __shared__ uint4 ring[];
    __shared__ float warpSum[2][kBlock / 32];
    __shared__ float warpSumSq[2][kBlock / 32];
    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const rowBegin = static_cast<int>(static_cast<int64_t>(blockIdx.x) * rows / gridDim.x);
    int const rowEnd = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * rows / gridDim.x);

    auto issue = [&](int row, int stage)
    {
        uint4* dst = ring + stage * kStage;
        int64_t const base = static_cast<int64_t>(row) * HIDDEN;
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            int const off = (c * kBlock + tid) * kVec;
            cpAsync16(dst + c * kBlock + tid, x + base + off);
            if constexpr (HAS_Y)
            {
                cpAsync16(dst + (kChunks + c) * kBlock + tid, y + base + off);
            }
        }
    };
    // One commit group per row slot, empty past the end of the range.
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++)
    {
        if (rowBegin + s < rowEnd)
        {
            issue(rowBegin + s, s);
        }
        cpAsyncCommit();
    }

    // ww = 1 + scale (AdaLN) or the LayerNorm weight, bb = shift or the LayerNorm bias, gg = gate.
    float ww[kElts];
    float bb[kElts];
    float gg[HAS_GATE ? kElts : 1];
    uint4 yb[HAS_YBIAS ? kChunks : 1];
    if constexpr (HAS_YBIAS)
    {
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            yb[c] = *reinterpret_cast<uint4 const*>(yBias + (c * kBlock + tid) * kVec);
        }
    }
    float const qmul = invOutScale[0];
    int curBatch = -1;

    int it = 0;
    for (int row = rowBegin; row < rowEnd; row++, it++)
    {
        if (row + STAGES - 1 < rowEnd)
        {
            issue(row + STAGES - 1, (it + STAGES - 1) % STAGES);
        }
        cpAsyncCommit();
        int const batch = kPerBatch ? row / rowsPerBatch : 0;
        if (batch != curBatch)
        {
            // Block-uniform: the first row, then only where the row range crosses a batch boundary.
            curBatch = batch;
            int64_t const modBase = kAda ? static_cast<int64_t>(batch) * HIDDEN : 0;
            int64_t const gateBase = static_cast<int64_t>(batch) * HIDDEN;
#pragma unroll
            for (int c = 0; c < kChunks; c++)
            {
                int const off = (c * kBlock + tid) * kVec;
                float4 const w0 = *reinterpret_cast<float4 const*>(weight + modBase + off);
                float4 const w1 = *reinterpret_cast<float4 const*>(weight + modBase + off + 4);
                float4 const b0 = *reinterpret_cast<float4 const*>(bias + modBase + off);
                float4 const b1 = *reinterpret_cast<float4 const*>(bias + modBase + off + 4);
                float const wv[kVec] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
                float const bv[kVec] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < kVec; i++)
                {
                    ww[c * kVec + i] = kAda ? 1.0F + wv[i] : wv[i];
                    bb[c * kVec + i] = bv[i];
                }
                if constexpr (HAS_GATE)
                {
                    float4 const g0 = *reinterpret_cast<float4 const*>(gate + gateBase + off);
                    float4 const g1 = *reinterpret_cast<float4 const*>(gate + gateBase + off + 4);
                    float const gv[kVec] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
#pragma unroll
                    for (int i = 0; i < kVec; i++)
                    {
                        gg[c * kVec + i] = gv[i];
                    }
                }
            }
        }
        cpAsyncWait<STAGES - 1>();
        uint4 const* src = ring + (it % STAGES) * kStage;
        int64_t const base = static_cast<int64_t>(row) * HIDDEN;
        uint4 xb[kChunks];
        float sum = 0.0F;
        float sumSq = 0.0F;
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            int const off = (c * kBlock + tid) * kVec;
            uint4 const xr = src[c * kBlock + tid];
            if constexpr (HAS_Y)
            {
                uint4 const yr = src[(kChunks + c) * kBlock + tid];
                auto const* xh = reinterpret_cast<__nv_bfloat162 const*>(&xr);
                auto const* yh = reinterpret_cast<__nv_bfloat162 const*>(&yr);
                __nv_bfloat162 o[4];
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    float2 const xx = __bfloat1622float2(xh[i]);
                    float2 yy = __bfloat1622float2(yh[i]);
                    if constexpr (HAS_YBIAS)
                    {
                        float2 const bbv = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const*>(&yb[c])[i]);
                        yy = make_float2(roundBf16(yy.x + bbv.x), roundBf16(yy.y + bbv.y));
                    }
                    float2 r;
                    // Round the product before the add; no FMA.
                    if constexpr (HAS_GATE)
                    {
                        r.x = __fadd_rn(xx.x, __fmul_rn(yy.x, gg[c * kVec + 2 * i]));
                        r.y = __fadd_rn(xx.y, __fmul_rn(yy.y, gg[c * kVec + 2 * i + 1]));
                    }
                    else
                    {
                        r.x = __fadd_rn(xx.x, yy.x);
                        r.y = __fadd_rn(xx.y, yy.y);
                    }
                    o[i] = __floats2bfloat162_rn(r.x, r.y);
                }
                xb[c] = *reinterpret_cast<uint4 const*>(o);
                *reinterpret_cast<uint4*>(xOut + base + off) = xb[c];
            }
            else
            {
                xb[c] = xr;
            }
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 const f = __bfloat1622float2(h[i]);
                sum += f.x;
                sumSq += f.x * f.x;
                sum += f.y;
                sumSq += f.y * f.y;
            }
        }
        int const parity = it & 1;
#pragma unroll
        for (int o = 16; o > 0; o /= 2)
        {
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
            sumSq += __shfl_xor_sync(0xffffffffu, sumSq, o);
        }
        if (lane == 0)
        {
            warpSum[parity][warp] = sum;
            warpSumSq[parity][warp] = sumSq;
        }
        __syncthreads();
        // Same operands and order as warp 0 in residualNormQuantKernel.
        float a = lane < kBlock / 32 ? warpSum[parity][lane] : 0.0F;
        float a2 = lane < kBlock / 32 ? warpSumSq[parity][lane] : 0.0F;
#pragma unroll
        for (int o = 16; o > 0; o /= 2)
        {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            a2 += __shfl_xor_sync(0xffffffffu, a2, o);
        }
        float const invHidden = 1.0F / static_cast<float>(HIDDEN);
        float const mean = a * invHidden;
        float const var = fmaxf(a2 * invHidden - mean * mean, 0.0F);
        float const rstd = rsqrtf(var + eps);
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            int const off = (c * kBlock + tid) * kVec;
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
            __nv_fp8x2_storage_t packed[4];
#pragma unroll
            for (int i = 0; i < kVec; i += 2)
            {
                float2 const xf = __bfloat1622float2(h[i / 2]);
                float const xv[2] = {xf.x, xf.y};
                float yv[2];
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    float const xn = (xv[j] - mean) * rstd;
                    yv[j] = roundBf16(xn * ww[c * kVec + i + j] + bb[c * kVec + i + j]) * qmul;
                }
                packed[i / 2] = __nv_cvt_float2_to_fp8x2(make_float2(yv[0], yv[1]), __NV_SATFINITE, __NV_E4M3);
            }
            *reinterpret_cast<uint2*>(normed + base + off) = *reinterpret_cast<uint2 const*>(packed);
        }
    }
    cpAsyncWait<0>();
#endif
}

// The persistent kernel wins where the one-row-per-block kernel is latency bound: normalization
// without a residual, or a gated residual before the norm. The other variants run at least about
// as fast one row per block. Wider rows than kMaxPersistentHidden would spill the per-thread
// modulation registers.
constexpr bool persistentWins(int hidden, ResidualNormMode mode, bool hasY, bool hasGate)
{
    return hidden <= kMaxPersistentHidden
        && (mode == ResidualNormMode::kLayerNorm || mode == ResidualNormMode::kAdaLayerNorm) && (!hasY || hasGate);
}

// Ring depth: rows in flight per block.
constexpr int persistentStages(bool hasY)
{
    return hasY ? 3 : 4;
}

template <int HIDDEN, bool HAS_Y>
constexpr size_t persistentSmemBytes()
{
    return static_cast<size_t>(persistentStages(HAS_Y)) * (HAS_Y ? 2 : 1) * HIDDEN * sizeof(__nv_bfloat16);
}

struct DeviceInfo
{
    int major = 0;
    int sms = 0;
    int maxSmemOptin = 0;
};

DeviceInfo const& deviceInfo(int device)
{
    static std::mutex mutex;
    static std::map<int, DeviceInfo> infos;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = infos.find(device);
    if (it == infos.end())
    {
        DeviceInfo info;
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&info.major, cudaDevAttrComputeCapabilityMajor, device));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, device));
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&info.maxSmemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        it = infos.emplace(device, info).first;
    }
    return it->second;
}

// Resident blocks of the persistent kernel on this device (0 when it does not fit).
template <int HIDDEN, ResidualNormMode MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS>
int persistentGrid(int device)
{
    constexpr int kStages = persistentStages(HAS_Y);
    constexpr size_t kSmem = persistentSmemBytes<HIDDEN, HAS_Y>();
    auto* kernel = residualNormQuantPersistentKernel<HIDDEN, MODE, HAS_Y, HAS_GATE, HAS_YBIAS, kStages>;
    static std::mutex mutex;
    static std::map<int, int> grids;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = grids.find(device);
    if (it == grids.end())
    {
        DeviceInfo const& info = deviceInfo(device);
        int grid = 0;
        // The ring must fit in the opt-in shared memory with headroom for the kernel's static
        // shared memory (the warp partials); occupancy below decides the blocks per SM.
        constexpr size_t kStaticSmemHeadroom = 1024;
        if (info.major >= 8 && kSmem + kStaticSmemHeadroom <= static_cast<size_t>(info.maxSmemOptin))
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem)));
            int perSm = 0;
            TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, kBlock, kSmem));
            grid = perSm * info.sms;
        }
        it = grids.emplace(device, grid).first;
    }
    return it->second;
}

template <int HIDDEN, ResidualNormMode MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS>
void launchKernel(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    if constexpr (persistentWins(HIDDEN, MODE, HAS_Y, HAS_GATE))
    {
        int device = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&device));
        int const resident = persistentGrid<HIDDEN, MODE, HAS_Y, HAS_GATE, HAS_YBIAS>(device);
        if (resident > 0)
        {
            constexpr int kStages = persistentStages(HAS_Y);
            residualNormQuantPersistentKernel<HIDDEN, MODE, HAS_Y, HAS_GATE, HAS_YBIAS, kStages>
                <<<std::min(p.rows, resident), kBlock, persistentSmemBytes<HIDDEN, HAS_Y>(), stream>>>(
                    static_cast<__nv_bfloat16 const*>(p.x), static_cast<__nv_bfloat16 const*>(p.y),
                    static_cast<__nv_bfloat16 const*>(p.yBias), p.gate, p.weight, p.bias, p.invOutScale, p.rows,
                    p.rowsPerBatch, p.eps, static_cast<__nv_bfloat16*>(p.xOut), static_cast<__nv_fp8_e4m3*>(p.normed));
            return;
        }
    }
    residualNormQuantKernel<HIDDEN, MODE, HAS_Y, HAS_GATE, HAS_YBIAS>
        <<<p.rows, kBlock, 0, stream>>>(static_cast<__nv_bfloat16 const*>(p.x), static_cast<__nv_bfloat16 const*>(p.y),
            static_cast<__nv_bfloat16 const*>(p.yBias), p.gate, p.weight, p.bias, p.invOutScale, p.rowsPerBatch, p.eps,
            static_cast<__nv_bfloat16*>(p.xOut), static_cast<__nv_fp8_e4m3*>(p.normed));
}

template <int HIDDEN, ResidualNormMode MODE>
void dispatchResidual(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    bool const hasGate = p.gate != nullptr;
    bool const hasYBias = p.yBias != nullptr;
    if (p.y == nullptr)
    {
        if constexpr (MODE != ResidualNormMode::kResidualOnly)
        {
            launchKernel<HIDDEN, MODE, false, false, false>(p, stream);
        }
    }
    else if (hasGate && hasYBias)
    {
        launchKernel<HIDDEN, MODE, true, true, true>(p, stream);
    }
    else if (hasGate)
    {
        launchKernel<HIDDEN, MODE, true, true, false>(p, stream);
    }
    else if (hasYBias)
    {
        launchKernel<HIDDEN, MODE, true, false, true>(p, stream);
    }
    else
    {
        launchKernel<HIDDEN, MODE, true, false, false>(p, stream);
    }
}

template <int HIDDEN>
void dispatchMode(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    switch (p.mode)
    {
    case ResidualNormMode::kResidualOnly: dispatchResidual<HIDDEN, ResidualNormMode::kResidualOnly>(p, stream); break;
    case ResidualNormMode::kLayerNorm: dispatchResidual<HIDDEN, ResidualNormMode::kLayerNorm>(p, stream); break;
    case ResidualNormMode::kAdaLayerNorm: dispatchResidual<HIDDEN, ResidualNormMode::kAdaLayerNorm>(p, stream); break;
    case ResidualNormMode::kRmsNorm: dispatchResidual<HIDDEN, ResidualNormMode::kRmsNorm>(p, stream); break;
    }
}

template <int HIDDEN>
void dispatchHidden(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    if (p.hidden == HIDDEN)
    {
        dispatchMode<HIDDEN>(p, stream);
    }
    else if constexpr (HIDDEN < kMaxHidden)
    {
        dispatchHidden<HIDDEN + kHiddenStep>(p, stream);
    }
}

constexpr int kModulateBlock = 256;

// Grid-stride loop over 8-element vectors.
__global__ void __launch_bounds__(kModulateBlock) adaModulateKernel(__nv_bfloat16 const* __restrict__ x,
    float const* __restrict__ scale, float const* __restrict__ shift, __nv_bfloat16* __restrict__ out, int64_t rows,
    int64_t rowsPerBatch, int vecPerRow)
{
    int64_t const total = rows * vecPerRow;
    int64_t const hidden = static_cast<int64_t>(vecPerRow) * kVec;
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride)
    {
        int64_t const row = i / vecPerRow;
        int64_t const col = (i - row * vecPerRow) * kVec;
        int64_t const mod = (row / rowsPerBatch) * hidden + col;
        uint4 const v = __ldcs(reinterpret_cast<uint4 const*>(x) + i);
        float4 const s0 = *reinterpret_cast<float4 const*>(scale + mod);
        float4 const s1 = *reinterpret_cast<float4 const*>(scale + mod + 4);
        float4 const h0 = *reinterpret_cast<float4 const*>(shift + mod);
        float4 const h1 = *reinterpret_cast<float4 const*>(shift + mod + 4);
        float const sv[kVec] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        float const hv[kVec] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        auto const* xh = reinterpret_cast<__nv_bfloat162 const*>(&v);
        __nv_bfloat162 o[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            float2 const f = __bfloat1622float2(xh[k]);
            float const a = __fadd_rn(__fmul_rn(f.x, __fadd_rn(1.0F, sv[2 * k])), hv[2 * k]);
            float const b = __fadd_rn(__fmul_rn(f.y, __fadd_rn(1.0F, sv[2 * k + 1])), hv[2 * k + 1]);
            o[k] = __floats2bfloat162_rn(a, b);
        }
        __stcs(reinterpret_cast<uint4*>(out) + i, *reinterpret_cast<uint4 const*>(o));
    }
}

} // namespace

void launchAdaModulate(AdaModulateParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        p.hidden > 0 && p.hidden % kVec == 0, "ada modulate: hidden %d must be a multiple of 8", p.hidden);
    TLLM_CHECK_WITH_INFO(p.rowsPerBatch > 0, "ada modulate: rows_per_batch must be positive");
    if (p.rows == 0)
    {
        return;
    }
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int const vecPerRow = p.hidden / kVec;
    int64_t const total = p.rows * vecPerRow;
    int64_t const maxGrid = static_cast<int64_t>(deviceInfo(device).sms) * 8;
    int const grid = static_cast<int>(std::min<int64_t>((total + kModulateBlock - 1) / kModulateBlock, maxGrid));
    adaModulateKernel<<<grid, kModulateBlock, 0, stream>>>(static_cast<__nv_bfloat16 const*>(p.x), p.scale, p.shift,
        static_cast<__nv_bfloat16*>(p.out), p.rows, p.rowsPerBatch, vecPerRow);
    sync_check_cuda_error(stream);
}

bool residualNormQuantSupportsHidden(int hidden)
{
    return hidden > 0 && hidden % kHiddenStep == 0 && hidden <= kMaxHidden;
}

void launchResidualNormQuant(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        residualNormQuantSupportsHidden(p.hidden), "residual norm quant: unsupported hidden %d", p.hidden);
    TLLM_CHECK_WITH_INFO(p.rowsPerBatch > 0, "residual norm quant: rows_per_batch must be positive");
    bool const normalizes = p.mode != ResidualNormMode::kResidualOnly;
    TLLM_CHECK_WITH_INFO(p.y != nullptr || normalizes, "residual norm quant: residual-only mode needs y");
    TLLM_CHECK_WITH_INFO(p.y == nullptr || p.xOut != nullptr, "residual norm quant: y needs x_out");
    bool const needsBias = normalizes && p.mode != ResidualNormMode::kRmsNorm;
    TLLM_CHECK_WITH_INFO(!normalizes
            || (p.weight != nullptr && (p.bias != nullptr || !needsBias) && p.invOutScale != nullptr
                && p.normed != nullptr),
        "residual norm quant: normalization needs weight, bias (except RMSNorm), inv_out_scale and normed");
    if (p.rows == 0)
    {
        return;
    }
    dispatchHidden<kHiddenStep>(p, stream);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
