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
#include "tensorrt_llm/kernels/fusedDiTQKNormRopeFp8Kernel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kHeadDim = 128;
constexpr int kMaxHeads = 64;
constexpr float kE4m3Max = 448.f;

__device__ __forceinline__ float warpReduceSum(float v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, m);
    }
    return v;
}

__device__ __forceinline__ float warpReduceMax(float v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
    {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    }
    return v;
}

// 8 values: round to bf16 (to match the unfused bf16 path), scale, saturating e4m3 store.
__device__ __forceinline__ void storeFp8x8(float const (&x)[8], __nv_fp8_e4m3* dst, float mul)
{
    __nv_fp8x2_storage_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float const a = __bfloat162float(__float2bfloat16_rn(x[2 * i]));
        float const b = __bfloat162float(__float2bfloat16_rn(x[2 * i + 1]));
        packed[i] = __nv_cvt_float2_to_fp8x2(make_float2(a * mul, b * mul), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(packed);
}

// Unpacks 8 bf16 values.
__device__ __forceinline__ void unpackBf16x8(uint4 const& v, float (&out)[8])
{
    uint32_t const* u = reinterpret_cast<uint32_t const*>(&v);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 const f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&u[i]));
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

// RMSNorm scale, weight and RoPE on 8 consecutive elements of one head (head_dim 128).
template <bool INTERLEAVE>
__device__ __forceinline__ void normRope8(
    float (&e)[8], float const (&w)[8], float const (&c)[8], float const (&s)[8], float rmsRcp, int rowLane)
{
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        e[i] *= rmsRcp * w[i];
    }
    if constexpr (INTERLEAVE)
    {
#pragma unroll
        for (int i = 0; i < 8; i += 2)
        {
            float const x = e[i], y = e[i + 1];
            e[i] = x * c[i] + (-y) * s[i];
            e[i + 1] = y * c[i + 1] + x * s[i + 1];
        }
    }
    else
    {
        // rotate_half: the partner element sits half a head away, i.e. 8 lanes over.
        constexpr int kXorMask = kHeadDim / 16;
        bool const negate = ((rowLane & kXorMask) == 0);
        unsigned const activeMask = __activemask();
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            float p = __shfl_xor_sync(activeMask, e[i], kXorMask);
            if (negate)
            {
                p = -p;
            }
            e[i] = e[i] * c[i] + p * s[i];
        }
    }
}

__device__ __forceinline__ void loadFloat8(float const* p, float (&out)[8])
{
    float4 const a = reinterpret_cast<float4 const*>(p)[0];
    float4 const b = reinterpret_cast<float4 const*>(p)[1];
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
    out[4] = b.x;
    out[5] = b.y;
    out[6] = b.z;
    out[7] = b.w;
}

// General path: two rows per 256-thread block; also handles per-head cos tables.
template <bool INTERLEAVE, bool PER_HEAD_COS>
__global__ void __launch_bounds__(256)
    fusedDiTQKNormRopeFp8Kernel(__nv_bfloat16 const* qkv, __nv_fp8_e4m3* qOut, __nv_fp8_e4m3* kOut, __nv_fp8_e4m3* vOut,
        float const* quantMul, int numHeads, float eps, __nv_bfloat16 const* qWeight, __nv_bfloat16 const* kWeight,
        float const* cosEmb, float const* sinEmb, int numTokens, int cosSeqPerBatch)
{
    constexpr int kRowsPerBlock = 2;
    constexpr int kThreadsPerRow = 128;
    constexpr int kWarpsPerRow = kThreadsPerRow / 32;
    constexpr int kChunk = 8;
    constexpr int kMaxChunks = (kMaxHeads * kHeadDim + kThreadsPerRow * kChunk - 1) / (kThreadsPerRow * kChunk);

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const rowInBlock = tid / kThreadsPerRow;
    int const laneInRow = tid % kThreadsPerRow;
    int const rowWarp = laneInRow >> 5;
    int const rowLane = laneInRow & 31;
    int const tokenIdx = blockIdx.x * kRowsPerBlock + rowInBlock;
    // Both rows of a block take part in every __syncthreads(); only valid rows touch memory.
    bool const validRow = tokenIdx < numTokens;

    int const n = numHeads * kHeadDim;
    int const chunksPerRow = (n + kThreadsPerRow * kChunk - 1) / (kThreadsPerRow * kChunk);
    int64_t const tokenBaseQ = static_cast<int64_t>(tokenIdx) * 3 * n;
    int64_t const tokenBaseK = tokenBaseQ + n;
    int const cosToken = (cosSeqPerBatch > 0) ? (tokenIdx % cosSeqPerBatch) : tokenIdx;
    int const cosStride = PER_HEAD_COS ? n : kHeadDim;
    int64_t const embBase = static_cast<int64_t>(cosToken) * cosStride;

    // SMEM: [Q row0][Q row1][K row0][K row1] bf16, [cos row0..1][sin row0..1] fp32, warp sums.
    extern __shared__ __align__(16) unsigned char smemRaw[];
    __nv_bfloat16* smemQ = reinterpret_cast<__nv_bfloat16*>(smemRaw);
    __nv_bfloat16* smemK = smemQ + kRowsPerBlock * n;
    float* smemCos = reinterpret_cast<float*>(smemK + kRowsPerBlock * n);
    float* smemSin = smemCos + kRowsPerBlock * cosStride;
    float* warpSums = smemSin + kRowsPerBlock * cosStride;

    if (validRow)
    {
#pragma unroll
        for (int chunk = 0; chunk < kMaxChunks; chunk++)
        {
            int const elemBase = chunk * kThreadsPerRow * kChunk + laneInRow * kChunk;
            if (chunk >= chunksPerRow || elemBase >= n)
            {
                continue;
            }
            __pipeline_memcpy_async(smemQ + rowInBlock * n + elemBase, qkv + tokenBaseQ + elemBase, 16);
            __pipeline_memcpy_async(smemK + rowInBlock * n + elemBase, qkv + tokenBaseK + elemBase, 16);
            int const cosOff = PER_HEAD_COS ? elemBase : elemBase % kHeadDim;
            bool const stagesCos = PER_HEAD_COS || elemBase < kHeadDim;
            if (stagesCos)
            {
                __pipeline_memcpy_async(smemCos + rowInBlock * cosStride + cosOff, cosEmb + embBase + cosOff, 16);
                __pipeline_memcpy_async(
                    smemCos + rowInBlock * cosStride + cosOff + 4, cosEmb + embBase + cosOff + 4, 16);
                __pipeline_memcpy_async(smemSin + rowInBlock * cosStride + cosOff, sinEmb + embBase + cosOff, 16);
                __pipeline_memcpy_async(
                    smemSin + rowInBlock * cosStride + cosOff + 4, sinEmb + embBase + cosOff + 4, 16);
            }
        }
    }
    __pipeline_commit();

    uint4 qW[kMaxChunks], kW[kMaxChunks];
#pragma unroll
    for (int chunk = 0; chunk < kMaxChunks; chunk++)
    {
        int const elemBase = chunk * kThreadsPerRow * kChunk + laneInRow * kChunk;
        if (chunk >= chunksPerRow || elemBase >= n)
        {
            continue;
        }
        qW[chunk] = *reinterpret_cast<uint4 const*>(&qWeight[elemBase]);
        kW[chunk] = *reinterpret_cast<uint4 const*>(&kWeight[elemBase]);
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    float qSum2 = 0.f, kSum2 = 0.f;
    if (validRow)
    {
#pragma unroll
        for (int chunk = 0; chunk < kMaxChunks; chunk++)
        {
            int const elemBase = chunk * kThreadsPerRow * kChunk + laneInRow * kChunk;
            if (chunk >= chunksPerRow || elemBase >= n)
            {
                continue;
            }
            float qv[8], kv[8];
            unpackBf16x8(*reinterpret_cast<uint4 const*>(&smemQ[rowInBlock * n + elemBase]), qv);
            unpackBf16x8(*reinterpret_cast<uint4 const*>(&smemK[rowInBlock * n + elemBase]), kv);
#pragma unroll
            for (int i = 0; i < 8; i += 2)
            {
                qSum2 += qv[i] * qv[i] + qv[i + 1] * qv[i + 1];
                kSum2 += kv[i] * kv[i] + kv[i + 1] * kv[i + 1];
            }
        }
    }
    qSum2 = warpReduceSum(qSum2);
    kSum2 = warpReduceSum(kSum2);
    if (rowLane == 0)
    {
        warpSums[rowInBlock * (2 * kWarpsPerRow) + 2 * rowWarp] = qSum2;
        warpSums[rowInBlock * (2 * kWarpsPerRow) + 2 * rowWarp + 1] = kSum2;
    }
    __syncthreads();
    if (!validRow)
    {
        return;
    }
    float qTotal = 0.f, kTotal = 0.f;
#pragma unroll
    for (int w = 0; w < kWarpsPerRow; w++)
    {
        qTotal += warpSums[rowInBlock * (2 * kWarpsPerRow) + 2 * w];
        kTotal += warpSums[rowInBlock * (2 * kWarpsPerRow) + 2 * w + 1];
    }
    float const qRmsRcp = rsqrtf(qTotal / static_cast<float>(n) + eps);
    float const kRmsRcp = rsqrtf(kTotal / static_cast<float>(n) + eps);
    float const qMul = quantMul[0], kMul = quantMul[1], vMul = quantMul[2];

#pragma unroll
    for (int chunk = 0; chunk < kMaxChunks; chunk++)
    {
        int const elemBase = chunk * kThreadsPerRow * kChunk + laneInRow * kChunk;
        if (chunk >= chunksPerRow || elemBase >= n)
        {
            continue;
        }
        int const cosIdx = rowInBlock * cosStride + (PER_HEAD_COS ? elemBase : elemBase % kHeadDim);
        float c[8], s[8], w[8], e[8];
        loadFloat8(smemCos + cosIdx, c);
        loadFloat8(smemSin + cosIdx, s);
        int64_t const outIdx = static_cast<int64_t>(tokenIdx) * n + elemBase;

        unpackBf16x8(*reinterpret_cast<uint4 const*>(&smemQ[rowInBlock * n + elemBase]), e);
        unpackBf16x8(qW[chunk], w);
        normRope8<INTERLEAVE>(e, w, c, s, qRmsRcp, rowLane);
        storeFp8x8(e, qOut + outIdx, qMul);

        unpackBf16x8(*reinterpret_cast<uint4 const*>(&smemK[rowInBlock * n + elemBase]), e);
        unpackBf16x8(kW[chunk], w);
        normRope8<INTERLEAVE>(e, w, c, s, kRmsRcp, rowLane);
        storeFp8x8(e, kOut + outIdx, kMul);

        unpackBf16x8(*reinterpret_cast<uint4 const*>(&qkv[tokenBaseK + n + elemBase]), e);
        storeFp8x8(e, vOut + outIdx, vMul);
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Rows in flight per block in the bulk-copy pipeline.
constexpr int kStages = 3;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(smemAddr(bar)), "r"(phase)
            : "memory");
    }
}

// Bulk global-to-shared copy whose completion is counted on the mbarrier.
__device__ __forceinline__ void bulkLoad(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar))
        : "memory");
}
#endif

// Fast path (SM90+): shared fp32 cos, persistent 128-thread blocks streaming
// rows through a kStages-deep ring of bulk copies. NCH = ceil(num_heads * 128 / 1024).
template <bool INTERLEAVE, int NCH>
__global__ void __launch_bounds__(128)
    fusedDiTQKNormRopeFp8PipeKernel(__nv_bfloat16 const* qkv, __nv_fp8_e4m3* qOut, __nv_fp8_e4m3* kOut,
        __nv_fp8_e4m3* vOut, float const* quantMul, int numHeads, float eps, __nv_bfloat16 const* qWeight,
        __nv_bfloat16 const* kWeight, float const* cosEmb, float const* sinEmb, int numTokens, int cosSeqPerBatch)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap(); // The launcher selects this kernel only on SM90+.
#else
    constexpr int kWarpsPerRow = 4;
    asm volatile("griddepcontrol.wait;");
    int const tid = threadIdx.x;
    int const rowWarp = tid >> 5;
    int const rowLane = tid & 31;
    int const n = numHeads * kHeadDim;
    uint32_t const rowBytes = 3 * n * sizeof(__nv_bfloat16);
    uint32_t const stageBytes = rowBytes + 2 * kHeadDim * sizeof(float);
    extern __shared__ __align__(128) unsigned char ring[];
    __shared__ __align__(8) uint64_t bars[kStages];
    __shared__ float warpSums[2 * kWarpsPerRow];
    int const stride = gridDim.x;
    auto issue = [&](int st, int t)
    {
        unsigned char* dst = ring + st * stageBytes;
        int const ct = (cosSeqPerBatch > 0) ? (t % cosSeqPerBatch) : t;
        mbarExpectTx(&bars[st], stageBytes);
        bulkLoad(dst, qkv + static_cast<int64_t>(t) * 3 * n, rowBytes, &bars[st]);
        bulkLoad(dst + rowBytes, cosEmb + static_cast<int64_t>(ct) * kHeadDim, kHeadDim * sizeof(float), &bars[st]);
        bulkLoad(dst + rowBytes + kHeadDim * sizeof(float), sinEmb + static_cast<int64_t>(ct) * kHeadDim,
            kHeadDim * sizeof(float), &bars[st]);
    };
    if (tid == 0)
    {
#pragma unroll
        for (int st = 0; st < kStages; st++)
        {
            mbarInit(&bars[st], 1);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
        for (int st = 0; st < kStages; st++)
        {
            int const t = blockIdx.x + st * stride;
            if (t < numTokens)
            {
                issue(st, t);
            }
        }
    }
    // The norm weights stay in registers across rows.
    uint4 qW[NCH], kW[NCH];
#pragma unroll
    for (int chunk = 0; chunk < NCH; chunk++)
    {
        int const elemBase = chunk * 1024 + tid * 8;
        if (elemBase >= n)
        {
            continue;
        }
        qW[chunk] = *reinterpret_cast<uint4 const*>(&qWeight[elemBase]);
        kW[chunk] = *reinterpret_cast<uint4 const*>(&kWeight[elemBase]);
    }
    float const qMul = quantMul[0], kMul = quantMul[1], vMul = quantMul[2];
    __syncthreads();
    int it = 0;
    for (int tokenIdx = blockIdx.x; tokenIdx < numTokens; tokenIdx += stride, it++)
    {
        int const st = it % kStages;
        mbarWait(&bars[st], (it / kStages) & 1);
        __nv_bfloat16 const* srow = reinterpret_cast<__nv_bfloat16 const*>(ring + st * stageBytes);
        float const* scos = reinterpret_cast<float const*>(ring + st * stageBytes + rowBytes);
        float const* ssin = scos + kHeadDim;
        float qSum2 = 0.f, kSum2 = 0.f;
#pragma unroll
        for (int chunk = 0; chunk < NCH; chunk++)
        {
            int const elemBase = chunk * 1024 + tid * 8;
            if (elemBase >= n)
            {
                continue;
            }
            float qv[8], kv[8];
            unpackBf16x8(*reinterpret_cast<uint4 const*>(&srow[elemBase]), qv);
            unpackBf16x8(*reinterpret_cast<uint4 const*>(&srow[n + elemBase]), kv);
#pragma unroll
            for (int i = 0; i < 8; i += 2)
            {
                qSum2 += qv[i] * qv[i] + qv[i + 1] * qv[i + 1];
                kSum2 += kv[i] * kv[i] + kv[i + 1] * kv[i + 1];
            }
        }
        qSum2 = warpReduceSum(qSum2);
        kSum2 = warpReduceSum(kSum2);
        if (rowLane == 0)
        {
            warpSums[2 * rowWarp] = qSum2;
            warpSums[2 * rowWarp + 1] = kSum2;
        }
        __syncthreads();
        float qTotal = 0.f, kTotal = 0.f;
#pragma unroll
        for (int w = 0; w < kWarpsPerRow; w++)
        {
            qTotal += warpSums[2 * w];
            kTotal += warpSums[2 * w + 1];
        }
        float const qRmsRcp = rsqrtf(qTotal / static_cast<float>(n) + eps);
        float const kRmsRcp = rsqrtf(kTotal / static_cast<float>(n) + eps);
        int64_t const outBase = static_cast<int64_t>(tokenIdx) * n;
#pragma unroll
        for (int chunk = 0; chunk < NCH; chunk++)
        {
            int const elemBase = chunk * 1024 + tid * 8;
            if (elemBase >= n)
            {
                continue;
            }
            int const cosOff = elemBase % kHeadDim;
            float c[8], s[8], w[8], e[8];
            loadFloat8(scos + cosOff, c);
            loadFloat8(ssin + cosOff, s);

            unpackBf16x8(*reinterpret_cast<uint4 const*>(&srow[elemBase]), e);
            unpackBf16x8(qW[chunk], w);
            normRope8<INTERLEAVE>(e, w, c, s, qRmsRcp, rowLane);
            storeFp8x8(e, qOut + outBase + elemBase, qMul);

            unpackBf16x8(*reinterpret_cast<uint4 const*>(&srow[n + elemBase]), e);
            unpackBf16x8(kW[chunk], w);
            normRope8<INTERLEAVE>(e, w, c, s, kRmsRcp, rowLane);
            storeFp8x8(e, kOut + outBase + elemBase, kMul);

            unpackBf16x8(*reinterpret_cast<uint4 const*>(&srow[2 * n + elemBase]), e);
            storeFp8x8(e, vOut + outBase + elemBase, vMul);
        }
        // The stage and the warp sums are consumed; refill the stage.
        __syncthreads();
        if (tid == 0 && tokenIdx + kStages * stride < numTokens)
        {
            issue(st, tokenIdx + kStages * stride);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Per-block |V| maxima: one warp per token, kVUnroll loads in flight.
constexpr int kVUnroll = 10;

__global__ void __launch_bounds__(256) vAmaxPartialKernel(__nv_bfloat16 const* qkv, int64_t tokens, int n, float* out)
{
    int const lane = threadIdx.x & 31;
    int64_t const nWarps = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
    int const chunksPerRow = n / 8;
    float m = 0.f;
    for (int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; t < tokens; t += nWarps)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(qkv + t * 3 * n + 2 * n);
        for (int c0 = 0; c0 < chunksPerRow; c0 += 32 * kVUnroll)
        {
            uint4 v[kVUnroll];
#pragma unroll
            for (int u = 0; u < kVUnroll; u++)
            {
                int const c = c0 + u * 32 + lane;
                v[u] = c < chunksPerRow ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < kVUnroll; u++)
            {
                float f[8];
                unpackBf16x8(v[u], f);
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    m = fmaxf(m, fabsf(f[j]));
                }
            }
        }
    }
    m = warpReduceMax(m);
    __shared__ float sm[8];
    if (lane == 0)
    {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMax(m);
        if (threadIdx.x == 0)
        {
            out[blockIdx.x] = m;
        }
    }
}

// Max of the partials into amax; optionally the V scale and its reciprocal.
__global__ void vScaleReduceKernel(float const* partial, int count, float* amax, float* mulV, float* scaleV)
{
    float m = 0.f;
    for (int i = threadIdx.x; i < count; i += blockDim.x)
    {
        m = fmaxf(m, partial[i]);
    }
    m = warpReduceMax(m);
    __shared__ float sm[32];
    if ((threadIdx.x & 31) == 0)
    {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMax(m);
        if (threadIdx.x == 0)
        {
            if (amax != nullptr)
            {
                amax[0] = m;
            }
            if (scaleV != nullptr)
            {
                float const s = fmaxf(m, 1e-12f) / kE4m3Max;
                scaleV[0] = s;
                mulV[0] = 1.f / s;
            }
        }
    }
}

template <typename Kernel, typename... Args>
void launchWithPdl(Kernel kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args)
{
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

int multiProcessorCount(int device)
{
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    return sms;
}

template <bool IL, int NCH>
void launchPipe(int tokens, int numHeads, float eps, cudaStream_t stream, __nv_bfloat16 const* qkv, __nv_fp8_e4m3* q8,
    __nv_fp8_e4m3* k8, __nv_fp8_e4m3* v8, float const* quantMul, __nv_bfloat16 const* qw, __nv_bfloat16 const* kw,
    float const* cosp, float const* sinp, int cosSeqPerBatch)
{
    int const n = numHeads * kHeadDim;
    auto* kernel = fusedDiTQKNormRopeFp8PipeKernel<IL, NCH>;
    size_t const smem = kStages * (3 * n * sizeof(__nv_bfloat16) + 2 * kHeadDim * sizeof(float));
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    // Resident blocks per device for this head count; one entry per (device, num_heads).
    static std::mutex mutex;
    static std::map<std::pair<int, int>, int> residentBlocks;
    int resident = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto const key = std::make_pair(device, numHeads);
        auto it = residentBlocks.find(key);
        if (it == residentBlocks.end())
        {
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(
                reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            int perSm = 0;
            TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, 128, smem));
            it = residentBlocks.emplace(key, std::max(1, perSm) * multiProcessorCount(device)).first;
        }
        resident = it->second;
    }
    int const grid = std::min(tokens, resident);
    launchWithPdl(kernel, dim3(grid), dim3(128), smem, stream, qkv, q8, k8, v8, quantMul, numHeads, eps, qw, kw, cosp,
        sinp, tokens, cosSeqPerBatch);
}

template <bool IL>
void dispatchPipe(int nch, int tokens, int numHeads, float eps, cudaStream_t stream, __nv_bfloat16 const* qkv,
    __nv_fp8_e4m3* q8, __nv_fp8_e4m3* k8, __nv_fp8_e4m3* v8, float const* quantMul, __nv_bfloat16 const* qw,
    __nv_bfloat16 const* kw, float const* cosp, float const* sinp, int cosSeqPerBatch)
{
#define TLLM_DIT_NRQ_CASE(C)                                                                                           \
    case C:                                                                                                            \
        launchPipe<IL, C>(                                                                                             \
            tokens, numHeads, eps, stream, qkv, q8, k8, v8, quantMul, qw, kw, cosp, sinp, cosSeqPerBatch);             \
        break
    switch (nch)
    {
        TLLM_DIT_NRQ_CASE(1);
        TLLM_DIT_NRQ_CASE(2);
        TLLM_DIT_NRQ_CASE(3);
        TLLM_DIT_NRQ_CASE(4);
        TLLM_DIT_NRQ_CASE(5);
        TLLM_DIT_NRQ_CASE(6);
        TLLM_DIT_NRQ_CASE(7);
        TLLM_DIT_NRQ_CASE(8);
    default: TLLM_THROW("fusedDiTQKNormRopeFp8: unsupported chunk count %d", nch);
    }
#undef TLLM_DIT_NRQ_CASE
}

bool aligned16(void const* p)
{
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
}

} // namespace

bool fusedDiTQKNormRopeFp8Supported(int num_heads, int head_dim, int sm)
{
    return sm >= 90 && head_dim == kHeadDim && num_heads > 0 && num_heads <= kMaxHeads;
}

void launchFusedDiTQKNormRopeFullDimFp8(void const* qkv, void* q8, void* k8, void* v8, int num_tokens, int num_heads,
    int head_dim, float eps, void const* q_weight, void const* k_weight, float const* cos_emb, float const* sin_emb,
    bool interleave, bool per_head_cos, int cos_seq_per_batch, float const* quant_mul, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(fusedDiTQKNormRopeFp8Supported(num_heads, head_dim, tensorrt_llm::common::getSMVersion()),
        "fusedDiTQKNormRopeFp8 requires SM90+, head_dim 128 and at most 64 heads (got %d heads, head_dim %d)",
        num_heads, head_dim);
    if (num_tokens == 0)
    {
        return;
    }
    auto const* src = static_cast<__nv_bfloat16 const*>(qkv);
    auto* q = static_cast<__nv_fp8_e4m3*>(q8);
    auto* k = static_cast<__nv_fp8_e4m3*>(k8);
    auto* v = static_cast<__nv_fp8_e4m3*>(v8);
    auto const* qw = static_cast<__nv_bfloat16 const*>(q_weight);
    auto const* kw = static_cast<__nv_bfloat16 const*>(k_weight);
    int const n = num_heads * kHeadDim;

    // Both kernels move 16-byte vectors.
    TLLM_CHECK_WITH_INFO(aligned16(qkv) && aligned16(q8) && aligned16(k8) && aligned16(v8) && aligned16(q_weight)
            && aligned16(k_weight) && aligned16(cos_emb) && aligned16(sin_emb),
        "fusedDiTQKNormRopeFp8: all tensors must be 16-byte aligned");

    if (!per_head_cos)
    {
        int const nch = (n + 1023) / 1024;
        if (interleave)
        {
            dispatchPipe<true>(nch, num_tokens, num_heads, eps, stream, src, q, k, v, quant_mul, qw, kw, cos_emb,
                sin_emb, cos_seq_per_batch);
        }
        else
        {
            dispatchPipe<false>(nch, num_tokens, num_heads, eps, stream, src, q, k, v, quant_mul, qw, kw, cos_emb,
                sin_emb, cos_seq_per_batch);
        }
        return;
    }

    constexpr int kRows = 2;
    int const cosStride = per_head_cos ? n : kHeadDim;
    size_t const smem
        = 2 * kRows * n * sizeof(__nv_bfloat16) + 2 * kRows * cosStride * sizeof(float) + kRows * 8 * sizeof(float);
    dim3 const grid((num_tokens + kRows - 1) / kRows);
#define TLLM_DIT_NRQ_LAUNCH(IL, PH)                                                                                    \
    launchWithPdl(fusedDiTQKNormRopeFp8Kernel<IL, PH>, grid, dim3(256), smem, stream, src, q, k, v, quant_mul,         \
        num_heads, eps, qw, kw, cos_emb, sin_emb, num_tokens, cos_seq_per_batch)
    if (interleave)
    {
        if (per_head_cos)
        {
            TLLM_DIT_NRQ_LAUNCH(true, true);
        }
        else
        {
            TLLM_DIT_NRQ_LAUNCH(true, false);
        }
    }
    else
    {
        if (per_head_cos)
        {
            TLLM_DIT_NRQ_LAUNCH(false, true);
        }
        else
        {
            TLLM_DIT_NRQ_LAUNCH(false, false);
        }
    }
#undef TLLM_DIT_NRQ_LAUNCH
}

int getDiTVAmaxWorkspaceSize(int device)
{
    int perSm = 0;
    TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, vAmaxPartialKernel, 256, 0));
    return std::max(1, perSm) * multiProcessorCount(device);
}

void launchDiTVAmax(void const* qkv, int num_tokens, int num_heads, int head_dim, float* workspace, int workspace_size,
    float* amax, float* mul_v, float* scale_v, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(workspace_size > 0, "launchDiTVAmax: empty workspace");
    TLLM_CHECK_WITH_INFO((mul_v == nullptr) == (scale_v == nullptr), "launchDiTVAmax: set both mul_v and scale_v");
    int const n = num_heads * head_dim;
    TLLM_CHECK_WITH_INFO(n % 8 == 0, "launchDiTVAmax: num_heads * head_dim must be a multiple of 8");
    vAmaxPartialKernel<<<workspace_size, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(qkv), num_tokens, n, workspace);
    TLLM_CUDA_CHECK(cudaGetLastError());
    vScaleReduceKernel<<<1, 1024, 0, stream>>>(workspace, workspace_size, amax, mul_v, scale_v);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchDiTVScaleFromAmax(float const* amax, float* mul_v, float* scale_v, cudaStream_t stream)
{
    vScaleReduceKernel<<<1, 32, 0, stream>>>(amax, 1, nullptr, mul_v, scale_v);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
