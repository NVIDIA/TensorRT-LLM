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

#include "tensorrt_llm/kernels/minimaxM31/markovEpilogueSm107.h"

#include <algorithm>
#include <cmath>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// torch's argmax order (aten SharedReduceOps GreaterOrNan): a NaN beats any number, two NaNs or two equal values keep
// the lower index. A total order, so the reduction's shape cannot change the result.
__device__ __forceinline__ bool better(float va, int32_t ia, float vb, int32_t ib)
{
    bool const na = isnan(va);
    bool const nb = isnan(vb);
    if (na || nb)
    {
        return na && (!nb || ia < ib);
    }
    return va == vb ? ia < ib : va > vb;
}

__device__ __forceinline__ void take(float& bv, int32_t& bi, float v, int32_t i)
{
    if (better(v, i, bv, bi))
    {
        bv = v;
        bi = i;
    }
}

__device__ __forceinline__ float bf16Bits(uint32_t lo16)
{
    return __uint_as_float(lo16 << 16); // exact bf16 -> fp32
}

__device__ __forceinline__ float4 loadBase(float const* src, int32_t i)
{
    return reinterpret_cast<float4 const*>(src)[i];
}

__device__ __forceinline__ float4 loadBase(__nv_bfloat16 const* src, int32_t i)
{
    uint2 const packed = reinterpret_cast<uint2 const*>(src)[i];
    return {
        bf16Bits(packed.x & 0xffffu), bf16Bits(packed.x >> 16), bf16Bits(packed.y & 0xffffu), bf16Bits(packed.y >> 16)};
}

template <typename Base>
__global__ void __launch_bounds__(kThreads) markovEpilogueKernel(Base const* __restrict__ base, int64_t baseRowStride,
    __nv_bfloat16 const* __restrict__ bias, float* __restrict__ out, int64_t outRowStride, int64_t* __restrict__ prev,
    __nv_bfloat16 const* __restrict__ w1, int32_t rank, __nv_bfloat16* __restrict__ embOut,
    MarkovEpilogueBest* __restrict__ partials, int32_t* __restrict__ counters, int32_t d, int32_t splits)
{
    __shared__ float sValue[kWarps];
    __shared__ int32_t sIndex[kWarps];
    __shared__ bool sLast;

    int32_t const row = blockIdx.y;
    int32_t const split = blockIdx.x;
    int32_t const d4 = d / 4;
    int32_t const per = (d4 + splits - 1) / splits;
    int32_t const begin = split * per;
    int32_t const end = min(d4, begin + per);

    Base const* src = base + row * baseRowStride;
    uint2 const* bsrc = reinterpret_cast<uint2 const*>(bias + static_cast<int64_t>(row) * d);
    float4* dst = reinterpret_cast<float4*>(out + row * outRowStride);

    float bv = -INFINITY;
    int32_t bi = 0x7fffffff;
    for (int32_t i = begin + threadIdx.x; i < end; i += kThreads)
    {
        float4 const x = loadBase(src, i);
        uint2 const w = bsrc[i];
        float4 y;
        y.x = __fadd_rn(x.x, bf16Bits(w.x & 0xffffu));
        y.y = __fadd_rn(x.y, bf16Bits(w.x >> 16));
        y.z = __fadd_rn(x.z, bf16Bits(w.y & 0xffffu));
        y.w = __fadd_rn(x.w, bf16Bits(w.y >> 16));
        dst[i] = y;
        int32_t const j = 4 * i;
        take(bv, bi, y.x, j);
        take(bv, bi, y.y, j + 1);
        take(bv, bi, y.z, j + 2);
        take(bv, bi, y.w, j + 3);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        float const ov = __shfl_xor_sync(0xffffffffu, bv, off);
        int32_t const oi = __shfl_xor_sync(0xffffffffu, bi, off);
        take(bv, bi, ov, oi);
    }
    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    if (lane == 0)
    {
        sValue[warp] = bv;
        sIndex[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int w = 1; w < kWarps; ++w)
        {
            take(bv, bi, sValue[w], sIndex[w]);
        }
        partials[row * splits + split] = MarkovEpilogueBest{bv, bi};
        __threadfence();
        int32_t const done = atomicAdd(counters + row, 1);
        sLast = done == splits - 1;
    }
    __syncthreads();
    if (!sLast)
    {
        return;
    }
    // The row's last CTA: combine the splits (any order: total order), publish prev and the next embedding.
    __threadfence();
    if (threadIdx.x == 0)
    {
        MarkovEpilogueBest const* p = partials + row * splits;
        float v = __ldcg(&p[0].value);
        int32_t idx = __ldcg(&p[0].index);
        for (int32_t s = 1; s < splits; ++s)
        {
            take(v, idx, __ldcg(&p[s].value), __ldcg(&p[s].index));
        }
        prev[row] = idx;
        sIndex[0] = idx;
        counters[row] = 0;
    }
    __syncthreads();
    if (embOut != nullptr)
    {
        int64_t const token = sIndex[0];
        for (int32_t c = threadIdx.x; c < rank; c += kThreads)
        {
            embOut[static_cast<int64_t>(row) * rank + c] = w1[token * rank + c];
        }
    }
}

} // namespace

int32_t markovEpilogueSplits(int32_t rows, int32_t d, int32_t smCount)
{
    int32_t const want = std::max(1, (2 * smCount + rows - 1) / std::max(1, rows));
    int32_t const cap = std::max(1, d / 4 / 1024);
    return std::min(want, cap);
}

void invokeMarkovEpilogue(float const* base, int64_t baseRowStride, __nv_bfloat16 const* bias, float* out,
    int64_t outRowStride, int64_t* prev, __nv_bfloat16 const* w1, int32_t rank, __nv_bfloat16* embOut,
    MarkovEpilogueBest* partials, int32_t* counters, int32_t rows, int32_t d, int32_t splits, cudaStream_t stream)
{
    dim3 const grid(splits, rows);
    markovEpilogueKernel<<<grid, kThreads, 0, stream>>>(
        base, baseRowStride, bias, out, outRowStride, prev, w1, rank, embOut, partials, counters, d, splits);
}

void invokeMarkovEpilogue(__nv_bfloat16 const* base, int64_t baseRowStride, __nv_bfloat16 const* bias, float* out,
    int64_t outRowStride, int64_t* prev, __nv_bfloat16 const* w1, int32_t rank, __nv_bfloat16* embOut,
    MarkovEpilogueBest* partials, int32_t* counters, int32_t rows, int32_t d, int32_t splits, cudaStream_t stream)
{
    dim3 const grid(splits, rows);
    markovEpilogueKernel<<<grid, kThreads, 0, stream>>>(
        base, baseRowStride, bias, out, outRowStride, prev, w1, rank, embOut, partials, counters, d, splits);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
