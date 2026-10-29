/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/packedVector.cuh"
#include "vaeFrameOpsKernel.h"
#include "vaeNormCommon.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;
using namespace vae_norm;

constexpr int kThreads = 256;
using T = __nv_bfloat16;
constexpr int V = static_cast<int>(sizeof(float4) / sizeof(T));
using Vec = PackedAligned<T, V>;

__device__ __forceinline__ float2 roundPair(float2 v)
{
    return Elem<T>::toFloat2(Elem<T>::fromFloat2(v));
}

// One thread per (output pixel, 8 output channels): y = x + mean over the group.
__global__ void __launch_bounds__(kThreads) vaeAvgDownAddKernel(
    T const* __restrict__ x, T const* __restrict__ r, T const* __restrict__ bias, T* __restrict__ y, VaeAvgDownParams p)
{
    int const cvec = p.cout / V;
    int64_t const total = static_cast<int64_t>(p.outFrames) * p.outPositions * cvec;
    int const factor = p.factorT * p.factorS * p.factorS;
    float const inv = 1.0f / static_cast<float>(p.groupSize);
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const o0 = static_cast<int>(idx % cvec) * V;
        int64_t const pix = idx / cvec; // (t', i', j') flattened
        int const tOut = static_cast<int>(pix / p.outPositions);
        int64_t const sp = pix - static_cast<int64_t>(tOut) * p.outPositions;
        int const iOut = static_cast<int>(sp / p.outWidth);
        int const jOut = static_cast<int>(sp - static_cast<int64_t>(iOut) * p.outWidth);
        Vec xv = *reinterpret_cast<Vec const*>(x + idx * V);
        Vec const bv = bias != nullptr ? *reinterpret_cast<Vec const*>(bias + o0) : common::make_zeros<Vec>();
        Vec out;
#pragma unroll
        for (int k = 0; k < V; ++k)
        {
            float acc = 0.0f;
            for (int g = 0; g < p.groupSize; ++g)
            {
                int const f = (o0 + k) * p.groupSize + g;
                int const c = f / factor;
                int const rem = f - c * factor;
                int const a = rem / (p.factorS * p.factorS);
                int const rem2 = rem - a * p.factorS * p.factorS;
                int const b = rem2 / p.factorS;
                int const d = rem2 - b * p.factorS;
                int const tIn = tOut * p.factorT + a - p.padT;
                if (tIn >= 0)
                {
                    int64_t const row
                        = (static_cast<int64_t>(tIn) * (p.outHeight * p.factorS) + iOut * p.factorS + b) * p.inWidth
                        + jOut * p.factorS + d;
                    acc += Elem<T>::toFloat(r[row * p.cin + c]);
                }
            }
            // mean in fp32 (PyTorch's bf16 mean accumulates in fp32), then the eager add
            float const m = roundPair(make_float2(acc * inv, 0.0f)).x;
            float const xb = Elem<T>::toFloat(xv.val[k]) + Elem<T>::toFloat(bv.val[k]);
            out.val[k] = Elem<T>::fromFloat2(make_float2(xb + m, 0.0f)).x;
        }
        *reinterpret_cast<Vec*>(y + idx * V) = out;
    }
}

__global__ void __launch_bounds__(kThreads) vaeInterleaveBiasKernel(
    T const* __restrict__ u, T const* __restrict__ bias, T* __restrict__ y, VaeInterleaveParams p)
{
    int const C = p.channels;
    int const cvec = C / V;
    int64_t const rowVecs = p.positions * cvec; // output vectors per frame
    int64_t const total = static_cast<int64_t>(2 * p.frames) * rowVecs;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const tOut = static_cast<int>(idx / rowVecs);
        int64_t const inFrame = idx - static_cast<int64_t>(tOut) * rowVecs;
        int64_t const pos = inFrame / cvec;
        int const c0 = static_cast<int>(inFrame - pos * cvec) * V;
        int const half = tOut & 1;
        int const tIn = tOut >> 1;
        // u row (tIn, pos) has 2C channels; this output takes channels [half*C, half*C + C).
        int64_t const uOff = (static_cast<int64_t>(tIn) * p.positions + pos) * (2 * C) + half * C + c0;
        Vec v = *reinterpret_cast<Vec const*>(u + uOff);
        if (bias != nullptr)
        {
            Vec const bv = *reinterpret_cast<Vec const*>(bias + half * C + c0);
#pragma unroll
            for (int i = 0; i < V / 2; ++i)
            {
                float2 const o = pairAt(v, i), b = pairAt(bv, i);
                setPair(v, i, make_float2(o.x + b.x, o.y + b.y));
            }
        }
        *reinterpret_cast<Vec*>(y + idx * V) = v;
    }
}

// One thread per chunk pixel: its C*patch*patch channels scatter to patch x patch output
// pixels of C channels each (6-byte pixels for RGB; scalar stores).
__global__ void __launch_bounds__(kThreads)
    vaeUnpatchifyClampKernel(T const* __restrict__ chunk, T* __restrict__ out, VaeUnpatchifyParams p)
{
    int const P = p.patch, C = p.channels;
    int const cpp = C * P * P;
    int64_t const pixels = static_cast<int64_t>(p.frames) * p.height * p.width;
    int const outW = p.width * P, outH = p.height * P;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; idx < pixels;
         idx += static_cast<int64_t>(gridDim.x) * kThreads)
    {
        int const w = static_cast<int>(idx % p.width);
        int64_t const th = idx / p.width;
        int const h = static_cast<int>(th % p.height);
        int const f = static_cast<int>(th / p.height);
        T const* src = chunk + idx * cpp;
        int64_t const rowBase = (static_cast<int64_t>(p.frame0 + f) * outH + static_cast<int64_t>(h) * P) * outW;
        for (int j = 0; j < P; ++j)
        {
            for (int i = 0; i < P; ++i)
            {
                T* dst = out + (rowBase + static_cast<int64_t>(j) * outW + static_cast<int64_t>(w) * P + i) * C;
                for (int c = 0; c < C; ++c)
                {
                    float const v = Elem<T>::toFloat(src[(c * P + i) * P + j]);
                    dst[c] = Elem<T>::fromFloat2(make_float2(fminf(fmaxf(v, -1.0f), 1.0f), 0.0f)).x;
                }
            }
        }
    }
}

unsigned gridFor(int64_t total)
{
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int64_t const blocks = (total + kThreads - 1) / kThreads;
    int64_t const cap = static_cast<int64_t>(sms) * 8;
    return static_cast<unsigned>(blocks < cap ? blocks : cap);
}

} // namespace

void launchVaeAvgDownAdd(
    void const* x, void const* r, void const* bias, void* y, VaeAvgDownParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.cout % V == 0, "vaeAvgDownAdd: cout must be a multiple of %d", V);
    TLLM_CHECK_WITH_INFO(p.groupSize >= 1 && p.factorT >= 1 && p.factorS >= 1, "vaeAvgDownAdd: bad factors");
    int64_t const total = static_cast<int64_t>(p.outFrames) * p.outPositions * (p.cout / V);
    if (total == 0)
    {
        return;
    }
    vaeAvgDownAddKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(x), static_cast<T const*>(r), static_cast<T const*>(bias), static_cast<T*>(y), p);
    sync_check_cuda_error(stream);
}

void launchVaeInterleaveBias(
    void const* u, void const* bias, void* y, VaeInterleaveParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        p.channels > 0 && p.channels % V == 0, "vaeInterleaveBias: channels must be a multiple of %d", V);
    int64_t const total = static_cast<int64_t>(2 * p.frames) * p.positions * (p.channels / V);
    if (total == 0)
    {
        return;
    }
    vaeInterleaveBiasKernel<<<gridFor(total), kThreads, 0, stream>>>(
        static_cast<T const*>(u), static_cast<T const*>(bias), static_cast<T*>(y), p);
    sync_check_cuda_error(stream);
}

void launchVaeUnpatchifyClamp(void const* chunk, void* out, VaeUnpatchifyParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.patch >= 1 && p.channels >= 1, "vaeUnpatchifyClamp: bad patch or channels");
    TLLM_CHECK_WITH_INFO(
        p.frame0 >= 0 && p.frame0 + p.frames <= p.outFrames, "vaeUnpatchifyClamp: frames out of range");
    int64_t const pixels = static_cast<int64_t>(p.frames) * p.height * p.width;
    vaeUnpatchifyClampKernel<<<gridFor(pixels), kThreads, 0, stream>>>(
        static_cast<T const*>(chunk), static_cast<T*>(out), p);
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
