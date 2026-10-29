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
#include "vaeUpsampleGatherKernel.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

using common::PackedAligned;

constexpr int kThreads = 256;
constexpr int kVec = 8; // output channels per thread: one 16-byte store

template <typename T>
struct Cvt;

template <>
struct Cvt<__nv_bfloat16>
{
    static __device__ __forceinline__ float toF(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }

    static __device__ __forceinline__ __nv_bfloat16 fromF(float v)
    {
        return __float2bfloat16_rn(v);
    }
};

template <>
struct Cvt<__half>
{
    static __device__ __forceinline__ float toF(__half v)
    {
        return __half2float(v);
    }

    static __device__ __forceinline__ __half fromF(float v)
    {
        return __float2half_rn(v);
    }
};

// The shortcut channels needed by one thread (output channels o..o+7, all phases of one
// source pixel) are the contiguous span starting at element o*FACTOR/REPEATS with
// max(1, 8*FACTOR/REPEATS) elements; load it with the widest aligned vectors available.
template <typename T, int FACTOR, int REPEATS>
struct ShortcutSpan
{
    static constexpr int kSpan = (kVec * FACTOR) / REPEATS > 0 ? (kVec* FACTOR) / REPEATS : 1;
    static constexpr int kChunk = kSpan < kVec ? kSpan : kVec;
    static constexpr int kChunks = kSpan / kChunk;
    using Chunk = PackedAligned<T, kChunk>;

    T val[kSpan];

    __device__ __forceinline__ void load(T const* __restrict__ base)
    {
#pragma unroll
        for (int c = 0; c < kChunks; ++c)
        {
            Chunk chunk = *reinterpret_cast<Chunk const*>(base + c * kChunk);
#pragma unroll
            for (int e = 0; e < kChunk; ++e)
            {
                val[c * kChunk + e] = chunk.val[e];
            }
        }
    }
};

template <typename T, int FACTOR, int REPEATS, int A, int P, int Q, typename Span>
__device__ __forceinline__ void addShortcut(PackedAligned<T, kVec>& v, Span const& sc)
{
    constexpr int S = (A * 2 + P) * 2 + Q;
#pragma unroll
    for (int k = 0; k < kVec; ++k)
    {
        T const s = sc.val[(k * FACTOR + S) / REPEATS]; // static after unrolling
        v.val[k] = Cvt<T>::fromF(Cvt<T>::toF(v.val[k]) + Cvt<T>::toF(s));
    }
}

// One thread owns one source pixel (b, tr, i, j) and 8 output channels, and produces
// all FACTOR_T x 2 x 2 output pixels that descend from it: FACTOR_T*4 16-byte loads
// from temp, one shortcut span load, FACTOR_T*4 16-byte stores in final channels-last
// order. Consecutive threads hold consecutive channel vectors of the same pixel, so
// every warp-wide load/store is 512 contiguous bytes.
// HAS_BIAS: the packed convolution ran without its bias; it is applied here as
// round(temp + bias[o]) before the shortcut add, like the eager bias add would be.
template <typename T, int FACTOR_T, int REPEATS, bool SHORTCUT, bool HAS_BIAS>
__global__ void __launch_bounds__(kThreads)
    vaeUpsampleGatherKernel(T const* __restrict__ temp, T const* __restrict__ shortcut, T const* __restrict__ bias,
        T* __restrict__ out, VaeUpsampleGatherParams p, int total)
{
    using Vec = PackedAligned<T, kVec>;
    constexpr int FACTOR = FACTOR_T * 4;
    int const idx = blockIdx.x * kThreads + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int const coutVec = p.cout / kVec;
    int const o = (idx % coutVec) * kVec;
    int pix = idx / coutVec;
    int const j = pix % p.width;
    pix /= p.width;
    int const i = pix % p.height;
    pix /= p.height;
    int const framesR = (p.frames + p.crop) / FACTOR_T;
    int const tr = pix % framesR;
    int const b = pix / framesR;

    Vec bv;
    if constexpr (HAS_BIAS)
    {
        bv = *reinterpret_cast<Vec const*>(bias + o);
    }
    ShortcutSpan<T, FACTOR, REPEATS> sc;
    if constexpr (SHORTCUT)
    {
        int64_t const pixBase = ((static_cast<int64_t>(b) * framesR + tr) * p.height + i) * p.width + j;
        sc.load(shortcut + pixBase * p.cin + (static_cast<int64_t>(o) * FACTOR) / REPEATS);
    }

    int64_t const tempPix = static_cast<int64_t>(p.height + 1) * (p.width + 1);
    int64_t const tempRow = static_cast<int64_t>(p.width + 1) * (4 * p.cout);
    int64_t const outRow = static_cast<int64_t>(2 * p.width) * p.cout;
    int64_t const outFrame = static_cast<int64_t>(2 * p.height) * outRow;

#pragma unroll
    for (int a = 0; a < FACTOR_T; ++a)
    {
        int const f = tr * FACTOR_T + a - p.crop; // output frame; negative on the cropped first chunk
        if (f < 0)
        {
            continue;
        }
        int64_t const bt = static_cast<int64_t>(b) * p.frames + f;
        T const* const tempBase
            = temp + (bt * tempPix + static_cast<int64_t>(i) * (p.width + 1) + j) * (4 * p.cout) + o;
        T* const outBase
            = out + bt * outFrame + static_cast<int64_t>(2 * i) * outRow + static_cast<int64_t>(2 * j) * p.cout + o;
        Vec v[4];
#pragma unroll
        for (int ph = 0; ph < 4; ++ph)
        {
            int const pp = ph >> 1, qq = ph & 1;
            v[ph] = *reinterpret_cast<Vec const*>(tempBase + pp * tempRow + qq * (4 * p.cout) + ph * p.cout);
        }
        if constexpr (HAS_BIAS)
        {
#pragma unroll
            for (int ph = 0; ph < 4; ++ph)
            {
#pragma unroll
                for (int k = 0; k < kVec; ++k)
                {
                    v[ph].val[k] = Cvt<T>::fromF(Cvt<T>::toF(v[ph].val[k]) + Cvt<T>::toF(bv.val[k]));
                }
            }
        }
        if constexpr (SHORTCUT)
        {
            if (a == 0)
            {
                addShortcut<T, FACTOR, REPEATS, 0, 0, 0>(v[0], sc);
                addShortcut<T, FACTOR, REPEATS, 0, 0, 1>(v[1], sc);
                addShortcut<T, FACTOR, REPEATS, 0, 1, 0>(v[2], sc);
                addShortcut<T, FACTOR, REPEATS, 0, 1, 1>(v[3], sc);
            }
            else
            {
                addShortcut<T, FACTOR, REPEATS, 1, 0, 0>(v[0], sc);
                addShortcut<T, FACTOR, REPEATS, 1, 0, 1>(v[1], sc);
                addShortcut<T, FACTOR, REPEATS, 1, 1, 0>(v[2], sc);
                addShortcut<T, FACTOR, REPEATS, 1, 1, 1>(v[3], sc);
            }
        }
#pragma unroll
        for (int ph = 0; ph < 4; ++ph)
        {
            int const pp = ph >> 1, qq = ph & 1;
            *reinterpret_cast<Vec*>(outBase + pp * outRow + qq * p.cout) = v[ph];
        }
    }
}

template <typename T, int FACTOR_T, int REPEATS, bool SHORTCUT>
void launchKernel(void const* temp, void const* shortcut, void const* bias, void* out, VaeUpsampleGatherParams const& p,
    cudaStream_t stream)
{
    int const framesR = (p.frames + p.crop) / FACTOR_T;
    int64_t const total64 = static_cast<int64_t>(p.batch) * framesR * p.height * p.width * (p.cout / kVec);
    TLLM_CHECK_WITH_INFO(total64 < (int64_t{1} << 31), "vaeUpsampleGather: too many source pixels");
    int const total = static_cast<int>(total64);
    if (total == 0)
    {
        return;
    }
    unsigned const blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    if (bias != nullptr)
    {
        vaeUpsampleGatherKernel<T, FACTOR_T, REPEATS, SHORTCUT, true>
            <<<blocks, kThreads, 0, stream>>>(reinterpret_cast<T const*>(temp), reinterpret_cast<T const*>(shortcut),
                reinterpret_cast<T const*>(bias), reinterpret_cast<T*>(out), p, total);
    }
    else
    {
        vaeUpsampleGatherKernel<T, FACTOR_T, REPEATS, SHORTCUT, false>
            <<<blocks, kThreads, 0, stream>>>(reinterpret_cast<T const*>(temp), reinterpret_cast<T const*>(shortcut),
                nullptr, reinterpret_cast<T*>(out), p, total);
    }
}

template <typename T, int FACTOR_T>
void launchRepeats(void const* temp, void const* shortcut, void const* bias, void* out,
    VaeUpsampleGatherParams const& p, cudaStream_t stream)
{
    if (shortcut == nullptr)
    {
        launchKernel<T, FACTOR_T, 1, false>(temp, nullptr, bias, out, p, stream);
        return;
    }
    switch (p.repeats)
    {
    case 1: launchKernel<T, FACTOR_T, 1, true>(temp, shortcut, bias, out, p, stream); break;
    case 2: launchKernel<T, FACTOR_T, 2, true>(temp, shortcut, bias, out, p, stream); break;
    case 4: launchKernel<T, FACTOR_T, 4, true>(temp, shortcut, bias, out, p, stream); break;
    case 8: launchKernel<T, FACTOR_T, 8, true>(temp, shortcut, bias, out, p, stream); break;
    case 16: launchKernel<T, FACTOR_T, 16, true>(temp, shortcut, bias, out, p, stream); break;
    case 32: launchKernel<T, FACTOR_T, 32, true>(temp, shortcut, bias, out, p, stream); break;
    default: TLLM_THROW("vaeUpsampleGather: unsupported repeats %d (1..32, power of two)", p.repeats);
    }
}

template <typename T>
void launchT(void const* temp, void const* shortcut, void const* bias, void* out, VaeUpsampleGatherParams const& p,
    cudaStream_t stream)
{
    switch (p.factorT)
    {
    case 1: launchRepeats<T, 1>(temp, shortcut, bias, out, p, stream); break;
    case 2: launchRepeats<T, 2>(temp, shortcut, bias, out, p, stream); break;
    default: TLLM_THROW("vaeUpsampleGather: unsupported factorT %d (1 or 2)", p.factorT);
    }
}

} // namespace

bool vaeUpsampleGatherSupported(VaeUpsampleGatherParams const& p)
{
    if (p.cout % kVec != 0 || (p.factorT != 1 && p.factorT != 2))
    {
        return false;
    }
    if (p.cin == 0)
    {
        return true;
    }
    bool const pow2 = p.repeats >= 1 && p.repeats <= 32 && (p.repeats & (p.repeats - 1)) == 0;
    return pow2 && (p.frames + p.crop) % p.factorT == 0;
}

void launchVaeUpsampleGather(void const* temp, void const* shortcut, void const* bias, void* out,
    VaeUpsampleGatherParams const& p, bool bf16, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(vaeUpsampleGatherSupported(p), "vaeUpsampleGather: unsupported parameters");
    VaeUpsampleGatherParams q = p;
    if (shortcut == nullptr)
    {
        // Without a shortcut every output frame is its own source frame.
        q.factorT = 1;
        q.crop = 0;
        q.repeats = 1;
    }
    if (bf16)
    {
        launchT<__nv_bfloat16>(temp, shortcut, bias, out, q, stream);
    }
    else
    {
        launchT<__half>(temp, shortcut, bias, out, q, stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
