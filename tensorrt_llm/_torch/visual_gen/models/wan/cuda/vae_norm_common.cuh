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

// Shared building blocks of the Wan VAE norm kernels (runtime-compiled copy of
// cpp/tensorrt_llm/kernels/vaeNormCommon.cuh): element traits with packed pair conversions,
// the per-element RMSNorm + SiLU math and the lane layout.

#pragma once

#include "packed_vector.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vaewino
{

namespace vae_norm
{

// Shared building blocks of the Wan VAE norm kernels: element traits with packed
// pair conversions, the per-element RMSNorm + SiLU math and the lane layout that
// maps a channel row onto (lanes per row, 16-byte vectors per lane).

template <typename T>
struct Elem;

template <>
struct Elem<__nv_bfloat16>
{
    using Pair = __nv_bfloat162;

    static __device__ __forceinline__ float toFloat(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }

    static __device__ __forceinline__ float2 toFloat2(__nv_bfloat162 v)
    {
        return __bfloat1622float2(v);
    }

    static __device__ __forceinline__ __nv_bfloat162 fromFloat2(float2 v)
    {
        return __float22bfloat162_rn(v);
    }
};

template <>
struct Elem<__half>
{
    using Pair = __half2;

    static __device__ __forceinline__ float toFloat(__half v)
    {
        return __half2float(v);
    }

    static __device__ __forceinline__ float2 toFloat2(__half2 v)
    {
        return __half22float2(v);
    }

    static __device__ __forceinline__ __half2 fromFloat2(float2 v)
    {
        return __float22half2_rn(v);
    }
};

template <typename T, int V>
__device__ __forceinline__ float2 pairAt(vaewino::common::PackedAligned<T, V> const& v, int i)
{
    static_assert(V % 2 == 0, "vector width must be even for pair conversion");
    return Elem<T>::toFloat2(*reinterpret_cast<typename Elem<T>::Pair const*>(&v.val[2 * i]));
}

template <typename T, int V>
__device__ __forceinline__ void setPair(vaewino::common::PackedAligned<T, V>& v, int i, float2 f)
{
    *reinterpret_cast<typename Elem<T>::Pair*>(&v.val[2 * i]) = Elem<T>::fromFloat2(f);
}

// Round through the activation dtype and back; the conversion intrinsics are opaque
// to the compiler, so a multiply and an add on either side cannot be contracted.
template <typename T>
__device__ __forceinline__ float roundTo(float v)
{
    return Elem<T>::toFloat2(Elem<T>::fromFloat2(make_float2(v, v))).x;
}

struct RowLayout
{
    int lanesPerRow; // L: lanes cooperating on one row (divides 32)
    int vecsPerLane; // K: vectors of V elements each lane handles
};

template <int V>
inline RowLayout makeRowLayout(int channels)
{
    int const nvec = channels / V;
    int lanes = 32;
    while (nvec % lanes != 0)
    {
        lanes >>= 1;
    }
    return RowLayout{lanes, nvec / lanes};
}

// One normalized (+ activated, when ACT) element. REF reproduces the eager rounding
// sequence; otherwise gamma arrives pre-multiplied by scale (by 0.5 * scale with ACT, so
// SiLU is h * (1 + tanh(h)) with h = r / 2).
template <typename TIn, bool REF, bool HAS_BIAS, bool ACT = true>
__device__ __forceinline__ float normElement(float x, float denom, float invDenom, float scale, float gamma, float bias)
{
    if constexpr (REF)
    {
        float n = roundTo<TIn>(x / denom);
        float r = roundTo<TIn>(n * scale);
        r = roundTo<TIn>(r * gamma);
        if constexpr (HAS_BIAS)
        {
            r = roundTo<TIn>(r + bias);
        }
        if constexpr (!ACT)
        {
            return r;
        }
        return r / (1.0f + expf(-r));
    }
    else
    {
        float h = x * invDenom * gamma;
        if constexpr (!ACT)
        {
            if constexpr (HAS_BIAS)
            {
                h += bias;
            }
            return h;
        }
        if constexpr (HAS_BIAS)
        {
            h += 0.5f * bias;
        }
        float t;
        asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(h));
        return fmaf(h, t, h);
    }
}

// The lane layouts the kernels are instantiated for: (L, K) for the Wan VAE widths,
// the small powers of two and a few generic ones. Mirrored in wino_vae_kernels.py.
#define VAE_NORM_FOR_EACH_LAYOUT(X)                                                                                    \
    X(32, 1)                                                                                                           \
    X(32, 2)                                                                                                           \
    X(32, 4)                                                                                                           \
    X(4, 5)                                                                                                            \
    X(8, 5)                                                                                                            \
    X(16, 5)                                                                                                           \
    X(2, 3)                                                                                                            \
    X(4, 3)                                                                                                            \
    X(1, 1)                                                                                                            \
    X(2, 1)                                                                                                            \
    X(4, 1)                                                                                                            \
    X(8, 1)                                                                                                            \
    X(16, 1)                                                                                                           \
    X(1, 3)                                                                                                            \
    X(32, 3)                                                                                                           \
    X(1, 2)                                                                                                            \
    X(2, 2)                                                                                                            \
    X(4, 2)                                                                                                            \
    X(8, 2)                                                                                                            \
    X(16, 2)

} // namespace vae_norm

} // namespace vaewino
