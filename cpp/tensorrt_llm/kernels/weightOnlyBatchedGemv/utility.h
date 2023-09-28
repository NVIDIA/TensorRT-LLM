/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass_extensions/interleaved_numeric_conversion.h"

namespace tensorrt_llm
{
namespace kernels
{

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T>
struct GeluActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (val + 0.044715f * val * val * val))));
        return val * cdf;
    }
};

template <typename T>
struct ReluActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
    }
};

template <typename T>
struct IdentityActivation
{
    static __device__ __forceinline__ T apply(const T& val)
    {
        return val;
    }
};

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void load(T0* dst, T1* src, size_t offset = 0)
{
    *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<const VecType*>(src) + offset);
}

template <typename AssignType, typename T>
__device__ __forceinline__ void assign(T* dst, const AssignType& val)
{
    *reinterpret_cast<AssignType*>(dst) = val;
}

template <typename VecType, typename T0, typename T1>
__device__ __forceinline__ void store(T0* src, T1* dst, size_t offset = 0)
{
    *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<const VecType*>(src);
}
} // namespace kernels
} // namespace tensorrt_llm
