/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <device_types.h>
#include <vector_types.h>

namespace trtllm
{
namespace dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ T clamp(T x, T lb, T ub)
{
    return (x < lb) ? lb : (x > ub ? ub : x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ void cpAsync(
    T* dst, T const* src, int32_t dstOffset = 0, int64_t srcOffset = 0, int const cpSize = sizeof(T))
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (cpSize == 4)
    {
        uint32_t* dstUInt32 = reinterpret_cast<uint32_t*>(dst + dstOffset);
        uint32_t const* srcUInt32 = reinterpret_cast<uint32_t const*>(src + srcOffset);
        uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt32));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dstU32), "l"(srcUInt32));
    }
    else if (cpSize == 8)
    {
        uint64_t* dstUInt64 = reinterpret_cast<uint64_t*>(dst + dstOffset);
        uint64_t const* srcUInt64 = reinterpret_cast<uint64_t const*>(src + srcOffset);
        uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt64));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dstU32), "l"(srcUInt64));
    }
    else if (cpSize == 16)
    {
        uint4* dstUInt128 = reinterpret_cast<uint4*>(dst + dstOffset);
        uint4 const* srcUInt128 = reinterpret_cast<uint4 const*>(src + srcOffset);
        uint32_t dstU32 = static_cast<uint32_t>(__cvta_generic_to_shared(dstUInt128));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dstU32), "l"(srcUInt128));
    }
    else
    {
        assert(0 && "cpSize is not supported"); // The compiler will eliminate that code.
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void cpAsyncCommitGroup()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ void cpAsyncWaitGroup()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Strong compiler hint to prevent code motion across the fence.
__forceinline__ __device__ void cfence()
{
#if defined(__CUDA_ARCH__)
    asm volatile(".pragma \"next knob FenceCode\";\n" : : : "memory");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Lesser compiler hint to prevent code motion across the fence.
__forceinline__ __device__ void ifence()
{
#if defined(__CUDA_ARCH__)
    asm volatile(".pragma \"next knob FenceInterference\";\n" : : : "memory");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
