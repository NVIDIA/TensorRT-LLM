/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
 * See the License for specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRTLLM_NCCL_DEVICE_MULTIMEM_H
#define TRTLLM_NCCL_DEVICE_MULTIMEM_H

#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include "constants.h"
#include "vector_types.h"
#include <cassert>
#include <cuda/std/cmath>
#include <cuda_fp8.h>
#include <type_traits>

namespace tensorrt_llm::kernels::nccl_device
{

// Architecture-specific feature detection based on PTX ISA documentation
// PTX ISA 8.1: Basic multimem support (sm_90+)
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
#define ARCH_HAS_MULTIMEM 1
#else
#define ARCH_HAS_MULTIMEM 0
#endif

// PTX ISA 8.2: .acc::f32 qualifier support (sm_90+)
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
#define ARCH_HAS_MULTIMEM_ACC_F32 1
#else
#define ARCH_HAS_MULTIMEM_ACC_F32 0
#endif

// PTX ISA 8.6: FP8 types and .acc::f16 qualifier support
// Supported on sm_100a, sm_101a (sm_110a), sm_120a, sm_121a
// And family-specific architectures sm_100f+, sm_101f+ (sm_110f+)
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
#define ARCH_HAS_MULTIMEM_FP8 1
#define ARCH_HAS_MULTIMEM_ACC_F16 1
#else
#define ARCH_HAS_MULTIMEM_FP8 0
#define ARCH_HAS_MULTIMEM_ACC_F16 0
#endif

// Basic data type support (independent of multimem)
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
#define ARCH_HAS_BF16 1
#else
#define ARCH_HAS_BF16 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890)
#define ARCH_HAS_FP8 1
#else
#define ARCH_HAS_FP8 0
#endif

// Base template for multimemLoadSum - device assert for SM < 90
template <typename ptrT, typename valT>
__device__ __forceinline__ valT multimemLoadSum(ptrT const* addr)
{
    assert(false && "multimemLoadSum requires SM90+ (Hopper) with multimem support. This operation cannot be emulated on older architectures.");
    return valT{}; // Unreachable, but satisfies return type requirement
}

// SM90+ specializations for supported types
#if ARCH_HAS_MULTIMEM
// Basic multimem support (PTX ISA 8.1)
template <>
__device__ __forceinline__ double multimemLoadSum<double, double>(double const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    double result;
    asm volatile("multimem.ld_reduce.global.add.f64 %0, [%1];" : "=d"(result) : "l"(multimem_addr) : "memory");
    return result;
}

template <>
__device__ __forceinline__ float multimemLoadSum<float, float>(float const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    float result;
    asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];" : "=f"(result) : "l"(multimem_addr) : "memory");
    return result;
}

template <>
__device__ __forceinline__ float2 multimemLoadSum<float, float2>(float const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    float2 result;
    asm volatile("multimem.ld_reduce.global.add.v2.f32 {%0,  %1}, [%2];"
                 : "=f"(result.x), "=f"(result.y)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}

template <>
__device__ __forceinline__ float4 multimemLoadSum<float, float4>(float const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    float4 result;
    asm volatile("multimem.ld_reduce.global.add.v4.f32 {%0,  %1, %2, %3}, [%4];"
                 : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}

#if ARCH_HAS_MULTIMEM_ACC_F32
// .acc::f32 qualifier support (PTX ISA 8.2)
template <>
__device__ __forceinline__ HalfVector multimemLoadSum<half, HalfVector>(half const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    HalfVector result;
    asm volatile("multimem.ld_reduce.global.add.v4.f16x2.acc::f32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(result.data.x), "=r"(result.data.y), "=r"(result.data.z), "=r"(result.data.w)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}

template <>
__device__ __forceinline__ BFloat16Vector multimemLoadSum<__nv_bfloat16, BFloat16Vector>(__nv_bfloat16 const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    BFloat16Vector result;
    asm volatile("multimem.ld_reduce.global.add.v4.bf16x2.acc::f32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(result.data.x), "=r"(result.data.y), "=r"(result.data.z), "=r"(result.data.w)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}
#endif // ARCH_HAS_MULTIMEM_ACC_F32

#if ARCH_HAS_MULTIMEM_FP8
// FP8 types and .acc::f16 qualifier support (PTX ISA 8.6)
template <>
__device__ __forceinline__ FP8E5M2x4Vector multimemLoadSum<__nv_fp8_e5m2, FP8E5M2x4Vector>(__nv_fp8_e5m2 const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    FP8E5M2x4Vector result;
    asm volatile("multimem.ld_reduce.global.add.v4.e5m2x4.acc::f16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(result.data.x), "=r"(result.data.y), "=r"(result.data.z), "=r"(result.data.w)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}

template <>
__device__ __forceinline__ FP8E4M3x4Vector multimemLoadSum<__nv_fp8_e4m3, FP8E4M3x4Vector>(__nv_fp8_e4m3 const* addr)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    FP8E4M3x4Vector result;
    asm volatile("multimem.ld_reduce.global.add.v4.e4m3x4.acc::f16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(result.data.x), "=r"(result.data.y), "=r"(result.data.z), "=r"(result.data.w)
                 : "l"(multimem_addr)
                 : "memory");
    return result;
}
#endif // ARCH_HAS_MULTIMEM_FP8
#endif // ARCH_HAS_MULTIMEM

// Base template for multimemStore - device assert for SM < 90
template <typename ptrT, typename valT>
__device__ __forceinline__ void multimemStore(ptrT* addr, valT const val)
{
    assert(false && "multimemStore requires SM90+ (Hopper) with multimem support. This operation cannot be emulated on older architectures.");
}

// SM90+ specializations for supported types
#if ARCH_HAS_MULTIMEM
// Basic multimem support (PTX ISA 8.1)
template <>
__device__ __forceinline__ void multimemStore<double, double>(double* addr, double const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.f64 [%0], %1;" : : "l"(multimem_addr), "d"(val) : "memory");
}

template <>
__device__ __forceinline__ void multimemStore<float, float>(float* addr, float const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.f32 [%0], %1;" : : "l"(multimem_addr), "f"(val) : "memory");
}

template <>
__device__ __forceinline__ void multimemStore<float, float2>(float* addr, float2 const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v2.f32 [%0], {%1, %2};" : : "l"(multimem_addr), "f"(val.x), "f"(val.y) : "memory");
}

template <>
__device__ __forceinline__ void multimemStore<float, float4>(float* addr, float4 const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v4.f32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(multimem_addr), "f"(val.x), "f"(val.y), "f"(val.z), "f"(val.w)
                 : "memory");
}

template <>
__device__ __forceinline__ void multimemStore<half, HalfVector>(half* addr, HalfVector const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(multimem_addr), "r"(val.data.x), "r"(val.data.y), "r"(val.data.z), "r"(val.data.w)
                 : "memory");
}

#if ARCH_HAS_MULTIMEM_ACC_F32
template <>
__device__ __forceinline__ void multimemStore<__nv_bfloat16, BFloat16Vector>(
    __nv_bfloat16* addr, const BFloat16Vector val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v4.bf16x2 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(multimem_addr), "r"(val.data.x), "r"(val.data.y), "r"(val.data.z), "r"(val.data.w)
                 : "memory");
}
#endif // ARCH_HAS_MULTIMEM_ACC_F32

#if ARCH_HAS_MULTIMEM_FP8
// FP8 types support (PTX ISA 8.6)
template <>
__device__ __forceinline__ void multimemStore<__nv_fp8_e5m2, FP8E5M2x4Vector>(
    __nv_fp8_e5m2* addr, FP8E5M2x4Vector const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v4.e5m2x4 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(multimem_addr), "r"(val.data.x), "r"(val.data.y), "r"(val.data.z), "r"(val.data.w)
                 : "memory");
}

template <>
__device__ __forceinline__ void multimemStore<__nv_fp8_e4m3, FP8E4M3x4Vector>(
    __nv_fp8_e4m3* addr, FP8E4M3x4Vector const val)
{
    uintptr_t const multimem_addr = reinterpret_cast<uintptr_t>(addr);
    asm volatile("multimem.st.global.v4.e4m3x4 [%0], {%1,%2,%3,%4};"
                 :
                 : "l"(multimem_addr), "r"(val.data.x), "r"(val.data.y), "r"(val.data.z), "r"(val.data.w)
                 : "memory");
}
#endif // ARCH_HAS_MULTIMEM_FP8
#endif // ARCH_HAS_MULTIMEM

} // namespace tensorrt_llm::kernels::nccl_device

#endif // TRTLLM_NCCL_DEVICE_MULTIMEM_H
