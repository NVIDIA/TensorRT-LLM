/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Unified internal kernel-side header for the Marlin NVFP4 kernels.
//
// Replaces:
//   - marlin.cuh         (compile-time params, helpers, cp.async wrappers)
//   - marlin_dtypes.cuh  (MarlinType<scalar_t> traits + fragment aliases)
//   - dequant.h          (FP4->BF16 and FP8->BF16 dequantization)
//   - marlin_mma.h       (BF16 m16n8k16 tensor-core MMA instructions)
//
// Layout, top to bottom:
//   1. Compile-time constants (tile / thread / stage sizes)
//   2. Vec<T,n> helper + div_ceil
//   3. cp.async wrappers (SM 7.x fallback / SM 8.x+ inline asm)
//   4. MarlinType<scalar_t> type traits + fragment aliases
//   5. FP4 / FP8 dequantization
//   6. BF16 m16n8k16 MMA wrappers (and transposed variant)

#pragma once

#ifndef _marlin_cuh
#define _marlin_cuh

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <type_traits>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME
{

// =========================================================================
// 1. Compile-time constants
// =========================================================================

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages = 4; // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// =========================================================================
// 2. Helpers
// =========================================================================

template <typename T, int n>
struct Vec
{
    T elems[n];

    __device__ T& operator[](int i)
    {
        return elems[i];
    }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b)
{
    return (a + b - 1) / b;
}

// =========================================================================
// 3. cp.async wrappers
// =========================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__device__ inline void cp_async1_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    if (pred)
    {
        reinterpret_cast<int32_t*>(smem_ptr)[0] = reinterpret_cast<int32_t const*>(glob_ptr)[0];
    }
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    if (pred)
    {
        reinterpret_cast<int64_t*>(smem_ptr)[0] = reinterpret_cast<int64_t const*>(glob_ptr)[0];
    }
}

__device__ inline void cp_async4_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    if (pred)
    {
        reinterpret_cast<int4*>(smem_ptr)[0] = reinterpret_cast<int4 const*>(glob_ptr)[0];
    }
}

__device__ inline void cp_async4_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    if (pred)
    {
        reinterpret_cast<int4*>(smem_ptr)[0] = reinterpret_cast<int4 const*>(glob_ptr)[0];
    }
}

__device__ inline void cp_async4(void* smem_ptr, void const* glob_ptr)
{
    reinterpret_cast<int4*>(smem_ptr)[0] = reinterpret_cast<int4 const*>(glob_ptr)[0];
}

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait()
{
}

#else

__device__ inline void cp_async1_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    int const BYTES = 4;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    int const BYTES = 8;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_ca_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    int const BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_pred(void* smem_ptr, void const* glob_ptr, bool pred = true)
{
    int const BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) pred),
        "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, void const* glob_ptr)
{
    int const BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

#endif

// =========================================================================
// 4. MarlinType<scalar_t> type traits + fragment aliases
//    (formerly marlin_dtypes.cuh)
//
//    Matrix fragment layouts documented at:
//    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//      #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
// =========================================================================

template <typename T>
struct MarlinType
{
};

template <>
struct MarlinType<nv_bfloat16>
{
    using scalar_t = nv_bfloat16;
    using scalar_t2 = nv_bfloat162;
    using scalar_t4 = nv_bfloat162;
    using scalar_32bit_t = nv_bfloat162;

    using FragA = Vec<nv_bfloat162, 4>;
    using FragB = Vec<nv_bfloat162, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<nv_bfloat162, 1>;
    using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
    using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static __device__ float inline num2float(const nv_bfloat16 x)
    {
        return __bfloat162float(x);
    }

    static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x)
    {
        return __bfloat162bfloat162(x);
    }

    static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1, const nv_bfloat16 x2)
    {
        return __halves2bfloat162(x1, x2);
    }

    static __host__ __device__ nv_bfloat16 inline float2num(float const x)
    {
        return __float2bfloat16(x);
    }

    static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x)
    {
        return __bfloat1622float2(x);
    }
#endif
};

// =========================================================================
// 5. FP4 / FP8 dequantization
//    (formerly dequant.h)
//
//    Fast dequantization for NVFP4 (FP4 E2M1 weights -> BF16) and
//    FP8 E4M3 scale dequantization -> BF16.
//
//    The FP4->BF16 dequant shifts the 3-bit FP4 value (sign + 2-bit
//    exponent + 1-bit mantissa) into the BF16 exponent/mantissa fields via
//    bitwise ops. A subsequent multiply by 2^(bias_offset) corrects the
//    exponent bias (skip_flop=false) or is deferred to fuse with scale
//    multiply (skip_flop=true).
// =========================================================================

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750

// Lookup-table based 3-input logical operation; the compiler does not always
// recognize the pattern automatically.
template <int lut>
__device__ inline int lop3(int a, int b, int c)
{
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// FP4 E2M1 -> BF16
//
// skip_flop=true: just place bits, caller multiplies exponent bias later.
template <bool skip_flop>
__device__ inline void dequant_fp4(int q, nv_bfloat162* frag_b)
{
    // Constants for FP4 (E2M1) -> BF16 (E8M7)
    constexpr int FP4_EXPONENT = 2, BF16_EXPONENT = 8;
    constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP4_EXPONENT;
    constexpr int MASK = 0x70007000;

    // Extract and shift FP4 values to BF16 format
    int Out1 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 4;
    int Out2 = (q & 0x80008000) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&Out2);

    if constexpr (!skip_flop)
    {
        // Apply exponent bias correction
        constexpr int BIAS_OFFSET = (1 << (BF16_EXPONENT - 1)) - (1 << (FP4_EXPONENT - 1));
        constexpr uint32_t BIAS = (BIAS_OFFSET + 127) << 23;
        const nv_bfloat162 bias_reg = __float2bfloat162_rn(*reinterpret_cast<float const*>(&BIAS));

        frag_b[1] = __hmul2(frag_b[1], bias_reg);
        frag_b[0] = __hmul2(frag_b[0], bias_reg);
    }
}

// FP8 E4M3 scale -> BF16
__device__ inline void dequant_fp8_scales(int q, nv_bfloat162* frag_b)
{
    constexpr int FP8_EXPONENT = 4, BF16_EXPONENT = 8;
    constexpr int RIGHT_SHIFT = BF16_EXPONENT - FP8_EXPONENT;
    constexpr int MASK = 0x7F007F00;

    // Extract and shift FP8 values to BF16 format
    int Out1 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);
    q <<= 8;
    int Out2 = ((q & 0x80008000) >> 1) | ((q & MASK) >> RIGHT_SHIFT);

    // Note: reverse indexing is intentional because weights are permuted
    frag_b[1] = *reinterpret_cast<nv_bfloat162 const*>(&Out1);
    frag_b[0] = *reinterpret_cast<nv_bfloat162 const*>(&Out2);
}

#endif // __CUDA_ARCH__ >= 750

// =========================================================================
// 6. BF16 m16n8k16 tensor-core MMA wrappers
//    (formerly marlin_mma.h)
// =========================================================================

// m16n8k16 tensor core mma: BF16 inputs, FP32 accumulation.
template <typename scalar_t>
__device__ inline void mma(const typename MarlinType<scalar_t>::FragA& a_frag,
    const typename MarlinType<scalar_t>::FragB& frag_b, typename MarlinType<scalar_t>::FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);

    static_assert(std::is_same<scalar_t, nv_bfloat16>::value, "Only BF16 is supported for Marlin NVFP4 MMA");

    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Transposed variant: B is transposed (used for column-major weight loading).
template <typename scalar_t>
__device__ inline void mma_trans(const typename MarlinType<scalar_t>::FragA& a_frag,
    const typename MarlinType<scalar_t>::FragB& frag_b, const typename MarlinType<scalar_t>::FragB& frag_b2,
    typename MarlinType<scalar_t>::FragC& frag_c)
{
    uint32_t const* a = reinterpret_cast<uint32_t const*>(&a_frag);
    uint32_t const* b = reinterpret_cast<uint32_t const*>(&frag_b);
    uint32_t const* b2 = reinterpret_cast<uint32_t const*>(&frag_b2);

    static_assert(std::is_same<scalar_t, nv_bfloat16>::value, "Only BF16 is supported for Marlin NVFP4 MMA");

    float* c = reinterpret_cast<float*>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]), "r"(a[0]), "r"(a[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]));
}

} // namespace MARLIN_NAMESPACE_NAME

// =========================================================================
// 7. Single-expert Marlin kernel: MARLIN_KERNEL_PARAMS + Marlin<> forward decl
//    (formerly marlin_nvfp4_kernel.h)
//
//    Opt in by `#define MARLIN_DECLARE_SINGLE_EXPERT_KERNEL` before
//    `#include "marlin.cuh"`. The MoE translation unit must NOT define
//    the guard because the MoE kernel uses a different parameter list,
//    declared in marlin_nvfp4_moe_template.h.
// =========================================================================

#ifdef MARLIN_DECLARE_SINGLE_EXPERT_KERNEL

#define MARLIN_KERNEL_PARAMS                                                                                           \
    const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
        const int4 *__restrict__ b_bias_ptr, const float *__restrict__ a_scales_ptr,                                   \
        const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ global_scale_ptr,                            \
        const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n,        \
        int prob_k, int lda, int *locks, bool has_bias, bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

namespace MARLIN_NAMESPACE_NAME
{

template <typename scalar_t,   // compute type (nv_bfloat16)
    int const threads,         // threads per block (128 or 256)
    int const thread_m_blocks, // 16x16 blocks in M dimension
    int const thread_n_blocks, // 16x16 blocks in N dimension
    int const thread_k_blocks, // 16x16 blocks in K dimension
    bool const m_block_size_8, // use 8-row M blocks (thread_m_blocks==1)
    int const stages,          // async pipeline stages
    int const group_blocks     // consecutive blocks per scale group
    >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

} // namespace MARLIN_NAMESPACE_NAME

#endif // MARLIN_DECLARE_SINGLE_EXPERT_KERNEL

#endif // _marlin_cuh
