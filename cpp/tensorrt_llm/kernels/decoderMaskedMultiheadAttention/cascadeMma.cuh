/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace mmha
{
namespace cascade
{
namespace mma
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Low-level PTX wrappers for Tensor Core attention on SM80+.  Adapted from
// `cpp/tensorrt_llm/kernels/selectiveScan/Common.h` and kept self-contained
// so the cascade module does not leak a dependency on the Mamba kernel dir.
//
// Supported ops (bf16 + f16 via mma<Tp_>):
//   - mma.sync.aligned.m16n8k16.row.col.f32.{bf16|f16}...   (16x8 output)
//   - cp.async.ca.shared.global {.b4, .b8, .b16}            (SM80+ pipelined)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert a generic pointer into a 32-bit .shared address suitable for PTX.
__device__ __forceinline__ unsigned smem_addr_of(void const* smem_ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Element packing helpers: two 16-bit elements -> one uint32 (little-endian).
// Used to manually build MMA A/B fragments without ldmatrix.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__device__ __forceinline__ unsigned pack2(T a, T b);

template <>
__device__ __forceinline__ unsigned pack2<half>(half a, half b)
{
    unsigned short ua = __half_as_ushort(a);
    unsigned short ub = __half_as_ushort(b);
    return (static_cast<unsigned>(ub) << 16) | static_cast<unsigned>(ua);
}

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ unsigned pack2<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b)
{
    unsigned short ua = __bfloat16_as_ushort(a);
    unsigned short ub = __bfloat16_as_ushort(b);
    return (static_cast<unsigned>(ub) << 16) | static_cast<unsigned>(ua);
}
#endif

// Read two 16-bit elements from SMEM at adjacent addresses (ptr[0], ptr[1]) as a
// packed uint32.  Assumes 32-bit alignment of ptr (i.e. even index in T array).
template <typename T>
__device__ __forceinline__ unsigned load_pack2(T const* ptr)
{
    return *reinterpret_cast<unsigned const*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// mma.sync.aligned.m16n8k16
//
// Thread-fragment layout (per PTX 8.x spec):
//   A (M=16, K=16, row-major, bf16):
//       Thread t owns 8 bf16 = 4 uint32:
//         a[0] -> (row = t/4,     col = 2*(t%4) + {0,1})  (within K=0..7)
//         a[1] -> (row = t/4 + 8, col = 2*(t%4) + {0,1})  (within K=0..7)
//         a[2] -> (row = t/4,     col = 2*(t%4) + {8,9})  (within K=8..15)
//         a[3] -> (row = t/4 + 8, col = 2*(t%4) + {8,9})  (within K=8..15)
//   B (K=16, N=8, col-major, bf16):
//       Thread t owns 4 bf16 = 2 uint32:
//         b[0] -> (col = t/4,     row = 2*(t%4) + {0,1})  covering K=0..7
//         b[1] -> (col = t/4,     row = 2*(t%4) + {8,9})  covering K=8..15
//   C (M=16, N=8, f32):
//       Thread t owns 4 f32:
//         c[0] -> (row = t/4,     col = 2*(t%4))
//         c[1] -> (row = t/4,     col = 2*(t%4) + 1)
//         c[2] -> (row = t/4 + 8, col = 2*(t%4))
//         c[3] -> (row = t/4 + 8, col = 2*(t%4) + 1)
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void mma_m16n8k16_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 \n"
        "{%0, %1, %2, %3},\n"
        "{%4, %5, %6, %7},\n"
        "{%8, %9},\n"
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_m16n8k16_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \n"
        "{%0, %1, %2, %3},\n"
        "{%4, %5, %6, %7},\n"
        "{%8, %9},\n"
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// Type-dispatched wrapper.  T is the element type (__nv_bfloat16 or half).
template <typename T>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1);

#ifdef ENABLE_BF16
template <>
__device__ __forceinline__ void mma_m16n8k16<__nv_bfloat16>(float& c0, float& c1, float& c2, float& c3, unsigned a0,
    unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
#endif

template <>
__device__ __forceinline__ void mma_m16n8k16<half>(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
    unsigned a2, unsigned a3, unsigned b0, unsigned b1)
{
    mma_m16n8k16_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// cp.async  (SM80+).  Transfers from global to shared memory without staging
// through registers, enabling SW pipelining of HBM LDG with MMA compute.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Issue a 16-byte async copy.  smem_addr must be 16B aligned (shared-space),
// global_ptr must be 16B aligned (generic).
//
// Optional `src_size` (PTX `cp.async` 4th operand) controls hardware zero-fill:
//   * src_size == 16 (default): full 16B copy, equivalent to omitting the operand.
//   * src_size <  16          : bytes [src_size, 16) of the destination SMEM
//                                are zeroed by the LSU.
//   * src_size ==  0          : no global load is issued at all (per PTX ISA:
//                                "if src-size is zero, the access is a no-op");
//                                the destination 16B are simply zeroed.
// Callers can therefore use a single uniform issue for both in-bounds and
// out-of-bounds chunks (passing 16 or 0 respectively) instead of branching
// on an explicit SMEM zero store.  When `src_size` is the immediate `16`,
// ptxas emits the compact form without the `src-size` operand.
__device__ __forceinline__ void cp_async_16B(unsigned smem_addr, void const* global_ptr, unsigned src_size = 16u)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(global_ptr), "r"(src_size));
#else
    unsigned tmp[4] = {0u, 0u, 0u, 0u};
    if (src_size >= 16u)
    {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
                     : "l"(global_ptr));
    }
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_addr), "r"(tmp[0]), "r"(tmp[1]), "r"(tmp[2]),
        "r"(tmp[3]));
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int remain>
__device__ __forceinline__ void cp_async_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"n"(remain));
#endif
}

} // namespace mma
} // namespace cascade
} // namespace mmha
} // namespace kernels

TRTLLM_NAMESPACE_END
