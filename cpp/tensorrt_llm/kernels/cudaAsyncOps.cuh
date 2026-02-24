/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <stdint.h>

#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

// ============================================================================
// Address Conversion Utilities
// ============================================================================

static __device__ __forceinline__ uint32_t __as_ptr_smem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
}

static __device__ __forceinline__ uint64_t __as_ptr_gmem(void const* __ptr)
{
    // Consider adding debug asserts here.
    return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
}

// ============================================================================
// Memory Fence Operations
// ============================================================================

__device__ __forceinline__ void fence_release_sys()
{
    asm volatile("fence.release.sys;" : : : "memory");
}

// ============================================================================
// Memory Barrier Operations (mbarrier)
// ============================================================================

__device__ __forceinline__ void mbarrier_init(uint64_t* addr, uint32_t const& count)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(addr)), "r"(count) : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* addr)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    uint64_t state;
    asm("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "r"(__as_ptr_smem(addr)) : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* addr, const uint32_t txCount)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint64_t state;
    asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(__as_ptr_smem(addr)), "r"(txCount)
        : "memory");
    return state;
#else
    return 0;
#endif
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* addr, uint32_t const& phaseParity)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    uint32_t waitComplete;
    asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(waitComplete)
        : "r"(__as_ptr_smem(addr)), "r"(phaseParity)
        : "memory");
    return static_cast<bool>(waitComplete);
#else
    return false;
#endif
}

// ============================================================================
// Async Copy Operations (cp.async for SM80+)
// ============================================================================

template <int COPY_SIZE = 4>
__device__ __forceinline__ void ldgsts(int* dstShm, int const* srcMem, bool predGuard)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int) predGuard),
        "r"(__as_ptr_smem(dstShm)), "l"(__as_ptr_gmem(srcMem)), "n"(COPY_SIZE));
#endif
}

__device__ __forceinline__ void cp_async_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" : : "n"(N) : "memory");
#endif
}

// ============================================================================
// Bulk Async Copy Operations (cp.async.bulk for SM90+)
// ============================================================================

__device__ __forceinline__ void cp_async_bulk_g2s(void* dstMem, void const* srcMem, int copySize, uint64_t* smemBar)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :
        : "r"(__as_ptr_smem(dstMem)), "l"(__as_ptr_gmem(srcMem)), "r"(copySize), "r"(__as_ptr_smem(smemBar))
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_s2g(void* dstMem, void const* srcMem, int copySize)
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(copySize)
        : "memory");
#endif
}

__device__ __forceinline__ void cp_async_bulk_commit_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.commit_group;" : : :);
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
#endif
}

template <int N = 0>
__device__ __forceinline__ void cp_async_bulk_wait_group_read()
{
#if defined(__CUDACC__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
#endif
}

// ============================================================================
// Shared Memory Barrier Helpers
// ============================================================================

__device__ __forceinline__ void initSmemBar(uint64_t* smemBar, int laneId)
{
    if (laneId == 0)
    {
        mbarrier_init(smemBar, WARP_SIZE);
    }
    __syncwarp();
}

__device__ __forceinline__ void smemBarWait(uint64_t* smemBar, uint32_t* phaseParity)
{
    while (!mbarrier_try_wait_parity(smemBar, *phaseParity))
    {
    }
    *phaseParity = 1 - *phaseParity;
}

} // namespace kernels
} // namespace tensorrt_llm
