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
#pragma once

// Shared-state helpers for the owned fast variants of the in-node one-shot Lamport kernels
// (allreduce_fusion_kernel_oneshot_lamport_fast in allReduceFusionKernels.cu,
// moefinalize_allreduce_fusion_kernel_oneshot_lamport_fast in moeAllReduceFusionKernels.cu). They use the workspace of
// allocate_allreduce_fusion_workspace exactly as the default kernels do: the same buffers, flag rotation, counter and
// sizes to clear, so either kind of kernel may follow the other on one workspace.

#include "tensorrt_llm/common/config.h"

#include <cooperative_groups.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::ipc
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

// Brings the line of ptr into L2 without reading it. Safe before a grid dependency wait for any address: no value
// reaches a register, and a later write of the line (L2 is the point of coherence) updates the cached copy.
__device__ __forceinline__ void prefetchL2(void const* ptr)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

// The Lamport state of the one-shot kernels, read in two steps. The workspace pointers and the Lamport buffer size are
// written by the host when the workspace is created and by no kernel, so they are read before the grid dependency
// wait, and thread 0 brings the lines of the flag and of the size to clear into L2. Those two are written by the
// previous one-shot kernel on the workspace and are read after the wait (loadFlags). arrive() then derives the buffers
// and counts this CTA's arrival, as LamportComm's constructor does.
template <int NRanks>
struct LamportCommSplit
{
    __device__ __forceinline__ LamportCommSplit(void* const* workspace, int rank, int32_t* flags, int64_t* layout)
    {
        counter_ptr = &flags[0];
        flag_ptr = &flags[2];
        clear_ptr = &layout[0];
        if (threadIdx.x == 0)
        {
            prefetchL2(flag_ptr);
            prefetchL2(clear_ptr);
        }
        comm_size = layout[1];
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            bases[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]);
        }
        local_base = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]);
    }

    __device__ __forceinline__ void loadFlags()
    {
        flag_value = *flag_ptr;
        clear_size = *clear_ptr;
    }

    __device__ __forceinline__ void arrive()
    {
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            data_bufs[r] = bases[r] + data_offset * comm_size;
        }
        local_data = local_base + data_offset * comm_size;
        clear_buf = local_base + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ int loadArrivals() const
    {
        return *reinterpret_cast<int volatile*>(counter_ptr);
    }

    // The rotation of LamportComm::update, by CTA 0's thread 0: once the count of arrived CTAs (arrivals, the last
    // value read) covers the grid, the next flag, the size the next kernel clears and a reset counter.
    __device__ __forceinline__ void rotate(int arrivals, int64_t new_clear_size)
    {
        while (arrivals != static_cast<int>(gridDim.x))
        {
            arrivals = loadArrivals();
        }
        *flag_ptr = (flag_value + 1) % 3;
        *clear_ptr = new_clear_size;
        *counter_ptr = 0;
    }

    int* counter_ptr;
    int* flag_ptr;
    int64_t* clear_ptr;
    int64_t comm_size;
    uint8_t* bases[NRanks];
    uint8_t* local_base;
    uint8_t* data_bufs[NRanks];
    // data_bufs[rank], without a runtime index into the array.
    uint8_t* local_data;
    uint8_t* clear_buf;
    int64_t clear_size;
    int flag_value;
};

int constexpr kNormExchangeMaxBlocks = 8;

// The RMSNorm partials of a cluster that covers one row: one float per CTA, and the transaction barrier that completes
// once every CTA's partial has landed.
struct NormExchangeStorage
{
    float partials[kNormExchangeMaxBlocks];
    uint64_t barrier;
};

__device__ __forceinline__ NormExchangeStorage& normExchangeStorage()
{
    __shared__ NormExchangeStorage storage;
    return storage;
}

// Before the grid dependency wait, by every thread of a cluster of 2 to kNormExchangeMaxBlocks CTAs: thread 0 sets up
// this CTA's barrier for one arrival (its own) and one partial from every CTA, then every thread arrives on the cluster
// barrier. That phase publishes the barrier before any CTA sends (exchangeNormPartials waits for it).
__device__ __forceinline__ void initNormExchange()
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (threadIdx.x == 0)
    {
        uint32_t const bytes = static_cast<uint32_t>(cooperative_groups::this_cluster().num_blocks() * sizeof(float));
        uint32_t const barrier = static_cast<uint32_t>(__cvta_generic_to_shared(&normExchangeStorage().barrier));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
#endif
}

// Called by every thread after initNormExchange; blockSum is this CTA's partial (thread 0's value is used). Thread 0
// stores it into slot [block rank] of every CTA of the cluster with st.async, which completes that CTA's barrier, then
// waits on its own barrier and returns the partials added in block-rank order starting from zero: the value the
// cluster.sync() loop of the default epilogue computes. Other threads return 0.
__device__ __forceinline__ float exchangeNormPartials(float blockSum)
{
    float sum = 0.F;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("barrier.cluster.wait;" ::: "memory");
    if (threadIdx.x == 0)
    {
        cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
        int const numBlocks = static_cast<int>(cluster.num_blocks());
        uint32_t const blockRank = cluster.block_rank();
        auto& storage = normExchangeStorage();
        uint32_t const slot = static_cast<uint32_t>(__cvta_generic_to_shared(&storage.partials[blockRank]));
        uint32_t const barrier = static_cast<uint32_t>(__cvta_generic_to_shared(&storage.barrier));
        for (int block = 0; block < numBlocks; ++block)
        {
            uint32_t peerSlot;
            uint32_t peerBarrier;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peerSlot) : "r"(slot), "r"(block));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peerBarrier) : "r"(barrier), "r"(block));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(peerSlot),
                         "r"(__float_as_uint(blockSum)), "r"(peerBarrier)
                         : "memory");
        }
        uint32_t complete = 0;
        while (!complete)
        {
            asm volatile(
                "{\n"
                ".reg .pred done;\n"
                "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%1], 0;\n"
                "selp.u32 %0, 1, 0, done;\n"
                "}\n"
                : "=r"(complete)
                : "r"(barrier)
                : "memory");
        }
        for (int block = 0; block < numBlocks; ++block)
        {
            sum += storage.partials[block];
        }
    }
#endif
    return sum;
}

#endif
} // namespace kernels::glm_moe_dsa::ipc

TRTLLM_NAMESPACE_END
