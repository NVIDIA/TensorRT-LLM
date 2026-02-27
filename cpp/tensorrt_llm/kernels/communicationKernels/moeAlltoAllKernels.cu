/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <nccl.h>
#include <nccl_device.h>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define SUPPORTS_PDL 1
#else
#define SUPPORTS_PDL 0
#endif

// Macros for concise launch-time specialization
#define SWITCH_BOOL(flag, NAME, ...)                                                                                   \
    if (flag)                                                                                                          \
    {                                                                                                                  \
        constexpr bool NAME = true;                                                                                    \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        constexpr bool NAME = false;                                                                                   \
        __VA_ARGS__                                                                                                    \
    }

#define SWITCH_TOP_K(top_k, TOP_K, ...)                                                                                \
    switch (top_k)                                                                                                     \
    {                                                                                                                  \
    case 22:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 22;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 16:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 16;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 10:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 10;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 8:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 8;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 6:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 6;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 4:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 4;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 2:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 2;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 1:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 1;                                                                                       \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported top_k");                                                              \
    }                                                                                                                  \
    }

#define SWITCH_DTYPE(dtype, TYPE, ...)                                                                                 \
    switch (dtype)                                                                                                     \
    {                                                                                                                  \
    case nvinfer1::DataType::kHALF:                                                                                    \
    {                                                                                                                  \
        using TYPE = half;                                                                                             \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case nvinfer1::DataType::kBF16:                                                                                    \
    {                                                                                                                  \
        using TYPE = __nv_bfloat16;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case nvinfer1::DataType::kFLOAT:                                                                                   \
    {                                                                                                                  \
        using TYPE = float;                                                                                            \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for moe_a2a_combine");                                          \
    }                                                                                                                  \
    }

#define SWITCH_POLICY(one_block_per_token, POLICY, ...)                                                                \
    if (one_block_per_token)                                                                                           \
    {                                                                                                                  \
        using POLICY = BlockPolicy;                                                                                    \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        using POLICY = WarpPolicy;                                                                                     \
        __VA_ARGS__                                                                                                    \
    }

// ============================================================================
// Helper Functions for Expert-to-Rank Mapping
// ============================================================================

__device__ int compute_target_rank_id(int expert_id, int num_experts_per_rank)
{
    // Compute which rank owns a given expert using contiguous partitioning
    // Experts are divided evenly across EP ranks:
    // - Rank 0 gets experts [0, num_experts_per_rank)
    // - Rank 1 gets experts [num_experts_per_rank, 2*num_experts_per_rank)
    // - etc.
    // Example: 32 experts, 4 ranks -> 8 experts per rank
    // - Rank 0: experts 0-7
    // - Rank 1: experts 8-15
    // - Rank 2: experts 16-23
    // - Rank 3: experts 24-31
    return expert_id / num_experts_per_rank;
}

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

struct WarpPolicy
{
    __device__ static int stride()
    {
        return warpSize;
    }

    __device__ static int offset()
    {
        return (threadIdx.x % warpSize);
    }

    __device__ static int token_idx()
    {
        return (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    }

    __device__ static void sync()
    {
        __syncwarp();
    }
};

struct BlockPolicy
{
    __device__ static int stride()
    {
        return blockDim.x;
    }

    __device__ static int offset()
    {
        return threadIdx.x;
    }

    __device__ static int token_idx()
    {
        return blockIdx.x;
    }

    __device__ static void sync()
    {
        __syncthreads();
    }
};

template <int VEC_SIZE, typename ThreadingPolicy>
__device__ void vectorized_copy_impl(void* dst, void const* src, int size)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;

    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < size; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> v;
        v.load(src_ptr + offset);
        v.store(dst_ptr + offset);
    }
}

template <typename ThreadingPolicy>
__device__ void vectorized_copy(void* dst, void const* src, int size)
{
    if (size % 16 == 0)
    {
        vectorized_copy_impl<16, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 8 == 0)
    {
        vectorized_copy_impl<8, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 4 == 0)
    {
        vectorized_copy_impl<4, ThreadingPolicy>(dst, src, size);
    }
    else if (size % 2 == 0)
    {
        vectorized_copy_impl<2, ThreadingPolicy>(dst, src, size);
    }
    else
    {
        vectorized_copy_impl<1, ThreadingPolicy>(dst, src, size);
    }
}

// Vectorized dispatch: load one vec from source and write to up to TOP_K destinations
template <int VEC_SIZE, int TOP_K, typename ThreadingPolicy>
__device__ void vectorized_dispatch_impl(uint8_t const* src_ptr, int bytes_per_token, int rank_id,
    int max_tokens_per_rank, int payload_idx, DispatchKernelPointers const& ptrs, int const* topk_target_ranks,
    int const* topk_send_indices)
{
    using flashinfer::vec_t;

    // Precompute destination base pointers per k
    uint8_t* dst_base_k[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k)
    {
        int dst_idx_k = topk_send_indices[k];
        int target_rank_k = topk_target_ranks[k];
        if (dst_idx_k < 0)
        {
            dst_base_k[k] = nullptr;
            continue;
        }
        uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank_k][payload_idx]);
        size_t base_source_rank
            = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank) + static_cast<size_t>(dst_idx_k);
        size_t base_token = base_source_rank * static_cast<size_t>(bytes_per_token);
        dst_base_k[k] = dst_data + base_token;
    }

    // TODO: process all payloads. index could be reused.
    int const stride = ThreadingPolicy::stride() * VEC_SIZE;
    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < bytes_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> v;
        v.load(src_ptr + offset);

#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            uint8_t* dst_base = dst_base_k[k];
            if (dst_base == nullptr)
            {
                continue;
            }
            v.store(dst_base + offset);
        }
    }
}

template <int TOP_K, typename ThreadingPolicy>
__device__ void vectorized_dispatch(uint8_t const* src_ptr, int bytes_per_token, int rank_id, int max_tokens_per_rank,
    int payload_idx, DispatchKernelPointers const& ptrs, int const* topk_target_ranks, int const* topk_send_indices)
{
    if (bytes_per_token % 16 == 0)
    {
        vectorized_dispatch_impl<16, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 8 == 0)
    {
        vectorized_dispatch_impl<8, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 4 == 0)
    {
        vectorized_dispatch_impl<4, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 2 == 0)
    {
        vectorized_dispatch_impl<2, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
    else
    {
        vectorized_dispatch_impl<1, TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
            payload_idx, ptrs, topk_target_ranks, topk_send_indices);
    }
}

// ============================================================================
// Dispatch Kernels
// ============================================================================

template <typename ThreadingPolicy, int TOP_K, bool ENABLE_EPLB>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
    ncclDevComm dev_comm,                                                   // NCCL device communicator
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts, int eplb_stats_num_experts)
{
    int thread_idx = ThreadingPolicy::offset();
    int local_token_idx = ThreadingPolicy::token_idx();

    if (local_num_tokens == 0)
    {
        // Special case: If local_num_tokens == 0,
        // we need to keep the threads where local_token_idx == 0 alive to participate in the synchronization.
        // Other threads should return.
        if (local_token_idx > 0)
            return;
#if SUPPORTS_PDL
        cudaGridDependencySynchronize();
#endif
    }
    else
    {
        // Threads that do not have a token to process should return.
        // TODO: Not needed for one-block-per-token policy. If one-warp-per-token is deprecated, remove this check.
        if (local_token_idx >= local_num_tokens)
            return;

        // Prepare per-policy shared-memory tiles for this token
        extern __shared__ int smem[];
        int* smem_topk_target_ranks;
        int* smem_topk_send_indices;
        int warps_per_block = blockDim.x / warpSize;
        if constexpr (std::is_same<ThreadingPolicy, WarpPolicy>::value)
        {
            int lane_id = threadIdx.x / warpSize;
            smem_topk_target_ranks = smem + lane_id * TOP_K;
            smem_topk_send_indices = smem + warps_per_block * TOP_K + lane_id * TOP_K;
        }
        else
        {
            smem_topk_target_ranks = smem;
            smem_topk_send_indices = smem + TOP_K;
        }

        uint64_t already_copied = 0;
        int num_experts_per_rank = num_experts / ep_size;
#if SUPPORTS_PDL
        cudaGridDependencySynchronize();
#endif
        for (int k = 0; k < TOP_K; k++)
        {
            int expert_id = token_selected_experts[local_token_idx * TOP_K + k];
            // Use contiguous partitioning to determine target rank
            int target_rank = compute_target_rank_id(expert_id, num_experts_per_rank);

            if (already_copied & (1ULL << target_rank))
            {
                if (thread_idx == 0)
                {
                    ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = -1;
                    ptrs.topk_send_indices[local_token_idx * TOP_K + k] = -1;
                    // Mirror to shared memory immediately
                    smem_topk_target_ranks[k] = -1;
                    smem_topk_send_indices[k] = -1;
                }
                continue;
            }

            // Only one thread per warp should increment the counter
            int dst_token_idx;
            if (thread_idx == 0)
            {
                dst_token_idx = atomicAdd(&ptrs.send_counters[target_rank], 1);

                ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank;
                ptrs.topk_send_indices[local_token_idx * TOP_K + k] = dst_token_idx;
                // Mirror to shared memory immediately
                smem_topk_target_ranks[k] = target_rank;
                smem_topk_send_indices[k] = dst_token_idx;
            }
            already_copied |= 1ULL << target_rank;
        }
        // Sync before dispatching data
        ThreadingPolicy::sync();

        // Read staged routing once into registers per thread
        int topk_target_ranks[TOP_K];
        int topk_send_indices[TOP_K];
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            topk_target_ranks[k] = smem_topk_target_ranks[k];
            topk_send_indices[k] = smem_topk_send_indices[k];
        }

        // Perform a single source load and TOP_K fanout per payload
        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
            int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

            vectorized_dispatch<TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank,
                payload_idx, ptrs, topk_target_ranks, topk_send_indices);
        }

        ThreadingPolicy::sync();
    }
#if SUPPORTS_PDL
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    bool is_first_warp = threadIdx.x / warpSize == 0;
    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;

        bool is_last_token = false;
        if (lane_id == 0)
        {
            if (local_num_tokens != 0)
            {
                int cnt = atomicAdd(ptrs.local_token_counter, 1);
                is_last_token = cnt + 1 == local_num_tokens;
            }
            else
            {
                is_last_token = true;
            }
        }
        is_last_token = __shfl_sync(0xffffffff, is_last_token, 0);

        if (is_last_token)
        {
            // Reset local_token_counter to 0 for using in the combine kernel.
            if (lane_id == 0)
            {
                *ptrs.local_token_counter = 0;
            }

// Store send_counters to recv_counters.
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                int send_count = ptrs.send_counters[target_rank];
                ptrs.recv_counters[target_rank][rank_id] = send_count;
            }

            if constexpr (ENABLE_EPLB)
            {
                // Write local stats into peer buffers before the release fence below.
#pragma unroll 1
                for (int target_rank = 0; target_rank < ep_size; ++target_rank)
                {
                    int* target_stats = ptrs.eplb_gathered_stats[target_rank];
                    for (int expert_id = lane_id; expert_id < eplb_stats_num_experts; expert_id += warpSize)
                    {
                        int stat_val = ptrs.eplb_local_stats[expert_id];
                        target_stats[rank_id * eplb_stats_num_experts + expert_id] = stat_val;
                    }
                }
            }

#if !DISABLE_SYNC_FOR_PROFILING
// Issue an release barrier to ensure that the data is visible to other ranks after the synchronization.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
            asm volatile("fence.release.sys;");
#else
            asm volatile("fence.acq_rel.sys;");
#endif

            // Use NCCL LSA barrier for inter-rank synchronization.
            // Barrier index 0 is reserved for dispatch.
            // Only the last-token warp (one warp across all CTAs) participates as a coop.
            ncclCoopWarp coop = ncclCoopWarp();
            ncclLsaBarrierSession barrier(coop, dev_comm, ncclTeamTagLsa{}, /*index=*/0, /*multimem=*/true);
            barrier.sync(coop, cuda::memory_order_relaxed);
#endif
        }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);

    // Prepare kernel pointers struct
    DispatchKernelPointers kernel_ptrs = {};

    // Fill source data pointers and payload sizes
    for (int i = 0; i < params.num_payloads; i++)
    {
        kernel_ptrs.src_data_ptrs[i] = params.payloads[i].src_data;
        kernel_ptrs.payload_bytes_per_token[i]
            = params.payloads[i].element_size * params.payloads[i].elements_per_token;
    }

    // Fill receive buffer pointers
    for (int target_rank = 0; target_rank < params.ep_size; target_rank++)
    {
        kernel_ptrs.recv_counters[target_rank] = params.recv_counters[target_rank];
        kernel_ptrs.eplb_gathered_stats[target_rank] = params.eplb_gathered_stats[target_rank];
        for (int payload = 0; payload < params.num_payloads; payload++)
        {
            kernel_ptrs.recv_buffers[target_rank][payload] = params.recv_buffers[target_rank][payload];
        }
    }

    // Copy communication tracking pointers
    kernel_ptrs.send_counters = params.send_counters;
    kernel_ptrs.local_token_counter = params.local_token_counter;
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;
    kernel_ptrs.eplb_local_stats = params.eplb_local_stats;

    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();
    constexpr int kWarpSize = 32;
    int const kWarpsPerBlock = kBlockSize / kWarpSize;

    // Configure kernel launch
    if (params.one_block_per_token)
    {
        int grid_size = params.local_num_tokens;
        // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
        if (grid_size == 0)
        {
            grid_size = 1;
        }
        int shared_bytes = 2 * params.top_k * (int) sizeof(int);
        SWITCH_BOOL(params.enable_eplb, EPLB_STATS, SWITCH_TOP_K(params.top_k, TOP_K, {
            auto kernel_fn = moeA2ADispatchKernel<BlockPolicy, TOP_K, EPLB_STATS>;
            launchWithPdlWhenEnabled("moeA2ADispatchKernel", kernel_fn, grid_size, kBlockSize, shared_bytes,
                params.stream, params.token_selected_experts, kernel_ptrs, *params.dev_comm, params.num_payloads,
                params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts,
                params.eplb_stats_num_experts);
        }))
    }
    else
    {
        int grid_size = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
        // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
        if (grid_size == 0)
        {
            grid_size = 1;
        }
        int shared_bytes = 2 * kWarpsPerBlock * params.top_k * (int) sizeof(int);
        SWITCH_BOOL(params.enable_eplb, EPLB_STATS, SWITCH_TOP_K(params.top_k, TOP_K, {
            auto kernel_fn = moeA2ADispatchKernel<WarpPolicy, TOP_K, EPLB_STATS>;
            launchWithPdlWhenEnabled("moeA2ADispatchKernel", kernel_fn, grid_size, kBlockSize, shared_bytes,
                params.stream, params.token_selected_experts, kernel_ptrs, *params.dev_comm, params.num_payloads,
                params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts,
                params.eplb_stats_num_experts);
        }))
    }
}

// ============================================================================
// Combine kernels
// ============================================================================

// Accumulate across all valid ranks into registers, then store once per segment
template <int VEC_SIZE, int TOP_K, typename ThreadingPolicy, typename T>
__device__ void vectorized_combine_impl(
    T* dst_typed_base, int size_per_token, int rank_id, int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
    using flashinfer::vec_t;

    uint8_t* dst_bytes = reinterpret_cast<uint8_t*>(dst_typed_base);

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;
    int const local_token_idx = ThreadingPolicy::token_idx();

    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> acc[TOP_K];

// Unrolled K accumulation using compact top-k lists
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_idx < 0)
            {
                acc[k].fill(0);
                continue;
            }

            uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
            size_t base_source_rank = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank)
                + static_cast<size_t>(dst_idx);
            size_t base_token = base_source_rank * static_cast<size_t>(size_per_token);

            // Load directly into the per-k accumulator; reduce across k below
            acc[k].load(recv_buffer + base_token + offset);
        }
        // Reduce acc[TOP_K] into acc[0]
        if constexpr (TOP_K == 22)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
            T* a10 = reinterpret_cast<T*>(&acc[10]);
            T* a11 = reinterpret_cast<T*>(&acc[11]);
            T* a12 = reinterpret_cast<T*>(&acc[12]);
            T* a13 = reinterpret_cast<T*>(&acc[13]);
            T* a14 = reinterpret_cast<T*>(&acc[14]);
            T* a15 = reinterpret_cast<T*>(&acc[15]);
            T* a16 = reinterpret_cast<T*>(&acc[16]);
            T* a17 = reinterpret_cast<T*>(&acc[17]);
            T* a18 = reinterpret_cast<T*>(&acc[18]);
            T* a19 = reinterpret_cast<T*>(&acc[19]);
            T* a20 = reinterpret_cast<T*>(&acc[20]);
            T* a21 = reinterpret_cast<T*>(&acc[21]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
                a10[j] += a11[j];
                a12[j] += a13[j];
                a14[j] += a15[j];
                a16[j] += a17[j];
                a18[j] += a19[j];
                a20[j] += a21[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
                a8[j] += a10[j];
                a12[j] += a14[j];
                a16[j] += a18[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a8[j] += a12[j];
                a16[j] += a20[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a8[j];
                a0[j] += a16[j];
            }
        }
        else if constexpr (TOP_K == 16)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
            T* a10 = reinterpret_cast<T*>(&acc[10]);
            T* a11 = reinterpret_cast<T*>(&acc[11]);
            T* a12 = reinterpret_cast<T*>(&acc[12]);
            T* a13 = reinterpret_cast<T*>(&acc[13]);
            T* a14 = reinterpret_cast<T*>(&acc[14]);
            T* a15 = reinterpret_cast<T*>(&acc[15]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
                a10[j] += a11[j];
                a12[j] += a13[j];
                a14[j] += a15[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
                a8[j] += a10[j];
                a12[j] += a14[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a8[j] += a12[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a8[j];
            }
        }
        else if constexpr (TOP_K == 10)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
            T* a8 = reinterpret_cast<T*>(&acc[8]);
            T* a9 = reinterpret_cast<T*>(&acc[9]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
                a8[j] += a9[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
                a0[j] += a8[j];
            }
        }
        else if constexpr (TOP_K == 8)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
            T* a6 = reinterpret_cast<T*>(&acc[6]);
            T* a7 = reinterpret_cast<T*>(&acc[7]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
                a6[j] += a7[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a4[j] += a6[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a4[j];
            }
        }
        else if constexpr (TOP_K == 6)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
            T* a4 = reinterpret_cast<T*>(&acc[4]);
            T* a5 = reinterpret_cast<T*>(&acc[5]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
                a4[j] += a5[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
                a0[j] += a4[j];
            }
        }
        else if constexpr (TOP_K == 4)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
            T* a2 = reinterpret_cast<T*>(&acc[2]);
            T* a3 = reinterpret_cast<T*>(&acc[3]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
                a2[j] += a3[j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a2[j];
            }
        }
        else if constexpr (TOP_K == 2)
        {
            T* a0 = reinterpret_cast<T*>(&acc[0]);
            T* a1 = reinterpret_cast<T*>(&acc[1]);
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                a0[j] += a1[j];
            }
        }
        else if constexpr (TOP_K == 1)
        {
            // nothing to do
        }
        else
        {
            // Generic fallback: accumulate all into acc[0]
            T* a0 = reinterpret_cast<T*>(&acc[0]);
#pragma unroll
            for (int k = 1; k < TOP_K; ++k)
            {
                T* ak = reinterpret_cast<T*>(&acc[k]);
#pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    a0[j] += ak[j];
                }
            }
        }

        acc[0].store(dst_bytes + offset);
    }
}

// Wrapper that selects vector width based on size_per_token alignment
template <int TOP_K, typename ThreadingPolicy, typename T>
__device__ void vectorized_combine(
    T* dst_typed_base, int size_per_token, int rank_id, int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    if (size_per_token % 16 == 0)
    {
        vectorized_combine_impl<16, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_impl<8, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_impl<4, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_impl<2, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else
    {
        vectorized_combine_impl<1, TOP_K, ThreadingPolicy, T>(
            dst_typed_base, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
}

// Copy payload to recv buffer using vectorized copy; supports warp/block token mapping
template <typename ThreadingPolicy>
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, uint8_t const* payload_bytes,
    int bytes_per_token, int ep_size, int max_tokens_per_rank, int const* recv_counters)
{
#if SUPPORTS_PDL
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    int global_token_idx = ThreadingPolicy::token_idx();

    int global_token_num = ep_size * max_tokens_per_rank;
    if (global_token_idx >= global_token_num)
        return;

    // Map global_token_idx to (rank_idx, local_token_idx)
    int rank_idx = global_token_idx / max_tokens_per_rank;
    int local_token_idx = global_token_idx % max_tokens_per_rank;

    // Skip invalid tokens beyond per-rank recv count
    if (local_token_idx >= recv_counters[rank_idx])
        return;

    // Calculate source and destination pointers for this token
    size_t offset = static_cast<size_t>(global_token_idx) * bytes_per_token;
    uint8_t* dst_ptr = recv_buffer_bytes + offset;
    uint8_t const* src_ptr = payload_bytes + offset;

    // Copy one token's data using vectorized copy with policy
    vectorized_copy<ThreadingPolicy>(dst_ptr, src_ptr, bytes_per_token);
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, typename ThreadingPolicy, int TOP_K>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs, // Combine-specific struct, src_data_ptrs[0] is output
    ncclDevComm dev_comm,             // NCCL device communicator
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size)
{
#if SUPPORTS_PDL
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

#if !DISABLE_SYNC_FOR_PROFILING

    // The first warp in each CTA performs the synchronization.
    {
        bool is_first_warp = threadIdx.x / warpSize == 0;

        // local_token_counter is reused as a 3-state flag:
        //   0 → no CTA elected yet
        //  -1 → one CTA elected, barrier in progress
        //   1 → barrier done, peer data is visible
        // The first CTA to arrive wins the election (CAS 0→-1) and performs the
        // NCCL inter-rank barrier (index 1 for combine). All other CTAs spin until the flag becomes 1.
        if (is_first_warp)
        {
            int lane_id = threadIdx.x % warpSize;
            bool elected = false;
            if (lane_id == 0)
            {
                int old = atomicCAS(ptrs.local_token_counter, 0, -1);
                elected = (old == 0);
            }
            elected = __shfl_sync(0xffffffff, elected, 0);

            if (elected)
            {
                ncclCoopWarp coop;
                ncclLsaBarrierSession barrier(coop, dev_comm, ncclTeamTagLsa{}, /*index=*/1);
                barrier.sync(coop, cuda::memory_order_relaxed);

                // Signal all other blocks that the inter-rank barrier is done.
                if (lane_id == 0)
                {
                    asm volatile("st.relaxed.gpu.s32 [%0], %1;" ::"l"(ptrs.local_token_counter), "r"(1));
                }

                // Zero send_counters for the next dispatch round.
                for (int i = lane_id; i < ep_size; i += warpSize)
                {
                    ptrs.send_counters[i] = 0;
                }
            }
            else
            {
                // Wait for the elected CTA to complete the inter-rank barrier.
                if (lane_id == 0)
                {
                    int local_token_counter_val;
                    do
                    {
                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];"
                                     : "=r"(local_token_counter_val)
                                     : "l"(ptrs.local_token_counter));
                    } while (local_token_counter_val != 1);
                }
            }

// Acquire system-scope visibility so this block sees peer data.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("fence.acquire.sys;");
#else
            asm volatile("fence.acq_rel.sys;");
#endif
        }
        // Synchronize the rest warps in the CTA with the first warp.
        __syncthreads();
    }
#endif

    int local_token_idx = ThreadingPolicy::token_idx();
    int const size_per_token = elements_per_token * sizeof(T);

    // Threads that do not have a token to process should return.
    // TODO: Not needed for one-block-per-token policy. If one-warp-per-token is deprecated, remove this check.
    if (local_token_idx >= local_num_tokens)
        return;

    // Get output location for this token (using src_data_ptrs[0] as output)
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;

    // Accumulate across ranks in registers, then store once per segment
    vectorized_combine<TOP_K, ThreadingPolicy, T>(token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs);
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    // If payload is already in workspace, nothing to copy.
    if (params.prepare_payload == nullptr)
    {
        return;
    }

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32; // 8 warps per block

    // Calculate bytes per token based on dtype
    int element_size;
    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF: element_size = sizeof(half); break;
    case nvinfer1::DataType::kBF16: element_size = sizeof(__nv_bfloat16); break;
    case nvinfer1::DataType::kFLOAT: element_size = sizeof(float); break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for combine prepare"); return;
    }

    int bytes_per_token = params.elements_per_token * element_size;
    int global_token_num = params.ep_size * params.max_tokens_per_rank;
    int grid_size_warp = ceilDiv(global_token_num, kWarpsPerBlock);
    int grid_size_block = global_token_num; // one block per token
    int grid = params.one_block_per_token ? grid_size_block : grid_size_warp;

    uint8_t* recv_buffer_bytes = static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank]));
    uint8_t const* payload_bytes = static_cast<uint8_t const*>(params.prepare_payload);

    auto kernel_fn
        = params.one_block_per_token ? moeA2APrepareCombineKernel<BlockPolicy> : moeA2APrepareCombineKernel<WarpPolicy>;
    launchWithPdlWhenEnabled("moeA2APrepareCombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
        recv_buffer_bytes, payload_bytes, bytes_per_token, params.ep_size, params.max_tokens_per_rank,
        params.recv_counters);
}

// ============================================================================
// Combine Launch Function
// ============================================================================

void moe_a2a_combine_launch(MoeA2ACombineParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.elements_per_token > 0);

    // Configure kernel launch
    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
    int const kWarpsPerBlock = kBlockSize / 32; // warpSize

    int grid_size;
    if (params.one_block_per_token)
    {
        grid_size = params.local_num_tokens;
    }
    else
    {
        grid_size = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
    }
    // Even if local_num_tokens is 0, we need at least 1 block for the barrier.
    if (grid_size == 0)
    {
        grid_size = 1;
    }

    // Prepare kernel pointers struct for combine
    CombineKernelPointers kernel_ptrs = {}; // Zero-initialize

    // Set output data pointer in src_data_ptrs[0]
    kernel_ptrs.src_data_ptrs[0] = params.output_data;

    // Fill recv buffer pointers
    for (int rank = 0; rank < params.ep_size; rank++)
    {
        kernel_ptrs.recv_buffers[rank][0] = params.recv_buffers[rank];
    }

    // Intra-rank barrier flag and send_counters for cleanup
    kernel_ptrs.local_token_counter = params.local_token_counter;
    kernel_ptrs.send_counters = params.send_counters;

    // Copy communication tracking pointers
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;

    // Launch appropriate kernel with compact macros
    SWITCH_DTYPE(params.dtype, TKernelType, {
        SWITCH_POLICY(params.one_block_per_token, Policy, {
            SWITCH_TOP_K(params.top_k, TOP_K, {
                auto kernel_fn = moeA2ACombineKernel<TKernelType, Policy, TOP_K>;
                launchWithPdlWhenEnabled("moeA2ACombineKernel", kernel_fn, grid_size, kBlockSize, 0, params.stream,
                    kernel_ptrs, *params.dev_comm, params.max_tokens_per_rank, params.elements_per_token,
                    params.local_num_tokens, params.ep_rank, params.ep_size);
            });
        });
    });
}

// Kernel to sanitize expert ids for invalid tokens
__global__ void moeA2ASanitizeExpertIdsKernel(int32_t* expert_ids_ptr, int32_t const* recv_counters_ptr, int ep_size,
    int max_tokens_per_rank, int top_k, int32_t invalid_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = ep_size * max_tokens_per_rank;
    if (tid >= total_tokens)
        return;

    int source_rank = tid / max_tokens_per_rank;
    int token_idx = tid % max_tokens_per_rank;

#if SUPPORTS_PDL
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (token_idx >= recv_counters_ptr[source_rank])
    {
        int32_t* token_expert_ids = expert_ids_ptr + tid * top_k;
        for (int k = 0; k < top_k; ++k)
        {
            token_expert_ids[k] = invalid_id;
        }
    }
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, int32_t invalid_id,
    int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    int total_tokens = ep_size * max_tokens_per_rank;
    int grid = ceilDiv(total_tokens, kBlockSize);
    launchWithPdlWhenEnabled("moeA2ASanitizeExpertIdsKernel", moeA2ASanitizeExpertIdsKernel, grid, kBlockSize, 0,
        stream, expert_ids, recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id);
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
