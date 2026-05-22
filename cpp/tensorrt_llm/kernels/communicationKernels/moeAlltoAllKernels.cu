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
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
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
    case 18:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 18;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 16:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 16;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 14:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 14;                                                                                      \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case 12:                                                                                                           \
    {                                                                                                                  \
        constexpr int TOP_K = 12;                                                                                      \
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
    case nvinfer1::DataType::kFP8:                                                                                     \
    {                                                                                                                  \
        using TYPE = __nv_fp8_e4m3;                                                                                    \
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

#if DISABLE_TIMEOUT
#define check_timeout(s) false
#else
// 300 * 2000 MHz - should be high enough on any GPU but will prevent a hang
#define check_timeout(s) ((clock64() - (s)) > (300ll * 2000ll * 1000ll * 1000ll))
#endif

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

__global__ void moeA2APrepareDispatchKernel(
    int* send_counters, int* local_token_counter, int ep_size, uint32_t* flag_val_ptr)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Zero send_counters
    if (idx < ep_size)
    {
        send_counters[idx] = 0;
    }
    // Zero local_token_counter and increment flag_val
    if (idx == 0)
    {
        *local_token_counter = 0;
        // Increment flag_val for this dispatch round
        *flag_val_ptr = *flag_val_ptr + 1;
    }
}

// ============================================================================
// Dispatch Kernels
// ============================================================================

template <typename ThreadingPolicy, int TOP_K, bool ENABLE_EPLB>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif
    }
    else
    {
        // Threads that do not have a token to process should return.
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
// Store send_counters to recv_counters
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
            uint32_t expected_value = *ptrs.flag_val;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
            asm volatile("fence.release.sys;");
#else
            asm volatile("fence.acq_rel.sys;");
#endif
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));

#if ENABLE_DEBUG_PRINT
                printf("dispatch: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    target_rank);
#endif
            }

#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                bool flag_set = false;
                auto s = clock64();
                do
                {
                    uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                    uint32_t flag_value;
                    // Acquire load to ensure visibility of peer's release-store
                    asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
                    printf(
                        "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, expected_value: "
                        "%d, address: %p\n",
                        rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
                    flag_set = flag_value == expected_value;
                } while (!flag_set && !check_timeout(s));

                if (__builtin_expect(!flag_set, 0))
                {
                    printf("dispatch: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id,
                        peer_rank);
                    asm volatile("trap;");
                    return;
                }
            }
#endif
        }
    }
}

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params)
{
    launchWithPdlWhenEnabled("moeA2APrepareDispatchKernel", moeA2APrepareDispatchKernel, 1, params.ep_size, 0,
        params.stream, params.send_counters, params.local_token_counter, params.ep_size, params.flag_val);
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

    // Copy completion flag pointers
    for (int i = 0; i < params.ep_size; i++)
    {
        kernel_ptrs.completion_flags[i] = params.completion_flags[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

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
                params.stream, params.token_selected_experts, kernel_ptrs, params.num_payloads,
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
                params.stream, params.token_selected_experts, kernel_ptrs, params.num_payloads,
                params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts,
                params.eplb_stats_num_experts);
        }))
    }
}

// ============================================================================
// Combine kernels
// ============================================================================

// Accumulate across all valid ranks into float32 registers, then store as T.
// InT: input element type in recv buffer (defaults to T for same-type accumulation).
// T:   output element type written to dst.
//
// Unified path: load VEC_SIZE bytes, reinterpret as InT[elems_per_vec], accumulate as float32,
// store as T.  Works for same-type (InT==T: half/bf16/float) and cross-type
// (e.g. InT=fp8_e4m3, T=bf16).  sizeof(InT) must divide VEC_SIZE.
template <int VEC_SIZE, int TOP_K, typename ThreadingPolicy, typename T, typename InT = T>
__device__ void vectorized_combine_impl(T* dst_typed_base, int size_per_token, int stride_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    using flashinfer::vec_t;

    // elems_per_vec: number of InT elements per VEC_SIZE-byte load (constexpr).
    constexpr int elems_per_vec = VEC_SIZE / static_cast<int>(sizeof(InT));

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;
    int const local_token_idx = ThreadingPolicy::token_idx();

    // offset is a byte offset into the recv buffer, stepping by VEC_SIZE bytes.
    for (int offset = ThreadingPolicy::offset() * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        // Per-k vec_t<float, elems_per_vec> accumulators, zero-initialised via fill().
        // Using vec_t enables cast_store() for the output, emitting a vectorized int4 write.
        vec_t<float, elems_per_vec> acc[TOP_K];

        // Pass 1: issue all TOP_K loads back-to-back without any type conversion.
        // Raw InT bytes are loaded directly into acc[k]'s register storage, reinterpreted as
        // vec_t<InT, elems_per_vec> (VEC_SIZE bytes, fitting in the low end of acc[k]'s
        // sizeof(float)*elems_per_vec allocation).  Separating load from cast lets the compiler
        // schedule all VEC_SIZE-byte global loads consecutively, hiding memory latency across k.
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_idx < 0)
            {
                acc[k].fill(0.0f);
                continue;
            }

            uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
            size_t base_source_rank = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank)
                + static_cast<size_t>(dst_idx);
            // stride_per_token: byte distance between tokens in the recv buffer.
            // Equals size_per_token for normal cases; may differ for FP8 in-place
            // (BF16-stride workspace but FP8-sized payload).
            size_t base_token = base_source_rank * static_cast<size_t>(stride_per_token);

            reinterpret_cast<vec_t<InT, elems_per_vec>&>(acc[k]).load(
                reinterpret_cast<InT const*>(recv_buffer + base_token + offset));
        }

        // Pass 2: in-place cast InT → float, iterating j in descending order.
        // float[j] occupies bytes [j*4, j*4+3]; InT[j] occupies [j*sizeof(InT), ...).
        // For sizeof(InT) < sizeof(float), high-j float writes land above all remaining
        // InT bytes, so descending order is always write-after-read safe.
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            if (ptrs.topk_send_indices[local_token_idx * TOP_K + k] < 0)
                continue; // acc[k] already holds 0.0f from fill() above
#pragma unroll
            for (int j = elems_per_vec - 1; j >= 0; --j)
                acc[k][j] = static_cast<float>(reinterpret_cast<InT const*>(&acc[k])[j]);
        }
        // Reduce acc[TOP_K] into acc[0] via unrolled tree-reduction.
        // acc[k][j] uses vec_t::operator[] which returns float& — no indirection overhead.
        if constexpr (TOP_K == 22)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
                acc[4][j] += acc[5][j];
                acc[6][j] += acc[7][j];
                acc[8][j] += acc[9][j];
                acc[10][j] += acc[11][j];
                acc[12][j] += acc[13][j];
                acc[14][j] += acc[15][j];
                acc[16][j] += acc[17][j];
                acc[18][j] += acc[19][j];
                acc[20][j] += acc[21][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
                acc[4][j] += acc[6][j];
                acc[8][j] += acc[10][j];
                acc[12][j] += acc[14][j];
                acc[16][j] += acc[18][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[4][j];
                acc[8][j] += acc[12][j];
                acc[16][j] += acc[20][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[8][j];
                acc[0][j] += acc[16][j];
            }
        }
        else if constexpr (TOP_K == 16)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
                acc[4][j] += acc[5][j];
                acc[6][j] += acc[7][j];
                acc[8][j] += acc[9][j];
                acc[10][j] += acc[11][j];
                acc[12][j] += acc[13][j];
                acc[14][j] += acc[15][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
                acc[4][j] += acc[6][j];
                acc[8][j] += acc[10][j];
                acc[12][j] += acc[14][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[4][j];
                acc[8][j] += acc[12][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[8][j];
            }
        }
        else if constexpr (TOP_K == 10)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
                acc[4][j] += acc[5][j];
                acc[6][j] += acc[7][j];
                acc[8][j] += acc[9][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
                acc[4][j] += acc[6][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[4][j];
                acc[0][j] += acc[8][j];
            }
        }
        else if constexpr (TOP_K == 8)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
                acc[4][j] += acc[5][j];
                acc[6][j] += acc[7][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
                acc[4][j] += acc[6][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[4][j];
            }
        }
        else if constexpr (TOP_K == 6)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
                acc[4][j] += acc[5][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
                acc[0][j] += acc[4][j];
            }
        }
        else if constexpr (TOP_K == 4)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
                acc[2][j] += acc[3][j];
            }
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[2][j];
            }
        }
        else if constexpr (TOP_K == 2)
        {
#pragma unroll
            for (int j = 0; j < elems_per_vec; ++j)
            {
                acc[0][j] += acc[1][j];
            }
        }
        else if constexpr (TOP_K == 1)
        {
            // nothing to do
        }
        else
        {
            // Generic fallback: accumulate all into acc[0]
#pragma unroll
            for (int k = 1; k < TOP_K; ++k)
            {
#pragma unroll
                for (int j = 0; j < elems_per_vec; ++j)
                {
                    acc[0][j] += acc[k][j];
                }
            }
        }

        // cast_store: converts float→T element-by-element then writes via vectorized int4 store.
        acc[0].cast_store(dst_typed_base + offset / static_cast<int>(sizeof(InT)));
    }
}

// Wrapper that selects vector width based on size_per_token alignment.
// stride_per_token: byte distance between tokens in the recv buffer (may differ from
// size_per_token when FP8 in-place uses BF16-stride workspace with FP8-sized payload).
// InT: input element type in recv buffer (defaults to T for same-type accumulation)
template <int TOP_K, typename ThreadingPolicy, typename T, typename InT = T>
__device__ void vectorized_combine(T* dst_typed_base, int size_per_token, int stride_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    // Each branch is guarded by if constexpr (sizeof(InT) <= VEC_SIZE) so that the compiler
    // never instantiates vectorized_combine_impl with elems_per_vec=0.
    // Branches where VEC_SIZE < sizeof(InT) are unreachable at runtime because size_per_token
    // is always a multiple of sizeof(InT), so a larger alignment branch is taken first.
    if (size_per_token % 16 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InT)) <= 16)
            vectorized_combine_impl<16, TOP_K, ThreadingPolicy, T, InT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 8 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InT)) <= 8)
            vectorized_combine_impl<8, TOP_K, ThreadingPolicy, T, InT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 4 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InT)) <= 4)
            vectorized_combine_impl<4, TOP_K, ThreadingPolicy, T, InT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 2 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InT)) <= 2)
            vectorized_combine_impl<2, TOP_K, ThreadingPolicy, T, InT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else
    {
        if constexpr (static_cast<int>(sizeof(InT)) <= 1)
            vectorized_combine_impl<1, TOP_K, ThreadingPolicy, T, InT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
}

// ---- vec_convert: per-vector type conversion, specialized by PTX where available ----
// Generic: SrcT → float → DstT (all architectures, all type combinations).
template <size_t VEC_SIZE, typename SrcT, typename DstT>
__device__ __forceinline__ void vec_convert(
    flashinfer::vec_t<DstT, VEC_SIZE>& out, flashinfer::vec_t<SrcT, VEC_SIZE> const& in)
{
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j)
        out[j] = DstT(static_cast<float>(in[j]));
}

// BF16 → FP8 e4m3: paired PTX cvt.rn.satfinite.e4m3x2.bf16x2 (SM100+, Blackwell).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
template <size_t VEC_SIZE, std::enable_if_t<(VEC_SIZE % 2 == 0), int> = 0>
__device__ __forceinline__ void vec_convert(
    flashinfer::vec_t<__nv_fp8_e4m3, VEC_SIZE>& out, flashinfer::vec_t<__nv_bfloat16, VEC_SIZE> const& in)
{
    uint32_t const* src_u32 = reinterpret_cast<uint32_t const*>(&in);
    uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
    for (int p = 0; p < VEC_SIZE / 2; ++p)
    {
        uint16_t d;
        asm volatile("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(src_u32[p]));
        dst_u16[p] = d;
    }
}
#endif

// FP16 → FP8 e4m3: paired PTX cvt.rn.satfinite.e4m3x2.f16x2 (SM89+, Hopper).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
template <size_t VEC_SIZE, std::enable_if_t<(VEC_SIZE % 2 == 0), int> = 0>
__device__ __forceinline__ void vec_convert(
    flashinfer::vec_t<__nv_fp8_e4m3, VEC_SIZE>& out, flashinfer::vec_t<half, VEC_SIZE> const& in)
{
    uint32_t const* src_u32 = reinterpret_cast<uint32_t const*>(&in);
    uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(&out);
#pragma unroll
    for (int p = 0; p < VEC_SIZE / 2; ++p)
    {
        uint16_t d;
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(src_u32[p]));
        dst_u16[p] = d;
    }
}
#endif

// ---- vectorized_quant_impl: load → sync → convert → store ----
// VEC_SIZE is in elements (not bytes), so both SrcT and DstT vectors hold VEC_SIZE values.
template <int VEC_SIZE, typename ThreadingPolicy, typename SrcT, typename DstT>
__device__ void vectorized_quant_impl(DstT* dst, SrcT const* src, int num_elements)
{
    using flashinfer::vec_t;

    int const stride = ThreadingPolicy::stride() * VEC_SIZE;

    for (int e = ThreadingPolicy::offset() * VEC_SIZE; e < num_elements; e += stride)
    {
        vec_t<SrcT, VEC_SIZE> in_vec;
        in_vec.load(src + e);

        // Sync to ensure all threads have loaded their input vectors before any thread starts writing output.
        // This avoids write-after-read hazards in the FP8 in-place case where the output of this kernel is
        // read by the next iteration as input. Without this sync, some threads might start writing their
        // output (DstT) before other threads have loaded their input (SrcT), causing the load to read partially
        // updated data.
        ThreadingPolicy::sync();

        vec_t<DstT, VEC_SIZE> out_vec;
        vec_convert(out_vec, in_vec);
        out_vec.store(dst + e);
    }
}

template <typename ThreadingPolicy, typename SrcT, typename DstT>
__device__ void vectorized_quant(DstT* dst, SrcT const* src, int num_elements)
{
    if (num_elements % 16 == 0)
        vectorized_quant_impl<16, ThreadingPolicy, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 8 == 0)
        vectorized_quant_impl<8, ThreadingPolicy, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 4 == 0)
        vectorized_quant_impl<4, ThreadingPolicy, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 2 == 0)
        vectorized_quant_impl<2, ThreadingPolicy, SrcT, DstT>(dst, src, num_elements);
    else
        vectorized_quant_impl<1, ThreadingPolicy, SrcT, DstT>(dst, src, num_elements);
}

// LOW_PRECISION=false: vectorized byte-copy (SrcT = payload dtype).
// LOW_PRECISION=true:  vectorized SrcT→FP8 quantization via vectorized_quant<SrcT, fp8_e4m3>.
// stride_per_token: byte distance between tokens in recv_buffer_bytes (host-computed, avoids
//   per-thread recomputation):
//   - FP8 external payload: elements_per_token × 1  (compact FP8 layout)
//   - FP8 in-place / byte-copy: elements_per_token × sizeof(SrcT)  (payload-dtype stride)
template <typename ThreadingPolicy, bool LOW_PRECISION, typename SrcT>
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, void const* payload, int elements_per_token,
    int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr, int const* recv_counters, int stride_per_token)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Increment flag_val for this combine round
        *flag_val_ptr = *flag_val_ptr + 1;
    }

    // Copy path: null payload means data is already in workspace — nothing to do.
    if (!LOW_PRECISION && payload == nullptr)
        return;

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

    size_t const token_offset = static_cast<size_t>(global_token_idx) * stride_per_token;

    if constexpr (LOW_PRECISION)
    {
        // Source pointer: external payload or in-place from workspace.
        SrcT const* src_ptr = (payload != nullptr)
            ? static_cast<SrcT const*>(payload) + static_cast<size_t>(global_token_idx) * elements_per_token
            : reinterpret_cast<SrcT const*>(recv_buffer_bytes + token_offset);

        // Destination: stride_per_token encodes the correct layout for both paths
        // (compact FP8 for external, payload-dtype stride for in-place).
        __nv_fp8_e4m3* dst_ptr = reinterpret_cast<__nv_fp8_e4m3*>(recv_buffer_bytes + token_offset);

        vectorized_quant<ThreadingPolicy, SrcT, __nv_fp8_e4m3>(dst_ptr, src_ptr, elements_per_token);
    }
    else
    {
        // Generic byte copy (payload guaranteed non-null by early return above).
        vectorized_copy<ThreadingPolicy>(
            recv_buffer_bytes + token_offset, static_cast<uint8_t const*>(payload) + token_offset, stride_per_token);
    }
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, typename ThreadingPolicy, int TOP_K>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs, // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size,
    int stride_per_token)
{
    int local_token_idx = ThreadingPolicy::token_idx();
    int const size_per_token = elements_per_token * static_cast<int>(sizeof(T));

    if (local_num_tokens == 0)
    {
        // Special case: If local_num_tokens == 0,
        // we need to keep the threads where local_token_idx == 0 alive to participate in the synchronization.
        // Other threads should return.
        if (local_token_idx > 0)
            return;
    }
    else
    {
        // Threads that do not have a token to process should return.
        if (local_token_idx >= local_num_tokens)
            return;
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

#if !DISABLE_SYNC_FOR_PROFILING
    // In-kernel readiness synchronization at start of combine:
    // - One warp signals readiness to all peers with current flag_val.
    // - The first warp of each block waits for all peers' readiness (equality), then __syncthreads.
    bool is_first_warp = threadIdx.x / warpSize == 0;
    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;
        uint32_t expected_value = *ptrs.flag_val;

        if (blockIdx.x == 0)
        {
#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
#if ENABLE_DEBUG_PRINT
                printf("combine: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    peer_rank);
#endif
            }
        }

#pragma unroll 1 // No unroll
        for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
        {
            bool flag_set = false;
            auto s = clock64();
            do
            {
                uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                uint32_t flag_value;
                // Acquire load to ensure visibility of peer's release-store
                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
                printf(
                    "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, expected_value: "
                    "%d, "
                    "address: %p\n",
                    rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
                flag_set = flag_value == expected_value;
            } while (!flag_set && !check_timeout(s));

            if (__builtin_expect(!flag_set, 0))
            {
                printf("combine: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id, peer_rank);
                asm volatile("trap;");
                return;
            }
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
        asm volatile("fence.acquire.sys;");
#else
        asm volatile("fence.acq_rel.sys;");
#endif
    }
    __syncthreads();
#endif

    if (local_num_tokens == 0)
        return;

    // Dispatch to FP8→BF16 or same-type combine path
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        // FP8 recv buffer → BF16 output
        // src_data_ptrs[0] points to a BF16 output buffer (set by moeA2ACombineOp)
        auto* token_output
            = reinterpret_cast<__nv_bfloat16*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;
        vectorized_combine<TOP_K, ThreadingPolicy, __nv_bfloat16, __nv_fp8_e4m3>(
            token_output, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else
    {
        // Get output location for this token (using src_data_ptrs[0] as output)
        T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;
        // Accumulate across ranks in registers, then store once per segment
        vectorized_combine<TOP_K, ThreadingPolicy, T>(
            token_output, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32; // 8 warps per block

    // FP8 in-place (payload_in_workspace=true, prepare_payload==nullptr): each CTA writes
    // FP8 at the BF16-stride position, so CTAs never race — all tokens must be processed.
    // Copy path with null payload is a no-op; 1 block suffices for the flag increment only.
    int global_token_num = (params.use_low_precision || params.prepare_payload != nullptr)
        ? params.ep_size * params.max_tokens_per_rank
        : 1;
    int grid_size_warp = ceilDiv(global_token_num, kWarpsPerBlock);
    int grid_size_block = global_token_num; // one block per token
    int grid = params.one_block_per_token ? grid_size_block : grid_size_warp;

    uint8_t* recv_buffer_bytes = static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank]));
    void const* payload = params.prepare_payload;

    // stride_per_token is computed once on the host and passed to the kernel to avoid
    // per-thread recomputation:
    //   FP8 external: EPT × 1        (compact FP8, dst packed tightly)
    //   FP8 in-place / byte-copy: EPT × sizeof(SrcT)  (payload-dtype stride)
    SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
        SWITCH_DTYPE(params.dtype, SrcT, {
            bool const low_precision_staged = LOW_PRECISION && (params.prepare_payload != nullptr);
            int const stride_per_token = low_precision_staged
                ? params.elements_per_token
                : params.elements_per_token * static_cast<int>(sizeof(SrcT));
            auto kernel_fn = params.one_block_per_token ? moeA2APrepareCombineKernel<BlockPolicy, LOW_PRECISION, SrcT>
                                                        : moeA2APrepareCombineKernel<WarpPolicy, LOW_PRECISION, SrcT>;
            launchWithPdlWhenEnabled("moeA2APrepareCombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
                recv_buffer_bytes, payload, params.elements_per_token, params.ep_size, params.max_tokens_per_rank,
                params.flag_val, params.recv_counters, stride_per_token);
        });
    });
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
    int grid_size_warp = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
    int grid_size_block = params.local_num_tokens;
    // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
    if (grid_size_warp == 0)
    {
        grid_size_warp = 1;
    }
    if (grid_size_block == 0)
    {
        grid_size_block = 1;
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

    // Copy completion flag pointers
    for (int i = 0; i < params.ep_size; i++)
    {
        kernel_ptrs.completion_flags[i] = params.completion_flags[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

    // Copy communication tracking pointers
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;

    int grid = params.one_block_per_token ? grid_size_block : grid_size_warp;

    // stride_per_token: byte distance between tokens in the recv buffer.
    //   FP8 external payload: EPT × 1            (compact FP8 layout)
    //   FP8 in-place / non-FP8: EPT × sizeof(PayloadT)  (payload-dtype stride)
    bool const low_precision_staged = params.use_low_precision && (params.prepare_payload != nullptr);
    int stride_per_token;
    SWITCH_DTYPE(params.dtype, PayloadT, {
        stride_per_token = low_precision_staged ? params.elements_per_token
                                                : params.elements_per_token * static_cast<int>(sizeof(PayloadT));
    });

    // When use_low_precision is set the recv buffers contain FP8 data regardless of params.dtype,
    // so dispatch the FP8 accumulation kernel in that case.
    auto const effective_dtype = params.use_low_precision ? nvinfer1::DataType::kFP8 : params.dtype;

    // Launch appropriate kernel with compact macros
    SWITCH_DTYPE(effective_dtype, TKernelType, {
        SWITCH_POLICY(params.one_block_per_token, Policy, {
            SWITCH_TOP_K(params.top_k, TOP_K, {
                auto kernel_fn = moeA2ACombineKernel<TKernelType, Policy, TOP_K>;
                launchWithPdlWhenEnabled("moeA2ACombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
                    kernel_ptrs, params.max_tokens_per_rank, params.elements_per_token, params.local_num_tokens,
                    params.ep_rank, params.ep_size, stride_per_token);
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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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
