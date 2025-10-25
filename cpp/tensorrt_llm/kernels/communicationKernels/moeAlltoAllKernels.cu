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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <type_traits>

namespace tensorrt_llm::kernels::moe_a2a
{

#define ENABLE_DEBUG_PRINT 0
#define DISABLE_SYNC_FOR_PROFILING 0

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
    case 8:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 8;                                                                                       \
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

__global__ void moeA2APrepareDispatchKernel(
    int* send_counters, int* local_token_counter, int ep_size, uint32_t* flag_val_ptr)
{
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
// Generic Dispatch Kernel Implementation
// One warp per token design:
// - Each CTA has 256 threads = 8 warps
// - Each warp independently processes one token and all its payloads
// - Better GPU utilization and reduced synchronization overhead
// ============================================================================

template <typename ThreadingPolicy, int TOP_K>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts_per_rank)
{

    int thread_idx = ThreadingPolicy::offset();
    int local_token_idx = ThreadingPolicy::token_idx();

    if (local_token_idx >= local_num_tokens)
    {
        return;
    }

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

        vectorized_dispatch<TOP_K, ThreadingPolicy>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx,
            ptrs, topk_target_ranks, topk_send_indices);
    }

    ThreadingPolicy::sync();

    bool is_first_warp = threadIdx.x / warpSize == 0;
    if (is_first_warp)
    {
        int lane_id = threadIdx.x % warpSize;

        bool is_last_token = false;
        if (lane_id == 0)
        {
            int cnt = atomicAdd(ptrs.local_token_counter, 1);
            is_last_token = cnt + 1 == local_num_tokens;
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

#if !DISABLE_SYNC_FOR_PROFILING
            uint32_t expected_value = *ptrs.flag_val;

            asm volatile("fence.release.sys;");
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
                } while (!flag_set);
            }
            // asm volatile("fence.acquire.sys;");
#endif
        }
    }
}

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params)
{
    moeA2APrepareDispatchKernel<<<1, params.ep_size, 0, params.stream>>>(
        params.send_counters, params.local_token_counter, params.ep_size, params.flag_val);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens > 0);
    TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);

    // Prepare kernel pointers struct
    DispatchKernelPointers kernel_ptrs = {}; // Zero-initialize

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

    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();
    constexpr int kWarpSize = 32;
    int const kWarpsPerBlock = kBlockSize / kWarpSize;

    // Configure kernel launch
    if (params.one_block_per_token)
    {
        int grid_size = params.local_num_tokens;
        int shared_bytes = 2 * params.top_k * (int) sizeof(int);
        SWITCH_TOP_K(params.top_k, TOP_K,
            moeA2ADispatchKernel<BlockPolicy, TOP_K><<<grid_size, kBlockSize, shared_bytes, params.stream>>>(
                params.token_selected_experts, kernel_ptrs, params.num_payloads, params.max_tokens_per_rank,
                params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts_per_rank))
    }
    else
    {
        int grid_size = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
        int shared_bytes = 2 * kWarpsPerBlock * params.top_k * (int) sizeof(int);
        SWITCH_TOP_K(params.top_k, TOP_K,
            moeA2ADispatchKernel<WarpPolicy, TOP_K><<<grid_size, kBlockSize, shared_bytes, params.stream>>>(
                params.token_selected_experts, kernel_ptrs, params.num_payloads, params.max_tokens_per_rank,
                params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts_per_rank))
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
        if constexpr (TOP_K == 8)
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
    int bytes_per_token, int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr, int const* recv_counters)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Increment flag_val for this combine round
        *flag_val_ptr = *flag_val_ptr + 1;
    }

    if (payload_bytes == nullptr)
        return;

    int slot_idx = ThreadingPolicy::token_idx();

    int total_slots = ep_size * max_tokens_per_rank;
    if (slot_idx >= total_slots)
        return;

    // Map global token to (source_rank, token_idx)
    int source_rank = slot_idx / max_tokens_per_rank;
    int token_idx = slot_idx % max_tokens_per_rank;

    // Skip invalid tokens beyond per-source recv count
    if (token_idx >= recv_counters[source_rank])
        return;

    // Calculate source and destination pointers for this token
    size_t slot_offset = static_cast<size_t>(slot_idx) * bytes_per_token;
    uint8_t* dst_ptr = recv_buffer_bytes + slot_offset;
    uint8_t const* src_ptr = payload_bytes + slot_offset;

    // Copy one token's data using vectorized copy with policy
    vectorized_copy<ThreadingPolicy>(dst_ptr, src_ptr, bytes_per_token);
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, typename ThreadingPolicy, int TOP_K>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs, // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size)
{
    int local_token_idx = ThreadingPolicy::token_idx();
    int const size_per_token = elements_per_token * sizeof(T);

    if (local_token_idx >= local_num_tokens)
    {
        return;
    }

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
            // asm volatile("fence.release.sys;");
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
            do
            {
                uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
                uint32_t flag_value;
                // Acquire load to ensure visibility of peer's release-store
                asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
#if ENABLE_DEBUG_PRINT
                printf(
                    "combine: ---Rank %d received completion flag from rank %d, flag_value: %d, expected_value: %d, "
                    "address: %p\n",
                    rank_id, peer_rank, flag_value, expected_value, flag_ptr);
#endif
                flag_set = flag_value == expected_value;
            } while (!flag_set);
        }
        asm volatile("fence.acquire.sys;");
    }
    __syncthreads();
#endif

    // Get output location for this token (using src_data_ptrs[0] as output)
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;

    // Accumulate across ranks in registers, then store once per segment
    vectorized_combine<TOP_K, ThreadingPolicy, T>(token_output, size_per_token, rank_id, max_tokens_per_rank, ptrs);
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
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
    int total_slots = params.prepare_payload == nullptr ? 1 : params.ep_size * params.max_tokens_per_rank;
    int grid_size_warp = ceilDiv(total_slots, kWarpsPerBlock);
    int grid_size_block = total_slots; // one block per token

    if (params.one_block_per_token)
    {
        moeA2APrepareCombineKernel<BlockPolicy><<<grid_size_block, kBlockSize, 0, params.stream>>>(
            static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])),
            static_cast<uint8_t const*>(params.prepare_payload), bytes_per_token, params.ep_size,
            params.max_tokens_per_rank, params.flag_val, params.recv_counters);
    }
    else
    {
        moeA2APrepareCombineKernel<WarpPolicy><<<grid_size_warp, kBlockSize, 0, params.stream>>>(
            static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])),
            static_cast<uint8_t const*>(params.prepare_payload), bytes_per_token, params.ep_size,
            params.max_tokens_per_rank, params.flag_val, params.recv_counters);
    }
}

// ============================================================================
// Combine Launch Function
// ============================================================================

void moe_a2a_combine_launch(MoeA2ACombineParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.local_num_tokens > 0);
    TLLM_CHECK(params.elements_per_token > 0);

    // Configure kernel launch
    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
    int const kWarpsPerBlock = kBlockSize / 32; // warpSize
    int grid_size_warp = ceilDiv(params.local_num_tokens, kWarpsPerBlock);
    int grid_size_block = params.local_num_tokens;

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

    // Launch appropriate kernel with compact macros
    SWITCH_DTYPE(params.dtype, TKernelType, {
        SWITCH_POLICY(params.one_block_per_token, Policy, {
            SWITCH_TOP_K(params.top_k, TOP_K, {
                auto launch = [&](int grid_blocks, int block_threads)
                {
                    moeA2ACombineKernel<TKernelType, Policy, TOP_K>
                        <<<grid_blocks, block_threads, 0, params.stream>>>(kernel_ptrs, params.max_tokens_per_rank,
                            params.elements_per_token, params.local_num_tokens, params.ep_rank, params.ep_size);
                };
                int grid = params.one_block_per_token ? grid_size_block : grid_size_warp;
                int cta = kBlockSize;
                launch(grid, cta);
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
    moeA2ASanitizeExpertIdsKernel<<<grid, kBlockSize, 0, stream>>>(
        expert_ids, recv_counters, ep_size, max_tokens_per_rank, top_k, invalid_id);
}

} // namespace tensorrt_llm::kernels::moe_a2a
