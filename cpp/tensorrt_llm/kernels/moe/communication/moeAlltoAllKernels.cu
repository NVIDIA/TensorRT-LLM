/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
#define TLLM_CUDA_HOST_PASS 1
#else
#define TLLM_CUDA_HOST_PASS 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890) || TLLM_CUDA_HOST_PASS
#define TLLM_MOE_A2A_COMPILE_SM89 1
#else
#define TLLM_MOE_A2A_COMPILE_SM89 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900) || TLLM_CUDA_HOST_PASS
#define TLLM_MOE_A2A_COMPILE_SM90 1
#else
#define TLLM_MOE_A2A_COMPILE_SM90 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000) || TLLM_CUDA_HOST_PASS
#define TLLM_MOE_A2A_COMPILE_SM100 1
#else
#define TLLM_MOE_A2A_COMPILE_SM100 0
#endif

#ifndef DISABLE_TIMEOUT
#define DISABLE_TIMEOUT 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

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
    case 3:                                                                                                            \
    {                                                                                                                  \
        constexpr int TOP_K = 3;                                                                                       \
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
    case tensorrt_llm::DataType::kHALF:                                                                                \
    {                                                                                                                  \
        using TYPE = half;                                                                                             \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    case tensorrt_llm::DataType::kBF16:                                                                                \
    {                                                                                                                  \
        using TYPE = __nv_bfloat16;                                                                                    \
        __VA_ARGS__;                                                                                                   \
        break;                                                                                                         \
    }                                                                                                                  \
    default:                                                                                                           \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for moe_a2a_combine");                                          \
    }                                                                                                                  \
    }

#if DISABLE_TIMEOUT
#define check_timeout(s, budget) false
#else
// The host supplies the wait budget in clock64() cycles as a launch argument.
#define check_timeout(s, budget) ((clock64() - (s)) > (budget))
#endif

// ============================================================================
// Helper Functions for Expert-to-Rank Mapping
// ============================================================================

// Compute which rank owns a given expert using contiguous ceil/floor partitioning.
// Supports non-divisible distribution when num_experts % ep_size != 0:
//   base      = num_experts / ep_size
//   remainder = num_experts % ep_size
//   - Ranks [0, remainder) each own (base + 1) experts.
//   - Ranks [remainder, ep_size) each own base experts.
__device__ __forceinline__ int compute_target_rank_id(int expert_id, int base, int remainder)
{
    if (remainder == 0)
    {
        return expert_id / base;
    }
    int const split = remainder * (base + 1);
    if (expert_id < split)
    {
        return expert_id / (base + 1);
    }
    return remainder + (expert_id - split) / base;
}

// Test bit `rank` in a kRankMaskWords-wide little-endian uint64 bitmask.
// Word 0 covers ranks 0..63, word 1 covers ranks 64..127, etc.
// `rank >> 6` and `rank & 63` divide / modulo by 64.
__device__ __forceinline__ bool is_rank_active(uint64_t const* mask, int rank)
{
    return (mask[rank >> 6] >> (rank & 63)) & 1ULL;
}

// Each A2A round uses two consecutive flag values: 2N + 1 for dispatch and
// 2N + 2 for combine. Subtracting one and dividing by two maps both values
// to round N; the low bit selects the alternating counter bank.
__device__ __forceinline__ uint32_t round_parity(uint32_t flag_val)
{
    return ((flag_val - 1U) >> 1U) & 1U;
}

// Round flags use system-scope relaxed accesses. Required payload visibility
// fences stay at the call sites.
__device__ __forceinline__ void publish_round_flag(uint32_t* address, uint32_t value)
{
    asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ bool wait_round_flag(
    uint32_t const* address, uint32_t expected, int64_t timeout_cycles, int rank_id, int peer_rank, char const* phase)
{
    auto const start = clock64();
    uint32_t observed;
    do
    {
        asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(observed) : "l"(address) : "memory");
#if ENABLE_DEBUG_PRINT
        printf("%s: rank %d waiting for rank %d flag=%u expected=%u address=%p\n", phase, rank_id, peer_rank, observed,
            expected, address);
#endif
        if (observed == expected)
        {
            return true;
        }
    } while (!check_timeout(start, timeout_cycles));

    printf("%s: rank %d timed out waiting for rank %d flag=%u expected=%u\n", phase, rank_id, peer_rank, observed,
        expected);
    asm volatile("trap;" ::: "memory");
    return false;
}

// Mask the first NUM_LANES lanes of a warp.
template <int NUM_LANES>
__host__ __device__ constexpr uint32_t make_warp_lane_mask()
{
    static_assert(NUM_LANES > 0 && NUM_LANES <= 32, "warp lane count must be in [1, 32]");
    return (NUM_LANES == 32) ? ~0U : ((1U << NUM_LANES) - 1U);
}

template <int TOP_K, bool ENABLE_RANK_MASK, bool USE_RANK_COMPACT_ROUTING>
__device__ __forceinline__ void route_dispatch_token(int32_t const* token_selected_experts,
    DispatchKernelPointers const& ptrs, int local_token_idx, int ep_size, int num_experts, int* topk_target_ranks,
    int* topk_target_indices, uint32_t lane_mask)
{
    static_assert(TOP_K <= 32, "warp-parallel routing requires TOP_K <= warpSize");
    int const k = threadIdx.x;

    if constexpr (USE_RANK_COMPACT_ROUTING)
    {
        // EP < TOP_K, so the routing lanes initialize every destination slot.
        topk_target_ranks[k] = -1;
        topk_target_indices[k] = -1;
        __syncwarp(lane_mask);
    }

    int const ep_base = num_experts / ep_size;
    int const ep_remainder = num_experts - ep_base * ep_size;
    int const expert_id = token_selected_experts[local_token_idx * TOP_K + k];
    // Invalid experts have no destination, but their lanes still participate in warp collectives.
    bool const valid_expert = expert_id >= 0 && expert_id < num_experts;
    int const target_rank = valid_expert ? compute_target_rank_id(expert_id, ep_base, ep_remainder) : -1;

    // Experts on the same rank share one token transfer; only their first lane allocates a send slot.
    uint32_t const same_target = __match_any_sync(lane_mask, target_rank);
    bool should_send = valid_expert && ((__ffs(same_target) - 1) == k);
    if constexpr (ENABLE_RANK_MASK)
    {
        should_send = should_send && is_rank_active(ptrs.active_rank_mask, target_rank);
    }

    int const target_rank_to_store = should_send ? target_rank : -1;
    int const target_index_to_store = should_send ? atomicAdd(&ptrs.send_counters[target_rank], 1) : -1;

    // Keep persistent routing metadata indexed by top-k slot for combine.
    ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank_to_store;
    ptrs.topk_target_indices[local_token_idx * TOP_K + k] = target_index_to_store;
    if constexpr (USE_RANK_COMPACT_ROUTING)
    {
        // Pack unique target ranks into consecutive shared-memory slots instead of leaving
        // holes at duplicate expert lanes. Dispatch then scans a rank-bounded list, not TOP_K slots.
        uint32_t const sending_lanes = __ballot_sync(lane_mask, should_send);
        if (should_send)
        {
            // Count sending lanes before this lane to obtain its packed slot.
            int const compact_index = __popc(sending_lanes & ((1U << k) - 1U));
            topk_target_ranks[compact_index] = target_rank;
            topk_target_indices[compact_index] = target_index_to_store;
        }
    }
    else
    {
        topk_target_ranks[k] = target_rank_to_store;
        topk_target_indices[k] = target_index_to_store;
    }
}

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

template <int VEC_SIZE>
__device__ void vectorized_copy_impl(void* dst, void const* src, int size)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int const stride = blockDim.x * VEC_SIZE;

    for (int offset = threadIdx.x * VEC_SIZE; offset < size; offset += stride)
    {
        vec_t<uint8_t, VEC_SIZE> v;
        v.load(src_ptr + offset);
        v.store(dst_ptr + offset);
    }
}

__device__ void vectorized_copy(void* dst, void const* src, int size)
{
    if (size % 16 == 0)
    {
        vectorized_copy_impl<16>(dst, src, size);
    }
    else if (size % 8 == 0)
    {
        vectorized_copy_impl<8>(dst, src, size);
    }
    else if (size % 4 == 0)
    {
        vectorized_copy_impl<4>(dst, src, size);
    }
    else if (size % 2 == 0)
    {
        vectorized_copy_impl<2>(dst, src, size);
    }
    else
    {
        vectorized_copy_impl<1>(dst, src, size);
    }
}

// Fan each source vector out through the CTA's shared destination table.
template <int VEC_SIZE, int TOP_K, bool USE_RANK_COMPACT_ROUTING>
__device__ void vectorized_dispatch_impl(
    uint8_t const* src_ptr, int bytes_per_token, int ep_size, uint8_t* const* destinations)
{
    using flashinfer::vec_t;

    int const num_destinations = USE_RANK_COMPACT_ROUTING ? ep_size : TOP_K;
    for (int offset = threadIdx.x * VEC_SIZE; offset < bytes_per_token; offset += blockDim.x * VEC_SIZE)
    {
        vec_t<uint8_t, VEC_SIZE> value;
        value.load(src_ptr + offset);
#pragma unroll(USE_RANK_COMPACT_ROUTING ? 2 : TOP_K)
        for (int k = 0; k < num_destinations; ++k)
        {
            uint8_t* const destination = destinations[k];
            if (destination != nullptr)
            {
                value.store(destination + offset);
            }
        }
    }
}

template <int TOP_K, bool USE_RANK_COMPACT_ROUTING>
__device__ void vectorized_dispatch(
    uint8_t const* src_ptr, int bytes_per_token, int ep_size, uint8_t* const* destinations)
{
    if (bytes_per_token % 16 == 0)
    {
        vectorized_dispatch_impl<16, TOP_K, USE_RANK_COMPACT_ROUTING>(src_ptr, bytes_per_token, ep_size, destinations);
    }
    else if (bytes_per_token % 8 == 0)
    {
        vectorized_dispatch_impl<8, TOP_K, USE_RANK_COMPACT_ROUTING>(src_ptr, bytes_per_token, ep_size, destinations);
    }
    else if (bytes_per_token % 4 == 0)
    {
        vectorized_dispatch_impl<4, TOP_K, USE_RANK_COMPACT_ROUTING>(src_ptr, bytes_per_token, ep_size, destinations);
    }
    else if (bytes_per_token % 2 == 0)
    {
        vectorized_dispatch_impl<2, TOP_K, USE_RANK_COMPACT_ROUTING>(src_ptr, bytes_per_token, ep_size, destinations);
    }
    else
    {
        vectorized_dispatch_impl<1, TOP_K, USE_RANK_COMPACT_ROUTING>(src_ptr, bytes_per_token, ep_size, destinations);
    }
}

__global__ void moeA2APrepareDispatchKernel(
    int* send_counters, int* recv_counters, int* local_token_counter, int ep_size, uint32_t* flag_val_ptr)
{
#if TLLM_MOE_A2A_COMPILE_SM90
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        uint32_t const next_round = *flag_val_ptr + 1;
        *flag_val_ptr = next_round;
        *local_token_counter = 0;
    }
    __syncthreads();

    if (idx < ep_size)
    {
        send_counters[idx] = 0;
        uint32_t const current_parity = round_parity(*flag_val_ptr);
        uint32_t const next_parity = current_parity ^ 1U;
        recv_counters[next_parity * ep_size + idx] = -1;
    }
}

// ============================================================================
// Dispatch Kernels
// ============================================================================

template <int TOP_K, bool ENABLE_EPLB, bool ENABLE_RANK_MASK, bool USE_RANK_COMPACT_ROUTING>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, TOP_K]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int num_experts, int eplb_stats_num_experts)
{
    int thread_idx = threadIdx.x;
    int local_token_idx = blockIdx.x;

    if (local_num_tokens == 0)
    {
        // Special case: If local_num_tokens == 0,
        // we need to keep the threads where local_token_idx == 0 alive to participate in the synchronization.
        // Other threads should return.
        if (local_token_idx > 0)
            return;
#if TLLM_MOE_A2A_COMPILE_SM90
        cudaGridDependencySynchronize();
#endif
    }
    else
    {
        // Threads that do not have a token to process should return.
        if (local_token_idx >= local_num_tokens)
            return;

        // Compact tiles keep valid destinations in top-k order and pad with -1 send indices.
        // Global routing metadata always retains all top-k slots.
        extern __shared__ int smem[];
        int* smem_topk_target_ranks = smem;
        int* smem_topk_target_indices = smem + TOP_K;

#if TLLM_MOE_A2A_COMPILE_SM90
        cudaGridDependencySynchronize();
#endif
        static_assert(TOP_K > 0 && TOP_K <= 32, "routing and destination setup require one warp");
        constexpr uint32_t kRoutingLaneMask = make_warp_lane_mask<TOP_K>();
        __shared__ uint8_t* destinations[kMaxPayloads][TOP_K];
        if (thread_idx < TOP_K)
        {
            route_dispatch_token<TOP_K, ENABLE_RANK_MASK, USE_RANK_COMPACT_ROUTING>(token_selected_experts, ptrs,
                local_token_idx, ep_size, num_experts, smem_topk_target_ranks, smem_topk_target_indices,
                kRoutingLaneMask);
            // Compact routing slots may be written by a different lane.
            __syncwarp(kRoutingLaneMask);

            // Resolve each payload's destinations once per CTA; unused routes remain null.
            int const target_index = smem_topk_target_indices[thread_idx];
            int const peer = smem_topk_target_ranks[thread_idx];
            for (int p = 0; p < num_payloads; ++p)
            {
                uint8_t* destination = nullptr;
                if (target_index >= 0)
                {
                    size_t const token = static_cast<size_t>(rank_id) * max_tokens_per_rank + target_index;
                    destination
                        = static_cast<uint8_t*>(ptrs.recv_buffers[peer][p]) + token * ptrs.payload_bytes_per_token[p];
                }
                destinations[p][thread_idx] = destination;
            }
        }
        // Publish the destination table to all payload-copying warps.
        __syncthreads();

        // Perform a single source load and TOP_K fanout per payload
        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
            int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;
            vectorized_dispatch<TOP_K, USE_RANK_COMPACT_ROUTING>(
                src_ptr, bytes_per_token, ep_size, destinations[payload_idx]);
        }

        __syncthreads();
    }
#if TLLM_MOE_A2A_COMPILE_SM90
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
            uint32_t const parity = round_parity(*ptrs.flag_val);
            int* const round_recv_counters = ptrs.recv_counters[rank_id] + parity * ep_size;
// Store send_counters to recv_counters.
// Skip masked target ranks: their symmetric memory may be inaccessible.
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, target_rank))
                        continue;
                }
                int send_count = ptrs.send_counters[target_rank];
                ptrs.recv_counters[target_rank][parity * ep_size + rank_id] = send_count;
            }

            if constexpr (ENABLE_EPLB)
            {
                // Write local stats into peer buffers before the release fence below.
                // Skip masked target ranks for the same reason as above.
#pragma unroll 1
                for (int target_rank = 0; target_rank < ep_size; ++target_rank)
                {
                    if constexpr (ENABLE_RANK_MASK)
                    {
                        if (!is_rank_active(ptrs.active_rank_mask, target_rank))
                            continue;
                    }
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

#if TLLM_MOE_A2A_COMPILE_SM90
            // .acquire and .release qualifiers for fence instruction require sm_90 or higher.
            asm volatile("fence.release.sys;");
#else
            asm volatile("fence.acq_rel.sys;");
#endif
            // Signal completion to all active peers; skip dead ranks (their symmetric memory
            // is unreachable).
#pragma unroll 1 // No unroll as one iter is typically enough
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, target_rank))
                        continue;
                }
                uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
                publish_round_flag(flag_addr, expected_value);

#if ENABLE_DEBUG_PRINT
                printf("dispatch: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    target_rank);
#endif
            }

            // Wait for all active peers to signal; skip dead ranks (otherwise we would
            // spin forever — this is the bug the rank-mask is here to prevent).
#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, peer_rank))
                    {
                        round_recv_counters[peer_rank] = 0;
                        continue;
                    }
                }
                if (!wait_round_flag(&ptrs.completion_flags[rank_id][peer_rank], expected_value, ptrs.timeout_cycles,
                        rank_id, peer_rank, "dispatch"))
                {
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
        params.stream, params.send_counters, params.recv_counters[params.ep_rank], params.local_token_counter,
        params.ep_size, params.flag_val);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
{
    constexpr int kBlockSize = 256;

    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);
    // The local rank must always be marked active in its own view of the mask;
    // otherwise the kernel itself would be running on a "dead" rank.
    if (params.enable_rank_mask)
    {
        TLLM_CHECK_WITH_INFO((params.active_rank_mask[params.ep_rank >> 6] >> (params.ep_rank & 63)) & 1ULL,
            "active_rank_mask must mark the local ep_rank (%d) as active", params.ep_rank);
    }

    // Prepare kernel pointers struct
    DispatchKernelPointers kernel_ptrs = {};
    kernel_ptrs.timeout_cycles = params.timeout_cycles;

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
    kernel_ptrs.topk_target_indices = params.topk_target_indices;
    kernel_ptrs.eplb_local_stats = params.eplb_local_stats;

    // Copy active-rank bitmask into the kernel pointers struct
    for (int w = 0; w < kRankMaskWords; ++w)
    {
        kernel_ptrs.active_rank_mask[w] = params.active_rank_mask[w];
    }

    int grid_size = params.local_num_tokens;
    if (grid_size == 0)
    {
        grid_size = 1;
    }

    int const shared_bytes = 2 * params.top_k * static_cast<int>(sizeof(int));

    SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
        SWITCH_BOOL(params.enable_eplb, EPLB_STATS, {
            SWITCH_TOP_K(params.top_k, TOP_K, {
                SWITCH_BOOL(params.ep_size < TOP_K, ENABLE_RANK_COMPACT_ROUTING, {
                    auto kernel_fn
                        = moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK, ENABLE_RANK_COMPACT_ROUTING>;
                    launchWithPdlWhenEnabled("moeA2ADispatchKernel", kernel_fn, grid_size, kBlockSize, shared_bytes,
                        params.stream, params.token_selected_experts, kernel_ptrs, params.num_payloads,
                        params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size,
                        params.num_experts, params.eplb_stats_num_experts);
                });
            });
        });
    });
}

// ============================================================================
// Combine kernels
// ============================================================================

// Load source vectors in their wire dtype and accumulate in FP32.
// Only the first source_count entries are valid, one per contributing rank.
template <int VEC_SIZE, typename OutputT, typename InputT>
__device__ void vectorized_combine_impl(
    OutputT* output, int size_per_token, uint8_t const* const* sources, int source_count)
{
    using flashinfer::vec_t;
    constexpr int kElements = VEC_SIZE / static_cast<int>(sizeof(InputT));
    if (source_count <= 2)
    {
        for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += blockDim.x * VEC_SIZE)
        {
            vec_t<InputT, kElements> left;
            vec_t<InputT, kElements> right;
            left.fill(InputT(0.0f));
            right.fill(InputT(0.0f));
            if (source_count > 0)
            {
                left.load(reinterpret_cast<InputT const*>(sources[0] + offset));
            }
            if (source_count > 1)
            {
                right.load(reinterpret_cast<InputT const*>(sources[1] + offset));
            }
            vec_t<float, kElements> value;
#pragma unroll
            for (int j = 0; j < kElements; ++j)
            {
                value[j] = static_cast<float>(left[j]) + static_cast<float>(right[j]);
            }
            value.cast_store(output + offset / static_cast<int>(sizeof(InputT)));
        }
        return;
    }

    // TOP_K <= 32: eight sources per lane require at most four cooperating lanes.
    int const rank_bits = (source_count > 8) + (source_count > 16);
    int const rank_lanes = 1 << rank_bits;
    int const lane = threadIdx.x & 31;
    int const rank_lane = lane & (rank_lanes - 1);
    int const output_lane = lane >> rank_bits;
    int const warp = threadIdx.x >> 5;
    int const vectors_per_warp = warpSize >> rank_bits;
    int const stride = (blockDim.x >> rank_bits) * VEC_SIZE;
    // Prefetch low-precision vectors, then combine adjacent sources in tree order.
    for (int base = warp * vectors_per_warp * VEC_SIZE; base < size_per_token; base += stride)
    {
        int const offset = base + output_lane * VEC_SIZE;
        bool const valid_output = offset < size_per_token;
        vec_t<InputT, kElements> packed[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            packed[k].fill(InputT(0.0f));
            int const source = 8 * rank_lane + k;
            if (valid_output && source < source_count)
            {
                packed[k].load(reinterpret_cast<InputT const*>(sources[source] + offset));
            }
        }
        vec_t<float, kElements> value;
        if constexpr (std::is_same_v<InputT, __nv_fp8_e4m3> && kElements % 2 == 0)
        {
#pragma unroll
            for (int j = 0; j < kElements; j += 2)
            {
                float2 pair[8];
#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    __nv_fp8x2_e4m3 input;
                    input.__x = reinterpret_cast<uint16_t const*>(&packed[k])[j / 2];
                    pair[k] = static_cast<float2>(input);
                }
                float const x0 = pair[0].x + pair[1].x;
                float const x1 = pair[2].x + pair[3].x;
                float const x2 = pair[4].x + pair[5].x;
                float const x3 = pair[6].x + pair[7].x;
                float const y0 = pair[0].y + pair[1].y;
                float const y1 = pair[2].y + pair[3].y;
                float const y2 = pair[4].y + pair[5].y;
                float const y3 = pair[6].y + pair[7].y;
                value[j] = (x0 + x1) + (x2 + x3);
                value[j + 1] = (y0 + y1) + (y2 + y3);
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElements; ++j)
            {
                float const p0 = static_cast<float>(packed[0][j]) + static_cast<float>(packed[1][j]);
                float const p1 = static_cast<float>(packed[2][j]) + static_cast<float>(packed[3][j]);
                float const p2 = static_cast<float>(packed[4][j]) + static_cast<float>(packed[5][j]);
                float const p3 = static_cast<float>(packed[6][j]) + static_cast<float>(packed[7][j]);
                value[j] = (p0 + p1) + (p2 + p3);
            }
        }
#pragma unroll
        for (int step = 1; step < 4; step *= 2)
        {
            if (step < rank_lanes)
            {
#pragma unroll
                for (int j = 0; j < kElements; ++j)
                {
                    float const next = __shfl_down_sync(~0U, value[j], step, rank_lanes);
                    if ((rank_lane & (2 * step - 1)) == 0)
                    {
                        value[j] += next;
                    }
                }
            }
        }
        if (rank_lane == 0 && valid_output)
        {
            value.cast_store(output + offset / static_cast<int>(sizeof(InputT)));
        }
    }
}

// Pack valid source pointers in routing order and count the contributing ranks.
// stride_per_token can exceed the wire size for in-place low-precision combine.
template <int TOP_K, typename OutputT, typename InputT>
__device__ void vectorized_combine(
    OutputT* output, int size_per_token, int stride_per_token, CombineKernelPointers const& ptrs)
{
    static_assert(TOP_K > 0 && TOP_K <= 32, "combine routing requires TOP_K <= warpSize");
    constexpr uint32_t kRoutingLaneMask = make_warp_lane_mask<TOP_K>();
    __shared__ uint8_t const* sources[TOP_K];
    __shared__ int source_count;
    if (threadIdx.x < TOP_K)
    {
        int const k = threadIdx.x;
        int const index = blockIdx.x * TOP_K + k;
        int const target_index = ptrs.topk_target_indices[index];
        uint32_t const valid = __ballot_sync(kRoutingLaneMask, target_index >= 0);
        if (k == 0)
        {
            source_count = __popc(valid);
        }
        if (target_index >= 0)
        {
            int const peer = ptrs.topk_target_ranks[index];
            int const compact_index = __popc(valid & ((1U << k) - 1U));
            sources[compact_index] = ptrs.source_buffers[peer] + static_cast<size_t>(target_index) * stride_per_token;
        }
    }
    __syncthreads();

    if (size_per_token % 16 == 0)
    {
        vectorized_combine_impl<16, OutputT, InputT>(output, size_per_token, sources, source_count);
    }
    else if (size_per_token % 8 == 0)
    {
        vectorized_combine_impl<8, OutputT, InputT>(output, size_per_token, sources, source_count);
    }
    else if (size_per_token % 4 == 0)
    {
        vectorized_combine_impl<4, OutputT, InputT>(output, size_per_token, sources, source_count);
    }
    else if (size_per_token % 2 == 0)
    {
        vectorized_combine_impl<2, OutputT, InputT>(output, size_per_token, sources, source_count);
    }
    else if constexpr (sizeof(InputT) == 1)
    {
        vectorized_combine_impl<1, OutputT, InputT>(output, size_per_token, sources, source_count);
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

// Paired BF16 -> FP8 conversion requires PTX 9.1 (CUDA 13.1) and an
// architecture- or family-specific Blackwell target.
#if TLLM_MOE_A2A_COMPILE_SM100 && (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 130100)                \
    && (TLLM_CUDA_HOST_PASS || defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
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
#if TLLM_MOE_A2A_COMPILE_SM89
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
template <int VEC_SIZE, typename SrcT, typename DstT>
__device__ void vectorized_quant_impl(DstT* dst, SrcT const* src, int num_elements)
{
    using flashinfer::vec_t;

    int const stride = blockDim.x * VEC_SIZE;

    for (int e = threadIdx.x * VEC_SIZE; e < num_elements; e += stride)
    {
        vec_t<SrcT, VEC_SIZE> in_vec;
        in_vec.load(src + e);

        // Sync to ensure all threads have loaded their input vectors before any thread starts writing output.
        // This avoids write-after-read hazards in the FP8 in-place case where the output of this kernel is
        // read by the next iteration as input. Without this sync, some threads might start writing their
        // output (DstT) before other threads have loaded their input (SrcT), causing the load to read partially
        // updated data.
        __syncthreads();

        vec_t<DstT, VEC_SIZE> out_vec;
        vec_convert(out_vec, in_vec);
        out_vec.store(dst + e);
    }
}

template <typename SrcT, typename DstT>
__device__ void vectorized_quant(DstT* dst, SrcT const* src, int num_elements)
{
    if (num_elements % 16 == 0)
        vectorized_quant_impl<16, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 8 == 0)
        vectorized_quant_impl<8, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 4 == 0)
        vectorized_quant_impl<4, SrcT, DstT>(dst, src, num_elements);
    else if (num_elements % 2 == 0)
        vectorized_quant_impl<2, SrcT, DstT>(dst, src, num_elements);
    else
        vectorized_quant_impl<1, SrcT, DstT>(dst, src, num_elements);
}

// Advance flag_val to the combine phase and prepare valid tokens in the requested range.
// Copy SrcT payloads, or quantize them to FP8 when LOW_PRECISION is enabled, into combine_input_base.
// This kernel performs no remote transfers or reduction.
template <bool LOW_PRECISION, typename SrcT>
__global__ void moeA2APrepareCombineKernel(uint8_t* combine_input_base, void const* source_payload,
    int elements_per_token, int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr, int const* recv_counters,
    int source_stride_per_token, int workspace_stride_per_token, int prepare_num_tokens)
{
#if TLLM_MOE_A2A_COMPILE_SM90
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    uint32_t const parity = round_parity(*flag_val_ptr);
    int const* const round_recv_counters = recv_counters + parity * ep_size;
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        *flag_val_ptr = *flag_val_ptr + 1;
    }

    if (blockIdx.x >= prepare_num_tokens)
        return;
    int const global_token_idx = blockIdx.x;

    // Map global_token_idx to (rank_idx, local_token_idx)
    int rank_idx = global_token_idx / max_tokens_per_rank;
    int local_token_idx = global_token_idx % max_tokens_per_rank;

    // Skip invalid tokens beyond per-rank recv count
    if (local_token_idx >= round_recv_counters[rank_idx])
        return;

    size_t const source_offset = static_cast<size_t>(global_token_idx) * source_stride_per_token;
    size_t const workspace_offset = static_cast<size_t>(global_token_idx) * workspace_stride_per_token;

    if constexpr (LOW_PRECISION)
    {
        SrcT const* src_ptr
            = reinterpret_cast<SrcT const*>(static_cast<uint8_t const*>(source_payload) + source_offset);
        __nv_fp8_e4m3* dst_ptr = reinterpret_cast<__nv_fp8_e4m3*>(combine_input_base + workspace_offset);
        vectorized_quant<SrcT, __nv_fp8_e4m3>(dst_ptr, src_ptr, elements_per_token);
    }
    else
    {
        // Same-type byte copy.
        uint8_t const* src = static_cast<uint8_t const*>(source_payload) + source_offset;
        uint8_t* dst = combine_input_base + workspace_offset;
        vectorized_copy(dst, src, elements_per_token * static_cast<int>(sizeof(SrcT)));
    }
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, int TOP_K, bool LOW_PRECISION, bool ENABLE_RANK_MASK>
__global__ void moeA2ACombineKernel(const CombineKernelPointers ptrs, int max_tokens_per_rank, int elements_per_token,
    int local_num_tokens, int rank_id, int ep_size, int stride_per_token)
{
    using InputT = std::conditional_t<LOW_PRECISION, __nv_fp8_e4m3, T>;

    int local_token_idx = blockIdx.x;
    int const size_per_token = elements_per_token * static_cast<int>(sizeof(InputT));

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

#if TLLM_MOE_A2A_COMPILE_SM90
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
            // Signal readiness to all active peers; skip dead ranks (their symmetric memory
            // is unreachable).
#pragma unroll 1 // No unroll
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, peer_rank))
                        continue;
                }
                uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
                publish_round_flag(flag_addr, expected_value);
#if ENABLE_DEBUG_PRINT
                printf("combine: +++Rank %d setting completion flag to %d for rank %d\n", rank_id, expected_value,
                    peer_rank);
#endif
            }
        }

        // Wait for all active peers to signal; skip dead ranks (otherwise we would spin
        // forever — this is the bug the rank-mask is here to prevent).
#pragma unroll 1 // No unroll
        for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
        {
            if constexpr (ENABLE_RANK_MASK)
            {
                if (!is_rank_active(ptrs.active_rank_mask, peer_rank))
                    continue;
            }
            if (!wait_round_flag(&ptrs.completion_flags[rank_id][peer_rank], expected_value, ptrs.timeout_cycles,
                    rank_id, peer_rank, "combine"))
            {
                return;
            }
        }
#if TLLM_MOE_A2A_COMPILE_SM90
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

    T* token_output = static_cast<T*>(ptrs.output) + local_token_idx * elements_per_token;
    vectorized_combine<TOP_K, T, InputT>(token_output, size_per_token, stride_per_token, ptrs);
#if TLLM_MOE_A2A_COMPILE_SM90
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;
    TLLM_CHECK(params.max_tokens_per_rank > 0);

    uint8_t* combine_input_base = params.combine_input_buffers[params.ep_rank];
    int const grid = std::max(params.prepare_num_tokens, 1);

    SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
        SWITCH_DTYPE(params.dtype, SrcT, {
            auto kernel_fn = moeA2APrepareCombineKernel<LOW_PRECISION, SrcT>;
            launchWithPdlWhenEnabled("moeA2APrepareCombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
                combine_input_base, params.source_payload, params.elements_per_token, params.ep_size,
                params.max_tokens_per_rank, params.flag_val, params.recv_counters, params.source_stride_per_token,
                params.workspace_stride_per_token, params.prepare_num_tokens);
        });
    });
}

// ============================================================================
// Combine Launch Function
// ============================================================================

void moe_a2a_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;

    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks);
    TLLM_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size);
    TLLM_CHECK(params.local_num_tokens >= 0);
    TLLM_CHECK(params.elements_per_token > 0);
    // The local rank must always be marked active in its own view of the mask;
    // otherwise the kernel itself would be running on a "dead" rank.
    if (params.enable_rank_mask)
    {
        TLLM_CHECK_WITH_INFO((params.active_rank_mask[params.ep_rank >> 6] >> (params.ep_rank & 63)) & 1ULL,
            "active_rank_mask must mark the local ep_rank (%d) as active", params.ep_rank);
    }

    // Configure kernel launch (one block per token).
    int grid = params.local_num_tokens;
    // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
    if (grid == 0)
    {
        grid = 1;
    }

    // Prepare kernel pointers struct for combine
    CombineKernelPointers kernel_ptrs = {}; // Zero-initialize
    kernel_ptrs.timeout_cycles = params.timeout_cycles;

    kernel_ptrs.output = params.output_data;

    // Each expert rank stores this origin rank's tokens in its own combine input region.
    int64_t const origin_offset
        = static_cast<int64_t>(params.ep_rank) * params.max_tokens_per_rank * params.workspace_stride_per_token;
    for (int rank = 0; rank < params.ep_size; rank++)
    {
        kernel_ptrs.source_buffers[rank] = params.combine_input_buffers[rank] + origin_offset;
    }

    // Copy completion flag pointers
    for (int i = 0; i < params.ep_size; i++)
    {
        kernel_ptrs.completion_flags[i] = params.completion_flags[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

    // Copy communication tracking pointers
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_target_indices = params.topk_target_indices;

    // Copy active-rank bitmask into the kernel pointers struct
    for (int w = 0; w < kRankMaskWords; ++w)
    {
        kernel_ptrs.active_rank_mask[w] = params.active_rank_mask[w];
    }

    SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
        SWITCH_DTYPE(params.dtype, T, {
            SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
                SWITCH_TOP_K(params.top_k, TOP_K, {
                    auto kernel_fn = moeA2ACombineKernel<T, TOP_K, LOW_PRECISION, ENABLE_RANK_MASK>;
                    launchWithPdlWhenEnabled("moeA2ACombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
                        kernel_ptrs, params.max_tokens_per_rank, params.elements_per_token, params.local_num_tokens,
                        params.ep_rank, params.ep_size, params.workspace_stride_per_token);
                });
            });
        });
    });
}

// Kernel to sanitize expert ids for invalid tokens
__global__ void moeA2ASanitizeExpertIdsKernel(int32_t* expert_ids_ptr, int32_t const* recv_counters_ptr,
    uint32_t const* flag_val, int ep_size, int max_tokens_per_rank, int top_k, int32_t invalid_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = ep_size * max_tokens_per_rank;
    if (tid >= total_tokens)
        return;

    int source_rank = tid / max_tokens_per_rank;
    int token_idx = tid % max_tokens_per_rank;

#if TLLM_MOE_A2A_COMPILE_SM90
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    uint32_t const parity = round_parity(*flag_val);
    if (token_idx >= recv_counters_ptr[parity * ep_size + source_rank])
    {
        int32_t* token_expert_ids = expert_ids_ptr + tid * top_k;
        // Vectorized invalid-id fill: 16B (int4) stores when top_k is a multiple of 4
        // (e.g. DSR1 top_k=8) -> 4x fewer stores; scalar fallback otherwise.
        if (top_k % 4 == 0)
        {
            int4 const invalid_ids = make_int4(invalid_id, invalid_id, invalid_id, invalid_id);
            int4* token_expert_ids_vec = reinterpret_cast<int4*>(token_expert_ids);
            for (int k = 0; k < top_k / 4; ++k)
            {
                token_expert_ids_vec[k] = invalid_ids;
            }
        }
        else
        {
            for (int k = 0; k < top_k; ++k)
            {
                token_expert_ids[k] = invalid_id;
            }
        }
    }
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters, uint32_t const* flag_val,
    int32_t invalid_id, int ep_size, int max_tokens_per_rank, int top_k, cudaStream_t stream)
{
    constexpr int kBlockSize = 256;
    int total_tokens = ep_size * max_tokens_per_rank;
    int grid = ceilDiv(total_tokens, kBlockSize);
    launchWithPdlWhenEnabled("moeA2ASanitizeExpertIdsKernel", moeA2ASanitizeExpertIdsKernel, grid, kBlockSize, 0,
        stream, expert_ids, recv_counters, flag_val, ep_size, max_tokens_per_rank, top_k, invalid_id);
}

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
