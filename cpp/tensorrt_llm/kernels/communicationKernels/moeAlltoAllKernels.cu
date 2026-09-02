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
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllCftSupport.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/cudaAsyncOps.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cerrno>
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

#define TLLM_MOE_A2A_COMPILE_CFT_DISPATCH TLLM_MOE_A2A_COMPILE_SM100

#if TLLM_MOE_A2A_COMPILE_SM90
#include <cuda/ptx>
#include <cuda_awbarrier_primitives.h>
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

// Resolve the completion-flag wait budget; see the header. Seconds are converted at
// an assumed 2 GHz SM clock, so they are nominal rather than wall-clock.
int64_t moeA2AGetTimeoutCycles(bool is_warmup)
{
    static constexpr int64_t kAssumedClockHz = 2000ll * 1000ll * 1000ll;
    static constexpr int64_t kDefaultTimeoutSec = 300;
    // Warmup contains one-time per-rank costs (JIT compilation, autotuning, module
    // loading) that can run for minutes and are not synchronized against this
    // collective, so it needs a larger budget than steady state.
    static constexpr int64_t kDefaultWarmupTimeoutSec = 1800;

    // Reject trailing garbage, out-of-range values and anything that would overflow
    // the cycle multiplication.
    auto const readEnv = [](char const* name, int64_t fallback) -> int64_t
    {
        static constexpr int64_t kMaxSec = 24 * 60 * 60; // 1 day; * 2e9 stays well inside int64
        char const* v = std::getenv(name);
        if (v == nullptr || *v == '\0')
        {
            return fallback;
        }
        errno = 0;
        char* end = nullptr;
        int64_t parsed = std::strtoll(v, &end, 10);
        bool const trailingGarbage = (end == v) || (*end != '\0');
        if (trailingGarbage || errno == ERANGE || parsed <= 0 || parsed > kMaxSec)
        {
            TLLM_LOG_WARNING("Ignoring invalid %s=\"%s\" (expected 1..%ld seconds); using %ld s", name, v,
                static_cast<long>(kMaxSec), static_cast<long>(fallback));
            return fallback;
        }
        return parsed;
    };

    static int64_t const sSteadySec = readEnv("TRTLLM_MOE_A2A_TIMEOUT_SEC", kDefaultTimeoutSec);
    static int64_t const sWarmupSec = readEnv("TRTLLM_MOE_A2A_WARMUP_TIMEOUT_SEC", kDefaultWarmupTimeoutSec);
    static bool const sLogged = []()
    {
        TLLM_LOG_INFO(
            "MoE all-to-all completion-flag budget: steady=%ld s, warmup=%ld s (nominal, at an "
            "assumed 2 GHz clock64 rate)",
            static_cast<long>(sSteadySec), static_cast<long>(sWarmupSec));
        return true;
    }();
    (void) sLogged;
    return (is_warmup ? sWarmupSec : sSteadySec) * kAssumedClockHz;
}

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
// `budget` is in clock64() cycles, resolved on the host by moeA2AGetTimeoutCycles().
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

template <int TOP_K, bool ENABLE_RANK_MASK>
__device__ __forceinline__ void route_dispatch_token(int32_t const* token_selected_experts,
    DispatchKernelPointers const& ptrs, int local_token_idx, int ep_size, int num_experts, int* topk_target_ranks,
    int* topk_send_indices)
{
    static_assert(TOP_K <= 32, "warp-parallel routing requires TOP_K <= warpSize");
    uint32_t const lane_mask = (TOP_K == 32) ? ~0U : ((1U << TOP_K) - 1U);
    int const k = threadIdx.x;

    int const ep_base = num_experts / ep_size;
    int const ep_remainder = num_experts - ep_base * ep_size;
    int const expert_id = token_selected_experts[local_token_idx * TOP_K + k];
    int const target_rank = compute_target_rank_id(expert_id, ep_base, ep_remainder);

    uint32_t const same_target = __match_any_sync(lane_mask, target_rank);
    bool keep = (__ffs(same_target) - 1) == k;
    if constexpr (ENABLE_RANK_MASK)
    {
        keep = keep && is_rank_active(ptrs.active_rank_mask, target_rank);
    }

    int const target_rank_to_store = keep ? target_rank : -1;
    int const send_index_to_store = keep ? atomicAdd(&ptrs.send_counters[target_rank], 1) : -1;

    ptrs.topk_target_ranks[local_token_idx * TOP_K + k] = target_rank_to_store;
    ptrs.topk_send_indices[local_token_idx * TOP_K + k] = send_index_to_store;
    topk_target_ranks[k] = target_rank_to_store;
    topk_send_indices[k] = send_index_to_store;
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

// Vectorized dispatch: load one vec from source and write to up to TOP_K destinations
template <int VEC_SIZE, int TOP_K>
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

    int const stride = blockDim.x * VEC_SIZE;
    for (int offset = threadIdx.x * VEC_SIZE; offset < bytes_per_token; offset += stride)
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

template <int TOP_K>
__device__ void vectorized_dispatch(uint8_t const* src_ptr, int bytes_per_token, int rank_id, int max_tokens_per_rank,
    int payload_idx, DispatchKernelPointers const& ptrs, int const* topk_target_ranks, int const* topk_send_indices)
{
    if (bytes_per_token % 16 == 0)
    {
        vectorized_dispatch_impl<16, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 8 == 0)
    {
        vectorized_dispatch_impl<8, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 4 == 0)
    {
        vectorized_dispatch_impl<4, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            topk_target_ranks, topk_send_indices);
    }
    else if (bytes_per_token % 2 == 0)
    {
        vectorized_dispatch_impl<2, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            topk_target_ranks, topk_send_indices);
    }
    else
    {
        vectorized_dispatch_impl<1, TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
            topk_target_ranks, topk_send_indices);
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
    // NOTE: LE-backed counters use cumulative baselines and are deliberately not zeroed
    // here, so that the kernel never issues SM stores to LE-backed memory.
}

// ============================================================================
// Dispatch Kernels
// ============================================================================

template <int TOP_K, bool ENABLE_EPLB, bool ENABLE_RANK_MASK>
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

        // One block per token: a single shared-memory tile is reused by the entire CTA.
        extern __shared__ int smem[];
        int* smem_topk_target_ranks = smem;
        int* smem_topk_send_indices = smem + TOP_K;

#if TLLM_MOE_A2A_COMPILE_SM90
        cudaGridDependencySynchronize();
#endif
        if (thread_idx < TOP_K)
        {
            route_dispatch_token<TOP_K, ENABLE_RANK_MASK>(token_selected_experts, ptrs, local_token_idx, ep_size,
                num_experts, smem_topk_target_ranks, smem_topk_send_indices);
        }
        // Sync before dispatching data
        __syncthreads();

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

            vectorized_dispatch<TOP_K>(src_ptr, bytes_per_token, rank_id, max_tokens_per_rank, payload_idx, ptrs,
                topk_target_ranks, topk_send_indices);
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
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));

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
                } while (!flag_set && !check_timeout(s, ptrs.timeout_cycles));

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

// ============================================================================
// CFT Counted Write Dispatch Kernel (one block per token, sm_100+)
//
//
// Data flow:
//   1. Route tokens to target ranks (same as fence-based)
//   2. Stage each token to shared memory, fabric.try_put.counted to peer LEs
//   3. Last block sends recv_counters through symmetric memory using the current round parity
//   4. Poll metadata + data counters from all peers (no fence.sys needed)
// ============================================================================
#if TLLM_MOE_A2A_COMPILE_CFT_DISPATCH
__device__ __forceinline__ void cft_barrier_wait_parity(__mbarrier_t* barrier, int parity)
{
    while (!::cuda::ptx::mbarrier_try_wait_parity(::cuda::ptx::sem_relaxed, ::cuda::ptx::scope_cta,
        reinterpret_cast<::cuda::std::uint64_t*>(barrier), static_cast<::cuda::std::uint32_t>(parity)))
    {
    }
}

__device__ __forceinline__ void cft_fabric_try_put_counted(
    uint32_t le, uint64_t offset, uint64_t counter_offset, void* src, int size, __mbarrier_t* mbar)
{
#if !TLLM_CFT_HAS_CUDA_13_4_SUPPORT
    (void) le;
    (void) offset;
    (void) counter_offset;
    (void) src;
    (void) size;
    (void) mbar;
    asm volatile("trap;" ::: "memory");
#else
    uint32_t src_smem = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t mbar_smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric"
        ".counted::bytes.relaxed.sys.b128 "
        "[%0, %1, %2], [%3], %4, [%5];"
        :
        : "r"(le), "l"(offset), "l"(counter_offset), "r"(src_smem), "r"(size), "r"(mbar_smem)
        : "memory");
#endif
}

__device__ __forceinline__ void cft_fabric_submit()
{
#if !TLLM_CFT_HAS_CUDA_13_4_SUPPORT
    asm volatile("trap;" ::: "memory");
#else
    asm volatile("fabric.submit;" ::: "memory");
#endif
}

__device__ __forceinline__ void cft_fabric_wait_reads()
{
#if !TLLM_CFT_HAS_CUDA_13_4_SUPPORT
    asm volatile("trap;" ::: "memory");
#else
    asm volatile("fabric.wait.sync_restrict::reads;" ::: "memory");
#endif
}

template <int TOP_K>
__device__ __forceinline__ void store_invalid_expert_ids(int32_t* expert_ids, int32_t invalid_expert_id)
{
    if constexpr (TOP_K % 4 == 0)
    {
        int4 const invalid_ids = make_int4(invalid_expert_id, invalid_expert_id, invalid_expert_id, invalid_expert_id);
        int4* expert_ids_vec = reinterpret_cast<int4*>(expert_ids);
#pragma unroll
        for (int k = 0; k < TOP_K / 4; ++k)
        {
            expert_ids_vec[k] = invalid_ids;
        }
    }
    else
    {
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            expert_ids[k] = invalid_expert_id;
        }
    }
}

template <int TOP_K, bool ENABLE_EPLB, bool ENABLE_RANK_MASK>
__global__ void moeA2ADispatchCountedWriteKernel(int32_t const* token_selected_experts,
    DispatchKernelPointers const ptrs, int num_payloads, int max_tokens_per_rank, int local_num_tokens, int rank_id,
    int ep_size, int num_experts, int eplb_stats_num_experts)
{
    int local_token_idx = blockIdx.x;
    uint32_t parity = 0;

    if (local_num_tokens == 0)
    {
        if (local_token_idx > 0)
            return;
        cudaGridDependencySynchronize();
        parity = round_parity(*ptrs.flag_val);
    }
    else
    {
        if (local_token_idx >= local_num_tokens)
            return;

        extern __shared__ int smem[];
        int* smem_topk_target_ranks = smem;
        int* smem_topk_send_indices = smem + TOP_K;

        // CFT smem layout (disjoint regions, kept stable across phases):
        //   [0 .. kRoutingBytes)                                       routing indices (above)
        //   [kRoutingBytes .. kRoutingBytes+kCftMbarrierSlotBytes)     mbarrier (tma_bar)
        //   [kRoutingBytes+kCftMbarrierSlotBytes ..)                   TMA staging buffer
        // Staging is populated by cp.async.bulk issued at kernel entry; the load
        // runs on the TMA engine in parallel with routing + self-send.
        constexpr int kRoutingBytes = 2 * TOP_K * static_cast<int>(sizeof(int));
        uint8_t* smem_bytes = reinterpret_cast<uint8_t*>(smem);
        __mbarrier_t* tma_bar = reinterpret_cast<__mbarrier_t*>(smem_bytes + kRoutingBytes);
        uint8_t* smem_staging = smem_bytes + kRoutingBytes + kCftMbarrierSlotBytes;

        cudaGridDependencySynchronize();
        parity = round_parity(*ptrs.flag_val);

        // ---- Payload staging: overlaps with routing ----
        int total_staged_bytes = 0;
        for (int p = 0; p < num_payloads; p++)
        {
            total_staged_bytes += ptrs.payload_bytes_per_token[p];
        }

        if (threadIdx.x == 0)
        {
            ::cuda::ptx::mbarrier_init(reinterpret_cast<::cuda::std::uint64_t*>(tma_bar), 1);
            int smem_offset = 0;
            for (int p = 0; p < num_payloads; p++)
            {
                uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[p]);
                int bytes_per_token = ptrs.payload_bytes_per_token[p];
                uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;
                cp_async_bulk_g2s(
                    smem_staging + smem_offset, src_ptr, bytes_per_token, reinterpret_cast<uint64_t*>(tma_bar));
                smem_offset += bytes_per_token;
            }
            // Sets expected tx count and consumes the single arrival.
            mbarrier_arrive_expect_tx(reinterpret_cast<uint64_t*>(tma_bar), static_cast<uint32_t>(total_staged_bytes));
        }

        // ---- Routing: map tokens to target ranks ----
        if (threadIdx.x < TOP_K)
        {
            route_dispatch_token<TOP_K, ENABLE_RANK_MASK>(token_selected_experts, ptrs, local_token_idx, ep_size,
                num_experts, smem_topk_target_ranks, smem_topk_send_indices);
        }
        __syncthreads();

        int topk_target_ranks[TOP_K];
        int topk_send_indices[TOP_K];
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            topk_target_ranks[k] = smem_topk_target_ranks[k];
            topk_send_indices[k] = smem_topk_send_indices[k];
        }

        // ---- Data dispatch: self via TMA s2g, remote via fabric.try_put.counted ----
        // Both are issued by thread 0 fire-and-forget. They run in parallel on
        // different HW units (TMA engine for s2g, fabric engine for puts), with a
        // single combined wait phase at the end.
        //
        // Self-send needs smem_staging populated, so it must come AFTER the TMA g2s
        // wait. Remote-send also reads smem_staging — both share the same source.
        bool has_remote = false;
        bool has_self = false;
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            if (topk_send_indices[k] < 0)
                continue;
            if (topk_target_ranks[k] == rank_id)
                has_self = true;
            else
                has_remote = true;
        }

        // Wait for all staged payloads before self/fabric sends consume smem_staging.
        if (threadIdx.x == 0)
        {
            cft_barrier_wait_parity(tma_bar, 0);
        }
        __syncthreads();

        // Use a different warp to issue s2g if there are at least 2 warps, so the s2g and the
        // fabric puts overlap instead of serializing on one thread.
        int const s2g_issuer = blockDim.x >= 64 ? 32 : 0;

        // Issue self-sends via TMA s2g (smem→gmem).
        if (threadIdx.x == s2g_issuer && has_self)
        {
            int smem_offset = 0;
            for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
            {
                int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
#pragma unroll
                for (int k = 0; k < TOP_K; ++k)
                {
                    int dst_idx_k = topk_send_indices[k];
                    int target_rank_k = topk_target_ranks[k];
                    if (dst_idx_k < 0 || target_rank_k != rank_id)
                        continue;
                    uint8_t* dst = static_cast<uint8_t*>(ptrs.recv_buffers[rank_id][payload_idx])
                        + (static_cast<size_t>(rank_id) * max_tokens_per_rank + dst_idx_k)
                            * static_cast<size_t>(bytes_per_token);
                    cp_async_bulk_s2g(dst, smem_staging + smem_offset, bytes_per_token);
                }
                smem_offset += bytes_per_token;
            }
            cp_async_bulk_commit_group();
        }

        // Issue remote-sends via fabric.try_put.counted — runs in parallel with the s2g above.
        if (threadIdx.x == 0 && has_remote)
        {
            // Re-arm tma_bar as the put report target. Past the __syncthreads() above this thread
            // is its only user, so the s2g issuer does not have to wait for it.
            ::cuda::ptx::mbarrier_init(reinterpret_cast<::cuda::std::uint64_t*>(tma_bar), 1);
            uint64_t counter_offset = ptrs.le_counter_base + static_cast<uint64_t>(rank_id) * kCftCounterStride;
            int smem_offset = 0;
            for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
            {
                int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];
#pragma unroll
                for (int k = 0; k < TOP_K; ++k)
                {
                    int dst_idx_k = topk_send_indices[k];
                    int target_rank_k = topk_target_ranks[k];
                    if (dst_idx_k < 0 || target_rank_k == rank_id)
                        continue;
                    uint64_t base_le_offset = ptrs.le_payload_offsets[payload_idx]
                        + (static_cast<uint64_t>(rank_id) * max_tokens_per_rank + dst_idx_k)
                            * static_cast<uint64_t>(bytes_per_token);
                    uint32_t le_id = ptrs.peer_le_ids[target_rank_k];
                    cft_fabric_try_put_counted(
                        le_id, base_le_offset, counter_offset, smem_staging + smem_offset, bytes_per_token, tma_bar);
                }
                smem_offset += bytes_per_token;
            }
            cft_fabric_submit();
        }

        // Wait for both s2g and fabric puts to complete (independent HW, in parallel).
        if (threadIdx.x == s2g_issuer && has_self)
            cp_async_bulk_wait_group<0>();

        if (threadIdx.x == 0 && has_remote)
            cft_fabric_wait_reads();
        __syncthreads();
    }

    cudaTriggerProgrammaticLaunchCompletion();

    // ---- is_last_token: metadata send + counter polling ----
    __shared__ int is_last_token_cta;
    if (threadIdx.x == 0)
    {
        is_last_token_cta = 0;
    }
    __syncthreads();

    bool is_first_warp = threadIdx.x / warpSize == 0;
    int lane_id = threadIdx.x % warpSize;
    if (is_first_warp)
    {
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
        if (lane_id == 0)
        {
            is_last_token_cta = static_cast<int>(is_last_token);
        }

        if (is_last_token)
        {
            if constexpr (ENABLE_EPLB)
            {
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
                        target_stats[rank_id * eplb_stats_num_experts + expert_id] = ptrs.eplb_local_stats[expert_id];
                }
            }

// Send recv_counters to active peers via direct MNNVL write.
#pragma unroll 1
            for (int target_rank = lane_id; target_rank < ep_size; target_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, target_rank))
                        continue;
                }
                int send_count = ptrs.send_counters[target_rank];
                int* slot = ptrs.recv_counters[target_rank] + static_cast<int>(parity) * ep_size + rank_id;
                asm volatile("st.relaxed.sys.b32 [%0], %1;" ::"l"(slot), "r"(send_count) : "memory");
            }
            __syncwarp();

#if !DISABLE_SYNC_FOR_PROFILING
            // Poll recv_counters from all peers (including self). The whole CTA
            // cooperatively sanitizes invalid expert-id payload slots after this.
#pragma unroll 1
            for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
            {
                if constexpr (ENABLE_RANK_MASK)
                {
                    if (!is_rank_active(ptrs.active_rank_mask, peer_rank))
                    {
                        ptrs.recv_counters[rank_id][static_cast<int>(parity) * ep_size + peer_rank] = 0;
                        continue;
                    }
                }
                int32_t* recvCountPtr = ptrs.recv_counters[rank_id] + static_cast<int>(parity) * ep_size + peer_rank;
                auto s = clock64();
                int recv_count;
                do
                {
                    asm volatile("ld.relaxed.sys.b32 %0, [%1];" : "=r"(recv_count) : "l"(recvCountPtr));
                    if (check_timeout(s, ptrs.timeout_cycles))
                    {
                        printf(
                            "dispatch: ---Rank %d timed out recv_counters from rank %d"
                            " parity=%u observed=%d\n",
                            rank_id, peer_rank, parity, recv_count);
                        asm volatile("trap;");
                    }
                } while (recv_count == -1);
            }
#endif // !DISABLE_SYNC_FOR_PROFILING
        }
    }

    __syncthreads();

#if !DISABLE_SYNC_FOR_PROFILING
    if (is_last_token_cta && ptrs.sanitize_expert_ids)
    {
        int32_t* expert_ids = static_cast<int32_t*>(ptrs.recv_buffers[rank_id][ptrs.expert_id_payload_index]);
        size_t const sanitize_tokens = static_cast<size_t>(ep_size) * static_cast<size_t>(max_tokens_per_rank);
        for (size_t linear_idx = threadIdx.x; linear_idx < sanitize_tokens; linear_idx += blockDim.x)
        {
            int const token_idx = static_cast<int>(linear_idx % max_tokens_per_rank);
            int const peer_rank = static_cast<int>(linear_idx / max_tokens_per_rank);
            int const recv_count = ptrs.recv_counters[rank_id][static_cast<int>(parity) * ep_size + peer_rank];
            if (token_idx >= recv_count)
            {
                int32_t* token_expert_ids
                    = expert_ids + (static_cast<size_t>(peer_rank) * max_tokens_per_rank + token_idx) * TOP_K;
                store_invalid_expert_ids<TOP_K>(token_expert_ids, ptrs.invalid_expert_id);
            }
        }
    }

    if (is_first_warp && is_last_token_cta)
    {
        // Compute expected data bytes for counter polling.
        // All payloads go through fabric.try_put.counted (16B alignment
        // guaranteed by the dispatch op; non-aligned falls back to fence path).
        int counted_payload_bytes_per_token = 0;
        for (int p = 0; p < num_payloads; p++)
        {
            counted_payload_bytes_per_token += ptrs.payload_bytes_per_token[p];
        }

        // Poll data counters from all active peers. Self data is placed locally, not via fabric.
#pragma unroll 1
        for (int peer_rank = lane_id; peer_rank < ep_size; peer_rank += warpSize)
        {
            if (peer_rank == rank_id)
                continue;
            if constexpr (ENABLE_RANK_MASK)
            {
                if (!is_rank_active(ptrs.active_rank_mask, peer_rank))
                    continue;
            }
            int tokens_from_peer = ptrs.recv_counters[rank_id][static_cast<int>(parity) * ep_size + peer_rank];
            uint64_t expected_data_bytes
                = static_cast<uint64_t>(tokens_from_peer) * static_cast<uint64_t>(counted_payload_bytes_per_token);

            if (expected_data_bytes > 0)
            {
                uint64_t* dataCounterPtr = &ptrs.le_dispatch_counters[rank_id][peer_rank * kCftCounterStrideU64];
                uint64_t data_base = ptrs.dispatch_counter_baseline[peer_rank];
                uint64_t data_target = data_base + expected_data_bytes;
                bool data_arrived = false;
                uint64_t current_data_counter = 0;
                auto s = clock64();
                do
                {
                    asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(current_data_counter) : "l"(dataCounterPtr));
                    data_arrived = current_data_counter >= data_target;
                } while (!data_arrived && !check_timeout(s, ptrs.timeout_cycles));

                if (__builtin_expect(!data_arrived, 0))
                {
                    printf("dispatch(counted): ---Rank %d timed out data from rank %d counter=%llu expected=%llu\n",
                        rank_id, peer_rank, (unsigned long long) current_data_counter,
                        (unsigned long long) data_target);
                    asm volatile("trap;");
                }
                ptrs.dispatch_counter_baseline[peer_rank] = data_target;
            }
        }
    }
#endif // !DISABLE_SYNC_FOR_PROFILING
}
#else  // TLLM_MOE_A2A_COMPILE_CFT_DISPATCH
template <int TOP_K, bool ENABLE_EPLB, bool ENABLE_RANK_MASK>
__global__ void moeA2ADispatchCountedWriteKernel(int32_t const* token_selected_experts,
    DispatchKernelPointers const ptrs, int num_payloads, int max_tokens_per_rank, int local_num_tokens, int rank_id,
    int ep_size, int num_experts, int eplb_stats_num_experts)
{
    (void) token_selected_experts;
    (void) ptrs;
    (void) num_payloads;
    (void) max_tokens_per_rank;
    (void) local_num_tokens;
    (void) rank_id;
    (void) ep_size;
    (void) num_experts;
    (void) eplb_stats_num_experts;
    asm volatile("trap;" ::: "memory");
}
#endif // TLLM_MOE_A2A_COMPILE_CFT_DISPATCH

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params)
{
    // NOTE: LE counters are NOT zeroed between iterations. They grow monotonically.
    // Cumulative baselines in regular device memory track the expected value.

    launchWithPdlWhenEnabled("moeA2APrepareDispatchKernel", moeA2APrepareDispatchKernel, 1, params.ep_size, 0,
        params.stream, params.send_counters, params.recv_counters[params.ep_rank], params.local_token_counter,
        params.ep_size, params.flag_val);
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params)
{
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
        kernel_ptrs.le_dispatch_counters[i] = params.le_dispatch_counters[i];
    }
    kernel_ptrs.flag_val = params.flag_val;

    // Copy communication tracking pointers
    kernel_ptrs.send_counters = params.send_counters;
    kernel_ptrs.local_token_counter = params.local_token_counter;
    kernel_ptrs.topk_target_ranks = params.topk_target_ranks;
    kernel_ptrs.topk_send_indices = params.topk_send_indices;
    kernel_ptrs.eplb_local_stats = params.eplb_local_stats;

    // CFT handle-based counted writes fields
    if (params.use_cft_counted_writes)
    {
        for (int i = 0; i < params.ep_size; i++)
        {
            kernel_ptrs.peer_le_ids[i] = params.cft_peer_le_ids[i];
        }
        for (int i = 0; i < params.num_payloads; i++)
        {
            kernel_ptrs.le_payload_offsets[i] = params.cft_le_payload_offsets[i];
        }
        kernel_ptrs.le_counter_base = params.cft_le_counter_base;
        kernel_ptrs.dispatch_counter_baseline = params.cft_dispatch_counter_baseline;
        kernel_ptrs.sanitize_expert_ids = params.sanitize_expert_ids;
        kernel_ptrs.expert_id_payload_index = params.expert_id_payload_index;
        kernel_ptrs.invalid_expert_id = params.invalid_expert_id;
    }

    // Copy active-rank bitmask into the kernel pointers struct
    for (int w = 0; w < kRankMaskWords; ++w)
    {
        kernel_ptrs.active_rank_mask[w] = params.active_rank_mask[w];
    }

    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ADispatchBlockSize();

    int grid_size = params.local_num_tokens;
    if (grid_size == 0)
    {
        grid_size = 1;
    }

    int const routing_bytes = 2 * params.top_k * static_cast<int>(sizeof(int));
    int shared_bytes = routing_bytes;

    if (params.use_cft_counted_writes)
    {
        // CFT path: routing indices + mbarrier (64B) + contiguous staging for all payloads.
        int total_payload_bytes = 0;
        for (int i = 0; i < params.num_payloads; i++)
        {
            total_payload_bytes += params.payloads[i].element_size * params.payloads[i].elements_per_token;
        }
        int const data_cft = routing_bytes + kCftMbarrierSlotBytes + total_payload_bytes;
        shared_bytes = shared_bytes > data_cft ? shared_bytes : data_cft;
    }

    if (params.use_cft_counted_writes)
    {
        // Per-block dynamic smem cap is 48KB by default on sm_90+; opt-in to use up to the
        // architectural max, as the CFT combine push launch does.
        if (shared_bytes > kDefaultDynamicSmemBytes)
        {
            int maxOptinBytes = 0;
            int deviceIdx = 0;
            TLLM_CUDA_CHECK(cudaGetDevice(&deviceIdx));
            TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxOptinBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceIdx));
            TLLM_CHECK_WITH_INFO(shared_bytes <= maxOptinBytes,
                "MoE all-to-all CFT dispatch needs %d bytes of shared memory per block, above the %d byte maximum on "
                "this device. Reduce the per-token payload size or disable CFT counted writes.",
                shared_bytes, maxOptinBytes);
        }

        SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK,
            {SWITCH_BOOL(params.enable_eplb, EPLB_STATS, SWITCH_TOP_K(params.top_k, TOP_K, {
                auto kernel_fn = moeA2ADispatchCountedWriteKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK>;
                if (shared_bytes > kDefaultDynamicSmemBytes)
                {
                    TLLM_CUDA_CHECK(
                        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
                }
                launchWithPdlWhenEnabled("moeA2ADispatchCountedWriteKernel", kernel_fn, grid_size, kBlockSize,
                    shared_bytes, params.stream, params.token_selected_experts, kernel_ptrs, params.num_payloads,
                    params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size,
                    params.num_experts, params.eplb_stats_num_experts);
            }))})
    }
    else
    {
        SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK,
            {SWITCH_BOOL(params.enable_eplb, EPLB_STATS, SWITCH_TOP_K(params.top_k, TOP_K, {
                auto kernel_fn = moeA2ADispatchKernel<TOP_K, EPLB_STATS, ENABLE_RANK_MASK>;
                launchWithPdlWhenEnabled("moeA2ADispatchKernel", kernel_fn, grid_size, kBlockSize, shared_bytes,
                    params.stream, params.token_selected_experts, kernel_ptrs, params.num_payloads,
                    params.max_tokens_per_rank, params.local_num_tokens, params.ep_rank, params.ep_size,
                    params.num_experts, params.eplb_stats_num_experts);
            }))})
    }
}

// ============================================================================
// Combine kernels
// ============================================================================

// Accumulate across all valid ranks into float32 registers, then store as OutputT.
// InputT is the wire element type in the receive buffer.
//
// Unified path: load VEC_SIZE bytes, reinterpret as InputT[elems_per_vec], accumulate as float32,
// store as OutputT. Works for same-type and FP8-to-payload-type accumulation.
// sizeof(InputT) must divide VEC_SIZE.
template <int VEC_SIZE, int TOP_K, typename OutputT, typename InputT = OutputT>
__device__ void vectorized_combine_impl(OutputT* dst_typed_base, int size_per_token, int stride_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    using flashinfer::vec_t;

    // elems_per_vec is the number of InputT elements per VEC_SIZE-byte load.
    constexpr int elems_per_vec = VEC_SIZE / static_cast<int>(sizeof(InputT));

    int const stride = blockDim.x * VEC_SIZE;
    int const local_token_idx = blockIdx.x;

    // offset is a byte offset into the recv buffer, stepping by VEC_SIZE bytes.
    for (int offset = threadIdx.x * VEC_SIZE; offset < size_per_token; offset += stride)
    {
        // Per-k vec_t<float, elems_per_vec> accumulators, zero-initialised via fill().
        // Using vec_t enables cast_store() for the output, emitting a vectorized int4 write.
        vec_t<float, elems_per_vec> acc[TOP_K];

        // Pass 1: issue all TOP_K loads back-to-back without any type conversion.
        // Raw InputT bytes are loaded directly into acc[k]'s register storage, reinterpreted as
        // vec_t<InputT, elems_per_vec> (VEC_SIZE bytes, fitting in the low end of acc[k]'s
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

            // Every contribution uses the same compact receive-buffer layout.
            uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
            size_t base_source_rank = static_cast<size_t>(rank_id) * static_cast<size_t>(max_tokens_per_rank)
                + static_cast<size_t>(dst_idx);
            size_t base_token = base_source_rank * static_cast<size_t>(stride_per_token);

            reinterpret_cast<vec_t<InputT, elems_per_vec>&>(acc[k]).load(
                reinterpret_cast<InputT const*>(recv_buffer + base_token + offset));
        }

        // Pass 2: in-place cast InputT to float, iterating j in descending order.
        // float[j] occupies bytes [j*4, j*4+3]; InputT[j] occupies
        // [j*sizeof(InputT), ...). For narrow inputs, high-j float writes land above all
        // remaining InputT bytes, so descending order is write-after-read safe.
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            int target_rank = ptrs.topk_target_ranks[local_token_idx * TOP_K + k];
            int dst_idx = ptrs.topk_send_indices[local_token_idx * TOP_K + k];
            if (dst_idx < 0)
            {
                continue; // acc[k] already holds 0.0f from fill() above
            }
#pragma unroll
            for (int j = elems_per_vec - 1; j >= 0; --j)
                acc[k][j] = static_cast<float>(reinterpret_cast<InputT const*>(&acc[k])[j]);
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

        // cast_store converts each accumulated element to OutputT before the vectorized store.
        acc[0].cast_store(dst_typed_base + offset / static_cast<int>(sizeof(InputT)));
    }
}

// Wrapper that selects vector width based on size_per_token alignment.
// stride_per_token: byte distance between tokens in the recv buffer (may differ from
// size_per_token when low-precision in-place data retains its payload-dtype workspace stride).
// InputT is the input element type in the receive buffer.
template <int TOP_K, typename OutputT, typename InputT = OutputT>
__device__ void vectorized_combine(OutputT* dst_typed_base, int size_per_token, int stride_per_token, int rank_id,
    int max_tokens_per_rank, CombineKernelPointers const& ptrs)
{
    // Each branch is guarded by if constexpr (sizeof(InputT) <= VEC_SIZE) so that the compiler
    // never instantiates vectorized_combine_impl with elems_per_vec=0.
    // Branches where VEC_SIZE < sizeof(InputT) are unreachable at runtime because size_per_token
    // is always a multiple of sizeof(InputT), so a larger alignment branch is taken first.
    if (size_per_token % 16 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InputT)) <= 16)
            vectorized_combine_impl<16, TOP_K, OutputT, InputT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 8 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InputT)) <= 8)
            vectorized_combine_impl<8, TOP_K, OutputT, InputT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 4 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InputT)) <= 4)
            vectorized_combine_impl<4, TOP_K, OutputT, InputT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else if (size_per_token % 2 == 0)
    {
        if constexpr (static_cast<int>(sizeof(InputT)) <= 2)
            vectorized_combine_impl<2, TOP_K, OutputT, InputT>(
                dst_typed_base, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
    }
    else
    {
        if constexpr (static_cast<int>(sizeof(InputT)) <= 1)
            vectorized_combine_impl<1, TOP_K, OutputT, InputT>(
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
#if TLLM_MOE_A2A_COMPILE_SM100
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

// LOW_PRECISION=false: vectorized byte-copy (SrcT = payload dtype).
// LOW_PRECISION=true:  vectorized SrcT→FP8 quantization via vectorized_quant<SrcT, fp8_e4m3>.
template <bool LOW_PRECISION, typename SrcT>
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, void const* source_payload,
    int elements_per_token, int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr, int const* recv_counters,
    int source_stride_per_token, int workspace_stride_per_token, int prepare_first_token, int prepare_num_tokens,
    uint8_t* region_c_base, int ep_rank)
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
    // NOTE: LE counters are NOT zeroed. They grow monotonically with cumulative baselines.

    if (blockIdx.x >= prepare_num_tokens)
        return;
    int const global_token_idx = prepare_first_token + blockIdx.x;

    // Map global_token_idx to (rank_idx, local_token_idx)
    int rank_idx = global_token_idx / max_tokens_per_rank;
    int local_token_idx = global_token_idx % max_tokens_per_rank;

    // Skip invalid tokens beyond per-rank recv count
    if (local_token_idx >= round_recv_counters[rank_idx])
        return;

    // CFT combine stages local tokens compactly into the dedicated receive region. This keeps
    // local and peer contributions in one uniform layout without an in-place write-after-read hazard.
    bool const stage_self = (region_c_base != nullptr && rank_idx == ep_rank);

    size_t const source_offset = static_cast<size_t>(global_token_idx) * source_stride_per_token;
    size_t const workspace_offset = static_cast<size_t>(global_token_idx) * workspace_stride_per_token;
    // The receive-region self slot is compact: FP8 uses one byte per element; same-type uses sizeof(SrcT).
    size_t const self_slot_offset = static_cast<size_t>(global_token_idx) * elements_per_token
        * static_cast<size_t>(LOW_PRECISION ? 1 : static_cast<int>(sizeof(SrcT)));

    if constexpr (LOW_PRECISION)
    {
        SrcT const* src_ptr
            = reinterpret_cast<SrcT const*>(static_cast<uint8_t const*>(source_payload) + source_offset);
        // Self -> region C (compact, separate buffer). Peer -> in-place workspace (push reads it).
        __nv_fp8_e4m3* dst_ptr = stage_self ? reinterpret_cast<__nv_fp8_e4m3*>(region_c_base + self_slot_offset)
                                            : reinterpret_cast<__nv_fp8_e4m3*>(recv_buffer_bytes + workspace_offset);
        vectorized_quant<SrcT, __nv_fp8_e4m3>(dst_ptr, src_ptr, elements_per_token);
    }
    else
    {
        // Same-type byte copy. CFT self tokens use the receive region; fence combine uses the workspace.
        uint8_t const* src = static_cast<uint8_t const*>(source_payload) + source_offset;
        uint8_t* dst = stage_self ? (region_c_base + self_slot_offset) : (recv_buffer_bytes + workspace_offset);
        vectorized_copy(dst, src, elements_per_token * static_cast<int>(sizeof(SrcT)));
    }
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T, int TOP_K, bool LOW_PRECISION, bool ENABLE_RANK_MASK>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs, // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size,
    int stride_per_token)
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
                asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(expected_value));
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
            } while (!flag_set && !check_timeout(s, ptrs.timeout_cycles));

            if (__builtin_expect(!flag_set, 0))
            {
                printf("combine: ---Rank %d timed out waiting for completion flag from rank %d\n", rank_id, peer_rank);
                asm volatile("trap;");
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

    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;
    vectorized_combine<TOP_K, T, InputT>(
        token_output, size_per_token, stride_per_token, rank_id, max_tokens_per_rank, ptrs);
#if TLLM_MOE_A2A_COMPILE_SM90
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// CFT counted-write combine path
// ----------------------------------------------------------------------------
// Gated by MoeA2ACombineParams::use_cft_for_combine. Peers push expert outputs back via
// fabric.try_put.counted; the reduce kernel polls per-token counters and reuses the shared
// vectorized_combine reduction. One block per token, matching the fence combine geometry.
// ============================================================================

// After expert compute, each processing rank pushes results back to the originating rank's LE
// combine region via fabric.try_put.counted. Tokens are fanned across all warps of all blocks for
// a given source rank; each warp stages one token to smem (TMA) and pushes it independently.
static constexpr int kCombinePushWarpsPerBlock = 4;

template <bool ENABLE_RANK_MASK>
__global__ void moeA2ACftCombinePushKernel(
    uint8_t const* local_payload, // Expert output (combine payload or dispatch recv_buffer)
    int const* recv_counters,     // [2, ep_size] tokens received from each source rank
    uint32_t const* flag_val,
    CftPeerLeIds peer_le_ids,     // LE IDs passed by value (no device pointer needed)
    int rank_id, int ep_size, int max_tokens_per_rank, int bytes_per_token, uint64_t combine_payload_base,
    uint64_t combine_counter_base, int combine_counter_ep_stride, int local_stride_per_token)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Wait for prepareCombine to finish writing the workspace we read from, then immediately
    // signal the next kernel (combineCountedWrite) that it can start. combineCountedWrite
    // polls for incoming counter writes from peers — it touches disjoint memory from our
    // local pushes, so it can run concurrently with the rest of this kernel.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    int source_rank = blockIdx.x;
    if (source_rank == rank_id)
        return;
    if constexpr (ENABLE_RANK_MASK)
    {
        if (!is_rank_active(peer_le_ids.active_rank_mask, source_rank))
            return;
    }

    uint32_t const parity = round_parity(*flag_val);
    int num_tokens = recv_counters[parity * ep_size + source_rank];
    // Nothing to push (0 tokens) or an out-of-range count (corrupt recv_counters): skip.
    if (num_tokens <= 0 || num_tokens > max_tokens_per_rank)
        return;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id != 0)
        return; // only lane 0 of each warp drives TMA + fabric

    // Per-warp smem layout: [mbarrier slot (kCftMbarrierSlotBytes) | staging (bytes_per_token)]
    // repeated kCombinePushWarpsPerBlock times. Each warp drives its own slot independently
    // and tracks its own put completion via put_bar — no __syncthreads, no CTA-scope drain
    // serialization. submit + wait_reads are issued per-warp; the fabric engine pipelines
    // drains across warps.
    extern __shared__ uint8_t smem_push[];
    int per_warp_bytes = kCftMbarrierSlotBytes + bytes_per_token;
    uint8_t* warp_smem = smem_push + warp_id * per_warp_bytes;
    __mbarrier_t* tma_bar = reinterpret_cast<__mbarrier_t*>(warp_smem);
    __mbarrier_t* put_bar = tma_bar + 1;
    uint8_t* staging = warp_smem + kCftMbarrierSlotBytes;

    uint32_t le_id = peer_le_ids.ids[source_rank]; // push back to source rank's LE

    // Tokens are fanned across all warps of all blocks for this source rank; blockIdx.y selects
    // the token-chunk. Each warp uses its own per-warp smem slot.
    int const warps_per_block = blockDim.x >> 5;
    int const warps_per_rank = gridDim.y * warps_per_block;
    int const global_warp = blockIdx.y * warps_per_block + warp_id;
    __mbarrier_init(tma_bar, 1); // init once; reused across tokens via phase parity
    __mbarrier_init(put_bar, 1);
    int tma_phase = 0;
    for (int t = global_warp; t < num_tokens; t += warps_per_rank)
    {
        uint8_t const* src
            = local_payload + (static_cast<size_t>(source_rank) * max_tokens_per_rank + t) * local_stride_per_token;

        // TMA g2s: stage one token's payload into our private smem slot.
        cp_async_bulk_g2s(staging, src, bytes_per_token, reinterpret_cast<uint64_t*>(tma_bar));
        mbarrier_arrive_expect_tx(reinterpret_cast<uint64_t*>(tma_bar), static_cast<uint32_t>(bytes_per_token));
        cft_barrier_wait_parity(tma_bar, tma_phase & 1);
        tma_phase++;

        // Issue the fabric put with put_bar tracking; arm put_bar to expect bytes_per_token
        // bytes of fabric.report::fabric.counted::bytes events.
        uint64_t data_offset
            = combine_payload_base + (static_cast<uint64_t>(rank_id) * max_tokens_per_rank + t) * bytes_per_token;
        uint64_t counter_offset = combine_counter_base
            + (static_cast<uint64_t>(rank_id) * combine_counter_ep_stride + t) * kCftCounterStride;
        // put_bar is a required mbarrier::report destination for the PTX, but it is not waited on.
        // Completion for smem reuse is enforced by the CTA-scope fabric.wait.sync_restrict::reads
        // below, which does not depend on mbarrier report delivery.
        cft_fabric_try_put_counted(le_id, data_offset, counter_offset, staging, bytes_per_token, put_bar);
        cft_fabric_submit();
        cft_fabric_wait_reads();
    }
#else
    // Launched only on the CFT path, which requires sm_100+; fail loudly rather than
    // completing with an empty body.
    asm volatile("trap;" ::: "memory");
#endif
}

template <typename T, int TOP_K, bool LOW_PRECISION, bool ENABLE_RANK_MASK>
__global__ void moeA2ACombineCountedWriteKernel(const CombineKernelPointers ptrs, int max_tokens_per_rank,
    int elements_per_token, int local_num_tokens, int rank_id)
{
    using InputT = std::conditional_t<LOW_PRECISION, __nv_fp8_e4m3, T>;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    int local_token_idx = blockIdx.x;
    int const size_per_token = elements_per_token * sizeof(InputT);

    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    // Empty rank: this block exists only for the PDL handshake above; no local token to reduce.
    if (local_num_tokens == 0)
        return;

#if !DISABLE_SYNC_FOR_PROFILING
    // Per-token readiness: warp 0 polls ONLY the k receive-slots its local token needs
    // (slot = target_rank*max + dst_idx), so a token reduces as soon as its own pieces land
    // (overlapping the still-running push under PDL) rather than waiting for every peer's counter.
    int lane_id = threadIdx.x % warpSize;
    // One token per block -> only warp 0 polls its receive-slots.
    if (threadIdx.x / warpSize == 0)
    {
        int const my_token = local_token_idx;
#pragma unroll 1
        for (int kk = lane_id; kk < TOP_K; kk += warpSize)
        {
            int tr = ptrs.topk_target_ranks[my_token * TOP_K + kk];
            int di = ptrs.topk_send_indices[my_token * TOP_K + kk];
            if (tr < 0 || di < 0)
                continue; // duplicate / invalid routing slot
            if constexpr (ENABLE_RANK_MASK)
            {
                if (!is_rank_active(ptrs.active_rank_mask, tr))
                    continue;
            }
            if (tr == rank_id)
                continue; // self contribution: not fabric-pushed

            int slot = tr * ptrs.combine_counter_ep_stride + di;
            uint64_t combine_base = ptrs.combine_counter_baseline[slot];
            uint64_t combine_target = combine_base + static_cast<uint64_t>(size_per_token);

            uint64_t* combineCounterPtr = &ptrs.combine_counters[static_cast<size_t>(slot) * kCftCounterStrideU64];
            uint64_t current_combine_counter = 0;
            auto s = clock64();
            while (true)
            {
                asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(current_combine_counter) : "l"(combineCounterPtr));
                if (current_combine_counter >= combine_target)
                {
                    break;
                }
                if (check_timeout(s, ptrs.timeout_cycles))
                {
                    printf(
                        "combine(cft): ---Rank %d tok %d k %d slot %d timed out counter=%llu base=%llu target=%llu\n",
                        rank_id, my_token, kk, slot, (unsigned long long) current_combine_counter,
                        (unsigned long long) combine_base, (unsigned long long) combine_target);
                    asm volatile("trap;");
                    return;
                }
            }
            ptrs.combine_counter_baseline[slot] = combine_target;
        }
#if TLLM_CFT_HAS_CUDA_13_4_SUPPORT
        asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
#endif
    }
    __syncthreads();
#endif

    __threadfence_system(); // system-scope fence before the gather (dispatch gets this from its kernel boundary)
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;
    vectorized_combine<TOP_K, T, InputT>(
        token_output, size_per_token, size_per_token, rank_id, max_tokens_per_rank, ptrs);
    cudaTriggerProgrammaticLaunchCompletion();
#else
    // Launched only on the CFT path, which requires sm_100+; fail loudly rather than
    // completing with an empty body.
    asm volatile("trap;" ::: "memory");
#endif // __CUDA_ARCH__ >= 1000
}

void moe_a2a_cft_combine_push_launch(MoeA2ACombineParams const& params)
{
    // grid.x selects a source rank; grid.y and the warps in each block fan out its tokens.
    int const bytes_per_token = params.wire_bytes_per_token;
    int const local_stride_per_token = params.cft_push_stride_per_token;
    uint8_t const* local_payload = static_cast<uint8_t const*>(params.cft_push_payload);

    // Pass peer metadata by value as a kernel argument.
    CftPeerLeIds le_ids = {};
    for (int i = 0; i < params.ep_size; i++)
        le_ids.ids[i] = params.cft_peer_le_ids[i];
    for (int w = 0; w < kRankMaskWords; ++w)
        le_ids.active_rank_mask[w] = params.active_rank_mask[w];

    // Push parallelism is env-overridable for tuning:
    //   TRTLLM_CFT_PUSH_WARPS           : warps per block (default kCombinePushWarpsPerBlock)
    //   TRTLLM_CFT_PUSH_BLOCKS_PER_RANK : blocks per source rank == grid.y
    int push_warps = kCombinePushWarpsPerBlock;
    if (char const* e = std::getenv("TRTLLM_CFT_PUSH_WARPS"))
    {
        int v = std::atoi(e);
        if (v >= 1)
            push_warps = v;
    }
    // Adaptive default: ~2 tokens/warp, capped at 32, so small batches do not launch empty blocks.
    int blocks_per_rank = (params.max_tokens_per_rank + push_warps * 2 - 1) / (push_warps * 2);
    if (blocks_per_rank < 1)
        blocks_per_rank = 1;
    if (blocks_per_rank > 32)
        blocks_per_rank = 32;
    if (char const* e = std::getenv("TRTLLM_CFT_PUSH_BLOCKS_PER_RANK"))
    {
        int v = std::atoi(e);
        if (v >= 1)
            blocks_per_rank = v;
    }
    int blockThreads = push_warps * 32;
    int per_warp_bytes = kCftMbarrierSlotBytes + bytes_per_token;
    int smem_size = push_warps * per_warp_bytes;

    // Per-block dynamic smem cap is 48KB by default on sm_90+; opt-in to use up to the
    // architectural max. DSR1 hidden=7168 BF16 needs ~57KB across 4 warps.
    if (smem_size > 48 * 1024)
    {
        auto set_attr = [&](auto* kernel_fn)
        { TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)); };
        if (params.enable_rank_mask)
            set_attr(moeA2ACftCombinePushKernel<true>);
        else
            set_attr(moeA2ACftCombinePushKernel<false>);
    }

    SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
        auto kernel_fn = moeA2ACftCombinePushKernel<ENABLE_RANK_MASK>;
        launchWithPdlWhenEnabled("moeA2ACftCombinePushKernel", kernel_fn, dim3(params.ep_size, blocks_per_rank),
            dim3(blockThreads), smem_size, params.stream, local_payload, params.recv_counters, params.flag_val, le_ids,
            params.ep_rank, params.ep_size, params.max_tokens_per_rank, bytes_per_token,
            params.cft_le_combine_payload_base, params.cft_le_combine_counter_base, params.combine_counter_ep_stride,
            local_stride_per_token);
    });
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;
    TLLM_CHECK(params.max_tokens_per_rank > 0);

    uint8_t* recv_buffer_bytes = static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank]));
    // CFT combine stages local contributions compactly into its dedicated receive region.
    uint8_t* const region_c_base
        = params.use_cft_for_combine ? static_cast<uint8_t*>(const_cast<void*>(params.cft_le_combine_recv)) : nullptr;
    int const grid = std::max(params.prepare_num_tokens, 1);

    // Zero LE-backed counters from HOST before kernel launch.
    // NOTE: Combine LE counters are zeroed in prepare_dispatch_launch (before any fabric activity).
    // Zeroing them here (after dispatch's fabric puts) corrupts subsequent counter increments
    // because cudaDeviceSynchronize does NOT wait for fabric engine completion.

    SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
        SWITCH_DTYPE(params.dtype, SrcT, {
            auto kernel_fn = moeA2APrepareCombineKernel<LOW_PRECISION, SrcT>;
            launchWithPdlWhenEnabled("moeA2APrepareCombineKernel", kernel_fn, grid, kBlockSize, 0, params.stream,
                recv_buffer_bytes, params.source_payload, params.elements_per_token, params.ep_size,
                params.max_tokens_per_rank, params.flag_val, params.recv_counters, params.source_stride_per_token,
                params.workspace_stride_per_token, params.prepare_first_token, params.prepare_num_tokens, region_c_base,
                params.ep_rank);
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

    // ---- CFT combine path (one block per token). ----
    // Bypasses the base fence combine below entirely. The CFT push is launched separately by
    // the op (moe_a2a_cft_combine_push_launch). This reduce kernel polls per-slot LE counters
    // and gathers local and peer contributions from the dedicated receive region.
    if (params.use_cft_for_combine)
    {
        int cft_grid = params.local_num_tokens;
        if (cft_grid == 0)
        {
            cft_grid = 1;
        }
        int const cft_block = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();

        CombineKernelPointers kp = {};
        kp.src_data_ptrs[0] = params.output_data;
        for (int rank = 0; rank < params.ep_size; rank++)
        {
            kp.recv_buffers[rank][0] = params.recv_buffers[rank];
        }
        for (int i = 0; i < params.ep_size; i++)
        {
            kp.completion_flags[i] = params.completion_flags[i];
        }
        kp.flag_val = params.flag_val;
        kp.topk_target_ranks = params.topk_target_ranks;
        kp.topk_send_indices = params.topk_send_indices;

        // CFT combine metadata.
        kp.combine_counters = params.cft_le_combine_counters;
        kp.combine_counter_baseline = params.cft_combine_counter_baseline;
        kp.combine_counter_ep_stride = params.combine_counter_ep_stride;
        kp.timeout_cycles = params.timeout_cycles;
        for (int w = 0; w < kRankMaskWords; ++w)
        {
            kp.active_rank_mask[w] = params.active_rank_mask[w];
        }

        // Offset-trick gather: peers' pushed data lands in THIS rank's region C
        // (cft_le_combine_recv). recv_buffers[P] = region_C_base + (P - S) * stride so the
        // reduce reads peer P's contribution at the same slot layout as the fence path.
        uint8_t const* combine_base = static_cast<uint8_t const*>(params.cft_le_combine_recv);
        int const element_size = static_cast<int>(tensorrt_llm::common::getDTypeSize(params.dtype));
        // Every contribution uses the same compact wire layout in the receive region.
        int const bytes_per_token = params.elements_per_token * (params.use_low_precision ? 1 : element_size);
        int64_t const peer_src_stride_per_rank = static_cast<int64_t>(params.max_tokens_per_rank) * bytes_per_token;
        for (int rank = 0; rank < params.ep_size; rank++)
        {
            // Local tokens occupy the zero-offset slice; peer slices are addressed relative to it.
            kp.recv_buffers[rank][0] = combine_base + (rank - params.ep_rank) * peer_src_stride_per_rank;
        }

        SWITCH_BOOL(params.enable_rank_mask, ENABLE_RANK_MASK, {
            SWITCH_DTYPE(params.dtype, T, {
                SWITCH_BOOL(params.use_low_precision, LOW_PRECISION, {
                    SWITCH_TOP_K(params.top_k, TOP_K, {
                        auto kernel_fn = moeA2ACombineCountedWriteKernel<T, TOP_K, LOW_PRECISION, ENABLE_RANK_MASK>;
                        launchWithPdlWhenEnabled("moeA2ACombineCountedWriteKernel", kernel_fn, cft_grid, cft_block, 0,
                            params.stream, kp, params.max_tokens_per_rank, params.elements_per_token,
                            params.local_num_tokens, params.ep_rank);
                    });
                });
            });
        });
        return;
    }

    // Configure kernel launch (one block per token).
    int const kBlockSize = tensorrt_llm::common::getEnvMoeA2ACombineBlockSize();
    int grid = params.local_num_tokens;
    // If local_num_tokens is 0, we still need to launch a minimal kernel to participate in the synchronization.
    if (grid == 0)
    {
        grid = 1;
    }

    // Prepare kernel pointers struct for combine
    CombineKernelPointers kernel_ptrs = {}; // Zero-initialize
    kernel_ptrs.timeout_cycles = params.timeout_cycles;

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
                        params.ep_rank, params.ep_size, params.reduce_stride_per_token);
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
