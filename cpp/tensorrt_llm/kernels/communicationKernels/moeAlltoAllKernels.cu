/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace tensorrt_llm::kernels::moe_a2a
{

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

template <int VEC_SIZE>
__device__ void warp_vectorized_copy_impl(void* dst, void const* src, int size, int lane_id)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int offset = lane_id * VEC_SIZE;

    while (offset < size)
    {
        vec_t<uint8_t, VEC_SIZE> vec_data;
        vec_data.load(src_ptr + offset);
        vec_data.store(dst_ptr + offset);
        offset += warpSize * VEC_SIZE;
    }
}

__device__ void warp_vectorized_copy(void* dst, void const* src, int size, int lane_id)
{
    if (size % 16 == 0)
    {
        warp_vectorized_copy_impl<16>(dst, src, size, lane_id);
    }
    else if (size % 8 == 0)
    {
        warp_vectorized_copy_impl<8>(dst, src, size, lane_id);
    }
    else if (size % 4 == 0)
    {
        warp_vectorized_copy_impl<4>(dst, src, size, lane_id);
    }
    else if (size % 2 == 0)
    {
        warp_vectorized_copy_impl<2>(dst, src, size, lane_id);
    }
    else
    {
        warp_vectorized_copy_impl<1>(dst, src, size, lane_id);
    }
}


__global__ void moeA2APrepareDispatchKernel(int* send_counters, int* send_indices,
    int* local_token_counter, int ep_size, int local_num_tokens, uint32_t* flag_val_ptr)
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
    // Fill send_indices with -1
    int total = local_num_tokens * ep_size;
    if (idx < total)
    {
        send_indices[idx] = -1;
    }
}

// ============================================================================
// Generic Dispatch Kernel Implementation
// One warp per token design:
// - Each CTA has 256 threads = 8 warps
// - Each warp independently processes one token and all its payloads
// - Better GPU utilization and reduced synchronization overhead
// ============================================================================

__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, top_k]
    const DispatchKernelPointers ptrs,                                      // Struct containing all kernel pointers
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int local_num_tokens, int rank_id, int ep_size, int top_k, int num_experts_per_rank)
{
    // Constants
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads / 32 threads per warp
    constexpr int WARP_SIZE = 32;

    // Determine which warp this thread belongs to and which token it handles
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int local_token_idx = global_warp_id;

    if (local_token_idx >= local_num_tokens)
    {
        return;
    }

    uint64_t already_copied = 0;
    for (int k = 0; k < top_k; k++)
    {
        int expert_id = token_selected_experts[local_token_idx * top_k + k];
        // Use contiguous partitioning to determine target rank
        int target_rank = compute_target_rank_id(expert_id, num_experts_per_rank);

        if (already_copied & (1ULL << target_rank))
            continue;

        // Only one thread per warp should increment the counter
        int dst_token_idx;
        if (lane_id == 0)
        {
            dst_token_idx = atomicAdd(&ptrs.send_counters[target_rank], 1);

            ptrs.send_indices[local_token_idx * ep_size + target_rank] = dst_token_idx;
        }
        // Broadcast the index to all threads in the warp
        dst_token_idx = __shfl_sync(0xffffffff, dst_token_idx, 0);

        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
            uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank][payload_idx]);

            int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];

            // Account for source rank offset in the 3D recv buffer layout [source_ranks, tokens, elements]
            uint8_t* dst_ptr = dst_data + (rank_id * max_tokens_per_rank + dst_token_idx) * bytes_per_token;
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

            warp_vectorized_copy(dst_ptr, src_ptr, bytes_per_token, lane_id);
        }

        already_copied |= 1ULL << target_rank;
    }

    // (A) __syncwarp guarantees: If lane0's sent data are visible, then all lanes' sent data are visible
    __syncwarp();

    // Finished sending this token. Check if we're the last token to complete.
    if (lane_id == 0)
    {
        int cnt;
        // (B) .release guarantees: If increment of local_token_counter is visible, then lane0's sent data are visible
        asm volatile("atom.add.release.sys.u32 %0, [%1], %2;"
                     : "=r"(cnt) 
                     : "l"(ptrs.local_token_counter), "r"(1));
        
        if (cnt + 1 == local_num_tokens)  // The last token dispatched
        {
            // Signal to each target rank that this source rank has completed sending
            for (int target_rank = 0; target_rank < ep_size; target_rank++)
            {
                uint32_t* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
                uint32_t flag_value = *ptrs.flag_val;
                // (C) .release guarantees: If flag setting is visible, then increment of local_token_counter is visible
                printf("$$$Rank %d setting completion flag to %d for rank %d\n", rank_id, flag_value, target_rank);
                asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(flag_addr), "r"(flag_value));
            }

            // Busy wait until all source ranks targeting this rank have completed sending
            uint32_t expected_value = *ptrs.flag_val;
            for (int source_rank = 0; source_rank < ep_size; source_rank++)
            {
                uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][source_rank];
                uint32_t flag_value;
                do
                {
                    // Acquire load to ensure visibility of peer's release-store
                    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
                    // printf("###Rank %d trying completion flag from rank %d, flag_value: %d, expected_value: %d\n", rank_id, source_rank, flag_value, expected_value);
                } while (flag_value != expected_value);
                printf("###Rank %d received completion flag from rank %d, flag_value: %d, completion_flags[rank_id][source_rank]: %d address: %p\n", rank_id, source_rank, flag_value, *flag_ptr, flag_ptr);
            }
        }
    }
}


void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params)
{
    constexpr int kBlockSize = 256;
    int n = params.local_num_tokens * params.ep_size;
    int grid = ceilDiv(n, kBlockSize);
    moeA2APrepareDispatchKernel<<<grid, kBlockSize, 0, params.stream>>>(
        params.send_counters, params.send_indices, params.local_token_counter,
        params.ep_size, params.local_num_tokens, params.flag_val);
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

    // Configure kernel launch
    constexpr int kBlockSize = 256;
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
    int grid_size = ceilDiv(params.local_num_tokens,kWarpsPerBlock);

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
    for (int rank = 0; rank < params.ep_size; rank++)
    {
        for (int payload = 0; payload < params.num_payloads; payload++)
        {
            kernel_ptrs.recv_buffers[rank][payload] = params.recv_buffers[rank][payload];
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
    kernel_ptrs.send_indices = params.send_indices;
    kernel_ptrs.local_token_counter = params.local_token_counter;

    moeA2ADispatchKernel<<<grid_size, kBlockSize, 0, params.stream>>>(params.token_selected_experts, kernel_ptrs,
        params.num_payloads, params.max_tokens_per_rank,
        params.local_num_tokens, params.ep_rank, params.ep_size, params.top_k, params.num_experts_per_rank);
}

// ============================================================================
// Helper Functions for Vectorized Summation
// ============================================================================

template <int VEC_SIZE, typename T>
__device__ void warp_vectorized_sum_impl(void* dst, void const* src, int size, int lane_id)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int offset = lane_id * VEC_SIZE;

    while (offset < size)
    {
        vec_t<uint8_t, VEC_SIZE> src_vec;
        vec_t<uint8_t, VEC_SIZE> dst_vec;

        src_vec.load(src_ptr + offset);
        dst_vec.load(dst_ptr + offset);

        // Reinterpret as typed arrays for arithmetic
        T* dst_typed = reinterpret_cast<T*>(&dst_vec);
        T const* src_typed = reinterpret_cast<T const*>(&src_vec);
        constexpr int elems_per_vec = VEC_SIZE / sizeof(T);

#pragma unroll
        for (int i = 0; i < elems_per_vec; i++)
        {
            dst_typed[i] += src_typed[i];
        }

        dst_vec.store(dst_ptr + offset);
        offset += warpSize * VEC_SIZE;
    }
}

template <typename T>
__device__ void warp_vectorized_sum(void* dst, void const* src, int size, int lane_id)
{
    // Choose vector size based on type alignment, but process as bytes
    if (size % 16 == 0)
    {
        warp_vectorized_sum_impl<16, T>(dst, src, size, lane_id);
    }
    else if (size % 8 == 0)
    {
        warp_vectorized_sum_impl<8, T>(dst, src, size, lane_id);
    }
    else if (size % 4 == 0)
    {
        warp_vectorized_sum_impl<4, T>(dst, src, size, lane_id);
    }
    else if (size % 2 == 0)
    {
        warp_vectorized_sum_impl<2, T>(dst, src, size, lane_id);
    }
    else
    {
        warp_vectorized_sum_impl<1, T>(dst, src, size, lane_id);
    }
}

template <int VEC_SIZE, typename T>
__device__ void warp_vectorized_fill_impl(void* dst, T value, int size, int lane_id)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    int offset = lane_id * VEC_SIZE;

    while (offset < size)
    {
        vec_t<uint8_t, VEC_SIZE> vec_data;

        // Fill the vector with the value of T
        T* vec_as_T = reinterpret_cast<T*>(&vec_data);
        constexpr int elems_per_vec = VEC_SIZE / sizeof(T);
#pragma unroll
        for (int i = 0; i < elems_per_vec; i++)
        {
            vec_as_T[i] = value;
        }

        vec_data.store(dst_ptr + offset);
        offset += warpSize * VEC_SIZE;
    }
}

template <typename T>
__device__ void warp_vectorized_fill(void* dst, T value, int size, int lane_id)
{
    if (size % 16 == 0)
    {
        warp_vectorized_fill_impl<16, T>(dst, value, size, lane_id);
    }
    else if (size % 8 == 0)
    {
        warp_vectorized_fill_impl<8, T>(dst, value, size, lane_id);
    }
    else if (size % 4 == 0)
    {
        warp_vectorized_fill_impl<4, T>(dst, value, size, lane_id);
    }
    else if (size % 2 == 0)
    {
        warp_vectorized_fill_impl<2, T>(dst, value, size, lane_id);
    }
    else
    {
        warp_vectorized_fill_impl<1, T>(dst, value, size, lane_id);
    }
}


// Copy payload to recv buffer using warp-based vectorized copy (one warp per token)
__global__ void moeA2APrepareCombineKernel(uint8_t* recv_buffer_bytes, const uint8_t* payload_bytes, 
    int bytes_per_token, int ep_size, int max_tokens_per_rank, uint32_t* flag_val_ptr)
{
    // One warp per token design
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads / 32
    
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    
    int total_tokens = ep_size * max_tokens_per_rank;
    if (global_warp_id >= total_tokens)
        return;
    
    // Calculate source and destination pointers for this token
    size_t token_offset = static_cast<size_t>(global_warp_id) * bytes_per_token;
    uint8_t* dst_ptr = recv_buffer_bytes + token_offset;
    const uint8_t* src_ptr = payload_bytes + token_offset;
    
    // Copy one token's data using vectorized copy
    warp_vectorized_copy(dst_ptr, src_ptr, bytes_per_token, lane_id);
    
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Increment flag_val for this combine round
        *flag_val_ptr = *flag_val_ptr + 1;
    }
}

// ============================================================================
// Generic Combine Kernel Implementation (Templated by data type)
// ============================================================================

template <typename T>
__global__ void moeA2ACombineKernel(
    const CombineKernelPointers ptrs,                        // Combine-specific struct, src_data_ptrs[0] is output
    int max_tokens_per_rank, int elements_per_token, int local_num_tokens, int rank_id, int ep_size)
{
    // Constants
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int WARP_SIZE = 32;

    // One warp per token
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int local_token_idx = global_warp_id;
    int const size_per_token = elements_per_token * sizeof(T);

    if (local_token_idx >= local_num_tokens)
    {
        return;
    }

    // In-kernel readiness synchronization at start of combine:
    // - Block 0, thread 0 signals readiness to all peers with current flag_val.
    // - Thread 0 of each block waits for all peers' readiness (equality), then __syncthreads.
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        uint32_t cur_val = *ptrs.flag_val;
        for (int peer_rank = 0; peer_rank < ep_size; ++peer_rank)
        {
            uint32_t* flag_addr = &ptrs.completion_flags[peer_rank][rank_id];
            asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(flag_addr), "r"(cur_val));
        }
    }

    if (threadIdx.x == 0)
    {
        uint32_t expected_value = *ptrs.flag_val;
        for (int peer_rank = 0; peer_rank < ep_size; ++peer_rank)
        {
            uint32_t* flag_ptr = &ptrs.completion_flags[rank_id][peer_rank];
            uint32_t flag_value;
            do
            {
                // Acquire load to ensure visibility of peer's release-store
                asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
            } while (flag_value != expected_value);
        }
    }
    __syncthreads();

    // Get output location for this token (using src_data_ptrs[0] as output)
    T* token_output = static_cast<T*>(ptrs.src_data_ptrs[0]) + local_token_idx * elements_per_token;

    // Zero initialize output using vectorized fill
    warp_vectorized_fill(token_output, T(0), size_per_token, lane_id);
    

    // For each possible target rank, check if this token was sent there
    for (int target_rank = 0; target_rank < ep_size; target_rank++)
    {
        // Check if this token was sent to target_rank
        int dst_idx = ptrs.send_indices[local_token_idx * ep_size + target_rank];
        if (dst_idx < 0)
            continue; // Wasn't sent to this rank

        // Get received data location from target rank's buffer
        // Using payload 0 (the only payload for combine)
        uint8_t const* recv_buffer = static_cast<uint8_t const*>(ptrs.recv_buffers[target_rank][0]);
        uint8_t const* recv_ptr = recv_buffer + (rank_id * max_tokens_per_rank + dst_idx) * size_per_token;

        if (lane_id == 0)
        {
            // Fused print: print all 5 values in one printf
            float vals[5];
            for (int i = 0; i < 5; ++i)
            {
                uint16_t bf16_val = *(reinterpret_cast<uint16_t const*>(recv_ptr) + i);
                vals[i] = __bfloat162float(*reinterpret_cast<__nv_bfloat16 const*>(&bf16_val));
            }
            int send_indices_vals[4];
            for (int i = 0; i < 4; ++i)
            {
                send_indices_vals[i] = ptrs.send_indices[local_token_idx * ep_size + i];
            }
            printf(
                "Rank %d token %d retrieved from rank %d, recv_ptr[0..4]=[%f, %f, %f, %f, %f] send_indices[0...3]=[%d, "
                "%d, %d, %d]\n",
                rank_id, local_token_idx, target_rank, vals[0], vals[1], vals[2], vals[3], vals[4],
                send_indices_vals[0], send_indices_vals[1], send_indices_vals[2], send_indices_vals[3]);
        }

        // Sum into output (distributed across warp)
        warp_vectorized_sum<T>(token_output, reinterpret_cast<T const*>(recv_ptr), size_per_token, lane_id);
        
    }
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params)
{
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32; // 8 warps per block
    
    // Calculate bytes per token based on dtype
    int element_size;
    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF:
        element_size = sizeof(half);
        break;
    case nvinfer1::DataType::kBF16:
        element_size = sizeof(__nv_bfloat16);
        break;
    case nvinfer1::DataType::kFLOAT:
        element_size = sizeof(float);
        break;
    default:
        TLLM_CHECK_WITH_INFO(false, "Unsupported dtype for combine prepare");
        return;
    }
    
    int bytes_per_token = params.elements_per_token * element_size;
    int total_tokens = params.ep_size * params.max_tokens_per_rank;
    int grid_size = ceilDiv(total_tokens, kWarpsPerBlock);
    
    moeA2APrepareCombineKernel<<<grid_size, kBlockSize, 0, params.stream>>>(
        static_cast<uint8_t*>(const_cast<void*>(params.recv_buffers[params.ep_rank])), 
        static_cast<const uint8_t*>(params.prepare_payload),
        bytes_per_token,
        params.ep_size,
        params.max_tokens_per_rank,
        params.flag_val);
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
    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32; // warpSize
    int grid_size = ceilDiv(params.local_num_tokens, kWarpsPerBlock);

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
    kernel_ptrs.send_indices = params.send_indices;
    kernel_ptrs.local_token_counter = params.local_token_counter;

    // Launch appropriate kernel based on data type
    switch (params.dtype)
    {
    case nvinfer1::DataType::kHALF:
        moeA2ACombineKernel<half><<<grid_size, kBlockSize, 0, params.stream>>>(kernel_ptrs,
            params.max_tokens_per_rank, params.elements_per_token, params.local_num_tokens,
            params.ep_rank, params.ep_size);
        break;

    case nvinfer1::DataType::kBF16:
        moeA2ACombineKernel<__nv_bfloat16><<<grid_size, kBlockSize, 0, params.stream>>>(kernel_ptrs,
            params.max_tokens_per_rank, params.elements_per_token, params.local_num_tokens,
            params.ep_rank, params.ep_size);
        break;

    case nvinfer1::DataType::kFLOAT:
        moeA2ACombineKernel<float><<<grid_size, kBlockSize, 0, params.stream>>>(kernel_ptrs,
            params.max_tokens_per_rank, params.elements_per_token, params.local_num_tokens,
            params.ep_rank, params.ep_size);
        break;

    default: TLLM_CHECK_WITH_INFO(false, "Unsupported data type for moe_a2a_combine");
    }
}

} // namespace tensorrt_llm::kernels::moe_a2a
