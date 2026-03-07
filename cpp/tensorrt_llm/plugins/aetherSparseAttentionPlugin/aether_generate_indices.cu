/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "aether_generate_indices.h"
#include "tensorrt_llm/common/cudaUtils.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm::plugins::aether
{

template <typename T>
__global__ void aetherGenerateIndicesKernel(
    T const* query,             // [B, H, D]
    T const* centroids,         // [B, H, num_blocks, D]
    int* active_indices,        // [B, H, max_active_blocks]
    int* active_counts,         // [B, H]
    float cos_threshold,
    int head_dim,
    int num_blocks,
    int max_active_blocks)
{
    // Each thread block processes one head for one batch (B * H grid)
    int const batch_head_idx = blockIdx.x;
    int const tid = threadIdx.x;
    
    // Shared memory for query caching and prefix sum
    extern __shared__ float s_query[];
    
    T const* q_ptr = query + batch_head_idx * head_dim;
    T const* c_ptr = centroids + batch_head_idx * num_blocks * head_dim;
    int* idx_out_ptr = active_indices + batch_head_idx * max_active_blocks;
    int* count_out_ptr = active_counts + batch_head_idx;
    
    // Load Query into shared memory (assuming head_dim <= blockDim.x for simplicity)
    if (tid < head_dim) {
        s_query[tid] = static_cast<float>(q_ptr[tid]);
    }
    __syncthreads();
    
    // Initialize active count
    __shared__ int s_active_count;
    if (tid == 0) {
        s_active_count = 0;
    }
    __syncthreads();

    // Iterate over blocks, assigning one or more blocks per thread
    for (int block_idx = tid; block_idx < num_blocks; block_idx += blockDim.x) {
        // Compute dot product of Q and Centroid[block_idx]
        float dot_val = 0.0f;
        T const* block_c_ptr = c_ptr + block_idx * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            dot_val += s_query[d] * static_cast<float>(block_c_ptr[d]);
        }
        
        // Angular pruning condition
        bool is_active = (dot_val >= cos_threshold);
        
        // Use warp-level primitives to compactly write indices
        unsigned int active_mask = __ballot_sync(0xffffffff, is_active);
        
        // Only one thread per warp performs the atomicAdd to get a base offset
        int warp_offset = 0;
        int lane_id = tid % 32;
        if (lane_id == 0 && active_mask != 0) {
            warp_offset = atomicAdd(&s_active_count, __popc(active_mask));
        }
        // Broadcast offset to other threads in warp
        warp_offset = __shfl_sync(0xffffffff, warp_offset, 0);
        
        if (is_active) {
            // Find our rank within the active threads of the warp
            int rank = __popc(active_mask & ((1 << lane_id) - 1));
            int out_idx = warp_offset + rank;
            if (out_idx < max_active_blocks) {
                idx_out_ptr[out_idx] = block_idx;
            }
        }
    }
    
    __syncthreads();
    if (tid == 0) {
        *count_out_ptr = min(s_active_count, max_active_blocks);
    }
}

void invokeAetherGenerateIndices(
    void const* query,
    void const* block_centroids,
    int* active_indices,
    int* active_counts,
    float cos_threshold,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_blocks,
    int max_active_blocks,
    cudaStream_t stream)
{
    // Simplified invocation, assuming float16 (half) for now.
    // In a full implementation, we switch over data types (FP16/BF16/FP32).
    dim3 grid(batch_size * num_heads);
    dim3 block(256); // 256 threads per block
    
    // Shared memory size: enough for query vector (head_dim * sizeof(float))
    size_t smem_size = head_dim * sizeof(float);
    
    aetherGenerateIndicesKernel<half><<<grid, block, smem_size, stream>>>(
        reinterpret_cast<half const*>(query),
        reinterpret_cast<half const*>(block_centroids),
        active_indices,
        active_counts,
        cos_threshold,
        head_dim,
        num_blocks,
        max_active_blocks
    );
    
    sync_check_cuda_error();
}

} // namespace tensorrt_llm::plugins::aether
