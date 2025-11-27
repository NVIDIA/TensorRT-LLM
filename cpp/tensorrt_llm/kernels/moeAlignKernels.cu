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

// Inspired by vLLM's moe_align_kernel.cu and ported to TensorRT-LLM

#include "moeAlignKernels.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeCommKernelsCommon.h"
#include <cub/cub.cuh>

#define CEILDIV(x, y) (((x) + (y) -1) / (y))
#define WARP_SIZE 32

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t padded_num_experts,
    int32_t experts_per_warp, int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded)
{
    extern __shared__ int32_t shared_counts[];

    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i)
    {
        if (my_expert_start + i < padded_num_experts)
        {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }

    __syncthreads();

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }

    __syncthreads();

    // Compute prefix sum over token counts per expert
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int expert_id = threadIdx.x;
    if (expert_id < num_experts)
    {
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
    if (expert_id <= num_experts)
    {
        cumsum[expert_id] = cumsum_val;
    }

    if (expert_id == num_experts)
    {
        *total_tokens_post_pad = cumsum_val;
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer, size_t numel)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded)
{
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = (int32_t*) (shared_mem + num_experts + 1);

    for (int i = 0; i < num_experts; ++i)
    {
        tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        tokens_cnts[threadIdx.x] = 0;
        for (int i = 1; i <= blockDim.x; ++i)
        {
            tokens_cnts[i * num_experts + threadIdx.x] += tokens_cnts[(i - 1) * num_experts + threadIdx.x];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i)
        {
            cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
        }
        *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[threadIdx.x * num_experts + expert_id];
    }
}

template <typename scalar_t>
void invokeMoeAlignBlockSizeTyped(scalar_t const* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
    int32_t* num_tokens_post_pad, int32_t num_experts, int32_t block_size, int32_t numel, int32_t max_num_tokens_padded,
    cudaStream_t stream)
{
    int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int experts_per_warp = WARP_SIZE;
    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // BlockScan uses 1024 threads and assigns one thread per expert.
    TLLM_CHECK_WITH_INFO(padded_num_experts < 1024, "padded_num_experts must be less than 1024");

    // Allocate temporary cumsum buffer
    int32_t* cumsum_buffer;
    cudaMallocAsync(&cumsum_buffer, (num_experts + 1) * sizeof(int32_t), stream);
    cudaMemsetAsync(cumsum_buffer, 0, (num_experts + 1) * sizeof(int32_t), stream);

    bool small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode)
    {
        const int32_t thread_count = std::max((int32_t) num_experts, (int32_t) WARP_SIZE);
        const int32_t shared_mem_size = ((thread_count + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

        moe_align_block_size_small_batch_expert_kernel<scalar_t><<<1, thread_count, shared_mem_size, stream>>>(topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts, block_size, numel, max_num_tokens_padded);
    }
    else
    {
        size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
        size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

        moe_align_block_size_kernel<scalar_t><<<1, threads, shared_mem_size, stream>>>(topk_ids, sorted_token_ids,
            expert_ids, num_tokens_post_pad, num_experts, padded_num_experts, experts_per_warp, block_size, numel,
            cumsum_buffer, max_num_tokens_padded);

        int const block_threads = std::min(256, (int) threads);
        int const num_blocks = (numel + block_threads - 1) / block_threads;
        int const max_blocks = 65535;
        int const actual_blocks = std::min(num_blocks, max_blocks);

        count_and_sort_expert_tokens_kernel<scalar_t>
            <<<actual_blocks, block_threads, 0, stream>>>(topk_ids, sorted_token_ids, cumsum_buffer, numel);
    }

    cudaFreeAsync(cumsum_buffer, stream);
}

void invokeMoeAlignBlockSize(void const* topk_ids, int32_t topk_ids_dtype_size, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* num_tokens_post_pad, int32_t num_experts, int32_t block_size, int32_t numel,
    int32_t max_num_tokens_padded, cudaStream_t stream)
{
    // Dispatch based on dtype size
    if (topk_ids_dtype_size == sizeof(int32_t))
    {
        invokeMoeAlignBlockSizeTyped(static_cast<int32_t const*>(topk_ids), sorted_token_ids, expert_ids,
            num_tokens_post_pad, num_experts, block_size, numel, max_num_tokens_padded, stream);
    }
    else if (topk_ids_dtype_size == sizeof(int64_t))
    {
        invokeMoeAlignBlockSizeTyped(static_cast<int64_t const*>(topk_ids), sorted_token_ids, expert_ids,
            num_tokens_post_pad, num_experts, block_size, numel, max_num_tokens_padded, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported topk_ids dtype size: %d", topk_ids_dtype_size);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
