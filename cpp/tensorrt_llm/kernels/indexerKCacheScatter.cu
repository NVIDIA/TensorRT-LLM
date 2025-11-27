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

#include "IndexerKCacheScatter.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
/**
 * Given a flat element index and tensor shape [d0, d1, d2, d3] with strides [s0, s1, s2, s3],
 * find the actual memory offset within the given k cache pool using the strides.
 */
__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
    int64_t flat_idx, int32_t d0, int32_t d1, int32_t d2, int32_t d3, int64_t s0, int64_t s1, int64_t s2, int64_t s3)
{
    // Unravel from innermost to outermost dimension
    int32_t i3 = flat_idx % d3;
    flat_idx /= d3;

    int32_t i2 = flat_idx % d2;
    flat_idx /= d2;

    int32_t i1 = flat_idx % d1;
    flat_idx /= d1;

    int32_t i0 = flat_idx;

    // Compute memory offset using strides
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
}

} // anonymous namespace

/**
 * CUDA kernel to scatter both FP8 K values and scales into the indexer k cache pool
 *
 * @param k_fp8_bytes       Quantized FP8 data [num_tokens, 128]
 * @param k_scale_bytes     Quantized scales (1 per token) [num_tokens, 4]
 * @param k_cache           Indexer k cache pool with shape [num_blocks, block_size, 1, per_token_size] (can be
 * non-contiguous)
 * @param slot_mapping_fp8  Flat element index for FP8 data start position [num_tokens]
 * @param slot_mapping_scale Flat element index for scale data start position [num_tokens]
 * @param num_tokens        Number of tokens
 * @param head_dim          Head dimension (must be 128)
 * @param scale_size        Scale size in bytes (must be 4)
 * @param cache_stride_0    Stride for k_cache dimension 0 (in bytes)
 * @param cache_stride_1    Stride for k_cache dimension 1 (in bytes)
 * @param cache_stride_2    Stride for k_cache dimension 2 (in bytes)
 * @param cache_stride_3    Stride for k_cache dimension 3 (in bytes)
 * @param cache_dim_0       Size of k_cache dimension 0
 * @param cache_dim_1       Size of k_cache dimension 1
 * @param cache_dim_2       Size of k_cache dimension 2
 * @param cache_dim_3       Size of k_cache dimension 3
 */
__global__ void indexerKCacheScatterUnifiedKernel(uint8_t const* __restrict__ k_fp8_bytes,
    uint8_t const* __restrict__ k_scale_bytes, uint8_t* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping_fp8, int64_t const* __restrict__ slot_mapping_scale, int32_t num_tokens,
    int32_t head_dim, int32_t scale_size, int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2,
    int64_t cache_stride_3, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3)
{
    // For head_dim=128, each thread handles 4 bytes/elements per read/write instruction
    constexpr int VEC_SIZE = 4;

    // Token index from block.x
    int32_t token_idx = blockIdx.x;

    if (token_idx >= num_tokens)
    {
        return;
    }

    int64_t flat_idx_fp8_base = slot_mapping_fp8[token_idx];
    int64_t flat_idx_scale_base = slot_mapping_scale[token_idx];

    if (flat_idx_fp8_base < 0 || flat_idx_scale_base < 0)
    {
        return;
    }

    int32_t head_dim_idx = threadIdx.x * VEC_SIZE;
    int64_t flat_idx = flat_idx_fp8_base + head_dim_idx;

    // Convert flat index to memory offset using strides (k cache pool from cpp kv cache manager is non-contiguous)
    int64_t dst_offset = flatIndexToMemoryOffset(flat_idx, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3,
        cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
    int64_t src_offset = token_idx * head_dim + head_dim_idx;

    // 4 bytes write
    *reinterpret_cast<uint32_t*>(&k_cache[dst_offset]) = *reinterpret_cast<uint32_t const*>(&k_fp8_bytes[src_offset]);

    // Only thread 0 writes the single 4 bytes scale value
    if (threadIdx.x == 0)
    {
        int64_t dst_offset_scale = flatIndexToMemoryOffset(flat_idx_scale_base, cache_dim_0, cache_dim_1, cache_dim_2,
            cache_dim_3, cache_stride_0, cache_stride_1, cache_stride_2, cache_stride_3);
        int64_t src_offset_scale = token_idx * scale_size; // scale_size = 4

        // 4 bytes write for scale
        *reinterpret_cast<uint32_t*>(&k_cache[dst_offset_scale])
            = *reinterpret_cast<uint32_t const*>(&k_scale_bytes[src_offset_scale]);
    }
}

void invokeIndexerKCacheScatter(uint8_t const* k_fp8_bytes, uint8_t const* k_scale_bytes, uint8_t* k_cache,
    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3, cudaStream_t stream)
{
    if (num_tokens == 0)
    {
        return;
    }

    // Assertions for DeepSeek-V3.2 configuration
    constexpr int32_t QUANT_BLOCK_SIZE = 128;
    TLLM_CHECK_WITH_INFO(
        head_dim == QUANT_BLOCK_SIZE, "head_dim must equal 128 for DeepSeek-V3 indexer cache (got %d)", head_dim);
    TLLM_CHECK_WITH_INFO(
        scale_size == 4, "scale_size must equal 4 bytes (1 float32 scale per token, got %d)", scale_size);

    // For head_dim=128, we use 32 threads to handle 128 bytes per token and extra 4 bytes for scale
    constexpr int32_t THREADS_PER_BLOCK = 32;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(num_tokens);

    indexerKCacheScatterUnifiedKernel<<<grid, block, 0, stream>>>(k_fp8_bytes, k_scale_bytes, k_cache, slot_mapping_fp8,
        slot_mapping_scale, num_tokens, head_dim, scale_size, cache_stride_0, cache_stride_1, cache_stride_2,
        cache_stride_3, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3);

    // Check for kernel launch errors
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
