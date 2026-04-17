/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Gather FP8 K values and scales from a non-contiguous paged indexer K cache
/// into contiguous output buffers. This is the inverse of invokeIndexerKCacheScatter.
///
/// @param k_fp8_out          Output FP8 data [num_tokens, head_dim], contiguous.
/// @param k_scale_out        Output scale data [num_tokens, scale_size], contiguous (uint8 view).
/// @param k_cache            Indexer K cache pool [num_blocks, block_size, 1, per_token_size] (may be non-contiguous).
/// @param slot_mapping_fp8   Flat element index for FP8 data start position [num_tokens].
/// @param slot_mapping_scale Flat element index for scale data start position [num_tokens].
/// @param num_tokens         Number of tokens to gather.
/// @param head_dim           Head dimension (must be 128).
/// @param scale_size         Scale size in bytes (must be 4).
/// @param cache_dim_0..3     Sizes of k_cache dimensions.
/// @param cache_stride_0..3  Strides of k_cache dimensions (in bytes).
/// @param stream             CUDA stream.
void invokeIndexerKCacheGather(uint8_t* k_fp8_out, uint8_t* k_scale_out, uint8_t const* k_cache,
    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3,
    cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
