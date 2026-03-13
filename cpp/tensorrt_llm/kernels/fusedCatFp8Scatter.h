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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Fused concat + FP8 quantization + scatter into paged K cache.
///
/// Combines the functionality of fusedCatFp8 and indexerKCacheScatter into a single kernel.
/// The FP8-quantized values are written to both:
///   1. A contiguous output buffer (for use by the prefill logits path)
///   2. The non-contiguous paged K cache (for later decode-phase gather)
///
/// This eliminates one global memory round trip compared to running the two kernels sequentially.
///
/// @param fp8_out            Output FP8 data [M, head_dim], row-major, contiguous.
/// @param scale_out          Output scales [M, 1], float32, contiguous.
/// @param k_cache            Paged K cache [num_blocks, block_size, 1, per_token_size] (may be non-contiguous).
/// @param pe                 Input PE part, BF16.
/// @param nope               Input non-PE part, BF16.
/// @param slot_mapping_fp8   Flat element index for FP8 data start in k_cache [M].
/// @param slot_mapping_scale Flat element index for scale start in k_cache [M].
/// @param M                  Number of rows.
/// @param pe_dim             PE dimension.
/// @param nope_dim           Non-PE dimension.
/// @param head_dim           Total head dimension (must be 128).
/// @param pe_row_stride      Row stride for pe input.
/// @param nope_row_stride    Row stride for nope input.
/// @param use_ue8m0          If true, use UE8M0 scale format.
/// @param cache_dim_0..3     Sizes of k_cache dimensions.
/// @param cache_stride_0..3  Strides of k_cache dimensions (in bytes).
/// @param stream             CUDA stream.
void invokeFusedCatFp8Scatter(__nv_fp8_e4m3* fp8_out, float* scale_out, uint8_t* k_cache, __nv_bfloat16 const* pe,
    __nv_bfloat16 const* nope, int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t M,
    int32_t pe_dim, int32_t nope_dim, int32_t head_dim, int32_t pe_row_stride, int32_t nope_row_stride, bool use_ue8m0,
    int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3, int64_t cache_stride_0,
    int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
