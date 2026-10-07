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

// Adapted from vLLM (Apache-2.0):
// https://github.com/vllm-project/vllm/blob/v0.25.0/csrc/libtorch_stable/quantization/w8a8/fp8/common.cu
// Extracted: dynamic_per_token_scaled_fp8_quant_kernel_strided only.

#pragma once

#include "vectorized_cub_helpers.h"
#include "vectorized_fp8_quant_compat.cuh"
#include "vectorized_vectorization_utils.cuh"

namespace vllm
{

inline constexpr int kPerTokenQuantBlockSize = 256;

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(fp8_type* __restrict__ out, float* __restrict__ scale,
    scalar_t const* __restrict__ input, float const* __restrict__ scale_ub, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride)
{
    static_assert(kPerTokenQuantBlockSize > 0, "block size must be positive");
    assert(blockDim.x == kPerTokenQuantBlockSize);

    const int64_t token_idx = blockIdx.x;
    int const tid = threadIdx.x;

    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
    int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
    scalar_t const* token_in = input + in_offset;
    fp8_type* token_out = out + out_offset;

    // 1) per-token absmax
    float absmax_val = 0.f;
    vectorize_read_with_alignment<16>(token_in, hidden_size, tid, blockDim.x,
        [&] __device__(scalar_t v) { absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v))); });

    using BlockReduce = cub::BlockReduce<float, kPerTokenQuantBlockSize>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float const block_max = BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

    __shared__ float token_scale;
    if (tid == 0)
    {
        token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
        token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>, min_scaling_factor<fp8_type>::val());
        scale[token_idx] = token_scale;
    }
    __syncthreads();

    // 2) quantize
    vectorize_with_alignment<16>(token_in, token_out, hidden_size, tid, blockDim.x,
        [=] __device__(fp8_type & dst, scalar_t const& src)
        { dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src), token_scale); });
}

} // namespace vllm
