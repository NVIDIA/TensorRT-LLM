/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "minimaxM3Fp8IndexerKernel.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 64;
constexpr int kElemsPerThread = kHeadDim / 32;

// Match the established vLLM contract exactly: the normalized/RoPE result is
// first materialized as BF16 and then cast, without an external FP8 scale.
__device__ __forceinline__ __nv_fp8_e4m3 bf16RoundedToFp8(float value)
{
    return __nv_fp8_e4m3(__bfloat162float(__float2bfloat16_rn(value)));
}

__global__ void minimaxM3Fp8IndexerQKNormRopeKernel(__nv_bfloat16 const* qk, __nv_fp8_e4m3* q_out,
    __nv_fp8_e4m3* k_cache, int const* out_cache_loc, int64_t page_stride, int64_t token_stride, int page_size,
    int num_tokens, int num_heads_q, float eps, __nv_bfloat16 const* q_weight, __nv_bfloat16 const* k_weight,
    float base, int const* position_ids)
{
    int const warps_per_block = blockDim.x / 32;
    int const warp_id = threadIdx.x / 32;
    int const lane_id = threadIdx.x % 32;
    int const global_warp = blockIdx.x * warps_per_block + warp_id;
    int const total_heads = num_heads_q + 1;
    int const token_idx = global_warp / total_heads;
    int const local_head = global_warp % total_heads;
    if (token_idx >= num_tokens)
    {
        return;
    }

    bool const is_q = local_head < num_heads_q;
    int64_t const input_offset
        = (static_cast<int64_t>(token_idx) * total_heads + local_head) * kHeadDim + lane_id * kElemsPerThread;

    uint2 const packed_input = *reinterpret_cast<uint2 const*>(qk + input_offset);
    float elements[kElemsPerThread];
    float sum_squares = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
    {
        auto const values = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const*>(&packed_input)[pair]);
        elements[pair * 2] = values.x;
        elements[pair * 2 + 1] = values.y;
        sum_squares += values.x * values.x + values.y * values.y;
    }

    sum_squares = tensorrt_llm::common::warpReduceSum(sum_squares);
    float const rms_rcp = rsqrtf(sum_squares / static_cast<float>(kHeadDim) + eps);
    auto const* weight = is_q ? q_weight : k_weight;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
    {
        int const dim = lane_id * kElemsPerThread + i;
        elements[i] *= rms_rcp * (1.0F + __bfloat162float(weight[dim]));
    }

    // MiniMax-M3 uses NeoX partial RoPE: rotate the first 64 of 128 channels.
    // Four elements per lane means the matching half is eight lanes away.
    __syncwarp();
    constexpr int kPairOffset = (kRotaryDim / 2) / kElemsPerThread;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
    {
        int const dim = lane_id * kElemsPerThread + i;
        float paired = __shfl_xor_sync(0xffffffff, elements[i], kPairOffset);
        if (dim < kRotaryDim)
        {
            if (lane_id < kPairOffset)
            {
                paired = -paired;
            }
            int const dim_idx = (dim * 2) % kRotaryDim;
            int const half_dim = dim_idx / 2;
            float const frequency = powf(base, -2.0F * half_dim / static_cast<float>(kRotaryDim));
            float sine;
            float cosine;
            __sincosf(static_cast<float>(position_ids[token_idx]) * frequency, &sine, &cosine);
            elements[i] = elements[i] * cosine + paired * sine;
        }
    }
    __syncwarp();

    uint32_t packed_output = 0;
    auto* fp8_values = reinterpret_cast<__nv_fp8_e4m3*>(&packed_output);
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
    {
        fp8_values[i] = bf16RoundedToFp8(elements[i]);
    }

    __nv_fp8_e4m3* output;
    if (is_q)
    {
        int64_t const output_offset
            = (static_cast<int64_t>(token_idx) * num_heads_q + local_head) * kHeadDim + lane_id * kElemsPerThread;
        output = q_out + output_offset;
    }
    else
    {
        int const slot = out_cache_loc[token_idx];
        int const page = slot / page_size;
        int const within_page = slot % page_size;
        output = k_cache + static_cast<int64_t>(page) * page_stride + static_cast<int64_t>(within_page) * token_stride
            + lane_id * kElemsPerThread;
    }
    *reinterpret_cast<uint32_t*>(output) = packed_output;
}

} // namespace

void launchMinimaxM3Fp8IndexerQKNormRope(void const* qk, void* q_out, void* k_cache, int const* out_cache_loc,
    int64_t page_stride, int64_t token_stride, int page_size, int num_tokens, int num_heads_q, int head_dim,
    int rotary_dim, float eps, void const* q_weight, void const* k_weight, float base, int const* position_ids,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(head_dim == kHeadDim, "MiniMax-M3 FP8 indexer requires head_dim=128");
    TLLM_CHECK_WITH_INFO(rotary_dim == kRotaryDim, "MiniMax-M3 FP8 indexer requires rotary_dim=64");
    TLLM_CHECK_WITH_INFO(num_heads_q > 0, "MiniMax-M3 FP8 indexer requires at least one query head");

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    int const total_warps = num_tokens * (num_heads_q + 1);
    int const grid_size = common::divUp(total_warps, kWarpsPerBlock);
    minimaxM3Fp8IndexerQKNormRopeKernel<<<grid_size, kBlockSize, 0, stream>>>(static_cast<__nv_bfloat16 const*>(qk),
        static_cast<__nv_fp8_e4m3*>(q_out), static_cast<__nv_fp8_e4m3*>(k_cache), out_cache_loc, page_stride,
        token_stride, page_size, num_tokens, num_heads_q, eps, static_cast<__nv_bfloat16 const*>(q_weight),
        static_cast<__nv_bfloat16 const*>(k_weight), base, position_ids);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
