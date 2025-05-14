/*
 * Copyright (c) 2025-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_bf16_gemm
{

// Grid size is num_tokens / TILE_TOKEN * hidden_out / TILE_OUT / WARP_PER_BLOCK.
// Each warp processes TILE_TOKEN tokens and TILE_OUT rows.
// Within each warp, it loops through hidden_in in steps of WARP_SIZE * VEC_SIZE.
template <int HIDDEN_IN, int TILE_TOKEN, int TILE_OUT, bool ALIGNED>
__launch_bounds__(BLOCK_SIZE) __global__ void llama4_fp8_bf16_gemm_per_warp_kernel(
    __nv_fp8_e4m3 const* __restrict__ A, // Input tensor [num_tokens][hidden_in]
    __nv_fp8_e4m3 const* __restrict__ B, // Input tensor [hidden_out][hidden_in]
    __nv_bfloat16* __restrict__ C,       // Output tensor [num_tokens][hidden_out]
    float* __restrict__ scaling_factor, int num_tokens, int hidden_in, int hidden_out)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    // Shared memory for tensor A values.
    __shared__ aligned_fp8x8 shared_a[TILE_TOKEN][BLOCK_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum[TILE_TOKEN][TILE_OUT];
#pragma unroll
    for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            thread_sum[tile_token_idx][tile_out_idx].x = 0.0f;
            thread_sum[tile_token_idx][tile_out_idx].y = 0.0f;
        }
    }

    int const token_idx = blockIdx.y;                           // Token ID in the unit of TILE_TOKEN
    int const warp_idx = threadIdx.x / WARP_SIZE;               // Warp ID within the block
    int const row_idx = blockIdx.x * WARP_PER_BLOCK + warp_idx; // row ID in the unit of TILE_OUT
    int const tid = threadIdx.x;
    int const lane_idx = tid % WARP_SIZE;

    int chunk = 0;
#if ENABLE_ACQBULK && ENABLE_PREFETCH
#pragma unroll
    for (; chunk < ((HIDDEN_IN > 0 ? HIDDEN_IN : hidden_in) / WARP_SIZE / VEC_SIZE - 1); chunk++)
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            int current_row = row_idx * TILE_OUT + tile_out_idx;
            int base_idx = chunk * WARP_SIZE + lane_idx;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(&B[current_row * HIDDEN_IN / VEC_SIZE + base_idx])
                         : "memory");
        }
    }
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            int current_row = row_idx * TILE_OUT + tile_out_idx;
            int base_idx = chunk * WARP_SIZE + lane_idx;
            if (ALIGNED || base_idx * VEC_SIZE < hidden_in)
            {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(&B[current_row * HIDDEN_IN / VEC_SIZE + base_idx])
                             : "memory");
            }
        }
    }
#endif

#if ENABLE_ACQBULK
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Processing 8 elements each
    chunk = 0;
#pragma unroll
    for (; chunk < ((HIDDEN_IN > 0 ? HIDDEN_IN : hidden_in) / BLOCK_SIZE / VEC_SIZE) - 1; chunk++)
    {
        int base_idx_a = chunk * BLOCK_SIZE + tid;
        int base_idx_b = chunk * BLOCK_SIZE + lane_idx;

        // Load values from tensor A into shared memory
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
            int current_token = token_idx * TILE_TOKEN + tile_token_idx;
            shared_a[tile_token_idx][tid]
                = reinterpret_cast<aligned_fp8x8 const*>(A)[current_token * hidden_in / VEC_SIZE + base_idx_a];
        }

        // Load values from tensor B
        aligned_fp8x8 b_vec[WARP_PER_BLOCK][TILE_OUT];
#pragma unroll
        for (int tile_warp_idx = 0; tile_warp_idx < WARP_PER_BLOCK; tile_warp_idx++)
        {
#pragma unroll
            for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
            {
                int current_col = base_idx_b + tile_warp_idx * WARP_SIZE;
                int current_row = row_idx * TILE_OUT + tile_out_idx;
                b_vec[tile_warp_idx][tile_out_idx]
                    = reinterpret_cast<aligned_fp8x8 const*>(B)[current_row * hidden_in / VEC_SIZE + current_col];
            }
        }

        __syncthreads();

        // Compute partial sum
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
#pragma unroll
            for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
            {
#pragma unroll
                for (int tile_warp_idx = 0; tile_warp_idx < WARP_PER_BLOCK; tile_warp_idx++)
                {
                    aligned_fp8x8 a_vec_current = shared_a[tile_token_idx][tile_warp_idx * WARP_SIZE + lane_idx];
                    aligned_fp8x8 b_vec_current = b_vec[tile_warp_idx][tile_out_idx];
#pragma unroll
                    for (int i = 0; i < VEC_SIZE / 4; i++)
                    {
                        float4 a_val = float4(a_vec_current.data[i]);
                        float4 b_val = float4(b_vec_current.data[i]);

                        thread_sum[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.x, a_val.y),
                            make_float2(b_val.x, b_val.y), thread_sum[tile_token_idx][tile_out_idx]);
                        thread_sum[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.z, a_val.w),
                            make_float2(b_val.z, b_val.w), thread_sum[tile_token_idx][tile_out_idx]);
                    }
                }
            }
        }

        __syncthreads();
    }
    {
        int base_idx_a = chunk * BLOCK_SIZE + tid;
        int base_idx_b = chunk * BLOCK_SIZE + lane_idx;

        if (ALIGNED || base_idx_a * VEC_SIZE < hidden_in)
        {
            // Load values from tensor A into shared memory
#pragma unroll
            for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
            {
                int current_token = token_idx * TILE_TOKEN + tile_token_idx;
                shared_a[tile_token_idx][tid]
                    = reinterpret_cast<aligned_fp8x8 const*>(A)[current_token * hidden_in / VEC_SIZE + base_idx_a];
            }
        }

        // Load values from tensor B
        aligned_fp8x8 b_vec[WARP_PER_BLOCK][TILE_OUT];
#pragma unroll
        for (int tile_warp_idx = 0; tile_warp_idx < WARP_PER_BLOCK; tile_warp_idx++)
        {
            int current_col = base_idx_b + tile_warp_idx * WARP_SIZE;
            if (ALIGNED || current_col < hidden_in)
            {
#pragma unroll
                for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
                {
                    int current_row = row_idx * TILE_OUT + tile_out_idx;
                    b_vec[tile_warp_idx][tile_out_idx]
                        = reinterpret_cast<aligned_fp8x8 const*>(B)[current_row * hidden_in / VEC_SIZE + current_col];
                }
            }
        }

        __syncthreads();

        // Compute partial sum
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
#pragma unroll
            for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
            {
#pragma unroll
                for (int tile_warp_idx = 0; tile_warp_idx < WARP_PER_BLOCK; tile_warp_idx++)
                {
                    int current_col = base_idx_b + tile_warp_idx * WARP_SIZE;
                    if (ALIGNED || current_col < hidden_in)
                    {
                        aligned_fp8x8 a_vec_current = shared_a[tile_token_idx][tile_warp_idx * WARP_SIZE + lane_idx];
                        aligned_fp8x8 b_vec_current = b_vec[tile_warp_idx][tile_out_idx];
#pragma unroll
                        for (int i = 0; i < VEC_SIZE / 4; i++)
                        {
                            float4 a_val = float4(a_vec_current.data[i]);
                            float4 b_val = float4(b_vec_current.data[i]);

                            thread_sum[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.x, a_val.y),
                                make_float2(b_val.x, b_val.y), thread_sum[tile_token_idx][tile_out_idx]);
                            thread_sum[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.z, a_val.w),
                                make_float2(b_val.z, b_val.w), thread_sum[tile_token_idx][tile_out_idx]);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    // Reduce partial sums using warp-level reduction.
#pragma unroll
    for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
    {
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
            float warp_sum = thread_sum[tile_token_idx][tile_out_idx].x + thread_sum[tile_token_idx][tile_out_idx].y;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }

            // Broadcast the result to all threads in the warp
            thread_sum[tile_token_idx][tile_out_idx].x = __shfl_sync(0xffffffff, warp_sum, 0);
        }
    }

    if (lane_idx == 0)
    {
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
#pragma unroll
            for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
            {
                int current_token = token_idx * TILE_TOKEN + tile_token_idx;
                int current_row = row_idx * TILE_OUT + tile_out_idx;
                C[current_token * hidden_out + current_row]
                    = __float2bfloat16(thread_sum[tile_token_idx][tile_out_idx].x * scaling_factor[0]);
            }
        }
    }

#if ENABLE_PREEXIT
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#endif
}

#define DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, TILE_OUT, ALIGNED)                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TILE_TOKEN == 1)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(llama4_fp8_bf16_gemm_per_warp_kernel<HIDDEN_IN, 1, TILE_OUT, ALIGNED>);     \
        }                                                                                                              \
        if (TILE_TOKEN == 2)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(llama4_fp8_bf16_gemm_per_warp_kernel<HIDDEN_IN, 2, TILE_OUT, ALIGNED>);     \
        }                                                                                                              \
        if (TILE_TOKEN == 3)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(llama4_fp8_bf16_gemm_per_warp_kernel<HIDDEN_IN, 3, TILE_OUT, ALIGNED>);     \
        }                                                                                                              \
        if (TILE_TOKEN == 4)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(llama4_fp8_bf16_gemm_per_warp_kernel<HIDDEN_IN, 4, TILE_OUT, ALIGNED>);     \
        }                                                                                                              \
        if (TILE_TOKEN == 8)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(llama4_fp8_bf16_gemm_per_warp_kernel<HIDDEN_IN, 8, TILE_OUT, ALIGNED>);     \
        }                                                                                                              \
        throw std::invalid_argument("Invalid tile token");                                                             \
    } while (0)

#define DISPATCH_PER_WARP_FC_FP8_BF16_TILE_OUT(HIDDEN_IN, TILE_TOKEN, TILE_OUT, ALIGNED)                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TILE_OUT == 1)                                                                                             \
        {                                                                                                              \
            DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 1, ALIGNED);                               \
        }                                                                                                              \
        if (TILE_OUT == 2)                                                                                             \
        {                                                                                                              \
            DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 2, ALIGNED);                               \
        }                                                                                                              \
        if (TILE_OUT == 3)                                                                                             \
        {                                                                                                              \
            DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 3, ALIGNED);                               \
        }                                                                                                              \
        if (TILE_OUT == 4)                                                                                             \
        {                                                                                                              \
            DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 4, ALIGNED);                               \
        }                                                                                                              \
        if (TILE_OUT == 8)                                                                                             \
        {                                                                                                              \
            DISPATCH_PER_WARP_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 8, ALIGNED);                               \
        }                                                                                                              \
        throw std::invalid_argument("Invalid tile token");                                                             \
    } while (0)

#define DEFINE_GET_PER_WARP_FUNC_PTR(HIDDEN_IN, ALIGNED)                                                               \
    void* get_per_warp_func_ptr_aligned_##ALIGNED##_##HIDDEN_IN##_(int tile_token, int tile_out)                       \
    {                                                                                                                  \
        DISPATCH_PER_WARP_FC_FP8_BF16_TILE_OUT(HIDDEN_IN, tile_token, tile_out, ALIGNED);                              \
    }

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_bf16_gemm
