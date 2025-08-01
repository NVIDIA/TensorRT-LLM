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

#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_fp8_gemm_swiglu
{

// Grid size is num_tokens / TILE_TOKEN * hidden_out / TILE_OUT.
// Each block processes TILE_TOKEN tokens and TILE_OUT rows.
// within each block, it steps through hidden_in in steps of BLOCK_SIZE * VEC_SIZE.
template <int HIDDEN_IN, int TILE_TOKEN, int TILE_OUT, bool ALIGNED = true>
__launch_bounds__(BLOCK_SIZE) __global__ void llama4_fp8_fp8_gemm_swiglu_per_block_kernel(
    __nv_fp8_e4m3 const* __restrict__ A, // Input tensor [num_tokens][hidden_in]
    __nv_fp8_e4m3 const* __restrict__ B, // Input tensor [2 * hidden_out][hidden_in]
    __nv_fp8_e4m3* __restrict__ C,       // Output tensor [num_tokens][hidden_out]
    float const* __restrict__ in_scale, float const* __restrict__ out_scale_inv, int num_tokens, int hidden_in,
    int hidden_out)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    // Shared memory for block reduction
    __shared__ float reduce_buffer_gate[TILE_TOKEN][TILE_OUT][BLOCK_SIZE];
    __shared__ float reduce_buffer_linear[TILE_TOKEN][TILE_OUT][BLOCK_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum_gate[TILE_TOKEN][TILE_OUT];
    float2 thread_sum_linear[TILE_TOKEN][TILE_OUT];
#pragma unroll
    for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            thread_sum_gate[tile_token_idx][tile_out_idx].x = 0.0f;
            thread_sum_gate[tile_token_idx][tile_out_idx].y = 0.0f;
            thread_sum_linear[tile_token_idx][tile_out_idx].x = 0.0f;
            thread_sum_linear[tile_token_idx][tile_out_idx].y = 0.0f;
        }
    }

    int const token_idx = blockIdx.y;
    int const row_idx = blockIdx.x;
    int const tid = threadIdx.x;

    int chunk = 0;
#if ENABLE_ACQBULK && ENABLE_PREFETCH
#pragma unroll 9
    for (; chunk < ((HIDDEN_IN > 0 ? HIDDEN_IN : hidden_in) / BLOCK_SIZE / VEC_SIZE - 1); chunk++)
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            int current_row = row_idx * TILE_OUT + tile_out_idx;
            int base_idx = chunk * BLOCK_SIZE + tid;
            asm volatile("prefetch.global.L2 [%0];" ::"l"(&B[current_row * hidden_in / VEC_SIZE + base_idx])
                         : "memory");
            asm volatile(
                "prefetch.global.L2 [%0];" ::"l"(&B[(current_row + hidden_out) * hidden_in / VEC_SIZE + base_idx])
                : "memory");
        }
    }
    {
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            int current_row = row_idx * TILE_OUT + tile_out_idx;
            int base_idx = chunk * BLOCK_SIZE + tid;
            if (ALIGNED || base_idx * VEC_SIZE < hidden_in)
            {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(&B[current_row * hidden_in / VEC_SIZE + base_idx])
                             : "memory");
                asm volatile(
                    "prefetch.global.L2 [%0];" ::"l"(&B[(current_row + hidden_out) * hidden_in / VEC_SIZE + base_idx])
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
#pragma unroll 9
    for (; chunk < ((HIDDEN_IN > 0 ? HIDDEN_IN : hidden_in) / BLOCK_SIZE / VEC_SIZE - 1); chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;
        // Load values from tensor B
        aligned_fp8x8 b_vec_gate[TILE_OUT];
        aligned_fp8x8 b_vec_linear[TILE_OUT];
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
            int current_row = row_idx * TILE_OUT + tile_out_idx;
            b_vec_gate[tile_out_idx]
                = reinterpret_cast<aligned_fp8x8 const*>(B)[current_row * hidden_in / VEC_SIZE + base_idx];
            b_vec_linear[tile_out_idx] = reinterpret_cast<aligned_fp8x8 const*>(
                B)[(current_row + hidden_out) * hidden_in / VEC_SIZE + base_idx];
        }

        // Load values from tensor A
        aligned_fp8x8 a_vec[TILE_TOKEN];
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
            int current_token = token_idx * TILE_TOKEN + tile_token_idx;
            a_vec[tile_token_idx]
                = reinterpret_cast<aligned_fp8x8 const*>(A)[current_token * hidden_in / VEC_SIZE + base_idx];
        }

        // Compute partial sum
#pragma unroll
        for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
        {
#pragma unroll
            for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
            {
                aligned_fp8x8 a_vec_current = a_vec[tile_token_idx];
                aligned_fp8x8 b_vec_current_gate = b_vec_gate[tile_out_idx];
                aligned_fp8x8 b_vec_current_linear = b_vec_linear[tile_out_idx];
#pragma unroll
                for (int i = 0; i < VEC_SIZE / 4; i++)
                {
                    float4 a_val = float4(a_vec_current.data[i]);
                    float4 b_val_gate = float4(b_vec_current_gate.data[i]);
                    float4 b_val_linear = float4(b_vec_current_linear.data[i]);

                    thread_sum_gate[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.x, a_val.y),
                        make_float2(b_val_gate.x, b_val_gate.y), thread_sum_gate[tile_token_idx][tile_out_idx]);
                    thread_sum_gate[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.z, a_val.w),
                        make_float2(b_val_gate.z, b_val_gate.w), thread_sum_gate[tile_token_idx][tile_out_idx]);
                    thread_sum_linear[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.x, a_val.y),
                        make_float2(b_val_linear.x, b_val_linear.y), thread_sum_linear[tile_token_idx][tile_out_idx]);
                    thread_sum_linear[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.z, a_val.w),
                        make_float2(b_val_linear.z, b_val_linear.w), thread_sum_linear[tile_token_idx][tile_out_idx]);
                }
            }
        }
    }

    // The last chunk may be partial.
    {
        int base_idx = chunk * BLOCK_SIZE + tid;
        if (ALIGNED || base_idx * VEC_SIZE < hidden_in)
        {
            // Load values from tensor B
            aligned_fp8x8 b_vec_gate[TILE_OUT];
            aligned_fp8x8 b_vec_linear[TILE_OUT];
#pragma unroll
            for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
            {
                int current_row = row_idx * TILE_OUT + tile_out_idx;
                b_vec_gate[tile_out_idx]
                    = reinterpret_cast<aligned_fp8x8 const*>(B)[current_row * hidden_in / VEC_SIZE + base_idx];
                b_vec_linear[tile_out_idx] = reinterpret_cast<aligned_fp8x8 const*>(
                    B)[(current_row + hidden_out) * hidden_in / VEC_SIZE + base_idx];
            }

            // Load values from tensor A
            aligned_fp8x8 a_vec[TILE_TOKEN];
#pragma unroll
            for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
            {
                int current_token = token_idx * TILE_TOKEN + tile_token_idx;
                a_vec[tile_token_idx]
                    = reinterpret_cast<aligned_fp8x8 const*>(A)[current_token * hidden_in / VEC_SIZE + base_idx];
            }

            // Compute partial sum
#pragma unroll
            for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
            {
#pragma unroll
                for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
                {
                    aligned_fp8x8 a_vec_current = a_vec[tile_token_idx];
                    aligned_fp8x8 b_vec_current_gate = b_vec_gate[tile_out_idx];
                    aligned_fp8x8 b_vec_current_linear = b_vec_linear[tile_out_idx];
#pragma unroll
                    for (int i = 0; i < VEC_SIZE / 4; i++)
                    {
                        float4 a_val = float4(a_vec_current.data[i]);
                        float4 b_val_gate = float4(b_vec_current_gate.data[i]);
                        float4 b_val_linear = float4(b_vec_current_linear.data[i]);

                        thread_sum_gate[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.x, a_val.y),
                            make_float2(b_val_gate.x, b_val_gate.y), thread_sum_gate[tile_token_idx][tile_out_idx]);
                        thread_sum_gate[tile_token_idx][tile_out_idx] = ffma2(make_float2(a_val.z, a_val.w),
                            make_float2(b_val_gate.z, b_val_gate.w), thread_sum_gate[tile_token_idx][tile_out_idx]);
                        thread_sum_linear[tile_token_idx][tile_out_idx]
                            = ffma2(make_float2(a_val.x, a_val.y), make_float2(b_val_linear.x, b_val_linear.y),
                                thread_sum_linear[tile_token_idx][tile_out_idx]);
                        thread_sum_linear[tile_token_idx][tile_out_idx]
                            = ffma2(make_float2(a_val.z, a_val.w), make_float2(b_val_linear.z, b_val_linear.w),
                                thread_sum_linear[tile_token_idx][tile_out_idx]);
                    }
                }
            }
        }
    }

    // Reduce partial sums using warp-level reduction.
#pragma unroll
    for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
    {
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
            float warp_sum_gate
                = thread_sum_gate[tile_token_idx][tile_out_idx].x + thread_sum_gate[tile_token_idx][tile_out_idx].y;
            float warp_sum_linear
                = thread_sum_linear[tile_token_idx][tile_out_idx].x + thread_sum_linear[tile_token_idx][tile_out_idx].y;
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                warp_sum_gate += __shfl_down_sync(0xffffffff, warp_sum_gate, offset);
                warp_sum_linear += __shfl_down_sync(0xffffffff, warp_sum_linear, offset);
            }
            // First thread in each warp writes to shared memory
            if (tid % WARP_SIZE == 0)
            {
                reduce_buffer_gate[tile_token_idx][tile_out_idx][tid / WARP_SIZE] = warp_sum_gate;
                reduce_buffer_linear[tile_token_idx][tile_out_idx][tid / WARP_SIZE] = warp_sum_linear;
            }
        }
    }

    __syncthreads();

    if (tid == 0)
    {
#pragma unroll
        for (int tile_token_idx = 0; tile_token_idx < TILE_TOKEN; tile_token_idx++)
        {
#pragma unroll
            for (int tile_out_idx = 0; tile_out_idx < TILE_OUT; tile_out_idx++)
            {
                float block_sum_gate = 0.0f;
                float block_sum_linear = 0.0f;
#pragma unroll
                for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; i++)
                {
                    block_sum_gate += reduce_buffer_gate[tile_token_idx][tile_out_idx][i];
                    block_sum_linear += reduce_buffer_linear[tile_token_idx][tile_out_idx][i];
                }
                int current_row = row_idx * TILE_OUT + tile_out_idx;
                int current_token = token_idx * TILE_TOKEN + tile_token_idx;
                float in_scale_val = in_scale[0];
                float out_scale_inv_val = out_scale_inv[0];
                C[current_token * hidden_out + current_row] = __nv_fp8_e4m3(
                    silu(block_sum_gate * in_scale_val) * block_sum_linear * in_scale_val * out_scale_inv_val);
            }
        }
    }

#if ENABLE_PREEXIT
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#endif
}

#define DISPATCH_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, TILE_OUT, ALIGNED)                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TILE_TOKEN == 1)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(                                                                            \
                llama4_fp8_fp8_gemm_swiglu_per_block_kernel<HIDDEN_IN, 1, TILE_OUT, ALIGNED>);                         \
        }                                                                                                              \
        if (TILE_TOKEN == 2)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(                                                                            \
                llama4_fp8_fp8_gemm_swiglu_per_block_kernel<HIDDEN_IN, 2, TILE_OUT, ALIGNED>);                         \
        }                                                                                                              \
        if (TILE_TOKEN == 3)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(                                                                            \
                llama4_fp8_fp8_gemm_swiglu_per_block_kernel<HIDDEN_IN, 3, TILE_OUT, ALIGNED>);                         \
        }                                                                                                              \
        if (TILE_TOKEN == 4)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(                                                                            \
                llama4_fp8_fp8_gemm_swiglu_per_block_kernel<HIDDEN_IN, 4, TILE_OUT, ALIGNED>);                         \
        }                                                                                                              \
        if (TILE_TOKEN == 8)                                                                                           \
        {                                                                                                              \
            return reinterpret_cast<void*>(                                                                            \
                llama4_fp8_fp8_gemm_swiglu_per_block_kernel<HIDDEN_IN, 8, TILE_OUT, ALIGNED>);                         \
        }                                                                                                              \
        throw std::invalid_argument("Invalid tile token");                                                             \
    } while (0)

#define DISPATCH_FC_FP8_BF16_TILE_OUT(HIDDEN_IN, TILE_TOKEN, TILE_OUT, ALIGNED)                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (TILE_OUT == 1)                                                                                             \
        {                                                                                                              \
            DISPATCH_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 1, ALIGNED);                                        \
        }                                                                                                              \
        if (TILE_OUT == 2)                                                                                             \
        {                                                                                                              \
            DISPATCH_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 2, ALIGNED);                                        \
        }                                                                                                              \
        if (TILE_OUT == 3)                                                                                             \
        {                                                                                                              \
            DISPATCH_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 3, ALIGNED);                                        \
        }                                                                                                              \
        if (TILE_OUT == 4)                                                                                             \
        {                                                                                                              \
            DISPATCH_FC_FP8_BF16_TILE_TOKEN(HIDDEN_IN, TILE_TOKEN, 4, ALIGNED);                                        \
        }                                                                                                              \
        throw std::invalid_argument("Invalid tile token");                                                             \
    } while (0)

#define DEFINE_GET_FUNC_PTR(HIDDEN_IN, ALIGNED)                                                                        \
    void* get_func_ptr_aligned_##ALIGNED##_##HIDDEN_IN##_(int tile_token, int tile_out)                                \
    {                                                                                                                  \
        DISPATCH_FC_FP8_BF16_TILE_OUT(HIDDEN_IN, tile_token, tile_out, ALIGNED);                                       \
    }

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_fp8_gemm_swiglu
