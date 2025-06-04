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

#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Bf16Gemm.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Bf16GemmAttnScalingPerBlockTemplate.cuh"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Bf16GemmPerBlockTemplate.cuh"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Fp8Bf16GemmPerWarpTemplate.cuh"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Utils.cuh"
#include <stdexcept>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_bf16_gemm
{

DEFINE_GET_PER_BLOCK_FUNC_PTR(/*HIDDEN_IN=*/5120, /*ALIGNED=*/true);
DEFINE_GET_PER_BLOCK_ATTN_SCALING_FUNC_PTR(/*HIDDEN_IN=*/5120, /*ALIGNED=*/true, /*POS_IDS_INT64=*/false);
DEFINE_GET_PER_BLOCK_ATTN_SCALING_FUNC_PTR(/*HIDDEN_IN=*/5120, /*ALIGNED=*/true, /*POS_IDS_INT64=*/true);

void llama4_fp8_bf16_gemm_launcher(void const* A, void const* B, void* C, void const* scaling_factor, int num_tokens,
    int hidden_in, int hidden_out, cudaStream_t stream)
{
    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &scaling_factor, (void*) &num_tokens,
        (void*) &hidden_in, (void*) &hidden_out};
    if (num_tokens == 1)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 1 and tile_out == 1.
        dim3 const grid_size = dim3(div_up(hidden_out, 1), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 1);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 2)
    {
        // When num_tokens == 2, the best tiling size is tile_token == 2 and tile_out == 1.
        dim3 const grid_size = dim3(div_up(hidden_out, 1), div_up(num_tokens, 2), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(2, 1);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 3)
    {
        // When num_tokens == 3, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 4);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 4)
    {
        // When num_tokens == 4, the best tiling size is tile_token == 2 and tile_out == 2.
        dim3 const grid_size = dim3(div_up(hidden_out, 2), div_up(num_tokens, 2), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(2, 2);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 5)
    {
        // When num_tokens == 5, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 4);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 6)
    {
        // When num_tokens == 6, the best tiling size is tile_token == 3 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 3), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(3, 4);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 7)
    {
        // When num_tokens == 7, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 4);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else if (num_tokens == 8)
    {
        // When num_tokens == 8, the best tiling size is tile_token == 2 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 2), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(2, 4);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 7);
    }
    else
    {
        throw std::runtime_error("llama4QKVGemm: num_tokens > 8 are not supported.");
    }
}

void* get_kernel_func(int tile_token, int tile_out, bool pos_ids_int64)
{
    if (pos_ids_int64)
    {
        return get_per_block_attn_scaling_func_ptr_aligned_true_pos_int64_true_5120_(tile_token, tile_out);
    }
    return get_per_block_attn_scaling_func_ptr_aligned_true_pos_int64_false_5120_(tile_token, tile_out);
}

void llama4_fp8_bf16_gemm_attn_scaling_launcher(void const* A, void const* B, void* C, void const* scaling_factor,
    void const* pos_ids, bool pos_ids_int64, float floor_scale, float attn_scale, int num_tokens, int hidden_in,
    int hidden_out, int q_hidden_out, cudaStream_t stream)
{
    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &scaling_factor, (void*) &pos_ids, (void*) &floor_scale,
        (void*) &attn_scale, (void*) &num_tokens, (void*) &hidden_in, (void*) &hidden_out, (void*) &q_hidden_out};
    if (num_tokens == 1)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 1 and tile_out == 1.
        dim3 const grid_size = dim3(div_up(hidden_out, 1), div_up(num_tokens, 1), 1);
        void* kernel_func = get_kernel_func(1, 1, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 2)
    {
        // When num_tokens == 2, the best tiling size is tile_token == 2 and tile_out == 2.
        dim3 const grid_size = dim3(div_up(hidden_out, 2), div_up(num_tokens, 2), 1);
        void* kernel_func = get_kernel_func(2, 2, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 3)
    {
        // When num_tokens == 3, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_kernel_func(1, 4, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 4)
    {
        // When num_tokens == 4, the best tiling size is tile_token == 2 and tile_out == 2.
        dim3 const grid_size = dim3(div_up(hidden_out, 2), div_up(num_tokens, 2), 1);
        void* kernel_func = get_kernel_func(2, 2, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 5)
    {
        // When num_tokens == 5, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_kernel_func(1, 4, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 6)
    {
        // When num_tokens == 6, the best tiling size is tile_token == 2 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 2), 1);
        void* kernel_func = get_kernel_func(2, 4, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 7)
    {
        // When num_tokens == 7, the best tiling size is tile_token == 1 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_kernel_func(1, 4, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else if (num_tokens == 8)
    {
        // When num_tokens == 8, the best tiling size is tile_token == 2 and tile_out == 4.
        dim3 const grid_size = dim3(div_up(hidden_out, 4), div_up(num_tokens, 2), 1);
        void* kernel_func = get_kernel_func(2, 4, pos_ids_int64);
        launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 11);
    }
    else
    {
        throw std::runtime_error("llama4_fp8_bf16_gemm: num_tokens > 8 are not supported.");
    }
}

void llama4_fp8_bf16_gemm_op(void const* A, void const* B, void* C, void const* scaling_factor, void const* pos_ids,
    bool pos_ids_int64, int num_tokens, int hidden_in, int hidden_out, cudaStream_t stream)
{
    if (pos_ids != nullptr)
    {
        llama4_fp8_bf16_gemm_attn_scaling_launcher(A, B, C, scaling_factor, pos_ids, pos_ids_int64, FLOOR_SCALE,
            ATTN_SCALE, num_tokens, hidden_in, hidden_out, Q_HIDDEN_OUT, stream);
    }
    else
    {
        llama4_fp8_bf16_gemm_launcher(A, B, C, scaling_factor, num_tokens, hidden_in, hidden_out, stream);
    }
}

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_fp8_bf16_gemm
