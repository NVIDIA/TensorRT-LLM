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

#include "tensorrt_llm/kernels/llama4QKVGemm.h"
#include "tensorrt_llm/kernels/llama4Fp8Bf16GemmPerBlockTemplate.cuh"
#include "tensorrt_llm/kernels/llama4Fp8Bf16GemmPerWarpTemplate.cuh"
#include <stdexcept>

#define GEMM_HIDDEN_IN 5120
#define GEMM_HIDDEN_OUT 896 // This is QKV_GEMM. Replaced with 4096 for MLP_FC1.

#define BLOCK_SIZE 128
#define WARP_SIZE 32

namespace tensorrt_llm::kernels::llama4_qkv_gemm
{

DEFINE_GET_PER_BLOCK_FUNC_PTR(5120, true);

// Function to launch kernel using FDL(Flexible Dispatch Layer)
void launch_kernel_fdl(
    dim3 grid_dim, dim3 block_dim, cudaStream_t stream, void* kernel_func, void* args[], int num_args)
{
    cudaLaunchConfig_t config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[1];
    config.attrs = attrs;
    config.numAttrs = 1;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchKernelExC(&config, (void const*) kernel_func, args);
}

inline int div_up(int x, int y) {
    return (x + y - 1) / y;
}

void llama4_qkv_gemv_kernel_launcher(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, __nv_bfloat16* C,
    float const* scaling_factor, int num_tokens, int hidden_in, int hidden_out, cudaStream_t stream)
{
    void* args[] = {(void*) &A, (void*) &B, (void*) &C, (void*) &scaling_factor, (void*) &num_tokens, (void*) &hidden_in, (void*) &hidden_out};
    if (num_tokens == 1)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 1 and tile_out == 1.
        const dim3 grid_size = dim3(div_up(GEMM_HIDDEN_OUT, 1), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 1);
        launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 5);
    }
    else if (num_tokens == 2)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 2 and tile_out == 1.
        const dim3 grid_size = dim3(div_up(GEMM_HIDDEN_OUT, 1), div_up(num_tokens, 2), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(2, 1);
        launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 5);
    }
    else if (num_tokens == 3)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 1 and tile_out == 4.
        const dim3 grid_size = dim3(div_up(GEMM_HIDDEN_OUT, 4), div_up(num_tokens, 1), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(1, 4);
        launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 5);
    }
    else if (num_tokens == 4)
    {
        // When num_tokens == 1, the best tiling size is tile_token == 2 and tile_out == 2.
        const dim3 grid_size = dim3(div_up(GEMM_HIDDEN_OUT, 2), div_up(num_tokens, 2), 1);
        void* kernel_func = get_per_block_func_ptr_aligned_true_5120_(2, 2);
        launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, kernel_func, args, 5);
    }
    else
    {
        throw std::runtime_error("llama4QKVGemm: num_tokens larger than 4 is unsupported.");
    }
}

void llama4_qkv_gemm_op(
    int num_tokens, void const* A, void const* B, void* C, void const* scaling_factor, cudaStream_t stream)
{
    __nv_fp8_e4m3 const* A_fp8 = static_cast<__nv_fp8_e4m3 const*>(A);
    __nv_fp8_e4m3 const* B_fp8 = static_cast<__nv_fp8_e4m3 const*>(B);
    __nv_bfloat16* C_bf16 = static_cast<__nv_bfloat16*>(C);
    float const* __restrict__ scaling_factor_float = static_cast<float const*>(scaling_factor);
    llama4_qkv_gemv_kernel_launcher(A_fp8, B_fp8, C_bf16, scaling_factor_float, num_tokens, GEMM_HIDDEN_IN, GEMM_HIDDEN_OUT, stream);
}

} // namespace tensorrt_llm::kernels::llama4_qkv_gemm
