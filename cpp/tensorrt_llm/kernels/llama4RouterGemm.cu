/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/llama4RouterGemm.h"

#define GEMM_K 5120
#define BLOCK_SIZE 256
#define NUM_EXPERTS 128

namespace tensorrt_llm::kernels::llama4_router_gemm
{

#define VEC_SIZE 4

struct __align__(8) aligned_bf16x4
{
    __align__(8) __nv_bfloat16 data[VEC_SIZE];
};

// This is the hand-optimized kernel by Po-Han.
__global__ void gemv_kernel(int num_tokens,
    __nv_bfloat16 const* __restrict__ A, // Input vector [num_tokens][5120]
    __nv_bfloat16 const* __restrict__ B, // Input matrix [128][5120]
    __nv_bfloat16* __restrict__ C        // Output vector [num_tokens][128]
)
{
    // Shared memory for block reduction
    __shared__ float reduce_buffer[BLOCK_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum;
    thread_sum.x = 0.0f;
    thread_sum.y = 0.0f;

    // Each thread processes 4 elements at a time, 5 times
    int const token_idx = blockIdx.x / NUM_EXPERTS;
    int const row = blockIdx.x % NUM_EXPERTS; // Matrix row / Output element index
    int const tid = threadIdx.x;              // Thread ID within the block

    // FDL prefetch all B data
    aligned_bf16x4 b_vec[GEMM_K / BLOCK_SIZE / VEC_SIZE];
#pragma unroll
    for (int chunk = 0; chunk < GEMM_K / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        // Base index for this chunk
        int base_idx = chunk * BLOCK_SIZE + tid;

        // Load 4 elements at once
        b_vec[chunk] = reinterpret_cast<aligned_bf16x4 const*>(B)[row * GEMM_K / VEC_SIZE + base_idx];
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // Process 5 chunks of 4 elements each
#pragma unroll
    for (int chunk = 0; chunk < GEMM_K / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        // Base index for this chunk
        int base_idx = chunk * BLOCK_SIZE + tid;

        // Load 4 elements at once
        aligned_bf16x4 a_vec = reinterpret_cast<aligned_bf16x4 const*>(A)[token_idx * GEMM_K / VEC_SIZE + base_idx];
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i += 2)
        {

            float2 a_val = make_float2(a_vec.data[i], a_vec.data[i + 1]);
            float2 b_val = make_float2(b_vec[chunk].data[i], b_vec[chunk].data[i + 1]);

#if __CUDA_ARCH__ >= 1000
            thread_sum = __ffma2_rn(a_val, b_val, thread_sum);
#else
            thread_sum.x += a_val.x * b_val.x;
            thread_sum.y += a_val.y * b_val.y;
#endif
        }
    }

    // Warp-level reduction
    float warp_sum = thread_sum.x + thread_sum.y;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    // First thread in each warp writes to shared memory
    if (tid % warpSize == 0)
    {
        reduce_buffer[tid / warpSize] = warp_sum;
    }
    __syncthreads();

    // Final thread reduces across warps and writes the result
    if (tid == 0)
    {
        float block_sum = 0.0f;
        for (int i = 0; i < BLOCK_SIZE / warpSize; i++)
        {
            block_sum += reduce_buffer[i];
        }
        C[token_idx * NUM_EXPERTS + row] = __float2bfloat16(block_sum);
    }
}

// Function to launch kernel using FDL (Flexible Dispatch Layer)
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
    config.numAttrs = 0;
    attrs[config.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[config.numAttrs++].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchKernelExC(&config, (void const*) kernel_func, args);
}

void gemv_kernel_launcher(
    int num_tokens, __nv_bfloat16 const* A, __nv_bfloat16 const* B, __nv_bfloat16* C, cudaStream_t stream)
{

    int const grid_size = NUM_EXPERTS * num_tokens;

    void* args[] = {(void*) &num_tokens, (void*) &A, (void*) &B, (void*) &C};
    launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) gemv_kernel, args, 4);
}

void llama4_router_gemm_op(int num_tokens, void const* A, void const* B, void* C, cudaStream_t stream)
{
    __nv_bfloat16 const* A_bf16 = static_cast<__nv_bfloat16 const*>(A);
    __nv_bfloat16 const* B_bf16 = static_cast<__nv_bfloat16 const*>(B);
    __nv_bfloat16* C_bf16 = static_cast<__nv_bfloat16*>(C);

    gemv_kernel_launcher(num_tokens, A_bf16, B_bf16, C_bf16, stream);
}

} // namespace tensorrt_llm::kernels::llama4_router_gemm
