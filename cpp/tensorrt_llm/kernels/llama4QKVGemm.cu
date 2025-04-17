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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define HIDDEN_IN 5120
#define HIDDEN_OUT 896 // This is QKV_GEMM. Replaced with 4096 for MLP_FC1.
#define BLOCK_SIZE 128
#define ENABLE_FDL 1

namespace tensorrt_llm::kernels::llama4_qkv_gemm
{

// GEMV kernel that processes fp8 inputs and produces bfloat16 output

// Use 8 for now, which results in LDG.64.
#define VEC_SIZE 8

struct __align__(8) aligned_fp8x8
{
    __align__(8) __nv_fp8x4_e4m3 data[2];
};

// This is the hand-optimized kernel by Po-Han.
__global__ void llama4_qkv_gemv_kernel(int num_tokens,
    __nv_fp8_e4m3 const* __restrict__ A,     // Input vector [num_tokens][5120]
    __nv_fp8_e4m3 const* __restrict__ B,     // Input matrix [896][5120]
    __nv_bfloat16* __restrict__ C,           // Output vector [num_tokens][896]
    float const* __restrict__ scaling_factor // New scaling factor parameter
)
{
    // Shared memory for block reduction
    __shared__ float reduce_buffer[BLOCK_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum;
    thread_sum.x = 0.0f;
    thread_sum.y = 0.0f;

    // Each thread processes 8 elements at a time, 5 times
    int const token_idx = blockIdx.x / HIDDEN_OUT;
    int const row = blockIdx.x % HIDDEN_OUT; // Matrix row / Output element index
    int const tid = threadIdx.x;             // Thread ID within the block

    {
        // FDL prefetch for chunk 0
        int base_idx = tid;

        // Load 8 elements at once
        // asm volatile("griddepcontrol.wait;" ::: "memory");
        aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec = reinterpret_cast<aligned_fp8x8 const*>(B)[row * HIDDEN_IN / VEC_SIZE + base_idx];

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val = float4(b_vec.data[i]);

#if __CUDA_ARCH__ >= 1000
            thread_sum = __ffma2_rn(make_float2(a_val.x, a_val.y), make_float2(b_val.x, b_val.y), thread_sum);
            thread_sum = __ffma2_rn(make_float2(a_val.z, a_val.w), make_float2(b_val.z, b_val.w), thread_sum);
#else
            thread_sum.x += a_val.x * b_val.x;
            thread_sum.y += a_val.y * b_val.y;
            thread_sum.x += a_val.z * b_val.z;
            thread_sum.y += a_val.w * b_val.w;
#endif
        }
    }

    // Process 5 chunks of 8 elements each
#pragma unroll
    for (int chunk = 1; chunk < HIDDEN_IN / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;

        // Load 8 elements at once
        aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec = reinterpret_cast<aligned_fp8x8 const*>(B)[row * HIDDEN_IN / VEC_SIZE + base_idx];

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val = float4(b_vec.data[i]);

#if __CUDA_ARCH__ >= 1000
            thread_sum = __ffma2_rn(make_float2(a_val.x, a_val.y), make_float2(b_val.x, b_val.y), thread_sum);
            thread_sum = __ffma2_rn(make_float2(a_val.z, a_val.w), make_float2(b_val.z, b_val.w), thread_sum);
#else
            thread_sum.x += a_val.x * b_val.x;
            thread_sum.y += a_val.y * b_val.y;
            thread_sum.x += a_val.z * b_val.z;
            thread_sum.y += a_val.w * b_val.w;
#endif
        }
    }

    float warp_sum = thread_sum.x + thread_sum.y;
#pragma unroll
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

    if (tid == 0)
    {
        float block_sum = warp_sum;
#pragma unroll
        for (int i = 1; i < BLOCK_SIZE / warpSize; i++)
        {
            block_sum += reduce_buffer[i];
        }
        C[token_idx * HIDDEN_OUT + row] = __float2bfloat16(block_sum * scaling_factor[0]);
    }
}

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

void llama4_qkv_gemv_kernel_launcher(int num_tokens, __nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, __nv_bfloat16* C,
    float const* scaling_factor, cudaStream_t stream)
{
    int const grid_size = HIDDEN_OUT * num_tokens;
    void* args[] = {(void*) &num_tokens, (void*) &A, (void*) &B, (void*) &C, (void*) &scaling_factor};
    launch_kernel_fdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) llama4_qkv_gemv_kernel, args, 5);
}

void llama4_qkv_gemm_op(
    int num_tokens, void const* A, void const* B, void* C, void const* scaling_factor, cudaStream_t stream)
{
    __nv_fp8_e4m3 const* A_fp8 = static_cast<__nv_fp8_e4m3 const*>(A);
    __nv_fp8_e4m3 const* B_fp8 = static_cast<__nv_fp8_e4m3 const*>(B);
    __nv_bfloat16* C_bf16 = static_cast<__nv_bfloat16*>(C);
    float const* __restrict__ scaling_factor_float = static_cast<float const*>(scaling_factor);
    llama4_qkv_gemv_kernel_launcher(num_tokens, A_fp8, B_fp8, C_bf16, scaling_factor_float, stream);
}

} // namespace tensorrt_llm::kernels::llama4_qkv_gemm
