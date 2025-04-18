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

#include "llama4GemmSwiGLU.h"

#define HIDDEN_IN 5120
#define BLOCK_SIZE 128
#define WARP_SIZE 32

#define ENABLE_ACQBULK 1
#define ENABLE_PREFETCH 1
#define ENABLE_PREEXIT 1

namespace tensorrt_llm::kernels::llama4_fc_swiglu
{

// Use 8 for now, which results in LDG.64.
#define VEC_SIZE 8

struct __align__(8) aligned_fp8x8
{
    __align__(8) __nv_fp8x4_e4m3 data[2];
};

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + __expf(-x));
}

// Hidden out is 2048 for MLP, and 1024 for shared expert.
// This is the hand-optimized kernel by Po-Han.
// fp8 in, fp8 out
__global__ void llama4_fc_swiglu_fp8_5120(int num_tokens, int hidden_out,
    __nv_fp8_e4m3 const* __restrict__ A,    // Input vector [num_tokens][5120]
    __nv_fp8_e4m3 const* __restrict__ B,    // Input matrix [4096][5120]
    __nv_fp8_e4m3* __restrict__ C,          // Output vector [num_tokens][2048]
    float const* __restrict__ in_scale,     // Input scaling factor
    float const* __restrict__ out_scale_inv // Output scaling factor inverse
)
{
    // Shared memory for block reduction
    __shared__ float reduce_buffer[2][BLOCK_SIZE / WARP_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum_gate;
    thread_sum_gate.x = 0.0f;
    thread_sum_gate.y = 0.0f;

    float2 thread_sum_linear;
    thread_sum_linear.x = 0.0f;
    thread_sum_linear.y = 0.0f;

    // Each thread processes 8 elements at a time, 5 times
    int const token_idx = blockIdx.x / hidden_out;
    int const row = blockIdx.x % hidden_out; // Matrix row / Output element index
    int const tid = threadIdx.x;             // Thread ID within the block

    // Prefetch for tensor B to L2
#if ENABLE_ACQBULK && ENABLE_PREFETCH && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#pragma unroll
    for (int chunk = 0; chunk < HIDDEN_IN / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;
        __nv_fp8_e4m3 const* ptr_b_gate = &B[row * HIDDEN_IN / VEC_SIZE + base_idx];
        __nv_fp8_e4m3 const* ptr_b_linear = &B[(row + hidden_out) * HIDDEN_IN / VEC_SIZE + base_idx];
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr_b_gate) : "memory");
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr_b_linear) : "memory");
    }
#endif

#if ENABLE_ACQBULK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Process 5 chunks of 8 elements each
#pragma unroll
    for (int chunk = 0; chunk < HIDDEN_IN / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;

        // Load 8 elements at once
        aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec_gate = reinterpret_cast<aligned_fp8x8 const*>(B)[row * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec_linear
            = reinterpret_cast<aligned_fp8x8 const*>(B)[(row + hidden_out) * HIDDEN_IN / VEC_SIZE + base_idx];

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val_gate = float4(b_vec_gate.data[i]);
            float4 b_val_linear = float4(b_vec_linear.data[i]);

#if __CUDA_ARCH__ >= 1000
            thread_sum_gate
                = __ffma2_rn(make_float2(a_val.x, a_val.y), make_float2(b_val_gate.x, b_val_gate.y), thread_sum_gate);
            thread_sum_gate
                = __ffma2_rn(make_float2(a_val.z, a_val.w), make_float2(b_val_gate.z, b_val_gate.w), thread_sum_gate);
            thread_sum_linear = __ffma2_rn(
                make_float2(a_val.x, a_val.y), make_float2(b_val_linear.x, b_val_linear.y), thread_sum_linear);
            thread_sum_linear = __ffma2_rn(
                make_float2(a_val.z, a_val.w), make_float2(b_val_linear.z, b_val_linear.w), thread_sum_linear);
#else
            thread_sum_gate.x += a_val.x * b_val_gate.x;
            thread_sum_gate.y += a_val.y * b_val_gate.y;
            thread_sum_gate.x += a_val.z * b_val_gate.z;
            thread_sum_gate.y += a_val.w * b_val_gate.w;
            thread_sum_linear.x += a_val.x * b_val_linear.x;
            thread_sum_linear.y += a_val.y * b_val_linear.y;
            thread_sum_linear.x += a_val.z * b_val_linear.z;
            thread_sum_linear.y += a_val.w * b_val_linear.w;
#endif
        }
    }

    // Block reduction
    float warp_sum_gate = thread_sum_gate.x + thread_sum_gate.y;
    float warp_sum_linear = thread_sum_linear.x + thread_sum_linear.y;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        warp_sum_gate += __shfl_down_sync(0xffffffff, warp_sum_gate, offset);
        warp_sum_linear += __shfl_down_sync(0xffffffff, warp_sum_linear, offset);
    }

    if (tid % WARP_SIZE == 0)
    {
        reduce_buffer[0][tid / WARP_SIZE] = warp_sum_gate;
        reduce_buffer[1][tid / WARP_SIZE] = warp_sum_linear;
    }
    __syncthreads();

    if (tid == 0)
    {
        float block_sum_gate = warp_sum_gate;
        float block_sum_linear = warp_sum_linear;
#pragma unroll
        for (int i = 1; i < BLOCK_SIZE / WARP_SIZE; i++)
        {
            block_sum_gate += reduce_buffer[0][i];
            block_sum_linear += reduce_buffer[1][i];
        }
        float in_scale_val = in_scale[0];
        C[token_idx * hidden_out + row]
            = __nv_fp8_e4m3(silu(block_sum_gate * in_scale_val) * block_sum_linear * in_scale_val * out_scale_inv[0]);
    }

#if ENABLE_PREEXIT && (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// fp8 in, bf16 out
__global__ void llama4_fc_swiglu_bf16_5120(int num_tokens, int hidden_out,
    __nv_fp8_e4m3 const* __restrict__ A,    // Input vector [num_tokens][5120]
    __nv_fp8_e4m3 const* __restrict__ B,    // Input matrix [4096][5120]
    __nv_bfloat16* __restrict__ C,          // Output vector [num_tokens][2048]
    float const* __restrict__ in_scale,     // Input scaling factor
    float const* __restrict__ out_scale_inv // Output scaling factor inverse
)
{
    // Shared memory for block reduction
    __shared__ float reduce_buffer[2][BLOCK_SIZE / WARP_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum_gate;
    thread_sum_gate.x = 0.0f;
    thread_sum_gate.y = 0.0f;

    float2 thread_sum_linear;
    thread_sum_linear.x = 0.0f;
    thread_sum_linear.y = 0.0f;

    // Each thread processes 8 elements at a time, 5 times
    int const token_idx = blockIdx.x / hidden_out;
    int const row = blockIdx.x % hidden_out; // Matrix row / Output element index
    int const tid = threadIdx.x;             // Thread ID within the block

    // Prefetch for tensor B to L2
#if ENABLE_ACQBULK && ENABLE_PREFETCH && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#pragma unroll
    for (int chunk = 0; chunk < HIDDEN_IN / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;
        __nv_fp8_e4m3 const* ptr_b_gate = &B[row * HIDDEN_IN / VEC_SIZE + base_idx];
        __nv_fp8_e4m3 const* ptr_b_linear = &B[(row + hidden_out) * HIDDEN_IN / VEC_SIZE + base_idx];
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr_b_gate) : "memory");
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr_b_linear) : "memory");
    }
#endif

#if ENABLE_ACQBULK && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Process 5 chunks of 8 elements each
#pragma unroll
    for (int chunk = 0; chunk < HIDDEN_IN / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;

        // Load 8 elements at once
        aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec_gate = reinterpret_cast<aligned_fp8x8 const*>(B)[row * HIDDEN_IN / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec_linear
            = reinterpret_cast<aligned_fp8x8 const*>(B)[(row + hidden_out) * HIDDEN_IN / VEC_SIZE + base_idx];

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val_gate = float4(b_vec_gate.data[i]);
            float4 b_val_linear = float4(b_vec_linear.data[i]);

#if __CUDA_ARCH__ >= 1000
            thread_sum_gate
                = __ffma2_rn(make_float2(a_val.x, a_val.y), make_float2(b_val_gate.x, b_val_gate.y), thread_sum_gate);
            thread_sum_gate
                = __ffma2_rn(make_float2(a_val.z, a_val.w), make_float2(b_val_gate.z, b_val_gate.w), thread_sum_gate);
            thread_sum_linear = __ffma2_rn(
                make_float2(a_val.x, a_val.y), make_float2(b_val_linear.x, b_val_linear.y), thread_sum_linear);
            thread_sum_linear = __ffma2_rn(
                make_float2(a_val.z, a_val.w), make_float2(b_val_linear.z, b_val_linear.w), thread_sum_linear);
#else
            thread_sum_gate.x += a_val.x * b_val_gate.x;
            thread_sum_gate.y += a_val.y * b_val_gate.y;
            thread_sum_gate.x += a_val.z * b_val_gate.z;
            thread_sum_gate.y += a_val.w * b_val_gate.w;
            thread_sum_linear.x += a_val.x * b_val_linear.x;
            thread_sum_linear.y += a_val.y * b_val_linear.y;
            thread_sum_linear.x += a_val.z * b_val_linear.z;
            thread_sum_linear.y += a_val.w * b_val_linear.w;
#endif
        }
    }

    // Block reduction
    float warp_sum_gate = thread_sum_gate.x + thread_sum_gate.y;
    float warp_sum_linear = thread_sum_linear.x + thread_sum_linear.y;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        warp_sum_gate += __shfl_down_sync(0xffffffff, warp_sum_gate, offset);
        warp_sum_linear += __shfl_down_sync(0xffffffff, warp_sum_linear, offset);
    }

    if (tid % WARP_SIZE == 0)
    {
        reduce_buffer[0][tid / WARP_SIZE] = warp_sum_gate;
        reduce_buffer[1][tid / WARP_SIZE] = warp_sum_linear;
    }
    __syncthreads();

    if (tid == 0)
    {
        float block_sum_gate = warp_sum_gate;
        float block_sum_linear = warp_sum_linear;
#pragma unroll
        for (int i = 1; i < BLOCK_SIZE / WARP_SIZE; i++)
        {
            block_sum_gate += reduce_buffer[0][i];
            block_sum_linear += reduce_buffer[1][i];
        }
        float in_scale_val = in_scale[0];
        C[token_idx * hidden_out + row]
            = __nv_bfloat16(silu(block_sum_gate * in_scale_val) * block_sum_linear * in_scale_val * out_scale_inv[0]);
    }

#if ENABLE_PREEXIT && (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
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
    config.numAttrs = 1;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchKernelExC(&config, (void const*) kernel_func, args);
}

void fc_swiglu_fp8_launcher(int num_tokens, int hidden_out, void const* __restrict__ A, void const* __restrict__ B,
    void* __restrict__ C, void const* in_scale, void const* out_scale_inv, cudaStream_t stream)
{
    dim3 grid_size = dim3(num_tokens * hidden_out);
    dim3 block_size = dim3(BLOCK_SIZE);
    void* args[] = {(void*) &num_tokens, (void*) &hidden_out, (void*) &A, (void*) &B, (void*) &C, (void*) &in_scale,
        (void*) &out_scale_inv};

    launch_kernel_fdl(dim3(grid_size), dim3(block_size), stream, (void*) llama4_fc_swiglu_fp8_5120, args, 7);
}

void llama4_fc_swiglu_fp8_op(int num_tokens, int hidden_out, void const* A, void const* B, void* C,
    void const* in_scale, void const* out_scale_inv, cudaStream_t stream)
{
    fc_swiglu_fp8_launcher(num_tokens, hidden_out, A, B, C, in_scale, out_scale_inv, stream);
}

void fc_swiglu_bf16_launcher(int num_tokens, int hidden_out, void const* __restrict__ A, void const* __restrict__ B,
    void* __restrict__ C, void const* in_scale, void const* out_scale_inv, cudaStream_t stream)
{
    dim3 grid_size = dim3(num_tokens * hidden_out);
    dim3 block_size = dim3(BLOCK_SIZE);
    void* args[] = {(void*) &num_tokens, (void*) &hidden_out, (void*) &A, (void*) &B, (void*) &C, (void*) &in_scale,
        (void*) &out_scale_inv};

    launch_kernel_fdl(dim3(grid_size), dim3(block_size), stream, (void*) llama4_fc_swiglu_bf16_5120, args, 7);
}

void llama4_fc_swiglu_bf16_op(int num_tokens, int hidden_out, void const* A, void const* B, void* C,
    void const* in_scale, void const* out_scale_inv, cudaStream_t stream)
{
    fc_swiglu_bf16_launcher(num_tokens, hidden_out, A, B, C, in_scale, out_scale_inv, stream);
}

} // namespace tensorrt_llm::kernels::llama4_fc_swiglu
