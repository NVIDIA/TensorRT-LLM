/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4MinLatencyMoEOp.h"
#include "tensorrt_llm/kernels/llama4MinLatencyKernels/llama4Utils.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#define NUM_EXPERTS 128
#define HIDDEN_SIZE 5120
#define INTER_SIZE 1024
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define TILE_ROW 4
#define VEC_SIZE 8

#define ENABLE_ACQBULK 1
#define ENABLE_PREFETCH 1
#define ENABLE_PREEXIT 1

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_moe
{

#define TOPK_VEC_SIZE 4
static_assert(NUM_EXPERTS == TOPK_VEC_SIZE * WARP_SIZE, "NUM_EXPERTS must be equal to TOPK_VEC_SIZE * WARP_SIZE");

// This is the hand-optimized kernel.
// The computation is:
//   C = silu(AxB_gated * in_scale * sigmoid(logit)) * (AxB_linear * in_scale * sigmoid(logit)) * out_scale_inv
// The out_scale_inv cannot be fused with in_scale because silu() is non-linear.
// Also, Llama-4 applies score scaling, which is sigmoid(logit), on tensor A.
__global__ void llama4_moe_fc13_swiglu_fp8_kernel(int num_tokens,
    __nv_fp8_e4m3 const* __restrict__ A,      // Input tensor [num_tokens][HIDDEN_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,      // Input tensor [num_experts][INTER_SIZE*2][HIDDEN_SIZE]
    __nv_bfloat16 const* __restrict__ logits, // Input tensor logits [num_tokens][num_experts]
    __nv_fp8_e4m3* __restrict__ C,            // Output tensor [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,                // Output tensor [num_tokens]
    float const* __restrict__ in_scales,      // Input scales [num_experts]
    float const* __restrict__ out_scale_inv   // Output scale [1]
)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    // Shared memory for block reduction
    __shared__ float reduce_buffer[2][BLOCK_SIZE / WARP_SIZE];

    // Each thread accumulates its partial sum
    float2 thread_sum_linear;
    thread_sum_linear.x = 0.0f;
    thread_sum_linear.y = 0.0f;

    float2 thread_sum_gate;
    thread_sum_gate.x = 0.0f;
    thread_sum_gate.y = 0.0f;

    // Each thread processes 8 elements at a time, 5 times
    int const token_idx = blockIdx.x / INTER_SIZE;
    int const row = blockIdx.x % INTER_SIZE; // Matrix row / Output element index
    int const tid = threadIdx.x;             // Thread ID within the block
    int const lane_id = tid % WARP_SIZE;     // Lane ID within the warp

    // Preload the scaling factors before ACQBULK.
    __shared__ float in_scales_shared[NUM_EXPERTS];
    in_scales_shared[tid] = in_scales[tid];

    // Logits depends on the previous kernel, so we cannot prefetch anything.
#if ENABLE_ACQBULK
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Perform top1 within the current thread, which processes 4 experts.
    aligned_bfloat16x4 logits_vec;
    logits_vec = reinterpret_cast<aligned_bfloat16x4 const*>(logits)[token_idx * NUM_EXPERTS / TOPK_VEC_SIZE + lane_id];

    __nv_bfloat16 best_logit = logits_vec.data[0];
    int base_exp = lane_id * TOPK_VEC_SIZE;
    int best_exp = base_exp;
#pragma unroll
    for (int i = 1; i < TOPK_VEC_SIZE; i++)
    {
        __nv_bfloat16 current_logit = logits_vec.data[i];
        if (current_logit >= best_logit)
        {
            best_logit = current_logit;
            best_exp = base_exp + i;
        }
    }

    // Perform top1 across threads using Warp reduction.
    // We pack logit and expert index into an int so that we can use integer max op for reduction.
    int best_result
        = ((int) (__bfloat16_as_short(best_logit) ^ (best_logit < __nv_bfloat16(0.f) ? 0x7fff : 0)) << 16) | best_exp;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        best_result = max(best_result, __shfl_down_sync(0xffffffff, best_result, offset));
    }

    // Broadcast the best result to all threads in the warp.
    best_result = __shfl_sync(0xffffffff, best_result, 0);

    // Extract the expert index and score.
    int expert_idx = best_result & 0xffff;
    float top_logit = __bfloat162float(__short_as_bfloat16(short((best_result >> 16) & 0xffff)));

    // Select the corresponding expert weight.
    int expert_weight_offset = expert_idx * 2 * INTER_SIZE * HIDDEN_SIZE / VEC_SIZE;

    // Process 5 chunks of 8 elements each
#pragma unroll
    for (int chunk = 0; chunk < HIDDEN_SIZE / BLOCK_SIZE / VEC_SIZE; chunk++)
    {
        int base_idx = chunk * BLOCK_SIZE + tid;
        // Load 8 elements at once
        aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * HIDDEN_SIZE / VEC_SIZE + base_idx];
        aligned_fp8x8 b_vec_linear
            = reinterpret_cast<aligned_fp8x8 const*>(B)[row * HIDDEN_SIZE / VEC_SIZE + base_idx + expert_weight_offset];
        aligned_fp8x8 b_vec_gate = reinterpret_cast<aligned_fp8x8 const*>(
            B)[(row + INTER_SIZE) * HIDDEN_SIZE / VEC_SIZE + base_idx + expert_weight_offset];

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val_linear = float4(b_vec_linear.data[i]);
            float4 b_val_gate = float4(b_vec_gate.data[i]);

            thread_sum_linear
                = ffma2(make_float2(a_val.x, a_val.y), make_float2(b_val_linear.x, b_val_linear.y), thread_sum_linear);
            thread_sum_linear
                = ffma2(make_float2(a_val.z, a_val.w), make_float2(b_val_linear.z, b_val_linear.w), thread_sum_linear);
            thread_sum_gate
                = ffma2(make_float2(a_val.x, a_val.y), make_float2(b_val_gate.x, b_val_gate.y), thread_sum_gate);
            thread_sum_gate
                = ffma2(make_float2(a_val.z, a_val.w), make_float2(b_val_gate.z, b_val_gate.w), thread_sum_gate);
        }
    }

    // Block reduction
    float warp_sum_linear = thread_sum_linear.x + thread_sum_linear.y;
    float warp_sum_gate = thread_sum_gate.x + thread_sum_gate.y;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        warp_sum_linear += __shfl_down_sync(0xffffffff, warp_sum_linear, offset);
        warp_sum_gate += __shfl_down_sync(0xffffffff, warp_sum_gate, offset);
    }

    if (tid % WARP_SIZE == 0)
    {
        reduce_buffer[0][tid / WARP_SIZE] = warp_sum_linear;
        reduce_buffer[1][tid / WARP_SIZE] = warp_sum_gate;
    }
    __syncthreads();

    if (tid == 0)
    {
        float block_sum_linear = warp_sum_linear;
        float block_sum_gate = warp_sum_gate;
#pragma unroll
        for (int i = 1; i < BLOCK_SIZE / WARP_SIZE; i++)
        {
            block_sum_linear += reduce_buffer[0][i];
            block_sum_gate += reduce_buffer[1][i];
        }
        float fused_in_scale = in_scales_shared[expert_idx] * sigmoid(top_logit);
        C[token_idx * INTER_SIZE + row] = __nv_fp8_e4m3(
            silu(block_sum_gate * fused_in_scale) * block_sum_linear * fused_in_scale * out_scale_inv[0]);
        if (row == 0)
        {
            exp_idx[token_idx] = expert_idx;
        }
    }

#if ENABLE_PREEXIT
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#endif
}

// Launch llama4_moe_fc13_swiglu_fp8_kernel
void launch_llama4_moe_fc13_swiglu_fp8_kernel(int num_tokens, int num_experts,
    void const* __restrict__ A,              // Input tensor A [num_tokens][HIDDEN_SIZE]
    void const* __restrict__ B,              // Input tensor B [num_experts][INTER_SIZE*2][HIDDEN_SIZE]
    void const* __restrict__ logits,         // Input tensor logits [num_tokens][num_experts]
    void* __restrict__ C,                    // Output tensor [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,               // Output tensor [num_tokens]
    float const* __restrict__ in_scales,     // Input scales [num_experts]
    float const* __restrict__ out_scale_inv, // Output scale [1]
    cudaStream_t stream)
{
    int const grid_size = num_tokens * INTER_SIZE;
    if (num_experts != NUM_EXPERTS)
    {
        printf("The implementation currently assumes num_experts = %d\n", NUM_EXPERTS);
        exit(1);
    }

    void* args[] = {(void*) &num_tokens, (void*) &A, (void*) &B, (void*) &logits, (void*) &C, (void*) &exp_idx,
        (void*) &in_scales, (void*) &out_scale_inv};
    launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) llama4_moe_fc13_swiglu_fp8_kernel, args, 8);
}

// This is the hand-optimized kernel.
__global__ void llama4_moe_fc2_fp8_kernel(int num_tokens,
    __nv_fp8_e4m3 const* __restrict__ A,      // Input tensor A [num_tokens][INTER_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,      // Input tensor B [num_experts][HIDDEN_SIZE][INTER_SIZE]
    int const* __restrict__ exp_idx,          // Input tensor exp_idx [num_tokens].
    __nv_bfloat16* __restrict__ C,            // Output tensor [num_tokens][HIDDEN_SIZE]
    float const* __restrict__ scaling_factors // Scaling factors [num_experts]
)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1200))
    // Shared memory for block reduction
    __shared__ float reduce_buffer[TILE_ROW][BLOCK_SIZE / WARP_SIZE];

    // Each thread processes 8 elements at a time, 5 times
    int const token_idx = blockIdx.x / (HIDDEN_SIZE / TILE_ROW);
    int const row = blockIdx.x % (HIDDEN_SIZE / TILE_ROW); // Matrix row / Output element index
    int const tid = threadIdx.x;                           // Thread ID within the block

    // Preload the scaling factors before ACQBULK.
    __shared__ float scaling_factors_shared[NUM_EXPERTS];
    scaling_factors_shared[tid] = scaling_factors[tid];

#if ENABLE_ACQBULK
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

    // Select the corresponding expert weight.
    int expert_idx = exp_idx[token_idx];
    int expert_weight_offset = expert_idx * HIDDEN_SIZE * INTER_SIZE / VEC_SIZE;
    int base_idx = tid;

    // Load 8 elements at once.
    aligned_fp8x8 a_vec = reinterpret_cast<aligned_fp8x8 const*>(A)[token_idx * INTER_SIZE / VEC_SIZE + base_idx];
    aligned_fp8x8 b_vec[TILE_ROW];
#pragma unroll
    for (int tile_row_idx = 0; tile_row_idx < TILE_ROW; tile_row_idx++)
    {
        int row_current = tile_row_idx + row * TILE_ROW;
        b_vec[tile_row_idx] = reinterpret_cast<aligned_fp8x8 const*>(
            B)[row_current * INTER_SIZE / VEC_SIZE + base_idx + expert_weight_offset];
    }

    // Loop over TILE_ROW times to compute the gemm result.
#pragma unroll
    for (int tile_row_idx = 0; tile_row_idx < TILE_ROW; tile_row_idx++)
    {
        // Each thread accumulates its partial sum
        float2 thread_sum = {0.0f, 0.0f};
        thread_sum.x = 0.0f;
        thread_sum.y = 0.0f;

#pragma unroll
        for (int i = 0; i < VEC_SIZE / 4; i++)
        {
            float4 a_val = float4(a_vec.data[i]);
            float4 b_val = float4(b_vec[tile_row_idx].data[i]);
            thread_sum = ffma2(make_float2(a_val.x, a_val.y), make_float2(b_val.x, b_val.y), thread_sum);
            thread_sum = ffma2(make_float2(a_val.z, a_val.w), make_float2(b_val.z, b_val.w), thread_sum);
        }

        // Warp reduction
        float warp_sum = thread_sum.x + thread_sum.y;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (tid % WARP_SIZE == 0)
        {
            reduce_buffer[tile_row_idx][tid / WARP_SIZE] = warp_sum;
        }
    }

    __syncthreads();

    // Use the first TILE_ROW threads to do block reduction and writes the result.
    if (tid < TILE_ROW)
    {
        int row_current = tid + row * TILE_ROW;
        float block_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; i++)
        {
            block_sum += reduce_buffer[tid][i];
        }
        float scaling_factor = scaling_factors_shared[expert_idx];
        C[token_idx * HIDDEN_SIZE + row_current] = __float2bfloat16(block_sum * scaling_factor);
    }

#if ENABLE_PREEXIT
    asm volatile("griddepcontrol.launch_dependents;");
#endif
#endif
}

void launch_llama4_moe_fc2_fp8_kernel(int num_tokens, int num_experts,
    void const* __restrict__ A,                // Input tensor A [num_tokens][INTER_SIZE]
    void const* __restrict__ B,                // Input tensor B [num_experts][HIDDEN_SIZE][INTER_SIZE]
    int const* __restrict__ exp_idx,           // Input tensor exp_idx [num_tokens].
    void* __restrict__ C,                      // Output tensor [num_tokens][HIDDEN_SIZE]
    float const* __restrict__ scaling_factors, // Scaling factors [num_experts]
    cudaStream_t stream)
{
    if (num_experts != NUM_EXPERTS)
    {
        printf("Current implementation assumes num_experts == %d\n", NUM_EXPERTS);
        exit(1);
    }
    int const grid_size = num_tokens * HIDDEN_SIZE / TILE_ROW;

    void* args[]
        = {(void*) &num_tokens, (void*) &A, (void*) &B, (void*) &exp_idx, (void*) &C, (void*) &scaling_factors};
    launch_kernel_pdl(dim3(grid_size), dim3(BLOCK_SIZE), stream, (void*) llama4_moe_fc2_fp8_kernel, args, 6);
}

void run_moe_llama4_tp8ep1_min_latency(int num_tokens, int num_experts,
    void const* __restrict__ input_activations_void,  // Input tensor FP8 [num_tokens][HIDDEN_SIZE]
    void const* __restrict__ router_logits_void,      // Router logits tensor BF16 [num_tokens][num_experts]
    void const* __restrict__ fc1_expert_weights_void, // FC13 weight tensor FP8 [num_experts][2*INTER_SIZE][HIDDEN_SIZE]
    void const* __restrict__ fc2_expert_weights_void, // FC2 weight tensor FP8 [num_experts][HIDDEN_SIZE][INTER_SIZE]
    float const* __restrict__ dequant_fc1,            // FC1 out scale factor FP32 [num_experts]
    float const* __restrict__ quant_fc2,              // FC2 input scaling factor FP32 [1]
    float const* __restrict__ dequant_fc2,            // FC2 out scaling factor FP32 [num_experts]
    void* __restrict__ fc2_input_activations_void,    // FC2 input tensor FP8 [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,                        // Expert indexes INT [num_tokens]
    void* __restrict__ output_void,                   // FC2 output tensor BF16 [num_tokens][HIDDEN_SIZE]
    cudaStream_t stream)
{
    launch_llama4_moe_fc13_swiglu_fp8_kernel(num_tokens, num_experts, input_activations_void, fc1_expert_weights_void,
        router_logits_void, fc2_input_activations_void, exp_idx, dequant_fc1, quant_fc2, stream);
    launch_llama4_moe_fc2_fp8_kernel(num_tokens, num_experts, fc2_input_activations_void, fc2_expert_weights_void,
        exp_idx, output_void, dequant_fc2, stream);
}

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_moe
