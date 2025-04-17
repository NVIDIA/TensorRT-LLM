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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>

#define ENABLE_FDL 1

namespace tensorrt_llm
{
namespace kernels
{

// Special struct for top1 result. Need to swap val and idx to because of little endian.
struct __attribute__((packed)) __align__(4) top1_result
{
    short idx;
    short val;
};

// Launch moe_mlp_fc13_swiglu_fp8_5120 kernel
void launch_moe_mlp_fc13_swiglu_fp8_5120(int num_tokens, int num_experts,
    __nv_fp8_e4m3 const* __restrict__ A,      // Input tensor A [num_tokens][HIDDEN_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,      // Input tensor B [num_experts][INTER_SIZE*2][HIDDEN_SIZE]
    __nv_bfloat16 const* __restrict__ logits, // Input tensor logits [num_tokens][num_experts]
    __nv_fp8_e4m3* __restrict__ C,            // Output tensor [num_tokens][INTER_SIZE]
    int* __restrict__ exp_idx,                // Output tensor [num_tokens]
    float const* __restrict__ in_scales,      // Input scales [num_experts]
    float const* __restrict__ out_scale_inv,  // Output scale [1]
    cudaStream_t stream);

// Launch moe_fc_fp8_bf16_1024 kernel
void launch_moe_fc_fp8_bf16_1024(int num_tokens, int num_experts,
    __nv_fp8_e4m3 const* __restrict__ A,       // Input tensor A [num_tokens][INTER_SIZE]
    __nv_fp8_e4m3 const* __restrict__ B,       // Input tensor B [num_experts][HIDDEN_SIZE][INTER_SIZE]
    int const* __restrict__ exp_idx,           // Input tensor exp_idx [num_tokens].
    __nv_bfloat16* __restrict__ C,             // Output tensor [num_tokens][HIDDEN_SIZE]
    float const* __restrict__ scaling_factors, // Scaling factors [num_experts]
    cudaStream_t stream);

// Launch moe_mlp_fc13_swiglu_fp8_5120 and moe_fc_fp8_bf16_1024.
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
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
