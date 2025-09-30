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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <vector>

namespace tensorrt_llm::kernels::llama4_min_latency::llama4_moe
{

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

} // namespace tensorrt_llm::kernels::llama4_min_latency::llama4_moe
