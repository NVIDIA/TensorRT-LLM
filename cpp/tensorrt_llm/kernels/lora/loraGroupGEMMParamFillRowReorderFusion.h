/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * @brief Fused kernel that fills group GEMM parameters, performs row reordering, zero fillings for CUDA graph
 * compatible LoRA.
 *
 * @param in_sizes Output: [module_count, max_lora_count, 3] problem sizes for first GEMM
 * @param out_sizes Output: [module_count, max_lora_count, 3] problem sizes for second GEMM
 * @param a_ptrs Output: [module_count, max_lora_count] input matrix pointers
 * @param d_ptrs Output: [module_count, max_lora_count] intermediate output pointers
 * @param d_prime_ptrs Output: [module_count, max_lora_count] final output pointers
 * @param lda Output: [module_count, max_lora_count] leading dimensions for A matrices
 * @param ldd Output: [module_count, max_lora_count] leading dimensions for D matrices
 * @param ldb_prime Output: [module_count, max_lora_count] leading dimensions for B' matrices
 * @param ldd_prime Output: [module_count, max_lora_count] leading dimensions for D' matrices
 * @param splitk_offsets Output: [module_count, max_lora_count] split-K work offsets
 * @param reordered_input Output: [batch_size, input_hidden_size] reordered input matrix
 * @param max_lora_count Maximum number of LoRA adapters
 * @param max_lora_rank Maximum rank of LoRA adapters
 * @param sum_output_hidden_size Sum of output hidden sizes across modules
 * @param input_hidden_size Input hidden dimension
 * @param dtype_element_size Size of data type in bytes
 * @param batch_size Batch size
 * @param a_base Base pointer for input matrices
 * @param d_base Base pointer for intermediate output matrices
 * @param d_prime_base Base pointer for final output matrices
 * @param slot_counts Input: [max_lora_count] number of requests per LoRA slot
 * @param slot_ranks Input: [max_lora_count] rank of each LoRA adapter
 * @param slot_offsets Input: [max_lora_count + 1] cumulative offsets (last element = total rows)
 * @param module_out_sizes Input: [module_count] output hidden size per module
 * @param module_out_prefix Input: [module_count] prefix sum of output hidden sizes
 * @param b_ptrs Input: [module_count, max_lora_count] weight pointers for first GEMM
 * @param b_prime_ptrs Input: [module_count, max_lora_count] weight pointers for second GEMM
 * @param input Input: [batch_size, input_hidden_size] original input matrix
 * @param sorted_ids Input: [batch_size] indices for row reordering
 * @param module_count Number of modules per layer
 * @param dtype Data type of matrices
 * @param stream CUDA stream
 */
void launchLoraGroupGEMMParamFillRowReorderFusion(int32_t* in_sizes, int32_t* out_sizes, int64_t* a_ptrs,
    int64_t* d_ptrs, int64_t* d_prime_ptrs, int64_t* lda, int64_t* ldd, int64_t* ldb_prime, int64_t* ldd_prime,
    int64_t* splitk_offsets, void* reordered_input, int32_t max_lora_count, int32_t max_lora_rank,
    int32_t sum_output_hidden_size, int32_t input_hidden_size, int64_t dtype_element_size, int64_t batch_size,
    int64_t a_base, int64_t d_base, int64_t d_prime_base, int32_t const* slot_counts, int32_t const* slot_ranks,
    int64_t const* slot_offsets, int32_t const* module_out_sizes, int64_t const* module_out_prefix,
    int64_t const* b_ptrs, int64_t const* b_prime_ptrs, void const* input, int64_t const* sorted_ids,
    int32_t module_count, nvinfer1::DataType dtype, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
