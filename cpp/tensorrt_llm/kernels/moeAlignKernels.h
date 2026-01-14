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
#include <cuda_runtime.h>
#include <stdint.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * @brief Aligns token distribution across experts to be compatible with block size for matrix multiplication.
 *
 * This kernel sorts tokens by expert assignment and pads the distribution to match block size requirements.
 * Inspired by vLLM's moe_align_kernel and ported to TensorRT-LLM.
 *
 * @param topk_ids Input tensor with expert IDs per token [total_tokens, top_k]
 * @param topk_ids_dtype_size Size of the dtype (e.g., sizeof(int32_t) or sizeof(int64_t))
 * @param sorted_token_ids Output tensor for sorted token indices
 * @param expert_ids Output tensor for expert IDs per block
 * @param num_tokens_post_pad Output tensor for total tokens after padding (single int32)
 * @param num_experts Total number of experts
 * @param block_size Block size for matrix multiplication alignment
 * @param numel Total number of elements in topk_ids (topk_ids.numel())
 * @param max_num_tokens_padded Maximum number of tokens after padding (sorted_token_ids.size(0))
 * @param stream CUDA stream for kernel execution
 */
void invokeMoeAlignBlockSize(void const* topk_ids, int32_t topk_ids_dtype_size, int32_t* sorted_token_ids,
    int32_t* expert_ids, int32_t* num_tokens_post_pad, int32_t num_experts, int32_t block_size, int32_t numel,
    int32_t max_num_tokens_padded, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
