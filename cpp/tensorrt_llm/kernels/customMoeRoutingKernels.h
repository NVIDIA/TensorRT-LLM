/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
template <typename InputT, typename OutputT, typename IdxT, bool DoSoftmaxBeforeTopK>
void invokeCustomMoeRouting(InputT* routerLogits, OutputT* topkValues, IdxT* topkIndices, int64_t const numTokens,
    int64_t const numExperts, int64_t const topK, cudaStream_t const stream);

// Gate forward function for custom MoE routing
// All tensors are expected to be float32 for scores/weights, int32 for indices
void gate_forward(void* scores_in, // [batch_size, nExperts] - pre-computed from linear(x, weight)
    void* bias,                    // nullptr if hash mode
    void* input_ids,               // nullptr if non-hash mode
    void* tid2eid,                 // nullptr if non-hash mode
    void* out_weights,             // [batch_size, topK] - pre-allocated
    void* out_indices,             // [batch_size, topK] - pre-allocated
    int batch_size, int n_experts, float route_scale, bool is_hash, cudaStream_t stream);
} // namespace kernels

TRTLLM_NAMESPACE_END
