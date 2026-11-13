/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime_api.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Base-2 log-sum-exp of every (token, head) row of a trtllm-gen FMHA call from the softmax stats it stored.
// stats: contiguous [numTokens, numHeads] (max, sum) pairs, the max in the natural-log domain of the scaled logits.
// lse[token * lseStrideTokens + head * lseStrideHeads] = max * log2(e) + log2(sum).
void invokeFmhaSoftmaxStatsToLse(float2 const* stats, float* lse, int32_t numTokens, int32_t numHeads,
    int64_t lseStrideTokens, int64_t lseStrideHeads, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
