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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! The router GEMV's K: the GLM-5 hidden size.
constexpr int kRouterHiddenSize = 6144;
//! Routed experts.
constexpr int kRouterNumExperts = 256;
//! Tokens per call at most: one kernel instance per token count.
constexpr int kRouterMaxTokens = 32;

//! logits[t, e] = sum_k hiddenStates[t, k] * weight[e, k]: BF16 operands, FP32 accumulation and FP32 logits, all
//! dense row-major. hiddenStates is [numTokens, 6144] with numTokens in [1, 32], weight [256, 6144] and logits
//! [numTokens, 256].1-16 tokens use one GEMV CTA per expert;17-32 use eight-CTA tensor-core clusters.
//!
//! Launched with the programmatic-dependent-launch attribute when TRTLLM_ENABLE_PDL is set. Before its grid-dependency
//! wait each CTA prefetches its weight row (GEMV) or stages its weight tile (MMA); right after the wait it triggers its
//! dependents. Every kernel that follows on the stream must therefore wait on its own grid dependency before it reads
//! the logits or writes hiddenStates.
void invokeRouterGemm(
    float* logits, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight, int numTokens, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
