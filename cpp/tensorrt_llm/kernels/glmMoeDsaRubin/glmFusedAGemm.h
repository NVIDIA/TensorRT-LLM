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

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! The GEMM's K: the GLM-5 hidden size, 24 k stages of 256, three per MMA warp.
constexpr int kFusedAHiddenSize = 6144;
//! Rows of the GLM-5 kv_a_proj_with_mqa weight: q_a (2048), compressed KV (512) and k_pe (64).
constexpr int kFusedAOutFeatures = 2624;
//! Rows of the same weight with the DSA indexer's wk (128) and weights_proj (32) rows appended.
constexpr int kFusedAIndexerOutFeatures = 2784;
//! Weight rows per CTA: the grid is outFeatures / 16 CTAs (164 or 174), one per SM.
constexpr int kFusedARowsPerCta = 16;
//! Tokens per call at most: one activation tile holds 8 token rows.
constexpr int kFusedAMaxTokens = 8;

//! Byte ranges that the GEMM prefetches into L2 before its grid-dependency wait: weights that later kernels of the
//! step read. Unused slots have zero bytes.
struct FusedAL2Prefetch
{
    static constexpr int kMaxRanges = 4;
    void const* ptr[kMaxRanges];
    size_t bytes[kMaxRanges];
};

//! output[t, r] = sum_k hiddenStates[t, k] * weight[r, k]: BF16 operands, FP32 accumulation, BF16 output, all three
//! dense row-major. hiddenStates is [numTokens, 6144] with numTokens in [1, 8]; weight is [outFeatures, 6144] with
//! outFeatures kFusedAOutFeatures or kFusedAIndexerOutFeatures; output is [numTokens, outFeatures].
//!
//! Launched with the programmatic-dependent-launch attribute when TRTLLM_ENABLE_PDL is set. Before its grid-dependency
//! wait the kernel reads only the weight (tensor copies into shared memory and registers) and prefetches the
//! l2Prefetch ranges into L2 (none when l2Prefetch is null); it reads hiddenStates and writes output after the wait,
//! and triggers its dependents right after the wait. Nothing launched before it in the step may write the weight.
void invokeFusedAGemm(__nv_bfloat16* output, __nv_bfloat16 const* hiddenStates, __nv_bfloat16 const* weight,
    int numTokens, int outFeatures, FusedAL2Prefetch const* l2Prefetch, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
