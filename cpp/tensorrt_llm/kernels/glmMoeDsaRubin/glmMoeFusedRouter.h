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
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Workspace of invokeFusedRouter, in 32-bit words: two exchange buffers of [8 tokens][256 experts][4 slices] fp32
//! (wide calls reuse each buffer for [tokens][256] reduced logits), whose idle value is -0.0 (bits 0x80000000), then
//! the epoch (0 before the first launch), padded to 16 bytes.
constexpr int32_t kFusedRouterWorkspaceWords = 2 * 8 * 256 * 4 + 4;

//! The router product and GLM-5's routing in one kernel (bf16
//! routerHiddenStates [numTokens, 6144] x routerWeight [256, 6144], fp32 logits, which are not returned). Through16
//! tokens the logits match the router GEMV;17-24 use the owned tensor-core reduction. The routing is
//! the TRTLLM-Gen MoE runner's DeepSeek-V3 routing without groups (sigmoid + bias, top-8, scaled sum normalization),
//! with the outputs of its small-token routing kernel: the bf16 expert weights and the permutation and token-tile
//! tables the TRTLLM-Gen FC1 and FC2 kernels read.
//!
//! Covers hiddenDim6144,256 experts,top-8:1-8 tokens require tileTokensDim >= numTokens;9-24 support
//! multiple tiles per expert. Anything else fails a check.
//! routerWorkspace: kFusedRouterWorkspaceWords words, owned by the caller and initialized once (both exchange buffers
//! -0.0, epoch 0); the kernel leaves it in that state for the next launch. Launches that share a workspace must run in
//! stream order.
void invokeFusedRouter(void const* routerHiddenStates, void const* routerWeight, int32_t hiddenDim,
    int32_t* routerWorkspace, void* routingBias, batchedGemm::trtllm::gen::Dtype dtypeRoutingBias, int32_t numTokens,
    int32_t numExperts, int32_t topK, int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
    int32_t tileTokensDim, int32_t* permutedIdxSize, int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToTokenIdx,
    void* expertWeights, int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas,
    cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
