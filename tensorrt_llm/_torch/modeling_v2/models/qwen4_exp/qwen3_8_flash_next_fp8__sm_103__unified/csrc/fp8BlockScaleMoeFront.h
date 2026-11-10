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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::fp8_block_scale_moe_front
{

// Renormalize routing (top-k of the bf16 logits, softmax over those k) in front of the trtllm-gen FP8 block-scale MoE
// runner for a decode-sized batch: one launch writes the runner's routing tables, optionally fused with the 1x128 e4m3
// quantization of the bf16 hidden states. Top-k ids, order and bf16 weights match the runner bitwise; a slot's offset
// within its expert may differ (atomic order), which no consumer depends on. Launches on one device must not overlap.
struct Data
{
    void const* ptrScores{nullptr};     // [numTokens, numExperts] bf16 logits
    void const* ptrHiddenBf16{nullptr}; // optional [numTokens, hiddenDim] bf16 hidden states to quantize
    int32_t numTokens{0};
    int32_t numExperts{0};
    int32_t topK{0};
    int32_t tileTokensDim{0};
    int32_t localExpertsStartIdx{0};
    int32_t numLocalExperts{0};
    int32_t hiddenDim{0};
    bool usePdl{false};

    // Outputs: the runner's routing tables.
    void* ptrTopKWeights{nullptr}; // [numTokens, topK] bf16
    int32_t* ptrExpandedIdxToPermutedIdx{nullptr};
    int32_t* ptrPermutedIdxToTokenIdx{nullptr};
    int32_t* ptrPermutedIdxSize{nullptr};
    int32_t* ptrCtaIdxXyToBatchIdx{nullptr};
    int32_t* ptrCtaIdxXyToMnLimit{nullptr};
    int32_t* ptrNumNonExitingCtas{nullptr};
    void* ptrHiddenFp8{nullptr};    // [numTokens, hiddenDim] e4m3 (with ptrHiddenBf16)
    float* ptrHiddenScale{nullptr}; // [hiddenDim / 128, numTokens] fp32 (with ptrHiddenBf16)
};

constexpr int MaxNumExperts = 512;
constexpr int MaxTopK = 16;
constexpr int MaxNumTokens = 1024;

constexpr int QuantHidden = 2560; // hidden width the fused quantization is compiled for

// Whether `run` takes this problem (bf16 logits, no fused shared experts, stride-1 local window).
bool supports(int32_t numTokens, int32_t numExperts, int32_t topK, int32_t numLocalExperts, int32_t hiddenDim);

void run(Data const& data, cudaStream_t stream);

} // namespace kernels::fp8_block_scale_moe_front

TRTLLM_MV2_NAMESPACE_END
