/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm
{
// namespace tensorrt_llm::kernels
namespace kernels
{

// Prepare drafter input
struct MTPPrepareDrafterInputsParam
{
    int numMTPModules;
    int batchSize;
    int numContextRequest;
    int hiddenSize;
    int* inputIds;
    int* seqLens;
    void** __restrict__ mtpPastHiddenStatesPtrs;
    int** mtpPastTokensPtrs;
    void* __restrict__ hiddenStates;
    int* acceptedTokens;
    int* numAcceptedTokens;
    int* returnInputIds;
    void* __restrict__ returnHiddenStates;
};

template <typename T>
void invokeMTPPrepareDrafterInputs(MTPPrepareDrafterInputsParam& params, cudaStream_t const stream = 0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Sample and accept draft tokens
struct MTPSampleAndAcceptDraftTokensParam
{
    int numMTPModules;
    int batchSize;
    int numContextRequest;
    int vocabSize;
    void* __restrict__ logits;
    int* draftTokens;
    int* targetTokens;
    int* acceptedTokens;
    int* numAcceptedTokens;
};

template <typename T>
void invokeMTPSampleAndAcceptDraftTokens(MTPSampleAndAcceptDraftTokensParam& params, cudaStream_t const stream = 0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Update hidden states
struct MTPUpdateHiddenStatesParam
{
    int numMTPModules;
    int batchSize;
    int numContextRequest;
    int hiddenSize;
    int* inputIds;
    int* seqLens;
    void* __restrict__ targetModelHiddenStates;
    void** __restrict__ mtpPastHiddenStatesPtrs;
    int** mtpPastTokensPtrs;
    int* numAcceptedTokens;
    int* acceptedTokens;
};

template <typename T>
void invokeMTPUpdateHiddenStates(MTPUpdateHiddenStatesParam& params, cudaStream_t const stream = 0);

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Relaxed acceptance
struct MTPRelaxedAcceptanceParam
{
    int numMTPModules;
    int batchSize;
    int numContextRequest;
    int relaxedTopK;
    float relaxedDelta;
    int beginThinkingTokens;
    int endThinkingTokens;
    void* __restrict__ topKValue;
    int* reqSlotIds;
    int64_t* topKIndices;
    int* draftTokens;
    float* mtpRelaxedDelta;
    int* numAcceptedTokens;
    int* acceptedTokens;
};

template <typename T>
void invokeMTPRelaxedAcceptance(MTPRelaxedAcceptanceParam& params, cudaStream_t const stream = 0);

} // namespace kernels

} // namespace tensorrt_llm
