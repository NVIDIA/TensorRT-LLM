/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{
using SizeType32 = MakeDecodingBatchInputOutput::SizeType32;
using TensorPtr = MakeDecodingBatchInputOutput::TensorPtr;

std::unique_ptr<tr::decoder_batch::Input> MakeDecodingBatchInputOutput::createDecoderBatchInputs(
    std::vector<SizeType32> const& activeSlots, runtime::decoder::DecoderState const& decoderState,
    std::vector<TensorPtr> const& decoderLogits, SizeType32 maxNumSequences, std::vector<TensorPtr> const& batchSlots)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& numDecodingEngineTokens = decoderState.getNumDecodingEngineTokens();
    auto const& maxDecodingEngineTokens = decoderState.getMaxDecodingEngineTokens();
    auto const& maxDecodingDecoderTokens = decoderState.getMaxDecodingDecoderTokens();
    auto const maxDecoderSteps = common::ceilDiv(maxDecodingEngineTokens, maxDecodingDecoderTokens);

    for (SizeType32 step = 0; step < maxDecoderSteps; ++step)
    {
        batchSlots.at(step)->resize(maxNumSequences);
    }

    auto constexpr singleRequest = 1;

    std::vector<SizeType32> batchSizes(maxDecoderSteps);
    std::vector<std::vector<tr::ITensor::SharedConstPtr>> batchLogits(maxDecoderSteps);
    auto maxActiveDecoderSteps = 1;
    for (size_t batchIdx = 0; batchIdx < activeSlots.size(); ++batchIdx)
    {
        auto const slot = activeSlots.at(batchIdx);
        auto const& logits = decoderLogits.at(batchIdx);

        auto const numDecoderSteps = common::ceilDiv(numDecodingEngineTokens.at(slot), maxDecodingDecoderTokens);
        maxActiveDecoderSteps = std::max(maxActiveDecoderSteps, numDecoderSteps);
        for (SizeType32 step = 0; step < numDecoderSteps; ++step)
        {
            auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlots.at(step));
            batchSlotsRange[batchSizes[step]] = slot;
            batchSizes[step]++;
            TensorPtr logitsSlice = tr::ITensor::slice(logits, step, singleRequest);
            batchLogits[step].emplace_back(std::move(logitsSlice));
        }
    }

    for (SizeType32 step = 0; step < maxDecoderSteps; ++step)
    {
        batchSlots.at(step)->resize(batchSizes[step]);
    }
    batchLogits.resize(maxActiveDecoderSteps);

    auto decodingInput = std::make_unique<tr::decoder_batch::Input>(batchLogits, maxActiveDecoderSteps);
    decodingInput->batchSlots = batchSlots;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return decodingInput;
}

namespace
{

std::pair<std::vector<SizeType32>, std::vector<SizeType32>> getActiveSlots(RequestVector const& decoderRequests)
{
    std::vector<SizeType32> activeSlots;
    std::vector<SizeType32> generationSteps;
    for (auto const& llmReq : decoderRequests)
    {
        activeSlots.push_back(llmReq->mSeqSlot.value());
        generationSteps.push_back(llmReq->getDecodingIter());
    }

    return {activeSlots, generationSteps};
}

//! @brief Sets inputs for explicit draft tokens.
void setExplicitDraftTokensInputs(tr::DecodingInput& dInput, RuntimeBuffers const& fusedRuntimeBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(fusedRuntimeBuffers.mExplicitDraftTokensBuffers);
    auto const& explicitDraftTokensInputs = fusedRuntimeBuffers.mExplicitDraftTokensBuffers->engineOutputs;
    auto const& explicitDraftTokensLastInputs = fusedRuntimeBuffers.mExplicitDraftTokensBuffers->engineInputs;

    dInput.explicitDraftTokensInputs = tr::DecodingInput::ExplicitDraftTokensInputs();
    dInput.explicitDraftTokensInputs->nextDraftTokens = explicitDraftTokensInputs.nextDraftTokens;
    dInput.explicitDraftTokensInputs->nextFlatTokens = explicitDraftTokensInputs.nextFlatTokens;
    dInput.explicitDraftTokensInputs->nextDraftIndices = explicitDraftTokensInputs.nextDraftIndices;
    dInput.explicitDraftTokensInputs->nextDraftProbs = explicitDraftTokensInputs.nextDraftProbs;
    dInput.explicitDraftTokensInputs->lastDraftTokens = explicitDraftTokensLastInputs.draftTokens;
    dInput.explicitDraftTokensInputs->lastDraftIndices = explicitDraftTokensLastInputs.draftIndices;
    dInput.explicitDraftTokensInputs->lastPositionIdsBase = explicitDraftTokensLastInputs.positionIdsBase;
    dInput.explicitDraftTokensInputs->masks = explicitDraftTokensInputs.masks;
    dInput.explicitDraftTokensInputs->packedPositionIds = explicitDraftTokensInputs.packedPositionIds;
    dInput.explicitDraftTokensInputs->bestPathLengths = explicitDraftTokensInputs.bestPathLengths;
    dInput.explicitDraftTokensInputs->bestPathIndices = explicitDraftTokensInputs.bestPathIndices;
    dInput.explicitDraftTokensInputs->nextGenerationLengths = explicitDraftTokensInputs.nextGenerationLengths;
    dInput.explicitDraftTokensInputs->lastGenerationLengths = explicitDraftTokensLastInputs.generationLengths;
    dInput.explicitDraftTokensInputs->maxGenLengthDevice = explicitDraftTokensInputs.maxGenToken;
    // Slots in request order
    dInput.explicitDraftTokensInputs->seqSlots = fusedRuntimeBuffers.seqSlots;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

//! @brief Sets inputs for eagle decoding.
void setEagleInputs(tr::DecodingInput& dInput, RuntimeBuffers const& fusedRuntimeBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(fusedRuntimeBuffers.mEagleBuffers);
    auto const& eagleInputs = fusedRuntimeBuffers.mEagleBuffers->engineOutputs;
    auto const& eagleLastInputs = fusedRuntimeBuffers.mEagleBuffers->engineInputs;

    dInput.eagleInputs = tr::DecodingInput::EagleInputs();
    dInput.eagleInputs->nextDraftTokens = eagleInputs.nextDraftTokens;
    dInput.eagleInputs->nextDraftLens = eagleInputs.nextDraftLens;
    dInput.eagleInputs->nextDraftPaths = eagleInputs.nextDraftPaths;
    dInput.eagleInputs->lastDraftTokens = eagleLastInputs.draftTokens;
    dInput.eagleInputs->lastDraftLens = eagleLastInputs.draftLens;
    dInput.eagleInputs->lastDraftPaths = eagleLastInputs.draftPaths;
    dInput.eagleInputs->acceptedTokens = eagleInputs.acceptedTokens;
    dInput.eagleInputs->acceptedLens = eagleInputs.acceptedLens;
    dInput.eagleInputs->acceptedPathIds = eagleInputs.acceptedPaths;
    dInput.eagleInputs->chunkedContextNextTokens = eagleInputs.chunkedContextNextTokens;
    // Slots in request order
    dInput.eagleInputs->seqSlots = fusedRuntimeBuffers.seqSlots;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace

std::unique_ptr<tr::decoder_batch::Input> MakeDecodingBatchInputOutput::operator()(DecoderInputBuffers& inputBuffers,
    runtime::decoder::DecoderState& decoderState, runtime::ModelConfig const& modelConfig, SizeType32 maxNumSequences,
    OptionalRef<RuntimeBuffers> fusedRuntimeBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto [activeSlots, generationSteps] = getActiveSlots(inputBuffers.decoderRequests);

    auto decodingInput = createDecoderBatchInputs(
        activeSlots, decoderState, inputBuffers.logits, maxNumSequences, inputBuffers.forwardBatchSlots);

    auto const maxBeamWidth = decoderState.getMaxBeamWidth();
    if (maxBeamWidth > 1)
    {
        // For Variable-Beam-Width-Search
        decoderState.getJointDecodingInput().generationSteps = generationSteps;
    }

    if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
    {
        decoderState.getJointDecodingInput().medusaInputs->medusaLogits = inputBuffers.predictedDraftLogits;
    }

    if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        setExplicitDraftTokensInputs(decoderState.getJointDecodingInput(), *fusedRuntimeBuffers);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        setEagleInputs(decoderState.getJointDecodingInput(), *fusedRuntimeBuffers);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return decodingInput;
}

} // namespace tensorrt_llm::batch_manager
