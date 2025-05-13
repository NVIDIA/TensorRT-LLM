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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{
using SizeType32 = MakeDecodingBatchInputOutput::SizeType32;
using TensorPtr = MakeDecodingBatchInputOutput::TensorPtr;

std::unique_ptr<tr::decoder_batch::Input> MakeDecodingBatchInputOutput::createDecoderBatchInputs(
    std::vector<SizeType32> const& activeSlots, runtime::decoder::DecoderState const& decoderState,
    std::vector<TensorPtr> const& logits, SizeType32 maxNumSequences, std::vector<TensorPtr> const& batchSlots,
    TensorPtr const& cacheIndirectionInput)
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

    std::vector<SizeType32> batchIdx(maxDecoderSteps);
    auto maxActiveDecoderSteps = 1;
    for (auto const slot : activeSlots)
    {
        auto const numDecoderSteps = common::ceilDiv(numDecodingEngineTokens.at(slot), maxDecodingDecoderTokens);
        maxActiveDecoderSteps = std::max(maxActiveDecoderSteps, numDecoderSteps);
        for (SizeType32 step = 0; step < numDecoderSteps; ++step)
        {
            auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlots.at(step));
            batchSlotsRange[batchIdx[step]] = slot;
            batchIdx[step]++;
        }
    }

    for (SizeType32 step = 0; step < maxDecoderSteps; ++step)
    {
        batchSlots.at(step)->resize(batchIdx[step]);
    }

    auto constexpr singleRequest = 1;
    std::vector<std::vector<tr::ITensor::SharedConstPtr>> logitsVec(maxActiveDecoderSteps);
    for (SizeType32 step = 0; step < maxActiveDecoderSteps; ++step)
    {
        auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlots.at(step));

        for (auto slot : batchSlotsRange)
        {
            auto const& targetLogits = logits.at(slot);
            TensorPtr logitsSlice = tr::ITensor::slice(targetLogits, step, singleRequest);
            logitsVec.at(step).push_back(logitsSlice);
        }
    }

    auto decodingInput = std::make_unique<tr::decoder_batch::Input>(logitsVec, maxActiveDecoderSteps);
    decodingInput->batchSlots = batchSlots;
    decodingInput->cacheIndirection = cacheIndirectionInput;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return decodingInput;
}

namespace
{

std::pair<std::vector<SizeType32>, std::vector<SizeType32>> getActiveSlots(
    RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    std::vector<std::pair<SizeType32, SizeType32>> slots;
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            if (llmReq->isGenerationInProgressState() || llmReq->isLastContextChunk())
            {
                slots.push_back({llmReq->mSeqSlot.value(), llmReq->getDecodingIter()});
            }
        }
    }

    std::sort(slots.begin(), slots.end(),
        [](std::pair<SizeType32, SizeType32> const& a, std::pair<SizeType32, SizeType32> const& b)
        { return a.first < b.first; });

    std::vector<SizeType32> activeSlots, generationSteps;
    for (auto const& slot : slots)
    {
        activeSlots.push_back(slot.first);
        generationSteps.push_back(slot.second);
    }

    return {activeSlots, generationSteps};
}

void copySequenceLengths(RequestVector const& contextRequests, RequestVector const& generationRequests,
    DecoderInputBuffers const& inputBuffers, TensorPtr const& sequenceLengths, SizeType32 beamWidth, bool isTrtOverlap,
    runtime::BufferManager const& manager, runtime::CudaStream const& stream)
{
    auto const batchSize = contextRequests.size() + generationRequests.size();
    auto batchSlotsView = tr::ITensor::slice(inputBuffers.forwardBatchSlotsRequestOrder, 0, batchSize);
    auto fillValuesView = tr::ITensor::slice(inputBuffers.fillValues, 0, batchSize);

    auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlotsView);
    auto fillValuesRange = tr::BufferRange<SizeType32>(*fillValuesView);

    // fill buffers on host
    SizeType32 batchIdx{0};
    for (auto const& llmReq : contextRequests)
    {
        auto const currentSequenceLen = llmReq->mPromptLen + llmReq->getMaxNumGeneratedTokens();
        // Get position of the current sequence in the decoder
        auto const seqSlot = llmReq->mSeqSlot.value();
        batchSlotsRange[batchIdx] = seqSlot;
        fillValuesRange[batchIdx] = currentSequenceLen;
        ++batchIdx;
    }
    for (auto const& llmReq : generationRequests)
    {
        auto const currentSequenceLen
            = llmReq->mPromptLen + llmReq->getMaxNumGeneratedTokens() + static_cast<SizeType32>(isTrtOverlap);
        // Get position of the current sequence in the decoder
        auto const seqSlot = llmReq->mSeqSlot.value();
        batchSlotsRange[batchIdx] = seqSlot;
        fillValuesRange[batchIdx] = currentSequenceLen;
        ++batchIdx;
    }

    // copy sequence lengths
    {
        auto batchSlotsDeviceView = tr::ITensor::slice(inputBuffers.forwardBatchSlotsRequestOrderDevice, 0, batchSize);
        auto fillValuesViewDevice = tr::ITensor::slice(inputBuffers.fillValuesDevice, 0, batchSize);

        manager.copy(*batchSlotsView, *batchSlotsDeviceView);
        manager.copy(*fillValuesView, *fillValuesViewDevice);
        tr::kernels::invokeFillBatch(*sequenceLengths, *batchSlotsDeviceView, beamWidth, *fillValuesViewDevice, stream);
    }
}
} // namespace

std::tuple<std::unique_ptr<tr::decoder_batch::Input>, std::unique_ptr<tr::decoder_batch::Output>>
MakeDecodingBatchInputOutput::operator()(RequestVector const& contextRequests, RequestVector const& generationRequests,
    DecoderBuffers& decoderBuffers, DecoderInputBuffers const& inputBuffers,
    runtime::decoder::DecoderState& decoderState, runtime::ModelConfig const& modelConfig, SizeType32 maxNumSequences,
    SizeType32 beamWidth, bool isTrtOverlap, runtime::BufferManager const& manager, runtime::CudaStream const& stream,
    OptionalRef<RuntimeBuffers> fusedRuntimeBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto [activeSlots, generationSteps] = getActiveSlots(contextRequests, generationRequests);

    auto decodingInput = createDecoderBatchInputs(activeSlots, decoderState, decoderBuffers.logits, maxNumSequences,
        inputBuffers.forwardBatchSlots, decoderBuffers.cacheIndirectionInput);
    decodingInput->generationSteps = generationSteps;

    if (modelConfig.getSpeculativeDecodingMode().hasDraftLogits())
    {
        decodingInput->predictedDraftLogits = decoderBuffers.draftBuffers.predictedDraftLogits;
    }

    if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        decodingInput->batchSlotsRequestOrder = fusedRuntimeBuffers->seqSlots;
        decodingInput->explicitDraftTokensInputs = fusedRuntimeBuffers->explicitDraftTokensBuffers->engineOutputs;
        decodingInput->explicitDraftTokensLastInputs = fusedRuntimeBuffers->explicitDraftTokensBuffers->engineInputs;
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        TLLM_CHECK(fusedRuntimeBuffers);
        // requires mCtxGenFusion == true
        decodingInput->batchSlotsRequestOrder = fusedRuntimeBuffers->seqSlots;
        decodingInput->eagleInputs = fusedRuntimeBuffers->eagleBuffers->engineOutputs;
        decodingInput->eagleLastInputs = fusedRuntimeBuffers->eagleBuffers->engineInputs;
    }

    copySequenceLengths(contextRequests, generationRequests, inputBuffers,
        decoderState.getJointDecodingOutput().lengths, beamWidth, isTrtOverlap, manager, stream);

    auto decodingOutput = std::make_unique<tr::decoder_batch::Output>();
    decodingOutput->cacheIndirection = decoderBuffers.cacheIndirectionOutput;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(decodingInput), std::move(decodingOutput)};
}

} // namespace tensorrt_llm::batch_manager
