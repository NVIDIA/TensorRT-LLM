/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/gptDecoderBatched.h"

#include "common.h"
#include "decoderState.h"
#include "iBuffer.h"
#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;
using TensorPtr = ITensor::SharedPtr;

GptDecoderBatched::GptDecoderBatched(GptDecoderBatched::CudaStreamPtr stream)
    : mRuntimeStream{std::move(stream)}
    , mBufferManager{mRuntimeStream}
{
}

void GptDecoderBatched::disableLookahead(RequestVector const& genRequests, TensorPtr const& batchSlots)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    std::vector<SamplingConfig> samplingConfigs;
    samplingConfigs.reserve(genRequests.size());
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlots);

    SizeType32 batchIdx = 0;
    for (auto const& llmReq : genRequests)
    {
        samplingConfigs.push_back(llmReq->mSamplingConfig);
        batchSlotsRange[batchIdx] = llmReq->mSeqSlot.value();
        batchIdx += 1;
    }
    auto const batchSize = batchIdx;
    std::optional<SamplingConfig> samplingConfig;
    if (batchSize > 0)
    {
        samplingConfig = SamplingConfig(samplingConfigs);
    }
    TensorPtr batchSlotsView = ITensor::slice(batchSlots, 0, batchSize);
    mDecoder->disableLookahead(samplingConfig, batchSize, batchSlots);

    CudaEvent event{};
    mDecoderStream->record(event);
    mRuntimeStream->wait(event);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setup(executor::DecodingMode const& mode, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
    nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxNumSequences > 0);
    TLLM_CHECK(maxBeamWidth > 0);

    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModulePtr = nullptr;
    if (modelConfig.getSpeculativeDecodingMode().predictsDraftTokens())
    {
        speculativeDecodingModulePtr = modelConfig.getSpeculativeDecodingModulePtr();
    }

    auto const device = mRuntimeStream->getDevice();
    mDecoderStream = std::make_shared<CudaStream>();
    TLLM_CHECK(mDecoderStream->getDevice() == device);

    auto const vocabSize = modelConfig.getVocabSize();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    mDecoder = IGptDecoder::create(mode, dtype, maxNumSequences, maxBeamWidth, vocabSize, vocabSizePadded,
        mDecoderStream, speculativeDecodingModulePtr);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
//! @brief Prepare Input and Output for decoder step.
// TODO: produce new input and output objects
void prepareForward(decoder::DecoderState const& decoderState, SizeType32 step, decoder_batch::Input const& input,
    BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const speculativeDecodingMode = decoderState.getSpeculativeDecodingMode();

    auto& dInput = decoderState.getJointDecodingInput();
    auto& dOutput = decoderState.getJointDecodingOutput();

    dInput.batchSlots = input.batchSlots.at(step);
    dInput.batchSize = static_cast<SizeType32>(dInput.batchSlots->getSize());
    dInput.logitsVec = input.logits.at(step);

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->step = step;

        // WAR: reset finished state for generation requests
        if (step == 0)
        {
            auto batchSlotsRange = BufferRange<SizeType32 const>(*dInput.batchSlots);
            for (auto batchSlot : batchSlotsRange)
            {
                TensorPtr finishedStepsSlice = ITensor::slice(decoderState.getFinishReasons(), batchSlot, 1);
                bufferManager.setZero(*finishedStepsSlice);
            }
        }
    }

    dOutput.newTokens = ITensor::slice(dOutput.newTokensSteps, step, decoderState.getMaxDecodingDecoderTokens());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace

void GptDecoderBatched::forwardDispatch(decoder::DecoderState const& decoderState, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    for (SizeType32 step = 0; step < input.maxDecoderSteps; ++step)
    {
        BufferManager manager{mDecoderStream};
        prepareForward(decoderState, step, input, manager);

        if (decoderState.getJointDecodingInput().batchSize > 0)
        {
            mDecoder->forwardAsync(decoderState.getJointDecodingOutput(), decoderState.getJointDecodingInput());
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatched::forwardAsync(decoder::DecoderState const& decoderState, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto eventStart = CudaEvent{};
    mRuntimeStream->record(eventStart);
    mDecoderStream->wait(eventStart.get());

    forwardDispatch(decoderState, input);

    CudaEvent event{};
    mDecoderStream->record(event);
    mRuntimeStream->wait(event);

    CudaEvent eventStop{};
    mRuntimeStream->record(eventStop);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return eventStop;
}

void GptDecoderBatched::forward(decoder::DecoderState const& decoderState, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto decoderFinishEvent = forwardAsync(decoderState, input);
    decoderFinishEvent.synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
std::pair<DecodingInput, DecodingOutput> prepareGatherTree(
    decoder::DecoderState const& decoderState, SizeType32 batchSlot, bool streaming, CudaStream const& stream)
{
    auto& dJointInput = decoderState.getJointDecodingInput();
    auto& dJointOutput = decoderState.getJointDecodingOutput();

    auto slice = [batchSlot](auto& a, auto const& b)
    {
        if (b && b->getShape().d[0] > 0)
        {
            a = ITensor::slice(b, batchSlot, 1);
        }
    };

    // Prepare a slice of dJointInput and dJointOutput for gatherTree
    DecodingInput dInput{dJointInput};
    slice(dInput.endIds, dJointInput.endIds);
    slice(dInput.lengths, dJointInput.lengths);

    DecodingOutput dOutput{
        ITensor::slice(dJointOutput.ids, batchSlot, 1), ITensor::slice(dJointOutput.gatheredIds, batchSlot, 1)};
    dOutput.beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, 1);
    slice(dOutput.parentIds, dJointOutput.parentIds);
    slice(dOutput.cumLogProbs, dJointOutput.cumLogProbs);
    slice(dOutput.cacheIndirection, dJointOutput.cacheIndirection);
    slice(dOutput.lengths, dJointOutput.lengths);
    slice(dOutput.finishReasons, dJointOutput.finishReasons);
    slice(dOutput.logProbs, dJointOutput.logProbs);

    dOutput.newTokens = ITensor::view(dJointOutput.newTokens);
    TLLM_CHECK(dOutput.newTokens->getShape().d[0] == 1);
    dOutput.newTokens->squeeze(0);
    dOutput.newTokens = ITensor::slice(dOutput.newTokens, batchSlot, 1);
    dOutput.logProbsTiled = dJointOutput.logProbsTiled;
    if (streaming)
    {
        // in case of streaming we shouldn't overwrite the data in beamHypotheses, since the beam search kernels expect
        // ungathered data but the kernels in gatherTree write in-place.
        // Thus, we need to make a copy of the beamHypotheses
        auto const& beamSearchBuffers = decoderState.getBeamSearchBuffers();
        tensorrt_llm::kernels::invokeCopyBeamHypotheses(dOutput.beamHypotheses, beamSearchBuffers.mOutputBeamHypotheses,
            *dOutput.cumLogProbs, *beamSearchBuffers.mCumLogProbsTmp, stream, beamSearchBuffers.mNumSMs);
        dOutput.beamHypotheses = beamSearchBuffers.mOutputBeamHypotheses;
        dOutput.cumLogProbs = beamSearchBuffers.mCumLogProbsTmp;
    }

    return {(std::move(dInput)), (std::move(dOutput))};
}
} // namespace

// TODO call this at the end of forward if mFinished[i] changes from false to true?
CudaEvent GptDecoderBatched::finalize(decoder::DecoderState const& decoderState, SizeType32 batchSlot,
    SamplingConfig const& samplingConfig, bool streaming) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto [dInput, dOutput] = prepareGatherTree(decoderState, batchSlot, streaming, *mRuntimeStream);

    kernels::gatherTree(dOutput, dInput, samplingConfig, *mRuntimeStream);

    CudaEvent event{};
    mRuntimeStream->record(event);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
