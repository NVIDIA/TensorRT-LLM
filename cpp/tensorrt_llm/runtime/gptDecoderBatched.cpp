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
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;

GptDecoderBatched::GptDecoderBatched(GptDecoderBatched::CudaStreamPtr stream,
    SpeculativeDecodingMode const& speculativeDecodingMode, nvinfer1::DataType dtype)
    : mRuntimeStream{std::move(stream)}
    , mBufferManager{mRuntimeStream}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecoderState = std::make_shared<decoder::DecoderState>(dtype, mBufferManager);

    if (!speculativeDecodingMode.isNone())
    {
        mDecoderState->allocateSpeculativeDecodingBuffers(speculativeDecodingMode, dtype, mBufferManager);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::disableLookahead(
    SizeType32 maxBatchSize, RequestVector const& genRequests, TensorPtr const& batchSlots)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecoderState->disableLookahead(maxBatchSize, genRequests);

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

void GptDecoderBatched::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    SizeType32 maxTokensPerEngineStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(maxTokensPerEngineStep > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mDecoderState->setup(maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength,
        modelConfig, worldConfig, mBufferManager);

    mDecoderState->setupSpeculativeDecoding(
        mDecoderState->getSpeculativeDecodingMode(), maxTokensPerEngineStep, modelConfig, worldConfig, mBufferManager);

    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModulePtr = nullptr;
    if (mDecoderState->getSpeculativeDecodingMode().predictsDraftTokens())
    {
        speculativeDecodingModulePtr = modelConfig.getSpeculativeDecodingModulePtr();
    }

    auto const device = mRuntimeStream->getDevice();
    mDecoderStream = std::make_shared<CudaStream>();
    TLLM_CHECK(mDecoderStream->getDevice() == device);

    auto const vocabSize = modelConfig.getVocabSize();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    mDecoder = IGptDecoder::create(mode, dtype, maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded,
        maxSequenceLength, mDecoderStream, speculativeDecodingModulePtr);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setExplicitDraftTokensInputs(decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto explicitDraftTokensInputs = DecodingInput::ExplicitDraftTokensInputs();
    TLLM_CHECK(input.explicitDraftTokensInputs.has_value());
    TLLM_CHECK(input.explicitDraftTokensLastInputs.has_value());

    explicitDraftTokensInputs.nextDraftTokens = input.explicitDraftTokensInputs->nextDraftTokens;
    explicitDraftTokensInputs.nextFlatTokens = input.explicitDraftTokensInputs->nextFlatTokens;
    explicitDraftTokensInputs.nextDraftIndices = input.explicitDraftTokensInputs->nextDraftIndices;
    explicitDraftTokensInputs.nextDraftProbs = input.explicitDraftTokensInputs->nextDraftProbs;
    explicitDraftTokensInputs.lastDraftTokens = input.explicitDraftTokensLastInputs->draftTokens;
    explicitDraftTokensInputs.lastDraftIndices = input.explicitDraftTokensLastInputs->draftIndices;
    explicitDraftTokensInputs.lastPositionIdsBase = input.explicitDraftTokensLastInputs->positionIdsBase;
    explicitDraftTokensInputs.masks = input.explicitDraftTokensInputs->masks;
    explicitDraftTokensInputs.packedPositionIds = input.explicitDraftTokensInputs->packedPositionIds;
    explicitDraftTokensInputs.bestPathLengths = input.explicitDraftTokensInputs->bestPathLengths;
    explicitDraftTokensInputs.bestPathIndices = input.explicitDraftTokensInputs->bestPathIndices;
    explicitDraftTokensInputs.nextGenerationLengths = input.explicitDraftTokensInputs->nextGenerationLengths;
    explicitDraftTokensInputs.lastGenerationLengths = input.explicitDraftTokensLastInputs->generationLengths;
    explicitDraftTokensInputs.maxGenLengthDevice = input.explicitDraftTokensInputs->maxGenToken;
    explicitDraftTokensInputs.seqSlots = input.batchSlotsRequestOrder;
    mDecoderState->getJointDecodingInput().explicitDraftTokensInputs = explicitDraftTokensInputs;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setEagleInputs(decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(input.eagleInputs.has_value());
    TLLM_CHECK(input.eagleLastInputs.has_value());

    auto eagleInputs = DecodingInput::EagleInputs(input.eagleInputs->nextDraftTokens, input.eagleInputs->nextDraftLens,
        input.eagleInputs->nextDraftPaths, input.eagleLastInputs->draftTokens, input.eagleLastInputs->draftLens,
        input.eagleLastInputs->draftPaths, input.eagleInputs->acceptedTokens, input.eagleInputs->acceptedLens,
        input.eagleInputs->acceptedPaths, input.eagleInputs->chunkedContextNextTokens, input.batchSlotsRequestOrder);

    mDecoderState->getJointDecodingInput().eagleInputs = eagleInputs;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
template <typename T>
T maxOfActiveSlots(std::vector<T> const& values, std::vector<bool> const& active)
{
    return std::transform_reduce(
        values.begin(), values.end(), active.begin(), std::numeric_limits<T>::min(),
        [](auto lhf, auto rhs) { return std::max(lhf, rhs); },
        [](auto numTokens, auto active) { return active ? numTokens : std::numeric_limits<T>::min(); });
}
} // namespace

void GptDecoderBatched::forwardDispatch(
    decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    auto eventStart = CudaEvent{};
    mRuntimeStream->record(eventStart);

    bool const async = forwardType == ForwardType::kASYNC;

    if (async)
    {
        mDecoderStream->wait(eventStart.get());
    }

    auto const maxDecodingEngineTokens
        = maxOfActiveSlots(mDecoderState->getJointDecodingInput().numDecodingEngineTokens, input.active);

    for (SizeType32 si = 0; si < maxDecodingEngineTokens; si += mDecoderState->getMaxDecodingDecoderTokens())
    {
        prepareForward(si, output, input);
        forwardDecoder(mDecoderState->getJointDecodingOutput(), mDecoderState->getJointDecodingInput(), forwardType);
    }

    if (async)
    {
        CudaEvent event{};
        mDecoderStream->record(event);
        mRuntimeStream->wait(event);
    }
}

GptDecoderBatched::DecoderFinishedEventPtr GptDecoderBatched::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    forwardDispatch(output, input, ForwardType::kASYNC);

    CudaEvent eventStop{};
    mRuntimeStream->record(eventStop);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::DecoderFinishedEvent>(std::move(eventStop), input.active);
}

// TODO(rkobus): produce new input and output
void GptDecoderBatched::prepareForward(
    SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& allTargetLogits = input.logits;
    auto const& jointOutputIdsShape = mDecoderState->getJointDecodingOutput().ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const speculativeDecodingMode = mDecoderState->getSpeculativeDecodingMode();

    auto constexpr singleRequest = 1;

    TLLM_CHECK(static_cast<SizeType32>(output.sequenceLengths->getSize())
        == mDecoderState->getActualBatchSize() * maxBeamWidth);
    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths = ITensor::view(
        output.sequenceLengths, ITensor::makeShape({mDecoderState->getActualBatchSize(), maxBeamWidth}));
    TLLM_CHECK(sequenceLengths);

    auto& dInput = mDecoderState->getJointDecodingInput();
    auto& dOutput = mDecoderState->getJointDecodingOutput();

    if (maxBeamWidth > 1)
    {
        dInput.cacheIndirection = input.cacheIndirection;
        dOutput.cacheIndirection = output.cacheIndirection;
    }

    if (speculativeDecodingMode.isExplicitDraftTokens())
    {
        setExplicitDraftTokensInputs(input);
    }
    else if (speculativeDecodingMode.isEagle())
    {
        setEagleInputs(input);
    }

    TensorPtr batchSlotsSlice = ITensor::at(input.batchSlots, {step});
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlotsSlice);
    SizeType32 localBatchDecoderIdx = 0;
    std::vector<SharedConstPtr> logitsVec;
    for (SizeType32 bi = 0; bi < mDecoderState->getActualBatchSize(); ++bi)
    {
        if (!input.active.at(bi) || step >= mDecoderState->getJointDecodingInput().numDecodingEngineTokens.at(bi))
        {
            continue;
        }
        batchSlotsRange[localBatchDecoderIdx] = bi;
        localBatchDecoderIdx++;

        auto const& targetLogits = allTargetLogits[bi];
        TensorPtr logitsSlice = ITensor::slice(targetLogits, step, singleRequest);
        logitsVec.push_back(logitsSlice);
    }
    batchSlotsSlice->resize(localBatchDecoderIdx);
    dInput.batchSlots = batchSlotsSlice;
    dInput.batchSize = localBatchDecoderIdx;
    dInput.logitsVec = logitsVec;

    auto const maxDecodingEngineTokens
        = maxOfActiveSlots(mDecoderState->getJointDecodingInput().numDecodingEngineTokens, input.active);

    TensorPtr finishedStepsInput = ITensor::slice(mDecoderState->getFinishedSteps(), step, 1);
    TensorPtr finishedStepsOutput
        = ITensor::slice(mDecoderState->getFinishedSteps(), std::min(maxDecodingEngineTokens - 1, step + 1), 1);
    finishedStepsInput->squeeze(0);
    finishedStepsOutput->squeeze(0);
    TensorPtr newTokensStepView
        = ITensor::slice(dOutput.newTokensSteps, step, mDecoderState->getMaxDecodingDecoderTokens());

    dInput.finishReasons = finishedStepsInput;

    if (speculativeDecodingMode.isMedusa())
    {
        dInput.medusaInputs->medusaLogits = input.predictedDraftLogits;
    }

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->step = step;

        // WAR: reset finished state for generation requests
        if (step == 0)
        {
            BufferManager manager{mDecoderStream};

            for (SizeType32 bi = 0; bi < mDecoderState->getActualBatchSize(); ++bi)
            {
                if (!input.active.at(bi))
                {
                    continue;
                }
                TensorPtr finishedStepsView = ITensor::slice(mDecoderState->getFinishedSteps(), 0, 1);
                finishedStepsView->squeeze(0);
                auto batchSlot = bi;
                TensorPtr finishedSteps = ITensor::slice(finishedStepsView, batchSlot, 1);
                manager.setZero(*finishedStepsView);
            }
        }
    }

    dOutput.newTokens = newTokensStepView;
    dOutput.finishReasons = finishedStepsOutput;
    dOutput.lengths = sequenceLengths;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardDecoder(DecodingOutput& output, DecodingInput const& input, ForwardType forwardType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (input.batchSize > 0)
    {
        if (forwardType == ForwardType::kASYNC)
        {
            mDecoder->forwardAsync(output, input);
        }
        else if (forwardType == ForwardType::kSYNC)
        {
            mDecoder->forwardSync(output, input);
        }
        else
        {
            TLLM_THROW("Unknown ForwardType");
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forward(decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto decoderFinishEvent = forwardAsync(output, input);
    decoderFinishEvent->event.synchronize();
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
CudaEvent GptDecoderBatched::finalize(SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& stream = mRuntimeStream;
    auto manager = BufferManager{stream};

    auto [dInput, dOutput] = prepareGatherTree(*mDecoderState, batchSlot, streaming, *stream);

    kernels::gatherTree(dOutput, dInput, manager, samplingConfig);

    CudaEvent event{};
    stream->record(event);
    mRuntimeStream->wait(event);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
