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

#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;

namespace tk = tensorrt_llm::kernels;

namespace
{
SamplingConfig extractSamplingConfig(SamplingConfig const& batchSamplingConfig, SizeType32 batchIdx)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    SamplingConfig samplingConfig{batchSamplingConfig.beamWidth};

    auto extractOptional = [&batchIdx](auto& single, auto const& batch)
    {
        using T = typename std::remove_reference_t<decltype(batch)>::value_type;
        if (batch)
        {
            if (batch->size() > 1)
                single.emplace(T{batch->at(batchIdx)});
            else
                single.emplace(T{batch->at(0)});
        }
    };

    extractOptional(samplingConfig.temperature, batchSamplingConfig.temperature);
    extractOptional(samplingConfig.originalTemperature, batchSamplingConfig.originalTemperature);
    extractOptional(samplingConfig.minLength, batchSamplingConfig.minLength);
    extractOptional(samplingConfig.repetitionPenalty, batchSamplingConfig.repetitionPenalty);
    extractOptional(samplingConfig.presencePenalty, batchSamplingConfig.presencePenalty);
    extractOptional(samplingConfig.frequencyPenalty, batchSamplingConfig.frequencyPenalty);
    extractOptional(samplingConfig.noRepeatNgramSize, batchSamplingConfig.noRepeatNgramSize);
    // sampling layers
    extractOptional(samplingConfig.topK, batchSamplingConfig.topK);
    extractOptional(samplingConfig.topP, batchSamplingConfig.topP);
    extractOptional(samplingConfig.randomSeed, batchSamplingConfig.randomSeed);
    extractOptional(samplingConfig.topPDecay, batchSamplingConfig.topPDecay);
    extractOptional(samplingConfig.topPMin, batchSamplingConfig.topPMin);
    extractOptional(samplingConfig.topPResetIds, batchSamplingConfig.topPResetIds);
    extractOptional(samplingConfig.minP, batchSamplingConfig.minP);

    // beam search layer
    extractOptional(samplingConfig.beamSearchDiversityRate, batchSamplingConfig.beamSearchDiversityRate);
    extractOptional(samplingConfig.lengthPenalty, batchSamplingConfig.lengthPenalty);
    extractOptional(samplingConfig.earlyStopping, batchSamplingConfig.earlyStopping);
    samplingConfig.normalizeLogProbs = batchSamplingConfig.normalizeLogProbs;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return samplingConfig;
}

} // namespace

GptDecoderBatched::GptDecoderBatched(std::size_t vocabSize, std::size_t vocabSizePadded,
    GptDecoderBatched::CudaStreamPtr stream, SpeculativeDecodingMode const& speculativeDecodingMode,
    nvinfer1::DataType dtype)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mRuntimeStream{std::move(stream)}
    , mBufferManager{mRuntimeStream}
    , mSpeculativeDecodingMode{speculativeDecodingMode}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    { // prevent reusing these vars after std::move
        auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
        auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto batchSlots = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        dInput = std::make_unique<DecodingInput>(
            0, 0, 0, 0, std::move(dummyLogits), std::move(endIds), std::move(batchSlots));
    }
    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    { // prevent reusing these vars after std::move
        auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto gatheredOutputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        dOutput = std::make_unique<DecodingOutput>(std::move(outputIds), std::move(gatheredOutputIds));
    }
    dOutput->newTokensSteps = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mFinishedSteps
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mBatchSlotsSetup = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mBatchSlotsDecoder = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mFinishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
    // we don't need dOutput->lengths because lengths are passed from outside
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);
    dOutput->finishReasons
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);

    dOutput->logProbsTiled = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    dInput->stopWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->badWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->embeddingBias = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    mNumSMs = deviceProp.multiProcessorCount;

    if (!mSpeculativeDecodingMode.isNone())
    {
        allocateSpeculativeDecodingBuffers(dtype);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::allocateSpeculativeDecodingBuffers(nvinfer1::DataType dtype)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;

    auto& dInput = mJointDecodingInput;
    auto& dOutput = mJointDecodingOutput;

    if (mSpeculativeDecodingMode.isMedusa())
    {
        DecodingInput::MedusaInputs medusaInputs;
        medusaInputs.medusaPaths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTreeIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaCurTokensPerStep = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTargetTokensPerStep = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        dInput->medusaInputs = medusaInputs;
    }

    DecodingOutput::SpeculativeDecodingOutputs speculativeDecodingOutputs;
    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingOutputs.nextDraftTokens
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            speculativeDecodingOutputs.nextDraftTokensLen
                = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            speculativeDecodingOutputs.prevDraftTokensLen
                = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        }
    }
    if (mSpeculativeDecodingMode.isLookaheadDecoding())
    {
        dInput->lookaheadInputs = DecodingInput::LookaheadInputs();
    }
    if (mSpeculativeDecodingMode.needsKVCacheRewind())
    {
        speculativeDecodingOutputs.acceptedTokensLen
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.acceptedLengthsCumSum
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.pathsOffsets
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }
    dOutput->speculativeDecodingOutputs = speculativeDecodingOutputs;

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        DecodingInput::ExternalDraftTokensInputs externalDraftTokensInputs;

        externalDraftTokensInputs.draftLogits = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.draftProbs = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.targetProbs = mBufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.numDraftTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        externalDraftTokensInputs.numDraftTokensHost = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        externalDraftTokensInputs.useDraftLogits
            = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
        externalDraftTokensInputs.useDraftLogitsHost
            = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<bool>::value);
        externalDraftTokensInputs.draftTokenIds
            = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

        dInput->externalDraftTokensInputs = externalDraftTokensInputs;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mSpeculativeDecodingMode.isExplicitDraftTokens());
    mJointDecodingOutput->explicitDraftTokensBuffers = std::move(explicitDraftTokensBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mSpeculativeDecodingMode.isLookaheadDecoding());
    mJointDecodingOutput->lookaheadOutputs = std::move(lookaheadDecodingBuffers);
    mJointDecodingInput->lookaheadInputs->tokensPerStep = mJointDecodingOutput->lookaheadOutputs->generationLengths;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupEagle(EagleBuffers::Inputs eagleBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mSpeculativeDecodingMode.isEagle());
    mJointDecodingOutput->eagleBuffers = std::move(eagleBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::disableLookahead(SizeType32 maxBatchSize, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = SpeculativeDecodingMode::None();
    mMaxDecodingEngineTokens = 1;
    mMaxDecodingDecoderTokens = 1;
    mDecodingMode = executor::DecodingMode::TopKTopP();
    mJointDecodingInput->lookaheadInputs.reset();
    mJointDecodingOutput->newTokensSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mFinishedSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mBatchSlotsDecoder->reshape(ITensor::makeShape({1, maxBatchSize}));
    mNumDecodingEngineTokens.clear();
    mNumDecodingEngineTokens.resize(maxBatchSize, 0);

    std::vector<SamplingConfig> samplingConfigs;
    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlotsSetup);
    SizeType32 bi = 0;
    for (auto const& llmReq : genRequests)
    {
        mNumDecodingEngineTokens[llmReq->mSeqSlot.value()] = 1;
        mMaxNewTokens[llmReq->mSeqSlot.value()] = mMaxSequenceLength - llmReq->getPromptLen();
        samplingConfigs.push_back(llmReq->mSamplingConfig);
        batchSlotsPtr[bi] = llmReq->mSeqSlot.value();
        bi += 1;
    }
    std::optional<SamplingConfig> samplingConfig;
    if (bi > 0)
    {
        samplingConfig = SamplingConfig(samplingConfigs);
    }
    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, bi);
    mDecoder->disableLookahead(samplingConfig, bi, batchSlotsView);

    CudaEvent event{};
    mDecoderStream->record(event);
    mRuntimeStream->wait(event);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    SizeType32 maxTokensPerEngineStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(maxTokensPerEngineStep > 0);
    TLLM_CHECK(maxSequenceLength > 0);
    mActualBatchSize = maxBatchSize;
    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mSinkTokenLength = sinkTokenLength;
    mMaxDecodingEngineTokens = maxTokensPerEngineStep;
    mDecodingMode = mode;

    TLLM_CHECK_WITH_INFO((mMaxDecodingEngineTokens == 1 && mSpeculativeDecodingMode.isNone())
            || (mMaxDecodingEngineTokens > 1 && !mSpeculativeDecodingMode.isNone()),
        "Max tokens per engine step must be equal to 1 when no speculative decoding is configured, "
        "or > 1 for any speculative decoding mode");

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});
    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize, maxBeamWidth});
    auto const maxBatchSizeXmaxTokensPerStep = ITensor::makeShape({maxBatchSize, maxTokensPerEngineStep});
    auto const jointOutputIdsShape = ITensor::makeShape({maxBatchSize, maxBeamWidth, maxSequenceLength});

    auto& dInput = *mJointDecodingInput;
    dInput.maxLength = mMaxSequenceLength;
    dInput.maxAttentionWindow = mMaxAttentionWindow;
    dInput.sinkTokenLength = mSinkTokenLength;
    dInput.stopWordsLists.resize(maxBatchSize);
    dInput.badWordsLists.resize(maxBatchSize);

    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.batchSlots).reshape(maxBatchSizeShape);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mRuntimeStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(inputLengths);

    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(jointOutputIdsShape);

    if (maxBeamWidth > 1)
    {
        dOutput.gatheredIds->reshape(jointOutputIdsShape);

        mOutputBeamHypotheses = std::make_shared<DecodingOutput::BeamHypotheses>();
        mOutputBeamHypotheses->empty(mBufferManager);
        mOutputBeamHypotheses->reshape(1, maxBeamWidth, mMaxSequenceLength);
        mCumLogProbsTmp = mBufferManager.gpu(ITensor::makeShape({1, maxBeamWidth}), nvinfer1::DataType::kFLOAT);
    }
    else
    {
        dOutput.gatheredIds = dOutput.ids;
    }

    mBufferManager.setZero(*dOutput.newTokensSteps);
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    dOutput.finishReasons->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.finishReasons);

    mBatchSlotsSetup->reshape(ITensor::makeShape({maxBatchSize}));
    mBatchSlotsDecoder->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->draftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->targetProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftLogits->reshape(
            ITensor::makeShape({maxBatchSize, maxTokensPerEngineStep, static_cast<SizeType32>(mVocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftTokenIds->reshape(maxBatchSizeXmaxTokensPerStep);
        dInput.externalDraftTokensInputs->numDraftTokens->reshape(ITensor::makeShape({maxBatchSize}));
        dInput.externalDraftTokensInputs->numDraftTokensHost->reshape(ITensor::makeShape({maxBatchSize}));
        dInput.externalDraftTokensInputs->useDraftLogits->reshape(ITensor::makeShape({maxBatchSize}));
        dInput.externalDraftTokensInputs->useDraftLogitsHost->reshape(ITensor::makeShape({maxBatchSize}));
    }

    dOutput.parentIds->reshape(jointOutputIdsShape);
    // use batchSize many entries instead of the usual 1
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    dOutput.newTokensSteps->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize, maxBeamWidth}));

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(jointOutputIdsShape);
    mBufferManager.setZero(*dOutput.logProbs);

    if (maxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(maxBatchSize, maxBeamWidth, mMaxSequenceLength);
    }

    dOutput.logProbsTiled->reshape(ITensor::makeShape({maxSequenceLength, maxBatchSize, maxBeamWidth}));
    mBufferManager.setZero(*dOutput.logProbsTiled);

    const_cast<ITensor&>(*dInput.embeddingBias)
        .reshape(ITensor::makeShape({maxBatchSize, static_cast<SizeType32>(mVocabSizePadded)}));
    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(ITensor::makeShape({maxBatchSize}));

    std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModulePtr = nullptr;
    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingModulePtr = modelConfig.getSpeculativeDecodingModulePtr();
        setupSpeculativeDecoding(modelConfig);
    }
    else
    {
        mMaxDecodingDecoderTokens = 1;
    }

    auto const device = mRuntimeStream->getDevice();
    mDecoderStream = std::make_shared<CudaStream>();
    TLLM_CHECK(mDecoderStream->getDevice() == device);

    mDecoder = IGptDecoder::create(mode, dtype, maxBatchSize, maxBeamWidth, mVocabSize, mVocabSizePadded,
        mMaxSequenceLength, mDecoderStream, speculativeDecodingModulePtr);

    mNbSteps.clear();
    mNbSteps.resize(maxBatchSize, 0);
    mFinished.clear();
    mFinished.resize(maxBatchSize, true);
    mMaxNewTokens.clear();
    mMaxNewTokens.resize(maxBatchSize, 0);
    mBeamWidths.clear();
    mBeamWidths.resize(maxBatchSize, 0);
    mNumDecodingEngineTokens.clear();
    mNumDecodingEngineTokens.resize(maxBatchSize, 0);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::setupSpeculativeDecoding(ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;

    auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
    if (mSpeculativeDecodingMode.isMedusa())
    {
        auto& medusaPaths = const_cast<ITensor&>(*dInput.medusaInputs->medusaPaths);
        medusaPaths.reshape(ITensor::makeShape({mActualBatchSize, speculativeDecodingModule->getMaxDecodingTokens(),
            speculativeDecodingModule->getMaxPathLen()}));
        mBufferManager.setMem(medusaPaths, -1);

        auto& medusaTreeIds = const_cast<ITensor&>(*dInput.medusaInputs->medusaTreeIds);
        medusaTreeIds.reshape(
            ITensor::makeShape({mActualBatchSize, speculativeDecodingModule->getMaxDecodingDraftTokens()}));
        mBufferManager.setZero(medusaTreeIds);
        auto& curTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaCurTokensPerStep);
        auto& targetTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaTargetTokensPerStep);
        curTokensPerStep.reshape(ITensor::makeShape({mActualBatchSize}));
        targetTokensPerStep.reshape(ITensor::makeShape({mActualBatchSize}));
        mBufferManager.setZero(curTokensPerStep);
        mBufferManager.setZero(targetTokensPerStep);
    }

    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        dOutput.speculativeDecodingOutputs->nextDraftTokens->reshape(
            ITensor::makeShape({mActualBatchSize, mMaxDecodingEngineTokens - 1}));
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            dOutput.speculativeDecodingOutputs->nextDraftTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
            dOutput.speculativeDecodingOutputs->prevDraftTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
        }
    }
    if (mSpeculativeDecodingMode.needsKVCacheRewind())
    {
        dOutput.speculativeDecodingOutputs->acceptedTokensLen->reshape(ITensor::makeShape({mActualBatchSize}));
        dOutput.speculativeDecodingOutputs->acceptedLengthsCumSum->reshape(ITensor::makeShape({mActualBatchSize + 1}));
        dOutput.speculativeDecodingOutputs->pathsOffsets->reshape(
            ITensor::makeShape({mActualBatchSize * speculativeDecodingModule->getMaxDraftPathLen()}));
    }

    mMaxDecodingDecoderTokens = mMaxDecodingEngineTokens;

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
    explicitDraftTokensInputs.seqSlots = input.seqSlots;
    mJointDecodingInput->explicitDraftTokensInputs = explicitDraftTokensInputs;

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
        input.eagleInputs->acceptedPaths, input.eagleInputs->chunkedContextNextTokens, input.seqSlots);

    mJointDecodingInput->eagleInputs = eagleInputs;

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
    auto const maxDecodingEngineTokens = maxOfActiveSlots(mNumDecodingEngineTokens, input.active);

    for (SizeType32 si = 0; si < maxDecodingEngineTokens; si += mMaxDecodingDecoderTokens)
    {
        forwardDecoder(si, output, input, forwardType);
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

void GptDecoderBatched::forwardDecoder(
    SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto eventStart = CudaEvent{};
    mRuntimeStream->record(eventStart);

    auto& allTargetLogits = input.logits;
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto constexpr singleRequest = 1;

    TLLM_CHECK(static_cast<SizeType32>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    TLLM_CHECK(sequenceLengths);

    auto batchSlotsDecoderPtr = maxBeamWidth > 1 && input.seqSlots ? bufferCast<SizeType32>(*input.seqSlots)
                                                                   : bufferCast<SizeType32>(*mBatchSlotsDecoder);
    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;
    auto& decoder = *mDecoder;

    if (maxBeamWidth > 1)
    {
        dInput.cacheIndirection = input.cacheIndirection;
        dOutput.cacheIndirection = output.cacheIndirection;
    }

    if (mSpeculativeDecodingMode.isExplicitDraftTokens())
    {
        setExplicitDraftTokensInputs(input);
    }
    else if (mSpeculativeDecodingMode.isEagle())
    {
        setEagleInputs(input);
    }

    bool const async = forwardType == ForwardType::kASYNC;

    if (async)
    {
        mDecoderStream->wait(eventStart.get());
    }

    SizeType32 localBatchDecoderIdx = 0;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        batchSlotsDecoderPtr[step * mActualBatchSize + localBatchDecoderIdx] = bi;
        localBatchDecoderIdx++;
    }

    auto const maxDecodingEngineTokens = maxOfActiveSlots(mNumDecodingEngineTokens, input.active);

    std::vector<SharedConstPtr> logitsVec;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        auto const& targetLogits = allTargetLogits[bi];
        TensorPtr logitsSlice = ITensor::slice(targetLogits, step, singleRequest);
        logitsVec.push_back(logitsSlice);
    }

    TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, step, 1);
    TensorPtr finishedStepsOutput = ITensor::slice(mFinishedSteps, std::min(maxDecodingEngineTokens - 1, step + 1), 1);
    finishedStepsInput->squeeze(0);
    finishedStepsOutput->squeeze(0);
    TensorPtr newTokensStepView = ITensor::slice(dOutput.newTokensSteps, step, mMaxDecodingDecoderTokens);

    dInput.logitsVec = logitsVec;
    dInput.finishReasons = finishedStepsInput;

    if (maxBeamWidth > 1 && input.seqSlots)
    {
        dInput.batchSlots = input.seqSlots;
    }
    else
    {
        TensorPtr batchSlotsDecoderSlice = ITensor::slice(mBatchSlotsDecoder, step, 1);
        batchSlotsDecoderSlice->squeeze(0);
        dInput.batchSlots = batchSlotsDecoderSlice;
    }

    dInput.batchSize = localBatchDecoderIdx;
    if (mSpeculativeDecodingMode.isMedusa())
    {
        dInput.medusaInputs->medusaLogits = input.predictedDraftLogits;
    }

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        dInput.externalDraftTokensInputs->step = step;

        // WAR: reset finished state for generation requests
        if (step == 0)
        {
            BufferManager manager{mDecoderStream};

            for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
            {
                if (mFinished[bi] || !input.active.at(bi))
                {
                    continue;
                }
                TensorPtr finishedStepsView = ITensor::slice(mFinishedSteps, 0, 1);
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

    if (localBatchDecoderIdx > 0)
    {
        if (forwardType == ForwardType::kASYNC)
        {
            decoder.forwardAsync(dOutput, dInput);
        }
        else if (forwardType == ForwardType::kSYNC)
        {
            decoder.forwardSync(dOutput, dInput);
        }
        else
        {
            TLLM_THROW("Unknown ForwardType");
        }
    }

    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        mNbSteps[bi] += 1;
        mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
        TLLM_LOG_DEBUG("decoder slot %d finished by steps: %d", bi, static_cast<std::int32_t>(mFinished[bi]));
    }

    // If last iteration
    if (async && step == maxDecodingEngineTokens - mMaxDecodingDecoderTokens)
    {
        CudaEvent event{};
        mDecoderStream->record(event);
        mRuntimeStream->wait(event);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::updateFinished(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (decoderFinishEvent.active[i] && !mFinished[i])
        {
            auto finishedSum = ITensor::slice(mJointDecodingOutput->finishedSum, i, 1);
            mFinished[i]
                = mFinished[i] || bufferCast<SizeType32>(*finishedSum)[0] == static_cast<SizeType32>(mBeamWidths[i]);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    decoderFinishEvent.event.synchronize();

    updateFinished(decoderFinishEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync(decoder_batch::DecoderFinishedEvent const& decoderFinishEvent,
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    decoderFinishEvent.event.synchronize();

    forwardDispatch(output, input, ForwardType::kSYNC);

    updateFinished(decoderFinishEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// TODO call this at the end of forward if mFinished[i] changes from false to true?
CudaEvent GptDecoderBatched::postProcessRequest(
    SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& stream = mRuntimeStream;
    auto manager = BufferManager{stream};

    auto& dJointInput = *mJointDecodingInput;
    auto& dJointOutput = *mJointDecodingOutput;

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
        tensorrt_llm::kernels::invokeCopyBeamHypotheses(
            dOutput.beamHypotheses, *mOutputBeamHypotheses, *dOutput.cumLogProbs, *mCumLogProbsTmp, *stream, mNumSMs);
        dOutput.beamHypotheses = *mOutputBeamHypotheses;
        dOutput.cumLogProbs = mCumLogProbsTmp;
    }

    kernels::gatherTree(dOutput, dInput, manager, samplingConfig);

    CudaEvent event{};
    stream->record(event);
    mRuntimeStream->wait(event);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatched::newBatch(GenerationInput const& inputs, GenerationOutput const& outputs,
    SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // split batch into single requests
    auto const& inputLengths = inputs.lengths;
    mActualBatchSize = inputLengths->getShape().d[0];
    mNumDecodingEngineTokens.resize(mActualBatchSize);

    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBatchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(mActualBatchSize <= maxBatchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK(samplingConfig.beamWidth <= maxBeamWidth);

    auto const inputIdsShape = inputs.ids->getShape();
    TensorPtr inputIdsFlatView = ITensor::view(inputs.ids);

    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, mActualBatchSize);
    auto batchSlots = BufferRange<SizeType32>(*batchSlotsView);
    std::iota(batchSlots.begin(), batchSlots.end(), 0);

    if (inputs.packed && inputIdsShape.nbDims == 2)
    { // For users still pass inputs.ids with shape [1, num_tokens], do squeeze for them.
        inputIdsFlatView->squeeze(0);
    }
    auto inputLengthsHost = mBufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType32>(*inputLengthsHost);
    auto inputOffset = 0;
    std::vector<SamplingConfig> samplingConfigs;
    for (auto batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        mNumDecodingEngineTokens[batchIdx] = 1;
        auto const inputLength = inputLengthsPtr[batchIdx];
        auto const inputShape = ITensor::makeShape({inputLength});
        TensorPtr inputView;
        if (inputs.packed)
        {
            TLLM_CHECK(inputIdsFlatView->getShape().nbDims == 1);
            inputView = ITensor::slice(inputIdsFlatView, inputOffset, inputLength);
            inputOffset += inputLength;
        }
        else
        {
            inputView = ITensor::slice(inputs.ids, batchIdx, 1);
            inputView->reshape(inputShape);
        }
        auto request = decoder_batch::Request{inputView, inputLength, inputs.maxNewTokens, inputs.endId};

        if (inputs.embeddingBias)
        {
            TLLM_THROW("newBatch doesn't support embeddingBias yet.");
        }
        if (inputs.badWordsList)
        {
            auto const& shape = inputs.badWordsList->getShape();
            if (shape.nbDims == 2)
            {
                request.badWordsList = inputs.badWordsList;
            }
            else
            {
                assert(shape.nbDims == 3);
                TensorPtr badWordsListView = ITensor::slice(inputs.badWordsList, batchIdx, 1);
                badWordsListView->squeeze(0);
                request.badWordsList = badWordsListView;
            }
        }
        if (inputs.stopWordsList)
        {
            TensorPtr stopWordsListView = ITensor::slice(inputs.stopWordsList, batchIdx, 1);
            stopWordsListView->squeeze(0);
            request.stopWordsList = stopWordsListView;
        }
        auto requestSamplingConfig = extractSamplingConfig(samplingConfig, batchIdx);
        requestSamplingConfig.cumLogProbs = {{outputs.cumLogProbs != nullptr}};
        requestSamplingConfig.outputLogProbs = {{outputs.logProbs != nullptr}};
        // Temporary usage of CreateNewDecoderRequests - only used for static batching.
        batch_manager::CreateNewDecoderRequests().newRequest(
            batchIdx, request, requestSamplingConfig, modelConfig, *this, mRuntimeStream, mMaxSequenceLength);
        samplingConfigs.push_back(requestSamplingConfig);
    }

    auto fusedSamplingConfig = samplingConfig;
    fusedSamplingConfig.cumLogProbs = std::vector<bool>(mActualBatchSize, outputs.cumLogProbs != nullptr);
    fusedSamplingConfig.outputLogProbs = std::vector<bool>(mActualBatchSize, outputs.logProbs != nullptr);

    mDecoder->setup(fusedSamplingConfig, mActualBatchSize, batchSlotsView, {*mJointDecodingOutput});

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& logitsShape = input.logits->getShape();
    auto const batchSize = logitsShape.d[0];
    auto constexpr singleRequest = 1;
    std::vector<ITensor::SharedPtr> logits;
    logits.reserve(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto logitsSlice = std::shared_ptr(ITensor::slice(input.logits, batchIdx, singleRequest));
        logits.emplace_back(
            ITensor::view(logitsSlice, ITensor::makeShape({singleRequest, mBeamWidths[batchIdx], logitsShape.d[2]})));
    }

    decoder_batch::Input batchInput{logits};
    batchInput.cacheIndirection = input.cacheIndirection;

    decoder_batch::Output batchOutput;
    batchOutput.cacheIndirection = output.cacheIndirection;
    batchOutput.sequenceLengths = output.sequenceLengths;

    mDecoderFinishEvent = forwardAsync(batchOutput, batchInput);
    mBufferManager.setZero(*mFinishedSum);
    kernels::reduce(
        *mFinishedSum, *ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize), *mRuntimeStream);
    mRuntimeStream->record(mForwardEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mDecoderFinishEvent);
    // wait for mFinishedSum to be updated
    mForwardEvent.synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatched::finalize(SamplingConfig const& samplingConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto batchSlots = bufferCast<SizeType32>(*mBatchSlotsSetup);
    for (SizeType32 batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        auto slot = batchSlots[batchIdx];
        auto requestSamplingConfig = extractSamplingConfig(samplingConfig, slot);
        auto event = postProcessRequest(slot, requestSamplingConfig, /*streaming*/ false);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatched::finalize(SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchSlot, samplingConfig, streaming);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
