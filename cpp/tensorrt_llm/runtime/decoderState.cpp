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

#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::runtime::decoder
{
using TensorPtr = DecoderState::TensorPtr;

BeamSearchBuffers::BeamSearchBuffers(BufferManager const& bufferManager)
    : mOutputBeamHypotheses{}
    , mCumLogProbsTmp(bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT))
{
    mOutputBeamHypotheses.empty(bufferManager);
    mCumLogProbsTmp = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    mNumSMs = deviceProp.multiProcessorCount;
}

void BeamSearchBuffers::reshape(SizeType32 maxBeamWidth, SizeType32 maxSequenceLength)
{
    mOutputBeamHypotheses.reshape(1, maxBeamWidth, maxSequenceLength);
    mCumLogProbsTmp->reshape(ITensor::makeShape({1, maxBeamWidth}));
}

DecoderState::DecoderState(nvinfer1::DataType dtype, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    { // prevent reusing these vars after std::move
        auto dummyLogits = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
        auto endIds = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto batchSlots = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        dInput = std::make_unique<DecodingInput>(
            0, 0, 0, 0, std::move(dummyLogits), std::move(endIds), std::move(batchSlots));
    }
    dInput->sequenceLimitLength = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    { // prevent reusing these vars after std::move
        auto outputIds = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        auto gatheredOutputIds = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
        dOutput = std::make_unique<DecodingOutput>(std::move(outputIds), std::move(gatheredOutputIds));
    }
    dOutput->newTokensSteps = bufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    // we don't need dOutput->lengths because lengths are passed from outside
    dOutput->cumLogProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(bufferManager);
    dOutput->finishReasons
        = bufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);

    dOutput->logProbsTiled = bufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    dInput->stopWordsPtrs = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->badWordsPtrs = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    dInput->embeddingBias = bufferManager.emptyTensor(MemoryType::kGPU, dtype);

    mFinishedSteps = bufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);

    mBeamSearchBuffers = std::make_unique<BeamSearchBuffers>(bufferManager);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::allocateSpeculativeDecodingBuffers(
    SpeculativeDecodingMode const speculativeDecodingMode, nvinfer1::DataType dtype, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = speculativeDecodingMode;

    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;

    auto& dInput = mJointDecodingInput;
    auto& dOutput = mJointDecodingOutput;

    if (speculativeDecodingMode.isMedusa())
    {
        DecodingInput::MedusaInputs medusaInputs;
        medusaInputs.medusaPaths = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTreeIds = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaCurTokensPerStep = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        medusaInputs.medusaTargetTokensPerStep = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        dInput->medusaInputs = medusaInputs;
    }

    DecodingOutput::SpeculativeDecodingOutputs speculativeDecodingOutputs;
    if (speculativeDecodingMode.predictsDraftTokens())
    {
        speculativeDecodingOutputs.nextDraftTokens
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        if (speculativeDecodingMode.variableDraftLength())
        {
            speculativeDecodingOutputs.nextDraftTokensLen
                = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
            speculativeDecodingOutputs.prevDraftTokensLen
                = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        }
    }
    if (speculativeDecodingMode.isLookaheadDecoding())
    {
        dInput->lookaheadInputs = DecodingInput::LookaheadInputs();
    }
    if (speculativeDecodingMode.needsKVCacheRewind())
    {
        speculativeDecodingOutputs.acceptedTokensLen
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.acceptedLengthsCumSum
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
        speculativeDecodingOutputs.pathsOffsets
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }
    dOutput->speculativeDecodingOutputs = speculativeDecodingOutputs;

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        DecodingInput::ExternalDraftTokensInputs externalDraftTokensInputs;

        externalDraftTokensInputs.draftLogits = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.draftProbs = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.targetProbs = bufferManager.emptyTensor(MemoryType::kGPU, dtype);
        externalDraftTokensInputs.numDraftTokens = bufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
        externalDraftTokensInputs.numDraftTokensHost = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
        externalDraftTokensInputs.useDraftLogits
            = bufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
        externalDraftTokensInputs.useDraftLogitsHost
            = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<bool>::value);
        externalDraftTokensInputs.draftTokenIds
            = bufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

        dInput->externalDraftTokensInputs = externalDraftTokensInputs;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setup(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& stream = bufferManager.getStream();

    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(mMaxDecodingEngineTokens > 0);
    TLLM_CHECK(maxSequenceLength > 0);
    mActualBatchSize = maxBatchSize;
    mMaxBatchSize = maxBatchSize;
    mMaxBeamWidth = maxBeamWidth;
    mMaxSequenceLength = maxSequenceLength;

    // setup input
    auto& dInput = *mJointDecodingInput;
    dInput.maxLength = mMaxSequenceLength;
    dInput.maxAttentionWindow = maxAttentionWindow;
    dInput.sinkTokenLength = sinkTokenLength;
    dInput.stopWordsLists.resize(mMaxBatchSize);
    dInput.badWordsLists.resize(mMaxBatchSize);

    auto const maxBatchSizeShape = ITensor::makeShape({mMaxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({mMaxBatchSize, mMaxBeamWidth});

    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.batchSlots).reshape(maxBatchSizeShape);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, stream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(inputLengths);

    dInput.beamWidths.clear();
    dInput.beamWidths.resize(mMaxBatchSize, 0);

    dInput.numDecodingEngineTokens.clear();
    dInput.numDecodingEngineTokens.resize(mMaxBatchSize, 0);

    auto const jointOutputIdsShape = ITensor::makeShape({mActualBatchSize, mMaxBeamWidth, mMaxSequenceLength});

    // setup output
    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(jointOutputIdsShape);

    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({mMaxDecodingEngineTokens, mMaxBatchSize, mMaxBeamWidth});
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*mFinishedSteps);

    dOutput.finishReasons->reshape(maxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*dOutput.finishReasons);

    dOutput.parentIds->reshape(jointOutputIdsShape);

    dOutput.finishedSum->reshape(maxBatchSizeShape);
    bufferManager.setZero(*dOutput.finishedSum);

    dOutput.newTokensSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*dOutput.newTokensSteps);

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(jointOutputIdsShape);
    bufferManager.setZero(*dOutput.logProbs);

    dOutput.logProbsTiled->reshape(ITensor::makeShape({mMaxSequenceLength, mMaxBatchSize, mMaxBeamWidth}));
    bufferManager.setZero(*dOutput.logProbsTiled);

    if (mMaxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(mMaxBatchSize, mMaxBeamWidth, mMaxSequenceLength);
        mBeamSearchBuffers->reshape(mMaxBeamWidth, mMaxSequenceLength);

        dOutput.gatheredIds->reshape(jointOutputIdsShape);
    }
    else
    {
        dOutput.gatheredIds = dOutput.ids;
    }

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    const_cast<ITensor&>(*dInput.embeddingBias)
        .reshape(ITensor::makeShape({mMaxBatchSize, static_cast<SizeType32>(vocabSizePadded)}));
    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(maxBatchSizeShape);
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(maxBatchSizeShape);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupSpeculativeDecoding(SpeculativeDecodingMode const& speculativeDecodingMode,
    SizeType32 maxTokensPerEngineStep, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    BufferManager const& bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;

    TLLM_CHECK(maxTokensPerEngineStep > 0);
    mMaxDecodingEngineTokens = maxTokensPerEngineStep;

    TLLM_CHECK_WITH_INFO((mMaxDecodingEngineTokens == 1 && speculativeDecodingMode.isNone())
            || (mMaxDecodingEngineTokens > 1 && !speculativeDecodingMode.isNone()),
        "Max tokens per engine step must be equal to 1 when no speculative decoding is configured, "
        "or > 1 for any speculative decoding mode");

    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({mMaxDecodingEngineTokens, mMaxBatchSize, mMaxBeamWidth});
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*mFinishedSteps);
    dOutput.newTokensSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    bufferManager.setZero(*dOutput.newTokensSteps);

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        mMaxDecodingDecoderTokens = mMaxDecodingEngineTokens;
    }
    else
    {
        mMaxDecodingDecoderTokens = 1;
    }

    if (speculativeDecodingMode.isNone())
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return;
    }

    auto const maxBatchSizeShape = ITensor::makeShape({mMaxBatchSize});
    auto const maxBatchSizeXmaxTokensPerStep = ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens});

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        auto const probsShape = ITensor::makeShape(
            {mMaxBatchSize, mMaxBeamWidth, mMaxSequenceLength, static_cast<SizeType32>(vocabSizePadded)});
        dInput.externalDraftTokensInputs->draftProbs->reshape(probsShape);
        dInput.externalDraftTokensInputs->targetProbs->reshape(probsShape);
        dInput.externalDraftTokensInputs->draftLogits->reshape(
            ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens, static_cast<SizeType32>(vocabSizePadded)}));
        dInput.externalDraftTokensInputs->draftTokenIds->reshape(maxBatchSizeXmaxTokensPerStep);
        dInput.externalDraftTokensInputs->numDraftTokens->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->numDraftTokensHost->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->useDraftLogits->reshape(maxBatchSizeShape);
        dInput.externalDraftTokensInputs->useDraftLogitsHost->reshape(maxBatchSizeShape);
    }

    if (speculativeDecodingMode.isMedusa())
    {
        auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
        auto& medusaPaths = const_cast<ITensor&>(*dInput.medusaInputs->medusaPaths);
        medusaPaths.reshape(ITensor::makeShape({mMaxBatchSize, speculativeDecodingModule->getMaxDecodingTokens(),
            speculativeDecodingModule->getMaxPathLen()}));
        bufferManager.setMem(medusaPaths, -1);

        auto& medusaTreeIds = const_cast<ITensor&>(*dInput.medusaInputs->medusaTreeIds);
        medusaTreeIds.reshape(
            ITensor::makeShape({mMaxBatchSize, speculativeDecodingModule->getMaxDecodingDraftTokens()}));
        bufferManager.setZero(medusaTreeIds);
        auto& curTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaCurTokensPerStep);
        auto& targetTokensPerStep = const_cast<ITensor&>(*dInput.medusaInputs->medusaTargetTokensPerStep);
        curTokensPerStep.reshape(maxBatchSizeShape);
        targetTokensPerStep.reshape(maxBatchSizeShape);
        bufferManager.setZero(curTokensPerStep);
        bufferManager.setZero(targetTokensPerStep);
    }

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        dOutput.speculativeDecodingOutputs->nextDraftTokens->reshape(
            ITensor::makeShape({mMaxBatchSize, mMaxDecodingEngineTokens - 1}));
        if (speculativeDecodingMode.variableDraftLength())
        {
            dOutput.speculativeDecodingOutputs->nextDraftTokensLen->reshape(maxBatchSizeShape);
            dOutput.speculativeDecodingOutputs->prevDraftTokensLen->reshape(maxBatchSizeShape);
        }
    }
    if (speculativeDecodingMode.needsKVCacheRewind())
    {
        auto const speculativeDecodingModule = modelConfig.getSpeculativeDecodingModulePtr();
        dOutput.speculativeDecodingOutputs->acceptedTokensLen->reshape(maxBatchSizeShape);
        dOutput.speculativeDecodingOutputs->acceptedLengthsCumSum->reshape(ITensor::makeShape({mMaxBatchSize + 1}));
        dOutput.speculativeDecodingOutputs->pathsOffsets->reshape(
            ITensor::makeShape({mMaxBatchSize * speculativeDecodingModule->getMaxDraftPathLen()}));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mJointDecodingOutput->explicitDraftTokensBuffers = std::move(explicitDraftTokensBuffers);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mJointDecodingOutput->lookaheadOutputs = std::move(lookaheadDecodingBuffers);
    mJointDecodingInput->lookaheadInputs->tokensPerStep = mJointDecodingOutput->lookaheadOutputs->generationLengths;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::setupEagle(EagleBuffers::Inputs eagleBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mJointDecodingOutput->eagleBuffers = std::move(eagleBuffers);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void DecoderState::disableLookahead(SizeType32 maxBatchSize, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mSpeculativeDecodingMode = SpeculativeDecodingMode::None();

    mMaxDecodingEngineTokens = 1;
    mMaxDecodingDecoderTokens = 1;
    mJointDecodingInput->lookaheadInputs.reset();
    mJointDecodingOutput->newTokensSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mFinishedSteps->reshape(ITensor::makeShape({1, maxBatchSize, 1}));
    mJointDecodingInput->numDecodingEngineTokens.clear();
    mJointDecodingInput->numDecodingEngineTokens.resize(maxBatchSize, 0);

    for (auto const& llmReq : genRequests)
    {
        mJointDecodingInput->numDecodingEngineTokens[llmReq->mSeqSlot.value()] = 1;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TensorPtr DecoderState::getFinishedSum() const
{
    return ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize);
}

TensorPtr DecoderState::getFinishReasons() const
{
    return ITensor::slice(mJointDecodingOutput->finishReasons, 0, mActualBatchSize);
}

TensorPtr DecoderState::getIds() const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto tensor = ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return tensor;
}

TensorPtr DecoderState::getIds(SizeType32 batchIdx) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
    tensor->squeeze(0);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return tensor;
}

TensorPtr DecoderState::getGatheredIds() const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, 0, mActualBatchSize);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return tensor;
}

TensorPtr DecoderState::getGatheredIds(SizeType32 batchIdx) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, batchIdx, 1);
    tensor->squeeze(0);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return tensor;
}

TensorPtr DecoderState::getParentIds() const
{
    return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
}

TensorPtr DecoderState::getCumLogProbs() const
{
    return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
}

TensorPtr DecoderState::getCumLogProbs(SizeType32 batchIdx) const
{
    auto tensor = ITensor::slice(mJointDecodingOutput->cumLogProbs, batchIdx, 1);
    tensor->squeeze(0);
    return tensor;
}

TensorPtr DecoderState::getLogProbs() const
{
    return ITensor::slice(mJointDecodingOutput->logProbs, 0, mActualBatchSize);
}

TensorPtr DecoderState::getLogProbs(SizeType32 batchIdx) const
{
    auto tensor = ITensor::slice(mJointDecodingOutput->logProbs, batchIdx, 1);
    tensor->squeeze(0);
    return tensor;
}

TensorPtr DecoderState::getAllNewTokens() const
{
    return mJointDecodingOutput->newTokensSteps;
}

TensorPtr DecoderState::getNextDraftTokens() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokens;
}

TensorPtr DecoderState::getPrevDraftTokensLengths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->prevDraftTokensLen;
}

TensorPtr DecoderState::getNextDraftTokensLengths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokensLen;
}

TensorPtr DecoderState::getAcceptedLengthsCumSum() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->acceptedLengthsCumSum;
}

TensorPtr DecoderState::getAcceptedPackedPaths() const
{
    return mJointDecodingOutput->speculativeDecodingOutputs->pathsOffsets;
}

TensorPtr DecoderState::getFinishedSteps() const
{
    return mFinishedSteps;
}

SizeType32 DecoderState::getActualBatchSize() const
{
    return mActualBatchSize;
}

void DecoderState::setActualBatchSize(SizeType32 actualBatchSize)
{
    TLLM_CHECK(actualBatchSize <= mMaxBatchSize);
    mActualBatchSize = actualBatchSize;
}

SizeType32 DecoderState::getMaxBeamWidth() const
{
    return mMaxBeamWidth;
}

SizeType32 DecoderState::getMaxSequenceLength() const
{
    return mMaxSequenceLength;
}

SizeType32 DecoderState::getMaxDecodingDecoderTokens() const
{
    return mMaxDecodingDecoderTokens;
}

SizeType32 DecoderState::getMaxDecodingEngineTokens() const
{
    return mMaxDecodingEngineTokens;
}

SpeculativeDecodingMode DecoderState::getSpeculativeDecodingMode() const
{
    return mSpeculativeDecodingMode;
}

BeamSearchBuffers const& DecoderState::getBeamSearchBuffers() const
{
    return *mBeamSearchBuffers;
}

DecodingInput& DecoderState::getJointDecodingInput() const
{
    return *mJointDecodingInput;
}

DecodingOutput& DecoderState::getJointDecodingOutput() const
{
    return *mJointDecodingOutput;
}

} // namespace tensorrt_llm::runtime::decoder
