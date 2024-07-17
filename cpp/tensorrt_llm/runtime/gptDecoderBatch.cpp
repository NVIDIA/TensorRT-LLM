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
#include "tensorrt_llm/runtime/gptDecoderBatch.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>
#include <cassert>
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
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

    // beam search layer
    samplingConfig.beamSearchDiversityRate = batchSamplingConfig.beamSearchDiversityRate;
    samplingConfig.lengthPenalty = batchSamplingConfig.lengthPenalty;
    samplingConfig.earlyStopping = batchSamplingConfig.earlyStopping;
    samplingConfig.normalizeLogProbs = batchSamplingConfig.normalizeLogProbs;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return samplingConfig;
}

} // namespace

GptDecoderBatch::GptDecoderBatch(std::size_t vocabSize, std::size_t vocabSizePadded,
    GptDecoderBatch::CudaStreamPtr stream, SpeculativeDecodingMode const& speculativeDecodingMode)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mStream{std::move(stream)}
    , mBufferManager{mStream}
    , mSpeculativeDecodingMode{speculativeDecodingMode}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mJointDecodingInput;
    auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dInput = std::make_unique<DecodingInput>(0, 0, 0, 0, std::move(dummyLogits), std::move(endIds));

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds));

    dOutput->newTokensSteps = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    mFinishedSteps
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    mDraftProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    mTargetProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    mBatchSlotsSetup = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    mBatchSlotsDecoder = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    mBatchSlotsAcceptTokens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    mBatchSlotsAcceptLogits = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    mFinishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
    // we don't need dOutput->lengths because lengths are passed from outside
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);

    mNumDraftTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mCurandStates = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT8);
    mDraftTokenIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    mDraftLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    mTargetLogitsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<float*>::value);

    dInput->stopWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    dInput->badWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType32>::value);
    dInput->embeddingBias = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    if (!mSpeculativeDecodingMode.isNone())
    {
        allocateSpeculativeDecodingBuffers();
    }

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::allocateSpeculativeDecodingBuffers()
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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mSpeculativeDecodingMode.isExplicitDraftTokens());
    mJointDecodingOutput->explicitDraftTokensBuffers = std::move(explicitDraftTokensBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    SizeType32 maxTokensPerEngineStep, bool fusedDecoder, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
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
    mFusedDecoder = fusedDecoder;
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

    auto& dInput = *mJointDecodingInput;
    dInput.maxLength = mMaxSequenceLength;
    dInput.maxAttentionWindow = mMaxAttentionWindow;
    dInput.sinkTokenLength = mSinkTokenLength;
    const_cast<ITensor&>(*dInput.endIds).reshape(maxBatchSizeXmaxBeamWidth);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(maxBatchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(inputLengths);

    auto const jointOutputIdsShape = ITensor::makeShape({maxBatchSize, maxBeamWidth, maxSequenceLength});

    auto& dOutput = *mJointDecodingOutput;
    dOutput.ids->reshape(jointOutputIdsShape);

    mBufferManager.setZero(*dOutput.newTokensSteps);
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    if (mFusedDecoder)
    {
        mBatchSlotsSetup->reshape(ITensor::makeShape({maxBatchSize}));
        mBatchSlotsDecoder->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));
        mBatchSlotsAcceptTokens->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));
        mBatchSlotsAcceptLogits->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));
    }

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        mDraftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
        mTargetProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerEngineStep, maxBeamWidth, static_cast<SizeType32>(mVocabSizePadded)}));
    }

    dOutput.parentIds->reshape(jointOutputIdsShape);
    // use batchSize many entries instead of the usual 1
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    dOutput.newTokensSteps->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize, maxBeamWidth}));

    dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.cumLogProbs);

    dOutput.logProbs->reshape(ITensor::makeShape({maxBatchSize, maxBeamWidth, mMaxSequenceLength}));
    mBufferManager.setZero(*dOutput.logProbs);

    if (maxBeamWidth > 1)
    {
        dOutput.beamHypotheses.reshape(maxBatchSize, maxBeamWidth, mMaxSequenceLength);
    }
    else
    {
        dOutput.beamHypotheses.release();
    }

    // speculative decoding only works for beam width == 1
    mDraftTokenIds->reshape(maxBatchSizeXmaxTokensPerStep);
    mDraftLogits->reshape(
        ITensor::makeShape({maxBatchSize, maxTokensPerEngineStep, static_cast<SizeType32>(mVocabSizePadded)}));
    mAcceptByLogits.resize(maxBatchSize);
    mNumDraftTokens->reshape(ITensor::makeShape({maxBatchSize, 1}));
    mCurandStates->reshape(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}));
    mTargetLogitsPtrs->reshape(ITensor::makeShape({maxTokensPerEngineStep, maxBatchSize}));

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

    auto const numOfDecoders = mFusedDecoder ? 1 : maxBatchSize;
    auto const maxBatchSizePerDecoder = mFusedDecoder ? maxBatchSize : 1;
    auto const device = mStream->getDevice();
    mStreams.resize(numOfDecoders);
    mDecoders.resize(numOfDecoders);
    for (SizeType32 i = 0; i < numOfDecoders; ++i)
    {
        mStreams[i] = std::make_shared<CudaStream>();
        TLLM_CHECK(mStreams[i]->getDevice() == device);

        mDecoders[i] = IGptDecoder::create(mode, dtype, maxBatchSizePerDecoder, maxBeamWidth, mVocabSize,
            mVocabSizePadded, mMaxSequenceLength, mStreams[i], speculativeDecodingModulePtr);
    }

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

    mDecodingInputs.resize(maxBatchSize);
    mDecodingOutputs.resize(maxBatchSize);
    for (SizeType32 i = 0; i < maxBatchSize; ++i)
    {
        mDecodingInputs[i].reset();
        mDecodingOutputs[i].reset();
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setupSpeculativeDecoding(ModelConfig const& modelConfig)
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

void GptDecoderBatch::newRequest(
    SizeType32 batchSlot, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchSlot >= 0);
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(0 <= batchSize && batchSlot < batchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const beamWidth = samplingConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
        tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (" FMT_DIM ") passed to decoder setup function.",
            beamWidth, maxBeamWidth));
    auto const& requestIds = request.ids;
    auto const inputLength = request.inputLen;
    auto const numDecodingEngineTokens = request.generatedTokensPerEngineStep;
    auto const numDecodingDraftEngineTokens = numDecodingEngineTokens - 1;
    auto const maxNewTokens
        = request.maxNewTokens.value_or(mMaxSequenceLength - inputLength - numDecodingDraftEngineTokens);

    TLLM_CHECK_WITH_INFO(inputLength + maxNewTokens + numDecodingDraftEngineTokens <= mMaxSequenceLength,
        tc::fmtstr(
            "Input length (%d) + max new tokens (%d) + draft tokens (%d) must be less than max sequence length (%d).",
            inputLength, maxNewTokens, numDecodingDraftEngineTokens, mMaxSequenceLength));
    TLLM_CHECK(requestIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = request.endId.value_or(-1);

    auto constexpr localBatchSize = 1;

    auto const decoderIdx = mFusedDecoder ? 0 : batchSlot;
    auto const& stream = mStreams.at(decoderIdx);
    BufferManager manager{stream};

    // input
    auto& dJointInput = *mJointDecodingInput;
    auto& dInput = mDecodingInputs.at(batchSlot);

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchSlot, localBatchSize)};
    kernels::invokeFill(*endIdTensorPtr, endId, *stream);
    dInput = std::make_unique<DecodingInput>(
        inputLength, mMaxAttentionWindow, mSinkTokenLength, localBatchSize, dJointInput.logits, endIdTensorPtr);

    TensorPtr embeddingBiasSlice
        = ITensor::slice(constPointerCast(dJointInput.embeddingBias), batchSlot, localBatchSize);
    if (request.embeddingBias)
    {
        TLLM_CHECK(request.embeddingBias->getShape().nbDims == 2);
        TLLM_CHECK(request.embeddingBias->getShape().d[0] == 1);
        TLLM_CHECK_WITH_INFO(request.embeddingBias->getShape().d[1] == static_cast<SizeType32>(mVocabSize),
            "The embedding bias shape is not as expected. Expected last dimension to be same as vocab size: %lu.",
            mVocabSize);

        manager.copy(*request.embeddingBias, *embeddingBiasSlice);
        dInput->embeddingBias = embeddingBiasSlice;
    }
    else
    {
        manager.setZero(*embeddingBiasSlice);
    }

    auto setupWords = [fusedDecoder = mFusedDecoder](SharedConstPtr& inputWordsList, TensorPtr const& requestWordsList,
                          SharedConstPtr& jointWordsPtrs, SharedConstPtr& jointWordsLens, SharedConstPtr& wordsPtrs,
                          SharedConstPtr& wordsLens, SizeType32& inputMaxStopWordsLen, SizeType32& maxWordsLen,
                          SizeType32 localBatchSize, SizeType32 batchSlot)
    {
        if (requestWordsList)
        {
            auto const wordsLen = requestWordsList->getShape().d[1];
            BufferRange<int32_t*>(*constPointerCast(jointWordsPtrs))[batchSlot]
                = bufferCast<TokenIdType>(*requestWordsList);
            bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = wordsLen;
            // FIXME(nkorobov): this is monotonically growing size
            maxWordsLen = std::max(static_cast<SizeType32>(wordsLen), maxWordsLen);
            if (!fusedDecoder)
            {
                wordsPtrs = ITensor::slice(jointWordsPtrs, batchSlot, localBatchSize);
                wordsLens = ITensor::slice(jointWordsLens, batchSlot, localBatchSize);
                inputMaxStopWordsLen = wordsLen;
            }
            // NOTE(nkorobov): dInput-><name>WordsList is not used in gptDecoder, but required to keep <name>WordsList's
            // memory allocated
            inputWordsList = requestWordsList;
        }
        else
        {
            bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = 0;
            inputMaxStopWordsLen = 0;
        }
    };

    setupWords(dInput->stopWordsList, request.stopWordsList, dJointInput.stopWordsPtrs, dJointInput.stopWordsLens,
        dInput->stopWordsPtrs, dInput->stopWordsLens, dInput->maxStopWordsLen, mMaxStopWordsLen, localBatchSize,
        batchSlot);
    dJointInput.maxStopWordsLen = mMaxStopWordsLen;

    setupWords(dInput->badWordsList, request.badWordsList, dJointInput.badWordsPtrs, dJointInput.badWordsLens,
        dInput->badWordsPtrs, dInput->badWordsLens, dInput->maxBadWordsLen, mMaxBadWordsLen, localBatchSize, batchSlot);
    dJointInput.maxBadWordsLen = mMaxBadWordsLen;

    TensorPtr sequenceLimitLength{
        ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchSlot, localBatchSize)};
    kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, *stream);
    dInput->sequenceLimitLength = std::move(sequenceLimitLength);
    TensorPtr inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchSlot, localBatchSize)};
    kernels::invokeFill(*inputLengths, inputLength, *stream);
    dInput->lengths = inputLengths;

    // output
    auto& dJointOutput = *mJointDecodingOutput;
    auto& dOutput = mDecodingOutputs.at(batchSlot);
    auto const outputIdsShape = ITensor::makeShape({localBatchSize, beamWidth, mMaxSequenceLength});

    TensorPtr outputIds = ITensor::slice(dJointOutput.ids, batchSlot, localBatchSize);
    outputIds->reshape(outputIdsShape);
    dOutput = std::make_unique<DecodingOutput>(outputIds);

    dOutput->finishedSum = ITensor::slice(dJointOutput.finishedSum, batchSlot, localBatchSize);
    manager.setZero(*dOutput->finishedSum);

    dOutput->newTokensVec.resize(mMaxDecodingEngineTokens);
    for (SizeType32 ti = 0; ti < mMaxDecodingEngineTokens; ++ti)
    {
        TensorPtr newTokensStepView = ITensor::slice(dJointOutput.newTokensSteps, ti, 1);
        newTokensStepView->squeeze(0);
        dOutput->newTokensVec[ti] = ITensor::slice(newTokensStepView, batchSlot, localBatchSize);
        manager.setZero(*dOutput->newTokensVec[ti]);
    }

    // FIXME(nkorobov): we call setZero mMaxDecodingEngineTokens times for only 1 element
    for (SizeType32 ti = 0; ti < mMaxDecodingEngineTokens; ++ti)
    {
        TensorPtr finishedStepsView = ITensor::slice(mFinishedSteps, ti, 1);
        finishedStepsView->squeeze(0);
        TensorPtr finishedSteps = ITensor::slice(finishedStepsView, batchSlot, localBatchSize);
        manager.setZero(*finishedSteps);
    }

    // cumLogProb is mandatory for beamWidth > 1
    dOutput->cumLogProbs = nullptr;
    if ((samplingConfig.cumLogProbs.has_value() && samplingConfig.cumLogProbs->at(0)) || beamWidth > 1)
    {
        dOutput->cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, localBatchSize);
        manager.setZero(*dOutput->cumLogProbs);
    }

    dOutput->logProbs = nullptr;
    if (samplingConfig.outputLogProbs.has_value() && samplingConfig.outputLogProbs->at(0))
    {
        dOutput->logProbs = ITensor::slice(dJointOutput.logProbs, batchSlot, localBatchSize);
        manager.setZero(*dOutput->logProbs);
    }

    if (beamWidth > 1)
    {
        kernels::invokeFill(
            *IBuffer::slice(dOutput->cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *stream);
        dOutput->parentIds = ITensor::slice(dJointOutput.parentIds, batchSlot, localBatchSize);
        dOutput->parentIds->reshape(outputIdsShape);
        manager.setZero(*dOutput->parentIds);
        dOutput->beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, localBatchSize);
        dOutput->beamHypotheses.init(manager, endId);
    }

    // Speculative execution
    if (numDecodingEngineTokens > 1)
    {
        TLLM_CHECK(beamWidth == 1);
        newRequestSpeculativeDecoding(batchSlot, request, samplingConfig);
    }

    // remaining
    if (!mFusedDecoder)
    {
        mDecoders[decoderIdx]->setup(samplingConfig, localBatchSize);
    }
    mBeamWidths[batchSlot] = beamWidth;
    mNbSteps[batchSlot] = 0;
    mFinished[batchSlot] = false;
    mMaxNewTokens[batchSlot] = maxNewTokens;
    mNumDecodingEngineTokens[batchSlot] = numDecodingEngineTokens;

    // copy the request ids into outputIds
    auto const requestIdsShape = requestIds->getShape();
    auto inputIdsView = ITensor::view(requestIds, ITensor::makeShape({localBatchSize, requestIdsShape.d[0]}));
    auto outputIdsView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, endId, *stream);
    kernels::tileTensor(*outputIdsView, *inputIdsView, beamWidth, *stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequestSpeculativeDecoding(
    SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mAcceptByLogits[batchIdx] = false;

    if (mSpeculativeDecodingMode.predictsDraftTokens())
    {
        auto constexpr decoderIdx = 0;
        auto const& stream = mStreams.at(decoderIdx);
        BufferManager manager{stream};

        auto& dJointOutput = *mJointDecodingOutput;

        auto constexpr localBatchSize = 1;
        TensorPtr nextDraftTokens
            = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokens, batchIdx, localBatchSize);
        // FIXME(nkorobov): can we skip this?
        manager.setZero(*nextDraftTokens);
        if (mSpeculativeDecodingMode.variableDraftLength())
        {
            TensorPtr nextDraftTokensLen
                = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokensLen, batchIdx, localBatchSize);
            manager.setZero(*nextDraftTokensLen);
        }
    }

    if (mSpeculativeDecodingMode.isDraftTokensExternal())
    {
        newRequestDraftTokensExternal(batchIdx, request, samplingConfig);
    }
    else if (mSpeculativeDecodingMode.isMedusa())
    {
        newRequestMedusa(batchIdx, request);
    }
    else if (mSpeculativeDecodingMode.isLookaheadDecoding())
    {
        newRequestLookahead(batchIdx, request);
    }
    else if (mSpeculativeDecodingMode.isExplicitDraftTokens())
    {
        newRequestExplicitDraftTokens(batchIdx, request);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequestDraftTokensExternal(
    SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mFusedDecoder, "Speculative decoding requires fused decoder");
    auto constexpr decoderIdx = 0;
    auto const& stream = mStreams.at(decoderIdx);
    BufferManager manager{stream};

    auto constexpr localBatchSize = 1;

    auto const numDraftTokens = request.generatedTokensPerEngineStep - 1;
    if (request.draftLogits.has_value())
    {
        TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());
        mAcceptByLogits[batchIdx] = true;

        TensorPtr draftLogitsReqBatchSlice = ITensor::slice(mDraftLogits, batchIdx, localBatchSize);
        draftLogitsReqBatchSlice->squeeze(0);
        TensorPtr draftLogitsReqTokensSlice = ITensor::slice(draftLogitsReqBatchSlice, 0, numDraftTokens);
        manager.copy(*draftLogitsView, *draftLogitsReqTokensSlice);
    }
    TensorPtr draftTokensReqBatchSlice = ITensor::slice(mDraftTokenIds, batchIdx, localBatchSize);
    draftTokensReqBatchSlice->squeeze(0);
    TensorPtr draftTokensReqTokensSlice = ITensor::slice(draftTokensReqBatchSlice, 0, numDraftTokens);
    TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({numDraftTokens}));
    manager.copy(*draftTokensView, *draftTokensReqTokensSlice);

    auto const curandStatesView = ITensor::slice(mCurandStates, batchIdx, localBatchSize);
    auto curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*curandStatesView));
    if (samplingConfig.randomSeed.has_value())
    {
        tk::invokeCurandInitialize(
            curandState, nullptr, localBatchSize, samplingConfig.randomSeed.value()[0], stream->get());
    }
    else
    {
        tk::invokeCurandInitialize(curandState, nullptr, localBatchSize, 0, stream->get());
    }
    auto numDraftTokensView = ITensor::slice(mNumDraftTokens, batchIdx, localBatchSize);
    kernels::invokeFill(*numDraftTokensView, numDraftTokens, *stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequestMedusa(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mFusedDecoder, "Medusa requires fused decoder");
    auto constexpr decoderIdx = 0;
    auto const& stream = mStreams.at(decoderIdx);
    BufferManager manager{stream};

    auto& dJointInput = *mJointDecodingInput;

    auto constexpr localBatchSize = 1;

    TensorPtr curTokensPerStepSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaCurTokensPerStep), batchIdx, localBatchSize);
    // Context phase Medusa processes 1 token only, new value from targetTokensPerStep will be filled at the end
    // of first decoder
    kernels::invokeFill(*curTokensPerStepSlice, 1, *stream);
    TensorPtr targetTokensPerStepSlice = ITensor::slice(
        constPointerCast(dJointInput.medusaInputs->medusaTargetTokensPerStep), batchIdx, localBatchSize);
    auto const generatedTokensPerEngineStep = request.generatedTokensPerEngineStep;
    TLLM_CHECK_WITH_INFO(generatedTokensPerEngineStep <= mMaxDecodingEngineTokens,
        "Tokens per step for (%d) is larger than maximum tokens per step (%d)", generatedTokensPerEngineStep,
        mMaxDecodingEngineTokens);
    kernels::invokeFill(*targetTokensPerStepSlice, generatedTokensPerEngineStep, *stream);

    TensorPtr pathsSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaPaths), batchIdx, localBatchSize);
    manager.copy(*request.medusaPaths, *pathsSlice);

    TensorPtr treeIdsSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaTreeIds), batchIdx, localBatchSize);
    manager.copy(*request.medusaTreeIds, *treeIdsSlice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequestLookahead(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mFusedDecoder, "Lookahead decoding requires fused decoder");
    // TODO(nkorobov) add lookahead layer
    TLLM_LOG_WARNING("Lookahead decoding is not supported yet.");
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequestExplicitDraftTokens(SizeType32 batchIdx, decoder_batch::Request const& request)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mFusedDecoder, "Explicit draft tokens decoding requires fused decoder");
    TLLM_CHECK(mJointDecodingOutput->explicitDraftTokensBuffers);

    auto constexpr localBatchSize = 1;
    auto& stream = mStream;

    TensorPtr positionIdsBaseSlice
        = ITensor::slice(mJointDecodingOutput->explicitDraftTokensBuffers->positionIdsBase, batchIdx, localBatchSize);
    kernels::invokeFill(*positionIdsBaseSlice, request.inputLen, *stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setExplicitDraftTokensInputs(decoder_batch::Input const& input)
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

void GptDecoderBatch::newRequests(std::vector<SizeType32> const& seqSlots,
    std::vector<decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlotsSetup);
    SizeType32 const localBatchSize = seqSlots.size();
    for (SizeType32 bi = 0; bi < localBatchSize; ++bi)
    {
        newRequest(seqSlots[bi], requests[bi], samplingConfigs[bi]);
        if (mFusedDecoder)
        {
            batchSlotsPtr[bi] = seqSlots[bi];
        }
    }
    if (mFusedDecoder)
    {
        TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, localBatchSize);
        auto fusedSamplingConfig = SamplingConfig(samplingConfigs);
        mDecoders[0]->setup(
            fusedSamplingConfig, localBatchSize, bufferCast<SizeType32>(*batchSlotsView), {*mJointDecodingOutput});

        auto const& stream = mStreams.at(0);
        CudaEvent event{};
        stream->record(event);
        mStream->wait(event);
    }
    else
    {
        for (SizeType32 bi = 0; bi < localBatchSize; ++bi)
        {
            auto const& stream = mStreams.at(bi);
            CudaEvent event{};
            stream->record(event);
            mStream->wait(event);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardDispatch(
    decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    auto const maxDecodingEngineTokens
        = *std::max_element(std::begin(mNumDecodingEngineTokens), std::end(mNumDecodingEngineTokens));

    for (SizeType32 si = 0; si < maxDecodingEngineTokens; si += mMaxDecodingDecoderTokens)
    {
        if (!mFusedDecoder)
        {
            TLLM_CHECK_WITH_INFO(forwardType == ForwardType::kASYNC, "Unfused decoder supports only async forward");
            forwardUnfusedDecoder(si, output, input, forwardType);
        }
        else
        {
            forwardFusedDecoder(si, output, input, forwardType);
        }
    }
}

GptDecoderBatch::TokenPtr GptDecoderBatch::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    forwardDispatch(output, input, ForwardType::kASYNC);

    CudaEvent eventStop{};
    mStream->record(eventStop);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::Token>(std::move(eventStop), input.active);
}

void GptDecoderBatch::forwardUnfusedDecoder(
    SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto eventStart = CudaEvent{};
    mStream->record(eventStart);

    auto& allTargetLogits = input.logits;
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    TLLM_CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType32>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType32>::value);

    TLLM_CHECK(static_cast<SizeType32>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    TLLM_CHECK(sequenceLengths);

    bool const async = forwardType == ForwardType::kASYNC;

    auto constexpr singleRequest = 1;

    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }

        auto const& stream = mStreams.at(bi);
        if (async)
        {
            stream->wait(eventStart);
        }

        auto& targetLogits = allTargetLogits[bi];
        auto& dInput = *mDecodingInputs[bi];
        auto& dOutput = *mDecodingOutputs[bi];
        auto& decoder = *mDecoders[bi];

        TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, step, 1);
        TensorPtr finishedStepsOutput
            = ITensor::slice(mFinishedSteps, std::min(step + 1, mNumDecodingEngineTokens[bi] - 1), 1);
        finishedStepsInput->squeeze(0);
        finishedStepsOutput->squeeze(0);

        if (srcCacheIndirection && tgtCacheIndirection)
        {
            auto srcView = std::shared_ptr(ITensor::slice(srcCacheIndirection, bi, singleRequest));
            auto tgtView = std::shared_ptr(ITensor::slice(tgtCacheIndirection, bi, singleRequest));
            dInput.cacheIndirection = ITensor::view(
                srcView, ITensor::makeShape({singleRequest, mBeamWidths[bi], srcView->getShape().d[2]}));
            dOutput.cacheIndirection = ITensor::view(
                tgtView, ITensor::makeShape({singleRequest, mBeamWidths[bi], tgtView->getShape().d[2]}));
        }

        auto sequenceLengthsView = std::shared_ptr(ITensor::slice(sequenceLengths, bi, singleRequest));
        dOutput.lengths = ITensor::view(sequenceLengthsView, ITensor::makeShape({singleRequest, mBeamWidths[bi]}));

        {
            dInput.logits = ITensor::slice(targetLogits, step, singleRequest);
            dOutput.newTokens = ITensor::view(dOutput.newTokensVec[step]);
            dInput.finished = ITensor::slice(finishedStepsInput, bi, singleRequest);
            dOutput.finished = ITensor::slice(finishedStepsOutput, bi, singleRequest);

            if (async)
            {
                decoder.forwardAsync(dOutput, dInput);
            }
            else
            {
                decoder.forwardSync(dOutput, dInput);
            }

            mNbSteps[bi] += 1;
            mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
            dInput.step += 1;
        }

        if (async)
        {
            if (step == mNumDecodingEngineTokens[bi] - 1)
            {
                auto const& stream = mStreams.at(bi);
                CudaEvent event{};
                stream->record(event);
                mStream->wait(event);
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardFusedDecoder(
    SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto eventStart = CudaEvent{};
    mStream->record(eventStart);

    auto& allTargetLogits = input.logits;
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto constexpr singleRequest = 1;

    TLLM_CHECK(static_cast<SizeType32>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    TLLM_CHECK(sequenceLengths);

    auto batchSlotsDecoderPtr
        = input.seqSlots ? bufferCast<SizeType32>(*input.seqSlots) : bufferCast<SizeType32>(*mBatchSlotsDecoder);
    auto batchSlotsAcceptTokensPtr = bufferCast<SizeType32>(*mBatchSlotsAcceptTokens);
    auto batchSlotsAcceptLogitsPtr = bufferCast<SizeType32>(*mBatchSlotsAcceptLogits);
    auto& dInput = *mJointDecodingInput;
    auto& dOutput = *mJointDecodingOutput;
    auto& decoder = *mDecoders[0];
    auto const& stream = mStreams.at(0);

    if (maxBeamWidth > 1)
    {
        dInput.cacheIndirection = input.cacheIndirection;
        dOutput.cacheIndirection = output.cacheIndirection;
    }

    if (mSpeculativeDecodingMode.isExplicitDraftTokens())
    {
        setExplicitDraftTokensInputs(input);
    }

    bool const async = forwardType == ForwardType::kASYNC;

    if (async)
    {
        stream->wait(eventStart.get());
    }

    SizeType32 localBatchDecoderIdx = 0;
    SizeType32 localBatchAcceptTokensIdx = 0;
    SizeType32 localBatchAcceptLogitsIdx = 0;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }

        if (mFusedDecoder)
        {
            if (!mAcceptByLogits[bi] && mMaxDecodingDecoderTokens == 1 && mNumDecodingEngineTokens[bi] > 1
                && step == mNumDecodingEngineTokens[bi] - 1)
            {
                batchSlotsAcceptTokensPtr[step * mActualBatchSize + localBatchAcceptTokensIdx] = bi;
                localBatchAcceptTokensIdx++;
            }
            else if (mAcceptByLogits[bi] && mMaxDecodingDecoderTokens == 1 && mNumDecodingEngineTokens[bi] > 1
                && step == 0)
            {
                batchSlotsAcceptLogitsPtr[step * mActualBatchSize + localBatchAcceptLogitsIdx] = bi;
                localBatchAcceptLogitsIdx++;
            }
            batchSlotsDecoderPtr[step * mActualBatchSize + localBatchDecoderIdx] = bi;
            localBatchDecoderIdx++;
        }
    }

    auto const maxDecodingEngineTokens
        = *std::max_element(std::begin(mNumDecodingEngineTokens), std::end(mNumDecodingEngineTokens));

    std::vector<SharedConstPtr> logitsVec;
    auto targetLogitsPtrsSlice = ITensor::slice(mTargetLogitsPtrs, step, 1);
    auto targetLogitsPtrsSlicePtr = reinterpret_cast<void const**>(bufferCast<int64_t>(*targetLogitsPtrsSlice));
    SizeType32 targetLogitsIdx = 0;
    for (SizeType32 bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi) || step >= mNumDecodingEngineTokens[bi])
        {
            continue;
        }
        auto& targetLogits = allTargetLogits[bi];
        SharedConstPtr logitsSlice = ITensor::slice(targetLogits, step, singleRequest);
        logitsVec.push_back(logitsSlice);
        targetLogitsPtrsSlicePtr[targetLogitsIdx++] = logitsSlice->data();
    }

    if (async && localBatchAcceptLogitsIdx > 0)
    {
        // These params are only used for testing. Thus, can be per batch instead of per request
        auto const& samplingConfig = decoder.getSamplingConfig();
        bool const useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
        float const randomAcceptanceThreshold
            = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

        TensorPtr batchSlotsAcceptLogitsStepSlice = ITensor::slice(mBatchSlotsAcceptLogits, step, 1);
        batchSlotsAcceptLogitsStepSlice->squeeze(0);
        TensorPtr batchSlotsAcceptLogitsSlice
            = ITensor::slice(batchSlotsAcceptLogitsStepSlice, 0, localBatchAcceptLogitsIdx);

        IGptDecoder::acceptDraftTokensByLogits(
            /* [maxBatchSize, maxDecodingTokens, vocabPadded] */ *mDraftLogits,
            /* [maxBatchSize][maxDecodingTokens, vocabPadded] */ *targetLogitsPtrsSlice,
            /* [maxBatchSize, maxDecodingTokens, vocabPadded] */ *mDraftProbs,
            /* [maxBatchSize, maxDecodingTokens, vocabPadded] */ *mTargetProbs,
            /* [maxBatchSize] */ *mNumDraftTokens,
            /* [maxDecodingTokens, maxBatchSize] */ *mFinishedSteps,
            /* [bs] */ *batchSlotsAcceptLogitsSlice, static_cast<SizeType32>(mVocabSize),
            static_cast<SizeType32>(mVocabSizePadded), useRandomAcceptanceThreshold, randomAcceptanceThreshold,
            reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), stream);
    }

    TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, step, 1);
    TensorPtr finishedStepsOutput = ITensor::slice(mFinishedSteps, std::min(maxDecodingEngineTokens - 1, step + 1), 1);
    finishedStepsInput->squeeze(0);
    finishedStepsOutput->squeeze(0);
    TensorPtr newTokensStepView = ITensor::slice(dOutput.newTokensSteps, step, mMaxDecodingDecoderTokens);

    dInput.logitsVec = logitsVec;
    dInput.finished = finishedStepsInput;

    if (input.seqSlots)
    {
        TensorPtr batchSlotsDecoderSlice = ITensor::slice(input.seqSlots, step, 1);
        dInput.batchSlots = batchSlotsDecoderSlice;
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

    dOutput.newTokens = newTokensStepView;
    dOutput.finished = finishedStepsOutput;
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
    }
    if (async && localBatchAcceptTokensIdx > 0)
    {
        TensorPtr batchSlotsAcceptTokensStepSlice = ITensor::slice(mBatchSlotsAcceptTokens, step, 1);
        batchSlotsAcceptTokensStepSlice->squeeze(0);
        auto batchSlotsAcceptTokensSlice
            = ITensor::slice(batchSlotsAcceptTokensStepSlice, 0, localBatchAcceptTokensIdx);

        // Update finished state for 0th step
        auto finishedFinal = ITensor::slice(mFinishedSteps, step, 1);
        IGptDecoder::acceptDraftTokensByIds(
            /* [maxBatchSize, maxBeamWidth, maxSeqLen] */ *dOutput.ids,
            /* [maxBatchSize, maxDecodingDraftTokens] */ *mDraftTokenIds,
            /* [maxBatchSize] */ *dInput.lengths,
            /* [maxBatchSize] */ *mNumDraftTokens,
            /* [maxBatchSize] */ *dOutput.lengths,
            /* [maxDecodingTokens, maxBatchSize] */ *mFinishedSteps,
            /* [maxBatchSize] */ *finishedFinal,
            /* [maxBatchSize] */ *dOutput.finishedSum,
            /* [bs] */ *batchSlotsAcceptTokensSlice, stream);
    }

    // If last iteration
    if (async && step == maxDecodingEngineTokens - mMaxDecodingDecoderTokens)
    {
        CudaEvent event{};
        stream->record(event);
        mStream->wait(event);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::updateFinished(decoder_batch::Token const& token)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (token.active[i] && !mFinished[i])
        {
            auto& dOutput = *mDecodingOutputs[i];
            mFinished[i] = mFinished[i]
                // This condition requires the synchronization above
                || bufferCast<SizeType32>(*dOutput.finishedSum)[0] == static_cast<SizeType32>(mBeamWidths[i]);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardSync(decoder_batch::Token const& token)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    token.event.synchronize();

    updateFinished(token);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardSync(
    decoder_batch::Token const& token, decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    token.event.synchronize();

    forwardDispatch(output, input, ForwardType::kSYNC);

    updateFinished(token);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// TODO call this at the end of forward if mFinished[i] changes from false to true?
CudaEvent GptDecoderBatch::postProcessRequest(
    SizeType32 batchSlot, std::optional<std::reference_wrapper<SamplingConfig const>> samplingConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& stream = mFusedDecoder ? mStream : mStreams[batchSlot];
    auto manager = BufferManager{stream};
    auto& decoder = mFusedDecoder ? *mDecoders[0] : *mDecoders[batchSlot];

    auto& dInput = *mDecodingInputs[batchSlot];
    auto& dOutput = *mDecodingOutputs[batchSlot];

    if (mFusedDecoder)
    {
        auto& dJointOutput = *mJointDecodingOutput;

        auto slice = [&batchSlot](auto& a, auto& b)
        {
            if (b && b->getShape().d[0] > 0)
            {
                a = ITensor::slice(b, batchSlot, 1);
            }
        };

        slice(dOutput.cacheIndirection, dJointOutput.cacheIndirection);
        slice(dOutput.lengths, dJointOutput.lengths);
        slice(dOutput.finished, dJointOutput.finished);
        slice(dOutput.logProbs, dJointOutput.logProbs);

        dOutput.newTokens = ITensor::view(dJointOutput.newTokens);
        TLLM_CHECK(dOutput.newTokens->getShape().d[0] == 1);
        dOutput.newTokens->squeeze(0);
        dOutput.newTokens = ITensor::slice(dOutput.newTokens, batchSlot, 1);
    }

    // TODO can we do this inplace?
    auto& outputIds = dOutput.ids;
    auto finalOutputIds = manager.gpu(outputIds->getShape(), outputIds->getDataType());
    decoder.gatherTree(*finalOutputIds, dOutput, dInput, manager, samplingConfig);
    manager.copy(*finalOutputIds, *outputIds);

    CudaEvent event{};
    stream->record(event);
    mStream->wait(event);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatch::newBatch(
    GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
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
    if (inputs.packed && inputIdsShape.nbDims == 2)
    { // For users still pass inputs.ids with shape [1, num_tokens], do squeeze for them.
        inputIdsFlatView->squeeze(0);
    }
    auto inputLengthsHost = mBufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType32>(*inputLengthsHost);
    auto inputOffset = 0;
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
        newRequest(batchIdx, request, requestSamplingConfig);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const& logitsShape = input.logits->getShape();
    auto const batchSize = logitsShape.d[0];
    auto constexpr singleRequest = 1;
    std::vector<ITensor::SharedConstPtr> logits;
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

    mForwardToken = forwardAsync(batchOutput, batchInput);
    mBufferManager.setZero(*mFinishedSum);
    kernels::reduce(*mFinishedSum, *ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize), *mStream);
    mStream->record(mForwardEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mForwardToken);
    // wait for mFinishedSum to be updated
    mForwardEvent.synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::finalize() const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto batchSlots = bufferCast<SizeType32>(*mBatchSlotsSetup);
    for (SizeType32 batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        auto event = postProcessRequest(batchSlots ? batchSlots[batchIdx] : batchIdx);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatch::finalize(SizeType32 batchSlot, SamplingConfig const& samplingConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchSlot, samplingConfig);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
