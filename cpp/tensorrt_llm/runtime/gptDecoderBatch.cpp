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
SamplingConfig extractSamplingConfig(SamplingConfig const& batchSamplingConfig, SizeType batchIdx)
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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return samplingConfig;
}

} // namespace

GptDecoderBatch::GptDecoderBatch(
    std::size_t vocabSize, std::size_t vocabSizePadded, GptDecoderBatch::CudaStreamPtr stream)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mStream{std::move(stream)}
    , mBufferManager{mStream}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType>::value;
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
    mBatchSlotsSetup = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    mBatchSlotsDecoder = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    mBatchSlotsAcceptTokens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    mBatchSlotsAcceptLogits = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    mFinishedSum = mBufferManager.pinned(ITensor::makeShape({1}), nvSizeType);
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
    dInput->stopWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    dInput->badWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNED, TRTDataType<SizeType>::value);
    dInput->embeddingBias = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setup(DecodingMode const& mode, SizeType maxBatchSize, SizeType maxBeamWidth,
    SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSequenceLength, SizeType maxTokensPerStep,
    bool fusedDecoder, nvinfer1::DataType dtype)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(maxTokensPerStep > 0);
    TLLM_CHECK(maxSequenceLength > 0);
    mActualBatchSize = maxBatchSize;
    mGeneratedTokensPerStep.resize(maxBatchSize);
    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mSinkTokenLength = sinkTokenLength;
    mMaxTokensPerStep = maxTokensPerStep;
    mFusedDecoder = fusedDecoder;

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});
    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({maxTokensPerStep, maxBatchSize, maxBeamWidth});
    auto const maxBatchSizeXmaxTokensPerStep = ITensor::makeShape({maxBatchSize, maxTokensPerStep});

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

    dOutput.newTokensSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.newTokensSteps);
    mFinishedSteps->reshape(maxTokensPerStepXmaxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    if (mFusedDecoder)
    {
        mBatchSlotsSetup->reshape(ITensor::makeShape({maxBatchSize}));
        mBatchSlotsDecoder->reshape(ITensor::makeShape({maxTokensPerStep, maxBatchSize}));
        mBatchSlotsAcceptTokens->reshape(ITensor::makeShape({maxTokensPerStep, maxBatchSize}));
        mBatchSlotsAcceptLogits->reshape(ITensor::makeShape({maxTokensPerStep, maxBatchSize}));
    }

    if (mMaxTokensPerStep > 1)
    {
        mDraftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerStep, maxBeamWidth, static_cast<SizeType>(mVocabSizePadded)}));
        mTargetProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerStep, maxBeamWidth, static_cast<SizeType>(mVocabSizePadded)}));
    }

    dOutput.parentIds->reshape(jointOutputIdsShape);
    // use batchSize many entries instead of the usual 1
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

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
        ITensor::makeShape({maxBatchSize, maxTokensPerStep, static_cast<SizeType>(mVocabSizePadded)}));
    mAcceptByLogits.resize(maxBatchSize);
    mNumDraftTokens->reshape(ITensor::makeShape({maxBatchSize, 1}));
    mCurandStates->reshape(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}));
    mTargetLogitsPtrs->reshape(ITensor::makeShape({maxTokensPerStep, maxBatchSize}));

    const_cast<ITensor&>(*dInput.embeddingBias)
        .reshape(ITensor::makeShape({maxBatchSize, static_cast<SizeType>(mVocabSizePadded)}));
    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(ITensor::makeShape({maxBatchSize}));
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(ITensor::makeShape({maxBatchSize}));

    auto const numOfDecoders = fusedDecoder ? 1 : maxBatchSize;
    mStreams.resize(maxBatchSize);
    mDecoders.resize(numOfDecoders);
    mDecodingInputs.resize(maxBatchSize);
    mDecodingOutputs.resize(maxBatchSize);
    mNbSteps.resize(maxBatchSize);
    mFinished.resize(maxBatchSize);
    mMaxNewTokens.resize(maxBatchSize);
    mBeamWidths.resize(maxBatchSize);
    auto const device = mStream->getDevice();
    for (SizeType i = 0; i < maxBatchSize; ++i)
    {
        mStreams[i] = std::make_shared<CudaStream>();
        TLLM_CHECK(mStreams[i]->getDevice() == device);
        if (i < numOfDecoders)
        {
            auto maxBatchSizePerDecoder = fusedDecoder ? maxBatchSize : 1;
            mDecoders[i] = IGptDecoder::create(mode, dtype, maxBatchSizePerDecoder, maxBeamWidth, mVocabSize,
                mVocabSizePadded, mMaxSequenceLength, mStreams[i]);
        }
        mDecodingInputs[i].reset();
        mDecodingOutputs[i].reset();
        mNbSteps[i] = 0;
        mFinished[i] = true;
        mMaxNewTokens[i] = 0;
        mBeamWidths[i] = 0;
        mGeneratedTokensPerStep[i] = 0;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequest(
    SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchIdx >= 0);
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(0 <= batchSize && batchIdx < batchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const beamWidth = samplingConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
        tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (%d) passed to decoder setup function.",
            beamWidth, maxBeamWidth));
    auto const& requestIds = request.ids;
    auto const inputLength = request.inputLen;
    auto const maxNewTokens = request.maxNewTokens.value_or(mMaxSequenceLength - inputLength);
    TLLM_CHECK_WITH_INFO(inputLength + maxNewTokens <= mMaxSequenceLength,
        tc::fmtstr("Input length (%d) + max new tokens (%d) must be less than max sequence length (%d).", inputLength,
            maxNewTokens, mMaxSequenceLength));
    TLLM_CHECK(requestIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = request.endId.value_or(mVocabSize - 1);

    auto constexpr localBatchSize = 1;

    auto const decoderIdx = mFusedDecoder ? 0 : batchIdx;
    auto& stream = mStreams[decoderIdx];
    BufferManager manager{stream};

    // input
    auto& dJointInput = *mJointDecodingInput;
    auto& dInput = mDecodingInputs.at(batchIdx);

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchIdx, localBatchSize)};
    kernels::invokeFill(*endIdTensorPtr, endId, *stream);
    dInput = std::make_unique<DecodingInput>(
        inputLength, mMaxAttentionWindow, mSinkTokenLength, localBatchSize, dJointInput.logits, endIdTensorPtr);

    TensorPtr embeddingBiasSlice
        = ITensor::slice(constPointerCast(dJointInput.embeddingBias), batchIdx, localBatchSize);
    if (request.embeddingBias)
    {
        TLLM_CHECK(request.embeddingBias->getShape().nbDims == 2);
        TLLM_CHECK(request.embeddingBias->getShape().d[0] == 1);
        TLLM_CHECK_WITH_INFO(request.embeddingBias->getShape().d[1] == static_cast<SizeType>(mVocabSize),
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
                          SharedConstPtr& wordsLens, SizeType& inputMaxStopWordsLen, SizeType& maxWordsLen,
                          SizeType localBatchSize, SizeType batchIdx)
    {
        if (requestWordsList)
        {
            auto const wordsLen = requestWordsList->getShape().d[1];
            BufferRange<int32_t*>(*constPointerCast(jointWordsPtrs))[batchIdx]
                = bufferCast<SizeType>(*requestWordsList);
            bufferCast<SizeType>(*constPointerCast(jointWordsLens))[batchIdx] = wordsLen;
            // FIXME(nkorobov): this is monotonically growing size
            maxWordsLen = std::max(wordsLen, maxWordsLen);
            if (!fusedDecoder)
            {
                wordsPtrs = ITensor::slice(jointWordsPtrs, batchIdx, localBatchSize);
                wordsLens = ITensor::slice(jointWordsLens, batchIdx, localBatchSize);
                inputMaxStopWordsLen = wordsLen;
            }
            // NOTE(nkorobov): dInput-><name>WordsList is not used in gptDecoder, but required to keep <name>WordsList's
            // memory allocated
            inputWordsList = requestWordsList;
        }
        else
        {
            bufferCast<SizeType>(*constPointerCast(jointWordsLens))[batchIdx] = 0;
            inputMaxStopWordsLen = 0;
        }
    };

    setupWords(dInput->stopWordsList, request.stopWordsList, dJointInput.stopWordsPtrs, dJointInput.stopWordsLens,
        dInput->stopWordsPtrs, dInput->stopWordsLens, dInput->maxStopWordsLen, mMaxStopWordsLen, localBatchSize,
        batchIdx);
    dJointInput.maxStopWordsLen = mMaxStopWordsLen;

    setupWords(dInput->badWordsList, request.badWordsList, dJointInput.badWordsPtrs, dJointInput.badWordsLens,
        dInput->badWordsPtrs, dInput->badWordsLens, dInput->maxBadWordsLen, mMaxBadWordsLen, localBatchSize, batchIdx);
    dJointInput.maxBadWordsLen = mMaxBadWordsLen;

    TensorPtr sequenceLimitLength{
        ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchIdx, localBatchSize)};
    kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, *stream);
    dInput->sequenceLimitLength = std::move(sequenceLimitLength);
    TensorPtr inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchIdx, localBatchSize)};
    kernels::invokeFill(*inputLengths, inputLength, *stream);
    dInput->lengths = inputLengths;

    // output
    auto& dJointOutput = *mJointDecodingOutput;
    auto& dOutput = mDecodingOutputs.at(batchIdx);
    auto const outputIdsShape = ITensor::makeShape({localBatchSize, beamWidth, mMaxSequenceLength});

    TensorPtr outputIds = ITensor::slice(dJointOutput.ids, batchIdx, localBatchSize);
    outputIds->reshape(outputIdsShape);
    dOutput = std::make_unique<DecodingOutput>(outputIds);

    dOutput->finishedSum = ITensor::slice(dJointOutput.finishedSum, batchIdx, localBatchSize);
    manager.setZero(*dOutput->finishedSum);

    dOutput->newTokensVec.resize(mMaxTokensPerStep);
    for (SizeType ti = 0; ti < mMaxTokensPerStep; ++ti)
    {
        TensorPtr newTokensStepView = ITensor::slice(dJointOutput.newTokensSteps, ti, localBatchSize);
        newTokensStepView->squeeze(0);
        dOutput->newTokensVec[ti] = ITensor::slice(newTokensStepView, batchIdx, localBatchSize);
        manager.setZero(*dOutput->newTokensVec[ti]);
    }

    // FIXME(nkorobov): we call setZero mMaxTokensPerStep times for only 1 element
    for (SizeType ti = 0; ti < mMaxTokensPerStep; ++ti)
    {
        TensorPtr finishedStepsView = std::move(ITensor::slice(mFinishedSteps, ti, 1));
        finishedStepsView->squeeze(0);
        TensorPtr finishedSteps = std::move(ITensor::slice(finishedStepsView, batchIdx, localBatchSize));
        manager.setZero(*finishedSteps);
    }

    // cumLogProb is mandatory for beamWidth > 1
    dOutput->cumLogProbs = nullptr;
    if (request.computeCumLogProbs || beamWidth > 1)
    {
        dOutput->cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchIdx, localBatchSize);
        manager.setZero(*dOutput->cumLogProbs);
    }

    dOutput->logProbs = nullptr;
    if (request.computeLogProbs)
    {
        dOutput->logProbs = ITensor::slice(dJointOutput.logProbs, batchIdx, localBatchSize);
        manager.setZero(*dOutput->logProbs);
    }

    if (beamWidth > 1)
    {
        kernels::invokeFill(
            *IBuffer::slice(dOutput->cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *stream);
        dOutput->parentIds = ITensor::slice(dJointOutput.parentIds, batchIdx, localBatchSize);
        dOutput->parentIds->reshape(outputIdsShape);
        manager.setZero(*dOutput->parentIds);
        dOutput->beamHypotheses = dJointOutput.beamHypotheses.slice(batchIdx, localBatchSize);
        dOutput->beamHypotheses.init(manager, endId);
    }

    auto generatedTokensPerStep = request.generatedTokensPerStep();
    if (generatedTokensPerStep > 1)
    {
        TLLM_CHECK(beamWidth == 1);
        auto numDraftTokens = generatedTokensPerStep - 1;
        TensorPtr draftTokensReqBatchSlice = std::move(ITensor::slice(mDraftTokenIds, batchIdx, 1));
        draftTokensReqBatchSlice->squeeze(0);
        TensorPtr draftTokensReqTokensSlice = ITensor::slice(draftTokensReqBatchSlice, 0, numDraftTokens);
        TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({numDraftTokens}));
        manager.copy(*draftTokensView, *draftTokensReqTokensSlice);
        mAcceptByLogits[batchIdx] = false;
        if (request.draftLogits.has_value())
        {
            TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());
            mAcceptByLogits[batchIdx] = true;

            TensorPtr draftLogitsReqBatchSlice = std::move(ITensor::slice(mDraftLogits, batchIdx, 1));
            draftLogitsReqBatchSlice->squeeze(0);
            TensorPtr draftLogitsReqTokensSlice = ITensor::slice(draftLogitsReqBatchSlice, 0, numDraftTokens);
            manager.copy(*draftLogitsView, *draftLogitsReqTokensSlice);
        }

        auto numDraftTokensView = ITensor::slice(mNumDraftTokens, batchIdx, localBatchSize);
        kernels::invokeFill(*numDraftTokensView, numDraftTokens, *stream);

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
    }

    // remaining
    if (!mFusedDecoder)
    {
        mDecoders[decoderIdx]->setup(samplingConfig, localBatchSize, mMaxSequenceLength);
    }
    mBeamWidths[batchIdx] = beamWidth;
    mNbSteps[batchIdx] = 0;
    mFinished[batchIdx] = false;
    mMaxNewTokens[batchIdx] = maxNewTokens;
    mGeneratedTokensPerStep[batchIdx] = generatedTokensPerStep;

    // copy the request ids into outputIds
    auto const requestIdsShape = requestIds->getShape();
    auto inputIdsView = ITensor::view(requestIds, ITensor::makeShape({localBatchSize, requestIdsShape.d[0]}));
    auto outputIdsView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, endId, *stream);
    kernels::tileTensor(*outputIdsView, *inputIdsView, beamWidth, *stream);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequests(std::vector<SizeType> const& seqSlots,
    std::vector<decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlotsPtr = bufferCast<SizeType>(*mBatchSlotsSetup);
    SizeType const localBatchSize = seqSlots.size();
    for (SizeType bi = 0; bi < localBatchSize; ++bi)
    {
        newRequest(seqSlots[bi], requests[bi], samplingConfigs[bi]);
        if (mFusedDecoder)
        {
            batchSlotsPtr[bi] = seqSlots[bi];
        }
    }
    if (mFusedDecoder)
    {
        TensorPtr batchSlotsView = std::move(ITensor::slice(mBatchSlotsSetup, 0, localBatchSize));
        auto fusedSamplingConfig = SamplingConfig(samplingConfigs);
        mDecoders[0]->setup(fusedSamplingConfig, localBatchSize, mMaxSequenceLength, {batchSlotsView});
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

GptDecoderBatch::TokenPtr GptDecoderBatch::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& allTargetLogits = input.logits;

    // TODO(nkorobov): check logits shape considering draft tokens
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    TLLM_CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    TLLM_CHECK(static_cast<SizeType>(output.sequenceLengths->getSize()) == mActualBatchSize * maxBeamWidth);
    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths
        = ITensor::view(output.sequenceLengths, ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    auto batchSlotsDecoderPtr = bufferCast<SizeType>(*mBatchSlotsDecoder);
    auto batchSlotsAcceptTokensPtr = bufferCast<SizeType>(*mBatchSlotsAcceptTokens);
    auto batchSlotsAcceptLogitsPtr = bufferCast<SizeType>(*mBatchSlotsAcceptLogits);
    TLLM_CHECK(sequenceLengths);
    auto constexpr singleRequest = 1;

    CudaEvent eventStart{};
    mStream->record(eventStart);

    auto const maxGeneratedTokensPerStep
        = *std::max_element(std::begin(mGeneratedTokensPerStep), std::end(mGeneratedTokensPerStep));

    for (SizeType si = 0; si < maxGeneratedTokensPerStep; ++si)
    {
        SizeType localBatchDecoderIdx = 0;
        SizeType localBatchAcceptTokensIdx = 0;
        SizeType localBatchAcceptLogitsIdx = 0;
        for (SizeType bi = 0; bi < mActualBatchSize; ++bi)
        {
            if (mFinished[bi] || !input.active.at(bi) || si >= mGeneratedTokensPerStep[bi])
            {
                continue;
            }

            if (mFusedDecoder)
            {
                if (!mAcceptByLogits[bi] && mGeneratedTokensPerStep[bi] > 1 && si == mGeneratedTokensPerStep[bi] - 1)
                {
                    batchSlotsAcceptTokensPtr[si * mActualBatchSize + localBatchAcceptTokensIdx] = bi;
                    localBatchAcceptTokensIdx++;
                }
                else if (mAcceptByLogits[bi] && mGeneratedTokensPerStep[bi] > 1 && si == 0)
                {
                    batchSlotsAcceptLogitsPtr[si * mActualBatchSize + localBatchAcceptLogitsIdx] = bi;
                    localBatchAcceptLogitsIdx++;
                }
                batchSlotsDecoderPtr[si * mActualBatchSize + localBatchDecoderIdx] = bi;
                localBatchDecoderIdx++;
            }
        }

        if (!mFusedDecoder)
        {
            for (SizeType bi = 0; bi < mActualBatchSize; ++bi)
            {
                if (mFinished[bi] || !input.active.at(bi) || si >= mGeneratedTokensPerStep[bi])
                {
                    continue;
                }

                auto& stream = mStreams[bi];
                stream->wait(eventStart.get());

                auto& targetLogits = allTargetLogits[bi];
                auto& dInput = *mDecodingInputs[bi];
                auto& dOutput = *mDecodingOutputs[bi];
                auto& decoder = *mDecoders[bi];

                TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, si, 1);
                TensorPtr finishedStepsOutput
                    = ITensor::slice(mFinishedSteps, std::min(si + 1, mGeneratedTokensPerStep[bi] - 1), 1);
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
                dOutput.lengths
                    = ITensor::view(sequenceLengthsView, ITensor::makeShape({singleRequest, mBeamWidths[bi]}));

                {
                    dInput.logits = ITensor::slice(targetLogits, si, singleRequest);
                    dOutput.newTokens = ITensor::view(dOutput.newTokensVec[si]);
                    dInput.finished = ITensor::slice(finishedStepsInput, bi, singleRequest);
                    dOutput.finished = ITensor::slice(finishedStepsOutput, bi, singleRequest);

                    decoder.forwardAsync(dOutput, dInput);

                    mNbSteps[bi] += 1;
                    mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
                    dInput.step += 1;
                }

                if (si == mGeneratedTokensPerStep[bi] - 1)
                {
                    auto& stream = mStreams[bi];
                    CudaEvent event{};
                    stream->record(event);
                    mStream->wait(event);
                }
            }
        }
        else
        {
            TLLM_CHECK_WITH_INFO(mBeamWidths[0] == 1, "Fused decoder is not supported for beam search yet.");

            auto& dInput = *mJointDecodingInput;
            auto& dOutput = *mJointDecodingOutput;
            auto& decoder = *mDecoders[0];
            auto& stream = mStreams[0];

            stream->wait(eventStart.get());

            BufferManager manager{stream};

            std::vector<SharedConstPtr> logitsVec;
            auto targetLogitsPtrsSlice = ITensor::slice(mTargetLogitsPtrs, si, 1);
            auto targetLogitsPtrsSlicePtr = reinterpret_cast<void const**>(bufferCast<int64_t>(*targetLogitsPtrsSlice));
            SizeType targetLogitsIdx = 0;
            for (SizeType bi = 0; bi < mActualBatchSize; ++bi)
            {
                if (mFinished[bi] || !input.active.at(bi) || si >= mGeneratedTokensPerStep[bi])
                {
                    continue;
                }
                auto& targetLogits = allTargetLogits[bi];
                SharedConstPtr logitsSlice = std::move(ITensor::slice(targetLogits, si, singleRequest));
                logitsVec.push_back(logitsSlice);
                targetLogitsPtrsSlicePtr[targetLogitsIdx++] = logitsSlice->data();
            }

            if (localBatchAcceptLogitsIdx > 0)
            {
                // These params are only used for testing. Thus, can be per batch instead of per request
                auto const& samplingConfig = decoder.getSamplingConfig();
                const bool useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
                const float randomAcceptanceThreshold
                    = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

                TensorPtr batchSlotsAcceptLogitsStepSlice = std::move(ITensor::slice(mBatchSlotsAcceptLogits, si, 1));
                batchSlotsAcceptLogitsStepSlice->squeeze(0);
                TensorPtr batchSlotsAcceptLogitsSlice
                    = std::move(ITensor::slice(batchSlotsAcceptLogitsStepSlice, 0, localBatchAcceptLogitsIdx));

                IGptDecoder::acceptDraftTokensByLogits(
                    /* [max_bs, max_tokens_per_step, vocabPadded] */ *mDraftLogits,
                    /* [max_bs][max_tokens_per_step, vocabPadded] */ *targetLogitsPtrsSlice,
                    /* [max_bs, max_tokens_per_step, vocabPadded] */ *mDraftProbs,
                    /* [max_bs, max_tokens_per_step, vocabPadded] */ *mTargetProbs,
                    /* [max_bs] */ *mNumDraftTokens,
                    /* [max_tokens_per_step, max_bs] */ *mFinishedSteps,
                    /* [bs] */ *batchSlotsAcceptLogitsSlice, static_cast<SizeType>(mVocabSize),
                    static_cast<SizeType>(mVocabSizePadded), useRandomAcceptanceThreshold, randomAcceptanceThreshold,
                    reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), stream);
            }

            TensorPtr finishedStepsInput = ITensor::slice(mFinishedSteps, si, 1);
            TensorPtr finishedStepsOutput
                = ITensor::slice(mFinishedSteps, std::min(maxGeneratedTokensPerStep - 1, si + 1), 1);
            finishedStepsInput->squeeze(0);
            finishedStepsOutput->squeeze(0);
            TensorPtr newTokensStepView = std::move(ITensor::slice(dOutput.newTokensSteps, si, 1));
            newTokensStepView->squeeze(0);

            dInput.logitsVec = logitsVec;
            dInput.finished = finishedStepsInput;
            TensorPtr batchSlotsDecoderSlice = std::move(ITensor::slice(mBatchSlotsDecoder, si, 1));
            batchSlotsDecoderSlice->squeeze(0);
            dInput.batchSlots = batchSlotsDecoderSlice;
            dInput.maxBatchSize = localBatchDecoderIdx;

            dOutput.newTokens = newTokensStepView;
            dOutput.finished = finishedStepsOutput;
            dOutput.lengths = sequenceLengths;

            if (localBatchDecoderIdx > 0)
            {
                decoder.forwardAsync(dOutput, dInput);
            }

            for (SizeType bi = 0; bi < mActualBatchSize; ++bi)
            {
                if (mFinished[bi] || !input.active.at(bi) || si >= mGeneratedTokensPerStep[bi])
                {
                    continue;
                }
                mNbSteps[bi] += 1;
                mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
            }
            if (localBatchAcceptTokensIdx > 0)
            {
                TensorPtr batchSlotsAcceptTokensStepSlice = std::move(ITensor::slice(mBatchSlotsAcceptTokens, si, 1));
                batchSlotsAcceptTokensStepSlice->squeeze(0);
                auto batchSlotsAcceptTokensSlice
                    = ITensor::slice(batchSlotsAcceptTokensStepSlice, 0, localBatchAcceptTokensIdx);

                // Update finished state for 0th step
                auto finishedFinal = ITensor::slice(mFinishedSteps, si, 1);
                IGptDecoder::acceptDraftTokensByIds(
                    /* [max_bs, max_seq_len] */ *dOutput.ids,
                    /* [max_bs, max_draft_tokens] */ *mDraftTokenIds,
                    /* [max_bs] */ *dInput.lengths,
                    /* [max_bs] */ *mNumDraftTokens,
                    /* [max_bs] */ *dOutput.lengths,
                    /* [max_tokens_per_step, max_bs] */ *mFinishedSteps,
                    /* [max_bs] */ *finishedFinal,
                    /* [max_bs] */ *dOutput.finishedSum,
                    /* [bs] */ *batchSlotsAcceptTokensSlice, stream);
            }

            if (si == maxGeneratedTokensPerStep - 1)
            {
                CudaEvent event{};
                stream->record(event);
                mStream->wait(event);
            }
        }
    }

    CudaEvent eventStop{};
    mStream->record(eventStop);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::Token>(std::move(eventStop), input.active);
}

void GptDecoderBatch::forwardSync(decoder_batch::Token const& token)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    token.event.synchronize();

    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (token.active[i] && !mFinished[i])
        {
            auto& dOutput = *mDecodingOutputs[i];
            mFinished[i] = mFinished[i]
                // This condition requires the synchronization above
                || bufferCast<SizeType>(*dOutput.finishedSum)[0] == static_cast<SizeType>(mBeamWidths[i]);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// TODO call this at the end of forward if mFinished[i] changes from false to true?
CudaEvent GptDecoderBatch::postProcessRequest(SizeType batchIdx) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& stream = mStreams[batchIdx];
    auto manager = BufferManager{stream};
    auto& decoder = *mDecoders[batchIdx];

    auto& dInput = *mDecodingInputs[batchIdx];
    auto& dOutput = *mDecodingOutputs[batchIdx];

    // TODO can we do this inplace?
    auto& outputIds = dOutput.ids;
    auto finalOutputIds = manager.gpu(outputIds->getShape(), outputIds->getDataType());
    decoder.gatherTree(*finalOutputIds, dOutput, dInput, manager);
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
    mGeneratedTokensPerStep.resize(mActualBatchSize);

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
    auto inputLengthsPtr = bufferCast<SizeType>(*inputLengthsHost);
    auto inputOffset = 0;
    for (auto batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        mGeneratedTokensPerStep[batchIdx] = 1;
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
        request.computeCumLogProbs = (outputs.cumLogProbs != nullptr);
        request.computeLogProbs = (outputs.logProbs != nullptr);

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
        newRequest(batchIdx, request, extractSamplingConfig(samplingConfig, batchIdx));
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
    for (SizeType batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        auto event = postProcessRequest(batchIdx);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatch::finalize(SizeType batchIdx) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchIdx);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return event;
}
