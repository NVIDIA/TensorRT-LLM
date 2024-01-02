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
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

namespace
{
SamplingConfig extractSamplingConfig(SamplingConfig const& batchSamplingConfig, SizeType batchIdx)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
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
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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
    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    mFinishedSum = mBufferManager.pinned(ITensor::makeShape({1}), nvSizeType);
    // we don't need dOutput->lengths because lengths are passed from outside
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->logProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);

    mNumDraftTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    mCurandStates = mBufferManager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT8);

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow,
    SizeType sinkTokenLength, SizeType maxSequenceLength, SizeType maxTokensPerStep, nvinfer1::DataType dtype)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});
    auto const maxBatchSizeXmaxTokensPerStepXmaxBeamWidth
        = ITensor::makeShape({maxBatchSize, maxTokensPerStep, maxBeamWidth});
    auto const maxTokensPerStepXmaxBatchSizeXmaxBeamWidth
        = ITensor::makeShape({maxTokensPerStep, maxBatchSize, maxBeamWidth});

    auto& dInput = *mJointDecodingInput;
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
    mFinishedSteps->reshape(maxBatchSizeXmaxTokensPerStepXmaxBeamWidth);
    mBufferManager.setZero(*mFinishedSteps);

    if (mMaxTokensPerStep > 1)
    {
        mDraftProbs->reshape(ITensor::makeShape(
            {maxBatchSize, maxTokensPerStep - 1, maxBeamWidth, static_cast<SizeType>(mVocabSizePadded)}));
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
    mDraftTokenIds.resize(maxBatchSize);
    mDraftLogits.resize(maxBatchSize);
    mAcceptByLogits.resize(maxBatchSize);
    mNumDraftTokens->reshape(ITensor::makeShape({maxBatchSize, 1}));
    mCurandStates->reshape(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}));

    mStreams.resize(maxBatchSize);
    mDecoders.resize(maxBatchSize);
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
        mDecoders[i] = IGptDecoder::create(dtype, mVocabSize, mVocabSizePadded, mStreams[i]);
        mDecodingInputs[i].reset();
        mDecodingOutputs[i].reset();
        mNbSteps[i] = 0;
        mFinished[i] = true;
        mMaxNewTokens[i] = 0;
        mBeamWidths[i] = 0;
        mGeneratedTokensPerStep[i] = 0;
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::newRequest(
    SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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

    auto& stream = mStreams[batchIdx];
    BufferManager manager{stream};

    // input
    auto& dJointInput = *mJointDecodingInput;
    auto& dInput = mDecodingInputs.at(batchIdx);

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchIdx, localBatchSize)};
    kernels::invokeFill(*endIdTensorPtr, endId, *stream);
    dInput = std::make_unique<DecodingInput>(
        inputLength, mMaxAttentionWindow, mSinkTokenLength, localBatchSize, dJointInput.logits, endIdTensorPtr);

    // Here, we need to add leading 1 dimension since decoderInput expects batchSize as leading dim
    // and decoder_batch::Request doesn't have batch dimension
    if (request.embeddingBias)
    {
        TensorPtr biasView = ITensor::view(request.embeddingBias);
        biasView->unsqueeze(0);
        dInput->embeddingBias = biasView;
    }
    if (request.badWordsList)
    {
        TensorPtr badWordsView = ITensor::view(request.badWordsList);
        badWordsView->unsqueeze(0);
        dInput->badWordsList = badWordsView;
    }
    if (request.stopWordsList)
    {
        TensorPtr stopWordsView = ITensor::view(request.stopWordsList);
        stopWordsView->unsqueeze(0);
        dInput->stopWordsList = stopWordsView;
    }

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
        TensorPtr newTokensStepView = std::move(ITensor::slice(dJointOutput.newTokensSteps, ti, localBatchSize));
        newTokensStepView->squeeze(0);
        dOutput->newTokensVec[ti] = ITensor::slice(newTokensStepView, batchIdx, localBatchSize);
        manager.setZero(*dOutput->newTokensVec[ti]);
    }

    TensorPtr finishedSteps = ITensor::slice(mFinishedSteps, batchIdx, localBatchSize);
    manager.setZero(*finishedSteps);

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
        TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({1, 1, numDraftTokens}));
        mDraftTokenIds[batchIdx] = draftTokensView;
        mAcceptByLogits[batchIdx] = false;
        if (request.draftLogits.has_value())
        {
            TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());
            mDraftLogits[batchIdx] = draftLogitsView;
            mAcceptByLogits[batchIdx] = true;
        }

        auto numDraftTokensView = ITensor::slice(mNumDraftTokens, batchIdx, localBatchSize);
        kernels::invokeFill(*numDraftTokensView, numDraftTokens, *stream);

        auto const curandStatesView = ITensor::slice(mCurandStates, batchIdx, localBatchSize);
        auto curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*curandStatesView));
        if (samplingConfig.randomSeed.has_value())
        {
            tk::invokeCurandInitialize(
                curandState, localBatchSize, samplingConfig.randomSeed.value()[0], stream->get());
        }
        else
        {
            tk::invokeCurandInitialize(curandState, localBatchSize, 0, stream->get());
        }
    }

    // remaining
    mDecoders[batchIdx]->setup(samplingConfig, localBatchSize, mMaxSequenceLength);
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
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

GptDecoderBatch::TokenPtr GptDecoderBatch::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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
    TLLM_CHECK(sequenceLengths);
    auto constexpr singleRequest = 1;

    CudaEvent eventStart{};
    mStream->record(eventStart);
    for (std::int32_t bi = 0; bi < mActualBatchSize; ++bi)
    {
        if (mFinished[bi] || !input.active.at(bi))
        {
            continue;
        }

        auto& targetLogits = allTargetLogits[bi];
        auto const& logitsShape = targetLogits->getShape();
        TLLM_CHECK_WITH_INFO(logitsShape.d[0] == mGeneratedTokensPerStep[bi],
            tc::fmtstr(
                "First dim (%d) does not match generated tokens (%d)", logitsShape.d[0], mGeneratedTokensPerStep[bi]));
        TLLM_CHECK_WITH_INFO(logitsShape.d[1] == mBeamWidths[bi],
            tc::fmtstr("Second dim (%d) does not match beam width (%d)", logitsShape.d[1], mBeamWidths[bi]));
        TLLM_CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

        auto& stream = mStreams[bi];
        stream->wait(eventStart.get());
        auto& dInput = *mDecodingInputs[bi];
        auto& dOutput = *mDecodingOutputs[bi];
        auto& decoder = *mDecoders[bi];

        TensorPtr finishedSteps = ITensor::slice(mFinishedSteps, bi, singleRequest);
        finishedSteps->squeeze(0);

        if (mGeneratedTokensPerStep[bi] > 1 && mAcceptByLogits[bi])
        {
            auto numDraftTokens = ITensor::slice(mNumDraftTokens, bi, singleRequest);
            auto const curandStatesView = ITensor::slice(mCurandStates, bi, singleRequest);
            auto curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*curandStatesView));
            auto const& samplingConfig = decoder.getSamplingConfig();
            const bool useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
            const float randomAcceptanceThreshold
                = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

            TensorPtr draftProbs = ITensor::slice(mDraftProbs, bi, singleRequest);
            TensorPtr targetProbs = ITensor::slice(mTargetProbs, bi, singleRequest);
            draftProbs = ITensor::view(draftProbs,
                ITensor::makeShape(
                    {mMaxTokensPerStep - 1, singleRequest, mBeamWidths[bi], static_cast<SizeType>(mVocabSizePadded)}));
            targetProbs = ITensor::view(targetProbs,
                ITensor::makeShape(
                    {mMaxTokensPerStep, singleRequest, mBeamWidths[bi], static_cast<SizeType>(mVocabSizePadded)}));

            IGptDecoder::acceptDraftTokensByLogits(
                /* [num_draft_tokens, bs, bw, vocabPadded] */ *mDraftLogits[bi],
                /* [num_draft_tokens+1, bs, bw, vocabPadded] */ *targetLogits,
                /* [max_draft_tokens, bs, bw, vocabPadded] */ *draftProbs,
                /* [max_tokens_per_step, bs, bw, vocabPadded] */ *targetProbs,
                /* [bs, bw] */ *numDraftTokens,
                /* [max_tokens_per_step, bs, bw] */ *finishedSteps, static_cast<SizeType>(mVocabSize),
                static_cast<SizeType>(mVocabSizePadded), useRandomAcceptanceThreshold, randomAcceptanceThreshold,
                curandState, stream);
        }

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

        for (std::int32_t di = 0; di < mGeneratedTokensPerStep[bi]; ++di)
        {
            dInput.logits = ITensor::slice(targetLogits, di, singleRequest);
            dOutput.newTokens = ITensor::view(dOutput.newTokensVec[di]);
            dInput.finished = ITensor::slice(finishedSteps, di, 1);
            dOutput.finished = ITensor::slice(finishedSteps, std::min(di + 1, mGeneratedTokensPerStep[bi] - 1), 1);

            decoder.forwardAsync(dOutput, dInput);

            mNbSteps[bi] += 1;
            mFinished[bi] = mNbSteps[bi] >= mMaxNewTokens[bi];
            dInput.step += 1;
        }

        if (mGeneratedTokensPerStep[bi] > 1 && !mAcceptByLogits[bi])
        {
            auto draftTokenIds = mDraftTokenIds[bi];
            auto numDraftTokens = ITensor::slice(mNumDraftTokens, bi, singleRequest);
            // Update finished state for 0th step
            auto finishedFinal = ITensor::slice(finishedSteps, 0, 1);
            IGptDecoder::acceptDraftTokensByIds(
                /* [bs=1, bw=1, max_seq_len] */ *dOutput.ids,
                /* [bs, bw, max_draft_tokens] */ *draftTokenIds,
                /* [bs, bw] */ *dInput.lengths,
                /* [bs, bw] */ *numDraftTokens,
                /* [bs, bw] */ *dOutput.lengths,
                /* [max_tokens_per_step, bs, bw] */ *finishedSteps,
                /* [bs, bw] */ *finishedFinal,
                /* [1] */ *dOutput.finishedSum, stream);
        }

        CudaEvent event{};
        stream->record(event);
        mStream->wait(event);
    }

    CudaEvent eventStop{};
    mStream->record(eventStop);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return std::make_unique<decoder_batch::Token>(std::move(eventStop), input.active);
}

void GptDecoderBatch::forwardSync(decoder_batch::Token const& token)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    token.event.synchronize();

    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (token.active[i] && !mFinished[i])
        {
            auto& dOutput = *mDecodingOutputs[i];
            mFinished[i] = mFinished[i]
                // This condition requires the synchronization above
                || *bufferCast<SizeType>(*dOutput.finishedSum) == static_cast<SizeType>(dOutput.lengths->getSize());
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

// TODO call this at the end of forward if mFinished[i] changes from false to true?
CudaEvent GptDecoderBatch::postProcessRequest(SizeType batchIdx) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatch::newBatch(
    GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
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
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

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

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardSync()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mForwardToken);
    // wait for mFinishedSum to be updated
    mForwardEvent.synchronize();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::finalize() const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    for (SizeType batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        postProcessRequest(batchIdx);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

CudaEvent GptDecoderBatch::finalize(SizeType batchIdx) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchIdx);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return event;
}
