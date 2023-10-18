/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

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
    dInput = std::make_unique<DecodingInput>(0, 0, std::move(dummyLogits), std::move(endIds));

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mJointDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds));

    dOutput->newTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->finished = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
    // use batchSize many entries instead of the usual 1
    dOutput->finishedSum = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    mFinishedSum = mBufferManager.pinned(ITensor::makeShape({1}), nvSizeType);
    dOutput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::setup(
    SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxBatchSize > 0);
    TLLM_CHECK(maxBeamWidth > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mActualBatchSize = maxBatchSize;
    mMaxSequenceLength = maxSequenceLength;

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    auto const maxBatchSizeXmaxBeamWidth = ITensor::makeShape({maxBatchSize, maxBeamWidth});

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
    dOutput.newTokens->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.newTokens);
    dOutput.parentIds->reshape(jointOutputIdsShape);
    dOutput.lengths->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.lengths);
    dOutput.finished->reshape(maxBatchSizeXmaxBeamWidth);
    mBufferManager.setZero(*dOutput.finished);
    mBufferManager.setZero(*dOutput.finishedSum);
    // use batchSize many entries instead of the usual 1
    dOutput.finishedSum->reshape(maxBatchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    if (maxBeamWidth > 1)
    {
        dOutput.cumLogProbs->reshape(maxBatchSizeXmaxBeamWidth);
        mBufferManager.setZero(*dOutput.cumLogProbs);
        dOutput.beamHypotheses.reshape(maxBatchSize, maxBeamWidth, mMaxSequenceLength);
    }
    else
    {
        dOutput.beamHypotheses.release();
    }

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
    auto const inputLength = requestIds->getShape().d[0];
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
    dInput = std::make_unique<DecodingInput>(inputLength, localBatchSize, dJointInput.logits, endIdTensorPtr);
    dInput->embeddingBias = request.embeddingBias;
    dInput->badWordsList = request.badWordsList;
    dInput->stopWordsList = request.stopWordsList;
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

    dOutput->finished = ITensor::slice(dJointOutput.finished, batchIdx, localBatchSize);
    manager.setZero(*dOutput->finished);
    dOutput->finishedSum = ITensor::slice(dJointOutput.finishedSum, batchIdx, localBatchSize);
    manager.setZero(*dOutput->finishedSum);
    dOutput->lengths = ITensor::slice(dJointOutput.lengths, batchIdx, localBatchSize);
    kernels::invokeFill(*dOutput->lengths, inputLength, *stream);
    dOutput->newTokens = ITensor::slice(dJointOutput.newTokens, batchIdx, localBatchSize);
    manager.setZero(*dOutput->newTokens);

    if (beamWidth > 1)
    {
        dOutput->cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchIdx, localBatchSize);
        manager.setZero(*IBuffer::slice(dOutput->cumLogProbs, 0, 1));
        kernels::invokeFill(
            *IBuffer::slice(dOutput->cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *stream);

        dOutput->parentIds = ITensor::slice(dJointOutput.parentIds, batchIdx, localBatchSize);
        dOutput->parentIds->reshape(outputIdsShape);
        manager.setZero(*dOutput->parentIds);
        dOutput->beamHypotheses = dJointOutput.beamHypotheses.slice(batchIdx, localBatchSize);
        dOutput->beamHypotheses.init(manager, endId);
    }

    // remaining
    mDecoders[batchIdx]->setup(samplingConfig, localBatchSize);
    mBeamWidths[batchIdx] = beamWidth;
    mNbSteps[batchIdx] = 0;
    mFinished[batchIdx] = false;
    mMaxNewTokens[batchIdx] = maxNewTokens;

    // copy the request ids into outputIds
    auto inputIdsView = ITensor::view(requestIds, ITensor::makeShape({localBatchSize, inputLength}));
    auto outputIdsView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, mMaxSequenceLength}));
    kernels::invokeFill(*outputIdsView, endId, *stream);
    kernels::tileTensor(*outputIdsView, *inputIdsView, beamWidth, *stream);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

GptDecoderBatch::TokenPtr GptDecoderBatch::forwardAsync(
    decoder_batch::Output& output, decoder_batch::Input const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& logits = input.logits;
    auto const& logitsShape = logits->getShape();

    TLLM_CHECK(logitsShape.d[0] == mActualBatchSize);
    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK(logitsShape.d[1] == maxBeamWidth);
    TLLM_CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    TLLM_CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    // TODO should remove this reshape and set shape to [batch_size, beam_width] outside
    TensorPtr sequenceLengths = ITensor::view(output.sequenceLengths);
    sequenceLengths->reshape(ITensor::makeShape({mActualBatchSize, maxBeamWidth}));
    TLLM_CHECK(sequenceLengths);
    auto constexpr singleRequest = 1;

    CudaEvent eventStart{};
    mStream->record(eventStart);
    for (std::int32_t i = 0; i < mActualBatchSize; ++i)
    {
        if (mFinished[i] || !input.active.at(i))
            continue;

        auto& stream = mStreams[i];
        stream->wait(eventStart.get());
        auto& dInput = *mDecodingInputs[i];
        auto& dOutput = *mDecodingOutputs[i];
        auto logitsView = std::shared_ptr(ITensor::slice(logits, i, singleRequest));
        dInput.logits
            = ITensor::view(logitsView, ITensor::makeShape({singleRequest, mBeamWidths[i], logitsShape.d[2]}));
        if (srcCacheIndirection && tgtCacheIndirection)
        {
            auto srcView = std::shared_ptr(ITensor::slice(srcCacheIndirection, i, singleRequest));
            auto tgtView = std::shared_ptr(ITensor::slice(tgtCacheIndirection, i, singleRequest));
            dInput.cacheIndirection
                = ITensor::view(srcView, ITensor::makeShape({singleRequest, mBeamWidths[i], srcView->getShape().d[2]}));
            dOutput.cacheIndirection
                = ITensor::view(tgtView, ITensor::makeShape({singleRequest, mBeamWidths[i], tgtView->getShape().d[2]}));
        }
        auto sequenceLengthsView = std::shared_ptr(ITensor::slice(sequenceLengths, i, singleRequest));
        dOutput.lengths = ITensor::view(sequenceLengthsView, ITensor::makeShape({singleRequest, mBeamWidths[i]}));

        auto& decoder = *mDecoders[i];
        decoder.forwardAsync(dOutput, dInput);

        auto manager = BufferManager{stream};

        auto jointOutputIdsView = ITensor::slice(mJointDecodingOutput->ids, i, singleRequest);
        auto const& jointOutputShape = jointOutputIdsView->getShape();
        // squeeze dim 0 and set beamWidth
        jointOutputIdsView->reshape(ITensor::makeShape({mBeamWidths[i], jointOutputShape.d[2]}));

        manager.copy(*dOutput.ids, *jointOutputIdsView);

        auto jointSequenceLengthsView = ITensor::slice(mJointDecodingOutput->lengths, i, singleRequest);
        jointSequenceLengthsView->reshape(ITensor::makeShape({1, mBeamWidths[i]}));
        manager.copy(*dOutput.lengths, *jointSequenceLengthsView);

        if (mBeamWidths[i] > 1)
        {
            auto jointOutputParentIdsView = ITensor::slice(mJointDecodingOutput->parentIds, i, singleRequest);
            auto const& jointOutputParentIdsShape = jointOutputParentIdsView->getShape();
            // squeeze dim 0 and set beamWidth
            jointOutputParentIdsView->reshape(ITensor::makeShape({mBeamWidths[i], jointOutputParentIdsShape.d[2]}));

            manager.copy(*dOutput.parentIds, *jointOutputParentIdsView);
        }

        CudaEvent event{};
        stream->record(event);
        mStream->wait(event);
        dInput.step += 1;
        mNbSteps[i] += 1;
        mFinished[i] = mNbSteps[i] >= mMaxNewTokens[i];
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
                || *bufferCast<SizeType>(*dOutput.finishedSum) == static_cast<SizeType>(dOutput.finished->getSize());
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

    auto& dInput = *mDecodingInputs[batchIdx];
    auto& dOutput = *mDecodingOutputs[batchIdx];

    // TODO can we do this inplace?
    auto& outputIds = dOutput.ids;
    auto finalOutputIds = manager.gpu(outputIds->getShape(), outputIds->getDataType());
    IGptDecoder::gatherTree(*finalOutputIds, dOutput, dInput, manager);
    manager.copy(*finalOutputIds, *outputIds);

    CudaEvent event{};
    stream->record(event);
    mStream->wait(event);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return event;
}

void GptDecoderBatch::newBatch(GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // split batch into single requests
    auto const& inputLengths = inputs.lengths;
    mActualBatchSize = inputLengths->getShape().d[0];

    auto const& jointOutputIdsShape = mJointDecodingOutput->ids->getShape();
    auto const maxBatchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(mActualBatchSize <= maxBatchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK(samplingConfig.beamWidth <= maxBeamWidth);

    auto const inputIdsShape = inputs.ids->getShape();
    TensorPtr inputIdsFlatView = ITensor::view(inputs.ids);
    inputIdsFlatView->reshape(ITensor::makeShape({inputIdsShape.d[1]}));
    auto inputLengthsHost = mBufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType>(*inputLengthsHost);
    auto inputOffset = 0;
    for (auto batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        auto const inputLength = inputLengthsPtr[batchIdx];
        auto const inputShape = ITensor::makeShape({inputLength});
        TensorPtr inputView;
        if (inputs.packed)
        {
            inputView = ITensor::slice(inputIdsFlatView, inputOffset, inputLength);
            inputOffset += inputLength;
        }
        else
        {
            inputView = ITensor::slice(inputs.ids, batchIdx, 1);
            inputView->reshape(inputShape);
        }
        auto request = decoder_batch::Request{inputView, std::nullopt, inputs.endId, inputs.padId};
        request.embeddingBias = inputs.embeddingBiasOpt;
        request.badWordsList = inputs.badWordsList;
        request.stopWordsList = inputs.stopWordsList;
        newRequest(batchIdx, request, extractSamplingConfig(samplingConfig, batchIdx));
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void GptDecoderBatch::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    decoder_batch::Input batchInput{input.logits};
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

bool GptDecoderBatch::isFinishedSync()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    forwardSync(*mForwardToken);
    auto const finished
        = std::all_of(mFinished.begin(), mFinished.begin() + mActualBatchSize, [](bool x) { return x; });
    // wait for mFinishedSum to be updated
    mStream->wait(mForwardEvent);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return finished;
}

IStatefulGptDecoder::TensorPtr GptDecoderBatch::getFinalOutputIds() const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    for (SizeType batchIdx = 0; batchIdx < mActualBatchSize; ++batchIdx)
    {
        postProcessRequest(batchIdx);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return getOutputIds();
}

std::tuple<CudaEvent, IStatefulGptDecoder::TensorPtr> GptDecoderBatch::getFinalOutputIds(SizeType batchIdx) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto event = postProcessRequest(batchIdx);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return {std::move(event), getOutputIds(batchIdx)};
}
