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

#include "tensorrt_llm/runtime/statefulGptDecoder.h"

#include <algorithm>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

using TensorPtr = ITensor::SharedPtr;

StatefulGptDecoder::StatefulGptDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream)
    : mVocabSize{vocabSize}
    , mVocabSizePadded{vocabSizePadded}
    , mStream{std::move(stream)}
    , mBufferManager{mStream}
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mDecodingInput;
    auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dInput = std::make_unique<DecodingInput>(0, 0, 0, 0, std::move(dummyLogits), std::move(endIds));

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds));

    dOutput->newTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->finished
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    dOutput->finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
    dOutput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxAttentionWindow,
    SizeType sinkTokenLength, SizeType maxSequenceLength, SizeType maxTokensPerStep, nvinfer1::DataType dtype)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxTokensPerStep == 1);
    mDecoder = IGptDecoder::create(dtype, mVocabSize, mVocabSizePadded, mStream);

    reshapeBuffers(maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::reshapeBuffers(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
    SizeType sinkTokenLength, SizeType maxSequenceLength)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchSize > 0);
    TLLM_CHECK(beamWidth > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mSinkTokenLength = sinkTokenLength;

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    auto const batchSizeXbeamWidth = ITensor::makeShape({batchSize, beamWidth});

    auto& dInput = *mDecodingInput;
    const_cast<ITensor&>(*dInput.endIds).reshape(batchSizeXbeamWidth);
    auto& sequenceLimitLength = const_cast<ITensor&>(*dInput.sequenceLimitLength);
    sequenceLimitLength.reshape(batchSizeShape);
    kernels::invokeFill(sequenceLimitLength, mMaxSequenceLength, *mStream);
    auto& inputLengths = const_cast<ITensor&>(*dInput.lengths);
    inputLengths.reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(inputLengths);

    auto const outputIdsShape = ITensor::makeShape({batchSize, beamWidth, maxSequenceLength});

    auto& dOutput = *mDecodingOutput;
    dOutput.ids->reshape(outputIdsShape);
    dOutput.newTokens->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*dOutput.newTokens);
    dOutput.parentIds->reshape(outputIdsShape);
    dOutput.finished->reshape(batchSizeXbeamWidth);
    dInput.finished = ITensor::view(dOutput.finished);
    mBufferManager.setZero(*dOutput.finished);
    mBufferManager.setZero(*dOutput.finishedSum);

    if (beamWidth > 1)
    {
        dOutput.cumLogProbs->reshape(batchSizeXbeamWidth);
        mBufferManager.setZero(*dOutput.cumLogProbs);
        dOutput.beamHypotheses.reshape(batchSize, beamWidth, mMaxSequenceLength);
    }
    else
    {
        dOutput.beamHypotheses.release();
    }

    mNbSteps = 0;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::newBatch(
    GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mBufferManager;
    auto& stream = mStream;

    auto const inputLengths = inputs.lengths;
    auto const& inputLengthsShape = inputLengths->getShape();
    auto const batchSize = inputLengthsShape.d[0];
    auto const beamWidth = samplingConfig.beamWidth;

    reshapeBuffers(batchSize, beamWidth, mMaxAttentionWindow, mSinkTokenLength, mMaxSequenceLength);
    mDecoder->setup(samplingConfig, batchSize, mMaxSequenceLength);

    // sanity checks, should always be true after reshape
    auto const& outputIdsShape = mDecodingOutput->ids->getShape();
    auto const maxBatchSize = outputIdsShape.d[0];
    TLLM_CHECK(batchSize == maxBatchSize);
    auto const maxBeamWidth = outputIdsShape.d[1];
    TLLM_CHECK(beamWidth == maxBeamWidth);

    auto const& inputIds = inputs.ids;
    auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto const* inputLengthsData = bufferCast<SizeType>(*inputLengthsHost);
    SizeType const maxInputLength = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());

    TensorPtr inputOffsets = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType>::value);
    if (inputs.packed)
    {
        inputOffsets->reshape(ITensor::makeShape({batchSize + 1}));
        manager.setZero(*inputOffsets);
        kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, manager, *stream);
    }

    TLLM_CHECK(inputIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = inputs.endId;
    auto const padId = inputs.padId;

    // inputs
    auto& dInput = *mDecodingInput;
    dInput.maxLength = maxInputLength;
    dInput.maxAttentionWindow = mMaxAttentionWindow;
    dInput.sinkTokenLength = mSinkTokenLength;
    dInput.batchSize = batchSize;
    kernels::invokeFill(const_cast<ITensor&>(*dInput.endIds), endId, *stream);
    dInput.embeddingBias = inputs.embeddingBias;
    dInput.badWordsList = inputs.badWordsList;
    dInput.stopWordsList = inputs.stopWordsList;
    auto inputLengthsView = ITensor::view(dInput.lengths, ITensor::makeShape({batchSize * beamWidth}));
    kernels::tileTensor(const_cast<ITensor&>(*inputLengthsView), *inputLengths, beamWidth, *stream);
    if (inputs.maxNewTokens)
    {
        auto const maxNewTokens = inputs.maxNewTokens.value();
        TLLM_CHECK_WITH_INFO(maxInputLength + maxNewTokens <= mMaxSequenceLength,
            tc::fmtstr("Input length (%d) + max new tokens (%d) must be less than max sequence length (%d).",
                maxInputLength, maxNewTokens, mMaxSequenceLength));
        manager.copy(*inputLengths, const_cast<ITensor&>(*dInput.sequenceLimitLength));
        kernels::invokeAdd(const_cast<ITensor&>(*dInput.sequenceLimitLength), maxNewTokens, *stream);
    }
    else
    {
        kernels::invokeFill(const_cast<ITensor&>(*dInput.sequenceLimitLength), mMaxSequenceLength, *stream);
    }

    // output
    auto& dOutput = *mDecodingOutput;
    manager.setZero(*dOutput.newTokens);
    manager.setZero(*dOutput.finished);
    manager.setZero(*dOutput.finishedSum);

    // If outputs contains cumLogProbs, use that
    if (outputs.cumLogProbs)
    {
        dOutput.cumLogProbs = outputs.cumLogProbs;
    }
    dOutput.logProbs = outputs.logProbs;

    if (dOutput.cumLogProbs)
        manager.setZero(*dOutput.cumLogProbs);

    if (dOutput.logProbs)
        manager.setZero(*dOutput.logProbs);

    if (beamWidth > 1)
    {
        std::vector<float> cumLogProbsHost(batchSize * beamWidth, DecodingOutput::kNegativeInfinity);
        // Set the entries for the first beam to 0
        for (SizeType i = 0; i < batchSize; ++i)
        {
            cumLogProbsHost[tc::flat_index2(i, 0, beamWidth)] = 0;
        }
        manager.copy(cumLogProbsHost.data(), *dOutput.cumLogProbs);

        manager.setZero(*dOutput.parentIds);
        dOutput.beamHypotheses.init(manager, endId);
    }
    else
    {
        // manager.setZero(*dOutput.cumLogProbs);
    }

    // copy the request ids into dOutput.ids (with tiling)
    kernels::initOutputIds(
        *dOutput.ids, *inputIds, *inputLengths, *inputOffsets, padId, endId, maxInputLength, inputs.packed, *stream);

    // remaining
    mNbSteps = 0;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& logits = input.logits;
    auto const& logitsShape = logits->getShape();

    auto const& outputIdsShape = mDecodingOutput->ids->getShape();
    auto const batchSize = outputIdsShape.d[0];
    TLLM_CHECK(logitsShape.d[0] == batchSize);
    auto const maxBeamWidth = outputIdsShape.d[1];
    TLLM_CHECK(logitsShape.d[1] == maxBeamWidth);
    TLLM_CHECK(static_cast<std::size_t>(logitsShape.d[2]) == mVocabSizePadded);

    auto& srcCacheIndirection = input.cacheIndirection;
    auto& tgtCacheIndirection = output.cacheIndirection;
    TLLM_CHECK_WITH_INFO((srcCacheIndirection && tgtCacheIndirection) || (!srcCacheIndirection && !tgtCacheIndirection),
        "Specify both srcCacheIndirection and tgtCacheIndirection or neither.");
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType>::value);

    auto& dInput = *mDecodingInput;
    auto& dOutput = *mDecodingOutput;
    dInput.logits = logits;
    if (srcCacheIndirection && tgtCacheIndirection)
    {
        dInput.cacheIndirection = srcCacheIndirection;
        dOutput.cacheIndirection = tgtCacheIndirection;
    }
    dOutput.lengths = output.sequenceLengths;

    mDecoder->forwardAsync(dOutput, dInput);
    mStream->record(mDecodedEvent.get());

    dInput.step += 1;
    mNbSteps += 1;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::forwardSync()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mDecodedEvent.synchronize();
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::finalize() const
{
    // TODO (rkobus) can we do this inplace?
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& outputIds = mDecodingOutput->ids;
    auto finalOutputIds = mBufferManager.gpu(outputIds->getShape(), outputIds->getDataType());
    mDecoder->gatherTree(*finalOutputIds, *mDecodingOutput, *mDecodingInput, mBufferManager);
    mBufferManager.copy(*finalOutputIds, *outputIds);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return;
}
