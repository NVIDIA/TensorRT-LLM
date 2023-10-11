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

#include "tensorrt_llm/runtime/statefulGptDecoder.h"

#include <algorithm>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

namespace tc = tensorrt_llm::common;
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
    dInput = std::make_unique<DecodingInput>(0, 0, std::move(dummyLogits), std::move(endIds));

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds));

    dOutput->newTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->finished = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<bool>::value);
    dOutput->finishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
    dOutput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::setup(
    SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mDecoder = IGptDecoder::create(dtype, mVocabSize, mVocabSizePadded, mStream);

    reshapeBuffers(maxBatchSize, maxBeamWidth, maxSequenceLength);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::reshapeBuffers(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchSize > 0);
    TLLM_CHECK(beamWidth > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mMaxSequenceLength = maxSequenceLength;

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

    mMaxNewTokens = 0;
    mNbSteps = 0;
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
void initOutputIds(TensorPtr const& outputIds, TensorPtr const& inputIds, TensorPtr const& inputLengths,
    TensorPtr const& inputOffsets, SizeType const padId, SizeType const endId, SizeType const maxInputLength,
    bool const inputPacked, CudaStream const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    kernels::invokeFill(*outputIds, endId, stream);

    if (inputPacked)
    {
        kernels::invokeCopyPackedInputToOutput(*outputIds, *inputIds, *inputOffsets, maxInputLength, padId, stream);
    }
    else
    {
        kernels::invokeCopyInputToOutput(*outputIds, *inputIds, *inputLengths, padId, stream);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}
} // namespace

void StatefulGptDecoder::newBatch(GenerationInput const& inputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& manager = mBufferManager;
    auto& stream = mStream;

    auto const inputLengths = inputs.lengths;
    auto const& inputLengthsShape = inputLengths->getShape();
    auto const batchSize = inputLengthsShape.d[0];
    auto const beamWidth = samplingConfig.beamWidth;

    reshapeBuffers(batchSize, beamWidth, mMaxSequenceLength);
    mDecoder->setup(samplingConfig, batchSize);

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

    mMaxNewTokens = inputs.maxNewTokens.value_or(mMaxSequenceLength - maxInputLength);
    TLLM_CHECK_WITH_INFO(maxInputLength + mMaxNewTokens <= mMaxSequenceLength,
        tc::fmtstr("Input length (%d) + max new tokens (%d) must be less than max sequence length (%d).",
            maxInputLength, mMaxNewTokens, mMaxSequenceLength));

    TLLM_CHECK(inputIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = inputs.endId;
    auto const padId = inputs.padId;

    // inputs
    auto& dInput = *mDecodingInput;
    dInput.maxLength = maxInputLength;
    dInput.batchSize = batchSize;
    kernels::invokeFill(const_cast<ITensor&>(*dInput.endIds), endId, *stream);
    dInput.embeddingBias = inputs.embeddingBiasOpt;
    dInput.badWordsList = inputs.badWordsList;
    dInput.stopWordsList = inputs.stopWordsList;
    kernels::invokeFill(const_cast<ITensor&>(*dInput.sequenceLimitLength), mMaxSequenceLength, *stream);
    auto inputLengthsView = ITensor::view(dInput.lengths, ITensor::makeShape({batchSize * beamWidth}));
    kernels::tileTensor(const_cast<ITensor&>(*inputLengthsView), *inputLengths, beamWidth, *stream);

    // output
    auto& dOutput = *mDecodingOutput;
    manager.setZero(*dOutput.newTokens);
    manager.setZero(*dOutput.finished);
    manager.setZero(*dOutput.finishedSum);

    if (beamWidth > 1)
    {
        std::vector<float> cumLogProbsHost(batchSize * beamWidth, DecodingOutput::kNegativeInfinity);
        // Set the entries for the first beam to 0
        for (SizeType i = 0; i < batchSize; ++i)
        {
            cumLogProbsHost[tc::flat_index2(i, 0, beamWidth)] = 0;
        }
        manager.copy(cumLogProbsHost.data(), *dOutput.cumLogProbs);

        // kernels::invokeFill(*dOutput.cumLogProbs, DecodingOutput::kNegativeInfinity, *stream);
        // for (SizeType batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        // {
        //     auto cumLogProbsSlice = ITensor::slice(dOutput.cumLogProbs, batchIdx, 1);
        //     manager.setZero(*IBuffer::slice(cumLogProbsSlice, 0, 1));
        // }

        manager.setZero(*dOutput.parentIds);
        dOutput.beamHypotheses.init(manager, endId);
    }
    else
    {
        // manager.setZero(*dOutput.cumLogProbs);
    }

    // copy the request ids into dOutput.ids (with tiling)
    initOutputIds(
        dOutput.ids, inputIds, inputLengths, inputOffsets, padId, endId, maxInputLength, inputs.packed, *stream);

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

bool StatefulGptDecoder::isFinishedSync()
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    mDecodedEvent.synchronize();

    auto& dOutput = *mDecodingOutput;
    auto finished = mNbSteps >= mMaxNewTokens
        // This condition requires the synchronization above
        || *bufferCast<SizeType>(*dOutput.finishedSum) == static_cast<SizeType>(dOutput.finished->getSize());

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return finished;
}

IStatefulGptDecoder::TensorPtr StatefulGptDecoder::getFinalOutputIds() const
{
    // TODO (rkobus) can we do this inplace?
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto& outputIds = mDecodingOutput->ids;
    auto finalOutputIds = mBufferManager.gpu(outputIds->getShape(), outputIds->getDataType());
    IGptDecoder::gatherTree(*finalOutputIds, *mDecodingOutput, *mDecodingInput, mBufferManager);
    mBufferManager.copy(*finalOutputIds, *outputIds);
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
    return outputIds;
}
