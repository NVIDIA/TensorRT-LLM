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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>

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
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFloatType = TRTDataType<float>::value;

    auto& dInput = mDecodingInput;
    auto dummyLogits = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    auto endIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    mSetupBatchSlots = mBufferManager.emptyTensor(MemoryType::kPINNED, nvSizeType);
    dInput = std::make_unique<DecodingInput>(0, 0, 0, 0, std::move(dummyLogits), std::move(endIds), mSetupBatchSlots);

    dInput->sequenceLimitLength = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dInput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);

    auto& dOutput = mDecodingOutput;
    auto outputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    auto gatheredOutputIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput = std::make_unique<DecodingOutput>(std::move(outputIds), std::move(gatheredOutputIds));

    dOutput->newTokens = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->parentIds = mBufferManager.emptyTensor(MemoryType::kGPU, nvTokenIdType);
    dOutput->finishReasons
        = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    dOutput->finishedSum = mBufferManager.pinnedPool(ITensor::makeShape({1}), nvSizeType);
    dOutput->lengths = mBufferManager.emptyTensor(MemoryType::kGPU, nvSizeType);
    dOutput->cumLogProbs = mBufferManager.emptyTensor(MemoryType::kGPU, nvFloatType);
    dOutput->beamHypotheses.empty(mBufferManager);
    dOutput->logProbsTiled = mBufferManager.emptyTensor(MemoryType::kGPU, TRTDataType<float>::value);

    dInput->stopWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->stopWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<SizeType32>::value);
    dInput->stopWordsLists.resize(1);
    dInput->badWordsPtrs = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<int32_t*>::value);
    dInput->badWordsLens = mBufferManager.emptyTensor(MemoryType::kPINNEDPOOL, TRTDataType<SizeType32>::value);
    dInput->badWordsLists.resize(1);

    mFinishedSum = mBufferManager.pinnedPool(ITensor::makeShape({1}), nvSizeType);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(maxTokensPerStep == 1);
    mDecoder = IGptDecoder::create(
        mode, dtype, maxBatchSize, maxBeamWidth, mVocabSize, mVocabSizePadded, maxSequenceLength, mStream);

    reshapeBuffers(maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::reshapeBuffers(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
    SizeType32 sinkTokenLength, SizeType32 maxSequenceLength)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchSize > 0);
    TLLM_CHECK(beamWidth > 0);
    TLLM_CHECK(maxSequenceLength > 0);

    mMaxSequenceLength = maxSequenceLength;
    mMaxAttentionWindow = maxAttentionWindow;
    mSinkTokenLength = sinkTokenLength;

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    auto const batchSizeXbeamWidth = ITensor::makeShape({batchSize, beamWidth});
    mSetupBatchSlots->reshape(batchSizeShape);
    auto setupBatchSlotsRange = runtime::BufferRange<runtime::SizeType32>(*mSetupBatchSlots);
    std::iota(setupBatchSlotsRange.begin(), setupBatchSlotsRange.end(), 0);

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
    if (beamWidth > 1)
    {
        dOutput.gatheredIds->reshape(outputIdsShape);
    }
    else
    {
        dOutput.gatheredIds = dOutput.ids;
    }
    dOutput.newTokens->reshape(batchSizeXbeamWidth);
    mBufferManager.setZero(*dOutput.newTokens);
    dOutput.parentIds->reshape(outputIdsShape);
    dOutput.finishReasons->reshape(batchSizeXbeamWidth);
    dInput.finishReasons = ITensor::view(dOutput.finishReasons);
    mBufferManager.setZero(*dOutput.finishReasons);

    dOutput.finishedSum->reshape(batchSizeShape);
    mBufferManager.setZero(*dOutput.finishedSum);

    const_cast<ITensor&>(*dInput.badWordsPtrs).reshape(ITensor::makeShape({batchSize}));
    const_cast<ITensor&>(*dInput.badWordsLens).reshape(ITensor::makeShape({batchSize}));
    const_cast<ITensor&>(*dInput.stopWordsPtrs).reshape(ITensor::makeShape({batchSize}));
    const_cast<ITensor&>(*dInput.stopWordsLens).reshape(ITensor::makeShape({batchSize}));

    if (beamWidth > 1)
    {
        dOutput.cumLogProbs->reshape(batchSizeXbeamWidth);
        mBufferManager.setZero(*dOutput.cumLogProbs);
        dOutput.beamHypotheses.reshape(batchSize, beamWidth, mMaxSequenceLength);
    }
    dOutput.logProbsTiled->reshape(ITensor::makeShape({maxSequenceLength, batchSize, beamWidth}));
    mBufferManager.setZero(*dOutput.logProbsTiled);

    mNbSteps = 0;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::newBatch(
    GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& manager = mBufferManager;
    auto& stream = mStream;

    auto const inputLengths = inputs.lengths;
    auto const& inputLengthsShape = inputLengths->getShape();
    auto const batchSize = inputLengthsShape.d[0];
    auto const beamWidth = samplingConfig.beamWidth;

    reshapeBuffers(batchSize, beamWidth, mMaxAttentionWindow, mSinkTokenLength, mMaxSequenceLength);
    mDecoder->setup(samplingConfig, batchSize, mSetupBatchSlots);

    // sanity checks, should always be true after reshape
    auto const& outputIdsShape = mDecodingOutput->ids->getShape();
    auto const maxBatchSize = outputIdsShape.d[0];
    TLLM_CHECK(batchSize == maxBatchSize);
    auto const maxBeamWidth = outputIdsShape.d[1];
    TLLM_CHECK(beamWidth == maxBeamWidth);

    auto const& inputIds = inputs.ids;
    auto const inputLengthsHost = manager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto const* inputLengthsData = bufferCast<SizeType32>(*inputLengthsHost);
    SizeType32 const maxInputLength = *std::max_element(inputLengthsData, inputLengthsData + inputLengths->getSize());

    TensorPtr inputOffsets = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType32>::value);
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

    if (inputs.badWordsList)
    {
        auto const& badWordsShape = inputs.badWordsList->getShape();
        auto badWordsLen = badWordsShape.d[1];
        if (badWordsShape.nbDims == 3)
        {
            badWordsLen = badWordsShape.d[2];
        }
        dInput.maxBadWordsLen = badWordsLen;

        TensorPtr badWordsList = ITensor::view(inputs.badWordsList);
        auto badWordsLensRange = BufferRange<SizeType32>(*constPointerCast(dInput.badWordsLens));
        auto badWordsPtrsRange = BufferRange<TokenIdType*>(*constPointerCast(dInput.badWordsPtrs));
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            if (badWordsShape.nbDims == 3)
            {
                badWordsPtrsRange[bi] = bufferCast<TokenIdType>(*badWordsList) + bi * 2 * badWordsLen;
            }
            else
            {
                badWordsPtrsRange[bi] = bufferCast<TokenIdType>(*badWordsList);
            }
            badWordsLensRange[bi] = badWordsLen;
        }
        // NOTE(nkorobov): dInput->badWordsList is not used in gptDecoder, but required to keep badWordsList memory
        // allocated
        dInput.badWordsLists.at(0) = badWordsList;
    }
    if (inputs.stopWordsList)
    {
        auto const stopWordsLen = inputs.stopWordsList->getShape().d[2];
        dInput.maxStopWordsLen = stopWordsLen;

        TensorPtr stopWordsList = ITensor::view(inputs.stopWordsList);
        auto stopWordsPtrsRange = BufferRange<TokenIdType*>(*constPointerCast(dInput.stopWordsPtrs));
        auto stopWordsLensRange = BufferRange<SizeType32>(*constPointerCast(dInput.stopWordsLens));
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            stopWordsPtrsRange[bi] = bufferCast<TokenIdType>(*stopWordsList) + bi * 2 * stopWordsLen;
            stopWordsLensRange[bi] = stopWordsLen;
        }
        // NOTE(nkorobov): dInput->stopWordsList is not used in gptDecoder, but required to keep stopWordsList memory
        // allocated
        dInput.stopWordsLists.at(0) = stopWordsList;
    }

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
    manager.setZero(*dOutput.finishReasons);
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
        for (SizeType32 i = 0; i < batchSize; ++i)
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
    mBufferManager.setZero(*dOutput.logProbsTiled);

    // copy the request ids into dOutput.ids (with tiling)
    kernels::initOutputIds(
        *dOutput.ids, *inputIds, *inputLengths, *inputOffsets, padId, endId, maxInputLength, inputs.packed, *stream);

    // remaining
    mNbSteps = 0;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::forwardAsync(decoder::Output& output, decoder::Input const& input)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
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
    TLLM_CHECK(!srcCacheIndirection || srcCacheIndirection->getDataType() == TRTDataType<SizeType32>::value);
    TLLM_CHECK(!tgtCacheIndirection || tgtCacheIndirection->getDataType() == TRTDataType<SizeType32>::value);

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
    kernels::reduce(*mFinishedSum, *ITensor::slice(mDecodingOutput->finishedSum, 0, batchSize), *mStream);
    mStream->record(mDecodedEvent.get());

    dInput.step += 1;
    mNbSteps += 1;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mDecodedEvent.synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoder::finalize(SamplingConfig const& samplingConfig) const
{
    // TODO (rkobus) can we do this inplace?
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& outputIds = mDecodingOutput->ids;
    kernels::gatherTree(*mDecodingOutput, *mDecodingInput, mBufferManager, samplingConfig);
    mBufferManager.copy(*(mDecodingOutput->gatheredIds), *outputIds);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return;
}
