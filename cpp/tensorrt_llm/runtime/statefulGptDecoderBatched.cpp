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

#include "tensorrt_llm/runtime/statefulGptDecoderBatched.h"

#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

using namespace tensorrt_llm::runtime;

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

StatefulGptDecoderBatched::StatefulGptDecoderBatched(CudaStreamPtr stream, nvinfer1::DataType dtype)
{
    mDecoder = std::make_unique<GptDecoderBatched>(stream, SpeculativeDecodingMode::None(), dtype);

    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;

    auto const& bufferManager = mDecoder->getBufferManager();

    mBatchSlotsSetup = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mBatchSlotsDecoder = bufferManager.emptyTensor(MemoryType::kPINNEDPOOL, nvSizeType);
    mFinishedSum = BufferManager::pinned(ITensor::makeShape({1}), nvSizeType);
}

StatefulGptDecoderBatched::~StatefulGptDecoderBatched() = default;

void StatefulGptDecoderBatched::setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize,
    SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
    nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    constexpr SizeType32 maxTokensPerStep = 1;
    mDecoder->setup(mode, maxBatchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSequenceLength,
        maxTokensPerStep, dtype, modelConfig, worldConfig);

    mBatchSlotsSetup->reshape(ITensor::makeShape({maxBatchSize}));
    mBatchSlotsDecoder->reshape(ITensor::makeShape({maxTokensPerStep, maxBatchSize}));
}

void StatefulGptDecoderBatched::newBatch(GenerationInput const& inputs, GenerationOutput const& outputs,
    SamplingConfig const& samplingConfig, ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // split batch into single requests
    auto const& inputLengths = inputs.lengths;
    mDecoder->getDecoderState().setActualBatchSize(inputLengths->getShape().d[0]);
    for (auto i = 0; i < mDecoder->getDecoderState().getActualBatchSize(); ++i)
    {
        mDecoder->getDecoderState().setNumDecodingEngineTokens(i, 1);
    }

    auto const& jointOutputIdsShape = mDecoder->getDecoderState().getJointDecodingOutput().ids->getShape();
    auto const maxBatchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(mDecoder->getDecoderState().getActualBatchSize() <= maxBatchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    TLLM_CHECK(samplingConfig.beamWidth <= maxBeamWidth);

    auto const inputIdsShape = inputs.ids->getShape();
    TensorPtr inputIdsFlatView = ITensor::view(inputs.ids);

    TensorPtr batchSlotsView = ITensor::slice(mBatchSlotsSetup, 0, mDecoder->getDecoderState().getActualBatchSize());
    auto batchSlots = BufferRange<SizeType32>(*batchSlotsView);
    std::iota(batchSlots.begin(), batchSlots.end(), 0);

    if (inputs.packed && inputIdsShape.nbDims == 2)
    { // For users still pass inputs.ids with shape [1, num_tokens], do squeeze for them.
        inputIdsFlatView->squeeze(0);
    }

    auto const& bufferManager = mDecoder->getBufferManager();
    auto const& runtimeStream = bufferManager.getStream();
    auto inputLengthsHost = bufferManager.copyFrom(*inputLengths, MemoryType::kCPU);
    auto inputLengthsPtr = bufferCast<SizeType32>(*inputLengthsHost);
    auto inputOffset = 0;
    for (auto batchIdx = 0; batchIdx < mDecoder->getDecoderState().getActualBatchSize(); ++batchIdx)
    {
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
        batch_manager::CreateNewDecoderRequests().newRequest(batchIdx, request, requestSamplingConfig, modelConfig,
            *mDecoder, runtimeStream, mDecoder->getDecoderState().getMaxSequenceLength());
    }

    auto fusedSamplingConfig = samplingConfig;
    fusedSamplingConfig.cumLogProbs
        = std::vector<bool>(mDecoder->getDecoderState().getActualBatchSize(), outputs.cumLogProbs != nullptr);
    fusedSamplingConfig.outputLogProbs
        = std::vector<bool>(mDecoder->getDecoderState().getActualBatchSize(), outputs.logProbs != nullptr);

    mDecoder->getUnderlyingDecoder().setup(fusedSamplingConfig, mDecoder->getDecoderState().getActualBatchSize(),
        batchSlotsView, {mDecoder->getDecoderState().getJointDecodingOutput()});

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoderBatched::forwardAsync(decoder::Output& output, decoder::Input const& input)
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
        logits.emplace_back(ITensor::view(logitsSlice,
            ITensor::makeShape({singleRequest,
                mDecoder->getDecoderState().getJointDecodingInput().beamWidths.at(batchIdx), logitsShape.d[2]})));
    }

    decoder_batch::Input batchInput{logits};
    batchInput.batchSlots = mBatchSlotsDecoder;
    batchInput.cacheIndirection = input.cacheIndirection;

    decoder_batch::Output batchOutput;
    batchOutput.cacheIndirection = output.cacheIndirection;
    batchOutput.sequenceLengths = output.sequenceLengths;

    mDecoderFinishEvent = mDecoder->forwardAsync(batchOutput, batchInput);

    auto const& bufferManager = mDecoder->getBufferManager();

    bufferManager.setZero(*mFinishedSum);
    auto const& runtimeStream = bufferManager.getStream();

    kernels::reduce(*mFinishedSum,
        *ITensor::slice(mDecoder->getDecoderState().getJointDecodingOutput().finishedSum, 0, batchSize), runtimeStream);
    runtimeStream.record(mForwardEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void StatefulGptDecoderBatched::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecoderFinishEvent.synchronize();

    // wait for mFinishedSum to be updated
    mForwardEvent.synchronize();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getIds() const
{
    return mDecoder->getDecoderState().getIds();
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getGatheredIds() const
{
    return mDecoder->getDecoderState().getGatheredIds();
}

void StatefulGptDecoderBatched::finalize(SamplingConfig const& samplingConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto batchSlots = bufferCast<SizeType32>(*mBatchSlotsSetup);
    for (SizeType32 batchIdx = 0; batchIdx < mDecoder->getDecoderState().getActualBatchSize(); ++batchIdx)
    {
        auto slot = batchSlots[batchIdx];
        auto requestSamplingConfig = extractSamplingConfig(samplingConfig, slot);
        auto event = mDecoder->finalize(mDecoder->getDecoderState(), slot, requestSamplingConfig, /*streaming*/ false);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getCumLogProbs() const
{
    return mDecoder->getDecoderState().getCumLogProbs();
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getLogProbs() const
{
    return mDecoder->getDecoderState().getLogProbs();
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getNewTokens(SizeType32 iter) const
{
    TensorPtr newTokensView
        = ITensor::slice(mDecoder->getDecoderState().getJointDecodingOutput().newTokensSteps, iter, 1);
    newTokensView->squeeze(0);
    return ITensor::slice(newTokensView, 0, mDecoder->getDecoderState().getActualBatchSize());
}

StatefulGptDecoderBatched::TensorPtr StatefulGptDecoderBatched::getNbFinished() const
{
    return mFinishedSum;
}
