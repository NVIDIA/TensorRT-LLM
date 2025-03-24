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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;

namespace tle = tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tb = tensorrt_llm::batch_manager;

using TensorPtr = decoder_batch::Input::TensorPtr;

namespace
{

void newRequests(TensorPtr const& batchSlots, std::vector<decoder_batch::Request> const& requests,
    std::vector<SamplingConfig> const& samplingConfigs, ModelConfig const& modelConfig, GptDecoderBatched& decoder,
    std::shared_ptr<CudaStream> runtimeStream, SizeType32 maxSequenceLength)
{
    auto newRequestsAlgo = tb::CreateNewDecoderRequests();
    newRequestsAlgo(batchSlots, requests, samplingConfigs, modelConfig, decoder, *runtimeStream, maxSequenceLength);

    // Setup underlying decoder.
    auto const localBatchSize = batchSlots->getSize();
    auto samplingConfig = SamplingConfig(samplingConfigs);
    decoder.getUnderlyingDecoder().setup(
        samplingConfig, localBatchSize, batchSlots, {decoder.getDecoderState().getJointDecodingOutput()}, {requests});

    auto const& stream = decoder.getDecoderStream();
    CudaEvent event{};
    stream->record(event);
    runtimeStream->wait(event);
}

decoder_batch::Input prepareDecoderInputs(SizeType32 batchSize, SizeType32 maxBeamWidth, SizeType32 maxSeqLength,
    SizeType32 vocabSizePadded, nvinfer1::DataType dataType, std::vector<SamplingConfig>& samplingConfigs,
    std::vector<SizeType32> const& generatedTokensPerSteps, bool computeLogProbs, BufferManager& manager)
{
    std::vector<decoder_batch::Input::TensorPtr> logits;
    logits.reserve(batchSize);
    SizeType32 maxGeneratedTokensPerSteps{1};
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto const beamWidth = samplingConfigs[batchIdx].beamWidth;
        samplingConfigs[batchIdx].outputLogProbs = {{computeLogProbs}};
        samplingConfigs[batchIdx].cumLogProbs = {{computeLogProbs}};
        logits.emplace_back(
            manager.gpu(ITensor::makeShape({generatedTokensPerSteps[batchIdx], beamWidth, vocabSizePadded}), dataType));
        manager.setZero(*logits.back());

        maxGeneratedTokensPerSteps = std::max(maxGeneratedTokensPerSteps, generatedTokensPerSteps[batchIdx]);
    }

    decoder_batch::Input inputs{logits};
    inputs.batchSlots = BufferManager::pinned(
        ITensor::makeShape({maxGeneratedTokensPerSteps, batchSize}), TRTDataType<SizeType32>::value);

    if (maxBeamWidth > 1)
    {
        auto srcCacheIndirection
            = manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType32>::value);
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = std::move(srcCacheIndirection);
    }

    return inputs;
}

decoder_batch::Output prepareDecoderOutputs(SizeType32 batchSize, SizeType32 maxBeamWidth, SizeType32 maxSeqLength,
    std::vector<SizeType32> const& tiledInputLengths, BufferManager& manager)
{
    decoder_batch::Output outputs{};

    auto sequenceLengths
        = manager.copyFrom(tiledInputLengths, ITensor::makeShape({batchSize, maxBeamWidth}), MemoryType::kGPU);
    outputs.sequenceLengths = std::move(sequenceLengths);

    if (maxBeamWidth > 1)
    {
        auto tgtCacheIndirection
            = manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType32>::value);
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = std::move(tgtCacheIndirection);
    }

    return outputs;
}

std::vector<decoder_batch::Request> prepareRequests(SizeType32 batchSize, SizeType32 maxNewTokens,
    std::vector<SizeType32> const& inputLengths, std::vector<SizeType32> const& generatedTokensPerSteps,
    std::vector<SizeType32> const& acceptedTokensPerStep, TokenIdType inputTokenId, TokenIdType expectedTokenId,
    TokenIdType endId, BufferManager const& manager)
{
    auto const& stream = manager.getStream();

    std::vector<decoder_batch::Request> requests;
    requests.reserve(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto shape = ITensor::makeShape({inputLengths[batchIdx]});
        auto input = manager.gpu(shape, TRTDataType<SizeType32>::value);
        kernels::invokeFill(*input, inputTokenId, stream);

        requests.emplace_back(std::move(input), inputLengths[batchIdx], maxNewTokens, endId);
        if (generatedTokensPerSteps[batchIdx] > 1)
        {
            TokenIdType constexpr tokenToReject{1};
            TLLM_CHECK(tokenToReject != expectedTokenId);
            // fill with tokens to reject
            std::vector<TokenIdType> draftTokens(generatedTokensPerSteps[batchIdx] - 1, tokenToReject);
            // fill with tokens to accept
            std::fill(draftTokens.begin(), draftTokens.begin() + acceptedTokensPerStep[batchIdx], expectedTokenId);
            requests.back().draftTokens = manager.copyFrom(draftTokens, MemoryType::kGPU);
            requests.back().generatedTokensPerEngineStep = generatedTokensPerSteps[batchIdx];
        }
    }

    return requests;
}

[[nodiscard]] std::vector<bool> getFinished(
    ITensor const& finishedSum, std::vector<SamplingConfig> const& samplingConfigs, BufferManager& manager)
{
    auto finishedSumHost = manager.copyFrom(finishedSum, MemoryType::kCPU);
    auto finishedSumHostRange = BufferRange<SizeType32>(*finishedSumHost);
    std::vector<bool> finished(finishedSumHostRange.size());
    std::transform(finishedSumHostRange.begin(), finishedSumHostRange.end(), samplingConfigs.begin(), finished.begin(),
        [](SizeType32 sum, SamplingConfig const& config) { return sum == config.beamWidth; });

    return finished;
}

void advanceSequenceLengths(std::vector<SizeType32>& sequenceLengths,
    std::vector<SizeType32> const& acceptedTokensPerStep, std::vector<SamplingConfig> const& samplingConfigs,
    std::vector<bool> const& finished, SizeType32 batchSize, SizeType32 maxBeamWidth)
{
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        if (!finished.at(batchIdx))
        {
            for (int beamId = 0; beamId < samplingConfigs.at(batchIdx).beamWidth; beamId++)
            {
                sequenceLengths.at(tc::flat_index2(batchIdx, beamId, maxBeamWidth))
                    += acceptedTokensPerStep.at(batchIdx) + 1;
            }
        }
    }
}

void checkSequenceLengths(
    ITensor const& sequenceLengths, std::vector<SizeType32> const& expectedLengths, BufferManager& manager)
{
    auto sequenceLengthsHost = manager.copyFrom(sequenceLengths, MemoryType::kCPU);
    auto sequenceLengthsHostRange = BufferRange<SizeType32>(*sequenceLengthsHost);
    EXPECT_THAT(sequenceLengthsHostRange, ::testing::ElementsAreArray(expectedLengths));
}

void verifyResults(BufferManager& manager, GptDecoderBatched const& decoder,
    std::vector<SamplingConfig> const& samplingConfigs, std::vector<SizeType32> const& inputLengths,
    std::vector<SizeType32> const& sequenceLengths, SizeType32 batchSize, SizeType32 maxBeamWidth,
    SizeType32 maxSeqLength, SizeType32 inputTokenId, SizeType32 expectedTokenId, SizeType32 endId)
{
    for (auto b = 0; b < batchSize; ++b)
    {
        auto outputsIds = decoder.getDecoderState().getIds(b);
        // TODO: test parentIds
        // parentIds = decoder.getParentIds();
        ASSERT_TRUE(outputsIds);
        auto outputShape = outputsIds->getShape();
        EXPECT_EQ(outputShape.nbDims, 2);
        EXPECT_EQ(outputShape.d[0], maxBeamWidth);
        EXPECT_EQ(outputShape.d[1], maxSeqLength);

        auto outputsIdsHost = manager.copyFrom(*outputsIds, MemoryType::kCPU);
        auto output = bufferCast<TokenIdType>(*outputsIdsHost);

        auto samplingConfig = samplingConfigs.at(b);

        for (auto bw = 0; bw < samplingConfig.beamWidth; ++bw)
        {
            auto const result = (samplingConfig.beamWidth == 1) ? expectedTokenId : bw;

            auto* const outputPtr = output + tc::flat_index(outputShape.d, bw, 0);

            auto const inputLength = inputLengths.at(b);
            auto* begin = outputPtr;
            auto* end = outputPtr + inputLength;
            ASSERT_LE(begin, end) << "bad input length " << inputLength;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(inputTokenId)) << "input tokens: "
                                                                                << "b:" << b << " bw: " << bw;
            auto const seqLength = sequenceLengths.at(tc::flat_index2(b, bw, maxBeamWidth));
            begin = end;
            end = outputPtr + seqLength;
            ASSERT_LE(begin, end) << "bad seq length " << seqLength;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(result)) << "new tokens: "
                                                                          << "b:" << b << " bw: " << bw;
            begin = end;
            end = outputPtr + maxSeqLength;
            ASSERT_LE(begin, end) << "bad max length " << maxSeqLength;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(endId)) << "padding: "
                                                                         << "b:" << b << " bw: " << bw;
        }
    }
}

void testDecoder(nvinfer1::DataType const dtype, std::vector<SamplingConfig>& samplingConfigs, SizeType32 maxBeamWidth,
    bool computeLogProbs, bool normalizeLogProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    SizeType32 constexpr tensorParallelism{1};
    SizeType32 constexpr pipelineParallelism{1};
    SizeType32 constexpr contextParallelism{1};
    SizeType32 constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, contextParallelism, localRank};

    SizeType32 constexpr vocabSize{51200};
    SizeType32 constexpr nbAttentionLayers{2};
    SizeType32 constexpr nbRnnLayers{0};
    SizeType32 constexpr nbHeads{16};
    SizeType32 constexpr hiddenSize{1024};
    ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType32>(samplingConfigs.size());
    SizeType32 constexpr maxInputLength{8};
    SizeType32 const maxNewTokens{2};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType32 constexpr maxGeneratedTokensPerStep{1};

    std::vector<SizeType32> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType32> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    std::vector<SizeType32> generatedTokensPerSteps(batchSize);
    std::vector<SizeType32> acceptedTokensPerStep(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        generatedTokensPerSteps[batchIdx] = maxGeneratedTokensPerStep;
        acceptedTokensPerStep[batchIdx] = generatedTokensPerSteps[batchIdx] - 1;
    }

    auto constexpr inputTokenId = 1;
    auto constexpr expectedTokenId = 1023;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, inputTokenId, expectedTokenId, endId, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, computeLogProbs, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType32 const sinkTokenLength{0};

    auto const decodingMode = maxBeamWidth == 1 ? tle::DecodingMode::TopKTopP() : tle::DecodingMode::BeamSearch();

    // set up decoder
    auto decoder = GptDecoderBatched(streamPtr, modelConfig.getSpeculativeDecodingMode(), dataType);
    decoder.setup(decodingMode, batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength,
        maxGeneratedTokensPerStep, dataType, modelConfig, worldConfig);

    std::vector<decoder_batch::Request> decoderRequests;
    TensorPtr batchSlots = BufferManager::pinnedPool(ITensor::makeShape({batchSize}), TRTDataType<SizeType32>::value);
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlots);
    std::iota(batchSlotsRange.begin(), batchSlotsRange.end(), 0);
    newRequests(batchSlots, requests, samplingConfigs, modelConfig, decoder, streamPtr, maxSeqLength);
    cudaDeviceSynchronize();

    auto expectedLengths = tiledInputLengths;
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    auto const& finished = getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager);
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);

    // run decoder for 1 step
    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs,
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), batchSize, maxBeamWidth);
    decoder.forward(outputs, inputs);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);

    // run decoder for 1 step
    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs,
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), batchSize, maxBeamWidth);
    decoder.forward(outputs, inputs);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), ::testing::Each(true));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);

    EXPECT_NO_THROW(decoder.forward(outputs, inputs));
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    TensorPtr batchSlotsView = ITensor::slice(batchSlots, 0, 1);
    std::vector<SamplingConfig> singleConfig = {samplingConfigs[0]};
    newRequests(batchSlotsView, {requests[0]}, singleConfig, modelConfig, decoder, streamPtr, maxSeqLength);
    EXPECT_FALSE(getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager)[0]);
}

void testDecoderWavefront(nvinfer1::DataType const dtype, std::vector<SamplingConfig>& samplingConfigs,
    SizeType32 maxBeamWidth, bool computeLogProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    SizeType32 constexpr tensorParallelism{1};
    SizeType32 constexpr pipelineParallelism{1};
    SizeType32 constexpr contextParallelism{1};
    SizeType32 constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, contextParallelism, localRank};

    SizeType32 constexpr vocabSize{51200};
    SizeType32 constexpr nbAttentionLayers{2};
    SizeType32 constexpr nbRnnLayers{0};
    SizeType32 constexpr nbHeads{16};
    SizeType32 constexpr hiddenSize{1024};
    ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType32>(samplingConfigs.size());
    SizeType32 constexpr maxInputLength{8};
    SizeType32 constexpr maxNewTokens{8};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;
    SizeType32 constexpr maxGeneratedTokensPerStep{1};

    std::vector<SizeType32> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType32> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    std::vector<SizeType32> generatedTokensPerSteps(batchSize);
    std::vector<SizeType32> acceptedTokensPerStep(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        generatedTokensPerSteps[batchIdx] = maxGeneratedTokensPerStep;
        acceptedTokensPerStep[batchIdx] = generatedTokensPerSteps[batchIdx] - 1;
    }

    auto constexpr inputTokenId = 1;
    auto constexpr expectedTokenId = 1023;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, inputTokenId, expectedTokenId, endId, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, computeLogProbs, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType32 const sinkTokenLength{0};

    auto const decodingMode = maxBeamWidth == 1 ? tle::DecodingMode::TopKTopP() : tle::DecodingMode::BeamSearch();

    // set up decoder
    auto decoder = GptDecoderBatched(streamPtr, modelConfig.getSpeculativeDecodingMode(), dataType);
    decoder.setup(decodingMode, batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength,
        maxGeneratedTokensPerStep, dataType, modelConfig, worldConfig);

    std::vector<SizeType32> expectedSteps(batchSize, 0);
    auto expectedLengths = tiledInputLengths;

    auto const& finished = getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager);
    EXPECT_EQ(finished.size(), batchSize);
    std::vector<bool> expectedFinished(batchSize, false);

    TensorPtr batchSlots = BufferManager::pinnedPool(ITensor::makeShape({batchSize}), TRTDataType<SizeType32>::value);
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlots);
    std::iota(batchSlotsRange.begin(), batchSlotsRange.end(), 0);

    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        TensorPtr batchSlotsView = ITensor::slice(batchSlots, batchIdx, 1);
        std::vector<SamplingConfig> singleConfig = {samplingConfigs[batchIdx]};
        newRequests(batchSlotsView, {requests[batchIdx]}, singleConfig, modelConfig, decoder, streamPtr, maxSeqLength);

        decoder.forward(outputs, inputs);

        advanceSequenceLengths(
            expectedLengths, acceptedTokensPerStep, samplingConfigs, expectedFinished, batchIdx + 1, maxBeamWidth);
        checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

        for (auto bi = 0; bi <= batchIdx; ++bi)
        {
            auto firstBeamIndex = tc::flat_index2(bi, 0, maxBeamWidth);
            expectedFinished.at(bi)
                = expectedLengths.at(firstBeamIndex) - tiledInputLengths.at(firstBeamIndex) >= maxNewTokens;
        }
        EXPECT_THAT(getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager),
            ::testing::ElementsAreArray(expectedFinished));
    }

    auto finishedVec = getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager);
    while (!std::all_of(expectedFinished.begin(), expectedFinished.end(), [](bool finish) { return finish; }))
    {
        decoder.forward(outputs, inputs);
        finishedVec = getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager);

        advanceSequenceLengths(
            expectedLengths, acceptedTokensPerStep, samplingConfigs, expectedFinished, batchSize, maxBeamWidth);
        checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

        for (auto bi = 0; bi < batchSize; ++bi)
        {
            auto firstBeamIndex = tc::flat_index2(bi, 0, maxBeamWidth);
            expectedFinished.at(bi)
                = expectedLengths.at(firstBeamIndex) - tiledInputLengths.at(firstBeamIndex) >= maxNewTokens;
        }
        EXPECT_THAT(finishedVec, ::testing::ElementsAreArray(expectedFinished));

        for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            inputs.active.at(batchIdx) = !finishedVec.at(batchIdx);
        }
    }

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);
}

void testDecoderDraft(nvinfer1::DataType const dtype, std::vector<SamplingConfig>& samplingConfigs,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& generatedTokensPerSteps,
    std::vector<SizeType32> const& acceptedTokensPerStep, SizeType32 maxGeneratedTokensPerStep)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(maxBeamWidth == 1);

    SizeType32 constexpr tensorParallelism{1};
    SizeType32 constexpr pipelineParallelism{1};
    SizeType32 constexpr contextParallelism{1};
    SizeType32 constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, contextParallelism, localRank};

    SizeType32 constexpr vocabSize{51200};
    SizeType32 constexpr nbAttentionLayers{2};
    SizeType32 constexpr nbRnnLayers{0};
    SizeType32 constexpr nbHeads{16};
    SizeType32 constexpr hiddenSize{1024};
    ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);
    modelConfig.setSpeculativeDecodingMode(SpeculativeDecodingMode::DraftTokensExternal());

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType32>(samplingConfigs.size());
    SizeType32 constexpr maxInputLength{8};
    SizeType32 const maxNewTokens{4};
    auto const maxSeqLength = maxInputLength + maxNewTokens;

    std::vector<SizeType32> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType32> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    auto constexpr inputTokenId = 1;
    auto constexpr expectedTokenId = 1023;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, inputTokenId, expectedTokenId, endId, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, false, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType32 const sinkTokenLength{0};

    auto const decodingMode = tle::DecodingMode::ExternalDraftTokens(); // only supports bw=1

    // set up decoder
    auto decoder = GptDecoderBatched(streamPtr, modelConfig.getSpeculativeDecodingMode(), dataType);
    decoder.setup(decodingMode, batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength,
        maxGeneratedTokensPerStep, dataType, modelConfig, worldConfig);

    TensorPtr batchSlots = BufferManager::pinnedPool(ITensor::makeShape({batchSize}), TRTDataType<SizeType32>::value);
    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlots);
    std::iota(batchSlotsRange.begin(), batchSlotsRange.end(), 0);

    newRequests(batchSlots, requests, samplingConfigs, modelConfig, decoder, streamPtr, maxSeqLength);
    cudaDeviceSynchronize();

    auto expectedLengths = tiledInputLengths;
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    auto const& finished = getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager);
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);

    // run decoder for 1 step
    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs,
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), batchSize, maxBeamWidth);
    decoder.forward(outputs, inputs);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(
        getFinished(*decoder.getDecoderState().getFinishedSum(), samplingConfigs, manager), ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, inputTokenId, expectedTokenId, endId);
}

} // namespace

struct BeamConfig
{
    SizeType32 maxBeamWidth;
    std::vector<SizeType32> beamWidths;
};

using ParamType = std::tuple<nvinfer1::DataType, BeamConfig, bool>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
    BeamConfig const beamConfig = std::get<1>(info.param);
    name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
    for (auto const beamWdith : beamConfig.beamWidths)
    {
        name.append("Bw" + std::to_string(beamWdith));
    }
    bool const computeLogProbs{std::get<2>(info.param)};
    if (computeLogProbs)
    {
        name.append("LogProbs");
    }
    return name;
}

class ParamTest : public ::testing::TestWithParam<ParamType>
{
};

TEST_P(ParamTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    BeamConfig const beamConfig{std::get<1>(GetParam())};
    bool const computeLogProbs{std::get<2>(GetParam())};
    std::vector<SamplingConfig> samplingConfigs;
    for (auto const beamWidth : beamConfig.beamWidths)
    {
        samplingConfigs.emplace_back(beamWidth);
    }

    testDecoder(dtype, samplingConfigs, beamConfig.maxBeamWidth, computeLogProbs, true);
}

INSTANTIATE_TEST_SUITE_P(DecoderBwTest, ParamTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF),
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {4, 4, 4}},
            BeamConfig{10, {10, 10, 10}}),
        testing::Values(false, true)),
    generateTestName);

class ParamWavefrontTest : public ::testing::TestWithParam<ParamType>
{
};

TEST_P(ParamWavefrontTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    BeamConfig const beamConfig{std::get<1>(GetParam())};
    bool const computeLogProbs{std::get<2>(GetParam())};
    bool const normalizeLogProbs{true};
    std::vector<SamplingConfig> samplingConfigs;
    for (auto const beamWidth : beamConfig.beamWidths)
    {
        samplingConfigs.emplace_back(beamWidth);
    }

    testDecoderWavefront(dtype, samplingConfigs, beamConfig.maxBeamWidth, computeLogProbs);
}

INSTANTIATE_TEST_SUITE_P(DecoderBwTest, ParamWavefrontTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF),
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {4, 4, 4}},
            BeamConfig{10, {10, 10, 10}}),
        testing::Values(false, true)),
    generateTestName);

struct DraftConfig
{
    SizeType32 maxGeneratedTokensPerStep;
    std::vector<SizeType32> generatedTokensPerSteps;
    std::vector<SizeType32> acceptedTokensPerStep;
};

using DraftTestParamType = std::tuple<nvinfer1::DataType, BeamConfig, DraftConfig>;

class ParamDraftTest : public ::testing::TestWithParam<DraftTestParamType>
{
};

TEST_P(ParamDraftTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    BeamConfig const beamConfig{std::get<1>(GetParam())};
    DraftConfig const draftConfig{std::get<2>(GetParam())};

    ASSERT_EQ(beamConfig.beamWidths.size(), draftConfig.acceptedTokensPerStep.size());
    ASSERT_EQ(beamConfig.beamWidths.size(), draftConfig.generatedTokensPerSteps.size());

    std::vector<SamplingConfig> samplingConfigs;
    for (auto const beamWidth : beamConfig.beamWidths)
    {
        samplingConfigs.emplace_back(beamWidth);
    }

    testDecoderDraft(dtype, samplingConfigs, beamConfig.maxBeamWidth, draftConfig.generatedTokensPerSteps,
        draftConfig.acceptedTokensPerStep, draftConfig.maxGeneratedTokensPerStep);
}

INSTANTIATE_TEST_SUITE_P(DecoderTest, ParamDraftTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF),
        testing::Values(BeamConfig{1, {1, 1, 1}}),
        testing::Values( //
            DraftConfig{2, {1, 1, 1}, {0, 0, 0}}, DraftConfig{2, {2, 2, 2}, {1, 1, 1}},
            DraftConfig{4, {1, 2, 3}, {0, 0, 1}}

            )),
    [](testing::TestParamInfo<DraftTestParamType> const& info)
    {
        std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
        BeamConfig const beamConfig = std::get<1>(info.param);
        DraftConfig const draftConfig = std::get<2>(info.param);
        name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
        auto const batchSize = beamConfig.beamWidths.size();
        for (auto const beamWdith : beamConfig.beamWidths)
        {
            name.append("Bw" + std::to_string(beamWdith));
        }
        name.append("PerStep" + std::to_string(draftConfig.maxGeneratedTokensPerStep));
        for (std::size_t i = 0; i < batchSize; ++i)
        {
            auto const acc = draftConfig.acceptedTokensPerStep.at(i);
            auto const gen = draftConfig.generatedTokensPerSteps.at(i);
            name.append("Acc" + std::to_string(acc) + "of" + std::to_string(gen));
        }
        return name;
    });
