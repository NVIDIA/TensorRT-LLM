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

#include <algorithm>
#include <random>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptDecoderBatch.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/worldConfig.h"

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

namespace
{

decoder_batch::Input prepareDecoderInputs(SizeType batchSize, SizeType maxBeamWidth, SizeType maxSeqLength,
    SizeType vocabSizePadded, nvinfer1::DataType dataType, std::vector<SamplingConfig> const& samplingConfigs,
    std::vector<SizeType> const& generatedTokensPerSteps, BufferManager& manager)
{
    std::vector<decoder_batch::Input::TensorPtr> logits;
    logits.reserve(batchSize);
    auto constexpr tokenId = 1;
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto const beamWidth = samplingConfigs[batchIdx].beamWidth;
        logits.emplace_back(
            manager.gpu(ITensor::makeShape({generatedTokensPerSteps[batchIdx], beamWidth, vocabSizePadded}), dataType));
        manager.setZero(*logits.back());
    }

    decoder_batch::Input inputs{logits};

    if (maxBeamWidth > 1)
    {
        auto srcCacheIndirection
            = manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value);
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = std::move(srcCacheIndirection);
    }

    return inputs;
}

decoder_batch::Output prepareDecoderOutputs(SizeType batchSize, SizeType maxBeamWidth, SizeType maxSeqLength,
    std::vector<SizeType> const& tiledInputLengths, BufferManager& manager)
{
    decoder_batch::Output outputs{};

    auto sequenceLengths
        = manager.copyFrom(tiledInputLengths, ITensor::makeShape({batchSize, maxBeamWidth}), MemoryType::kGPU);
    outputs.sequenceLengths = std::move(sequenceLengths);

    if (maxBeamWidth > 1)
    {
        auto tgtCacheIndirection
            = manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value);
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = std::move(tgtCacheIndirection);
    }

    return outputs;
}

std::vector<decoder_batch::Request> prepareRequests(SizeType batchSize, SizeType maxNewTokens,
    std::vector<SizeType> const& inputLengths, std::vector<SizeType> const& generatedTokensPerSteps,
    std::vector<SizeType> const& acceptedTokensPerStep, TokenIdType tokenId, TokenIdType endId, TokenIdType padId,
    bool computeLogProbs, BufferManager& manager)
{
    auto& stream = manager.getStream();

    std::vector<decoder_batch::Request> requests;
    requests.reserve(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        auto shape = ITensor::makeShape({inputLengths[batchIdx]});
        auto input = manager.gpu(shape, TRTDataType<SizeType>::value);
        kernels::invokeFill(*input, tokenId, stream);

        requests.emplace_back(decoder_batch::Request{std::move(input), inputLengths[batchIdx], maxNewTokens, endId});
        if (generatedTokensPerSteps[batchIdx] > 1)
        {
            std::vector<TokenIdType> draftTokens(generatedTokensPerSteps[batchIdx] - 1);
            std::fill(draftTokens.begin(), draftTokens.begin() + acceptedTokensPerStep[batchIdx], 1023);
            requests.back().draftTokens = manager.copyFrom(draftTokens, MemoryType::kGPU);
        }
        requests.back().computeCumLogProbs = computeLogProbs;
        requests.back().computeLogProbs = computeLogProbs;
    }

    return requests;
}

void advanceSequenceLengths(std::vector<SizeType>& sequenceLengths, std::vector<SizeType> const& acceptedTokensPerStep,
    std::vector<SamplingConfig> const& samplingConfigs, SizeType batchSize, SizeType maxBeamWidth)
{
    for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
    {
        for (int beamId = 0; beamId < samplingConfigs.at(batchIdx).beamWidth; beamId++)
        {
            sequenceLengths.at(tc::flat_index2(batchIdx, beamId, maxBeamWidth))
                += acceptedTokensPerStep.at(batchIdx) + 1;
        }
    }
}

void checkSequenceLengths(
    ITensor const& sequenceLengths, std::vector<SizeType> const& expectedLengths, BufferManager& manager)
{
    auto sequenceLengthsHost = manager.copyFrom(sequenceLengths, MemoryType::kCPU);
    auto sequenceLengthsHostRange = BufferRange<SizeType>(*sequenceLengthsHost);
    EXPECT_THAT(sequenceLengthsHostRange, ::testing::ElementsAreArray(expectedLengths));
}

void verifyResults(BufferManager& manager, GptDecoderBatch const& decoder,
    std::vector<SamplingConfig> const& samplingConfigs, std::vector<SizeType> const& inputLengths,
    std::vector<SizeType> const& sequenceLengths, SizeType batchSize, SizeType maxBeamWidth, SizeType maxSeqLength,
    SizeType tokenId, SizeType padId)
{
    auto outputsIds = decoder.getOutputIds();
    // TODO: test parentIds
    // parentIds = decoder.getParentIds();
    ASSERT_TRUE(outputsIds);
    auto outputShape = outputsIds->getShape();
    EXPECT_EQ(outputShape.nbDims, 3);
    EXPECT_EQ(outputShape.d[0], batchSize);
    EXPECT_EQ(outputShape.d[1], maxBeamWidth);
    EXPECT_EQ(outputShape.d[2], maxSeqLength);

    auto outputsIdsHost = manager.copyFrom(*outputsIds, MemoryType::kCPU);
    auto output = bufferCast<TokenIdType>(*outputsIdsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        auto samplingConfig = samplingConfigs.at(b);
        for (auto bw = 0; bw < samplingConfig.beamWidth; ++bw)
        {
            auto const result = (samplingConfig.beamWidth == 1) ? 1023 : bw;

            auto const outputPtr = output + tc::flat_index(outputShape.d, b, bw, 0);
            auto begin = outputPtr;
            auto end = outputPtr + inputLengths.at(b);
            ASSERT_LE(begin, end) << "bad input length " << inputLengths.at(b);
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(tokenId)) << "input tokens: "
                                                                           << "b:" << b << " bw: " << bw;
            begin = end;
            end = outputPtr + sequenceLengths.at(tc::flat_index2(b, bw, maxBeamWidth));
            ASSERT_LE(begin, end) << "bad seq length " << sequenceLengths.at(b);
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(result)) << "new tokens: "
                                                                          << "b:" << b << " bw: " << bw;
            begin = end;
            end = outputPtr + maxSeqLength;
            ASSERT_LE(begin, end) << "bad max length " << maxSeqLength;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(padId)) << "padding: "
                                                                         << "b:" << b << " bw: " << bw;
        }
    }
}

void testDecoder(nvinfer1::DataType const dtype, std::vector<SamplingConfig> const& samplingConfigs,
    SizeType maxBeamWidth, bool computeLogProbs, bool normalizeLogProbs)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};
    TokenIdType constexpr padId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType>(samplingConfigs.size());
    SizeType constexpr maxInputLength{8};
    SizeType const maxNewTokens{2};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr maxGeneratedTokensPerStep{1};

    std::vector<SizeType> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    std::vector<SizeType> generatedTokensPerSteps(batchSize);
    std::vector<SizeType> acceptedTokensPerStep(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        generatedTokensPerSteps[batchIdx] = maxGeneratedTokensPerStep;
        acceptedTokensPerStep[batchIdx] = generatedTokensPerSteps[batchIdx] - 1;
    }

    auto constexpr tokenId = 1;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, tokenId, endId, padId, computeLogProbs, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType const sinkTokenLength{0};

    // set up decoder
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, streamPtr);
    decoder.setup(batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength, maxGeneratedTokensPerStep,
        dataType);

    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        decoder.newRequest(batchIdx, requests[batchIdx], samplingConfigs[batchIdx]);
    }
    cudaDeviceSynchronize();

    auto expectedLengths = tiledInputLengths;
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    auto const& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);

    // run decoder for 1 step
    decoder.forward(outputs, inputs);

    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs, batchSize, maxBeamWidth);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);

    // run decoder for 1 step
    decoder.forward(outputs, inputs);

    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs, batchSize, maxBeamWidth);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(true));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);

    EXPECT_NO_THROW(decoder.forward(outputs, inputs));
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    decoder.newRequest(0, requests[0], samplingConfigs[0]);
    EXPECT_FALSE(decoder.getFinished()[0]);
}

void testDecoderWavefront(nvinfer1::DataType const dtype, std::vector<SamplingConfig> const& samplingConfigs,
    SizeType maxBeamWidth, bool computeLogProbs)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};
    TokenIdType constexpr padId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType>(samplingConfigs.size());
    SizeType constexpr maxInputLength{8};
    SizeType constexpr maxNewTokens{8};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr maxGeneratedTokensPerStep{1};

    std::vector<SizeType> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    std::vector<SizeType> generatedTokensPerSteps(batchSize);
    std::vector<SizeType> acceptedTokensPerStep(batchSize);
    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        generatedTokensPerSteps[batchIdx] = maxGeneratedTokensPerStep;
        acceptedTokensPerStep[batchIdx] = generatedTokensPerSteps[batchIdx] - 1;
    }

    auto constexpr tokenId = 1;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, tokenId, endId, padId, computeLogProbs, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType const sinkTokenLength{0};

    // set up decoder
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, streamPtr);
    decoder.setup(batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength, maxGeneratedTokensPerStep,
        dataType);

    std::vector<SizeType> expectedSteps(batchSize, 0);
    auto expectedLengths = tiledInputLengths;

    auto const& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    std::vector<bool> expectedFinished(batchSize, true);

    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        decoder.newRequest(batchIdx, requests[batchIdx], samplingConfigs[batchIdx]);

        decoder.forward(outputs, inputs);

        advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs, batchIdx + 1, maxBeamWidth);
        checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

        for (auto bi = 0; bi <= batchIdx; ++bi)
        {
            auto firstBeamIndex = tc::flat_index2(bi, 0, maxBeamWidth);
            expectedFinished.at(bi)
                = expectedLengths.at(firstBeamIndex) - tiledInputLengths.at(firstBeamIndex) == maxNewTokens;
        }
        EXPECT_THAT(decoder.getFinished(), ::testing::ElementsAreArray(expectedFinished));
    }

    auto finishedVec = decoder.getFinished();
    while (!std::any_of(finishedVec.begin(), finishedVec.end(), [](bool finish) { return finish; }))
    {
        decoder.forward(outputs, inputs);
        finishedVec = decoder.getFinished();

        advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs, batchSize, maxBeamWidth);
        checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

        for (auto bi = 0; bi < batchSize; ++bi)
        {
            auto firstBeamIndex = tc::flat_index2(bi, 0, maxBeamWidth);
            expectedFinished.at(bi)
                = expectedLengths.at(firstBeamIndex) - tiledInputLengths.at(firstBeamIndex) == maxNewTokens;
        }
        EXPECT_THAT(finishedVec, ::testing::ElementsAreArray(expectedFinished));
    }

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);
}

void testDecoderDraft(nvinfer1::DataType const dtype, std::vector<SamplingConfig> const& samplingConfigs,
    SizeType maxBeamWidth, std::vector<SizeType> const& generatedTokensPerSteps,
    std::vector<SizeType> const& acceptedTokensPerStep, SizeType maxGeneratedTokensPerStep)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    TokenIdType constexpr endId{50257};
    TokenIdType constexpr padId{50257};

    auto const dataType = modelConfig.getDataType();
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

    auto const batchSize = static_cast<SizeType>(samplingConfigs.size());
    SizeType constexpr maxInputLength{8};
    SizeType const maxNewTokens{4};
    auto const maxSeqLength = maxInputLength + maxNewTokens;

    std::vector<SizeType> inputLengths(batchSize);
    std::iota(inputLengths.begin(), inputLengths.end(), 4);

    std::vector<SizeType> tiledInputLengths;
    for (int batchIdx = 0; batchIdx < inputLengths.size(); batchIdx++)
    {
        for (int beamId = 0; beamId < maxBeamWidth; beamId++)
        {
            tiledInputLengths.push_back(inputLengths.at(batchIdx));
        }
    }

    std::vector<SizeType> advancedTokensPerStep{generatedTokensPerSteps};
    std::for_each(advancedTokensPerStep.begin(), advancedTokensPerStep.end(), [](auto& x) { x -= 1; });

    auto constexpr tokenId = 1;
    auto requests = prepareRequests(batchSize, maxNewTokens, inputLengths, generatedTokensPerSteps,
        acceptedTokensPerStep, tokenId, endId, padId, false, manager);

    // set up inputs and outputs
    auto inputs = prepareDecoderInputs(batchSize, maxBeamWidth, maxSeqLength, vocabSizePadded, dataType,
        samplingConfigs, generatedTokensPerSteps, manager);
    auto outputs = prepareDecoderOutputs(batchSize, maxBeamWidth, maxSeqLength, tiledInputLengths, manager);

    // We set maxAttentionWindow = maxSeqLength, but it can be smaller than maxSeqLength (cyclic kv cache).
    auto const maxAttentionWindow = maxSeqLength;
    SizeType const sinkTokenLength{0};

    // set up decoder
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, streamPtr);
    decoder.setup(batchSize, maxBeamWidth, maxAttentionWindow, sinkTokenLength, maxSeqLength, maxGeneratedTokensPerStep,
        dataType);

    for (auto batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        decoder.newRequest(batchIdx, requests[batchIdx], samplingConfigs[batchIdx]);
    }
    cudaDeviceSynchronize();

    auto expectedLengths = tiledInputLengths;
    auto generatedLengths = tiledInputLengths;
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);

    auto const& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, expectedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);

    // run decoder for 1 step
    decoder.forward(outputs, inputs);

    advanceSequenceLengths(expectedLengths, acceptedTokensPerStep, samplingConfigs, batchSize, maxBeamWidth);
    // WAR: we don't write endId back into outputIds when we rejected tokens,
    // so we adjust the lengths for verifyResults here
    advanceSequenceLengths(generatedLengths, advancedTokensPerStep, samplingConfigs, batchSize, maxBeamWidth);
    checkSequenceLengths(*outputs.sequenceLengths, expectedLengths, manager);
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(false));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, generatedLengths, batchSize, maxBeamWidth,
        maxSeqLength, tokenId, padId);
}

} // namespace

struct BeamConfig
{
    SizeType maxBeamWidth;
    std::vector<SizeType> beamWidths;
};

using ParamType = std::tuple<nvinfer1::DataType, BeamConfig, bool>;

std::string generateTestName(const testing::TestParamInfo<ParamType>& info)
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
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {1, 1}},
            BeamConfig{4, {3, 3, 3}}, BeamConfig{4, {1, 2, 3, 4}}),
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
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {1, 1}},
            BeamConfig{4, {3, 3, 3}}, BeamConfig{4, {1, 2, 3, 4}}),
        testing::Values(false, true)),
    generateTestName);

struct DraftConfig
{
    SizeType maxGeneratedTokensPerStep;
    std::vector<SizeType> generatedTokensPerSteps;
    std::vector<SizeType> acceptedTokensPerStep;
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
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{4, {1, 1, 1}}),
        testing::Values( //
            DraftConfig{2, {1, 1, 1}, {0, 0, 0}}, DraftConfig{2, {2, 2, 2}, {1, 1, 1}},
            DraftConfig{4, {1, 2, 3}, {0, 0, 1}}

            )),
    [](const testing::TestParamInfo<DraftTestParamType>& info)
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
