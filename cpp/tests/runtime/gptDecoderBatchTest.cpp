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

void verifyResults(BufferManager& manager, GptDecoderBatch const& decoder,
    std::vector<SamplingConfig> const& samplingConfigs, std::vector<SizeType> const& inputLengths, SizeType batchSize,
    SizeType maxBeamWidth, SizeType maxSeqLength, SizeType nbNewTokens, int tokenId, int padId)
{
    auto sequenceLengths = decoder.getOutputLengths();
    ASSERT_TRUE(sequenceLengths);
    EXPECT_EQ(sequenceLengths->getSize(), batchSize * maxBeamWidth);
    auto sequenceLengthsHost = manager.copyFrom(*sequenceLengths, MemoryType::kCPU);
    auto sequenceLengthsPtr = bufferCast<SizeType>(*sequenceLengthsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        auto samplingConfig = samplingConfigs[b];
        for (auto bw = 0; bw < samplingConfig.beamWidth; ++bw)
        {
            auto index = tc::flat_index(sequenceLengths->getShape().d, b, bw);
            EXPECT_EQ(sequenceLengthsPtr[index], inputLengths[b] + nbNewTokens);
        }
    }

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
        auto samplingConfig = samplingConfigs[b];
        for (auto bw = 0; bw < samplingConfig.beamWidth; ++bw)
        {
            auto const result = (samplingConfig.beamWidth == 1) ? 1023 : bw;

            auto const outputPtr = output + tc::flat_index(outputShape.d, b, bw, 0);
            auto begin = outputPtr;
            auto end = outputPtr + inputLengths[b];
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(tokenId)) << "input tokens: "
                                                                           << "b:" << b << " bw: " << bw;
            begin = end;
            end = begin + nbNewTokens;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(result)) << "new tokens: "
                                                                          << "b:" << b << " bw: " << bw;
            begin = end;
            end = outputPtr + maxSeqLength;
            ASSERT_THAT(std::vector(begin, end), ::testing::Each(padId)) << "padding: "
                                                                         << "b:" << b << " bw: " << bw;
        }
    }
}

void testDecoder(nvinfer1::DataType const dtype, std::vector<SamplingConfig> const& samplingConfigs, int maxBeamWidth)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig constexpr worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    // create decoder
    int constexpr endId{50257};
    int constexpr padId{50257};

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, streamPtr);

    // setup decoder
    auto const batchSize = static_cast<SizeType>(samplingConfigs.size());
    SizeType constexpr maxInputLength{8};
    SizeType constexpr maxNewTokens{2};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;

    decoder.setup(batchSize, maxBeamWidth, maxSeqLength, modelConfig.getDataType());

    std::vector<SizeType> const inputLengths{4, 5, 6, 7};
    std::vector<SizeType> tiledInputLengths;
    for (int batch_id = 0; batch_id < inputLengths.size(); batch_id++)
    {
        for (int beam_id = 0; beam_id < maxBeamWidth; beam_id++)
        {
            tiledInputLengths.push_back(inputLengths.at(batch_id));
        }
    }

    // set up inputs
    auto logits = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, vocabSizePadded}), modelConfig.getDataType()));
    manager.setZero(*logits);

    decoder_batch::Input inputs{logits};
    if (maxBeamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    decoder_batch::Output outputs{};

    if (maxBeamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;
    }
    auto sequenceLengths
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize * maxBeamWidth}), TRTDataType<SizeType>::value));
    manager.copy(tiledInputLengths.data(), *sequenceLengths);
    outputs.sequenceLengths = sequenceLengths;

    auto constexpr tokenId = 1;
    std::vector<decoder_batch::Input::TensorPtr> inputIds;
    for (auto b = 0; b < batchSize; ++b)
    {
        auto shape = ITensor::makeShape({inputLengths[b]});
        auto input = std::shared_ptr(manager.gpu(shape, TRTDataType<SizeType>::value));
        kernels::invokeFill(*input, tokenId, *streamPtr);
        inputIds.emplace_back(input);
        decoder.newRequest(b, decoder_batch::Request{inputIds[b], maxNewTokens, endId, padId}, samplingConfigs[b]);
    }
    cudaDeviceSynchronize();

    auto const& nbSteps = decoder.getNbSteps();
    EXPECT_EQ(nbSteps.size(), batchSize);
    EXPECT_THAT(nbSteps, ::testing::Each(0));

    auto const& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    EXPECT_THAT(finished, ::testing::Each(false));

    verifyResults(
        manager, decoder, samplingConfigs, inputLengths, batchSize, maxBeamWidth, maxSeqLength, 0, tokenId, padId);

    // run decoder for 1 step
    decoder.forward(outputs, inputs);
    EXPECT_THAT(decoder.getNbSteps(), ::testing::Each(1));
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(false));

    verifyResults(
        manager, decoder, samplingConfigs, inputLengths, batchSize, maxBeamWidth, maxSeqLength, 1, tokenId, padId);

    // run decoder for 1 step
    decoder.forward(outputs, inputs);
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(true));
    EXPECT_THAT(decoder.getNbSteps(), ::testing::Each(maxNewTokens));

    verifyResults(
        manager, decoder, samplingConfigs, inputLengths, batchSize, maxBeamWidth, maxSeqLength, 2, tokenId, padId);

    EXPECT_NO_THROW(decoder.forward(outputs, inputs));
    EXPECT_THAT(decoder.getNbSteps(), ::testing::Each(maxNewTokens));

    decoder.newRequest(0, decoder_batch::Request{inputIds[0], maxNewTokens}, samplingConfigs[0]);
    EXPECT_FALSE(decoder.getFinished()[0]);
    EXPECT_EQ(decoder.getNbSteps()[0], 0);
}

void testDecoderWavefront(
    nvinfer1::DataType const dtype, std::vector<SamplingConfig> const& samplingConfigs, int maxBeamWidth)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    SizeType constexpr tensorParallelism{1};
    SizeType constexpr pipelineParallelism{1};
    SizeType constexpr localRank{0};
    WorldConfig constexpr worldConfig{tensorParallelism, pipelineParallelism, localRank};

    SizeType constexpr vocabSize{51200};
    SizeType constexpr nbLayers{2};
    SizeType constexpr nbHeads{16};
    SizeType constexpr hiddenSize{1024};
    GptModelConfig modelConfig{vocabSize, nbLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    // create decoder
    int constexpr endId{50257};
    int constexpr padId{50257};

    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = GptDecoderBatch(vocabSize, vocabSizePadded, streamPtr);

    // setup decoder
    auto const batchSize = static_cast<SizeType>(samplingConfigs.size());
    SizeType constexpr maxInputLength{8};
    SizeType constexpr maxNewTokens{8};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;

    decoder.setup(batchSize, maxBeamWidth, maxSeqLength, modelConfig.getDataType());

    std::vector<SizeType> const inputLengths{4, 5, 6, 7};
    std::vector<SizeType> tiledInputLengths;
    for (int batch_id = 0; batch_id < inputLengths.size(); batch_id++)
    {
        for (int beam_id = 0; beam_id < maxBeamWidth; beam_id++)
        {
            tiledInputLengths.push_back(inputLengths.at(batch_id));
        }
    }

    // set up inputs
    auto logits = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, vocabSizePadded}), modelConfig.getDataType()));
    manager.setZero(*logits);

    decoder_batch::Input inputs{logits};
    if (maxBeamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    decoder_batch::Output outputs{};

    if (maxBeamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, maxBeamWidth, maxSeqLength}), TRTDataType<SizeType>::value));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;
    }
    auto sequenceLengths
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize * maxBeamWidth}), TRTDataType<SizeType>::value));
    manager.copy(tiledInputLengths.data(), *sequenceLengths);
    outputs.sequenceLengths = sequenceLengths;

    auto const& nbSteps = decoder.getNbSteps();
    EXPECT_EQ(nbSteps.size(), batchSize);
    std::vector<SizeType> expectedSteps(batchSize, 0);

    auto const& finished = decoder.getFinished();
    EXPECT_EQ(finished.size(), batchSize);
    std::vector<bool> expectedFinished(batchSize, true);

    auto constexpr tokenId = 1;
    std::vector<decoder_batch::Input::TensorPtr> inputIds;
    for (auto b = 0; b < batchSize; ++b)
    {
        auto shape = ITensor::makeShape({inputLengths[b]});
        auto input = std::shared_ptr(manager.gpu(shape, TRTDataType<SizeType>::value));
        kernels::invokeFill(*input, tokenId, *streamPtr);
        inputIds.emplace_back(input);
        decoder.newRequest(b, decoder_batch::Request{inputIds[b], maxNewTokens, endId, padId}, samplingConfigs[b]);

        decoder.forward(outputs, inputs);

        for (auto i = 0; i < inputIds.size(); ++i)
        {
            expectedSteps[i] = std::min(expectedSteps[i] + 1, maxNewTokens);
            expectedFinished[i] = expectedSteps[i] == maxNewTokens;
        }

        EXPECT_THAT(decoder.getNbSteps(), ::testing::ElementsAreArray(expectedSteps));
        EXPECT_THAT(decoder.getFinished(), ::testing::ElementsAreArray(expectedFinished));
    }

    while (!decoder.getFinished().back())
    {
        decoder.forward(outputs, inputs);
    }
    EXPECT_THAT(decoder.getFinished(), ::testing::Each(true));
    EXPECT_THAT(decoder.getNbSteps(), ::testing::Each(maxNewTokens));

    verifyResults(manager, decoder, samplingConfigs, inputLengths, batchSize, maxBeamWidth, maxSeqLength, maxNewTokens,
        tokenId, padId);
}

} // namespace

struct BeamConfig
{
    SizeType maxBeamWidth;
    std::vector<SizeType> beamWidths;
};

class ParamTest : public ::testing::TestWithParam<std::tuple<nvinfer1::DataType, BeamConfig>>
{
};

TEST_P(ParamTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    BeamConfig const beamConfig{std::get<1>(GetParam())};
    std::vector<SamplingConfig> samplingConfigs;
    for (auto const beamWidth : beamConfig.beamWidths)
    {
        samplingConfigs.emplace_back(beamWidth);
    }

    testDecoder(dtype, samplingConfigs, beamConfig.maxBeamWidth);
}

INSTANTIATE_TEST_SUITE_P(GptDecoderTest, ParamTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF),
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {1, 1}},
            BeamConfig{4, {3, 3, 3}}, BeamConfig{4, {1, 2, 3, 4}})),
    [](const testing::TestParamInfo<ParamTest::ParamType>& info)
    {
        std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
        BeamConfig const beamConfig = std::get<1>(info.param);
        name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
        for (auto const beamWdith : beamConfig.beamWidths)
        {
            name.append("Bw" + std::to_string(beamWdith));
        }
        return name;
    });

class ParamWavefrontTest : public ::testing::TestWithParam<std::tuple<nvinfer1::DataType, BeamConfig>>
{
};

TEST_P(ParamWavefrontTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    BeamConfig const beamConfig{std::get<1>(GetParam())};
    std::vector<SamplingConfig> samplingConfigs;
    for (auto const beamWidth : beamConfig.beamWidths)
    {
        samplingConfigs.emplace_back(beamWidth);
    }

    testDecoderWavefront(dtype, samplingConfigs, beamConfig.maxBeamWidth);
}

INSTANTIATE_TEST_SUITE_P(GptDecoderTest, ParamWavefrontTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF),
        testing::Values(BeamConfig{1, {1, 1, 1}}, BeamConfig{3, {3, 3, 3, 3}}, BeamConfig{4, {1, 1}},
            BeamConfig{4, {3, 3, 3}}, BeamConfig{4, {1, 2, 3, 4}})),
    [](const testing::TestParamInfo<ParamTest::ParamType>& info)
    {
        std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
        BeamConfig const beamConfig = std::get<1>(info.param);
        name.append("MaxBeamWidth" + std::to_string(beamConfig.maxBeamWidth));
        for (auto const beamWdith : beamConfig.beamWidths)
        {
            name.append("Bw" + std::to_string(beamWdith));
        }
        return name;
    });
