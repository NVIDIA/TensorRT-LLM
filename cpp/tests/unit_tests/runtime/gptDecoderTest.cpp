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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

namespace
{

bool forwardAndSync(std::unique_ptr<IGptDecoder> const& decoder, DecodingOutput& output, DecodingInput const& input,
    std::shared_ptr<CudaStream> stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBatchSize = input.batchSize;

    BufferManager::ITensorPtr finishedSum;
    std::int32_t* finishedSumHost = nullptr;
    if (input.sequenceLimitLength && output.finishReasons)
    {
        finishedSumHost = bufferCast<std::int32_t>(*output.finishedSum);
        for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
        {
            finishedSumHost[bi] = 0;
        }
    }

    decoder->forwardAsync(output, input);

    if (finishedSumHost)
    {
        auto const numToFinish = output.finishReasons->getSize();
        TLLM_CUDA_CHECK(::cudaStreamSynchronize(stream->get()));

        SizeType32 finishedSum = 0;
        for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
        {
            finishedSum += finishedSumHost[bi];
        }
        return numToFinish == static_cast<std::size_t>(finishedSum);
    }
    else
    {
        return false;
    }
}

void testDecoder(nvinfer1::DataType const dtype, SamplingConfig const& samplingConfig)
{
    SizeType32 constexpr tensorParallelism{1};
    SizeType32 constexpr pipelineParallelism{1};
    SizeType32 constexpr contextParallelism{1};
    SizeType32 constexpr localRank{0};
    WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, contextParallelism, localRank};

    SizeType32 constexpr vocabSize{51200};
    SizeType32 constexpr nbLayers{2};
    SizeType32 constexpr nbRnnLayers{0};
    SizeType32 constexpr nbHeads{16};
    SizeType32 constexpr hiddenSize{1024};
    SizeType32 constexpr batchSize{4};
    ModelConfig modelConfig{vocabSize, nbLayers + nbRnnLayers, nbLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.useGptAttentionPlugin(false);

    SizeType32 constexpr maxInputLength{8};
    SizeType32 constexpr maxNewTokens{2};
    SizeType32 constexpr sinkTokenLength{0};
    auto constexpr maxSeqLength = maxInputLength + maxNewTokens;

    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);

    // setup decoder
    auto const beamWidth = samplingConfig.beamWidth;

    auto const decodingMode = beamWidth == 1 ? tle::DecodingMode::TopKTopP() : tle::DecodingMode::BeamSearch();

    // create decoder
    auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
    auto decoder = IGptDecoder::create(
        decodingMode, modelConfig.getDataType(), batchSize, beamWidth, vocabSize, vocabSizePadded, streamPtr);
    ASSERT_TRUE(static_cast<bool>(decoder));

    auto batchSlots = getDefaultBatchSlots(batchSize);
    decoder->setup(samplingConfig, batchSize, batchSlots);

    // set up inputs
    std::vector<std::shared_ptr<ITensor const>> logitsVec;
    for (auto i = 0; i < batchSize; ++i)
    {
        auto logits = manager.gpu(ITensor::makeShape({1, beamWidth, vocabSizePadded}), modelConfig.getDataType());
        manager.setZero(*logits);
        logitsVec.push_back(std::move(logits));
    }

    int constexpr endId{50257};
    std::vector<int> const endIdsVec(batchSize * beamWidth, endId);
    auto endIds
        = std::shared_ptr(manager.copyFrom(endIdsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU));

    DecodingInput inputs;
    inputs.maxLength = maxInputLength;
    inputs.maxAttentionWindow = maxSeqLength;
    inputs.sinkTokenLength = sinkTokenLength;
    inputs.batchSize = batchSize;
    inputs.logitsVec = logitsVec;
    inputs.endIds = endIds;
    inputs.batchSlots = batchSlots;

    std::vector<std::int32_t> inputLengthsVec(batchSize * beamWidth, 0);
    inputs.lengths = manager.copyFrom(inputLengthsVec, ITensor::makeShape({batchSize * beamWidth}), MemoryType::kGPU);

    std::vector<std::int32_t> sequenceLimitLengthsVec(batchSize, maxSeqLength);
    inputs.sequenceLimitLength
        = manager.copyFrom(sequenceLimitLengthsVec, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    if (beamWidth > 1)
    {
        auto srcCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*srcCacheIndirection);
        inputs.cacheIndirection = srcCacheIndirection;
    }

    // set up outputs
    auto outputIds = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
    manager.setZero(*outputIds);
    auto gatheredOutputIds = std::shared_ptr(
        manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
    manager.setZero(*gatheredOutputIds);
    DecodingOutput outputs{outputIds, gatheredOutputIds};
    auto newTokens
        = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kINT32));
    manager.setZero(*newTokens);
    outputs.newTokens = newTokens;

    std::vector<int> sequenceLengthsVec(batchSize * beamWidth, maxInputLength);
    outputs.lengths
        = manager.copyFrom(sequenceLengthsVec, ITensor::makeShape({batchSize, beamWidth}), MemoryType::kGPU);
    outputs.finishReasons = manager.gpu(ITensor::makeShape({batchSize, beamWidth}),
        TRTDataType<tensorrt_llm::kernels::FinishedState::UnderlyingType>::value);
    inputs.finishReasons = ITensor::view(outputs.finishReasons);
    manager.setZero(*outputs.finishReasons);
    outputs.finishedSum = BufferManager::pinnedPool(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    auto finishedSumHost = bufferCast<std::int32_t>(*outputs.finishedSum);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        finishedSumHost[bi] = -1;
    }

    if (beamWidth > 1)
    {
        auto tgtCacheIndirection = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*tgtCacheIndirection);
        outputs.cacheIndirection = tgtCacheIndirection;

        auto cumLogProbs
            = std::shared_ptr(manager.gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kFLOAT));
        manager.setZero(*cumLogProbs);
        outputs.cumLogProbs = cumLogProbs;

        auto parentIds = std::shared_ptr(
            manager.gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32));
        manager.setZero(*parentIds);
        outputs.parentIds = parentIds;
    }

    // run decoder
    EXPECT_FALSE(forwardAndSync(decoder, outputs, inputs, streamPtr));
    inputs.step += 1;

    {
        SizeType32 finishedSum = 0;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            finishedSum += finishedSumHost[bi];
        }
        EXPECT_EQ(finishedSum, 0);
    }

    // verify results
    auto outputsIdsHost = manager.copyFrom(*outputs.ids, MemoryType::kCPU);
    auto output = bufferCast<std::int32_t>(*outputsIdsHost);
    manager.getStream().synchronize();

    for (auto b = 0; b < batchSize; ++b)
    {
        for (auto bw = 0; bw < beamWidth; ++bw)
        {
            auto const result = (beamWidth == 1) ? 1023 : bw;

            bool anyMismatch = false;
            for (auto i = 0; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, bw, i, beamWidth, maxSeqLength);
                EXPECT_EQ(output[outputIndex], 0) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[outputIndex] != 0);
            }
            for (auto i = 0; i < maxNewTokens - 1; ++i)
            {
                auto const index = tc::flat_index3(b, bw, maxInputLength + i, beamWidth, maxSeqLength);
                EXPECT_EQ(output[index], result) << " b: " << b << " bw: " << bw << " i: " << i;
                anyMismatch |= (output[index] != result);
            }
            ASSERT_FALSE(anyMismatch);
        }
    }

    // run decoder again
    EXPECT_TRUE(forwardAndSync(decoder, outputs, inputs, streamPtr));
    {
        SizeType32 finishedSum = 0;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            finishedSum += finishedSumHost[bi];
        }
        EXPECT_EQ(finishedSum, outputs.finishReasons->getSize());
    }
}

} // namespace

class ParamTest : public ::testing::TestWithParam<std::tuple<nvinfer1::DataType, SizeType32>>
{
};

TEST_P(ParamTest, Test)
{
    nvinfer1::DataType const dtype{std::get<0>(GetParam())};
    SizeType32 const beamWidth{std::get<1>(GetParam())};
    SamplingConfig const samplingConfig{beamWidth};

    testDecoder(dtype, samplingConfig);
}

INSTANTIATE_TEST_SUITE_P(DecoderTest, ParamTest,
    testing::Combine(testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF), testing::Values(1, 3)),
    [](testing::TestParamInfo<ParamTest::ParamType> const& info)
    {
        std::string name{std::get<0>(info.param) == nvinfer1::DataType::kFLOAT ? "Float" : "Half"};
        auto const beamWidth = std::get<1>(info.param);
        name.append(beamWidth == 1 ? "Sampling" : "BeamWidth" + std::to_string(beamWidth));
        return name;
    });
