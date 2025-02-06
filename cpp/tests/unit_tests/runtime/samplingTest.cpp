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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tl = tensorrt_llm::layers;
namespace tle = tensorrt_llm::executor;

class SamplingTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP() << "No GPUs found";

        mLogger = std::make_shared<TllmLogger>();
    }

    void TearDown() override {}

    int mDeviceCount;
    std::shared_ptr<nvinfer1::ILogger> mLogger;
};

std::shared_ptr<tl::BaseDecodingOutputs> dynamicDecodeTest(std::shared_ptr<BufferManager> manager, size_t vocabSize,
    size_t vocabSizePadded, size_t batchSize, size_t beamWidth, int step, int ite, int maxInputLength,
    size_t maxSeqLength, int localBatchSize, std::vector<int>& cpuOutputIds, std::vector<float> cpuLogits,
    int noRepeatNgramSizeValue = 0)
{
    constexpr int endId = 1;
    auto signedBatchSize = static_cast<int32_t>(batchSize);
    auto signedBeamWidth = static_cast<int32_t>(beamWidth);
    auto signedMaxSeqLength = static_cast<int32_t>(maxSeqLength);
    cudaDeviceProp prop{};
    tc::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    std::vector<int> cpuEndIds(batchSize, endId);
    std::vector<int> cpuSequenceLengths(batchSize, maxInputLength);
    std::vector<int> cpuNoRepeatNgramSize(batchSize, noRepeatNgramSizeValue);

    tk::FinishedState::UnderlyingType* gpuFinished = nullptr;

    ITensor::SharedPtr gpuEndIds = manager->gpu(ITensor::makeShape({signedBatchSize}), nvinfer1::DataType::kINT32);
    manager->copy(cpuEndIds.data(), *gpuEndIds, MemoryType::kCPU);
    ITensor::SharedPtr gpuOutputIds = manager->gpu(
        ITensor::makeShape({signedBatchSize, signedBeamWidth, signedMaxSeqLength}), nvinfer1::DataType::kINT32);
    manager->copy(cpuOutputIds.data(), *gpuOutputIds, MemoryType::kCPU);

    auto const decodingMode = beamWidth == 1 ? tle::DecodingMode::TopKTopP() : tle::DecodingMode::BeamSearch();
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(batchSize, beamWidth, vocabSize, vocabSizePadded);
    auto ddLayer = tl::DynamicDecodeLayer<float>(decodingMode, decodingDomain, manager);

    auto setupParams = std::make_shared<tl::DynamicDecodeSetupParams>();
    setupParams->banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    setupParams->banWordsParams->noRepeatNgramSize = cpuNoRepeatNgramSize;

    setupParams->penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    setupParams->decodingParams = std::make_shared<tl::SamplingSetupParams>();

    auto batchSlots = getDefaultBatchSlots(batchSize);
    auto workspace = std::make_shared<tensorrt_llm::runtime::DecodingLayerWorkspace>(
        manager, decodingDomain, TRTDataType<float>::value, ddLayer.getWorkspaceSize());
    ddLayer.setup(batchSize, beamWidth, batchSlots, setupParams, workspace);

    auto forwardParams = std::make_shared<tl::SamplingInputs>(gpuEndIds, batchSlots, step, ite, localBatchSize);
    auto logitsShape
        = ITensor::makeShape({signedBatchSize, static_cast<int64_t>(beamWidth), static_cast<int64_t>(vocabSizePadded)});
    ITensor::SharedPtr inputLogits = manager->gpu(logitsShape, nvinfer1::DataType::kFLOAT);
    forwardParams->logits = inputLogits;
    manager->copy(cpuLogits.data(), *inputLogits, MemoryType::kCPU);

    forwardParams->banWordsInputs = std::make_shared<tl::BanWordsDecodingInputs>(localBatchSize);

    forwardParams->stopCriteriaInputs = std::make_shared<tl::StopCriteriaDecodingInputs>(localBatchSize);

    auto outputParams = std::make_shared<tl::BaseDecodingOutputs>(gpuOutputIds);
    outputParams->sequenceLength = manager->gpu(ITensor::makeShape({signedBatchSize}), nvinfer1::DataType::kINT32);
    manager->copy(cpuSequenceLengths.data(), *outputParams->sequenceLength.value(), MemoryType::kCPU);
    outputParams->newTokens
        = manager->gpu(ITensor::makeShape({signedBatchSize, signedBeamWidth}), nvinfer1::DataType::kINT32);
    outputParams->finished = manager->gpu(
        ITensor::makeShape({signedBatchSize, signedBeamWidth}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

    ddLayer.forwardAsync(outputParams, forwardParams, workspace);

    return outputParams;
}

TEST_F(SamplingTest, SamplingWithNoRepeatNGramSize)
{
    auto streamPtr = std::make_shared<CudaStream>();
    auto manager = std::make_shared<BufferManager>(streamPtr);

    constexpr size_t vocabSize{200};
    constexpr size_t vocabSizePadded{256};
    constexpr size_t batchSize{1};
    constexpr size_t beamWidth{1};
    constexpr int step{8};
    constexpr int ite{0};
    constexpr int maxInputLength{8};
    constexpr int maxSeqLength{9};
    constexpr int localBatchSize{batchSize};
    constexpr int noRepeatNgramSize{3};

    std::vector<int> cpuOutputIds(batchSize * beamWidth * maxSeqLength);
    int ids[maxInputLength] = {10, 11, 12, 40, 41, 42, 40, 41};
    for (int i = 0; i < maxInputLength; i++)
    {
        cpuOutputIds[i] = ids[i];
    }

    std::vector<float> cpuLogits(batchSize * beamWidth * vocabSizePadded, 1.0);
    // We're setting 42 as favorite for greeding sampling but it should be banned because of no_repeat_ngram_size
    // 43 is the expected fallback
    cpuLogits[42] = 10.0;
    cpuLogits[43] = 5.0;

    auto outputParams = dynamicDecodeTest(manager, vocabSize, vocabSizePadded, batchSize, beamWidth, step, ite,
        maxInputLength, maxSeqLength, localBatchSize, cpuOutputIds, cpuLogits, noRepeatNgramSize);

    manager->copy(*outputParams->outputIds, cpuOutputIds.data(), MemoryType::kCPU);
    EXPECT_EQ(cpuOutputIds[maxSeqLength - 1], 43);
}
