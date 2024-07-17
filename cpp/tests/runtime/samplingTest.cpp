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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

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
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

std::shared_ptr<tl::BaseDecodingOutputs> dynamicDecodeTest(BufferManager& manager,
    std::shared_ptr<tc::CudaAllocator> allocator, size_t vocabSize, size_t vocabSizePadded, size_t batchSize,
    size_t beamWidth, int step, int ite, int maxInputLength, size_t maxSeqLength, size_t sinkTokenLength,
    int localBatchSize, std::vector<int>& cpuOutputIds, std::vector<float> cpuLogits, int noRepeatNgramSizeValue = 0)
{
    constexpr int endId = 1;
    cudaDeviceProp prop;
    tc::check_cuda_error(cudaGetDeviceProperties(&prop, 0));

    std::vector<int> cpuEndIds(batchSize, endId);
    std::vector<int> cpuSequenceLengths(batchSize, maxInputLength);
    std::vector<int> cpuNoRepeatNgramSize(batchSize, noRepeatNgramSizeValue);

    float* gpuLogits = nullptr;
    int* gpuEndIds = nullptr;
    int* gpuOutputIds = nullptr;
    int* gpuSequenceLengths = nullptr;
    int* gpuNewTokens = nullptr;
    tk::FinishedState::UnderlyingType* gpuFinished = nullptr;

    gpuLogits = allocator->reMalloc(gpuLogits, batchSize * beamWidth * vocabSizePadded * sizeof(float));
    gpuEndIds = allocator->reMalloc(gpuEndIds, batchSize * sizeof(int));
    gpuOutputIds = allocator->reMalloc(gpuOutputIds, batchSize * beamWidth * maxSeqLength * sizeof(int));
    gpuSequenceLengths = allocator->reMalloc(gpuSequenceLengths, batchSize * sizeof(int));
    gpuNewTokens = allocator->reMalloc(gpuNewTokens, batchSize * beamWidth * sizeof(int));
    gpuFinished = allocator->reMalloc(gpuFinished, batchSize * beamWidth * sizeof(tk::FinishedState::UnderlyingType));

    cudaMemcpy(gpuLogits, cpuLogits.data(), cpuLogits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuEndIds, cpuEndIds.data(), cpuEndIds.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        gpuSequenceLengths, cpuSequenceLengths.data(), cpuSequenceLengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuOutputIds, cpuOutputIds.data(), cpuOutputIds.size() * sizeof(int), cudaMemcpyHostToDevice);

    tc::Tensor logits{tc::MEMORY_GPU, tc::TYPE_FP32, {batchSize, beamWidth, vocabSizePadded}, gpuLogits};
    tc::Tensor endIds{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuEndIds};
    tc::Tensor outputIds{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize, beamWidth, maxSeqLength}, gpuOutputIds};
    tc::Tensor sequenceLengths{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuSequenceLengths};
    tc::Tensor newTokens{tc::MEMORY_GPU, tc::TYPE_INT32, {batchSize}, gpuNewTokens};
    tc::Tensor finished{tc::MEMORY_GPU, tc::TYPE_INT8, {batchSize, beamWidth}, gpuFinished};

    auto const decodingMode = beamWidth == 1 ? tle::DecodingMode::TopKTopP() : tle::DecodingMode::BeamSearch();
    auto const decodingDomain = tensorrt_llm::layers::DecoderDomain(batchSize, beamWidth, vocabSize, vocabSizePadded);
    auto ddLayer = tl::DynamicDecodeLayer<float>(decodingMode, decodingDomain, manager.getStream().get(), allocator);

    auto setupParams = std::make_shared<tl::DynamicDecodeSetupParams>();
    setupParams->banWordsParams = std::make_shared<tl::BanWordsSetupParams>();
    setupParams->banWordsParams->noRepeatNgramSize = cpuNoRepeatNgramSize;

    setupParams->penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    setupParams->decodingParams = std::make_shared<tl::SamplingSetupParams>();

    ddLayer.setup(batchSize, beamWidth, nullptr, setupParams);

    auto forwardParams = std::make_shared<tl::SamplingInputs>(endIds, step, ite, localBatchSize);
    forwardParams->logits = logits;

    forwardParams->banWordsInputs = std::make_shared<tl::BanWordsDecodingInputs>(localBatchSize);

    forwardParams->stopCriteriaInputs = std::make_shared<tl::StopCriteriaDecodingInputs>(localBatchSize);

    auto outputParams = std::make_shared<tl::BaseDecodingOutputs>(outputIds);
    outputParams->sequenceLength = sequenceLengths;
    outputParams->newTokens = newTokens;
    outputParams->finished = finished;

    ddLayer.forwardAsync(outputParams, forwardParams);

    return outputParams;
}

TEST_F(SamplingTest, SamplingWithNoRepeatNGramSize)
{
    auto streamPtr = std::make_shared<CudaStream>();
    BufferManager manager(streamPtr);
    auto allocator = std::make_shared<tc::CudaAllocator>(manager);

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
    constexpr int sinkTokenLength{0};

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

    auto outputParams = dynamicDecodeTest(manager, allocator, vocabSize, vocabSizePadded, batchSize, beamWidth, step,
        ite, maxInputLength, maxSeqLength, sinkTokenLength, localBatchSize, cpuOutputIds, cpuLogits, noRepeatNgramSize);

    cudaMemcpy(cpuOutputIds.data(), outputParams->outputIds.getPtr<int>(), cpuOutputIds.size() * sizeof(int),
        cudaMemcpyDeviceToHost);

    EXPECT_EQ(cpuOutputIds[maxSeqLength - 1], 43);
}
