/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/trtGptModelInflightBatching.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/testing/modelSpec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

using ::testing::ElementsAre;
using namespace tensorrt_llm::runtime;
namespace fs = std::filesystem;
using tensorrt_llm::testing::ModelSpec;
using tensorrt_llm::testing::KVCacheType;

using TensorPtr = ITensor::SharedPtr;

namespace
{
auto const TEST_RESOURCE_PATH = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const ENGINE_PATH = TEST_RESOURCE_PATH / "models/rt_engine";
auto const GPT_MODEL_PATH = ENGINE_PATH / "gpt2";
auto const LLAMA_MODEL_PATH = ENGINE_PATH / "Llama-3.2-1B";
} // namespace

namespace tensorrt_llm::batch_manager
{

class TrtGptModelTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    TrtGptModelTest(std::filesystem::path const& modelPath)
        : mModelConfig(1, 1, 1, 0, 1, 1, nvinfer1::DataType::kFLOAT)
        , mModelPath(modelPath)
    {
    }

    TrtGptModelTest()
        : TrtGptModelTest(GPT_MODEL_PATH / GetModelSpec().getModelPath() / "tp1-pp1-cp1-gpu")
    {
    }

    static ModelSpec& GetModelSpec()
    {
        static ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED);
        return modelSpec;
    }

    void SetUp() override
    {
        std::filesystem::path trtEnginePath = mModelPath;

        mBeamWidth = 1;

        mLogger = std::make_shared<TllmLogger>();

        initTrtLlmPlugins(mLogger.get());

        auto const json = GptJsonConfig::parse(trtEnginePath / "config.json");
        mModelConfig = json.getModelConfig();
        mMaxNumRequests = mModelConfig.getMaxBatchSize();
        mMaxSeqLen = mModelConfig.getMaxSequenceLen();
        mWorldConfig = WorldConfig::mpi();
        mVocabSizePadded = mModelConfig.getVocabSizePadded(mWorldConfig.getSize());

        auto const enginePath = trtEnginePath / json.engineFilename(mWorldConfig);
        auto const dtype = mModelConfig.getDataType();

        mRawEngine.reset(new RawEngine(enginePath));

        mSamplingConfig.temperature = std::vector{1.0f};
        mSamplingConfig.minLength = std::vector{1};
        mSamplingConfig.randomSeed = std::vector{static_cast<uint64_t>(42ul)};
        mSamplingConfig.topK = std::vector{0};
        mSamplingConfig.topP = std::vector{0.0f};
        mSamplingConfig.noRepeatNgramSize = std::vector{1 << 30};

        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    void TearDown() override {}

    void forwardRequestsToCompletion(
        std::shared_ptr<TrtGptModel> const& trtGptModel, RequestList& requestList, SizeType32 maxNumIterations)
    {
        SizeType32 numFinished = 0;
        SizeType32 numIterations = 0;
        while (numFinished < requestList.size() && numIterations < maxNumIterations)
        {
            if (numIterations > maxNumIterations)
            {
                FAIL() << "Iterations never finished";
            }
            trtGptModel->forwardAsync(requestList);
            trtGptModel->forwardSync();
            numFinished = 0;
            for (auto& request : requestList)
            {
                if (request->isGenerationCompleteState())
                {
                    ++numFinished;
                }
            }
            ++numIterations;
        }
    }

    int32_t mMaxNumRequests;
    int32_t mMaxSeqLen;
    int32_t mBeamWidth;
    int32_t mVocabSizePadded;
    SamplingConfig mSamplingConfig;
    std::string mDataPath;
    std::shared_ptr<nvinfer1::ILogger> mLogger;
    ModelConfig mModelConfig;
    WorldConfig mWorldConfig;
    std::unique_ptr<RawEngine> mRawEngine;
    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
    std::filesystem::path mModelPath;
};

class TrtGptModelLoraTest : public TrtGptModelTest
{
protected:
    TrtGptModelLoraTest()
        : TrtGptModelTest(GPT_MODEL_PATH / GetModelSpec().getModelPath() / "tp1-pp1-cp1-gpu")
    {
    }

    static ModelSpec& GetModelSpec()
    {
        static ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED).useLoraPlugin();
        return modelSpec;
    }
};

TEST_F(TrtGptModelTest, Forward)
{
    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);

    RequestList requestList{llmRequest};

    auto& manager = *mManager;
    std::vector<int32_t> newTokensHost(mMaxNumRequests, 5);
    TensorPtr const fakeNewTokens
        = manager.copyFrom(newTokensHost, ITensor::makeShape({mMaxNumRequests, 1}), MemoryType::kGPU);

    std::vector<bool> finished(mMaxNumRequests, false);

    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kMAX_UTILIZATION};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    // Generate one token for the requests in request_table
    // We need to sync with decoder
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 5);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(1, 2, 3, 4, 2));
}

TEST_F(TrtGptModelLoraTest, Forward)
{
    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);

    RequestList requestList{llmRequest};

    auto& manager = *mManager;
    std::vector<int32_t> newTokensHost(mMaxNumRequests, 5);
    TensorPtr const fakeNewTokens
        = manager.copyFrom(newTokensHost, ITensor::makeShape({mMaxNumRequests, 1}), MemoryType::kGPU);

    std::vector<bool> finished(mMaxNumRequests, false);

    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kMAX_UTILIZATION};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    // Generate one token for the requests in request_table
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 5);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(1, 2, 3, 4, 2));
}

TEST_F(TrtGptModelTest, ForwardMaxNewTokens)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(256);
    std::iota(std::begin(*tokens), std::end(*tokens), 1);
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);

    int correlationId2 = 2;
    auto maxNewTokens2 = 8;
    auto llmRequest2 = std::make_shared<LlmRequest>(correlationId2, maxNewTokens2, tokens, inSamplingConfig, false);

    RequestList requestList{llmRequest, llmRequest2};

    auto& manager = *mManager;
    std::vector<bool> finished(mMaxNumRequests, false);

    // Generate one token for the requests in request_table
    // We call forward twice because the first call doesn't sync with decoder
    SizeType32 maxNumIterations = 13;
    forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

    for (auto& request : requestList)
    {
        auto outputTokens = request->getTokens(0);
        if (request->mRequestId == correlationId)
        {
            EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
        }
        if (request->mRequestId == correlationId2)
        {
            EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens2);
        }
    }
}

TEST_F(TrtGptModelTest, MaxNumTokensInChunked)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.enableChunkedContext = true;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto modelConfig = mModelConfig;
    mModelConfig.setMaxNumTokens(200);

    auto trtGptModelIfb = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);
    std::vector<std::shared_ptr<TrtGptModel>> trtGptModels{trtGptModelIfb};

    for (auto trtGptModel : trtGptModels)
    {
        SamplingConfig inSamplingConfig;
        inSamplingConfig.temperature = std::vector{2.0f};
        int correlationId = 0;
        auto maxNewTokens = 4;
        auto tokens = std::make_shared<std::vector<int32_t>>(256);
        std::iota(std::begin(*tokens), std::end(*tokens), 1);
        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);

        int correlationId2 = 2;
        auto maxNewTokens2 = 8;
        auto llmRequest2 = std::make_shared<LlmRequest>(correlationId2, maxNewTokens2, tokens, inSamplingConfig, false);

        RequestList requestList{llmRequest, llmRequest2};

        auto& manager = *mManager;
        std::vector<bool> finished(mMaxNumRequests, false);

        // Generate one token for the requests in request_table
        // We call forward twice because the first call doesn't sync with decoder
        SizeType32 maxNumIterations = 13;
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

        for (auto& request : requestList)
        {
            auto outputTokens = request->getTokens(0);
            if (request->mRequestId == correlationId)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
            }
            if (request->mRequestId == correlationId2)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens2);
            }
        }
    }
}

TEST_F(TrtGptModelTest, ForwardEndId)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto endId = 107;
    auto tokens = std::make_shared<std::vector<int32_t>>(256);
    std::iota(std::begin(*tokens), std::end(*tokens), 1);
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false, endId);

    int correlationId2 = 2;
    auto maxNewTokens2 = 8;
    auto llmRequest2
        = std::make_shared<LlmRequest>(correlationId2, maxNewTokens2, tokens, inSamplingConfig, false, endId);

    RequestList requestList{llmRequest, llmRequest2};

    auto& manager = *mManager;
    std::vector<bool> finished(mMaxNumRequests, false);

    // Generate one token for the requests in request_table
    // We call forward twice because the first call doesn't sync with decoder
    SizeType32 maxNumIterations = 13;
    forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

    for (auto& request : requestList)
    {
        auto outputTokens = request->getTokens(0);
        // endId token is generated at 2nd iteration, so expect 1 output token
        if (request->mRequestId == correlationId)
        {
            EXPECT_EQ(outputTokens.size(), tokens->size() + 1);
        }
        if (request->mRequestId == correlationId2)
        {
            EXPECT_EQ(outputTokens.size(), tokens->size() + 1);
        }
    }
}

TEST_F(TrtGptModelTest, ForwardNoEoS)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kSTATIC_BATCH};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    SamplingConfig inSamplingConfig;
    inSamplingConfig.topP = {0.9};
    inSamplingConfig.temperature = {0.6};
    inSamplingConfig.minLength = {5};

    auto tokens = std::make_shared<std::vector<int32_t>>(256);
    std::iota(std::begin(*tokens), std::end(*tokens), 1);

    RequestList requestList;
    for (auto requestIdx = 0; requestIdx < mMaxNumRequests; requestIdx++)
    {
        auto llmRequest = std::make_shared<LlmRequest>(requestIdx, 8, tokens, inSamplingConfig, false, -1);
        requestList.push_back(llmRequest);
    }

    auto& manager = *mManager;
    std::vector<bool> finished(mMaxNumRequests, false);

    // Generate one token for the requests in request_table
    // We call forward twice because the first call doesn't sync with decoder
    SizeType32 maxNumIterations = 13;
    forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
}

TEST_F(TrtGptModelTest, ForwardFinished)
{
    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 2;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{10, 9, 8, 7, 6});
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);

    RequestList requestList{llmRequest};

    int mForwardCount = 0;

    auto& manager = *mManager;
    std::vector<int32_t> newTokensHost(mMaxNumRequests, 5);
    TensorPtr const fakeNewTokens
        = manager.copyFrom(newTokensHost, ITensor::makeShape({mMaxNumRequests, 1}), MemoryType::kGPU);

    std::vector<int32_t> newTokensHost2(mMaxNumRequests, 4);
    TensorPtr const fakeNewTokens2
        = manager.copyFrom(newTokensHost2, ITensor::makeShape({mMaxNumRequests, 1}), MemoryType::kGPU);

    // Below are only used if beam > 1
    // So we are just returning tensors with the correct shape, content is not important
    std::vector<int32_t> outputIdsHost(mMaxNumRequests * (5 + 2), 5);
    TensorPtr const fakeOutputIds
        = manager.copyFrom(outputIdsHost, ITensor::makeShape({mMaxNumRequests, 1, 5 + 2}), MemoryType::kGPU);

    std::vector<bool> finishedFalse(mMaxNumRequests, false);
    std::vector<bool> finishedTrue(mMaxNumRequests, true);

    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kMAX_UTILIZATION};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    // Generate one token for the requests in request_table
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 6);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10));

    // Generate one more token
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 7);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 2);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6));
}

TEST_F(TrtGptModelTest, ForwardStopWords)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{10, 9, 8, 7, 6});
    std::optional<SizeType32> endId(std::nullopt);
    std::optional<SizeType32> padId(std::nullopt);
    std::optional<TensorPtr> embeddingBias(std::nullopt);
    std::optional<TensorPtr> badWordsList(std::nullopt);

    auto& manager = *mManager;
    // No stop words
    {
        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
        RequestList requestList{llmRequest};
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
    }
    // With stop words
    {
        TensorPtr stopWordsList = manager.cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);
        auto stopWordsPtr = bufferCast<int32_t>(*stopWordsList);
        // make 10, 6 10 the tokens for the stop word:
        stopWordsPtr[0] = 10;
        stopWordsPtr[1] = 6;
        stopWordsPtr[2] = 10;
        stopWordsPtr[3] = 3;
        stopWordsPtr[4] = -1;
        stopWordsPtr[5] = -1;

        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
            endId, padId, embeddingBias, badWordsList, stopWordsList);
        RequestList requestList{llmRequest};
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10));
    }

    // With stop words
    {
        TensorPtr stopWordsList = manager.cpu(ITensor::makeShape({1, 2, 1}), nvinfer1::DataType::kINT32);
        auto stopWordsPtr = bufferCast<int32_t>(*stopWordsList);
        // make 10 is the token for the stop word:
        stopWordsPtr[0] = 10;
        stopWordsPtr[1] = 1;

        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
            endId, padId, embeddingBias, badWordsList, stopWordsList);
        RequestList requestList{llmRequest};
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10));
    }

    // Multiple requests, each with different stop words
    {
        // Request w/o stop words
        auto llmRequest = std::make_shared<LlmRequest>(1, maxNewTokens, tokens, inSamplingConfig, false);

        TensorPtr stopWordsList2 = manager.cpu(ITensor::makeShape({1, 2, 1}), nvinfer1::DataType::kINT32);
        {
            auto stopWordsPtr = bufferCast<int32_t>(*stopWordsList2);
            stopWordsPtr[0] = 10;
            stopWordsPtr[1] = 1;
        }
        auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, tokens, inSamplingConfig, false, endId, padId,
            embeddingBias, badWordsList, stopWordsList2);

        TensorPtr stopWordsList3 = manager.cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);
        {
            auto stopWordsPtr = bufferCast<int32_t>(*stopWordsList3);
            stopWordsPtr[0] = 10;
            stopWordsPtr[1] = 6;
            stopWordsPtr[2] = 10;
            stopWordsPtr[3] = 3;
            stopWordsPtr[4] = -1;
            stopWordsPtr[5] = -1;
        }
        auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, tokens, inSamplingConfig, false, endId, padId,
            embeddingBias, badWordsList, stopWordsList3);

        RequestList requestList{llmRequest, llmRequest2, llmRequest3};

        SizeType32 maxNumIterations(5);
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

        for (auto& request : requestList)
        {
            auto outputTokens = request->getTokens(0);
            if (request->mRequestId == 1)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
            }
            if (request->mRequestId == 2)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + 1);
                EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10));
            }
            if (request->mRequestId == 3)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + 3);
                EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10));
            }
        }
    }
}

TEST_F(TrtGptModelTest, ForwardBadWords)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 4;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{10, 9, 8, 7, 6});
    std::optional<SizeType32> endId(std::nullopt);
    std::optional<SizeType32> padId(std::nullopt);
    std::optional<TensorPtr> embeddingBias(std::nullopt);
    std::optional<TensorPtr> stopWordsList(std::nullopt);

    auto& manager = *mManager;
    // No bad words
    {
        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
        RequestList requestList{llmRequest};

        SizeType32 maxNumIterations = 5;
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
    }
    // With bad words, multiple tokens
    {
        TensorPtr badWordsList = manager.cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);
        auto badWordsPtr = bufferCast<int32_t>(*badWordsList);
        // make 10, 6 10 the tokens for the bad word:
        badWordsPtr[0] = 10;
        badWordsPtr[1] = 6;
        badWordsPtr[2] = 10;
        badWordsPtr[3] = 3;
        badWordsPtr[4] = -1;
        badWordsPtr[5] = -1;

        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
            endId, padId, embeddingBias, badWordsList, stopWordsList);
        RequestList requestList{llmRequest};
        SizeType32 maxNumIterations = 5;
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        // Token at position 7 should be different than 10
        EXPECT_NE(requestList.front()->getTokens(0).at(7), 10);
    }

    // With bad words single token
    {
        TensorPtr badWordsList = manager.cpu(ITensor::makeShape({1, 2, 1}), nvinfer1::DataType::kINT32);
        auto badWordsPtr = bufferCast<int32_t>(*badWordsList);
        // make 10 is the token for the bad word:
        badWordsPtr[0] = 10;
        badWordsPtr[1] = 1;

        auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
            endId, padId, embeddingBias, badWordsList, stopWordsList);
        RequestList requestList{llmRequest};
        SizeType32 maxNumIterations = 5;
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
        EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
        EXPECT_NE(requestList.front()->getTokens(0).at(5), 10);
    }

    // Multiple requests, each with different bad words
    {
        // Request w/o bad words
        auto llmRequest = std::make_shared<LlmRequest>(1, maxNewTokens, tokens, inSamplingConfig, false);

        TensorPtr badWordsList2 = manager.cpu(ITensor::makeShape({1, 2, 1}), nvinfer1::DataType::kINT32);
        {
            auto badWordsPtr = bufferCast<int32_t>(*badWordsList2);
            badWordsPtr[0] = 10;
            badWordsPtr[1] = 1;
        }
        auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, tokens, inSamplingConfig, false, endId, padId,
            embeddingBias, badWordsList2, stopWordsList);

        TensorPtr badWordsList3 = manager.cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);
        {
            auto badWordsPtr = bufferCast<int32_t>(*badWordsList3);
            badWordsPtr[0] = 10;
            badWordsPtr[1] = 6;
            badWordsPtr[2] = 10;
            badWordsPtr[3] = 3;
            badWordsPtr[4] = -1;
            badWordsPtr[5] = -1;
        }
        auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, tokens, inSamplingConfig, false, endId, padId,
            embeddingBias, badWordsList3, stopWordsList);

        RequestList requestList{llmRequest, llmRequest2, llmRequest3};

        SizeType32 maxNumIterations(6);
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

        for (auto& request : requestList)
        {
            auto outputTokens = request->getTokens(0);
            if (request->mRequestId == 1)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
            }
            if (request->mRequestId == 2)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                EXPECT_NE(request->getTokens(0).at(5), 10);
            }
            if (request->mRequestId == 3)
            {
                EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                EXPECT_NE(request->getTokens(0).at(7), 10);
            }
        }
    }
}

TEST_F(TrtGptModelTest, ForwardEmbeddingBias)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.kvCacheConfig.maxTokens = 10000;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto trtGptModelIfb = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    std::vector<std::shared_ptr<TrtGptModel>> trtGptModels{trtGptModelIfb};

    for (auto& trtGptModel : trtGptModels)
    {
        SamplingConfig inSamplingConfig;
        inSamplingConfig.temperature = std::vector{2.0f};
        int correlationId = 0;
        auto maxNewTokens = 4;
        auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{10, 9, 8, 7, 6});
        std::optional<SizeType32> endId(std::nullopt);
        std::optional<SizeType32> padId(std::nullopt);
        std::optional<TensorPtr> badWordsList(std::nullopt);
        std::optional<TensorPtr> stopWordsList(std::nullopt);

        auto& manager = *mManager;
        // No bad words
        {
            auto llmRequest
                = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
            RequestList requestList{llmRequest};

            SizeType32 maxNumIterations = 5;
            forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
            EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
            EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
        }
        // With embedding bias
        {
            TensorPtr embeddingBias
                = manager.cpu(ITensor::makeShape({1, mVocabSizePadded}), nvinfer1::DataType::kFLOAT);
            auto embeddingBiasPtr = bufferCast<float>(*embeddingBias);
            for (SizeType32 vi = 0; vi < mVocabSizePadded; ++vi)
            {
                embeddingBiasPtr[vi] = 0.f;
            }
            // bias all words to the 10th token
            embeddingBiasPtr[10] = std::numeric_limits<float>::max();

            auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
                endId, padId, embeddingBias, badWordsList, stopWordsList);
            RequestList requestList{llmRequest};
            SizeType32 maxNumIterations = 5;
            forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
            EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
            // All tokens should become 10 after applying bias
            EXPECT_EQ(requestList.front()->getTokens(0).at(5), 10);
            EXPECT_EQ(requestList.front()->getTokens(0).at(6), 10);
            EXPECT_EQ(requestList.front()->getTokens(0).at(7), 10);
            EXPECT_EQ(requestList.front()->getTokens(0).at(8), 10);
        }

        // Multiple requests, each with different bias
        {
            // Request w/o bias
            auto llmRequest = std::make_shared<LlmRequest>(1, maxNewTokens, tokens, inSamplingConfig, false);

            TensorPtr embeddingBias1
                = manager.cpu(ITensor::makeShape({1, mVocabSizePadded}), nvinfer1::DataType::kFLOAT);
            auto embeddingBias1Ptr = bufferCast<float>(*embeddingBias1);
            for (SizeType32 vi = 0; vi < mVocabSizePadded; ++vi)
            {
                embeddingBias1Ptr[vi] = 0.f;
            }
            // bias all words to the 10th token
            embeddingBias1Ptr[10] = std::numeric_limits<float>::max();

            auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, tokens, inSamplingConfig, false, endId,
                padId, embeddingBias1, badWordsList, stopWordsList);

            TensorPtr embeddingBias2
                = manager.cpu(ITensor::makeShape({1, mVocabSizePadded}), nvinfer1::DataType::kFLOAT);
            auto embeddingBias2Ptr = bufferCast<float>(*embeddingBias2);
            for (SizeType32 vi = 0; vi < mVocabSizePadded; ++vi)
            {
                embeddingBias2Ptr[vi] = 0.f;
            }
            // bias all words to the 100th token
            embeddingBias2Ptr[100] = std::numeric_limits<float>::max();

            auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, tokens, inSamplingConfig, false, endId,
                padId, embeddingBias2, badWordsList, stopWordsList);

            RequestList requestList{llmRequest, llmRequest2, llmRequest3};

            SizeType32 maxNumIterations(6);
            forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

            for (auto& request : requestList)
            {
                auto outputTokens = request->getTokens(0);
                if (request->mRequestId == 1)
                {
                    EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                    EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 6, 10, 6));
                }
                if (request->mRequestId == 2)
                {
                    EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                    EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 10, 10, 10, 10));
                }
                if (request->mRequestId == 3)
                {
                    EXPECT_EQ(outputTokens.size(), tokens->size() + maxNewTokens);
                    EXPECT_THAT(request->getTokens(0), ElementsAre(10, 9, 8, 7, 6, 100, 100, 100, 100));
                }
            }
        }
    }
}

class TrtGptModelIfbHelper : public TrtGptModelInflightBatching
{
public:
    using TrtGptModelInflightBatching::TrtGptModelInflightBatching;

    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager const> getKVCacheManager() const
    {
        return TrtGptModelInflightBatching::getKVCacheManager();
    }
};

TEST_F(TrtGptModelTest, KVCacheReuseChunked)
{
    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.enableChunkedContext = true;
    optionalParams.kvCacheConfig.enableBlockReuse = true;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

    auto modelConfig = mModelConfig;
    mModelConfig.setMaxNumTokens(384);

    for (int const numBlocksExpectedReused : {1, 2})
    {
        for (int const maxNumIterations : {2, 6})
        {
            auto trtGptModelIfb = std::make_shared<TrtGptModelIfbHelper>(
                mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);
            auto const cacheManager = trtGptModelIfb->getKVCacheManager();
            auto const tokensPerBlock = cacheManager->getTokensPerBlock();
            constexpr int numPrefillBlocks = 2;

            SamplingConfig inSamplingConfig;
            inSamplingConfig.temperature = std::vector{2.0f};
            int correlationId = 0;
            constexpr int maxNewTokens = 4;

            auto tokens = std::make_shared<std::vector<int32_t>>(tokensPerBlock * numPrefillBlocks);
            std::iota(std::begin(*tokens), std::end(*tokens), 1);
            auto subTokens = std::make_shared<std::vector<int32_t>>(
                tokens->begin(), tokens->begin() + numBlocksExpectedReused * tokensPerBlock);
            // Add new token to "start" a new block.
            subTokens->push_back(0);
            {
                auto llmRequest
                    = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
                RequestList requests{llmRequest};
                forwardRequestsToCompletion(trtGptModelIfb, requests, maxNumIterations);
                EXPECT_EQ(llmRequest->isGenerationCompleteState(), maxNumIterations >= maxNewTokens);
                if (!llmRequest->isGenerationCompleteState())
                {
                    correlationId++;
                }
            }
            for (size_t i = 1; i <= 2; ++i)
            {
                auto llmRequest
                    = std::make_shared<LlmRequest>(correlationId, maxNewTokens, subTokens, inSamplingConfig, false);
                RequestList req{llmRequest};
                forwardRequestsToCompletion(trtGptModelIfb, req, 5);
                EXPECT_EQ(cacheManager->getBlockManager().getNumReusedBlocks(), i * numBlocksExpectedReused);
            }
        }
    }
}

TEST_F(TrtGptModelTest, PauseRequestStats)
{
    SamplingConfig inSamplingConfig;
    inSamplingConfig.temperature = std::vector{2.0f};
    int correlationId = 0;
    auto maxNewTokens = 3;
    auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
    auto llmRequest = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, executor::Request::kDefaultPriority, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, 1, std::nullopt,
        std::nullopt, true /* returnPerfMetrics */);

    RequestList requestList{llmRequest};

    TrtGptModelOptionalParams optionalParams;
    optionalParams.enableTrtOverlap = false;
    optionalParams.maxBeamWidth = mBeamWidth;
    optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kMAX_UTILIZATION};

    auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
        mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

    // Generate one token for the requests in request_table
    // We need to sync with decoder
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 5);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(1, 2, 3, 4, 2));

    auto perfMetrics = requestList.front()->getPerfMetrics();
    auto zero = executor::RequestPerfMetrics::TimePoint{};

    EXPECT_NE(perfMetrics.timingMetrics.arrivalTime, zero);
    EXPECT_NE(perfMetrics.timingMetrics.firstScheduledTime, zero);
    EXPECT_NE(perfMetrics.timingMetrics.firstTokenTime, zero);
    EXPECT_EQ(perfMetrics.timingMetrics.lastTokenTime, zero);
    EXPECT_EQ(perfMetrics.firstIter, 0);
    EXPECT_EQ(perfMetrics.iter, 0);
    EXPECT_EQ(perfMetrics.lastIter, std::nullopt);

    // Pause the request
    trtGptModel->terminateRequest(llmRequest, true);

    // Resume work
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    // Generate one more token
    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_IN_PROGRESS);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 6);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(1, 2, 3, 4, 2, 4));

    auto newPerfMetrics = requestList.front()->getPerfMetrics();
    EXPECT_EQ(newPerfMetrics.firstIter, 0);
    EXPECT_EQ(newPerfMetrics.iter, 1);
    EXPECT_EQ(newPerfMetrics.lastIter, std::nullopt);

    // Check that firstScheduledTime and firstTokenTime are the same
    EXPECT_EQ(perfMetrics.timingMetrics.firstScheduledTime, newPerfMetrics.timingMetrics.firstScheduledTime);
    EXPECT_EQ(perfMetrics.timingMetrics.firstTokenTime, newPerfMetrics.timingMetrics.firstTokenTime);

    // Pause the request
    trtGptModel->terminateRequest(llmRequest, true);

    // Resume work
    trtGptModel->forwardAsync(requestList);
    trtGptModel->forwardSync();

    // Generate last token
    EXPECT_EQ(requestList.size(), 1);
    EXPECT_EQ(requestList.front()->getState(), LlmRequestState::kGENERATION_COMPLETE);
    EXPECT_EQ(requestList.front()->getNumTokens(0), 7);
    EXPECT_EQ(requestList.front()->getMaxNumGeneratedTokens(), 1);
    EXPECT_THAT(requestList.front()->getTokens(0), ElementsAre(1, 2, 3, 4, 2, 4, 2));

    auto endPerfMetrics = requestList.front()->getPerfMetrics();
    EXPECT_EQ(endPerfMetrics.firstIter, 0);
    EXPECT_EQ(endPerfMetrics.iter, 2);
    EXPECT_EQ(endPerfMetrics.lastIter, 2);

    // Check that firstScheduledTime and firstTokenTime are the same
    EXPECT_EQ(perfMetrics.timingMetrics.firstScheduledTime, endPerfMetrics.timingMetrics.firstScheduledTime);
    EXPECT_EQ(perfMetrics.timingMetrics.firstTokenTime, endPerfMetrics.timingMetrics.firstTokenTime);
}

class TrtGptModelLogitsTest : public TrtGptModelTest
{
protected:
    TrtGptModelLogitsTest()
        : TrtGptModelTest(GPT_MODEL_PATH / GetModelSpec().getModelPath() / "tp1-pp1-cp1-gpu")
    {
    }

    static ModelSpec& GetModelSpec()
    {
        static ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().usePackedInput().setKVCacheType(KVCacheType::kPAGED).gatherLogits();
        return modelSpec;
    }
};

TEST_F(TrtGptModelLogitsTest, ReturnContextLogitsWithChunkedContext)
{
    // General config
    int correlationId = 0;
    auto maxNewTokens = 4;
    int const worldSize = 1;
    auto const vocabSizePadded = mModelConfig.getVocabSizePadded(worldSize);

    SamplingConfig inSamplingConfig;

    // Different prompt length
    for (int const promptLength : {10, 128, 200, 250, 256})
    {
        RequestList finishList;
        for (bool enableChunkedContext : {false, true})
        {
            auto modelConfig = mModelConfig;
            if (enableChunkedContext)
            {
                modelConfig.setMaxNumTokens(128);
            }

            TrtGptModelOptionalParams optionalParams;
            optionalParams.enableTrtOverlap = false;
            optionalParams.kvCacheConfig.enableBlockReuse = true;
            optionalParams.enableChunkedContext = enableChunkedContext;
            optionalParams.maxBeamWidth = mBeamWidth;
            optionalParams.schedulerConfig
                = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};

            auto trtGptModelIfb = std::make_shared<TrtGptModelIfbHelper>(
                mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

            // Prepare input tokens
            std::vector<int32_t> input_ids;
            for (int i = 1; i <= promptLength; i++)
            {
                input_ids.push_back(i);
            }
            auto tokens = std::make_shared<std::vector<int32_t>>(input_ids);

            auto llmRequest
                = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
            TensorPtr contextLogitsHost = BufferManager::cpu(
                ITensor::makeShape({llmRequest->mPromptLen, vocabSizePadded}), nvinfer1::DataType::kFLOAT);

            llmRequest->setContextLogitsHost(contextLogitsHost);
            llmRequest->setReturnContextLogits(true);

            RequestList requestList{llmRequest};
            forwardRequestsToCompletion(trtGptModelIfb, requestList, 6);

            finishList.push_back(llmRequest);
        }
        EXPECT_EQ(finishList.size(), 2);

        float const* const disableChunkedContextLogits
            = bufferCast<float>(*(finishList.front()->getContextLogitsHost()));
        float const* const enableChunkedContextLogits = bufferCast<float>(*(finishList.back()->getContextLogitsHost()));

        for (int i = 0; i < promptLength; i++)
        {
            for (int j = 0; j < vocabSizePadded; j++)
            {
                size_t idx = i * vocabSizePadded + j;
                EXPECT_EQ(disableChunkedContextLogits[idx], enableChunkedContextLogits[idx]);
            }
        }
        finishList.clear();
    }
}

class LlamaModelLADTest : public TrtGptModelTest
{
protected:
    LlamaModelLADTest()
        : TrtGptModelTest(LLAMA_MODEL_PATH / GetModelSpec().getModelPath() / "tp1-pp1-cp1-gpu")
    {
    }

    static ModelSpec& GetModelSpec()
    {
        static ModelSpec modelSpec = ModelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF}
                                         .useGptAttentionPlugin()
                                         .usePackedInput()
                                         .setKVCacheType(KVCacheType::kPAGED)
                                         .useLookaheadDecoding();
        return modelSpec;
    }
};

TEST_F(LlamaModelLADTest, SeamlessLookaheadDecoding)
{
    GTEST_SKIP() << "Will enable this test when we have a force LAD support.";
    SizeType32 requestId = 0;
    for (bool const initLADConfig : {true, false})
    {
        RequestList requestList{};
        for (SizeType32 i = 0; i < 8; ++i)
        {
            SamplingConfig inSamplingConfig;
            int correlationId = requestId;
            auto maxNewTokens = 8;
            auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
            auto llmRequest
                = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
            requestList.emplace_back(std::move(llmRequest));
            requestId += 1;
        }

        TrtGptModelOptionalParams optionalParams;
        optionalParams.enableChunkedContext = false;
        optionalParams.enableTrtOverlap = false;
        optionalParams.maxBeamWidth = 1;
        optionalParams.schedulerConfig = executor::SchedulerConfig{executor::CapacitySchedulerPolicy::kMAX_UTILIZATION};
        if (initLADConfig)
        {
            optionalParams.decodingConfig.setLookaheadDecodingConfig(executor::LookaheadDecodingConfig(5, 5, 5));
        }

        auto trtGptModel = std::make_shared<TrtGptModelInflightBatching>(
            mLogger, mModelConfig, mWorldConfig, *mRawEngine, true, optionalParams);

        // Generate tokens for the requests in request_table
        // We need to sync with decoder
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(trtGptModel->getSpeculativeDecodingMode().isLookaheadDecoding(), true);

        // Add new requests
        for (SizeType32 i = 0; i < 4; ++i)
        {
            SamplingConfig inSamplingConfig;
            int correlationId = requestId;
            auto maxNewTokens = 8;
            auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
            auto llmRequest
                = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
            requestList.emplace_back(std::move(llmRequest));
            requestId += 1;
        }
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(trtGptModel->getSpeculativeDecodingMode().isLookaheadDecoding(), false);

        // Complete all of the requests
        SizeType32 maxNumIterations = 8;
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);

        // Run new requests with lookahead
        requestList.clear();
        for (SizeType32 i = 0; i < 4; ++i)
        {
            SamplingConfig inSamplingConfig;
            int correlationId = requestId;
            auto maxNewTokens = 8;
            auto tokens = std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4});
            auto llmRequest
                = std::make_shared<LlmRequest>(correlationId, maxNewTokens, tokens, inSamplingConfig, false);
            requestList.emplace_back(std::move(llmRequest));
            requestId += 1;
        }
        trtGptModel->forwardAsync(requestList);
        trtGptModel->forwardSync();
        EXPECT_EQ(trtGptModel->getSpeculativeDecodingMode().isLookaheadDecoding(), true);
        forwardRequestsToCompletion(trtGptModel, requestList, maxNumIterations);
        requestList.clear();
    }
}

} // namespace tensorrt_llm::batch_manager
