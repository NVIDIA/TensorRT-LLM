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

#include "executorTest.h"

#include "tensorrt_llm/batch_manager/trtGptModel.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/testing/modelSpec.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using ::testing::_;
using ::testing::Invoke;

namespace tr = tensorrt_llm::runtime;
namespace tb = tensorrt_llm::batch_manager;

using namespace tensorrt_llm::testing;
using namespace tensorrt_llm::executor;
using namespace std::chrono_literals;
using tensorrt_llm::testing::KVCacheType;

class MockedModel : public Model
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

public:
    MOCK_METHOD(void, forwardSync, (), ());
    MOCK_METHOD(void, forwardAsync, (RequestList const&), ());
    MOCK_METHOD(void, terminateRequest, (std::shared_ptr<tb::LlmRequest> const& llmRequest, bool pause), ());
    MOCK_METHOD(
        void, terminateRequestSync, (std::shared_ptr<tb::LlmRequest> const& llmRequest, FinishReason finishReason), ());
    MOCK_METHOD(SizeType32, getMaxNumSequences, (), (const));
    MOCK_METHOD(SizeType32, getMaxInputLen, (), (const));
    MOCK_METHOD(SizeType32, getHiddenSize, (), (const));
    MOCK_METHOD(SizeType32, getMaxSequenceLen, (), (const));
    MOCK_METHOD(SizeType32, getVocabSizePadded, (), (const));
    MOCK_METHOD(SizeType32, getMaxDraftLen, (), (const));
    MOCK_METHOD(SizeType32, getNumMicroBatches, (), (const));
    MOCK_METHOD(SizeType32, getOperatingBeamWidth, (), (const));
    MOCK_METHOD(nvinfer1::DataType, getLogitDataType, (), (const));
    MOCK_METHOD(nvinfer1::DataType, getTensorDataType, (std::string const&), (const));
    MOCK_METHOD(nvinfer1::Dims, getTensorShape, (std::string const&), (const));
    MOCK_METHOD(void, getCurrentIterationStats, (IterationStats&), (const));
    MOCK_METHOD(void, getCurrentRequestStats, (RequestStatsPerIteration&), (const));
    MOCK_METHOD(DebugTensorsPerIteration, getCurrentDebugTensors, (), (const));
    MOCK_METHOD(tr::WorldConfig const&, getWorldConfig, (), (const));
    MOCK_METHOD(tr::ModelConfig const&, getModelConfig, (), (const));
    MOCK_METHOD(tr::BufferManager const&, getBufferManager, (), (const));
    MOCK_METHOD(tr::BufferManager::CudaStreamPtr, getRuntimeStreamPtr, (), (const));
    MOCK_METHOD(IterationType, getIterCounter, (), (const, noexcept));
    MOCK_METHOD(bool, hasSpeculativeDecodingFastLogits, (), (const, noexcept));
    MOCK_METHOD(bool, getGatherGenerationLogits, (), (const));
    MOCK_METHOD(void, updatePeftCache, (LlmRequestPtr const& llmReqeust), ());
    MOCK_METHOD(void, setLogitsPostProcessorBatched, (std::optional<LogitsPostProcessorBatched>), ());
    MOCK_METHOD(void, setReplicateLogitsPostProcessor, (bool), ());
    MOCK_METHOD(bool, getReplicateLogitsPostProcessor, (), (const));
    MOCK_METHOD(bool, hasGuidedDecoder, (), (const, noexcept));
    MOCK_METHOD(void, resetIterationStats, (), ());
    MOCK_METHOD(
        std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager>, getKVCacheManager, (), ());
    MOCK_METHOD(std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager const>,
        getKVCacheManager, (), (const));
    MOCK_METHOD(SizeType32, getMaxCapacityBatchSize, (SizeType32, SizeType32), (const));
};

using ParamType = std::tuple<bool, bool, int>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const streaming = std::get<0>(info.param);
    auto const excludeInputFromOutput = std::get<1>(info.param);
    auto const beamWidth = std::get<2>(info.param);
    std::string name = "ExecutorTest";
    if (streaming)
    {
        name += "Streaming";
    }
    if (excludeInputFromOutput)
    {
        name += "ExclInput";
    }
    name.append("BW" + std::to_string(beamWidth));
    return name;
}

class ParamTest : public GptExecutorTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, MockedModel)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens(VecTokens(beamWidth, 1));
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));

    ExecutorConfig const executorConfig(beamWidth);
    auto executor = Executor(model, executorConfig);

    // Create the request
    constexpr SizeType32 maxNewTokens = 5;
    VecTokens const inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(request);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {
            auto const& result = response.getResult();
            done = result.isFinal;
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(callCount, maxNewTokens);
}

TEST_F(GptExecutorTest, MockedModelMaxQueueSize)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, terminateRequestSync(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);

    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    // Sleep to allow queue to fill up
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
    SizeType32 maxQueueSize = 6;
    ExecutorConfig executorConfig;
    executorConfig.setMaxQueueSize(maxQueueSize);

    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 const maxNewTokens = 5;
    VecTokens const inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens);

    // Enqueue as many requests as the queue can manage
    for (int i = 0; i < maxQueueSize; i++)
    {
        auto requestId = executor.enqueueRequest(request);
    }
    try
    {
        auto requestId = executor.enqueueRequest(request);

        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Maximum queue size of 6 has been reached, please try again later"));
    }

    // Wait for requests to get scheduled to free up space in queue
    std::this_thread::sleep_for(std::chrono::milliseconds(maxQueueSize * 200));
    auto requestId = executor.enqueueRequest(request);

    try
    {
        auto samplingConfig = SamplingConfig(1);
        samplingConfig.setNumReturnSequences(maxQueueSize);
        auto request = Request(inputTokens, maxNewTokens, false, samplingConfig);
        auto requestId = executor.enqueueRequest(request);
        FAIL() << "Expected TllmException";
    }
    catch (std::exception const& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Maximum queue size of 6 has been reached, please try again later"));
    }
}

TEST_F(GptExecutorTest, MockedModelReqStatsBug)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool streaming = false;
    bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    EXPECT_CALL(*model, updatePeftCache(_)).WillRepeatedly(Invoke([&]() { return; }));

    SizeType32 callCount = 0;
    RequestList currentReq;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                currentReq = requestList;
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                }
                callCount++;
            }));

    EXPECT_CALL(*model, forwardSync())
        .WillRepeatedly(Invoke(
            [&]()
            {
                for (auto const& llmReq : currentReq)
                {
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                }
                return;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

    SizeType32 beamWidth = 1;
    ExecutorConfig executorConfig(beamWidth);
    executorConfig.setRequestStatsMaxIterations(1000);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 maxNewTokens = 5;
    VecTokens inputTokens{1, 2, 3, 4};
    int numRequests = 10000;
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    auto done = std::atomic<bool>{false};
    auto statsThreadDone = false;
    // Spawn a thread that continuously get stats
    auto statsThread = std::thread(
        [&executor, &done, &statsThreadDone]()
        {
            while (!done)
            {
                auto reqStats = executor.getLatestRequestStats();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            statsThreadDone = true;
        });

    // Spawn a thread that enqueues the requests
    std::vector<IdType> requestIds;
    auto enqueueThread = std::thread(
        [&executor, &requestIds, &request, &done, numRequests]()
        {
            for (int i = 0; i < numRequests; ++i)
            {
                requestIds.push_back(executor.enqueueRequest(request));
            }
            done = true;
        });
    enqueueThread.join();
    ASSERT_EQ(requestIds.size(), numRequests);

    // Wait for stats thread to be done, fail otherwise
    int iter = 0;
    while (!statsThreadDone && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        iter++;
    }
    ASSERT_TRUE(statsThreadDone);
    statsThread.join();
}

TEST_F(GptExecutorTest, MockedModelEvictRestartValidityTest)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    constexpr bool excludeInputFromOutput = false;
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    EXPECT_CALL(*model, updatePeftCache(_)).WillRepeatedly(Invoke([&]() { return; }));
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
    SizeType32 callCount = 0;
    RequestList currentReq;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                currentReq = requestList;
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                }
                callCount++;
            }));

    EXPECT_CALL(*model, forwardSync())
        .WillRepeatedly(Invoke(
            [&]()
            {
                for (auto const& llmReq : currentReq)
                {
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                }
                return;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 6; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));

    SizeType32 const beamWidth = 1;
    ExecutorConfig executorConfig(beamWidth,
        SchedulerConfig(CapacitySchedulerPolicy::kMAX_UTILIZATION)); // Condition 1 : MAX_UTILIZATION scheduling policy
    executorConfig.setEnableChunkedContext(false);                   // Condition 2 : Chunked context disabled
    executorConfig.setRequestStatsMaxIterations(1000);
    auto executor = Executor(model, executorConfig);

    // Create the request
    constexpr bool streaming = true;                   // Condition 3 : Streaming enabled
    SizeType32 const maxNewTokens = 5;
    VecTokens const tooLongInputTokens{1, 2, 3, 4, 5}; // Condition 4 : prompt input len + maxNewTokens > MaxInputLen
    auto tooLongRequest = Request(
        tooLongInputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    // Enqueue the request
    auto longRequestId = executor.enqueueRequest(tooLongRequest);
    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(longRequestId, waitTime);
        for (auto& response : responses)
        {
            EXPECT_EQ(response.hasError(), true);
            EXPECT_THAT(response.getErrorMsg(),
                testing::HasSubstr("sequence length is potentially greater than max input length"));
            done = true;
        }
        ++iter;
    }
}

#if ENABLE_MULTI_DEVICE
// This test can be run manually to test multiGPU execution
// mpirun --allow-run-as-root -n 5 ./executorTest --gtest_filter="*MockedModelMultiGpu/ExecutorTest"
// Number of MPI ranks can be greater than tp

TEST_P(ParamTest, MockedModelMultiGpu)
{
    auto const& world = tensorrt_llm::mpi::MpiComm::world();
    auto const worldRank = world.getRank();
    auto const worldSize = world.getSize();

    // In this test, allow worldSize to be greater than tp = 4
    // If so, set participant ids to be the last 4 ranks
    SizeType32 const tp = std::min(4, worldSize);

    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    bool const streaming = std::get<0>(GetParam());
    bool const excludeInputFromOutput = std::get<1>(GetParam());
    auto const beamWidth = std::get<2>(GetParam());
    OutputConfig outConfig;
    outConfig.excludeInputFromOutput = excludeInputFromOutput;
    auto model = std::make_shared<MockedModel>();

    // Create the request
    constexpr SizeType32 maxNewTokens = 5;
    VecTokens const inputTokens{1, 2, 3, 4};
    auto request
        = Request(inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
    SizeType32 callCount = 0;
    SizeType32 reqCallCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    EXPECT_EQ(llmReq->getTokens().size(), beamWidth);
                    // Verify that all MPI ranks get the expected request, even though only rank 0 actually gets the
                    // request
                    if (reqCallCount == 0)
                    {
                        EXPECT_EQ(llmReq->getOrigPromptLen(), request.getInputTokenIds().size());
                        for (int i = 0; i < llmReq->getOrigPromptLen(); ++i)
                        {
                            EXPECT_EQ(llmReq->getTokens(beamWidth - 1).at(i), request.getInputTokenIds().at(i));
                        }
                    }
                    EXPECT_EQ(llmReq->isStreaming(), request.getStreaming());
                    EXPECT_EQ(llmReq->mMaxNewTokens, request.getMaxTokens());
                    EXPECT_EQ(
                        llmReq->getTokens(beamWidth - 1).size(), request.getInputTokenIds().size() + reqCallCount);

                    SizeType32 tokenId = 1;
                    COMM_SESSION.bcastValue(tokenId, 0);
                    // Don't add any tokens to simulate no output tokens
                    // Simulate leader rank communicating with comm session
                    VecTokens const newTokens(beamWidth, tokenId);
                    llmReq->addNewTokens(newTokens);
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                    reqCallCount++;
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));

    tr::WorldConfig dummyWorldConfig = tr::WorldConfig(tp, 1, 1, worldRank, tp);
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));

    ParallelConfig parallelConfig;

    // Set participant ids to be of size tp, starting at worldSize - 1
    std::vector<SizeType32> participantIds;
    participantIds.reserve(tp);
    for (int i = 0; i < tp; ++i)
    {
        participantIds.push_back(worldSize - tp + i);
    }
    bool const isLeader = (worldRank == participantIds.front());
    parallelConfig.setParticipantIds(participantIds);

    bool const isWorker = (std::find(participantIds.begin(), participantIds.end(), worldRank) != participantIds.end());

    // Set device ids
    std::vector<SizeType32> deviceIds(tp);
    std::iota(deviceIds.begin(), deviceIds.end(), 0);
    parallelConfig.setDeviceIds(deviceIds);

    ExecutorConfig executorConfig(beamWidth);
    executorConfig.setParallelConfig(parallelConfig);
    auto executor = Executor(model, executorConfig);

    EXPECT_EQ(isWorker, executor.isParticipant());

    // Enqueue the request
    IdType requestId = 0;
    if (isLeader)
    {
        requestId = executor.enqueueRequest(request);

        SizeType32 numResponses{0};
        bool done = false;
        int iter = 0;
        while (!done && iter < mMaxWaitMs)
        {
            std::chrono::milliseconds waitTime(1);
            auto responses = executor.awaitResponses(waitTime);
            for (auto& response : responses)
            {
                ++numResponses;
                auto const& result = response.getResult();
                EXPECT_EQ(result.outputTokenIds.size(), beamWidth);
                auto expectedSize = streaming ? (beamWidth > 1 ? numResponses : 1)
                                              : (maxNewTokens + (excludeInputFromOutput ? 0 : inputTokens.size()));
                EXPECT_EQ(result.outputTokenIds.at(beamWidth - 1).size(), expectedSize);
                done = result.isFinal;
            }
            ++iter;
        }

        EXPECT_LT(iter, mMaxWaitMs);
        EXPECT_EQ(numResponses, streaming ? maxNewTokens : 1);
        EXPECT_EQ(callCount, maxNewTokens);
    }
}
#endif // ENABLE_MULTI_DEVICE

TEST_F(GptExecutorTest, MockedModelWithError)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    struct MockedModelParams
    {
        SizeType32 maxInputLen;
        SizeType32 maxSeqLen;
        SizeType32 expectedTerminateCnt;
        SizeType32 expectedForwardCnt;
        bool computeGenLogits;
        bool computeContextLogits;
        std::string expectedError;
    };

    std::vector<MockedModelParams> mockedModelParams;
    // Mocked error in forward call
    mockedModelParams.emplace_back(MockedModelParams{10, 20, 1, 1, true, true, "mocked error"});
    // prompt longer than maxInputLen
    mockedModelParams.emplace_back(MockedModelParams{1, 20, 0, 0, true, true, "exceeds maximum input length"});
    // Model doesn't support context logits output
    mockedModelParams.emplace_back(
        MockedModelParams{10, 20, 0, 0, false, true, "gather_generation_logits must be enabled"});
    // Model doesn't support gen logits output
    mockedModelParams.emplace_back(
        MockedModelParams{10, 20, 0, 0, true, false, "need to build engine with gather_context"});

    for (auto const& mockedModelParam : mockedModelParams)
    {
        auto model = std::make_shared<MockedModel>();
        SizeType32 beamWidth = 1;

        // One request should be terminated
        EXPECT_CALL(*model, terminateRequest(_, _)).Times(mockedModelParam.expectedTerminateCnt);
        EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 1024; }));
        EXPECT_CALL(*model, getLogitDataType()).WillRepeatedly(Invoke([&]() { return nvinfer1::DataType::kFLOAT; }));
        EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
        EXPECT_CALL(*model, getCurrentRequestStats(_))
            .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));

        SizeType32 callCount = 0;
        EXPECT_CALL(*model, forwardAsync(_))
            .WillRepeatedly(Invoke(
                [&](RequestList const&)
                {
                    callCount++;
                    // There was a bug where we were missing a notify call when errors were encountered
                    // and this test was not catching it, probably because the error was reported
                    // before the first call to awaitResponses. So we add a sleep here to make sure
                    // the awaitResponses is called before the error is thrown
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    throw std::runtime_error("mocked error");
                }));

        EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
        EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return mockedModelParam.maxInputLen; }));
        EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return mockedModelParam.maxSeqLen; }));
        EXPECT_CALL(*model, getMaxDraftLen()).WillRepeatedly(Invoke([&]() { return 0; }));
        tr::WorldConfig const dummyWorldConfig;
        EXPECT_CALL(*model, getWorldConfig())
            .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
        tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
        dummyModelConfig.computeContextLogits(mockedModelParam.computeContextLogits);
        dummyModelConfig.computeGenerationLogits(mockedModelParam.computeGenLogits);
        EXPECT_CALL(*model, getModelConfig())
            .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
        EXPECT_CALL(*model, getGatherGenerationLogits())
            .WillRepeatedly(Invoke([&]() -> bool { return mockedModelParam.computeGenLogits; }));
        EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& stats) { return; }));
        EXPECT_CALL(*model, getCurrentRequestStats(_))
            .WillRepeatedly(Invoke([&](RequestStatsPerIteration& stats) { return; }));
        EXPECT_CALL(*model, getIterCounter()).WillRepeatedly(Invoke([&]() -> IterationType { return 0; }));

        ExecutorConfig executorConfig(beamWidth);
        auto executor = Executor(model, executorConfig);

        // Create the request
        SizeType32 maxNewTokens = 5;
        VecTokens inputTokens{1, 2, 3, 4};

        OutputConfig outConfig;
        outConfig.returnContextLogits = true;
        outConfig.returnGenerationLogits = true;

        auto streaming = false;
        auto request = Request(
            inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(beamWidth), outConfig);

        // Enqueue the request
        auto requestId = executor.enqueueRequest(std::move(request));

        bool done = false;
        auto responses = executor.awaitResponses(requestId);
        for (auto& response : responses)
        {
            if (!response.hasError())
            {
                FAIL() << "Expecting an error to be received";
            }
            else
            {
                auto err = response.getErrorMsg();
                EXPECT_THAT(err, testing::HasSubstr(mockedModelParam.expectedError));
                done = true;
            }
        }

        EXPECT_TRUE(done);
        EXPECT_EQ(callCount, mockedModelParam.expectedForwardCnt);
    }
}

TEST_F(GptExecutorTest, MockedModelCancelRequest)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    constexpr bool streaming = true;
    auto model = std::make_shared<MockedModel>();

    std::unordered_map<IdType, tensorrt_llm::executor::FinishReason> reqIdsToTerminate;
    // Two requests with one child request (3 in total) should be terminated
    EXPECT_CALL(*model, terminateRequestSync(_, _))
        .Times(3)
        .WillRepeatedly(Invoke([&](LlmRequestPtr const& llmRequest, FinishReason finishReason)
            { reqIdsToTerminate.try_emplace(llmRequest->mRequestId, finishReason); }));
    EXPECT_CALL(*model, terminateRequest(_, _)).Times(3);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));

    SizeType32 callCount = 0;
    std::unordered_map<IdType, SizeType32> callCountPerSeq;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                for (auto const& llmReq : requestList)
                {
                    if (llmReq->isGenerationCompleteState())
                    {
                        continue;
                    }
                    // Don't add any tokens to simulate no output tokens
                    llmReq->addNewTokens({1});
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                    if (callCountPerSeq.find(llmReq->mRequestId) != callCountPerSeq.end())
                    {
                        callCountPerSeq[llmReq->mRequestId]++;
                    }
                    else
                    {
                        callCountPerSeq[llmReq->mRequestId] = 1;
                    }

                    if (reqIdsToTerminate.count(llmReq->mRequestId) != 0U)
                    {
                        if (!llmReq->isGenerationToCompleteState())
                        {
                            model->terminateRequest(llmReq, false);
                            llmReq->finishByReason(reqIdsToTerminate[llmReq->mRequestId]);
                            llmReq->clearGeneratedTokens();
                        }
                        reqIdsToTerminate.erase(llmReq->mRequestId);
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 100; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 200; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));

    SizeType32 const beamWidth = 1;
    ExecutorConfig const executorConfig(beamWidth);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 const maxNewTokens = 150;
    VecTokens const inputTokens{1, 2, 3, 4};
    auto request = Request(inputTokens, maxNewTokens, streaming);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Cancel the request
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    executor.cancelRequest(requestId);

    bool done = false;
    int iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId, waitTime);
        for (auto& response : responses)
        {

            if (response.hasError())
            {
                FAIL() << "Not expecting an error to be received";
            }

            auto const& result = response.getResult();
            done = result.isFinal;
            if (done)
            {
                for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
                {
                    EXPECT_EQ(result.finishReasons[beamIdx], FinishReason::kCANCELLED);
                }
            }
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    // Expecting to receiving fewer tokens than maxNewTokens
    EXPECT_LT(callCount, maxNewTokens);

    // Create the request having child requests.
    auto samplingConfig2 = SamplingConfig(1);
    samplingConfig2.setNumReturnSequences(2);
    auto request2 = Request(inputTokens, maxNewTokens, streaming, samplingConfig2);

    // Reset call count.
    callCount = 0;
    callCountPerSeq.clear();

    // Enqueue the request
    auto requestId2 = executor.enqueueRequest(request2);

    // Cancel the request
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    executor.cancelRequest(requestId2);

    done = false;
    iter = 0;
    while (!done && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(requestId2, waitTime);
        for (auto& response : responses)
        {

            if (response.hasError())
            {
                FAIL() << "Not expecting an error to be received";
            }

            auto const& result = response.getResult();
            done = result.isFinal;
            if (done)
            {
                EXPECT_EQ(result.finishReasons[0], FinishReason::kCANCELLED);
            }
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    for (auto& [reqId, count] : callCountPerSeq)
    {
        // Expecting to receiving fewer tokens than maxNewTokens
        EXPECT_LT(count, maxNewTokens) << "Failed at request id: " << reqId;
    }
}

TEST_F(GptExecutorTest, MockedModelNumReturns)
{
    using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    SizeType32 const maxBeamWidth = 4;
    OutputConfig const outConfig;
    auto model = std::make_shared<MockedModel>();

    EXPECT_CALL(*model, terminateRequest(_, _)).Times(0);
    EXPECT_CALL(*model, getVocabSizePadded()).Times(0);
    EXPECT_CALL(*model, getLogitDataType()).Times(0);
    tr::WorldConfig const dummyWorldConfig;
    EXPECT_CALL(*model, getWorldConfig())
        .WillRepeatedly(Invoke([&]() -> tr::WorldConfig const& { return dummyWorldConfig; }));
    EXPECT_CALL(*model, getCurrentIterationStats(_)).WillRepeatedly(Invoke([&](IterationStats& /*stats*/) { return; }));
    EXPECT_CALL(*model, getCurrentRequestStats(_))
        .WillRepeatedly(Invoke([&](RequestStatsPerIteration& /*stats*/) { return; }));
    tr::ModelConfig dummyModelConfig(0, 0, 0, 0, 1, 0, nvinfer1::DataType::kHALF);
    EXPECT_CALL(*model, getModelConfig())
        .WillRepeatedly(Invoke([&]() -> tr::ModelConfig const& { return dummyModelConfig; }));
    SizeType32 callCount = 0;
    EXPECT_CALL(*model, forwardAsync(_))
        .WillRepeatedly(Invoke(
            [&](RequestList const& requestList)
            {
                for (auto const& llmReq : requestList)
                {
                    // Don't add any tokens to simulate no output tokens
                    auto numBeams = llmReq->mSamplingConfig.getNumReturnBeams();
                    llmReq->addNewTokens(VecTokens(numBeams, 1));
                    llmReq->setState(tb::LlmRequestState::kGENERATION_IN_PROGRESS);
                    if (llmReq->getMaxNumGeneratedTokens() >= llmReq->mMaxNewTokens)
                    {
                        llmReq->setState(tb::LlmRequestState::kGENERATION_COMPLETE);
                    }
                }
                callCount++;
            }));

    EXPECT_CALL(*model, getMaxNumSequences()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxInputLen()).WillRepeatedly(Invoke([&]() { return 10; }));
    EXPECT_CALL(*model, getMaxSequenceLen()).WillRepeatedly(Invoke([&]() { return 20; }));
    EXPECT_CALL(*model, getVocabSizePadded()).WillRepeatedly(Invoke([&]() { return 80000; }));

    ExecutorConfig const executorConfig(maxBeamWidth);
    auto executor = Executor(model, executorConfig);

    // Create the request
    SizeType32 const maxNewTokens = 5;
    VecTokens const inputTokens{1, 2, 3, 4};
    constexpr bool streaming = false;

    auto samplingConfig1 = SamplingConfig(1);
    samplingConfig1.setNumReturnSequences(3);
    auto request1 = Request(inputTokens, maxNewTokens, streaming, samplingConfig1, outConfig);
    auto samplingConfig2 = SamplingConfig(4);
    auto request2 = Request(inputTokens, maxNewTokens, streaming, samplingConfig2, outConfig);
    auto samplingConfig3 = SamplingConfig(4);
    samplingConfig3.setNumReturnSequences(2);
    auto request3 = Request(inputTokens, maxNewTokens, streaming, samplingConfig3, outConfig);

    // Enqueue the request
    auto requestId1 = executor.enqueueRequest(request1);
    auto requestId2 = executor.enqueueRequest(request2);
    auto requestId3 = executor.enqueueRequest(request3);

    // Expecting one response in beam search. Instead, numReturnSequences limits the number of beams to return.
    std::unordered_map<IdType, SizeType32> expectedNumResponses{{requestId1, 3}, {requestId2, 1}, {requestId3, 1}};
    std::unordered_map<IdType, SizeType32> const expectedNumBeams{{requestId1, 1}, {requestId2, 4}, {requestId3, 2}};

    std::unordered_map<IdType, SizeType32> numResponses{{requestId1, 0}, {requestId2, 0}, {requestId3, 0}};
    std::unordered_map<IdType, SizeType32> numBeams{{requestId1, 0}, {requestId2, 0}, {requestId3, 0}};
    int numFinished = 0;
    int iter = 0;
    while (numFinished < 3 && iter < mMaxWaitMs)
    {
        std::chrono::milliseconds waitTime(1);
        auto responses = executor.awaitResponses(waitTime);
        for (auto& response : responses)
        {
            auto const& result = response.getResult();
            auto reqId = response.getRequestId();
            numFinished += result.isFinal;
            numResponses[reqId]++;
            numBeams[reqId] = result.outputTokenIds.size();
        }
        ++iter;
    }

    EXPECT_LT(iter, mMaxWaitMs);
    EXPECT_EQ(numFinished, 3);
    for (auto& [reqId, numResp] : numResponses)
    {
        EXPECT_EQ(numResp, expectedNumResponses[reqId]);
    }
    for (auto& [reqId, numResp] : numResponses)
    {
        EXPECT_EQ(numResp, expectedNumResponses[reqId]);
    }
}

INSTANTIATE_TEST_SUITE_P(GptExecutorTest, ParamTest,
    testing::Combine(testing::Values(false, true), // streaming
        testing::Values(false, true),              // excludeInputFromOutput
        testing::Values(1, 2)                      // beamWidth
        ),
    generateTestName);
