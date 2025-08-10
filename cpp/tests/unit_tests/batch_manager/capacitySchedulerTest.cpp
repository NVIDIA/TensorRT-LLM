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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestUtils.h"
#include "tensorrt_llm/executor/types.h"

#include <NvInferPlugin.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
using tensorrt_llm::executor::insertRequestInOrder;
namespace tc = tensorrt_llm::common;

using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using VecTokens = std::vector<TokenIdType>;

struct RequestState
{
    int32_t mRequestId;
    int32_t mMaxNewTokens;
    int32_t mPromptLen;
    int32_t mNumGenTokensBegin;
    int32_t mContextCurrentPosition{0};
};

struct ExpectedState
{
    int32_t itBegin;
    int32_t itEnd;
    std::vector<uint64_t> activeIds;
    std::vector<RequestState> scheduledState;
    std::vector<RequestState> scheduledDisaggGenInitState;
};

class MockPeftCacheManager : public BasePeftCacheManager
{
public:
    MockPeftCacheManager(SizeType32 numPages = 15, SizeType32 maxDevicePages = 100, SizeType32 maxHostPages = 1000)
        : mNumPages(numPages)
        , mMaxDevicePages(maxDevicePages)
        , mMaxHostPages(maxHostPages)
    {
    }

    void addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache = true) override {}

    PeftTable ensureBatch(RequestVector const&, RequestVector const&, bool resetGpuCache = false) override
    {
        return PeftTable{};
    }

    void resetDeviceCache() override {}

    void markRequestDone(LlmRequest const& llmReq, bool pause = false) override {}

    [[nodiscard]] SizeType32 getMaxDevicePages() const override
    {
        return mMaxDevicePages;
    }

    [[nodiscard]] SizeType32 getMaxHostPages() const override
    {
        return mMaxHostPages;
    }

    [[nodiscard]] SizeType32 determineNumPages(std::shared_ptr<LlmRequest> llmReqeust) const override
    {
        return mNumPages;
    }

    inline bool enabled() const override
    {
        return true;
    }

private:
    SizeType32 mNumPages;
    SizeType32 mMaxDevicePages;
    SizeType32 mMaxHostPages;
};

class CapacitySchedulerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{

protected:
    using CapacitySchedulerPolicy = tensorrt_llm::executor::CapacitySchedulerPolicy;

    CapacitySchedulerTest() {}

    void SetUp() override {}

    void TearDown() override {}

    static std::shared_ptr<kv_cache_manager::BaseKVCacheManager> getKvCacheManager(SizeType32 maxNumRequests,
        SizeType32 tokensPerBlock, SizeType32 maxNumTokens, SizeType32 maxNumTokensPerSeq,
        SizeType32 sinkTokenLength = 0, bool enableReuse = false,
        kv_cache_manager::CacheType cacheType = kv_cache_manager::CacheType::kSELF)
    {
        auto const numLayers = 10;
        auto const nbKvHeads = 10;
        auto constexpr sizePerHead = 1;
        auto const maxNumBlocks = tc::divUp(maxNumTokens, tokensPerBlock);
        auto const kvDtype = nvinfer1::DataType::kHALF;
        bool onboardBlocks = true;
        CudaStreamPtr streamPtr = std::make_shared<tensorrt_llm::runtime::CudaStream>();

        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        auto const blocksPerWindow = BlocksPerWindow{{maxNumTokensPerSeq, {maxNumBlocks, 0}}};

        // init KV cache block manager
        return std::make_shared<kv_cache_manager::KVCacheManager>(numLayers, nbKvHeads, sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumRequests, 1, std::vector<SizeType32>{maxNumTokensPerSeq}, std::nullopt, kvDtype,
            sinkTokenLength, streamPtr, maxNumTokensPerSeq, enableReuse, onboardBlocks, cacheType);
    }

    static std::shared_ptr<BasePeftCacheManager> getPeftCacheManager()
    {
        return std::make_shared<MockPeftCacheManager>();
    }

    static std::shared_ptr<LlmRequest> createRequest(std::shared_ptr<std::vector<int32_t>> inputTokens,
        int32_t maxNewTokens, std::optional<uint64_t> optionalReqId, std::optional<uint64_t> loraTaskId = std::nullopt,
        tensorrt_llm::executor::PriorityType priority = tensorrt_llm::executor::Request::kDefaultPriority,
        LlmRequestState state = LlmRequestState::kCONTEXT_INIT)
    {
        tensorrt_llm::runtime::SamplingConfig samplingConfig;
        uint64_t reqId = optionalReqId.value_or((rand() % INT64_MAX) + 1);
        auto req = std::make_shared<LlmRequest>(reqId, maxNewTokens, inputTokens, samplingConfig, false);
        req->setPriority(priority);
        req->setState(state);

        if (loraTaskId.has_value())
        {
            req->setLoraTaskId(loraTaskId.value());
        }
        return req;
    }

    static std::shared_ptr<LlmRequest> createRequest(int32_t promptLen, int32_t maxNewTokens,
        std::optional<uint64_t> optionalReqId, std::optional<uint64_t> loraTaskId = std::nullopt,
        tensorrt_llm::executor::PriorityType priority = tensorrt_llm::executor::Request::kDefaultPriority,
        LlmRequestState state = LlmRequestState::kCONTEXT_INIT)
    {
        auto inputTokens = std::make_shared<std::vector<int32_t>>(promptLen, 1);
        return createRequest(inputTokens, maxNewTokens, optionalReqId, loraTaskId, priority, state);
    }

    static std::shared_ptr<LlmRequest> createFromExecutorRequest(int32_t promptLen, int32_t maxNewTokens,
        int32_t encoderInputLen, std::optional<uint64_t> optionalReqId,
        std::optional<uint64_t> loraTaskId = std::nullopt,
        tensorrt_llm::executor::PriorityType priority = tensorrt_llm::executor::Request::kDefaultPriority)
    {

        auto inputTokens = VecTokens(promptLen, 1);
        auto encoderInputTokens = VecTokens(encoderInputLen, 1);

        tensorrt_llm::executor::OutputConfig outConfig;
        outConfig.excludeInputFromOutput = false;
        outConfig.returnLogProbs = false;
        outConfig.returnGenerationLogits = false;
        outConfig.returnContextLogits = false;
        outConfig.returnEncoderOutput = false;

        uint64_t reqId = optionalReqId.value_or((rand() % INT64_MAX) + 1);
        bool streaming = false;
        auto executorReq = tensorrt_llm::executor::Request(
            inputTokens, maxNewTokens, streaming, tensorrt_llm::executor::SamplingConfig(), outConfig);
        executorReq.setEncoderInputTokenIds(encoderInputTokens);
        auto req = std::make_shared<LlmRequest>(reqId, executorReq);
        req->setPriority(priority);

        if (loraTaskId.has_value())
        {
            req->setLoraTaskId(loraTaskId.value());
        }
        return req;
    }
};

using RequestTable = std::map<RequestIdType, std::shared_ptr<LlmRequest>>;
using CheckCallback = std::function<void(RequestTable& scheduledRequest, RequestList& activeRequests, int itCount)>;
using AddNewRequestsCallback = std::function<void(RequestList& activeRequests, int itCount)>;

void prepRequestsForEncoderSkip(RequestList& activeRequests)
{

    for (auto& req : activeRequests)
    {

        if (req->isEncoderInitState() && req->getEncoderTokens())
        {
            TLLM_LOG_INFO("Changing state of request and setting encoder output to skip encoder run");
            req->setState(tensorrt_llm::batch_manager::LlmRequestState::kCONTEXT_INIT);
        }
    }
}

int runTest(CapacityScheduler& capacityScheduler,
    std::shared_ptr<kv_cache_manager::BaseKVCacheManager> const& kvCacheManager, RequestList& activeRequests,
    std::vector<ExpectedState> const& expectedStates, AddNewRequestsCallback const& addNewRequestsCb,
    SizeType32 maxInputLen, std::shared_ptr<BasePeftCacheManager> const& peftCacheManager = nullptr,
    std::shared_ptr<kv_cache_manager::BaseKVCacheManager> const& crossKvCacheManager = nullptr,
    bool hasDisaggGenInit = false)
{
    int itCount = 0;
    while (!activeRequests.empty())
    {
        addNewRequestsCb(activeRequests, itCount);
        prepRequestsForEncoderSkip(activeRequests);
        auto [scheduledRequestsList, scheduledDisaggGenInitRequestsLists, pausedRequests]
            = capacityScheduler(activeRequests, kvCacheManager, peftCacheManager, crossKvCacheManager);

        RequestTable scheduledRequests;
        for (auto& req : scheduledRequestsList)
        {
            scheduledRequests.emplace(req->mRequestId, req);
        }
        RequestTable scheduledDisaggGenInitRequests;
        for (auto& req : scheduledDisaggGenInitRequestsLists)
        {
            scheduledDisaggGenInitRequests.emplace(req->mRequestId, req);
        }

        // ------------------
        // Check state
        // ------------------

        for (int i = 0; i < expectedStates.size(); ++i)
        {
            auto const& expectedState = expectedStates.at(i);

            // If iteration falls within a range, check state
            if (itCount >= expectedState.itBegin && itCount < expectedState.itEnd)
            {

                std::set<uint64_t> previousScheduled;
                if (i > 0)
                {
                    for (auto const& state : expectedStates[i - 1].scheduledState)
                    {
                        previousScheduled.insert(state.mRequestId);
                    }
                }

                EXPECT_EQ(activeRequests.size(), expectedState.activeIds.size()) << "itCount: " << itCount;
                EXPECT_EQ(scheduledRequests.size(), expectedState.scheduledState.size()) << "itCount: " << itCount;
                EXPECT_EQ(scheduledDisaggGenInitRequests.size(), expectedState.scheduledDisaggGenInitState.size())
                    << "itCount: " << itCount;

                // Check that active requests are as expected
                int reqCount = 0;
                for (auto const& activeReq : activeRequests)
                {
                    EXPECT_EQ(activeReq->mRequestId, expectedState.activeIds[reqCount]) << "itCount: " << itCount;
                    reqCount++;
                }

                // Check that scheduled requests are as expected
                for (auto scheduledReqState : expectedState.scheduledState)
                {
                    // Check that scheduleId is found in scheduled Requests
                    EXPECT_NE(scheduledRequests.find(scheduledReqState.mRequestId), scheduledRequests.end())
                        << "itCount: " << itCount << "mRequestId:" << scheduledReqState.mRequestId;
                }

                // Check that scheduled requests are as expected
                for (auto scheduledReqState : expectedState.scheduledDisaggGenInitState)
                {
                    // Check that scheduleId is found in scheduled Requests
                    EXPECT_NE(
                        scheduledRequests.find(scheduledReqState.mRequestId), scheduledDisaggGenInitRequests.end())
                        << "itCount: " << itCount << "mRequestId:" << scheduledReqState.mRequestId;
                }

                // Check that all new scheduled are in context init state
                if (itCount == expectedState.itBegin)
                {
                    for (auto scheduledReqState : expectedState.scheduledState)
                    {
                        auto scheduledId = scheduledReqState.mRequestId;
                        if (!hasDisaggGenInit)
                        {
                            if (previousScheduled.find(scheduledId) == previousScheduled.end())
                            {
                                EXPECT_EQ(scheduledRequests.at(scheduledId)->getState(), LlmRequestState::kCONTEXT_INIT)
                                    << "itCount: " << itCount << "reqId: " << scheduledId;
                            }
                            else if (!scheduledRequests.at(scheduledId)->getContextRemainingLength())
                            {
                                EXPECT_EQ(scheduledRequests.at(scheduledId)->getState(),
                                    LlmRequestState::kGENERATION_IN_PROGRESS)
                                    << "itCount: " << itCount << "reqId: " << scheduledId;
                            }
                        }

                        // Check that request parameters are as expected
                        EXPECT_EQ(scheduledRequests.at(scheduledId)->mMaxNewTokens, scheduledReqState.mMaxNewTokens)
                            << "itCount: " << itCount << ", reqId: " << scheduledId;
                        EXPECT_EQ(scheduledRequests.at(scheduledId)->mPromptLen, scheduledReqState.mPromptLen)
                            << "itCount: " << itCount << ", reqId: " << scheduledId;
                        EXPECT_EQ(scheduledRequests.at(scheduledId)->getMaxNumGeneratedTokens(),
                            scheduledReqState.mNumGenTokensBegin)
                            << "itCount: " << itCount << ", reqId: " << scheduledId;
                        EXPECT_EQ(scheduledRequests.at(scheduledId)->getContextCurrentPosition(),
                            scheduledReqState.mContextCurrentPosition)
                            << "itCount: " << itCount << ", reqId: " << scheduledId;
                    }
                }
            }
        }

        // Mock the behavior of TrtModelInflightBatching

        // pause all requests that haven't been scheduled
        for (auto const& llmReq : pausedRequests)
        {
            kvCacheManager->removeSequence(llmReq->mRequestId);
            if (crossKvCacheManager)
            {
                crossKvCacheManager->removeSequence(llmReq->mRequestId);
            }

            llmReq->pause(maxInputLen);

            // All non-scheduled requests should be in state CONTEXT_INIT
            EXPECT_EQ(llmReq->getState(), LlmRequestState::kCONTEXT_INIT);
        }

        // Move scheduled disagg gen init to generation in progress
        for (auto& [reqId, llmReq] : scheduledDisaggGenInitRequests)
        {
            if (llmReq->isDisaggGenerationInitState())
            {
                kvCacheManager->addSequence(
                    llmReq->mRequestId, llmReq->mPromptLen, llmReq->mSamplingConfig.beamWidth, llmReq);
                llmReq->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
                llmReq->setContextCurrentPosition(llmReq->mPromptLen);
                llmReq->setDecodingIter(1);
                llmReq->addNewTokens({itCount});
            }
        }
        // Append a token for all scheduled requests
        for (auto& [reqId, llmReq] : scheduledRequests)
        {
            auto const promptLen = llmReq->mPromptLen;
            if (llmReq->isContextInitState())
            {
                if (!llmReq->isLastContextChunk())
                {
                    TLLM_CHECK_WITH_INFO(llmReq->getContextChunkSize() % kvCacheManager->getTokensPerBlock() == 0,
                        "To prevent cache fragmentation, the context chunk size should be divisible by the number "
                        "of "
                        "tokens per block, except for the last chunk.");
                }
                if (llmReq->isFirstContextChunk())
                {
                    // We need to perform initialization work for the first context chunk.
                    kvCacheManager->addSequence(
                        llmReq->mRequestId, promptLen, llmReq->mSamplingConfig.beamWidth, llmReq);
                    if (crossKvCacheManager)
                    {
                        crossKvCacheManager->addSequence(llmReq->mRequestId, llmReq->getEncoderOutputLen(),
                            llmReq->mSamplingConfig.beamWidth, llmReq);
                    }
                }
                auto preContextLength = llmReq->getContextChunkSize();
                // Values returned by isFirstContextChunk and isLastContextChunk will change after this call.
                // This call resets context chunk size to zero for some reason, so need to set it again.
                llmReq->moveToNextContextChunk();
                llmReq->setContextChunkSize(preContextLength);

                if (llmReq->getContextRemainingLength() == 0)
                {
                    kvCacheManager->storeContextBlocks(*llmReq);
                    if (crossKvCacheManager)
                    {
                        crossKvCacheManager->storeContextBlocks(*llmReq);
                    }
                    llmReq->addNewTokens({itCount});
                    llmReq->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
                }
            }
            else
            {
                kvCacheManager->addToken(llmReq->mRequestId);
                llmReq->addNewTokens({itCount});
            }
            if (llmReq->getNumTokens(0) == promptLen + llmReq->mMaxNewTokens)
            {
                llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
                kvCacheManager->removeSequence(llmReq->mRequestId, llmReq);
                if (crossKvCacheManager)
                {
                    crossKvCacheManager->removeSequence(llmReq->mRequestId, llmReq);
                }
            }
        }

        // Remove completed requests
        for (auto it = activeRequests.cbegin(); it != activeRequests.cend();)
        {
            auto const& llmReq = (*it);
            if (llmReq->getState() == LlmRequestState::kGENERATION_COMPLETE)
            {
                activeRequests.erase(it++);
            }
            else
            {
                it++;
            }
        }
        itCount++;
    }
    return itCount;
}

TEST_F(CapacitySchedulerTest, SimpleShouldFit)
{
    SizeType32 kvCacheMaxNumTokens = 200;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        CapacitySchedulerPolicy::kMAX_UTILIZATION, CapacitySchedulerPolicy::kSTATIC_BATCH};
    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(
                maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
            auto peftCacheManager = getPeftCacheManager();
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 80;
            int32_t promptLen = 10;

            RequestList activeRequests;
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1234));
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 6789));

            std::vector<ExpectedState> expectedStates;
            expectedStates.push_back(ExpectedState{0, 80, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager);

            EXPECT_EQ(numIterations, 80);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleShouldFitWithCrossBlocks)
{
    SizeType32 kvCacheMaxNumTokens
        = 200; // 90*2=180 + 20 (for one request 10 for sink token as one whole block is reserved so 20 for 2 requests)
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;

    SizeType32 crossKvCacheMaxNumTokens = 20; // 20 for 2 requests , encoder input len = 2*10
    SizeType32 crossKvCacheMaxNumTokensPerSeq = 10;
    SizeType32 crossKvCacheTokensPerBlock = 10;
    int32_t encoderInputLen = 10;
    SizeType32 crossKvsinkTokenLen = 0; // sink tokens not accounted in cross kv cache

    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;
    kv_cache_manager::CacheType cacheType = kv_cache_manager::CacheType::kCROSS;
    bool enableReuse = false;

    auto capacitySchedulerPolicies
        = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(
                maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
            auto crossKvCacheManager = getKvCacheManager(maxNumRequests, crossKvCacheTokensPerBlock,
                crossKvCacheMaxNumTokens, crossKvCacheMaxNumTokensPerSeq, crossKvsinkTokenLen, enableReuse, cacheType);
            auto peftCacheManager = getPeftCacheManager();
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 80;
            int32_t promptLen = 10;
            RequestList activeRequests;
            activeRequests.push_back(createFromExecutorRequest(promptLen, maxNewTokens, encoderInputLen, 0, 1234));
            activeRequests.push_back(createFromExecutorRequest(promptLen, maxNewTokens, encoderInputLen, 1, 6789));

            std::vector<ExpectedState> expectedStates;
            expectedStates.push_back(ExpectedState{0, 80, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager, crossKvCacheManager);

            EXPECT_EQ(numIterations, 80);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleLoraFitsDuplicateTask)
{
    SizeType32 kvCacheMaxNumTokens = 210;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        CapacitySchedulerPolicy::kMAX_UTILIZATION, CapacitySchedulerPolicy::kSTATIC_BATCH};
    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(
                maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
            auto peftCacheManager = std::make_shared<MockPeftCacheManager>(15, 30, 30);
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 50;
            int32_t promptLen = 10;

            RequestList activeRequests;
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1234));
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 5678));
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1234));

            std::vector<ExpectedState> expectedStates;
            expectedStates.push_back(ExpectedState{0, 50, {0, 1, 2}, {{0, 50, 10, 0}, {1, 50, 10, 0}, {2, 50, 10, 0}}});

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager);

            EXPECT_EQ(numIterations, 50);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleLoraDoesntFitDuplicateTask)
{
    SizeType32 kvCacheMaxNumTokens = 210;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        CapacitySchedulerPolicy::kMAX_UTILIZATION, CapacitySchedulerPolicy::kSTATIC_BATCH};
    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};

    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        std::vector<ExpectedState> expectedStates;
        uint32_t expectedIterations{0};
        if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kMAX_UTILIZATION)
        {
            expectedStates.emplace_back(ExpectedState{0, 50, {0, 1, 2}, {{0, 50, 10, 0}}});
            expectedStates.emplace_back(ExpectedState{50, 100, {1, 2}, {{1, 50, 10, 0}}});
            expectedStates.emplace_back(ExpectedState{100, 150, {2}, {{2, 50, 10, 0}}});
            expectedIterations = 150;
        }
        else if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        {
            // NO_EVICT can guarantee no eviction with requests 0 and 2, so it will allocate them concurrently
            expectedStates.emplace_back(ExpectedState{0, 50, {0, 1, 2}, {{0, 50, 10, 0}, {2, 50, 10, 0}}});
            expectedStates.emplace_back(ExpectedState{50, 100, {1}, {{1, 50, 10, 0}}});
            expectedIterations = 100;
        }
        else if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kSTATIC_BATCH)
        {
            // STATIC_BATCH can guarantee no eviction with requests 0 and 2, so it will allocate them concurrently
            expectedStates.emplace_back(ExpectedState{0, 50, {0, 1, 2}, {{0, 50, 10, 0}, {2, 50, 10, 0}}});
            expectedStates.emplace_back(ExpectedState{50, 100, {1}, {{1, 50, 10, 0}}});
            expectedIterations = 100;
        }

        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(
                maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
            auto peftCacheManager = std::make_shared<MockPeftCacheManager>(20, 30, 30);
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, static_cast<bool>(kvCacheManager));

            // Create two requests that should not fit in kvCache for entire duration
            int32_t maxNewTokens = 50;
            int32_t promptLen = 10;

            RequestList activeRequests;
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1234));
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 5678));
            activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1234));

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager);

            EXPECT_EQ(numIterations, expectedIterations);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleShouldFitInChunk)
{
    SizeType32 kvCacheMaxNumTokens = 200;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT,
        CapacitySchedulerPolicy::kMAX_UTILIZATION, CapacitySchedulerPolicy::kSTATIC_BATCH};
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        auto kvCacheManager
            = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that should fit in kvCache for entire duration
        int32_t maxNewTokens = 40;
        int32_t promptLen = 50;

        RequestList activeRequests;
        auto request0 = createRequest(promptLen, maxNewTokens, 0, 1234);
        auto request1 = createRequest(promptLen, maxNewTokens, 1, 6789);
        request0->setContextChunkSize(20);
        request1->setContextChunkSize(20);
        activeRequests.push_back(std::move(request0));
        activeRequests.push_back(std::move(request1));

        std::vector<ExpectedState> expectedStates;
        expectedStates.push_back(ExpectedState{0, 1, {0, 1}, {{0, 40, 50, 0, 0}, {1, 40, 50, 0, 0}}});
        expectedStates.push_back(ExpectedState{1, 2, {0, 1}, {{0, 40, 50, 0, 20}, {1, 40, 50, 0, 20}}});
        expectedStates.push_back(ExpectedState{2, 3, {0, 1}, {{0, 40, 50, 0, 40}, {1, 40, 50, 0, 40}}});
        expectedStates.push_back(ExpectedState{3, 4, {0, 1}, {{0, 40, 50, 1, 50}, {1, 40, 50, 1, 50}}});
        expectedStates.push_back(ExpectedState{4, 42, {0, 1}, {{0, 40, 50, 2, 50}, {1, 40, 50, 2, 50}}});

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, 42);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitMaxUtilization)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList activeRequests;
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1));

        std::vector<ExpectedState> expectedStates;
        if (sinkTokenLen == 0)
        {
            // Up to iteration 41, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 41, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
            // At iteration 41, running out of space, only one scheduled (req 0)
            expectedStates.push_back(ExpectedState{41, 80, {0, 1}, {{0, 80, 10, 41, 10}}});
            // At iteration 80, req 0 is done
            expectedStates.push_back(ExpectedState{80, 120, {1}, {{1, 39, 51, 0}}});
        }
        else if (sinkTokenLen == 4)
        {
            // Up to iteration 35, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 35, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
            // At iteration 35, running out of space, only one scheduled (req 0)
            expectedStates.push_back(ExpectedState{35, 80, {0, 1}, {{0, 80, 10, 35, 10}}});
            // At iteration 80, req 0 is done
            expectedStates.push_back(ExpectedState{80, 126, {1}, {{1, 45, 45, 0}}});
        }

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        int expectedNumIters = (sinkTokenLen == 0) ? 119 : 125;
        EXPECT_EQ(numIterations, expectedNumIters);
    }
}

TEST_F(CapacitySchedulerTest, DisaggGenInitMaxUtilization)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    SizeType32 sinkTokenLen = 0;
    auto kvCacheManager = getKvCacheManager(
        maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
    auto peftCacheManager = getPeftCacheManager();
    CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
    auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

    // Create two requests that will not fit in kvCache for entire duration
    int32_t maxNewTokens = 80;
    int32_t promptLen = 10;

    RequestList activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, std::nullopt,
        tensorrt_llm::executor::Request::kDefaultPriority, LlmRequestState::kDISAGG_GENERATION_INIT));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, std::nullopt,
        tensorrt_llm::executor::Request::kDefaultPriority, LlmRequestState::kDISAGG_GENERATION_INIT));

    std::vector<ExpectedState> expectedStates;
    expectedStates.push_back(ExpectedState{0, 1, {0, 1}, {}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
    // Up to iteration 41, kvCache is big enough, expect 2 requests
    expectedStates.push_back(ExpectedState{1, 41, {0, 1}, {{0, 80, 10, 1, 10}, {1, 80, 10, 1, 10}}});
    // At iteration 41, running out of space, only one scheduled (req 0)
    expectedStates.push_back(ExpectedState{41, 80, {0, 1}, {{0, 80, 10, 41, 10}}});
    // At iteration 80, req 0 is done
    expectedStates.push_back(ExpectedState{80, 120, {1}, {{1, 39, 51, 0}}});

    // Callback to call at each iteration, to have option to add new active Requests
    auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

    int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
        maxInputLen, peftCacheManager, nullptr, true);

    int expectedNumIters = (sinkTokenLen == 0) ? 119 : 125;
    EXPECT_EQ(numIterations, expectedNumIters);
}

TEST_F(CapacitySchedulerTest, RequestsSortedByPriorities)
{
    RequestList activeRequests;
    insertRequestInOrder(activeRequests, createRequest(10, 20, 0, std::nullopt));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 1, std::nullopt));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 2, std::nullopt, 0.6));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 3, std::nullopt, 0.6));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 4, std::nullopt, 0.6));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 5, std::nullopt, 1.0));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 6, std::nullopt, 0.3));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 7, std::nullopt, 0.3));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 8, std::nullopt, 0.3));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 9, std::nullopt, 0.6));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 10, std::nullopt, 1.0));
    insertRequestInOrder(activeRequests, createRequest(10, 20, 11, std::nullopt));

    std::vector<RequestIdType> expectedOrder = {5, 10, 2, 3, 4, 9, 0, 1, 11, 6, 7, 8};

    int i = 0;
    for (auto const& a : activeRequests)
    {
        EXPECT_EQ(a->mRequestId, expectedOrder[i++]);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitPriorities)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    // Removed configuration:
    // {CapacitySchedulerPolicy::kMAX_UTILIZATION, 4, 125}
    // {CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, 1, 160}
    auto sinkTokenLens = std::vector<SizeType32>{0};
    auto configurations = std::vector<std::tuple<CapacitySchedulerPolicy, SizeType32, int>>{
        {CapacitySchedulerPolicy::kMAX_UTILIZATION, 0, 119}};

    for (auto [capacitySchedulerPolicy, sinkTokenLen, expectedNumIters] : configurations)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList activeRequests;
        insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 0, std::nullopt));
        insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 1, std::nullopt, 0.6));

        std::vector<ExpectedState> expectedStates;
        if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kMAX_UTILIZATION && sinkTokenLen == 0)
        {
            // Up to iteration 41, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 41, {1, 0}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
            // At iteration 41, running out of space, only one scheduled (req 1, as it has more priority)
            expectedStates.push_back(ExpectedState{41, 80, {1, 0}, {{1, 80, 10, 41, 10}}});
            // At iteration 80, req 1 is done
            expectedStates.push_back(ExpectedState{80, 120, {0}, {{0, 39, 51, 0}}});
        }
        else if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kMAX_UTILIZATION && sinkTokenLen == 4)
        {
            // Up to iteration 35, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 35, {1, 0}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
            // At iteration 35, running out of space, only one scheduled (req 1, as it has more priority)
            expectedStates.push_back(ExpectedState{35, 80, {1, 0}, {{1, 80, 10, 35, 10}}});
            // At iteration 80, req 1 is done
            expectedStates.push_back(ExpectedState{80, 126, {0}, {{0, 45, 45, 0}}});
        }
        else if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
        {
            // It cannot fit both for all iterations, so it should pick req 1 with higher priority
            expectedStates.push_back(ExpectedState{0, 80, {1, 0}, {{1, 80, 10, 0}}});
            // At iteration 80, req 1 is done
            expectedStates.push_back(ExpectedState{80, 160, {0}, {{0, 80, 10, 0}}});
        }

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, expectedNumIters);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitMaxUtilizationInChunk)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto kvCacheManager
        = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq);
    auto peftCacheManager = getPeftCacheManager();
    CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
    auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

    // Create two requests that will not fit in kvCache for entire duration
    int32_t maxNewTokens = 60;
    int32_t promptLen = 30;

    RequestList activeRequests;
    auto request0 = createRequest(promptLen, maxNewTokens, 0);
    auto request1 = createRequest(promptLen, maxNewTokens, 1);
    request0->setContextChunkSize(20);
    request1->setContextChunkSize(20);
    activeRequests.push_back(std::move(request0));
    activeRequests.push_back(std::move(request1));

    std::vector<ExpectedState> expectedStates;
    expectedStates.push_back(ExpectedState{0, 2, {0, 1}, {{0, 60, 30, 0, 0}, {1, 60, 30, 0, 0}}});
    expectedStates.push_back(ExpectedState{2, 22, {0, 1}, {{0, 60, 30, 1, 30}, {1, 60, 30, 1, 30}}});
    expectedStates.push_back(ExpectedState{22, 61, {0, 1}, {{0, 60, 30, 21, 30}}});
    // The context may be further chunked at runtime, which is distinct from the current situation.
    expectedStates.push_back(ExpectedState{61, 100, {1}, {{1, 39, 51, 0, 0}}});

    // Callback to call at each iteration, to have option to add new active Requests
    auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

    int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
        maxInputLen, peftCacheManager);

    EXPECT_EQ(numIterations, 100);
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitMaxUtilizationInChunkedCache)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto kvCacheManager
        = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq);
    auto peftCacheManager = getPeftCacheManager();
    CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
    auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

    // Create two requests that will not fit in kvCache for entire duration
    int32_t maxNewTokens = 20;
    int32_t promptLen = 70;

    RequestList activeRequests;
    auto request0 = createRequest(promptLen, maxNewTokens, 0);
    auto request1 = createRequest(promptLen, maxNewTokens, 1);
    request0->setContextChunkSize(40);
    request1->setContextChunkSize(40);
    activeRequests.push_back(std::move(request0));
    activeRequests.push_back(std::move(request1));

    std::vector<ExpectedState> expectedStates;
    expectedStates.push_back(ExpectedState{0, 1, {0, 1}, {{0, 20, 70, 0, 0}}});
    expectedStates.push_back(ExpectedState{1, 2, {0, 1}, {{0, 20, 70, 0, 40}}});
    expectedStates.push_back(ExpectedState{2, 21, {0, 1}, {{0, 20, 70, 1, 70}}});
    expectedStates.push_back(ExpectedState{21, 22, {1}, {{1, 20, 70, 0, 0}}});
    expectedStates.push_back(ExpectedState{22, 23, {1}, {{1, 20, 70, 0, 40}}});
    expectedStates.push_back(ExpectedState{23, 24, {1}, {{1, 20, 70, 1, 70}}});
    expectedStates.push_back(ExpectedState{24, 42, {1}, {{1, 20, 70, 2, 70}}});

    // Callback to call at each iteration, to have option to add new active Requests
    auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

    int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
        maxInputLen, peftCacheManager);

    EXPECT_EQ(numIterations, 42);
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitMaxUtilizationDraftTokens)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;
    SizeType32 numDraftTokens1 = 5;
    SizeType32 numDraftTokens2 = 10;

    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, 0);
        auto peftCacheManager = getPeftCacheManager();
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kMAX_UTILIZATION;
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        auto const draftTokens1 = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>(numDraftTokens1));
        auto const draftTokens2 = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>(numDraftTokens2));
        RequestList activeRequests;
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        activeRequests.back()->setDraftTokens(draftTokens1);
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1));
        activeRequests.back()->setDraftTokens(draftTokens2);

        std::vector<ExpectedState> expectedStates;
        // Up to iteration 31, kvCache is big enough, expect 2 requests
        expectedStates.push_back(ExpectedState{0, 31, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
        // At iteration 31, running out of space, only one scheduled (req 0)
        expectedStates.push_back(ExpectedState{31, 80, {0, 1}, {{0, 80, 10, 31, 10}}});
        // At iteration 80, req 0 is done
        expectedStates.push_back(ExpectedState{80, 129, {1}, {{1, 49, 41, 0}}});

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        int expectedNumIters = 129;
        EXPECT_EQ(numIterations, expectedNumIters);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitGuaranteedCompletion)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0. (e.g. 4)
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList activeRequests;
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1));

        std::vector<ExpectedState> expectedStates;
        expectedStates.push_back(ExpectedState{0, 80, {0, 1}, {{0, 80, 10, 0}}});
        // At iteration 80, req 0 is done
        expectedStates.push_back(ExpectedState{80, 160, {1}, {{1, 80, 10, 0}}});

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, 160);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitWithCrossBlocks)
{
    SizeType32 kvCacheMaxNumTokens
        = 200; // 90*2=180 + 20 (for one request 10 for sink token as one whole block is reserved so 20 for 2 requests)
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;

    int32_t encoderInputLen = 10;
    SizeType32 crossKvCacheMaxNumTokens
        = 10; // Only one request should be able to fit so 10 is encoder input len, hence 1 block.
    SizeType32 crossKvCacheTokensPerBlock = 10;
    SizeType32 crossKvCacheMaxNumTokensPerSeq = 10;
    SizeType32 crossKvsinkTokenLen = 0;

    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;
    kv_cache_manager::CacheType cacheType = kv_cache_manager::CacheType::kCROSS;
    bool enableReuse = false;

    auto capacitySchedulerPolicies
        = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};

    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(
                maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
            auto crossKvCacheManager = getKvCacheManager(maxNumRequests, crossKvCacheTokensPerBlock,
                crossKvCacheMaxNumTokens, crossKvCacheMaxNumTokensPerSeq, crossKvsinkTokenLen, enableReuse, cacheType);
            auto peftCacheManager = getPeftCacheManager();
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 80;
            int32_t promptLen = 10;

            RequestList activeRequests;
            activeRequests.push_back(createFromExecutorRequest(promptLen, maxNewTokens, encoderInputLen, 0, 1234));
            activeRequests.push_back(createFromExecutorRequest(promptLen, maxNewTokens, encoderInputLen, 1, 6789));

            std::vector<ExpectedState> expectedStates;
            expectedStates.push_back(ExpectedState{0, 80, {0, 1}, {{0, 80, 10, 0}}});
            // At iteration 80, req 0 is done
            expectedStates.push_back(ExpectedState{80, 160, {1}, {{1, 80, 10, 0}}});

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager, crossKvCacheManager);

            EXPECT_EQ(numIterations, 160);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitGuaranteedCompletionInChunk)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto kvCacheManager
        = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq);
    auto peftCacheManager = getPeftCacheManager();
    CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

    // Create two requests that should fit in kvCache for entire duration
    int32_t maxNewTokens = 60;
    int32_t promptLen = 30;

    RequestList activeRequests;
    auto request0 = createRequest(promptLen, maxNewTokens, 0);
    auto request1 = createRequest(promptLen, maxNewTokens, 1);
    request0->setContextChunkSize(20);
    request1->setContextChunkSize(20);
    activeRequests.push_back(std::move(request0));
    activeRequests.push_back(std::move(request1));

    std::vector<ExpectedState> expectedStates;
    expectedStates.push_back(ExpectedState{0, 1, {0, 1}, {{0, 60, 30, 0, 0}}});
    expectedStates.push_back(ExpectedState{1, 2, {0, 1}, {{0, 60, 30, 0, 20}}});
    expectedStates.push_back(ExpectedState{2, 3, {0, 1}, {{0, 60, 30, 1, 30}}});
    expectedStates.push_back(ExpectedState{3, 61, {0, 1}, {{0, 60, 30, 2, 30}}});
    // At iteration 61, req 0 is done
    expectedStates.push_back(ExpectedState{61, 62, {1}, {{1, 60, 30, 0, 0}}});
    expectedStates.push_back(ExpectedState{62, 63, {1}, {{1, 60, 30, 0, 20}}});
    expectedStates.push_back(ExpectedState{63, 64, {1}, {{1, 60, 30, 1, 30}}});
    expectedStates.push_back(ExpectedState{64, 122, {1}, {{1, 60, 30, 2, 30}}});

    // Callback to call at each iteration, to have option to add new active Requests
    auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

    int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
        maxInputLen, peftCacheManager);

    EXPECT_EQ(numIterations, 122);
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitAddingNewRequestsMaxUtilization)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler
            = CapacityScheduler(maxNumRequests, CapacitySchedulerPolicy::kMAX_UTILIZATION, kvCacheManager != nullptr);

        // Initially two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList initActiveRequests;
        initActiveRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        initActiveRequests.push_back(createRequest(promptLen, maxNewTokens, 1));

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this, promptLen, maxNewTokens](RequestList& activeRequests, int itCount)
        {
            if (itCount == 10)
            {
                activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2));
                activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3));
            }
        };

        std::vector<ExpectedState> expectedStates;
        if (sinkTokenLen == 0)
        {
            // Up to iteration 10, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});
            // At iteration 10, two more request get added, fits in KV cache
            expectedStates.push_back(ExpectedState{
                10, 21, {0, 1, 2, 3}, {{0, 80, 10, 10, 10}, {1, 80, 10, 10, 10}, {2, 80, 10, 0}, {3, 80, 10, 0}}});

            // At iteration 21, running out of kv cache
            // Req 0 and 1 used 31 tokens (62 tokens)
            // Req 2 and 3 used 21 tokens (42 tokens)
            // Each sequence need 1 block -> Need to drop two requests
            expectedStates.push_back(ExpectedState{21, 41, {0, 1, 2, 3}, {{0, 80, 10, 21, 10}, {1, 80, 10, 21, 10}}});

            // At iteration 41, running out of Kv cache again
            // Req 0 used 51 tokens
            // Req 1 used 51 tokens
            expectedStates.push_back(ExpectedState{41, 80, {0, 1, 2, 3}, {{0, 80, 10, 41, 10}}});
            // At it 80, req 0 is done, 100 free kv tokens, req 1 has 51 tokens (needs 6 blks), req 2 21 tokens (needs 3
            // blks), req 3 21 tokens (needs 3 blocks)
            expectedStates.push_back(ExpectedState{80, 90, {1, 2, 3}, {{1, 39, 51, 0, 0}, {2, 69, 21, 0}}});

            // At it 90, running out of kv cache again
            // req 1 used 61 tokens
            // req 2 used 31 tokens
            // So we need two more blocks -> need to drop req 2
            expectedStates.push_back(ExpectedState{90, 119, {1, 2, 3}, {{1, 39, 51, 10, 51}}});

            // At it 119, req 1 is done
            // Req 2 used 31 tokens
            // Req 3 used 21 tokens
            // Need total of 7 blocks
            expectedStates.push_back(ExpectedState{119, 139, {2, 3}, {{2, 59, 31, 0}, {3, 69, 21, 0}}});

            // At it 139, run out of kv cache again
            // Req 2 used 51 tokens
            // Req 3 used 41 tokens
            // Need two more blocks -> need to drop req 3
            expectedStates.push_back(ExpectedState{139, 177, {2, 3}, {{2, 59, 31, 20, 31}}});

            // At it 178, req 2 is done
            // Only req 3 is remaining, -> 41 tokens generated
            expectedStates.push_back(ExpectedState{178, 227, {3}, {{3, 49, 41, 0}}});
        }
        else if (sinkTokenLen == 4)
        {
            // Up to iteration 10, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 80, 10, 0, 0}, {1, 80, 10, 0, 0}}});
            // At iteration 10, two more request get added, fits in KV cache
            expectedStates.push_back(ExpectedState{
                10, 15, {0, 1, 2, 3}, {{0, 80, 10, 10, 10}, {1, 80, 10, 10, 10}, {2, 80, 10, 0}, {3, 80, 10, 0}}});

            // At iteration 15, running out of kv cache
            // Req 0 and 1 used 31 tokens (62 tokens), including 6 bubble tokens
            // Req 2 and 3 used 21 tokens (42 tokens), including 6 bubble tokens
            // Each sequence need 1 block -> Need to drop two requests
            expectedStates.push_back(ExpectedState{15, 35, {0, 1, 2, 3}, {{0, 80, 10, 15, 10}, {1, 80, 10, 15, 10}}});

            // At iteration 35, running out of Kv cache again
            // Req 0 used 51 tokens, including 6 bubble tokens
            // Req 1 used 51 tokens, including 6 bubble tokens
            expectedStates.push_back(ExpectedState{35, 80, {0, 1, 2, 3}, {{0, 80, 10, 35, 10}}});
            // At it 80, req 0 is done, 100 free kv tokens, req 1 has 45 tokens (needs 6 blks), req 2 15 tokens (needs 3
            // blks), req 3 15 tokens (needs 3 blocks)
            expectedStates.push_back(ExpectedState{80, 90, {1, 2, 3}, {{1, 45, 45, 0, 0}, {2, 75, 15, 0}}});

            // At it 90, running out of kv cache again
            // req 1 used 61 tokens, including 6 bubble tokens
            // req 2 used 31 tokens, including 6 bubble tokens
            // So we need two more blocks -> need to drop req 2
            expectedStates.push_back(ExpectedState{90, 125, {1, 2, 3}, {{1, 45, 45, 10, 45}}});

            // At it 119, req 1 is done
            // Req 2 used 31 tokens, including 6 bubble tokens
            // Req 3 used 21 tokens, including 6 bubble tokens
            // Need total of 7 blocks
            expectedStates.push_back(ExpectedState{125, 145, {2, 3}, {{2, 65, 25, 0, 0}, {3, 75, 15, 0}}});

            // At it 145, run out of kv cache again
            // Req 2 used 51 tokens, including 6 bubble tokens
            // Req 3 used 41 tokens, including 6 bubble tokens
            // Need two more blocks -> need to drop req 3
            expectedStates.push_back(ExpectedState{145, 189, {2, 3}, {{2, 65, 25, 20, 25}}});

            // At it 190, req 2 is done
            // Only req 3 is remaining, -> 35 tokens generated
            expectedStates.push_back(ExpectedState{190, 245, {3}, {{3, 55, 35, 0}}});
        }

        // Callback to call after scheduling requests
        int numIterations = runTest(capacityScheduler, kvCacheManager, initActiveRequests, expectedStates,
            addNewRequestsCb, maxInputLen, peftCacheManager);

        int expectedNumIters = (sinkTokenLen == 0) ? 227 : 245;
        EXPECT_EQ(numIterations, expectedNumIters);
    }
}

TEST_F(CapacitySchedulerTest, SimpleSurpassMaxNumRequestsWithPriorities)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    auto sinkTokenLen = 0;
    auto kvCacheManager = getKvCacheManager(
        maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
    auto peftCacheManager = getPeftCacheManager();
    auto capacityScheduler
        = CapacityScheduler(maxNumRequests, CapacitySchedulerPolicy::kMAX_UTILIZATION, kvCacheManager != nullptr);

    int32_t maxNewTokens = 10;
    int32_t promptLen = 10;

    RequestList initActiveRequests;
    insertRequestInOrder(initActiveRequests, createRequest(promptLen, maxNewTokens, 0));
    insertRequestInOrder(initActiveRequests, createRequest(promptLen, maxNewTokens, 1));

    // Callback to call at each iteration, to have option to add new active Requests
    auto addNewRequestsCb = [this, promptLen, maxNewTokens](RequestList& activeRequests, int itCount)
    {
        if (itCount == 2)
        {
            insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 2, std::nullopt, 0.6));
            insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 3, std::nullopt, 0.7));
        }
    };

    std::vector<ExpectedState> expectedStates;
    // Up to iteration 2, expect 2 requests
    expectedStates.push_back(ExpectedState{0, 2, {0, 1}, {{0, 10, 10, 0}, {1, 10, 10, 0}}});

    // At iteration 2, two more request get added. They have more priority than
    // running requests, therefore they are added to activeRequests. The maxNumRequests is 2,
    // so only the newly added requests should run.
    expectedStates.push_back(ExpectedState{2, 12, {3, 2, 0, 1}, {{2, 10, 10, 0}, {3, 10, 10, 0}}});

    // At iteration 12, reqs 2 and 3 have completed.
    expectedStates.push_back(ExpectedState{12, 20, {0, 1}, {{0, 8, 12, 0}, {1, 8, 12, 0}}});

    // Callback to call after scheduling requests
    int numIterations = runTest(capacityScheduler, kvCacheManager, initActiveRequests, expectedStates, addNewRequestsCb,
        maxInputLen, peftCacheManager);

    EXPECT_EQ(numIterations, 20);
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitAddingNewRequestsMaxUtilizationPriorities)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler
            = CapacityScheduler(maxNumRequests, CapacitySchedulerPolicy::kMAX_UTILIZATION, kvCacheManager != nullptr);

        // Initially two requests that will not fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList initActiveRequests;
        insertRequestInOrder(initActiveRequests, createRequest(promptLen, maxNewTokens, 0));
        insertRequestInOrder(initActiveRequests, createRequest(promptLen, maxNewTokens, 1));

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this, promptLen, maxNewTokens](RequestList& activeRequests, int itCount)
        {
            if (itCount == 10)
            {
                insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 2, std::nullopt, 0.6));
                insertRequestInOrder(activeRequests, createRequest(promptLen, maxNewTokens, 3, std::nullopt, 0.7));
            }
        };

        std::vector<ExpectedState> expectedStates;
        if (sinkTokenLen == 0)
        {
            // Up to iteration 10, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 80, 10, 0}, {1, 80, 10, 0}}});

            // At iteration 10, two more request get added, fits in KV cache
            expectedStates.push_back(ExpectedState{
                10, 21, {3, 2, 0, 1}, {{0, 80, 10, 10, 10}, {1, 80, 10, 10, 10}, {2, 80, 10, 0}, {3, 80, 10, 0}}});

            // At iteration 21, running out of kv cache
            // Req 0 and 1 used 31 tokens (62 tokens, 6 blocks)
            // Req 2 and 3 used 21 tokens (42 tokens, 4 blocks)
            // Each sequence needs 1 block -> Need to drop a sequence. The lowest priority one is req 1,
            // which has priority 0.5 and arrived later than req 0. It liberates 3 blocks, enough to continue
            // with the three ongoing reqs for another 10 tokens for 3 reqs.
            expectedStates.push_back(
                ExpectedState{21, 31, {3, 2, 0, 1}, {{0, 80, 10, 21, 10}, {2, 80, 10, 11, 10}, {3, 80, 10, 11, 10}}});

            // At iteration 31, running out of Kv cache again
            // Req 0 used 41 tokens (4 blocks)
            // Req 2 and 3 used 31 tokens (62 tokens, 6 blocks)
            // Freeing req 0, with priority 0.5, liberates 4 blocks, good for another 20 tokens for 2 reqs.
            expectedStates.push_back(ExpectedState{31, 51, {3, 2, 0, 1}, {{2, 80, 10, 21, 10}, {3, 80, 10, 21, 10}}});

            // At iteration 51, running out of Kv cache again
            // Reqs 2 and 3 used 51 tokens (102 tokens, 10 blocks)
            // Req 2 has priority 0.6 and req 3 has priority 0.7, so req 2 will be freed, liberating 5 blocks.
            expectedStates.push_back(ExpectedState{51, 90, {3, 2, 0, 1}, {{3, 80, 10, 41, 10}}});

            // At it 90, req 3 is done, 100 free kv tokens (10 blocks).
            // req 2 has 51 tokens (needs 6 blocks)
            // req 0 has 41 tokens (needs 5 blocks)
            // req 1 has 31 tokens (needs 4 blocks)
            // Therefore, only req 2 will be scheduled (req 1 will not be scheduled as it would jump the order).
            expectedStates.push_back(ExpectedState{90, 129, {2, 0, 1}, {{2, 39, 51, 0}}});

            // At it 129, req 2 is done
            // req 0 has 41 tokens (needs 5 blocks)
            // req 1 has 31 tokens (needs 4 blocks)
            // Both req 0 and 1 are scheduled.
            expectedStates.push_back(ExpectedState{129, 139, {0, 1}, {{0, 49, 41, 0}, {1, 59, 31, 0}}});

            // At it 139
            // req 0 has 51 tokens (needs 6 blocks)
            // req 1 has 41 tokens (needs 5 blocks)
            // Therefore we need to drop one, which is req 1 due to same priority / later arrival.
            expectedStates.push_back(ExpectedState{139, 178, {0, 1}, {{0, 49, 41, 10, 41}}});

            // At it 178, req 0 is done
            // req 1 has 41 tokens and is scheduled at long last.
            expectedStates.push_back(ExpectedState{178, 227, {1}, {{1, 49, 41, 0}}});
        }
        else if (sinkTokenLen == 4)
        {
            // Up to iteration 10, kvCache is big enough, expect 2 requests
            expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 80, 10, 0, 0}, {1, 80, 10, 0, 0}}});
            // At iteration 10, two more request get added, fits in KV cache
            expectedStates.push_back(ExpectedState{
                10, 15, {3, 2, 0, 1}, {{0, 80, 10, 10, 10}, {1, 80, 10, 10, 10}, {2, 80, 10, 0}, {3, 80, 10, 0}}});

            // At iteration 15, running out of kv cache
            // Req 0 and 1 used 31 tokens (62 tokens), including 6 bubble tokens
            // Req 2 and 3 used 21 tokens (42 tokens), including 6 bubble tokens
            // Each sequence need 1 block -> Need to drop the lowest prio: req 1
            expectedStates.push_back(
                ExpectedState{15, 25, {3, 2, 0, 1}, {{0, 80, 10, 15, 10}, {2, 80, 10, 5, 10}, {3, 80, 10, 5, 10}}});

            // At iteration 25, running out of Kv cache again
            // Req 0 used 41 tokens, including 6 bubble tokens
            // Req 2 and 3 used 31 tokens, including 6 bubble tokens
            // Free req 0, which has the lowest priority.
            expectedStates.push_back(ExpectedState{25, 45, {3, 2, 0, 1}, {{2, 80, 10, 15, 10}, {3, 80, 10, 15, 10}}});

            // At it 45, running out of Kv cache again
            // Req 2 and 3 used 51 tokens, including 6 bubble tokens
            // Free req 2 (lowest priority)
            expectedStates.push_back(ExpectedState{45, 90, {3, 2, 0, 1}, {{3, 80, 10, 35, 10}}});

            // At it 90, req 3 finished, 100 tokens up for grabs
            // req 2 used 51 tokens (including 6 bubble tokens, always assumed from now) (requires 6 blocks)
            // req 0 used 41 tokens (requires 5 blocks)
            // req 1 used 31 tokens (requires 4 blocks)
            // Only req 2 can be scheduled now (higher priority).
            expectedStates.push_back(ExpectedState{90, 135, {2, 0, 1}, {{2, 45, 45, 0}}});

            // At it 135, req 2 is done
            // req 0 used 41 tokens (requires 5 blocks)
            // req 1 used 31 tokens (requires 4 blocks)
            // Both can be scheduled for 10 tokens
            expectedStates.push_back(ExpectedState{135, 145, {0, 1}, {{0, 55, 35, 0}, {1, 65, 25, 0}}});

            // At it 139, run out of kv cache again
            // req 0 used 51 tokens (requires 6 blocks)
            // req 1 used 41 tokens (requires 5 blocks)
            // Free req 1 (lowest priority)
            expectedStates.push_back(ExpectedState{145, 190, {0, 1}, {{0, 55, 35, 10, 35}}});

            // At it 200, req 0 is done
            // Only req 1 is remaining -> 41 tokens generated
            expectedStates.push_back(ExpectedState{190, 245, {1}, {{1, 55, 35, 0}}});
        }

        // Callback to call after scheduling requests
        int numIterations = runTest(capacityScheduler, kvCacheManager, initActiveRequests, expectedStates,
            addNewRequestsCb, maxInputLen, peftCacheManager);

        int expectedNumIters = (sinkTokenLen == 0) ? 227 : 245;
        EXPECT_EQ(numIterations, expectedNumIters);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitAddingNewRequestsGuaranteedCompletion)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler = CapacityScheduler(
            maxNumRequests, CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, kvCacheManager != nullptr);

        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList initActiveRequests;
        initActiveRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        initActiveRequests.push_back(createRequest(promptLen, maxNewTokens, 1));

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this, promptLen, maxNewTokens](RequestList& activeRequests, int itCount)
        {
            if (itCount == 10)
            {
                activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2));
                activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3));
            }
        };

        std::vector<ExpectedState> expectedStates;
        // Up to iteration 10, expect 1 scheduled, 2 active
        expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 80, 10, 0}}});
        // At iteration 80, 1 request is done
        expectedStates.push_back(ExpectedState{10, 80, {0, 1, 2, 3}, {{0, 80, 10, 10, 10}}});
        expectedStates.push_back(ExpectedState{80, 160, {1, 2, 3}, {{1, 80, 10, 0}}});
        expectedStates.push_back(ExpectedState{160, 240, {2, 3}, {{2, 80, 10, 0}}});
        expectedStates.push_back(ExpectedState{240, 320, {3}, {{3, 80, 10, 0}}});

        // Callback to call after scheduling requests
        int numIterations = runTest(capacityScheduler, kvCacheManager, initActiveRequests, expectedStates,
            addNewRequestsCb, maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, 320);
    }
}

TEST_F(CapacitySchedulerTest, SimpleDoesntFitAddingNewRequestsGuaranteedCompletionInChunk)
{
    SizeType32 kvCacheMaxNumTokens = 100;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 4;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler = CapacityScheduler(
            maxNumRequests, CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, kvCacheManager != nullptr);

        int32_t maxNewTokens = 60;
        int32_t promptLen = 30;

        RequestList initActiveRequests;
        auto request0 = createRequest(promptLen, maxNewTokens, 0);
        auto request1 = createRequest(promptLen, maxNewTokens, 1);
        request0->setContextChunkSize(20);
        request1->setContextChunkSize(20);
        initActiveRequests.push_back(std::move(request0));
        initActiveRequests.push_back(std::move(request1));

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this, promptLen, maxNewTokens](RequestList& activeRequests, int itCount)
        {
            if (itCount == 10)
            {
                auto request2 = createRequest(promptLen, maxNewTokens, 2);
                auto request3 = createRequest(promptLen, maxNewTokens, 3);
                request2->setContextChunkSize(20);
                request3->setContextChunkSize(20);
                activeRequests.push_back(std::move(request2));
                activeRequests.push_back(std::move(request3));
            }
        };

        std::vector<ExpectedState> expectedStates;
        // Up to iteration 10, expect 1 scheduled, 2 active
        expectedStates.push_back(ExpectedState{0, 10, {0, 1}, {{0, 60, 30, 0}}});
        // At iteration 80, 1 request is done
        expectedStates.push_back(ExpectedState{10, 61, {0, 1, 2, 3}, {{0, 60, 30, 9, 30}}});
        expectedStates.push_back(ExpectedState{61, 122, {1, 2, 3}, {{1, 60, 30, 0}}});
        expectedStates.push_back(ExpectedState{122, 183, {2, 3}, {{2, 60, 30, 0}}});
        expectedStates.push_back(ExpectedState{183, 244, {3}, {{3, 60, 30, 0}}});

        // Callback to call after scheduling requests
        int numIterations = runTest(capacityScheduler, kvCacheManager, initActiveRequests, expectedStates,
            addNewRequestsCb, maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, 244);
    }
}

TEST_F(CapacitySchedulerTest, DelayDuplicateRequest)
{
    SizeType32 kvCacheMaxNumTokens = 200;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 50;
    SizeType32 maxNumRequests = 3;
    SizeType32 maxInputLen = 1000;
    bool enableReuse = true;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{CapacitySchedulerPolicy::kMAX_UTILIZATION,
        CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, CapacitySchedulerPolicy::kSTATIC_BATCH};
    auto sinkTokenLens = std::vector<SizeType32>{0}; // sinkTokenLen > 0 is not supported with KV cache reuse
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens,
                kvCacheMaxNumTokensPerSeq, sinkTokenLen, enableReuse);
            auto peftCacheManager = getPeftCacheManager();
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 2;
            int32_t promptLen = 2 * kvCacheTokensPerBlock
                + 1; // must be one greater than kvCacheTokensPerBlock because we don't reuse last input token

            RequestList activeRequests;
            auto inputTokens = std::make_shared<std::vector<int32_t>>(promptLen, 1);
            std::iota(inputTokens->begin(), inputTokens->end(), 0);

            activeRequests.push_back(createRequest(inputTokens, maxNewTokens, 0, 1234));
            activeRequests.push_back(createRequest(inputTokens, maxNewTokens, 1, 1234));
            activeRequests.push_back(createRequest(inputTokens, maxNewTokens, 2, 1234));

            std::vector<ExpectedState> expectedStates;
            // No delay in static batching.
            if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kSTATIC_BATCH)
            {
                expectedStates.push_back(ExpectedState{0, maxNewTokens, {0, 1, 2},
                    {{0, maxNewTokens, promptLen, 0}, {1, maxNewTokens, promptLen, 0},
                        {2, maxNewTokens, promptLen, 0}}});
            }
            else
            {
                expectedStates.push_back(ExpectedState{0, 1, {0, 1, 2}, {{0, maxNewTokens, promptLen, 0}}});
                expectedStates.push_back(ExpectedState{1, maxNewTokens, {0, 1, 2},
                    {{0, maxNewTokens, promptLen, 1, promptLen}, {1, maxNewTokens, promptLen, 0},
                        {2, maxNewTokens, promptLen, 0}}});
                expectedStates.push_back(ExpectedState{maxNewTokens, maxNewTokens + 1, {1, 2},
                    {{1, maxNewTokens, promptLen, maxNewTokens - 1, promptLen},
                        {2, maxNewTokens, promptLen, maxNewTokens - 1, promptLen}}});
            }

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager);

            if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kSTATIC_BATCH)
            {
                EXPECT_EQ(numIterations, maxNewTokens);
            }
            else
            {
                EXPECT_EQ(numIterations, maxNewTokens + 1);
            }
            EXPECT_EQ(kvCacheManager->getNumReusedBlocks(), promptLen / kvCacheTokensPerBlock * 2);
        }
    }
}

TEST_F(CapacitySchedulerTest, DelayDuplicateRequestChunked)
{
    SizeType32 kvCacheMaxNumTokens = 200;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;
    bool enableReuse = true;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{
        CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT, CapacitySchedulerPolicy::kMAX_UTILIZATION};
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, 0, enableReuse);
        auto peftCacheManager = getPeftCacheManager();
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that should fit in kvCache for entire duration
        int32_t maxNewTokens = 40;
        int32_t promptLen = 50;

        RequestList activeRequests;
        auto inputTokens0 = std::make_shared<std::vector<int32_t>>(promptLen, 1);
        std::iota(inputTokens0->begin(), inputTokens0->end(), 0);
        auto request0 = createRequest(inputTokens0, maxNewTokens, 0, 1234);
        auto inputTokens1 = std::make_shared<std::vector<int32_t>>(promptLen, 1);
        std::iota(inputTokens1->begin(), inputTokens1->end(), 0);
        auto request1 = createRequest(inputTokens1, maxNewTokens, 1, 1234);
        request0->setContextChunkSize(20);
        request1->setContextChunkSize(20);
        activeRequests.push_back(std::move(request0));
        activeRequests.push_back(std::move(request1));

        std::vector<ExpectedState> expectedStates;
        // No delay in static batching.
        if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kSTATIC_BATCH)
        {
            expectedStates.push_back(ExpectedState{0, 41, {0, 1}, {{0, 40, 50, 0, 0}, {1, 40, 50, 0, 0}}});
        }
        else
        {
            expectedStates.push_back(ExpectedState{0, 1, {0, 1}, {{0, 40, 50, 0, 0}}});
            expectedStates.push_back(ExpectedState{1, 2, {0, 1}, {{0, 40, 50, 0, 20}}});
            expectedStates.push_back(ExpectedState{2, 3, {0, 1}, {{0, 40, 50, 0, 40}}});
            expectedStates.push_back(ExpectedState{3, 4, {0, 1}, {{0, 40, 50, 1, 50}, {1, 40, 50, 0, 0}}});
            expectedStates.push_back(ExpectedState{4, 42, {0, 1}, {{0, 40, 50, 2, 50}, {1, 40, 50, 1, 50}}});
            expectedStates.push_back(ExpectedState{42, 43,
                {
                    1,
                },
                {{1, 40, 50, 39, 50}}});
        }
        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        if (capacitySchedulerPolicy == CapacitySchedulerPolicy::kSTATIC_BATCH)
        {
            EXPECT_EQ(numIterations, 41);
            EXPECT_EQ(kvCacheManager->getNumReusedBlocks(), 0);
        }
        else
        {
            EXPECT_EQ(numIterations, 43);
            EXPECT_EQ(kvCacheManager->getNumReusedBlocks(), 4);
        }
    }
}

TEST_F(CapacitySchedulerTest, DelayFiveRequestsComplicated)
{
    SizeType32 kvCacheMaxNumTokens = 1000;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 150;
    SizeType32 maxNumRequests = 5;
    SizeType32 maxInputLen = 1000;
    bool enableReuse = true;

    auto capacitySchedulerPolicies = std::vector<CapacitySchedulerPolicy>{
        CapacitySchedulerPolicy::kMAX_UTILIZATION, CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT};
    auto sinkTokenLens = std::vector<SizeType32>{0}; // sinkTokenLen > 0 is not supported with KV cache reuse
    for (auto capacitySchedulerPolicy : capacitySchedulerPolicies)
    {
        for (auto sinkTokenLen : sinkTokenLens)
        {
            auto kvCacheManager = getKvCacheManager(maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens,
                kvCacheMaxNumTokensPerSeq, sinkTokenLen, enableReuse);
            auto peftCacheManager = getPeftCacheManager();
            auto capacityScheduler
                = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

            // Create two requests that should fit in kvCache for entire duration
            int32_t maxNewTokens = 80;
            int32_t promptLen = kvCacheTokensPerBlock
                + 1; // must be one greater than kvCacheTokensPerBlock because we don't reuse last input token

            std::vector<int32_t> promptLens(5);
            RequestList activeRequests;
            // produce 2 requests with unique system prompts
            for (int i = 0; i < 2; ++i)
            {
                int32_t promptLen = kvCacheTokensPerBlock * (i + 1) + 1;
                promptLens[i] = promptLen;
                auto inputTokens = std::make_shared<std::vector<int32_t>>(promptLen, 1);
                std::iota(inputTokens->begin(), inputTokens->end(), i + 1);
                activeRequests.push_back(createRequest(inputTokens, maxNewTokens, i, 1234));
            }
            // produce 3 requests with matching system prompts
            for (int i = 0; i < 3; ++i)
            {
                int32_t promptLen = kvCacheTokensPerBlock * (i + 1) + 1;
                promptLens[2 + i] = promptLen;
                auto inputTokens = std::make_shared<std::vector<int32_t>>(promptLen, 1);
                std::iota(inputTokens->begin(), inputTokens->end(), 0);
                activeRequests.push_back(createRequest(inputTokens, maxNewTokens, 2 + i, 1234));
            }

            std::vector<ExpectedState> expectedStates;
            expectedStates.push_back(ExpectedState{0, 1, {0, 1, 2, 3, 4},
                {{0, maxNewTokens, promptLens[0], 0, 0}, {1, maxNewTokens, promptLens[1], 0, 0},
                    {2, maxNewTokens, promptLens[2], 0, 0}}});
            expectedStates.push_back(ExpectedState{1, 2, {0, 1, 2, 3, 4},
                {{0, maxNewTokens, promptLens[0], 1, promptLens[0]}, {1, maxNewTokens, promptLens[1], 1, promptLens[1]},
                    {2, maxNewTokens, promptLens[2], 1, promptLens[2]}, {3, maxNewTokens, promptLens[3], 0, 0}}});
            expectedStates.push_back(ExpectedState{2, 80, {0, 1, 2, 3, 4},
                {{0, maxNewTokens, promptLens[0], 2, promptLens[0]}, {1, maxNewTokens, promptLens[1], 2, promptLens[1]},
                    {2, maxNewTokens, promptLens[2], 2, promptLens[2]},
                    {3, maxNewTokens, promptLens[3], 1, promptLens[3]}, {4, maxNewTokens, promptLens[4], 0, 0}}});
            expectedStates.push_back(ExpectedState{80, 81, {3, 4},
                {{3, maxNewTokens, promptLens[3], 79, promptLens[3]},
                    {4, maxNewTokens, promptLens[4], 78, promptLens[4]}}});
            expectedStates.push_back(ExpectedState{81, 82, {4}, {{4, maxNewTokens, promptLens[4], 79, promptLens[4]}}});

            // Callback to call at each iteration, to have option to add new active Requests
            auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

            int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates,
                addNewRequestsCb, maxInputLen, peftCacheManager);

            EXPECT_EQ(numIterations, 82);
            EXPECT_EQ(kvCacheManager->getNumReusedBlocks(), 3);
        }
    }
}

TEST_F(CapacitySchedulerTest, SimpleFitsStaticBatch)
{
    SizeType32 kvCacheMaxNumTokens = 200;
    SizeType32 kvCacheTokensPerBlock = 10;
    SizeType32 kvCacheMaxNumTokensPerSeq = 90;
    SizeType32 maxNumRequests = 2;
    SizeType32 maxInputLen = 1000;

    // TODO: Support and add coverage for sinkTokenLen > 0
    auto sinkTokenLens = std::vector<SizeType32>{0};
    for (auto sinkTokenLen : sinkTokenLens)
    {
        auto kvCacheManager = getKvCacheManager(
            maxNumRequests, kvCacheTokensPerBlock, kvCacheMaxNumTokens, kvCacheMaxNumTokensPerSeq, sinkTokenLen);
        auto peftCacheManager = getPeftCacheManager();
        CapacitySchedulerPolicy capacitySchedulerPolicy = CapacitySchedulerPolicy::kSTATIC_BATCH;
        auto capacityScheduler = CapacityScheduler(maxNumRequests, capacitySchedulerPolicy, kvCacheManager != nullptr);

        // Create two requests that will fit in kvCache for entire duration
        int32_t maxNewTokens = 80;
        int32_t promptLen = 10;

        RequestList activeRequests;
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
        activeRequests.push_back(createRequest(promptLen, maxNewTokens / 2, 1)); // This request finishes earlier.
        activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2));

        std::vector<ExpectedState> expectedStates;
        // Two requests are scheduled together. When both of them finish, the 3rd one is scheduled.
        expectedStates.push_back(ExpectedState{0, 40, {0, 1, 2}, {{0, 80, 10, 0}, {1, 40, 10, 0}}});
        expectedStates.push_back(ExpectedState{41, 79, {0, 2}, {{0, 80, 10, 41, 10}}});
        expectedStates.push_back(ExpectedState{80, 160, {2}, {{2, 80, 10, 0}}});

        // Callback to call at each iteration, to have option to add new active Requests
        auto addNewRequestsCb = [this](RequestList& activeRequests, int itCount) {};

        int numIterations = runTest(capacityScheduler, kvCacheManager, activeRequests, expectedStates, addNewRequestsCb,
            maxInputLen, peftCacheManager);

        EXPECT_EQ(numIterations, 160);
    }
}
