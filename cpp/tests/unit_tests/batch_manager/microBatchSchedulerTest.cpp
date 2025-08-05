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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"

#include <optional>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::batch_scheduler;
using namespace tensorrt_llm::executor;

using RequestTable = std::map<RequestIdType, std::shared_ptr<LlmRequest>>;
using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;

class MicroBatchSchedulerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{

protected:
    MicroBatchSchedulerTest() {}

    void SetUp() override {}

    void TearDown() override {}

    static std::shared_ptr<LlmRequest> createRequest(int32_t promptLen, int32_t maxNewTokens,
        std::optional<uint64_t> optionalReqId, SizeType32 beamWidth = 1, int32_t draftTokensLen = 0)
    {
        tensorrt_llm::runtime::SamplingConfig samplingConfig;
        samplingConfig.beamWidth = beamWidth;
        uint64_t reqId = optionalReqId.value_or((rand() % INT64_MAX) + 1);
        auto inputTokens = std::make_shared<std::vector<int32_t>>(promptLen, 1);
        std::optional<std::shared_ptr<std::vector<int32_t>>> draftTokens = std::nullopt;
        std::optional<LlmRequest::TensorPtr> draftLogits = std::nullopt;
        if (draftTokensLen > 0)
        {
            draftTokens = std::make_shared<std::vector<int32_t>>(draftTokensLen, 2);
            draftLogits = BufferManager::cpu(
                ITensor::makeShape({draftTokensLen, /* vocabSizePadded*/ 42}), nvinfer1::DataType::kFLOAT);
        }
        return std::make_shared<LlmRequest>(reqId, maxNewTokens, inputTokens, samplingConfig,
            /*isStreaming=*/false,
            /*endId=*/std::nullopt,
            /*padId=*/std::nullopt, /*embeddingBias=*/std::nullopt,
            /*badWordsList=*/std::nullopt, /*stopWordsList=*/std::nullopt, /*positionIds=*/std::nullopt,
            /*promptEmbeddingTable=*/std::nullopt, /*promptVocabSize=*/std::nullopt,
            /*multimodalHashes=*/std::nullopt, /*multimodalPos=*/std::nullopt, /*multimodalLength=*/std::nullopt,
            /*multimodalEmbedding=*/std::nullopt,
            /*mropeRotaryCosSin=*/std::nullopt, /*mropePositionDeltas*/ std::nullopt,
            /*loraTaskId=*/std::nullopt, /*loraWeights=*/std::nullopt,
            /*loraConfig=*/std::nullopt, /*lookaheadConfig=*/std::nullopt, /*kvCacheRetentionConfig=*/std::nullopt,
            /*returnLogProbs=*/false,
            /*returnContextLogits=*/false, /*returnGenerationLogits=*/false, draftTokens, draftLogits);
    }

    RequestTable forward(
        RequestVector& activeRequests, SizeType32 maxBatchSizeRuntime, std::optional<SizeType32> maxNumTokensRuntime)
    {
        for (auto const& [reqId, req] : mContextRequests.at(mRuntimeContextId))
        {
            mInflightReqIds.erase(reqId);
        }

        auto const [contextRequests, genRequests]
            = (*mMicroBatchScheduler)(activeRequests, mInflightReqIds, maxBatchSizeRuntime, maxNumTokensRuntime);

        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                mInflightReqIds.insert(llmReq->mRequestId);
            }
        }

        // ----------------------------------------
        // Mock the behavior of TrtModelInflightBatching
        // ----------------------------------------

        // Append a token for all exec requests
        for (auto const& llmReq : contextRequests)
        {
            llmReq->moveToNextContextChunk();
            if (!llmReq->getContextRemainingLength())
            {
                llmReq->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
                llmReq->addNewTokens({mItCount});
            }
            if (llmReq->getNumTokens(0) == llmReq->mPromptLen + llmReq->mMaxNewTokens)
            {
                llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
            }
        }
        for (auto const& llmReq : genRequests)
        {
            llmReq->addNewTokens({mItCount});
            if (llmReq->getNumTokens(0) == llmReq->mPromptLen + llmReq->mMaxNewTokens)
            {
                llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
            }
        }

        // Remove completed requests
        auto const activeEnd = std::remove_if(activeRequests.begin(), activeRequests.end(),
            [](auto const& llmReq) { return llmReq->isGenerationCompleteState(); });
        activeRequests.erase(activeEnd, activeRequests.end());

        auto& currContextRequests = mContextRequests.at(mRuntimeContextId);
        currContextRequests.clear();
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                currContextRequests.emplace(llmReq->mRequestId, llmReq);
            }
        }

        // Update the context ID
        mRuntimeContextId = (mRuntimeContextId + 1) % mNumContexts;
        mItCount++;
        return currContextRequests;
    }

    ReqIdsSet mInflightReqIds;
    SizeType32 mItCount{0};
    SizeType32 mRuntimeContextId{0};
    SizeType32 mMaxInputLen = 1000;
    SizeType32 mNumContexts;

    std::vector<RequestTable> mContextRequests;
    std::shared_ptr<MicroBatchScheduler> mMicroBatchScheduler;
};

TEST_F(MicroBatchSchedulerTest, SimpleNoOverlap)
{
    constexpr SizeType32 maxBatchSize = 2;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 80;
    constexpr int32_t promptLen = 10;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3));

    for (int it = 0; it < 170; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, std::nullopt);
        if (it < 80)
        {
            EXPECT_EQ(newRequests.size(), 2) << " in iteration " << it;
            EXPECT_NE(newRequests.find(0), newRequests.end()) << " in iteration " << it;
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), it + 1) << " in iteration " << it;
            EXPECT_NE(newRequests.find(1), newRequests.end()) << " in iteration " << it;
            EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), it + 1) << " in iteration " << it;
        }
        else if (it < 160)
        {
            EXPECT_EQ(newRequests.size(), 2) << " in iteration " << it;
            EXPECT_NE(newRequests.find(2), newRequests.end()) << " in iteration " << it;
            EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), (it - 80 + 1)) << " in iteration " << it;
            EXPECT_NE(newRequests.find(3), newRequests.end()) << " in iteration " << it;
            EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), (it - 80 + 1)) << " in iteration " << it;
        }
        else
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, SimpleNoOverlapMaxNumTokens)
{
    constexpr SizeType32 maxBatchSize = 2;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;
    constexpr SizeType32 maxNumTokens = 7;
    constexpr SizeType32 chunkUnitSize = 5;
    constexpr ContextChunkingPolicy ctxChunkPolicy{ContextChunkingPolicy::kEQUAL_PROGRESS};

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler
        = std::make_shared<MicroBatchScheduler>(ContextChunkingConfig{ctxChunkPolicy, chunkUnitSize}, std::nullopt);

    constexpr int32_t maxNewTokens = 5;

    // Use numbers to represent context tokens and letters to represent generated tokens.
    // Req 0: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, A, B, C, D, E)
    // Req 1: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, A, B, C, D, E)
    RequestVector activeRequests;
    constexpr int32_t promptLen0 = 12;
    constexpr int32_t promptLen1 = 12;
    activeRequests.push_back(createRequest(promptLen0, maxNewTokens, 0));
    activeRequests.push_back(createRequest(promptLen1, maxNewTokens, 1));

    for (int it = 0; it < 9; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it == 0)
        {
            // When it equals 0:
            // Req 0: (0, 1, 2, 3, 4)
            // Req 1: ()
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), 0);
            EXPECT_EQ(newRequests.at(0)->getContextCurrentPosition(), 5);
        }
        else if (it == 1)
        {
            // When it equals 1:
            // Req 0: The last context chunk can be larger than the chunk unit size
            //        and it also satisfies the total token count limit.
            // Req 0: 0, 1, 2, 3, 4; (5, 6, 7, 8, 9, 10, 11, A)
            // Req 1: ()
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), 1);
            EXPECT_EQ(newRequests.at(0)->getContextCurrentPosition(), promptLen0);
        }
        else if (it == 2)
        {
            // When it equals 2:
            // Req 0: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9, 10, 11, A; (B)
            // Req 1: (0, 1, 2, 3, 4)
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), 2);
            EXPECT_EQ(newRequests.at(0)->getContextCurrentPosition(), promptLen0);
            EXPECT_EQ(newRequests.at(1)->getContextCurrentPosition(), chunkUnitSize);
        }
        else if (it == 3)
        {
            // When it equals 3:
            // Req 1: Although the last chunk can be larger than the chunk unit size, it
            //        does not meet the total number of tokens, so it is chunked again.
            // Req 0: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9, 10, 11, A; B; (C)
            // Req 1: 0, 1, 2, 3, 4; (5, 6, 7, 8, 9)
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), 3);
            EXPECT_EQ(newRequests.at(1)->getContextCurrentPosition(), chunkUnitSize * 2);
        }
        else if (it <= 5)
        {
            // When it equals 4:
            // Req 0: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9, A; B; C; D;
            // Req 1: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9; (10, 11, A)
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), it);
            EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), it - 3);
            EXPECT_EQ(newRequests.at(1)->getContextCurrentPosition(), promptLen1);
        }
        else if (it <= 8)
        {
            // When it equals 6:
            // Req 0: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9, A; B; C; D; E;
            // Req 1: 0, 1, 2, 3, 4; 5, 6, 7, 8, 9, 10, 11, A; B; C; (D)
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), it - 3);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, SimpleNoOverlapMaxContextLength)
{
    constexpr SizeType32 maxBatchSize = 2;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;
    constexpr SizeType32 chunkUnitSize = 5;
    constexpr SizeType32 maxContextLength = 12;
    ContextChunkingPolicy ctxChunkPolicy{ContextChunkingPolicy::kEQUAL_PROGRESS};

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler
        = std::make_shared<MicroBatchScheduler>(ContextChunkingConfig{ctxChunkPolicy, chunkUnitSize}, maxContextLength);

    constexpr int32_t maxNewTokens = 5;

    // Use numbers to represent context tokens and letters to represent generated tokens.
    // Req 0, 1, 2: (0, 1, ..., 9, A, B, C, D, E)
    // Req 3: (0, 1, ..., 16, A, B, C, D, E)
    RequestVector activeRequests;
    activeRequests.push_back(createRequest(10, maxNewTokens, 0));
    activeRequests.push_back(createRequest(10, maxNewTokens, 1));
    activeRequests.push_back(createRequest(10, maxNewTokens, 2));
    activeRequests.push_back(createRequest(17, maxNewTokens, 3));

    RequestTable newRequests;
    for (int it = 0; it < 12; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, std::nullopt);
        if (it == 0)
        {
            // The context for requests 0 and 1 can be processed in one batch.
            // Req 0, 1: (0, 1, ..., 9, A)
            // Req 2, 3: ()
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(0)->getContextCurrentPosition(), 10);
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), 1);
            EXPECT_EQ(newRequests.at(1)->getContextCurrentPosition(), 10);
            EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), 1);
        }
        else if (it < 5)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), it + 1);
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), it + 1);
        }
        else if (it == 5)
        {
            // Limited by `maxContextLength`, continued chunking is required for request 3.
            // Req 2: (0, 1, ..., 9, A)
            // Req 3: (0, 1, ..., 9)
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(2)->getContextCurrentPosition(), 10);
            EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), 1);
            EXPECT_EQ(newRequests.at(3)->getContextCurrentPosition(), 10);
            EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), 0);
        }
        else if (it == 6)
        {
            // Req 2: 0, 1, ..., 9, A; (B)
            // Req 3: 0, 1, ..., 9; (10, 11, ..., 16, A)
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_EQ(newRequests.at(2)->getContextCurrentPosition(), 10);
            EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), 2);
            EXPECT_EQ(newRequests.at(3)->getContextCurrentPosition(), 17);
            EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), 1);
        }
        else if (it <= 9)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), (it - 4));
            EXPECT_NE(newRequests.find(3), newRequests.end());
            EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), (it - 5));
        }
        else if (it == 10)
        {
            // Req 3: 0, 1, ..., 9; 10, 11, ..., 16; A; B; C; D; (E)
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_NE(newRequests.find(3), newRequests.end());
            EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), 5);
        }
        else
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, SimpleWithOverlap)
{
    constexpr SizeType32 maxBatchSize = 2;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 2;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 80;
    constexpr int32_t promptLen = 10;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3));

    for (int it = 0; it < 170; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, std::nullopt);
        if (it < 160)
        {
            if (it % 2 == 1)
            {
                // new: 2,3
                EXPECT_EQ(newRequests.size(), 2);
                EXPECT_NE(newRequests.find(2), newRequests.end());
                // EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), (it+1) / 2);
                EXPECT_EQ(newRequests.at(2)->getMaxNumGeneratedTokens(), (it / 2) + 1);
                EXPECT_NE(newRequests.find(3), newRequests.end());
                // EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), (it+1) / 2);
                EXPECT_EQ(newRequests.at(3)->getMaxNumGeneratedTokens(), (it / 2) + 1);
            }
            else
            {
                // new: 0,1
                EXPECT_EQ(newRequests.size(), 2);
                EXPECT_NE(newRequests.find(0), newRequests.end());
                EXPECT_EQ(newRequests.at(0)->getMaxNumGeneratedTokens(), (it / 2) + 1);
                EXPECT_NE(newRequests.find(1), newRequests.end());
                EXPECT_EQ(newRequests.at(1)->getMaxNumGeneratedTokens(), (it / 2) + 1);
            }
        }
        else
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, SimpleMaxNumTokensBW1)
{
    constexpr SizeType32 maxNumTokens = 12;
    constexpr SizeType32 maxBatchSize = 4;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 10;
    constexpr int32_t promptLen = 10;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3, 1));

    for (int it = 0; it < 21; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it < 3)
        {
            EXPECT_EQ(newRequests.size(), it + 1);
        }
        else if (it < 10)
        {
            // Due to limit of 12 tokens, we can only have 1 context and 2 gen, or 3 gen
            // we can't have 1 context + 3 gen
            EXPECT_EQ(newRequests.size(), 3);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
        }
        // At iteration 10, request 0 is done, 3 finally enters
        else if (it < 11)
        {
            EXPECT_EQ(newRequests.size(), 3);
            EXPECT_EQ(newRequests.find(0), newRequests.end());

            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        // At iteration 11, request 1 is done
        else if (it < 12)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        // By iteration 20, all requests are done
        else if (it == 20)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, SimpleMaxNumTokensBW4)
{
    constexpr SizeType32 maxNumTokens = 15;
    constexpr SizeType32 maxBatchSize = 4;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 10;
    constexpr int32_t promptLen = 10;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 4));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 4));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 4));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3, 4));

    //
    for (int it = 0; it < 22; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it < 2)
        {
            EXPECT_EQ(newRequests.size(), it + 1);
        }
        // At iteration 2, we should be limited to only 2 requests (since we have 2 gen, each needing 4 input ids, so
        // adding one more gen would violate constraint)
        else if (it < 10)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
        }
        // At iteration 10, request 0 is done, 2 enters
        else if (it < 11)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
        }
        // At iteration 11, request 1 is done, 3 enters
        else if (it < 20)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        else if (it == 20)
        {
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        else if (it == 21)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, DraftTokensMaxNumTokens)
{
    // This test checks that draft tokens will not cause maxNumTokens to be exceeded.
    constexpr SizeType32 maxNumTokens = 4096;
    constexpr SizeType32 maxBatchSize = 64;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 10;
    constexpr int32_t promptLen = 1024;
    constexpr int32_t draftLen = 17;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3, 1, draftLen));

    for (int it = 0; it < 12; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it == 0)
        {
            // Due to draft tokens, only 3 requests can fit.
            EXPECT_EQ(newRequests.size(), 3);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
        }
        else if (it < 10)
        {
            EXPECT_EQ(newRequests.size(), 4);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        else if (it == 10)
        {
            EXPECT_EQ(newRequests.size(), 1);
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        else if (it == 11)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, GenDraftTokensMaxNumTokens)
{
    // This test checks that draft tokens will not cause maxNumTokens to be exceeded.
    constexpr SizeType32 maxNumTokens = 128;
    constexpr SizeType32 maxBatchSize = 64;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>();

    constexpr int32_t maxNewTokens = 2;
    constexpr int32_t promptLen = 2;
    constexpr int32_t genDraftLen = 63;

    RequestVector activeRequests;
    // No ctx draft tokens.
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3, 1));

    for (int it = 0; it < 4; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it == 0)
        {
            EXPECT_EQ(newRequests.size(), 4);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());

            // Set gen draft tokens.
            newRequests.find(0)->second->setDraftTokens(std::make_shared<std::vector<SizeType32>>(genDraftLen));
            newRequests.find(1)->second->setDraftTokens(std::make_shared<std::vector<SizeType32>>(genDraftLen));
            newRequests.find(2)->second->setDraftTokens(std::make_shared<std::vector<SizeType32>>(genDraftLen));
            newRequests.find(3)->second->setDraftTokens(std::make_shared<std::vector<SizeType32>>(genDraftLen));
        }
        if (it == 1)
        {
            // Due to draft tokens, only 2 gen requests can fit
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
        }
        if (it == 2)
        {
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
        }
        else if (it == 3)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, ChunkedContextDraftTokensMaxNumTokens)
{
    // This test checks that draft tokens will not cause maxNumTokens to be exceeded.
    constexpr SizeType32 maxNumTokens = 8192;
    constexpr SizeType32 maxBatchSize = 64;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    auto ctxChunkConfig = batch_scheduler::ContextChunkingConfig{ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED, 64};
    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>(ctxChunkConfig);

    constexpr int32_t maxNewTokens = 9;
    constexpr int32_t promptLen = 2041;
    constexpr int32_t draftLen = 8;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 3, 1, draftLen));

    for (int it = 0; it < 10; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it < 9)
        {
            // Some draft tokens are discarded so that all requests can fit.
            EXPECT_EQ(newRequests.size(), 4);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_NE(newRequests.find(3), newRequests.end());
            EXPECT_EQ(newRequests.find(0)->second->getNumDraftTokens(), 7);
            EXPECT_EQ(newRequests.find(1)->second->getNumDraftTokens(), 7);
            EXPECT_EQ(newRequests.find(2)->second->getNumDraftTokens(), 7);
            EXPECT_EQ(newRequests.find(3)->second->getNumDraftTokens(), 7);
        }
        else if (it == 9)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, ChunkedContextDraftTokensMaxContextLength)
{
    // This test checks that draft tokens will not cause maxContextLength to be exceeded.
    constexpr SizeType32 maxContextLength = 10;
    constexpr SizeType32 maxBatchSize = 64;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;
    constexpr SizeType32 maxNumTokens = 8192;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    auto ctxChunkConfig = batch_scheduler::ContextChunkingConfig{ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED, 64};
    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>(ctxChunkConfig, maxContextLength);

    constexpr int32_t maxNewTokens = 6;
    constexpr int32_t promptLen = 6;
    constexpr int32_t draftLen = 5;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1, draftLen));

    for (int it = 0; it < 7; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it < 6)
        {
            // Some draft tokens are discarded so that all requests can fit.
            EXPECT_EQ(newRequests.size(), 2);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_EQ(newRequests.find(0)->second->getNumDraftTokens(), 4);
            EXPECT_EQ(newRequests.find(1)->second->getNumDraftTokens(), 4);
        }
        else if (it == 6)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

TEST_F(MicroBatchSchedulerTest, DraftTokensGreaterThanChunkSize)
{
    // This test checks that draft tokens are dropped to fit the request within
    // a chunk unit.
    // TODO(tmorris): This behavior may not be desired and might be changed soon.
    constexpr SizeType32 maxNumTokens = 40;
    constexpr SizeType32 maxBatchSize = 64;
    constexpr SizeType32 chunkUnitSize = 16;
    constexpr uint64_t maxSeqIdleMicroseconds = 60 * 1000 * 1000;

    mNumContexts = 1;
    mContextRequests.resize(mNumContexts);

    auto ctxChunkConfig
        = batch_scheduler::ContextChunkingConfig{ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED, chunkUnitSize};
    mMicroBatchScheduler = std::make_shared<MicroBatchScheduler>(ctxChunkConfig, maxBatchSize);

    constexpr int32_t maxNewTokens = 21;
    constexpr int32_t promptLen = 3;
    constexpr int32_t draftLen = 17;

    RequestVector activeRequests;
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 0, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 1, 1, draftLen));
    activeRequests.push_back(createRequest(promptLen, maxNewTokens, 2, 1, draftLen));

    for (int it = 0; it < 22; ++it)
    {
        RequestTable newRequests = forward(activeRequests, maxBatchSize, maxNumTokens);
        if (it < 21)
        {
            EXPECT_EQ(newRequests.size(), 3);
            EXPECT_NE(newRequests.find(0), newRequests.end());
            EXPECT_NE(newRequests.find(1), newRequests.end());
            EXPECT_NE(newRequests.find(2), newRequests.end());
            EXPECT_EQ(newRequests.find(0)->second->getNumDraftTokens(), 13);
            EXPECT_EQ(newRequests.find(1)->second->getNumDraftTokens(), 13);
            EXPECT_EQ(newRequests.find(2)->second->getNumDraftTokens(), 5);
        }
        else if (it == 21)
        {
            EXPECT_EQ(newRequests.size(), 0);
        }
    }
}

class ContextChunkingTest : public MicroBatchSchedulerTest
{
protected:
    using Policy = ContextChunkingPolicy;

    ~ContextChunkingTest()
    {
        for (auto ctxChunkPolicy : {Policy::kEQUAL_PROGRESS, Policy::kFIRST_COME_FIRST_SERVED})
        {
            auto reqs = initContextLengths(mLengths, mDraftLengths);
            auto const& statesMap = mExpectedStates.at(ctxChunkPolicy);
            EXPECT_GT(statesMap.size(), 0);
            SizeType32 const endIter = statesMap.rbegin()->first + 1;
            for (SizeType32 i = 0; i < endIter; ++i)
            {
                forward(ctxChunkPolicy, reqs, i, statesMap);
            }
            // Check final draft lengths.
            if (mExpectedDraftLengthsMap.count(ctxChunkPolicy))
            {
                auto const& expectedDraftLengths = mExpectedDraftLengthsMap.at(ctxChunkPolicy);
                EXPECT_EQ(reqs.size(), expectedDraftLengths.size());
                for (size_t i = 0; i < reqs.size(); i++)
                {
                    EXPECT_EQ(reqs[i]->getNumDraftTokens(), expectedDraftLengths[i]) << "policy = " << ctxChunkPolicy;
                }
            }
        }
    }

    struct ChunkState
    {
        SizeType32 mContextCurrentPosition;
        LlmRequest::RequestIdType mRequestId;

        friend std::ostream& operator<<(std::ostream& os, ChunkState const& obj)
        {
            os << "pos = " << obj.mContextCurrentPosition << ", id = " << obj.mRequestId;
            return os;
        }
    };

    static RequestVector initContextLengths(
        std::vector<SizeType32> const& lengths, std::vector<SizeType32> const& draftLengths)
    {
        RequestVector reqs;
        constexpr SizeType32 maxNewTokens = 1;
        for (size_t i = 0; i < lengths.size(); ++i)
        {
            auto draftLen = draftLengths.size() > 0 ? draftLengths[i] : 0;
            reqs.push_back(createRequest(lengths[i], maxNewTokens, i, /*beamWidth=*/1, draftLen));
        }
        return reqs;
    }

    static std::string debugStr(LlmRequest const& val)
    {
        std::ostringstream os;
        os << "pos = " << val.getContextCurrentPosition() << ", id = " << val.mRequestId;
        return os.str();
    }

    static bool isEqual(LlmRequest const& source, ChunkState const& target)
    {
        bool ret = true;
        ret = ret && (source.mRequestId == target.mRequestId);
        ret = ret && (source.getContextCurrentPosition() == target.mContextCurrentPosition);
        return ret;
    }

    template <Policy tPolicy>
    void setExpectedPositions(std::vector<std::vector<SizeType32>> const& positions)
    {
        std::vector<ChunkState> stateVec;
        for (size_t iter = 0; iter < positions.size(); ++iter)
        {
            for (size_t i = 0; i < positions[iter].size(); ++i)
            {
                stateVec.emplace_back(ChunkState{positions[iter][i], i});
            }
            mExpectedStates[tPolicy].insert({iter, std::move(stateVec)});
        }
    }

    void forward(Policy ctxChunkPolicy, RequestVector const& reqs, SizeType32 itCount,
        std::map<SizeType32, std::vector<ChunkState>> const& statesMap)
    {

        // Don't process already completed requests
        RequestVector activeRequests;
        std::copy_if(reqs.begin(), reqs.end(), std::back_inserter(activeRequests),
            [](auto const& llmReq) { return llmReq->getContextRemainingLength() > 0; });

        MicroBatchScheduler::setCtxRequestsChunkSize(
            activeRequests, ctxChunkPolicy, mCtxTokensCapacity, mChunkUnitSize, mMaxContextLength);
        for (auto const& req : activeRequests)
        {
            req->moveToNextContextChunk();
        }
        auto stateIt = statesMap.find(itCount);
        if (stateIt != statesMap.end())
        {
            std::vector<ChunkState> const& states = stateIt->second;
            EXPECT_EQ(reqs.size(), states.size());

            auto reqIt = reqs.begin();
            SizeType32 i = 0;
            while (reqIt != reqs.end())
            {
                EXPECT_TRUE(isEqual(**reqIt, states[i]))
                    << "policy = " << ctxChunkPolicy << "; mItCount = " << mItCount << "; actual: " << debugStr(**reqIt)
                    << "; expect: " << states[i];
                ++reqIt;
                ++i;
            }
        }
    }

    void setCtxTokenCapacity(SizeType32 ctxTokensCapacity)
    {
        mCtxTokensCapacity = ctxTokensCapacity;
    }

    void setChunkUnitSize(SizeType32 chunkUnitSize)
    {
        mChunkUnitSize = chunkUnitSize;
    }

    void setMaxContextLength(SizeType32 maxContextLength)
    {
        mMaxContextLength = maxContextLength;
    }

    void setContextLengths(std::vector<SizeType32> lengths)
    {
        mLengths = std::move(lengths);
    }

    void setDraftLengths(std::vector<SizeType32> draftLengths)
    {
        mDraftLengths = std::move(draftLengths);
    }

    template <Policy tPolicy>
    void setExpectedFinalDraftLengths(std::vector<SizeType32> expectedDraftLengths)
    {
        mExpectedDraftLengthsMap[tPolicy] = std::move(expectedDraftLengths);
    }

private:
    std::vector<SizeType32> mLengths;
    std::vector<SizeType32> mDraftLengths;
    std::map<Policy, std::vector<SizeType32>> mExpectedDraftLengthsMap;
    std::map<Policy, std::map<SizeType32, std::vector<ChunkState>>> mExpectedStates;
    std::optional<SizeType32> mCtxTokensCapacity;
    SizeType32 mChunkUnitSize{0};
    std::optional<SizeType32> mMaxContextLength;
};

TEST_F(ContextChunkingTest, NoLimit)
{
    setContextLengths({25, 25});
    setChunkUnitSize(20);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 25}});
}

TEST_F(ContextChunkingTest, ContextLengthNeverSatisfied)
{
    setContextLengths({25, 25});
    setMaxContextLength(20);
    setChunkUnitSize(100);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{0, 0}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{0, 0}});
}

TEST_F(ContextChunkingTest, ChunkLongerThanContext)
{
    setContextLengths({25, 25});
    setMaxContextLength(25);
    setChunkUnitSize(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 25}});
}

TEST_F(ContextChunkingTest, ContextLengthSatisfied)
{
    setContextLengths({10, 25});
    setMaxContextLength(20);
    setChunkUnitSize(10);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{10, 20}, {10, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{10, 20}, {10, 25}});
}

TEST_F(ContextChunkingTest, TokenCapacitySmallerThanContext)
{
    setContextLengths({25, 25});
    setChunkUnitSize(20);
    setCtxTokenCapacity(20);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{20, 0}, {25, 0}, {25, 20}, {25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{20, 0}, {25, 0}, {25, 20}, {25, 25}});
}

TEST_F(ContextChunkingTest, TokenCapacitySmallerThanChunkUnit)
{
    setContextLengths({25, 25});
    setChunkUnitSize(20);
    setCtxTokenCapacity(10);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{0, 0}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{0, 0}});
}

TEST_F(ContextChunkingTest, SchedulingOrder)
{
    setContextLengths({25, 25});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{15, 15}, {25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 5}, {25, 25}});
}

TEST_F(ContextChunkingTest, CompletionOrder)
{
    setContextLengths({25, 15});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{15, 15}, {25, 15}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 5}, {25, 15}});
}

TEST_F(ContextChunkingTest, LongFirstShortLater)
{
    setContextLengths({25, 15});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setMaxContextLength(10);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{10, 10}, {20, 15}, {25, 15}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{10, 10}, {20, 15}, {25, 15}});
}

TEST_F(ContextChunkingTest, FrontPriority)
{
    setContextLengths({25, 25});
    setChunkUnitSize(5);
    setCtxTokenCapacity(15);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{10, 5}, {20, 10}, {25, 20}, {25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{15, 0}, {25, 5}, {25, 20}, {25, 25}});
}

TEST_F(ContextChunkingTest, DraftTokensDiscard)
{
    setContextLengths({27, 27});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{15, 15}, {27, 27}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{27, 0}, {27, 27}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({3, 3});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({3, 3});
}

TEST_F(ContextChunkingTest, DraftTokensDiscard2)
{
    setContextLengths({17, 17});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{15, 15}, {17, 17}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({3, 3});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{17, 10}, {17, 17}});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({3, 3});
}

TEST_F(ContextChunkingTest, DraftTokensDiscard3)
{
    setContextLengths({27, 27});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(20);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{10, 10}, {20, 20}, {27, 27}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{20, 0}, {27, 10}, {27, 27}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({3, 3});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({3, 3});
}

TEST_F(ContextChunkingTest, DraftTokensDiscardDueToTokenCapacity)
{
    setContextLengths({23, 17});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(20);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{10, 10}, {23, 17}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({0, 0});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{20, 0}, {23, 17}});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({0, 0});
}

TEST_F(ContextChunkingTest, DraftTokensDiscardDueToMaxContextLength)
{
    setContextLengths({6, 6});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(30);
    setMaxContextLength(10);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{6, 6}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{6, 6}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({4, 4});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({4, 4});
}

TEST_F(ContextChunkingTest, DraftTokensDiscardAll)
{
    setContextLengths({25, 25});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(50);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 25}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({0, 0});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({0, 0});
}

TEST_F(ContextChunkingTest, DraftTokensDiscardAll2)
{
    setContextLengths({25, 25});
    setDraftLengths({5, 5});
    setChunkUnitSize(5);
    setCtxTokenCapacity(25);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{15, 10}, {25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 0}, {25, 25}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({0, 0});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({0, 0});
}

TEST_F(ContextChunkingTest, DraftTokensNoDiscard)
{
    setContextLengths({25, 25});
    setDraftLengths({5, 5});
    setChunkUnitSize(10);
    setCtxTokenCapacity(30);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{20, 10}, {25, 25}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{25, 0}, {25, 25}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({5, 5});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({5, 5});
}

TEST_F(ContextChunkingTest, DraftTokensNoChunkingDiscardAll)
{
    setContextLengths({4128});
    setDraftLengths({3});
    setChunkUnitSize(64);
    setMaxContextLength(4128);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{4128}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{4128}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({0});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({0});
}

TEST_F(ContextChunkingTest, DraftTokensNoChunkingDiscardSome)
{
    setContextLengths({4127});
    setDraftLengths({3});
    setChunkUnitSize(64);
    setMaxContextLength(4128);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{4127}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{4127}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({1});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({1});
}

TEST_F(ContextChunkingTest, DraftTokensNoChunkingDiscardNone)
{
    setContextLengths({4125});
    setDraftLengths({3});
    setChunkUnitSize(64);
    setMaxContextLength(4128);
    setExpectedPositions<Policy::kEQUAL_PROGRESS>({{4125}});
    setExpectedPositions<Policy::kFIRST_COME_FIRST_SERVED>({{4125}});
    setExpectedFinalDraftLengths<Policy::kEQUAL_PROGRESS>({3});
    setExpectedFinalDraftLengths<Policy::kFIRST_COME_FIRST_SERVED>({3});
}
