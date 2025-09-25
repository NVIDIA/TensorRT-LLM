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

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::executor::kv_cache;

bool needSkipTest(std::string& skipReason)
{
    bool skip = false;
    try
    {
        auto& loader = tensorrt_llm::executor::kv_cache::DynLibLoader::getInstance();

        using CreateNixlFuncType = std::unique_ptr<tensorrt_llm::executor::kv_cache::BaseTransferAgent> (*)(
            tensorrt_llm::executor::kv_cache::BaseAgentConfig const*);
        auto* func = loader.getFunctionPointer<CreateNixlFuncType>(
            "libtensorrt_llm_nixl_wrapper.so", "createNixlTransferAgent");
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (error.find("libtensorrt_llm_nixl_wrapper.so") != std::string::npos)
        {
            skip = true;
            skipReason = error;
        }
    }
    return skip;
}

class AgentCommTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::string skipReason;
        if (needSkipTest(skipReason))
        {
            GTEST_SKIP() << skipReason;
        }
        setenv("TRTLLM_USE_NIXL_KVCACHE", "1", 1);

        auto constexpr numLayers = 8;
        auto constexpr numHeads = 16;
        auto constexpr sizePerHead = 1024;
        auto constexpr tokensPerBlock = 32;
        auto constexpr maxBlocksPerSeq = 10;
        auto constexpr maxBeamWidth = 4;
        auto constexpr sinkTokenLength = 0;
        auto constexpr maxNumSequences = 8;
        auto constexpr cacheType = CacheType::kSELF;
        auto const stream = std::make_shared<CudaStream>();

        auto kvMaxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto maxAttentionWindow = kvMaxNumTokens;
        auto inputLength = kvMaxNumTokens - tokensPerBlock - 1;
        auto numSharedBlocks = inputLength / tokensPerBlock;
        auto numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = maxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = true;
        auto constexpr onboardBlocks = true;
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        BlocksPerWindow const blocksPerWindow
            = {{maxAttentionWindow, std::make_tuple(totalNumBlocks, blocksInSecondaryPool)}};

        mCacheManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            std::nullopt, dataType, sinkTokenLength, stream, kvMaxNumTokens, enableBlockReuse, onboardBlocks, cacheType,
            std::nullopt, nullptr, true);

        mCacheManager->allocatePools(false);

        size_t maxNumTokens = 1024;
        mTransBufferManager = std::make_unique<CacheTransBufferManager>(mCacheManager.get(), maxNumTokens);
        mCacheState = std::make_unique<CacheState>(
            numLayers, numHeads, sizePerHead, tokensPerBlock, 1, 1, 1, std::vector<SizeType32>{numLayers}, dataType);
    }

    void TearDown() override
    {
        mTransBufferManager.reset();
        mCacheManager.reset();
        mCacheState.reset();
    }

    std::unique_ptr<CacheTransBufferManager> mTransBufferManager;
    std::unique_ptr<KVCacheManager> mCacheManager;
    std::unique_ptr<CacheState> mCacheState;
};

TEST_F(AgentCommTest, AgentConnectionManagerBasic)
{
    auto connectionManager = std::make_unique<AgentConnectionManager>(mTransBufferManager.get(), *mCacheState);
    ASSERT_TRUE(connectionManager != nullptr);
    ASSERT_TRUE(connectionManager->getCacheTransBufferManager() != nullptr);
    ASSERT_EQ(connectionManager->getDeviceId(), 0);
    ASSERT_TRUE(!connectionManager->getAgentName().empty());
    ASSERT_TRUE(connectionManager->getAgent() != nullptr);
    CommState commState = connectionManager->getCommState();
    ASSERT_TRUE(commState.isAgentState());
    ASSERT_EQ(commState.getAgentState().size(), 1);
}

TEST_F(AgentCommTest, AgentConnectionManagerConnect)
{
    auto connectionManager0 = std::make_unique<AgentConnectionManager>(mTransBufferManager.get(), *mCacheState);
    auto connectionManager1 = std::make_unique<AgentConnectionManager>(mTransBufferManager.get(), *mCacheState);
    auto agentName0 = connectionManager0->getAgentName();
    auto agentName1 = connectionManager1->getAgentName();
    ASSERT_TRUE(!agentName0.empty());
    ASSERT_TRUE(!agentName1.empty());
    ASSERT_TRUE(agentName0 != agentName1);

    auto commState0 = connectionManager0->getCommState();
    auto commState1 = connectionManager1->getCommState();
    ASSERT_TRUE(commState0.isAgentState());
    ASSERT_TRUE(commState1.isAgentState());
    ASSERT_EQ(commState0.getAgentState().size(), 1);
    ASSERT_EQ(commState1.getAgentState().size(), 1);

    auto connection0 = connectionManager0->getConnections(commState1).at(0);

    uint64_t requestId = 2;
    auto cacheState0 = *mCacheState;
    auto cacheState1 = *mCacheState;
    tensorrt_llm::executor::DataTransceiverState dataTransceiverState0{cacheState0, commState0};
    tensorrt_llm::executor::DataTransceiverState dataTransceiverState1{cacheState1, commState1};
    tensorrt_llm::batch_manager::RequestInfo sendRequestInfo{requestId, dataTransceiverState0};
    size_t cacheBufferId = 0;
    int validConnectionIdx = 0;
    // convert to AgentConnection
    auto agentConnection0 = const_cast<tensorrt_llm::executor::kv_cache::AgentConnection*>(
        dynamic_cast<tensorrt_llm::executor::kv_cache::AgentConnection const*>(connection0));
    agentConnection0->sendRequestAndBufferInfo(sendRequestInfo, cacheBufferId, validConnectionIdx);

    tensorrt_llm::batch_manager::RequestInfo recvRequestInfo;
    auto connection1 = connectionManager1->recvConnectionAndRequestInfo(recvRequestInfo);
    ASSERT_EQ(recvRequestInfo.getRequestId(), requestId);

    auto sendBuffer = mTransBufferManager->getSendBuffer(cacheBufferId);
    auto sendSize = 1024;
    std::vector<char> sendData(sendSize);
    std::fill(sendData.begin(), sendData.end(), 'a');

    TLLM_CUDA_CHECK(cudaMemcpy(sendBuffer->data(), sendData.data(), sendSize, cudaMemcpyHostToDevice));
    DataContext dataContext{static_cast<int>(requestId)};
    auto future = std::async(std::launch::async,
        [&]()
        {
            TLLM_CUDA_CHECK(cudaSetDevice(0));
            connection1->send(dataContext, sendBuffer->data(), sendSize);
        });
    connection0->recv(dataContext, nullptr, 0);

    future.wait();

    auto recvBuffer = mTransBufferManager->getRecvBuffer(cacheBufferId);
    std::vector<char> recvData(sendSize);
    TLLM_CUDA_CHECK(cudaMemcpy(recvData.data(), recvBuffer->data(), sendSize, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sendSize; i++)
    {
        ASSERT_EQ(recvData[i], 'a');
    }
    TLLM_LOG_INFO("after finish");
}
