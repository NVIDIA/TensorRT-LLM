/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::executor::kv_cache;

std::vector<std::string> getAvailableBackends()
{
    std::vector<std::string> backends;

#ifdef TEST_NIXL_BACKEND
    backends.push_back("nixl");
#endif

#ifdef TEST_MOONCAKE_BACKEND
    backends.push_back("mooncake");
#endif

    return backends;
}

bool needSkipTest(std::string const& backend, std::string& skipReason)
{
    bool skip = false;
    try
    {
        auto& loader = tensorrt_llm::executor::kv_cache::DynLibLoader::getInstance();

        if (backend == "nixl")
        {
            using CreateNixlFuncType = std::unique_ptr<tensorrt_llm::executor::kv_cache::BaseTransferAgent> (*)(
                tensorrt_llm::executor::kv_cache::BaseAgentConfig const*);
            auto* func = loader.getFunctionPointer<CreateNixlFuncType>(
                "libtensorrt_llm_nixl_wrapper.so", "createNixlTransferAgent");
        }
        else if (backend == "mooncake")
        {
            using CreateMooncakeFuncType = std::unique_ptr<tensorrt_llm::executor::kv_cache::BaseTransferAgent> (*)(
                tensorrt_llm::executor::kv_cache::BaseAgentConfig const*);
            auto* func = loader.getFunctionPointer<CreateMooncakeFuncType>(
                "libtensorrt_llm_mooncake_wrapper.so", "createMooncakeTransferAgent");
        }
        else
        {
            skip = true;
            skipReason = "Unknown backend: " + backend;
        }
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        std::string libName
            = (backend == "nixl") ? "libtensorrt_llm_nixl_wrapper.so" : "libtensorrt_llm_mooncake_wrapper.so";
        if (error.find(libName) != std::string::npos)
        {
            skip = true;
            skipReason = error;
        }
    }
    return skip;
}

class AgentCommTest : public ::testing::TestWithParam<std::string>
{
protected:
    void SetUp() override
    {
        backend = GetParam();
        std::string skipReason;
        if (needSkipTest(backend, skipReason))
        {
            GTEST_SKIP() << skipReason;
        }

        if (backend == "nixl")
        {
            setenv("TRTLLM_USE_NIXL_KVCACHE", "1", 1);
        }
        else if (backend == "mooncake")
        {
            setenv("TRTLLM_USE_MOONCAKE_KVCACHE", "1", 1);
        }

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

    std::string backend;
    std::unique_ptr<CacheTransBufferManager> mTransBufferManager;
    std::unique_ptr<KVCacheManager> mCacheManager;
    std::unique_ptr<CacheState> mCacheState;
};

TEST_P(AgentCommTest, AgentConnectionManagerBasic)
{
    std::vector<CacheTransBufferManager*> bufferManagers{mTransBufferManager.get()};
    auto connectionManager = std::make_unique<AgentConnectionManager>(bufferManagers, *mCacheState, backend);
    ASSERT_TRUE(connectionManager != nullptr);
    ASSERT_EQ(connectionManager->getCacheTransBufferManagers().size(), bufferManagers.size());
    ASSERT_TRUE(connectionManager->getCacheTransBufferManagers().front() != nullptr);
    ASSERT_EQ(connectionManager->getDeviceId(), 0);
    ASSERT_TRUE(!connectionManager->getAgentName().empty());
    ASSERT_TRUE(connectionManager->getAgent() != nullptr);
    CommState commState = connectionManager->getCommState();
    ASSERT_TRUE(commState.isAgentState());
    ASSERT_EQ(commState.getAgentState().size(), 1);
}

TEST_P(AgentCommTest, AgentConnectionManagerConnect)
{
    std::vector<CacheTransBufferManager*> bufferManagers{mTransBufferManager.get()};
    auto connectionManager0 = std::make_unique<AgentConnectionManager>(bufferManagers, *mCacheState, backend);
    auto connectionManager1 = std::make_unique<AgentConnectionManager>(bufferManagers, *mCacheState, backend);
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
    std::vector<std::optional<size_t>> cacheBufferIds{std::optional<size_t>{0}};
    int validConnectionIdx = 0;
    // convert to AgentConnection
    auto agentConnection0 = const_cast<tensorrt_llm::executor::kv_cache::AgentConnection*>(
        dynamic_cast<tensorrt_llm::executor::kv_cache::AgentConnection const*>(connection0));
    agentConnection0->sendRequestAndBufferInfo(sendRequestInfo, cacheBufferIds, validConnectionIdx);

    tensorrt_llm::batch_manager::RequestInfo recvRequestInfo;
    auto connection1 = connectionManager1->recvConnectionAndRequestInfo(recvRequestInfo, std::atomic<bool>(false));
    ASSERT_EQ(recvRequestInfo.getRequestId(), requestId);

    auto sendBuffer = mTransBufferManager->getSendBuffer(cacheBufferIds[0].value());
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

    auto recvBuffer = mTransBufferManager->getRecvBuffer(cacheBufferIds[0].value());
    std::vector<char> recvData(sendSize);
    TLLM_CUDA_CHECK(cudaMemcpy(recvData.data(), recvBuffer->data(), sendSize, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < sendSize; i++)
    {
        ASSERT_EQ(recvData[i], 'a');
    }
    TLLM_LOG_INFO("after finish");
}

INSTANTIATE_TEST_SUITE_P(AvailableBackends, AgentCommTest, ::testing::ValuesIn(getAvailableBackends()),
    [](::testing::TestParamInfo<AgentCommTest::ParamType> const& info) { return info.param; });
