/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/agentTree.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/executor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace tb = tensorrt_llm::batch_manager;
namespace tbat = tensorrt_llm::batch_manager::agent_tree;

using VecTokens = tb::LlmRequest::VecTokens;
using SizeType32 = tb::LlmRequest::SizeType32;
using LlmRequestPtr = std::shared_ptr<tb::LlmRequest>;
using RequestVector = std::vector<LlmRequestPtr>;
using TensorPtr = std::shared_ptr<tr::ITensor>;
using StreamPtr = std::shared_ptr<tr::CudaStream>;

class AgentTreeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    void TearDown() override {}

    LlmRequestPtr createRequestWithHierarchy(
        SizeType32 requestId, std::optional<std::vector<std::tuple<std::string, int>>> agentHierarchy)
    {
        VecTokens inputTokens{1, 2, 3, 4, 5};
        SizeType32 maxNewTokens = 60;
        tr::SamplingConfig samplingConfig(1);

        return std::make_shared<tb::LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, false,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
            std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt,
            tensorrt_llm::executor::Request::kDefaultPriority, std::nullopt, std::nullopt, std::nullopt,
            tb::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, 1, std::nullopt, std::nullopt,
            false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, agentHierarchy);
    }

    LlmRequestPtr createAgentLatencyRequest(SizeType32 nodeId, SizeType32 requestId)
    {
        std::vector<std::tuple<std::string, int>> agentHierarchy;
        agentHierarchy.emplace_back("agent_latency", nodeId);
        return createRequestWithHierarchy(requestId, std::make_optional(std::move(agentHierarchy)));
    }

    LlmRequestPtr createAgentDeepResearchRequest(SizeType32 nodeId, SizeType32 requestId)
    {
        std::vector<std::tuple<std::string, int>> agentHierarchy;
        agentHierarchy.emplace_back("agent_deep_research", nodeId);
        return createRequestWithHierarchy(requestId, std::make_optional(std::move(agentHierarchy)));
    }

    LlmRequestPtr createChatbotRequest(SizeType32 nodeId, SizeType32 requestId)
    {
        std::vector<std::tuple<std::string, int>> agentHierarchy;
        agentHierarchy.emplace_back("chatbot", nodeId);
        return createRequestWithHierarchy(requestId, std::make_optional(std::move(agentHierarchy)));
    }

    LlmRequestPtr createRequestWithoutHierarchy(SizeType32 requestId)
    {
        return createRequestWithHierarchy(requestId, std::nullopt);
    }

    SizeType32 getRandomNodeId()
    {
        std::uniform_int_distribution<SizeType32> dist(0, 1000000);
        return dist(gen);
    }

    RequestVector createAgentChatbotRequests(SizeType32 agentReqNum, SizeType32 chatbotReqNum)
    {
        RequestVector requests;
        requests.reserve(agentReqNum + chatbotReqNum);

        for (SizeType32 i = 0; i < agentReqNum; ++i)
        {
            SizeType32 nodeId = i;
            SizeType32 requestId = getRandomNodeId();
            requests.push_back(createAgentLatencyRequest(nodeId, requestId));
        }

        for (SizeType32 i = 0; i < chatbotReqNum; ++i)
        {
            SizeType32 nodeId = getRandomNodeId();
            SizeType32 requestId = getRandomNodeId();
            requests.push_back(createChatbotRequest(nodeId, requestId));
        }

        return requests;
    }

    RequestVector sortByAgentTree(float agentRatio, std::optional<std::vector<std::string>> const& agentTypes,
        RequestVector const& requests, SizeType32 reservedCount = 0)
    {
        auto rootNode = tbat::createAgentTreeRoot(agentRatio, agentTypes);
        return tbat::sortAndTruncateRequestsByAgentTree(rootNode, requests, reservedCount);
    }

    void checkRequestSequence(
        RequestVector const& sortedReqs, float agentRatio, SizeType32 agentReqNum, SizeType32 chatbotReqNum)
    {
        SizeType32 checkNum;
        if (agentReqNum * agentRatio < chatbotReqNum * (1.0f - agentRatio))
        {
            checkNum = static_cast<SizeType32>(chatbotReqNum / (1.0f - agentRatio));
        }
        else
        {
            checkNum = static_cast<SizeType32>(agentReqNum / agentRatio);
        }

        ASSERT_GT(checkNum, 0) << "checkNum should be greater than 0";
        ASSERT_LE(checkNum, static_cast<SizeType32>(sortedReqs.size()))
            << "checkNum should be less than the number of requests";

        SizeType32 agentCount = 0;
        for (SizeType32 i = 0; i < checkNum; ++i)
        {
            auto const& req = sortedReqs[i];
            auto nodeType = tbat::getNodeType(req, 0);
            if (nodeType == tbat::NodeType::kAGENT_LATENCY)
            {
                agentCount++;
            }
        }

        float realAgentRatio = static_cast<float>(agentCount) / static_cast<float>(checkNum);

        float atol = 0.1f;
        float rtol = 0.1f;
        float tolerance = atol + rtol * std::abs(agentRatio);

        EXPECT_NEAR(realAgentRatio, agentRatio, tolerance)
            << "real_agent_ratio=" << realAgentRatio << ", agent_ratio=" << agentRatio;

        std::vector<SizeType32> agentNodeIds;
        for (auto const& req : sortedReqs)
        {
            auto nodeType = tbat::getNodeType(req, 0);
            if (nodeType == tbat::NodeType::kAGENT_LATENCY)
            {
                SizeType32 nodeId = tbat::getNodeId(req, 0);
                agentNodeIds.push_back(nodeId);
            }
        }

        bool isSorted = std::is_sorted(agentNodeIds.begin(), agentNodeIds.end());
        EXPECT_TRUE(isSorted) << "Agent node_ids should be sorted in ascending order";
    }

    std::mt19937 gen;
};

TEST_F(AgentTreeTest, AgentChatbotNodeBasic)
{
    float agentRatio = 0.7f;
    SizeType32 agentReqNum = 100;
    SizeType32 chatbotReqNum = 100;

    auto agentTypes = std::make_optional<std::vector<std::string>>({"agent_latency"});
    auto requests = createAgentChatbotRequests(agentReqNum, chatbotReqNum);
    auto sortedReqs = sortByAgentTree(agentRatio, agentTypes, requests);

    EXPECT_EQ(sortedReqs.size(), agentReqNum + chatbotReqNum);
    checkRequestSequence(sortedReqs, agentRatio, agentReqNum, chatbotReqNum);
}

TEST_F(AgentTreeTest, EmptyRequestList)
{
    float agentRatio = 0.7f;
    auto agentTypes = std::make_optional<std::vector<std::string>>({"agent_latency"});

    RequestVector emptyRequests;
    auto sortedReqs = sortByAgentTree(agentRatio, agentTypes, emptyRequests);

    EXPECT_TRUE(sortedReqs.empty()) << "Sorted requests should be empty when input is empty";
}

TEST_F(AgentTreeTest, RequestWithoutAgentHierarchy)
{
    auto reqWithoutHierarchy = createRequestWithoutHierarchy(0);

    EXPECT_FALSE(reqWithoutHierarchy->getAgentHierarchy().has_value()) << "Request should not have agent_hierarchy";

    auto agentReq1 = createAgentLatencyRequest(1, 1);
    auto agentReq2 = createAgentLatencyRequest(2, 2);
    auto chatbotReq = createChatbotRequest(3, 3);

    auto agentTypes = std::make_optional<std::vector<std::string>>({"agent_latency"});
    RequestVector requests = {reqWithoutHierarchy, agentReq1, chatbotReq, agentReq2};

    auto sortedReqs = sortByAgentTree(0.5f, agentTypes, requests);
    EXPECT_EQ(sortedReqs.size(), requests.size()) << "Should return all requests including one without hierarchy";

    bool foundReqWithoutHierarchy = false;
    for (auto const& req : sortedReqs)
    {
        if (req->mRequestId == reqWithoutHierarchy->mRequestId)
        {
            foundReqWithoutHierarchy = true;
            break;
        }
    }
    EXPECT_TRUE(foundReqWithoutHierarchy) << "Request without hierarchy should still be in the sorted results";
}

TEST_F(AgentTreeTest, EmptyAgentTypes)
{
    float agentRatio = 0.7f;
    auto emptyAgentTypes = std::make_optional<std::vector<std::string>>(std::vector<std::string>{});

    auto requests = createAgentChatbotRequests(10, 10);
    auto sortedReqs = sortByAgentTree(agentRatio, emptyAgentTypes, requests);

    EXPECT_EQ(sortedReqs.size(), requests.size()) << "Should return all requests";
    EXPECT_EQ(sortedReqs, requests) << "Should return requests unchanged when agentTypes is empty";
}

TEST_F(AgentTreeTest, NulloptAgentTypes)
{
    float agentRatio = 0.7f;
    std::optional<std::vector<std::string>> nulloptAgentTypes = std::nullopt;

    auto requests = createAgentChatbotRequests(5, 5);
    auto sortedReqs = sortByAgentTree(agentRatio, nulloptAgentTypes, requests);

    EXPECT_EQ(sortedReqs.size(), requests.size()) << "Should return all requests";
    EXPECT_EQ(sortedReqs, requests) << "Should return requests unchanged when agentTypes is nullopt";
}

TEST_F(AgentTreeTest, AgentDeepResearchNodeBasic)
{
    // Test that AgentDeepResearchNode inherits AgentLatencyNode behavior correctly
    auto agentTypes = std::make_optional<std::vector<std::string>>({"agent_deep_research"});

    // Create deep research requests with ascending node IDs
    RequestVector requests;
    for (SizeType32 i = 0; i < 10; ++i)
    {
        requests.push_back(createAgentDeepResearchRequest(i, getRandomNodeId()));
    }

    auto sortedReqs = sortByAgentTree(0.5f, agentTypes, requests);

    // Verify all requests are returned
    EXPECT_EQ(sortedReqs.size(), requests.size());

    // Verify node IDs are sorted (inherited from AgentLatencyNode)
    std::vector<SizeType32> nodeIds;
    for (auto const& req : sortedReqs)
    {
        auto nodeType = tbat::getNodeType(req, 0);
        EXPECT_EQ(nodeType, tbat::NodeType::kAGENT_DEEP_RESEARCH) << "Node type should be AGENT_DEEP_RESEARCH";

        SizeType32 nodeId = tbat::getNodeId(req, 0);
        nodeIds.push_back(nodeId);
    }

    bool isSorted = std::is_sorted(nodeIds.begin(), nodeIds.end());
    EXPECT_TRUE(isSorted) << "Deep research node IDs should be sorted in ascending order";
}
