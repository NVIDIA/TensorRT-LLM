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

#include <algorithm>
#include <queue>
#include <random>
#include <tuple>

namespace tensorrt_llm::batch_manager
{
namespace agent_tree
{

static SizeType32 ROOT_LEVEL = -1;
static SizeType32 RANDOM_SEED = 42;

RequestVector AgentTreeNode::getRequests()
{
    std::unordered_map<NodeType, RequestVector> typeToChildReqs;
    typeToChildReqs.reserve(mTypeToChild.size());
    for (auto const& [nodeType, childNode] : mTypeToChild)
    {
        typeToChildReqs[nodeType] = childNode->getRequests();
    }

    return mergeNodesSequence(typeToChildReqs, mReqs);
}

void AgentTreeNode::insertRequest(LlmRequestPtr const& request)
{
    TLLM_CHECK_WITH_INFO(request != nullptr, "Request cannot be null");

    // req does not have agent hierarchy
    if (getLevelNum(request) == 0)
    {
        if (mLevel == -1)
        {
            TLLM_CHECK_WITH_INFO(mTypeToChild.find(NodeType::kCHATBOT) != mTypeToChild.end(), "Chatbot node not found");
            mTypeToChild[NodeType::kCHATBOT]->insertRequest(request);
        }
        else
        {
            mReqs.push_back(request);
        }
    }
    else if (mLevel == getLevelNum(request) - 1)
    {
        mReqs.push_back(request);
    }
    else
    {
        SizeType32 nextLevel = mLevel + 1;
        NodeType nodeType = getNodeType(request, nextLevel);

        auto it = mTypeToChild.find(nodeType);
        TLLM_CHECK_WITH_INFO(it != mTypeToChild.end(), "Node type %d not found in children at level %d",
            static_cast<SizeType32>(nodeType), nextLevel);

        it->second->insertRequest(request);
    }
}

void AgentTreeNode::clear()
{
    mReqs.clear();
    for (auto const& [nodeType, childNode] : mTypeToChild)
    {
        childNode->clear();
    }
}

std::shared_ptr<NodeParams> AgentTreeNode::getChildNodeParams(
    NodeType childNodeType, std::shared_ptr<NodeParams> const& nodeParams)
{
    return nullptr;
}

AgentChatbotNode::AgentChatbotNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams)
    : AgentTreeNode(level, nodeParams)
{
    TLLM_CHECK_WITH_INFO(nodeParams != nullptr, "AgentChatbotNode requires non-null nodeParams");

    auto agentChatbotParams = std::dynamic_pointer_cast<AgentChatbotNodeParams>(nodeParams);
    TLLM_CHECK_WITH_INFO(agentChatbotParams != nullptr, "Invalid node params for AgentChatbotNode");

    // Set agent ratio from the converted params
    mAgentRatio = agentChatbotParams->getAgentRatio();

    // Create child nodes for all specified agent types (skip chatbot as it will be added separately)
    for (NodeType childType : agentChatbotParams->getChildNodeTypes())
    {
        auto childParams = getChildNodeParams(childType, nodeParams);
        mTypeToChild.emplace(childType, createNode(childType, mLevel + 1, childParams));
    }

    // Always add chatbot node as the default fallback child (ensures uniqueness and proper placement)
    mTypeToChild.emplace(NodeType::kCHATBOT, createNode(NodeType::kCHATBOT, mLevel + 1, nullptr));
}

std::shared_ptr<NodeParams> AgentChatbotNode::getChildNodeParams(
    NodeType childNodeType, std::shared_ptr<NodeParams> const& nodeParams)
{
    return nullptr;
}

RequestVector AgentChatbotNode::mergeRequestsByReqId(std::vector<RequestVector const*> const& reqVectors)
{
    if (reqVectors.size() == 1)
    {
        return *reqVectors[0];
    }

    std::vector<RequestVector const*> nonEmptyVectors;
    nonEmptyVectors.reserve(reqVectors.size());
    for (auto const* reqVec : reqVectors)
    {
        if (!reqVec->empty())
        {
            nonEmptyVectors.push_back(reqVec);
        }
    }

    SizeType32 totalReqNum = 0;
    for (auto const* reqVec : nonEmptyVectors)
    {
        totalReqNum += reqVec->size();
    }
    RequestVector result;
    result.reserve(totalReqNum);

    using MergeItem = std::tuple<uint64_t, size_t, size_t, LlmRequestPtr>;
    auto cmp = [](MergeItem const& a, MergeItem const& b) { return std::get<0>(a) > std::get<0>(b); };
    std::priority_queue<MergeItem, std::vector<MergeItem>, decltype(cmp)> pq(cmp);

    size_t activeVectors = nonEmptyVectors.size();
    for (size_t i = 0; i < nonEmptyVectors.size(); ++i)
    {
        auto const& firstReq = (*nonEmptyVectors[i])[0];
        pq.emplace(firstReq->mRequestId, i, 0, firstReq);
    }

    while (!pq.empty())
    {
        auto [reqId, listIdx, itemIdx, req] = pq.top();
        pq.pop();

        result.push_back(req);

        size_t nextIdx = itemIdx + 1;

        if (nextIdx < nonEmptyVectors[listIdx]->size())
        {
            auto const& nextReq = (*nonEmptyVectors[listIdx])[nextIdx];
            pq.emplace(nextReq->mRequestId, listIdx, nextIdx, nextReq);
        }
        else
        {
            --activeVectors;
            if (activeVectors == 1 && !pq.empty())
            {
                auto [reqId, listIdx, itemIdx, req] = pq.top();
                auto const* lastVec = nonEmptyVectors[listIdx];
                result.insert(result.end(), lastVec->begin() + itemIdx, lastVec->end());
            }
        }
    }

    return result;
}

RequestVector AgentChatbotNode::mixRequestsByRatio(
    RequestVector const& primary, RequestVector const& secondary, float primaryRatio)
{
    RequestVector result;
    result.reserve(primary.size() + secondary.size());

    auto primaryIt = primary.begin();
    auto secondaryIt = secondary.begin();

    // Use fixed seed to ensure deterministic behavior across different machines
    static thread_local std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    while (primaryIt != primary.end() && secondaryIt != secondary.end())
    {
        if (dist(gen) < primaryRatio)
        {
            result.push_back(*primaryIt++);
        }
        else
        {
            result.push_back(*secondaryIt++);
        }
    }

    result.insert(result.end(), primaryIt, primary.end());
    result.insert(result.end(), secondaryIt, secondary.end());

    return result;
}

RequestVector AgentChatbotNode::mergeNodesSequence(
    std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs)
{
    TLLM_CHECK_WITH_INFO(reqs.empty(), "AgentChatbotNode should not have self requests");

    std::vector<RequestVector const*> agentReqsVector;
    for (auto const& [type, reqList] : typeToChildReqs)
    {
        if (type != NodeType::kCHATBOT)
        {
            agentReqsVector.push_back(&reqList);
        }
    }

    RequestVector agentReqs = mergeRequestsByReqId(agentReqsVector);
    RequestVector chatbotReqs = typeToChildReqs.at(NodeType::kCHATBOT);

    return mixRequestsByRatio(agentReqs, chatbotReqs, mAgentRatio);
}

AgentLatencyNode::AgentLatencyNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams)
    : AgentTreeNode(level, nodeParams)
{
}

RequestVector AgentLatencyNode::mergeNodesSequence(
    std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs)
{
    TLLM_CHECK_WITH_INFO(typeToChildReqs.empty(), "AgentLatencyNode should not have child nodes");

    RequestVector allReqs = reqs;

    std::sort(allReqs.begin(), allReqs.end(),
        [this](LlmRequestPtr const& a, LlmRequestPtr const& b) { return getNodeId(a, mLevel) < getNodeId(b, mLevel); });

    return allReqs;
}

AgentDeepResearchNode::AgentDeepResearchNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams)
    : AgentTreeNode(level, nodeParams)
{
}

RequestVector AgentDeepResearchNode::mergeNodesSequence(
    std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs)
{
    TLLM_CHECK_WITH_INFO(typeToChildReqs.empty(), "AgentDeepResearchNode should not have child nodes");
    RequestVector allReqs = reqs;

    std::sort(allReqs.begin(), allReqs.end(),
        [this](LlmRequestPtr const& a, LlmRequestPtr const& b) { return getNodeId(a, mLevel) < getNodeId(b, mLevel); });

    return allReqs;
}

ChatbotNode::ChatbotNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams)
    : AgentTreeNode(level, nodeParams)
{
}

RequestVector ChatbotNode::mergeNodesSequence(
    std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs)
{
    TLLM_CHECK_WITH_INFO(typeToChildReqs.empty(), "ChatbotNode should not have child nodes");
    return reqs;
}

std::shared_ptr<AgentTreeNode> createAgentTreeRoot(
    std::optional<batch_scheduler::AgentTreeConfig> const& agentTreeConfig)
{
    if (!agentTreeConfig.has_value())
    {
        return nullptr;
    }

    auto const& config = agentTreeConfig.value();
    if (!config.agentTypes.has_value() || config.agentTypes->empty())
    {
        return nullptr;
    }

    auto childNodeTypes = convertAgentTypes(config.agentTypes);
    auto nodeParams = std::make_shared<AgentChatbotNodeParams>(config.agentPercentage, childNodeTypes);
    auto root = createNode(NodeType::kAGENT_CHATBOT, ROOT_LEVEL, nodeParams);

    // Store the max requests limit in the root node
    root->setMaxRequests(config.agentInflightSeqNum);

    return root;
}

RequestVector sortAndTruncateRequestsByAgentTree(
    std::shared_ptr<AgentTreeNode> const& root, RequestVector const& requests, SizeType32 reservedCount)
{
    if (requests.empty() || !root)
    {
        return requests;
    }

    for (auto const& req : requests)
    {
        root->insertRequest(req);
    }

    auto sortedRequests = root->getRequests();

    root->clear();

    // Limit the number of returned requests based on root's maxRequests setting.
    // reservedCount is the number of requests already reserved (e.g., generation requests
    // that must be scheduled), so we can only accept (maxRequests - reservedCount) more.
    auto const maxRequests = root->getMaxRequests();
    auto const availableSlots = std::max(0, maxRequests - reservedCount);
    if (sortedRequests.size() > static_cast<size_t>(availableSlots))
    {
        sortedRequests.resize(availableSlots);
        TLLM_LOG_DEBUG("sortAndTruncateRequestsByAgentTree: Limited to %d requests (maxRequests=%d, reserved=%d)",
            availableSlots, maxRequests, reservedCount);
    }

    return sortedRequests;
}

std::shared_ptr<AgentTreeNode> createNode(NodeType nodeType, SizeType32 level, std::shared_ptr<NodeParams> nodeParams)
{
    return NodeFactory::instance().createNode(nodeType, level, nodeParams);
}

NodeType getNodeType(LlmRequestPtr const& request, SizeType32 level)
{
    TLLM_CHECK_WITH_INFO(request != nullptr, "Request cannot be null");

    auto const& agentHierarchyRaw = request->getAgentHierarchy();
    TLLM_CHECK_WITH_INFO(agentHierarchyRaw.has_value(), "Request must have agent hierarchy");

    return stringToNodeType(std::get<0>(agentHierarchyRaw->at(level)));
}

SizeType32 getNodeId(LlmRequestPtr const& request, SizeType32 level)
{
    auto const& agentHierarchyRaw = request->getAgentHierarchy();
    TLLM_CHECK_WITH_INFO(agentHierarchyRaw.has_value(), "Request must have agent hierarchy");

    return std::get<1>(agentHierarchyRaw->at(level));
}

SizeType32 getLevelNum(LlmRequestPtr const& request)
{
    auto const& agentHierarchy = request->getAgentHierarchy();
    if (!agentHierarchy.has_value() || agentHierarchy->size() == 0)
    {
        return 0;
    }
    return static_cast<SizeType32>(agentHierarchy->size());
}

// Register all node types - uses AgentTreeNode::createInstance<T> automatically
// To add a new node type: just add one line here!
REGISTER_NODE_TYPE(ChatbotNode, NodeType::kCHATBOT, "chatbot")
REGISTER_NODE_TYPE(AgentChatbotNode, NodeType::kAGENT_CHATBOT, "agent_chatbot")
REGISTER_NODE_TYPE(AgentLatencyNode, NodeType::kAGENT_LATENCY, "agent_latency")
REGISTER_NODE_TYPE(AgentDeepResearchNode, NodeType::kAGENT_DEEP_RESEARCH, "agent_deep_research")

} // namespace agent_tree
} // namespace tensorrt_llm::batch_manager
