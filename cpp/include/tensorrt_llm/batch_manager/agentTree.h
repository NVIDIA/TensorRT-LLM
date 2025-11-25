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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager
{
namespace agent_tree
{

using LlmRequestPtr = std::shared_ptr<LlmRequest>;
using SizeType32 = runtime::SizeType32;

enum class NodeType : SizeType32
{
    kCHATBOT = 0,
    kAGENT_CHATBOT = 1,
    kAGENT_LATENCY = 2,
    kAGENT_DEEP_RESEARCH = 3
};

using AgentHierarchyVector = std::vector<std::pair<NodeType, SizeType32>>;

// Forward declarations
class AgentTreeNode;
class NodeParams;

// Node factory with auto-registration support
class NodeFactory
{
public:
    using Creator = std::function<std::shared_ptr<AgentTreeNode>(SizeType32, std::shared_ptr<NodeParams>)>;

    static NodeFactory& instance()
    {
        static NodeFactory factory;
        return factory;
    }

    void registerNode(std::string const& name, NodeType type, Creator creator)
    {
        mStringToType[name] = type;
        mCreators[type] = std::move(creator);
    }

    NodeType stringToNodeType(std::string const& typeStr) const
    {
        auto it = mStringToType.find(typeStr);
        if (it != mStringToType.end())
        {
            return it->second;
        }
        TLLM_CHECK_WITH_INFO(false, "Unknown NodeType string '%s'", typeStr.c_str());
        return NodeType::kCHATBOT;
    }

    std::shared_ptr<AgentTreeNode> createNode(NodeType type, SizeType32 level, std::shared_ptr<NodeParams> params) const
    {
        auto it = mCreators.find(type);
        if (it != mCreators.end())
        {
            return it->second(level, params);
        }
        TLLM_CHECK_WITH_INFO(false, "No creator registered for node type: %d", static_cast<SizeType32>(type));
        return nullptr;
    }

private:
    std::unordered_map<std::string, NodeType> mStringToType;
    std::unordered_map<NodeType, Creator> mCreators;
};

inline NodeType stringToNodeType(std::string const& typeStr)
{
    return NodeFactory::instance().stringToNodeType(typeStr);
}

inline std::vector<NodeType> convertAgentTypes(std::optional<std::vector<std::string>> const& agentTypes)
{
    if (!agentTypes.has_value())
    {
        return {};
    }

    std::vector<NodeType> result;
    result.reserve(agentTypes->size());

    for (auto const& typeStr : agentTypes.value())
    {
        result.emplace_back(stringToNodeType(typeStr));
    }

    return result;
}

class NodeParams
{
public:
    virtual ~NodeParams() = default;
};

class AgentChatbotNodeParams : public NodeParams
{
public:
    AgentChatbotNodeParams(float agentRatio, std::vector<NodeType> const& childNodeTypes)
        : mAgentRatio(agentRatio)
        , mChildNodeTypes(childNodeTypes)
    {
        TLLM_CHECK_WITH_INFO(agentRatio >= 0.0f && agentRatio <= 1.0f, "Invalid agent ratio: %f", agentRatio);
    }

    [[nodiscard]] float getAgentRatio() const noexcept
    {
        return mAgentRatio;
    }

    [[nodiscard]] std::vector<NodeType> const& getChildNodeTypes() const noexcept
    {
        return mChildNodeTypes;
    }

private:
    float mAgentRatio;
    std::vector<NodeType> mChildNodeTypes;
};

class AgentTreeNode
{
public:
    using NodePtr = std::shared_ptr<AgentTreeNode>;
    using NodeMap = std::unordered_map<NodeType, NodePtr>;

    explicit AgentTreeNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams = nullptr)
        : mLevel(level)
    {
    }

    virtual ~AgentTreeNode() = default;

    [[nodiscard]] RequestVector getRequests();

    void insertRequest(LlmRequestPtr const& request);

    void clear();

    [[nodiscard]] SizeType32 getLevel() const noexcept
    {
        return mLevel;
    }

protected:
    [[nodiscard]] virtual RequestVector mergeNodesSequence(
        std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs)
        = 0;

    [[nodiscard]] virtual std::shared_ptr<NodeParams> getChildNodeParams(
        NodeType childNodeType, std::shared_ptr<NodeParams> const& nodeParams);

    NodeMap mTypeToChild;
    RequestVector mReqs;
    SizeType32 mLevel;
};

class AgentChatbotNode : public AgentTreeNode
{
public:
    explicit AgentChatbotNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams);

protected:
    [[nodiscard]] RequestVector mergeNodesSequence(
        std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs) override;

    [[nodiscard]] std::shared_ptr<NodeParams> getChildNodeParams(
        NodeType childNodeType, std::shared_ptr<NodeParams> const& nodeParams) override;

private:
    [[nodiscard]] static RequestVector mergeRequestsByReqId(std::vector<RequestVector const*> const& reqVectors);

    [[nodiscard]] static RequestVector mixRequestsByRatio(
        RequestVector const& primary, RequestVector const& secondary, float primaryRatio);

    float mAgentRatio;
};

class AgentLatencyNode : public AgentTreeNode
{
public:
    explicit AgentLatencyNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams = nullptr);

protected:
    [[nodiscard]] RequestVector mergeNodesSequence(
        std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs) override;
};

class AgentDeepResearchNode : public AgentTreeNode
{
public:
    explicit AgentDeepResearchNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams = nullptr);

protected:
    [[nodiscard]] RequestVector mergeNodesSequence(
        std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs) override;
};

class ChatbotNode : public AgentTreeNode
{
public:
    explicit ChatbotNode(SizeType32 level, std::shared_ptr<NodeParams> nodeParams = nullptr);

protected:
    [[nodiscard]] RequestVector mergeNodesSequence(
        std::unordered_map<NodeType, RequestVector> const& typeToChildReqs, RequestVector const& reqs) override;
};

[[nodiscard]] std::shared_ptr<AgentTreeNode> createNode(
    NodeType nodeType, SizeType32 level, std::shared_ptr<NodeParams> nodeParams = nullptr);

[[nodiscard]] std::shared_ptr<AgentTreeNode> createAgentTreeRoot(
    float agentRatio, std::optional<std::vector<std::string>> const& agentTypes);

[[nodiscard]] RequestVector sortRequestsByAgentTree(
    std::shared_ptr<AgentTreeNode> const& root, RequestVector const& requests);

[[nodiscard]] SizeType32 getLevelNum(LlmRequestPtr const& request);

[[nodiscard]] NodeType getNodeType(LlmRequestPtr const& request, SizeType32 level);

[[nodiscard]] SizeType32 getNodeId(LlmRequestPtr const& request, SizeType32 level);

// Auto-registration macro: call this in .cpp after all class definitions
// Usage: REGISTER_NODE_TYPE(ChatbotNode, NodeType::kCHATBOT, "chatbot")
#define REGISTER_NODE_TYPE(ClassName, EnumType, StringName)                                                            \
    namespace                                                                                                          \
    {                                                                                                                  \
    struct ClassName##Registrar                                                                                        \
    {                                                                                                                  \
        ClassName##Registrar()                                                                                         \
        {                                                                                                              \
            NodeFactory::instance().registerNode(StringName, EnumType,                                                 \
                [](SizeType32 level, std::shared_ptr<NodeParams> params) -> std::shared_ptr<AgentTreeNode>             \
                { return std::make_shared<ClassName>(level, params); });                                               \
        }                                                                                                              \
    };                                                                                                                 \
    static ClassName##Registrar g_##ClassName##_registrar;                                                             \
    }

} // namespace agent_tree
} // namespace tensorrt_llm::batch_manager
