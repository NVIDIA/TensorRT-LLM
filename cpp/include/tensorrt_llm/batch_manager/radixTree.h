/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <algorithm>
#include <optional>
#include <cstddef>
#include "tensorrt_llm/batch_manager/common.h"

//
// This file implements a templated radix search tree.
// It is used to retrieve KV cache blocks with reusable content during context phase.
//
// Template arguments:
// NodeKey - Single node key. An edge key is needed for a search from root. Edge key is a vector of node keys.
// NodeKeyHashFunctor - Hash functor for node key. If node key is a type supported by std::hash, hash functor can be std::hash<NodeKey>.
// ValueKey - Value key for single node. Think of this as a channel selector, for instance if you have one set of values for each sliding window size, ValueKey can be an int that is the window size.
// ValueKeyHashFunctor - Hash functor for value key. If value key is a type supported by std::hash, hash functor can be std::hash<ValueKey>.
// Value - Value returned for a given edge and value key.
// 

namespace tensorrt_llm::batch_manager::radix_tree
{

template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
class RadixTreeNode;

template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
class RadixTree;

template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
struct RadixTreeMatch
{
    using Node = RadixTreeNode<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;
    using NodePtr = std::shared_ptr<Node>;

    RadixTreeMatch() = default;

    explicit RadixTreeMatch(NodeKey const& key, NodePtr node, bool exactMatch, bool wasCreated)
        : key(key)
          , node(node)
          , exactMatch(exactMatch)
          , wasCreated(wasCreated)
    {
    }

    NodeKey key;
    NodePtr node;
    bool exactMatch;
    bool wasCreated;
};

template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
class RadixTreeNode
{
    public:
        using PrefixKey = std::vector<NodeKey>;
        using NodePtr = std::shared_ptr<RadixTreeNode>;
        using Match = RadixTreeMatch<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;
        using MatchPtr = std::shared_ptr<Match>;

        RadixTreeNode() = default;

        explicit RadixTreeNode(NodeKey const& key, NodePtr& prevNode)
            : mKey{key}
            , mPrevNode{prevNode}
        {
        }

        void setPrevNode(NodePtr& prevNode)
        {
            if (mPrevNode != nullptr)
            {
                mPrevNode->clearValue(mKey);
            }
        }

        //! \brief Insert child node. Will overwrite exisiting node with same nkey.
        //! \param nkey Node key of child node.
        //! \param node Child node.
        //! \return true if node was overwritten in child node map, false if it was inserted.
        [[nodiscard]] bool insertNode(NodeKey const& nkey, NodePtr node)
        {
            auto const& [itr, inserted] = mNextNodes.insert_or_assign(nkey, node);
            return !inserted;
        }

        [[nodiscard]] bool clearNode(NodeKey const& nkey)
        {
            auto itr = mNextNodes.find(nkey);
            if (itr != mNextNodes.end())
            {
                mNextNodes.erase(itr);
                auto const isRoot = mPrevNode == nullptr;
                auto const canBeDeleted = !isRoot && mValue.empty() && mNextNodes.empty();
                if (canBeDeleted)
                {
                    // Node has no values and no descendants. Delete it
                    [[maybe_unused]] auto const wasDeleted = mPrevNode->clearNode(mKey);
                }
                return true;
            }
            return false;
        }

        //! \brief Set value for vkey.
        //! \param vkey Key.
        //! \param value Value.
        //! \param overwrite True to allow overwrite.
        //! \return True if value was overwritten, false otherwise.
        [[nodiscard]] bool setValue(ValueKey const& vkey, Value const& value, bool overwrite)
        {
            if (overwrite)
            {
                auto const& [itr, inserted] = mValue.insert_or_assign(vkey, value);
                return !inserted;
            }
            else
            {
                mValue.try_emplace(vkey, value);
                return false;
            }
        }

        //! \brief Clear value for vkey.
        //! \param vKey Key.
        //! \return True if value was cleared.
        [[nodiscard]] bool clearValue(ValueKey const& vkey)
        {
            auto itr = mValue.find(vkey);
            if (itr != mValue.end())
            {
                mValue.erase(itr);
                auto const isRoot = mPrevNode == nullptr;
                auto const canBeDeleted = !isRoot && mValue.empty() && mNextNodes.empty();
                if (canBeDeleted)
                {
                    // Node has no values and no descendants. Delete it
                    [[maybe_unused]] auto const wasDeleted = mPrevNode->clearNode(mKey);
                }
                return true;
            }
            return false;
        }

        //! \brief Get Value held for vkey.
        //! \param vkey Key to lookup value.
        //! \return Pointer to shared_ptr holding Value. nullptr is returned when vkey is not found.
        [[nodiscard]] std::optional<Value> getValue(ValueKey const& vkey) const
        {
            auto itr = mValue.find(vkey);
            if (itr != mValue.end())
            {
                return itr->second;
            }
            else
            {
                return std::nullopt;
            }
        }

        //! \brief Find exact match
        //! \param key Key we are looking for.
        //! \return Matching node or null_opt if no match
        [[nodiscard]] std::optional<Match> findMatchingNode(NodeKey const& key) const
        {
            auto itr = mNextNodes.find(key);
            if (itr != mNextNodes.end())
            {
                return RadixTreeMatch(itr->first, itr->second, true, false);
            }
            else
            {
                return std::nullopt;
            }
        }

        //! \brief Find all partially matching nodes
        //! \param key The key we're matching.
        //! \return vector of matching nodes, sorted in descending order of number of matched tokens.
        [[nodiscard]] std::vector<Match> findPartiallyMatchingNodes(NodeKey const& key) const
        {
            if constexpr (supportsPartialMatching)
            {
                std::vector<std::tuple<int,NodePtr>> matches;
                for (auto nn : mNextNodes)
                {
                    int numMatched = key.numMatchingTokens(nn->first);
                    if (numMatched > 0)
                    {
                        matches.emplace_back(std::make_tuple<int,NodePtr>(numMatched,nn->second));
                    }
                }
                auto results = std::vector<Match>();
                if (!matches.empty())
                {
                    // Sort in descending order of number of matched tokens (longest match first)
                    std::sort(matches.begin(), matches.end(),
                            [](std::tuple<int, NodePtr> const& a,
                                std::tuple<int, NodePtr> const& b)
                            {
                            [[maybe_unused]] auto [numMatchedA, dummy1] = a;
                            [[maybe_unused]] auto [numMatchedB, dummy2] = b;
                            return numMatchedA > numMatchedB;
                            });
                    // Include meta-data in output
                    for (auto match : matches)
                    {
                        auto numMatched = match->first;
                        auto fullMatch = numMatched == key.getNumTokens();
                        if (fullMatch)
                        {
                            results.emplace_back(key, match->second, true, false);
                        }
                        else
                        {
                            auto partialKey = key.shorten(numMatched);
                            results.emplace_back(partialKey, match->second, false, false);
                        }
                    }
                }
                return results;
            }
            else
            {
                return {};
            }
        }

    private:
        friend RadixTree<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;

        // Private debugging method.
        void _getEdges(std::vector<NodeKey> edge, std::vector<std::vector<NodeKey>>& edges) const
        {
            auto const isRoot = mPrevNode == nullptr;
            if (!isRoot)
            {
                edge.emplace_back(mKey);
            }
            if (!mValue.empty())
            {
                edges.emplace_back(edge);
            }
            else
            {
                for (auto const& [key, node] : mNextNodes)
                {
                    node->_getEdges(edge, edges);
                }
            }
        }

        // Private debugging method.
        void _countNodes(int& count) const
        {
            auto const isRoot = mPrevNode == nullptr;
            if (!isRoot)
            {
                ++count;
            }
            for (auto const& [key, node] : mNextNodes)
            {
                node->_countNodes(count);
            }
        }

    private:
        NodeKey mKey;
        std::unordered_map<ValueKey,Value,ValueKeyHashFunctor> mValue;

        NodePtr mPrevNode = nullptr;
        std::unordered_map<NodeKey,NodePtr,NodeKeyHashFunctor> mNextNodes;
};

template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
struct RadixTreeLookupNodesResult
{
    using Match = RadixTreeMatch<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;

    // Nodes matched exactly from beginning of prefix
    std::vector<Match> exactMatches;
    // All partial matches for last node key, sorted in descending order of number of matched tokens
    std::vector<Match> partialMatches;
};

//! \brief A radix search tree used to look up values for given prefixes.
//! \tparam NodeKey Single node key. A prefix key is required for lookup from root and is a vector of node key.
//! \tparam NodeKeyHashFunctor Hash functor for node key. Can be std::hash<NodeKey> if std::hash is implemented for NodeKey.
//! \tparam ValueKey Value key. Think of this as a channel selector, for instance multiple window sizes can be handled if ValueKey is an int.
//! \tparam ValueKeyHashFunctor Hash functor for value key. Can be std::hash<ValueKey> if std::hash is implemented for ValueKey.
//! \tparam Value Object holding value.
//! \tparam supportsPartialMatching Set to true if NodeKey supports partial matching and you know how to extract a subset of matched tokens from Value.
template<class NodeKey,class NodeKeyHashFunctor,class ValueKey,class ValueKeyHashFunctor,class Value,bool supportsPartialMatching>
class RadixTree
{
    public:
        using PrefixKey = std::vector<NodeKey>;
        using Node = RadixTreeNode<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;
        using NodePtr = std::shared_ptr<Node>;
        using Match = RadixTreeMatch<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;
        using LookupNodesResult = RadixTreeLookupNodesResult<NodeKey,NodeKeyHashFunctor,ValueKey,ValueKeyHashFunctor,Value,supportsPartialMatching>;
        using LookupNodesResultPtr = std::shared_ptr<LookupNodesResult>;
        using Values = std::vector<std::optional<Value>>;

        explicit RadixTree()
            : mRoot{std::make_shared<Node>()}
        {
        }

        //! \brief Insert nodes for new prefix, or return existing nodes.
        //! \param key Key for new prefix.
        //! \return  An object containing results + meta-data about how nodes were matched.
        LookupNodesResult insertNodes(PrefixKey const& pkey)
        {
            // Return value
            LookupNodesResult results;
            // State variables
            bool lookForMatch = true;
            auto prevNode = mRoot;
            for (auto const& key : pkey)
            {
                auto matchedNode = lookForMatch ? prevNode->findMatchingNode(key) : std::nullopt;
                // Create new node if no match was found
                if (!matchedNode.has_value())
                {
                    lookForMatch = false;
                    auto newNode = std::make_shared<Node>(key, prevNode);
                    [[maybe_unused]] auto const overwritten = prevNode->insertNode(key, newNode);
                    matchedNode = Match(key, newNode, true, true);
                }
                prevNode = matchedNode.value().node;
                results.exactMatches.emplace_back(std::move(matchedNode.value()));
            }
            return results;
        }

        //! \brief Look up nodes matching given key.
        //! \param key Key for prefix we are looking for.
        //! \param allowPartialMatch If true, consider partial match for last node.
        //! \return An object containing results of the lookup + meta-data about how nodes were matched.
        [[nodiscard]] LookupNodesResult lookupNodes(PrefixKey const& pkey, bool allowPartialMatch) const
        {
            // Return value
            LookupNodesResult results;
            // State variables
            auto prevNode = mRoot;
            for (auto const& key : pkey)
            {
                auto matchedNode = prevNode->findMatchingNode(key);
                if (matchedNode.has_value())
                {
                    prevNode = matchedNode.value().node;
                    results.exactMatches.push_back(std::move(matchedNode.value()));
                }
                else
                {
                    if (supportsPartialMatching && allowPartialMatch)
                    {
                        auto partialMatches = prevNode->findPartiallyMatchingNodes(key);
                        results.partialMatches.insert(results.partialMatches.end(), partialMatches.begin(), partialMatches.end());
                    }
                    // Stop lookup since no exact match was found
                    break;
                }
            }
            return results;
        }

        //! \brief Get all edges of radix tree. Intended for debugging. An edge is a prefix leading to a value.
        //! \return Vector of edges.
        [[nodiscard]] std::vector<std::vector<NodeKey>> getEdges() const
        {
            std::vector<NodeKey> edge;
            std::vector<std::vector<NodeKey>> edges;
            mRoot->_getEdges(edge, edges);
            return edges;
        }

        //! \brief Get total number of unique nodes in tree. Intended for debugging.
        //! \return Total number of unique nodes in tree.
        [[nodiscard]] int countNumberOfNodes() const
        {
            int count = 0;
            mRoot->_countNodes(count);
            return count;
        }

    private:
        NodePtr mRoot;
};

}

