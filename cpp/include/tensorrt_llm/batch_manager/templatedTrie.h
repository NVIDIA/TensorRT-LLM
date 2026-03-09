/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>

//
// This file implements a templated trie.
// It is used to implement a constant radix search tree for reusable KV cache blocks.
//
// Template arguments:
//
// The trie is maintained as a set of nodes.
// Each node has a map with pointers to the next node(s) [the children] and a single pointer to the previous node [the
// parent]. The key is of type 'NodeKey'. A hash functor for 'NodeKey' is provided by 'NodeKeyHashFunctor'. The hash
// functor can be std::hash for simple types.
//
// Each node stores a value of type 'Value' for multiple channels. The channel is selected with an argument of type
// 'ValueKey'. For example, if we have one value per window size, ValueKey can be of type int. ValueKey can be any
// custom struct or class that implements == operator and has a hash functor. The hash functor is passed with argument
// 'ValueKeyHashFunctor'. For simple data types, hash functor can be std::hash.
//
// Note that value is stored by value, not by reference. If your value is an object instance, you should use a pointer
// type as your Value argument, for instance std::shared_ptr<KVCacheBlock> can be used to point to the objects that
// contain meta-data for KV cache blocks. If used Value = KVCacheBlock instead of std::shared_ptr<KVCacheBlock>, the
// value that is returned by Node::getValue would be a copy of the meta-data in the KVCacheBlock instance, not a pointer
// to it. This is fine for reading, but will not work if you plan on modifying anything.
//
// NodeKey is a custom data type with a notion of tokens. The NodeKey we use for KV cache block lookups is struct
// BlockKey, which has a property called uniqueTokens. uniqueTokens is a vector of fixed size, the size is equal to
// block length, which is sometimes referred to as 'tokens_per_block'. BlockKey supports partial matching. Partial
// matching means the entire BlockKey did not match, but the first 'N' tokens did. Not all NodeKeys support partial
// matching, you enable or disable this feature with template argument 'EnablePartialMatching'.
//

namespace tensorrt_llm::batch_manager::templated_trie
{

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
class Node;

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
class Trie;

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
struct NodeMatch
{
    using _Node = Node<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using NodePtr = std::shared_ptr<_Node>;

    NodeMatch() = default;

    explicit NodeMatch(NodeKey const& key, NodePtr node, bool exactMatch, bool wasCreated)
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

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
struct NodeMatches
{
    using _NodeMatch
        = NodeMatch<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;

    // Nodes matched exactly from beginning of prefix
    std::vector<_NodeMatch> exactMatches;
    // All partial matches for last node key, sorted in descending order of number of matched tokens
    std::vector<_NodeMatch> partialMatches;
};

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
struct ValueMatch
{
    using _Node = Node<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using NodePtr = std::shared_ptr<_Node>;

    ValueMatch() = default;

    explicit ValueMatch(NodeKey const& nk, NodePtr const& n, ValueKey vk)
        : isValid{false}
        , key{nk}
        , node{n}
        , vkey{vk}
        , exactMatch{false}
    {
    }

    explicit ValueMatch(NodeKey const& nk, NodePtr const& n, ValueKey vk, Value const& v, bool em)
        : isValid{true}
        , key{nk}
        , node{n}
        , vkey{vk}
        , value{v}
        , exactMatch{em}
    {
    }

    bool isValid = false;
    NodeKey key;
    NodePtr node = nullptr;
    ValueKey vkey;
    Value value;
    bool exactMatch = false;
};

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
struct ValueMatches
{
    using _ValueMatch
        = ValueMatch<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    std::vector<_ValueMatch> matches;
};

template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
class Node
{
public:
    using PrefixKey = std::vector<NodeKey>;
    using NodePtr = std::shared_ptr<Node>;
    using _NodeMatch
        = NodeMatch<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using NodeMatchPtr = std::shared_ptr<_NodeMatch>;

    Node() = default;

    explicit Node(NodeKey const& key, NodePtr& prevNode)
        : mKey{key}
        , mPrevNode{prevNode}
    {
    }

    //! \brief Update the back-pointer to this node's parent.
    //! \details Only updates mPrevNode (the back-edge). The caller is responsible for also
    //! updating the old and new parent's mNextNodes forward maps: remove this node from the old
    //! parent's map, and insert it into the new parent's map.
    //! Typical use: detaching a node before eviction, or reparenting during tree restructuring.
    //! Pass an empty NodePtr{} to make this node behave as a root — the cascade-delete in
    //! clearValue / clearNode will stop here instead of propagating upward.
    //! \param prevNode The new parent node. Pass NodePtr{} to detach from the current parent.
    void setPrevNode(NodePtr const& prevNode)
    {
        mPrevNode = prevNode;
    }

    //! \brief Insert child node. Will overwrite existing node with same nkey.
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
            auto const isRoot = mPrevNode.expired();
            auto const canBeDeleted = !isRoot && mValue.empty() && mNextNodes.empty();
            if (canBeDeleted)
            {
                // Node has no values and no descendants. Delete it
                if (auto parent = mPrevNode.lock())
                {
                    auto const wasDeleted = parent->clearNode(mKey);
                    TLLM_CHECK_WITH_INFO(wasDeleted, "cascade prune: parent did not find this node as a child");
                }
            }
            return true;
        }
        return false;
    }

    //! \brief Try to set value for vkey.
    //! \param vkey Key.
    //! \param value Value.
    //! \param overwrite True to allow overwriting an existing value.
    //! \return True if the node was updated (key inserted or existing value overwritten).
    //!         False if the key already existed and overwrite=false (no change made).
    [[nodiscard]] bool trySetValue(ValueKey const& vkey, Value const& value, bool overwrite)
    {
        auto itr = mValue.find(vkey);
        bool priorExists = (itr != mValue.end());
        if (!priorExists)
        {
            mValue.emplace(vkey, value);
        }
        else if (overwrite)
        {
            itr->second = value;
        }
        return !priorExists || overwrite;
    }

    //! \brief Clear value for vkey.
    //! \param vkey Key.
    //! \return True if value was cleared.
    [[nodiscard]] bool clearValue(ValueKey const& vkey)
    {
        auto itr = mValue.find(vkey);
        if (itr != mValue.end())
        {
            mValue.erase(itr);
            auto const isRoot = mPrevNode.expired();
            auto const canBeDeleted = !isRoot && mValue.empty() && mNextNodes.empty();
            if (canBeDeleted)
            {
                // Node has no values and no descendants. Delete it
                if (auto parent = mPrevNode.lock())
                {
                    auto const wasDeleted = parent->clearNode(mKey);
                    TLLM_CHECK_WITH_INFO(wasDeleted, "cascade prune: parent did not find this node as a child");
                }
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
    [[nodiscard]] std::optional<_NodeMatch> findMatchingNode(NodeKey const& key) const
    {
        auto itr = mNextNodes.find(key);
        if (itr != mNextNodes.end())
        {
            return _NodeMatch(itr->first, itr->second, true, false);
        }
        else
        {
            return std::nullopt;
        }
    }

    //! \brief Get the parent node of this node.
    //! \return Shared pointer to parent node, or nullptr if this is the root.
    [[nodiscard]] NodePtr getParentNode() const
    {
        return mPrevNode.lock();
    }

    //! \brief Check if this node has any children.
    //! \return true if this node has at least one child node.
    [[nodiscard]] bool hasChildren() const
    {
        return !mNextNodes.empty();
    }

    //! \brief Get all (key, value) pairs for direct child nodes that have a value for vkey.
    //! \param vkey Value key to look up in each child.
    //! \return Vector of (NodeKey, Value) pairs for children that have a value for vkey.
    [[nodiscard]] std::vector<std::pair<NodeKey, Value>> getChildKeyValues(ValueKey const& vkey) const
    {
        std::vector<std::pair<NodeKey, Value>> results;
        for (auto const& [childKey, childNode] : mNextNodes)
        {
            auto optVal = childNode->getValue(vkey);
            if (optVal.has_value())
            {
                results.emplace_back(childKey, optVal.value());
            }
        }
        return results;
    }

    //! \brief Find an existing child node by key, or insert a new one.
    //! \details If a child with \p key already exists it is returned unchanged.
    //! Otherwise a new child is created, linked to \p self as its parent, inserted into
    //! mNextNodes and returned. The caller is responsible for providing \p self as the
    //! shared_ptr that owns *this (i.e. the caller's NodePtr).
    //! \param key Key of the child to find or create.
    //! \param self shared_ptr to *this node (used as the parent pointer for a new child).
    //! \return NodePtr to the (existing or newly created) child node.
    [[nodiscard]] NodePtr findOrInsertChild(NodeKey const& key, NodePtr const& self)
    {
        auto existing = findMatchingNode(key);
        if (existing.has_value())
        {
            return existing.value().node;
        }
        auto newNode = std::make_shared<Node>(key, const_cast<NodePtr&>(self));
        auto const overwritten = insertNode(key, newNode);
        TLLM_CHECK_WITH_INFO(!overwritten, "findOrInsertChild: inserted a node that already existed");
        return newNode;
    }

    //! \brief Find all partially matching nodes
    //! \param key The key we're matching.
    //! \return vector of matching nodes, sorted in descending order of number of matched tokens.
    [[nodiscard]] std::vector<_NodeMatch> findPartiallyMatchingNodes(NodeKey const& key) const
    {
        if constexpr (EnablePartialMatching)
        {
            // Runtime check: the specific key instance may disable partial matching (e.g. BlockKey with extraKeys
            // for multimodal content). This is by design — returning empty here is correct, not an error, because
            // multimodal blocks require an exact key match and cannot be partially reused.
            if (!key.supportsPartialMatching())
            {
                return {};
            }
            std::vector<std::pair<int, NodePtr>> matches;
            for (auto const& [nodeKey, node] : mNextNodes)
            {
                int numMatched = key.numMatchingTokens(nodeKey);
                if (numMatched > 0)
                {
                    matches.emplace_back(numMatched, node);
                }
            }
            auto results = std::vector<_NodeMatch>();
            if (!matches.empty())
            {
                // Sort in descending order of number of matched tokens (longest match first)
                std::sort(matches.begin(), matches.end(),
                    [](std::pair<int, NodePtr> const& a, std::pair<int, NodePtr> const& b)
                    { return a.first > b.first; });
                // Include meta-data in output
                for (auto match : matches)
                {
                    auto numMatched = match.first;
                    auto fullMatch = numMatched == key.getNumTokens();
                    if (fullMatch)
                    {
                        results.emplace_back(key, match.second, true, false);
                    }
                    else
                    {
                        auto partialKey = key.shorten(numMatched);
                        results.emplace_back(partialKey, match.second, false, false);
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
    friend Trie<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;

    // Private debugging method.
    // Returns the prefix path to every node that holds a value, including nodes that
    // are both terminal (have a value) and internal (have children). Used only in
    // unit tests via getEdges().
    void _getEdges(std::vector<NodeKey> edge, std::vector<std::vector<NodeKey>>& edges) const
    {
        auto const isRoot = mPrevNode.expired();
        if (!isRoot)
        {
            edge.emplace_back(mKey);
        }
        if (!mValue.empty())
        {
            edges.emplace_back(edge);
        }
        for (auto const& [key, node] : mNextNodes)
        {
            node->_getEdges(edge, edges);
        }
    }

    // Private debugging method.
    void _countNodes(int& count) const
    {
        auto const isRoot = mPrevNode.expired();
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
    std::unordered_map<ValueKey, Value, ValueKeyHashFunctor> mValue;

    std::weak_ptr<Node> mPrevNode;
    std::unordered_map<NodeKey, NodePtr, NodeKeyHashFunctor> mNextNodes;
};

//! \brief A radix search tree used to look up values for given prefixes.
//! \tparam NodeKey Single node key. A prefix key is required for lookup from root and is a vector of node key.
//! \tparam NodeKeyHashFunctor Hash functor for node key. Can be std::hash<NodeKey> if std::hash is implemented for
//! NodeKey.
//! \tparam ValueKey Value key. Think of this as a channel selector, for instance multiple window sizes can be
//! handled if ValueKey is an int.
//! \tparam ValueKeyHashFunctor Hash functor for value key. Can be std::hash<ValueKey> if std::hash is implemented for
//! ValueKey.
//! \tparam Value Object holding value.
//! \tparam EnablePartialMatching Set to true to allow the tree to attempt partial matching.
template <class NodeKey, class NodeKeyHashFunctor, class ValueKey, class ValueKeyHashFunctor, class Value,
    bool EnablePartialMatching>
class Trie
{
public:
    using PrefixKey = std::vector<NodeKey>;
    using _Node = Node<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using NodePtr = std::shared_ptr<_Node>;
    using _NodeMatch
        = NodeMatch<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using _NodeMatches
        = NodeMatches<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using NodeMatchesPtr = std::shared_ptr<_NodeMatches>;
    using _ValueMatch
        = ValueMatch<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using _ValueMatches
        = ValueMatches<NodeKey, NodeKeyHashFunctor, ValueKey, ValueKeyHashFunctor, Value, EnablePartialMatching>;
    using ValueMatchesPtr = std::shared_ptr<_ValueMatches>;

    Trie()
        : mRoot{std::make_shared<_Node>()}
    {
    }

    //! \brief Get the root node of the trie.
    //! \return Shared pointer to the root node.
    [[nodiscard]] NodePtr getRoot() const
    {
        return mRoot;
    }

    //! \brief Insert nodes for new prefix, or return existing nodes.
    //! \param key Key for new prefix.
    //! \return  An object containing results + meta-data about how nodes were matched.
    _NodeMatches insertNodes(PrefixKey const& pkey)
    {
        // Return value
        _NodeMatches results;
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
                auto newNode = std::make_shared<_Node>(key, prevNode);
                auto const overwritten = prevNode->insertNode(key, newNode);
                TLLM_CHECK_WITH_INFO(!overwritten, "insertNodes: inserted a node that already existed");
                matchedNode = _NodeMatch(key, newNode, true, true);
            }
            prevNode = matchedNode.value().node;
            results.exactMatches.emplace_back(std::move(matchedNode.value()));
        }
        return results;
    }

    //! \brief Look up nodes matching given key.
    //! \param pkey Key for prefix we are looking for.
    //! \param allowPartialMatch If true, consider partial match for last node.
    //! \return An object containing results of the lookup + meta-data about how nodes were matched.
    [[nodiscard]] _NodeMatches lookupNodes(PrefixKey const& pkey, bool allowPartialMatch) const
    {
        // Return value
        _NodeMatches results;
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
                if (EnablePartialMatching && allowPartialMatch)
                {
                    auto partialMatches = prevNode->findPartiallyMatchingNodes(key);
                    results.partialMatches.insert(
                        results.partialMatches.end(), partialMatches.begin(), partialMatches.end());
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

    //! \brief Get values for given prefix and value key.
    //! \param nodeMatches Nodes matching a given prefix.
    //! \param vkey Value key.
    //! \return ValueMatches struct containing values matching prefix and window size. Might be nullptr for some or all.
    [[nodiscard]] _ValueMatches lookupValues(_NodeMatches const& nodeMatches, ValueKey const& vkey) const
    {
        _ValueMatches valueMatches;
        for (auto const& nodeMatch : nodeMatches.exactMatches)
        {
            auto optValue = nodeMatch.node->getValue(vkey);
            if (optValue.has_value())
            {
                valueMatches.matches.emplace_back(
                    nodeMatch.key, nodeMatch.node, vkey, optValue.value(), nodeMatch.exactMatch);
            }
            else
            {
                valueMatches.matches.emplace_back(nodeMatch.key, nodeMatch.node, vkey);
            }
        }
        for (auto const& nodeMatch : nodeMatches.partialMatches)
        {
            auto optValue = nodeMatch.node->getValue(vkey);
            if (optValue.has_value())
            {
                valueMatches.matches.emplace_back(
                    nodeMatch.key, nodeMatch.node, vkey, optValue.value(), nodeMatch.exactMatch);
                break; // Found longest match, we are done searching for partial match
            }
        }
        return valueMatches;
    }

    //! \brief Get values for given prefix and value key.
    //! \param pkey prefix.
    //! \param allowPartialMatch If partial matching of tokens is allowed for last matching block.
    //! \param vkey Value key.
    //! \return ValueMatches struct containing values matching prefix and window size. Might be nullptr for some or all.
    [[nodiscard]] _ValueMatches lookupValues(PrefixKey const& pkey, bool allowPartialMatch, ValueKey const& vkey) const
    {
        auto nodeMatches = lookupNodes(pkey, allowPartialMatch);
        return lookupValues(nodeMatches, vkey);
    }

private:
    NodePtr mRoot;
};

} // namespace tensorrt_llm::batch_manager::templated_trie
