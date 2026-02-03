/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#include <cassert>

#include "saFlatHashMap.h"
#include "saTypes.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

template <typename Key, typename NodeData, size_t MaxSize>
struct SAFlatGraph
{

    using Allocator = SAHashMapAllocator<Key, NodeIndex, 10 * MaxSize * (sizeof(Key) + sizeof(NodeIndex))>;

    using HashMap = typename Allocator::HashMap;

    struct Node
    {

        NodeData data;

        bool mIsEmpty = true;
        bool mIsInlined = false;

        Key mInlinedKey;
        NodeIndex mInlinedNodeIndex;
        typename Allocator::Ptr edgesPtr;

        SA_CUDA_CALLABLE bool isEmpty()
        {
            return mIsEmpty;
        }

        SA_CUDA_CALLABLE bool isInlined()
        {
            return mIsInlined;
        }

        SA_CUDA_CALLABLE void setInlinedEdge(std::pair<Key, NodeIndex> edge)
        {
            mIsEmpty = false;
            mIsInlined = true;
            mInlinedKey = edge.first;
            mInlinedNodeIndex = edge.second;
        }

        SA_CUDA_CALLABLE void setEdgesPtr(typename Allocator::Ptr ptr)
        {
            mIsEmpty = false;
            mIsInlined = false;
            edgesPtr = ptr;
        }

        static_assert(std::is_trivially_copyable<NodeData>::value, "NodeData must be trivially copyable");
        static_assert(
            std::is_trivially_copyable<typename Allocator::Ptr>::value, "Allocator::Ptr must be trivially copyable");
        static_assert(std::is_trivially_copyable<NodeIndex>::value, "NodeIndex must be trivially copyable");
    };

    static_assert(std::is_trivially_copyable<Node>::value, "Node must be trivially copyable");

    SADynamicBuffer<Node, MaxSize, NodeIndex> mNodes;
    Allocator mAllocator;

    static_assert(std::is_trivially_copyable<decltype(mNodes)>::value, "mNodes must be trivially copyable");
    static_assert(std::is_trivially_copyable<decltype(mAllocator)>::value, "mAllocator must be trivially copyable");

    template <typename Func>
    void visitChunks(Func&& func) const
    {
        mNodes.visitChunks(func);
        mAllocator.visitChunks(func);
    }

    SA_CUDA_CALLABLE NodeIndex size() const
    {
        return mNodes.size();
    }

    SA_CUDA_CALLABLE bool empty() const
    {
        return mNodes.empty();
    }

    SA_CUDA_CALLABLE NodeIndex* at(NodeIndex nodeId, Key key, bool insertIfNotExists = false)
    {

        assert(nodeId < mNodes.size());

        auto& node = mNodes.at(nodeId);

        if (node.isEmpty())
        {
            if (insertIfNotExists)
            {
                node.setInlinedEdge(std::make_pair(key, NodeIndex(0)));
                return &node.mInlinedNodeIndex;
            }
            else
            {
                return nullptr;
            }
        }

        if (node.isInlined())
        {
            if (node.mInlinedKey == key)
            {
                return &node.mInlinedNodeIndex;
            }

            if (!insertIfNotExists)
            {
                return nullptr;
            }

            auto newEdgesPtr = mAllocator.alloc(6);
            auto& newEdges = mAllocator.at(newEdgesPtr);

            auto* valPtr = newEdges.at(node.mInlinedKey, true);
            assert(valPtr != nullptr);
            *valPtr = node.mInlinedNodeIndex;

            node.setEdgesPtr(newEdgesPtr);
        }

        typename Allocator::Ptr& edgesPtr = node.edgesPtr;
        HashMap& edges = mAllocator.at(edgesPtr);

        NodeIndex* res = nullptr;

        // lookup the key in the hashmap
        res = edges.at(key, insertIfNotExists);

        // if failed to insert, we need to size up
        if (res == nullptr && insertIfNotExists)
        {
            // create a larger hashmap
            size_t newCapacity = edges.capacity() * 5;

            auto newEdgesPtr = mAllocator.alloc(newCapacity);
            HashMap& newEdges = mAllocator.at(newEdgesPtr);

            // copy to the new hashmap
            for (auto kvPair : edges)
            {
                auto* valPtr = newEdges.at(kvPair.first, true);
                // must have space
                assert(valPtr != nullptr);
                *valPtr = kvPair.second;
            }

            // insert the new key
            res = newEdges.at(key, true);
            assert(res != nullptr);

            // free the old hashmap
            mAllocator.free(edgesPtr);

            // set pointer to the new hashmap
            edgesPtr = newEdgesPtr;
        }

        return res;
    }

    SA_CUDA_CALLABLE NodeData& at(NodeIndex nodeId)
    {
        return mNodes.at(nodeId).data;
    }

    SA_CUDA_CALLABLE NodeData const& at(NodeIndex nodeId) const
    {
        return mNodes.at(nodeId).data;
    }

    SA_CUDA_CALLABLE NodeData& pushBack()
    {
        return mNodes.pushBack(Node{}).data;
    }

    SA_CUDA_CALLABLE NodeData& pushBackClone(NodeIndex oldNodeId)
    {
        assert(oldNodeId < mNodes.size());

        auto& oldNode = mNodes.at(oldNodeId);

        if (oldNode.isEmpty() || oldNode.isInlined())
        {
            return mNodes.pushBack(oldNode).data;
        }

        auto& oldEdges = mAllocator.at(oldNode.edgesPtr);

        // allocate hashmap with equal size
        typename Allocator::Ptr newEdgesPtr = mAllocator.alloc(oldEdges.capacity());

        // copy data
        mAllocator.at(newEdgesPtr) = oldEdges;

        Node newNode;
        newNode.data = oldNode.data;
        newNode.setEdgesPtr(newEdgesPtr);

        return mNodes.pushBack(std::move(newNode)).data;
    }

    SA_CUDA_CALLABLE void clear()
    {
        mNodes.clear();
        mAllocator.clear();
    }
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
