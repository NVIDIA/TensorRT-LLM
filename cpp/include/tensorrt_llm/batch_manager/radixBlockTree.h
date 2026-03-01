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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/templatedTrie.h"

//
// Implementation of constant radix search tree for KV cache blocks.
// It is a unified tree, a single instane of search tree services
// all WindowBlockManager instances. This is achieved by using
// window size as value key.
//

namespace tensorrt_llm::batch_manager::radix_block_tree
{
struct BlockMatch
{
    using BlockPtr = std::shared_ptr<kv_cache_manager::KVCacheBlock>;
    using Node = templated_trie::Node<BlockKey,BlockKeyHasher,int,std::hash<int>,kv_cache_manager::KVCacheBlock,true>;
    using NodePtr = std::shared_ptr<Node>;

    BlockMatch() = default;

    explicit BlockMatch(NodePtr& n, BlockPtr& b, BlocKey& bk, bool em, int ws)
        : node{n}
    , block{b}
    , key{bk}
    , exactMatch{em}
    , windowSize{ws}
    {
    }

    //! Claim the matched block. Block is removed from search tree to prevent further reuse.
    //! \return true if block was removed from search tree.
    [[nodiscard]] bool claim()
    {
        if (node != nullptr && !block->hasRefs() && node != nullptr)
        {
            return node->clearValue(windowSize);
        }
        return false;
    }

    NodePtr node = nullptr;
    BlockPtr block = nullptr;
    BlocKey key;
    bool exactMatch = false;
    int windowSize = -1;
};
using BlockMatches = std::vector<BlockMatch>;

// The following template arguments are used:
// NodeKey = BlockKey
// NodeKeyHashFunctor = BlockKeyHasher
// ValueKey = int (the 'channel' is window size, so a simple integer suffices as value key)
// ValueKeyHashFunctor = std::hash<int> since that already exists.
// Value = std::shared_ptr<KVCacheBlock> very important to use a pointer here since we are planning to modify KVCacheBlock state.
// supportsPartialMatching = true, because BlockKey supports partial matching.
class UnifiedBlockTree : public templated_trie::Trie<BlockKey,BlockKeyHasher,int,std::hash<int>,std::shared_ptr<kv_cache_manager::KVCacheBlock>,true>
{
    public:
        using PrefixKey = std::vector<BlockKey>;
        using NodeMatches = templated_trie::NodeMatches<BlockKey,BlockKeyHasher,int,std::hash<int>,kv_cache_manager::KVCacheBlock,true>;

        UnifiedBlockTree() = default;

        //! \brief Find reusable blocks for given window size.
        //! \param nodeMatches Nodes matching a given prefix.
        //! \param windowSize Window size.
        //! \return Blocks BlockMatch struct containing blocks matching prefix and window size. For some of these, matchedBlock field may be nullptr.
        BlockMatches lookupBlocks(NodeMatches const& nodeMatches, int windowSize) const
        {
            BlockMatches blockMatches;
            for (auto const& nodeMatch : nodeMatches.exactMatches)
            {
                blockMatches.emplace_back(nodeMatch.node, nodeMatch.node->getValue(windowSize), nodeMatch.key, nodeMatch.exactMatch, windowSize);
            }
            for (auto const& nodeMatch : nodeMatches.partialMatches)
            {
                auto block = nodeMatch.node->getValue();
                if (block != nullptr)
                {
                    blockMatches.emplace_back(nodeMatch.node, nodeMatch.node->getValue(windowSize), nodeMatch.key, nodeMatch.exactMatch, windowSize);
                    break; // Found longest partial match, we are done
                }
            }
            return blockMatches;
        }

        //! \brief Find reusable blocks for given prefix and window size.
        //! \param pkey prefix.
        //! \param allowPartialMatching If partial matching of tokens is allowed for last matching block.
        //! \param windowSize Window size.
        //! \return Blocks BlockMatch struct containing blocks matching prefix and window size. For some of these, matchedBlock field may be nullptr.
        BlockMatches lookupBlocks(PrefixKey const& pkey, bool allowPartialMatching, int windowSize) const
        {
            auto nodeMatches = lookupNodes(pkey, allowPartialMatching);
            return lookupBlocks(nodeMatches, windowSize);
        }
};
} // namespace tensorrt_llm::batch_manager::unified_block_tree
