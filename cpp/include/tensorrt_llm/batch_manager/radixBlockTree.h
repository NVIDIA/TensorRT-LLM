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

#include "tensorrt_llm/batch_manager/blockKey.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/templatedTrie.h"

//
// Implementation of constant radix search tree for KV cache blocks.
// It is a unified tree, a single instance of search tree services
// all WindowBlockManager instances. This is achieved by using
// window size as value key.
//

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheBlock;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::batch_manager::radix_block_tree
{

using BlockPtr = std::shared_ptr<kv_cache_manager::KVCacheBlock>;
using BlockKey = kv_cache_manager::BlockKey;
using BlockKeyHasher = kv_cache_manager::BlockKeyHasher;

using BlockMatch = templated_trie::ValueMatch<BlockKey, BlockKeyHasher, int, std::hash<int>, BlockPtr, true>;
using BlockMatches = std::vector<BlockMatch>;

//! \brief Node type used in the unified block tree.
//! One node per token-prefix stores block pointers for every window size.
using LookupNode = templated_trie::Node<BlockKey, BlockKeyHasher, int, std::hash<int>, BlockPtr, true>;
using LookupNodePtr = std::shared_ptr<LookupNode>;

// The following template arguments are used:
// NodeKey = BlockKey
// NodeKeyHashFunctor = BlockKeyHasher
// ValueKey = int (the 'channel' is window size, so a simple integer suffices as value key)
// ValueKeyHashFunctor = std::hash<int> since that already exists.
// Value = std::shared_ptr<KVCacheBlock> very important to use a pointer here since we are planning to modify
// KVCacheBlock state. supportsPartialMatching = true, because BlockKey supports partial matching.
class UnifiedBlockTree : public templated_trie::Trie<BlockKey, BlockKeyHasher, int, std::hash<int>, BlockPtr, true>
{
public:
    UnifiedBlockTree() = default;

    //! \brief Insert a block into the tree at the given prefix position for a specific window size.
    //! \param prefix Sequence of BlockKeys leading to the node where the block is stored.
    //! \param windowSize Value key (window size) under which the block is stored at the target node.
    //! \param block The KVCacheBlock to store.
    void insertBlock(PrefixKey const& prefix, int windowSize, BlockPtr const& block)
    {
        auto nodeMatches = insertNodes(prefix);
        if (!nodeMatches.exactMatches.empty())
        {
            [[maybe_unused]] auto const wasOverwritten
                = nodeMatches.exactMatches.back().node->setValue(windowSize, block, /*overwrite=*/false);
        }
    }

    //! \brief Look up a cached block for a given prefix and window size.
    //! \param prefix Sequence of BlockKeys identifying the prefix.
    //! \param windowSize Value key (window size) to retrieve the block for.
    //! \param allowPartialMatch If true, a partial token match on the last block key is accepted.
    //! \return The cached block if found, std::nullopt otherwise.
    [[nodiscard]] std::optional<BlockPtr> lookupBlock(
        PrefixKey const& prefix, int windowSize, bool allowPartialMatch) const
    {
        auto valueMatches = lookupValues(prefix, allowPartialMatch, windowSize);
        for (auto const& vm : valueMatches.matches)
        {
            if (vm.isValid && vm.value)
            {
                return vm.value;
            }
        }
        return std::nullopt;
    }
};

} // namespace tensorrt_llm::batch_manager::radix_block_tree
