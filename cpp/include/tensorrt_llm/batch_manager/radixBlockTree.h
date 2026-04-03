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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <optional>
#include <vector>

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

//! \brief Sentinel windowSize for the linear-attention (Mamba) WindowBlockManager.
//! Negative to distinguish from all valid full-attention window sizes (>= 1).
//! Usage: `WindowBlockManager` created with windowSize = kRecurrentStates manages
//! Mamba/SSM state blocks for hybrid models.
inline constexpr int kRecurrentStates = -1;

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
    //! \details This is a tree-only insertion: it does NOT set block->mLookupNode. The block is
    //! stored as a value in the trie node but carries no back-reference to that node. Use this for testing. For
    //! full-attention blocks that need bidirectional wiring (getPrevBlock, detachFromLookupNode, etc.), use
    //! addNextBlock() instead. \param prefix Sequence of BlockKeys leading to the node where the block is stored.
    //! \param windowSize Value key (window size) under which the block is stored at the target node.
    //! \param block The KVCacheBlock to store.
    void insertBlock(PrefixKey const& prefix, int windowSize, BlockPtr const& block)
    {
        auto nodeMatches = insertNodes(prefix);
        if (!nodeMatches.exactMatches.empty())
        {
            auto const wasInserted
                = nodeMatches.exactMatches.back().node->trySetValue(windowSize, block, /*overwrite=*/false);
            if (!wasInserted)
            {
                TLLM_LOG_DEBUG("insertBlock: slot for windowSize=%d already occupied; insertion skipped", windowSize);
            }
        }
    }

    //! \brief Look up a cached block for a given prefix and window size.
    //! \details Returns the deepest (most specific) valid match found along the prefix path.
    //! When \p allowPartialMatch is false, also requires that the trie contains nodes for
    //! every step in \p prefix (i.e., the chain must be complete) before returning a block.
    //! \param prefix Sequence of BlockKeys identifying the prefix.
    //! \param windowSize Value key (window size) to retrieve the block for.
    //! \param allowPartialMatch If true, a partial token match on the last block key is accepted.
    //! \return The cached block if found, std::nullopt otherwise.
    [[nodiscard]] std::optional<BlockPtr> lookupBlock(
        PrefixKey const& prefix, int windowSize, bool allowPartialMatch) const
    {
        auto valueMatches = lookupValues(prefix, allowPartialMatch, windowSize);
        if (!allowPartialMatch)
        {
            // Exact lookup: all prefix nodes must exist AND the target node must have a value.
            // We must not fall back to an ancestor block even if one exists, because that
            // would silently return a cached block for a shorter prefix than requested.
            // lookupValues stops early when a node is missing, so matches.size() < prefix.size()
            // means the prefix chain is broken.
            if (valueMatches.matches.size() != prefix.size())
            {
                return std::nullopt;
            }
            auto const& exactMatch = valueMatches.matches.back();
            if (exactMatch.isValid && exactMatch.value)
            {
                return exactMatch.value;
            }
            return std::nullopt;
        }
        // Partial match allowed: return the deepest (last) valid match.
        for (auto itr = valueMatches.matches.rbegin(); itr != valueMatches.matches.rend(); ++itr)
        {
            if (itr->isValid && itr->value)
            {
                return itr->value;
            }
        }
        return std::nullopt;
    }

    //! \brief Look up cached blocks at every position of the given prefix.
    //! \details Returns one entry per prefix step. The entry is nullopt when no block exists
    //! for \p windowSize at that prefix position (either the trie node is absent or its slot
    //! is empty). Trailing positions not represented in the trie are padded with nullopt.
    //!
    //! This is the primary API for Mamba / linear-attention support: use it in
    //! getCacheBlockIndices to determine which Mamba state block slots are real vs. nil.
    //! Mamba snapshot blocks are inserted only at specific prefix positions; positions
    //! without a snapshot (placeholder KVCacheBlocks) appear as nullopt here.
    //!
    //! \param prefix Sequence of BlockKeys for the full sequence prefix.
    //! \param windowSize Value key (window size) — use kRecurrentStates for Mamba layers.
    //! \return Vector of length prefix.size(); nullopt at positions with no block.
    [[nodiscard]] std::vector<std::optional<BlockPtr>> lookupBlocksAtAllPositions(
        PrefixKey const& prefix, int windowSize) const
    {
        auto valueMatches = lookupValues(prefix, /*allowPartialMatch=*/false, windowSize);
        std::vector<std::optional<BlockPtr>> result;
        result.reserve(prefix.size());
        for (auto const& vm : valueMatches.matches)
        {
            if (vm.isValid && vm.value)
            {
                result.emplace_back(vm.value);
            }
            else
            {
                result.emplace_back(std::nullopt);
            }
        }
        // Pad with nullopt for any prefix positions that have no trie node.
        while (result.size() < prefix.size())
        {
            result.emplace_back(std::nullopt);
        }
        return result;
    }

    //! \brief Insert blocks at selected positions in the prefix, creating all intermediate nodes.
    //! \details Creates trie nodes for every step in \p prefix. For each position \p i where
    //! \p blocks[i] is non-null, stores that block under \p windowSize at node \p i. nullptr
    //! entries are placeholder positions: the trie node is created (to preserve prefix
    //! structure for future lookups) but no value is attached for \p windowSize.
    //!
    //! Use this for Mamba storeContextBlocks: pass the full per-window-size block vector
    //! with nullptr for positions that have no Mamba state snapshot (placeholder blocks).
    //!
    //! \param prefix Full prefix (one BlockKey per block position in the sequence).
    //! \param windowSize Value key under which real blocks are stored (e.g. kRecurrentStates).
    //! \param blocks Parallel to prefix; nullptr entries denote placeholder positions.
    void insertBlocks(PrefixKey const& prefix, int windowSize, std::vector<BlockPtr> const& blocks)
    {
        TLLM_CHECK_WITH_INFO(blocks.size() == prefix.size(),
            "insertBlocks: blocks.size()=%zu must equal prefix.size()=%zu", blocks.size(), prefix.size());
        auto nodeMatches = insertNodes(prefix);
        for (size_t i = 0; i < nodeMatches.exactMatches.size(); ++i)
        {
            if (i < blocks.size() && blocks[i])
            {
                auto const wasInserted
                    = nodeMatches.exactMatches[i].node->trySetValue(windowSize, blocks[i], /*overwrite=*/false);
                if (!wasInserted)
                {
                    TLLM_LOG_DEBUG(
                        "insertBlocks: slot at index %zu for windowSize=%d already occupied; "
                        "insertion skipped",
                        i, windowSize);
                }
            }
        }
    }
};

} // namespace tensorrt_llm::batch_manager::radix_block_tree
