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
using BlockMatch = ValueMatch<kv_cache_manager::BlockKey, kv_cache_manager::BlockKeyHasher, int, std::hash<int>, std::shared_ptr<kv_cache_manager::KVCacheBlock>>;
using BlockMatches = std::vector<BlockMatch>;

// Convenience method to claim a block. May not be used, mostly here to show how to claim blocks.
bool claimBlock(BlockMatch& match)
{
    if (match.node != nullptr && match.value != nullptr && !match.value->hasRefs())
    {
        return match.node->clearValue(match.vkey);
    }
    return false;
}

// The following template arguments are used:
// NodeKey = BlockKey
// NodeKeyHashFunctor = BlockKeyHasher
// ValueKey = int (the 'channel' is window size, so a simple integer suffices as value key)
// ValueKeyHashFunctor = std::hash<int> since that already exists.
// Value = std::shared_ptr<KVCacheBlock> very important to use a pointer here since we are planning to modify
// KVCacheBlock state. supportsPartialMatching = true, because BlockKey supports partial matching.
class UnifiedBlockTree : public templated_trie::Trie<kv_cache_manager::BlockKey, kv_cache_manager::BlockKeyHasher, int, std::hash<int>,
                             std::shared_ptr<kv_cache_manager::KVCacheBlock>, true>
{
public:
    UnifiedBlockTree() = default;
};
} // namespace tensorrt_llm::batch_manager::radix_block_tree
