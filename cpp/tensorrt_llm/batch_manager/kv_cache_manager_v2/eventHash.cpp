/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#include "kv_cache_manager_v2/eventManager.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "tensorrt_llm/common/logger.h"

#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

constexpr uint32_t kUint32HashConst = 0x045D9F3BU;
constexpr uint64_t kUint64HashConst1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kUint64HashConst2 = 0x94D049BB133111EBULL;
constexpr uint32_t kHashCombineConst = 0x9E3779B9U;
constexpr uint64_t kParentHashConst = 0xBF58476D1CE4E5B9ULL;

uint64_t hash32Mix(int64_t input, uint64_t seed)
{
    uint32_t value = static_cast<uint32_t>(input);
    value = ((value >> 16U) ^ value) * kUint32HashConst;
    value = ((value >> 16U) ^ value) * kUint32HashConst;
    value = (value >> 16U) ^ value;
    value += kHashCombineConst;
    return seed ^ (static_cast<uint64_t>(value) + (seed << 6U) + (seed >> 2U));
}

uint64_t hash64Mix(int64_t input, uint64_t seed)
{
    uint64_t value = static_cast<uint64_t>(input);
    value = (value ^ (value >> 30U)) * kUint64HashConst1;
    value = (value ^ (value >> 27U)) * kUint64HashConst2;
    value ^= value >> 31U;
    return seed ^ (value + static_cast<uint64_t>(kHashCombineConst) + (seed << 6U) + (seed >> 2U));
}

} // namespace

uint64_t EventManager::hashV1BlockKey(std::vector<TokenId> const& tokens, uint64_t parentHash,
    std::optional<LoraTaskIdType> loraTaskId, std::optional<std::uint64_t> cacheSaltId)
{
    uint64_t seed = static_cast<uint64_t>(tokens.size()) ^ (parentHash * kParentHashConst);
    if (parentHash == 0 && cacheSaltId.has_value())
    {
        seed = hash64Mix(*cacheSaltId, seed);
    }
    for (TokenId token : tokens)
    {
        seed = hash32Mix(token, seed);
    }
    if (loraTaskId.has_value())
    {
        seed = hash64Mix(*loraTaskId, seed);
    }
    return seed;
}

std::optional<uint64_t> EventManager::hashV1BlockTokens(std::vector<TokenIdExt> const& tokens, uint64_t parentHash,
    std::optional<LoraTaskIdType> loraTaskId, std::optional<std::uint64_t> cacheSaltId,
    std::vector<UniqueToken>* eventTokens)
{
    uint64_t seed = static_cast<uint64_t>(tokens.size()) ^ (parentHash * kParentHashConst);
    if (parentHash == 0 && cacheSaltId.has_value())
    {
        seed = hash64Mix(*cacheSaltId, seed);
    }
    if (eventTokens != nullptr)
    {
        eventTokens->clear();
        eventTokens->reserve(tokens.size());
    }
    for (auto const& token : tokens)
    {
        if (token.isDigest())
        {
            if (eventTokens != nullptr)
            {
                eventTokens->clear();
            }
            return std::nullopt;
        }
        TokenId const tokenId = token.tokenId();
        seed = hash32Mix(tokenId, seed);
        if (eventTokens != nullptr)
        {
            UniqueToken eventToken;
            eventToken.tokenId = EventTokenId{std::in_place_index<0>, tokenId};
            eventTokens->push_back(std::move(eventToken));
        }
    }
    if (loraTaskId.has_value())
    {
        seed = hash64Mix(*loraTaskId, seed);
    }
    return seed;
}

uint64_t EventManager::v1HashFromBlock(Block const& block, std::vector<UniqueToken>* eventTokens)
{
    if (eventTokens != nullptr)
    {
        eventTokens->clear();
    }
    if (auto const cached = mV1HashStates.find(block.key); cached != mV1HashStates.end())
    {
        if (eventTokens != nullptr && cached->second.compatible && !cached->second.rootAttrs.first.has_value()
            && !cached->second.rootAttrs.second.has_value())
        {
            eventTokens->reserve(block.tokens.size());
            for (auto const& token : block.tokens)
            {
                UniqueToken eventToken;
                eventToken.tokenId = EventTokenId{std::in_place_index<0>, token.tokenId()};
                eventTokens->push_back(std::move(eventToken));
            }
        }
        return cached->second.hash;
    }

    uint64_t parentHash = 0;
    bool parentIsV1Compatible = true;
    V1RootAttrs rootAttrs;
    auto hashAndCache = [&](Block const& currentBlock, std::vector<UniqueToken>* currentEventTokens)
    {
        if (parentIsV1Compatible)
        {
            auto const hash = hashV1BlockTokens(
                currentBlock.tokens, parentHash, rootAttrs.first, rootAttrs.second, currentEventTokens);
            if (hash.has_value())
            {
                parentHash = *hash;
            }
            else
            {
                parentIsV1Compatible = false;
            }
        }
        if (!parentIsV1Compatible)
        {
            parentHash = fallbackV1Hash(currentBlock.key);
        }
        mV1HashStates.insert_or_assign(currentBlock.key, V1HashState{parentHash, parentIsV1Compatible, rootAttrs});
    };

    NodeBase const* parent = block.prev;
    if (parent == nullptr)
    {
        throw std::logic_error("Cannot hash an orphan KV cache block");
    }
    if (parent->type() == NodeBase::Type::kROOT_BLOCK)
    {
        auto const& reuseScope = static_cast<RootBlock const*>(parent)->reuseScope;
        rootAttrs = {reuseScope.loraId, reuseScope.salt};
        hashAndCache(block, !rootAttrs.first.has_value() && !rootAttrs.second.has_value() ? eventTokens : nullptr);
        return parentHash;
    }
    auto const* parentBlock = static_cast<Block const*>(parent);
    if (auto const cached = mV1HashStates.find(parentBlock->key); cached != mV1HashStates.end())
    {
        parentHash = cached->second.hash;
        parentIsV1Compatible = cached->second.compatible;
        rootAttrs = cached->second.rootAttrs;
        hashAndCache(block, !rootAttrs.first.has_value() && !rootAttrs.second.has_value() ? eventTokens : nullptr);
        return parentHash;
    }

    std::vector<Block const*> chain{&block};
    NodeBase const* current = parent;
    while (current->type() == NodeBase::Type::kBLOCK)
    {
        auto const* currentBlock = static_cast<Block const*>(current);
        if (auto const cached = mV1HashStates.find(currentBlock->key); cached != mV1HashStates.end())
        {
            parentHash = cached->second.hash;
            parentIsV1Compatible = cached->second.compatible;
            rootAttrs = cached->second.rootAttrs;
            break;
        }
        chain.push_back(currentBlock);
        current = currentBlock->prev;
        if (current == nullptr)
        {
            throw std::logic_error("Cannot hash an orphan KV cache block");
        }
    }
    if (current->type() == NodeBase::Type::kROOT_BLOCK)
    {
        auto const& reuseScope = static_cast<RootBlock const*>(current)->reuseScope;
        rootAttrs = {reuseScope.loraId, reuseScope.salt};
    }

    for (auto chainIter = chain.rbegin(); chainIter != chain.rend(); ++chainIter)
    {
        Block const& currentBlock = **chainIter;
        auto* currentEventTokens
            = &currentBlock == &block && !rootAttrs.first.has_value() && !rootAttrs.second.has_value() ? eventTokens
                                                                                                       : nullptr;
        hashAndCache(currentBlock, currentEventTokens);
    }
    return parentHash;
}

uint64_t EventManager::fallbackV1Hash(Digest const& blockKey)
{
    if (!mWarnedV1HashFallback)
    {
        TLLM_LOG_WARNING(
            "V2 KV cache event hash algorithm v1_block_key only matches v1 for text-token radix blocks. "
            "Falling back to truncated SHA-256 block hash for unsupported blocks.");
        mWarnedV1HashFallback = true;
    }
    return truncateDigestToInt64(blockKey);
}

void EventManager::dropHashCache(Digest const& blockKey)
{
    mV1HashStates.erase(blockKey);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
