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
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using UniqueToken = tensorrt_llm::runtime::UniqueToken;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using LoraTaskIdType = tensorrt_llm::runtime::LoraTaskIdType;
using CacheSaltIDType = tensorrt_llm::runtime::CacheSaltIDType;
using MmKey = tensorrt_llm::executor::MmKey;

std::vector<MmKey> generateBlockHashExtraKeys(
    tensorrt_llm::batch_manager::LlmRequest const& llmRequest, SizeType32 startTokenIdx, SizeType32 endTokenIdx);

struct BlockKey
{
    bool usesExtraIds = false;
    std::optional<LoraTaskIdType> loraTaskId = std::nullopt;
    VecUniqueTokens uniqueTokens;

    // Extra keys for multimodal data (similar to VLLM's approach)
    // Each extra key is a pair of (mm_hash, start_offset_in_block)
    std::vector<MmKey> extraKeys;
    std::optional<CacheSaltIDType> cacheSaltID = std::nullopt;

    BlockKey() = default;

    explicit BlockKey(VecTokens const& tokens, std::optional<LoraTaskIdType> loraTaskId = std::nullopt)
        : loraTaskId{loraTaskId}
    {
        uniqueTokens.reserve(tokens.size());
        for (auto const& token : tokens)
        {
            uniqueTokens.push_back(UniqueToken{token, 0});
        }
    }

    explicit BlockKey(bool usesExtraIds, std::optional<LoraTaskIdType> loraTaskId, VecUniqueTokens uniqueTokens,
        std::vector<MmKey> extraKeys = {}, std::optional<CacheSaltIDType> cacheSaltID = std::nullopt)
        : usesExtraIds{usesExtraIds}
        , loraTaskId{loraTaskId}
        , uniqueTokens{std::move(uniqueTokens)}
        , extraKeys{std::move(extraKeys)}
        , cacheSaltID{cacheSaltID}
    {
    }

    bool operator==(BlockKey const& other) const noexcept;

    bool supportsPartialMatching() const noexcept
    {
        return true;
    }

    int numMatchingTokens(BlockKey const& other) const noexcept
    {
        SizeType32 numMatched{0};
        if (loraTaskId == other.loraTaskId && extraKeys == other.extraKeys && cacheSaltID == other.cacheSaltID)
        {
            auto [matchEnd, otherMatchEnd] = std::mismatch(
                uniqueTokens.begin(), uniqueTokens.end(), other.uniqueTokens.begin(), other.uniqueTokens.end());
            numMatched = std::distance(uniqueTokens.begin(), matchEnd);
        }
        return numMatched;
    }

    int getNumTokens() const noexcept
    {
        return static_cast<int>(uniqueTokens.size());
    }

    struct BlockKey shorten(int newNumTokens) const;
};

std::vector<BlockKey> buildBlockKeys(std::list<VecUniqueTokens>& blockedUniqueTokens, LlmRequest const& llmRequest);

// Implement hash functor for BlockKey.
// This allows us to use unordered_map with BlockKey as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
struct BlockKeyHasher
{
    [[nodiscard]] static size_t hash(BlockKey const& blockKey, std::size_t parentHash = 0) noexcept;

    std::size_t operator()(BlockKey const& blockKey, std::size_t parentHash = 0) const noexcept
    {
        return hash(blockKey, parentHash);
    }
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
