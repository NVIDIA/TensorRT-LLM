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

#include "tensorrt_llm/batch_manager/blockKey.h"

#include <algorithm>

namespace
{
inline uint8_t getNthByte(tensorrt_llm::runtime::SizeType32 hashPart, uint8_t byteIdx) noexcept
{
    return static_cast<uint8_t>((hashPart >> (24 - byteIdx * 8)) & 0xFF);
}

std::array<uint8_t, 32> makeHashArray(std::vector<tensorrt_llm::runtime::SizeType32> const& mmHashVector)
{
    TLLM_CHECK_WITH_INFO(
        mmHashVector.size() == 8, "Multimodal hash vector has unexpected size: %zu (expected 8)", mmHashVector.size());

    std::array<uint8_t, 32> mmHashArray;
    for (size_t j = 0; j < 8; ++j)
    {
        auto const& hashPart = mmHashVector[j];
        for (uint8_t byteIdx = 0; byteIdx < 4; ++byteIdx)
        {
            mmHashArray[j * 4 + byteIdx] = getNthByte(hashPart, byteIdx);
        }
    }
    return mmHashArray;
}
} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
std::vector<MmKey> generateBlockHashExtraKeys(
    tensorrt_llm::batch_manager::LlmRequest const& llmRequest, SizeType32 startTokenIdx, SizeType32 endTokenIdx)
{
    auto const multimodalHashes = llmRequest.getMultimodalHashes();
    auto const multimodalPositions = llmRequest.getMultimodalPositions();
    auto const multimodalLengths = llmRequest.getMultimodalLengths();
    auto const multimodalUuids = llmRequest.getMultimodalUuids();
    auto const multimodalHashPositions = llmRequest.getMultimodalHashPositions();

    if (!multimodalHashes || !(*multimodalHashes) || (*multimodalHashes)->empty())
    {
        return {};
    }

    auto const hasExactHashPositions = multimodalHashPositions && *multimodalHashPositions;
    if (hasExactHashPositions)
    {
        TLLM_CHECK_WITH_INFO((*multimodalHashes)->size() == (*multimodalHashPositions)->size(),
            "Multimodal hash arrays and hash-position arrays have mismatched sizes");
        TLLM_CHECK_WITH_INFO(multimodalLengths && *multimodalLengths,
            "Multimodal hash positions require multimodal lengths for validation");
        TLLM_CHECK_WITH_INFO((*multimodalHashes)->size() == (*multimodalLengths)->size(),
            "Multimodal hash arrays and length arrays have mismatched sizes");

        std::vector<MmKey> extraKeys;
        extraKeys.reserve((*multimodalHashes)->size());

        for (size_t i = 0; i < (*multimodalHashes)->size(); ++i)
        {
            auto const mmHashArray = makeHashArray((*(*multimodalHashes))[i]);
            auto const& promptPositions = (*(*multimodalHashPositions))[i];
            auto const expectedLength = (*(*multimodalLengths))[i];
            TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(promptPositions.size()) == expectedLength,
                "Multimodal hash positions and lengths have mismatched sizes");

            std::optional<std::string> uuid = std::nullopt;
            if (multimodalUuids && *multimodalUuids && i < (*multimodalUuids)->size())
            {
                uuid = (*(*multimodalUuids))[i];
            }

            auto emitRegion = [&](SizeType32 regionStart, SizeType32 regionLength, SizeType32 hashStartOffset)
            {
                auto const regionEnd = regionStart + regionLength;
                if (endTokenIdx > regionStart && startTokenIdx < regionEnd)
                {
                    auto const overlapStart = std::max(startTokenIdx, regionStart);
                    auto const startOffset = hashStartOffset + overlapStart - regionStart;
                    extraKeys.emplace_back(mmHashArray, startOffset, uuid);
                }
            };

            bool hasRegion = false;
            SizeType32 regionStart = 0;
            SizeType32 regionEnd = 0;
            SizeType32 regionHashStartOffset = 0;
            for (SizeType32 hashOffset = 0; hashOffset < static_cast<SizeType32>(promptPositions.size()); ++hashOffset)
            {
                auto const position = promptPositions[hashOffset];
                if (!hasRegion)
                {
                    regionStart = position;
                    regionEnd = position + 1;
                    regionHashStartOffset = hashOffset;
                    hasRegion = true;
                    continue;
                }
                if (position < regionEnd)
                {
                    TLLM_CHECK_WITH_INFO(false, "Multimodal hash positions must be strictly increasing");
                }
                if (position == regionEnd)
                {
                    ++regionEnd;
                    continue;
                }

                emitRegion(regionStart, regionEnd - regionStart, regionHashStartOffset);
                regionStart = position;
                regionEnd = position + 1;
                regionHashStartOffset = hashOffset;
            }
            if (hasRegion)
            {
                emitRegion(regionStart, regionEnd - regionStart, regionHashStartOffset);
            }
        }

        return extraKeys;
    }

    if (!multimodalPositions || !multimodalLengths || !(*multimodalPositions) || (*multimodalPositions)->empty()
        || !(*multimodalLengths) || (*multimodalLengths)->empty())
    {
        return {};
    }

    if ((*multimodalHashes)->size() != (*multimodalPositions)->size()
        || (*multimodalPositions)->size() != (*multimodalLengths)->size())
    {
        TLLM_LOG_WARNING("Multimodal data arrays have mismatched sizes");
        return {};
    }

    std::vector<MmKey> extraKeys;
    extraKeys.reserve((*multimodalPositions)->size());

    for (size_t i = 0; i < (*multimodalPositions)->size(); ++i)
    {
        auto const& startPos = (*(*multimodalPositions))[i];
        auto const& length = (*(*multimodalLengths))[i];
        auto const& mmHashVector = (*(*multimodalHashes))[i];
        auto const mmHashArray = makeHashArray(mmHashVector);

        // Check if this multimodal content overlaps with the current block
        if (endTokenIdx > startPos && startTokenIdx < startPos + length)
        {
            uint64_t mmStartInBlock = (startPos >= startTokenIdx) ? 0 : static_cast<uint64_t>(startTokenIdx - startPos);

            // Get UUID if available
            std::optional<std::string> uuid = std::nullopt;
            if (multimodalUuids && *multimodalUuids && i < (*multimodalUuids)->size())
            {
                uuid = (*(*multimodalUuids))[i];
            }

            extraKeys.emplace_back(mmHashArray, mmStartInBlock, std::move(uuid));
        }
    }

    return extraKeys;
}

std::vector<BlockKey> buildBlockKeys(
    std::list<VecUniqueTokens>& blockedUniqueTokens, tensorrt_llm::batch_manager::LlmRequest const& llmRequest)
{
    std::vector<BlockKey> blockKeys;

    SizeType32 currentTokenIdx = 0;
    for (auto& uniqueTokens : blockedUniqueTokens)
    {
        auto extraKeys = generateBlockHashExtraKeys(llmRequest, currentTokenIdx, currentTokenIdx + uniqueTokens.size());
        currentTokenIdx += uniqueTokens.size();

        blockKeys.emplace_back(llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(),
            std::move(uniqueTokens), std::move(extraKeys), llmRequest.getCacheSaltID());
    }
    return blockKeys;
}

bool BlockKey::operator==(BlockKey const& other) const noexcept
{
    return (usesExtraIds == other.usesExtraIds && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens
        && extraKeys == other.extraKeys && cacheSaltID == other.cacheSaltID);
}

BlockKey BlockKey::shorten(int newNumTokens) const
{
    TLLM_CHECK_WITH_INFO(extraKeys.empty(),
        "shorten() cannot be called on a BlockKey with extraKeys (partial matching is disabled when multimodal data is "
        "present).");
    TLLM_CHECK_WITH_INFO(
        newNumTokens >= 0 && newNumTokens <= getNumTokens(), "newNumTokens must be >= 0 and <= getNumTokens()");
    BlockKey result(*this);
    result.uniqueTokens.resize(newNumTokens);
    return result;
}

size_t BlockKeyHasher::hash(BlockKey const& blockKey, std::size_t parentHash) noexcept
{
    // Hashing algorithm adapted from StackOverflow:
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    // Constants provide very good distribution - each input bit affects each output bit with ~50% probability.
    size_t seed = blockKey.uniqueTokens.size() ^ parentHash * UINT64_C(0xbf58476d1ce4e5b9);

    if (parentHash == 0 && blockKey.cacheSaltID)
    {
        // Only hashing the cache salt ID for the first block in the sequence
        uint64_t c = blockKey.cacheSaltID.value();
        seed = hash64Mix(c, seed);
    }

    for (auto const& uniqueToken : blockKey.uniqueTokens)
    {
        uint32_t a = static_cast<uint32_t>(uniqueToken.tokenId);
        seed = hash32Mix(a, seed);
        if (blockKey.usesExtraIds)
        {
            uint64_t b = uniqueToken.tokenExtraId;
            seed = hash64Mix(b, seed);
        }
    }

    if (blockKey.loraTaskId)
    {
        uint64_t c = blockKey.loraTaskId.value();
        seed = hash64Mix(c, seed);
    }

    // Add extra keys for multimodal data mixing in external multimodal item hash and token offset within this sequence
    // block
    if (!blockKey.extraKeys.empty())
    {
        for (auto const& mmKey : blockKey.extraKeys)
        {
            auto const& mmHash = mmKey.hash;
            auto const& startOffset = mmKey.startOffset;
            // Hash the multimodal hash array in 32-bit chunks (more efficient)
            for (size_t i = 0; i < 32; i += 4)
            {
                // Combine 4 bytes into a 32-bit word (construct as little endian order)
                uint32_t word = static_cast<uint32_t>(mmHash[i]) | (static_cast<uint32_t>(mmHash[i + 1]) << 8)
                    | (static_cast<uint32_t>(mmHash[i + 2]) << 16) | (static_cast<uint32_t>(mmHash[i + 3]) << 24);

                // Mix the word into the seed
                seed = hash32Mix(word, seed);
            }

            // Hash the start offset
            uint64_t e = static_cast<uint64_t>(startOffset);
            seed = hash64Mix(e, seed);
        }
    }

    return seed;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
