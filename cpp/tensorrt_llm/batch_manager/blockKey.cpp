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
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using MmKey = tensorrt_llm::executor::MmKey;
using MultimodalHashes = std::vector<std::vector<SizeType32>>;
using MultimodalSizeVector = std::vector<SizeType32>;
using OptionalSizeVectorPtr = std::optional<std::shared_ptr<MultimodalSizeVector>>;
using OptionalUuidVectorPtr = std::optional<std::shared_ptr<std::vector<std::optional<std::string>>>>;

enum class RunMetadataStatus
{
    kAbsent,
    kInvalid,
    kPresent,
};

struct RunMetadataView
{
    MultimodalSizeVector const* itemRunCuSeqlen{nullptr};
    MultimodalSizeVector const* runPositions{nullptr};
    MultimodalSizeVector const* runLengths{nullptr};
};

struct RunMetadataResolution
{
    RunMetadataStatus status{RunMetadataStatus::kAbsent};
    RunMetadataView metadata{};
};

inline uint8_t getNthByte(SizeType32 hashPart, uint8_t byteIdx) noexcept
{
    return static_cast<uint8_t>((hashPart >> (24 - byteIdx * 8)) & 0xFF);
}

std::array<uint8_t, 32> makeHashArray(std::vector<SizeType32> const& mmHashVector)
{
    TLLM_CHECK_WITH_INFO(
        mmHashVector.size() == 8, "Multimodal hash vector has unexpected size: %zu (expected 8)", mmHashVector.size());

    std::array<uint8_t, 32> mmHashArray;
    // Python int(hex_chunk, 16) stores the first hash byte in the most significant byte.
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

std::optional<std::string> getUuid(OptionalUuidVectorPtr const& multimodalUuids, size_t itemIdx)
{
    if (multimodalUuids && *multimodalUuids && itemIdx < (*multimodalUuids)->size())
    {
        return (*(*multimodalUuids))[itemIdx];
    }
    return std::nullopt;
}

std::vector<MmKey> warnAndDisableExtraKeys(char const* warning)
{
    TLLM_LOG_WARNING("%s", warning);
    return {};
}

RunMetadataResolution resolveRunMetadata(OptionalSizeVectorPtr const& multimodalItemRunCuSeqlen,
    OptionalSizeVectorPtr const& multimodalRunPositions, OptionalSizeVectorPtr const& multimodalRunLengths,
    size_t itemCount)
{
    auto const hasRunMetadataFields = multimodalItemRunCuSeqlen || multimodalRunPositions || multimodalRunLengths;
    if (!hasRunMetadataFields)
    {
        return {};
    }

    auto const hasCompleteRunMetadata = multimodalItemRunCuSeqlen && *multimodalItemRunCuSeqlen
        && multimodalRunPositions && *multimodalRunPositions && multimodalRunLengths && *multimodalRunLengths;
    if (!hasCompleteRunMetadata)
    {
        TLLM_LOG_WARNING("Incomplete multimodal run metadata; disabling multimodal cache hash extra keys");
        return {RunMetadataStatus::kInvalid, {}};
    }

    auto const& itemRunCuSeqlen = *(*multimodalItemRunCuSeqlen);
    auto const& runPositions = *(*multimodalRunPositions);
    auto const& runLengths = *(*multimodalRunLengths);
    if (itemRunCuSeqlen.size() != itemCount + 1 || itemRunCuSeqlen.empty() || itemRunCuSeqlen.front() != 0
        || itemRunCuSeqlen.back() < 0 || static_cast<size_t>(itemRunCuSeqlen.back()) != runPositions.size()
        || runPositions.size() != runLengths.size())
    {
        TLLM_LOG_WARNING("Invalid multimodal run metadata shape; disabling multimodal cache hash extra keys");
        return {RunMetadataStatus::kInvalid, {}};
    }

    return {RunMetadataStatus::kPresent, {&itemRunCuSeqlen, &runPositions, &runLengths}};
}

std::optional<SizeType32> getValidRunEnd(SizeType32 runStart, SizeType32 runLength)
{
    auto const runEnd64 = static_cast<int64_t>(runStart) + static_cast<int64_t>(runLength);
    if (runStart < 0 || runLength <= 0 || runEnd64 > std::numeric_limits<SizeType32>::max())
    {
        return std::nullopt;
    }
    return static_cast<SizeType32>(runEnd64);
}

std::vector<MmKey> generateRunBasedExtraKeys(MultimodalHashes const& multimodalHashes,
    OptionalUuidVectorPtr const& multimodalUuids, RunMetadataView const& runMetadata, SizeType32 startTokenIdx,
    SizeType32 endTokenIdx)
{
    auto const& itemRunCuSeqlen = *runMetadata.itemRunCuSeqlen;
    auto const& runPositions = *runMetadata.runPositions;
    auto const& runLengths = *runMetadata.runLengths;

    std::vector<MmKey> extraKeys;
    extraKeys.reserve(multimodalHashes.size());

    for (size_t itemIdx = 0; itemIdx < multimodalHashes.size(); ++itemIdx)
    {
        auto const runBegin32 = itemRunCuSeqlen[itemIdx];
        auto const runEnd32 = itemRunCuSeqlen[itemIdx + 1];
        if (runBegin32 < 0 || runEnd32 < runBegin32 || static_cast<size_t>(runEnd32) > runPositions.size())
        {
            return warnAndDisableExtraKeys(
                "Invalid multimodal run prefix sum; disabling multimodal cache hash extra keys");
        }

        auto const mmHashArray = makeHashArray(multimodalHashes[itemIdx]);
        auto const uuid = getUuid(multimodalUuids, itemIdx);
        SizeType32 itemOffset = 0;
        for (size_t runIdx = static_cast<size_t>(runBegin32); runIdx < static_cast<size_t>(runEnd32); ++runIdx)
        {
            auto const runStart = runPositions[runIdx];
            auto const runLength = runLengths[runIdx];
            auto const runEnd = getValidRunEnd(runStart, runLength);
            if (!runEnd)
            {
                return warnAndDisableExtraKeys(
                    "Invalid multimodal run range; disabling multimodal cache hash extra keys");
            }

            if (endTokenIdx > runStart && startTokenIdx < *runEnd)
            {
                auto const overlapStart = std::max(startTokenIdx, runStart);
                auto const startOffset = itemOffset + overlapStart - runStart;
                extraKeys.emplace_back(mmHashArray, startOffset, uuid);
            }
            itemOffset += runLength;
        }
    }

    return extraKeys;
}

std::vector<MmKey> generateLegacyContiguousExtraKeys(MultimodalHashes const& multimodalHashes,
    OptionalSizeVectorPtr const& multimodalPositions, OptionalSizeVectorPtr const& multimodalLengths,
    OptionalUuidVectorPtr const& multimodalUuids, SizeType32 startTokenIdx, SizeType32 endTokenIdx)
{
    if (!multimodalPositions || !multimodalLengths || !(*multimodalPositions) || (*multimodalPositions)->empty()
        || !(*multimodalLengths) || (*multimodalLengths)->empty())
    {
        return {};
    }

    auto const& positions = *(*multimodalPositions);
    auto const& lengths = *(*multimodalLengths);
    if (multimodalHashes.size() != positions.size() || positions.size() != lengths.size())
    {
        return warnAndDisableExtraKeys("Multimodal data arrays have mismatched sizes");
    }

    std::vector<MmKey> extraKeys;
    extraKeys.reserve(positions.size());

    for (size_t itemIdx = 0; itemIdx < positions.size(); ++itemIdx)
    {
        auto const& startPos = positions[itemIdx];
        auto const& length = lengths[itemIdx];

        if (endTokenIdx > startPos && startTokenIdx < startPos + length)
        {
            auto const mmStartInBlock = startPos >= startTokenIdx ? 0 : startTokenIdx - startPos;
            extraKeys.emplace_back(
                makeHashArray(multimodalHashes[itemIdx]), mmStartInBlock, getUuid(multimodalUuids, itemIdx));
        }
    }

    return extraKeys;
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
    auto const multimodalItemRunCuSeqlen = llmRequest.getMultimodalItemRunCuSeqlen();
    auto const multimodalRunPositions = llmRequest.getMultimodalRunPositions();
    auto const multimodalRunLengths = llmRequest.getMultimodalRunLengths();

    if (!multimodalHashes || !(*multimodalHashes) || (*multimodalHashes)->empty())
    {
        return {};
    }

    auto const& hashes = *(*multimodalHashes);
    auto const runMetadata
        = resolveRunMetadata(multimodalItemRunCuSeqlen, multimodalRunPositions, multimodalRunLengths, hashes.size());
    if (runMetadata.status == RunMetadataStatus::kInvalid)
    {
        return {};
    }
    if (runMetadata.status == RunMetadataStatus::kPresent)
    {
        return generateRunBasedExtraKeys(hashes, multimodalUuids, runMetadata.metadata, startTokenIdx, endTokenIdx);
    }

    return generateLegacyContiguousExtraKeys(
        hashes, multimodalPositions, multimodalLengths, multimodalUuids, startTokenIdx, endTokenIdx);
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
