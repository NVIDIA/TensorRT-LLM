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
    MultimodalSizeVector const* itemRunCuOffsets{nullptr};
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
    // Missing UUIDs intentionally fall back to content-hash cache keys.
    return std::nullopt;
}

size_t getOptionalVectorSize(OptionalSizeVectorPtr const& values)
{
    return values && *values ? (*values)->size() : 0;
}

RunMetadataResolution resolveRunMetadata(OptionalSizeVectorPtr const& multimodalItemRunCuOffsets,
    OptionalSizeVectorPtr const& multimodalRunPositions, OptionalSizeVectorPtr const& multimodalRunLengths,
    size_t itemCount)
{
    auto const hasRunMetadataFields = multimodalItemRunCuOffsets || multimodalRunPositions || multimodalRunLengths;
    if (!hasRunMetadataFields)
    {
        return {};
    }

    auto const hasItemRunCuOffsets = multimodalItemRunCuOffsets && *multimodalItemRunCuOffsets;
    auto const hasRunPositions = multimodalRunPositions && *multimodalRunPositions;
    auto const hasRunLengths = multimodalRunLengths && *multimodalRunLengths;
    auto const hasCompleteRunMetadata = hasItemRunCuOffsets && hasRunPositions && hasRunLengths;
    if (!hasCompleteRunMetadata)
    {
        TLLM_LOG_WARNING(
            "Incomplete multimodal run metadata; falling back to legacy multimodal cache hash extra keys: "
            "hasItemRunCuOffsets=%d, hasRunPositions=%d, hasRunLengths=%d, "
            "itemRunCuOffsets.size=%zu, runPositions.size=%zu, runLengths.size=%zu",
            static_cast<int>(hasItemRunCuOffsets), static_cast<int>(hasRunPositions), static_cast<int>(hasRunLengths),
            getOptionalVectorSize(multimodalItemRunCuOffsets), getOptionalVectorSize(multimodalRunPositions),
            getOptionalVectorSize(multimodalRunLengths));
        return {RunMetadataStatus::kInvalid, {}};
    }

    auto const& itemRunCuOffsets = *(*multimodalItemRunCuOffsets);
    auto const& runPositions = *(*multimodalRunPositions);
    auto const& runLengths = *(*multimodalRunLengths);
    // - itemRunCuOffsets are exclusive-prefix-sum offsets: item i reads runs
    //   [offsets[i], offsets[i + 1]). so its size should be 1 + the number of items.
    // - The first item must start at run 0.
    // - The final cumulative offset must be non-negative and equal the number of runs.
    // - runPositions and runLengths describe the same runs, so sizes match.
    if (itemRunCuOffsets.size() != itemCount + 1 || itemRunCuOffsets.front() != 0 || itemRunCuOffsets.back() < 0
        || static_cast<size_t>(itemRunCuOffsets.back()) != runPositions.size()
        || runPositions.size() != runLengths.size())
    {
        TLLM_LOG_WARNING(
            "Invalid multimodal run metadata shape; falling back to legacy multimodal cache hash extra keys: "
            "itemCount=%zu, itemRunCuOffsets.size=%zu, expectedItemRunCuOffsets.size=%zu, "
            "itemRunCuOffsets.front=%d, itemRunCuOffsets.back=%d, runPositions.size=%zu, runLengths.size=%zu",
            itemCount, itemRunCuOffsets.size(), itemCount + 1, itemRunCuOffsets.empty() ? 0 : itemRunCuOffsets.front(),
            itemRunCuOffsets.empty() ? 0 : itemRunCuOffsets.back(), runPositions.size(), runLengths.size());
        return {RunMetadataStatus::kInvalid, {}};
    }

    return {RunMetadataStatus::kPresent, {&itemRunCuOffsets, &runPositions, &runLengths}};
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

std::optional<std::vector<MmKey>> generateRunBasedExtraKeys(MultimodalHashes const& multimodalHashes,
    OptionalUuidVectorPtr const& multimodalUuids, RunMetadataView const& runMetadata, SizeType32 startTokenIdx,
    SizeType32 endTokenIdx)
{
    auto const& itemRunCuOffsets = *runMetadata.itemRunCuOffsets;
    auto const& runPositions = *runMetadata.runPositions;
    auto const& runLengths = *runMetadata.runLengths;

    // Worked example for one logical multimodal item split by text:
    //
    //   prompt index: 0    1      2      3    4      5
    //   prompt token: T0   MM0:0  MM0:1  T1   MM0:2  MM0:3
    //   flat run:          run0   run0        run1   run1
    //
    // Inputs encode the prompt layout as:
    //
    //   itemRunCuOffsets = [0, 2]  // item 0 owns flat runs [0, 2)
    //   runPositions     = [1, 4]  // run0 starts at prompt 1, run1 at 4
    //   runLengths       = [2, 2]  // token counts for run0 and run1
    //
    // The loop below accumulates `itemOffset` across an item's flat runs. For
    // run1, itemOffset is 2, so a block overlapping prompt [4, 6) mixes MM0's
    // hash with startOffset=2. A block overlapping prompt [5, 6) mixes the
    // same hash with startOffset=3.
    // This makes the block hash calculation accurate to the actual multimodal token locations.
    std::vector<MmKey> extraKeys;
    extraKeys.reserve(multimodalHashes.size());

    for (size_t itemIdx = 0; itemIdx < multimodalHashes.size(); ++itemIdx)
    {
        auto const runBegin32 = itemRunCuOffsets[itemIdx];
        auto const runEnd32 = itemRunCuOffsets[itemIdx + 1];
        if (runBegin32 < 0 || runEnd32 < runBegin32 || static_cast<size_t>(runEnd32) > runPositions.size())
        {
            TLLM_LOG_WARNING(
                "Invalid multimodal item run offsets; falling back to legacy multimodal cache hash extra keys: "
                "itemIdx=%zu, runBegin=%d, runEnd=%d, runPositions.size=%zu",
                itemIdx, runBegin32, runEnd32, runPositions.size());
            return std::nullopt;
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
                auto const runEnd64 = static_cast<int64_t>(runStart) + static_cast<int64_t>(runLength);
                TLLM_LOG_WARNING(
                    "Invalid multimodal run range; falling back to legacy multimodal cache hash extra keys: "
                    "itemIdx=%zu, runIdx=%zu, runStart=%d, runLength=%d, runEnd=%lld, maxSizeType32=%d",
                    itemIdx, runIdx, runStart, runLength, static_cast<long long>(runEnd64),
                    std::numeric_limits<SizeType32>::max());
                return std::nullopt;
            }

            if (endTokenIdx > runStart && startTokenIdx < *runEnd)
            {
                auto const overlapStart = std::max(startTokenIdx, runStart);
                // Offset is relative to the logical multimodal item, not this
                // prompt run. For the example above, run1 starts from MM0:2.
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
        TLLM_LOG_WARNING(
            "Multimodal data arrays have mismatched sizes; disabling multimodal cache hash extra keys: "
            "multimodalHashes.size=%zu, multimodalPositions.size=%zu, multimodalLengths.size=%zu",
            multimodalHashes.size(), positions.size(), lengths.size());
        return {};
    }

    std::vector<MmKey> extraKeys;
    extraKeys.reserve(positions.size());

    for (size_t itemIdx = 0; itemIdx < positions.size(); ++itemIdx)
    {
        auto const& startPos = positions[itemIdx];
        auto const& length = lengths[itemIdx];

        auto const runEnd = getValidRunEnd(startPos, length);
        if (!runEnd)
        {
            auto const runEnd64 = static_cast<int64_t>(startPos) + static_cast<int64_t>(length);
            TLLM_LOG_WARNING(
                "Invalid legacy multimodal range; disabling multimodal cache hash extra keys: "
                "itemIdx=%zu, startPos=%d, length=%d, runEnd=%lld, maxSizeType32=%d",
                itemIdx, startPos, length, static_cast<long long>(runEnd64), std::numeric_limits<SizeType32>::max());
            return {};
        }

        if (endTokenIdx > startPos && startTokenIdx < *runEnd)
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
    auto const multimodalItemRunCuOffsets = llmRequest.getMultimodalItemRunCuOffsets();
    auto const multimodalRunPositions = llmRequest.getMultimodalRunPositions();
    auto const multimodalRunLengths = llmRequest.getMultimodalRunLengths();

    if (!multimodalHashes || !(*multimodalHashes) || (*multimodalHashes)->empty())
    {
        return {};
    }

    auto const& hashes = *(*multimodalHashes);
    auto const runMetadata
        = resolveRunMetadata(multimodalItemRunCuOffsets, multimodalRunPositions, multimodalRunLengths, hashes.size());
    if (runMetadata.status == RunMetadataStatus::kInvalid)
    {
        return generateLegacyContiguousExtraKeys(
            hashes, multimodalPositions, multimodalLengths, multimodalUuids, startTokenIdx, endTokenIdx);
    }
    if (runMetadata.status == RunMetadataStatus::kPresent)
    {
        auto runBasedExtraKeys
            = generateRunBasedExtraKeys(hashes, multimodalUuids, runMetadata.metadata, startTokenIdx, endTokenIdx);
        if (runBasedExtraKeys)
        {
            return std::move(runBasedExtraKeys.value());
        }
        return generateLegacyContiguousExtraKeys(
            hashes, multimodalPositions, multimodalLengths, multimodalUuids, startTokenIdx, endTokenIdx);
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
            std::move(uniqueTokens), std::move(extraKeys), llmRequest.getCacheSalt());
    }
    return blockKeys;
}

bool BlockKey::operator==(BlockKey const& other) const noexcept
{
    return (usesExtraIds == other.usesExtraIds && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens
        && extraKeys == other.extraKeys && cacheSalt == other.cacheSalt);
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

    if (parentHash == 0 && blockKey.cacheSalt)
    {
        // Only mix the cache salt into the hash for the first block in the sequence.
        uint64_t c = static_cast<uint64_t>(std::hash<std::string>{}(blockKey.cacheSalt.value()));
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
