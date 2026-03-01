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

namespace
{
inline uint8_t getNthByte(tensorrt_llm::runtime::SizeType32 hashPart, uint8_t byteIdx) noexcept
{
    return static_cast<uint8_t>((hashPart >> (24 - byteIdx * 8)) & 0xFF);
}
}

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
std::vector<MmKey> generateBlockHashExtraKeys(
    tensorrt_llm::batch_manager::LlmRequest const& llmRequest, SizeType32 startTokenIdx, SizeType32 endTokenIdx)
{
    auto const multimodalHashes = llmRequest.getMultimodalHashes();
    auto const multimodalPositions = llmRequest.getMultimodalPositions();
    auto const multimodalLengths = llmRequest.getMultimodalLengths();

    if (!multimodalHashes || !multimodalPositions || !multimodalLengths || !(*multimodalHashes)
        || (*multimodalHashes)->empty() || !(*multimodalPositions) || (*multimodalPositions)->empty()
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

    std::vector<MmKey> extraKeys; // MmKey = std::pair<std::array<uint8_t, 32>, SizeType32>
    extraKeys.reserve((*multimodalPositions)->size());
    std::array<uint8_t, 32> mmHashArray;

    for (size_t i = 0; i < (*multimodalPositions)->size(); ++i)
    {
        auto const& startPos = (*(*multimodalPositions))[i];
        auto const& length = (*(*multimodalLengths))[i];
        auto const& mmHashVector = (*(*multimodalHashes))[i];

        TLLM_CHECK_WITH_INFO(mmHashVector.size() == 8, "Multimodal hash vector has unexpected size: %zu (expected 8)",
            mmHashVector.size());

        // mmHashVector[j] comes from Python's int(hex_chunk, 16)
        // where hex_chunk like "00010203" means 0x00 is MSB and 0x03 is LSB (big endian)
        // Convert 8x 32-bit integers into a 32-byte array preserving Blake3 hash byte order
        // Example: hashPart = 0x00010203 → mmHashArray[0:3] = [0x00, 0x01, 0x02, 0x03]
        for (size_t j = 0; j < 8; ++j)
        {
            auto const& hashPart = mmHashVector[j];
            for (uint8_t byteIdx = 0; byteIdx < 4; ++byteIdx)
            {
                mmHashArray[j * 4 + byteIdx] = getNthByte(hashPart, byteIdx);
            }
        }

        // Check if this multimodal content overlaps with the current block
        if (endTokenIdx > startPos && startTokenIdx < startPos + length)
        {
            uint64_t mmStartInBlock = (startPos >= startTokenIdx) ? 0 : static_cast<uint64_t>(startTokenIdx - startPos);
            extraKeys.emplace_back(mmHashArray, mmStartInBlock);
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
        seed = __hash64(blockKey.cacheSaltID.value(), seed);
    }

    for (auto const& uniqueToken : blockKey.uniqueTokens)
    {
        seed = __hash32(static_cast<uint32_t>(uniqueToken.tokenId), seed);
        if (blockKey.usesExtraIds)
        {
            seed = __hash64(uniqueToken.tokenExtraId, seed);
        }
    }

    if (blockKey.loraTaskId)
    {
        seed = __hash64(blockKey.loraTaskId.value(), seed);
    }

    // Add extra keys for multimodal data mixing in external multimodal item hash and token offset within this sequence
    // block
    if (!blockKey.extraKeys.empty())
    {
        for (auto const& [mmHash, startOffset] : blockKey.extraKeys)
        {
            // Hash the multimodal hash array in 32-bit chunks (more efficient)
            for (size_t i = 0; i < 32; i += 4)
            {
                // Combine 4 bytes into a 32-bit word (construct as little endian order)
                uint32_t word = static_cast<uint32_t>(mmHash[i]) | (static_cast<uint32_t>(mmHash[i + 1]) << 8)
                    | (static_cast<uint32_t>(mmHash[i + 2]) << 16) | (static_cast<uint32_t>(mmHash[i + 3]) << 24);

                // Mix the word into the seed
                seed = __hash32(word, seed);
            }

            // Hash the start offset
            seed = __hash64(static_cast<uint64_t>(startOffset), seed);
        }
    }

    return seed;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
