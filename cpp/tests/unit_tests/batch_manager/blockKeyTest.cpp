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

#include "tensorrt_llm/batch_manager/blockKey.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

class BlockKeyTest : public ::testing::Test
{
protected:
    // Convenience: build an MmKey with a recognisable hash byte pattern.
    static MmKey makeMmKey(uint8_t fill, SizeType32 startOffset)
    {
        std::array<uint8_t, 32> hashBytes{};
        hashBytes.fill(fill);
        return MmKey{hashBytes, startOffset};
    }

    static std::array<uint8_t, 32> makeHashBytes(std::vector<SizeType32> const& hashParts)
    {
        std::array<uint8_t, 32> hashBytes{};
        for (size_t partIdx = 0; partIdx < hashParts.size(); ++partIdx)
        {
            for (uint8_t byteIdx = 0; byteIdx < 4; ++byteIdx)
            {
                hashBytes[partIdx * 4 + byteIdx]
                    = static_cast<uint8_t>((hashParts[partIdx] >> (24 - byteIdx * 8)) & 0xFF);
            }
        }
        return hashBytes;
    }
};

TEST_F(BlockKeyTest, PartialMatch)
{
    VecUniqueTokens tokens0 = {{0, 0}, {0, 0}};
    VecUniqueTokens tokens1 = {{0, 0}};
    BlockKey bk0(false, 0, tokens0);
    BlockKey bk1(false, 0, tokens1);

    bk1.uniqueTokens.reserve(2);
    auto ptr = reinterpret_cast<char*>(bk1.uniqueTokens.data());
    std::fill(ptr, ptr + bk1.uniqueTokens.capacity() * sizeof(UniqueToken), 0);

    EXPECT_EQ(bk0.numMatchingTokens(bk1), 1);
}

// ---------------------------------------------------------------------------
// Equality
// ---------------------------------------------------------------------------

TEST_F(BlockKeyTest, EqualityIdentical)
{
    BlockKey bk0(false, std::nullopt, {{1, 0}, {2, 0}});
    BlockKey bk1(false, std::nullopt, {{1, 0}, {2, 0}});
    EXPECT_EQ(bk0, bk1);
}

TEST_F(BlockKeyTest, EqualityDiffersInUsesExtraIds)
{
    BlockKey bk0(false, std::nullopt, {{1, 0}});
    BlockKey bk1(true, std::nullopt, {{1, 0}});
    EXPECT_FALSE(bk0 == bk1);
}

TEST_F(BlockKeyTest, EqualityDiffersInLoraTaskId)
{
    BlockKey bk0(false, static_cast<LoraTaskIdType>(1), {{1, 0}});
    BlockKey bk1(false, static_cast<LoraTaskIdType>(2), {{1, 0}});
    EXPECT_FALSE(bk0 == bk1);
}

TEST_F(BlockKeyTest, EqualityDiffersInTokens)
{
    BlockKey bk0(false, std::nullopt, {{1, 0}, {2, 0}});
    BlockKey bk1(false, std::nullopt, {{1, 0}, {3, 0}});
    EXPECT_FALSE(bk0 == bk1);
}

TEST_F(BlockKeyTest, EqualityDiffersInExtraKeys)
{
    BlockKey bk0(false, std::nullopt, {{1, 0}}, {makeMmKey(0xAA, 0)});
    BlockKey bk1(false, std::nullopt, {{1, 0}}, {makeMmKey(0xBB, 0)});
    EXPECT_FALSE(bk0 == bk1);
}

TEST_F(BlockKeyTest, EqualityIdenticalWithExtraKeys)
{
    // Two keys that are bit-for-bit identical including extraKeys must compare equal.
    BlockKey bk0(false, std::nullopt, {{1, 0}}, {makeMmKey(0xAA, 5)});
    BlockKey bk1(false, std::nullopt, {{1, 0}}, {makeMmKey(0xAA, 5)});
    EXPECT_EQ(bk0, bk1);
}

// ---------------------------------------------------------------------------
// shorten
// ---------------------------------------------------------------------------

TEST_F(BlockKeyTest, ShortenReducesTokens)
{
    VecUniqueTokens tokens = {{1, 0}, {2, 0}, {3, 0}, {4, 0}};
    BlockKey bk(false, static_cast<LoraTaskIdType>(42), tokens);
    auto shortened = bk.shorten(2);

    EXPECT_EQ(shortened.getNumTokens(), 2);
    EXPECT_EQ(shortened.uniqueTokens[0].tokenId, 1);
    EXPECT_EQ(shortened.uniqueTokens[1].tokenId, 2);
    // Non-token fields must be preserved.
    ASSERT_TRUE(shortened.loraTaskId.has_value());
    EXPECT_EQ(shortened.loraTaskId.value(), static_cast<LoraTaskIdType>(42));
    EXPECT_EQ(shortened.usesExtraIds, false);
}

TEST_F(BlockKeyTest, ShortenToZero)
{
    BlockKey bk(false, std::nullopt, {{1, 0}, {2, 0}});
    auto shortened = bk.shorten(0);
    EXPECT_EQ(shortened.getNumTokens(), 0);
}

TEST_F(BlockKeyTest, ShortenWithExtraKeysFails)
{
    // shorten() must assert when extraKeys is non-empty, because such keys
    // have supportsPartialMatching()==false and must never be shortened.
    BlockKey bk(false, std::nullopt, {{1, 0}, {2, 0}}, {makeMmKey(0x01, 0)});
    EXPECT_THROW(bk.shorten(1), std::exception);
}

// ---------------------------------------------------------------------------
// supportsPartialMatching
// ---------------------------------------------------------------------------

TEST_F(BlockKeyTest, SupportsPartialMatchingNoExtraKeys)
{
    BlockKey bk(false, std::nullopt, {{0, 0}});
    EXPECT_TRUE(bk.supportsPartialMatching());
}

TEST_F(BlockKeyTest, SupportsPartialMatchingWithExtraKeys)
{
    BlockKey bk(false, std::nullopt, {{0, 0}}, {makeMmKey(0x01, 0)});
    EXPECT_FALSE(bk.supportsPartialMatching());
}

// ---------------------------------------------------------------------------
// BlockKeyHasher
// ---------------------------------------------------------------------------

TEST_F(BlockKeyTest, HashIsConsistent)
{
    BlockKey bk(false, std::nullopt, {{1, 0}, {2, 0}});
    EXPECT_EQ(BlockKeyHasher::hash(bk), BlockKeyHasher::hash(bk));
}

TEST_F(BlockKeyTest, HashDiffersForDifferentTokens)
{
    BlockKey bk0(false, std::nullopt, {{1, 0}, {2, 0}});
    BlockKey bk1(false, std::nullopt, {{1, 0}, {3, 0}});
    EXPECT_FALSE(BlockKeyHasher::hash(bk0) == BlockKeyHasher::hash(bk1));
}

TEST_F(BlockKeyTest, HashWithExtraKeys)
{
    // Two keys with identical tokens but different multimodal hashes must differ.
    BlockKey bk0(false, std::nullopt, {{1, 0}}, {makeMmKey(0xAA, 0)});
    BlockKey bk1(false, std::nullopt, {{1, 0}}, {makeMmKey(0xBB, 0)});
    EXPECT_FALSE(BlockKeyHasher::hash(bk0) == BlockKeyHasher::hash(bk1));
}

TEST_F(BlockKeyTest, MultimodalExtraKeysRespectSparsePromptPositions)
{
    using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;

    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};

    // Sparse semantic layout: MM tokens at prompt positions {1, 3}; position 2 is text/framing.
    // Current fields can only approximate this as one contiguous span: position=1, length=2.
    // The exact hash positions keep cache hashing on the sparse MM tokens, not the approximation.
    auto const request
        = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12}, tensorrt_llm::runtime::SamplingConfig{1},
            false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::vector<std::vector<SizeType32>>{hashParts}, std::vector<SizeType32>{1},
            std::vector<SizeType32>{2}, std::nullopt, std::vector<std::vector<SizeType32>>{{1, 3}});
    auto const expectedHash = makeHashBytes(hashParts);

    auto const textGapKeys = generateBlockHashExtraKeys(request, 2, 3);
    EXPECT_TRUE(textGapKeys.empty()) << "Text/framing tokens must not inherit the multimodal content hash.";

    auto const secondSparseMmTokenKeys = generateBlockHashExtraKeys(request, 3, 4);
    ASSERT_EQ(secondSparseMmTokenKeys.size(), 1);
    EXPECT_EQ(secondSparseMmTokenKeys.front().hash, expectedHash);
    EXPECT_EQ(secondSparseMmTokenKeys.front().startOffset, 1);
}

TEST_F(BlockKeyTest, MultimodalExtraKeysFallbackToContiguousSpan)
{
    using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;

    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};
    auto const request = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12},
        tensorrt_llm::runtime::SamplingConfig{1}, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::vector<std::vector<SizeType32>>{hashParts},
        std::vector<SizeType32>{1}, std::vector<SizeType32>{2});
    auto const expectedHash = makeHashBytes(hashParts);

    auto const textGapKeys = generateBlockHashExtraKeys(request, 2, 3);
    ASSERT_EQ(textGapKeys.size(), 1);
    EXPECT_EQ(textGapKeys.front().hash, expectedHash);
    EXPECT_EQ(textGapKeys.front().startOffset, 1);

    auto const secondSparseMmTokenKeys = generateBlockHashExtraKeys(request, 3, 4);
    EXPECT_TRUE(secondSparseMmTokenKeys.empty());
}

TEST_F(BlockKeyTest, MultimodalExtraKeysRejectInvalidExactHashPositions)
{
    using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;

    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};

    auto const emptyHashPositions = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12},
        tensorrt_llm::runtime::SamplingConfig{1}, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::vector<std::vector<SizeType32>>{hashParts},
        std::vector<SizeType32>{1}, std::vector<SizeType32>{2}, std::nullopt, std::vector<std::vector<SizeType32>>{});
    EXPECT_THROW((void) generateBlockHashExtraKeys(emptyHashPositions, 1, 2), std::exception);

    auto const shortHashPositions
        = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12}, tensorrt_llm::runtime::SamplingConfig{1},
            false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::vector<std::vector<SizeType32>>{hashParts}, std::vector<SizeType32>{1},
            std::vector<SizeType32>{2}, std::nullopt, std::vector<std::vector<SizeType32>>{{1}});
    EXPECT_THROW((void) generateBlockHashExtraKeys(shortHashPositions, 1, 2), std::exception);

    auto const unsortedHashPositions
        = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12}, tensorrt_llm::runtime::SamplingConfig{1},
            false, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::vector<std::vector<SizeType32>>{hashParts}, std::vector<SizeType32>{1},
            std::vector<SizeType32>{2}, std::nullopt, std::vector<std::vector<SizeType32>>{{3, 1}});
    EXPECT_THROW((void) generateBlockHashExtraKeys(unsortedHashPositions, 1, 2), std::exception);
}

// ---------------------------------------------------------------------------
// numMatchingTokens edge cases
// ---------------------------------------------------------------------------

TEST_F(BlockKeyTest, NumMatchingTokensLoraIdMismatch)
{
    // Differing loraTaskId → 0 matching tokens regardless of token content.
    BlockKey bk0(false, static_cast<LoraTaskIdType>(1), {{0, 0}, {1, 0}});
    BlockKey bk1(false, static_cast<LoraTaskIdType>(2), {{0, 0}, {1, 0}});
    EXPECT_EQ(bk0.numMatchingTokens(bk1), 0);
}

TEST_F(BlockKeyTest, NumMatchingTokensExtraKeysMismatch)
{
    // Differing extraKeys → 0 matching tokens regardless of token content.
    BlockKey bk0(false, std::nullopt, {{0, 0}, {1, 0}}, {makeMmKey(0xAA, 0)});
    BlockKey bk1(false, std::nullopt, {{0, 0}, {1, 0}}, {makeMmKey(0xAA, 1)});
    EXPECT_EQ(bk0.numMatchingTokens(bk1), 0);
}

TEST_F(BlockKeyTest, NumMatchingTokensFullMatch)
{
    BlockKey bk0(false, std::nullopt, {{0, 0}, {1, 0}, {2, 0}});
    BlockKey bk1(false, std::nullopt, {{0, 0}, {1, 0}, {2, 0}});
    EXPECT_EQ(bk0.numMatchingTokens(bk1), 3);
}
