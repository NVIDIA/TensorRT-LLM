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

#include <utility>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

class BlockKeyTest : public ::testing::Test
{
protected:
    using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
    using ItemRun = LlmRequest::MultimodalItemRun;
    using ItemRuns = LlmRequest::MultimodalItemRuns;

    // Convenience: build an MmKey with a recognisable hash byte pattern.
    static MmKey makeMmKey(uint8_t fill, SizeType32 startOffset)
    {
        std::array<uint8_t, 32> hashBytes{};
        hashBytes.fill(fill);
        return MmKey{hashBytes, startOffset};
    }

    static std::vector<SizeType32> makeHashParts(SizeType32 base)
    {
        std::vector<SizeType32> hashParts;
        hashParts.reserve(8);
        for (SizeType32 partIdx = 0; partIdx < 8; ++partIdx)
        {
            hashParts.push_back(base + partIdx);
        }
        return hashParts;
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

    static LlmRequest makeMultimodalRequest(std::vector<TokenIdType> inputTokens,
        std::vector<std::vector<SizeType32>> multimodalHashes, std::optional<ItemRuns> multimodalItemRuns)
    {
        return LlmRequest(0, 1, inputTokens, tensorrt_llm::runtime::SamplingConfig{1}, false, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::move(multimodalHashes), std::nullopt, std::nullopt, std::move(multimodalItemRuns));
    }

    static ItemRun makeRun(SizeType32 runStart, SizeType32 runLength, std::vector<SizeType32> nonEmbedOffsets = {})
    {
        return ItemRun{runStart, runLength, std::move(nonEmbedOffsets)};
    }

    static ItemRuns makeRunsFromPositions(std::vector<std::vector<SizeType32>> const& positions)
    {
        ItemRuns runs;
        runs.reserve(positions.size());
        for (auto const& itemPositions : positions)
        {
            std::vector<ItemRun> itemRuns;
            if (!itemPositions.empty())
            {
                auto runStart = itemPositions.front();
                auto runLength = static_cast<SizeType32>(1);
                auto previousPosition = runStart;
                for (size_t positionIdx = 1; positionIdx < itemPositions.size(); ++positionIdx)
                {
                    auto const position = itemPositions[positionIdx];
                    if (position == previousPosition + 1)
                    {
                        ++runLength;
                    }
                    else
                    {
                        itemRuns.push_back(makeRun(runStart, runLength));
                        runStart = position;
                        runLength = 1;
                    }
                    previousPosition = position;
                }
                itemRuns.push_back(makeRun(runStart, runLength));
            }
            runs.push_back(std::move(itemRuns));
        }
        return runs;
    }

    static std::list<VecUniqueTokens> makeBlockedUniqueTokens(std::vector<VecTokens> const& tokenBlocks)
    {
        std::list<VecUniqueTokens> blockedUniqueTokens;
        for (auto const& tokenBlock : tokenBlocks)
        {
            VecUniqueTokens uniqueTokens;
            uniqueTokens.reserve(tokenBlock.size());
            for (auto const& token : tokenBlock)
            {
                uniqueTokens.push_back(UniqueToken{token, 0});
            }
            blockedUniqueTokens.push_back(std::move(uniqueTokens));
        }
        return blockedUniqueTokens;
    }

    static std::vector<size_t> hashBlockKeySequence(std::vector<BlockKey> const& blockKeys)
    {
        std::vector<size_t> hashes;
        hashes.reserve(blockKeys.size());
        size_t parentHash = 0;
        for (auto const& blockKey : blockKeys)
        {
            parentHash = BlockKeyHasher::hash(blockKey, parentHash);
            hashes.push_back(parentHash);
        }
        return hashes;
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

TEST_F(BlockKeyTest, MultimodalExtraKeysItemRunsCoverContiguousSpan)
{
    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};

    auto const request = makeMultimodalRequest(std::vector<TokenIdType>{11, 999, 999, 12},
        std::vector<std::vector<SizeType32>>{hashParts}, ItemRuns{{makeRun(1, 2)}});
    auto const expectedHash = makeHashBytes(hashParts);

    auto const firstTokenKeys = generateBlockHashExtraKeys(request, 1, 2);
    ASSERT_EQ(firstTokenKeys.size(), 1);
    EXPECT_EQ(firstTokenKeys.front().hash, expectedHash);
    EXPECT_EQ(firstTokenKeys.front().startOffset, 0);

    auto const secondTokenKeys = generateBlockHashExtraKeys(request, 2, 3);
    ASSERT_EQ(secondTokenKeys.size(), 1);
    EXPECT_EQ(secondTokenKeys.front().hash, expectedHash);
    EXPECT_EQ(secondTokenKeys.front().startOffset, 1);
}

TEST_F(BlockKeyTest, MultimodalExtraKeysRespectSparsePromptPositionRuns)
{
    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};

    auto const request = LlmRequest(0, 1, std::vector<TokenIdType>{11, 999, 77, 999, 12},
        tensorrt_llm::runtime::SamplingConfig{1}, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::vector<std::vector<SizeType32>>{hashParts},
        std::nullopt, std::nullopt, ItemRuns{{makeRun(1, 1), makeRun(3, 1)}});
    auto const expectedHash = makeHashBytes(hashParts);

    auto const textGapKeys = generateBlockHashExtraKeys(request, 2, 3);
    EXPECT_TRUE(textGapKeys.empty()) << "Text/framing tokens must not inherit the multimodal content hash.";

    auto const secondSparseMmTokenKeys = generateBlockHashExtraKeys(request, 3, 4);
    ASSERT_EQ(secondSparseMmTokenKeys.size(), 1);
    EXPECT_EQ(secondSparseMmTokenKeys.front().hash, expectedHash);
    EXPECT_EQ(secondSparseMmTokenKeys.front().startOffset, 1);
}

TEST_F(BlockKeyTest, MultimodalExtraKeysKeepNonEmbedOffsetsInHashCoverage)
{
    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};
    auto const request = makeMultimodalRequest(std::vector<TokenIdType>{11, 999, 998, 999, 12},
        std::vector<std::vector<SizeType32>>{hashParts}, ItemRuns{{makeRun(1, 3, {1})}});
    auto const expectedHash = makeHashBytes(hashParts);

    auto const nonEmbedOffsetKeys = generateBlockHashExtraKeys(request, 2, 3);
    ASSERT_EQ(nonEmbedOffsetKeys.size(), 1);
    EXPECT_EQ(nonEmbedOffsetKeys.front().hash, expectedHash);
    EXPECT_EQ(nonEmbedOffsetKeys.front().startOffset, 1);

    auto const followingMmTokenKeys = generateBlockHashExtraKeys(request, 3, 4);
    ASSERT_EQ(followingMmTokenKeys.size(), 1);
    EXPECT_EQ(followingMmTokenKeys.front().hash, expectedHash);
    EXPECT_EQ(followingMmTokenKeys.front().startOffset, 2);
}

TEST_F(BlockKeyTest, MultimodalExtraKeysStressMultipleSparseItemsAcrossBlockBoundaries)
{
    auto const firstHashParts = makeHashParts(0x01020300);
    auto const secondHashParts = makeHashParts(0x11121300);
    auto const request = makeMultimodalRequest(std::vector<TokenIdType>{10, 999, 20, 998, 999, 999, 30, 31, 998, 40},
        std::vector<std::vector<SizeType32>>{firstHashParts, secondHashParts},
        makeRunsFromPositions({{1, 4, 5}, {3, 8}}));
    auto const firstHash = makeHashBytes(firstHashParts);
    auto const secondHash = makeHashBytes(secondHashParts);

    auto const firstBlockKeys = generateBlockHashExtraKeys(request, 0, 4);
    ASSERT_EQ(firstBlockKeys.size(), 2);
    EXPECT_EQ(firstBlockKeys[0].hash, firstHash);
    EXPECT_EQ(firstBlockKeys[0].startOffset, 0);
    EXPECT_EQ(firstBlockKeys[1].hash, secondHash);
    EXPECT_EQ(firstBlockKeys[1].startOffset, 0);

    auto const secondBlockKeys = generateBlockHashExtraKeys(request, 4, 8);
    ASSERT_EQ(secondBlockKeys.size(), 1);
    EXPECT_EQ(secondBlockKeys.front().hash, firstHash);
    EXPECT_EQ(secondBlockKeys.front().startOffset, 1);

    auto const blockGapKeys = generateBlockHashExtraKeys(request, 6, 8);
    EXPECT_TRUE(blockGapKeys.empty());

    auto const thirdBlockKeys = generateBlockHashExtraKeys(request, 8, 12);
    ASSERT_EQ(thirdBlockKeys.size(), 1);
    EXPECT_EQ(thirdBlockKeys.front().hash, secondHash);
    EXPECT_EQ(thirdBlockKeys.front().startOffset, 1);
}

TEST_F(BlockKeyTest, BuildBlockKeysStressRepeatedHashesAndSparsePositionIdentity)
{
    auto const repeatedHashParts = makeHashParts(0x01010100);
    auto const differentHashParts = makeHashParts(0x02020200);

    auto makeKeys
        = [&](std::vector<std::vector<SizeType32>> hashes, std::vector<std::vector<SizeType32>> exactPositions)
    {
        auto const request
            = makeMultimodalRequest(std::vector<TokenIdType>{10, 999, 20, 21, 999, 22, 23, 24, 999, 25, 26, 27},
                std::move(hashes), makeRunsFromPositions(exactPositions));
        auto blockedUniqueTokens
            = makeBlockedUniqueTokens(std::vector<VecTokens>{{10, 999, 20, 21}, {999, 22, 23, 24}, {999, 25, 26, 27}});
        return buildBlockKeys(blockedUniqueTokens, request);
    };

    auto const keysWithRepeatedHash = makeKeys({repeatedHashParts, repeatedHashParts}, {{1}, {4}});
    auto const identicalKeys = makeKeys({repeatedHashParts, repeatedHashParts}, {{1}, {4}});
    auto const differentPositionKeys = makeKeys({repeatedHashParts, repeatedHashParts}, {{1}, {8}});
    auto const differentHashKeys = makeKeys({repeatedHashParts, differentHashParts}, {{1}, {4}});
    auto const repeatedHash = makeHashBytes(repeatedHashParts);

    ASSERT_EQ(keysWithRepeatedHash.size(), 3);
    ASSERT_EQ(keysWithRepeatedHash[0].extraKeys.size(), 1);
    ASSERT_EQ(keysWithRepeatedHash[1].extraKeys.size(), 1);
    EXPECT_EQ(keysWithRepeatedHash[0].extraKeys.front().hash, repeatedHash);
    EXPECT_EQ(keysWithRepeatedHash[0].extraKeys.front().startOffset, 0);
    EXPECT_EQ(keysWithRepeatedHash[1].extraKeys.front().hash, repeatedHash);
    EXPECT_EQ(keysWithRepeatedHash[1].extraKeys.front().startOffset, 0);

    EXPECT_EQ(keysWithRepeatedHash, identicalKeys);
    EXPECT_EQ(hashBlockKeySequence(keysWithRepeatedHash), hashBlockKeySequence(identicalKeys));

    EXPECT_NE(keysWithRepeatedHash, differentPositionKeys);
    EXPECT_NE(hashBlockKeySequence(keysWithRepeatedHash), hashBlockKeySequence(differentPositionKeys));

    EXPECT_NE(keysWithRepeatedHash, differentHashKeys);
    EXPECT_NE(hashBlockKeySequence(keysWithRepeatedHash), hashBlockKeySequence(differentHashKeys));
}

TEST_F(BlockKeyTest, MultimodalExtraKeysRejectInvalidItemRuns)
{
    auto const hashParts = std::vector<SizeType32>{
        0x01020304, 0x05060708, 0x11121314, 0x15161718, 0x21222324, 0x25262728, 0x31323334, 0x35363738};

    auto makeInvalidRequest = [&](ItemRuns runs)
    {
        return makeMultimodalRequest(std::vector<TokenIdType>{11, 999, 77, 999, 12},
            std::vector<std::vector<SizeType32>>{hashParts}, std::move(runs));
    };

    auto const emptyOuterRuns = makeMultimodalRequest(
        std::vector<TokenIdType>{11, 999, 77, 999, 12}, std::vector<std::vector<SizeType32>>{hashParts}, ItemRuns{});
    EXPECT_THROW((void) generateBlockHashExtraKeys(emptyOuterRuns, 1, 2), std::exception);

    EXPECT_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{}}), 1, 2), std::exception);
    EXPECT_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 0), makeRun(3, 2)}}), 1, 2),
        std::exception);
    EXPECT_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(-1, 1), makeRun(3, 1)}}), 1, 2),
        std::exception);
    EXPECT_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 1), makeRun(5, 1)}}), 1, 2),
        std::exception);
    EXPECT_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 2), makeRun(2, 1)}}), 1, 2),
        std::exception);
    auto const overlappingItems = makeMultimodalRequest(std::vector<TokenIdType>{11, 999, 77, 999, 12},
        std::vector<std::vector<SizeType32>>{hashParts, makeHashParts(0x11121300)},
        ItemRuns{{makeRun(1, 2)}, {makeRun(2, 1)}});
    EXPECT_THROW((void) generateBlockHashExtraKeys(overlappingItems, 1, 3), std::exception);
    EXPECT_NO_THROW(
        (void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(2, 1), makeRun(3, 1)}}), 1, 2));
    EXPECT_NO_THROW((void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 1)}}), 1, 2));
    EXPECT_THROW(
        (void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 2, {-1})}}), 1, 2), std::exception);
    EXPECT_THROW(
        (void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 2, {2})}}), 1, 2), std::exception);
    EXPECT_THROW(
        (void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 3, {2, 1})}}), 1, 2), std::exception);
    EXPECT_THROW(
        (void) generateBlockHashExtraKeys(makeInvalidRequest(ItemRuns{{makeRun(1, 3, {1, 1})}}), 1, 2), std::exception);

    auto const outerMismatch = makeMultimodalRequest(std::vector<TokenIdType>{11, 999, 77, 999, 12},
        std::vector<std::vector<SizeType32>>{hashParts, makeHashParts(0x11121300)}, ItemRuns{{makeRun(1, 1)}});
    EXPECT_THROW((void) generateBlockHashExtraKeys(outerMismatch, 1, 2), std::exception);
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
