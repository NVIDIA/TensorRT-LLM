/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file kvCacheManagerTestNew.cpp
 * @brief Behavioral unit tests for KVCacheManager with improved coverage.
 *
 * These tests focus on observable, semantic behavior rather than internal block-ID
 * assignments. The guiding principles are:
 *
 *   - Verify free/used block counts and their invariants (free + used == total).
 *   - Verify context-position (prepopulated prompt length) as a proxy for how many
 *     tokens were satisfied from the cache rather than re-computed.
 *   - Verify per-request and aggregate statistics (reused blocks, missed blocks,
 *     cache hit rate).
 *   - Verify capacity constraints and eviction semantics.
 *   - Verify static utility calculations.
 *
 * Block-ID assertions are deliberately avoided because IDs are an implementation
 * detail that can legitimately change with eviction-order or tree-rebalancing
 * changes.  Behavior-oriented assertions (counts, positions, rates) are stable
 * across refactors while still catching real regressions.
 */

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

// ============================================================================
// Test fixture
// ============================================================================

class KVCacheManagerNewTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    // -----------------------------------------------------------------------
    // Factory helpers
    // -----------------------------------------------------------------------

    /**
     * Build a full-attention KVCacheManager with block reuse enabled.
     *
     * @param numBlocks       Total primary blocks in the pool.
     * @param tokensPerBlock  Tokens per block.
     * @param maxSeqs         Maximum concurrent sequences.
     * @param numLayers       Number of model layers.
     * @param numKvHeads      KV heads per layer.
     * @param sizePerHead     Head dimension.
     */
    std::unique_ptr<KVCacheManager> makeManager(SizeType32 numBlocks, SizeType32 tokensPerBlock, SizeType32 maxSeqs,
        SizeType32 numLayers = 4, SizeType32 numKvHeads = 2, SizeType32 sizePerHead = 64)
    {
        auto const maxAttentionWindow = numBlocks * tokensPerBlock;
        auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {numBlocks, 0}}};
        auto const stream = std::make_shared<tr::CudaStream>();

        auto mgr = std::make_unique<KVCacheManager>(numLayers, numKvHeads, sizePerHead, tokensPerBlock,
            blocksPerWindow, maxSeqs, /*maxBeamWidth=*/1,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            /*tempAttentionWindowInputs=*/std::nullopt, nvinfer1::DataType::kHALF,
            /*sinkTokenLength=*/0, stream, /*maxSequenceLength=*/maxAttentionWindow,
            /*enableBlockReuse=*/true, /*onboardBlocks=*/true);
        mgr->allocatePools(/*useUvm=*/false);
        return mgr;
    }

    /**
     * Build a manager with sliding-window attention (block reuse disabled so
     * that window-slide tests are not confounded by reuse).
     */
    std::unique_ptr<KVCacheManager> makeWindowedManager(
        SizeType32 numBlocks, SizeType32 tokensPerBlock, SizeType32 maxSeqs, SizeType32 attentionWindow)
    {
        auto const blocksPerWindow = BlocksPerWindow{{attentionWindow, {numBlocks, 0}}};
        auto const stream = std::make_shared<tr::CudaStream>();
        auto const maxSequenceLength = numBlocks * tokensPerBlock;

        auto mgr = std::make_unique<KVCacheManager>(/*numLayers=*/4, /*numKvHeads=*/2, /*sizePerHead=*/64,
            tokensPerBlock, blocksPerWindow, maxSeqs, /*maxBeamWidth=*/1,
            std::vector<BlockManager::SizeType32>{attentionWindow},
            /*tempAttentionWindowInputs=*/std::nullopt, nvinfer1::DataType::kHALF,
            /*sinkTokenLength=*/0, stream, maxSequenceLength,
            /*enableBlockReuse=*/false, /*onboardBlocks=*/true);
        mgr->allocatePools(false);
        return mgr;
    }

    /**
     * Create a simple LlmRequest for a given token sequence.
     */
    static std::shared_ptr<LlmRequest> makeRequest(
        LlmRequest::RequestIdType id, std::vector<TokenIdType> const& tokens, SizeType32 maxNewTokens = 0)
    {
        tr::SamplingConfig const samplingConfig{1};
        auto tokenVec = std::make_shared<VecTokens>(tokens);
        return std::make_shared<LlmRequest>(id, maxNewTokens, tokenVec, samplingConfig, /*isStreaming=*/false);
    }

    /**
     * Add a sequence to the manager and return the context position
     * (tokens pre-populated from the KV cache).
     */
    static SizeType32 addSeq(KVCacheManager& mgr, std::shared_ptr<LlmRequest> const& req)
    {
        auto const id = req->getRequestId();
        auto const inputLen = static_cast<SizeType32>(req->getNumTokens(0));
        mgr.addSequence(id, inputLen, /*beamWidth=*/1, req);
        return req->getContextCurrentPosition();
    }

    /**
     * Remove a sequence from the manager (discarding the optional last-block ID).
     */
    static void removeSeq(KVCacheManager& mgr, std::shared_ptr<LlmRequest> const& req)
    {
        (void) mgr.removeSequence(req->getRequestId(), req);
    }

    // -----------------------------------------------------------------------
    // Pool invariant checker
    // -----------------------------------------------------------------------

    /**
     * Assert the fundamental invariant: free + used == total.
     */
    static void checkPoolInvariant(KVCacheManager const& mgr)
    {
        auto const total = mgr.getMaxNumBlocks();
        auto const used  = mgr.getUsedNumBlocks();
        auto const free  = mgr.getNumFreeBlocks();
        EXPECT_EQ(used + free, total)
            << "Pool invariant violated: used=" << used << " free=" << free << " total=" << total;
    }
};

// ============================================================================
// Section 1 – Pool invariants
// ============================================================================

/**
 * The pool invariant (free + used == total) must hold after construction,
 * after addSequence, after addToken, and after removeSequence.
 */
TEST_F(KVCacheManagerNewTest, PoolInvariantHoldsAtAllStages)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 8;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    checkPoolInvariant(*mgr);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
    EXPECT_EQ(mgr->getUsedNumBlocks(), 0);

    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7});
    addSeq(*mgr, req0);
    checkPoolInvariant(*mgr);
    EXPECT_GT(mgr->getUsedNumBlocks(), 0);

    mgr->addToken(0);
    req0->addNewToken(8, 0);
    checkPoolInvariant(*mgr);

    removeSeq(*mgr, req0);
    checkPoolInvariant(*mgr);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks) << "All blocks must be free after removal";
}

/**
 * Pool invariant must hold with multiple concurrent sequences.
 */
TEST_F(KVCacheManagerNewTest, PoolInvariantWithMultipleConcurrentSequences)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 16;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    auto req0 = makeRequest(0, {10, 11, 12, 13});
    auto req1 = makeRequest(1, {20, 21, 22, 23, 24, 25, 26, 27});
    auto req2 = makeRequest(2, {30, 31, 32, 33, 34, 35});

    addSeq(*mgr, req0);
    checkPoolInvariant(*mgr);
    addSeq(*mgr, req1);
    checkPoolInvariant(*mgr);
    addSeq(*mgr, req2);
    checkPoolInvariant(*mgr);

    removeSeq(*mgr, req1);
    checkPoolInvariant(*mgr);
    removeSeq(*mgr, req0);
    checkPoolInvariant(*mgr);
    removeSeq(*mgr, req2);
    checkPoolInvariant(*mgr);

    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
}

// ============================================================================
// Section 2 – Cache hits detected via context position
// ============================================================================

/**
 * The first request ever has nothing to reuse: context position must be 0.
 */
TEST_F(KVCacheManagerNewTest, FirstRequestNeverReuses)
{
    auto mgr = makeManager(/*numBlocks=*/8, /*tokensPerBlock=*/4, /*maxSeqs=*/4);
    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    SizeType32 ctxPos = addSeq(*mgr, req);
    EXPECT_EQ(ctxPos, 0) << "First request must start at context position 0 (empty cache)";
    removeSeq(*mgr, req);
}

/**
 * A request with the same tokens as a previously-released request must show a
 * non-zero context position (some blocks were served from the cache).
 */
TEST_F(KVCacheManagerNewTest, SamePrefixIsReused)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    SizeType32 ctxPos = addSeq(*mgr, req1);
    EXPECT_GT(ctxPos, 0)
        << "Second identical request should reuse cached blocks (ctxPos > 0)";
    removeSeq(*mgr, req1);
}

/**
 * A request with completely different tokens must not reuse any cached blocks.
 */
TEST_F(KVCacheManagerNewTest, DifferentTokensAreNotReused)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    auto req0 = makeRequest(0, {10, 11, 12, 13, 14, 15, 16, 17});
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, {90, 91, 92, 93, 94, 95, 96, 97});
    SizeType32 ctxPos = addSeq(*mgr, req1);
    EXPECT_EQ(ctxPos, 0) << "Entirely different tokens should not reuse any blocks";
    removeSeq(*mgr, req1);
}

/**
 * Partial prefix match: context position must equal the number of tokens in the
 * matching full blocks only (the last partial block is not reused as-is).
 */
TEST_F(KVCacheManagerNewTest, PartialPrefixReuseMatchesFullBlocksOnly)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    // Cache two full blocks: [0,1,2,3] and [4,5,6,7]
    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7});
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    // Second request shares only the first block
    auto req1 = makeRequest(1, {0, 1, 2, 3, 99, 100, 101, 102});
    SizeType32 ctxPos = addSeq(*mgr, req1);
    EXPECT_EQ(ctxPos, tokensPerBlock)
        << "Only the matching full block should be pre-populated from cache";
    removeSeq(*mgr, req1);
}

/**
 * When a request exactly matches a full chain of blocks, context position must
 * be (total tokens - 1): the last token is never pre-populated.
 */
TEST_F(KVCacheManagerNewTest, FullMatchPrepopulatesAllButLastToken)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numTokens = 8; // exactly 2 full blocks
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens(numTokens);
    std::iota(tokens.begin(), tokens.end(), 0);

    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, tokens);
    SizeType32 ctxPos = addSeq(*mgr, req1);
    EXPECT_EQ(ctxPos, numTokens - 1)
        << "Full match should pre-populate all tokens except the last one";
    removeSeq(*mgr, req1);
}

// ============================================================================
// Section 3 – Reuse statistics
// ============================================================================

/**
 * With no reuse opportunity, per-request stats must report:
 *   reused == 0, allocNew > 0, missed == allocNew.
 */
TEST_F(KVCacheManagerNewTest, PerRequestStatsShowZeroReuseOnFirstRequest)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    addSeq(*mgr, req);

    EXPECT_EQ(req->getReusedBlocksPerRequest(), 0);
    EXPECT_GT(req->getAllocNewBlocksPerRequest(), 0);
    EXPECT_EQ(req->getMissedBlocksPerRequest(), req->getAllocNewBlocksPerRequest());

    removeSeq(*mgr, req);
}

/**
 * After a successful prefix reuse, per-request stats must show:
 *   reused > 0 and cacheHitRate in (0, 1].
 */
TEST_F(KVCacheManagerNewTest, PerRequestStatsShowReuseOnSecondRequest)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, tokens);
    addSeq(*mgr, req1);

    EXPECT_GT(req1->getReusedBlocksPerRequest(), 0)
        << "At least one block should be reused on the second identical request";
    float hitRate = req1->getKVCacheHitRatePerRequest();
    EXPECT_GT(hitRate, 0.0f) << "Cache hit rate must be positive after reuse";
    EXPECT_LE(hitRate, 1.0f) << "Cache hit rate must not exceed 1.0";

    removeSeq(*mgr, req1);
}

/**
 * Aggregate KvCacheStats must be internally consistent:
 *   free + used == total
 *   cacheHitRate == reused / (reused + missed)  when reused > 0
 */
TEST_F(KVCacheManagerNewTest, AggregateStatsInternallyConsistent)
{
    auto mgr = makeManager(/*numBlocks=*/8, /*tokensPerBlock=*/4, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, tokens);
    addSeq(*mgr, req1);

    KvCacheStats stats = mgr->getKvCacheStats();

    EXPECT_EQ(stats.maxNumBlocks, mgr->getMaxNumBlocks());
    EXPECT_EQ(stats.freeNumBlocks + stats.usedNumBlocks, stats.maxNumBlocks);
    EXPECT_EQ(stats.toksPerBlock, mgr->getTokensPerBlock());

    if (stats.reusedBlocks > 0)
    {
        float expectedHitRate = static_cast<float>(stats.reusedBlocks)
            / static_cast<float>(stats.reusedBlocks + stats.missedBlocks);
        EXPECT_NEAR(stats.cacheHitRate, expectedHitRate, 1e-5f)
            << "cacheHitRate must equal reused/(reused+missed)";
    }
    else
    {
        EXPECT_FLOAT_EQ(stats.cacheHitRate, 0.0f);
    }

    removeSeq(*mgr, req1);
}

/**
 * allocTotalBlocks == allocNewBlocks + reusedBlocks at all times.
 */
TEST_F(KVCacheManagerNewTest, TotalAllocEqualsNewPlusReused)
{
    auto mgr = makeManager(/*numBlocks=*/8, /*tokensPerBlock=*/4, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto req1 = makeRequest(1, tokens);
    addSeq(*mgr, req1);

    KvCacheStats stats = mgr->getKvCacheStats();
    EXPECT_EQ(stats.allocTotalBlocks, stats.allocNewBlocks + stats.reusedBlocks);

    removeSeq(*mgr, req1);
}

// ============================================================================
// Section 4 – Concurrent sequences and prefix block sharing
// ============================================================================

/**
 * Two concurrent requests sharing a common prefix must use fewer total blocks
 * than if they were allocated independently (sharing reduces pool pressure).
 */
TEST_F(KVCacheManagerNewTest, SharedPrefixReducesBlockUsage)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 16;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    // Each request: 2-block shared prefix + 1-block unique suffix (= 3 blocks isolated).
    // With sharing: 2 shared + 1 + 1 = 4 blocks total.
    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto req1 = makeRequest(1, {0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23});

    addSeq(*mgr, req0);
    addSeq(*mgr, req1);

    // Without sharing: 3 + 3 = 6 blocks; with sharing should be < 6.
    SizeType32 usedWithSharing = mgr->getUsedNumBlocks();
    EXPECT_LT(usedWithSharing, 6)
        << "Shared prefix should reduce total allocation; expected < 6, got " << usedWithSharing;

    removeSeq(*mgr, req0);
    removeSeq(*mgr, req1);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
}

/**
 * Two concurrent sequences with entirely different tokens must not share any
 * blocks.
 */
TEST_F(KVCacheManagerNewTest, NoSharingForDifferentTokens)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 16;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7});
    auto req1 = makeRequest(1, {100, 101, 102, 103, 104, 105, 106, 107});

    addSeq(*mgr, req0);
    SizeType32 after0 = mgr->getUsedNumBlocks();
    addSeq(*mgr, req1);
    SizeType32 req1Blocks = mgr->getUsedNumBlocks() - after0;

    EXPECT_EQ(req1->getReusedBlocksPerRequest(), 0)
        << "Entirely different tokens should not reuse any blocks";
    EXPECT_EQ(req1Blocks, after0)
        << "Same-length, different-token sequences should consume equal block counts";

    removeSeq(*mgr, req0);
    removeSeq(*mgr, req1);
}

// ============================================================================
// Section 5 – Generation phase (addToken)
// ============================================================================

/**
 * addToken must eventually trigger a new block allocation once the current
 * block becomes full.
 */
TEST_F(KVCacheManagerNewTest, AddTokenAllocatesBlockWhenCurrentFull)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 8;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    auto req = makeRequest(0, {0, 1, 2, 3}, /*maxNewTokens=*/16);
    addSeq(*mgr, req);
    SizeType32 blocksAfterContext = mgr->getUsedNumBlocks();

    // Fill the current block and overflow
    for (int i = 0; i < tokensPerBlock; ++i)
    {
        mgr->addToken(0);
        req->addNewToken(static_cast<TokenIdType>(10 + i), 0);
    }

    EXPECT_GT(mgr->getUsedNumBlocks(), blocksAfterContext)
        << "A new block must be allocated when the current block overflows";
    checkPoolInvariant(*mgr);

    removeSeq(*mgr, req);
}

/**
 * All blocks (context + generation) must be released when a generating
 * sequence is removed.
 */
TEST_F(KVCacheManagerNewTest, GenerationBlocksReleasedOnRemove)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 8;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7}, /*maxNewTokens=*/8);
    addSeq(*mgr, req);

    for (int i = 0; i < 6; ++i)
    {
        mgr->addToken(0);
        req->addNewToken(static_cast<TokenIdType>(100 + i), 0);
    }

    checkPoolInvariant(*mgr);
    EXPECT_GT(mgr->getUsedNumBlocks(), 0);

    removeSeq(*mgr, req);

    checkPoolInvariant(*mgr);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks)
        << "All blocks must be returned to the pool after removing a generating sequence";
}

// ============================================================================
// Section 6 – Capacity and eviction
// ============================================================================

/**
 * When all blocks are held in the reuse pool (released, not active), a new
 * sequence with a different prefix should succeed by evicting the cached blocks.
 */
TEST_F(KVCacheManagerNewTest, EvictionSatisfiesNewSequenceWhenPoolFull)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 4;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    // Populate the reuse pool with 3 blocks
    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);

    // New request with entirely different tokens, needs all 4 blocks
    auto req1 = makeRequest(1, {50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65});
    EXPECT_NO_THROW(addSeq(*mgr, req1))
        << "New sequence should succeed by evicting cached blocks";
    checkPoolInvariant(*mgr);

    removeSeq(*mgr, req1);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
}

/**
 * Attempting to add a sequence when all blocks are actively held and capacity
 * is fully consumed must throw.
 */
TEST_F(KVCacheManagerNewTest, ExceedingCapacityThrows)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 4;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    // Occupy all 4 blocks with an active sequence
    auto req0 = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    addSeq(*mgr, req0);
    EXPECT_EQ(mgr->getNumFreeBlocks(), 0);

    // Adding even a single-token sequence must throw
    auto req1 = makeRequest(1, {99});
    EXPECT_THROW(addSeq(*mgr, req1), std::exception)
        << "Adding a sequence when all blocks are actively held must throw";

    removeSeq(*mgr, req0);
}

// ============================================================================
// Section 7 – Reuse state reset
// ============================================================================

/**
 * After resetReuseState(), the cache is cleared: a request with previously
 * cached tokens must start from context position 0.
 */
TEST_F(KVCacheManagerNewTest, ResetReuseStateClearsCache)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/8, tokensPerBlock, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7};

    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    // Verify reuse occurs before reset
    auto req1 = makeRequest(1, tokens);
    SizeType32 ctxBeforeReset = addSeq(*mgr, req1);
    removeSeq(*mgr, req1);
    EXPECT_GT(ctxBeforeReset, 0) << "Pre-condition: reuse must occur before reset";

    mgr->resetReuseState();

    // After reset the same tokens must not be found
    auto req2 = makeRequest(2, tokens);
    SizeType32 ctxAfterReset = addSeq(*mgr, req2);
    EXPECT_EQ(ctxAfterReset, 0)
        << "After resetReuseState() no cached blocks should be found for the same tokens";
    removeSeq(*mgr, req2);
}

// ============================================================================
// Section 8 – Static utility functions
// ============================================================================

/**
 * calculateMaxBlockRequirements must be monotonically non-decreasing in both
 * inputLength and outputLength.
 */
TEST_F(KVCacheManagerNewTest, MaxBlockRequirementsMonotoneInLengths)
{
    constexpr SizeType32 sinkLen = 0;
    constexpr SizeType32 windowSize = 1024;
    constexpr SizeType32 beamWidth = 1;
    constexpr SizeType32 tokensPerBlock = 4;

    SizeType32 prev = 0;
    for (SizeType32 inputLen : {1, 4, 8, 16, 32, 64})
    {
        SizeType32 blocks = KVCacheManager::calculateMaxBlockRequirements(
            inputLen, /*outputLen=*/4, sinkLen, windowSize, beamWidth, tokensPerBlock);
        EXPECT_GE(blocks, prev)
            << "Block count must not decrease as inputLength grows";
        prev = blocks;
    }

    prev = 0;
    for (SizeType32 outputLen : {1, 4, 8, 16, 32, 64})
    {
        SizeType32 blocks = KVCacheManager::calculateMaxBlockRequirements(
            /*inputLen=*/8, outputLen, sinkLen, windowSize, beamWidth, tokensPerBlock);
        EXPECT_GE(blocks, prev)
            << "Block count must not decrease as outputLength grows";
        prev = blocks;
    }
}

/**
 * calculateMaxBlockRequirements must return at least 1 for any positive lengths.
 */
TEST_F(KVCacheManagerNewTest, MaxBlockRequirementsAtLeastOne)
{
    SizeType32 blocks = KVCacheManager::calculateMaxBlockRequirements(
        /*inputLen=*/1, /*outputLen=*/1, /*sinkLen=*/0,
        /*windowSize=*/1024, /*beamWidth=*/1, /*tokensPerBlock=*/4);
    EXPECT_GE(blocks, 1);
}

/**
 * getMaxCapacityBatchSize must return at least 1 when sequences are short.
 */
TEST_F(KVCacheManagerNewTest, MaxCapacityBatchSizeAtLeastOne)
{
    auto mgr = makeManager(/*numBlocks=*/16, /*tokensPerBlock=*/4, /*maxSeqs=*/4);
    SizeType32 cap = mgr->getMaxCapacityBatchSize(/*inputLength=*/4, /*outputLength=*/4);
    EXPECT_GE(cap, 1);
}

/**
 * getMaxCapacityBatchSize must decrease (or stay equal) as sequences get longer.
 */
TEST_F(KVCacheManagerNewTest, MaxCapacityBatchSizeDecreasesWithLongerSequences)
{
    auto mgr = makeManager(/*numBlocks=*/16, /*tokensPerBlock=*/4, /*maxSeqs=*/8);

    SizeType32 capShort = mgr->getMaxCapacityBatchSize(/*inputLength=*/4,  /*outputLength=*/4);
    SizeType32 capLong  = mgr->getMaxCapacityBatchSize(/*inputLength=*/16, /*outputLength=*/16);
    EXPECT_GE(capShort, capLong)
        << "Shorter sequences should allow at least as many concurrent seqs as longer ones";
}

// ============================================================================
// Section 9 – Sliding-window attention
// ============================================================================

/**
 * With SWA, once the sequence substantially exceeds the attention window, the
 * number of actively held blocks must stabilize (old blocks are detached).
 */
TEST_F(KVCacheManagerNewTest, SlidingWindowStabilizesBlockCount)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 attentionWindow = 8; // 2 blocks
    constexpr SizeType32 numBlocks = 16;
    auto mgr = makeWindowedManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4, attentionWindow);

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7}, /*maxNewTokens=*/32);
    addSeq(*mgr, req);

    for (int i = 0; i < 16; ++i)
    {
        mgr->addToken(0);
        req->addNewToken(static_cast<TokenIdType>(100 + i), 0);
        checkPoolInvariant(*mgr);
    }

    // kSWAExtraBlock == 1, so max held ≈ windowBlocks + 1
    SizeType32 windowBlocks = tc::ceilDiv(attentionWindow, tokensPerBlock) + 1;
    EXPECT_LE(mgr->getUsedNumBlocks(), windowBlocks + 1)
        << "SWA must not allow unbounded block growth beyond the window";

    removeSeq(*mgr, req);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
}

// ============================================================================
// Section 10 – Multiple reuse cycles and diverging paths
// ============================================================================

/**
 * Across multiple add/remove cycles with the same prefix, the aggregate reused
 * block count must grow monotonically.
 */
TEST_F(KVCacheManagerNewTest, RepeatedReuseAccumulatesInStats)
{
    auto mgr = makeManager(/*numBlocks=*/16, /*tokensPerBlock=*/4, /*maxSeqs=*/4);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    // Warm the cache
    auto req0 = makeRequest(0, tokens);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    SizeType32 prevReused = mgr->getNumReusedBlocks();

    for (int cycle = 1; cycle <= 3; ++cycle)
    {
        auto req = makeRequest(static_cast<LlmRequest::RequestIdType>(cycle), tokens);
        addSeq(*mgr, req);
        EXPECT_GE(mgr->getNumReusedBlocks(), prevReused)
            << "Aggregate reused block count must not decrease";
        prevReused = mgr->getNumReusedBlocks();
        removeSeq(*mgr, req);
    }
}

/**
 * Diverging requests (shared stem, unique suffixes) must each reuse the stem but
 * not each other's unique suffix blocks.
 */
TEST_F(KVCacheManagerNewTest, DivergingContinuationsReuseStemOnly)
{
    constexpr SizeType32 tokensPerBlock = 4;
    auto mgr = makeManager(/*numBlocks=*/16, tokensPerBlock, /*maxSeqs=*/4);

    std::vector<TokenIdType> stem = {0, 1, 2, 3, 4, 5, 6, 7}; // 2 full blocks
    auto req0 = makeRequest(0, stem);
    addSeq(*mgr, req0);
    removeSeq(*mgr, req0);

    auto pathA = stem;
    pathA.insert(pathA.end(), {10, 11, 12, 13});
    auto reqA = makeRequest(1, pathA);
    SizeType32 ctxA = addSeq(*mgr, reqA);
    removeSeq(*mgr, reqA);

    auto pathB = stem;
    pathB.insert(pathB.end(), {20, 21, 22, 23});
    auto reqB = makeRequest(2, pathB);
    SizeType32 ctxB = addSeq(*mgr, reqB);
    removeSeq(*mgr, reqB);

    EXPECT_GE(ctxA, static_cast<SizeType32>(stem.size() - 1))
        << "Path A should reuse the full cached stem";
    EXPECT_GE(ctxB, static_cast<SizeType32>(stem.size() - 1))
        << "Path B should reuse the full cached stem";
}

// ============================================================================
// Section 11 – Rewind
// ============================================================================

/**
 * rewindKVCache must not throw and must preserve the pool invariant. The
 * number of used blocks after rewind must be <= the count before rewind.
 */
TEST_F(KVCacheManagerNewTest, RewindPreservesPoolInvariant)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 8;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/4);

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7}, /*maxNewTokens=*/8);
    addSeq(*mgr, req);

    for (int i = 0; i < 4; ++i)
    {
        mgr->addToken(0);
        req->addNewToken(static_cast<TokenIdType>(10 + i), 0);
    }
    checkPoolInvariant(*mgr);
    SizeType32 blocksBeforeRewind = mgr->getUsedNumBlocks();

    EXPECT_NO_THROW(mgr->rewindKVCache(0, /*rewindLengths=*/4));
    checkPoolInvariant(*mgr);

    EXPECT_LE(mgr->getUsedNumBlocks(), blocksBeforeRewind)
        << "Rewind must not increase block usage";

    removeSeq(*mgr, req);
    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks);
}

// ============================================================================
// Section 12 – Beam width
// ============================================================================

/**
 * With beam width > 1, shared context blocks reduce total allocation to less
 * than beamWidth * numContextBlocks.
 */
TEST_F(KVCacheManagerNewTest, BeamWidthSharesContextBlocks)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 32;
    constexpr SizeType32 beamWidth = 4;
    constexpr SizeType32 maxSequenceLength = numBlocks * tokensPerBlock;

    auto const maxAttentionWindow = maxSequenceLength;
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {numBlocks, 0}}};
    auto const stream = std::make_shared<tr::CudaStream>();

    KVCacheManager mgr(/*numLayers=*/4, /*numKvHeads=*/2, /*sizePerHead=*/64, tokensPerBlock, blocksPerWindow,
        /*maxNumSequences=*/4, beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
        nvinfer1::DataType::kHALF, /*sinkTokenLength=*/0, stream, maxSequenceLength, /*enableBlockReuse=*/true);
    mgr.allocatePools(false);

    std::vector<TokenIdType> tokens = {0, 1, 2, 3, 4, 5, 6, 7}; // 2 context blocks
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto tokenVec = std::make_shared<VecTokens>(tokens);
    auto req = std::make_shared<LlmRequest>(0, /*maxNewTokens=*/0, tokenVec, samplingConfig, false);
    mgr.addSequence(0, static_cast<SizeType32>(tokens.size()), beamWidth, req);

    SizeType32 numContextBlocks = tc::ceilDiv(static_cast<SizeType32>(tokens.size()), tokensPerBlock);
    EXPECT_LT(mgr.getUsedNumBlocks(), beamWidth * numContextBlocks)
        << "Shared context blocks should reduce total allocation for multi-beam sequences";

    (void) mgr.removeSequence(0, req);
    checkPoolInvariant(mgr);
}

// ============================================================================
// Section 13 – allocatedBytes
// ============================================================================

/**
 * allocatedBytes must be > 0 after pool allocation and must remain constant
 * across add/remove cycles (it reflects reserved pool memory, not live blocks).
 */
TEST_F(KVCacheManagerNewTest, AllocatedBytesNonzeroAndStable)
{
    auto mgr = makeManager(/*numBlocks=*/8, /*tokensPerBlock=*/4, /*maxSeqs=*/4);

    std::size_t bytes0 = mgr->getKvCacheStats().allocatedBytes;
    EXPECT_GT(bytes0, 0u) << "allocatedBytes must be non-zero after pool allocation";

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7});
    addSeq(*mgr, req);
    EXPECT_EQ(mgr->getKvCacheStats().allocatedBytes, bytes0)
        << "allocatedBytes must not change when a sequence is added";

    removeSeq(*mgr, req);
    EXPECT_EQ(mgr->getKvCacheStats().allocatedBytes, bytes0)
        << "allocatedBytes must not change when a sequence is removed";
}

// ============================================================================
// Section 14 – Rapid-cycle stress
// ============================================================================

/**
 * Many rapid add/remove cycles with varying prefixes must preserve the pool
 * invariant and leave the pool fully free at the end.
 */
TEST_F(KVCacheManagerNewTest, RapidAddRemoveCyclesPreserveInvariant)
{
    constexpr SizeType32 tokensPerBlock = 4;
    constexpr SizeType32 numBlocks = 16;
    auto mgr = makeManager(numBlocks, tokensPerBlock, /*maxSeqs=*/8);

    for (int round = 0; round < 20; ++round)
    {
        // Cycle through three base tokens to exercise both reuse and fresh-alloc paths
        TokenIdType baseToken = static_cast<TokenIdType>((round % 3) * 100);
        std::vector<TokenIdType> tokens(8);
        std::iota(tokens.begin(), tokens.end(), baseToken);

        auto req = makeRequest(static_cast<LlmRequest::RequestIdType>(round + 1), tokens);
        EXPECT_NO_THROW(addSeq(*mgr, req));
        checkPoolInvariant(*mgr);
        removeSeq(*mgr, req);
        checkPoolInvariant(*mgr);
    }

    EXPECT_EQ(mgr->getNumFreeBlocks(), numBlocks)
        << "All blocks must be free after all rapid-cycle sequences are removed";
}

// ============================================================================
// Section 15 – numFreeBlocksPerWindowSize consistency
// ============================================================================

/**
 * For a single-window manager the per-window free-block entry must match the
 * aggregate free-block count.
 */
TEST_F(KVCacheManagerNewTest, FreeBlocksPerWindowSizeConsistentWithTotal)
{
    auto mgr = makeManager(/*numBlocks=*/8, /*tokensPerBlock=*/4, /*maxSeqs=*/4);

    KvCacheStats stats = mgr->getKvCacheStats();
    ASSERT_EQ(stats.numFreeBlocksPerWindowSize.size(), 1u)
        << "Single-window manager should report exactly one window entry";

    SizeType32 perWindowFree = stats.numFreeBlocksPerWindowSize.begin()->second;
    EXPECT_EQ(perWindowFree, stats.freeNumBlocks)
        << "Per-window free count must equal total free count for a single window";

    auto req = makeRequest(0, {0, 1, 2, 3, 4, 5, 6, 7});
    addSeq(*mgr, req);

    KvCacheStats stats2 = mgr->getKvCacheStats();
    SizeType32 perWindowFree2 = stats2.numFreeBlocksPerWindowSize.begin()->second;
    EXPECT_EQ(perWindowFree2, stats2.freeNumBlocks)
        << "Per-window free count must remain consistent after addSequence";

    removeSeq(*mgr, req);
}
