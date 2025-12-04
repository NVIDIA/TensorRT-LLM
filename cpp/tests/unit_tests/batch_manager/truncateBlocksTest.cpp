/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
using VecTokens = LlmRequest::VecTokens;

class TruncateBlocksTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto const deviceCount = tc::getDeviceCount();
        if (deviceCount == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}

    std::shared_ptr<LlmRequest> createLlmRequest(
        LlmRequest::RequestIdType requestId, std::shared_ptr<VecTokens> inputTokens)
    {
        SizeType32 constexpr maxNewTokens{0};
        tr::SamplingConfig const samplingConfig{1};
        bool constexpr isStreaming{false};
        return std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    }
};

/**
 * Test truncateBlocks with multi-turn conversation simulation.
 *
 * This test simulates two conversation traces with partial blocks (tokens don't align with block boundaries):
 *   Trace_1: System Prompt (SP) + User Input 1 (UI1) + User Output 1 (UO1)
 *   Trace_2: System Prompt (SP) + User Input 2 (UI2) + User Output 2 (UO2) + User Input 3 (UI3) + User Output 3 (UO3)
 *
 * Token layout with tokensPerBlock=4:
 *   System Prompt: 10 tokens -> 2 full blocks + 2 tokens in partial block
 *   The partial block after SP will contain different tokens in Trace_1 vs Trace_2
 *
 * After truncating the tokens in Trace_2 (keeping only SP), Trace_1 should remain intact
 * because it has different tokens after the shared System Prompt.
 */
TEST_F(TruncateBlocksTest, MultiTurnConversationTruncation)
{
    // KV Cache configuration
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 16;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 32;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    // Create KVCacheManager with block reuse enabled
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Define token sequences for multi-turn conversations with PARTIAL BLOCKS
    // System Prompt: 10 tokens -> blocks [0-3], [4-7], [8-9, ?, ?] (2 full + 1 partial)
    // User Input 1: 5 tokens (continues partial block, creates new partial)
    // User Output 1: 3 tokens (continues partial block)
    // User Input 2: 6 tokens (continues partial block, creates new partial)
    // User Output 2: 3 tokens (continues partial block)
    // User Input 3: 5 tokens (continues partial block, creates new partial)
    // User Output 3: 3 tokens (continues partial block)

    VecTokens systemPrompt = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 10 tokens
    VecTokens userInput1 = {100, 101, 102, 103, 104};        // 5 tokens
    VecTokens userOutput1 = {200, 201, 202};                 // 3 tokens
    VecTokens userInput2 = {300, 301, 302, 303, 304, 305};   // 6 tokens
    VecTokens userOutput2 = {400, 401, 402};                 // 3 tokens
    VecTokens userInput3 = {500, 501, 502, 503, 504};        // 5 tokens
    VecTokens userOutput3 = {600, 601, 602};                 // 3 tokens

    // Trace_1: System Prompt + User Input 1 + User Output 1
    // Total: 10 + 5 + 3 = 18 tokens -> 5 blocks (last block has 2 tokens)
    VecTokens trace1Tokens;
    trace1Tokens.insert(trace1Tokens.end(), systemPrompt.begin(), systemPrompt.end());
    trace1Tokens.insert(trace1Tokens.end(), userInput1.begin(), userInput1.end());
    trace1Tokens.insert(trace1Tokens.end(), userOutput1.begin(), userOutput1.end());

    // Trace_2: System Prompt + User Input 2 + User Output 2 + User Input 3 + User Output 3
    // Total: 10 + 6 + 3 + 5 + 3 = 27 tokens -> 7 blocks (last block has 3 tokens)
    VecTokens trace2Tokens;
    trace2Tokens.insert(trace2Tokens.end(), systemPrompt.begin(), systemPrompt.end());
    trace2Tokens.insert(trace2Tokens.end(), userInput2.begin(), userInput2.end());
    trace2Tokens.insert(trace2Tokens.end(), userOutput2.begin(), userOutput2.end());
    trace2Tokens.insert(trace2Tokens.end(), userInput3.begin(), userInput3.end());
    trace2Tokens.insert(trace2Tokens.end(), userOutput3.begin(), userOutput3.end());

    auto trace1TokensPtr = std::make_shared<VecTokens>(trace1Tokens);
    auto trace2TokensPtr = std::make_shared<VecTokens>(trace2Tokens);

    auto const trace1Length = static_cast<SizeType32>(trace1Tokens.size());
    auto const trace2Length = static_cast<SizeType32>(trace2Tokens.size());

    // Verify our token counts create partial blocks
    EXPECT_EQ(trace1Length, 18); // 18 tokens = 4 full blocks + 2 tokens in partial
    EXPECT_EQ(trace2Length, 27); // 27 tokens = 6 full blocks + 3 tokens in partial

    // Step 1: Add Trace_1 to the cache
    LlmRequest::RequestIdType requestId1{0};
    auto llmRequest1 = createLlmRequest(requestId1, trace1TokensPtr);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId1, trace1Length, beamWidth, llmRequest1));

    // Trace_1 should have no reused blocks (first request)
    EXPECT_EQ(llmRequest1->getReusedBlocksPerRequest(), 0);
    auto numBlocksTrace1 = tc::ceilDiv(trace1Length, tokensPerBlock);
    EXPECT_EQ(numBlocksTrace1, 5); // 18 tokens / 4 = 5 blocks (ceil)
    EXPECT_EQ(llmRequest1->getAllocTotalBlocksPerRequest(), numBlocksTrace1);

    // Remove Trace_1 (blocks go to cache for reuse)
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId1, llmRequest1));

    // Step 2: Add Trace_2 to the cache
    LlmRequest::RequestIdType requestId2{1};
    auto llmRequest2 = createLlmRequest(requestId2, trace2TokensPtr);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId2, trace2Length, beamWidth, llmRequest2));

    // Trace_2 reuse system prompt blocks as well as a partial match block
    auto numMatchingBlocks = (systemPrompt.size() + tokensPerBlock - 1) / tokensPerBlock;
    EXPECT_EQ(llmRequest2->getReusedBlocksPerRequest(), numMatchingBlocks);

    // Remove Trace_2 (blocks go to cache for reuse)
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId2, llmRequest2));

    // Step 3: Truncate Trace_2 tokens, keeping only System Prompt
    // This should mark blocks after System Prompt in Trace_2 as low priority
    auto numTokensToKeep = static_cast<SizeType32>(systemPrompt.size());
    kvCacheManager.truncateBlocks(trace2Tokens, numTokensToKeep);

    // Step 4: Verify that Trace_1 is still intact (can be reused)
    LlmRequest::RequestIdType requestId3{2};
    auto llmRequest3 = createLlmRequest(requestId3, trace1TokensPtr);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId3, trace1Length, beamWidth, llmRequest3));

    // Trace_1 should reuse ALL its blocks (including the last partial block)
    // Because Trace_1 and Trace_2 share System Prompt, but have different subsequent tokens
    // Trace_1's blocks after System Prompt should still be intact
    // 18 tokens = 4 full blocks (tokens 0-15) + 1 partial block (tokens 16-17)
    auto numAllBlocksTrace1 = tc::ceilDiv(trace1Length, tokensPerBlock);
    EXPECT_EQ(numAllBlocksTrace1, 5); // 18 / 4 = 5 blocks (including partial)
    EXPECT_EQ(llmRequest3->getReusedBlocksPerRequest(), numAllBlocksTrace1);
    // Context position is the actual number of reused tokens (all 18)

    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId3, llmRequest3));
}

/**
 * Test truncateBlocks with shared prefix truncation.
 *
 * This test verifies that when truncating a trace, only the specific branch is affected,
 * not all branches sharing the same prefix.
 */
TEST_F(TruncateBlocksTest, SharedPrefixTruncation)
{
    // KV Cache configuration
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 16;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 32;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Shared prefix: tokens 0-10 (11 tokens)
    // Block layout: [0,1,2,3], [4,5,6,7], [8,9,10,?] where ? differs between branches
    // All 3 blocks (including the partial block) can be reused based on prefix matching
    VecTokens sharedPrefix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto const numSharedBlocks = (sharedPrefix.size() + tokensPerBlock - 1) / tokensPerBlock;

    // Branch A: prefix + tokens 11-18 (19 total tokens = 5 blocks, last is partial)
    VecTokens branchA = sharedPrefix;
    branchA.insert(branchA.end(), {11, 12, 13, 14, 15, 16, 17, 18});

    // Branch B: prefix + tokens 21-28 (19 total tokens = 5 blocks, last is partial)
    VecTokens branchB = sharedPrefix;
    branchB.insert(branchB.end(), {21, 22, 23, 24, 25, 26, 27, 28});

    auto branchAPtr = std::make_shared<VecTokens>(branchA);
    auto branchBPtr = std::make_shared<VecTokens>(branchB);

    auto const branchALength = static_cast<SizeType32>(branchA.size());
    auto const branchBLength = static_cast<SizeType32>(branchB.size());

    // Add Branch A
    LlmRequest::RequestIdType requestIdA{0};
    auto llmRequestA = createLlmRequest(requestIdA, branchAPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestIdA, branchALength, beamWidth, llmRequestA));
    EXPECT_EQ(llmRequestA->getReusedBlocksPerRequest(), 0); // First request
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestIdA, llmRequestA));

    // Add Branch B
    LlmRequest::RequestIdType requestIdB{1};
    auto llmRequestB = createLlmRequest(requestIdB, branchBPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestIdB, branchBLength, beamWidth, llmRequestB));
    // Branch B should reuse all blocks from the shared prefix (3 blocks)
    // This includes the partial block [8,9,10,?] based on prefix matching
    EXPECT_EQ(llmRequestB->getReusedBlocksPerRequest(), numSharedBlocks); // 3 blocks
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestIdB, llmRequestB));

    // Truncate Branch B, keeping only the shared prefix
    auto numTokensToKeep = static_cast<SizeType32>(sharedPrefix.size());
    kvCacheManager.truncateBlocks(branchB, numTokensToKeep);

    // Verify Branch A is still fully reusable
    LlmRequest::RequestIdType requestIdA2{2};
    auto llmRequestA2 = createLlmRequest(requestIdA2, branchAPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestIdA2, branchALength, beamWidth, llmRequestA2));
    // Branch A should reuse ALL its blocks (5 blocks including the partial one)
    auto numBlocksBranchA = tc::ceilDiv(branchALength, tokensPerBlock);
    EXPECT_EQ(llmRequestA2->getReusedBlocksPerRequest(), numBlocksBranchA);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestIdA2, llmRequestA2));

    // Verify Branch B with truncated suffix cannot reuse the truncated blocks
    LlmRequest::RequestIdType requestIdB2{3};
    auto llmRequestB2 = createLlmRequest(requestIdB2, branchBPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestIdB2, branchBLength, beamWidth, llmRequestB2));
    // After truncation, Branch B should only reuse the shared prefix blocks (3 blocks)
    // The blocks after the prefix were marked with low priority and may be evicted
    EXPECT_LE(llmRequestB2->getReusedBlocksPerRequest(), numSharedBlocks); // <= 3 blocks
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestIdB2, llmRequestB2));
}

/**
 * Test truncateBlocks with complete sequence truncation.
 *
 * This test verifies that truncating an entire sequence (numTokensToKeep = 0)
 * properly handles the edge case.
 */
TEST_F(TruncateBlocksTest, CompleteTruncation)
{
    // KV Cache configuration
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 8;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Simple sequence: tokens 0-17 (5 blocks)
    VecTokens sequence = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    auto sequencePtr = std::make_shared<VecTokens>(sequence);
    auto const sequenceLength = static_cast<SizeType32>(sequence.size());

    // Add the sequence
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest = createLlmRequest(requestId, sequencePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, sequenceLength, beamWidth, llmRequest));
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId, llmRequest));

    SizeType32 numTokensToKeep = 0;
    kvCacheManager.truncateBlocks(sequence, numTokensToKeep);

    // Add the same sequence again
    LlmRequest::RequestIdType requestId2{1};
    auto llmRequest2 = createLlmRequest(requestId2, sequencePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId2, sequenceLength, beamWidth, llmRequest2));

    // After truncation, only the first block should be reusable
    // This is intentional in the implementation of truncateBlocks, in order to save the partial block
    // at the border of the partial system prompt block.
    EXPECT_LE(llmRequest2->getReusedBlocksPerRequest(), 1);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId2, llmRequest2));
}

/**
 * Test truncateBlocks with non-existent tokens.
 *
 * This test verifies that truncating tokens that don't exist in the cache
 * doesn't cause any issues.
 */
TEST_F(TruncateBlocksTest, NonExistentTokensTruncation)
{
    // KV Cache configuration
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 8;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Actual sequence in cache
    VecTokens existingSequence = {0, 1, 2, 3, 4, 5, 6, 7};
    auto existingSequencePtr = std::make_shared<VecTokens>(existingSequence);
    auto const existingLength = static_cast<SizeType32>(existingSequence.size());

    // Add the sequence
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest = createLlmRequest(requestId, existingSequencePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, existingLength, beamWidth, llmRequest));
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId, llmRequest));

    // Try to truncate a non-existent sequence (different tokens)
    VecTokens nonExistentSequence = {100, 101, 102, 103, 104, 105, 106, 107};
    EXPECT_NO_THROW(kvCacheManager.truncateBlocks(nonExistentSequence, 4));

    // Verify the original sequence is still reusable
    LlmRequest::RequestIdType requestId2{1};
    auto llmRequest2 = createLlmRequest(requestId2, existingSequencePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId2, existingLength, beamWidth, llmRequest2));
    EXPECT_EQ(llmRequest2->getReusedBlocksPerRequest(), 2); // 8 tokens / 4 tokens per block = 2 blocks
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId2, llmRequest2));
}

/**
 * Test truncateBlocks with complex multi-turn conversation simulation.
 *
 * This test simulates two parallel conversation traces that share the same system_prompt + user_input_1,
 * but diverge into different CoT (chain-of-thought) reasoning and outputs. Each trace then continues
 * with a second turn of user input and response.
 *
 * Events timeline:
 *   1. trace_1 prefill done:  system_prompt + user_input_1
 *   2. trace_1 decode done:   system_prompt + user_input_1 + cot_1 + assistant_output_1
 *   3. trace_2 prefill done:  system_prompt + user_input_1
 *   4. trace_2 decode done:   system_prompt + user_input_1 + cot_2 + assistant_output_2
 *   5. trace_1 prefill done:  system_prompt + user_input_1 + assistant_output_1 + user_input_3
 *   6. trace_1 decode done:   system_prompt + user_input_1 + assistant_output_1 + user_input_3 + cot_3 +
 * assistant_output_3
 *   7. trace_2 prefill done:  system_prompt + user_input_1 + assistant_output_2 + user_input_4
 *   8. trace_2 decode done:   system_prompt + user_input_1 + assistant_output_2 + user_input_4 + cot_4 +
 * assistant_output_4
 *
 * After all events, truncate trace_1 retaining only system_prompt + user_input_1.
 * Verify:
 *   - Tokens are released for trace_1's unique blocks
 *   - trace_2 remains completely intact and reusable
 */
TEST_F(TruncateBlocksTest, ComplexMultiTurnConversationTruncation)
{
    // KV Cache configuration
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 32;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 16;
    auto constexpr blocksInPrimaryPool = 64;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    // Create KVCacheManager with block reuse enabled
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Define token sequences for multi-turn conversations
    // Using distinct token ranges to make debugging easier
    VecTokens systemPrompt = {1, 2, 3, 4, 5, 6, 7, 8}; // 8 tokens (2 blocks)
    VecTokens userInput1 = {10, 11, 12, 13, 14, 15};   // 6 tokens

    // trace_1 turn 1 output (CoT hidden from user, shown here for block computation)
    VecTokens cot1 = {100, 101, 102, 103, 104};        // 5 tokens
    VecTokens assistantOutput1 = {110, 111, 112, 113}; // 4 tokens

    // trace_2 turn 1 output (different CoT path)
    VecTokens cot2 = {200, 201, 202, 203, 204, 205}; // 6 tokens
    VecTokens assistantOutput2 = {210, 211, 212};    // 3 tokens

    // trace_1 turn 2
    VecTokens userInput3 = {300, 301, 302, 303};       // 4 tokens
    VecTokens cot3 = {310, 311, 312, 313, 314};        // 5 tokens
    VecTokens assistantOutput3 = {320, 321, 322, 323}; // 4 tokens

    // trace_2 turn 2
    VecTokens userInput4 = {400, 401, 402, 403, 404};       // 5 tokens
    VecTokens cot4 = {410, 411, 412, 413};                  // 4 tokens
    VecTokens assistantOutput4 = {420, 421, 422, 423, 424}; // 5 tokens

    // Shared prefix: system_prompt + user_input_1 (14 tokens = 3 full blocks + 2 tokens in partial)
    VecTokens sharedPrefix;
    sharedPrefix.insert(sharedPrefix.end(), systemPrompt.begin(), systemPrompt.end());
    sharedPrefix.insert(sharedPrefix.end(), userInput1.begin(), userInput1.end());
    auto const sharedPrefixLength = static_cast<SizeType32>(sharedPrefix.size());
    EXPECT_EQ(sharedPrefixLength, 14);

    // --- Event 1: trace_1 prefill done (system_prompt + user_input_1) ---
    VecTokens trace1Turn1Prefill = sharedPrefix;
    auto trace1Turn1PrefillPtr = std::make_shared<VecTokens>(trace1Turn1Prefill);
    auto const trace1Turn1PrefillLength = static_cast<SizeType32>(trace1Turn1Prefill.size());

    LlmRequest::RequestIdType requestId1{0};
    auto llmRequest1 = createLlmRequest(requestId1, trace1Turn1PrefillPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId1, trace1Turn1PrefillLength, beamWidth, llmRequest1));
    EXPECT_EQ(llmRequest1->getReusedBlocksPerRequest(), 0); // First request, no reuse
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId1, llmRequest1));

    // --- Event 2: trace_1 decode done (system_prompt + user_input_1 + cot_1 + assistant_output_1) ---
    VecTokens trace1Turn1Decode = sharedPrefix;
    trace1Turn1Decode.insert(trace1Turn1Decode.end(), cot1.begin(), cot1.end());
    trace1Turn1Decode.insert(trace1Turn1Decode.end(), assistantOutput1.begin(), assistantOutput1.end());
    auto trace1Turn1DecodePtr = std::make_shared<VecTokens>(trace1Turn1Decode);
    auto const trace1Turn1DecodeLength = static_cast<SizeType32>(trace1Turn1Decode.size());
    EXPECT_EQ(trace1Turn1DecodeLength, 23); // 14 + 5 + 4 = 23 tokens

    LlmRequest::RequestIdType requestId2{1};
    auto llmRequest2 = createLlmRequest(requestId2, trace1Turn1DecodePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId2, trace1Turn1DecodeLength, beamWidth, llmRequest2));
    auto trace1Turn1DecodeReusedBlocks = llmRequest2->getReusedBlocksPerRequest();
    // Should reuse shared prefix blocks (at least 3 blocks from 14 tokens)
    EXPECT_EQ(trace1Turn1DecodeReusedBlocks, 4);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId2, llmRequest2));

    // --- Event 3: trace_2 prefill done (system_prompt + user_input_1) ---
    VecTokens trace2Turn1Prefill = sharedPrefix;
    auto trace2Turn1PrefillPtr = std::make_shared<VecTokens>(trace2Turn1Prefill);
    auto const trace2Turn1PrefillLength = static_cast<SizeType32>(trace2Turn1Prefill.size());

    LlmRequest::RequestIdType requestId3{2};
    auto llmRequest3 = createLlmRequest(requestId3, trace2Turn1PrefillPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId3, trace2Turn1PrefillLength, beamWidth, llmRequest3));
    // Should reuse all blocks from shared prefix
    auto sharedPrefixBlocks = tc::ceilDiv(sharedPrefixLength, tokensPerBlock);
    EXPECT_EQ(llmRequest3->getReusedBlocksPerRequest(), sharedPrefixBlocks);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId3, llmRequest3));

    // --- Event 4: trace_2 decode done (system_prompt + user_input_1 + cot_2 + assistant_output_2) ---
    VecTokens trace2Turn1Decode = sharedPrefix;
    trace2Turn1Decode.insert(trace2Turn1Decode.end(), cot2.begin(), cot2.end());
    trace2Turn1Decode.insert(trace2Turn1Decode.end(), assistantOutput2.begin(), assistantOutput2.end());
    auto trace2Turn1DecodePtr = std::make_shared<VecTokens>(trace2Turn1Decode);
    auto const trace2Turn1DecodeLength = static_cast<SizeType32>(trace2Turn1Decode.size());
    EXPECT_EQ(trace2Turn1DecodeLength, 23); // 14 + 6 + 3 = 23 tokens

    LlmRequest::RequestIdType requestId4{3};
    auto llmRequest4 = createLlmRequest(requestId4, trace2Turn1DecodePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId4, trace2Turn1DecodeLength, beamWidth, llmRequest4));
    // trace_2 diverges from trace_1 after shared prefix (different cot tokens)
    // Should reuse shared prefix blocks but not cot blocks
    EXPECT_EQ(llmRequest4->getReusedBlocksPerRequest(), 4);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId4, llmRequest4));

    // --- Event 5: trace_1 prefill done (system_prompt + user_input_1 + assistant_output_1 + user_input_3) ---
    // Note: In multi-turn, CoT is typically stripped, keeping only assistant_output
    VecTokens trace1Turn2Prefill = sharedPrefix;
    trace1Turn2Prefill.insert(trace1Turn2Prefill.end(), assistantOutput1.begin(), assistantOutput1.end());
    trace1Turn2Prefill.insert(trace1Turn2Prefill.end(), userInput3.begin(), userInput3.end());
    auto trace1Turn2PrefillPtr = std::make_shared<VecTokens>(trace1Turn2Prefill);
    auto const trace1Turn2PrefillLength = static_cast<SizeType32>(trace1Turn2Prefill.size());
    EXPECT_EQ(trace1Turn2PrefillLength, 22); // 14 + 4 + 4 = 22 tokens

    LlmRequest::RequestIdType requestId5{4};
    auto llmRequest5 = createLlmRequest(requestId5, trace1Turn2PrefillPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId5, trace1Turn2PrefillLength, beamWidth, llmRequest5));
    // Should reuse shared prefix blocks
    EXPECT_EQ(llmRequest5->getReusedBlocksPerRequest(), 4);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId5, llmRequest5));

    // --- Event 6: trace_1 decode done (+ cot_3 + assistant_output_3) ---
    VecTokens trace1Turn2Decode = trace1Turn2Prefill;
    trace1Turn2Decode.insert(trace1Turn2Decode.end(), cot3.begin(), cot3.end());
    trace1Turn2Decode.insert(trace1Turn2Decode.end(), assistantOutput3.begin(), assistantOutput3.end());
    auto trace1Turn2DecodePtr = std::make_shared<VecTokens>(trace1Turn2Decode);
    auto const trace1Turn2DecodeLength = static_cast<SizeType32>(trace1Turn2Decode.size());
    EXPECT_EQ(trace1Turn2DecodeLength, 31); // 22 + 5 + 4 = 31 tokens

    LlmRequest::RequestIdType requestId6{5};
    auto llmRequest6 = createLlmRequest(requestId6, trace1Turn2DecodePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId6, trace1Turn2DecodeLength, beamWidth, llmRequest6));
    auto trace1Turn2DecodeReusedBlocks = llmRequest6->getReusedBlocksPerRequest();
    auto trace1Turn2DecodeAllocBlocks = llmRequest6->getAllocTotalBlocksPerRequest();
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId6, llmRequest6));

    // --- Event 7: trace_2 prefill done (system_prompt + user_input_1 + assistant_output_2 + user_input_4) ---
    VecTokens trace2Turn2Prefill = sharedPrefix;
    trace2Turn2Prefill.insert(trace2Turn2Prefill.end(), assistantOutput2.begin(), assistantOutput2.end());
    trace2Turn2Prefill.insert(trace2Turn2Prefill.end(), userInput4.begin(), userInput4.end());
    auto trace2Turn2PrefillPtr = std::make_shared<VecTokens>(trace2Turn2Prefill);
    auto const trace2Turn2PrefillLength = static_cast<SizeType32>(trace2Turn2Prefill.size());
    EXPECT_EQ(trace2Turn2PrefillLength, 22); // 14 + 3 + 5 = 22 tokens

    LlmRequest::RequestIdType requestId7{6};
    auto llmRequest7 = createLlmRequest(requestId7, trace2Turn2PrefillPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId7, trace2Turn2PrefillLength, beamWidth, llmRequest7));
    EXPECT_EQ(llmRequest7->getReusedBlocksPerRequest(), 4);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId7, llmRequest7));

    // --- Event 8: trace_2 decode done (+ cot_4 + assistant_output_4) ---
    VecTokens trace2Turn2Decode = trace2Turn2Prefill;
    trace2Turn2Decode.insert(trace2Turn2Decode.end(), cot4.begin(), cot4.end());
    trace2Turn2Decode.insert(trace2Turn2Decode.end(), assistantOutput4.begin(), assistantOutput4.end());
    auto trace2Turn2DecodePtr = std::make_shared<VecTokens>(trace2Turn2Decode);
    auto const trace2Turn2DecodeLength = static_cast<SizeType32>(trace2Turn2Decode.size());
    EXPECT_EQ(trace2Turn2DecodeLength, 31); // 22 + 4 + 5 = 31 tokens

    LlmRequest::RequestIdType requestId8{7};
    auto llmRequest8 = createLlmRequest(requestId8, trace2Turn2DecodePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId8, trace2Turn2DecodeLength, beamWidth, llmRequest8));
    auto trace2Turn2DecodeReusedBlocks = llmRequest8->getReusedBlocksPerRequest();
    auto trace2Turn2DecodeAllocBlocks = llmRequest8->getAllocTotalBlocksPerRequest();
    auto const trace2TotalBlocks = tc::ceilDiv(trace2Turn2DecodeLength, tokensPerBlock);
    EXPECT_EQ(trace2TotalBlocks, 8); // 31 tokens / 4 = 8 blocks
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId8, llmRequest8));

    // Record free blocks before truncation
    auto freeBlocksBefore = kvCacheManager.getNumFreeBlocks();

    // --- Truncate trace_1, retaining only system_prompt + user_input_1 ---
    // This should mark all trace_1 specific blocks (after shared prefix) as low priority
    kvCacheManager.truncateBlocks(trace1Turn2Decode, sharedPrefixLength);

    // Record free blocks after truncation
    auto freeBlocksAfter = kvCacheManager.getNumFreeBlocks();

    // Calculate expected released blocks from trace_1
    // trace_1 turn 2 decode: 31 tokens = 8 blocks
    // Keeping first 14 tokens = 4 blocks (ceil(14/4))
    // Released: blocks after shared prefix that are unique to trace_1
    auto trace1TotalBlocks = tc::ceilDiv(trace1Turn2DecodeLength, tokensPerBlock);
    auto sharedPrefixBlockCount = tc::ceilDiv(sharedPrefixLength, tokensPerBlock);
    // Note: Some blocks may be shared, so released blocks <= trace1TotalBlocks - sharedPrefixBlockCount
    EXPECT_EQ(freeBlocksAfter, freeBlocksBefore + trace1TotalBlocks - sharedPrefixBlockCount);

    // --- Verify trace_2 remains completely intact and reusable ---
    LlmRequest::RequestIdType requestId9{8};
    auto llmRequest9 = createLlmRequest(requestId9, trace2Turn2DecodePtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId9, trace2Turn2DecodeLength, beamWidth, llmRequest9));

    // trace_2's full sequence should still be fully reusable
    // All 8 blocks (31 tokens) should be reused
    EXPECT_EQ(llmRequest9->getReusedBlocksPerRequest(), trace2TotalBlocks);

    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId9, llmRequest9));

    // --- Additional verification: Shared prefix is still intact ---
    LlmRequest::RequestIdType requestId10{9};
    auto sharedPrefixPtr = std::make_shared<VecTokens>(sharedPrefix);
    auto llmRequest10 = createLlmRequest(requestId10, sharedPrefixPtr);
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId10, sharedPrefixLength, beamWidth, llmRequest10));
    EXPECT_EQ(llmRequest10->getReusedBlocksPerRequest(), sharedPrefixBlockCount);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId10, llmRequest10));
}
