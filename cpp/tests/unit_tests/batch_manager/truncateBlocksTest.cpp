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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/testing/kvCacheManagerTestUtil.h"

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
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    // Create KVCacheManager with block reuse enabled
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true /* enableBlockReuse */);
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

    EXPECT_NO_THROW(
        kvCacheManager.addSequenceBatch({{{requestId1, trace1Length, beamWidth}}}, {std::ref(*llmRequest1)}));

    // Trace_1 should have no reused blocks (first request)
    EXPECT_EQ(llmRequest1->getReusedBlocksPerRequest(), 0);
    auto numBlocksTrace1 = tc::ceilDiv(trace1Length, tokensPerBlock);
    EXPECT_EQ(numBlocksTrace1, 5); // 18 tokens / 4 = 5 blocks (ceil)
    EXPECT_EQ(llmRequest1->getAllocTotalBlocksPerRequest(), numBlocksTrace1);

    // Remove Trace_1 (blocks go to cache for reuse)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId1, llmRequest1));

    // Step 2: Add Trace_2 to the cache
    LlmRequest::RequestIdType requestId2{1};
    auto llmRequest2 = createLlmRequest(requestId2, trace2TokensPtr);

    EXPECT_NO_THROW(
        kvCacheManager.addSequenceBatch({{{requestId2, trace2Length, beamWidth}}}, {std::ref(*llmRequest2)}));

    // Trace_2 reuse system prompt blocks as well as a partial match block
    auto numMatchingBlocks = (systemPrompt.size() + tokensPerBlock - 1) / tokensPerBlock;
    EXPECT_EQ(llmRequest2->getReusedBlocksPerRequest(), numMatchingBlocks);

    // Remove Trace_2 (blocks go to cache for reuse)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId2, llmRequest2));

    // Step 3: Truncate Trace_2 tokens, keeping only System Prompt
    // This should mark blocks after System Prompt in Trace_2 as low priority
    auto numTokensToKeep = static_cast<SizeType32>(systemPrompt.size());
    kvCacheManager.truncateBlocks(trace2Tokens, numTokensToKeep);

    // Step 4: Verify that Trace_1 is still intact (can be reused)
    LlmRequest::RequestIdType requestId3{2};
    auto llmRequest3 = createLlmRequest(requestId3, trace1TokensPtr);

    EXPECT_NO_THROW(
        kvCacheManager.addSequenceBatch({{{requestId3, trace1Length, beamWidth}}}, {std::ref(*llmRequest3)}));

    // Trace_1 should reuse ALL its blocks (including the last partial block)
    // Because Trace_1 and Trace_2 share System Prompt, but have different subsequent tokens
    // Trace_1's blocks after System Prompt should still be intact
    // 18 tokens = 4 full blocks (tokens 0-15) + 1 partial block (tokens 16-17)
    auto numAllBlocksTrace1 = tc::ceilDiv(trace1Length, tokensPerBlock);
    EXPECT_EQ(numAllBlocksTrace1, 5); // 18 / 4 = 5 blocks (including partial)
    EXPECT_EQ(llmRequest3->getReusedBlocksPerRequest(), numAllBlocksTrace1);
    // Context position is the actual number of reused tokens (all 18)

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId3, llmRequest3));
}
