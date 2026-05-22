/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"

#include <random>
#include <tuple>

namespace texec = tensorrt_llm::executor;
namespace tbm = tensorrt_llm::batch_manager;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

class RnnTargetIRanksTest : public ::testing::Test
{
protected:
    // Helper to create CacheState with RNN config
    static texec::kv_cache::CacheState makeRnnCacheState(
        SizeType32 numLayers, SizeType32 tp, SizeType32 pp, std::vector<SizeType32> const& layersPerPP)
    {
        // Create a CacheState with dummy KV fields and proper RNN config
        std::vector<SizeType32> kvLayersPerPP(pp, 0); // No attention layers
        auto state = texec::kv_cache::CacheState(
            /*nbAttentionLayers=*/0, /*nbKvHeads=*/1, /*sizePerHead=*/64, /*tokensPerBlock=*/32, tp, pp,
            /*contextParallelism=*/1, kvLayersPerPP, nvinfer1::DataType::kFLOAT);

        texec::kv_cache::CacheState::RnnModelConfig rnnModelConfig{/*mDState=*/16, /*mDConv=*/4, /*mHiddenSize=*/256,
            /*mHeadDim=*/64,
            /*mConvDimSize=*/128, /*mNGroups=*/1, /*mNumLayers=*/numLayers, /*mNumHeads=*/4};
        state.setRnnConfig(rnnModelConfig, layersPerPP, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT);
        return state;
    }
};

// Test: Same PP, same TP - trivial 1:1 mapping
TEST_F(RnnTargetIRanksTest, SamePPSameTP)
{
    SizeType32 numLayers = 8;
    SizeType32 tp = 2;
    SizeType32 pp = 2;
    std::vector<SizeType32> layersPerPP = {4, 4};

    auto contextState = makeRnnCacheState(numLayers, tp, pp, layersPerPP);
    auto genState = makeRnnCacheState(numLayers, tp, pp, layersPerPP);

    // Rank 0: PP=0, TP=0 -> should communicate with gen rank 0
    auto result0 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mDomainPPSize, 1);
    EXPECT_EQ(result0.mDomainTPSize, 1);

    // Rank 1: PP=0, TP=1 -> should communicate with gen rank 1
    auto result1 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 1);
    EXPECT_EQ(result1.mIRanks, std::vector<int>({1}));

    // Rank 2: PP=1, TP=0 -> should communicate with gen rank 2
    auto result2 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 2);
    EXPECT_EQ(result2.mIRanks, std::vector<int>({2}));

    // Rank 3: PP=1, TP=1 -> should communicate with gen rank 3
    auto result3 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 3);
    EXPECT_EQ(result3.mIRanks, std::vector<int>({3}));
}

// Test: Context PP=1, Gen PP=2 (main disaggregated use case)
TEST_F(RnnTargetIRanksTest, ContextPP1GenPP2)
{
    SizeType32 numLayers = 8;
    SizeType32 tp = 2;

    // Context: PP=1 (all 8 layers on one PP rank)
    std::vector<SizeType32> contextLayersPerPP = {8};
    auto contextState = makeRnnCacheState(numLayers, tp, /*pp=*/1, contextLayersPerPP);

    // Gen: PP=2 (4 layers each)
    std::vector<SizeType32> genLayersPerPP = {4, 4};
    auto genState = makeRnnCacheState(numLayers, tp, /*pp=*/2, genLayersPerPP);

    // Context rank 0 (PP=0, TP=0, has layers 0-7) needs data from:
    //   Gen PP=0 (layers 0-3) at TP=0 -> rank 0
    //   Gen PP=1 (layers 4-7) at TP=0 -> rank 2
    auto result0 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 2}));
    EXPECT_EQ(result0.mDomainPPSize, 2);
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, (std::vector<int>{4, 4}));

    // Context rank 1 (PP=0, TP=1, has layers 0-7) needs:
    //   Gen PP=0 at TP=1 -> rank 1
    //   Gen PP=1 at TP=1 -> rank 3
    auto result1 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 1);
    EXPECT_EQ(result1.mIRanks, (std::vector<int>{1, 3}));
    EXPECT_EQ(result1.mDomainPPSize, 2);
}

// Test: Context PP=2, Gen PP=1 (reverse direction)
TEST_F(RnnTargetIRanksTest, ContextPP2GenPP1)
{
    SizeType32 numLayers = 8;
    SizeType32 tp = 2;

    // Context: PP=2
    std::vector<SizeType32> contextLayersPerPP = {4, 4};
    auto contextState = makeRnnCacheState(numLayers, tp, /*pp=*/2, contextLayersPerPP);

    // Gen: PP=1
    std::vector<SizeType32> genLayersPerPP = {8};
    auto genState = makeRnnCacheState(numLayers, tp, /*pp=*/1, genLayersPerPP);

    // Context rank 0 (PP=0, TP=0, has layers 0-3) needs:
    //   Gen PP=0 (layers 0-7) at TP=0 -> rank 0 (but only 4 layers overlap)
    auto result0 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mDomainPPSize, 1);
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, std::vector<int>({4}));

    // Context rank 2 (PP=1, TP=0, has layers 4-7) also needs gen rank 0
    auto result2 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 2);
    EXPECT_EQ(result2.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result2.mPeerLayerNumInDomainPP, std::vector<int>({4}));
}

// Test: Non-uniform layer distribution with CTX 8 GPUs, Gen 16 GPUs
TEST_F(RnnTargetIRanksTest, NonUniformLayers)
{
    SizeType32 numLayers = 16;
    SizeType32 tp = 4; // Same TP required for RNN

    // Context: TP=4, PP=2 -> 8 GPUs, non-uniform (10 + 6 layers)
    std::vector<SizeType32> contextLayersPerPP = {10, 6};
    auto contextState = makeRnnCacheState(numLayers, tp, /*pp=*/2, contextLayersPerPP);

    // Gen: TP=4, PP=4 -> 16 GPUs, uniform (4 layers each)
    std::vector<SizeType32> genLayersPerPP = {4, 4, 4, 4};
    auto genState = makeRnnCacheState(numLayers, tp, /*pp=*/4, genLayersPerPP);

    // Rank formula: rank = ppRank * tpNum + tpRank
    //
    // Context layout (8 GPUs):
    //   Rank 0-3: PP=0, TP=0-3, layers 0-9
    //   Rank 4-7: PP=1, TP=0-3, layers 10-15
    //
    // Gen layout (16 GPUs):
    //   Rank 0-3:   PP=0, TP=0-3, layers 0-3
    //   Rank 4-7:   PP=1, TP=0-3, layers 4-7
    //   Rank 8-11:  PP=2, TP=0-3, layers 8-11
    //   Rank 12-15: PP=3, TP=0-3, layers 12-15

    // Context rank 0 (PP=0, TP=0, has layers 0-9) needs:
    //   Gen PP=0 (layers 0-3)  at TP=0 -> rank 0  (4 layers overlap)
    //   Gen PP=1 (layers 4-7)  at TP=0 -> rank 4  (4 layers overlap)
    //   Gen PP=2 (layers 8-11) at TP=0 -> rank 8  (2 layers overlap: 8-9)
    auto result0 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(result0.mDomainPPSize, 3);
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, (std::vector<int>{4, 4, 2}));

    // Context rank 1 (PP=0, TP=1, has layers 0-9) needs:
    //   Gen PP=0 at TP=1 -> rank 1
    //   Gen PP=1 at TP=1 -> rank 5
    //   Gen PP=2 at TP=1 -> rank 9
    auto result1 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 1);
    EXPECT_EQ(result1.mIRanks, (std::vector<int>{1, 5, 9}));
    EXPECT_EQ(result1.mDomainPPSize, 3);

    // Context rank 4 (PP=1, TP=0, has layers 10-15) needs:
    //   Gen PP=2 (layers 8-11)  at TP=0 -> rank 8  (2 layers overlap: 10-11)
    //   Gen PP=3 (layers 12-15) at TP=0 -> rank 12 (4 layers overlap)
    auto result4 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 4);
    EXPECT_EQ(result4.mIRanks, (std::vector<int>{8, 12}));
    EXPECT_EQ(result4.mDomainPPSize, 2);
    EXPECT_EQ(result4.mPeerLayerNumInDomainPP, (std::vector<int>{2, 4}));

    // Context rank 7 (PP=1, TP=3, has layers 10-15) needs:
    //   Gen PP=2 at TP=3 -> rank 11
    //   Gen PP=3 at TP=3 -> rank 15
    auto result7 = tensorrt_llm::executor::kv_cache::targetIRanksForRnn(genState, contextState, 7);
    EXPECT_EQ(result7.mIRanks, (std::vector<int>{11, 15}));
    EXPECT_EQ(result7.mDomainPPSize, 2);
}

// Test: inquireSupport
TEST_F(RnnTargetIRanksTest, inquireSupport)
{
    SizeType32 numLayers = 8;

    auto state1 = makeRnnCacheState(numLayers, /*tp=*/2, /*pp=*/1, {8});
    auto state2 = makeRnnCacheState(numLayers, /*tp=*/2, /*pp=*/2, {4, 4});
    auto state3 = makeRnnCacheState(numLayers, /*tp=*/4, /*pp=*/1, {8}); // Different TP

    // Use reinterpret_cast to pass a non-null dummy pointer (formatter only stores it, doesn't use it in
    // inquireSupport)
    tbm::RnnCacheFormatter formatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // Same TP, different PP -> should be supported
    EXPECT_TRUE(formatter.inquireSupport(state1, state2));

    // Different TP is also supported for RNN cache transfer (only model config and data types are checked)
    EXPECT_TRUE(formatter.inquireSupport(state1, state3));
}

// Add to rnnCacheFormatterTest.cpp

// Test fixture for hybrid model counterparts merging
class HybridModelCounterpartsTest : public ::testing::Test
{
protected:
    // Helper to create KvCacheState
    static texec::kv_cache::CacheState makeKvCacheState(SizeType32 numLayers, SizeType32 tp, SizeType32 pp,
        std::vector<SizeType32> const& layersPerPP, SizeType32 numHeads = 8, SizeType32 sizePerHead = 64,
        SizeType32 tokensPerBlock = 32)
    {
        return texec::kv_cache::CacheState(numLayers, numHeads, sizePerHead, tokensPerBlock, tp, pp,
            /*contextParallelism=*/1, layersPerPP, nvinfer1::DataType::kFLOAT,
            texec::kv_cache::CacheState::AttentionType::kDEFAULT, /*kvFactor=*/2,
            /*enableAttentionDP=*/false, /*DPrank=*/0, /*DPsize=*/1);
    }

    // Helper to create a hybrid CacheState with both KV and RNN config
    static texec::kv_cache::CacheState makeHybridState(SizeType32 kvNumLayers, SizeType32 rnnNumLayers, SizeType32 tp,
        SizeType32 pp, std::vector<SizeType32> const& kvLayersPerPP, std::vector<SizeType32> const& rnnLayersPerPP,
        SizeType32 numHeads = 8, SizeType32 sizePerHead = 64, SizeType32 tokensPerBlock = 32)
    {
        auto state = makeKvCacheState(kvNumLayers, tp, pp, kvLayersPerPP, numHeads, sizePerHead, tokensPerBlock);
        texec::kv_cache::CacheState::RnnModelConfig rnnModelConfig{16, 4, 256, 64, 128, 1, rnnNumLayers, 4};
        state.setRnnConfig(rnnModelConfig, rnnLayersPerPP, nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT);
        return state;
    }

    // Helper to compute merged counterparts (union of KV and RNN)
    static std::vector<SizeType32> mergeCounterparts(
        std::vector<SizeType32> const& kvCounterParts, std::vector<SizeType32> const& rnnCounterParts)
    {
        std::vector<SizeType32> allCounterParts = kvCounterParts;
        for (auto rank : rnnCounterParts)
        {
            if (std::find(allCounterParts.begin(), allCounterParts.end(), rank) == allCounterParts.end())
            {
                allCounterParts.push_back(rank);
            }
        }
        return allCounterParts;
    }
};

// Test: Hybrid model with different PP distributions for KV and RNN
// Scenario: Context has PP=1, Gen has PP=2 for KV, but RNN layers distributed differently
TEST_F(HybridModelCounterpartsTest, DifferentPPDistributionKvRnn)
{
    // ============= Setup =============
    // Model: 16 total layers - 10 attention (KV), 6 RNN (Mamba)
    // Context executor: PP=1, TP=2 (2 GPUs)
    // Gen executor: PP=2, TP=2 (4 GPUs)
    //
    // KV layers: 10 attention layers
    //   Context: PP=1, layersPerPP = {10}
    //   Gen: PP=2, layersPerPP = {5, 5}  (layers 0-4 on PP0, layers 5-9 on PP1)
    //
    // RNN layers: 6 RNN layers
    //   Context: PP=1, layersPerPP = {6}
    //   Gen: PP=2, layersPerPP = {3, 3}  (layers 0-2 on PP0, layers 3-5 on PP1)

    SizeType32 const tp = 2;

    // Unified states with both KV and RNN config
    auto contextState = makeHybridState(/*kvNumLayers=*/10, /*rnnNumLayers=*/6, tp, /*pp=*/1, {10}, {6});
    auto genState = makeHybridState(/*kvNumLayers=*/10, /*rnnNumLayers=*/6, tp, /*pp=*/2, {5, 5}, {3, 3});

    // Use dummy formatter pointers (we only need them to call getCounterparts)
    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // ============= Test from Context rank 0 (PP=0, TP=0) =============
    // Context rank 0 has ALL layers for both KV and RNN (since PP=1)
    // KV counterparts from context rank 0:
    //   Needs layers 0-9 from Gen. Gen PP=0 has layers 0-4 (at TP=0 -> rank 0)
    //   Gen PP=1 has layers 5-9 (at TP=0 -> rank 2)
    //   So KV counterparts = {0, 2}
    //
    // RNN counterparts from context rank 0:
    //   Needs layers 0-5 from Gen. Gen PP=0 has layers 0-2 (at TP=0 -> rank 0)
    //   Gen PP=1 has layers 3-5 (at TP=0 -> rank 2)
    //   So RNN counterparts = {0, 2}

    SizeType32 contextRank0 = 0;

    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);

    // Both should need the same ranks (0, 2) in this symmetric case
    // Note: targetIRanks returns std::vector<int>, getCounterparts returns std::vector<SizeType32>
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2}));
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2}));

    auto mergedCounterParts = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(mergedCounterParts, (std::vector<SizeType32>{0, 2}));

    auto rnnTargetInfo = texec::kv_cache::targetIRanksForRnn(genState, contextState, contextRank0);
    auto [rnnPickUp, rnnLocalRankIndices] = tensorrt_llm::batch_manager::cache_formatter_utils::pickRecvConnections(
        mergedCounterParts.size(), contextState, contextRank0, genState, mergedCounterParts, rnnTargetInfo);
    EXPECT_EQ(rnnPickUp, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(rnnLocalRankIndices, (std::vector<size_t>{0, 1}));
}

// Test: Asymmetric hybrid PP distribution where KV and RNN have different counterparts
TEST_F(HybridModelCounterpartsTest, AsymmetricKvRnnDistribution)
{
    // ============= Setup =============
    // Model with different PP distributions for attention and RNN layers
    // Context executor: PP=1, TP=2 (2 GPUs)
    // Gen executor: PP=4, TP=2 (8 GPUs)
    //
    // KV layers: 8 attention layers
    //   Context: PP=1, layersPerPP = {8}
    //   Gen: PP=4, layersPerPP = {2, 2, 2, 2}
    //        PP0 (ranks 0,1): layers 0-1
    //        PP1 (ranks 2,3): layers 2-3
    //        PP2 (ranks 4,5): layers 4-5
    //        PP3 (ranks 6,7): layers 6-7
    //
    // RNN layers: 4 RNN layers distributed differently
    //   Context: PP=1, layersPerPP = {4}
    //   Gen: PP=4, layersPerPP = {2, 2, 0, 0}  (only PP0 and PP1 have RNN layers)
    //        PP0 (ranks 0,1): layers 0-1
    //        PP1 (ranks 2,3): layers 2-3
    //        PP2 (ranks 4,5): no RNN layers
    //        PP3 (ranks 6,7): no RNN layers

    SizeType32 const tp = 2;

    // Unified states with both KV and RNN config
    auto contextState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/4, tp, /*pp=*/1, {8}, {4});
    auto genState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/4, tp, /*pp=*/4, {2, 2, 2, 2}, {2, 2, 0, 0});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // ============= Test from Context rank 0 (PP=0, TP=0) =============
    SizeType32 contextRank0 = 0;

    // KV: Needs all 8 layers from Gen
    //   Gen PP0 (layers 0-1) at TP=0 -> rank 0
    //   Gen PP1 (layers 2-3) at TP=0 -> rank 2
    //   Gen PP2 (layers 4-5) at TP=0 -> rank 4
    //   Gen PP3 (layers 6-7) at TP=0 -> rank 6
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2, 4, 6}));

    // RNN: Needs 4 layers from Gen (only PP0 and PP1 have RNN layers)
    //   Gen PP0 (layers 0-1) at TP=0 -> rank 0
    //   Gen PP1 (layers 2-3) at TP=0 -> rank 2
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2}));

    // Merged counterparts: union of {0,2,4,6} and {0,2} = {0,2,4,6}
    auto mergedCounterParts = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(mergedCounterParts, (std::vector<SizeType32>{0, 2, 4, 6}));

    auto rnnTargetInfo = texec::kv_cache::targetIRanksForRnn(genState, contextState, contextRank0);
    auto [rnnPickUp, rnnLocalRankIndices] = tensorrt_llm::batch_manager::cache_formatter_utils::pickRecvConnections(
        mergedCounterParts.size(), contextState, contextRank0, genState, mergedCounterParts, rnnTargetInfo);
    EXPECT_EQ(rnnPickUp, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(rnnLocalRankIndices, (std::vector<size_t>{0, 1}));

    // ============= Test from Context rank 1 (PP=0, TP=1) =============
    SizeType32 contextRank1 = 1;

    // KV: Same PP stages, but TP=1
    //   Gen PP0 at TP=1 -> rank 1
    //   Gen PP1 at TP=1 -> rank 3
    //   Gen PP2 at TP=1 -> rank 5
    //   Gen PP3 at TP=1 -> rank 7
    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank1).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{1, 3, 5, 7}));

    // RNN:
    //   Gen PP0 at TP=1 -> rank 1
    //   Gen PP1 at TP=1 -> rank 3
    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank1, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{1, 3}));

    mergedCounterParts = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(mergedCounterParts, (std::vector<SizeType32>{1, 3, 5, 7}));

    rnnTargetInfo = texec::kv_cache::targetIRanksForRnn(genState, contextState, contextRank1);
    std::tie(rnnPickUp, rnnLocalRankIndices) = tensorrt_llm::batch_manager::cache_formatter_utils::pickRecvConnections(
        mergedCounterParts.size(), contextState, contextRank1, genState, mergedCounterParts, rnnTargetInfo);
    EXPECT_EQ(rnnPickUp, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(rnnLocalRankIndices, (std::vector<size_t>{0, 1}));
}

// Test: Completely disjoint KV and RNN counterparts
TEST_F(HybridModelCounterpartsTest, DisjointKvRnnCounterparts)
{
    // ============= Setup =============
    // Scenario where KV and RNN layers are on completely different PP stages
    // Context: PP=2 (each PP stage handles different layer types)
    // Gen: PP=4
    //
    // Context PP0: Only KV layers (layers 0-3)
    // Context PP1: Only RNN layers (layers 0-3)
    //
    // Gen:
    //   PP0-1: KV layers spread across PP0 (layers 0-1) and PP1 (layers 2-3)
    //   PP2-3: RNN layers spread across PP2 (layers 0-1) and PP3 (layers 2-3)

    SizeType32 const tp = 2;

    // Context rank 0 (PP=0, TP=0) only has KV layers
    // Context rank 2 (PP=1, TP=0) only has RNN layers

    // Unified states: KV layers on Context PP=0, RNN layers on Context PP=1
    auto contextState = makeHybridState(
        /*kvNumLayers=*/4, /*rnnNumLayers=*/4, tp, /*pp=*/2, {4, 0}, {0, 4});
    auto genState = makeHybridState(
        /*kvNumLayers=*/4, /*rnnNumLayers=*/4, tp, /*pp=*/4, {2, 2, 0, 0}, {0, 0, 2, 2});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // ============= Test from Context rank 0 (PP=0, TP=0) - has KV only =============
    SizeType32 contextRank0 = 0;

    // KV counterparts: needs layers 0-3 from Gen PP0 (rank 0) and PP1 (rank 2)
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2}));

    // RNN counterparts: Context PP=0 has 0 RNN layers, should need nothing from Gen
    // But targetIRanks for RNN should still work correctly
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    // Context PP0 has 0 RNN layers, so it doesn't need any Gen RNN data
    EXPECT_TRUE(rnnCounterParts.empty());

    auto mergedCounterParts = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(mergedCounterParts, (std::vector<SizeType32>{0, 2})); // Only KV ranks

    // ============= Test from Context rank 2 (PP=1, TP=0) - has RNN only =============
    SizeType32 contextRank2 = 2;

    // KV counterparts: Context PP=1 has 0 KV layers
    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank2).mIRanks;
    EXPECT_TRUE(kvCounterParts.empty());

    // RNN counterparts: needs layers 0-3 from Gen PP2 (rank 4) and PP3 (rank 6)
    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank2, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{4, 6}));

    mergedCounterParts = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(mergedCounterParts, (std::vector<SizeType32>{4, 6})); // Only RNN ranks

    auto rnnTargetInfo = texec::kv_cache::targetIRanksForRnn(genState, contextState, contextRank2);
    auto [rnnPickUp, rnnLocalRankIndices] = tensorrt_llm::batch_manager::cache_formatter_utils::pickRecvConnections(
        mergedCounterParts.size(), contextState, contextRank2, genState, mergedCounterParts, rnnTargetInfo);
    EXPECT_EQ(rnnPickUp, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(rnnLocalRankIndices, (std::vector<size_t>{0, 1}));
}

// ==================== ATTENTION-ONLY MODEL TESTS ====================

// Test fixture for attention-only models (KV cache only, no RNN)
class AttentionOnlyModelTest : public ::testing::Test
{
protected:
    static texec::kv_cache::CacheState makeKvCacheState(SizeType32 numLayers, SizeType32 tp, SizeType32 pp,
        std::vector<SizeType32> const& layersPerPP, SizeType32 numHeads = 8, SizeType32 sizePerHead = 64,
        SizeType32 tokensPerBlock = 32)
    {
        return texec::kv_cache::CacheState(numLayers, numHeads, sizePerHead, tokensPerBlock, tp, pp,
            /*contextParallelism=*/1, layersPerPP, nvinfer1::DataType::kFLOAT,
            texec::kv_cache::CacheState::AttentionType::kDEFAULT, /*kvFactor=*/2,
            /*enableAttentionDP=*/false, /*DPrank=*/0, /*DPsize=*/1);
    }
};

// Test: Attention-only model, same PP on both sides
TEST_F(AttentionOnlyModelTest, SamePPConfig)
{
    SizeType32 const tp = 2;
    SizeType32 const pp = 2;
    SizeType32 const numLayers = 32;
    std::vector<SizeType32> layersPerPP = {16, 16};

    auto contextKvState = makeKvCacheState(numLayers, tp, pp, layersPerPP);
    auto genKvState = makeKvCacheState(numLayers, tp, pp, layersPerPP);

    // Rank 0 (PP=0, TP=0) should only talk to Gen rank 0
    auto result0 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mDomainPPSize, 1);

    // Rank 2 (PP=1, TP=0) should only talk to Gen rank 2
    auto result2 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 2);
    EXPECT_EQ(result2.mIRanks, std::vector<int>({2}));
}

// Test: Attention-only model, Context PP=1, Gen PP=4 (typical disagg scenario)
TEST_F(AttentionOnlyModelTest, ContextPP1GenPP4)
{
    SizeType32 const tp = 2;
    SizeType32 const numLayers = 32;

    // Context: PP=1, all 32 layers on one PP rank
    auto contextKvState = makeKvCacheState(numLayers, tp, /*pp=*/1, {32});

    // Gen: PP=4, 8 layers each
    auto genKvState = makeKvCacheState(numLayers, tp, /*pp=*/4, {8, 8, 8, 8});

    // Context rank 0 (PP=0, TP=0) needs data from all Gen PP ranks at TP=0
    // Gen: PP0->rank0, PP1->rank2, PP2->rank4, PP3->rank6
    auto result0 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 2, 4, 6}));
    EXPECT_EQ(result0.mDomainPPSize, 4);
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, (std::vector<int>{8, 8, 8, 8}));

    // Context rank 1 (PP=0, TP=1) needs from all Gen PP ranks at TP=1
    auto result1 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 1);
    EXPECT_EQ(result1.mIRanks, (std::vector<int>{1, 3, 5, 7}));
}

// Test: Attention-only model, Gen PP=1, Context PP=4 (reverse direction)
TEST_F(AttentionOnlyModelTest, GenPP1ContextPP4)
{
    SizeType32 const tp = 2;
    SizeType32 const numLayers = 32;

    // Context: PP=4, 8 layers each
    auto contextKvState = makeKvCacheState(numLayers, tp, /*pp=*/4, {8, 8, 8, 8});

    // Gen: PP=1, all 32 layers
    auto genKvState = makeKvCacheState(numLayers, tp, /*pp=*/1, {32});

    // Context rank 0 (PP=0, TP=0) has layers 0-7, needs from Gen rank 0
    auto result0 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, std::vector<int>({8}));

    // Context rank 4 (PP=2, TP=0) has layers 16-23, still needs from Gen rank 0
    auto result4 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 4);
    EXPECT_EQ(result4.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result4.mPeerLayerNumInDomainPP, std::vector<int>({8}));
}

// Test: Attention-only model with non-uniform layer distribution
TEST_F(AttentionOnlyModelTest, NonUniformLayerDistribution)
{
    SizeType32 const tp = 2;
    SizeType32 const numLayers = 40;

    // Context: PP=2, non-uniform (24 + 16 layers)
    auto contextKvState = makeKvCacheState(numLayers, tp, /*pp=*/2, {24, 16});

    // Gen: PP=4, non-uniform (12, 12, 8, 8 layers)
    auto genKvState = makeKvCacheState(numLayers, tp, /*pp=*/4, {12, 12, 8, 8});

    // Context rank 0 (PP=0, TP=0) has layers 0-23
    //   Gen PP0 (layers 0-11)  -> rank 0, 12 layers overlap
    //   Gen PP1 (layers 12-23) -> rank 2, 12 layers overlap
    auto result0 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 2}));
    EXPECT_EQ(result0.mPeerLayerNumInDomainPP, (std::vector<int>{12, 12}));

    // Context rank 2 (PP=1, TP=0) has layers 24-39
    //   Gen PP2 (layers 24-31) -> rank 4, 8 layers overlap
    //   Gen PP3 (layers 32-39) -> rank 6, 8 layers overlap
    auto result2 = texec::kv_cache::targetIRanks(genKvState, contextKvState, 2);
    EXPECT_EQ(result2.mIRanks, (std::vector<int>{4, 6}));
    EXPECT_EQ(result2.mPeerLayerNumInDomainPP, (std::vector<int>{8, 8}));
}

// ==================== MORE HYBRID MODEL TESTS ====================

// Test: Hybrid model with interleaved layers (like Jamba)
// In Jamba, attention and Mamba layers are interleaved: Att-Mamba-Att-Mamba-...
TEST_F(HybridModelCounterpartsTest, InterleavedLayers)
{
    // Model with 16 total layers: 8 attention, 8 RNN (interleaved)
    // Context: PP=1, Gen: PP=2
    SizeType32 const tp = 2;

    // Unified states: both attention and RNN have 8 layers, distributed evenly
    auto contextState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/8, tp, /*pp=*/1, {8}, {8});
    auto genState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/8, tp, /*pp=*/2, {4, 4}, {4, 4});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    SizeType32 contextRank0 = 0;

    // Both KV and RNN should have same counterparts
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);

    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2}));
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2}));

    // Merged should be same as individual (no new ranks)
    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, (std::vector<SizeType32>{0, 2}));
}

// Test: Hybrid model where RNN has more PP stages than KV
TEST_F(HybridModelCounterpartsTest, RnnMorePPThanKv)
{
    SizeType32 const tp = 2;

    // Unified states: PP=4 (the physical deployment), KV only on first 2 PP stages
    auto contextState = makeHybridState(/*kvNumLayers=*/4, /*rnnNumLayers=*/8, tp, /*pp=*/1, {4}, {8});
    auto genState = makeHybridState(/*kvNumLayers=*/4, /*rnnNumLayers=*/8, tp, /*pp=*/4, {2, 2, 0, 0}, {2, 2, 2, 2});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    SizeType32 contextRank0 = 0;

    // KV needs ranks 0, 2 (from Gen PP0, PP1)
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2}));

    // RNN needs ranks 0, 2, 4, 6 (from Gen PP0, PP1, PP2, PP3)
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2, 4, 6}));

    // Merged should include all RNN ranks (superset)
    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, (std::vector<SizeType32>{0, 2, 4, 6}));
}

// Test: Hybrid model where KV has more PP stages than RNN
TEST_F(HybridModelCounterpartsTest, KvMorePPThanRnn)
{
    SizeType32 const tp = 2;

    // Unified states: PP=4, RNN only on first 2 PP stages
    auto contextState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/4, tp, /*pp=*/1, {8}, {4});
    auto genState = makeHybridState(/*kvNumLayers=*/8, /*rnnNumLayers=*/4, tp, /*pp=*/4, {2, 2, 2, 2}, {2, 2, 0, 0});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    SizeType32 contextRank0 = 0;

    // KV needs ranks 0, 2, 4, 6 (from Gen PP0, PP1, PP2, PP3)
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 2, 4, 6}));

    // RNN needs ranks 0, 2 only (from Gen PP0, PP1 - PP2,PP3 have 0 RNN layers)
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2}));

    // Merged should be KV ranks (superset)
    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, (std::vector<SizeType32>{0, 2, 4, 6}));

    auto rnnTargetInfo = texec::kv_cache::targetIRanksForRnn(genState, contextState, contextRank0);
    auto [rnnPickUp, rnnLocalRankIndices] = tensorrt_llm::batch_manager::cache_formatter_utils::pickRecvConnections(
        merged.size(), contextState, contextRank0, genState, merged, rnnTargetInfo);
    EXPECT_EQ(rnnPickUp, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(rnnLocalRankIndices, (std::vector<size_t>{0, 1}));
}

// Test: Edge case - RNN-only model (0 attention layers)
TEST_F(HybridModelCounterpartsTest, RnnOnlyModel)
{
    SizeType32 const tp = 2;

    // Unified states: 0 KV layers, 8 RNN layers
    auto contextState = makeHybridState(/*kvNumLayers=*/0, /*rnnNumLayers=*/8, tp, /*pp=*/1, {0}, {8});
    auto genState = makeHybridState(/*kvNumLayers=*/0, /*rnnNumLayers=*/8, tp, /*pp=*/2, {0, 0}, {4, 4});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    SizeType32 contextRank0 = 0;

    // KV counterparts should be empty
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_TRUE(kvCounterParts.empty());

    // RNN counterparts should have ranks
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 2}));

    // Merged is just RNN (kvCounterParts is empty)
    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, (std::vector<SizeType32>{0, 2}));
}

// Test: Large scale - 8 GPU Gen with mixed layer counts
TEST_F(HybridModelCounterpartsTest, LargeScaleMixedLayers)
{
    SizeType32 const tp = 4;

    // Context: TP=4, PP=1 (4 GPUs total)
    // Gen: TP=4, PP=4 (16 GPUs total)

    // Unified states: KV 32 layers, RNN 16 layers (only on first 2 PP stages of Gen)
    auto contextState = makeHybridState(/*kvNumLayers=*/32, /*rnnNumLayers=*/16, tp, /*pp=*/1, {32}, {16});
    auto genState = makeHybridState(/*kvNumLayers=*/32, /*rnnNumLayers=*/16, tp, /*pp=*/4, {8, 8, 8, 8}, {8, 8, 0, 0});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // Context rank 0 (PP=0, TP=0)
    SizeType32 contextRank0 = 0;

    // KV needs all 4 Gen PP stages at TP=0: ranks 0, 4, 8, 12
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{0, 4, 8, 12}));

    // RNN needs only first 2 Gen PP stages at TP=0: ranks 0, 4
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{0, 4}));

    // Merged
    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, (std::vector<SizeType32>{0, 4, 8, 12}));

    // Context rank 3 (PP=0, TP=3) - different TP rank
    SizeType32 contextRank3 = 3;

    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank3).mIRanks;
    EXPECT_EQ(kvCounterParts, (std::vector<int>{3, 7, 11, 15}));

    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank3, genState);
    EXPECT_EQ(rnnCounterParts, (std::vector<SizeType32>{3, 7}));
}

// Test: Hybrid model with Context PP > Gen PP (reverse direction)
TEST_F(HybridModelCounterpartsTest, ContextPPGreaterThanGenPP)
{
    // ============= Setup =============
    // Reverse scenario: Context has MORE PP stages than Gen
    // Context: PP=4, TP=2 (8 GPUs)
    // Gen: PP=1, TP=2 (2 GPUs) - simpler Gen setup
    //
    // KV: 16 attention layers
    //   Context: PP=4, layersPerPP = {4, 4, 4, 4}
    //   Gen: PP=1, layersPerPP = {16}
    //
    // RNN: 8 Mamba layers
    //   Context: PP=4, layersPerPP = {2, 2, 2, 2}
    //   Gen: PP=1, layersPerPP = {8}

    SizeType32 const tp = 2;

    // Unified states
    auto contextState = makeHybridState(
        /*kvNumLayers=*/16, /*rnnNumLayers=*/8, tp, /*pp=*/4, {4, 4, 4, 4}, {2, 2, 2, 2});
    auto genState = makeHybridState(
        /*kvNumLayers=*/16, /*rnnNumLayers=*/8, tp, /*pp=*/1, {16}, {8});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // ============= Test from Context rank 0 (PP=0, TP=0) =============
    // Context PP=0 has layers 0-3 for KV, layers 0-1 for RNN
    // Gen PP=0 (rank 0) has ALL layers
    // So both KV and RNN should only need Gen rank 0
    SizeType32 contextRank0 = 0;

    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, std::vector<int>({0}));

    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_EQ(rnnCounterParts, std::vector<SizeType32>({0}));

    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, std::vector<SizeType32>({0}));

    // ============= Test from Context rank 4 (PP=2, TP=0) =============
    // Context PP=2 has layers 8-11 for KV, layers 4-5 for RNN
    // Still needs from Gen rank 0 only
    SizeType32 contextRank4 = 4;

    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank4).mIRanks;
    EXPECT_EQ(kvCounterParts, std::vector<int>({0}));

    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank4, genState);
    EXPECT_EQ(rnnCounterParts, std::vector<SizeType32>({0}));

    // ============= Test from Context rank 1 (PP=0, TP=1) =============
    // Should need Gen rank 1 (same TP rank)
    SizeType32 contextRank1 = 1;

    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank1).mIRanks;
    EXPECT_EQ(kvCounterParts, std::vector<int>({1}));

    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank1, genState);
    EXPECT_EQ(rnnCounterParts, std::vector<SizeType32>({1}));
}

// Test: Hybrid model with asymmetric CTX PP > Gen PP
// (KV and RNN have different layer distributions in Context)
TEST_F(HybridModelCounterpartsTest, AsymmetricContextPPGreaterThanGenPP)
{
    // Context: PP=4, TP=2
    // Gen: PP=2, TP=2
    //
    // KV: Context has layers only on PP0,PP1; Gen distributes across PP0,PP1
    // RNN: Context has layers only on PP2,PP3; Gen distributes across PP0,PP1

    SizeType32 const tp = 2;

    // Unified states: KV on PP0,PP1; RNN on PP2,PP3 for Context
    auto contextState = makeHybridState(
        /*kvNumLayers=*/8, /*rnnNumLayers=*/6, tp, /*pp=*/4, {4, 4, 0, 0}, {0, 0, 3, 3});
    auto genState = makeHybridState(
        /*kvNumLayers=*/8, /*rnnNumLayers=*/6, tp, /*pp=*/2, {4, 4}, {3, 3});

    tbm::RnnCacheFormatter rnnFormatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1),
        reinterpret_cast<tbm::rnn_state_manager::RnnCacheTransBufferManager*>(0x2));

    // ============= Test from Context rank 0 (PP=0, TP=0) - KV only =============
    SizeType32 contextRank0 = 0;

    // KV: Context PP0 has layers 0-3, Gen PP0 has 0-3 -> rank 0
    auto kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank0).mIRanks;
    EXPECT_EQ(kvCounterParts, std::vector<int>({0}));

    // RNN: Context PP0 has 0 RNN layers
    auto rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank0, genState);
    EXPECT_TRUE(rnnCounterParts.empty());

    auto merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, std::vector<SizeType32>({0})); // Only KV

    // ============= Test from Context rank 4 (PP=2, TP=0) - RNN only =============
    SizeType32 contextRank4 = 4;

    // KV: Context PP2 has 0 KV layers
    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank4).mIRanks;
    EXPECT_TRUE(kvCounterParts.empty());

    // RNN: Context PP2 has layers 0-2, Gen PP0 has 0-2 -> rank 0
    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank4, genState);
    EXPECT_EQ(rnnCounterParts, std::vector<SizeType32>({0}));

    merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, std::vector<SizeType32>({0})); // Only RNN

    // ============= Test from Context rank 6 (PP=3, TP=0) - RNN only =============
    SizeType32 contextRank6 = 6;

    // KV: Context PP3 has 0 KV layers
    kvCounterParts = texec::kv_cache::targetIRanks(genState, contextState, contextRank6).mIRanks;
    EXPECT_TRUE(kvCounterParts.empty());

    // RNN: Context PP3 has layers 3-5, Gen PP1 has 3-5 -> rank 2
    rnnCounterParts = rnnFormatter.getCounterparts(contextState, contextRank6, genState);
    EXPECT_EQ(rnnCounterParts, std::vector<SizeType32>({2}));

    merged = mergeCounterparts(kvCounterParts, rnnCounterParts);
    EXPECT_EQ(merged, std::vector<SizeType32>({2}));
}
