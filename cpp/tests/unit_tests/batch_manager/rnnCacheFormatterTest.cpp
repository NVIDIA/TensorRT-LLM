/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"

namespace texec = tensorrt_llm::executor;
namespace tbm = tensorrt_llm::batch_manager;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

class RnnTargetIRanksTest : public ::testing::Test
{
protected:
    // Helper to create RnnCacheState
    static texec::rnn_cache::RnnCacheState makeRnnCacheState(
        SizeType32 numLayers, SizeType32 tp, SizeType32 pp, std::vector<SizeType32> const& layersPerPP)
    {
        // Model config (use consistent values)
        SizeType32 dState = 16;
        SizeType32 dConv = 4;
        SizeType32 hiddenSize = 256;
        SizeType32 headDim = 64;
        SizeType32 convDimSize = 128;
        SizeType32 nGroups = 1;
        SizeType32 numHeads = 4;
        auto convDtype = nvinfer1::DataType::kFLOAT;
        auto ssmDtype = nvinfer1::DataType::kFLOAT;

        return texec::rnn_cache::RnnCacheState(dState, dConv, hiddenSize, headDim, convDimSize, nGroups, numLayers,
            numHeads, tp, pp, /*enableAttentionDP=*/false, /*DPrank=*/0, /*DPsize=*/1, layersPerPP, convDtype,
            ssmDtype);
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
    auto result0 = tbm::rnnTargetIRanks(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mDomainPPSize, 1);
    EXPECT_EQ(result0.mDomainTPSize, 1);

    // Rank 1: PP=0, TP=1 -> should communicate with gen rank 1
    auto result1 = tbm::rnnTargetIRanks(genState, contextState, 1);
    EXPECT_EQ(result1.mIRanks, std::vector<int>({1}));

    // Rank 2: PP=1, TP=0 -> should communicate with gen rank 2
    auto result2 = tbm::rnnTargetIRanks(genState, contextState, 2);
    EXPECT_EQ(result2.mIRanks, std::vector<int>({2}));

    // Rank 3: PP=1, TP=1 -> should communicate with gen rank 3
    auto result3 = tbm::rnnTargetIRanks(genState, contextState, 3);
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
    auto result0 = tbm::rnnTargetIRanks(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 2}));
    EXPECT_EQ(result0.mDomainPPSize, 2);
    EXPECT_EQ(result0.mPeerAttentionLayerNumInDomainPP, (std::vector<int>{4, 4}));

    // Context rank 1 (PP=0, TP=1, has layers 0-7) needs:
    //   Gen PP=0 at TP=1 -> rank 1
    //   Gen PP=1 at TP=1 -> rank 3
    auto result1 = tbm::rnnTargetIRanks(genState, contextState, 1);
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
    auto result0 = tbm::rnnTargetIRanks(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result0.mDomainPPSize, 1);
    EXPECT_EQ(result0.mPeerAttentionLayerNumInDomainPP, std::vector<int>({4}));

    // Context rank 2 (PP=1, TP=0, has layers 4-7) also needs gen rank 0
    auto result2 = tbm::rnnTargetIRanks(genState, contextState, 2);
    EXPECT_EQ(result2.mIRanks, std::vector<int>({0}));
    EXPECT_EQ(result2.mPeerAttentionLayerNumInDomainPP, std::vector<int>({4}));
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
    auto result0 = tbm::rnnTargetIRanks(genState, contextState, 0);
    EXPECT_EQ(result0.mIRanks, (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(result0.mDomainPPSize, 3);
    EXPECT_EQ(result0.mPeerAttentionLayerNumInDomainPP, (std::vector<int>{4, 4, 2}));

    // Context rank 1 (PP=0, TP=1, has layers 0-9) needs:
    //   Gen PP=0 at TP=1 -> rank 1
    //   Gen PP=1 at TP=1 -> rank 5
    //   Gen PP=2 at TP=1 -> rank 9
    auto result1 = tbm::rnnTargetIRanks(genState, contextState, 1);
    EXPECT_EQ(result1.mIRanks, (std::vector<int>{1, 5, 9}));
    EXPECT_EQ(result1.mDomainPPSize, 3);

    // Context rank 4 (PP=1, TP=0, has layers 10-15) needs:
    //   Gen PP=2 (layers 8-11)  at TP=0 -> rank 8  (2 layers overlap: 10-11)
    //   Gen PP=3 (layers 12-15) at TP=0 -> rank 12 (4 layers overlap)
    auto result4 = tbm::rnnTargetIRanks(genState, contextState, 4);
    EXPECT_EQ(result4.mIRanks, (std::vector<int>{8, 12}));
    EXPECT_EQ(result4.mDomainPPSize, 2);
    EXPECT_EQ(result4.mPeerAttentionLayerNumInDomainPP, (std::vector<int>{2, 4}));

    // Context rank 7 (PP=1, TP=3, has layers 10-15) needs:
    //   Gen PP=2 at TP=3 -> rank 11
    //   Gen PP=3 at TP=3 -> rank 15
    auto result7 = tbm::rnnTargetIRanks(genState, contextState, 7);
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
    tbm::RnnCacheFormatter formatter(reinterpret_cast<tbm::rnn_state_manager::RnnStateManager*>(0x1), state1);

    // Same TP, different PP -> should be supported
    EXPECT_TRUE(formatter.inquireSupport(state1, state2));

    // Different TP -> should NOT be supported (for now)
    EXPECT_FALSE(formatter.inquireSupport(state1, state3));
}
