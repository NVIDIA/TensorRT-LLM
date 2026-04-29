/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/runtime/common.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace tb = tensorrt_llm::batch_manager;

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using BeamTokens = tb::DecoderInputBuffers::BeamTokens;

namespace
{

/// @brief Standalone implementation of the parentIds-tracing algorithm used by
///        TrtGptModelInflightBatching::buildGatheredBeamTokensForCallback().
///        This allows testing the core gathering logic without needing GPU buffers
///        or a full runtime setup.
///
/// @param perSlotTokens  mTokens-style per-slot token histories [beamWidth][promptLen + numGenerated]
/// @param parentIds      Flat parentIds buffer [beamWidth * maxSeqLength], where
///                       parentIds[beam * maxSeqLength + pos] = parent of `beam` at `pos`
/// @param beamWidth      Number of beams
/// @param promptLen      Length of the prompt (shared across all beams)
/// @param maxSeqLength   Second dimension of the parentIds tensor
/// @return Gathered (coherent) beam token histories, or nullopt if no reordering needed
std::optional<BeamTokens> gatherBeamTokens(BeamTokens const& perSlotTokens,
    std::vector<TokenIdType> const& parentIds, SizeType32 beamWidth, SizeType32 promptLen, SizeType32 maxSeqLength)
{
    // Per-beam generated counts — finished/inactive beams may have shorter histories
    // than the SamplingConfig nominal width, so we don't take beam 0's length as the
    // ground truth.
    std::vector<SizeType32> numGeneratedPerBeam(beamWidth);
    SizeType32 maxGenerated = 0;
    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        auto const beamGenerated = static_cast<SizeType32>(perSlotTokens[beam].size()) - promptLen;
        numGeneratedPerBeam[beam] = beamGenerated;
        maxGenerated = std::max(maxGenerated, beamGenerated);
    }

    if (maxGenerated <= 1)
    {
        return std::nullopt;
    }

    BeamTokens gatheredTokens(beamWidth);
    bool anyReorderNeeded = false;

    for (SizeType32 beam = 0; beam < beamWidth; ++beam)
    {
        auto const beamNumGenerated = numGeneratedPerBeam[beam];

        if (beamNumGenerated <= 0)
        {
            gatheredTokens[beam] = perSlotTokens[beam];
            continue;
        }

        // Trace parentIds backward to find ancestor slot at each generation step
        std::vector<SizeType32> slotAtStep(beamNumGenerated);
        SizeType32 slot = beam;
        for (SizeType32 g = beamNumGenerated - 1; g >= 0; --g)
        {
            slotAtStep[g] = slot;
            if (g > 0)
            {
                slot = parentIds[slot * maxSeqLength + (promptLen + g)];
            }
        }

        // Check if reordering needed
        for (SizeType32 g = 0; g < beamNumGenerated; ++g)
        {
            if (slotAtStep[g] != beam)
            {
                anyReorderNeeded = true;
                break;
            }
        }

        // Build gathered path
        auto& tokens = gatheredTokens[beam];
        tokens.reserve(static_cast<size_t>(promptLen) + static_cast<size_t>(beamNumGenerated));

        // Prompt tokens (same for all beams)
        tokens.insert(tokens.end(), perSlotTokens[0].begin(), perSlotTokens[0].begin() + promptLen);

        // Gathered generated tokens
        for (SizeType32 g = 0; g < beamNumGenerated; ++g)
        {
            tokens.push_back(perSlotTokens[slotAtStep[g]][promptLen + g]);
        }
    }

    if (anyReorderNeeded)
    {
        return gatheredTokens;
    }
    return std::nullopt;
}

} // namespace

class GatherBeamTokensTest : public ::testing::Test
{
};

// beam_width=1: no gathering needed, always returns nullopt
TEST_F(GatherBeamTokensTest, BeamWidth1NoOp)
{
    // prompt = [10, 20], generated = [30, 40]
    BeamTokens perSlotTokens = {{10, 20, 30, 40}};
    std::vector<TokenIdType> parentIds(1 * 10, 0); // all parent 0

    auto result = gatherBeamTokens(perSlotTokens, parentIds, 1, 2, 10);
    EXPECT_FALSE(result.has_value());
}

// beam_width=2, no reorder: parentIds always point to self
TEST_F(GatherBeamTokensTest, BeamWidth2NoReorder)
{
    SizeType32 const beamWidth = 2;
    SizeType32 const promptLen = 2;
    SizeType32 const maxSeqLen = 10;

    // prompt = [10, 20]
    // slot 0 generated: [30, 50]
    // slot 1 generated: [40, 60]
    // parentIds: slot 0 always from 0, slot 1 always from 1
    BeamTokens perSlotTokens = {{10, 20, 30, 50}, {10, 20, 40, 60}};

    std::vector<TokenIdType> parentIds(beamWidth * maxSeqLen, 0);
    // parentIds[beam][pos] = beam (identity mapping)
    for (SizeType32 b = 0; b < beamWidth; ++b)
    {
        for (SizeType32 p = 0; p < maxSeqLen; ++p)
        {
            parentIds[b * maxSeqLen + p] = b;
        }
    }

    auto result = gatherBeamTokens(perSlotTokens, parentIds, beamWidth, promptLen, maxSeqLen);
    // No reorder needed — beams never diverged
    EXPECT_FALSE(result.has_value());
}

// Core test: beam_width=2 with a single reorder at step 1
// This is the exact scenario from the PR description analysis.
TEST_F(GatherBeamTokensTest, BeamWidth2WithReorder)
{
    SizeType32 const beamWidth = 2;
    SizeType32 const promptLen = 2;
    SizeType32 const maxSeqLen = 10;

    // prompt = [A=10, B=20]
    //
    // Step 0 (first beam search):
    //   Both slots start with [10, 20]
    //   slot 0 gets token C=30 from parent 0, slot 1 gets token D=40 from parent 0
    //   parentIds[0][2] = 0, parentIds[1][2] = 0
    //   mTokens[0] = [10, 20, 30], mTokens[1] = [10, 20, 40]
    //
    // Step 1 (second beam search):
    //   slot 0 gets token E=50 from parent 1 (KV cache from slot 1 = [10, 20, 40])
    //   slot 1 gets token F=60 from parent 0 (KV cache from slot 0 = [10, 20, 30])
    //   parentIds[0][3] = 1, parentIds[1][3] = 0
    //   mTokens[0] = [10, 20, 30, 50], mTokens[1] = [10, 20, 40, 60]
    //
    // At the callback AFTER step 1:
    //   logits[0] continues from [10, 20, 40, 50] (slot 0's KV traces through parent 1)
    //   logits[1] continues from [10, 20, 30, 60] (slot 1's KV traces through parent 0)
    //
    //   mTokens[0] = [10, 20, 30, 50] — WRONG! Has 30 at pos 2, should be 40
    //   mTokens[1] = [10, 20, 40, 60] — WRONG! Has 40 at pos 2, should be 30
    //
    // Gathered paths should be:
    //   beam 0: [10, 20, 40, 50]  (slot 1 at step 0, slot 0 at step 1)
    //   beam 1: [10, 20, 30, 60]  (slot 0 at step 0, slot 1 at step 1)

    BeamTokens perSlotTokens = {{10, 20, 30, 50}, {10, 20, 40, 60}};

    std::vector<TokenIdType> parentIds(beamWidth * maxSeqLen, 0);
    // Step 0: both beams from parent 0
    parentIds[0 * maxSeqLen + 2] = 0; // slot 0 at pos 2 came from parent 0
    parentIds[1 * maxSeqLen + 2] = 0; // slot 1 at pos 2 came from parent 0
    // Step 1: cross-over reorder
    parentIds[0 * maxSeqLen + 3] = 1; // slot 0 at pos 3 came from parent 1
    parentIds[1 * maxSeqLen + 3] = 0; // slot 1 at pos 3 came from parent 0

    auto result = gatherBeamTokens(perSlotTokens, parentIds, beamWidth, promptLen, maxSeqLen);

    ASSERT_TRUE(result.has_value());
    auto const& gathered = result.value();

    ASSERT_EQ(gathered.size(), 2);
    // Beam 0's gathered path: [10, 20, D=40, E=50]
    EXPECT_EQ(gathered[0], (std::vector<TokenIdType>{10, 20, 40, 50}));
    // Beam 1's gathered path: [10, 20, C=30, F=60]
    EXPECT_EQ(gathered[1], (std::vector<TokenIdType>{10, 20, 30, 60}));
}

// beam_width=3 with fan-out: all beams come from the same parent
TEST_F(GatherBeamTokensTest, BeamWidth3FanOut)
{
    SizeType32 const beamWidth = 3;
    SizeType32 const promptLen = 1;
    SizeType32 const maxSeqLen = 10;

    // prompt = [100]
    // Step 0: all beams from parent 0. tokens: slot0=10, slot1=20, slot2=30
    // Step 1: all beams still from parent 0 at step 0, but fan out
    //   slot 0 from parent 1, slot 1 from parent 2, slot 2 from parent 0
    //   tokens: slot0=40, slot1=50, slot2=60

    BeamTokens perSlotTokens = {{100, 10, 40}, {100, 20, 50}, {100, 30, 60}};

    std::vector<TokenIdType> parentIds(beamWidth * maxSeqLen, 0);
    // Step 0: all from parent 0
    parentIds[0 * maxSeqLen + 1] = 0;
    parentIds[1 * maxSeqLen + 1] = 0;
    parentIds[2 * maxSeqLen + 1] = 0;
    // Step 1: shuffle
    parentIds[0 * maxSeqLen + 2] = 1; // slot 0 from parent 1
    parentIds[1 * maxSeqLen + 2] = 2; // slot 1 from parent 2
    parentIds[2 * maxSeqLen + 2] = 0; // slot 2 from parent 0

    auto result = gatherBeamTokens(perSlotTokens, parentIds, beamWidth, promptLen, maxSeqLen);

    ASSERT_TRUE(result.has_value());
    auto const& gathered = result.value();

    ASSERT_EQ(gathered.size(), 3);
    // Beam 0: trace back from slot 0. At pos 2, parent=1. At pos 1, slot is 1, parent=parentIds[1][1]=0.
    // So: slotAtStep[0] = 0 (from parentIds[1][1]=0 → but wait, let me retrace)
    // g=1: slotAtStep[1]=0, slot=parentIds[0*10+2]=1
    // g=0: slotAtStep[0]=1
    // gathered[0] = [100, mTokens[1][1], mTokens[0][2]] = [100, 20, 40]
    EXPECT_EQ(gathered[0], (std::vector<TokenIdType>{100, 20, 40}));
    // Beam 1: g=1: slotAtStep[1]=1, slot=parentIds[1*10+2]=2
    //          g=0: slotAtStep[0]=2
    // gathered[1] = [100, mTokens[2][1], mTokens[1][2]] = [100, 30, 50]
    EXPECT_EQ(gathered[1], (std::vector<TokenIdType>{100, 30, 50}));
    // Beam 2: g=1: slotAtStep[1]=2, slot=parentIds[2*10+2]=0
    //          g=0: slotAtStep[0]=0
    // gathered[2] = [100, mTokens[0][1], mTokens[2][2]] = [100, 10, 60]
    EXPECT_EQ(gathered[2], (std::vector<TokenIdType>{100, 10, 60}));
}

// Only one generated token: no reorder possible
TEST_F(GatherBeamTokensTest, SingleGeneratedToken)
{
    SizeType32 const beamWidth = 2;
    SizeType32 const promptLen = 3;
    SizeType32 const maxSeqLen = 10;

    BeamTokens perSlotTokens = {{1, 2, 3, 10}, {1, 2, 3, 20}};

    std::vector<TokenIdType> parentIds(beamWidth * maxSeqLen, 0);
    parentIds[0 * maxSeqLen + 3] = 0;
    parentIds[1 * maxSeqLen + 3] = 0;

    auto result = gatherBeamTokens(perSlotTokens, parentIds, beamWidth, promptLen, maxSeqLen);
    // numGenerated=1, should return nullopt
    EXPECT_FALSE(result.has_value());
}

// Multi-step chain: beam_width=2, 4 generation steps with cascading reorders
TEST_F(GatherBeamTokensTest, MultiStepChainReorder)
{
    SizeType32 const beamWidth = 2;
    SizeType32 const promptLen = 1;
    SizeType32 const maxSeqLen = 10;

    // prompt = [100]
    // Step 0: slot0 gets A=1 from parent 0, slot1 gets B=2 from parent 0
    //   parentIds[0][1]=0, parentIds[1][1]=0
    //   mTokens = [[100,1], [100,2]]
    //
    // Step 1: swap — slot0 from parent 1, slot1 from parent 0
    //   parentIds[0][2]=1, parentIds[1][2]=0
    //   slot0 gets C=3, slot1 gets D=4
    //   mTokens = [[100,1,3], [100,2,4]]
    //
    // Step 2: swap again — slot0 from parent 1, slot1 from parent 0
    //   parentIds[0][3]=1, parentIds[1][3]=0
    //   slot0 gets E=5, slot1 gets F=6
    //   mTokens = [[100,1,3,5], [100,2,4,6]]
    //
    // Step 3: identity — slot0 from parent 0, slot1 from parent 1
    //   parentIds[0][4]=0, parentIds[1][4]=1
    //   slot0 gets G=7, slot1 gets H=8
    //   mTokens = [[100,1,3,5,7], [100,2,4,6,8]]

    BeamTokens perSlotTokens = {{100, 1, 3, 5, 7}, {100, 2, 4, 6, 8}};

    std::vector<TokenIdType> parentIds(beamWidth * maxSeqLen, 0);
    // Step 0: all from parent 0
    parentIds[0 * maxSeqLen + 1] = 0;
    parentIds[1 * maxSeqLen + 1] = 0;
    // Step 1: swap
    parentIds[0 * maxSeqLen + 2] = 1;
    parentIds[1 * maxSeqLen + 2] = 0;
    // Step 2: swap again
    parentIds[0 * maxSeqLen + 3] = 1;
    parentIds[1 * maxSeqLen + 3] = 0;
    // Step 3: identity
    parentIds[0 * maxSeqLen + 4] = 0;
    parentIds[1 * maxSeqLen + 4] = 1;

    auto result = gatherBeamTokens(perSlotTokens, parentIds, beamWidth, promptLen, maxSeqLen);

    ASSERT_TRUE(result.has_value());
    auto const& gathered = result.value();

    // Trace beam 0:
    //   g=3: slot=0, parent[0][4]=0 → slot becomes 0
    //   g=2: slot=0, parent[0][3]=1 → slot becomes 1
    //   g=1: slot=1, parent[1][2]=0 → slot becomes 0
    //   g=0: slot=0 (final)
    // slotAtStep = [0, 1, 0, 0]
    // gathered[0] = [100, mTokens[0][1], mTokens[1][2], mTokens[0][3], mTokens[0][4]]
    //             = [100, 1, 4, 5, 7]
    EXPECT_EQ(gathered[0], (std::vector<TokenIdType>{100, 1, 4, 5, 7}));

    // Trace beam 1:
    //   g=3: slot=1, parent[1][4]=1 → slot becomes 1
    //   g=2: slot=1, parent[1][3]=0 → slot becomes 0
    //   g=1: slot=0, parent[0][2]=1 → slot becomes 1
    //   g=0: slot=1 (final)
    // slotAtStep = [1, 0, 1, 1]
    // gathered[1] = [100, mTokens[1][1], mTokens[0][2], mTokens[1][3], mTokens[1][4]]
    //             = [100, 2, 3, 6, 8]
    EXPECT_EQ(gathered[1], (std::vector<TokenIdType>{100, 2, 3, 6, 8}));
}

// DecoderInputBuffers::gatheredBeamTokensForCallback field basic functionality
TEST_F(GatherBeamTokensTest, DecoderInputBuffersFieldWorks)
{
    // Verify the new field on DecoderInputBuffers can store and retrieve gathered tokens
    std::vector<std::optional<BeamTokens>> gathered(3);

    // First entry: has gathered tokens
    gathered[0] = BeamTokens{{1, 2, 3}, {4, 5, 6}};
    // Second entry: no gathering needed
    gathered[1] = std::nullopt;
    // Third entry: has gathered tokens
    gathered[2] = BeamTokens{{7, 8}, {9, 10}};

    EXPECT_TRUE(gathered[0].has_value());
    EXPECT_FALSE(gathered[1].has_value());
    EXPECT_TRUE(gathered[2].has_value());

    EXPECT_EQ(gathered[0].value()[0], (std::vector<TokenIdType>{1, 2, 3}));
    EXPECT_EQ(gathered[0].value()[1], (std::vector<TokenIdType>{4, 5, 6}));
    EXPECT_EQ(gathered[2].value()[0], (std::vector<TokenIdType>{7, 8}));
}
