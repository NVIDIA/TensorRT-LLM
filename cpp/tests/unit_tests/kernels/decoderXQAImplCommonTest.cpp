/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include "tensorrt_llm/common/envUtils.h"

#include <gtest/gtest.h>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

namespace
{

tk::XQAParams makeSpecDecXqaParams(int numKvHeads, int maxPastKvLength)
{
    tk::XQAParams params{};
    params.num_kv_heads = numKvHeads;
    params.max_past_kv_length = maxPastKvLength;
    return params;
}

void skipIfXqaBlocksPerSequenceIsForced()
{
    if (tc::getEnvXqaBlocksPerSequence().has_value())
    {
        GTEST_SKIP() << "TRTLLM_XQA_BLOCKS_PER_SEQUENCE overrides multi-block tuning.";
    }
}

TEST(DecoderXQAImplCommonTest, computeMultiBlockCountSpecDecPopulatesLowCtaLaunch)
{
    skipIfXqaBlocksPerSequenceIsForced();

    int constexpr kBatchSize = 1;
    int constexpr kMultiprocessorCount = 128;
    int constexpr kSpecDecBlocks = 1;
    auto const params = makeSpecDecXqaParams(/*numKvHeads=*/1, /*maxPastKvLength=*/4096);

    EXPECT_EQ(tk::computeMultiBlockCountSpecDec(params, kBatchSize, kMultiprocessorCount, kSpecDecBlocks), 16);
}

TEST(DecoderXQAImplCommonTest, computeMultiBlockCountSpecDecAccountsForTokenBlocks)
{
    skipIfXqaBlocksPerSequenceIsForced();

    int constexpr kBatchSize = 1;
    int constexpr kMultiprocessorCount = 128;
    int constexpr kSpecDecBlocks = 8;
    auto const params = makeSpecDecXqaParams(/*numKvHeads=*/1, /*maxPastKvLength=*/4096);

    EXPECT_EQ(tk::computeMultiBlockCountSpecDec(params, kBatchSize, kMultiprocessorCount, kSpecDecBlocks), 8);
}

TEST(DecoderXQAImplCommonTest, computeMultiBlockCountSpecDecKeepsShortHistorySingleBlock)
{
    skipIfXqaBlocksPerSequenceIsForced();

    int constexpr kBatchSize = 1;
    int constexpr kMultiprocessorCount = 128;
    int constexpr kSpecDecBlocks = 1;
    auto const params = makeSpecDecXqaParams(/*numKvHeads=*/1, /*maxPastKvLength=*/1024);

    EXPECT_EQ(tk::computeMultiBlockCountSpecDec(params, kBatchSize, kMultiprocessorCount, kSpecDecBlocks), 1);
}

TEST(DecoderXQAImplCommonTest, getSpecDecHmmaMTileSizeUsesHeadTokens)
{
    EXPECT_EQ(tk::getSpecDecHmmaMTileSize(/*headGrpSize=*/8, /*qSeqLen=*/4), 32U);
}

TEST(DecoderXQAImplCommonTest, getSpecDecHmmaTokenBlocksPerGroupMatchesCompiledTile)
{
    EXPECT_EQ(tk::getSpecDecHmmaTokenBlocksPerGroup(/*headGrpSize=*/8, /*qSeqLen=*/4), 1U);
    EXPECT_EQ(tk::getSpecDecHmmaTokenBlocksPerGroup(/*headGrpSize=*/16, /*qSeqLen=*/4), 2U);
}

} // namespace
