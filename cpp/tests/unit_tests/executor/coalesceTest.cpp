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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"
#include <gtest/gtest.h>

using namespace tensorrt_llm::executor::kv_cache;

// ==================== coalesceMemoryDescs tests ====================

TEST(CoalesceMemoryDescsTest, EmptyInput)
{
    MemoryDescs descs{MemoryType::kVRAM, {}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    EXPECT_EQ(result.getDescs().size(), 0);
}

TEST(CoalesceMemoryDescsTest, SingleEntry)
{
    MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 256);
    EXPECT_EQ(result.getDescs()[0].getDeviceId(), 0);
}

TEST(CoalesceMemoryDescsTest, TwoContiguous)
{
    // [0x1000, 256) then [0x1100, 256) — adjacent, should merge into one
    MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 512);
    EXPECT_EQ(result.getDescs()[0].getDeviceId(), 0);
}

TEST(CoalesceMemoryDescsTest, TwoNonContiguous)
{
    // Gap between blocks — should stay as two
    MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x2000, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 2);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 256);
    EXPECT_EQ(result.getDescs()[1].getAddr(), 0x2000);
    EXPECT_EQ(result.getDescs()[1].getLen(), 256);
}

TEST(CoalesceMemoryDescsTest, ThreeContiguous)
{
    MemoryDescs descs{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1200, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 768);
}

TEST(CoalesceMemoryDescsTest, UnsortedInput)
{
    // Same three contiguous blocks but in reverse order — sorting should fix it
    MemoryDescs descs{
        MemoryType::kVRAM, {MemoryDesc{0x1200, 256, 0}, MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 1);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 768);
}

TEST(CoalesceMemoryDescsTest, DifferentDevices)
{
    // Contiguous addresses but different devices — should NOT merge
    MemoryDescs descs{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 1}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 2);
}

TEST(CoalesceMemoryDescsTest, MixedContiguousAndGaps)
{
    // First two are contiguous, then a gap before the third
    MemoryDescs descs{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x3000, 128, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 2);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 512);
    EXPECT_EQ(result.getDescs()[1].getAddr(), 0x3000);
    EXPECT_EQ(result.getDescs()[1].getLen(), 128);
}

TEST(CoalesceMemoryDescsTest, MultipleDevicesEachContiguous)
{
    // Two contiguous on device 0, two contiguous on device 1
    MemoryDescs descs{MemoryType::kVRAM,
        {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x2000, 128, 1},
            MemoryDesc{0x2080, 128, 1}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 2);
    EXPECT_EQ(result.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(result.getDescs()[0].getLen(), 512);
    EXPECT_EQ(result.getDescs()[0].getDeviceId(), 0);
    EXPECT_EQ(result.getDescs()[1].getAddr(), 0x2000);
    EXPECT_EQ(result.getDescs()[1].getLen(), 256);
    EXPECT_EQ(result.getDescs()[1].getDeviceId(), 1);
}

TEST(CoalesceMemoryDescsTest, PreservesMemoryType)
{
    MemoryDescs descs{MemoryType::kDRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    EXPECT_EQ(result.getType(), MemoryType::kDRAM);
}

TEST(CoalesceMemoryDescsTest, AllSeparate)
{
    // Nothing can be merged — all have gaps
    MemoryDescs descs{MemoryType::kVRAM,
        {MemoryDesc{0x1000, 100, 0}, MemoryDesc{0x2000, 100, 0}, MemoryDesc{0x3000, 100, 0},
            MemoryDesc{0x4000, 100, 0}}};
    auto result = NixlHelper::coalesceMemoryDescs(descs);
    ASSERT_EQ(result.getDescs().size(), 4);
}

// ==================== coalesceTransferDescs tests ====================

TEST(CoalesceTransferDescsTest, EmptyInput)
{
    TransferDescs src{MemoryType::kVRAM, {}};
    TransferDescs dst{MemoryType::kVRAM, {}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    EXPECT_EQ(resSrc.getDescs().size(), 0);
    EXPECT_EQ(resDst.getDescs().size(), 0);
}

TEST(CoalesceTransferDescsTest, SinglePair)
{
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
}

TEST(CoalesceTransferDescsTest, BothSidesContiguous)
{
    // src contiguous AND dst contiguous — should merge into one transfer
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
}

TEST(CoalesceTransferDescsTest, SrcContiguousDstNot)
{
    // src is contiguous but dst has a gap — can't merge
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x6000, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(CoalesceTransferDescsTest, DstContiguousSrcNot)
{
    // dst is contiguous but src has a gap — can't merge
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x2000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(CoalesceTransferDescsTest, DifferentDevicesOnSrc)
{
    // src addresses look contiguous but are on different devices — can't merge
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 1}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 0}, MemoryDesc{0x5100, 256, 0}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(CoalesceTransferDescsTest, DifferentDevicesOnDst)
{
    // dst addresses look contiguous but are on different devices — can't merge
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 0}, MemoryDesc{0x5100, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
}

TEST(CoalesceTransferDescsTest, MismatchedSizes)
{
    // src has 2 entries, dst has 1 — sizes don't match, return as-is
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 1);
}

TEST(CoalesceTransferDescsTest, ThreePairsAllContiguous)
{
    TransferDescs src{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1200, 256, 0}}};
    TransferDescs dst{
        MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x5200, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 768);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 768);
}

TEST(CoalesceTransferDescsTest, PartialMerge)
{
    // First two pairs: both sides contiguous — merge
    // Third pair: src contiguous but dst has gap — stays separate
    TransferDescs src{
        MemoryType::kVRAM, {MemoryDesc{0x1000, 256, 0}, MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1200, 256, 0}}};
    TransferDescs dst{
        MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 1}, MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x9000, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 2);
    ASSERT_EQ(resDst.getDescs().size(), 2);
    // Merged pair
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
    // Separate pair
    EXPECT_EQ(resSrc.getDescs()[1].getAddr(), 0x1200);
    EXPECT_EQ(resSrc.getDescs()[1].getLen(), 256);
    EXPECT_EQ(resDst.getDescs()[1].getAddr(), 0x9000);
    EXPECT_EQ(resDst.getDescs()[1].getLen(), 256);
}

TEST(CoalesceTransferDescsTest, UnsortedInput)
{
    // Same as BothSidesContiguous but in reverse order — sorting should fix it
    TransferDescs src{MemoryType::kVRAM, {MemoryDesc{0x1100, 256, 0}, MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5100, 256, 1}, MemoryDesc{0x5000, 256, 1}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    ASSERT_EQ(resSrc.getDescs().size(), 1);
    ASSERT_EQ(resDst.getDescs().size(), 1);
    EXPECT_EQ(resSrc.getDescs()[0].getAddr(), 0x1000);
    EXPECT_EQ(resSrc.getDescs()[0].getLen(), 512);
    EXPECT_EQ(resDst.getDescs()[0].getAddr(), 0x5000);
    EXPECT_EQ(resDst.getDescs()[0].getLen(), 512);
}

TEST(CoalesceTransferDescsTest, PreservesMemoryType)
{
    TransferDescs src{MemoryType::kDRAM, {MemoryDesc{0x1000, 256, 0}}};
    TransferDescs dst{MemoryType::kVRAM, {MemoryDesc{0x5000, 256, 0}}};
    auto [resSrc, resDst] = NixlHelper::coalesceTransferDescs(src, dst);
    EXPECT_EQ(resSrc.getType(), MemoryType::kDRAM);
    EXPECT_EQ(resDst.getType(), MemoryType::kVRAM);
}
