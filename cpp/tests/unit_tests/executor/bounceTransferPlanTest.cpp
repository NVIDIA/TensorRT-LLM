/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferPlan.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{
// Build a TransferDescs from (addr,len,dev) tuples. Addresses are synthetic; the planner never
// dereferences them, it only bins by length / device.
kvc::TransferDescs makeDescs(std::vector<std::tuple<std::uintptr_t, std::size_t, std::uint32_t>> const& t)
{
    std::vector<kvc::MemoryDesc> v;
    v.reserve(t.size());
    for (auto const& [a, l, d] : t)
    {
        v.emplace_back(a, l, d);
    }
    return kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(v)};
}
} // namespace

TEST(BounceTransferPlan, EmptyYieldsNoChunks)
{
    auto plan = b::BounceTransferPlan::build(makeDescs({}), makeDescs({}), /*slot=*/1024, /*maxDescs=*/64);
    EXPECT_EQ(plan.numChunks(), 0u);
    EXPECT_EQ(plan.totalDescs(), 0u);
    EXPECT_EQ(plan.totalBytes(), 0u);
}

TEST(BounceTransferPlan, SingleDescOneChunk)
{
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 100, 0}}), makeDescs({{0x9000, 100, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    EXPECT_EQ(c.srcPtrs.size(), 1u);
    EXPECT_EQ(c.bounceOffsets[0], 0u);
    EXPECT_EQ(c.sizes[0], 100u);
    EXPECT_EQ(c.totalBytes, 100u);
    EXPECT_EQ(c.dstPtrs[0], 0x9000u);
}

TEST(BounceTransferPlan, TwoDescsPackOneChunkWith32ByteAlignedOffsets)
{
    // len 100 -> next offset aligns up to 128 (multiple of 32).
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 100, 0}, {0x2000, 50, 0}}), makeDescs({{0x9000, 100, 0}, {0xA000, 50, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    EXPECT_EQ(c.bounceOffsets[0], 0u);
    EXPECT_EQ(c.bounceOffsets[1], 128u); // alignUp(100,32)=128
    EXPECT_EQ(c.totalBytes, 150u);
}

TEST(BounceTransferPlan, OverflowSplitsIntoTwoChunks)
{
    // slot=256; two 200-byte descs cannot share a slot.
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 200, 0}, {0x2000, 200, 0}}), makeDescs({{0x9000, 200, 0}, {0xA000, 200, 0}}), 256, 64);
    EXPECT_EQ(plan.numChunks(), 2u);
    EXPECT_EQ(plan.chunks()[0].srcPtrs.size(), 1u);
    EXPECT_EQ(plan.chunks()[1].srcPtrs.size(), 1u);
}

TEST(BounceTransferPlan, DescExactlySlotBytesIsOneChunk)
{
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 256, 0}}), makeDescs({{0x9000, 256, 0}}), 256, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    EXPECT_EQ(plan.chunks()[0].totalBytes, 256u);
}

TEST(BounceTransferPlan, DescLargerThanSlotThrows)
{
    EXPECT_ANY_THROW(
        (void) b::BounceTransferPlan::build(makeDescs({{0x1000, 257, 0}}), makeDescs({{0x9000, 257, 0}}), 256, 64));
}

TEST(BounceTransferPlan, MaxChunkBytesAboveU32Throws)
{
    // A chunk's packed size travels in 32-bit wire fields, so maxChunkBytes must fit in 32 bits even
    // though arena offsets are 64-bit. Building with a >4 GiB cap must be rejected, not silently wrap.
    EXPECT_ANY_THROW((void) b::BounceTransferPlan::build(
        makeDescs({{0x1000, 8, 0}}), makeDescs({{0x9000, 8, 0}}), /*maxChunkBytes=*/(std::size_t{1} << 32), 64));
    // Exactly 4 GiB - 1 is allowed.
    EXPECT_NO_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}}), makeDescs({{0x9000, 8, 0}}),
        /*maxChunkBytes=*/(std::size_t{1} << 32) - 1, 64));
}

TEST(BounceTransferPlan, MaxDescsPerChunkBoundary)
{
    // 3 tiny descs, maxDescs=2 -> first chunk holds 2, second holds 1.
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}, {0x2000, 8, 0}, {0x3000, 8, 0}}),
        makeDescs({{0x9000, 8, 0}, {0xA000, 8, 0}, {0xB000, 8, 0}}), 4096, /*maxDescs=*/2);
    ASSERT_EQ(plan.numChunks(), 2u);
    EXPECT_EQ(plan.chunks()[0].srcPtrs.size(), 2u);
    EXPECT_EQ(plan.chunks()[1].srcPtrs.size(), 1u);
}

TEST(BounceTransferPlan, DeviceMismatchSplits)
{
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}, {0x2000, 8, 0}}),
        makeDescs({{0x9000, 8, /*dev=*/0}, {0xA000, 8, /*dev=*/1}}), 4096, 64);
    ASSERT_EQ(plan.numChunks(), 2u);
    EXPECT_EQ(plan.chunks()[0].dstDeviceId, 0u);
    EXPECT_EQ(plan.chunks()[1].dstDeviceId, 1u);
}

TEST(BounceTransferPlan, ZeroLengthDescSkippedButCounted)
{
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 0, 0}, {0x2000, 16, 0}}), makeDescs({{0x9000, 0, 0}, {0xA000, 16, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    EXPECT_EQ(plan.chunks()[0].srcPtrs.size(), 1u); // zero-len skipped from packing
    EXPECT_EQ(plan.totalDescs(), 2u);               // but still counted as seen
    EXPECT_EQ(plan.totalBytes(), 16u);
}

TEST(BounceTransferPlan, CountMismatchThrows)
{
    EXPECT_ANY_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}}), makeDescs({}), 1024, 64));
}
