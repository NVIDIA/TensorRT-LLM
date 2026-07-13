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

TEST(BounceTransferPlan, ContiguousSrcAndDstDescsMergeInPlace)
{
    // Both src and dst advance contiguously (and 32 divides 32, so the bounce cursor has no align
    // gap) -> the two descs collapse into ONE plan desc covering 64 bytes.
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 32, 0}, {0x1020, 32, 0}}), makeDescs({{0x9000, 32, 0}, {0x9020, 32, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    ASSERT_EQ(c.srcPtrs.size(), 1u);
    EXPECT_EQ(c.sizes[0], 64u);
    EXPECT_EQ(c.totalBytes, 64u);
    EXPECT_EQ(c.packedBytes, 64u);
    EXPECT_EQ(plan.totalDescs(), 2u); // both input descs still counted as seen
    EXPECT_EQ(plan.totalBytes(), 64u);
}

TEST(BounceTransferPlan, ContiguousSrcOnlyDoesNotMergeDescs)
{
    // src contiguous but dst jumps -> per-desc arrays must stay separate (the gather is strided).
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 32, 0}, {0x1020, 32, 0}}), makeDescs({{0x9000, 32, 0}, {0xA000, 32, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    EXPECT_EQ(plan.chunks()[0].srcPtrs.size(), 2u);
}

TEST(BounceTransferPlan, ScatterRunsCoalesceContiguousDst)
{
    // dst contiguous, src strided (e.g. ctx tp1 -> gen tp4: dst is the gen rank's dense head-slice
    // pool): per-desc arrays keep 3 entries for the gather, but the scatter view collapses to ONE
    // count==1 run whose pieceSize grew over the whole extent.
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 32, 0}, {0x3000, 32, 0}, {0x5000, 32, 0}}),
        makeDescs({{0x9000, 32, 0}, {0x9020, 32, 0}, {0x9040, 32, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    EXPECT_EQ(c.srcPtrs.size(), 3u);
    ASSERT_EQ(c.scatterRuns.size(), 1u);
    EXPECT_EQ(c.scatterRuns[0].dstAddr, 0x9000u);
    EXPECT_EQ(c.scatterRuns[0].bounceOffset, 0u);
    EXPECT_EQ(c.scatterRuns[0].pieceSize, 96u);
    EXPECT_EQ(c.scatterRuns[0].count, 1u);
}

TEST(BounceTransferPlan, ScatterRunsCoalesceUniformlyStridedDst)
{
    // dst uniformly strided (e.g. ctx tp-slice -> gen DP full-head pool: each 32B piece lands every
    // 128B in the peer pool): ONE strided run of count 3. Bounce packing steps by exactly 32
    // (aligned), so bounceStride == pieceSize.
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 32, 0}, {0x3000, 32, 0}, {0x5000, 32, 0}}),
        makeDescs({{0x9000, 32, 0}, {0x9080, 32, 0}, {0x9100, 32, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    ASSERT_EQ(c.scatterRuns.size(), 1u);
    EXPECT_EQ(c.scatterRuns[0].dstAddr, 0x9000u);
    EXPECT_EQ(c.scatterRuns[0].dstStride, 0x80u);
    EXPECT_EQ(c.scatterRuns[0].bounceStride, 32u);
    EXPECT_EQ(c.scatterRuns[0].pieceSize, 32u);
    EXPECT_EQ(c.scatterRuns[0].count, 3u);
}

TEST(BounceTransferPlan, ScatterRunsBreakOnDstHoleOrAlignGap)
{
    // First pair: dst steps forward but the second desc's SIZE differs -> no stride latch -> two
    // runs. Second pair: dst contiguous but the 100-byte desc aligns the cursor up to 128, leaving a
    // bounce gap -> contiguous growth fails; the stride latch still absorbs it ONLY if sizes match —
    // they don't (100 vs 32) -> two runs.
    auto planHole = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 32, 0}, {0x3000, 16, 0}}), makeDescs({{0x9000, 32, 0}, {0xA000, 16, 0}}), 1024, 64);
    ASSERT_EQ(planHole.numChunks(), 1u);
    EXPECT_EQ(planHole.chunks()[0].scatterRuns.size(), 2u);

    auto planGap = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 100, 0}, {0x3000, 32, 0}}), makeDescs({{0x9000, 100, 0}, {0x9064, 32, 0}}), 1024, 64);
    ASSERT_EQ(planGap.numChunks(), 1u);
    auto const& c = planGap.chunks()[0];
    ASSERT_EQ(c.scatterRuns.size(), 2u);
    EXPECT_EQ(c.scatterRuns[1].bounceOffset, 128u); // alignUp(100,32)
}

TEST(BounceTransferPlan, ScatterRunsIrregularStrideBreaks)
{
    // Same sizes but NON-uniform dst steps (+0x80 then +0x40): the latch fixes stride 0x80 from the
    // first pair; the third desc doesn't land on it -> it opens a new run (2 runs total, 3 pieces).
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 32, 0}, {0x3000, 32, 0}, {0x5000, 32, 0}}),
        makeDescs({{0x9000, 32, 0}, {0x9080, 32, 0}, {0x90C0, 32, 0}}), 1024, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    auto const& c = plan.chunks()[0];
    ASSERT_EQ(c.scatterRuns.size(), 2u);
    EXPECT_EQ(c.scatterRuns[0].count, 2u);
    EXPECT_EQ(c.scatterRuns[1].count, 1u);
    EXPECT_EQ(c.scatterRuns[1].dstAddr, 0x90C0u);
}
