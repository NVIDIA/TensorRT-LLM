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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
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
    auto plan = b::BounceTransferPlan::build(makeDescs({}), makeDescs({}), /*maxChunkSizeBytes=*/1024, /*maxDescs=*/64);
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
    EXPECT_EQ(c.sizes, (std::vector<std::uint32_t>{100, 50}));
    EXPECT_EQ(c.totalBytes, 150u);
}

TEST(BounceTransferPlan, OverflowSplitsIntoTwoChunks)
{
    // With a 256-byte chunk cap, two 200-byte descriptors cannot share a chunk.
    auto plan = b::BounceTransferPlan::build(
        makeDescs({{0x1000, 200, 0}, {0x2000, 200, 0}}), makeDescs({{0x9000, 200, 0}, {0xA000, 200, 0}}), 256, 64);
    EXPECT_EQ(plan.numChunks(), 2u);
    EXPECT_EQ(plan.chunks()[0].srcPtrs.size(), 1u);
    EXPECT_EQ(plan.chunks()[1].srcPtrs.size(), 1u);
}

TEST(BounceTransferPlan, DescExactlyChunkSizeIsOneChunk)
{
    auto plan = b::BounceTransferPlan::build(makeDescs({{0x1000, 256, 0}}), makeDescs({{0x9000, 256, 0}}), 256, 64);
    ASSERT_EQ(plan.numChunks(), 1u);
    EXPECT_EQ(plan.chunks()[0].totalBytes, 256u);
}

TEST(BounceTransferPlan, DescLargerThanChunkThrows)
{
    EXPECT_ANY_THROW(
        (void) b::BounceTransferPlan::build(makeDescs({{0x1000, 257, 0}}), makeDescs({{0x9000, 257, 0}}), 256, 64));
}

TEST(BounceTransferPlan, MaxChunkSizeBytesAboveU32Throws)
{
    // A chunk's packed size travels in 32-bit wire fields, so maxChunkSizeBytes must fit in 32 bits
    // even though arena offsets are 64-bit. Building with a >4 GiB cap must be rejected.
    EXPECT_ANY_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}}), makeDescs({{0x9000, 8, 0}}),
        /*maxChunkSizeBytes=*/(std::size_t{1} << 32), 64));
    // Exactly 4 GiB - 1 is allowed.
    EXPECT_NO_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, 0}}), makeDescs({{0x9000, 8, 0}}),
        /*maxChunkSizeBytes=*/(std::size_t{1} << 32) - 1, 64));
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

TEST(BounceTransferPlan, MixedSourceDeviceIdsThrow)
{
    EXPECT_ANY_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, /*dev=*/0}, {0x2000, 8, /*dev=*/1}}),
        makeDescs({{0x9000, 8, /*dev=*/0}, {0xA000, 8, /*dev=*/0}}), 4096, 64));
}

TEST(BounceTransferPlan, MixedDestinationDeviceIdsThrow)
{
    EXPECT_ANY_THROW((void) b::BounceTransferPlan::build(makeDescs({{0x1000, 8, /*dev=*/0}, {0x2000, 8, /*dev=*/0}}),
        makeDescs({{0x9000, 8, /*dev=*/0}, {0xA000, 8, /*dev=*/1}}), 4096, 64));
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

// ---- parallel (segmented) plan build ----

namespace
{
// A request large enough for the parallel planner. src AND dst are strided (no in-place merges, no
// scatter-run growth) and neighbouring descs differ in size (no stride latch), so every plan desc and
// every scatter piece maps 1:1 to one input desc. Every 997th desc is zero-length (skipped, counted).
// `badIdx` (if < n) gets a dst length that does not match its src.
std::pair<std::vector<kvc::MemoryDesc>, std::vector<kvc::MemoryDesc>> makeLargeRequest(
    std::size_t n, std::size_t badIdx = std::numeric_limits<std::size_t>::max())
{
    std::vector<kvc::MemoryDesc> src;
    std::vector<kvc::MemoryDesc> dst;
    src.reserve(n);
    dst.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        std::size_t const len = (i % 997 == 0) ? 0 : 64 + (i % 7) * 96; // 64..640 B
        src.emplace_back(static_cast<std::uintptr_t>(0x10000000ULL + i * 1024), len, 0);
        dst.emplace_back(static_cast<std::uintptr_t>(0x80000000ULL + i * 2048), i == badIdx ? len + 1 : len, 0);
    }
    return {std::move(src), std::move(dst)};
}

kvc::TransferDescs vram(std::vector<kvc::MemoryDesc> descs)
{
    return kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(descs)};
}

// Every chunk honours the planner invariants, the chunks' per-desc entries read in order are exactly
// the request's non-empty descs (each once, with its src/dst/size), offsets are 32 B-aligned and
// non-overlapping, and each chunk's scatter runs expand to exactly its descs.
void expectPlanCoversRequest(b::BounceTransferPlan const& plan, std::vector<kvc::MemoryDesc> const& src,
    std::vector<kvc::MemoryDesc> const& dst, std::size_t maxChunkSizeBytes, std::size_t maxDescsPerChunk)
{
    std::size_t next = 0;
    std::uint64_t bytes = 0;
    auto skipEmpty = [&]
    {
        while (next < src.size() && src[next].getLen() == 0)
        {
            ++next;
        }
    };
    for (auto const& c : plan.chunks())
    {
        ASSERT_FALSE(c.srcPtrs.empty());
        ASSERT_LE(c.srcPtrs.size(), maxDescsPerChunk);
        ASSERT_LE(c.packedBytes, maxChunkSizeBytes);
        ASSERT_EQ(c.dstPtrs.size(), c.srcPtrs.size());
        ASSERT_EQ(c.sizes.size(), c.srcPtrs.size());
        ASSERT_EQ(c.bounceOffsets.size(), c.srcPtrs.size());
        std::uint64_t chunkBytes = 0;
        for (std::size_t k = 0; k < c.srcPtrs.size(); ++k)
        {
            skipEmpty();
            ASSERT_LT(next, src.size());
            ASSERT_EQ(c.srcPtrs[k], src[next].getAddr()) << "desc " << next;
            ASSERT_EQ(c.dstPtrs[k], dst[next].getAddr()) << "desc " << next;
            ASSERT_EQ(c.sizes[k], src[next].getLen()) << "desc " << next;
            ASSERT_EQ(c.bounceOffsets[k] % 32, 0u);
            if (k == 0)
            {
                ASSERT_EQ(c.bounceOffsets[k], 0u);
            }
            else
            {
                ASSERT_GE(c.bounceOffsets[k], c.bounceOffsets[k - 1] + c.sizes[k - 1]);
            }
            chunkBytes += c.sizes[k];
            ++next;
        }
        EXPECT_EQ(c.totalBytes, chunkBytes);
        EXPECT_EQ(c.packedBytes, c.bounceOffsets.back() + c.sizes.back());
        std::size_t piece = 0;
        for (auto const& r : c.scatterRuns)
        {
            for (std::uint32_t p = 0; p < r.count; ++p, ++piece)
            {
                ASSERT_LT(piece, c.srcPtrs.size());
                ASSERT_EQ(r.bounceOffset + static_cast<std::uint64_t>(p) * r.bounceStride, c.bounceOffsets[piece]);
                ASSERT_EQ(r.dstAddr + static_cast<std::uint64_t>(p) * r.dstStride, c.dstPtrs[piece]);
                ASSERT_EQ(r.pieceSize, c.sizes[piece]);
            }
        }
        EXPECT_EQ(piece, c.srcPtrs.size());
        bytes += chunkBytes;
    }
    skipEmpty();
    EXPECT_EQ(next, src.size()); // every desc consumed exactly once
    EXPECT_EQ(plan.totalBytes(), bytes);
    EXPECT_EQ(plan.totalDescs(), src.size());
}

std::string buildError(std::vector<kvc::MemoryDesc> const& src, std::vector<kvc::MemoryDesc> const& dst,
    std::size_t buildSegments, b::HostWorkerPool* pool = nullptr)
{
    try
    {
        (void) b::BounceTransferPlan::build(vram(src), vram(dst), 16384, 40, buildSegments, pool);
    }
    catch (std::exception const& e)
    {
        return e.what();
    }
    return {};
}

// The two plans are identical chunk for chunk (same per-desc arrays, scatter runs and extents).
void expectSamePlan(b::BounceTransferPlan const& a, b::BounceTransferPlan const& b2)
{
    ASSERT_EQ(a.numChunks(), b2.numChunks());
    EXPECT_EQ(a.totalBytes(), b2.totalBytes());
    EXPECT_EQ(a.totalDescs(), b2.totalDescs());
    for (std::size_t k = 0; k < a.numChunks(); ++k)
    {
        auto const& x = a.chunks()[k];
        auto const& y = b2.chunks()[k];
        ASSERT_EQ(x.srcPtrs, y.srcPtrs) << "chunk " << k;
        ASSERT_EQ(x.dstPtrs, y.dstPtrs) << "chunk " << k;
        ASSERT_EQ(x.sizes, y.sizes) << "chunk " << k;
        ASSERT_EQ(x.bounceOffsets, y.bounceOffsets) << "chunk " << k;
        ASSERT_EQ(x.scatterRuns.size(), y.scatterRuns.size()) << "chunk " << k;
        for (std::size_t r = 0; r < x.scatterRuns.size(); ++r)
        {
            auto const& u = x.scatterRuns[r];
            auto const& v = y.scatterRuns[r];
            ASSERT_TRUE(u.bounceOffset == v.bounceOffset && u.dstAddr == v.dstAddr && u.dstStride == v.dstStride
                && u.bounceStride == v.bounceStride && u.pieceSize == v.pieceSize && u.count == v.count)
                << "chunk " << k << " run " << r;
        }
        EXPECT_EQ(x.totalBytes, y.totalBytes);
        EXPECT_EQ(x.packedBytes, y.packedBytes);
        EXPECT_EQ(x.maxDescBytes, y.maxDescBytes);
    }
}
} // namespace

TEST(HostWorkerPool, BulkSegmentCount)
{
    EXPECT_EQ(b::bulkSegmentCount(0), 1u);
    EXPECT_EQ(b::bulkSegmentCount(b::kBulkSegmentItems - 1), 1u);
    EXPECT_EQ(b::bulkSegmentCount(b::kBulkSegmentItems), 2u); // the threshold itself already splits
    EXPECT_EQ(b::bulkSegmentCount(2 * b::kBulkSegmentItems), 2u);
    EXPECT_EQ(b::bulkSegmentCount(3 * b::kBulkSegmentItems + 1), 3u);
    EXPECT_EQ(b::bulkSegmentCount(std::size_t{1} << 20), b::kMaxBulkSegments); // 1M descs: capped
}

TEST(HostWorkerPool, DefaultThreadCountIsBounded)
{
    EXPECT_GE(b::HostWorkerPool::defaultThreadCount(), 1u);
    EXPECT_LE(b::HostWorkerPool::defaultThreadCount(), 16u);
}

TEST(HostWorkerPool, RunsEveryIndexOnce)
{
    for (std::size_t threads : {std::size_t{0}, std::size_t{1}, std::size_t{4}})
    {
        b::HostWorkerPool pool(threads);
        EXPECT_LE(pool.threadCount(), threads);
        constexpr std::size_t kCount = 1000;
        std::vector<std::atomic<int>> hits(kCount);
        pool.parallelFor(kCount, [&](std::size_t i) { hits[i].fetch_add(1); });
        for (std::size_t i = 0; i < kCount; ++i)
        {
            ASSERT_EQ(hits[i].load(), 1) << "threads=" << threads << " index " << i;
        }
        pool.parallelFor(0, [](std::size_t) { FAIL() << "no index to run"; });
    }
}

TEST(HostWorkerPool, RethrowsLowestIndexError)
{
    b::HostWorkerPool pool(4);
    std::atomic<int> ran{0};
    try
    {
        pool.parallelFor(16,
            [&](std::size_t i)
            {
                ran.fetch_add(1);
                if (i == 11 || i == 5)
                {
                    throw std::runtime_error("index " + std::to_string(i));
                }
            });
        FAIL() << "expected an exception";
    }
    catch (std::runtime_error const& e)
    {
        EXPECT_STREQ(e.what(), "index 5"); // the lowest failing index wins, like a sequential loop
    }
    EXPECT_EQ(ran.load(), 16);             // the other indices still ran
}

TEST(HostWorkerPool, SharedByConcurrentCallers)
{
    b::HostWorkerPool pool(2); // fewer workers than callers: callers must still finish on their own
    constexpr int kCallers = 6;
    std::atomic<std::size_t> total{0};
    std::vector<std::thread> callers;
    for (int c = 0; c < kCallers; ++c)
    {
        callers.emplace_back(
            [&]
            {
                for (int round = 0; round < 50; ++round)
                {
                    pool.parallelFor(8, [&](std::size_t) { total.fetch_add(1); });
                }
            });
    }
    for (auto& t : callers)
    {
        t.join();
    }
    EXPECT_EQ(total.load(), std::size_t{kCallers} * 50 * 8);
}

TEST(BounceTransferPlan, ParallelBuildMatchesSequential)
{
    constexpr std::size_t kMaxChunk = 16384;
    constexpr std::size_t kMaxDescs = 40; // both the byte and the desc cap close chunks here
    std::size_t const n = b::kBulkSegmentItems + 12345;
    auto const [src, dst] = makeLargeRequest(n);
    auto const seq = b::BounceTransferPlan::build(vram(src), vram(dst), kMaxChunk, kMaxDescs, /*buildSegments=*/1);
    auto const par = b::BounceTransferPlan::build(vram(src), vram(dst), kMaxChunk, kMaxDescs, /*buildSegments=*/4);
    auto const autoPlan = b::BounceTransferPlan::build(vram(src), vram(dst), kMaxChunk, kMaxDescs); // default
    b::HostWorkerPool pool(4);
    auto const parPool
        = b::BounceTransferPlan::build(vram(src), vram(dst), kMaxChunk, kMaxDescs, /*buildSegments=*/4, &pool);
    auto const autoPool = b::BounceTransferPlan::build(vram(src), vram(dst), kMaxChunk, kMaxDescs, 0, &pool);
    // The pool only changes who plans the segments, never the plan.
    expectSamePlan(parPool, par);
    expectSamePlan(autoPool, autoPlan);
    for (auto const* plan : {&seq, &par, &autoPlan, &parPool})
    {
        expectPlanCoversRequest(*plan, src, dst, kMaxChunk, kMaxDescs);
        EXPECT_EQ(plan->totalBytes(), seq.totalBytes());
        EXPECT_EQ(plan->totalDescs(), seq.totalDescs());
        // Each extra segment can add at most one partially filled chunk.
        EXPECT_GE(plan->numChunks(), seq.numChunks());
        EXPECT_LE(plan->numChunks(), seq.numChunks() + b::kMaxBulkSegments - 1);
    }
}

TEST(BounceTransferPlan, SegmentBoundaryStartsFreshChunk)
{
    // Eight contiguous 32 B descs: one merged 256 B desc sequentially; with 4 segments each segment
    // merges its own pair and starts a fresh chunk, so the plan has 4 chunks of one 64 B desc each.
    std::vector<std::tuple<std::uintptr_t, std::size_t, std::uint32_t>> s;
    std::vector<std::tuple<std::uintptr_t, std::size_t, std::uint32_t>> d;
    for (std::uintptr_t i = 0; i < 8; ++i)
    {
        s.emplace_back(0x1000 + i * 32, 32, 0);
        d.emplace_back(0x9000 + i * 32, 32, 0);
    }
    auto const seq = b::BounceTransferPlan::build(makeDescs(s), makeDescs(d), 1024, 64, /*buildSegments=*/1);
    auto const par = b::BounceTransferPlan::build(makeDescs(s), makeDescs(d), 1024, 64, /*buildSegments=*/4);
    ASSERT_EQ(seq.numChunks(), 1u);
    EXPECT_EQ(seq.chunks()[0].sizes, (std::vector<std::uint32_t>{256}));
    ASSERT_EQ(par.numChunks(), 4u);
    for (std::size_t k = 0; k < 4; ++k)
    {
        EXPECT_EQ(par.chunks()[k].srcPtrs, (std::vector<std::uint64_t>{0x1000 + k * 64}));
        EXPECT_EQ(par.chunks()[k].sizes, (std::vector<std::uint32_t>{64}));
        EXPECT_EQ(par.chunks()[k].bounceOffsets, (std::vector<std::uint64_t>{0}));
    }
    EXPECT_EQ(par.totalBytes(), seq.totalBytes());
    EXPECT_EQ(par.totalDescs(), seq.totalDescs());
    // More segments than descs is capped at one desc per segment.
    auto const capped = b::BounceTransferPlan::build(makeDescs(s), makeDescs(d), 1024, 64, /*buildSegments=*/100);
    EXPECT_EQ(capped.numChunks(), 8u);
    EXPECT_EQ(capped.totalBytes(), 256u);
}

TEST(BounceTransferPlan, ParallelBuildErrorInLateSegmentThrows)
{
    std::size_t const n = b::kBulkSegmentItems + 1000;
    std::size_t const bad = n - 5; // in the last segment
    auto const [src, dst] = makeLargeRequest(n, bad);
    std::string const expect = "len mismatch at idx " + std::to_string(bad);
    b::HostWorkerPool pool(4);
    for (std::size_t segments : {std::size_t{1}, std::size_t{4}, std::size_t{0}})
    {
        auto const err = buildError(src, dst, segments);
        EXPECT_NE(err.find(expect), std::string::npos) << "segments=" << segments << ": " << err;
        auto const errPool = buildError(src, dst, segments, &pool);
        EXPECT_NE(errPool.find(expect), std::string::npos) << "pool, segments=" << segments << ": " << errPool;
    }

    // Two bad descs in different segments: the parallel build reports the lower index, like the
    // sequential one (results are collected in segment order).
    std::size_t const early = n / 4 + 7; // second segment of four
    auto [src2, dst2] = makeLargeRequest(n, bad);
    dst2[early] = kvc::MemoryDesc{dst2[early].getAddr(), dst2[early].getLen() + 1, 0};
    std::string const expectEarly = "len mismatch at idx " + std::to_string(early);
    for (std::size_t segments : {std::size_t{1}, std::size_t{4}})
    {
        auto const err = buildError(src2, dst2, segments);
        EXPECT_NE(err.find(expectEarly), std::string::npos) << "segments=" << segments << ": " << err;
        auto const errPool = buildError(src2, dst2, segments, &pool);
        EXPECT_NE(errPool.find(expectEarly), std::string::npos) << "pool, segments=" << segments << ": " << errPool;
    }
}
