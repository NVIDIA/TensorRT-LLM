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
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/HostWorkerPool.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
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
// Every desc starts at a multiple of this offset within its chunk's bounce region.
constexpr std::uint64_t kBounceAlignment = 32;

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
    auto plan = b::BounceTransferPlan::build(
        makeDescs({}), makeDescs({}), /*maxChunkSizeBytes=*/1024, /*maxDescsPerChunk=*/64);
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
        makeDescs({{0x9000, 8, 0}, {0xA000, 8, 0}, {0xB000, 8, 0}}), 4096, /*maxDescsPerChunk=*/2);
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
constexpr std::size_t kSequential = 1;
constexpr std::size_t kFourSegments = 4;
constexpr std::size_t kPoolWorkers = 4;

// Chunk caps of the large-request plans: both the byte cap and the desc cap close chunks there.
constexpr std::size_t kLargeRequestChunkBytes = 16384;
constexpr std::size_t kLargeRequestDescsPerChunk = 40;

kvc::TransferDescs vram(std::vector<kvc::MemoryDesc> descs)
{
    return kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(descs)};
}

struct DescPairs
{
    std::vector<kvc::MemoryDesc> src;
    std::vector<kvc::MemoryDesc> dst;
};

// A request large enough for the parallel planner in which every plan desc and every scatter piece maps
// 1:1 to one input desc: src and dst are strided, so nothing merges, and neighbouring descs differ in
// size, so no strided scatter run forms.
DescPairs makeLargeRequest(std::size_t n)
{
    constexpr std::size_t kEmptyDescPeriod = 997; // zero-length descs are skipped but still counted
    constexpr std::size_t kDescSizeCount = 7;
    constexpr std::size_t kSmallestDescBytes = 64;
    constexpr std::size_t kDescBytesStep = 96;
    constexpr std::size_t kLargestDescBytes = kSmallestDescBytes + (kDescSizeCount - 1) * kDescBytesStep;
    constexpr std::uintptr_t kSrcBase = 0x10000000ULL;
    constexpr std::uintptr_t kDstBase = 0x80000000ULL;
    constexpr std::uintptr_t kSrcPitch = 1024;
    constexpr std::uintptr_t kDstPitch = 2048;
    static_assert(kSrcPitch > kLargestDescBytes && kDstPitch > kLargestDescBytes);

    DescPairs request;
    request.src.reserve(n);
    request.dst.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        std::size_t const len
            = (i % kEmptyDescPeriod == 0) ? 0 : kSmallestDescBytes + (i % kDescSizeCount) * kDescBytesStep;
        request.src.emplace_back(kSrcBase + i * kSrcPitch, len, 0);
        request.dst.emplace_back(kDstBase + i * kDstPitch, len, 0);
    }
    return request;
}

void mismatchDstLen(std::vector<kvc::MemoryDesc>& dst, std::size_t idx)
{
    dst[idx] = kvc::MemoryDesc{dst[idx].getAddr(), dst[idx].getLen() + 1, dst[idx].getDeviceId()};
}

b::BounceTransferPlan planLargeRequest(
    DescPairs const& request, std::size_t segments, b::HostWorkerPool* pool = nullptr)
{
    return b::BounceTransferPlan::build(
        vram(request.src), vram(request.dst), kLargeRequestChunkBytes, kLargeRequestDescsPerChunk, segments, pool);
}

auto throwsLenMismatchAt(std::size_t idx)
{
    return testing::Throws<std::exception>(
        testing::Property(&std::exception::what, testing::HasSubstr("len mismatch at idx " + std::to_string(idx))));
}

// Every chunk honours the planner invariants, the chunks' per-desc entries read in order are exactly
// the request's non-empty descs (each once, with its src/dst/size), offsets are aligned and
// non-overlapping, and each chunk's scatter runs expand to exactly its descs.
void expectPlanCoversRequest(b::BounceTransferPlan const& plan, DescPairs const& request)
{
    auto const& src = request.src;
    auto const& dst = request.dst;
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
        ASSERT_LE(c.srcPtrs.size(), kLargeRequestDescsPerChunk);
        ASSERT_LE(c.packedBytes, kLargeRequestChunkBytes);
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
            ASSERT_EQ(c.bounceOffsets[k] % kBounceAlignment, 0u);
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

// The two plans are identical chunk for chunk (same per-desc arrays, scatter runs and extents).
void expectSamePlan(b::BounceTransferPlan const& expected, b::BounceTransferPlan const& actual)
{
    ASSERT_EQ(actual.numChunks(), expected.numChunks());
    EXPECT_EQ(actual.totalBytes(), expected.totalBytes());
    EXPECT_EQ(actual.totalDescs(), expected.totalDescs());
    for (std::size_t k = 0; k < expected.numChunks(); ++k)
    {
        auto const& want = expected.chunks()[k];
        auto const& got = actual.chunks()[k];
        ASSERT_EQ(got.srcPtrs, want.srcPtrs) << "chunk " << k;
        ASSERT_EQ(got.dstPtrs, want.dstPtrs) << "chunk " << k;
        ASSERT_EQ(got.sizes, want.sizes) << "chunk " << k;
        ASSERT_EQ(got.bounceOffsets, want.bounceOffsets) << "chunk " << k;
        ASSERT_EQ(got.scatterRuns.size(), want.scatterRuns.size()) << "chunk " << k;
        for (std::size_t r = 0; r < want.scatterRuns.size(); ++r)
        {
            auto const& u = want.scatterRuns[r];
            auto const& v = got.scatterRuns[r];
            ASSERT_TRUE(u.bounceOffset == v.bounceOffset && u.dstAddr == v.dstAddr && u.dstStride == v.dstStride
                && u.bounceStride == v.bounceStride && u.pieceSize == v.pieceSize && u.count == v.count)
                << "chunk " << k << " run " << r;
        }
        EXPECT_EQ(got.totalBytes, want.totalBytes);
        EXPECT_EQ(got.packedBytes, want.packedBytes);
        EXPECT_EQ(got.maxDescBytes, want.maxDescBytes);
    }
}

// `count` back-to-back descs of `descBytes` each, the first at `base`.
kvc::TransferDescs contiguousDescs(std::uintptr_t base, std::size_t count, std::size_t descBytes)
{
    std::vector<kvc::MemoryDesc> descs;
    descs.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        descs.emplace_back(base + i * descBytes, descBytes, 0);
    }
    return vram(std::move(descs));
}

constexpr std::size_t kThrowingIndices = 16;
constexpr std::size_t kFirstFailingIdx = 5;
constexpr std::size_t kLaterFailingIdx = 11;

std::string failureAt(std::size_t idx)
{
    return "index " + std::to_string(idx);
}

// parallelFor over kThrowingIndices indices, counting each run in `runs`; kFirstFailingIdx and
// kLaterFailingIdx throw failureAt(index).
void runThrowingIndices(b::HostWorkerPool& pool, std::vector<std::atomic<int>>& runs)
{
    pool.parallelFor(kThrowingIndices,
        [&](std::size_t i)
        {
            runs[i].fetch_add(1);
            if (i == kFirstFailingIdx || i == kLaterFailingIdx)
            {
                throw std::runtime_error(failureAt(i));
            }
        });
}

auto throwsFailureAt(std::size_t idx)
{
    return testing::Throws<std::runtime_error>(
        testing::Property(&std::runtime_error::what, testing::StrEq(failureAt(idx))));
}
} // namespace

TEST(HostWorkerPool, BulkSegmentCount)
{
    EXPECT_EQ(b::bulkSegmentCount(0), 1u);
    EXPECT_EQ(b::bulkSegmentCount(b::kBulkSegmentItems - 1), 1u);
    EXPECT_EQ(b::bulkSegmentCount(b::kBulkSegmentItems), b::kMinBulkSegments);
    EXPECT_EQ(b::bulkSegmentCount(2 * b::kBulkSegmentItems), 2u);
    EXPECT_EQ(b::bulkSegmentCount(3 * b::kBulkSegmentItems + 1), 3u);
    EXPECT_EQ(b::bulkSegmentCount((b::kMaxBulkSegments + 1) * b::kBulkSegmentItems), b::kMaxBulkSegments);
}

TEST(HostWorkerPool, DefaultThreadCountIsBounded)
{
    EXPECT_GE(b::HostWorkerPool::defaultThreadCount(), 1u);
    EXPECT_LE(b::HostWorkerPool::defaultThreadCount(), b::HostWorkerPool::kMaxDefaultWorkers);
}

TEST(HostWorkerPool, RunsEveryIndexOnce)
{
    constexpr std::size_t kIndices = 1000;
    for (std::size_t threads : {std::size_t{0}, std::size_t{1}, std::size_t{4}})
    {
        b::HostWorkerPool pool(threads);
        EXPECT_LE(pool.threadCount(), threads);
        std::vector<std::atomic<int>> runs(kIndices);
        pool.parallelFor(kIndices, [&](std::size_t i) { runs[i].fetch_add(1); });
        for (std::size_t i = 0; i < kIndices; ++i)
        {
            ASSERT_EQ(runs[i].load(), 1) << "threads=" << threads << " index " << i;
        }
    }
}

TEST(HostWorkerPool, ZeroCountRunsNothing)
{
    for (std::size_t threads : {std::size_t{0}, std::size_t{1}, std::size_t{4}})
    {
        b::HostWorkerPool pool(threads);
        std::atomic<int> runs{0};
        pool.parallelFor(/*count=*/0, [&](std::size_t) { runs.fetch_add(1); });
        EXPECT_EQ(runs.load(), 0) << "threads=" << threads;
    }
}

// Like a sequential loop, the error of the lowest failing index wins.
TEST(HostWorkerPool, RethrowsLowestIndexErrorAfterEveryIndexRan)
{
    b::HostWorkerPool pool(kPoolWorkers);
    ASSERT_GT(pool.threadCount(), 0u) << "needs workers to hand indices to";
    std::vector<std::atomic<int>> runs(kThrowingIndices);

    EXPECT_THAT([&] { runThrowingIndices(pool, runs); }, throwsFailureAt(kFirstFailingIdx));
    for (std::size_t i = 0; i < kThrowingIndices; ++i)
    {
        EXPECT_EQ(runs[i].load(), 1) << "index " << i;
    }
}

TEST(HostWorkerPool, WithoutWorkersStopsAtLowestIndexError)
{
    b::HostWorkerPool pool(/*threads=*/0);
    std::vector<std::atomic<int>> runs(kThrowingIndices);

    EXPECT_THAT([&] { runThrowingIndices(pool, runs); }, throwsFailureAt(kFirstFailingIdx));
    for (std::size_t i = 0; i < kThrowingIndices; ++i)
    {
        EXPECT_EQ(runs[i].load(), i <= kFirstFailingIdx ? 1 : 0) << "index " << i;
    }
}

TEST(HostWorkerPool, SharedByConcurrentCallers)
{
    constexpr std::size_t kWorkers = 2;
    constexpr std::size_t kCallers = 6;
    constexpr std::size_t kCallsPerCaller = 50;
    constexpr std::size_t kIndicesPerCall = 8;
    static_assert(kWorkers < kCallers, "some callers find no free worker and must run their indices themselves");
    b::HostWorkerPool pool(kWorkers);
    std::atomic<std::size_t> runs{0};
    std::vector<std::thread> callers;
    for (std::size_t c = 0; c < kCallers; ++c)
    {
        callers.emplace_back(
            [&]
            {
                for (std::size_t call = 0; call < kCallsPerCaller; ++call)
                {
                    pool.parallelFor(kIndicesPerCall, [&](std::size_t) { runs.fetch_add(1); });
                }
            });
    }
    for (auto& t : callers)
    {
        t.join();
    }
    EXPECT_EQ(runs.load(), kCallers * kCallsPerCaller * kIndicesPerCall);
}

TEST(BounceTransferPlan, ParallelBuildMatchesSequential)
{
    std::size_t const n = b::kBulkSegmentItems + 12345;
    std::size_t const autoSegments = b::bulkSegmentCount(n);
    ASSERT_GT(autoSegments, kSequential);
    auto const request = makeLargeRequest(n);
    b::HostWorkerPool pool(kPoolWorkers);

    // The pool only changes who plans the segments, never the plan.
    expectSamePlan(/*expected=*/planLargeRequest(request, kFourSegments),
        /*actual=*/planLargeRequest(request, kFourSegments, &pool));
    expectSamePlan(/*expected=*/planLargeRequest(request, autoSegments),
        /*actual=*/planLargeRequest(request, autoSegments, &pool));

    auto const sequential = planLargeRequest(request, kSequential);
    for (std::size_t segments : {kSequential, kFourSegments, autoSegments})
    {
        auto const plan = planLargeRequest(request, segments);
        expectPlanCoversRequest(plan, request);
        EXPECT_EQ(plan.totalBytes(), sequential.totalBytes());
        EXPECT_EQ(plan.totalDescs(), sequential.totalDescs());
        // Each segment boundary can add at most one partially filled chunk.
        EXPECT_GE(plan.numChunks(), sequential.numChunks()) << "segments=" << segments;
        EXPECT_LE(plan.numChunks(), sequential.numChunks() + segments - 1) << "segments=" << segments;
    }
}

TEST(BounceTransferPlan, SegmentBoundaryStartsFreshChunk)
{
    constexpr std::size_t kDescs = 8;
    constexpr std::uint32_t kDescBytes = kBounceAlignment; // so contiguous descs merge with no alignment gap
    constexpr std::size_t kSegments = 4;
    constexpr std::size_t kDescsPerSegment = kDescs / kSegments;
    constexpr std::uint32_t kSegmentBytes = kDescsPerSegment * kDescBytes;
    constexpr std::uintptr_t kSrcBase = 0x1000;
    auto const src = contiguousDescs(kSrcBase, kDescs, kDescBytes);
    auto const dst = contiguousDescs(/*base=*/0x9000, kDescs, kDescBytes);

    auto const sequential
        = b::BounceTransferPlan::build(src, dst, /*maxChunkSizeBytes=*/1024, /*maxDescsPerChunk=*/64, kSequential);
    auto const segmented
        = b::BounceTransferPlan::build(src, dst, /*maxChunkSizeBytes=*/1024, /*maxDescsPerChunk=*/64, kSegments);

    ASSERT_EQ(sequential.numChunks(), 1u);
    EXPECT_EQ(sequential.chunks()[0].sizes, (std::vector<std::uint32_t>{kDescs * kDescBytes}));
    ASSERT_EQ(segmented.numChunks(), kSegments);
    for (std::size_t k = 0; k < kSegments; ++k)
    {
        auto const& chunk = segmented.chunks()[k];
        EXPECT_EQ(chunk.srcPtrs, (std::vector<std::uint64_t>{kSrcBase + k * kSegmentBytes}));
        EXPECT_EQ(chunk.sizes, (std::vector<std::uint32_t>{kSegmentBytes}));
        EXPECT_EQ(chunk.bounceOffsets, (std::vector<std::uint64_t>{0}));
    }
    EXPECT_EQ(segmented.totalBytes(), sequential.totalBytes());
    EXPECT_EQ(segmented.totalDescs(), sequential.totalDescs());
}

TEST(BounceTransferPlan, MoreSegmentsThanDescsPlansOneDescPerSegment)
{
    constexpr std::size_t kDescs = 8;
    constexpr std::size_t kDescBytes = kBounceAlignment;
    constexpr std::size_t kSegments = 100;
    static_assert(kSegments > kDescs);

    auto const plan = b::BounceTransferPlan::build(contiguousDescs(/*base=*/0x1000, kDescs, kDescBytes),
        contiguousDescs(/*base=*/0x9000, kDescs, kDescBytes), /*maxChunkSizeBytes=*/1024, /*maxDescsPerChunk=*/64,
        kSegments);

    ASSERT_EQ(plan.numChunks(), kDescs);
    for (auto const& chunk : plan.chunks())
    {
        EXPECT_EQ(chunk.srcPtrs.size(), 1u);
    }
    EXPECT_EQ(plan.totalBytes(), kDescs * kDescBytes);
}

TEST(BounceTransferPlan, ParallelBuildReportsMismatchInLastSegment)
{
    std::size_t const n = b::kBulkSegmentItems + 1000;
    std::size_t const lastSegmentIdx = n - 5;
    ASSERT_GE(lastSegmentIdx, b::segmentBegin(n, kFourSegments, /*segment=*/kFourSegments - 1));
    auto request = makeLargeRequest(n);
    mismatchDstLen(request.dst, lastSegmentIdx);
    b::HostWorkerPool pool(kPoolWorkers);

    for (std::size_t segments : {kSequential, kFourSegments, b::bulkSegmentCount(n)})
    {
        EXPECT_THAT([&] { (void) planLargeRequest(request, segments); }, throwsLenMismatchAt(lastSegmentIdx))
            << "segments=" << segments;
        EXPECT_THAT([&] { (void) planLargeRequest(request, segments, &pool); }, throwsLenMismatchAt(lastSegmentIdx))
            << "pool, segments=" << segments;
    }
}

TEST(BounceTransferPlan, ParallelBuildReportsLowestMismatchAcrossSegments)
{
    std::size_t const n = b::kBulkSegmentItems + 1000;
    std::size_t const secondSegmentIdx = b::segmentBegin(n, kFourSegments, /*segment=*/1) + 7;
    std::size_t const lastSegmentIdx = n - 5;
    ASSERT_LT(secondSegmentIdx, b::segmentBegin(n, kFourSegments, /*segment=*/2));
    ASSERT_GE(lastSegmentIdx, b::segmentBegin(n, kFourSegments, /*segment=*/kFourSegments - 1));
    auto request = makeLargeRequest(n);
    mismatchDstLen(request.dst, secondSegmentIdx);
    mismatchDstLen(request.dst, lastSegmentIdx);
    b::HostWorkerPool pool(kPoolWorkers);

    for (std::size_t segments : {kSequential, kFourSegments})
    {
        EXPECT_THAT([&] { (void) planLargeRequest(request, segments); }, throwsLenMismatchAt(secondSegmentIdx))
            << "segments=" << segments;
        EXPECT_THAT([&] { (void) planLargeRequest(request, segments, &pool); }, throwsLenMismatchAt(secondSegmentIdx))
            << "pool, segments=" << segments;
    }
}
