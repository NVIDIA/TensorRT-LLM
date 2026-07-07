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

// End-to-end transport tests over REAL NIXL RDMA (two in-process agents, real UCX backend): drive
// the full bounce pipeline (gather -> RDMA write -> scatter + credit recycling) and verify every
// byte arrives. Sizing is chosen so the chunk count exceeds the per-flow window, forcing credit
// recycling (R1). Skips if no CUDA device or the NIXL backend can't init.

#include "bounceTestNixlNode.h"

#include <gtest/gtest.h>

#include <chrono>
#include <string>

namespace kvc = tensorrt_llm::executor::kv_cache;
namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
// One end-to-end transfer of `nDescs` x `descBytes` through the bounce pipeline between two real
// NIXL nodes. maxChunkBytes/windowDepth are chosen so the chunk count exceeds the window (forcing
// credit recycling). `tag` gives the two agents unique names (NIXL agents register by name).
void runTransfer(std::string const& tag, std::uint32_t nDescs, std::uint32_t descBytes, std::size_t maxChunkBytes,
    std::uint32_t windowDepth)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    b::BounceConfig cfg;
    cfg.maxChunkBytes = maxChunkBytes; // per-chunk byte cap
    cfg.windowDepth = windowDepth;
    cfg.window = windowDepth;          // per-flow in-flight region cap -> still forces credit recycling
    cfg.scatterWorkers = 2;
    cfg.minBlock = 256;                // buddy granularity (matches the 256-aligned desc layout)
    // Arena large enough to hold a full window of max-size regions for BOTH roles with headroom for
    // buddy rounding; rounded to a power of two so the BuddyAllocator uses all of it.
    std::size_t arenaBytes = 1ULL << 20;
    while (arenaBytes < static_cast<std::size_t>(windowDepth) * maxChunkBytes * 4ULL)
    {
        arenaBytes <<= 1;
    }
    cfg.arenaBytes = arenaBytes;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, maxChunkBytes / 256ULL);

    auto A = bounce_test::makeNode(tag + "A", cfg, maxDescs);
    auto B = bounce_test::makeNode(tag + "B", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    auto bufs = bounce_test::makeXferBufs(nDescs, descBytes, /*seed=*/1);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}
} // namespace

TEST(BounceTransport, SmallTransferFitsOneWindow)
{
    // 4 descs, chunk holds a few -> chunks <= windowDepth.
    runTransfer("btSmall", /*nDescs=*/4, /*descBytes=*/512, /*maxChunkBytes=*/8192, /*windowDepth=*/4);
}

TEST(BounceTransport, LargeTransferRecyclesCredits)
{
    // 40 descs of ~700B, chunk=4KB (a few descs/chunk) -> chunks >> windowDepth(2): forces credit
    // recycling through a 2-region window. Exercises R1 + the recycling loop.
    runTransfer("btLarge", /*nDescs=*/40, /*descBytes=*/700, /*maxChunkBytes=*/4096, /*windowDepth=*/2);
}

TEST(BounceTransport, ManySmallDescs)
{
    // Closer to the real KV pattern: many tiny descs.
    runTransfer("btMany", /*nDescs=*/500, /*descBytes=*/256, /*maxChunkBytes=*/16384, /*windowDepth=*/4);
}

TEST(BounceTransport, ConcurrentRequestsToSameReceiver)
{
    // Two independent transfers (distinct rids) over one transport pair run as separate flows.
    runTransfer("btConc1", 8, 1024, 8192, 3);
    runTransfer("btConc2", 8, 1024, 8192, 3);
}

// Regression: a maxChunkBytes that passes the naive "<= arenaBytes" check but exceeds the arena's
// USABLE capacity must be clamped at construction. The buddy allocator rounds usable capacity DOWN
// (to minBlock<<maxOrder) and rounds each request UP to a power of two, so an unclamped chunk sized
// to the whole arena can never be granted and the flow hangs to leaseTimeout. Here arena=96KiB has
// only 64KiB usable, yet maxChunkBytes=96KiB (96KiB <= 96KiB passes). A transfer larger than the
// usable capacity must still complete byte-exact — proving the cap was clamped so chunks split to fit.
TEST(BounceTransport, MaxChunkBytesClampedToUsableArena)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.arenaBytes = 96 * 1024;    // buddy usable rounds DOWN to 64KiB (256<<8)
    cfg.maxChunkBytes = 96 * 1024; // exceeds the 64KiB usable -> must be clamped to 64KiB
    cfg.minBlock = 256;
    cfg.windowDepth = 2;
    cfg.window = 2;
    cfg.scatterWorkers = 2;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkBytes / 256ULL);

    auto A = bounce_test::makeNode("btClampA", cfg, maxDescs);
    auto B = bounce_test::makeNode("btClampB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    // 4 x 20KiB = 80KiB total > 64KiB usable. Unclamped, the planner packs all 80KiB into ONE chunk
    // (<= 96KiB cap) that can never be allocated (rounds to 128KiB > 64KiB usable) -> hang. Clamped to
    // 64KiB, it splits into chunks that each fit a drained arena and recycle through.
    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/4, /*descBytes=*/20480, /*seed=*/9);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready)
        << "transfer hung -> maxChunkBytes was NOT clamped to usable arena capacity";
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// Sender-side arena backpressure: the receiver's arena/window is generous (grants every chunk's
// credit up front) but the SENDER's arena only fits a few concurrent gather regions, so most
// credits get parked in pendingCredits and drain via drainPendingPosts as ACKs free regions. The
// transfer must still complete byte-exact (parked != dropped). This is also the path the
// `arenaStarved` NVTX span instruments.
TEST(BounceTransport, SenderArenaBackpressureParksCredits)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig small; // sender: 64KiB usable -> at most 4 in-flight 16KiB gather regions
    small.maxChunkBytes = 16 * 1024;
    small.minBlock = 256;
    small.windowDepth = 8;
    small.window = 8;
    small.scatterWorkers = 2;
    small.arenaBytes = 64 * 1024;
    b::BounceConfig big = small; // receiver: room to grant the full 8-credit window at once
    big.arenaBytes = 1ULL << 20;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, small.maxChunkBytes / 256ULL);

    auto A = bounce_test::makeNode("btParkA", small, maxDescs);
    auto B = bounce_test::makeNode("btParkB", big, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    // 32 x 4KiB = 128KiB in ~8 chunks of 16KiB: double the sender's usable arena, so at least half
    // the granted credits must park and retry.
    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/32, /*descBytes=*/4096, /*seed=*/11);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}
