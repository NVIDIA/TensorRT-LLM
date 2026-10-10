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
// byte arrives. Skips if no CUDA device or the NIXL backend cannot initialize.

#include "bounceTestNixlNode.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferPlan.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/HostWorkerPool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;
namespace b = tensorrt_llm::executor::kv_cache::bounce;

using bounce_test::Node;

namespace
{
// One end-to-end transfer of `nDescs` x `descBytes` through the bounce pipeline between two real
// NIXL nodes: a sender built from `senderCfg` and a receiver built from `receiverCfg` (asymmetric
// configs let a test pin clamping/backpressure to one side). `tag` gives the two agents unique names.
void runTransfer(std::string const& tag, std::uint32_t nDescs, std::uint32_t descBytes,
    b::BounceConfig const& senderCfg, b::BounceConfig const& receiverCfg, std::uint32_t seed,
    bounce_test::BackendParams const& backendParams = bounce_test::dedicatedWorkerBackendParams())
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    auto nodes = bounce_test::makeWiredPair(tag, senderCfg, receiverCfg, backendParams);
    if (!nodes)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    Node& sender = *nodes->sender;
    Node& receiver = *nodes->receiver;

    auto bufs = bounce_test::makeXferBufs(nDescs, descBytes, seed);
    auto fut = sender.tx->submit(bufs.srcDescs, bufs.dstDescs, receiver.name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    sender.tx->shutdown();
    receiver.tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// Power-of-two arena (so BuddyAllocator can use all of it) that holds the in-flight chunks of both roles
// with headroom for buddy rounding.
std::size_t arenaBytesFor(std::size_t maxChunkSizeBytes, std::uint32_t maxInflightChunksPerRequest)
{
    constexpr std::size_t kRoles = 2;
    constexpr std::size_t kBuddyRoundingHeadroom = 2;
    constexpr std::size_t kMinArenaBytes = 1ULL << 20;
    std::size_t const needed = kRoles * kBuddyRoundingHeadroom * maxInflightChunksPerRequest * maxChunkSizeBytes;
    return std::bit_ceil(std::max(kMinArenaBytes, needed));
}

// The same config on both ends: chunks of at most `maxChunkSizeBytes`, `maxInflightChunksPerRequest` of
// them in flight per request.
void runTransfer(std::string const& tag, std::uint32_t nDescs, std::uint32_t descBytes, std::size_t maxChunkSizeBytes,
    std::uint32_t maxInflightChunksPerRequest,
    bounce_test::BackendParams const& backendParams = bounce_test::dedicatedWorkerBackendParams())
{
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = maxChunkSizeBytes;
    cfg.maxInflightChunksPerRequest = maxInflightChunksPerRequest;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = bounce_test::kDescAlignmentBytes;
    cfg.arenaSizeBytes = arenaBytesFor(maxChunkSizeBytes, maxInflightChunksPerRequest);
    runTransfer(tag, nDescs, descBytes, cfg, cfg, /*seed=*/1, backendParams);
}

// Number of chunks a transport configured with `cfg` splits `bufs` into (as long as its arena does not
// clamp the chunk size).
std::size_t plannedChunkCount(bounce_test::XferBufs const& bufs, b::BounceConfig const& cfg)
{
    return b::BounceTransferPlan::build(bufs.srcDescs, bufs.dstDescs, cfg.maxChunkSizeBytes, b::maxDescsPerChunk(cfg),
        b::bulkSegmentCount(bufs.sizes.size()))
        .numChunks();
}
} // namespace

TEST(BounceTransport, SmallTransferFitsInflightLimit)
{
    runTransfer("btSmall", /*nDescs=*/4, /*descBytes=*/512, /*maxChunkSizeBytes=*/8192,
        /*maxInflightChunksPerRequest=*/4);
}

TEST(BounceTransport, LargeTransferRecyclesCredits)
{
    // Forty descriptors produce more chunks than the limit of two, forcing credit recycling.
    runTransfer("btLarge", /*nDescs=*/40, /*descBytes=*/700, /*maxChunkSizeBytes=*/4096,
        /*maxInflightChunksPerRequest=*/2);
}

TEST(BounceTransport, LargeTransferOnSharedNixlWorker)
{
    runTransfer("btShared", /*nDescs=*/40, /*descBytes=*/700, /*maxChunkSizeBytes=*/4096,
        /*maxInflightChunksPerRequest=*/2, bounce_test::sharedWorkerBackendParams());
}

TEST(BounceTransport, ManySmallDescs)
{
    // Closer to the real KV pattern: many tiny descs.
    runTransfer("btMany", /*nDescs=*/500, /*descBytes=*/256, /*maxChunkSizeBytes=*/16384,
        /*maxInflightChunksPerRequest=*/4);
}

TEST(BounceTransport, ConcurrentRequestsToSameReceiver)
{
    // Two independent transfers (distinct rids) over ONE transport pair, both submitted before
    // either future is waited on, so the flows are truly concurrent on the same A->B connection.
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = 8192;
    cfg.maxInflightChunksPerRequest = 3;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.arenaSizeBytes = 1ULL << 20;
    auto nodes = bounce_test::makeWiredPair("btConc", cfg);
    if (!nodes)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    Node& sender = *nodes->sender;
    Node& receiver = *nodes->receiver;

    // Seed-distinct payloads so any cross-talk between the two in-flight flows fails verification.
    auto bufs1 = bounce_test::makeXferBufs(/*nDescs=*/8, /*descBytes=*/1024, /*seed=*/1);
    auto bufs2 = bounce_test::makeXferBufs(/*nDescs=*/8, /*descBytes=*/1024, /*seed=*/2);
    auto fut1 = sender.tx->submit(bufs1.srcDescs, bufs1.dstDescs, receiver.name);
    auto fut2 = sender.tx->submit(bufs2.srcDescs, bufs2.dstDescs, receiver.name);
    ASSERT_EQ(fut1.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "first transfer hung";
    ASSERT_EQ(fut2.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "second transfer hung";
    EXPECT_EQ(fut1.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_EQ(fut2.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs1)) << "byte mismatch (first flow)";
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs2)) << "byte mismatch (second flow)";

    sender.tx->shutdown();
    receiver.tx->shutdown();
    bounce_test::freeXferBufs(bufs1);
    bounce_test::freeXferBufs(bufs2);
}

TEST(BounceTransport, ReplacementHandshakeClearsStaleCompatibility)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.arenaSizeBytes = 1ULL << 20;
    cfg.maxChunkSizeBytes = 4096;
    cfg.arenaAllocationGranularityBytes = 256;
    auto node = bounce_test::makeNode("btHandshake", cfg, b::maxDescsPerChunk(cfg));
    if (!node)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }

    auto compatible = node->tx->localHandshakeBlob();
    ASSERT_TRUE(node->tx->registerPeerHandshake("peer", compatible));
    EXPECT_TRUE(node->tx->hasPeerHandshake("peer"));

    b::BounceHandshake incompatible;
    ASSERT_TRUE(b::decodeHandshake(compatible, incompatible));
    incompatible.maxChunkSizeBytes += 1;
    EXPECT_FALSE(node->tx->registerPeerHandshake("peer", b::encodeHandshake(incompatible)));
    EXPECT_FALSE(node->tx->hasPeerHandshake("peer"));

    // request_timeout_ms is handshake-enforced too (receiver lease/quarantine derive from it).
    b::BounceHandshake timeoutMismatch;
    ASSERT_TRUE(b::decodeHandshake(compatible, timeoutMismatch));
    EXPECT_EQ(timeoutMismatch.requestTimeoutMs, cfg.requestTimeoutMs);
    timeoutMismatch.requestTimeoutMs += 1;
    ASSERT_TRUE(node->tx->registerPeerHandshake("peer", compatible));
    EXPECT_FALSE(node->tx->registerPeerHandshake("peer", b::encodeHandshake(timeoutMismatch)));
    EXPECT_FALSE(node->tx->hasPeerHandshake("peer"));

    ASSERT_TRUE(node->tx->registerPeerHandshake("peer", compatible));
    EXPECT_FALSE(node->tx->registerPeerHandshake("peer", {}));
    EXPECT_FALSE(node->tx->hasPeerHandshake("peer"));

    // Peer input must never throw out of the metadata-exchange path: a decodable handshake with an
    // empty or un-connectable endpoint just disables bounce for that peer.
    b::BounceHandshake invalidEndpoint;
    ASSERT_TRUE(b::decodeHandshake(compatible, invalidEndpoint));
    invalidEndpoint.endpoint.clear();
    bool registered = true;
    EXPECT_NO_THROW(registered = node->tx->registerPeerHandshake("peer", b::encodeHandshake(invalidEndpoint)));
    EXPECT_FALSE(registered);
    EXPECT_FALSE(node->tx->hasPeerHandshake("peer"));

    invalidEndpoint.endpoint = "not-a-zmq-endpoint";
    registered = true;
    EXPECT_NO_THROW(registered = node->tx->registerPeerHandshake("peer", b::encodeHandshake(invalidEndpoint)));
    EXPECT_FALSE(registered);
    EXPECT_FALSE(node->tx->hasPeerHandshake("peer"));
    node->tx->shutdown();
}

// Regression: maxChunkSizeBytes can be no larger than the buddy allocator's usable capacity, which
// may be smaller than arenaSizeBytes. A 96 KiB arena with 256-byte granularity has only one 64 KiB
// top-level buddy block. The effective chunk cap must therefore become 64 KiB.
TEST(BounceTransport, MaxChunkSizeBytesClampedToUsableArena)
{
    b::BounceConfig cfg;
    cfg.arenaSizeBytes = 96 * 1024;    // buddy usable rounds DOWN to 64KiB (256<<8)
    cfg.maxChunkSizeBytes = 96 * 1024; // exceeds the 64KiB usable -> must be clamped to 64KiB
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.maxInflightChunksPerRequest = 2;
    cfg.scatterWorkerCount = 2;
    // 4 x 20KiB = 80KiB total > 64KiB usable. Unclamped, the planner packs all 80KiB into ONE chunk
    // (<= 96KiB cap) that can never be allocated (rounds to 128KiB > 64KiB usable) -> hang. Clamped to
    // 64KiB, it splits into chunks that each fit a drained arena and recycle through.
    runTransfer("btClamp", /*nDescs=*/4, /*descBytes=*/20480, cfg, cfg, /*seed=*/9);
}

// Sender-side arena backpressure: the receiver's arena and in-flight limit are generous (it grants
// every credit up front) but the SENDER's arena only fits a few concurrent gather regions, so most
// credits get parked in pendingCredits and drain via drainPendingPosts as ACKs free regions. The
// transfer must still complete byte-exact (parked != dropped). This is also the path the
// `arenaStarved` NVTX span instruments.
TEST(BounceTransport, SenderArenaBackpressureParksCredits)
{
    b::BounceConfig small; // sender: 64KiB usable -> at most 4 in-flight 16KiB gather regions
    small.maxChunkSizeBytes = 16 * 1024;
    small.arenaAllocationGranularityBytes = 256;
    small.maxInflightChunksPerRequest = 8;
    small.scatterWorkerCount = 2;
    small.arenaSizeBytes = 64 * 1024;
    b::BounceConfig big = small; // receiver: room to grant all eight allowed credits at once
    big.arenaSizeBytes = 1ULL << 20;
    // 32 x 4KiB = 128KiB in ~8 chunks of 16KiB: double the sender's usable arena, so at least half
    // the granted credits must park and retry.
    runTransfer("btPark", /*nDescs=*/32, /*descBytes=*/4096, small, big, /*seed=*/11);
}

// Mixed descriptor sizes in one request, so both gather/scatter copy paths run on each side: chunks
// whose descs all fit in one copy entry (the bulk "no split" path) and chunks holding a desc above
// b::kCopySplitBytes, which is copied in kCopySplitBytes pieces.
TEST(BounceTransport, MixedSizeDescsAreByteExact)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr std::size_t kMaxChunkBytes = 512U << 10;
    // 1 B and 31 B sit below the 32 B bounce alignment.
    constexpr std::uint32_t kOneByteDesc = 1;
    constexpr std::uint32_t kBelowAlignmentDesc = 31;
    constexpr std::uint32_t kPageDesc = 4096;
    constexpr std::uint32_t kLargestUnsplitDesc = b::kCopySplitBytes;
    constexpr std::uint32_t kSmallestSplitDesc = b::kCopySplitBytes + 1;
    constexpr std::uint32_t kMultiPieceDesc = 300U << 10;
    constexpr std::uint32_t kMergedRunDescBytes = 2U << 10;
    constexpr std::uint32_t kMergedRunDescs = 48;
    constexpr std::uint32_t kUnsplitGroups = 10;
    constexpr std::size_t kUnsplitGroupBytes = kOneByteDesc + kBelowAlignmentDesc + kPageDesc + kLargestUnsplitDesc;
    static_assert(kUnsplitGroups * kUnsplitGroupBytes > kMaxChunkBytes, "the unsplit groups alone fill a chunk");
    static_assert(kMultiPieceDesc > 2 * b::kCopySplitBytes && kMultiPieceDesc <= kMaxChunkBytes,
        "a desc of several copy pieces that still fits one chunk");
    static_assert(
        kMergedRunDescBytes <= b::kCopySplitBytes && kMergedRunDescs * kMergedRunDescBytes > b::kCopySplitBytes,
        "the run's descs need splitting only once merged");

    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = kMaxChunkBytes;
    cfg.maxInflightChunksPerRequest = 4;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 4096;
    cfg.arenaSizeBytes = 8ULL << 20;
    auto nodes = bounce_test::makeWiredPair("btMixed", cfg);
    if (!nodes)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    Node& sender = *nodes->sender;
    Node& receiver = *nodes->receiver;

    std::vector<std::uint32_t> sizes;
    std::vector<bool> contiguousWithPrev;
    auto add = [&](std::uint32_t len)
    {
        sizes.push_back(len);
        contiguousWithPrev.push_back(false);
    };
    auto addContiguousRun = [&](std::uint32_t count, std::uint32_t len)
    {
        for (std::uint32_t i = 0; i < count; ++i)
        {
            sizes.push_back(len);
            contiguousWithPrev.push_back(i > 0);
        }
    };
    auto addSmallDescs = [&]
    {
        add(kOneByteDesc);
        add(kBelowAlignmentDesc);
        add(kPageDesc);
    };
    for (std::uint32_t g = 0; g < kUnsplitGroups; ++g)
    {
        addSmallDescs();
        add(kLargestUnsplitDesc);
    }
    add(kSmallestSplitDesc);
    add(kMultiPieceDesc);
    addContiguousRun(kMergedRunDescs, kMergedRunDescBytes);
    addSmallDescs();
    auto bufs = bounce_test::makeXferBufsSized(sizes, /*seed=*/51, contiguousWithPrev);

    auto fut = sender.tx->submit(bufs.srcDescs, bufs.dstDescs, receiver.name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    sender.tx->shutdown();
    receiver.tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// Several multi-chunk requests submitted at once from different threads to one receiver, with more
// chunks allowed in flight than the sender has exec contexts: GRANTs race the submits, and exec
// contexts are reused across requests. Every request must land byte-exact.
TEST(BounceTransport, ConcurrentMultiChunkSubmitsShareExecContexts)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = 4096;
    cfg.maxInflightChunksPerRequest = 4;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.arenaSizeBytes = 1ULL << 20;
    auto nodes = bounce_test::makeWiredPair("btConcMulti", cfg);
    if (!nodes)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    Node& sender = *nodes->sender;
    Node& receiver = *nodes->receiver;

    constexpr std::uint32_t kRequests = 4;
    constexpr std::uint32_t kDescsPerRequest = 40;
    constexpr std::uint32_t kDescBytes = 700;
    constexpr std::uint32_t kFirstSeed = 60;
    ASSERT_LT(sender.exec->size(), kRequests * cfg.maxInflightChunksPerRequest)
        << "the in-flight chunks must outnumber the sender's exec contexts";
    std::vector<bounce_test::XferBufs> bufs;
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        bufs.push_back(bounce_test::makeXferBufs(kDescsPerRequest, kDescBytes, kFirstSeed + r));
    }
    ASSERT_GE(plannedChunkCount(bufs.front(), cfg), cfg.maxInflightChunksPerRequest)
        << "each request must have enough chunks to fill its in-flight limit";

    std::vector<std::shared_future<b::BounceResult>> futs(kRequests);
    std::vector<std::thread> submitters;
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        submitters.emplace_back(
            [&, r] { futs[r] = sender.tx->submit(bufs[r].srcDescs, bufs[r].dstDescs, receiver.name); });
    }
    for (auto& t : submitters)
    {
        t.join();
    }
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        ASSERT_EQ(futs[r].wait_for(std::chrono::seconds(30)), std::future_status::ready) << "request " << r << " hung";
        EXPECT_EQ(futs[r].get().state, kvc::TransferState::kSUCCESS) << "request " << r;
        EXPECT_TRUE(bounce_test::verifyXferBufs(bufs[r])) << "byte mismatch, request " << r;
    }

    sender.tx->shutdown();
    receiver.tx->shutdown();
    for (auto& x : bufs)
    {
        bounce_test::freeXferBufs(x);
    }
}

// Many tiny requests at once keep the control queues full (WANTs on the receiver; GRANTs and ACKs on
// the sender) while the shared arena and exec contexts are recycled across requests. Every request
// must still complete byte-exact.
TEST(BounceTransport, ManyTinyConcurrentRequestsAllComplete)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = 4096;
    cfg.maxInflightChunksPerRequest = 2;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.arenaSizeBytes = 1ULL << 20;
    auto nodes = bounce_test::makeWiredPair("btTiny", cfg);
    if (!nodes)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    Node& sender = *nodes->sender;
    Node& receiver = *nodes->receiver;

    constexpr std::uint32_t kRequests = 200;
    ASSERT_GT(kRequests, sender.exec->size()) << "the requests must outnumber the sender's exec contexts";
    std::vector<bounce_test::XferBufs> bufs;
    bufs.reserve(kRequests);
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        bufs.push_back(bounce_test::makeXferBufs(/*nDescs=*/2, /*descBytes=*/100, /*seed=*/r));
    }
    std::vector<std::shared_future<b::BounceResult>> futs;
    futs.reserve(kRequests);
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        futs.push_back(sender.tx->submit(bufs[r].srcDescs, bufs[r].dstDescs, receiver.name));
    }
    for (std::uint32_t r = 0; r < kRequests; ++r)
    {
        ASSERT_EQ(futs[r].wait_for(std::chrono::seconds(60)), std::future_status::ready) << "request " << r << " hung";
        EXPECT_EQ(futs[r].get().state, kvc::TransferState::kSUCCESS) << "request " << r;
        EXPECT_TRUE(bounce_test::verifyXferBufs(bufs[r])) << "byte mismatch, request " << r;
    }

    sender.tx->shutdown();
    receiver.tx->shutdown();
    for (auto& x : bufs)
    {
        bounce_test::freeXferBufs(x);
    }
}
