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
// byte arrives. Sizing is chosen so the chunk count exceeds the per-request in-flight limit,
// forcing credit recycling. Skips if no CUDA device or the NIXL backend cannot initialize.

#include "bounceTestNixlNode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;
namespace b = tensorrt_llm::executor::kv_cache::bounce;

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
    std::size_t const maxDescs
        = std::max<std::size_t>(1024ULL, std::max(senderCfg.maxChunkSizeBytes, receiverCfg.maxChunkSizeBytes) / 256ULL);

    auto A = bounce_test::makeNode(tag + "A", senderCfg, maxDescs, nullptr, backendParams);
    auto B = bounce_test::makeNode(tag + "B", receiverCfg, maxDescs, nullptr, backendParams);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    auto bufs = bounce_test::makeXferBufs(nDescs, descBytes, seed);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// Thin wrapper: the same config on both ends, with maxChunkSizeBytes/maxInflightChunksPerRequest
// chosen so the chunk count exceeds the in-flight limit (forcing credit recycling).
void runTransfer(std::string const& tag, std::uint32_t nDescs, std::uint32_t descBytes, std::size_t maxChunkSizeBytes,
    std::uint32_t maxInflightChunksPerRequest,
    bounce_test::BackendParams const& backendParams = bounce_test::dedicatedWorkerBackendParams())
{
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = maxChunkSizeBytes;
    cfg.maxInflightChunksPerRequest = maxInflightChunksPerRequest;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 256; // matches the 256-aligned desc layout
    // Arena large enough for the configured in-flight chunk count in both roles, with headroom for
    // buddy rounding. A power-of-two size lets BuddyAllocator use the whole allocation.
    std::size_t arenaSizeBytes = 1ULL << 20;
    while (arenaSizeBytes < static_cast<std::size_t>(maxInflightChunksPerRequest) * maxChunkSizeBytes * 4ULL)
    {
        arenaSizeBytes <<= 1;
    }
    cfg.arenaSizeBytes = arenaSizeBytes;
    runTransfer(tag, nDescs, descBytes, cfg, cfg, /*seed=*/1, backendParams);
}
} // namespace

TEST(BounceTransport, SmallTransferFitsInflightLimit)
{
    // Four descriptors fit within the configured in-flight chunk limit.
    runTransfer("btSmall", /*nDescs=*/4, /*descBytes=*/512, /*maxChunkSizeBytes=*/8192, /*maxInflightChunks=*/4);
}

TEST(BounceTransport, LargeTransferRecyclesCredits)
{
    // Forty descriptors produce more chunks than the limit of two, forcing credit recycling.
    runTransfer("btLarge", /*nDescs=*/40, /*descBytes=*/700, /*maxChunkSizeBytes=*/4096, /*maxInflightChunks=*/2);
}

TEST(BounceTransport, LargeTransferOnSharedNixlWorker)
{
    // Plugin-default backend params (no num_threads, no split_batch_size): every chunk write runs on
    // NIXL's shared UCX worker instead of a dedicated worker thread.
    runTransfer("btShared", /*nDescs=*/40, /*descBytes=*/700, /*maxChunkSizeBytes=*/4096, /*maxInflightChunks=*/2,
        /*backendParams=*/{});
}

TEST(BounceTransport, ManySmallDescs)
{
    // Closer to the real KV pattern: many tiny descs.
    runTransfer("btMany", /*nDescs=*/500, /*descBytes=*/256, /*maxChunkSizeBytes=*/16384, /*maxInflightChunks=*/4);
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
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);

    auto A = bounce_test::makeNode("btConcA", cfg, maxDescs);
    auto B = bounce_test::makeNode("btConcB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    // Seed-distinct payloads so any cross-talk between the two in-flight flows fails verification.
    auto bufs1 = bounce_test::makeXferBufs(/*nDescs=*/8, /*descBytes=*/1024, /*seed=*/1);
    auto bufs2 = bounce_test::makeXferBufs(/*nDescs=*/8, /*descBytes=*/1024, /*seed=*/2);
    auto fut1 = A->tx->submit(bufs1.srcDescs, bufs1.dstDescs, B->name);
    auto fut2 = A->tx->submit(bufs2.srcDescs, bufs2.dstDescs, B->name);
    ASSERT_EQ(fut1.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "first transfer hung";
    ASSERT_EQ(fut2.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "second transfer hung";
    EXPECT_EQ(fut1.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_EQ(fut2.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs1)) << "byte mismatch (first flow)";
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs2)) << "byte mismatch (second flow)";

    A->tx->shutdown();
    B->tx->shutdown();
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
    auto node = bounce_test::makeNode("btHandshake", cfg, /*maxDescs=*/1024);
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

// Sender-side arena backpressure: the receiver's arena and in-flight limit are generous
// credit up front) but the SENDER's arena only fits a few concurrent gather regions, so most
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

// Mixed descriptor sizes in one request, so both gather/scatter plan paths run on each side: chunks
// whose descs are all <= 64 KiB (the bulk "no split" path) and chunks with a larger one — a 64 KiB+1
// desc, a 300 KiB desc, and a contiguous run of 2 KiB descs that the planner merges into one 96 KiB
// desc — which are split into 64 KiB copy pieces. Odd sizes (1 B, 31 B) exercise the 32 B alignment.
TEST(BounceTransport, MixedSizeDescsAreByteExact)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = 512 * 1024;
    cfg.maxInflightChunksPerRequest = 4;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 4096;
    cfg.arenaSizeBytes = 8ULL << 20;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);
    auto A = bounce_test::makeNode("btMixedA", cfg, maxDescs);
    auto B = bounce_test::makeNode("btMixedB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    std::vector<std::uint32_t> sizes;
    std::vector<bool> contiguous;
    auto add = [&](std::uint32_t len, bool contiguousWithPrev = false)
    {
        sizes.push_back(len);
        contiguous.push_back(contiguousWithPrev);
    };
    for (int g = 0; g < 10; ++g) // ~680 KiB of descs <= 64 KiB: fills a whole chunk on the no-split path
    {
        add(1);
        add(31);
        add(4096);
        add(64 * 1024);
    }
    add(64 * 1024 + 1);
    add(300 * 1024);
    for (int i = 0; i < 48; ++i) // contiguous in src and dst: merged into one 96 KiB plan desc
    {
        add(2048, /*contiguousWithPrev=*/i > 0);
    }
    add(1);
    add(31);
    add(4096);
    auto bufs = bounce_test::makeXferBufsSized(sizes, contiguous, /*seed=*/51);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// Several multi-chunk requests submitted at once from different threads to one receiver, with more
// chunks allowed in flight (4 per request x 4 requests) than the sender has exec contexts (8): GRANTs
// race the submits, and exec contexts are reused across requests. Every request must land byte-exact.
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
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);
    auto A = bounce_test::makeNode("btConcMultiA", cfg, maxDescs); // exec pool: maxInflight + 4 = 8 contexts
    auto B = bounce_test::makeNode("btConcMultiB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    constexpr int kRequests = 4;
    std::vector<bounce_test::XferBufs> bufs;
    for (int r = 0; r < kRequests; ++r)
    {
        // 40 x 700 B -> 11 chunks of <= 4 KiB each.
        bufs.push_back(bounce_test::makeXferBufs(/*nDescs=*/40, /*descBytes=*/700, /*seed=*/60 + r));
    }
    std::vector<std::shared_future<b::BounceResult>> futs(kRequests);
    std::vector<std::thread> submitters;
    for (int r = 0; r < kRequests; ++r)
    {
        submitters.emplace_back([&, r] { futs[r] = A->tx->submit(bufs[r].srcDescs, bufs[r].dstDescs, B->name); });
    }
    for (auto& t : submitters)
    {
        t.join();
    }
    for (int r = 0; r < kRequests; ++r)
    {
        ASSERT_EQ(futs[r].wait_for(std::chrono::seconds(30)), std::future_status::ready) << "request " << r << " hung";
        EXPECT_EQ(futs[r].get().state, kvc::TransferState::kSUCCESS) << "request " << r;
        EXPECT_TRUE(bounce_test::verifyXferBufs(bufs[r])) << "byte mismatch, request " << r;
    }

    A->tx->shutdown();
    B->tx->shutdown();
    for (auto& x : bufs)
    {
        bounce_test::freeXferBufs(x);
    }
}

// Many tiny requests at once keep the control queues full (WANTs on the receiver; GRANTs and ACKs on
// the sender) while the shared arena and exec contexts are recycled across requests. Every request must
// still complete byte-exact.
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
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);
    auto A = bounce_test::makeNode("btTinyA", cfg, maxDescs);
    auto B = bounce_test::makeNode("btTinyB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*A, *B);

    constexpr int kRequests = 200;
    std::vector<bounce_test::XferBufs> bufs;
    bufs.reserve(kRequests);
    for (int r = 0; r < kRequests; ++r)
    {
        bufs.push_back(
            bounce_test::makeXferBufs(/*nDescs=*/2, /*descBytes=*/100, /*seed=*/static_cast<std::uint32_t>(r)));
    }
    std::vector<std::shared_future<b::BounceResult>> futs;
    futs.reserve(kRequests);
    for (int r = 0; r < kRequests; ++r)
    {
        futs.push_back(A->tx->submit(bufs[r].srcDescs, bufs[r].dstDescs, B->name));
    }
    int ok = 0;
    for (int r = 0; r < kRequests; ++r)
    {
        ASSERT_EQ(futs[r].wait_for(std::chrono::seconds(60)), std::future_status::ready) << "request " << r << " hung";
        if (futs[r].get().state == kvc::TransferState::kSUCCESS && bounce_test::verifyXferBufs(bufs[r]))
        {
            ++ok;
        }
    }
    EXPECT_EQ(ok, kRequests);

    A->tx->shutdown();
    B->tx->shutdown();
    for (auto& x : bufs)
    {
        bounce_test::freeXferBufs(x);
    }
}
