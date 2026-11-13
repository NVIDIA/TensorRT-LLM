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

#include "bounceGateKernel.h"
#include "bounceTestNixlNode.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;
namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
// One end-to-end transfer of `nDescs` x `descBytes` through the bounce pipeline between two real
// NIXL nodes: a sender built from `senderCfg` and a receiver built from `receiverCfg` (asymmetric
// configs let a test pin clamping/backpressure to one side). `tag` gives the two agents unique names.
void runTransfer(std::string const& tag, std::uint32_t nDescs, std::uint32_t descBytes,
    b::BounceConfig const& senderCfg, b::BounceConfig const& receiverCfg, std::uint32_t seed)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    std::size_t const maxDescs
        = std::max<std::size_t>(1024ULL, std::max(senderCfg.maxChunkSizeBytes, receiverCfg.maxChunkSizeBytes) / 256ULL);

    auto A = bounce_test::makeNode(tag + "A", senderCfg, maxDescs);
    auto B = bounce_test::makeNode(tag + "B", receiverCfg, maxDescs);
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
    std::uint32_t maxInflightChunksPerRequest)
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
    runTransfer(tag, nDescs, descBytes, cfg, cfg, /*seed=*/1);
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

namespace
{
// Two nodes (A sends, B receives) sized like the many-small-descriptor KV pattern, plus a
// non-blocking stream standing in for the receiver's forward stream.
struct OrderFixture
{
    std::unique_ptr<bounce_test::Node> A;
    std::unique_ptr<bounce_test::Node> B;
    cudaStream_t stream{nullptr};

    ~OrderFixture()
    {
        if (A)
        {
            A->tx->shutdown();
        }
        if (B)
        {
            B->tx->shutdown();
        }
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }
};

constexpr std::uint32_t kOrderDescs = 1200; // above the default min_descriptor_count admission gate
constexpr std::uint32_t kOrderDescBytes = 256;

b::BounceConfig orderTestConfig(bool scatterOnOrderStream, bool copyEngine)
{
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = 64 << 10;
    cfg.maxInflightChunksPerRequest = 4;
    cfg.scatterWorkerCount = 2;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.arenaSizeBytes = 4ULL << 20;
    cfg.scatterOnOrderStream = scatterOnOrderStream;
    cfg.scatterCopyEngine = copyEngine;
    return cfg;
}

// Returns false (after GTEST_SKIP) when CUDA or NIXL is unavailable.
// copyEngine defaults to off: the order-stream tests below exercise the kernel scatter paths.
bool makeOrderFixture(OrderFixture& f, std::string const& tag, bool scatterOnOrderStream, bool copyEngine = false)
{
    if (!bounce_test::hasCuda())
    {
        return false;
    }
    auto const cfg = orderTestConfig(scatterOnOrderStream, copyEngine);
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, cfg.maxChunkSizeBytes / 256ULL);
    f.A = bounce_test::makeNode(tag + "A", cfg, maxDescs);
    f.B = bounce_test::makeNode(tag + "B", cfg, maxDescs);
    if (!f.A || !f.B)
    {
        return false;
    }
    bounce_test::wirePair(*f.A, *f.B);
    // One untimed transfer first, so lazy transport setup (connections, registration caches) is not
    // part of the timed window below.
    auto warm = bounce_test::makeXferBufs(64, kOrderDescBytes, /*seed=*/3);
    auto warmFut = f.A->tx->submit(warm.srcDescs, warm.dstDescs, f.B->name);
    bool const warmOk = warmFut.wait_for(std::chrono::seconds(30)) == std::future_status::ready
        && warmFut.get().state == kvc::TransferState::kSUCCESS;
    bounce_test::freeXferBufs(warm);
    EXPECT_TRUE(warmOk) << "warm-up transfer failed";
    return warmOk && cudaStreamCreateWithFlags(&f.stream, cudaStreamNonBlocking) == cudaSuccess;
}

// True when no byte of the destination was written yet (makeXferBufs zero-fills it).
bool dstUntouched(bounce_test::XferBufs const& x)
{
    std::vector<unsigned char> got(x.total, 0xEE);
    if (cudaMemcpy(got.data(), x.dst, x.total, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    for (unsigned char c : got)
    {
        if (c != 0)
        {
            return false;
        }
    }
    return true;
}

} // namespace

// scatter_copy_engine without a registered order stream: the receive scatter runs on the copy engines (counted as
// copy-engine scatters, none ordered, none on the private kernel path) and the transfer is byte-exact.
TEST(BounceTransport, ScatterCopyEngineWithoutOrderStream)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btCe", /*scatterOnOrderStream=*/true, /*copyEngine=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    auto const ceBefore = f.B->tx->copyEngineScatterCount();
    auto const privateBefore = f.B->tx->privateScatterCount();
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/13);
    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_GT(f.B->tx->copyEngineScatterCount(), ceBefore);
    EXPECT_EQ(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_EQ(f.B->tx->privateScatterCount(), privateBefore);
    bounce_test::freeXferBufs(bufs);
}

// With an order stream registered, the order stream wins over the copy engines: scatters stay ordered.
TEST(BounceTransport, ScatterCopyEngineYieldsToOrderStream)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btCeOrd", /*scatterOnOrderStream=*/true, /*copyEngine=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(f.stream));
    auto const ceBefore = f.B->tx->copyEngineScatterCount();
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/17);
    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_GT(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_EQ(f.B->tx->copyEngineScatterCount(), ceBefore);
    bounce_test::freeXferBufs(bufs);
}

// The receive scatter is enqueued on the registered order stream: while a kernel occupying every SM
// holds that stream (a forward in flight), no chunk is copied into the destination and no ACK is
// sent; once it finishes, the transfer completes byte-exact.
TEST(BounceTransport, ScatterOrderedOnOrderStream)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btOrd", /*scatterOnOrderStream=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(f.stream));
    auto const privateBefore = f.B->tx->privateScatterCount(); // the warm-up transfer ran before the stream was set
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/11);
    bounce_test::GateKernel gate;
    ASSERT_EQ(gate.launch(f.stream), cudaSuccess);

    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout)
        << "transfer completed while the order stream was still busy";
    EXPECT_TRUE(gate.running());
    EXPECT_TRUE(dstUntouched(bufs)) << "scatter wrote the destination while the order stream was busy";

    gate.release();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_GT(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_EQ(f.B->tx->privateScatterCount(), privateBefore);
    bounce_test::freeXferBufs(bufs);
}

// scatter_on_order_stream=0: the stream is refused and scatters keep the private copy streams, so
// the transfer completes while the "forward" still holds the stream (the overlap the knob removes).
TEST(BounceTransport, ScatterOrderStreamKillSwitch)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btOrdOff", /*scatterOnOrderStream=*/false))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    EXPECT_FALSE(f.B->tx->setScatterOrderStream(f.stream));
    auto const privateBefore = f.B->tx->privateScatterCount();
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/12);
    bounce_test::GateKernel gate;
    ASSERT_EQ(gate.launch(f.stream), cudaSuccess);

    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_TRUE(gate.running()) << "the scatter should not have waited for the order stream";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_EQ(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_GT(f.B->tx->privateScatterCount(), privateBefore);
    gate.release();
    bounce_test::freeXferBufs(bufs);
}

// Clearing the order stream (nullptr) restores the private copy streams.
TEST(BounceTransport, ScatterOrderStreamClearRestoresPrivateStreams)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btOrdClr", /*scatterOnOrderStream=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(f.stream));
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(nullptr));
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/13);
    bounce_test::GateKernel gate;
    ASSERT_EQ(gate.launch(f.stream), cudaSuccess);

    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "transfer hung";
    EXPECT_TRUE(gate.running());
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_EQ(f.B->tx->orderedScatterCount(), 0u);
    gate.release();
    bounce_test::freeXferBufs(bufs);
}

// While the order stream is being captured, scatters fall back to the private copy streams: the
// captured graph holds only the capturing thread's own work and the data is exact.
TEST(BounceTransport, ScatterOrderStreamCaptureFallback)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btOrdCap", /*scatterOnOrderStream=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(f.stream));
    auto const privateBefore = f.B->tx->privateScatterCount();
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/14);
    void* scratch = nullptr;
    ASSERT_EQ(cudaMalloc(&scratch, 256), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(scratch, 0, 256, f.stream), cudaSuccess); // the test's one captured node
    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    auto const status = fut.wait_for(std::chrono::seconds(30));
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(f.stream, &graph), cudaSuccess);
    ASSERT_EQ(status, std::future_status::ready) << "transfer hung";
    std::size_t numNodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &numNodes), cudaSuccess);
    EXPECT_EQ(numNodes, 1u) << "a scatter was captured into the graph";
    cudaGraphDestroy(graph);

    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_EQ(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_GT(f.B->tx->privateScatterCount(), privateBefore);
    cudaFree(scratch);
    bounce_test::freeXferBufs(bufs);
}

// scatter_copy_engine with an order stream that is capturing at launch: the scatters take the copy engines
// (none on the private kernel path, which could wait for SMs a persistent forward grid holds), nothing is
// captured into the graph and the data is exact.
TEST(BounceTransport, ScatterCopyEngineCoversCapturingOrderStream)
{
    OrderFixture f;
    if (!makeOrderFixture(f, "btCeCap", /*scatterOnOrderStream=*/true, /*copyEngine=*/true))
    {
        GTEST_SKIP() << "CUDA or NIXL agent/backend unavailable";
    }
    ASSERT_TRUE(f.B->tx->setScatterOrderStream(f.stream));
    auto const ceBefore = f.B->tx->copyEngineScatterCount();
    auto const privateBefore = f.B->tx->privateScatterCount();
    auto bufs = bounce_test::makeXferBufs(kOrderDescs, kOrderDescBytes, /*seed=*/15);
    void* scratch = nullptr;
    ASSERT_EQ(cudaMalloc(&scratch, 256), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(f.stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(scratch, 0, 256, f.stream), cudaSuccess); // the test's one captured node
    auto fut = f.A->tx->submit(bufs.srcDescs, bufs.dstDescs, f.B->name);
    auto const status = fut.wait_for(std::chrono::seconds(30));
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(f.stream, &graph), cudaSuccess);
    ASSERT_EQ(status, std::future_status::ready) << "transfer hung";
    std::size_t numNodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &numNodes), cudaSuccess);
    EXPECT_EQ(numNodes, 1u) << "a scatter was captured into the graph";
    cudaGraphDestroy(graph);

    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(bufs)) << "byte mismatch";
    EXPECT_GT(f.B->tx->copyEngineScatterCount(), ceBefore);
    EXPECT_EQ(f.B->tx->orderedScatterCount(), 0u);
    EXPECT_EQ(f.B->tx->privateScatterCount(), privateBefore);
    cudaFree(scratch);
    bounce_test::freeXferBufs(bufs);
}
