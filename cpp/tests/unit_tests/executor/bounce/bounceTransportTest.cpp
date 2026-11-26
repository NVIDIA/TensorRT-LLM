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

#include "tensorrt_llm/runtime/cudaStream.h"

#include <gtest/gtest.h>

#include <cuda.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

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
// A device-side wait keeps legacy work pending without occupying an SM. Release before joining
// transport workers or freeing CUDA allocations, including when an ASSERT returns from the test.
class LegacyDefaultGate
{
public:
    LegacyDefaultGate() = default;
    LegacyDefaultGate(LegacyDefaultGate const&) = delete;
    LegacyDefaultGate& operator=(LegacyDefaultGate const&) = delete;

    void hold()
    {
        ASSERT_EQ(cudaHostAlloc(&mFlag, sizeof(*mFlag), cudaHostAllocMapped), cudaSuccess);
        *mFlag = 0;
        void* deviceFlag = nullptr;
        ASSERT_EQ(cudaHostGetDevicePointer(&deviceFlag, mFlag, 0), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&mDone, cudaEventDisableTiming), cudaSuccess);
        // Materialize the event before the gate so first-record setup cannot synchronize the hold.
        ASSERT_EQ(cudaEventRecord(mDone, cudaStreamLegacy), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(mDone), cudaSuccess);
        mWatchdog = std::thread(
            [this]
            {
                std::unique_lock<std::mutex> lock(mMutex);
                if (!mReleased.wait_for(lock, std::chrono::seconds(120), [this] { return mRelease; }))
                {
                    mWatchdogFired.store(true);
                    *static_cast<std::uint32_t volatile*>(mFlag) = 1;
                }
            });
        ASSERT_EQ(cuStreamWaitValue32(
                      CU_STREAM_LEGACY, reinterpret_cast<CUdeviceptr>(deviceFlag), 1, CU_STREAM_WAIT_VALUE_GEQ),
            CUDA_SUCCESS);
        mHeld = true;
        ASSERT_EQ(cudaEventRecord(mDone, cudaStreamLegacy), cudaSuccess);
        expectPending();
    }

    void expectPending() const
    {
        EXPECT_FALSE(mWatchdogFired.load()) << "cleanup watchdog released unrelated legacy work";
        EXPECT_EQ(cudaEventQuery(mDone), cudaErrorNotReady) << "unrelated legacy work completed";
    }

    ~LegacyDefaultGate()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mFlag)
            {
                *static_cast<std::uint32_t volatile*>(mFlag) = 1;
            }
            mRelease = true;
        }
        mReleased.notify_one();
        if (mWatchdog.joinable())
        {
            mWatchdog.join();
        }
        // The mapped flag must stay alive until the device wait has retired.
        if (mHeld)
        {
            EXPECT_EQ(cudaStreamSynchronize(cudaStreamLegacy), cudaSuccess);
        }
        if (mDone)
        {
            EXPECT_EQ(cudaEventDestroy(mDone), cudaSuccess);
        }
        if (mFlag)
        {
            EXPECT_EQ(cudaFreeHost(mFlag), cudaSuccess);
        }
        EXPECT_FALSE(mWatchdogFired.load()) << "cleanup watchdog released unrelated legacy work";
    }

private:
    std::uint32_t* mFlag{nullptr};
    cudaEvent_t mDone{};
    bool mHeld{false};
    bool mRelease{false};
    std::atomic<bool> mWatchdogFired{false};
    std::mutex mMutex;
    std::condition_variable mReleased;
    std::thread mWatchdog;
};

class BounceStreamDependencies : public ::testing::TestWithParam<std::uint32_t>
{
};
} // namespace

TEST_P(BounceStreamDependencies, RoundTripSettlesWhileUnrelatedLegacyWorkIsPending)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    b::BounceConfig cfg;
    cfg.arenaSizeBytes = 1ULL << 20;
    cfg.maxChunkSizeBytes = 4096;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.maxInflightChunksPerRequest = 2;
    cfg.scatterWorkerCount = 1;
    auto const freeBuffers = [](bounce_test::XferBufs* buffers)
    {
        bounce_test::freeXferBufs(*buffers);
        delete buffers;
    };
    using Buffers = std::unique_ptr<bounce_test::XferBufs, decltype(freeBuffers)>;
    // Separate buffers and payloads ensure the warmup cannot satisfy the gated transfer's check.
    Buffers warmup(new bounce_test::XferBufs(bounce_test::makeXferBufs(4, GetParam(), 1)), freeBuffers);
    Buffers payload(new bounce_test::XferBufs(bounce_test::makeXferBufs(4, GetParam(), 2)), freeBuffers);
    std::unique_ptr<void, decltype(&cudaFreeHost)> readback(nullptr, cudaFreeHost);
    void* pinned = nullptr;
    ASSERT_EQ(cudaHostAlloc(&pinned, payload->total, cudaHostAllocDefault), cudaSuccess);
    readback.reset(pinned);
    tensorrt_llm::runtime::CudaStream reader(cudaStreamNonBlocking);

    // Destroy the transports before their buffers on every return, including timeout failures.
    auto sender = bounce_test::makeNode("btStreamSender", cfg, 1024);
    auto receiver = bounce_test::makeNode("btStreamReceiver", cfg, 1024);
    // The H100 executor_bounce CI group requires the real NIXL tier; losing its backend must fail.
    ASSERT_NE(sender, nullptr) << "NIXL agent/backend unavailable";
    ASSERT_NE(receiver, nullptr) << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*sender, *receiver);

    auto warmupDone = sender->tx->submit(warmup->srcDescs, warmup->dstDescs, receiver->name);
    ASSERT_EQ(warmupDone.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    ASSERT_EQ(warmupDone.get().state, kvc::TransferState::kSUCCESS);
    ASSERT_EQ(
        cudaMemcpyAsync(readback.get(), warmup->dst, warmup->total, cudaMemcpyDeviceToHost, reader.get()), cudaSuccess);
    reader.synchronize();

    // Declared last: fatal assertions release the gate before other CUDA resources are destroyed.
    LegacyDefaultGate gate;
    gate.hold();
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto done = sender->tx->submit(payload->srcDescs, payload->dstDescs, receiver->name);
    ASSERT_EQ(done.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "bounce transfer hung";
    ASSERT_EQ(done.get().state, kvc::TransferState::kSUCCESS);
    gate.expectPending();
    ASSERT_EQ(cudaMemcpyAsync(readback.get(), payload->dst, payload->total, cudaMemcpyDeviceToHost, reader.get()),
        cudaSuccess);
    reader.synchronize();
    gate.expectPending();
    auto const* actual = static_cast<unsigned char const*>(readback.get());
    for (std::uint32_t fragment = 0; fragment < payload->nDescs; ++fragment)
    {
        for (std::uint32_t byte = 0; byte < payload->descBytes; ++byte)
        {
            ASSERT_EQ(actual[payload->off[fragment] + byte], bounce_test::patSeed(2, fragment, byte))
                << "fragment=" << fragment << " byte=" << byte;
        }
    }
    gate.expectPending();
}

INSTANTIATE_TEST_SUITE_P(AlignedAndUnaligned, BounceStreamDependencies, ::testing::Values(32U, 19U));
