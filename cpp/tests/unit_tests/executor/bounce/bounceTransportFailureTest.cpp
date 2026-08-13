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

// Boundary / failure-path tests for BounceTransport: the reactor must always resolve a request
// (SUCCESS or FAILURE) and never hang — including when the peer never grants, the transfer engine
// fails, or shutdown races in-flight requests. Plus a multi-threaded concurrent-submit test.
// The data plane is real NIXL (via bounceTestNixlNode helpers); the one exception is the
// transfer-engine-failure path, which uses a tiny FailingTransferEngine fault injector because a
// real NIXL engine cannot be made to deterministically fail a write.

#include "bounceTestNixlNode.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/TransferEngine.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{
// Engine whose writes always report failure -> exercises the reactor's transfer-failure path. NIXL
// can't be coerced into a deterministic write failure, so this stays a local fault injector (it is
// NOT a loopback data mover — no data ever flows, so it needs no real agent / RDMA).
class FailingTransferEngine : public b::TransferEngine
{
public:
    bool registerRegion(void*, std::size_t) override
    {
        return true;
    }

    std::uint64_t postWrite(
        std::string const&, void const*, std::uint64_t, std::uint32_t, std::uint32_t, cudaStream_t) override
    {
        return 1; // pretend a write was posted...
    }

    b::XferState poll(std::uint64_t) override
    {
        return b::XferState::kFailed; // ...but it never lands
    }

    bool release(std::uint64_t) override
    {
        return true;
    }
};

// Deterministic engine for protocol and shutdown tests. It can leave a write active and reject its
// first release, modeling a backend that cannot cancel immediately while preserving the handle for
// a later retry.
class ControllableTransferEngine : public b::TransferEngine
{
public:
    bool registerRegion(void*, std::size_t) override
    {
        return true;
    }

    std::uint64_t postWrite(
        std::string const&, void const*, std::uint64_t, std::uint32_t, std::uint32_t, cudaStream_t) override
    {
        auto const handle = nextHandle.fetch_add(1, std::memory_order_relaxed);
        outstandingHandle.store(handle, std::memory_order_release);
        postCount.fetch_add(1, std::memory_order_relaxed);
        return handle;
    }

    b::XferState poll(std::uint64_t) override
    {
        return allowTerminal.load(std::memory_order_acquire) ? b::XferState::kDone : b::XferState::kInProgress;
    }

    bool release(std::uint64_t handle) override
    {
        releaseCount.fetch_add(1, std::memory_order_relaxed);
        if (!releaseSucceeds.load(std::memory_order_acquire))
        {
            return false;
        }
        std::uint64_t expected = handle;
        return outstandingHandle.compare_exchange_strong(expected, 0, std::memory_order_acq_rel) || expected == 0;
    }

    std::atomic<bool> allowTerminal{true};
    std::atomic<bool> releaseSucceeds{true};
    std::atomic<std::uint64_t> postCount{0};
    std::atomic<std::uint64_t> releaseCount{0};
    std::atomic<std::uint64_t> outstandingHandle{0};

private:
    std::atomic<std::uint64_t> nextHandle{1};
};

b::BounceConfig cfg(int timeoutMs)
{
    b::BounceConfig c;
    c.maxChunkSizeBytes = 4096;
    c.maxInflightChunksPerRequest = 2;
    c.scatterWorkerCount = 2;
    c.arenaAllocationGranularityBytes = 256;
    c.arenaSizeBytes = 1ULL << 20;
    c.requestTimeoutMs = timeoutMs;
    return c;
}

// The shared data arena + exec contexts for one manually-built transport (used only by the
// FailingTransferEngine test, which doesn't use a NIXL node). Kept alive by the caller.
struct Backend
{
    std::unique_ptr<b::BounceArena> arena;
    std::unique_ptr<b::ExecPool> exec;
};

// Arena holds exactly `regionCap` max-size regions. Matching the allocation granularity to the
// maximum chunk size makes every region one buddy block.
Backend makeBackend(b::BounceConfig& c, std::uint32_t regionCap)
{
    c.arenaAllocationGranularityBytes = c.maxChunkSizeBytes;
    std::size_t arenaSizeBytes = c.arenaAllocationGranularityBytes;
    while (arenaSizeBytes < static_cast<std::size_t>(regionCap) * c.maxChunkSizeBytes)
    {
        arenaSizeBytes <<= 1;
    }
    c.arenaSizeBytes = arenaSizeBytes;
    std::uint32_t const execCount = regionCap + c.scatterWorkerCount + 4;
    return Backend{std::make_unique<b::BounceArena>(arenaSizeBytes, 0, /*allowFabric=*/false),
        std::make_unique<b::ExecPool>(execCount, 1024, 0, c.useZeroCopyArguments, c.useCubCopy)};
}
} // namespace

// Peer never grants (not added to the channel) -> requestTimeoutMs must fail the request.
TEST(BounceTransportFailure, NoGrantTimesOutNotHang)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/500);
    auto t = bounce_test::makeNode("ngSolo", c, 1024);
    if (!t)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "nobody"); // no peer "nobody" -> WANT dropped
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "request hung with no grant";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    t->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// The transfer engine reports failure -> the request must FAIL (not hang, not falsely succeed).
// Uses the FailingTransferEngine fault injector (no NIXL agent: no data ever lands).
TEST(BounceTransportFailure, EngineFailureFailsRequest)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    b::ZmqControlChannel chA("feA");
    b::ZmqControlChannel chB("feB");
    FailingTransferEngine engA; // sender's writes fail
    FailingTransferEngine engB; // receiver never writes; engine is unused on its side
    auto beA = makeBackend(c, c.maxInflightChunksPerRequest);
    auto beB = makeBackend(c, c.maxInflightChunksPerRequest);
    auto A = std::make_unique<b::BounceTransport>("feA", c, 0, &chA, &engA, beA.arena.get(), beA.exec.get());
    auto B = std::make_unique<b::BounceTransport>("feB", c, 0, &chB, &engB, beB.arena.get(), beB.exec.get());
    A->addPeer("feB", chB.localEndpoint());
    B->addPeer("feA", chA.localEndpoint());

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = A->submit(bufs.srcDescs, bufs.dstDescs, "feB");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "engine failure hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    A->shutdown();
    B->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// DATA has no retransmission path in this protocol. Receiving it twice for one granted region must
// therefore produce one scatter and one ACK, rather than adding replay state for an impossible
// normal-flow event.
TEST(BounceTransportFailure, DuplicateDataProducesOneScatterAndAck)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    b::ZmqControlChannel sender("dupDataSender");
    b::ZmqControlChannel receiverChannel("dupDataReceiver");
    FailingTransferEngine receiverEngine;
    auto receiverBackend = makeBackend(c, c.maxInflightChunksPerRequest);
    auto receiver = std::make_unique<b::BounceTransport>("dupDataReceiver", c, 0, &receiverChannel, &receiverEngine,
        receiverBackend.arena.get(), receiverBackend.exec.get());
    ASSERT_TRUE(sender.addPeer("dupDataReceiver", receiverChannel.localEndpoint()));

    // Hold every context so both DATA messages reach the reactor before the first scatter can finish.
    std::vector<b::ExecCtx*> heldExecContexts;
    while (auto* ctx = receiverBackend.exec->tryAcquire())
    {
        heldExecContexts.push_back(ctx);
    }
    ASSERT_EQ(heldExecContexts.size(), receiverBackend.exec->size());

    constexpr std::uint64_t rid = 17;
    sender.sendTo("dupDataReceiver", b::encodeWant(rid, {256}, sender.localEndpoint()));

    b::BounceMsgHeader grantHeader{};
    std::vector<b::BounceCreditEntry> credits;
    auto const grantDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < grantDeadline && credits.empty())
    {
        std::string peer;
        std::string blob;
        if (sender.recv(peer, blob, 20) && b::decodeHeader(blob, grantHeader)
            && static_cast<b::BounceMsgType>(grantHeader.msgType) == b::BounceMsgType::kGRANT)
        {
            EXPECT_EQ(peer, "dupDataReceiver");
            EXPECT_TRUE(b::decodeCredits(blob, grantHeader, credits));
        }
    }
    ASSERT_EQ(credits.size(), 1);

    void* dst = nullptr;
    ASSERT_EQ(cudaMalloc(&dst, 256), cudaSuccess);
    b::BounceScatterRun run{0, reinterpret_cast<std::uintptr_t>(dst), 0, 0, 256, 1};
    auto const data = b::encodeData(rid, /*chunkIdx=*/0, /*numChunks=*/1, credits.front().regionHandle, {run});
    sender.sendTo("dupDataReceiver", data);
    sender.sendTo("dupDataReceiver", data);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (auto* ctx : heldExecContexts)
    {
        receiverBackend.exec->release(ctx);
    }

    int ackCount = 0;
    auto const ackDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < ackDeadline)
    {
        std::string peer;
        std::string blob;
        b::BounceMsgHeader header{};
        if (sender.recv(peer, blob, 20) && b::decodeHeader(blob, header)
            && static_cast<b::BounceMsgType>(header.msgType) == b::BounceMsgType::kACK && header.requestId == rid)
        {
            ++ackCount;
        }
    }
    EXPECT_EQ(ackCount, 1);

    receiver->shutdown();
    EXPECT_EQ(cudaFree(dst), cudaSuccess);
}

// GRANT and ACK messages carry peer-owned capabilities. Only the intended peer may grant, and an
// ACK is valid only for the matching region after that chunk has reached Sent.
TEST(BounceTransportFailure, RejectsWrongPeerGrantAndInvalidAcks)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    b::ZmqControlChannel senderChannel("validateSender");
    b::ZmqControlChannel legitimate("validateLegitimate");
    b::ZmqControlChannel attacker("validateAttacker");
    ControllableTransferEngine senderEngine;
    auto senderBackend = makeBackend(c, c.maxInflightChunksPerRequest);
    auto sender = std::make_unique<b::BounceTransport>(
        "validateSender", c, 0, &senderChannel, &senderEngine, senderBackend.arena.get(), senderBackend.exec.get());
    ASSERT_TRUE(sender->addPeer("validateLegitimate", legitimate.localEndpoint()));
    ASSERT_TRUE(legitimate.addPeer("validateSender", senderChannel.localEndpoint()));
    ASSERT_TRUE(attacker.addPeer("validateSender", senderChannel.localEndpoint()));

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/4);
    auto fut = sender->submit(bufs.srcDescs, bufs.dstDescs, "validateLegitimate");
    b::BounceCreditEntry credit{/*addr=*/0, /*len=*/256, /*devId=*/0, /*regionHandle=*/77};

    attacker.sendTo("validateSender", b::encodeGrant(/*requestId=*/1, {credit}));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(senderEngine.postCount.load(std::memory_order_acquire), 0);

    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeGrant(/*requestId=*/1, {credit}));

    // Seeing DATA guarantees the matching sender chunk has transitioned to Sent.
    bool sawData = false;
    auto const dataDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < dataDeadline && !sawData)
    {
        std::string peer;
        std::string blob;
        b::BounceMsgHeader header{};
        if (legitimate.recv(peer, blob, 20) && b::decodeHeader(blob, header)
            && static_cast<b::BounceMsgType>(header.msgType) == b::BounceMsgType::kDATA)
        {
            sawData = true;
        }
    }
    ASSERT_TRUE(sawData);
    EXPECT_EQ(senderEngine.postCount.load(std::memory_order_acquire), 1);

    attacker.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle + 1));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);

    sender->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// shutdown() with an in-flight (stuck) request must resolve its future FAILURE, never leave wait() hanging.
TEST(BounceTransportFailure, ShutdownFailsInflight)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/0); // timeout disabled -> only shutdown can resolve it
    auto t = bounce_test::makeNode("sdSolo", c, 1024);
    if (!t)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "nobody"); // stuck (no grant, no timeout)
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
    t->tx->shutdown();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "shutdown left request hanging";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    bounce_test::freeXferBufs(bufs);
}

// Shutdown uses the backend's bounded cancel/release operation instead of polling forever. If the
// backend cannot abort, the future still fails promptly and the engine retains the handle for retry.
TEST(BounceTransportFailure, ShutdownReleaseFailureDoesNotHangOrLoseHandle)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/0);
    b::ZmqControlChannel senderChannel("shutdownReleaseSender");
    b::ZmqControlChannel peer("shutdownReleasePeer");
    ControllableTransferEngine senderEngine;
    senderEngine.allowTerminal.store(false, std::memory_order_release);
    senderEngine.releaseSucceeds.store(false, std::memory_order_release);
    auto senderBackend = makeBackend(c, c.maxInflightChunksPerRequest);
    auto sender = std::make_unique<b::BounceTransport>("shutdownReleaseSender", c, 0, &senderChannel, &senderEngine,
        senderBackend.arena.get(), senderBackend.exec.get());
    ASSERT_TRUE(sender->addPeer("shutdownReleasePeer", peer.localEndpoint()));
    ASSERT_TRUE(peer.addPeer("shutdownReleaseSender", senderChannel.localEndpoint()));

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/5);
    auto fut = sender->submit(bufs.srcDescs, bufs.dstDescs, "shutdownReleasePeer");
    b::BounceCreditEntry credit{/*addr=*/0, /*len=*/256, /*devId=*/0, /*regionHandle=*/91};
    peer.sendTo("shutdownReleaseSender", b::encodeGrant(/*requestId=*/1, {credit}));

    auto const postDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (
        std::chrono::steady_clock::now() < postDeadline && senderEngine.postCount.load(std::memory_order_acquire) == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(senderEngine.postCount.load(std::memory_order_acquire), 1);

    auto const start = std::chrono::steady_clock::now();
    sender->shutdown();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);
    EXPECT_EQ(senderEngine.releaseCount.load(std::memory_order_acquire), 2);
    EXPECT_EQ(senderEngine.outstandingHandle.load(std::memory_order_acquire), 1);

    senderEngine.releaseSucceeds.store(true, std::memory_order_release);
    EXPECT_TRUE(senderEngine.release(1));
    EXPECT_EQ(senderEngine.outstandingHandle.load(std::memory_order_acquire), 0);
    bounce_test::freeXferBufs(bufs);
}

// forgetPeer() (the invalidateRemoteAgent path) must fail any in-flight request to the gone peer,
// even with the request timeout disabled, modeling a peer that drops out mid-transfer.
TEST(BounceTransportFailure, ForgetPeerFailsInflightRequest)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/0); // timeout disabled -> only forgetPeer can resolve it
    auto t = bounce_test::makeNode("fpSolo", c, 1024);
    if (!t)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "gonePeer");                   // stuck: never granted
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
    t->tx->forgetPeer("gonePeer");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "forgetPeer left request hanging";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    // forgetPeer for an unrelated/unknown peer must be a harmless no-op (no crash).
    t->tx->forgetPeer("someoneElse");
    t->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// One sender -> TWO receivers sharing ONE small outgoing arena, deliberately oversubscribed: B and C
// can each grant up to their per-request limit, so aggregate grants exceed sender staging capacity.
// The IO thread must NOT block in acquireLocal (it would deadlock: posted chunks could never send
// DATA -> no ACK -> no region freed). With non-blocking acquire + parked credits, both transfers
// must still complete. (Models KV_TRANSFER_NUM_THREADS>1 fanning out to many peers.)
TEST(BounceTransportFailure, MultiPeerSharedOutgoingPoolNoDeadlock)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/15000);
    c.maxChunkSizeBytes = 4096;
    c.maxInflightChunksPerRequest = 2;
    // Arena holds only 2 max-size regions on every node: the sender's gather arena (2) is
    // oversubscribed by B+C's aggregate grants (up to 4).
    c.arenaAllocationGranularityBytes = c.maxChunkSizeBytes;
    c.arenaSizeBytes = 2ULL * c.maxChunkSizeBytes; // exactly 2 regions (power of two)
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, c.maxChunkSizeBytes / 256ULL);

    auto A = bounce_test::makeNode("mpA", c, maxDescs);
    auto B = bounce_test::makeNode("mpB", c, maxDescs);
    auto C = bounce_test::makeNode("mpC", c, maxDescs);
    if (!A || !B || !C)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*A, *B);
    bounce_test::wirePair(*A, *C);

    // 48 descs * 512B = 24KiB -> ~6 chunks of 4KiB each, so each transfer needs many credits.
    auto toB = bounce_test::makeXferBufs(48, 512, /*seed=*/2);
    auto toC = bounce_test::makeXferBufs(48, 512, /*seed=*/3);
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.emplace_back(
        [&]
        {
            auto fut = A->tx->submit(toB.srcDescs, toB.dstDescs, B->name);
            if (fut.wait_for(std::chrono::seconds(40)) == std::future_status::ready
                && fut.get() == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    threads.emplace_back(
        [&]
        {
            auto fut = A->tx->submit(toC.srcDescs, toC.dstDescs, C->name);
            if (fut.wait_for(std::chrono::seconds(40)) == std::future_status::ready
                && fut.get() == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    for (auto& th : threads)
        th.join();
    EXPECT_EQ(ok.load(), 2) << "multi-peer oversubscribed outgoing arena did not both complete (deadlock?)";
    EXPECT_TRUE(bounce_test::verifyXferBufs(toB)) << "byte mismatch to B";
    EXPECT_TRUE(bounce_test::verifyXferBufs(toC)) << "byte mismatch to C";

    A->tx->shutdown();
    B->tx->shutdown();
    C->tx->shutdown();
    bounce_test::freeXferBufs(toB);
    bounce_test::freeXferBufs(toC);
}

// Many threads submit concurrently to the same transport pair -> all complete (thread-safety).
TEST(BounceTransportFailure, ConcurrentMultiThreadedSubmit)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/10000);
    c.maxInflightChunksPerRequest = 4;
    c.arenaAllocationGranularityBytes = 256;
    c.arenaSizeBytes = 1ULL << 20;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, c.maxChunkSizeBytes / 256ULL);

    auto A = bounce_test::makeNode("cmtA", c, maxDescs);
    auto B = bounce_test::makeNode("cmtB", c, maxDescs);
    if (!A || !B)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*A, *B);

    constexpr int kThreads = 8;
    std::vector<bounce_test::XferBufs> bufs(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs[i] = bounce_test::makeXferBufs(6, 300, /*seed=*/static_cast<std::uint32_t>(10 + i));
    }
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto fut = A->tx->submit(bufs[i].srcDescs, bufs[i].dstDescs, B->name);
                if (fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready
                    && fut.get() == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& th : threads)
    {
        th.join();
    }
    EXPECT_EQ(ok.load(), kThreads) << "not all concurrent submits succeeded";
    for (auto& bb : bufs)
    {
        EXPECT_TRUE(bounce_test::verifyXferBufs(bb));
    }

    A->tx->shutdown();
    B->tx->shutdown();
    for (auto& bb : bufs)
    {
        bounce_test::freeXferBufs(bb);
    }
}
