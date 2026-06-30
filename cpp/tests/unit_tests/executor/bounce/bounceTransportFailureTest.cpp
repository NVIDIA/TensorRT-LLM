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
// fails, or shutdown races in-flight requests. Plus a multi-threaded concurrent-submit test (R4).
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

    void release(std::uint64_t) override {}
};

b::BounceConfig cfg(int timeoutMs)
{
    b::BounceConfig c;
    c.maxChunkBytes = 4096;
    c.windowDepth = 2;
    c.window = 2;
    c.scatterWorkers = 2;
    c.minBlock = 256;
    c.arenaBytes = 1ULL << 20;
    c.leaseTimeoutMs = timeoutMs;
    return c;
}

// The shared data arena + exec contexts for one manually-built transport (used only by the
// FailingTransferEngine test, which doesn't use a NIXL node). Kept alive by the caller.
struct Backend
{
    std::unique_ptr<b::BounceArena> arena;
    std::unique_ptr<b::ExecPool> exec;
};

// Arena holds exactly `regionCap` max-size (maxChunkBytes) regions — minBlock == maxChunkBytes makes
// every region one buddy block. Sets c.arenaBytes/minBlock so the scheduler matches the arena.
Backend makeBackend(b::BounceConfig& c, std::uint32_t regionCap)
{
    c.minBlock = c.maxChunkBytes;
    std::size_t arenaBytes = c.minBlock;
    while (arenaBytes < static_cast<std::size_t>(regionCap) * c.maxChunkBytes)
    {
        arenaBytes <<= 1;
    }
    c.arenaBytes = arenaBytes;
    std::uint32_t const execCount = regionCap + c.scatterWorkers + 4;
    return Backend{std::make_unique<b::BounceArena>(arenaBytes, 0, /*allowFabric=*/false),
        std::make_unique<b::ExecPool>(execCount, 1024, 0)};
}
} // namespace

// Peer never grants (not addPeer'd) -> the request must FAIL on the lease timeout, not hang.
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
    auto beA = makeBackend(c, c.windowDepth);
    auto beB = makeBackend(c, c.windowDepth);
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

// forgetPeer() (the invalidateRemoteAgent path) must fail any in-flight request to the gone peer,
// even with the lease timeout disabled -> models a peer dropping out mid-transfer (R5).
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
// can each grant up to their incoming depth, so aggregate grants > sender's gather-staging capacity.
// The IO thread must NOT block in acquireLocal (it would deadlock: posted chunks could never send
// DATA -> no ACK -> no region freed). With non-blocking acquire + parked credits, both transfers
// must still complete. (Models KV_TRANSFER_NUM_THREADS>1 fanning out to many peers.)
TEST(BounceTransportFailure, MultiPeerSharedOutgoingPoolNoDeadlock)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/15000);
    c.maxChunkBytes = 4096;
    c.window = 2;
    c.windowDepth = 2;
    // Arena holds only 2 max-size regions on every node: the sender's gather arena (2) is
    // oversubscribed by B+C's aggregate grants (up to 4).
    c.minBlock = c.maxChunkBytes;
    c.arenaBytes = 2ULL * c.maxChunkBytes; // exactly 2 regions (power of two)
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, c.maxChunkBytes / 256ULL);

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

// Many threads submit concurrently to the same transport pair -> all complete (thread-safety, R4).
TEST(BounceTransportFailure, ConcurrentMultiThreadedSubmit)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/10000);
    c.windowDepth = 4;
    c.window = 4;
    c.minBlock = 256;
    c.arenaBytes = 1ULL << 20;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, c.maxChunkBytes / 256ULL);

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
