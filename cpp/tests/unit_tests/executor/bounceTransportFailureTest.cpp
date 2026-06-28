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
// (SUCCESS or FAILURE) and never hang — including when the peer never grants, the transfer
// engine fails, or shutdown races in-flight requests. Plus a multi-threaded concurrent-submit
// test for thread-safety (R4).

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/LocalCopyTransferEngine.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{
bool hasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Engine whose writes always report failure -> exercises the reactor's transfer-failure path.
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

// Build src/dst descs over freshly-allocated device buffers (caller frees).
struct Buffers
{
    void* src{nullptr};
    void* dst{nullptr};
    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {}};
    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {}};
};

Buffers makeBuffers(std::uint32_t nDescs, std::uint32_t descBytes)
{
    std::size_t const total = static_cast<std::size_t>(nDescs) * descBytes;
    Buffers b;
    EXPECT_EQ(cudaMalloc(&b.src, total), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&b.dst, total), cudaSuccess);
    std::vector<kvc::MemoryDesc> sd;
    std::vector<kvc::MemoryDesc> dd;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        sd.emplace_back(reinterpret_cast<std::uintptr_t>(static_cast<char*>(b.src) + i * descBytes), descBytes, 0);
        dd.emplace_back(reinterpret_cast<std::uintptr_t>(static_cast<char*>(b.dst) + i * descBytes), descBytes, 0);
    }
    b.srcDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(sd)};
    b.dstDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(dd)};
    return b;
}

b::BounceConfig cfg(int timeoutMs)
{
    b::BounceConfig c;
    c.maxChunkBytes = 4096;
    c.windowDepth = 2;
    c.window = 2;
    c.scatterWorkers = 2;
    c.leaseTimeoutMs = timeoutMs;
    return c;
}

// The shared data arena + exec contexts for one transport. Kept alive by the caller (must outlive
// the BounceTransport that borrows them).
struct Backend
{
    std::unique_ptr<b::BounceArena> arena;
    std::unique_ptr<b::ExecPool> exec;
};

// Build a backend whose arena holds exactly `regionCap` max-size (maxChunkBytes) regions — minBlock ==
// maxChunkBytes makes every region one buddy block, so the arena behaves like the old fixed `regionCap`-
// slot pool (predictable, and lets a small cap deliberately oversubscribe the gather staging). Sets
// c.arenaBytes/minBlock so the transport's scheduler matches the arena it is handed.
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
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/500);
    b::ZmqControlChannel ch("solo");
    b::LocalCopyTransferEngine eng;
    auto be = makeBackend(c, c.windowDepth);
    auto t = std::make_unique<b::BounceTransport>("solo", c, 0, &ch, &eng, be.arena.get(), be.exec.get());

    auto bufs = makeBuffers(8, 256);
    auto fut = t->submit(bufs.srcDescs, bufs.dstDescs, "nobody"); // no peer "nobody" -> WANT dropped
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "request hung with no grant";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    t->shutdown();
    cudaFree(bufs.src);
    cudaFree(bufs.dst);
}

// The transfer engine reports failure -> the request must FAIL (not hang, not falsely succeed).
TEST(BounceTransportFailure, EngineFailureFailsRequest)
{
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/5000);
    b::ZmqControlChannel chA("fA");
    b::ZmqControlChannel chB("fB");
    FailingTransferEngine engA; // sender's writes fail
    b::LocalCopyTransferEngine engB;
    auto beA = makeBackend(c, c.windowDepth);
    auto beB = makeBackend(c, c.windowDepth);
    auto A = std::make_unique<b::BounceTransport>("fA", c, 0, &chA, &engA, beA.arena.get(), beA.exec.get());
    auto B = std::make_unique<b::BounceTransport>("fB", c, 0, &chB, &engB, beB.arena.get(), beB.exec.get());
    A->addPeer("fB", chB.localEndpoint());
    B->addPeer("fA", chA.localEndpoint());

    auto bufs = makeBuffers(8, 256);
    auto fut = A->submit(bufs.srcDescs, bufs.dstDescs, "fB");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "engine failure hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    A->shutdown();
    B->shutdown();
    cudaFree(bufs.src);
    cudaFree(bufs.dst);
}

// shutdown() with an in-flight (stuck) request must resolve its future FAILURE, never leave wait() hanging.
TEST(BounceTransportFailure, ShutdownFailsInflight)
{
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/0); // timeout disabled -> only shutdown can resolve it
    b::ZmqControlChannel ch("sd");
    b::LocalCopyTransferEngine eng;
    auto be = makeBackend(c, c.windowDepth);
    auto t = std::make_unique<b::BounceTransport>("sd", c, 0, &ch, &eng, be.arena.get(), be.exec.get());

    auto bufs = makeBuffers(8, 256);
    auto fut = t->submit(bufs.srcDescs, bufs.dstDescs, "nobody"); // stuck (no grant, no timeout)
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
    t->shutdown();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "shutdown left request hanging";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    cudaFree(bufs.src);
    cudaFree(bufs.dst);
}

// forgetPeer() (the invalidateRemoteAgent path) must fail any in-flight request to the gone
// peer, even with the lease timeout disabled -> models a peer dropping out mid-transfer (R5).
TEST(BounceTransportFailure, ForgetPeerFailsInflightRequest)
{
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/0); // timeout disabled -> only forgetPeer can resolve it
    b::ZmqControlChannel ch("fp");
    b::LocalCopyTransferEngine eng;
    auto be = makeBackend(c, c.windowDepth);
    auto t = std::make_unique<b::BounceTransport>("fp", c, 0, &ch, &eng, be.arena.get(), be.exec.get());

    auto bufs = makeBuffers(8, 256);
    auto fut = t->submit(bufs.srcDescs, bufs.dstDescs, "gonePeer");                       // stuck: never granted
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
    t->forgetPeer("gonePeer");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "forgetPeer left request hanging";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    // forgetPeer for an unrelated/unknown peer must be a harmless no-op (no crash).
    t->forgetPeer("someoneElse");
    t->shutdown();
    cudaFree(bufs.src);
    cudaFree(bufs.dst);
}

// One sender -> TWO receivers sharing ONE small outgoing pool, deliberately oversubscribed:
// B and C can each grant up to their incoming depth, so aggregate grants > sender outgoing depth.
// The IO thread must NOT block in checkout (it would deadlock: posted chunks could never send
// DATA -> no ACK -> no slot freed). With non-blocking try-checkout + parked credits, both
// transfers must still complete. (Models KV_TRANSFER_NUM_THREADS>1 fanning out to many peers.)
TEST(BounceTransportFailure, MultiPeerSharedOutgoingPoolNoDeadlock)
{
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/15000);
    c.maxChunkBytes = 4096;
    c.window = 2;
    b::ZmqControlChannel chA("mpA");
    b::ZmqControlChannel chB("mpB");
    b::ZmqControlChannel chC("mpC");
    b::LocalCopyTransferEngine engA;
    b::LocalCopyTransferEngine engB;
    b::LocalCopyTransferEngine engC;
    // Sender A: ONE shared arena holding only 2 max-size regions, used for gather staging across
    // BOTH sends (the oversubscribed resource — aggregate grants from B+C can reach 4 > 2).
    auto beA = makeBackend(c, /*regionCap=*/2);
    // Receivers B,C: 2 regions each -> together can grant up to 4 > A's arena of 2.
    auto beB = makeBackend(c, /*regionCap=*/2);
    auto beC = makeBackend(c, /*regionCap=*/2);
    auto A = std::make_unique<b::BounceTransport>("mpA", c, 0, &chA, &engA, beA.arena.get(), beA.exec.get());
    auto B = std::make_unique<b::BounceTransport>("mpB", c, 0, &chB, &engB, beB.arena.get(), beB.exec.get());
    auto C = std::make_unique<b::BounceTransport>("mpC", c, 0, &chC, &engC, beC.arena.get(), beC.exec.get());
    A->addPeer("mpB", chB.localEndpoint());
    A->addPeer("mpC", chC.localEndpoint());
    B->addPeer("mpA", chA.localEndpoint());
    C->addPeer("mpA", chA.localEndpoint());

    // 48 descs * 512B = 24KiB -> ~6 chunks of 4KiB each, so each transfer needs many credits.
    auto toB = makeBuffers(48, 512);
    auto toC = makeBuffers(48, 512);
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.emplace_back(
        [&]
        {
            auto fut = A->submit(toB.srcDescs, toB.dstDescs, "mpB");
            if (fut.wait_for(std::chrono::seconds(25)) == std::future_status::ready
                && fut.get() == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    threads.emplace_back(
        [&]
        {
            auto fut = A->submit(toC.srcDescs, toC.dstDescs, "mpC");
            if (fut.wait_for(std::chrono::seconds(25)) == std::future_status::ready
                && fut.get() == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    for (auto& th : threads)
        th.join();
    EXPECT_EQ(ok.load(), 2) << "multi-peer oversubscribed outgoing pool did not both complete (deadlock?)";

    A->shutdown();
    B->shutdown();
    C->shutdown();
    cudaFree(toB.src);
    cudaFree(toB.dst);
    cudaFree(toC.src);
    cudaFree(toC.dst);
}

// Many threads submit concurrently to the same transport pair -> all complete (thread-safety, R4).
TEST(BounceTransportFailure, ConcurrentMultiThreadedSubmit)
{
    if (!hasCuda())
        GTEST_SKIP();
    auto c = cfg(/*timeoutMs=*/10000);
    c.windowDepth = 4;
    c.window = 4;
    b::ZmqControlChannel chA("cA");
    b::ZmqControlChannel chB("cB");
    b::LocalCopyTransferEngine engA;
    b::LocalCopyTransferEngine engB;
    auto beA = makeBackend(c, c.windowDepth);
    auto beB = makeBackend(c, c.windowDepth);
    auto A = std::make_unique<b::BounceTransport>("cA", c, 0, &chA, &engA, beA.arena.get(), beA.exec.get());
    auto B = std::make_unique<b::BounceTransport>("cB", c, 0, &chB, &engB, beB.arena.get(), beB.exec.get());
    A->addPeer("cB", chB.localEndpoint());
    B->addPeer("cA", chA.localEndpoint());

    constexpr int kThreads = 8;
    std::vector<Buffers> bufs(kThreads);
    for (auto& bb : bufs)
    {
        bb = makeBuffers(6, 300);
    }
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto fut = A->submit(bufs[i].srcDescs, bufs[i].dstDescs, "cB");
                if (fut.wait_for(std::chrono::seconds(20)) == std::future_status::ready
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

    A->shutdown();
    B->shutdown();
    for (auto& bb : bufs)
    {
        cudaFree(bb.src);
        cudaFree(bb.dst);
    }
}
