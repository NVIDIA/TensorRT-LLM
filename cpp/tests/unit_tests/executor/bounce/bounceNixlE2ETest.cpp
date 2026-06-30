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

// Real-NIXL end-to-end loopback for the bounce v2 pipeline. Two NixlTransferAgents in one
// process provide the agent + UCX backend + metadata bootstrap (via the public getRawAgent());
// we register the bounce arena on each raw agent, exchange metadata, and drive two
// BounceTransports over ACTUAL NIXL RDMA. This is the production data path (NixlTransferEngine),
// verifying byte-exact movement + credit recycling, without surgery on the legacy agent.

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/NixlTransferEngine.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
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

std::uint64_t alignUp(std::uint64_t v, std::uint64_t a)
{
    return (v + a - 1) / a * a;
}

unsigned char pattern(std::size_t d, std::size_t i)
{
    return static_cast<unsigned char>((d * 191 + i * 23 + 5) & 0xFF);
}

// A `seed`-distinct byte pattern so concurrent transfers can't masquerade as each other.
unsigned char patSeed(std::uint32_t seed, std::size_t d, std::size_t i)
{
    return static_cast<unsigned char>((seed * 131 + d * 191 + i * 23 + 5) & 0xFF);
}

// One transfer's src (gather source) + dst (scatter target) device buffers + descriptor lists.
// These KV buffers are NOT NIXL-registered — only the bounce arena traverses RDMA.
struct XferBufs
{
    void* src{nullptr};
    void* dst{nullptr};
    std::uint32_t nDescs{};
    std::uint32_t descBytes{};
    std::uint32_t seed{};
    std::vector<std::uint64_t> off;
    std::uint64_t total{};
    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {}};
    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {}};
};

XferBufs makeXferBufs(std::uint32_t nDescs, std::uint32_t descBytes, std::uint32_t seed)
{
    XferBufs x;
    x.nDescs = nDescs;
    x.descBytes = descBytes;
    x.seed = seed;
    x.off.resize(nDescs);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        x.off[i] = cur;
        cur = alignUp(cur + descBytes, 256);
    }
    x.total = cur;
    EXPECT_EQ(cudaMalloc(&x.src, x.total), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&x.dst, x.total), cudaSuccess);
    std::vector<unsigned char> h(x.total, 0);
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        for (std::uint32_t j = 0; j < descBytes; ++j)
        {
            h[x.off[i] + j] = patSeed(seed, i, j);
        }
    }
    EXPECT_EQ(cudaMemcpy(x.src, h.data(), x.total, cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(x.dst, 0, x.total), cudaSuccess);
    auto a2 = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uintptr_t>(static_cast<char*>(p) + o); };
    std::vector<kvc::MemoryDesc> sd;
    std::vector<kvc::MemoryDesc> dd;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        sd.emplace_back(a2(x.src, x.off[i]), descBytes, 0);
        dd.emplace_back(a2(x.dst, x.off[i]), descBytes, 0);
    }
    x.srcDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(sd)};
    x.dstDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(dd)};
    return x;
}

bool verifyXferBufs(XferBufs const& x)
{
    std::vector<unsigned char> got(x.total, 0xEE);
    if (cudaMemcpy(got.data(), x.dst, x.total, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    for (std::uint32_t i = 0; i < x.nDescs; ++i)
    {
        for (std::uint32_t j = 0; j < x.descBytes; ++j)
        {
            if (got[x.off[i] + j] != patSeed(x.seed, i, j))
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(BounceNixlE2E, RealRdmaLoopback)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    // Stand up two real NIXL agents (UCX backend). Skip if the backend can't init in this env.
    std::unique_ptr<kvc::NixlTransferAgent> agentA;
    std::unique_ptr<kvc::NixlTransferAgent> agentB;
    try
    {
        kvc::BaseAgentConfig cfgA{"bounceA", /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true};
        kvc::BaseAgentConfig cfgB{"bounceB", true, false, true};
        agentA = std::make_unique<kvc::NixlTransferAgent>(cfgA);
        agentB = std::make_unique<kvc::NixlTransferAgent>(cfgB);
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    constexpr std::size_t kMaxChunkBytes = 4096; // per-chunk byte cap
    constexpr std::uint32_t kWindowDepth = 2;    // per-flow window -> forces credit recycling below
    constexpr std::size_t kArenaBytes = 1 << 20; // shared data arena (holds many small regions)
    constexpr std::size_t kMinBlock = 256;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkBytes / 256ULL);

    b::NixlTransferEngine engA(agentA->getRawAgent(), 0);
    b::NixlTransferEngine engB(agentB->getRawAgent(), 0);

    // ONE shared arena per agent (both roles). A is sender (gathers into arenaA regions, RDMA src);
    // B is receiver (arenaB regions are the RDMA-write targets).
    b::BounceArena arenaA(kArenaBytes, 0, /*allowFabric=*/false);
    b::BounceArena arenaB(kArenaBytes, 0, /*allowFabric=*/false);
    b::ExecPool execA(kWindowDepth + 4, maxDescs, 0);
    b::ExecPool execB(kWindowDepth + 4, maxDescs, 0);

    // Register each arena BEFORE exchanging metadata, so each peer's metadata includes the buffer
    // and createXferReq can resolve the remote region addresses.
    ASSERT_TRUE(engA.registerRegion(arenaA.base(), arenaA.bytes()));
    ASSERT_TRUE(engB.registerRegion(arenaB.base(), arenaB.bytes()));

    agentA->loadRemoteAgent("bounceB", agentB->getLocalConnectionInfo());
    agentB->loadRemoteAgent("bounceA", agentA->getLocalConnectionInfo());

    b::BounceConfig cfg;
    cfg.maxChunkBytes = kMaxChunkBytes;
    cfg.arenaBytes = kArenaBytes;
    cfg.minBlock = kMinBlock;
    cfg.windowDepth = kWindowDepth;
    cfg.window = kWindowDepth;
    cfg.scatterWorkers = 2;

    b::ZmqControlChannel chA("bounceA");
    b::ZmqControlChannel chB("bounceB");
    auto A = std::make_unique<b::BounceTransport>("bounceA", cfg, 0, &chA, &engA, &arenaA, &execA);
    auto B = std::make_unique<b::BounceTransport>("bounceB", cfg, 0, &chB, &engB, &arenaB, &execB);
    A->addPeer("bounceB", chB.localEndpoint());
    B->addPeer("bounceA", chA.localEndpoint());

    // Source KV data on A, destination buffers on B (these are NOT NIXL-registered; they are
    // gathered/scattered locally — only the bounce arena regions traverse RDMA).
    constexpr std::uint32_t kNDescs = 24;
    constexpr std::uint32_t kDescBytes = 600;
    std::vector<std::uint64_t> off(kNDescs);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < kNDescs; ++i)
    {
        off[i] = cur;
        cur = alignUp(cur + kDescBytes, 256);
    }
    std::uint64_t const total = cur;

    void* dSrc = nullptr; // on A
    void* dDst = nullptr; // on B
    ASSERT_EQ(cudaMalloc(&dSrc, total), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dDst, total), cudaSuccess);
    std::vector<unsigned char> host(total, 0);
    for (std::uint32_t i = 0; i < kNDescs; ++i)
    {
        for (std::uint32_t j = 0; j < kDescBytes; ++j)
        {
            host[off[i] + j] = pattern(i, j);
        }
    }
    ASSERT_EQ(cudaMemcpy(dSrc, host.data(), total, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(dDst, 0, total), cudaSuccess);

    auto a2 = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uintptr_t>(static_cast<char*>(p) + o); };
    std::vector<kvc::MemoryDesc> sd;
    std::vector<kvc::MemoryDesc> dd;
    for (std::uint32_t i = 0; i < kNDescs; ++i)
    {
        sd.emplace_back(a2(dSrc, off[i]), kDescBytes, 0);
        dd.emplace_back(a2(dDst, off[i]), kDescBytes, 0);
    }
    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, std::move(sd)};
    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, std::move(dd)};

    auto fut = A->submit(srcDescs, dstDescs, "bounceB");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(30)), std::future_status::ready) << "real-NIXL transfer hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kSUCCESS);

    std::vector<unsigned char> got(total, 0xEE);
    ASSERT_EQ(cudaMemcpy(got.data(), dDst, total, cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::uint32_t i = 0; i < kNDescs; ++i)
    {
        for (std::uint32_t j = 0; j < kDescBytes; ++j)
        {
            ASSERT_EQ(got[off[i] + j], pattern(i, j)) << "mismatch desc=" << i << " byte=" << j;
        }
    }

    A->shutdown();
    B->shutdown();
    cudaFree(dSrc);
    cudaFree(dDst);
}

// Concurrent, BIDIRECTIONAL real-NIXL RDMA: many threads submit transfers at once, half A->B and
// half B->A, so BOTH agents' shared arenas serve sender (gather) AND receiver (RDMA target +
// scatter) roles simultaneously. With a small window + arena this exercises the fair scheduler,
// credit recycling, and ExecPool contention over actual RDMA — every transfer must complete
// SUCCESS and land byte-exact, with no deadlock/hang/cross-talk (R2/R3/R4 over the production path).
TEST(BounceNixlE2E, ConcurrentBidirectionalRealRdma)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    std::unique_ptr<kvc::NixlTransferAgent> agentA;
    std::unique_ptr<kvc::NixlTransferAgent> agentB;
    try
    {
        kvc::BaseAgentConfig cfgA{"concA", /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true};
        kvc::BaseAgentConfig cfgB{"concB", true, false, true};
        agentA = std::make_unique<kvc::NixlTransferAgent>(cfgA);
        agentB = std::make_unique<kvc::NixlTransferAgent>(cfgB);
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    constexpr std::size_t kMaxChunkBytes = 4096;
    constexpr std::uint32_t kWindowDepth = 2;    // small -> forces recycling under concurrency
    constexpr std::size_t kArenaBytes = 2 << 20; // 2 MiB shared arena per agent (both roles)
    constexpr std::size_t kMinBlock = 256;
    constexpr int kThreads = 8;                  // 4 flows A->B + 4 flows B->A, all concurrent
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkBytes / 256ULL);

    b::NixlTransferEngine engA(agentA->getRawAgent(), 0);
    b::NixlTransferEngine engB(agentB->getRawAgent(), 0);
    b::BounceArena arenaA(kArenaBytes, 0, /*allowFabric=*/false);
    b::BounceArena arenaB(kArenaBytes, 0, /*allowFabric=*/false);
    b::ExecPool execA(kThreads + 4, maxDescs, 0);
    b::ExecPool execB(kThreads + 4, maxDescs, 0);
    ASSERT_TRUE(engA.registerRegion(arenaA.base(), arenaA.bytes()));
    ASSERT_TRUE(engB.registerRegion(arenaB.base(), arenaB.bytes()));

    agentA->loadRemoteAgent("concB", agentB->getLocalConnectionInfo());
    agentB->loadRemoteAgent("concA", agentA->getLocalConnectionInfo());

    b::BounceConfig cfg;
    cfg.maxChunkBytes = kMaxChunkBytes;
    cfg.arenaBytes = kArenaBytes;
    cfg.minBlock = kMinBlock;
    cfg.windowDepth = kWindowDepth;
    cfg.window = kWindowDepth;
    cfg.scatterWorkers = 2;

    b::ZmqControlChannel chA("concA");
    b::ZmqControlChannel chB("concB");
    auto A = std::make_unique<b::BounceTransport>("concA", cfg, 0, &chA, &engA, &arenaA, &execA);
    auto B = std::make_unique<b::BounceTransport>("concB", cfg, 0, &chB, &engB, &arenaB, &execB);
    A->addPeer("concB", chB.localEndpoint());
    B->addPeer("concA", chA.localEndpoint());

    // One distinct (seeded) buffer set per thread; ~24KiB / 16 descs each -> several chunks per
    // transfer so each flow recycles its window over RDMA.
    std::vector<XferBufs> bufs;
    bufs.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs.push_back(makeXferBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(i + 1)));
    }

    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                // Even threads send A->B; odd threads send B->A (both arenas act as both roles).
                bool const a2b = (i % 2 == 0);
                auto* tx = a2b ? A.get() : B.get();
                char const* peer = a2b ? "concB" : "concA";
                auto fut = tx->submit(bufs[i].srcDescs, bufs[i].dstDescs, peer);
                if (fut.wait_for(std::chrono::seconds(60)) == std::future_status::ready
                    && fut.get() == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(ok.load(), kThreads) << "not all concurrent bidirectional RDMA transfers completed";

    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyXferBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    A->shutdown();
    B->shutdown();
    for (auto& x : bufs)
    {
        cudaFree(x.src);
        cudaFree(x.dst);
    }
}

// One bounce node = a full agent stack (agent + engine + arena + exec + control channel + transport).
struct Node
{
    std::string name;
    std::unique_ptr<kvc::NixlTransferAgent> agent;
    std::unique_ptr<b::NixlTransferEngine> eng;
    std::unique_ptr<b::BounceArena> arena;
    std::unique_ptr<b::ExecPool> exec;
    std::unique_ptr<b::ZmqControlChannel> ch;
    std::unique_ptr<b::BounceTransport> tx;
};

// MULTI-AGENT (N>2) real-NIXL RDMA: 1 receiver + S senders, each a separate UCX agent. All senders
// write to the one receiver CONCURRENTLY (the disagg "many context workers -> one gen" shape). Each
// sender's data lands in its own distinct receiver buffers and is verified BYTE-EXACT (seed-distinct
// patterns -> any cross-talk or precision bug fails). Exercises the receiver's fair scheduler +
// credit recycling across multiple real peers (R3) over actual RDMA.
TEST(BounceNixlE2E, MultiAgentManySendersToOneReceiver)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    constexpr int kSenders = 3; // total agents = 1 receiver + 3 senders = 4
    constexpr std::size_t kMaxChunkBytes = 4096;
    constexpr std::uint32_t kWindowDepth = 2;
    constexpr std::size_t kArenaBytes = 2 << 20;
    constexpr std::size_t kMinBlock = 256;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkBytes / 256ULL);

    b::BounceConfig cfg;
    cfg.maxChunkBytes = kMaxChunkBytes;
    cfg.arenaBytes = kArenaBytes;
    cfg.minBlock = kMinBlock;
    cfg.windowDepth = kWindowDepth;
    cfg.window = kWindowDepth;
    cfg.scatterWorkers = 2;

    // nodes[0] = receiver "R", nodes[1..kSenders] = senders "S<i>".
    std::vector<std::unique_ptr<Node>> nodes;
    for (int i = 0; i <= kSenders; ++i)
    {
        auto n = std::make_unique<Node>();
        n->name = (i == 0) ? std::string("mnR") : ("mnS" + std::to_string(i));
        try
        {
            kvc::BaseAgentConfig c{n->name, /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true};
            n->agent = std::make_unique<kvc::NixlTransferAgent>(c);
        }
        catch (std::exception const& e)
        {
            GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
        }
        n->eng = std::make_unique<b::NixlTransferEngine>(n->agent->getRawAgent(), 0);
        n->arena = std::make_unique<b::BounceArena>(kArenaBytes, 0, /*allowFabric=*/false);
        n->exec = std::make_unique<b::ExecPool>(kWindowDepth + 4, maxDescs, 0);
        ASSERT_TRUE(n->eng->registerRegion(n->arena->base(), n->arena->bytes()));
        n->ch = std::make_unique<b::ZmqControlChannel>(n->name);
        n->tx = std::make_unique<b::BounceTransport>(
            n->name, cfg, 0, n->ch.get(), n->eng.get(), n->arena.get(), n->exec.get());
        nodes.push_back(std::move(n));
    }

    // Star wiring: connect each sender Si <-> receiver R (NIXL metadata both ways + bounce addPeer).
    Node& R = *nodes[0];
    for (int i = 1; i <= kSenders; ++i)
    {
        Node& S = *nodes[i];
        S.agent->loadRemoteAgent(R.name, R.agent->getLocalConnectionInfo());
        R.agent->loadRemoteAgent(S.name, S.agent->getLocalConnectionInfo());
        S.tx->addPeer(R.name, R.ch->localEndpoint());
        R.tx->addPeer(S.name, S.ch->localEndpoint());
    }

    // Each sender gets its own seeded buffer set (src on the sender, dst on the receiver).
    std::vector<XferBufs> bufs;
    bufs.reserve(kSenders);
    for (int i = 1; i <= kSenders; ++i)
    {
        bufs.push_back(makeXferBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(100 + i)));
    }

    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kSenders);
    for (int i = 1; i <= kSenders; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto fut = nodes[i]->tx->submit(bufs[i - 1].srcDescs, bufs[i - 1].dstDescs, R.name);
                if (fut.wait_for(std::chrono::seconds(60)) == std::future_status::ready
                    && fut.get() == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(ok.load(), kSenders) << "not all senders completed to the shared receiver";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyXferBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    for (auto& n : nodes)
    {
        n->tx->shutdown();
    }
    for (auto& x : bufs)
    {
        cudaFree(x.src);
        cudaFree(x.dst);
    }
}

namespace
{
// Build one full bounce node (agent+engine+arena+exec+channel+transport). Returns nullptr if the
// NIXL agent/backend can't init or the arena can't be registered (caller GTEST_SKIPs).
std::unique_ptr<Node> makeNode(std::string const& name, b::BounceConfig const& cfg, std::size_t maxDescs)
{
    auto n = std::make_unique<Node>();
    n->name = name;
    try
    {
        kvc::BaseAgentConfig c{name, /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true};
        n->agent = std::make_unique<kvc::NixlTransferAgent>(c);
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
    n->eng = std::make_unique<b::NixlTransferEngine>(n->agent->getRawAgent(), 0);
    n->arena = std::make_unique<b::BounceArena>(cfg.arenaBytes, 0, /*allowFabric=*/false);
    n->exec = std::make_unique<b::ExecPool>(cfg.windowDepth + 4, maxDescs, 0);
    if (!n->eng->registerRegion(n->arena->base(), n->arena->bytes()))
    {
        return nullptr;
    }
    n->ch = std::make_unique<b::ZmqControlChannel>(name);
    n->tx = std::make_unique<b::BounceTransport>(
        n->name, cfg, 0, n->ch.get(), n->eng.get(), n->arena.get(), n->exec.get());
    return n;
}

// Bidirectional connect: NIXL metadata both ways + bounce addPeer both ways.
void wirePair(Node& a, Node& b)
{
    a.agent->loadRemoteAgent(b.name, b.agent->getLocalConnectionInfo());
    b.agent->loadRemoteAgent(a.name, a.agent->getLocalConnectionInfo());
    a.tx->addPeer(b.name, b.ch->localEndpoint());
    b.tx->addPeer(a.name, a.ch->localEndpoint());
}
} // namespace

// FAILURE PATH over the real stack: a request whose peer never GRANTs must FAIL on leaseTimeout, not
// hang. The sender's WANT goes to a control endpoint with no live receiver transport (nobody grants),
// so checkTimeouts must resolve the future kFAILURE within ~leaseTimeoutMs (R5 over the real
// ZmqControlChannel + NixlTransferEngine wiring).
TEST(BounceNixlE2E, NoGrantTimesOutNotHang)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr std::size_t kMaxChunkBytes = 4096;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkBytes / 256ULL);
    b::BounceConfig cfg;
    cfg.maxChunkBytes = kMaxChunkBytes;
    cfg.arenaBytes = 1 << 20;
    cfg.minBlock = 256;
    cfg.windowDepth = 2;
    cfg.window = 2;
    cfg.scatterWorkers = 2;
    cfg.leaseTimeoutMs = 1500; // short: a no-grant request must fail fast, not hang

    auto A = makeNode("toGrantA", cfg, maxDescs);
    if (!A)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    // A bound ROUTER that no transport ever recv()s on -> A's WANT is delivered but never granted.
    b::ZmqControlChannel ghost("ghostPeer");
    A->tx->addPeer("ghostPeer", ghost.localEndpoint());

    auto bufs = makeXferBufs(/*nDescs=*/8, /*descBytes=*/500, /*seed=*/7);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, "ghostPeer");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "no-grant request hung";
    EXPECT_EQ(fut.get(), kvc::TransferState::kFAILURE);

    A->tx->shutdown();
    cudaFree(bufs.src);
    cudaFree(bufs.dst);
}

// FAILURE PATH over real RDMA: forgetPeer() while a transfer is in flight must not hang, leak, or
// corrupt. We submit then immediately forgetPeer the target; the request must RESOLVE (SUCCESS if it
// beat the queued reclaim, else FAILURE) — never hang. Then several FRESH transfers to the same peer
// must still complete byte-exact, proving forgetPeer's reclaim returned the regions to the arena and
// left the reactor healthy (small arena + window -> a leak would soon exhaust it and stall these).
TEST(BounceNixlE2E, ForgetPeerInFlightRecovers)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr std::size_t kMaxChunkBytes = 4096;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkBytes / 256ULL);
    b::BounceConfig cfg;
    cfg.maxChunkBytes = kMaxChunkBytes;
    cfg.arenaBytes = 1 << 20;
    cfg.minBlock = 256;
    cfg.windowDepth = 2;
    cfg.window = 2;
    cfg.scatterWorkers = 2;
    cfg.leaseTimeoutMs = 5000;

    auto A = makeNode("fpA", cfg, maxDescs);
    auto B = makeNode("fpB", cfg, maxDescs);
    if (!A || !B)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    wirePair(*A, *B);

    auto bufs = makeXferBufs(/*nDescs=*/24, /*descBytes=*/600, /*seed=*/1);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, "fpB");
    A->tx->forgetPeer("fpB"); // drop the peer (queued; applied on A's IO thread)
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "request hung after forgetPeer";
    auto const st = fut.get();
    EXPECT_TRUE(st == kvc::TransferState::kSUCCESS || st == kvc::TransferState::kFAILURE)
        << "unexpected state " << static_cast<int>(st);
    cudaFree(bufs.src);
    cudaFree(bufs.dst);

    // Recovery + no-leak: forgetPeer is a one-shot event; the scheduler/request reclaim is drained by
    // the time fut resolved, so these new flows aren't reclaimed. forgetPeer ALSO drops the
    // control-channel DEALER to fpB synchronously (on this thread), so re-establish it with addPeer
    // before recovering — deterministic because forgetPeer's removePeer happens-before this addPeer
    // (no async removePeer can race/erase the freshly re-added dealer). NIXL metadata persists, so
    // only the dealer is re-added (no loadRemoteAgent / full re-wire).
    A->tx->addPeer("fpB", B->ch->localEndpoint());
    for (int k = 0; k < 5; ++k)
    {
        auto rb = makeXferBufs(/*nDescs=*/20, /*descBytes=*/600, /*seed=*/static_cast<std::uint32_t>(50 + k));
        auto rf = A->tx->submit(rb.srcDescs, rb.dstDescs, "fpB");
        ASSERT_EQ(rf.wait_for(std::chrono::seconds(30)), std::future_status::ready)
            << "post-forget transfer hung k=" << k;
        EXPECT_EQ(rf.get(), kvc::TransferState::kSUCCESS) << "post-forget transfer failed k=" << k;
        EXPECT_TRUE(verifyXferBufs(rb)) << "post-forget byte mismatch k=" << k;
        cudaFree(rb.src);
        cudaFree(rb.dst);
    }

    A->tx->shutdown();
    B->tx->shutdown();
}
