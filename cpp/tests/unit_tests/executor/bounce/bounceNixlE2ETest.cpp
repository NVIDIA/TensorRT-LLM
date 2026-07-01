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

// Real-NIXL FAILURE/EDGE paths for the bounce v2 pipeline that need a hand-built (white-box)
// BounceTransport to inject faults the public agent API can't easily reach — a peer that never
// GRANTs (leaseTimeout) and forgetPeer() while a transfer is in flight. Each test stands up real
// NixlTransferAgents (agent + UCX backend + metadata via getRawAgent()), registers the bounce arena
// on each raw agent, and drives standalone BounceTransports over ACTUAL NIXL RDMA.
//
// The happy-path / concurrency / bidirectional / multi-agent coverage lives in bounceAgentE2ETest,
// which drives the SAME pipeline through the production entry point (NixlTransferAgent::
// submitTransferRequests) with the production one-directional AgentDesc bootstrap — a strictly more
// faithful setup, so it is not duplicated here.

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

// Bidirectional connect for the white-box harness. Two SEPARATE wirings are needed here because the
// transport under test is hand-built (b::BounceTransport with its own b::ZmqControlChannel) and lives
// OUTSIDE the agent — so, unlike production (bounceAgentE2ETest), loadRemoteAgent cannot wire it:
//   - loadRemoteAgent(AgentDesc) exchanges only the NIXL metadata layer (so createXferReq can resolve
//     the remote arena). We use the AgentDesc path — the one production disagg uses — not the
//     connection-info path.
//   - addPeer() wires the control-channel layer (the DEALER to the peer's ROUTER) on the standalone
//     transport. Production folds this into loadRemoteAgent + WANT self-bootstrap; here it is manual.
void wirePair(Node& a, Node& b)
{
    a.agent->loadRemoteAgent(b.name, b.agent->getLocalAgentDesc());
    b.agent->loadRemoteAgent(a.name, a.agent->getLocalAgentDesc());
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
