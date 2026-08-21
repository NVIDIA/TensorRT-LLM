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

// Production-path e2e: bounce engaged transparently through NixlTransferAgent::submitTransferRequests
// with BaseAgentConfig::agentBufferSizeMb > 0. Real agents (2..N), real RDMA, wired exactly as production disagg
// does (tensorrt_llm/_torch/disaggregation/native/transfer.py): one-directional AgentDesc exchange
// (senders load the receiver, never the reverse) and the receiver self-bootstraps each sender from
// its WANT — no manual addPeer, no connection-info path. NOTE: the KV src/dst buffers are
// intentionally NOT NIXL-registered, so the standard path's createXferReq would fail on them --
// a SUCCESS here proves the bounce fast path was taken (only the bounce arena is registered).

#include "bounceTestNixlNode.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;

using bounce_test::freeXferBufs;
using bounce_test::hasCuda;
using bounce_test::makeXferBufs;
using bounce_test::verifyXferBufs;
using bounce_test::XferBufs;

namespace
{
kvc::TransferRequest makeReq(XferBufs const& x, char const* dstPeer)
{
    return kvc::TransferRequest{kvc::TransferOp::kWRITE, x.srcDescs, x.dstDescs, dstPeer, std::nullopt};
}

// Poll a transfer status to a terminal state (bounce futures resolve once all chunks are
// scattered+ACKed); returns the last observed state (kIN_PROGRESS on deadline).
template <typename StatusPtr>
kvc::TransferState waitTerminal(StatusPtr& status, int seconds)
{
    auto st = kvc::TransferState::kIN_PROGRESS;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (std::chrono::steady_clock::now() < deadline)
    {
        st = status->wait(100);
        if (st != kvc::TransferState::kIN_PROGRESS)
        {
            break;
        }
    }
    return st;
}

// Fan out one transfer per (sender agent, dst peer) route, each on its own thread, and return how
// many completed SUCCESS. bufs[i] backs routes[i].
int runConcurrentFlows(
    std::vector<std::pair<kvc::NixlTransferAgent*, char const*>> const& routes, std::vector<XferBufs> const& bufs)
{
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.reserve(routes.size());
    for (std::size_t i = 0; i < routes.size(); ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto req = makeReq(bufs[i], routes[i].second);
                auto status = routes[i].first->submitTransferRequests(req);
                if (status == nullptr)
                {
                    return;
                }
                if (waitTerminal(status, 60) == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    return ok.load();
}

// Bounce-enabled agent config: agentBufferSizeMb switches bounce on, and the expert knobs ride
// bounceParams (dict > env > default) with thresholds tuned so a modest transfer engages bounce
// (small regions -> recycling). sizeMb == 0 keeps bounce off (no knobs attached).
kvc::BaseAgentConfig makeBounceConfig(
    std::string name, std::size_t arenaSizeMb = 2, char const* granularityBytes = "256")
{
    kvc::BaseAgentConfig cfg{std::move(name), true, false, true};
    cfg.agentBufferSizeMb = arenaSizeMb;
    if (arenaSizeMb > 0)
    {
        cfg.bounceParams = {
            {"min_descriptor_count", "4"},
            {"max_chunk_size", "4096"},
            {"arena_allocation_granularity", granularityBytes},
            {"max_inflight_chunks_per_request", "2"},
        };
    }
    return cfg;
}
} // namespace

// BaseAgentConfig::agentBufferSizeMb (from CacheTransceiverConfig's agent_buffer_size_mb) is the
// ONLY on/off switch: 0 keeps bounce disabled, >0 enables it. The legacy TRTLLM_NIXL_BOUNCE_ENABLE
// environment variable must have no effect anymore.
TEST(BounceAgentE2E, AgentBufferSizeControlsBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    setenv("TRTLLM_NIXL_BOUNCE_ENABLE", "1", 1);
    try
    {
        EXPECT_FALSE(std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("cfgOffAgent", 0))->isBounceEnabled());
        EXPECT_TRUE(std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("cfgOnAgent", 2))->isBounceEnabled());
        // size 0 + non-empty params: the params must be ignored (warned about, not honored) and
        // bounce must stay off. (The llm_args validator rejects this combination upfront; this
        // covers the direct BaseAgentConfig entry point.)
        auto orphanParams = makeBounceConfig("cfgOrphanAgent", 0);
        orphanParams.bounceParams = {{"copy_stream_count", "2"}};
        EXPECT_FALSE(std::make_unique<kvc::NixlTransferAgent>(orphanParams)->isBounceEnabled());
    }
    catch (std::exception const& e)
    {
        unsetenv("TRTLLM_NIXL_BOUNCE_ENABLE");
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }
    unsetenv("TRTLLM_NIXL_BOUNCE_ENABLE");
}

TEST(BounceAgentE2E, SubmitTransferRequestsUsesBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        a = std::make_unique<kvc::NixlTransferAgent>(
            makeBounceConfig("bAgentA", /*arenaSizeMb=*/1, /*granularityBytes=*/"4096"));
        b = std::make_unique<kvc::NixlTransferAgent>(
            makeBounceConfig("bAgentB", /*arenaSizeMb=*/1, /*granularityBytes=*/"4096"));
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // Connect exactly as production disagg does (tensorrt_llm/_torch/disaggregation/native/transfer.py):
    // the metadata exchange is ONE-DIRECTIONAL — only the KV sender loads the receiver's AgentDesc
    // (get_local_agent_desc / loadRemoteAgent(AgentDesc)); the receiver never loads the sender. The
    // bounce control endpoint rides inside that AgentDesc. B's reverse control path (GRANT/ACK back
    // to A) is self-bootstrapped from A's WANT (BounceReceiver::onWant) — exercising it
    // here is the whole point. (We deliberately do NOT touch the connection-info path.)
    a->loadRemoteAgent("bAgentB", b->getLocalAgentDesc());

    auto bufs = makeXferBufs(/*nDescs=*/24, /*descBytes=*/600, /*seed=*/7);
    auto req = makeReq(bufs, "bAgentB");
    auto status = a->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(waitTerminal(status, 30), kvc::TransferState::kSUCCESS)
        << "bounce transfer via submitTransferRequests did not succeed";
    EXPECT_TRUE(verifyXferBufs(bufs));

    a->shutdown();
    b->shutdown();
    freeXferBufs(bufs);
}

// A non-power-of-two arena can clamp the effective chunk cap below the configured cap. A descriptor
// between those limits must use ordinary NIXL instead of entering bounce and failing plan creation.
TEST(BounceAgentE2E, EffectiveChunkCapFallsBackToStandardNixl)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    // 3 MiB arena (non-power-of-two) -> buddy capacity is 2 MiB, below the configured 3 MiB chunk
    // cap. The descriptor below sits between the two limits.
    auto makeCapConfig = [](char const* name)
    {
        kvc::BaseAgentConfig cfg{name, true, false, true};
        cfg.agentBufferSizeMb = 3;
        cfg.bounceParams = {
            {"min_descriptor_count", "1"},
            {"max_chunk_size", "3145728"}, // configured 3 MiB
            {"arena_allocation_granularity", "256"},
            {"max_average_descriptor_size", "8MB"},
        };
        return cfg;
    };

    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        a = std::make_unique<kvc::NixlTransferAgent>(makeCapConfig("capAgentA"));
        b = std::make_unique<kvc::NixlTransferAgent>(makeCapConfig("capAgentB"));
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // 2.5 MiB: above the effective (buddy) 2 MiB cap, below the configured 3 MiB cap.
    auto bufs = makeXferBufs(/*nDescs=*/1, /*descBytes=*/2560 * 1024, /*seed=*/8);
    auto req = makeReq(bufs, "capAgentB");
    a->registerMemory(req.getSrcDescs());
    b->registerMemory(req.getDstDescs());
    a->loadRemoteAgent("capAgentB", b->getLocalAgentDesc());

    auto status = a->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(waitTerminal(status, 30), kvc::TransferState::kSUCCESS);
    EXPECT_EQ(a->getBounceSubmitCount(), 0);
    EXPECT_TRUE(verifyXferBufs(bufs));
    EXPECT_TRUE(status->release());
    status.reset();

    a->deregisterMemory(req.getSrcDescs());
    b->deregisterMemory(req.getDstDescs());
    a->shutdown();
    b->shutdown();
    freeXferBufs(bufs);
}

// Production-path CONCURRENCY: many threads call submitTransferRequests on the SAME sender agent at
// once (mirrors transfer.py's KV_TRANSFER_NUM_THREADS>1 worker pool fanning out to one receiver).
// Each gets its own seeded buffers; all must complete SUCCESS + land byte-exact with no cross-talk,
// no hang/deadlock — the production-API counterpart of the transport-level concurrency tests.
TEST(BounceAgentE2E, ConcurrentSubmitUsesBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        a = std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("cAgentA"));
        b = std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("cAgentB"));
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // One-directional bootstrap, exactly like transfer.py: only the sender (A) loads the receiver
    // (B); B self-bootstraps A from the first WANT (BounceReceiver::onWant). No manual addPeer — the agent's own
    // bounce transport is wired through loadRemoteAgent(AgentDesc).
    a->loadRemoteAgent("cAgentB", b->getLocalAgentDesc());

    constexpr int kThreads = 8;
    std::vector<XferBufs> bufs;
    std::vector<std::pair<kvc::NixlTransferAgent*, char const*>> routes;
    bufs.reserve(kThreads);
    routes.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs.push_back(makeXferBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(i + 1)));
        routes.emplace_back(a.get(), "cAgentB");
    }

    EXPECT_EQ(runConcurrentFlows(routes, bufs), kThreads) << "not all concurrent submitTransferRequests completed";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyXferBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    a->shutdown();
    b->shutdown();
    for (auto& x : bufs)
    {
        freeXferBufs(x);
    }
}

// Production-path BIDIRECTIONAL concurrency: two bounce-enabled agents each submit to the OTHER at
// once, so both arenas serve sender (gather) AND receiver (scatter) roles simultaneously. Both are
// senders here, so — exactly like transfer.py when two ranks each write to each other — each loads
// the other's AgentDesc; there is still NO manual addPeer (loadRemoteAgent wires each agent's own
// bounce transport). Every transfer must land byte-exact with no cross-talk or hang over the
// production API.
TEST(BounceAgentE2E, ConcurrentBidirectionalUsesBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        a = std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("biAgentA"));
        b = std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("biAgentB"));
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // Both agents send, so both load the other's AgentDesc (each direction's WANT self-bootstraps
    // the reverse control path anyway; the redundant load is harmless and mirrors a two-way flow).
    a->loadRemoteAgent("biAgentB", b->getLocalAgentDesc());
    b->loadRemoteAgent("biAgentA", a->getLocalAgentDesc());

    constexpr int kThreads = 8; // 4 flows A->B + 4 flows B->A, all concurrent
    std::vector<XferBufs> bufs;
    std::vector<std::pair<kvc::NixlTransferAgent*, char const*>> routes;
    bufs.reserve(kThreads);
    routes.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs.push_back(makeXferBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(i + 1)));
        // Even threads send A->B; odd threads send B->A (both arenas act as both roles).
        bool const a2b = (i % 2 == 0);
        routes.emplace_back(a2b ? a.get() : b.get(), a2b ? "biAgentB" : "biAgentA");
    }

    EXPECT_EQ(runConcurrentFlows(routes, bufs), kThreads) << "not all bidirectional submitTransferRequests completed";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyXferBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    a->shutdown();
    b->shutdown();
    for (auto& x : bufs)
    {
        freeXferBufs(x);
    }
}

// Production-path MULTI-AGENT (N>2): 1 receiver + S sender agents, every sender writing to the one
// receiver concurrently (the disagg "many context workers -> one gen" shape). This is the REAL
// one-directional bootstrap that transfer.py uses: each sender loads the receiver's AgentDesc; the
// receiver loads NOBODY and self-bootstraps every sender from its WANT (BounceReceiver::onWant) —
// so the reverse-control self-bootstrap is exercised across N distinct peers at once.
// Seed-distinct patterns -> any cross-talk fails; all must complete SUCCESS + land byte-exact.
TEST(BounceAgentE2E, MultiAgentManySendersToOneReceiver)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr int kSenders = 3; // total agents = 1 receiver + 3 senders
    std::string const recvName = "mnAgentR";

    std::unique_ptr<kvc::NixlTransferAgent> recv;
    std::vector<std::unique_ptr<kvc::NixlTransferAgent>> senders;
    try
    {
        recv = std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig(recvName));
        for (int i = 1; i <= kSenders; ++i)
        {
            senders.push_back(
                std::make_unique<kvc::NixlTransferAgent>(makeBounceConfig("mnAgentS" + std::to_string(i))));
        }
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // One-directional wiring: each sender loads the receiver; the receiver loads NOBODY. The
    // receiver only ever hears about a sender when that sender's WANT arrives (self-bootstrap).
    for (auto& s : senders)
    {
        s->loadRemoteAgent(recvName, recv->getLocalAgentDesc());
    }

    std::vector<XferBufs> bufs;
    std::vector<std::pair<kvc::NixlTransferAgent*, char const*>> routes;
    bufs.reserve(kSenders);
    routes.reserve(kSenders);
    for (int i = 0; i < kSenders; ++i)
    {
        bufs.push_back(makeXferBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(100 + i)));
        routes.emplace_back(senders[i].get(), recvName.c_str());
    }

    EXPECT_EQ(runConcurrentFlows(routes, bufs), kSenders) << "not all senders completed to the shared receiver";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyXferBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    for (auto& s : senders)
    {
        s->shutdown();
    }
    recv->shutdown();
    for (auto& x : bufs)
    {
        freeXferBufs(x);
    }
}
