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
// with TRTLLM_NIXL_BOUNCE_ENABLE=1. Real agents (2..N), real RDMA, wired exactly as production disagg
// does (tensorrt_llm/_torch/disaggregation/native/transfer.py): one-directional AgentDesc exchange
// (senders load the receiver, never the reverse) and the receiver self-bootstraps each sender from
// its WANT — no manual addPeer, no connection-info path. NOTE: the KV src/dst buffers are
// intentionally NOT NIXL-registered, so the standard path's createXferReq would fail on them --
// a SUCCESS here proves the bounce fast path was taken (only the bounce arena is registered).

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

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
    return static_cast<unsigned char>((d * 211 + i * 17 + 3) & 0xFF);
}

unsigned char patSeed(std::uint32_t seed, std::size_t d, std::size_t i)
{
    return static_cast<unsigned char>((seed * 101 + d * 211 + i * 17 + 3) & 0xFF);
}

// One transfer's device buffers + descriptors (KV buffers, intentionally NOT NIXL-registered).
struct AgentBufs
{
    void* src{nullptr};
    void* dst{nullptr};
    std::uint32_t nDescs{};
    std::uint32_t descBytes{};
    std::uint32_t seed{};
    std::vector<std::uint64_t> off;
    std::uint64_t total{};
};

AgentBufs makeAgentBufs(std::uint32_t nDescs, std::uint32_t descBytes, std::uint32_t seed)
{
    AgentBufs x;
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
    return x;
}

kvc::TransferRequest makeReq(AgentBufs const& x, char const* dstPeer)
{
    auto a2 = [](void* p, std::uint64_t o) { return reinterpret_cast<std::uintptr_t>(static_cast<char*>(p) + o); };
    std::vector<kvc::MemoryDesc> sd;
    std::vector<kvc::MemoryDesc> dd;
    for (std::uint32_t i = 0; i < x.nDescs; ++i)
    {
        sd.emplace_back(a2(x.src, x.off[i]), x.descBytes, 0);
        dd.emplace_back(a2(x.dst, x.off[i]), x.descBytes, 0);
    }
    return kvc::TransferRequest{kvc::TransferOp::kWRITE, kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(sd)},
        kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(dd)}, dstPeer, std::nullopt};
}

bool verifyAgentBufs(AgentBufs const& x)
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

void setBounceEnv()
{
    setenv("TRTLLM_NIXL_BOUNCE_ENABLE", "1", 1);
    setenv("TRTLLM_NIXL_BOUNCE_MIN_DESC", "4", 1);
    setenv("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES", "4096", 1);
    setenv("TRTLLM_NIXL_BOUNCE_ARENA_BYTES", "2097152", 1); // 2 MiB
    setenv("TRTLLM_NIXL_BOUNCE_MIN_BLOCK", "256", 1);
    setenv("TRTLLM_NIXL_BOUNCE_DEPTH", "2", 1);
}

void clearBounceEnv()
{
    unsetenv("TRTLLM_NIXL_BOUNCE_ENABLE");
    unsetenv("TRTLLM_NIXL_BOUNCE_MIN_DESC");
    unsetenv("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES");
    unsetenv("TRTLLM_NIXL_BOUNCE_ARENA_BYTES");
    unsetenv("TRTLLM_NIXL_BOUNCE_MIN_BLOCK");
    unsetenv("TRTLLM_NIXL_BOUNCE_DEPTH");
}
} // namespace

TEST(BounceAgentE2E, SubmitTransferRequestsUsesBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    // Enable bounce + tune thresholds so a modest transfer engages it (small regions -> recycling).
    setenv("TRTLLM_NIXL_BOUNCE_ENABLE", "1", 1);
    setenv("TRTLLM_NIXL_BOUNCE_MIN_DESC", "4", 1);
    setenv("TRTLLM_NIXL_BOUNCE_MAX_CHUNK_BYTES", "4096", 1); // per-chunk byte cap
    setenv("TRTLLM_NIXL_BOUNCE_ARENA_BYTES", "1048576", 1);  // 1 MiB shared arena
    setenv("TRTLLM_NIXL_BOUNCE_MIN_BLOCK", "4096", 1);       // buddy granularity == chunk cap
    setenv("TRTLLM_NIXL_BOUNCE_DEPTH", "2", 1);              // per-flow window

    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        kvc::BaseAgentConfig ca{"bAgentA", true, false, true};
        kvc::BaseAgentConfig cb{"bAgentB", true, false, true};
        a = std::make_unique<kvc::NixlTransferAgent>(ca);
        b = std::make_unique<kvc::NixlTransferAgent>(cb);
    }
    catch (std::exception const& e)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // Connect exactly as production disagg does (tensorrt_llm/_torch/disaggregation/native/transfer.py):
    // the metadata exchange is ONE-DIRECTIONAL — only the KV sender loads the receiver's AgentDesc
    // (get_local_agent_desc / loadRemoteAgent(AgentDesc)); the receiver never loads the sender. The
    // bounce control endpoint rides inside that AgentDesc. B's reverse control path (GRANT/ACK back
    // to A) is self-bootstrapped from A's WANT (BounceReceiver::onWant, DESIGN §7) — exercising it
    // here is the whole point. (We deliberately do NOT touch the connection-info path.)
    a->loadRemoteAgent("bAgentB", b->getLocalAgentDesc());

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

    void* dSrc = nullptr;
    void* dDst = nullptr;
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
    kvc::TransferRequest req{kvc::TransferOp::kWRITE, kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(sd)},
        kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(dd)}, "bAgentB", std::nullopt};

    auto status = a->submitTransferRequests(req);
    ASSERT_NE(status, nullptr);
    // Poll to completion (bounce future resolves SUCCESS once all chunks are scattered+ACKed).
    kvc::TransferState st = kvc::TransferState::kIN_PROGRESS;
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline)
    {
        st = status->wait(100);
        if (st != kvc::TransferState::kIN_PROGRESS)
        {
            break;
        }
    }
    EXPECT_EQ(st, kvc::TransferState::kSUCCESS) << "bounce transfer via submitTransferRequests did not succeed";

    std::vector<unsigned char> got(total, 0xEE);
    ASSERT_EQ(cudaMemcpy(got.data(), dDst, total, cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::uint32_t i = 0; i < kNDescs; ++i)
    {
        for (std::uint32_t j = 0; j < kDescBytes; ++j)
        {
            ASSERT_EQ(got[off[i] + j], pattern(i, j)) << "mismatch desc=" << i << " byte=" << j;
        }
    }

    a->shutdown();
    b->shutdown();
    cudaFree(dSrc);
    cudaFree(dDst);
    clearBounceEnv();
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
    setBounceEnv();

    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        kvc::BaseAgentConfig ca{"cAgentA", true, false, true};
        kvc::BaseAgentConfig cb{"cAgentB", true, false, true};
        a = std::make_unique<kvc::NixlTransferAgent>(ca);
        b = std::make_unique<kvc::NixlTransferAgent>(cb);
    }
    catch (std::exception const& e)
    {
        clearBounceEnv();
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // One-directional bootstrap, exactly like transfer.py: only the sender (A) loads the receiver
    // (B); B self-bootstraps A from the first WANT (DESIGN §7). No manual addPeer — the agent's own
    // bounce transport is wired through loadRemoteAgent(AgentDesc).
    a->loadRemoteAgent("cAgentB", b->getLocalAgentDesc());

    constexpr int kThreads = 8;
    std::vector<AgentBufs> bufs;
    bufs.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs.push_back(makeAgentBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(i + 1)));
    }

    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto req = makeReq(bufs[i], "cAgentB");
                auto status = a->submitTransferRequests(req);
                if (status == nullptr)
                {
                    return;
                }
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                kvc::TransferState st = kvc::TransferState::kIN_PROGRESS;
                while (std::chrono::steady_clock::now() < deadline)
                {
                    st = status->wait(100);
                    if (st != kvc::TransferState::kIN_PROGRESS)
                    {
                        break;
                    }
                }
                if (st == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(ok.load(), kThreads) << "not all concurrent submitTransferRequests completed";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyAgentBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    a->shutdown();
    b->shutdown();
    for (auto& x : bufs)
    {
        cudaFree(x.src);
        cudaFree(x.dst);
    }
    clearBounceEnv();
}

// Production-path BIDIRECTIONAL concurrency: two bounce-enabled agents each submit to the OTHER at
// once, so both arenas serve sender (gather) AND receiver (scatter) roles simultaneously. Both are
// senders here, so — exactly like transfer.py when two ranks each write to each other — each loads
// the other's AgentDesc; there is still NO manual addPeer (loadRemoteAgent wires each agent's own
// bounce transport). Every transfer must land byte-exact with no cross-talk / hang (R2/R3/R4 over
// the production API).
TEST(BounceAgentE2E, ConcurrentBidirectionalUsesBounce)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    setBounceEnv();

    std::unique_ptr<kvc::NixlTransferAgent> a;
    std::unique_ptr<kvc::NixlTransferAgent> b;
    try
    {
        kvc::BaseAgentConfig ca{"biAgentA", true, false, true};
        kvc::BaseAgentConfig cb{"biAgentB", true, false, true};
        a = std::make_unique<kvc::NixlTransferAgent>(ca);
        b = std::make_unique<kvc::NixlTransferAgent>(cb);
    }
    catch (std::exception const& e)
    {
        clearBounceEnv();
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // Both agents send, so both load the other's AgentDesc (each direction's WANT self-bootstraps
    // the reverse control path anyway; the redundant load is harmless and mirrors a two-way flow).
    a->loadRemoteAgent("biAgentB", b->getLocalAgentDesc());
    b->loadRemoteAgent("biAgentA", a->getLocalAgentDesc());

    constexpr int kThreads = 8; // 4 flows A->B + 4 flows B->A, all concurrent
    std::vector<AgentBufs> bufs;
    bufs.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        bufs.push_back(makeAgentBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(i + 1)));
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
                auto* tx = a2b ? a.get() : b.get();
                char const* peer = a2b ? "biAgentB" : "biAgentA";
                auto req = makeReq(bufs[i], peer);
                auto status = tx->submitTransferRequests(req);
                if (status == nullptr)
                {
                    return;
                }
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                kvc::TransferState st = kvc::TransferState::kIN_PROGRESS;
                while (std::chrono::steady_clock::now() < deadline)
                {
                    st = status->wait(100);
                    if (st != kvc::TransferState::kIN_PROGRESS)
                    {
                        break;
                    }
                }
                if (st == kvc::TransferState::kSUCCESS)
                {
                    ok.fetch_add(1);
                }
            });
    }
    for (auto& t : threads)
    {
        t.join();
    }
    EXPECT_EQ(ok.load(), kThreads) << "not all bidirectional submitTransferRequests completed";
    for (auto const& x : bufs)
    {
        EXPECT_TRUE(verifyAgentBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    a->shutdown();
    b->shutdown();
    for (auto& x : bufs)
    {
        cudaFree(x.src);
        cudaFree(x.dst);
    }
    clearBounceEnv();
}

// Production-path MULTI-AGENT (N>2): 1 receiver + S sender agents, every sender writing to the one
// receiver concurrently (the disagg "many context workers -> one gen" shape). This is the REAL
// one-directional bootstrap that transfer.py uses: each sender loads the receiver's AgentDesc; the
// receiver loads NOBODY and self-bootstraps every sender from its WANT (BounceReceiver::onWant,
// DESIGN §7) — so the reverse-control self-bootstrap is exercised across N distinct peers at once.
// Seed-distinct patterns -> any cross-talk fails; all must complete SUCCESS + land byte-exact.
TEST(BounceAgentE2E, MultiAgentManySendersToOneReceiver)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    setBounceEnv();

    constexpr int kSenders = 3; // total agents = 1 receiver + 3 senders
    std::string const recvName = "mnAgentR";

    std::unique_ptr<kvc::NixlTransferAgent> recv;
    std::vector<std::unique_ptr<kvc::NixlTransferAgent>> senders;
    try
    {
        recv = std::make_unique<kvc::NixlTransferAgent>(kvc::BaseAgentConfig{recvName, true, false, true});
        for (int i = 1; i <= kSenders; ++i)
        {
            senders.push_back(std::make_unique<kvc::NixlTransferAgent>(
                kvc::BaseAgentConfig{"mnAgentS" + std::to_string(i), true, false, true}));
        }
    }
    catch (std::exception const& e)
    {
        clearBounceEnv();
        GTEST_SKIP() << "NIXL agent/backend unavailable: " << e.what();
    }

    // One-directional wiring: each sender loads the receiver; the receiver loads NOBODY. The
    // receiver only ever hears about a sender when that sender's WANT arrives (self-bootstrap).
    for (auto& s : senders)
    {
        s->loadRemoteAgent(recvName, recv->getLocalAgentDesc());
    }

    std::vector<AgentBufs> bufs;
    bufs.reserve(kSenders);
    for (int i = 0; i < kSenders; ++i)
    {
        bufs.push_back(makeAgentBufs(/*nDescs=*/16, /*descBytes=*/500, /*seed=*/static_cast<std::uint32_t>(100 + i)));
    }

    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    threads.reserve(kSenders);
    for (int i = 0; i < kSenders; ++i)
    {
        threads.emplace_back(
            [&, i]
            {
                auto req = makeReq(bufs[i], recvName.c_str());
                auto status = senders[i]->submitTransferRequests(req);
                if (status == nullptr)
                {
                    return;
                }
                auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                kvc::TransferState st = kvc::TransferState::kIN_PROGRESS;
                while (std::chrono::steady_clock::now() < deadline)
                {
                    st = status->wait(100);
                    if (st != kvc::TransferState::kIN_PROGRESS)
                    {
                        break;
                    }
                }
                if (st == kvc::TransferState::kSUCCESS)
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
        EXPECT_TRUE(verifyAgentBufs(x)) << "byte mismatch / cross-talk for seed=" << x.seed;
    }

    for (auto& s : senders)
    {
        s->shutdown();
    }
    recv->shutdown();
    for (auto& x : bufs)
    {
        cudaFree(x.src);
        cudaFree(x.dst);
    }
    clearBounceEnv();
}
