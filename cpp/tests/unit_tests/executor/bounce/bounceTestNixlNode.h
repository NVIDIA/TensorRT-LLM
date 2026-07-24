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

#pragma once

// Shared helpers for the bounce v2 tests that drive the FULL pipeline over REAL NIXL RDMA: a "node"
// is a complete agent stack (NixlTransferAgent + NixlTransferEngine + arena + exec + zmq control
// channel + BounceTransport), plus seeded device buffers and a byte-exact verifier. Used by
// bounceTransportTest, bounceTransportFailureTest, and bounceNixlE2ETest so they share one NIXL
// setup (no per-file copy, and no LocalCopy loopback fake — the data plane is always real NIXL).

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/NixlTransferEngine.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace bounce_test
{

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

inline bool hasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

inline std::uint64_t alignUp(std::uint64_t v, std::uint64_t a)
{
    return (v + a - 1) / a * a;
}

// A `seed`-distinct byte pattern so concurrent transfers can't masquerade as each other.
inline unsigned char patSeed(std::uint32_t seed, std::size_t d, std::size_t i)
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

inline XferBufs makeXferBufs(std::uint32_t nDescs, std::uint32_t descBytes, std::uint32_t seed)
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

inline bool verifyXferBufs(XferBufs const& x)
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

inline void freeXferBufs(XferBufs& x)
{
    if (x.src)
    {
        cudaFree(x.src);
        x.src = nullptr;
    }
    if (x.dst)
    {
        cudaFree(x.dst);
        x.dst = nullptr;
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

// Build one full bounce node (agent+engine+arena+exec+channel+transport). Returns nullptr if the
// NIXL agent/backend can't init or the arena can't be registered (caller GTEST_SKIPs). `cfg` supplies
// arenaBytes/minBlock/windowDepth etc., so callers control the scheduler/arena sizing.
inline std::unique_ptr<Node> makeNode(std::string const& name, b::BounceConfig const& cfg, std::size_t maxDescs)
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
// OUTSIDE the agent, so — unlike production (bounceAgentE2ETest) — loadRemoteAgent cannot wire it:
//   - loadRemoteAgent(AgentDesc) exchanges only the NIXL metadata layer (so createXferReq can resolve
//     the remote arena). We use the AgentDesc path — the one production disagg uses
//     (tensorrt_llm/_torch/disaggregation/native/transfer.py) — NOT the connection-info path.
//   - addPeer() wires the control-channel layer (the DEALER to the peer's ROUTER) on the standalone
//     transport. Production folds this into loadRemoteAgent + WANT self-bootstrap; here it is manual.
inline void wirePair(Node& a, Node& b)
{
    a.agent->loadRemoteAgent(b.name, b.agent->getLocalAgentDesc());
    b.agent->loadRemoteAgent(a.name, a.agent->getLocalAgentDesc());
    a.tx->addPeer(b.name, b.ch->localEndpoint());
    b.tx->addPeer(a.name, a.ch->localEndpoint());
}

} // namespace bounce_test
