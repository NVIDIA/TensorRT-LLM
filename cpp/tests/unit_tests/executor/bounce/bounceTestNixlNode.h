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
// is a complete agent stack (NixlTransferAgent as the data plane + arena + exec + zmq control
// channel + BounceTransport), plus seeded device buffers and a byte-exact verifier. Used by
// bounceTransportTest, bounceTransportFailureTest, and bounceAgentE2ETest so they share one NIXL
// setup (no per-file copy, and no LocalCopy loopback fake — the data plane is always real NIXL).

#include "bounceTestUtils.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bounce_test
{

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

using BackendParams = std::unordered_map<std::string, std::string>;

// Each dedicated NIXL worker adds ~0.2 s to agent setup; production defaults to 8.
inline constexpr char kTestNixlThreads[] = "2";

// What the Python transceiver passes; a bounce agent then derives split_batch_size=1 itself.
inline BackendParams transceiverBackendParams()
{
    return {{"num_threads", kTestNixlThreads}};
}

// Every chunk write goes to one of NIXL's dedicated workers, as on a production bounce agent. makeNode's
// agents are not bounce agents (the transport is built beside them), so they set split_batch_size=1 here.
inline BackendParams dedicatedWorkerBackendParams()
{
    auto params = transceiverBackendParams();
    params.emplace("split_batch_size", "1");
    return params;
}

// The plugin defaults, where chunk writes run on NIXL's shared UCX worker.
inline BackendParams sharedWorkerBackendParams()
{
    return {};
}

// makeXferBufs starts every desc on this boundary, in src and dst alike.
inline constexpr std::uint64_t kDescAlignmentBytes = 256;
// The smallest hole makeXferBufsSized leaves before a desc that is not contiguous with its predecessor.
inline constexpr std::uint64_t kDescGapBytes = 64;

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
    std::vector<std::uint32_t> sizes;
    std::uint32_t seed{};
    std::vector<std::uint64_t> off; // per-desc offset, the same in src and dst
    std::uint64_t total{};
    kvc::TransferDescs srcDescs{kvc::MemoryType::kVRAM, {}};
    kvc::TransferDescs dstDescs{kvc::MemoryType::kVRAM, {}};
};

// Allocate src/dst of x.total bytes, seed the src pattern and build both desc lists from x.off and
// x.sizes. src and dst share the layout, so a desc contiguous with its predecessor in src is contiguous
// in dst too.
inline void allocXferBufs(XferBufs& x)
{
    EXPECT_EQ(cudaMalloc(&x.src, x.total), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&x.dst, x.total), cudaSuccess);
    std::vector<unsigned char> h(x.total, 0);
    for (std::size_t i = 0; i < x.sizes.size(); ++i)
    {
        for (std::uint32_t j = 0; j < x.sizes[i]; ++j)
        {
            h[x.off[i] + j] = patSeed(x.seed, i, j);
        }
    }
    EXPECT_EQ(cudaMemcpy(x.src, h.data(), x.total, cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(cudaMemset(x.dst, 0, x.total), cudaSuccess);
    auto addrAt = [](void* base, std::uint64_t offset)
    { return reinterpret_cast<std::uintptr_t>(static_cast<char*>(base) + offset); };
    std::vector<kvc::MemoryDesc> sd;
    std::vector<kvc::MemoryDesc> dd;
    for (std::size_t i = 0; i < x.sizes.size(); ++i)
    {
        sd.emplace_back(addrAt(x.src, x.off[i]), x.sizes[i], 0);
        dd.emplace_back(addrAt(x.dst, x.off[i]), x.sizes[i], 0);
    }
    x.srcDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(sd)};
    x.dstDescs = kvc::TransferDescs{kvc::MemoryType::kVRAM, std::move(dd)};
}

// `nDescs` descs of `descBytes` each, every one starting at the next kDescAlignmentBytes boundary (so
// descs whose size is a multiple of kDescAlignmentBytes are back to back and the planner may merge them).
inline XferBufs makeXferBufs(std::uint32_t nDescs, std::uint32_t descBytes, std::uint32_t seed)
{
    XferBufs x;
    x.sizes.assign(nDescs, descBytes);
    x.seed = seed;
    x.off.resize(nDescs);
    std::uint64_t cur = 0;
    for (std::uint32_t i = 0; i < nDescs; ++i)
    {
        x.off[i] = cur;
        cur = alignUp(cur + descBytes, kDescAlignmentBytes);
    }
    x.total = cur;
    allocXferBufs(x);
    return x;
}

// One desc per entry of `sizes`. contiguousWithPrev[i] (false when absent) places desc i right after
// desc i-1, in src and dst alike, so the planner may merge the two; otherwise a hole of at least
// kDescGapBytes separates them.
inline XferBufs makeXferBufsSized(
    std::vector<std::uint32_t> sizes, std::uint32_t seed, std::vector<bool> const& contiguousWithPrev = {})
{
    XferBufs x;
    x.sizes = std::move(sizes);
    x.seed = seed;
    x.off.resize(x.sizes.size());
    std::uint64_t end = 0;
    for (std::size_t i = 0; i < x.sizes.size(); ++i)
    {
        bool const contiguous = i > 0 && i < contiguousWithPrev.size() && contiguousWithPrev[i];
        x.off[i] = contiguous ? end : alignUp(end + kDescGapBytes, kDescAlignmentBytes);
        end = x.off[i] + x.sizes[i];
    }
    x.total = alignUp(end, kDescAlignmentBytes);
    allocXferBufs(x);
    return x;
}

inline bool verifyXferBufs(XferBufs const& x)
{
    std::vector<unsigned char> got(x.total, 0xEE);
    if (cudaMemcpy(got.data(), x.dst, x.total, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        return false;
    }
    for (std::size_t i = 0; i < x.sizes.size(); ++i)
    {
        for (std::uint32_t j = 0; j < x.sizes[i]; ++j)
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

// One bounce node = a full agent stack (agent + arena + exec + control channel + transport). The
// data plane is the agent itself (postXferRequest / registerRegionImpl); `agent` may be a test
// subclass of NixlTransferAgent that overrides postXferRequest to inject transfer faults.
struct Node
{
    std::string name;
    std::unique_ptr<kvc::NixlTransferAgent> agent;
    std::unique_ptr<b::BounceArena> arena;
    std::unique_ptr<b::ExecPool> exec;
    std::unique_ptr<b::ZmqControlChannel> ch;
    std::unique_ptr<b::BounceTransport> tx;

    ~Node()
    {
        tx.reset(); // join the transport threads before the arena/agent go away
        if (agent && arena)
        {
            agent->deregisterRegionImpl(arena->base(), arena->bytes(), /*deviceId=*/0);
        }
    }
};

// Build one full bounce node (agent+arena+exec+channel+transport). Returns nullptr if the NIXL
// agent/backend can't init or the arena can't be registered (caller GTEST_SKIPs). `cfg` supplies
// arenaSizeBytes/arenaAllocationGranularityBytes/maxInflightChunksPerRequest etc., so callers
// control the scheduler/arena sizing. `makeAgent` lets failure tests substitute a fault-injecting
// NixlTransferAgent subclass. `backendParams` configures the agent's NIXL backend.
inline std::unique_ptr<Node> makeNode(std::string const& name, b::BounceConfig const& cfg, std::size_t maxDescs,
    std::function<std::unique_ptr<kvc::NixlTransferAgent>(kvc::BaseAgentConfig const&)> const& makeAgent = nullptr,
    BackendParams const& backendParams = dedicatedWorkerBackendParams())
{
    auto n = std::make_unique<Node>();
    n->name = name;
    try
    {
        kvc::BaseAgentConfig c{name, /*useProgThread=*/true, /*multiThread=*/false, /*useListenThread=*/true};
        c.backendParams = backendParams;
        n->agent = makeAgent ? makeAgent(c) : std::make_unique<kvc::NixlTransferAgent>(c);
    }
    catch (std::exception const&)
    {
        return nullptr;
    }
    n->arena = std::make_unique<b::BounceArena>(cfg.arenaSizeBytes, 0, /*allowFabric=*/false);
    n->exec = std::make_unique<b::ExecPool>(cfg.maxInflightChunksPerRequest + 4, maxDescs, 0, cfg.useZeroCopyArguments);
    if (!n->agent->registerRegionImpl(n->arena->base(), n->arena->bytes(), /*deviceId=*/0))
    {
        return nullptr;
    }
    n->ch = std::make_unique<b::ZmqControlChannel>(name);
    n->tx
        = std::make_unique<b::BounceTransport>(n->name, cfg, 0, n->ch.get(), *n->agent, n->arena.get(), n->exec.get());
    return n;
}

// Bidirectional connect for the white-box harness. Two SEPARATE wirings are needed here because the
// transport under test is hand-built (b::BounceTransport with its own b::ZmqControlChannel) and lives
// OUTSIDE the agent, so loadRemoteAgent cannot wire this standalone transport:
//   - loadRemoteAgent(AgentDesc) exchanges only the NIXL metadata layer (so createXferReq can resolve
//     the remote arena). The connection-info overload would not carry the structured metadata.
//   - addPeer() wires the control-channel layer (the DEALER to the peer's ROUTER) on the standalone
//     transport. NixlTransferAgent normally folds this into handshake registration plus WANT
//     self-bootstrap; here it is manual.
inline void wirePair(Node& a, Node& b)
{
    a.agent->loadRemoteAgent(b.name, b.agent->getLocalAgentDesc());
    b.agent->loadRemoteAgent(a.name, a.agent->getLocalAgentDesc());
    a.tx->addPeer(b.name, b.ch->localEndpoint());
    b.tx->addPeer(a.name, a.ch->localEndpoint());
}

struct NodePair
{
    std::unique_ptr<Node> sender;
    std::unique_ptr<Node> receiver;
};

// Sender `tag`+"A" built from `senderCfg` and receiver `tag`+"B" from `receiverCfg`, wired to each other.
// nullopt when either node can't be built (caller GTEST_SKIPs).
inline std::optional<NodePair> makeWiredPair(std::string const& tag, b::BounceConfig const& senderCfg,
    b::BounceConfig const& receiverCfg, BackendParams const& backendParams = dedicatedWorkerBackendParams())
{
    auto sender = makeNode(tag + "A", senderCfg, b::maxDescsPerChunk(senderCfg), /*makeAgent=*/nullptr, backendParams);
    auto receiver
        = makeNode(tag + "B", receiverCfg, b::maxDescsPerChunk(receiverCfg), /*makeAgent=*/nullptr, backendParams);
    if (!sender || !receiver)
    {
        return std::nullopt;
    }
    wirePair(*sender, *receiver);
    return NodePair{std::move(sender), std::move(receiver)};
}

// Both nodes built from the same `cfg`.
inline std::optional<NodePair> makeWiredPair(std::string const& tag, b::BounceConfig const& cfg,
    BackendParams const& backendParams = dedicatedWorkerBackendParams())
{
    return makeWiredPair(tag, cfg, cfg, backendParams);
}

} // namespace bounce_test
