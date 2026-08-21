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
// GRANTs (request timeout) and forgetPeer() while a transfer is in flight. Each test stands up real
// NixlTransferAgents (agent + UCX backend + metadata via getRawAgent()), registers the bounce arena
// on each raw agent, and drives standalone BounceTransports over ACTUAL NIXL RDMA.
//
// The happy-path / concurrency / bidirectional / multi-agent coverage lives in bounceAgentE2ETest,
// which drives the SAME pipeline through the production entry point (NixlTransferAgent::
// submitTransferRequests) with one-directional AgentDesc bootstrap, so it is not duplicated here.

#include "bounceTestNixlNode.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ZmqControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Node/XferBufs plumbing (hasCuda, makeXferBufs, verifyXferBufs, Node, makeNode, wirePair) is shared
// with bounceTransportTest / bounceTransportFailureTest via bounceTestNixlNode.h.
using namespace bounce_test;

// FAILURE PATH over the real stack: a request whose peer never GRANTs must FAIL on requestTimeout, not
// hang. The sender's WANT goes to a control endpoint with no live receiver transport (nobody grants),
// so checkTimeouts must resolve the future kFAILURE within ~requestTimeoutMs over the real
// ZmqControlChannel + NixlTransferEngine wiring.
TEST(BounceNixlE2E, NoGrantTimesOutNotHang)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr std::size_t kMaxChunkSizeBytes = 4096;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkSizeBytes / 256ULL);
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = kMaxChunkSizeBytes;
    cfg.arenaSizeBytes = 1 << 20;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.maxInflightChunksPerRequest = 2;
    cfg.scatterWorkerCount = 2;
    cfg.requestTimeoutMs = 1500; // short: a no-grant request must fail fast, not hang

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
// left the reactor healthy (a small arena would quickly expose a leaked allocation).
TEST(BounceNixlE2E, ForgetPeerInFlightRecovers)
{
    if (!hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr std::size_t kMaxChunkSizeBytes = 4096;
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, kMaxChunkSizeBytes / 256ULL);
    b::BounceConfig cfg;
    cfg.maxChunkSizeBytes = kMaxChunkSizeBytes;
    cfg.arenaSizeBytes = 1 << 20;
    cfg.arenaAllocationGranularityBytes = 256;
    cfg.maxInflightChunksPerRequest = 2;
    cfg.scatterWorkerCount = 2;
    cfg.requestTimeoutMs = 5000;

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
