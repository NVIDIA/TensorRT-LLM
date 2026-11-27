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
// fails, forgetPeer() drops a peer mid-transfer, or shutdown races in-flight requests. Plus a
// multi-threaded concurrent-submit test.
// The data plane is real NIXL (via bounceTestNixlNode helpers); the one exception is the
// transfer-failure path, which uses a FakeXferAgent (a NixlTransferAgent subclass overriding the
// postXferRequest primitive) because real NIXL cannot be made to deterministically fail a write.
//
// The happy-path / concurrency / bidirectional / multi-agent coverage lives in bounceAgentE2ETest,
// which drives the SAME pipeline through the production entry point (NixlTransferAgent::
// submitTransferRequests) with one-directional AgentDesc bootstrap, so it is not duplicated here.

#include "bounceTestNixlNode.h"

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
// Test-controlled behavior for every write a FakeXferAgent posts. Shared by the agent and all the
// statuses it hands out.
struct XferControls
{
    std::atomic<bool> useNixl{false};
    std::atomic<bool> failWrites{false};     // every posted write polls as FAILURE
    std::atomic<bool> allowTerminal{true};   // false -> posted writes stay IN_PROGRESS
    std::atomic<bool> releaseSucceeds{true}; // false -> release() fails (handle retained)
    std::atomic<std::uint64_t> postCount{0};
    std::atomic<std::uint64_t> releaseCount{0};
};

// A TransferStatus driven by XferControls instead of a real backend handle. Unlike
// NixlTransferStatus it does NOT release in its destructor, keeping releaseCount deterministic.
class FakeXferStatus final : public kvc::TransferStatus
{
public:
    explicit FakeXferStatus(std::shared_ptr<XferControls> ctl)
        : mCtl(std::move(ctl))
    {
    }

    [[nodiscard]] bool isCompleted() const override
    {
        return wait(0) != kvc::TransferState::kIN_PROGRESS;
    }

    [[nodiscard]] kvc::TransferState wait(int64_t) const override
    {
        if (mCtl->failWrites.load(std::memory_order_acquire))
        {
            return kvc::TransferState::kFAILURE;
        }
        return mCtl->allowTerminal.load(std::memory_order_acquire) ? kvc::TransferState::kSUCCESS
                                                                   : kvc::TransferState::kIN_PROGRESS;
    }

    [[nodiscard]] bool release() override
    {
        if (mReleased)
        {
            return true;
        }
        mCtl->releaseCount.fetch_add(1, std::memory_order_relaxed);
        if (!mCtl->releaseSucceeds.load(std::memory_order_acquire))
        {
            return false;
        }
        mReleased = true;
        return true;
    }

private:
    std::shared_ptr<XferControls> mCtl;
    bool mReleased{false};
};

// The fault-injection seam: a REAL NixlTransferAgent (control plane / metadata untouched) whose
// low-level write primitive is overridden per XferControls — real NIXL cannot be coerced into a
// deterministic write failure.
class FakeXferAgent final : public kvc::NixlTransferAgent
{
public:
    FakeXferAgent(kvc::BaseAgentConfig const& config, std::shared_ptr<XferControls> ctl)
        : kvc::NixlTransferAgent(config)
        , mCtl(std::move(ctl))
    {
    }

    [[nodiscard]] std::unique_ptr<kvc::TransferStatus> postXferRequest(kvc::TransferOp op,
        kvc::TransferDescs const& local, kvc::TransferDescs const& remote, std::string const& peer,
        std::optional<kvc::SyncMessage> const& notification) override
    {
        mCtl->postCount.fetch_add(1, std::memory_order_relaxed);
        if (mCtl->useNixl.load(std::memory_order_acquire))
        {
            return kvc::NixlTransferAgent::postXferRequest(op, local, remote, peer, notification);
        }
        return std::make_unique<FakeXferStatus>(mCtl);
    }

private:
    std::shared_ptr<XferControls> mCtl;
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

// Shrink the arena to exactly `regionCap` max-size regions. Matching the allocation granularity to
// the maximum chunk size makes every region one buddy block.
void capRegions(b::BounceConfig& c, std::uint32_t regionCap)
{
    c.arenaAllocationGranularityBytes = c.maxChunkSizeBytes;
    std::size_t arenaSizeBytes = c.arenaAllocationGranularityBytes;
    while (arenaSizeBytes < static_cast<std::size_t>(regionCap) * c.maxChunkSizeBytes)
    {
        arenaSizeBytes <<= 1;
    }
    c.arenaSizeBytes = arenaSizeBytes;
}

// makeNode with a FakeXferAgent wired to `ctl`.
std::unique_ptr<bounce_test::Node> makeFakeNode(
    std::string const& name, b::BounceConfig const& c, std::shared_ptr<XferControls> ctl, std::size_t maxDescs = 1024)
{
    return bounce_test::makeNode(name, c, maxDescs,
        [&ctl](kvc::BaseAgentConfig const& agentConfig) { return std::make_unique<FakeXferAgent>(agentConfig, ctl); });
}

} // namespace

// A peer that never GRANTs must fail the request on requestTimeoutMs, not hang. The WANT is
// DELIVERED to a live ROUTER that no transport ever recv()s on (a "ghost"), so this covers the
// stronger case (WANT accepted, nobody grants) as well as the weaker unknown-peer one (WANT
// dropped at send) — checkTimeouts must resolve the future kFAILURE either way.
TEST(BounceTransportFailure, NoGrantTimesOutNotHang)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/500);
    auto t = bounce_test::makeNode("ngSolo", c, 1024);
    if (!t)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    // A bound ROUTER that no transport ever recv()s on -> the WANT is delivered but never granted.
    b::ZmqControlChannel ghost("ghostPeer");
    t->tx->addPeer("ghostPeer", ghost.localEndpoint());

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "ghostPeer", bounce_test::testWindowKey());
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "request hung with no grant";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kNoProgressTimeout);

    t->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// A backend write failure settles the request. The same sender and receiver can service the next transfer.
TEST(BounceTransportFailure, WriteFailureFailsRequest)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    capRegions(c, c.maxInflightChunksPerRequest);
    auto ctl = std::make_shared<XferControls>();
    ctl->failWrites.store(true, std::memory_order_release);
    auto A = makeFakeNode("feA", c, ctl);
    auto B = bounce_test::makeNode("feB", c, 1024);
    if (!A || !B)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*A, *B);

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = bounce_test::submitTransfer(*A, *B, bufs);
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "write failure hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kWriteFailed);
    EXPECT_GE(ctl->postCount.load(std::memory_order_acquire), 1u);

    ctl->useNixl.store(true, std::memory_order_release);
    auto followup = bounce_test::makeXferBufs(8, 256, /*seed=*/2);
    auto recovered = bounce_test::submitTransfer(*A, *B, followup);
    ASSERT_EQ(recovered.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(recovered.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(followup));

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
    bounce_test::freeXferBufs(followup);
}

class BounceCapacityFailure : public ::testing::TestWithParam<bool>
{
};

TEST_P(BounceCapacityFailure, ResolvesAndReleasesTheRegionForTheNextRequest)
{
    if (!bounce_test::hasCuda())
    {
        GTEST_SKIP() << "no CUDA device";
    }
    bool const senderLimited = GetParam();
    auto config = cfg(senderLimited ? 1000 : 20000);
    capRegions(config, 1);
    auto controls = std::make_shared<XferControls>();
    controls->useNixl.store(true, std::memory_order_release);
    auto sender = makeFakeNode("capacitySender", config, controls, senderLimited ? 1 : 8);
    auto receiver = bounce_test::makeNode("capacityReceiver", config, senderLimited ? 8 : 1);
    if (!sender || !receiver)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*sender, *receiver);

    // Two non-contiguous descriptors cannot share a single scratch entry.
    auto twoPieces = bounce_test::makeXferBufs(2, 128, 1);
    auto const freeContexts = sender->exec->freeCount();
    auto const start = std::chrono::steady_clock::now();
    auto refused = bounce_test::submitTransfer(*sender, *receiver, twoPieces);
    ASSERT_EQ(refused.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(refused.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(
        refused.get().reason, senderLimited ? b::BounceFailReason::kProtocolError : b::BounceFailReason::kPeerNack);
    EXPECT_EQ(controls->postCount.load(), senderLimited ? 0U : 1U);
    if (!senderLimited)
    {
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(config.requestTimeoutMs / 2));
    }
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (sender->exec->freeCount() != freeContexts && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
    EXPECT_EQ(sender->exec->freeCount(), freeContexts);

    auto onePiece = bounce_test::makeXferBufs(1, 128, 2);
    auto accepted = bounce_test::submitTransfer(*sender, *receiver, onePiece);
    ASSERT_EQ(accepted.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(accepted.get().state, kvc::TransferState::kSUCCESS);
    EXPECT_TRUE(bounce_test::verifyXferBufs(onePiece));

    sender->tx->shutdown();
    receiver->tx->shutdown();
    bounce_test::freeXferBufs(twoPieces);
    bounce_test::freeXferBufs(onePiece);
}

INSTANTIATE_TEST_SUITE_P(SenderOrReceiver, BounceCapacityFailure, ::testing::Bool());

namespace tensorrt_llm::executor::kv_cache::bounce
{
class BounceSenderStateTest : public ::testing::Test
{
protected:
    using Table = BounceSender::RequestTable;
    using Request = BounceSender::Request;
    using PostState = BounceSender::PostState;

    static auto reply(Table& requests, std::string const& peer, std::uint64_t rid)
    {
        return BounceSender::findReply(requests, peer, rid);
    }
};

TEST_F(BounceSenderStateTest, OnlyTheMatchingSentChunkAdvancesTheRequest)
{
    Table requests;
    auto& request = requests[1];
    request.peer = "receiver";
    request.numChunks = 1;
    auto& post = request.posted.emplace_back();
    post.chunkIdx = 0;
    post.remoteHandle = 42;
    auto released = 0;
    auto recycle = [&](auto const&) { ++released; };
    EXPECT_EQ(reply(requests, "other", 1), requests.end());
    EXPECT_EQ(reply(requests, "receiver", 2), requests.end());
    ASSERT_NE(reply(requests, "receiver", 1), requests.end());

    EXPECT_FALSE(request.acknowledge(0, 42, recycle));
    post.state = PostState::Sent;
    EXPECT_FALSE(request.acknowledge(0, 43, recycle));
    EXPECT_FALSE(request.acknowledge(1, 42, recycle));
    EXPECT_EQ(released, 0);
    EXPECT_EQ(request.acked, 0U);
    EXPECT_TRUE(request.acknowledge(0, 42, recycle));
    EXPECT_FALSE(request.acknowledge(0, 42, recycle));
    EXPECT_EQ(released, 1);
    EXPECT_EQ(request.acked, request.numChunks);
    EXPECT_TRUE(request.posted.empty());
}

TEST_F(BounceSenderStateTest, CreditSizeFailureRemainsTerminal)
{
    Request request;
    EXPECT_TRUE(request.acceptsCredit(256, 256));
    EXPECT_EQ(request.failureReason(BounceFailReason::kNoProgressTimeout), BounceFailReason::kNoProgressTimeout);
    request.pendingCredits.emplace_back();
    EXPECT_FALSE(request.acceptsCredit(256, 128));
    EXPECT_TRUE(request.pendingCredits.empty());
    request.pendingCredits.emplace_back();
    EXPECT_FALSE(request.canPost());
    EXPECT_FALSE(request.acceptsCredit(256, 256));
    EXPECT_TRUE(request.pendingCredits.empty());
    EXPECT_EQ(request.failureReason(BounceFailReason::kNoProgressTimeout), BounceFailReason::kProtocolError);
}
} // namespace tensorrt_llm::executor::kv_cache::bounce

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
    auto fut = t->tx->submit(
        bufs.srcDescs, bufs.dstDescs, "nobody", bounce_test::testWindowKey()); // stuck (no grant, no timeout)
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
    t->tx->shutdown();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "shutdown left request hanging";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kShutdown);

    bounce_test::freeXferBufs(bufs);
}

// Shutdown uses the status object's bounded release instead of polling forever. If the backend
// cannot abort, the future still fails promptly; the release is attempted once in failAll and once
// in its retry pass (the production NixlTransferStatus destructor makes a final attempt).
TEST(BounceTransportFailure, ShutdownReleaseFailureDoesNotHangOrLoseHandle)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/0);
    capRegions(c, c.maxInflightChunksPerRequest);
    auto ctl = std::make_shared<XferControls>();
    ctl->allowTerminal.store(false, std::memory_order_release);
    ctl->releaseSucceeds.store(false, std::memory_order_release);
    auto sender = makeFakeNode("shutdownReleaseSender", c, ctl);
    auto receiver = bounce_test::makeNode("shutdownReleasePeer", c, 1024);
    if (!sender || !receiver)
    {
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    }
    bounce_test::wirePair(*sender, *receiver);

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/5);
    auto fut = bounce_test::submitTransfer(*sender, *receiver, bufs);

    auto const postDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < postDeadline && ctl->postCount.load(std::memory_order_acquire) == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(ctl->postCount.load(std::memory_order_acquire), 1u);

    auto const start = std::chrono::steady_clock::now();
    sender->tx->shutdown();
    // A write that never completes is drained for the fixed kShutdownWriteDrainMs (5 s) — intended
    // teardown behaviour — so the bound only asserts "bounded, no hang", not "instant".
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(10));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kShutdown);
    // failAll attempts the failed release exactly twice (initial + bounded retry), never spins.
    EXPECT_EQ(ctl->releaseCount.load(std::memory_order_acquire), 2u);
    receiver->tx->shutdown();
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
    auto fut
        = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "gonePeer", bounce_test::testWindowKey()); // stuck: never granted
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);        // still pending
    t->tx->forgetPeer("gonePeer");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "forgetPeer left request hanging";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kPeerDropped);

    // forgetPeer for an unrelated/unknown peer must be a harmless no-op (no crash).
    t->tx->forgetPeer("someoneElse");
    t->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// forgetPeer() while a transfer is in flight over REAL RDMA must not hang, leak, or corrupt. We
// submit then immediately forgetPeer the target; the request must RESOLVE (SUCCESS if it beat the
// queued reclaim, else FAILURE) — never hang. Then several FRESH transfers to the same peer must
// still complete byte-exact, proving forgetPeer's reclaim returned the regions to the arena and
// left the reactor healthy (a small arena would quickly expose a leaked allocation).
TEST(BounceTransportFailure, ForgetPeerInFlightRecovers)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    std::size_t const maxDescs = std::max<std::size_t>(1024ULL, c.maxChunkSizeBytes / 256ULL);
    auto A = bounce_test::makeNode("fpA", c, maxDescs);
    auto B = bounce_test::makeNode("fpB", c, maxDescs);
    if (!A || !B)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*A, *B);

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/24, /*descBytes=*/600, /*seed=*/1);
    auto fut = bounce_test::submitTransfer(*A, *B, bufs);
    A->tx->forgetPeer("fpB"); // drop the peer (queued; applied on A's IO thread)
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "request hung after forgetPeer";
    auto const res = fut.get();
    EXPECT_TRUE(res.state == kvc::TransferState::kSUCCESS || res.state == kvc::TransferState::kFAILURE)
        << "unexpected state " << static_cast<int>(res.state);
    if (res.state == kvc::TransferState::kFAILURE)
    {
        EXPECT_EQ(res.reason, b::BounceFailReason::kPeerDropped);
    }
    bounce_test::freeXferBufs(bufs);

    // Recovery + no-leak: forgetPeer is a one-shot event; the scheduler/request reclaim is drained by
    // the time fut resolved, so these new flows aren't reclaimed. forgetPeer ALSO drops the
    // control-channel DEALER to fpB synchronously (on this thread), so re-establish it with addPeer
    // before recovering — deterministic because forgetPeer's removePeer happens-before this addPeer
    // (no async removePeer can race/erase the freshly re-added dealer). NIXL metadata persists, so
    // only the dealer is re-added (no loadRemoteAgent / full re-wire).
    A->tx->addPeer("fpB", B->ch->localEndpoint());
    for (int k = 0; k < 5; ++k)
    {
        auto rb
            = bounce_test::makeXferBufs(/*nDescs=*/20, /*descBytes=*/600, /*seed=*/static_cast<std::uint32_t>(50 + k));
        auto rf = bounce_test::submitTransfer(*A, *B, rb);
        ASSERT_EQ(rf.wait_for(std::chrono::seconds(30)), std::future_status::ready)
            << "post-forget transfer hung k=" << k;
        EXPECT_EQ(rf.get().state, kvc::TransferState::kSUCCESS) << "post-forget transfer failed k=" << k;
        EXPECT_TRUE(bounce_test::verifyXferBufs(rb)) << "post-forget byte mismatch k=" << k;
        bounce_test::freeXferBufs(rb);
    }

    A->tx->shutdown();
    B->tx->shutdown();
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
            auto fut = bounce_test::submitTransfer(*A, *B, toB);
            if (fut.wait_for(std::chrono::seconds(40)) == std::future_status::ready
                && fut.get().state == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    threads.emplace_back(
        [&]
        {
            auto fut = bounce_test::submitTransfer(*A, *C, toC);
            if (fut.wait_for(std::chrono::seconds(40)) == std::future_status::ready
                && fut.get().state == kvc::TransferState::kSUCCESS)
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
                auto fut = bounce_test::submitTransfer(*A, *B, bufs[i]);
                if (fut.wait_for(std::chrono::seconds(30)) == std::future_status::ready
                    && fut.get().state == kvc::TransferState::kSUCCESS)
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
