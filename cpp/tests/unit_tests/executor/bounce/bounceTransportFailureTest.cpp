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

    [[nodiscard]] std::unique_ptr<kvc::TransferStatus> postXferRequest(kvc::TransferOp, kvc::TransferDescs const&,
        kvc::TransferDescs const&, std::string const&, std::optional<kvc::SyncMessage> const&) override
    {
        mCtl->postCount.fetch_add(1, std::memory_order_relaxed);
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
    std::string const& name, b::BounceConfig const& c, std::shared_ptr<XferControls> ctl)
{
    return bounce_test::makeNode(name, c, 1024,
        [&ctl](kvc::BaseAgentConfig const& agentConfig) { return std::make_unique<FakeXferAgent>(agentConfig, ctl); });
}

// Poll `ch` until `budget` elapses, invoking `onMsg(header, blob)` for each decoded message;
// stops early when onMsg returns true (returns whether it did).
template <typename Fn>
bool pumpChannel(b::ZmqControlChannel& ch, std::chrono::milliseconds budget, Fn&& onMsg)
{
    auto const deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline)
    {
        std::string peer;
        std::string blob;
        b::BounceMsgHeader h{};
        if (ch.recv(peer, blob, 20) && b::decodeHeader(blob, h) && onMsg(h, blob))
        {
            return true;
        }
    }
    return false;
}

// Wait up to `budget` for a GRANT for `rid`, decoding its credits into `out`.
bool waitGrant(b::ZmqControlChannel& ch, std::uint64_t rid, std::chrono::milliseconds budget,
    std::vector<b::BounceCreditEntry>& out)
{
    return pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const& blob)
        {
            if (static_cast<b::BounceMsgType>(h.msgType) != b::BounceMsgType::kGRANT || h.requestId != rid)
            {
                return false;
            }
            EXPECT_TRUE(b::decodeCredits(blob, h, out));
            return !out.empty();
        });
}

// Count ACKs for `rid` over the FULL `budget` window (a negative-assertion helper: always runs to
// the deadline so extra/duplicate ACKs have time to show up).
int countAcks(b::ZmqControlChannel& ch, std::chrono::milliseconds budget, std::uint64_t rid)
{
    int n = 0;
    pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const&)
        {
            if (static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kACK && h.requestId == rid)
            {
                ++n;
            }
            return false;
        });
    return n;
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
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "ghostPeer");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "request hung with no grant";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kNoProgressTimeout);

    t->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// The posted write reports failure -> the request must FAIL (not hang, not falsely succeed).
// The sender is a FakeXferAgent whose writes always poll as FAILURE.
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
    A->tx->addPeer("feB", B->ch->localEndpoint());
    B->tx->addPeer("feA", A->ch->localEndpoint());

    auto bufs = bounce_test::makeXferBufs(8, 256, /*seed=*/1);
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, "feB");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready) << "write failure hung";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kWriteFailed);
    EXPECT_GE(ctl->postCount.load(std::memory_order_acquire), 1u);

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// DATA has no retransmission path in this protocol. Receiving it twice for one granted region must
// therefore produce one scatter and one ACK, rather than adding replay state for an impossible
// normal-flow event.
TEST(BounceTransportFailure, DuplicateDataProducesOneScatterAndAck)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    capRegions(c, c.maxInflightChunksPerRequest);
    b::ZmqControlChannel sender("dupDataSender");
    auto receiver = bounce_test::makeNode("dupDataReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender.addPeer("dupDataReceiver", receiver->ch->localEndpoint()));

    // Hold every context so both DATA messages reach the reactor before the first scatter can finish.
    std::vector<b::ExecCtx*> heldExecContexts;
    while (auto* ctx = receiver->exec->tryAcquire())
    {
        heldExecContexts.push_back(ctx);
    }
    ASSERT_EQ(heldExecContexts.size(), receiver->exec->size());

    constexpr std::uint64_t rid = 17;
    sender.sendTo("dupDataReceiver", b::encodeWant(rid, {256}, sender.localEndpoint()));

    std::vector<b::BounceCreditEntry> credits;
    ASSERT_TRUE(waitGrant(sender, rid, std::chrono::seconds(5), credits));
    ASSERT_EQ(credits.size(), 1);

    void* dst = nullptr;
    ASSERT_EQ(cudaMalloc(&dst, 256), cudaSuccess);
    b::BounceScatterRun run{0, reinterpret_cast<std::uintptr_t>(dst), 0, 0, 256, 1};
    auto const data = b::encodeData(rid, /*chunkIdx=*/0, /*numChunks=*/1, credits.front().regionHandle, {run});
    sender.sendTo("dupDataReceiver", data);
    sender.sendTo("dupDataReceiver", data);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (auto* ctx : heldExecContexts)
    {
        receiver->exec->release(ctx);
    }

    EXPECT_EQ(countAcks(sender, std::chrono::seconds(2), rid), 1);

    receiver->tx->shutdown();
    EXPECT_EQ(cudaFree(dst), cudaSuccess);
}

// A non-empty WANT has no retransmission path either (submit() sends it exactly once per fresh
// rid), so a replayed one for a live flow must be dropped: re-queueing would grant fresh regions
// over the still-held ones — the sender never writes the extras, so they leak — and the lease
// refresh would keep the flow off staleFlows() forever, defeating the reclaim that exists for
// exactly this state.
TEST(BounceTransportFailure, DuplicateWantIsDroppedWithoutRegrant)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    capRegions(c, c.maxInflightChunksPerRequest);
    b::ZmqControlChannel sender("dupWantSender");
    auto receiver = bounce_test::makeNode("dupWantReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender.addPeer("dupWantReceiver", receiver->ch->localEndpoint()));

    constexpr std::uint64_t rid = 23;
    sender.sendTo("dupWantReceiver", b::encodeWant(rid, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credits;
    ASSERT_TRUE(waitGrant(sender, rid, std::chrono::seconds(5), credits));
    ASSERT_EQ(credits.size(), 1u);

    // Replay the same WANT. The arena and the per-request cap both have room for a second region,
    // so a re-queue WOULD be granted — the negative assertion below therefore fails pre-fix.
    sender.sendTo("dupWantReceiver", b::encodeWant(rid, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> extra;
    EXPECT_FALSE(waitGrant(sender, rid, std::chrono::milliseconds(500), extra)) << "duplicate WANT was re-granted";

    receiver->tx->shutdown();
}

// The scatter worker sizes its plan by iterating the DATA run list; a hostile/corrupt run whose
// count is near 2^32 (bounceStride 0 keeps the span check happy) must be rejected up front, not
// counted piece by piece — pre-fix the worker (and the flow's region) is pinned for the whole
// count, which the rid=2 re-grant deadline below catches.
TEST(BounceTransportFailure, OversizedScatterRunListIsRejectedNotCounted)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    capRegions(c, /*regionCap=*/1);                                   // arena holds exactly ONE region
    b::ZmqControlChannel sender("bigRunSender");
    auto receiver = bounce_test::makeNode("bigRunReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender.addPeer("bigRunReceiver", receiver->ch->localEndpoint()));

    sender.sendTo("bigRunReceiver", b::encodeWant(/*rid=*/1, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credits;
    ASSERT_TRUE(waitGrant(sender, 1, std::chrono::seconds(5), credits));
    ASSERT_EQ(credits.size(), 1u);

    void* dst = nullptr;
    ASSERT_EQ(cudaMalloc(&dst, 256), cudaSuccess);
    // count=2^32-1 with bounceStride=0: every piece reads the same 256 in-bounds bytes, so the
    // span check alone admits it; only the piece-count guard can reject it cheaply.
    b::BounceScatterRun run{/*bounceOffset=*/0, reinterpret_cast<std::uintptr_t>(dst), /*dstStride=*/0,
        /*bounceStride=*/0, /*pieceSize=*/256, /*count=*/0xFFFFFFFFu};
    sender.sendTo("bigRunReceiver",
        b::encodeData(/*rid=*/1, /*chunkIdx=*/0, /*numChunks=*/1, credits.front().regionHandle, {run}));

    // The rejected scatter must not ACK...
    EXPECT_EQ(countAcks(sender, std::chrono::seconds(2), /*rid=*/1), 0);
    // ...and must release the single region promptly: rid=2 can only be granted once rid=1's
    // region is freed, which a worker stuck counting 2^32 pieces cannot do within the deadline.
    sender.sendTo("bigRunReceiver", b::encodeWant(/*rid=*/2, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credits2;
    EXPECT_TRUE(waitGrant(sender, 2, std::chrono::seconds(5), credits2)) << "region pinned by oversized run list";

    receiver->tx->shutdown();
    EXPECT_EQ(cudaFree(dst), cudaSuccess);
}

// A sender that takes a GRANT and then dies emits neither DATA nor a cancel, so nothing
// event-driven can ever reclaim its region — the receiver's grant lease must. After
// receiverFlowTimeoutMs of silence the flow is reclaimed and its region quarantined for
// quarantineMs, after which it must serve the next waiting flow; a late DATA for the expired
// grant must be dropped (no scatter, no ACK).
TEST(BounceTransportFailure, GrantLeaseExpiryReclaimsSilentSendersRegion)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000); // sender-side request timeout: irrelevant here
    c.receiverFlowTimeoutMs = 400;
    c.quarantineMs = 200;
    capRegions(c, /*regionCap=*/1);                               // arena holds exactly ONE region
    b::ZmqControlChannel sender("glSender");
    auto receiver = bounce_test::makeNode("glReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender.addPeer("glReceiver", receiver->ch->localEndpoint()));

    // rid=1 takes the single region... and goes silent (models a dead/unreachable sender).
    sender.sendTo("glReceiver", b::encodeWant(/*rid=*/1, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credit1;
    ASSERT_TRUE(waitGrant(sender, 1, std::chrono::seconds(5), credit1));
    ASSERT_EQ(credit1.size(), 1u);

    // rid=2 wants the region too. Before the lease (400ms) + quarantine (200ms) can possibly have
    // elapsed, it must NOT be granted (premature reclaim would race rid=1's hypothetical write).
    sender.sendTo("glReceiver", b::encodeWant(/*rid=*/2, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credit2;
    EXPECT_FALSE(waitGrant(sender, 2, std::chrono::milliseconds(200), credit2))
        << "region re-granted before lease expiry";
    // ...but once the silent flow's lease expires and the quarantine passes, it must be.
    ASSERT_TRUE(waitGrant(sender, 2, std::chrono::seconds(10), credit2)) << "silent sender's region never reclaimed";
    ASSERT_EQ(credit2.size(), 1u);
    EXPECT_EQ(credit2.front().regionHandle, credit1.front().regionHandle); // the one region, recycled

    // Late DATA for the EXPIRED grant must be dropped (flow reclaimed -> heldByFlow false): exactly
    // one ACK total, and it belongs to rid=2's legitimate DATA.
    void* dst = nullptr;
    ASSERT_EQ(cudaMalloc(&dst, 256), cudaSuccess);
    b::BounceScatterRun run{0, reinterpret_cast<std::uintptr_t>(dst), 0, 0, 256, 1};
    sender.sendTo("glReceiver", b::encodeData(/*rid=*/1, 0, 1, credit1.front().regionHandle, {run}));
    sender.sendTo("glReceiver", b::encodeData(/*rid=*/2, 0, 1, credit2.front().regionHandle, {run}));
    int ackRid1 = 0;
    int ackRid2 = 0;
    pumpChannel(sender, std::chrono::seconds(2),
        [&](b::BounceMsgHeader const& h, std::string const&)
        {
            if (static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kACK)
            {
                ackRid1 += h.requestId == 1 ? 1 : 0;
                ackRid2 += h.requestId == 2 ? 1 : 0;
            }
            return false;
        });
    EXPECT_EQ(ackRid1, 0) << "late DATA for an expired grant was scattered/ACKed";
    EXPECT_EQ(ackRid2, 1);

    receiver->tx->shutdown();
    EXPECT_EQ(cudaFree(dst), cudaSuccess);
}

// GRANT and ACK messages carry peer-owned capabilities. Only the intended peer may grant, and an
// ACK is valid only for the matching region after that chunk has reached Sent.
TEST(BounceTransportFailure, RejectsWrongPeerGrantAndInvalidAcks)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    capRegions(c, c.maxInflightChunksPerRequest);
    b::ZmqControlChannel legitimate("validateLegitimate");
    b::ZmqControlChannel attacker("validateAttacker");
    auto ctl = std::make_shared<XferControls>(); // defaults: writes complete instantly
    auto sender = makeFakeNode("validateSender", c, ctl);
    if (!sender)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender->tx->addPeer("validateLegitimate", legitimate.localEndpoint()));
    ASSERT_TRUE(legitimate.addPeer("validateSender", sender->ch->localEndpoint()));
    ASSERT_TRUE(attacker.addPeer("validateSender", sender->ch->localEndpoint()));

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/4);
    auto fut = sender->tx->submit(bufs.srcDescs, bufs.dstDescs, "validateLegitimate");
    b::BounceCreditEntry credit{/*addr=*/0, /*len=*/256, /*devId=*/0, /*regionHandle=*/77};

    attacker.sendTo("validateSender", b::encodeGrant(/*requestId=*/1, {credit}));
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(ctl->postCount.load(std::memory_order_acquire), 0u);

    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeGrant(/*requestId=*/1, {credit}));

    // Seeing DATA guarantees the matching sender chunk has transitioned to Sent.
    ASSERT_TRUE(pumpChannel(legitimate, std::chrono::seconds(5),
        [](b::BounceMsgHeader const& h, std::string const&)
        { return static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kDATA; }));
    EXPECT_EQ(ctl->postCount.load(std::memory_order_acquire), 1u);

    attacker.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle + 1));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    legitimate.sendTo("validateSender", b::encodeAck(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(fut.get().state, kvc::TransferState::kSUCCESS);

    sender->tx->shutdown();
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
    b::ZmqControlChannel peer("shutdownReleasePeer");
    auto ctl = std::make_shared<XferControls>();
    ctl->allowTerminal.store(false, std::memory_order_release);
    ctl->releaseSucceeds.store(false, std::memory_order_release);
    auto sender = makeFakeNode("shutdownReleaseSender", c, ctl);
    if (!sender)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender->tx->addPeer("shutdownReleasePeer", peer.localEndpoint()));
    ASSERT_TRUE(peer.addPeer("shutdownReleaseSender", sender->ch->localEndpoint()));

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/5);
    auto fut = sender->tx->submit(bufs.srcDescs, bufs.dstDescs, "shutdownReleasePeer");
    b::BounceCreditEntry credit{/*addr=*/0, /*len=*/256, /*devId=*/0, /*regionHandle=*/91};
    peer.sendTo("shutdownReleaseSender", b::encodeGrant(/*requestId=*/1, {credit}));

    auto const postDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < postDeadline && ctl->postCount.load(std::memory_order_acquire) == 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(ctl->postCount.load(std::memory_order_acquire), 1u);

    auto const start = std::chrono::steady_clock::now();
    sender->tx->shutdown();
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kShutdown);
    // failAll attempts the failed release exactly twice (initial + bounded retry), never spins.
    EXPECT_EQ(ctl->releaseCount.load(std::memory_order_acquire), 2u);
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
    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "gonePeer");                   // stuck: never granted
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout); // still pending
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
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, "fpB");
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
        auto rf = A->tx->submit(rb.srcDescs, rb.dstDescs, "fpB");
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
            auto fut = A->tx->submit(toB.srcDescs, toB.dstDescs, B->name);
            if (fut.wait_for(std::chrono::seconds(40)) == std::future_status::ready
                && fut.get().state == kvc::TransferState::kSUCCESS)
                ok.fetch_add(1);
        });
    threads.emplace_back(
        [&]
        {
            auto fut = A->tx->submit(toC.srcDescs, toC.dstDescs, C->name);
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
    c.arenaAllocationGranularityBytes = 256;
    c.arenaSizeBytes = 1ULL << 20;
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
                auto fut = A->tx->submit(bufs[i].srcDescs, bufs[i].dstDescs, B->name);
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
