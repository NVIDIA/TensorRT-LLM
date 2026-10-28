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
// The data plane is real NIXL (via bounceTestNixlNode helpers), except where a test must script
// individual writes (complete / stall / fail): those use a FakeXferAgent (a NixlTransferAgent subclass
// overriding postXferRequest), because real NIXL cannot be made to do that.
//
// The happy-path / concurrency / bidirectional / multi-agent coverage lives in bounceAgentE2ETest,
// which drives the SAME pipeline through the production entry point (NixlTransferAgent::
// submitTransferRequests) with one-directional AgentDesc bootstrap, so it is not duplicated here.

#include "bounceTestNixlNode.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferStatus.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{
// How one scripted write polls (see XferControls::scriptWrite).
enum class WriteScript
{
    kComplete,
    kStall,
    kFail,
};

// Test-controlled behavior for every write a FakeXferAgent posts. Shared by the agent and all the
// statuses it hands out.
struct XferControls
{
    std::atomic<bool> failWrites{false};     // every posted write polls as FAILURE
    std::atomic<bool> allowTerminal{true};   // false -> posted writes stay IN_PROGRESS
    std::atomic<bool> releaseSucceeds{true}; // false -> release() fails (handle retained)
    std::atomic<std::uint64_t> postCount{0};
    std::atomic<std::uint64_t> releaseCount{0};

    // Override how the write with this post index (0 = the agent's first write) polls; may be switched
    // while that write is in flight. Unscripted writes follow the switches above.
    void scriptWrite(std::uint64_t postIndex, WriteScript script)
    {
        std::lock_guard<std::mutex> lk(mScriptMu);
        mScripts[postIndex] = script;
    }

    [[nodiscard]] std::optional<WriteScript> scriptFor(std::uint64_t postIndex)
    {
        std::lock_guard<std::mutex> lk(mScriptMu);
        auto const it = mScripts.find(postIndex);
        if (it == mScripts.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

private:
    std::mutex mScriptMu;
    std::unordered_map<std::uint64_t, WriteScript> mScripts;
};

// A TransferStatus driven by XferControls instead of a real backend handle. Unlike
// NixlTransferStatus it does NOT release in its destructor, keeping releaseCount deterministic.
class FakeXferStatus final : public kvc::TransferStatus
{
public:
    FakeXferStatus(std::shared_ptr<XferControls> ctl, std::uint64_t postIndex)
        : mCtl(std::move(ctl))
        , mPostIndex(postIndex)
    {
    }

    [[nodiscard]] bool isCompleted() const override
    {
        return wait(0) != kvc::TransferState::kIN_PROGRESS;
    }

    [[nodiscard]] kvc::TransferState wait(int64_t) const override
    {
        if (auto const script = mCtl->scriptFor(mPostIndex))
        {
            switch (*script)
            {
            case WriteScript::kComplete: return kvc::TransferState::kSUCCESS;
            case WriteScript::kStall: return kvc::TransferState::kIN_PROGRESS;
            case WriteScript::kFail: return kvc::TransferState::kFAILURE;
            }
        }
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
    std::uint64_t mPostIndex{};
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
        auto const postIndex = mCtl->postCount.fetch_add(1, std::memory_order_acq_rel);
        return std::make_unique<FakeXferStatus>(mCtl, postIndex);
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

// Wait up to `budget` for a NACK for `rid`.
bool waitNack(b::ZmqControlChannel& ch, std::uint64_t rid, std::chrono::milliseconds budget)
{
    return pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const&)
        { return static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kNACK && h.requestId == rid; });
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

// ---- Helpers for the FAILURE-contract tests (last section) ----

// Poll `done` every 1 ms until it holds or `budget` elapses; returns whether it held. For state the
// test cannot be notified of (exec-pool and fake-agent counters).
template <typename Pred>
bool waitUntil(Pred&& done, std::chrono::milliseconds budget = std::chrono::seconds(5))
{
    auto const deadline = std::chrono::steady_clock::now() + budget;
    while (!done())
    {
        if (std::chrono::steady_clock::now() >= deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Takes exec contexts away from a node's reactor and gives them back on destruction, so a failed
// ASSERT cannot strand them. Holding every receiver context stops scatters before they launch.
// Holding all but one sender context serializes the gathers, so writes post in chunk order and a
// FakeXferAgent post index names its chunk.
class ExecHold
{
public:
    ExecHold(b::ExecPool& pool, std::size_t leaveFree)
        : mPool(pool)
        , mLeaveFree(leaveFree)
    {
        while (mPool.freeCount() > leaveFree)
        {
            auto* ctx = mPool.tryAcquire();
            if (ctx == nullptr)
            {
                break;
            }
            mHeld.push_back(ctx);
        }
    }

    ~ExecHold()
    {
        releaseAll();
    }

    ExecHold(ExecHold const&) = delete;
    ExecHold& operator=(ExecHold const&) = delete;

    // Whether it holds every context but the `leaveFree` it was asked to leave (i.e. the pool was idle).
    [[nodiscard]] bool complete() const
    {
        return mHeld.size() + mLeaveFree == mPool.size();
    }

    [[nodiscard]] b::ExecCtx* front() const
    {
        return mHeld.front();
    }

    void releaseFront()
    {
        mPool.release(mHeld.front());
        mHeld.erase(mHeld.begin());
    }

    void releaseAll()
    {
        for (auto* ctx : mHeld)
        {
            mPool.release(ctx);
        }
        mHeld.clear();
    }

private:
    b::ExecPool& mPool;
    std::size_t mLeaveFree;
    std::vector<b::ExecCtx*> mHeld;
};

// Launch the batched-copy kernel once (a zero-byte copy) so its module is loaded. Under
// CUDA_MODULE_LOADING=LAZY, the default, the first launch in the process loads the module, which
// needs a device-wide lock and so would wait forever behind a closed StreamGate. Gather and scatter
// both launch this kernel (launchPrepared -> launchBatchedCopy), and the test executable links the
// same copy, so this one launch covers both.
bool loadBatchedCopyKernel()
{
    // One all-zero plan entry (src = dst = 0, size = 0), packed srcs|dsts|sizes like the plan buffers:
    // the kernel touches no memory.
    constexpr std::size_t kPlanBytes = 3 * sizeof(std::uint64_t);
    void* plan = nullptr;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&plan, kPlanBytes) != cudaSuccess) // every node is on device 0
    {
        return false;
    }
    auto const* entry = static_cast<std::uint64_t const*>(plan);
    bool const ok = cudaMemset(plan, 0, kPlanBytes) == cudaSuccess
        && b::launchBatchedCopy(/*srcs=*/entry, /*dsts=*/entry + 1,
               /*sizes=*/reinterpret_cast<std::uint32_t const*>(entry + 2), /*n=*/1, /*stream=*/nullptr)
            == cudaSuccess
        && cudaStreamSynchronize(nullptr) == cudaSuccess;
    return cudaFree(plan) == cudaSuccess && ok;
}

// Blocks a CUDA stream: a host function enqueued on it spins until open(), so everything enqueued
// later on that stream (a scatter kernel) waits behind it. The destructor opens the gate and drains
// the stream, so a failed ASSERT can never leave the stream blocked (the receiver's shutdown would
// hang in cudaDeviceSynchronize). While it is closed, anything that needs a device-wide lock waits
// for it and deadlocks: cudaFree, cudaDeviceSynchronize, and lazy module loading on a kernel's first
// launch. So allocate before closing a gate and free after opening it; the constructor loads the
// bounce kernel before it closes.
class StreamGate
{
public:
    explicit StreamGate(cudaStream_t stream)
        : mStream(stream)
    {
        if (loadBatchedCopyKernel()) // before closing: see the class comment
        {
            mInstalled = cudaLaunchHostFunc(mStream, &StreamGate::holdUntilOpen, &mOpen) == cudaSuccess;
        }
    }

    ~StreamGate()
    {
        open();
        if (mInstalled)
        {
            (void) cudaStreamSynchronize(mStream);
        }
    }

    StreamGate(StreamGate const&) = delete;
    StreamGate& operator=(StreamGate const&) = delete;

    // Whether the gate was placed on the stream (it then blocks the stream until open()).
    [[nodiscard]] bool installed() const
    {
        return mInstalled;
    }

    void open()
    {
        mOpen.store(true, std::memory_order_release);
    }

private:
    static void CUDART_CB holdUntilOpen(void* openFlag)
    {
        auto const& isOpen = *static_cast<std::atomic<bool> const*>(openFlag);
        while (!isOpen.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    cudaStream_t mStream;
    std::atomic<bool> mOpen{false};
    bool mInstalled{false};
};

// XferBufs freed on scope exit. cudaFree synchronizes the device, so declare it before any StreamGate.
struct OwnedXferBufs : bounce_test::XferBufs
{
    OwnedXferBufs(std::uint32_t nDescs, std::uint32_t descBytes, std::uint32_t seed)
        : bounce_test::XferBufs(bounce_test::makeXferBufs(nDescs, descBytes, seed))
    {
    }

    ~OwnedXferBufs()
    {
        bounce_test::freeXferBufs(*this);
    }

    OwnedXferBufs(OwnedXferBufs const&) = delete;
    OwnedXferBufs& operator=(OwnedXferBufs const&) = delete;
};

// Whether every byte of the device range [base + offset, base + offset + bytes) equals `value`.
bool deviceBytesEqual(void const* base, std::uint64_t offset, std::uint64_t bytes, unsigned char value)
{
    std::vector<unsigned char> host(bytes);
    if (cudaMemcpy(host.data(), static_cast<unsigned char const*>(base) + offset, bytes, cudaMemcpyDeviceToHost)
        != cudaSuccess)
    {
        return false;
    }
    return std::all_of(host.begin(), host.end(), [value](unsigned char v) { return v == value; });
}

// What a receiver's scatter copies into dst: its arena is prefilled with this byte, because fake
// writes never deliver the source.
constexpr unsigned char kArenaFill = 0xA5;
// What the caller writes into dst once the request has resolved.
constexpr unsigned char kReusedFill = 0x3C;

// Wait for the sender's (non-cancel) WANT and store its rid in `rid`.
bool waitWant(b::ZmqControlChannel& ch, std::uint64_t& rid, std::chrono::milliseconds budget = std::chrono::seconds(5))
{
    return pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const& blob)
        {
            std::vector<std::uint32_t> chunkBytes;
            std::string endpoint;
            if (static_cast<b::BounceMsgType>(h.msgType) != b::BounceMsgType::kWANT
                || !b::decodeWant(blob, h, chunkBytes, endpoint) || b::isCancelWant(chunkBytes))
            {
                return false;
            }
            rid = h.requestId;
            return true;
        });
}

// Wait until DATA for `chunks` distinct chunks of `rid` has arrived, i.e. the sender marked them Sent.
bool waitData(b::ZmqControlChannel& ch, std::uint64_t rid, std::size_t chunks,
    std::chrono::milliseconds budget = std::chrono::seconds(5))
{
    std::set<std::uint32_t> seen;
    return pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const&)
        {
            if (static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kDATA && h.requestId == rid)
            {
                seen.insert(h.chunkIdx);
            }
            return seen.size() >= chunks;
        });
}

// Wait for the sender's cancel (an empty WANT) for `rid`, i.e. proof that it failed the request.
bool waitCancel(b::ZmqControlChannel& ch, std::uint64_t rid, std::chrono::milliseconds budget = std::chrono::seconds(5))
{
    return pumpChannel(ch, budget,
        [&](b::BounceMsgHeader const& h, std::string const& blob)
        {
            std::vector<std::uint32_t> chunkBytes;
            std::string endpoint;
            return static_cast<b::BounceMsgType>(h.msgType) == b::BounceMsgType::kWANT && h.requestId == rid
                && b::decodeWant(blob, h, chunkBytes, endpoint) && b::isCancelWant(chunkBytes);
        });
}

// The region handle a scripted receiver grants to chunk `chunkIdx`.
std::uint64_t fakeRegion(std::uint32_t chunkIdx)
{
    return 100 + chunkIdx;
}

// A FakeXferAgent sender wired to a scripted receiver: a bare control channel through which the test
// plays the receiver by hand. Writes are fake, so the credits carry fake addresses.
struct ScriptedPeers
{
    // Declared first, so destroyed last: freed only after the sender has shut down.
    std::deque<OwnedXferBufs> requestBufs;
    std::shared_ptr<XferControls> ctl{std::make_shared<XferControls>()};
    std::string senderName;
    std::string receiverName;
    std::uint32_t chunkBytes{};
    std::unique_ptr<b::ZmqControlChannel> receiver;
    std::unique_ptr<bounce_test::Node> sender; // declared last: shut down before the channel goes away

    // GRANT chunks [firstChunk, endChunk) of `rid` in one message; chunk i gets fakeRegion(i).
    void grant(std::uint64_t rid, std::uint32_t firstChunk, std::uint32_t endChunk) const
    {
        std::vector<b::BounceCreditEntry> credits;
        for (std::uint32_t i = firstChunk; i < endChunk; ++i)
        {
            credits.push_back({/*addr=*/0, chunkBytes, /*devId=*/0, fakeRegion(i)});
        }
        receiver->sendTo(senderName, b::encodeGrant(rid, credits));
    }

    void ack(std::uint64_t rid, std::uint32_t chunkIdx) const
    {
        receiver->sendTo(senderName, b::encodeAck(rid, chunkIdx, fakeRegion(chunkIdx)));
    }

    void nack(std::uint64_t rid, std::uint32_t chunkIdx) const
    {
        receiver->sendTo(senderName, b::encodeNack(rid, chunkIdx, fakeRegion(chunkIdx)));
    }
};

// nullptr when the NIXL agent/backend is unavailable (the caller GTEST_SKIPs).
std::unique_ptr<ScriptedPeers> makeScriptedPeers(std::string const& name, b::BounceConfig const& c)
{
    auto peers = std::make_unique<ScriptedPeers>();
    peers->senderName = name + "Sender";
    peers->receiverName = name + "Receiver";
    peers->chunkBytes = static_cast<std::uint32_t>(c.maxChunkSizeBytes);
    peers->receiver = std::make_unique<b::ZmqControlChannel>(peers->receiverName);
    peers->sender = makeFakeNode(peers->senderName, c, peers->ctl);
    if (!peers->sender)
    {
        return nullptr;
    }
    EXPECT_TRUE(peers->sender->tx->addPeer(peers->receiverName, peers->receiver->localEndpoint()));
    EXPECT_TRUE(peers->receiver->addPeer(peers->senderName, peers->sender->ch->localEndpoint()));
    return peers;
}

// One request to the scripted receiver: `chunks` descriptors of one full chunk each, so chunk i is
// descriptor i. Its buffers live in `peers`. The rid is filled in from the WANT (waitWant).
struct ScriptedRequest
{
    ScriptedRequest(ScriptedPeers& peers, std::uint32_t chunks, std::uint32_t seed)
        : chunks(chunks)
        , bufs(peers.requestBufs.emplace_back(chunks, peers.chunkBytes, seed))
        , fut(peers.sender->tx->submit(bufs.srcDescs, bufs.dstDescs, peers.receiverName))
    {
    }

    std::uint32_t chunks;
    OwnedXferBufs const& bufs;
    std::shared_future<b::BounceResult> fut;
    std::uint64_t rid{};
};

// Learn `req`'s rid from its WANT, grant all its chunks at once and wait until every one is Sent.
bool grantUntilAllSent(ScriptedPeers& peers, ScriptedRequest& req)
{
    if (!waitWant(*peers.receiver, req.rid))
    {
        return false;
    }
    peers.grant(req.rid, /*firstChunk=*/0, /*endChunk=*/req.chunks);
    return waitData(*peers.receiver, req.rid, req.chunks);
}

// FIFO barrier: once this returns true, the sender's reactor has dispatched every message the scripted
// receiver sent before it. A never-granted sentinel request is NACKed and must resolve; the reactor reads
// one peer's messages in order.
bool syncSender(ScriptedPeers& peers)
{
    ScriptedRequest sentinel(peers, /*chunks=*/1, /*seed=*/0);
    if (!waitWant(*peers.receiver, sentinel.rid))
    {
        return false;
    }
    peers.nack(sentinel.rid, 0);
    return sentinel.fut.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
}

// A status over a future already resolved with `result`.
b::BounceTransferStatus resolvedStatus(b::BounceResult const& result)
{
    std::promise<b::BounceResult> promise;
    promise.set_value(result);
    return b::BounceTransferStatus{promise.get_future().share()};
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
    EXPECT_TRUE(fut.get().quiesced) << "nothing was sent, so nothing can write dst";

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
    EXPECT_TRUE(fut.get().quiesced) << "no write completed, so no DATA was sent";
    EXPECT_GE(ctl->postCount.load(std::memory_order_acquire), 1u);

    A->tx->shutdown();
    B->tx->shutdown();
    bounce_test::freeXferBufs(bufs);
}

// A receiver that KNOWS a request cannot complete NACKs it, so the sender fails right away
// (kPeerNack) instead of waiting out requestTimeoutMs (kNoProgressTimeout). Provoked via the WANT
// chunk-size rejection — the request-level NACK source (a rejected WANT): the sender's chunk cap
// exceeds the receiver's, so the single 8 KiB chunk is outside the receiver's (0, 4 KiB]. The
// cap-equality handshake gate lives in
// NixlTransferAgent::shouldUseBounce, which BounceTransport::submit never consults (wirePair only does
// loadRemoteAgent + addPeer), so the mismatched WANT reaches onWant directly.
TEST(BounceTransportFailure, ReceiverNackFailsSenderBeforeTimeout)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr int kTimeoutMs = 20000; // generous: the NACK bound below is timeout/2, the hang guard is 10 s
    auto senderCfg = cfg(kTimeoutMs);
    senderCfg.maxChunkSizeBytes = 8192;
    auto receiverCfg = cfg(kTimeoutMs); // maxChunkSizeBytes = 4096
    auto A = bounce_test::makeNode("nackA", senderCfg, 1024);
    auto B = bounce_test::makeNode("nackB", receiverCfg, 1024);
    if (!A || !B)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    bounce_test::wirePair(*A, *B);

    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/8192, /*seed=*/12);
    auto const execBaseline = A->exec->freeCount(); // every gather ctx idle before submit
    auto const start = std::chrono::steady_clock::now();
    auto fut = A->tx->submit(bufs.srcDescs, bufs.dstDescs, "nackB");
    // Hang guard = the full request timeout, so a slow NACK fails the timing bound below, not this.
    ASSERT_EQ(fut.wait_for(std::chrono::milliseconds(kTimeoutMs)), std::future_status::ready) << "NACKed request hung";
    auto const elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kPeerNack);
    EXPECT_TRUE(fut.get().quiesced) << "the WANT was rejected, so no DATA was sent";
    EXPECT_LT(elapsed, std::chrono::milliseconds(kTimeoutMs / 2)) << "NACK did not short-circuit the timeout";

    // The eager gather of the failed chunk holds an ExecCtx + staging region until failRequest (or,
    // for a still-running gather, the IO loop's drainOrphanGather()) releases both in one step. The
    // ctx assertion is a leak guard, not parked-branch coverage: it cannot tell the parked
    // (Gathering) branch from the immediate (Gathered) one — an 8 KiB eager gather finishes before
    // the ZMQ round trip, and if the NACK wins the race nothing was posted at all — so the
    // parked-gather branch is not deterministically exercised here (it would need a gather-stall
    // hook).
    auto const releaseDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (A->exec->freeCount() != execBaseline && std::chrono::steady_clock::now() < releaseDeadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(A->exec->freeCount(), execBaseline) << "orphaned gather ctx/staging not released after NACK";

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
    // FIFO sentinel instead of a sleep: the reactor is a single FIFO consumer, so a GRANT for a
    // fresh rid sent after both DATA messages proves it already consumed them (the second region is
    // free under capRegions(2), so the sentinel WANT is grantable while rid=17 holds its region).
    constexpr std::uint64_t sentinelRid = 18;
    sender.sendTo("dupDataReceiver", b::encodeWant(sentinelRid, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> sentinelCredits;
    ASSERT_TRUE(waitGrant(sender, sentinelRid, std::chrono::seconds(5), sentinelCredits))
        << "sentinel WANT never granted; DATA messages may not have reached the reactor";
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

// WANT chunk sizes are untrusted peer input, and each of the two out-of-contract shapes is its own
// DoS pre-fix: a chunk of 0 can never be allocated, so the flow blocks forever, and once
// maybeActivateDrain() latches it as the drain flow, schedule() short-circuits before sweeping any
// other flow — every grant to every peer stops with no reclaim path (a pending-only flow holds no
// regions, so the lease sweep skips it). A chunk above the handshake-pinned maxChunkSizeBytes
// instead buddy-rounds up to the WHOLE usable arena (test config: 4097 -> 8192 = capacity), gets
// granted, and squats on it, starving all peers until the lease sweep. The receiver must reject
// both up front. The test drives each mode separately: park the zero-size flow, push the grant
// sequence past the drain bypass threshold with legitimate traffic, require fresh demand to still
// be granted; then send the oversized WANT and require the same.
TEST(BounceTransportFailure, MalformedWantChunkSizeIsRejectedNotWedged)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    capRegions(c, c.maxInflightChunksPerRequest);
    b::ZmqControlChannel sender("badWantSender");
    auto receiver = bounce_test::makeNode("badWantReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender.addPeer("badWantReceiver", receiver->ch->localEndpoint()));

    sender.sendTo("badWantReceiver", b::encodeWant(/*rid=*/1, {0}, sender.localEndpoint()));

    // Drive enough legitimate grant/cancel cycles to push mGrantSequence past the drain bypass
    // threshold (max(kMinimumBypassGrants, kBypassRounds * ringSize) = 8 here); pre-fix this
    // latches rid=1 as the drain flow and the cycle after the eighth grant starves.
    for (std::uint64_t rid = 100; rid < 112; ++rid)
    {
        sender.sendTo("badWantReceiver", b::encodeWant(rid, {256}, sender.localEndpoint()));
        std::vector<b::BounceCreditEntry> credits;
        ASSERT_TRUE(waitGrant(sender, rid, std::chrono::seconds(5), credits)) << "legit WANT starved, rid=" << rid;
        // Cancel to release the granted region and drop the flow from the ring.
        sender.sendTo("badWantReceiver", b::encodeWant(rid, {}, sender.localEndpoint()));
    }

    // The receiver must still grant fresh legitimate demand — pre-fix the drain latch wedges this.
    sender.sendTo("badWantReceiver", b::encodeWant(/*rid=*/200, {256}, sender.localEndpoint()));
    std::vector<b::BounceCreditEntry> credits;
    EXPECT_TRUE(waitGrant(sender, 200, std::chrono::seconds(5), credits)) << "receiver wedged after zero-size WANT";
    sender.sendTo("badWantReceiver", b::encodeWant(200, {}, sender.localEndpoint()));

    // Oversized mode: pre-fix this grant monopolizes the whole arena, so the follow-up legitimate
    // WANT starves.
    sender.sendTo("badWantReceiver",
        b::encodeWant(/*rid=*/2, {static_cast<std::uint32_t>(c.maxChunkSizeBytes) + 1}, sender.localEndpoint()));
    sender.sendTo("badWantReceiver", b::encodeWant(/*rid=*/201, {256}, sender.localEndpoint()));
    credits.clear();
    EXPECT_TRUE(waitGrant(sender, 201, std::chrono::seconds(5), credits)) << "arena monopolized by oversized WANT";

    // The malformed flows themselves must never have been granted. The reactor is a single FIFO
    // consumer, so the completed rid=200/201 grants above prove both malformed WANTs were already
    // processed — these are not timing-window assertions.
    std::vector<b::BounceCreditEntry> badCredits;
    EXPECT_FALSE(waitGrant(sender, 1, std::chrono::milliseconds(200), badCredits)) << "zero-size WANT was granted";
    EXPECT_FALSE(waitGrant(sender, 2, std::chrono::milliseconds(200), badCredits)) << "oversized WANT was granted";

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

    // The rejected scatter must NACK the chunk (so a real sender fails at once) and never ACK it.
    // One full-window pump covers both: the NACK must arrive, and no ACK may show up afterwards.
    int acks = 0;
    bool nacked = false;
    pumpChannel(sender, std::chrono::seconds(2),
        [&](b::BounceMsgHeader const& h, std::string const&)
        {
            if (h.requestId != 1)
            {
                return false;
            }
            auto const type = static_cast<b::BounceMsgType>(h.msgType);
            if (type == b::BounceMsgType::kACK)
            {
                ++acks;
            }
            if (type == b::BounceMsgType::kNACK)
            {
                nacked = true;
                EXPECT_EQ(h.chunkIdx, 0u);
                EXPECT_EQ(h.regionHandle, credits.front().regionHandle);
            }
            return false;
        });
    EXPECT_TRUE(nacked) << "plan overflow was not NACKed";
    EXPECT_EQ(acks, 0);
    // The rejection must also release the single region promptly: rid=2 can only be granted once rid=1's
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

    // A wrong-peer NACK (and one for an unknown rid) must not kill the request: still in flight.
    attacker.sendTo("validateSender", b::encodeNack(/*requestId=*/1, /*chunkIdx=*/0, credit.regionHandle));
    legitimate.sendTo("validateSender", b::encodeNack(/*requestId=*/999, /*chunkIdx=*/0, 0));
    EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(200)), std::future_status::timeout);

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

// A GRANT whose credit is smaller than its chunk (mispair/reorder) abandons the flow. Abandon must
// be terminal: pumpRequest ignores the request from then on, so (a) a LATER well-formed GRANT must
// not resurrect it — pre-fix it would attach, post the RDMA write, and refresh lastProgress, pushing
// the failure out by one extra timeout period per revived chunk (up to numChunks periods on a
// multi-chunk request with a peer that keeps granting) — and (b) checkTimeouts resolves the future
// kFAILURE with the abandon reason (kProtocolError, not the generic timeout).
TEST(BounceTransportFailure, CreditMispairAbandonIsTerminalAndIgnoresLaterGrants)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    // Generous timeout: the mispair GRANT must be DELIVERED and dispatched before checkTimeouts
    // can fire kNoProgressTimeout, or the reason assertion below flakes on a loaded machine.
    auto c = cfg(/*timeoutMs=*/2000);
    capRegions(c, c.maxInflightChunksPerRequest);
    b::ZmqControlChannel peer("mispairPeer");
    auto ctl = std::make_shared<XferControls>(); // defaults: writes complete instantly
    auto sender = makeFakeNode("mispairSender", c, ctl);
    if (!sender)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender->tx->addPeer("mispairPeer", peer.localEndpoint()));
    ASSERT_TRUE(peer.addPeer("mispairSender", sender->ch->localEndpoint()));

    // One 256-byte chunk (packedBytes = 256).
    auto bufs = bounce_test::makeXferBufs(/*nDescs=*/1, /*descBytes=*/256, /*seed=*/6);
    auto fut = sender->tx->submit(bufs.srcDescs, bufs.dstDescs, "mispairPeer");

    // Undersized credit (128 < 256) -> mispair -> abandon; then a well-formed one that pre-fix
    // would attach and post. FIFO ordering to the single reactor consumer guarantees the mispair
    // is processed first.
    peer.sendTo(
        "mispairSender", b::encodeGrant(/*requestId=*/1, {{/*addr=*/0, /*len=*/128, /*devId=*/0, /*regionHandle=*/7}}));
    peer.sendTo(
        "mispairSender", b::encodeGrant(/*requestId=*/1, {{/*addr=*/0, /*len=*/256, /*devId=*/0, /*regionHandle=*/8}}));

    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "abandoned flow never resolved";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kProtocolError);
    EXPECT_TRUE(fut.get().quiesced) << "no write was posted, so no DATA was sent";
    // The later valid GRANT must not have resurrected the abandoned flow into posting a write.
    EXPECT_EQ(ctl->postCount.load(std::memory_order_acquire), 0u) << "abandoned flow posted an RDMA write";

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
    EXPECT_TRUE(fut.get().quiesced) << "never granted, so no DATA was sent";

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
    // A write that never completes is drained for the fixed kShutdownWriteDrainMs (5 s) — intended
    // teardown behaviour — so the bound only asserts "bounded, no hang", not "instant".
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(10));
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kShutdown);
    EXPECT_TRUE(fut.get().quiesced) << "the write never completed, so no DATA was sent";
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

// forget() must drop the gone peer's flows BEFORE freeing the orphaned regions of its queued (not yet
// started) scatter jobs: freeOrphanRegion() runs schedule(), so with free-first the region is
// granted straight back to a pending flow of the very peer being forgotten, and the reclaim that
// follows then quarantines it (quarantineMs) instead of returning it to the arena. Setup: the
// receiver's ExecPool is drained from the test so the single scatter worker spins in tryAcquire on
// job A while job B stays queued (mScattering set for both); a cancel flags both regions orphaned; a
// second WANT from the same peer is left pending (arena full); then forgetPeer. Correct order: the
// pending flow is gone before region B is freed, so a second peer's WANT gets it at once. Old order:
// region B goes to the forgotten peer's pending flow and is quarantined for 30 s, so the second peer
// starves (and the forgotten peer sees a GRANT after forgetPeer).
TEST(BounceTransportFailure, ForgetPeerFreesOrphanedRegionsToOtherPeersOnly)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    c.scatterWorkerCount = 1;                     // exactly one job can be dequeued; the other must stay in mJobs
    c.quarantineMs = 30000;                       // the old-order symptom: the mis-granted region is parked this long
    capRegions(c, c.maxInflightChunksPerRequest); // arena holds exactly TWO regions
    b::ZmqControlChannel gone("fgGone");
    b::ZmqControlChannel other("fgOther");
    auto receiver = bounce_test::makeNode("fgReceiver", c, 1024); // receiver posts no writes
    if (!receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(gone.addPeer("fgReceiver", receiver->ch->localEndpoint()));
    ASSERT_TRUE(other.addPeer("fgReceiver", receiver->ch->localEndpoint()));
    // A zero-size WANT is NACKed inline by dispatch(); on the same DEALER it proves every earlier
    // message was processed, and the tick that dispatched it runs drainForgets() before recv()ing again.
    auto syncReactor = [&](std::uint64_t rid)
    {
        gone.sendTo("fgReceiver", b::encodeWant(rid, {0}, gone.localEndpoint()));
        return waitNack(gone, rid, std::chrono::seconds(5));
    };

    std::vector<b::ExecCtx*> heldExecContexts;
    while (auto* ctx = receiver->exec->tryAcquire())
    {
        heldExecContexts.push_back(ctx);
    }
    ASSERT_EQ(heldExecContexts.size(), receiver->exec->size());

    auto const chunk = static_cast<std::uint32_t>(c.maxChunkSizeBytes);
    gone.sendTo("fgReceiver", b::encodeWant(/*rid=*/1, {chunk, chunk}, gone.localEndpoint()));
    std::vector<b::BounceCreditEntry> credits;
    while (credits.size() < 2)
    {
        std::vector<b::BounceCreditEntry> more;
        ASSERT_TRUE(waitGrant(gone, 1, std::chrono::seconds(5), more));
        credits.insert(credits.end(), more.begin(), more.end());
    }
    ASSERT_EQ(credits.size(), 2u);

    void* dst = nullptr;
    ASSERT_EQ(cudaMalloc(&dst, 256), cudaSuccess);
    b::BounceScatterRun run{0, reinterpret_cast<std::uintptr_t>(dst), 0, 0, 256, 1};
    for (std::uint32_t i = 0; i < 2; ++i)
    {
        gone.sendTo("fgReceiver", b::encodeData(/*rid=*/1, i, /*numChunks=*/2, credits[i].regionHandle, {run}));
    }
    gone.sendTo("fgReceiver", b::encodeWant(/*rid=*/1, {}, gone.localEndpoint()));             // cancel -> orphans
    gone.sendTo("fgReceiver", b::encodeWant(/*rid=*/2, {chunk, chunk}, gone.localEndpoint())); // pending: arena full
    ASSERT_TRUE(syncReactor(/*rid=*/3)) << "cancel / pending WANT not processed";

    receiver->tx->forgetPeer("fgGone");
    ASSERT_TRUE(syncReactor(/*rid=*/4)) << "forget not applied"; // also re-adds fgGone's DEALER: a mis-grant is visible

    other.sendTo("fgReceiver", b::encodeWant(/*rid=*/10, {chunk}, other.localEndpoint()));
    std::vector<b::BounceCreditEntry> otherCredits;
    EXPECT_TRUE(waitGrant(other, 10, std::chrono::seconds(2), otherCredits))
        << "orphaned region not returned to the arena by forgetPeer (granted back to the forgotten peer?)";
    std::vector<b::BounceCreditEntry> stray;
    EXPECT_FALSE(waitGrant(gone, 2, std::chrono::milliseconds(200), stray)) << "forgotten peer was granted a region";

    for (auto* ctx : heldExecContexts)
    {
        receiver->exec->release(ctx);
    }
    receiver->tx->shutdown();
    EXPECT_EQ(cudaFree(dst), cudaSuccess);
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

// ============================================================================
// A FAILURE resolves only after the receiver has answered every chunk whose DATA was sent
// ----------------------------------------------------------------------------
// Only the receiver's scatter of a received DATA writes dst, and the receiver answers (ACK or NACK)
// after that scatter's last write. So a failed request waits for those answers before it resolves
// kFAILURE with quiesced == true; a receiver that sends no answer for requestTimeoutMs is assumed dead
// and the request resolves in-doubt (quiesced == false). These tests order events with exec-context
// holds, a stream gate, scripted writes and FIFO barriers, not by sleeping.
// ============================================================================

// The status contract the agent hands out (pure: hand-built futures, no NIXL). isCompleted() is true
// exactly when the request is over AND nothing will write the destination again.
TEST(BounceTransferStatus, CompletedOnlyOnceTheDestinationIsQuiesced)
{
    std::promise<b::BounceResult> pending;
    b::BounceTransferStatus const inFlight{pending.get_future().share()};
    EXPECT_FALSE(inFlight.isCompleted());
    EXPECT_EQ(inFlight.wait(0), kvc::TransferState::kIN_PROGRESS);
    EXPECT_EQ(inFlight.getLastStatusStr(), "");

    auto const success = resolvedStatus({kvc::TransferState::kSUCCESS, b::BounceFailReason::kNone, /*quiesced=*/true});
    EXPECT_TRUE(success.isCompleted());
    EXPECT_EQ(success.wait(0), kvc::TransferState::kSUCCESS);
    EXPECT_EQ(success.getLastStatusStr(), "");

    auto const quiescedFailure
        = resolvedStatus({kvc::TransferState::kFAILURE, b::BounceFailReason::kWriteFailed, /*quiesced=*/true});
    EXPECT_TRUE(quiescedFailure.isCompleted()) << "a destination proven quiescent may be reused";
    EXPECT_EQ(quiescedFailure.wait(0), kvc::TransferState::kFAILURE);
    EXPECT_EQ(quiescedFailure.getLastStatusStr(), b::toString(b::BounceFailReason::kWriteFailed));

    auto const inDoubt
        = resolvedStatus({kvc::TransferState::kFAILURE, b::BounceFailReason::kNoProgressTimeout, /*quiesced=*/false});
    EXPECT_FALSE(inDoubt.isCompleted()) << "the receiver may still write the destination";
    EXPECT_EQ(inDoubt.wait(0), kvc::TransferState::kFAILURE);
    EXPECT_EQ(inDoubt.getLastStatusStr(),
        std::string(b::toString(b::BounceFailReason::kNoProgressTimeout)) + "; destination not proven quiescent");

    pending.set_value({kvc::TransferState::kSUCCESS, b::BounceFailReason::kNone, /*quiesced=*/true});
    EXPECT_TRUE(inFlight.isCompleted());
}

// A request rejected at plan time never sent anything, so its FAILURE is quiesced.
TEST(BounceTransportFailure, PlanRejectionIsQuiesced)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/5000);
    // One descriptor larger than maxChunkSizeBytes: BounceTransferPlan::build rejects it.
    OwnedXferBufs bufs(/*nDescs=*/1, /*descBytes=*/2 * static_cast<std::uint32_t>(c.maxChunkSizeBytes), /*seed=*/81);
    auto t = bounce_test::makeNode("prSolo", c, 1024);
    if (!t)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    auto fut = t->tx->submit(bufs.srcDescs, bufs.dstDescs, "nobody");
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::ready)
        << "plan rejection resolves in submit()";
    EXPECT_EQ(fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(fut.get().reason, b::BounceFailReason::kPlanRejected);
    EXPECT_TRUE(fut.get().quiesced);
    t->tx->shutdown();
}

// A write fails while an earlier chunk's scatter is RUNNING on the receiver. The FAILURE must wait for
// that scatter's answer; resolving first lets the caller reuse dst while the scatter still writes it.
// The scatter is held on the receiver's only scatter worker by a stream gate placed ahead of it on the
// one exec context the worker can get.
TEST(BounceTransportFailure, FailureWaitsForRunningScatter)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    // The timeout never fires here: only the receiver's answers may resolve the failed request.
    auto c = cfg(/*timeoutMs=*/30000);
    c.scatterWorkerCount = 1;
    capRegions(c, c.maxInflightChunksPerRequest); // two chunks, both granted at once
    // Chunk i is descriptor i. Declared before the nodes, so it is freed after they shut down.
    OwnedXferBufs bufs(/*nDescs=*/2, static_cast<std::uint32_t>(c.maxChunkSizeBytes), /*seed=*/91);
    auto ctl = std::make_shared<XferControls>();
    auto sender = makeFakeNode("runScatterSender", c, ctl);
    auto receiver = bounce_test::makeNode("runScatterReceiver", c, 1024);
    if (!sender || !receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender->tx->addPeer(receiver->name, receiver->ch->localEndpoint()));
    ASSERT_TRUE(receiver->tx->addPeer(sender->name, sender->ch->localEndpoint()));
    ASSERT_EQ(cudaMemset(receiver->arena->base(), kArenaFill, receiver->arena->bytes()), cudaSuccess);

    ExecHold senderGathers(*sender->exec, /*leaveFree=*/1); // post index == chunk index
    ExecHold receiverScatters(*receiver->exec, /*leaveFree=*/0);
    ASSERT_TRUE(senderGathers.complete());
    ASSERT_TRUE(receiverScatters.complete());
    StreamGate gate(receiverScatters.front()->stream);
    ASSERT_TRUE(gate.installed());
    receiverScatters.releaseFront(); // the worker can get only the gated context
    ctl->scriptWrite(0, WriteScript::kComplete);
    ctl->scriptWrite(1, WriteScript::kStall);

    auto fut = sender->tx->submit(bufs.srcDescs, bufs.dstDescs, receiver->name);
    // The worker took the gated context: chunk 0 is Sent and its scatter waits behind the gate.
    ASSERT_TRUE(waitUntil([&] { return receiver->exec->freeCount() == 0; })) << "chunk 0 never reached the scatter";
    ASSERT_TRUE(waitUntil([&] { return ctl->postCount.load(std::memory_order_acquire) == 2; }));
    ctl->scriptWrite(1, WriteScript::kFail);
    // Write 0's handle was released when its DATA went out; write 1's is released by the orphan
    // drain that runs only after failRequest.
    ASSERT_TRUE(waitUntil([&] { return ctl->releaseCount.load(std::memory_order_acquire) == 2; }))
        << "the failed write was never handled";
    EXPECT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "FAILURE reported while chunk 0's scatter could still write dst";

    gate.open();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "the scatter's ACK was not awaited";
    auto const result = fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kWriteFailed);
    EXPECT_TRUE(result.quiesced);
    EXPECT_TRUE(deviceBytesEqual(bufs.dst, bufs.off[0], bufs.descBytes, kArenaFill))
        << "chunk 0's scatter had not finished when the FAILURE resolved";
    EXPECT_TRUE(deviceBytesEqual(bufs.dst, bufs.off[1], bufs.descBytes, 0)) << "chunk 1 was never sent";
    // Reuse dst, then shut the receiver down (joins its scatter workers, syncs the device): nothing may
    // write it again.
    ASSERT_EQ(cudaMemset(bufs.dst, kReusedFill, bufs.total), cudaSuccess);
    receiver->tx->shutdown();
    EXPECT_TRUE(deviceBytesEqual(bufs.dst, 0, bufs.total, kReusedFill)) << "dst was written after it was reused";
    sender->tx->shutdown();
}

// Same as above, but the sent chunks are still QUEUED on the receiver: chunk 0 waits for an exec
// context in the only scatter worker and chunk 1 waits in the job queue, while chunk 2's write fails.
// Without an exec context no answer is possible, so the order is deterministic.
TEST(BounceTransportFailure, FailureWaitsForQueuedScatters)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    // The timeout never fires here: only the receiver's answers may resolve the failed request.
    auto c = cfg(/*timeoutMs=*/30000);
    c.scatterWorkerCount = 1;
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest); // three chunks, all granted at once
    // Chunk i is descriptor i. Declared before the nodes, so it is freed after they shut down.
    OwnedXferBufs bufs(/*nDescs=*/3, static_cast<std::uint32_t>(c.maxChunkSizeBytes), /*seed=*/92);
    auto ctl = std::make_shared<XferControls>();
    auto sender = makeFakeNode("queuedScatterSender", c, ctl);
    auto receiver = bounce_test::makeNode("queuedScatterReceiver", c, 1024);
    if (!sender || !receiver)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ASSERT_TRUE(sender->tx->addPeer(receiver->name, receiver->ch->localEndpoint()));
    ASSERT_TRUE(receiver->tx->addPeer(sender->name, sender->ch->localEndpoint()));
    ASSERT_EQ(cudaMemset(receiver->arena->base(), kArenaFill, receiver->arena->bytes()), cudaSuccess);

    ExecHold senderGathers(*sender->exec, /*leaveFree=*/1); // post index == chunk index
    ExecHold receiverScatters(*receiver->exec, /*leaveFree=*/0);
    ASSERT_TRUE(senderGathers.complete());
    ASSERT_TRUE(receiverScatters.complete());
    ctl->scriptWrite(0, WriteScript::kComplete);
    ctl->scriptWrite(1, WriteScript::kComplete);
    // Polled in chunk order, writes 0 and 1 are Sent (DATA out) before write 2's failure is seen.
    ctl->scriptWrite(2, WriteScript::kFail);

    auto fut = sender->tx->submit(bufs.srcDescs, bufs.dstDescs, receiver->name);
    // Writes 0 and 1 were released when their DATA went out; write 2 after failRequest.
    ASSERT_TRUE(waitUntil([&] { return ctl->releaseCount.load(std::memory_order_acquire) == 3; }))
        << "the failed write was never handled";
    EXPECT_EQ(fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "FAILURE reported while chunks 0 and 1 could still be scattered into dst";

    receiverScatters.releaseAll();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "the scatters' ACKs were not awaited";
    auto const result = fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kWriteFailed);
    EXPECT_TRUE(result.quiesced);
    for (std::uint32_t chunk = 0; chunk < 2; ++chunk)
    {
        EXPECT_TRUE(deviceBytesEqual(bufs.dst, bufs.off[chunk], bufs.descBytes, kArenaFill))
            << "chunk " << chunk << " was not scattered before the FAILURE resolved";
    }
    EXPECT_TRUE(deviceBytesEqual(bufs.dst, bufs.off[2], bufs.descBytes, 0)) << "chunk 2 was never sent";
    // Reuse dst, then shut the receiver down (joins its scatter workers, syncs the device): nothing may
    // write it again.
    ASSERT_EQ(cudaMemset(bufs.dst, kReusedFill, bufs.total), cudaSuccess);
    receiver->tx->shutdown();
    EXPECT_TRUE(deviceBytesEqual(bufs.dst, 0, bufs.total, kReusedFill)) << "dst was written after it was reused";
    sender->tx->shutdown();
}

// A NACK fails the request but answers only its own chunk: the request waits for the other Sent
// chunk's answer, and not for a chunk whose write is still in flight (its DATA was never sent, so the
// receiver will never scatter or answer it).
TEST(BounceTransportFailure, NackWaitsOnlyForOtherSentChunks)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000); // waiting for chunk 2 would show up as a missed 5 s bound
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("nackWait", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ExecHold senderGathers(*peers->sender->exec, /*leaveFree=*/1); // post index == chunk index
    ASSERT_TRUE(senderGathers.complete());
    peers->ctl->scriptWrite(0, WriteScript::kComplete);
    peers->ctl->scriptWrite(1, WriteScript::kComplete);
    peers->ctl->scriptWrite(2, WriteScript::kStall);

    ScriptedRequest req(*peers, /*chunks=*/3, /*seed=*/93);
    ASSERT_TRUE(waitWant(*peers->receiver, req.rid));
    peers->grant(req.rid, /*firstChunk=*/0, /*endChunk=*/3);
    ASSERT_TRUE(waitData(*peers->receiver, req.rid, 2)) << "chunks 0 and 1 never reached Sent";
    ASSERT_TRUE(waitUntil([&] { return peers->ctl->postCount.load(std::memory_order_acquire) == 3; }))
        << "chunk 2's write was never posted";

    peers->nack(req.rid, 1);
    ASSERT_TRUE(syncSender(*peers));
    EXPECT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "resolved before chunk 0 was answered";
    peers->ack(req.rid, 0);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "waited for chunk 2, whose DATA was never sent";
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerNack);
    EXPECT_TRUE(result.quiesced);
    EXPECT_EQ(peers->ctl->releaseCount.load(std::memory_order_acquire), 2u)
        << "chunk 2's write must still be in flight";

    // End the orphaned write so shutdown skips its 5 s write drain; its deferred cancel follows.
    peers->ctl->scriptWrite(2, WriteScript::kComplete);
    EXPECT_TRUE(waitCancel(*peers->receiver, req.rid));
}

// NACKs are answers too: a failed request whose remaining chunks are all NACKed (e.g. a receiver with a
// CUDA fault NACKs every queued scatter) resolves quiesced at once, not in-doubt at requestTimeoutMs.
TEST(BounceTransportFailure, NackAnswersEndTheWait)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000); // ignoring the NACKs would show up as a missed 5 s bound
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("nackAnswers", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/3, /*seed=*/106);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    peers->nack(req.rid, 2); // fails the request: chunks 0 and 1 are awaited
    ASSERT_TRUE(waitCancel(*peers->receiver, req.rid));
    peers->nack(req.rid, 0);
    peers->nack(req.rid, 1);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "NACKs to a failed request were not taken as answers";
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerNack);
    EXPECT_TRUE(result.quiesced);
}

// Only an answer for an awaited chunk (same chunk index AND region) ends a failed request's wait. A
// duplicate, an answer for a chunk that is not awaited and a wrong-region answer are all ignored.
TEST(BounceTransportFailure, OnlyMatchingAnswersEndTheWait)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("matchAnswers", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/3, /*seed=*/102);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    peers->nack(req.rid, 2); // fails the request: chunks 0 and 1 are awaited
    ASSERT_TRUE(waitCancel(*peers->receiver, req.rid));

    peers->ack(req.rid, 1); // a real answer: only chunk 0 is awaited now
    peers->ack(req.rid, 1); // duplicate
    peers->ack(req.rid, 2); // chunk 2 is not awaited: its NACK already answered it
    peers->receiver->sendTo(peers->senderName, b::encodeAck(req.rid, 0, fakeRegion(0) + 1)); // wrong region
    ASSERT_TRUE(syncSender(*peers));
    EXPECT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "an answer that matches no awaited chunk ended the wait";

    peers->ack(req.rid, 0);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerNack);
    EXPECT_TRUE(result.quiesced);
}

// A failed request that still has unposted chunks (more chunks than maxInflightChunksPerRequest, the
// production norm) stays inert while it awaits answers: a GRANT that crossed its cancel must not make
// it gather, post or send DATA, and once it resolves it must hold no staging region.
TEST(BounceTransportFailure, FailedRequestWithUnpostedChunksStaysInert)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);            // maxInflightChunksPerRequest = 2
    capRegions(c, c.maxInflightChunksPerRequest); // the sender's staging holds exactly two chunks
    auto peers = makeScriptedPeers("unpostedChunks", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ExecHold senderGathers(*peers->sender->exec, /*leaveFree=*/1); // post index == chunk index
    ASSERT_TRUE(senderGathers.complete());
    peers->ctl->scriptWrite(0, WriteScript::kComplete);
    peers->ctl->scriptWrite(1, WriteScript::kStall);

    ScriptedRequest req(*peers, /*chunks=*/4, /*seed=*/107);
    ASSERT_TRUE(waitWant(*peers->receiver, req.rid));
    peers->grant(req.rid, /*firstChunk=*/0, /*endChunk=*/2);
    ASSERT_TRUE(waitData(*peers->receiver, req.rid, 1));
    ASSERT_TRUE(waitUntil([&] { return peers->ctl->postCount.load(std::memory_order_acquire) == 2; }));
    peers->ctl->scriptWrite(1, WriteScript::kFail);
    // Failed: chunk 0 is awaited, and chunks 2 and 3 were never posted.
    ASSERT_TRUE(waitCancel(*peers->receiver, req.rid));

    peers->grant(req.rid, /*firstChunk=*/2, /*endChunk=*/4);
    ASSERT_TRUE(syncSender(*peers));
    EXPECT_EQ(peers->ctl->postCount.load(std::memory_order_acquire), 2u) << "the failed request posted a write";
    EXPECT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    peers->ack(req.rid, 0);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kWriteFailed);
    EXPECT_TRUE(result.quiesced);

    // No staging may stay held: a fresh 2-chunk request needs both regions to get both chunks Sent.
    ScriptedRequest next(*peers, /*chunks=*/2, /*seed=*/108);
    ASSERT_TRUE(grantUntilAllSent(*peers, next)) << "the failed request leaked a staging region";
    EXPECT_EQ(peers->ctl->postCount.load(std::memory_order_acquire), 4u)
        << "the failed request posted a write (only the fresh request's two may follow its own two)";
}

// A receiver that stops answering is assumed dead requestTimeoutMs after its LAST answer: the request
// then resolves in-doubt. Each answer restarts that bound. Bounds are measured from the test's own
// send times, so they hold however the threads are scheduled.
TEST(BounceTransportFailure, SilentReceiverResolvesInDoubtAfterTimeoutSinceLastAnswer)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr auto kTimeout = std::chrono::milliseconds(2000);
    // Room for thread scheduling on top of the bound; far below a second kTimeout.
    constexpr auto kSchedulingSlack = std::chrono::seconds(3);
    auto c = cfg(static_cast<int>(kTimeout.count()));
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("silentReceiver", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/3, /*seed=*/94);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));

    // Fails the request; chunks 0 and 1 still owe an answer.
    peers->nack(req.rid, 2);
    // Let a quarter of the bound pass first: if the ACK below did not restart the deadline, the request
    // would resolve kTimeout/4 early and fail EXPECT_GE(waited, kTimeout).
    std::this_thread::sleep_for(kTimeout / 4);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "resolved before requestTimeoutMs passed since the failure";
    auto const lastAnswerSent = std::chrono::steady_clock::now();
    peers->ack(req.rid, 0); // the last answer: chunk 1 never gets one

    ASSERT_EQ(req.fut.wait_for(kTimeout + kSchedulingSlack), std::future_status::ready)
        << "an unanswered chunk held the request past its bound";
    auto const waited = std::chrono::steady_clock::now() - lastAnswerSent;
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerNack);
    EXPECT_FALSE(result.quiesced) << "chunk 1 was never answered";
    EXPECT_GE(waited, kTimeout) << "the last answer did not restart the requestTimeoutMs bound";
    EXPECT_LT(waited, kTimeout + kSchedulingSlack);
}

// The answer wait also restarts at the failure itself, however long the failing write took: the
// receiver gets a full requestTimeoutMs after the failure to answer a sent chunk.
TEST(BounceTransportFailure, AnswerWaitStartsAtTheFailure)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr auto kTimeout = std::chrono::milliseconds(1500);
    auto c = cfg(static_cast<int>(kTimeout.count()));
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("answerWaitStart", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";
    ExecHold senderGathers(*peers->sender->exec, /*leaveFree=*/1); // post index == chunk index
    ASSERT_TRUE(senderGathers.complete());
    peers->ctl->scriptWrite(0, WriteScript::kComplete);
    peers->ctl->scriptWrite(1, WriteScript::kStall);

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/109);
    ASSERT_TRUE(waitWant(*peers->receiver, req.rid));
    peers->grant(req.rid, /*firstChunk=*/0, /*endChunk=*/2);
    ASSERT_TRUE(waitData(*peers->receiver, req.rid, 1));
    ASSERT_TRUE(waitUntil([&] { return peers->ctl->postCount.load(std::memory_order_acquire) == 2; }));
    // Write 1 stalls for most of requestTimeoutMs (its post was the request's last progress), then fails.
    std::this_thread::sleep_for(kTimeout * 6 / 10);
    peers->ctl->scriptWrite(1, WriteScript::kFail);
    ASSERT_TRUE(waitCancel(*peers->receiver, req.rid));
    // Counted from the stalled write's post, the bound would already be over here.
    std::this_thread::sleep_for(kTimeout / 2);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "resolved in-doubt less than requestTimeoutMs after the failure";
    peers->ack(req.rid, 0);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kWriteFailed);
    EXPECT_TRUE(result.quiesced);
}

// A kNoProgressTimeout means no progress for requestTimeoutMs, so a request with a sent chunk still
// unanswered resolves in-doubt at once instead of waiting another requestTimeoutMs.
TEST(BounceTransportFailure, NoProgressTimeoutWithUnansweredChunkResolvesInDoubtAtOnce)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr auto kTimeout = std::chrono::milliseconds(2000);
    auto c = cfg(static_cast<int>(kTimeout.count()));
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("noProgress", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/95);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    auto const lastAnswerSent = std::chrono::steady_clock::now();
    peers->ack(req.rid, 0); // the request's last progress; chunk 1 stays unanswered

    ASSERT_EQ(req.fut.wait_for(2 * kTimeout), std::future_status::ready)
        << "the no-progress timeout waited another requestTimeoutMs for answers";
    auto const waited = std::chrono::steady_clock::now() - lastAnswerSent;
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kNoProgressTimeout);
    EXPECT_FALSE(result.quiesced) << "chunk 1 was never answered";
    EXPECT_GE(waited, kTimeout);
    EXPECT_LT(waited, 2 * kTimeout);
}

// An abandoned flow (GRANT mispair) dies through checkTimeouts' kNoProgressTimeout. With a Sent chunk
// unanswered it resolves in-doubt within one requestTimeoutMs of its last progress, reporting the
// abandon cause; it must not wait another requestTimeoutMs for answers.
TEST(BounceTransportFailure, AbandonWithSentChunkResolvesInDoubtWithinOneTimeout)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr auto kTimeout = std::chrono::milliseconds(1500);
    auto c = cfg(static_cast<int>(kTimeout.count()));
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("abandonSent", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/110);
    ASSERT_TRUE(waitWant(*peers->receiver, req.rid));
    peers->grant(req.rid, /*firstChunk=*/0, /*endChunk=*/1);
    ASSERT_TRUE(waitData(*peers->receiver, req.rid, 1));
    auto const chunk0Sent = std::chrono::steady_clock::now(); // after the request's last progress
    // Chunk 1's credit is half the chunk: a mispair, so the flow is abandoned.
    std::vector<b::BounceCreditEntry> const undersized{{/*addr=*/0, peers->chunkBytes / 2, /*devId=*/0, fakeRegion(1)}};
    peers->receiver->sendTo(peers->senderName, b::encodeGrant(req.rid, undersized));

    ASSERT_EQ(req.fut.wait_for(kTimeout + kTimeout / 2), std::future_status::ready)
        << "the abandoned flow waited another requestTimeoutMs for answers";
    EXPECT_LT(std::chrono::steady_clock::now() - chunk0Sent, kTimeout + kTimeout / 2);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kProtocolError);
    EXPECT_FALSE(result.quiesced) << "chunk 0 was never answered";
}

// A kNoProgressTimeout with every chunk Sent and none answered resolves in-doubt. ACKs arriving
// afterwards, enough to count every chunk, hit an erased request: the reactor drops them and keeps
// serving (the FIFO barrier below).
TEST(BounceTransportFailure, LateAcksAfterNoProgressTimeoutChangeNothing)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    constexpr auto kTimeout = std::chrono::milliseconds(1000);
    auto c = cfg(static_cast<int>(kTimeout.count()));
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("lateAcks", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/96);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    ASSERT_EQ(req.fut.wait_for(kTimeout + std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(req.fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(req.fut.get().reason, b::BounceFailReason::kNoProgressTimeout);
    EXPECT_FALSE(req.fut.get().quiesced);

    peers->ack(req.rid, 0);
    peers->ack(req.rid, 1);
    EXPECT_TRUE(syncSender(*peers)) << "the reactor stopped after the late ACKs";
}

// forgetPeer while a failed request awaits answers must not fail it again: it keeps its first reason
// and keeps waiting, and a later answer resolves it with that ORIGINAL reason.
TEST(BounceTransportFailure, ForgetPeerWhileAwaitingAnswersKeepsWaiting)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("forgetAwaiting", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/98);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    // Never granted: forgetPeer fails it, which proves the forget was applied.
    ScriptedRequest sentinel(*peers, /*chunks=*/1, /*seed=*/99);

    peers->nack(req.rid, 1);
    // No write was in flight, so the cancel went out at once: the NACK failed the request.
    ASSERT_TRUE(waitCancel(*peers->receiver, req.rid));
    peers->sender->tx->forgetPeer(peers->receiverName);
    ASSERT_EQ(sentinel.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "forgetPeer not applied";
    EXPECT_EQ(sentinel.fut.get().reason, b::BounceFailReason::kPeerDropped);
    EXPECT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "forgetPeer resolved a request that still awaits an answer";

    peers->ack(req.rid, 0); // answers still reach the sender's ROUTER after forgetPeer
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerNack) << "forgetPeer failed the request a second time";
    EXPECT_TRUE(result.quiesced);
}

// forgetPeer fails a request whose chunks were all Sent; it then waits for their answers. ACKs for
// every chunk must resolve it FAILURE (kPeerDropped, quiesced), never SUCCESS, even though the ACK
// count reaches numChunks.
TEST(BounceTransportFailure, AnswersAfterForgetPeerResolveFailureNotSuccess)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("forgetAllSent", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest req(*peers, /*chunks=*/2, /*seed=*/100);
    ASSERT_TRUE(grantUntilAllSent(*peers, req));
    // Never granted: forgetPeer fails it in the same pass, which proves the forget was applied.
    ScriptedRequest sentinel(*peers, /*chunks=*/1, /*seed=*/101);

    peers->sender->tx->forgetPeer(peers->receiverName);
    ASSERT_EQ(sentinel.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready) << "forgetPeer not applied";
    EXPECT_EQ(req.fut.wait_for(std::chrono::seconds(0)), std::future_status::timeout)
        << "forgetPeer resolved a request whose sent chunks were unanswered";

    peers->ack(req.rid, 0);
    peers->ack(req.rid, 1);
    ASSERT_EQ(req.fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto const result = req.fut.get();
    EXPECT_EQ(result.state, kvc::TransferState::kFAILURE) << "a failed request resolved SUCCESS";
    EXPECT_EQ(result.reason, b::BounceFailReason::kPeerDropped);
    EXPECT_TRUE(result.quiesced);
}

// Shutdown resolves every pending request. One already awaiting answers keeps its own reason, and any
// request with a sent chunk unanswered is in-doubt: the receiver may still scatter into its dst.
TEST(BounceTransportFailure, ShutdownResolvesUnansweredRequestsInDoubtWithTheirOwnReason)
{
    if (!bounce_test::hasCuda())
        GTEST_SKIP() << "no CUDA device";
    auto c = cfg(/*timeoutMs=*/30000);
    c.maxInflightChunksPerRequest = 3;
    capRegions(c, c.maxInflightChunksPerRequest);
    auto peers = makeScriptedPeers("shutdownAwaiting", c);
    if (!peers)
        GTEST_SKIP() << "NIXL agent/backend unavailable";

    ScriptedRequest failed(*peers, /*chunks=*/2, /*seed=*/104);
    ASSERT_TRUE(grantUntilAllSent(*peers, failed));
    peers->nack(failed.rid, 1); // fails it: chunk 0 is awaited
    ASSERT_TRUE(waitCancel(*peers->receiver, failed.rid));
    // Still live: its only chunk is Sent and unanswered.
    ScriptedRequest live(*peers, /*chunks=*/1, /*seed=*/105);
    ASSERT_TRUE(grantUntilAllSent(*peers, live));

    peers->sender->tx->shutdown();
    ASSERT_EQ(failed.fut.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_EQ(failed.fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(failed.fut.get().reason, b::BounceFailReason::kPeerNack) << "shutdown replaced the original reason";
    EXPECT_FALSE(failed.fut.get().quiesced) << "chunk 0 was never answered";
    ASSERT_EQ(live.fut.wait_for(std::chrono::seconds(0)), std::future_status::ready);
    EXPECT_EQ(live.fut.get().state, kvc::TransferState::kFAILURE);
    EXPECT_EQ(live.fut.get().reason, b::BounceFailReason::kShutdown);
    EXPECT_FALSE(live.fut.get().quiesced) << "its sent chunk was never answered";
}
