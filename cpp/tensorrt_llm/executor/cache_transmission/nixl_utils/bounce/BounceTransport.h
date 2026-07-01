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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceConfig.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceMessage.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferPlan.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ControlChannel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/CreditScheduler.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/ExecPool.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/TransferEngine.h"
#include "tensorrt_llm/executor/transferAgent.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

// ============================================================================
// Bounce v2 reactor
// ----------------------------------------------------------------------------
// One BounceTransport per agent. Concurrency is collapsed onto ONE IO thread +
// M scatter workers, so the credit/request state needs almost no
// locking. The reactor is decomposed into three collaborators that all live on
// (and are owned by) that single IO thread:
//
//   BounceContext  — the shared, single-IO-thread-owned dependencies both roles
//                    operate on: the injected channel/engine/arena/exec, the one
//                    CreditScheduler (region allocator), and sendGrants().
//   BounceSender   — the [S] role: submit() -> WANT, GRANT -> gather+write, ACK
//                    -> resolve. Owns the request table + sender-side orphan state.
//   BounceReceiver — the [R] role: WANT -> grant regions, DATA -> scatter, ACK.
//                    Owns the scatter workers + scatter job/done queues.
//   BounceTransport— the thin reactor: owns the three above + the IO thread, and
//                    routes control messages / drains to sender vs receiver.
//
// IO thread loop (single-threaded owner of scheduler + sender request table):
//   1. recv() one control message (short timeout) and dispatch it:
//        [S] GRANT(rid, credits)  -> per credit: acquireLocal(bytes) a shared-arena region + borrow
//                                    an ExecCtx, launch gather + cudaEventRecord (NO sync); postWrite
//                                    is deferred to drainGatherReady() once the gather event signals
//                                    (so a gather delayed behind other GPU work never blocks IO).
//        [S] ACK(rid, chunk, h)   -> releaseLocal() that chunk's region; acked++;
//                                    all acked -> resolve the request's future SUCCESS.
//        [R] WANT(rid, sizes[])   -> scheduler.onWant(peer:rid, sizes) -> send GRANT(s).
//        [R] DATA(rid, chunk, h, plan) -> enqueue a scatter job over region `h`.
//   2. poll TransferEngine on every in-flight write; on Done send DATA (data has
//      landed at the remote) and mark on-wire. THIS is why no notifMsg is needed.
//   3. drain scatter completions posted by workers: scheduler.onScatterDone -> re-GRANT,
//      and send ACK back to the sender.
//
// Scatter workers (receiver): pop a job, scatter region->dst via the kernel, sync, then
// post a completion back to the IO thread.
//
// Sender requests are keyed by a monotonic rid; the receiver-side scheduler is keyed by
// "peer\x1f rid" so multiple concurrent requests from one peer are independent flows.
//
// Lifetime: submit() returns a shared_future<TransferState>; the IO thread resolves it
// SUCCESS on full ACK, FAILURE on transfer error / shutdown — never hangs.
//
// TransferEngine & ControlChannel are injected (not owned) so the same reactor runs
// unchanged in tests and production (both over the real NIXL/zmq stack; tests may inject a
// fault-injecting engine to exercise the failure path).
// ============================================================================

/// Shared, single-IO-thread-owned dependencies both roles operate on. Holds the injected
/// channel/engine/arena/exec (borrowed, not owned), the one CreditScheduler that carves the shared
/// arena for BOTH roles, and sendGrants(). Owned by value by BounceTransport; referenced by the
/// sender and receiver. Not thread-safe by itself — every member is touched only on the IO thread
/// (the scheduler) or is itself internally synchronized (channel/engine/exec/arena).
class BounceContext
{
public:
    BounceContext(std::string selfName, BounceConfig cfg, int deviceId, ControlChannel* channel, TransferEngine* engine,
        BounceArena* arena, ExecPool* exec)
        : selfName(std::move(selfName))
        , cfg(cfg)
        , deviceId(deviceId)
        , channel(channel)
        , engine(engine)
        , arena(arena)
        , exec(exec)
        , scheduler(arena->baseAddr(), cfg.arenaBytes, cfg.minBlock, cfg.effectiveWindow())
    {
    }

    BounceContext(BounceContext const&) = delete;
    BounceContext& operator=(BounceContext const&) = delete;

    /// Split a scheduler GRANT batch (keyed by flow id "peer<sep>rid") back to (peer, rid) and send a
    /// GRANT control message to each peer. Used by BOTH roles (the receiver grants incoming regions;
    /// the sender's releaseLocal/reclaim can free arena bytes that re-grant a waiting remote flow).
    void sendGrants(std::vector<Grant> const& grants);

    std::string selfName;
    BounceConfig cfg;
    int deviceId{};
    ControlChannel* channel{};
    TransferEngine* engine{};
    BounceArena* arena{};      // ONE shared data buffer: receiver grants + local gather both carve regions from it
    ExecPool* exec{};          // gather/scatter exec contexts (streams/scratch), borrowed per kernel
    CreditScheduler scheduler; // the single region allocator; touched only by the IO thread
    std::atomic<bool> stop{false};
};

/// Receiver role ([R]): WANT -> grant regions, DATA -> scatter into the caller's KV, then ACK. Owns
/// the M scatter workers and the IO<->worker job/done queues. All public methods run on the IO thread.
class BounceReceiver
{
public:
    explicit BounceReceiver(BounceContext& ctx);

    /// Launch the scatter workers (call once after construction, before the IO loop starts).
    void startWorkers();
    /// Wake workers so they observe ctx.stop and exit (notify; call before joinWorkers()).
    void wake();
    /// Join the scatter workers (call during shutdown after wake()).
    void joinWorkers();

    void onWant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob);
    void onData(std::string const& peer, BounceMsgHeader const& h, std::string const& blob);
    /// Drain scatter completions posted by workers (ACK + free region). Returns true if any.
    bool drainScatterDone();
    /// A peer is gone: reclaim every receiver-side flow of that peer (deferring regions a worker is
    /// still reading) and drop its not-yet-started scatter jobs.
    void forget(std::string const& peer);

    /// True while any scatter is enqueued-or-running (drives the IO loop's 0ms busy-poll).
    [[nodiscard]] bool busy() const
    {
        return !mScattering.empty();
    }

private:
    struct ScatterJob
    {
        std::string key; // "peer\x1f rid"
        std::string peer;
        std::uint64_t rid{};
        std::uint32_t chunkIdx{};
        std::uint64_t offset{};      // receiver arena region offset (the granted region handle) to scatter from
        std::uint64_t regionBytes{}; // byte size of the buddy block backing `offset` (bounds scatter reads)
        std::vector<BounceScatterEntry> entries;
    };

    struct ScatterDone
    {
        std::string key;
        std::string peer;
        std::uint64_t rid{};
        std::uint32_t chunkIdx{};
        std::uint64_t offset{}; // receiver arena region offset freed once scattered
        bool ok{true};          // scatter kernel succeeded -> ACK; false -> skip ACK (sender times out)
    };

    /// The set of incoming regions currently scattering — the `busy` set passed to the scheduler's
    /// reclaim calls so a region a worker is still reading is deferred, not re-granted.
    std::unordered_set<std::uint64_t> scatteringRegions() const;
    void scatterWorkerLoop();

    BounceContext& mCtx;
    std::vector<std::thread> mWorkers;

    // scatter job queue (IO thread -> workers)
    std::mutex mJobMu;
    std::condition_variable mJobCv;
    std::deque<ScatterJob> mJobs;

    // scatter completion queue (workers -> IO thread)
    std::mutex mDoneMu;
    std::deque<ScatterDone> mDone;

    // Every incoming region with a scatter job enqueued-or-running, mapped to whether its flow was
    // reclaimed (peer gone / cancel) mid-scatter (orphaned == true). Touched only by the IO thread
    // (onData / drainScatterDone / forget / onWant) so it needs no lock. Membership stops forget()
    // from re-granting an incoming region a worker is still reading; the orphaned flag decides whether
    // drainScatterDone frees the region via freeOrphanRegion (flow already gone) or onScatterDone
    // (normal completion).
    std::unordered_map<std::uint64_t, bool> mScattering;
};

/// Sender role ([S]): submit() announces a WRITE via WANT; GRANT -> gather into a local arena region
/// + RDMA-write; ACK -> resolve the chunk; all chunks ACKed -> resolve the request SUCCESS. Owns the
/// request table + the sender-side deferred-cleanup state. All methods except submit() run on the IO
/// thread; submit() is called from app threads and is mutex-guarded against the IO thread.
class BounceSender
{
public:
    explicit BounceSender(BounceContext& ctx);

    /// Submit a WRITE of (src -> dst) descriptors to `peer`. Returns a future that resolves
    /// kSUCCESS once every chunk is scattered+ACKed, or kFAILURE on error/shutdown.
    [[nodiscard]] std::shared_future<TransferState> submit(
        TransferDescs const& srcDescs, TransferDescs const& dstDescs, std::string const& peer);

    void onGrant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob);
    void onAck(std::string const& peer, BounceMsgHeader const& h);
    /// Issue the RDMA write for any chunk whose gather kernel has now completed (poll ctx->event,
    /// non-blocking). Decouples the IO thread from gather latency on a shared GPU. Returns true if it
    /// posted a write (or failed a request) this pass — used to drive the IO loop's idle backoff.
    bool drainGatherReady();
    bool pollSenderHandles();
    /// Free local regions whose failed request still had an RDMA write in flight, once that write
    /// reaches a terminal state (so the NIC is done reading the source). Returns true if any freed.
    bool drainOrphanLocal();
    /// Retry parked credits for every request (called on the IO loop after ACKs free regions).
    void drainPendingPosts();
    /// Fail requests that have made no progress within leaseTimeoutMs (e.g. peer never granted).
    void checkTimeouts();
    /// A peer is gone: fail any in-flight request targeting it so its wait() returns.
    void forget(std::string const& peer);
    /// Shutdown: fail every still-pending request (releasing in-flight handles + exec contexts) and
    /// release deferred orphan-local handles. Called after the device has been synced.
    void failAll();

    /// True while a local gather region is held or an orphan-local write is in flight (drives the IO
    /// loop's 0ms busy-poll). Called on the IO thread.
    [[nodiscard]] bool busy() const
    {
        return mCtx.scheduler.localHeldCount() > 0 || !mOrphanLocal.empty();
    }

private:
    // A chunk's transfer state — the per-chunk view of the sender Request state machine.
    // Linear progression Gathering -> Writing -> Sent, with GatherFailed as the one
    // off-ramp (a gather whose launch/event-record failed). Replaces a former trio of bools whose
    // illegal combinations (e.g. !writePosted && dataSent) were only ruled out by convention.
    enum class PostState
    {
        Gathering,    // gather kernel launched + event recorded; write deferred until the event signals
        GatherFailed, // gather launch / event-record failed -> drainGatherReady fails the request
                      // (an unrecorded event queries as "complete", so its event must NOT be trusted)
        Writing,      // gather done; RDMA write issued (xfer valid); polling getXferStatus
        Sent,         // getXferStatus==Done -> DATA emitted; xfer released; awaiting ACK
    };

    struct Posted
    {
        std::uint32_t chunkIdx{};
        std::uint64_t localOffset{};  // shared-arena region held for gather/write until ACK (OUTGOING_HELD)
        ExecCtx* ctx{nullptr};        // gather exec context, borrowed while the gather runs (until Writing)
        std::uint64_t xfer{};         // TransferEngine handle (valid once state == Writing)
        std::uint64_t remoteHandle{}; // receiver's region handle (its arena offset); echoed in DATA
        // Remote write target (from the GRANT), kept so postWrite can be issued LATER — after the
        // gather kernel completes — instead of blocking the IO thread on cudaStreamSynchronize.
        // The gather may be delayed behind unrelated GPU work (shared device), so we never sync;
        // drainGatherReady() polls ctx->event and posts the write only when the gather is done.
        std::uint64_t remoteAddr{};
        std::uint32_t remoteDevId{};
        std::uint32_t writeBytes{};
        PostState state{PostState::Gathering};
    };

    struct Request
    {
        std::string peer;
        std::uint32_t numChunks{};
        BounceTransferPlan plan;
        std::uint32_t nextPost{0};
        std::uint32_t acked{0};
        std::vector<Posted> posted;
        // Credits granted by the receiver but not yet posted because no gather region / exec context
        // was free (the shared arena is used across peers and can be oversubscribed). The IO thread
        // NEVER blocks on acquire; instead it parks the credit here and retries each loop iteration
        // as ACKs free regions. FIFO: paired with chunk indices in order.
        std::deque<BounceCreditEntry> pendingCredits;
        std::shared_ptr<std::promise<TransferState>> promise;
        // Last time this request made forward progress (granted+posted a chunk, or got an ACK).
        // A request stuck with no progress for leaseTimeoutMs (e.g. the peer never GRANTs because
        // it is unreachable / not bounce-ready) is failed rather than hanging wait() forever.
        std::chrono::steady_clock::time_point lastProgress;
    };

    // Regions of a FAILED request whose RDMA write was still in flight (state == Writing).
    // The NIC may still be reading the source region, so recycling is deferred until the write reaches
    // a terminal state — drainOrphanLocal() polls and only then releases the xfer handle + returns the
    // region. IO-thread-only (no lock). (xfer handle, arena offset).
    struct OrphanLocal
    {
        std::uint64_t xfer{};
        std::uint64_t offset{};
        std::string peer{};
        std::uint64_t rid{};
    };

    /// Post as many of `req`'s parked credits as free regions + exec contexts allow (non-blocking).
    /// Pairs each credit with the next chunk; stops when the arena/ExecPool is empty (rest parked).
    /// Caller MUST hold mReqMu.
    void pumpRequest(std::uint64_t rid, Request& req);
    void failRequest(std::uint64_t rid, Request& req);

    BounceContext& mCtx;

    // request table (IO thread + submit() from app threads)
    std::mutex mReqMu;
    std::unordered_map<std::uint64_t, Request> mRequests;
    std::atomic<std::uint64_t> mNextRid{1};

    std::vector<OrphanLocal> mOrphanLocal;
    // A failed request's cancel (empty WANT) must be DEFERRED while any of its RDMA writes are still
    // in flight: those writes are landing on the RECEIVER's regions, and an early cancel would let the
    // receiver reclaim + re-grant those regions under the in-flight write -> cross-node corruption.
    // drainOrphanLocal() sends the cancel once the flow's last in-flight write reaches terminal.
    // rid -> peer. IO-thread-only.
    std::unordered_map<std::uint64_t, std::string> mPendingCancel;
};

/// The thin reactor: owns the shared context, the sender + receiver, and the IO thread. Routes each
/// control message to the right role and drives both roles' per-loop drains. One per agent.
class BounceTransport
{
public:
    /// @param arena ONE shared data buffer serving BOTH roles: receiver (remote senders' RDMA-write
    /// targets, granted as variable regions by the scheduler) and sender (local gather staging, via
    /// acquireLocal). It must already be registered with `engine` by the caller (or be a no-op
    /// engine). @param exec the gather/scatter execution contexts (streams/scratch) borrowed for the
    /// duration of one kernel. `channel`/`engine`/`arena`/`exec` are borrowed for the transport's
    /// lifetime. (Most disagg agents are sender-only OR receiver-only, so a single arena avoids
    /// wasting a second one.)
    BounceTransport(std::string selfName, BounceConfig cfg, int deviceId, ControlChannel* channel,
        TransferEngine* engine, BounceArena* arena, ExecPool* exec);
    ~BounceTransport();

    BounceTransport(BounceTransport const&) = delete;
    BounceTransport& operator=(BounceTransport const&) = delete;

    /// Register where to reach `peer` (its ControlChannel endpoint).
    void addPeer(std::string const& peer, std::string const& endpoint);

    /// A peer is gone (NixlTransferAgent::invalidateRemoteAgent). Reclaim its receiver-side
    /// credits and fail any in-flight sender requests to it. Thread-safe: queued and applied on
    /// the IO thread so the scheduler stays single-threaded.
    void forgetPeer(std::string const& peer);

    /// Submit a WRITE of (src -> dst) descriptors to `peer`. Returns a future that resolves
    /// kSUCCESS once every chunk is scattered+ACKed, or kFAILURE on error/shutdown.
    [[nodiscard]] std::shared_future<TransferState> submit(
        TransferDescs const& srcDescs, TransferDescs const& dstDescs, std::string const& peer)
    {
        return mSender.submit(srcDescs, dstDescs, peer);
    }

    /// Stop threads, fail any in-flight requests, join. Safe to call once.
    void shutdown();

private:
    void ioLoop();
    void dispatch(std::string const& peer, std::string const& blob);
    /// Apply queued forgetPeer() requests on the IO thread (reclaim receiver credits + fail
    /// sender requests to the gone peer).
    void drainForgets();

    BounceContext mCtx; // declared first: sender/receiver hold a reference to it
    BounceReceiver mReceiver;
    BounceSender mSender;

    std::thread mIoThread;
    // IO-loop idle backoff: consecutive "busy poll but nothing happened" iterations. When in-flight
    // work exists (so the poll timeout is 0ms) but a gather is stalled behind unrelated GPU kernels,
    // the loop would otherwise spin a core at 100% on cudaEventQuery. After a threshold of no-progress
    // spins we sleep briefly. Reset on any control message or forward progress. IO-thread-only.
    std::uint32_t mIdleSpins{0};

    // forgetPeer() requests queued from app threads -> applied on the IO thread
    std::mutex mForgetMu;
    std::vector<std::string> mForgetPeers;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
