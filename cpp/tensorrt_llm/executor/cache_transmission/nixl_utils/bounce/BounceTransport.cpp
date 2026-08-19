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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransport.h"

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceNvtx.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

char const* toString(BounceFailReason reason)
{
    // Prefixed with the module name: these strings travel up through TransferStatus::
    // getLastStatusStr() into upper-layer error logs, where "RDMA write failed" alone would not
    // say WHICH transport failed.
    switch (reason)
    {
    case BounceFailReason::kNone: return "bounce: none";
    case BounceFailReason::kPlanRejected: return "bounce: plan rejected (request did not fit a transfer plan)";
    case BounceFailReason::kNoProgressTimeout: return "bounce: no GRANT/ACK progress within requestTimeoutMs";
    case BounceFailReason::kPeerDropped: return "bounce: peer dropped (forgetPeer/invalidateRemoteAgent)";
    case BounceFailReason::kGatherFailed: return "bounce: gather kernel failed (CUDA error)";
    case BounceFailReason::kWriteFailed: return "bounce: RDMA write failed";
    case BounceFailReason::kProtocolError: return "bounce: protocol error (GRANT mispair/plan overflow)";
    case BounceFailReason::kShutdown: return "bounce: transport shut down while pending";
    }
    return "bounce: unknown";
}

namespace
{
constexpr char kSep = '\x1f'; // unit separator: agent names won't contain it

// Split large copy runs into pieces of this size when building the batched-copy plan arrays. The
// copy kernel assigns ONE thread block per plan entry, so a plan of a few huge coalesced runs would
// use only a few SMs; splitting restores the grid-level parallelism the pre-coalescing per-desc plan
// had, without giving up the small wire messages.
constexpr std::uint32_t kCopySplitBytes = 64U << 10;

// This entry's split budget: the scratch slots still free after reserving ONE slot for every raw
// entry not yet appended (each needs at least one). Never below 1.
std::size_t splitBudget(std::size_t appended, std::size_t rawRemaining, std::size_t maxEntries)
{
    std::size_t const reserved = appended + rawRemaining;
    return maxEntries > reserved ? maxEntries - reserved : 1;
}

// Number of pieces appendSplitInto() will emit for one raw entry under `maxPieces`: one per full
// kCopySplitBytes piece up to the budget, the remainder as a single (possibly oversized) entry.
// Used for the exact-count pass that lets callers write the plan arrays straight into the pinned
// buffer (the [srcs|dsts|sizes] layout needs the total BEFORE the first write).
std::size_t piecesFor(std::uint32_t size, std::size_t maxPieces)
{
    std::size_t const want = (static_cast<std::size_t>(size) + kCopySplitBytes - 1) / kCopySplitBytes;
    return std::min(std::max<std::size_t>(want, 1), std::max<std::size_t>(maxPieces, 1));
}

// Plan-array views into an exec context's pinned buffer: [srcs(n) | dsts(n) | sizes(n)] — the
// layout launchPrepared() consumes. Callers fill these IN PLACE: one write pass replaces the old
// build-std::vectors-then-memcpy-into-pinned flow (two full passes over the plan arrays per chunk).
struct PlanBufs
{
    std::uint64_t* srcs;
    std::uint64_t* dsts;
    std::uint32_t* sizes;
};

PlanBufs planBufs(ExecCtx* ctx, std::size_t n)
{
    auto* host = static_cast<std::uint8_t*>(ctx->hostPinned);
    std::size_t const b64 = n * sizeof(std::uint64_t);
    return {reinterpret_cast<std::uint64_t*>(host), reinterpret_cast<std::uint64_t*>(host + b64),
        reinterpret_cast<std::uint32_t*>(host + 2 * b64)};
}

// Write (src, dst, size) into the plan buffers at `idx`, split into <= kCopySplitBytes pieces but
// at most `maxPieces` entries (>= 1) — when the budget runs out the remainder goes in as ONE
// oversized entry (the kernel's strided loop handles any size, so an unsplit entry only costs
// parallelism, never correctness). Emits exactly piecesFor(size, maxPieces) entries.
void appendSplitInto(PlanBufs const& bufs, std::size_t& idx, std::uint64_t src, std::uint64_t dst, std::uint32_t size,
    std::size_t maxPieces)
{
    while (size > kCopySplitBytes && maxPieces > 1)
    {
        bufs.srcs[idx] = src;
        bufs.dsts[idx] = dst;
        bufs.sizes[idx] = kCopySplitBytes;
        ++idx;
        src += kCopySplitBytes;
        dst += kCopySplitBytes;
        size -= kCopySplitBytes;
        --maxPieces;
    }
    bufs.srcs[idx] = src;
    bufs.dsts[idx] = dst;
    bufs.sizes[idx] = size;
    ++idx;
}

// Number of plan entries an exec context's scratch can hold ([srcs|dsts|sizes] packed).
std::size_t maxPlanEntries(ExecCtx const* ctx)
{
    return ctx->scratchBytes / (2 * sizeof(std::uint64_t) + sizeof(std::uint32_t));
}

std::string makeKey(std::string const& peer, std::uint64_t rid)
{
    return peer + kSep + std::to_string(rid);
}

std::pair<std::string, std::uint64_t> splitKey(std::string const& key)
{
    auto pos = key.rfind(kSep);
    return {key.substr(0, pos), std::strtoull(key.c_str() + pos + 1, nullptr, 10)};
}

// Launch the batched copy over plan arrays the CALLER already wrote into the exec context's pinned
// host buffer (via planBufs()/appendSplitInto(): [srcs(n)|dsts(n)|sizes(n)]). Direction-agnostic
// (gather or scatter); the data region (arena offset) is encoded into the srcs/dsts by the caller.
// zeroCopy (default on) skips the H2D of the plan arrays — the kernel reads them straight from
// pinned host via ctx->hostPinnedDev, removing the H2D-then-kernel serialization. Off stages them
// in device scratch first (faster on machines where mapped-host reads are slow).
cudaError_t launchPrepared(ExecCtx* ctx, std::size_t n, bool zeroCopy)
{
    if (n == 0)
    {
        return cudaSuccess;
    }
    std::size_t const b64 = n * sizeof(std::uint64_t);
    std::size_t const b32 = n * sizeof(std::uint32_t);
    // The plan arrays must fit this context's scratch/hostPinned (both sized for maxDescsPerChunk).
    // Callers bound n before writing (planBufs is capacity-unchecked), so this is defense in depth.
    if (2 * b64 + b32 > ctx->scratchBytes)
    {
        return cudaErrorInvalidValue;
    }
    auto* host = static_cast<std::uint8_t*>(ctx->hostPinned);
    // Pick the DEVICE-accessible base for the plan arrays: the pinned buffer's device alias (zeroCopy)
    // or the device scratch we H2D-copy into.
    std::uint8_t* base = nullptr;
    if (zeroCopy && ctx->hostPinnedDev != nullptr)
    {
        base = static_cast<std::uint8_t*>(ctx->hostPinnedDev);
    }
    else
    {
        base = static_cast<std::uint8_t*>(ctx->scratch);
        cudaError_t const st = cudaMemcpyAsync(base, host, 2 * b64 + b32, cudaMemcpyHostToDevice, ctx->stream);
        if (st != cudaSuccess)
        {
            return st;
        }
    }
    auto* dsrcs = reinterpret_cast<std::uint64_t*>(base);
    auto* ddsts = reinterpret_cast<std::uint64_t*>(base + b64);
    auto* dsizes = reinterpret_cast<std::uint32_t*>(base + 2 * b64);
    auto const n32 = static_cast<std::uint32_t>(n); // n <= scratchBytes/20, far below u32 max
    return launchBatchedCopy(dsrcs, ddsts, dsizes, n32, ctx->stream);
}
} // namespace

// ============================================================================
// BounceContext
// ============================================================================

void BounceContext::sendGrants(std::vector<Grant> const& grants)
{
    if (grants.empty())
    {
        return;
    }
    // grants are keyed by flow id ("peer<sep>rid"); split back to (agent name, rid) to address them.
    std::unordered_map<std::string, std::vector<BounceCreditEntry>> byFlow;
    for (auto const& g : grants)
    {
        // Carry OUR (receiver) device id so the sender writes the remote desc to the right GPU
        // even if the two agents don't share a device index. `regionHandle` (our arena offset) is
        // echoed back in DATA so we can locate + free the region.
        byFlow[g.flow].push_back(BounceCreditEntry{g.addr, g.len, static_cast<std::uint32_t>(deviceId), g.offset});
    }
    for (auto const& [flow, creds] : byFlow)
    {
        auto [peer, rid] = splitKey(flow);
        channel->sendTo(peer, encodeGrant(rid, creds));
    }
}

// ============================================================================
// BounceReceiver — [R] role
// ============================================================================

BounceReceiver::BounceReceiver(BounceContext& ctx)
    : mCtx(ctx)
{
}

void BounceReceiver::startWorkers()
{
    std::uint32_t const workers = mCtx.cfg.scatterWorkerCount > 0 ? mCtx.cfg.scatterWorkerCount : 1;
    mWorkers.reserve(workers);
    for (std::uint32_t i = 0; i < workers; ++i)
    {
        mWorkers.emplace_back(&BounceReceiver::scatterWorkerLoop, this);
    }
}

void BounceReceiver::wake()
{
    // Take mJobMu so the notify is ordered against a worker's predicate check. The wait predicate
    // reads mCtx.stop, which shutdown() sets WITHOUT holding mJobMu — a naked notify_all can then
    // fire in the window between a worker evaluating the predicate false and parking, and is lost
    // forever (no later notifier exists once the IO thread is joined), hanging joinWorkers().
    std::lock_guard<std::mutex> lk(mJobMu);
    mJobCv.notify_all();
}

void BounceReceiver::joinWorkers()
{
    for (auto& t : mWorkers)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    // Workers are gone; any never-dequeued job still holds an open queue-wait span — close them so
    // shutdown doesn't leave dangling NVTX ranges (their regions die with the arena right after).
    std::lock_guard<std::mutex> lk(mJobMu);
    for (auto& j : mJobs)
    {
        bounceRangeEnd(j.nvtxQueue);
    }
}

void BounceReceiver::onWant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    std::vector<std::uint32_t> chunkBytes;
    std::string endpoint;
    if (!decodeWant(blob, h, chunkBytes, endpoint))
    {
        return;
    }
    // Self-bootstrap the reverse control path. Disaggregated serving supports one-directional
    // metadata exchange: the KV sender may load our agent metadata without us loading the sender's.
    // In that case we have no reverse route for GRANT/ACK, so register the sender endpoint carried
    // by WANT here (addPeer is idempotent). A cancel is still honored when registration fails so it
    // can reclaim any flow state left by an earlier, valid WANT.
    bool reversePathReady = false;
    try
    {
        reversePathReady = mCtx.channel->addPeer(peer, endpoint);
    }
    catch (tensorrt_llm::common::TllmException const& e)
    {
        // A malformed WANT is peer input on the reactor thread. Reject it without allowing an
        // exception to escape the thread; handshake registration still propagates this error to
        // the caller because a ZMQ endpoint is mandatory there.
        TLLM_LOG_WARNING(
            "BounceTransport(%s): rejected WANT from peer %s: %s", mCtx.selfName.c_str(), peer.c_str(), e.what());
    }
    auto const key = makeKey(peer, h.requestId);
    if (isCancelWant(chunkBytes))
    {
        // Explicit cancel/abort (the sender failed or retracted): precisely free this flow's
        // granted-but-unwritten regions now — otherwise they stay held until peer loss and a
        // long-running receiver leaks up to one request's in-flight allocation cap per failed rid.
        // Any region whose scatter is still running is deferred (flagged orphaned in mScattering,
        // freed on completion).
        // Immediate free (no quarantine) is safe HERE because a cancel is sender-initiated: the
        // sender defers it until its last in-flight RDMA write reached a terminal state
        // (mPendingCancel / drainOrphanLocal), so nothing can still be writing these regions.
        std::vector<std::uint64_t> deferred;
        mCtx.sendGrants(mCtx.scheduler.reclaimFlow(key, scatteringRegions(), deferred));
        for (auto off : deferred)
        {
            mScattering[off] = true;
        }
        return;
    }
    if (!reversePathReady)
    {
        return;
    }
    // A non-empty WANT has no retransmission path (submit() sends it exactly once per fresh rid),
    // so one for an already-tracked flow is a replay or rid collision. Re-queueing would re-grant
    // over the still-held regions — the sender never writes the extras, so they leak — and the
    // lease refresh inside onWant would keep the flow forever off staleFlows(), defeating the
    // reclaim path that exists for exactly this state. Drop it, mirroring the duplicate-DATA and
    // stale-ACK handling. (Check-then-act is safe: all flow-state mutation happens on this IO
    // thread; app threads only acquireLocal().)
    if (mCtx.scheduler.knowsFlow(key))
    {
        TLLM_LOG_WARNING("BounceTransport(%s): dropped duplicate WANT from peer %s rid=%llu", mCtx.selfName.c_str(),
            peer.c_str(), static_cast<unsigned long long>(h.requestId));
        return;
    }
    mCtx.sendGrants(mCtx.scheduler.onWant(key, chunkBytes));
}

void BounceReceiver::onData(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    // Covers DATA decode + scatter-job enqueue on the receiver IO thread — the receiver-side
    // software leg of the sender's ackWait (the message copy cost scales with the entry count).
    BounceNvtxScope onDataScope(kNvtxOnData, "onData rid=%llu chunk=%u bytes=%zu",
        static_cast<unsigned long long>(h.requestId), h.chunkIdx, blob.size());
    std::vector<BounceScatterRun> entries;
    if (!decodeScatter(blob, h, entries))
    {
        return;
    }
    auto const key = makeKey(peer, h.requestId);
    // Drop a DATA whose region this flow no longer holds — it was cancelled/reclaimed (e.g. an empty
    // WANT raced ahead of this DATA), and the region may have been re-granted to another flow.
    // Scattering it would read a freed/re-owned region and corrupt that other flow's data.
    if (!mCtx.scheduler.heldByFlow(key, h.regionHandle))
    {
        return;
    }
    ScatterJob job;
    job.key = key;
    job.peer = peer;
    job.rid = h.requestId;
    job.chunkIdx = h.chunkIdx;
    job.offset = h.regionHandle;
    // Capture the granted region's byte size HERE (IO thread) — the scatter worker cannot query the
    // IO-thread-only scheduler. Used to bound scatter reads to THIS flow's region (below).
    job.regionBytes = mCtx.scheduler.regionBytes(h.regionHandle);
    job.entries = std::move(entries);
    job.nvtxQueue = bounceRangeStart(
        kNvtxScatterQueue, "scatterQueue rid=%llu chunk=%u", static_cast<unsigned long long>(h.requestId), h.chunkIdx);
    // DATA is not retransmitted by this protocol. An occupied region therefore indicates stale or
    // malformed input; drop it instead of adding replay state or risking two readers of one grant.
    if (!mScattering.emplace(job.offset, false).second)
    {
        TLLM_LOG_WARNING("BounceTransport(%s): dropping duplicate DATA peer=%s rid=%llu chunk=%u region=%llu",
            mCtx.selfName.c_str(), peer.c_str(), static_cast<unsigned long long>(h.requestId), h.chunkIdx,
            static_cast<unsigned long long>(h.regionHandle));
        bounceRangeEnd(job.nvtxQueue);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mJobMu);
        mJobs.emplace_back(std::move(job));
    }
    mJobCv.notify_one();
}

std::unordered_set<std::uint64_t> BounceReceiver::scatteringRegions() const
{
    std::unordered_set<std::uint64_t> busy;
    busy.reserve(mScattering.size());
    for (auto const& [off, orphaned] : mScattering)
    {
        busy.insert(off);
    }
    return busy;
}

bool BounceReceiver::drainScatterDone()
{
    std::deque<ScatterDone> done;
    {
        std::lock_guard<std::mutex> lk(mDoneMu);
        done.swap(mDone);
    }
    if (done.empty())
    {
        return false;
    }
    bool const didWork = true;
    // Bookkeeping only (the ACK itself was sent by the worker): region frees + re-grants.
    BounceNvtxScope drainScope(kNvtxDoneDrain, "doneDrain n=%zu", done.size());
    for (auto& d : done)
    {
        // The ACK was already sent by the scatter worker itself (latency: it is on the sender's
        // ackWait critical path); only the region bookkeeping happens here, on the IO thread.
        // Worker finished reading this region. Was its flow reclaimed (peer gone / cancel) mid-scatter?
        auto it = mScattering.find(d.offset);
        bool const orphaned = (it != mScattering.end()) && it->second;
        if (it != mScattering.end())
        {
            mScattering.erase(it);
        }
        if (orphaned)
        {
            // Flow was reclaimed while this scatter ran; the region was kept out of the arena so it
            // couldn't be re-granted under the worker. Now it's safe to free.
            mCtx.sendGrants(mCtx.scheduler.freeOrphanRegion(d.offset));
        }
        else
        {
            mCtx.sendGrants(mCtx.scheduler.onScatterDone(d.key, d.offset));
        }
    }
    return didWork;
}

void BounceReceiver::forget(std::string const& peer)
{
    // Reclaim every flow this peer was granted ("peer\x1f rid").
    // (1) Drop this peer's not-yet-started scatter jobs — no point scattering for a gone peer; their
    //     incoming regions are no longer busy and get freed by the reclaim below.
    {
        std::lock_guard<std::mutex> lk(mJobMu);
        std::deque<ScatterJob> keep;
        for (auto& j : mJobs)
        {
            if (j.peer == peer)
            {
                bounceRangeEnd(j.nvtxQueue); // job dropped, close its queue-wait span
                mScattering.erase(j.offset);
            }
            else
            {
                keep.push_back(std::move(j));
            }
        }
        mJobs.swap(keep);
    }
    // (2) Regions of this peer still scattering are reads already RUNNING in a worker — they must not
    //     be re-granted until the worker finishes. reclaimByPrefix defers those; we flag them orphaned
    //     in mScattering so drainScatterDone frees them via freeOrphanRegion on completion.
    // (3) Every OTHER held region may STILL be RDMA-written by the peer: forget() is
    //     receiver-initiated (invalidateRemoteAgent), so — unlike an explicit cancel — no
    //     sender-side drain guarantees those writes ended, and a one-sided write cannot be aborted.
    //     quarantineFor > 0 keeps them out of the arena until checkTimeouts()'s reapQuarantine.
    std::vector<std::uint64_t> deferred;
    mCtx.sendGrants(mCtx.scheduler.reclaimByPrefix(
        peer + kSep, scatteringRegions(), deferred, std::chrono::milliseconds(std::max(0, mCtx.cfg.quarantineMs))));
    for (auto off : deferred)
    {
        mScattering[off] = true;
    }
}

void BounceReceiver::checkTimeouts()
{
    auto const now = std::chrono::steady_clock::now();
    if (now < mNextSweep)
    {
        return; // throttled: no need to scan on every ~1ms tick
    }
    // Sweep granularity follows the smallest ENABLED timeout (a tenth of it, clamped to
    // [50ms, 1s]) instead of a fixed constant: the 60s/30s defaults yield a 1s sweep, while a
    // config that shrinks the timeouts to sub-second (tests, debugging) automatically gets a
    // proportionally finer sweep — no hidden floor under the configured values.
    int smallestMs = mCtx.cfg.receiverFlowTimeoutMs > 0 ? mCtx.cfg.receiverFlowTimeoutMs : 0;
    if (mCtx.cfg.quarantineMs > 0)
    {
        smallestMs = smallestMs > 0 ? std::min(smallestMs, mCtx.cfg.quarantineMs) : mCtx.cfg.quarantineMs;
    }
    if (smallestMs <= 0)
    {
        // Lease disabled AND quarantine disabled: nothing is ever quarantined (reclaims free
        // immediately) and no flow can go stale — nothing to sweep for.
        mNextSweep = now + std::chrono::seconds(1);
        return;
    }
    mNextSweep = now
        + std::clamp(
            std::chrono::milliseconds(smallestMs / 10), std::chrono::milliseconds(50), std::chrono::milliseconds(1000));
    // Quarantine over for any expired region: no write posted before its flow's lease expired can
    // plausibly still be in flight, so it may re-enter circulation (and may grant a waiter).
    mCtx.sendGrants(mCtx.scheduler.reapQuarantine());
    if (mCtx.cfg.receiverFlowTimeoutMs <= 0)
    {
        return; // lease disabled (mirror of requestTimeoutMs <= 0 on the sender)
    }
    for (auto const& flow : mCtx.scheduler.staleFlows(std::chrono::milliseconds(mCtx.cfg.receiverFlowTimeoutMs)))
    {
        // The flow's sender went silent after taking grants: it is dead or unreachable (a LIVE
        // sender either makes progress or abandons via requestTimeoutMs + cancel well before this
        // lease expires). Reclaim the whole flow — regions a worker still reads defer as orphans,
        // all others quarantine (the silent peer's NIC may still be writing them).
        auto const [peer, rid] = splitKey(flow);
        TLLM_LOG_WARNING(
            "BounceTransport(%s): flow lease expired (no progress within %d ms) peer=%s rid=%llu -> reclaiming "
            "(regions quarantined for %d ms before reuse)",
            mCtx.selfName.c_str(), mCtx.cfg.receiverFlowTimeoutMs, peer.c_str(), static_cast<unsigned long long>(rid),
            mCtx.cfg.quarantineMs);
        std::vector<std::uint64_t> deferred;
        mCtx.sendGrants(mCtx.scheduler.reclaimFlow(
            flow, scatteringRegions(), deferred, std::chrono::milliseconds(std::max(0, mCtx.cfg.quarantineMs))));
        for (auto off : deferred)
        {
            mScattering[off] = true;
        }
    }
}

void BounceReceiver::scatterWorkerLoop()
{
    bounceNameThread("bounceScatter");
    // Pin this worker to our device. Can't throw out of a thread fn -> warn-only (the loop's CUDA ops
    // would then target the wrong device, so this is a real fault, just non-recoverable here).
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
    while (true)
    {
        ScatterJob job;
        {
            std::unique_lock<std::mutex> lk(mJobMu);
            mJobCv.wait(lk, [this] { return !mJobs.empty() || mCtx.stop.load(std::memory_order_acquire); });
            if (mJobs.empty())
            {
                if (mCtx.stop.load(std::memory_order_acquire))
                {
                    break;
                }
                continue;
            }
            job = std::move(mJobs.front());
            mJobs.pop_front();
        }
        bounceRangeEnd(job.nvtxQueue); // dequeued: the queue-wait leg ends here
        // Covers exec-context acquire + scatter launch + stream sync (the scatter's real GPU wait).
        BounceNvtxScope scatterScope(kNvtxScatter, "scatter rid=%llu chunk=%u n=%zu",
            static_cast<unsigned long long>(job.rid), job.chunkIdx, job.entries.size());
        // Borrow an exec context (stream/scratch) for this scatter. The arena region (job.offset) is
        // held by the scheduler until ACK; the exec context is needed only while the kernel runs, so
        // it comes from the small shared pool. If all are busy, briefly retry (backpressure, never
        // deadlock — senders release contexts independently of local scatter progress).
        ExecCtx* ctx = nullptr;
        while ((ctx = mCtx.exec->tryAcquire()) == nullptr)
        {
            if (mCtx.stop.load(std::memory_order_acquire))
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (ctx == nullptr)
        {
            break; // shutting down
        }
        auto const n = static_cast<std::uint32_t>(job.entries.size());
        // Validate every scatter SOURCE stays inside THIS flow's granted region before launching.
        // bounceOffset/size come from the peer's DATA message; a buggy/hostile peer (or a reordered
        // GRANT) could point them past the region and, if we only bounded against the whole arena,
        // read from an ADJACENT flow's region and copy its bytes into our KV — silent cross-flow
        // corruption. The region is one buddy block [regionBase, regionBase+regionBytes) owned solely
        // by this flow, so bounding to it prevents any cross-flow read. (dstAddr is the caller's own KV
        // target by design, so it isn't bounded here.) Any bad entry -> skip launch, no ACK (sender
        // times out). regionBytes==0 means the region wasn't allocated (stale) -> reject the whole job.
        std::uint64_t const arenaLo = mCtx.arena->baseAddr();
        auto const regionBase = arenaLo + job.offset;
        bool srcInBounds = (job.regionBytes > 0 && regionBase + job.regionBytes <= arenaLo + mCtx.arena->bytes());
        // Scatter into the final dst, then wait so the data is at dst before we ACK. A bad source, a
        // failed launch, OR a stream error must NOT produce an ACK — an ACK tells the sender its KV
        // data landed, so a false ACK here is silent corruption. On error the worker sends no ACK,
        // but the done queue still releases the region; the sender then times out -> FAILURE.
        cudaError_t launchErr = cudaErrorInvalidValue;
        {
            // Prep leg: bounds-check + plan-array build + kernel launch (host-side cost).
            BounceNvtxScope prepScope(kNvtxScatterPrep, "scatterPrep rid=%llu chunk=%u n=%u",
                static_cast<unsigned long long>(job.rid), job.chunkIdx, n);
            // Entries arrive as COALESCED runs (contiguous or strided; see BounceScatterRun); expand
            // them back to <= kCopySplitBytes pieces so the copy kernel keeps its grid-level
            // parallelism. Exact-count pass first (it also validates every run stays inside THIS
            // flow's granted region), then fill the pinned plan buffers DIRECTLY (no intermediate
            // vectors). Piece counts come from the peer's DATA message — a peer built with a larger
            // maxChunkSizeBytes can carry more pieces than our pinned/scratch hold; reject rather than
            // overflow (no launch -> no ACK -> the sender times out, never a false ACK).
            std::size_t const maxEntries = maxPlanEntries(ctx);
            std::uint64_t rawPieces = 0;
            for (std::uint32_t i = 0; i < n; ++i)
            {
                auto const& e = job.entries[i];
                // Run-level source bounds: every piece p reads region[bounceOffset + p*bounceStride
                // .. +pieceSize). count-1 and bounceStride are both u32 so the span product cannot
                // overflow u64. A count of 0 is malformed (a run always carries >= 1 piece).
                std::uint64_t const span = static_cast<std::uint64_t>(e.count - 1) * e.bounceStride + e.pieceSize;
                srcInBounds = srcInBounds && e.count >= 1 && e.bounceOffset <= job.regionBytes
                    && span <= job.regionBytes - e.bounceOffset;
                rawPieces += std::max<std::uint32_t>(e.count, 1);
            }
            // Reject an oversized run list BEFORE the exact-count pass below: that pass iterates
            // once per PIECE, so a hostile/corrupt DATA (per-run count near 2^32 passes the span
            // check when bounceStride is 0) could otherwise pin this worker — and its region —
            // for days. piecesFor() emits at least one entry per piece, so rawPieces > maxEntries
            // implies the nTotal <= maxEntries check would reject the job anyway.
            if (srcInBounds && rawPieces > maxEntries)
            {
                TLLM_LOG_WARNING("BounceTransport(%s): rejected scatter with %llu pieces (max %zu) rid=%llu chunk=%u",
                    mCtx.selfName.c_str(), static_cast<unsigned long long>(rawPieces), maxEntries,
                    static_cast<unsigned long long>(job.rid), job.chunkIdx);
                srcInBounds = false;
            }
            std::size_t nTotal = 0;
            std::uint64_t seen = 0;
            for (std::uint32_t i = 0; i < n && srcInBounds; ++i)
            {
                auto const& e = job.entries[i];
                for (std::uint32_t p = 0; p < e.count; ++p)
                {
                    ++seen;
                    nTotal += piecesFor(
                        e.pieceSize, splitBudget(nTotal, static_cast<std::size_t>(rawPieces - seen), maxEntries));
                }
            }
            if (srcInBounds && nTotal > 0 && nTotal <= maxEntries)
            {
                auto const bufs = planBufs(ctx, nTotal);
                std::size_t idx = 0;
                seen = 0;
                for (std::uint32_t i = 0; i < n; ++i)
                {
                    auto const& e = job.entries[i];
                    for (std::uint32_t p = 0; p < e.count; ++p)
                    {
                        ++seen;
                        appendSplitInto(bufs, idx,
                            regionBase + e.bounceOffset + static_cast<std::uint64_t>(p) * e.bounceStride,
                            e.dstAddr + static_cast<std::uint64_t>(p) * e.dstStride, e.pieceSize,
                            splitBudget(idx, static_cast<std::size_t>(rawPieces - seen), maxEntries));
                    }
                }
                launchErr = launchPrepared(ctx, nTotal, mCtx.cfg.useZeroCopyArguments);
            }
            else if (srcInBounds && nTotal == 0)
            {
                launchErr = cudaSuccess; // empty plan: nothing to scatter (0 runs) -> vacuous success
            }
        }
        cudaError_t syncErr = cudaSuccess;
        if (launchErr == cudaSuccess)
        {
            // Sync leg: the actual GPU wait (kernel queueing + run time on the exec stream).
            BounceNvtxScope syncScope(kNvtxScatterSync, "scatterSync rid=%llu chunk=%u",
                static_cast<unsigned long long>(job.rid), job.chunkIdx);
            syncErr = cudaStreamSynchronize(ctx->stream);
        }
        bool const ok = srcInBounds && launchErr == cudaSuccess && syncErr == cudaSuccess;
        if (!ok)
        {
            (void) cudaGetLastError(); // clear sticky error so the reused context isn't poisoned
            TLLM_LOG_WARNING(
                "BounceTransport(%s): scatter failed (srcInBounds=%d launch=%d sync=%d) rid=%llu chunk=%u -> no ACK",
                mCtx.selfName.c_str(), static_cast<int>(srcInBounds), static_cast<int>(launchErr),
                static_cast<int>(syncErr), static_cast<unsigned long long>(job.rid), job.chunkIdx);
        }
        mCtx.exec->release(ctx); // kernel done (or failed) -> return the context
        // ACK straight from the worker (ControlChannel::sendTo is thread-safe): the data IS at its
        // final dst here, and skipping the done-queue -> IO-thread hop shaves its drain latency off
        // the sender's ackWait critical path. Region bookkeeping (scheduler free / re-grant) still
        // goes through drainScatterDone on the IO thread. A failed scatter sends NO ACK — a false
        // ACK would tell the sender corrupt/absent data landed; it must time out instead.
        if (ok)
        {
            BounceNvtxScope ackScope(
                kNvtxAckSend, "ackSend rid=%llu chunk=%u", static_cast<unsigned long long>(job.rid), job.chunkIdx);
            mCtx.channel->sendTo(job.peer, encodeAck(job.rid, job.chunkIdx, job.offset));
        }
        {
            std::lock_guard<std::mutex> lk(mDoneMu);
            mDone.push_back(ScatterDone{job.key, job.offset});
        }
    }
}

// ============================================================================
// BounceSender — [S] role
// ============================================================================

BounceSender::BounceSender(BounceContext& ctx)
    : mCtx(ctx)
{
}

std::shared_future<BounceResult> BounceSender::submit(
    TransferDescs const& srcDescs, TransferDescs const& dstDescs, std::string const& peer)
{
    auto promise = std::make_shared<std::promise<BounceResult>>();
    auto fut = promise->get_future().share();
    BounceTransferPlan plan;
    try
    {
        BounceNvtxScope planScope(kNvtxBuildPlan, "buildPlan nDesc=%zu", srcDescs.getDescs().size());
        plan = BounceTransferPlan::build(srcDescs, dstDescs, mCtx.cfg.maxChunkSizeBytes,
            std::max<std::size_t>(1024ULL, mCtx.cfg.maxChunkSizeBytes / 256ULL));
    }
    catch (std::exception const& e)
    {
        // The eligibility gate (shouldUseBounce) screens the plan's preconditions, so this is a
        // should-not-happen defense: resolve the future to FAILURE instead of letting the exception
        // unwind out of submit(). Bounce admission is final for a request — a failure here (or later
        // in the protocol) fails the transfer; there is deliberately no automatic re-submit on the
        // standard NIXL path, so a bounce-layer fault surfaces to the caller instead of being
        // silently absorbed as a slow success. Whether to retry is the caller's decision.
        TLLM_LOG_WARNING(
            "BounceTransport(%s): plan build for peer %s rejected: %s", mCtx.selfName.c_str(), peer.c_str(), e.what());
        promise->set_value({TransferState::kFAILURE, BounceFailReason::kPlanRejected});
        return fut;
    }
    auto const numChunks = static_cast<std::uint32_t>(plan.numChunks());
    if (numChunks == 0)
    {
        promise->set_value({TransferState::kSUCCESS, BounceFailReason::kNone});
        return fut;
    }

    // Per-chunk packed byte sizes: the receiver allocates a region of each size as it grants, so the
    // WANT both announces how many chunks we have and how big each one's bounce region must be.
    std::vector<std::uint32_t> chunkBytes(numChunks);
    for (std::uint32_t i = 0; i < numChunks; ++i)
    {
        chunkBytes[i] = static_cast<std::uint32_t>(plan.chunks()[i].packedBytes);
    }

    std::uint64_t const rid = mNextRid.fetch_add(1, std::memory_order_relaxed);
    std::uint64_t const planBytes = plan.totalBytes();
    {
        std::lock_guard<std::mutex> lk(mReqMu);
        Request req;
        req.peer = peer;
        req.numChunks = numChunks;
        req.plan = std::move(plan);
        req.promise = promise;
        req.lastProgress = std::chrono::steady_clock::now();
        req.nvtxReq = bounceRangeStart(kNvtxRequest, "req rid=%llu chunks=%u bytes=%llu",
            static_cast<unsigned long long>(rid), numChunks, static_cast<unsigned long long>(planBytes));
        // Ends at the FIRST GRANT (onGrant) — the credit-wait leg of the request.
        req.nvtxGrantWait
            = bounceRangeStart(kNvtxGrantWait, "grantWait rid=%llu", static_cast<unsigned long long>(rid));
        mRequests.emplace(rid, std::move(req));
    }
    // Ask the receiver to grant a region for each chunk of this request flow. The WANT carries our
    // own control endpoint so the receiver can addPeer us and send GRANT/ACK back (self-bootstrap).
    mCtx.channel->sendTo(peer, encodeWant(rid, chunkBytes, mCtx.channel->localEndpoint()));
    if (mCtx.cfg.enableEagerGather)
    {
        // Overlap the WANT->GRANT control round-trip with the gather: launch this request's first
        // chunks NOW instead of waiting for the GRANT (they were measured back-to-back at roughly
        // the same duration, so eager gather hides one of the two). Running the launch on the
        // caller's thread also keeps its prep cost off the IO thread. The exec streams live on our
        // device but the caller thread's current device is not guaranteed — pin it first (warn-only:
        // on failure the pump's CUDA calls fail and the request degrades to the classic GRANT path
        // or a deterministic failure, never a hang).
        TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
        std::lock_guard<std::mutex> lk(mReqMu);
        auto it = mRequests.find(rid);
        if (it != mRequests.end()) // a racing GRANT may have already pumped (or failed) the request
        {
            pumpRequest(rid, it->second);
        }
    }
    return fut;
}

void BounceSender::onGrant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    std::vector<BounceCreditEntry> credits;
    if (!decodeCredits(blob, h, credits))
    {
        return;
    }
    std::lock_guard<std::mutex> lk(mReqMu);
    auto it = mRequests.find(h.requestId);
    if (it == mRequests.end())
    {
        return; // late grant for a finished/cancelled request
    }
    Request& req = it->second;
    // A credit contains a peer-owned address. Never let an unrelated peer redirect this request's
    // RDMA write, even if it guesses the request id.
    if (peer != req.peer)
    {
        TLLM_LOG_WARNING("BounceTransport(%s): dropping wrong-peer GRANT peer=%s expected=%s rid=%llu",
            mCtx.selfName.c_str(), peer.c_str(), req.peer.c_str(), static_cast<unsigned long long>(h.requestId));
        return;
    }
    bounceRangeEnd(req.nvtxGrantWait);     // first GRANT ends the credit-wait span (no-op on later GRANTs)
    bounceRangeEnd(req.nvtxCreditStarved); // a GRANT ends the current starvation period (pump may reopen one)
    for (auto const& credit : credits)
    {
        req.pendingCredits.push_back(credit);
    }
    pumpRequest(h.requestId, req);
}

void BounceSender::attachCredits(std::uint64_t rid, Request& req)
{
    // Credits pair with chunks strictly in order (the receiver serves the WANT size list FIFO), so
    // pendingCredits.front() is always chunk `nextCredit`'s credit. Attach parked credits to chunks
    // that eager gather already posted credit-less; chunks not yet posted keep their credit parked
    // for pumpRequest to consume at gather-launch time.
    while (!req.pendingCredits.empty() && req.nextCredit < req.nextPost)
    {
        BounceCreditEntry const& credit = req.pendingCredits.front();
        Posted* target = nullptr;
        for (auto& p : req.posted)
        {
            if (p.chunkIdx == req.nextCredit)
            {
                target = &p;
                break;
            }
        }
        if (target == nullptr || target->hasCredit)
        {
            // A posted chunk is only erased on ACK, and it can only be ACKed after consuming its
            // credit (nextCredit already passed it) — so a missing or already-credited target is a
            // protocol anomaly (dup GRANT after reconnect). Drop the credit, never mispair it.
            TLLM_LOG_WARNING("BounceTransport(%s): rid=%llu chunk=%u unexpected credit (dup GRANT?); dropping",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), req.nextCredit);
            req.pendingCredits.pop_front();
            req.nextCredit += 1;
            continue;
        }
        auto const& chunk = req.plan.chunks()[target->chunkIdx];
        // Same mispair guard as pumpRequest: a credit smaller than the chunk would make the RDMA
        // write overflow the granted region into an adjacent flow's region on the peer. Abandon the
        // flow (fails via checkTimeouts) rather than corrupt.
        if (chunk.packedBytes > credit.len)
        {
            TLLM_LOG_WARNING(
                "BounceTransport(%s): rid=%llu chunk=%u packedBytes=%zu > granted region len=%u (GRANT "
                "mispair/reorder); abandoning flow",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), target->chunkIdx,
                static_cast<std::size_t>(chunk.packedBytes), static_cast<unsigned int>(credit.len));
            req.abandonReason = BounceFailReason::kProtocolError;
            req.pendingCredits.clear();
            return;
        }
        target->remoteHandle = credit.regionHandle;
        target->remoteAddr = credit.addr;
        target->remoteDevId = credit.devId;
        target->hasCredit = true;
        // Now credit-backed: its staging region stops counting against the eager budget.
        mCtx.scheduler.promoteLocal(target->localOffset);
        req.pendingCredits.pop_front();
        req.nextCredit += 1;
        req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk got its credit
    }
    if (req.nextCredit >= req.numChunks && !req.pendingCredits.empty())
    {
        // Over-grant: receiver handed more credits than we have chunks. Shouldn't happen under
        // the protocol (receiver grants at most numChunks). Log it — silently dropping would
        // mask an upstream bug and leak the receiver-side regions backing these credits.
        TLLM_LOG_WARNING("BounceTransport(%s): rid=%llu over-grant, dropping %zu extra credit(s)",
            mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), req.pendingCredits.size());
        req.pendingCredits.clear();
    }
}

void BounceSender::pumpRequest(std::uint64_t rid, Request& req)
{
    attachCredits(rid, req);
    while (req.nextPost < req.numChunks)
    {
        std::uint32_t const chunkIdx = req.nextPost;
        auto const& chunk = req.plan.chunks()[chunkIdx];
        // After attachCredits, a non-empty pendingCredits implies nextCredit == nextPost, i.e. the
        // front credit is exactly THIS chunk's credit.
        bool const haveCredit = !req.pendingCredits.empty();
        if (!haveCredit)
        {
            if (!mCtx.cfg.enableEagerGather)
            {
                break; // classic path: a chunk's gather starts only once its GRANT arrived
            }
            if (req.posted.size() >= mCtx.cfg.maxInflightChunksPerRequest)
            {
                break; // cap eager gathers by this request's configured in-flight chunk limit
            }
        }
        // Defensive pairing check BEFORE committing resources. Credits pair with chunks by FIFO
        // order, and the receiver sizes each granted region to chunkBytes[chunkIdx] in WANT, so
        // packedBytes always fits when the channel honors its FIFO contract. A malformed or
        // misordered GRANT would make us RDMA-write packedBytes into a smaller region, overflowing
        // into an adjacent flow's region on the peer. Detect that protocol violation and abandon the
        // flow (it then fails via checkTimeouts) rather than corrupt the peer.
        if (haveCredit && chunk.packedBytes > req.pendingCredits.front().len)
        {
            TLLM_LOG_WARNING(
                "BounceTransport(%s): rid=%llu chunk=%u packedBytes=%zu > granted region len=%u (GRANT "
                "mispair/reorder); abandoning flow",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), chunkIdx,
                static_cast<std::size_t>(chunk.packedBytes), static_cast<unsigned int>(req.pendingCredits.front().len));
            req.abandonReason = BounceFailReason::kProtocolError;
            req.pendingCredits.clear();
            break;
        }
        // Non-blocking: borrow an exec context (cheap to return), then a gather-staging region sized
        // to this chunk from the SHARED arena. If either is unavailable, leave the credit parked and
        // bail; the IO loop retries via drainPendingPosts() once an ACK frees a region / context —
        // never blocks here, so an oversubscribed arena (many peers, or both roles) degrades to
        // backpressure, not deadlock. Credit-less (eager) staging is additionally capped by the
        // scheduler's eager budget (half the arena) so it can never starve incoming grants.
        ExecCtx* ctx = mCtx.exec->tryAcquire();
        if (ctx == nullptr)
        {
            break;
        }
        auto localOff = mCtx.scheduler.acquireLocal(chunk.packedBytes, /*eager=*/!haveCredit);
        if (!localOff)
        {
            mCtx.exec->release(ctx);
            break;
        }
        BounceCreditEntry credit{};
        if (haveCredit)
        {
            credit = req.pendingCredits.front();
            req.pendingCredits.pop_front();
            req.nextCredit += 1;
        }
        auto const nDesc = static_cast<std::uint32_t>(chunk.srcPtrs.size());
        // Covers plan-array prep + gather launch + event record (the synchronous launch cost;
        // the gather's GPU time is the async `gather` span ended in drainGatherReady).
        BounceNvtxScope gatherLaunchScope(kNvtxGatherLaunch, "gatherLaunch rid=%llu chunk=%u n=%u bytes=%llu",
            static_cast<unsigned long long>(rid), chunkIdx, nDesc, static_cast<unsigned long long>(chunk.packedBytes));
        auto const regionBase = mCtx.arena->baseAddr() + *localOff;
        // Coalescing in the plan can leave very large per-desc runs; split them so the copy kernel
        // keeps its one-thread-block-per-entry parallelism (bounded by the scratch capacity).
        // Two passes: an exact piece count first (the packed [srcs|dsts|sizes] pinned layout needs
        // the total before the first write), then fill the pinned buffer DIRECTLY — one write pass
        // instead of building std::vectors and memcpy'ing them in.
        std::size_t const maxEntries = maxPlanEntries(ctx);
        std::size_t nTotal = 0;
        for (std::uint32_t i = 0; i < nDesc; ++i)
        {
            nTotal += piecesFor(chunk.sizes[i], splitBudget(nTotal, nDesc - 1 - i, maxEntries));
        }
        // Same capacity guard as the scatter path: planBufs is capacity-unchecked, so writing more
        // than maxEntries would overflow the pinned/scratch plan buffers. The bound holds by
        // construction today (submit's maxDescsPerChunk and the ExecPool's maxDescs derive from the
        // same expression), but a divergence must fail the flow, not corrupt the heap.
        if (nTotal > maxEntries)
        {
            TLLM_LOG_WARNING(
                "BounceTransport(%s): rid=%llu chunk=%u plan entries %zu > exec capacity %zu; abandoning flow",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), chunkIdx, nTotal, maxEntries);
            mCtx.exec->release(ctx);
            mCtx.sendGrants(mCtx.scheduler.releaseLocal(*localOff));
            req.abandonReason = BounceFailReason::kProtocolError;
            req.pendingCredits.clear(); // abandon: the request then fails via checkTimeouts
            break;
        }
        auto const bufs = planBufs(ctx, nTotal);
        std::size_t idx = 0;
        for (std::uint32_t i = 0; i < nDesc; ++i)
        {
            appendSplitInto(bufs, idx, chunk.srcPtrs[i], regionBase + chunk.bounceOffsets[i], chunk.sizes[i],
                splitBudget(idx, nDesc - 1 - i, maxEntries));
        }
        // gather into the region (cfg knobs select the H2D-vs-zero-copy arg path + custom-vs-cub copy)
        cudaError_t const gatherErr = launchPrepared(ctx, nTotal, mCtx.cfg.useZeroCopyArguments);
        // Record an event for gather completion and DEFER the write. The gather must finish before
        // NIXL reads the region, but on a shared GPU the gather can be delayed behind model kernels
        // — blocking the IO thread on cudaStreamSynchronize here would stall the whole reactor
        // (no recv/poll/ACK) for that delay. Instead drainGatherReady() polls this event and posts
        // the write only once the gather is done (NIXL's postXferReq is not stream-ordered anyway).
        cudaError_t const recordErr = cudaEventRecord(ctx->event, ctx->stream);
        bool const gatherFailed = (gatherErr != cudaSuccess || recordErr != cudaSuccess);
        if (gatherFailed)
        {
            // If the launch or the event-record failed we must NOT trust the event: an event never
            // successfully recorded queries as "complete", which would make drainGatherReady post a
            // write of an UN-gathered region (garbage). Clear the sticky error and flag the Posted so
            // drainGatherReady fails the request deterministically (region/ctx released there).
            (void) cudaGetLastError();
            TLLM_LOG_WARNING("BounceTransport(%s): gather launch/record failed (launch=%d record=%d) rid=%llu chunk=%u",
                mCtx.selfName.c_str(), static_cast<int>(gatherErr), static_cast<int>(recordErr),
                static_cast<unsigned long long>(rid), chunkIdx);
        }
        Posted p;
        p.chunkIdx = chunkIdx;
        p.localOffset = *localOff;
        p.ctx = ctx;
        p.hasCredit = haveCredit;
        if (haveCredit)
        {
            p.remoteHandle = credit.regionHandle;
            p.remoteAddr = credit.addr;
            p.remoteDevId = credit.devId;
        }
        p.writeBytes = static_cast<std::uint32_t>(chunk.packedBytes);
        // Gather in flight; the write is issued later by drainGatherReady once the event signals. If
        // the gather launch/record failed, go straight to GatherFailed so drainGatherReady fails the
        // request without ever trusting the (un)recorded event.
        p.state = gatherFailed ? PostState::GatherFailed : PostState::Gathering;
        if (!gatherFailed)
        {
            p.nvtxGather = bounceRangeStart(kNvtxGather, "gather rid=%llu chunk=%u bytes=%llu",
                static_cast<unsigned long long>(rid), chunkIdx, static_cast<unsigned long long>(chunk.packedBytes));
        }
        req.posted.push_back(std::move(p));
        req.nextPost += 1;
        req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk's gather launched
    }
    // Reconcile the pipeline-starvation NVTX spans after every pump pass (perf visibility only):
    // - creditStarved: every granted credit is consumed but chunks still lack one -> the flow is
    //   waiting on the receiver's next re-GRANT (with eager gather a chunk may be posted yet still
    //   credit-less). Ended in onGrant, so each wait period is its own range.
    // - arenaStarved: exited the loop with credits still parked -> blocked on LOCAL resources (gather
    //   region / exec ctx). One continuous range per park period, ended here once the park drains.
    // Both are idempotent across repeated pump attempts (drainPendingPosts retries every IO pass).
    if (req.pendingCredits.empty())
    {
        bounceRangeEnd(req.nvtxArenaStarved);
        if (req.nextCredit < req.numChunks && req.nvtxCreditStarved == 0)
        {
            req.nvtxCreditStarved = bounceRangeStart(kNvtxCreditStarved, "creditStarved rid=%llu posted=%u/%u",
                static_cast<unsigned long long>(rid), req.nextPost, req.numChunks);
        }
    }
    else if (req.nvtxArenaStarved == 0)
    {
        req.nvtxArenaStarved = bounceRangeStart(kNvtxArenaStarved, "arenaStarved rid=%llu parked=%zu",
            static_cast<unsigned long long>(rid), req.pendingCredits.size());
    }
}

void BounceSender::drainPendingPosts()
{
    std::lock_guard<std::mutex> lk(mReqMu);
    for (auto& [rid, req] : mRequests)
    {
        // Retry parked credits, and (with eager gather) chunks that couldn't start earlier because
        // the arena/ExecPool/eager budget was exhausted — ACKs may have freed resources since.
        if (!req.pendingCredits.empty() || (mCtx.cfg.enableEagerGather && req.nextPost < req.numChunks))
        {
            pumpRequest(rid, req);
        }
    }
}

bool BounceSender::drainGatherReady()
{
    std::lock_guard<std::mutex> lk(mReqMu);
    bool didWork = false;
    std::vector<std::uint64_t> toFail;
    for (auto& [rid, req] : mRequests)
    {
        for (auto& p : req.posted)
        {
            if (p.state == PostState::Writing || p.state == PostState::Sent)
            {
                continue; // gather already done + write issued
            }
            if (p.state == PostState::GatherFailed)
            {
                // Gather launch / event-record failed in pumpRequest -> never trust the event; fail.
                toFail.push_back(rid);
                break;
            }
            if (p.state == PostState::Gathering)
            {
                // Poll the gather event (non-blocking).
                cudaError_t const ev = cudaEventQuery(p.ctx->event);
                if (ev == cudaErrorNotReady)
                {
                    continue; // gather still running (possibly delayed behind other GPU work) — no block
                }
                if (ev != cudaSuccess)
                {
                    // Gather kernel / stream error -> fail the request deterministically (never hang).
                    (void) cudaGetLastError();
                    toFail.push_back(rid);
                    break;
                }
                // Gather done — return the exec context immediately for another chunk to reuse (the
                // write path never needs it: postXferReq is not stream-ordered). The region stays
                // held until ACK.
                bounceRangeEnd(p.nvtxGather);
                mCtx.exec->release(p.ctx);
                p.ctx = nullptr;
                p.state = PostState::Gathered;
                didWork = true;
            }
            // state == Gathered: issue the RDMA write once the credit is here (an eagerly-gathered
            // chunk may finish before its GRANT arrives; it then waits in place until attachCredits
            // fills in the remote target).
            if (!p.hasCredit)
            {
                continue;
            }
            p.nvtxWrite = bounceRangeStart(kNvtxNixlWrite, "nixlWrite rid=%llu chunk=%u bytes=%u",
                static_cast<unsigned long long>(rid), p.chunkIdx, p.writeBytes);
            {
                // One already-final (src, dst) pair — the remote address came from the credit, so
                // this goes through the agent's below-the-splitter primitive, not the public path.
                // A nullptr result (submission failure, logged by the agent) polls as kFAILURE.
                TransferDescs const src{MemoryType::kVRAM,
                    {MemoryDesc{reinterpret_cast<std::uintptr_t>(mCtx.arena->at(p.localOffset)), p.writeBytes,
                        static_cast<std::uint32_t>(mCtx.deviceId)}}};
                TransferDescs const dst{MemoryType::kVRAM, {MemoryDesc{p.remoteAddr, p.writeBytes, p.remoteDevId}}};
                p.xfer = mCtx.agent.postXferRequest(TransferOp::kWRITE, src, dst, req.peer, std::nullopt);
            }
            p.state = PostState::Writing;
            didWork = true;
            req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk was posted
        }
    }
    for (auto rid : toFail)
    {
        auto it = mRequests.find(rid);
        if (it != mRequests.end())
        {
            failRequest(rid, it->second, BounceFailReason::kGatherFailed);
            didWork = true;
        }
    }
    return didWork;
}

bool BounceSender::pollSenderHandles()
{
    std::lock_guard<std::mutex> lk(mReqMu);
    bool didWork = false;
    std::vector<std::uint64_t> toFail;
    for (auto& [rid, req] : mRequests)
    {
        for (auto& p : req.posted)
        {
            if (p.state != PostState::Writing)
            {
                continue; // still gathering (drainGatherReady handles it) or DATA already sent
            }
            // wait(0) is one non-blocking status query: IN_PROGRESS / SUCCESS / FAILURE.
            // A failed post left p.xfer null -> report FAILURE.
            TransferState const st = p.xfer != nullptr ? p.xfer->wait(0) : TransferState::kFAILURE;
            if (st == TransferState::kSUCCESS)
            {
                // End the write span BEFORE building the DATA message so nixlWrite measures only the
                // RDMA in-flight time; the DATA build/encode/enqueue cost gets its own span (it used
                // to hide inside nixlWrite).
                bounceRangeEnd(p.nvtxWrite);
                auto const& chunk = req.plan.chunks()[p.chunkIdx];
                // The DATA scatter plan is the chunk's COALESCED run list (built once at plan time):
                // same bytes, but typically orders of magnitude fewer entries than the per-desc view
                // — this message sits on the ACK critical path. Sent as-is, no per-send rebuild.
                auto const nRuns = static_cast<std::uint32_t>(chunk.scatterRuns.size());
                {
                    BounceNvtxScope dataScope(kNvtxDataSend, "dataSend rid=%llu chunk=%u n=%u bytes=%zu",
                        static_cast<unsigned long long>(rid), p.chunkIdx, nRuns,
                        static_cast<std::size_t>(nRuns) * sizeof(BounceScatterRun));
                    mCtx.channel->sendTo(
                        req.peer, encodeData(rid, p.chunkIdx, req.numChunks, p.remoteHandle, chunk.scatterRuns));
                }
                if (!p.xfer->release())
                {
                    // The write is terminal, so progress is safe; the status object keeps the
                    // handle and its destructor retries the backend release.
                    TLLM_LOG_WARNING("BounceTransport(%s): terminal write handle release deferred rid=%llu chunk=%u",
                        mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), p.chunkIdx);
                }
                p.xfer.reset();
                p.state = PostState::Sent;
                p.nvtxAckWait = bounceRangeStart(
                    kNvtxAckWait, "ackWait rid=%llu chunk=%u", static_cast<unsigned long long>(rid), p.chunkIdx);
                didWork = true;
            }
            else if (st == TransferState::kFAILURE)
            {
                toFail.push_back(rid);
                break;
            }
        }
    }
    for (auto rid : toFail)
    {
        auto it = mRequests.find(rid);
        if (it != mRequests.end())
        {
            failRequest(rid, it->second, BounceFailReason::kWriteFailed);
            didWork = true;
        }
    }
    return didWork;
}

void BounceSender::onAck(std::string const& peer, BounceMsgHeader const& h)
{
    // Starts BEFORE taking mReqMu: the span exposes ACK-processing latency INCLUDING lock wait —
    // pumpRequest holds mReqMu during gather launches, so a long onAck here means the ACK stalled
    // behind another flow's launch prep.
    BounceNvtxScope ackScope(
        kNvtxOnAck, "onAck rid=%llu chunk=%u", static_cast<unsigned long long>(h.requestId), h.chunkIdx);
    std::lock_guard<std::mutex> lk(mReqMu);
    auto it = mRequests.find(h.requestId);
    if (it == mRequests.end())
    {
        return;
    }
    Request& req = it->second;
    if (peer != req.peer)
    {
        TLLM_LOG_WARNING("BounceTransport(%s): dropping wrong-peer ACK peer=%s expected=%s rid=%llu chunk=%u",
            mCtx.selfName.c_str(), peer.c_str(), req.peer.c_str(), static_cast<unsigned long long>(h.requestId),
            h.chunkIdx);
        return;
    }
    bool found = false;
    for (auto pit = req.posted.begin(); pit != req.posted.end(); ++pit)
    {
        // Only a Sent chunk has finished reading its local staging region and may recycle it on ACK.
        if (pit->chunkIdx == h.chunkIdx && pit->remoteHandle == h.regionHandle && pit->state == PostState::Sent)
        {
            bounceRangeEnd(pit->nvtxAckWait);
            // Return the gather-staging region to the shared arena; re-schedule may hand the freed
            // bytes to a waiting remote flow.
            mCtx.sendGrants(mCtx.scheduler.releaseLocal(pit->localOffset));
            req.posted.erase(pit);
            found = true;
            break;
        }
    }
    if (!found)
    {
        TLLM_LOG_WARNING("BounceTransport(%s): dropping stale/invalid ACK peer=%s rid=%llu chunk=%u region=%llu",
            mCtx.selfName.c_str(), peer.c_str(), static_cast<unsigned long long>(h.requestId), h.chunkIdx,
            static_cast<unsigned long long>(h.regionHandle));
        // Duplicate / unknown ACK (zmq reconnect, retransmit). Do NOT count it — an over-count
        // could push acked past numChunks and resolve SUCCESS before all chunks actually landed.
        return;
    }
    req.acked += 1;
    req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk was ACKed
    if (req.acked >= req.numChunks)
    {
        bounceRangeEnd(req.nvtxGrantWait);
        bounceRangeEnd(req.nvtxCreditStarved);
        bounceRangeEnd(req.nvtxArenaStarved);
        bounceRangeEnd(req.nvtxReq);
        try
        {
            req.promise->set_value({TransferState::kSUCCESS, BounceFailReason::kNone});
        }
        catch (...)
        {
            // set_value throws std::future_error ONLY if the promise is already satisfied — a benign
            // double-resolve (a request resolves exactly once); intentionally ignored, not a failure.
        }
        mRequests.erase(it);
    }
}

void BounceSender::checkTimeouts()
{
    if (mCtx.cfg.requestTimeoutMs <= 0)
    {
        return; // timeout disabled (e.g. tests that intentionally wait forever)
    }
    auto const now = std::chrono::steady_clock::now();
    auto const limit = std::chrono::milliseconds(mCtx.cfg.requestTimeoutMs);
    std::vector<std::uint64_t> stuck;
    {
        std::lock_guard<std::mutex> lk(mReqMu);
        for (auto& [rid, req] : mRequests)
        {
            if (now - req.lastProgress > limit)
            {
                stuck.push_back(rid);
            }
        }
        for (auto rid : stuck)
        {
            auto it = mRequests.find(rid);
            if (it != mRequests.end())
            {
                // Peer never granted or stopped making progress (unreachable / not bounce-ready /
                // congested). Fail the request so wait() returns FAILURE instead of hanging.
                failRequest(rid, it->second, BounceFailReason::kNoProgressTimeout);
            }
        }
    }
}

bool BounceSender::drainOrphanLocal()
{
    if (mOrphanLocal.empty())
    {
        return false;
    }
    bool didWork = false;
    std::vector<OrphanLocal> keep;
    keep.reserve(mOrphanLocal.size());
    for (auto& o : mOrphanLocal)
    {
        if (o.xfer != nullptr && o.xfer->wait(0) == TransferState::kIN_PROGRESS)
        {
            keep.push_back(std::move(o)); // write still in flight -> the NIC may still read the region; wait
            continue;
        }
        // Terminal (Done or Failed): the NIC is finished with the region (source AND the receiver's
        // destination) -> recycle the local source now.
        if (o.xfer != nullptr && !o.xfer->release())
        {
            // The write is terminal, so recycling is safe; only the backend handle remains retained
            // (the status object's destructor retries the release).
            TLLM_LOG_WARNING("BounceTransport(%s): terminal orphan handle release deferred rid=%llu",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(o.rid));
        }
        mCtx.sendGrants(mCtx.scheduler.releaseLocal(o.offset));
        didWork = true;
    }
    mOrphanLocal.swap(keep);
    // Send any deferred cancel whose flow now has NO in-flight write left: the receiver may safely
    // reclaim its regions (the writes have landed/failed, no more DMA targets them).
    for (auto it = mPendingCancel.begin(); it != mPendingCancel.end();)
    {
        std::uint64_t const rid = it->first;
        bool const stillInFlight = std::any_of(
            mOrphanLocal.begin(), mOrphanLocal.end(), [rid](OrphanLocal const& o) { return o.rid == rid; });
        if (stillInFlight)
        {
            ++it;
            continue;
        }
        mCtx.channel->sendTo(it->second, encodeCancel(rid, mCtx.channel->localEndpoint()));
        it = mPendingCancel.erase(it);
        didWork = true;
    }
    return didWork;
}

void BounceSender::failRequest(std::uint64_t rid, Request& req, BounceFailReason reason)
{
    // An abandoned flow (GRANT mispair / plan overflow) reaches here through the timeout path;
    // report the specific abandon cause, not the generic timeout.
    if (req.abandonReason != BounceFailReason::kNone)
    {
        reason = req.abandonReason;
    }
    // The one log line every sender-side failure passes through (timeout, peer drop, write/gather
    // failure) — keep it WARNING and keep the progress context.
    TLLM_LOG_WARNING("BounceTransport(%s): request FAILED rid=%llu peer=%s reason=\"%s\" chunks acked=%u posted=%u/%u",
        mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), req.peer.c_str(), toString(reason), req.acked,
        req.nextPost, req.numChunks);
    // Release in-flight transfer handles and return each gather-staging region to the shared arena —
    // but only once nothing is still touching the region's memory, else recycling races a live DMA:
    //   - Writing: the RDMA write may still be reading the region as its source. Defer to mOrphanLocal;
    //     drainOrphanLocal() releases the xfer + region once poll() is terminal.
    //   - Gathering / GatherFailed: our gather kernel may still be WRITING the region; sync its stream
    //     before recycling (else an abandoned gather scribbles a re-granted region — the write is not
    //     ordered against the new owner). Rare failure path, so a sync here is fine.
    //   - Sent: write landed (poll==kDone), xfer already released in pollSenderHandles, NIC done
    //     reading -> recycle now.
    bool deferredWrite = false;
    for (auto& p : req.posted)
    {
        // Close this chunk's NVTX spans (whichever leg it died in); 0 handles are no-ops.
        bounceRangeEnd(p.nvtxGather);
        bounceRangeEnd(p.nvtxWrite);
        bounceRangeEnd(p.nvtxAckWait);
        if (p.state == PostState::Writing)
        {
            mOrphanLocal.push_back(OrphanLocal{std::move(p.xfer), p.localOffset, req.peer, rid});
            deferredWrite = true;
            continue; // do NOT release xfer or recycle the region yet
        }
        if ((p.state == PostState::Gathering || p.state == PostState::GatherFailed) && p.ctx != nullptr)
        {
            cudaError_t const se = cudaStreamSynchronize(p.ctx->stream);
            if (se != cudaSuccess)
            {
                TLLM_LOG_WARNING("BounceTransport(%s): failRequest stream sync error rid=%llu chunk=%u: %s",
                    mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), p.chunkIdx, cudaGetErrorString(se));
                (void) cudaGetLastError();
            }
            mCtx.exec->release(p.ctx);
            p.ctx = nullptr;
        }
        mCtx.sendGrants(mCtx.scheduler.releaseLocal(p.localOffset));
    }
    // Retract the credit request so the receiver stops holding/granting for it. If any RDMA write is
    // still in flight it is landing on the receiver's region; sending the cancel NOW would let the
    // receiver reclaim+re-grant that region under the write -> corruption. Defer the cancel until the
    // flow's writes drain (drainOrphanLocal sends it). With no in-flight write, send it immediately.
    if (deferredWrite)
    {
        mPendingCancel[rid] = req.peer;
    }
    else
    {
        mCtx.channel->sendTo(req.peer, encodeCancel(rid, mCtx.channel->localEndpoint()));
    }
    bounceRangeEnd(req.nvtxGrantWait);
    bounceRangeEnd(req.nvtxCreditStarved);
    bounceRangeEnd(req.nvtxArenaStarved);
    bounceRangeEnd(req.nvtxReq);
    try
    {
        req.promise->set_value({TransferState::kFAILURE, reason});
    }
    catch (...)
    {
        // set_value throws std::future_error ONLY if the promise is already satisfied — a benign
        // double-resolve (a request resolves exactly once); intentionally ignored, not a failure.
    }
    mRequests.erase(rid);
}

void BounceSender::forget(std::string const& peer)
{
    // Fail any in-flight request targeting the gone peer so its wait() returns.
    std::lock_guard<std::mutex> lk(mReqMu);
    std::vector<std::uint64_t> toFail;
    for (auto const& [rid, req] : mRequests)
    {
        if (req.peer == peer)
        {
            toFail.push_back(rid);
        }
    }
    for (auto rid : toFail)
    {
        auto it = mRequests.find(rid);
        if (it != mRequests.end())
        {
            failRequest(rid, it->second, BounceFailReason::kPeerDropped);
        }
    }
}

void BounceSender::failAll()
{
    // Fail any still-pending requests so no submit() future hangs, releasing their in-flight
    // transfer handles first (same handle-leak fix as failRequest). Called after the device has been
    // synced and the IO thread joined, so no lock contention — but keep mReqMu for consistency.
    // cudaDeviceSynchronize covers only local kernels. NIXL_SUCCESS from releaseXferReq means an
    // active write was canceled/released; retain failures for retry instead of polling forever and
    // letting a failed peer/backend hang teardown.
    std::lock_guard<std::mutex> lk(mReqMu);
    std::vector<std::unique_ptr<TransferStatus>> releaseRetry;
    for (auto& [rid, req] : mRequests)
    {
        for (auto& p : req.posted)
        {
            bounceRangeEnd(p.nvtxGather);
            bounceRangeEnd(p.nvtxWrite);
            bounceRangeEnd(p.nvtxAckWait);
            if (p.state == PostState::Writing && p.xfer != nullptr)
            {
                if (!p.xfer->release())
                {
                    releaseRetry.push_back(std::move(p.xfer));
                }
            }
            if (p.ctx != nullptr)
            {
                mCtx.exec->release(p.ctx); // still-gathering chunk: return its borrowed exec context
                p.ctx = nullptr;
            }
        }
        bounceRangeEnd(req.nvtxGrantWait);
        bounceRangeEnd(req.nvtxCreditStarved);
        bounceRangeEnd(req.nvtxArenaStarved);
        bounceRangeEnd(req.nvtxReq);
        try
        {
            req.promise->set_value({TransferState::kFAILURE, BounceFailReason::kShutdown});
        }
        catch (...)
        {
            // set_value throws std::future_error ONLY if the promise is already satisfied — a benign
            // double-resolve (a request resolves exactly once); intentionally ignored, not a failure.
        }
    }
    mRequests.clear();
    // Cancel/release deferred writes without recycling their regions or sending deferred control
    // messages: no producer remains, and shutdown must not grant work against an arena being torn down.
    for (auto& o : mOrphanLocal)
    {
        if (o.xfer != nullptr && !o.xfer->release())
        {
            releaseRetry.push_back(std::move(o.xfer));
        }
    }
    mOrphanLocal.clear();
    // Retry failed cancellations once after all producers/futures have been stopped. A persistent
    // backend failure gets a final bounded attempt from the status object's destructor.
    for (auto& status : releaseRetry)
    {
        if (!status->release())
        {
            TLLM_LOG_WARNING(
                "BounceTransport(%s): NIXL handle still retained after shutdown retry", mCtx.selfName.c_str());
        }
    }
    // Deferred cancels aren't sent at shutdown: in-flight RDMA writes aren't drained by the device
    // sync (they're NIXL, not CUDA-stream), so a cancel could still race a write. The receiver
    // reclaims those regions via its own teardown / peer-loss path.
    mPendingCancel.clear();
}

// ============================================================================
// BounceTransport — the reactor
// ============================================================================

BounceTransport::BounceTransport(std::string selfName, BounceConfig cfg, int deviceId, ControlChannel* channel,
    NixlTransferAgent& agent, BounceArena* arena, ExecPool* exec)
    : mCtx(std::move(selfName), cfg, deviceId, channel, agent, arena, exec)
    , mReceiver(mCtx)
    , mSender(mCtx)
{
    // A bounce chunk must fit a fully-drained arena, or its GRANT can never succeed and the flow
    // stalls until requestTimeoutMs. The buddy allocator rounds usable capacity down and each allocation up
    // to a power of two, so comparing maxChunkSizeBytes directly with arenaSizeBytes is insufficient
    // (e.g. a 96 MiB arena has only 64 MiB usable, so a 65 MiB chunk never fits). Clamp to the largest
    // block the drained arena can actually hand out.
    std::size_t const cap = mCtx.scheduler.arenaCapacity();
    if (mCtx.cfg.maxChunkSizeBytes > cap)
    {
        TLLM_LOG_WARNING(
            "BounceTransport(%s): maxChunkSizeBytes=%zu exceeds usable arena capacity=%zu; clamping to %zu",
            mCtx.selfName.c_str(), static_cast<std::size_t>(mCtx.cfg.maxChunkSizeBytes), cap, cap);
        mCtx.cfg.maxChunkSizeBytes = cap;
    }
    try
    {
        mReceiver.startWorkers();
        mIoThread = std::thread(&BounceTransport::ioLoop, this);
    }
    catch (...)
    {
        // Thread creation failed partway (resource exhaustion). The destructor won't run — the
        // constructor is throwing — so join any scatter workers already spawned here, or their
        // std::thread destructors call std::terminate.
        mCtx.stop.store(true, std::memory_order_release);
        mReceiver.wake();
        mReceiver.joinWorkers();
        throw;
    }
}

BounceTransport::~BounceTransport()
{
    shutdown();
}

void BounceTransport::shutdown()
{
    bool expected = false;
    if (!mCtx.stop.compare_exchange_strong(expected, true))
    {
        return;       // already shut down
    }
    mReceiver.wake(); // wake scatter workers so they observe stop
    if (mIoThread.joinable())
    {
        mIoThread.join();
    }
    mReceiver.joinWorkers();
    // Threads are joined, but in-flight gather/scatter kernels may still be queued on ExecPool
    // streams referencing the arena. Drain the device BEFORE we release contexts and let the caller
    // tear down ExecPool/BounceArena — otherwise ~ExecPool's cudaStreamDestroy + ~BounceArena's
    // cudaFree could race a kernel still reading the arena. (Teardown-only; a full-device sync is
    // acceptable here.) Warn-only (we're in teardown / a dtor path -> must not throw); the user
    // should still see a GPU fault. shutdown() may run on a thread whose current device isn't ours.
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
    TLLM_CUDA_CHECK_WARN(cudaDeviceSynchronize());
    mSender.failAll();
}

bool BounceTransport::addPeer(std::string const& peer, std::string const& endpoint)
{
    return mCtx.channel->addPeer(peer, endpoint);
}

std::string BounceTransport::localHandshakeBlob() const
{
    auto endpoint = mCtx.channel->localEndpoint();
    if (endpoint.empty())
    {
        TLLM_THROW("BounceTransport(%s): ZMQ control requires a non-empty local endpoint", mCtx.selfName.c_str());
    }
    BounceHandshake handshake;
    handshake.wireVersion = kBounceVersion;
    handshake.controlKind = BounceControlKind::kZMQ;
    handshake.arenaUsableCapacityBytes = mCtx.scheduler.arenaCapacity();
    handshake.maxChunkSizeBytes = mCtx.cfg.maxChunkSizeBytes; // post-ctor-clamp effective value
    handshake.endpoint = std::move(endpoint);
    return encodeHandshake(handshake);
}

bool BounceTransport::registerPeerHandshake(std::string const& peer, std::string const& blob)
{
    // Treat registration as replacement, not an additive update. A peer may be reloaded under the
    // same name with bounce disabled or with changed settings; clear the previously validated route
    // first so any missing/malformed/incompatible replacement immediately falls back to NIXL.
    mCtx.channel->removePeer(peer);
    {
        std::lock_guard<std::mutex> lk(mPeerMu);
        mHandshakedPeers.erase(peer);
    }
    if (blob.empty())
    {
        return false; // bounce not advertised by this peer
    }

    BounceHandshake handshake;
    if (!decodeHandshake(blob, handshake))
    {
        TLLM_LOG_WARNING(
            "BounceTransport(%s): peer %s advertised an unparsable bounce handshake -> bounce disabled for "
            "this peer (NIXL fallback)",
            mCtx.selfName.c_str(), peer.c_str());
        return false;
    }
    auto const localControlKind = BounceControlKind::kZMQ;
    // STRICT equality. maxChunkSizeBytes is compared on the effective (post-clamp) values both
    // sides advertise; each side already clamped its own value to its usable arena capacity, so
    // equality also guarantees our chunks always fit the peer's arena and its scatter scratch
    // (sized for its own maxChunkSizeBytes). Local-only knobs (worker/stream counts, timeouts,
    // copy backends, granularity, arena size) intentionally do NOT have to match.
    if (handshake.wireVersion != kBounceVersion || handshake.controlKind != localControlKind
        || handshake.maxChunkSizeBytes != mCtx.cfg.maxChunkSizeBytes)
    {
        TLLM_LOG_WARNING(
            "BounceTransport(%s): peer %s bounce handshake incompatible (wireVersion %u vs %u, controlKind "
            "%u vs %u, maxChunkSizeBytes %llu vs %zu) -> bounce disabled for this peer (NIXL fallback)",
            mCtx.selfName.c_str(), peer.c_str(), static_cast<unsigned>(handshake.wireVersion),
            static_cast<unsigned>(kBounceVersion), static_cast<unsigned>(handshake.controlKind),
            static_cast<unsigned>(localControlKind), static_cast<unsigned long long>(handshake.maxChunkSizeBytes),
            static_cast<std::size_t>(mCtx.cfg.maxChunkSizeBytes));
        return false;
    }
    if (!mCtx.channel->addPeer(peer, handshake.endpoint))
    {
        TLLM_LOG_WARNING(
            "BounceTransport(%s): peer %s bounce endpoint could not be registered -> bounce disabled for "
            "this peer (NIXL fallback)",
            mCtx.selfName.c_str(), peer.c_str());
        return false;
    }
    std::lock_guard<std::mutex> lk(mPeerMu);
    mHandshakedPeers.insert(peer);
    return true;
}

bool BounceTransport::hasPeerHandshake(std::string const& peer) const
{
    std::lock_guard<std::mutex> lk(mPeerMu);
    return mHandshakedPeers.count(peer) > 0;
}

void BounceTransport::forgetPeer(std::string const& peer)
{
    // Drop the control-channel DEALER to this peer SYNCHRONOUSLY here (ControlChannel::removePeer is
    // thread-safe). Doing it on the caller thread — rather than on the IO thread in drainForgets —
    // gives a deterministic happens-before for any addPeer() the caller issues after forgetPeer()
    // returns (e.g. re-establishing a peer that came back): the dealer is already gone, so that
    // addPeer() rebuilds it instead of racing an async removePeer that would otherwise erase the
    // freshly re-added dealer. A pending send to the now-removed peer is dropped (it is being
    // invalidated), which degrades any in-flight request to a FAILURE — never corruption.
    mCtx.channel->removePeer(peer);
    {
        // Also drop it from the bounce-capable set so the fast path stops engaging it immediately;
        // a fresh loadRemoteAgent must re-validate its handshake.
        std::lock_guard<std::mutex> lk(mPeerMu);
        mHandshakedPeers.erase(peer);
    }
    // The scheduler / request-table reclaim still runs on the IO thread (drainForgets) so that state
    // stays owned by a single thread. Safe to call from invalidateRemoteAgent.
    std::lock_guard<std::mutex> lk(mForgetMu);
    mForgetPeers.push_back(peer);
}

void BounceTransport::drainForgets()
{
    std::vector<std::string> peers;
    {
        std::lock_guard<std::mutex> lk(mForgetMu);
        if (mForgetPeers.empty())
        {
            return;
        }
        peers.swap(mForgetPeers);
    }
    for (auto const& peer : peers)
    {
        mReceiver.forget(peer); // reclaim receiver-side credits/jobs of the gone peer
        mSender.forget(peer);   // fail in-flight sender requests to the gone peer
        // NOTE: the control-channel DEALER for this peer was already dropped synchronously in
        // forgetPeer() (see there); we only reclaim scheduler/request state on this (IO) thread.
    }
}

void BounceTransport::ioLoop()
{
    bounceNameThread("bounceIO");
    // Pin this thread to our device up front; if it fails, every CUDA op in the loop targets the wrong
    // device (the transport is effectively broken). Can't throw out of a thread fn -> warn-only.
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
    std::string peer;
    std::string blob;
    while (!mCtx.stop.load(std::memory_order_acquire))
    {
        // Exception boundary: this is a thread entry, so anything escaping here would
        // std::terminate the process. recv() can throw zmq::error_t (e.g. EINTR/ETERM from
        // zmq_poll) and the handlers may throw on peer input; log, back off briefly (so a
        // persistent error can't hot-spin the core), and keep serving — in-flight requests still
        // resolve via checkTimeouts, never a hang (R5).
        try
        {
            tick(peer, blob);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("BounceTransport(%s): IO tick failed: %s", mCtx.selfName.c_str(), e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void BounceTransport::tick(std::string& peer, std::string& blob)
{
    // Adaptive poll wait. When there is GPU/RDMA work to poll — gather/write in flight on the
    // sender (localHeldCount>0) or scatter in flight on the receiver (mScattering) — use a 0ms
    // timeout so cudaEventQuery / getXferStatus / scatter completions are detected ASAP (low
    // latency on the critical path; also keeps UCX progress driven). When fully idle (only
    // waiting on a control message), sleep up to 1ms so the IO thread doesn't busy-spin a core.
    // A request merely waiting for a GRANT (nothing posted yet) is NOT "busy", so a stalled /
    // unreachable peer won't spin a core until requestTimeoutMs. Both checks are IO-thread-only.
    bool const busy = mSender.busy() || mReceiver.busy();
    int const timeoutMs = busy ? 0 : 1;
    bool work = false;
    if (mCtx.channel->recv(peer, blob, timeoutMs))
    {
        dispatch(peer, blob);
        work = true;
    }
    work |= mSender.drainGatherReady(); // post writes for chunks whose gather kernel just finished
    work |= mSender.pollSenderHandles();
    work |= mReceiver.drainScatterDone();
    work |= mSender.drainOrphanLocal(); // recycle failed-but-in-flight write regions once their write ends
    drainForgets();
    mSender.drainPendingPosts();        // retry credits parked when the arena was full (onAck freed regions)
    mSender.checkTimeouts();
    mReceiver.checkTimeouts();          // expire grant leases of silent senders + free post-quarantine regions
    // Idle backoff: when there IS in-flight work (busy → 0ms poll) but nothing actually advanced
    // this pass — the classic case being a gather stalled behind unrelated model kernels (gather
    // event stays NotReady) — keep latency low for the first few spins, then sleep briefly so we
    // don't peg a core at 100%. Any control message or forward progress resets the counter.
    if (work)
    {
        mIdleSpins = 0;
    }
    else if (busy)
    {
        constexpr std::uint32_t kSpinBeforeBackoff = 64;
        if (mIdleSpins < kSpinBeforeBackoff)
        {
            ++mIdleSpins;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

void BounceTransport::dispatch(std::string const& peer, std::string const& blob)
{
    BounceMsgHeader h{};
    if (!decodeHeader(blob, h))
    {
        return;
    }
    switch (static_cast<BounceMsgType>(h.msgType))
    {
    case BounceMsgType::kWANT: mReceiver.onWant(peer, h, blob); break; // [R]
    case BounceMsgType::kGRANT: mSender.onGrant(peer, h, blob); break; // [S]
    case BounceMsgType::kDATA: mReceiver.onData(peer, h, blob); break; // [R]
    case BounceMsgType::kACK: mSender.onAck(peer, h); break;           // [S]
    default: break;
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
