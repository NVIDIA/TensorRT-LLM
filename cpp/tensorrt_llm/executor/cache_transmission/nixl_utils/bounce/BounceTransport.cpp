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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/GatherScatterKernel.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

namespace
{
constexpr char kSep = '\x1f'; // unit separator: agent names won't contain it

std::string makeKey(std::string const& peer, std::uint64_t rid)
{
    return peer + kSep + std::to_string(rid);
}

std::pair<std::string, std::uint64_t> splitKey(std::string const& key)
{
    auto pos = key.rfind(kSep);
    return {key.substr(0, pos), std::strtoull(key.c_str() + pos + 1, nullptr, 10)};
}

// Lay out plan arrays [srcs|dsts|sizes] into the exec context's pinned host buffer, make them
// device-accessible, then launch the batched copy. Direction-agnostic (gather or scatter); the data
// region (arena offset) is encoded into hsrcs/hdsts by the caller. Two opt-in knobs (default off):
//   - zeroCopy: skip the H2D of the plan arrays — the kernel reads them straight from pinned host via
//     ctx->hostPinnedDev (saves a small copy; likely a loss for large n — PCIe reads in-kernel).
//   - cub: use cub::DeviceMemcpy::Batched (ctx->cubTemp workspace) instead of the custom kernel.
// The two compose: arg source (scratch H2D vs pinned) x copy backend (custom vs cub).
cudaError_t launchPacked(ExecCtx* ctx, std::vector<std::uint64_t> const& hsrcs, std::vector<std::uint64_t> const& hdsts,
    std::vector<std::uint32_t> const& hsizes, bool zeroCopy, bool cub)
{
    auto const n = static_cast<std::uint32_t>(hsrcs.size());
    if (n == 0)
    {
        return cudaSuccess;
    }
    std::size_t const b64 = static_cast<std::size_t>(n) * sizeof(std::uint64_t);
    std::size_t const b32 = static_cast<std::size_t>(n) * sizeof(std::uint32_t);
    // The plan arrays must fit this context's scratch/hostPinned (both sized for maxDescsPerChunk).
    // On the RECEIVER, `n` comes from the peer's DATA message — if that peer was built with a larger
    // maxChunkBytes it can carry more entries than our scratch holds. Reject rather than overflow the
    // host/device buffers (caller turns this into "no ACK" -> sender times out, never a false ACK).
    if (2 * b64 + b32 > ctx->scratchBytes)
    {
        return cudaErrorInvalidValue;
    }
    // Pack the plan into pinned host (the H2D source, and what the kernel reads under zeroCopy).
    auto* host = static_cast<std::uint8_t*>(ctx->hostPinned);
    std::memcpy(host, hsrcs.data(), b64);
    std::memcpy(host + b64, hdsts.data(), b64);
    std::memcpy(host + 2 * b64, hsizes.data(), b32);
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
    if (cub && ctx->cubTemp != nullptr)
    {
        std::size_t need = 0;
        cudaError_t const st = batchedCopyCubTempBytes(n, need);
        if (st != cudaSuccess)
        {
            return st;
        }
        if (need > ctx->cubTempBytes)
        {
            return cudaErrorMemoryAllocation; // shouldn't happen: n <= maxDescs the temp was sized for
        }
        return launchBatchedCopyCub(dsrcs, ddsts, dsizes, n, ctx->stream, ctx->cubTemp, need);
    }
    return launchBatchedCopy(dsrcs, ddsts, dsizes, n, ctx->stream);
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
    std::uint32_t const workers = mCtx.cfg.scatterWorkers > 0 ? mCtx.cfg.scatterWorkers : 1;
    mWorkers.reserve(workers);
    for (std::uint32_t i = 0; i < workers; ++i)
    {
        mWorkers.emplace_back(&BounceReceiver::scatterWorkerLoop, this);
    }
}

void BounceReceiver::wake()
{
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
}

void BounceReceiver::onWant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    std::vector<std::uint32_t> chunkBytes;
    std::string endpoint;
    if (!decodeWant(blob, h, chunkBytes, endpoint))
    {
        return;
    }
    // Self-bootstrap the reverse control path. The disagg metadata exchange is one-directional — the
    // KV sender loads OUR agent metadata (so it can WANT us), but we never load the sender's, so we'd
    // have no DEALER to send GRANT/ACK back and every transfer would stall to leaseTimeout. The WANT
    // carries the sender's bounce endpoint; register it here (addPeer is idempotent). See DESIGN.md §7.
    if (!endpoint.empty())
    {
        mCtx.channel->addPeer(peer, endpoint);
    }
    auto const key = makeKey(peer, h.requestId);
    if (isCancelWant(chunkBytes))
    {
        // Explicit cancel/abort (the sender failed or retracted): precisely free this flow's
        // granted-but-unwritten regions now — otherwise they stay held until peer loss and a
        // long-running receiver leaks a window's worth per failed rid. Any region whose scatter is
        // still running is deferred (flagged orphaned in mScattering, freed on completion).
        std::vector<std::uint64_t> deferred;
        mCtx.sendGrants(mCtx.scheduler.reclaimFlow(key, scatteringRegions(), deferred));
        for (auto off : deferred)
        {
            mScattering[off] = true;
        }
        return;
    }
    mCtx.sendGrants(mCtx.scheduler.onWant(key, chunkBytes));
}

void BounceReceiver::onData(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    std::vector<BounceScatterEntry> entries;
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
    job.entries = std::move(entries);
    mScattering.emplace(job.offset, false); // a worker is about to read this incoming region (not yet orphaned)
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
    bool const didWork = !done.empty();
    for (auto& d : done)
    {
        // Only ACK a SUCCESSFUL scatter — a false ACK would tell the sender corrupt/absent data
        // landed. On failure (d.ok==false) we still free the region below (no leak) but send no ACK,
        // so the sender's chunk stalls and the request fails via leaseTimeout instead of corrupting.
        if (d.ok)
        {
            mCtx.channel->sendTo(d.peer, encodeAck(d.rid, d.chunkIdx, d.offset));
        }
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
    std::vector<std::uint64_t> deferred;
    mCtx.sendGrants(mCtx.scheduler.reclaimByPrefix(peer + kSep, scatteringRegions(), deferred));
    for (auto off : deferred)
    {
        mScattering[off] = true;
    }
}

void BounceReceiver::scatterWorkerLoop()
{
    // Pin this worker to our device. Can't throw out of a thread fn -> warn-only (the loop's CUDA ops
    // would then target the wrong device, so this is a real fault, just non-recoverable here).
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
    std::vector<std::uint64_t> hsrcs;
    std::vector<std::uint64_t> hdsts;
    std::vector<std::uint32_t> hsizes;
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
        hsrcs.resize(n);
        hdsts.resize(n);
        hsizes.resize(n);
        // Validate every scatter SOURCE stays inside the arena before launching. bounceOffset/size
        // come from the peer's DATA message; a buggy/hostile peer could point them outside the
        // region/arena and make the kernel read out of bounds. (dstAddr is the caller's own KV target
        // by design, so it isn't bounded here.) Any bad entry -> skip launch, no ACK (sender times out).
        std::uint64_t const arenaLo = mCtx.arena->baseAddr();
        std::uint64_t const arenaHi = arenaLo + mCtx.arena->bytes();
        auto const regionBase = arenaLo + job.offset;
        bool srcInBounds = (regionBase >= arenaLo && regionBase <= arenaHi);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            std::uint64_t const s = regionBase + job.entries[i].bounceOffset;
            srcInBounds = srcInBounds && s >= regionBase && s + job.entries[i].size <= arenaHi;
            hsrcs[i] = s;
            hdsts[i] = job.entries[i].dstAddr;
            hsizes[i] = job.entries[i].size;
        }
        // Scatter into the final dst, then wait so the data is at dst before we ACK. A bad source, a
        // failed launch, OR a stream error must NOT produce an ACK — an ACK tells the sender its KV
        // data landed, so a false ACK here is silent corruption. On error we still free the region
        // (below) but mark !ok so drainScatterDone skips the ACK; the sender then times out -> FAILURE.
        cudaError_t const launchErr = srcInBounds
            ? launchPacked(ctx, hsrcs, hdsts, hsizes, mCtx.cfg.zeroCopyArgs, mCtx.cfg.cubCopy)
            : cudaErrorInvalidValue;
        cudaError_t const syncErr = (launchErr == cudaSuccess) ? cudaStreamSynchronize(ctx->stream) : cudaSuccess;
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
        {
            std::lock_guard<std::mutex> lk(mDoneMu);
            mDone.push_back(ScatterDone{job.key, job.peer, job.rid, job.chunkIdx, job.offset, ok});
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

std::shared_future<TransferState> BounceSender::submit(
    TransferDescs const& srcDescs, TransferDescs const& dstDescs, std::string const& peer)
{
    auto plan = BounceTransferPlan::build(
        srcDescs, dstDescs, mCtx.cfg.maxChunkBytes, std::max<std::size_t>(1024ULL, mCtx.cfg.maxChunkBytes / 256ULL));
    auto const numChunks = static_cast<std::uint32_t>(plan.numChunks());

    auto promise = std::make_shared<std::promise<TransferState>>();
    auto fut = promise->get_future().share();
    if (numChunks == 0)
    {
        promise->set_value(TransferState::kSUCCESS);
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
    {
        std::lock_guard<std::mutex> lk(mReqMu);
        Request req;
        req.peer = peer;
        req.numChunks = numChunks;
        req.plan = std::move(plan);
        req.promise = promise;
        req.lastProgress = std::chrono::steady_clock::now();
        mRequests.emplace(rid, std::move(req));
    }
    // Ask the receiver to grant a region for each chunk of this request flow. The WANT carries our
    // own control endpoint so the receiver can addPeer us and send GRANT/ACK back (self-bootstrap).
    mCtx.channel->sendTo(peer, encodeWant(rid, chunkBytes, mCtx.channel->localEndpoint()));
    return fut;
}

void BounceSender::onGrant(std::string const& peer, BounceMsgHeader const& h, std::string const& blob)
{
    (void) peer; // the GRANT sender is this request's peer; we post to req.peer below
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
    for (auto const& credit : credits)
    {
        req.pendingCredits.push_back(credit);
    }
    pumpRequest(h.requestId, req);
}

void BounceSender::pumpRequest(std::uint64_t rid, Request& req)
{
    while (!req.pendingCredits.empty())
    {
        if (req.nextPost >= req.numChunks)
        {
            // Over-grant: receiver handed more credits than we have chunks. Shouldn't happen under
            // the protocol (receiver grants at most numChunks). Log it — silently dropping would
            // mask an upstream bug and leak the receiver-side regions backing these credits.
            TLLM_LOG_WARNING("BounceTransport(%s): rid=%llu over-grant, dropping %zu extra credit(s)",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), req.pendingCredits.size());
            req.pendingCredits.clear();
            break;
        }
        std::uint32_t const chunkIdx = req.nextPost;
        auto const& chunk = req.plan.chunks()[chunkIdx];
        // Non-blocking: borrow an exec context (cheap to return), then a gather-staging region sized
        // to this chunk from the SHARED arena. If either is unavailable, leave the credit parked and
        // bail; the IO loop retries via drainPendingPosts() once an ACK frees a region / context —
        // never blocks here, so an oversubscribed arena (many peers, or both roles) degrades to
        // backpressure, not deadlock.
        ExecCtx* ctx = mCtx.exec->tryAcquire();
        if (ctx == nullptr)
        {
            break;
        }
        auto localOff = mCtx.scheduler.acquireLocal(chunk.packedBytes);
        if (!localOff)
        {
            mCtx.exec->release(ctx);
            break;
        }
        BounceCreditEntry const credit = req.pendingCredits.front();
        req.pendingCredits.pop_front();
        auto const nDesc = static_cast<std::uint32_t>(chunk.srcPtrs.size());
        std::vector<std::uint64_t> hsrcs(nDesc);
        std::vector<std::uint64_t> hdsts(nDesc);
        std::vector<std::uint32_t> hsizes(nDesc);
        auto const regionBase = mCtx.arena->baseAddr() + *localOff;
        for (std::uint32_t i = 0; i < nDesc; ++i)
        {
            hsrcs[i] = chunk.srcPtrs[i];
            hdsts[i] = regionBase + chunk.bounceOffsets[i];
            hsizes[i] = chunk.sizes[i];
        }
        // gather into the region (cfg knobs select the H2D-vs-zero-copy arg path + custom-vs-cub copy)
        cudaError_t const gatherErr = launchPacked(ctx, hsrcs, hdsts, hsizes, mCtx.cfg.zeroCopyArgs, mCtx.cfg.cubCopy);
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
        p.remoteHandle = credit.regionHandle;
        p.remoteAddr = credit.addr;
        p.remoteDevId = credit.devId;
        p.writeBytes = static_cast<std::uint32_t>(chunk.packedBytes);
        // Gather in flight; the write is issued later by drainGatherReady once the event signals. If
        // the gather launch/record failed, go straight to GatherFailed so drainGatherReady fails the
        // request without ever trusting the (un)recorded event.
        p.state = gatherFailed ? PostState::GatherFailed : PostState::Gathering;
        req.posted.push_back(std::move(p));
        req.nextPost += 1;
        req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk's gather launched
    }
}

void BounceSender::drainPendingPosts()
{
    std::lock_guard<std::mutex> lk(mReqMu);
    for (auto& [rid, req] : mRequests)
    {
        if (!req.pendingCredits.empty())
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
            // state == Gathering: poll the gather event (non-blocking).
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
            // Gather done: NOW issue the RDMA write (postXferReq is not stream-ordered, so it was
            // correct to wait for the gather before posting). The gather has completed, so the write
            // no longer needs the gather stream's ordering — return the exec context immediately for
            // another chunk to reuse (the region stays held until ACK; see ExecPool DESIGN).
            p.xfer = mCtx.engine->postWrite(
                req.peer, mCtx.arena->at(p.localOffset), p.remoteAddr, p.remoteDevId, p.writeBytes, p.ctx->stream);
            mCtx.exec->release(p.ctx);
            p.ctx = nullptr;
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
            failRequest(rid, it->second);
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
            XferState const st = mCtx.engine->poll(p.xfer);
            if (st == XferState::kDone)
            {
                auto const& chunk = req.plan.chunks()[p.chunkIdx];
                auto const nDesc = static_cast<std::uint32_t>(chunk.srcPtrs.size());
                std::vector<BounceScatterEntry> entries(nDesc);
                for (std::uint32_t i = 0; i < nDesc; ++i)
                {
                    entries[i].bounceOffset = chunk.bounceOffsets[i];
                    entries[i].dstAddr = chunk.dstPtrs[i];
                    entries[i].size = chunk.sizes[i];
                    entries[i].deviceId = chunk.dstDeviceId;
                }
                mCtx.channel->sendTo(req.peer, encodeData(rid, p.chunkIdx, req.numChunks, p.remoteHandle, entries));
                mCtx.engine->release(p.xfer);
                p.state = PostState::Sent;
                didWork = true;
            }
            else if (st == XferState::kFailed)
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
            failRequest(rid, it->second);
            didWork = true;
        }
    }
    return didWork;
}

void BounceSender::onAck(std::string const& peer, BounceMsgHeader const& h)
{
    (void) peer;
    std::lock_guard<std::mutex> lk(mReqMu);
    auto it = mRequests.find(h.requestId);
    if (it == mRequests.end())
    {
        return;
    }
    Request& req = it->second;
    bool found = false;
    for (auto pit = req.posted.begin(); pit != req.posted.end(); ++pit)
    {
        if (pit->chunkIdx == h.chunkIdx)
        {
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
        // Duplicate / unknown ACK (zmq reconnect, retransmit). Do NOT count it — an over-count
        // could push acked past numChunks and resolve SUCCESS before all chunks actually landed.
        return;
    }
    req.acked += 1;
    req.lastProgress = std::chrono::steady_clock::now(); // forward progress: a chunk was ACKed
    if (req.acked >= req.numChunks)
    {
        try
        {
            req.promise->set_value(TransferState::kSUCCESS);
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
    if (mCtx.cfg.leaseTimeoutMs <= 0)
    {
        return; // timeout disabled (e.g. tests that intentionally wait forever)
    }
    auto const now = std::chrono::steady_clock::now();
    auto const limit = std::chrono::milliseconds(mCtx.cfg.leaseTimeoutMs);
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
                // Peer never granted / stalled (unreachable / not bounce-ready / congested).
                // Fail the request so wait() returns FAILURE instead of hanging (R5).
                failRequest(rid, it->second);
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
    for (auto const& o : mOrphanLocal)
    {
        if (mCtx.engine->poll(o.xfer) == XferState::kInProgress)
        {
            keep.push_back(o); // write still in flight -> the NIC may still read the region; wait
            continue;
        }
        // Terminal (Done or Failed): the NIC is finished with the region (source AND the receiver's
        // destination) -> recycle the local source now.
        mCtx.engine->release(o.xfer);
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

void BounceSender::failRequest(std::uint64_t rid, Request& req)
{
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
        if (p.state == PostState::Writing)
        {
            mOrphanLocal.push_back(OrphanLocal{p.xfer, p.localOffset, req.peer, rid});
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
    try
    {
        req.promise->set_value(TransferState::kFAILURE);
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
            failRequest(rid, it->second);
        }
    }
}

void BounceSender::failAll()
{
    // Fail any still-pending requests so no submit() future hangs, releasing their in-flight
    // transfer handles first (same handle-leak fix as failRequest). Called after the device has been
    // synced and the IO thread joined, so no lock contention — but keep mReqMu for consistency.
    std::lock_guard<std::mutex> lk(mReqMu);
    for (auto& [rid, req] : mRequests)
    {
        for (auto& p : req.posted)
        {
            if (p.state == PostState::Writing)
            {
                mCtx.engine->release(p.xfer); // RDMA write in flight -> release its transfer handle
            }
            if (p.ctx != nullptr)
            {
                mCtx.exec->release(p.ctx); // still-gathering chunk: return its borrowed exec context
                p.ctx = nullptr;
            }
        }
        try
        {
            req.promise->set_value(TransferState::kFAILURE);
        }
        catch (...)
        {
            // set_value throws std::future_error ONLY if the promise is already satisfied — a benign
            // double-resolve (a request resolves exactly once); intentionally ignored, not a failure.
        }
    }
    mRequests.clear();
    // Release any deferred orphan-local xfer handles (their writes are done — device was synced
    // before this call; the arena is torn down by the caller next, so no need to recycle the regions).
    for (auto const& o : mOrphanLocal)
    {
        mCtx.engine->release(o.xfer);
    }
    mOrphanLocal.clear();
    // Deferred cancels aren't sent at shutdown: in-flight RDMA writes aren't drained by the device
    // sync (they're NIXL, not CUDA-stream), so a cancel could still race a write. The receiver
    // reclaims those regions via its own teardown / peer-loss path.
    mPendingCancel.clear();
}

// ============================================================================
// BounceTransport — the reactor
// ============================================================================

BounceTransport::BounceTransport(std::string selfName, BounceConfig cfg, int deviceId, ControlChannel* channel,
    TransferEngine* engine, BounceArena* arena, ExecPool* exec)
    : mCtx(std::move(selfName), cfg, deviceId, channel, engine, arena, exec)
    , mReceiver(mCtx)
    , mSender(mCtx)
{
    mReceiver.startWorkers();
    mIoThread = std::thread(&BounceTransport::ioLoop, this);
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

void BounceTransport::addPeer(std::string const& peer, std::string const& endpoint)
{
    mCtx.channel->addPeer(peer, endpoint);
}

void BounceTransport::forgetPeer(std::string const& peer)
{
    // Queue; the actual reclaim runs on the IO thread (drainForgets) so the scheduler and the
    // request table stay owned by a single thread. Safe to call from invalidateRemoteAgent.
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
    }
}

void BounceTransport::ioLoop()
{
    // Pin this thread to our device up front; if it fails, every CUDA op in the loop targets the wrong
    // device (the transport is effectively broken). Can't throw out of a thread fn -> warn-only.
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mCtx.deviceId));
    std::string peer;
    std::string blob;
    while (!mCtx.stop.load(std::memory_order_acquire))
    {
        // Adaptive poll wait. When there is GPU/RDMA work to poll — gather/write in flight on the
        // sender (localHeldCount>0) or scatter in flight on the receiver (mScattering) — use a 0ms
        // timeout so cudaEventQuery / getXferStatus / scatter completions are detected ASAP (low
        // latency on the critical path; also keeps UCX progress driven). When fully idle (only
        // waiting on a control message), sleep up to 1ms so the IO thread doesn't busy-spin a core.
        // A request merely waiting for a GRANT (nothing posted yet) is NOT "busy", so a stalled /
        // unreachable peer won't spin a core until leaseTimeout. Both checks are IO-thread-only.
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
