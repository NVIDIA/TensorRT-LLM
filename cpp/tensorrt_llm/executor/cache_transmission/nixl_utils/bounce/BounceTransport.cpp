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
void appendSplitInto(PlanBufs const& bufs, std::size_t& idx, std::uint64_t src, std::uint64_t dst,
    std::uint32_t size, std::size_t maxPieces)
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
// Two opt-in knobs (default off):
//   - zeroCopy: skip the H2D of the plan arrays — the kernel reads them straight from pinned host via
//     ctx->hostPinnedDev (saves a small copy; likely a loss for large n — PCIe reads in-kernel).
//   - cub: use cub::DeviceMemcpy::Batched (ctx->cubTemp workspace) instead of the custom kernel.
// The two compose: arg source (scratch H2D vs pinned) x copy backend (custom vs cub).
cudaError_t launchPrepared(ExecCtx* ctx, std::size_t n, bool zeroCopy, bool cub)
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
    if (cub && ctx->cubTemp != nullptr)
    {
        std::size_t need = 0;
        cudaError_t const st = batchedCopyCubTempBytes(n32, need);
        if (st != cudaSuccess)
        {
            return st;
        }
        if (need > ctx->cubTempBytes)
        {
            return cudaErrorMemoryAllocation; // shouldn't happen: n <= maxDescs the temp was sized for
        }
        return launchBatchedCopyCub(dsrcs, ddsts, dsizes, n32, ctx->stream, ctx->cubTemp, need);
    }
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
    // Self-bootstrap the reverse control path. The disagg metadata exchange is one-directional — the
    // KV sender loads OUR agent metadata (so it can WANT us), but we never load the sender's, so we'd
    // have no DEALER to send GRANT/ACK back and every transfer would stall to leaseTimeout. The WANT
    // carries the sender's bounce endpoint; register it here (addPeer is idempotent).
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
    std::vector<std::uint64_t> deferred;
    mCtx.sendGrants(mCtx.scheduler.reclaimByPrefix(peer + kSep, scatteringRegions(), deferred));
    for (auto off : deferred)
    {
        mScattering[off] = true;
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
        std::uint64_t const regionHi = regionBase + job.regionBytes;
        bool srcInBounds = (job.regionBytes > 0 && regionBase + job.regionBytes <= arenaLo + mCtx.arena->bytes());
        // Scatter into the final dst, then wait so the data is at dst before we ACK. A bad source, a
        // failed launch, OR a stream error must NOT produce an ACK — an ACK tells the sender its KV
        // data landed, so a false ACK here is silent corruption. On error we still free the region
        // (below) but mark !ok so drainScatterDone skips the ACK; the sender then times out -> FAILURE.
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
            // maxChunkBytes can carry more pieces than our pinned/scratch hold; reject rather than
            // overflow (no launch -> no ACK -> the sender times out, never a false ACK).
            std::size_t const maxEntries = maxPlanEntries(ctx);
            std::uint64_t rawPieces = 0;
            for (std::uint32_t i = 0; i < n; ++i)
            {
                auto const& e = job.entries[i];
                // Run-level source bounds: every piece p reads region[bounceOffset + p*bounceStride
                // .. +pieceSize). count-1 and bounceStride are both u32 so the span product cannot
                // overflow u64. A count of 0 is malformed (a run always carries >= 1 piece).
                std::uint64_t const span
                    = static_cast<std::uint64_t>(e.count - 1) * e.bounceStride + e.pieceSize;
                srcInBounds = srcInBounds && e.count >= 1 && e.bounceOffset <= job.regionBytes
                    && span <= job.regionBytes - e.bounceOffset;
                rawPieces += std::max<std::uint32_t>(e.count, 1);
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
                launchErr = launchPrepared(ctx, nTotal, mCtx.cfg.zeroCopyArgs, mCtx.cfg.cubCopy);
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
            BounceNvtxScope ackScope(kNvtxAckSend, "ackSend rid=%llu chunk=%u",
                static_cast<unsigned long long>(job.rid), job.chunkIdx);
            mCtx.channel->sendTo(job.peer, encodeAck(job.rid, job.chunkIdx, job.offset));
        }
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
    BounceTransferPlan plan;
    {
        BounceNvtxScope planScope(kNvtxBuildPlan, "buildPlan nDesc=%zu", srcDescs.getDescs().size());
        plan = BounceTransferPlan::build(srcDescs, dstDescs, mCtx.cfg.maxChunkBytes,
            std::max<std::size_t>(1024ULL, mCtx.cfg.maxChunkBytes / 256ULL), !mCtx.cfg.noRunMerge);
    }
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
    if (mCtx.cfg.eagerGather)
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
    // Credits pair with chunks strictly in order (the receiver serves the WANT's size list FIFO), so
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
            if (!mCtx.cfg.eagerGather)
            {
                break; // classic path: a chunk's gather starts only once its GRANT arrived
            }
            if (req.posted.size() >= mCtx.cfg.effectiveWindow())
            {
                break; // eager depth cap: mirror the receiver's per-flow window
            }
        }
        // Defensive pairing check BEFORE committing resources. Credits pair with chunks by FIFO order,
        // and the receiver sizes each granted region to the chunkBytes[chunkIdx] in the WANT, so packedBytes
        // always fits. But the control channel does NOT guarantee GRANT ordering (a reconnect can
        // reorder messages), and a mispaired credit would make us RDMA-write packedBytes into a
        // smaller granted region — overflowing into an adjacent flow's region on the peer (silent
        // cross-flow corruption). Detect the mispair and abandon the flow (it then fails via
        // checkTimeouts) rather than corrupt the peer.
        if (haveCredit && chunk.packedBytes > req.pendingCredits.front().len)
        {
            TLLM_LOG_WARNING(
                "BounceTransport(%s): rid=%llu chunk=%u packedBytes=%zu > granted region len=%u (GRANT "
                "mispair/reorder); abandoning flow",
                mCtx.selfName.c_str(), static_cast<unsigned long long>(rid), chunkIdx,
                static_cast<std::size_t>(chunk.packedBytes), static_cast<unsigned int>(req.pendingCredits.front().len));
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
        auto const bufs = planBufs(ctx, nTotal);
        std::size_t idx = 0;
        for (std::uint32_t i = 0; i < nDesc; ++i)
        {
            appendSplitInto(bufs, idx, chunk.srcPtrs[i], regionBase + chunk.bounceOffsets[i], chunk.sizes[i],
                splitBudget(idx, nDesc - 1 - i, maxEntries));
        }
        // gather into the region (cfg knobs select the H2D-vs-zero-copy arg path + custom-vs-cub copy)
        cudaError_t const gatherErr = launchPrepared(ctx, nTotal, mCtx.cfg.zeroCopyArgs, mCtx.cfg.cubCopy);
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
        if (!req.pendingCredits.empty() || (mCtx.cfg.eagerGather && req.nextPost < req.numChunks))
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
            p.xfer = mCtx.engine->postWrite(
                req.peer, mCtx.arena->at(p.localOffset), p.remoteAddr, p.remoteDevId, p.writeBytes, nullptr);
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
                mCtx.engine->release(p.xfer);
                p.state = PostState::Sent;
                p.nvtxAckWait = bounceRangeStart(
                    kNvtxAckWait, "ackWait rid=%llu chunk=%u", static_cast<unsigned long long>(rid), p.chunkIdx);
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
    bool found = false;
    for (auto pit = req.posted.begin(); pit != req.posted.end(); ++pit)
    {
        if (pit->chunkIdx == h.chunkIdx)
        {
            // nvtxGather/nvtxWrite are 0 by Sent state; ending them too keeps a protocol-anomalous
            // early ACK (which erases this Posted) from leaving their spans dangling.
            bounceRangeEnd(pit->nvtxGather);
            bounceRangeEnd(pit->nvtxWrite);
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
        // Close this chunk's NVTX spans (whichever leg it died in); 0 handles are no-ops.
        bounceRangeEnd(p.nvtxGather);
        bounceRangeEnd(p.nvtxWrite);
        bounceRangeEnd(p.nvtxAckWait);
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
    bounceRangeEnd(req.nvtxGrantWait);
    bounceRangeEnd(req.nvtxCreditStarved);
    bounceRangeEnd(req.nvtxArenaStarved);
    bounceRangeEnd(req.nvtxReq);
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
            bounceRangeEnd(p.nvtxGather);
            bounceRangeEnd(p.nvtxWrite);
            bounceRangeEnd(p.nvtxAckWait);
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
        bounceRangeEnd(req.nvtxGrantWait);
        bounceRangeEnd(req.nvtxCreditStarved);
        bounceRangeEnd(req.nvtxArenaStarved);
        bounceRangeEnd(req.nvtxReq);
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
    // A bounce chunk must fit a fully-drained arena, or its GRANT can never succeed and the flow
    // stalls to leaseTimeout. The buddy allocator rounds usable capacity DOWN (to minBlock<<maxOrder)
    // and rounds each request UP to a power of two, so the naive "maxChunkBytes <= arenaBytes" is NOT
    // sufficient (e.g. a 96 MiB arena has only 64 MiB usable, so a 65 MiB chunk never fits). Clamp to
    // the largest block the drained arena can actually hand out.
    std::size_t const cap = mCtx.scheduler.arenaCapacity();
    if (mCtx.cfg.maxChunkBytes > cap)
    {
        TLLM_LOG_WARNING("BounceTransport(%s): maxChunkBytes=%zu exceeds usable arena capacity=%zu; clamping to %zu",
            mCtx.selfName.c_str(), static_cast<std::size_t>(mCtx.cfg.maxChunkBytes), cap, cap);
        mCtx.cfg.maxChunkBytes = cap;
    }
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
    // Drop the control-channel DEALER to this peer SYNCHRONOUSLY here (ControlChannel::removePeer is
    // thread-safe). Doing it on the caller thread — rather than on the IO thread in drainForgets —
    // gives a deterministic happens-before for any addPeer() the caller issues after forgetPeer()
    // returns (e.g. re-establishing a peer that came back): the dealer is already gone, so that
    // addPeer() rebuilds it instead of racing an async removePeer that would otherwise erase the
    // freshly re-added dealer. A pending send to the now-removed peer is dropped (it is being
    // invalidated), which degrades any in-flight request to a FAILURE — never corruption.
    mCtx.channel->removePeer(peer);
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
