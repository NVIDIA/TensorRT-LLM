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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/BatchedCopyPool.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/GatherScatterKernel.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

namespace
{

// Split large copy runs into pieces of this size when building the batched-copy plan arrays. The
// copy kernel assigns ONE thread block per plan entry, so a plan of a few huge coalesced runs would
// use only a few SMs; splitting restores the grid-level parallelism a per-desc plan has, without
// giving up the small wire messages. Python never sees this: it hands whole runs to submitCopy().
constexpr std::uint32_t kCopySplitBytes = 64U << 10;

// This entry's split budget: the plan slots still free after reserving ONE slot for every raw
// entry not yet appended (each needs at least one). Never below 1.
std::size_t splitBudget(std::size_t appended, std::size_t rawRemaining, std::size_t maxEntries)
{
    std::size_t const reserved = appended + rawRemaining;
    return maxEntries > reserved ? maxEntries - reserved : 1;
}

// Number of pieces appendSplitInto() will emit for one raw entry under `maxPieces`: one per full
// kCopySplitBytes piece up to the budget, the remainder as a single (possibly oversized) entry.
// Used for the exact-count pass that lets submitCopy() write the plan arrays straight into the
// pinned buffer (the [srcs|dsts|sizes] layout needs the total BEFORE the first write).
std::size_t piecesFor(std::uint32_t size, std::size_t maxPieces)
{
    std::size_t const want = (static_cast<std::size_t>(size) + kCopySplitBytes - 1) / kCopySplitBytes;
    return std::min(std::max<std::size_t>(want, 1), std::max<std::size_t>(maxPieces, 1));
}

// Plan-array views into a context's pinned buffer: [srcs(n) | dsts(n) | sizes(n)] — the layout the
// copy kernel consumes (through the buffer's device alias). Filled IN PLACE, one write pass.
struct PlanBufs
{
    std::uint64_t* srcs;
    std::uint64_t* dsts;
    std::uint32_t* sizes;
};

PlanBufs planBufs(void* hostPinned, std::size_t n)
{
    auto* host = static_cast<std::uint8_t*>(hostPinned);
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

} // namespace

BatchedCopyPool::BatchedCopyPool(
    std::uint32_t numStreams, std::size_t maxPlanEntries, int deviceId, CompletionPoller& poller)
    : mDeviceId(deviceId)
    , mMaxPlanEntries(maxPlanEntries)
    , mPlanBytes(maxPlanEntries * (2 * sizeof(std::uint64_t) + sizeof(std::uint32_t)))
    , mPoller(poller)
{
    TLLM_CHECK_WITH_INFO(numStreams > 0, "BatchedCopyPool: numStreams must be > 0");
    TLLM_CHECK_WITH_INFO(maxPlanEntries > 0, "BatchedCopyPool: maxPlanEntries must be > 0");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
    // Copies sit on the KV-transfer critical path but share the GPU with model kernels
    // (prefill/decode). At default priority a copy kernel queues behind those; greatest priority
    // lets its blocks be scheduled as soon as any SM frees up, without preempting running work.
    int leastPriority = 0;
    int greatestPriority = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    mCtxs.resize(numStreams);
    // A mid-loop TLLM_CUDA_CHECK throw would skip the destructor (the object is not fully
    // constructed), leaking every stream/event/buffer allocated so far -> clean up and rethrow.
    try
    {
        for (std::uint32_t i = 0; i < numStreams; ++i)
        {
            auto& c = mCtxs[i];
            c.id = i;
            // Mapped so the copy kernel reads the plan arrays straight from pinned host memory via
            // the device alias — no H2D staging, no H2D-then-kernel serialization.
            TLLM_CUDA_CHECK(cudaHostAlloc(&c.hostPinned, mPlanBytes, cudaHostAllocMapped));
            TLLM_CUDA_CHECK(cudaHostGetDevicePointer(&c.hostPinnedDev, c.hostPinned, 0));
            TLLM_CUDA_CHECK(cudaStreamCreateWithPriority(&c.stream, cudaStreamNonBlocking, greatestPriority));
            TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&c.event, cudaEventDisableTiming));
            mFree.push_back(i);
        }
    }
    catch (...)
    {
        destroyContexts();
        throw;
    }
}

BatchedCopyPool::~BatchedCopyPool()
{
    // A context in flight has its event registered with the CompletionPoller, which queries it from
    // the poll thread — destroying the event/stream now would hand CUDA a dangling handle and route
    // onTerminal into a destructed pool (use-after-free). Give in-flight contexts a short bounded
    // window to drain normally (event fires -> onTerminal returns the context), then UNREGISTER
    // every event of this pool from the poller BEFORE destroying anything: unregisterEvents holds
    // the poller's sweep mutex AND waits out any chain poster still executing outside it, so after
    // it returns no entry of ours is being queried or called back. The unregistered ids —
    // including the reserved id of an armed-but-unposted gather->RDMA chain — simply never report
    // to Python; waiters unblock via the reactor's request-timeout/stall watchdogs. Acceptable at
    // teardown only; the ordered path (poller.shutdown() BEFORE the pool is dropped) terminates
    // every id properly and makes this the GC-order fallback.
    // (A chain poster references only the agent/poller, never this pool, so a poster that slips
    // past the freeCount fast path below — its context was recycled at gather-completion — cannot
    // touch destroyed pool state.)
    // NOTE: this destructor may run under the GIL (nanobind tp_dealloc); the poll thread never
    // takes the GIL, so waiting on the poller mutex here cannot deadlock.
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (freeCount() < mCtxs.size() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (freeCount() < mCtxs.size())
    {
        std::vector<cudaEvent_t> events;
        events.reserve(mCtxs.size());
        for (auto const& c : mCtxs)
        {
            if (c.event != nullptr)
            {
                events.push_back(c.event);
            }
        }
        auto const removed = mPoller.unregisterEvents(events);
        TLLM_LOG_WARNING(
            "BatchedCopyPool: %zu context(s) still in flight at destruction; unregistered %zu pending "
            "event(s) from the CompletionPoller before teardown",
            mCtxs.size() - freeCount(), removed);
    }
    // destroyContexts() stream-synchronizes each context before freeing its pinned plan buffer, so
    // a still-running kernel of an unregistered entry cannot read freed memory.
    destroyContexts();
}

void BatchedCopyPool::destroyContexts()
{
    // Select the owning device before freeing; otherwise CUDA teardown targets the current device of
    // this thread. Must not throw (runs from the destructor and the constructor's failure path), so
    // every cleanup uses the warn-only check. Fields of never-initialized contexts are nullptr.
    TLLM_CUDA_CHECK_WARN(cudaSetDevice(mDeviceId));
    for (auto& c : mCtxs)
    {
        // Belt-and-suspenders: a context may still have a live kernel (a completion terminated
        // without waiting for the event, or the destructor's deadline-break path) — drain its
        // stream before freeing the pinned plan buffer that kernel reads.
        if (c.stream != nullptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(c.stream));
        }
        if (c.hostPinned != nullptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaFreeHost(c.hostPinned)); // also frees its hostPinnedDev alias
        }
        if (c.stream != nullptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaStreamDestroy(c.stream));
        }
        if (c.event != nullptr)
        {
            TLLM_CUDA_CHECK_WARN(cudaEventDestroy(c.event));
        }
    }
    mCtxs.clear();
    mFree.clear();
}

BatchedCopyPool::Ctx* BatchedCopyPool::tryAcquire()
{
    std::lock_guard<std::mutex> lk(mMu);
    if (mFree.empty())
    {
        return nullptr;
    }
    std::uint32_t const id = mFree.front();
    mFree.pop_front();
    return &mCtxs[id];
}

void BatchedCopyPool::release(Ctx* ctx)
{
    if (ctx == nullptr)
    {
        return;
    }
    std::lock_guard<std::mutex> lk(mMu);
    mFree.push_back(ctx->id);
}

std::size_t BatchedCopyPool::freeCount()
{
    std::lock_guard<std::mutex> lk(mMu);
    return mFree.size();
}

std::int64_t BatchedCopyPool::submitCopy(std::uint64_t const* srcs, std::uint64_t const* dsts,
    std::uint32_t const* sizes, std::size_t n, std::uint64_t* reserveChainId)
{
    return launchGather(srcs, dsts, /*dstBase=*/0, sizes, n, reserveChainId);
}

std::uint64_t BatchedCopyPool::registerPlan(std::uint64_t const* srcs, std::uint64_t const* bounceOffsets,
    std::uint32_t const* sizes, std::uint64_t const* chunkStarts, std::size_t nDescs, std::size_t nChunks)
{
    if (nChunks == 0 || chunkStarts[0] != 0 || chunkStarts[nChunks] != nDescs)
    {
        throw std::invalid_argument("BatchedCopyPool::registerPlan: malformed chunk boundaries");
    }
    for (std::size_t c = 0; c < nChunks; ++c)
    {
        if (chunkStarts[c + 1] < chunkStarts[c] || chunkStarts[c + 1] - chunkStarts[c] > mMaxPlanEntries)
        {
            throw std::invalid_argument(
                "BatchedCopyPool::registerPlan: chunk boundaries non-monotonic or chunk exceeds maxPlanEntries");
        }
    }
    auto plan = std::make_shared<RequestPlan>();
    plan->srcs.assign(srcs, srcs + nDescs);
    plan->bounceOffsets.assign(bounceOffsets, bounceOffsets + nDescs);
    plan->sizes.assign(sizes, sizes + nDescs);
    plan->chunkStarts.assign(chunkStarts, chunkStarts + nChunks + 1);
    std::lock_guard<std::mutex> lk(mPlanMu);
    std::uint64_t const handle = mNextPlanId++;
    mPlans.emplace(handle, std::move(plan));
    return handle;
}

void BatchedCopyPool::releasePlan(std::uint64_t handle)
{
    std::lock_guard<std::mutex> lk(mPlanMu);
    mPlans.erase(handle);
}

std::int64_t BatchedCopyPool::launchChunk(
    std::uint64_t handle, std::size_t chunkIdx, std::uint64_t stagingBase, std::uint64_t* reserveChainId)
{
    std::shared_ptr<RequestPlan const> plan;
    {
        std::lock_guard<std::mutex> lk(mPlanMu);
        auto const it = mPlans.find(handle);
        if (it == mPlans.end())
        {
            // Deterministic terminal for a launch racing the plan's release (the request already
            // failed): the caller's launch-error path handles it (no kernel was launched).
            throw std::invalid_argument("BatchedCopyPool::launchChunk: unknown plan handle");
        }
        plan = it->second;
    }
    if (chunkIdx >= plan->chunkStarts.size() - 1)
    {
        throw std::invalid_argument("BatchedCopyPool::launchChunk: chunk index out of range");
    }
    std::size_t const lo = static_cast<std::size_t>(plan->chunkStarts[chunkIdx]);
    std::size_t const hi = static_cast<std::size_t>(plan->chunkStarts[chunkIdx + 1]);
    return launchGather(plan->srcs.data() + lo, plan->bounceOffsets.data() + lo, stagingBase, plan->sizes.data() + lo,
        hi - lo, reserveChainId);
}

std::int64_t BatchedCopyPool::launchGather(std::uint64_t const* srcs, std::uint64_t const* dstOffsets,
    std::uint64_t dstBase, std::uint32_t const* sizes, std::size_t n, std::uint64_t* reserveChainId)
{
    TLLM_CHECK_WITH_INFO(
        n <= mMaxPlanEntries, "BatchedCopyPool::launchGather: n (%zu) > maxPlanEntries (%zu)", n, mMaxPlanEntries);
    Ctx* ctx = tryAcquire();
    if (ctx == nullptr)
    {
        return kBusy; // every stream context busy — caller retries on its next tick
    }
    try
    {
        // The stream/event belong to mDeviceId's context; the calling (Python reactor) thread may
        // have any current device, so pin it for the launch+record.
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        // Exact-count pass first: the packed [srcs|dsts|sizes] pinned layout needs the total number
        // of (post-split) entries before the first write.
        std::size_t nTotal = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            nTotal += piecesFor(sizes[i], splitBudget(nTotal, n - 1 - i, mMaxPlanEntries));
        }
        TLLM_CHECK_WITH_INFO(nTotal <= mMaxPlanEntries,
            "BatchedCopyPool::launchGather: split plan (%zu) exceeds maxPlanEntries (%zu)", nTotal, mMaxPlanEntries);

        // Fill the plan arrays in place in the context's pinned buffer, splitting runs so the
        // one-block-per-entry kernel keeps its grid-level parallelism.
        auto const bufs = planBufs(ctx->hostPinned, nTotal);
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            appendSplitInto(
                bufs, idx, srcs[i], dstBase + dstOffsets[i], sizes[i], splitBudget(idx, n - 1 - i, mMaxPlanEntries));
        }

        // The kernel reads the plan straight from pinned memory through the device alias.
        std::size_t const b64 = nTotal * sizeof(std::uint64_t);
        auto* base = static_cast<std::uint8_t*>(ctx->hostPinnedDev);
        auto const* dSrcs = reinterpret_cast<std::uint64_t const*>(base);
        auto const* dDsts = reinterpret_cast<std::uint64_t const*>(base + b64);
        auto const* dSizes = reinterpret_cast<std::uint32_t const*>(base + 2 * b64);
        TLLM_CUDA_CHECK(launchBatchedCopy(dSrcs, dDsts, dSizes, static_cast<std::uint32_t>(nTotal), ctx->stream));
        TLLM_CUDA_CHECK(cudaEventRecord(ctx->event, ctx->stream));
        // The poller's onTerminal returns the context to the free list from the poll thread —
        // contexts recycle in C++ without waiting for Python to drain the completion. Registered
        // inside the try so a throwing registration still releases the context below.
        // `reserveChainId` (gather->RDMA chain) is taken atomically WITH the registration, so the
        // poll thread can never complete the event before the reservation exists.
        return static_cast<std::int64_t>(mPoller.registerEvent(
            ctx->event, [this, ctx] { release(ctx); }, reserveChainId));
    }
    catch (...)
    {
        // The launch may have succeeded even though a later step (event record, poller
        // registration) threw — drain the stream so the recycled context cannot have a live kernel
        // still reading its pinned plan buffer. Warn-only: we are already unwinding.
        TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(ctx->stream));
        release(ctx);
        throw;
    }
}

std::int64_t BatchedCopyPool::submitScatterRuns(
    std::uint64_t regionBase, std::uint64_t regionBytes, ScatterRunWire const* runs, std::size_t nRuns)
{
    // ---- validation: a check-by-check replica of the Python reactor's _validate_and_expand_runs
    // (reactor.py), which guards against a malicious/corrupt DATA scattering outside the granted
    // region. Mapping (Python -> here):
    //   (1) region inside the arena (region_bytes <= 0 / region beyond arena end): stays in the
    //       PYTHON caller — it owns the arena bounds and checks BEFORE calling this.
    //   (2) n_runs > max_plan_entries or total_pieces > max_plan_entries: the two size checks
    //       below (nRuns first so the summing loop is itself bounded).
    //   (3) per run: count < 1, or bounce_offset > region_bytes, or
    //       span = (count-1)*bounce_stride + piece_size > region_bytes - bounce_offset:
    //       the per-run loop below. Arithmetic is exact like Python's int math: count,
    //       bounceStride and pieceSize are u32, so (count-1)*stride + piece < 2^64 — the u64
    //       span can never wrap, and bounceOffset <= regionBytes is established before the
    //       regionBytes - bounceOffset subtraction.
    // Destination addresses are NOT range-checked, exactly like the Python fallback: dst comes
    // from the sender's plan against addresses the receiver handed out; the region checks above
    // are the hostile-input guard.
    if (nRuns > mMaxPlanEntries)
    {
        TLLM_LOG_WARNING("BatchedCopyPool::submitScatterRuns: rejected %zu runs (max %zu)", nRuns, mMaxPlanEntries);
        return kScatterRejected;
    }
    std::uint64_t totalPieces = 0;
    for (std::size_t i = 0; i < nRuns; ++i)
    {
        ScatterRunWire const& r = runs[i];
        std::uint64_t const span
            = static_cast<std::uint64_t>(r.count - 1) * r.bounceStride + static_cast<std::uint64_t>(r.pieceSize);
        if (r.count < 1 || r.bounceOffset > regionBytes || span > regionBytes - r.bounceOffset)
        {
            TLLM_LOG_WARNING(
                "BatchedCopyPool::submitScatterRuns: run %zu out of region bounds "
                "(off=%llu span=%llu region=%llu count=%u)",
                i, static_cast<unsigned long long>(r.bounceOffset), static_cast<unsigned long long>(span),
                static_cast<unsigned long long>(regionBytes), r.count);
            return kScatterRejected;
        }
        totalPieces += r.count;
    }
    if (totalPieces > mMaxPlanEntries)
    {
        TLLM_LOG_WARNING("BatchedCopyPool::submitScatterRuns: rejected %llu pieces over %zu runs (max %zu)",
            static_cast<unsigned long long>(totalPieces), nRuns, mMaxPlanEntries);
        return kScatterRejected;
    }

    Ctx* ctx = tryAcquire();
    if (ctx == nullptr)
    {
        return kBusy; // every stream context busy — caller retries on its next tick
    }
    try
    {
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        // Exact-count pass over the EXPANDED pieces (same budgeted <= 64 KiB splitting as
        // submitCopy): the packed [srcs|dsts|sizes] pinned layout needs the total before the
        // first write. `remaining` counts the pre-split pieces not yet appended, so splitBudget
        // reserves one slot for each of them exactly like submitCopy's n - 1 - i.
        std::size_t nTotal = 0;
        std::size_t remaining = static_cast<std::size_t>(totalPieces);
        for (std::size_t i = 0; i < nRuns; ++i)
        {
            for (std::uint32_t p = 0; p < runs[i].count; ++p)
            {
                --remaining;
                nTotal += piecesFor(runs[i].pieceSize, splitBudget(nTotal, remaining, mMaxPlanEntries));
            }
        }
        TLLM_CHECK_WITH_INFO(nTotal <= mMaxPlanEntries,
            "BatchedCopyPool::submitScatterRuns: split plan (%zu) exceeds maxPlanEntries (%zu)", nTotal,
            mMaxPlanEntries);

        // Fill pass: expand each run to its pieces in place in the pinned plan buffer. The
        // destination address dstAddr + p*dstStride intentionally wraps modulo 2^64 like the
        // Python fallback's np.uint64 arithmetic (sources cannot wrap: bounds-checked above).
        auto const bufs = planBufs(ctx->hostPinned, nTotal);
        std::size_t idx = 0;
        remaining = static_cast<std::size_t>(totalPieces);
        for (std::size_t i = 0; i < nRuns; ++i)
        {
            ScatterRunWire const& r = runs[i];
            for (std::uint32_t p = 0; p < r.count; ++p)
            {
                --remaining;
                appendSplitInto(bufs, idx, regionBase + r.bounceOffset + static_cast<std::uint64_t>(p) * r.bounceStride,
                    r.dstAddr + static_cast<std::uint64_t>(p) * r.dstStride, r.pieceSize,
                    splitBudget(idx, remaining, mMaxPlanEntries));
            }
        }

        std::size_t const b64 = nTotal * sizeof(std::uint64_t);
        auto* base = static_cast<std::uint8_t*>(ctx->hostPinnedDev);
        auto const* dSrcs = reinterpret_cast<std::uint64_t const*>(base);
        auto const* dDsts = reinterpret_cast<std::uint64_t const*>(base + b64);
        auto const* dSizes = reinterpret_cast<std::uint32_t const*>(base + 2 * b64);
        TLLM_CUDA_CHECK(launchBatchedCopy(dSrcs, dDsts, dSizes, static_cast<std::uint32_t>(nTotal), ctx->stream));
        TLLM_CUDA_CHECK(cudaEventRecord(ctx->event, ctx->stream));
        return static_cast<std::int64_t>(mPoller.registerEvent(ctx->event, [this, ctx] { release(ctx); }));
    }
    catch (...)
    {
        // Same unwind contract as submitCopy: the launch may have succeeded before a later step
        // threw — drain the stream so the recycled context cannot have a live kernel still
        // reading its pinned plan buffer. Warn-only: we are already unwinding.
        TLLM_CUDA_CHECK_WARN(cudaStreamSynchronize(ctx->stream));
        release(ctx);
        throw;
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
