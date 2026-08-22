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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce_v2/CompletionPoller.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <exception>

namespace tensorrt_llm::executor::kv_cache::bounce_v2
{

CompletionPoller::CompletionPoller(std::uint32_t pollIntervalUs)
    : mPollIntervalUs(pollIntervalUs)
{
    mThread = std::thread(&CompletionPoller::pollLoop, this);
}

CompletionPoller::~CompletionPoller()
{
    shutdown();
    // Best-effort retry for handles whose release() failed on the poll thread.
    std::lock_guard<std::mutex> lk(mMu);
    for (auto& status : mRetired)
    {
        try
        {
            if (!status->release())
            {
                TLLM_LOG_WARNING("CompletionPoller: TransferStatus release failed again at destruction");
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("CompletionPoller: TransferStatus release threw at destruction: %s", e.what());
        }
    }
    mRetired.clear();
}

std::uint64_t CompletionPoller::registerEvent(
    cudaEvent_t event, std::function<void()> onTerminal, std::uint64_t* reserveChainId)
{
    std::uint64_t const id = mNextId.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mMu);
    if (mStop.load(std::memory_order_acquire))
    {
        if (reserveChainId != nullptr)
        {
            *reserveChainId = 0; // no reservation after shutdown; the event id resolves classically
        }
        // Late registration after shutdown: terminate immediately so the id still resolves. The
        // kernel behind the event may still be running — wait for it BEFORE onTerminal() recycles
        // the context, or its pinned plan buffer could be reused/freed under a live kernel.
        // Warn-only: a broken CUDA context must not turn this into a throw.
        TLLM_CUDA_CHECK_WARN(cudaEventSynchronize(event));
        // push_back (the only throwing statement) runs BEFORE onTerminal(): if it threw AFTER
        // onTerminal released the context, the caller's catch would release it a second time and
        // hand the same context to two submitters. Both statements are under mMu, so drain()
        // cannot observe the completion until unlock — ordering is invisible to callers.
        mDone.push_back(Completion{id, kKindEvent, 0});
        if (onTerminal)
        {
            onTerminal();
        }
        mCv.notify_all();
        return id;
    }
    EventEntry entry{id, event, std::move(onTerminal)};
    if (reserveChainId != nullptr)
    {
        // Atomic chain reservation: created in the SAME critical section as the registration, so
        // the poll sweep can never complete the event before the reservation exists (guaranteed
        // chain — the reserve race is structurally impossible).
        entry.chainId = mNextId.fetch_add(1, std::memory_order_relaxed);
        *reserveChainId = entry.chainId;
    }
    mEvents.push_back(std::move(entry));
    return id;
}

std::int64_t CompletionPoller::fulfillChain(std::uint64_t reservedId, ChainPoster poster)
{
    if (!poster)
    {
        return kFulfillDeclined; // caller bug; safe: declining never posts and never double-publishes
    }
    std::vector<PendingChain> ready;
    {
        std::lock_guard<std::mutex> lk(mMu);
        if (!mStop.load(std::memory_order_acquire))
        {
            for (auto& e : mEvents)
            {
                if (e.chainId == reservedId)
                {
                    if (e.chainPoster)
                    {
                        // Already fulfilled: that chain resolves the reserved id on its own.
                        return kFulfillDeclined;
                    }
                    e.chainPoster = std::move(poster);
                    return kFulfillArmed;
                }
            }
            auto it = std::find(mGatherDoneChains.begin(), mGatherDoneChains.end(), reservedId);
            if (it != mGatherDoneChains.end())
            {
                // Gather already done OK: post inline on THIS thread via the executeChains path
                // (outside mMu; mChainsInFlight keeps unregisterEvents' teardown wait correct).
                mGatherDoneChains.erase(it);
                ++mChainsInFlight;
                ready.push_back(PendingChain{reservedId, std::move(poster)});
            }
        }
        // mStop / unknown reservation: the gather failed ({reserved, kKindEvent, 0} is
        // published/pending) or the shutdown sweep terminated it ({reserved, kKindXfer, 0}) —
        // either way exactly one terminal row exists for the reserved id; decline below.
    }
    if (ready.empty())
    {
        return kFulfillDeclined;
    }
    executeChains(ready);
    return kFulfillPosted;
}

std::uint64_t CompletionPoller::registerXfer(std::unique_ptr<TransferStatus> status)
{
    std::uint64_t const id = mNextId.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mMu);
    if (mStop.load(std::memory_order_acquire))
    {
        XferEntry entry{id, std::move(status)};
        releaseXferLocked(entry);
        mDone.push_back(Completion{id, kKindXfer, 0});
        mCv.notify_all();
        return id;
    }
    mXfers.push_back(XferEntry{id, std::move(status)});
    return id;
}

std::vector<CompletionPoller::Completion> CompletionPoller::drain(int timeoutMs)
{
    std::unique_lock<std::mutex> lk(mMu);
    if (mDone.empty() && timeoutMs > 0 && !mStop.load(std::memory_order_acquire))
    {
        mCv.wait_for(lk, std::chrono::milliseconds(timeoutMs), [this] { return !mDone.empty() || mStop.load(); });
    }
    std::vector<Completion> out;
    out.swap(mDone);
    return out;
}

std::size_t CompletionPoller::unregisterEvents(std::vector<cudaEvent_t> const& events)
{
    // Taking mMu synchronizes with pollLoop's sweep (pollOnceLocked runs under it): once the erase
    // is done, no removed entry can be cudaEventQuery'd or have its onTerminal invoked. Erasing an
    // entry with an ARMED chain also drops its reserved id and poster — neither ever reports
    // (consistent with the entry's own id; waiters unblock via the reactor's timeout/stall paths).
    std::unique_lock<std::mutex> lk(mMu);
    auto const before = mEvents.size();
    mEvents.erase(std::remove_if(mEvents.begin(), mEvents.end(),
                      [&events](EventEntry const& e)
                      { return std::find(events.begin(), events.end(), e.event) != events.end(); }),
        mEvents.end());
    // Chain posters run OUTSIDE mMu (executeChains, by design — drain()/register* must not stall
    // behind postXferRequest), so the erase alone does not prove the poll thread is out of its
    // sweep: wait — bounded and GIL-free (the poll thread never takes the GIL; it only needs mMu,
    // which this wait releases) — for every in-flight poster to finish before returning, so the
    // caller may safely destroy anything a poster could still reference. Posters are short (one
    // RDMA post), so the bound only guards against a wedged agent; warn-and-proceed mirrors the
    // other bounded teardown paths.
    if (!mCv.wait_for(lk, std::chrono::seconds(2), [this] { return mChainsInFlight == 0; }))
    {
        TLLM_LOG_WARNING(
            "CompletionPoller: %zu chain poster(s) still in flight after unregisterEvents wait; proceeding",
            mChainsInFlight);
    }
    return before - mEvents.size();
}

void CompletionPoller::shutdown() noexcept
{
    // Serializes concurrent shutdown() callers (join + cleanup must run exactly once).
    std::lock_guard<std::mutex> shutdownLk(mShutdownMu);
    if (mShutdownDone)
    {
        return;
    }
    mShutdownDone = true;
    mStop.store(true, std::memory_order_release);
    if (mThread.joinable())
    {
        mThread.join();
    }
    // Terminate everything still pending so no drain()/wait path hangs. Runs after the poll thread
    // exited, so no concurrent poll sweep touches these lists.
    std::lock_guard<std::mutex> lk(mMu);
    for (auto& e : mEvents)
    {
        // The kernel behind a pending event may still be running — wait for it BEFORE onTerminal()
        // recycles the context (BatchedCopyPool returns it to the free list, after which its pinned
        // plan buffer may be reused by a new submit or freed by the pool's destructor). Warn-only:
        // a broken CUDA context must not wedge shutdown.
        TLLM_CUDA_CHECK_WARN(cudaEventSynchronize(e.event));
        if (e.onTerminal)
        {
            e.onTerminal();
        }
        if (e.chainId != 0)
        {
            // Armed chain, write never posted: terminate the RESERVED id (that is the only id the
            // caller still routes; kKindXfer because the chunk's classic event route was dropped
            // when the arm succeeded).
            mDone.push_back(Completion{e.chainId, kKindXfer, 0});
        }
        else
        {
            mDone.push_back(Completion{e.id, kKindEvent, 0});
        }
    }
    mEvents.clear();
    for (auto const id : mGatherDoneChains)
    {
        // Reserved chain whose gather completed but whose fulfill never arrived: terminate the
        // reserved id like an armed chain (gather OK, write never posted -> kKindXfer).
        mDone.push_back(Completion{id, kKindXfer, 0});
    }
    mGatherDoneChains.clear();
    for (auto& x : mXfers)
    {
        releaseXferLocked(x);
        mDone.push_back(Completion{x.id, kKindXfer, 0});
    }
    mXfers.clear();
    mCv.notify_all();
}

void CompletionPoller::releaseXferLocked(XferEntry& entry)
{
    if (entry.status == nullptr)
    {
        return;
    }
    bool released = false;
    try
    {
        released = entry.status->release();
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("CompletionPoller: TransferStatus release threw: %s", e.what());
    }
    if (!released)
    {
        // Keep the object so the destructor can retry (a busy backend may release later).
        mRetired.push_back(std::move(entry.status));
    }
    entry.status.reset();
}

bool CompletionPoller::pollOnceLocked(std::vector<PendingChain>& chainsOut)
{
    bool published = false;

    // CUDA events: cudaSuccess => done, cudaErrorNotReady => keep polling, anything else => failure.
    for (auto it = mEvents.begin(); it != mEvents.end();)
    {
        cudaError_t const st = cudaEventQuery(it->event);
        if (st == cudaErrorNotReady)
        {
            ++it;
            continue;
        }
        if (st != cudaSuccess)
        {
            TLLM_LOG_WARNING("CompletionPoller: cudaEventQuery failed: %s", cudaGetErrorString(st));
        }
        if (it->onTerminal)
        {
            it->onTerminal();
        }
        if (it->chainId != 0)
        {
            if (st == cudaSuccess)
            {
                if (it->chainPoster)
                {
                    // Successful gather with an armed chain: publish NOTHING for the event — the
                    // chunk's one completion is the reserved xfer id, resolved after the post. The
                    // in-flight count (under mMu, same critical section as the erase) lets
                    // unregisterEvents wait out the poster that will run outside mMu.
                    chainsOut.push_back(PendingChain{it->chainId, std::move(it->chainPoster)});
                    ++mChainsInFlight;
                }
                else
                {
                    // RESERVED but not yet fulfilled (the credit is still in flight): remember
                    // gather-done; fulfillChain posts inline later. Publish NOTHING.
                    mGatherDoneChains.push_back(it->chainId);
                }
            }
            else
            {
                // Gather failed: the write is never posted. kKindEvent tells the caller the
                // chain died at the gather stage (FAIL_GATHER, not FAIL_WRITE).
                mDone.push_back(Completion{it->chainId, kKindEvent, 0});
                published = true;
            }
        }
        else
        {
            mDone.push_back(Completion{it->id, kKindEvent, st == cudaSuccess ? 1 : 0});
            published = true;
        }
        it = mEvents.erase(it);
    }

    // RDMA transfers: wait(0) is a non-blocking status probe.
    for (auto it = mXfers.begin(); it != mXfers.end();)
    {
        TransferState state = TransferState::kFAILURE;
        try
        {
            state = it->status->wait(0);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("CompletionPoller: TransferStatus wait threw: %s", e.what());
        }
        if (state == TransferState::kIN_PROGRESS)
        {
            ++it;
            continue;
        }
        releaseXferLocked(*it);
        mDone.push_back(Completion{it->id, kKindXfer, state == TransferState::kSUCCESS ? 1 : 0});
        published = true;
        it = mXfers.erase(it);
    }

    return published;
}

void CompletionPoller::executeChains(std::vector<PendingChain>& chains)
{
    for (auto& chain : chains)
    {
        std::unique_ptr<TransferStatus> status;
        try
        {
            status = chain.poster();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING("CompletionPoller: armed chain post threw: %s", e.what());
        }
        std::lock_guard<std::mutex> lk(mMu);
        // This chain's outside-mMu section is over; wake any unregisterEvents caller waiting for
        // in-flight posters to drain. Decremented on EVERY path (success, failed post, shutdown).
        --mChainsInFlight;
        if (status == nullptr)
        {
            mDone.push_back(Completion{chain.id, kKindXfer, 0});
            mCv.notify_all();
            continue;
        }
        if (mStop.load(std::memory_order_acquire))
        {
            // Raced shutdown(): its sweep already drained mXfers and will never poll this handle —
            // release it here and terminate the reserved id so no drain() caller waits forever.
            XferEntry entry{chain.id, std::move(status)};
            releaseXferLocked(entry);
            mDone.push_back(Completion{chain.id, kKindXfer, 0});
            mCv.notify_all();
            continue;
        }
        mXfers.push_back(XferEntry{chain.id, std::move(status)});
        mCv.notify_all();
    }
}

void CompletionPoller::pollLoop()
{
    while (!mStop.load(std::memory_order_acquire))
    {
        std::vector<PendingChain> chains;
        {
            std::lock_guard<std::mutex> lk(mMu);
            if (pollOnceLocked(chains))
            {
                mCv.notify_all();
            }
        }
        if (!chains.empty())
        {
            executeChains(chains);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(mPollIntervalUs));
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
