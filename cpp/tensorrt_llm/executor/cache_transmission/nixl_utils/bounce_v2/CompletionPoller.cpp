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

std::uint64_t CompletionPoller::registerEvent(cudaEvent_t event, std::function<void()> onTerminal)
{
    std::uint64_t const id = mNextId.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mMu);
    if (mStop.load(std::memory_order_acquire))
    {
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
    mEvents.push_back(EventEntry{id, event, std::move(onTerminal)});
    return id;
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
    // Taking mMu synchronizes with pollLoop's sweep (pollOnceLocked runs under it): once this
    // returns, no removed entry can be cudaEventQuery'd or have its onTerminal invoked.
    std::lock_guard<std::mutex> lk(mMu);
    auto const before = mEvents.size();
    mEvents.erase(std::remove_if(mEvents.begin(), mEvents.end(),
                      [&events](EventEntry const& e)
                      { return std::find(events.begin(), events.end(), e.event) != events.end(); }),
        mEvents.end());
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
        mDone.push_back(Completion{e.id, kKindEvent, 0});
    }
    mEvents.clear();
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

bool CompletionPoller::pollOnceLocked()
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
        mDone.push_back(Completion{it->id, kKindEvent, st == cudaSuccess ? 1 : 0});
        published = true;
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

void CompletionPoller::pollLoop()
{
    while (!mStop.load(std::memory_order_acquire))
    {
        {
            std::lock_guard<std::mutex> lk(mMu);
            if (pollOnceLocked())
            {
                mCv.notify_all();
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(mPollIntervalUs));
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce_v2
