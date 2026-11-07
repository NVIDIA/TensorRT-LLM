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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// Bulk host passes on the submit path (admission shape scan, transfer-plan build) are split into
/// segments from this many items on; below it, one pass on the caller is cheaper than waking helpers.
inline constexpr std::size_t kBulkSegmentItems = 8192;
inline constexpr std::size_t kMinBulkSegments = 2;
inline constexpr std::size_t kMaxBulkSegments = 16;

[[nodiscard]] inline std::size_t bulkSegmentCount(std::size_t itemCount) noexcept
{
    if (itemCount < kBulkSegmentItems)
    {
        return 1;
    }
    return std::clamp(itemCount / kBulkSegmentItems, kMinBulkSegments, kMaxBulkSegments);
}

/// First item of `segment` when `itemCount` items are cut into `segmentCount` near-equal contiguous segments.
[[nodiscard]] inline std::size_t segmentBegin(
    std::size_t itemCount, std::size_t segmentCount, std::size_t segment) noexcept
{
    return itemCount * segment / segmentCount;
}

/// Long-lived host threads that help the calling thread run the segments of a bulk pass.
class HostWorkerPool
{
public:
    static constexpr std::size_t kCpusPerDefaultWorker = 4;
    static constexpr std::size_t kMaxDefaultWorkers = 16;

    /// Starts up to `threads` workers and keeps as many as the system allows (0 is valid).
    explicit HostWorkerPool(std::size_t threads)
    {
        mWorkers.reserve(threads);
        for (std::size_t i = 0; i < threads; ++i)
        {
            try
            {
                mWorkers.emplace_back([this] { workerLoop(); });
            }
            catch (std::exception const&)
            {
                break;
            }
        }
    }

    /// Precondition: no parallelFor is running.
    ~HostWorkerPool()
    {
        stopAndJoin();
    }

    HostWorkerPool(HostWorkerPool const&) = delete;
    HostWorkerPool& operator=(HostWorkerPool const&) = delete;

    [[nodiscard]] static std::size_t defaultThreadCount() noexcept
    {
        return std::clamp<std::size_t>(usableCpuCount() / kCpusPerDefaultWorker, 1, kMaxDefaultWorkers);
    }

    [[nodiscard]] std::size_t threadCount() const noexcept
    {
        return mWorkers.size();
    }

    /// Runs body(0) .. body(count - 1) on the calling thread plus up to count - 1 idle workers and
    /// returns once all of them ran. Rethrows the exception of the lowest index that threw.
    template <typename Body>
    void parallelFor(std::size_t count, Body&& body)
    {
        std::size_t const helpers = count > 1 ? std::min(count - 1, mWorkers.size()) : 0;
        std::shared_ptr<Job> const job = helpers > 0 ? tryMakeJob(count, body) : nullptr;
        if (job == nullptr)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                body(i);
            }
            return;
        }
        handOut(job, helpers);
        job->run();
        job->waitAll();
        job->rethrowLowestIndexError();
    }

private:
    /// One parallelFor call, shared with the workers it was handed to. A worker that dequeues it after
    /// every index was claimed runs nothing, so it never calls `body`, which refers to the caller's frame
    /// and is valid only until waitAll() returns; the shared_ptr keeps the counters alive for that worker.
    class Job
    {
    public:
        Job(std::size_t count, std::function<void(std::size_t)> body)
            : mCount(count)
            , mBody(std::move(body))
            , mExceptions(count)
        {
        }

        /// Claims and runs indices until none are left.
        void run() noexcept
        {
            for (std::size_t i = mNextIndex.fetch_add(1); i < mCount; i = mNextIndex.fetch_add(1))
            {
                try
                {
                    mBody(i);
                }
                catch (...)
                {
                    mExceptions[i] = std::current_exception();
                }
                recordFinished();
            }
        }

        void waitAll()
        {
            std::unique_lock<std::mutex> lock(mFinishedMutex);
            mAllFinished.wait(lock, [this] { return mFinishedCount.load() == mCount; });
        }

        void rethrowLowestIndexError() const
        {
            for (auto const& error : mExceptions)
            {
                if (error)
                {
                    std::rethrow_exception(error);
                }
            }
        }

    private:
        void recordFinished()
        {
            if (mFinishedCount.fetch_add(1) + 1 == mCount)
            {
                // Notify under the mutex: waitAll() reads the count under it, so the wake-up cannot fall
                // between its check and its wait.
                std::lock_guard<std::mutex> lock(mFinishedMutex);
                mAllFinished.notify_all();
            }
        }

        std::size_t const mCount;
        std::function<void(std::size_t)> const mBody;
        std::vector<std::exception_ptr> mExceptions; ///< mExceptions[i] is written only by the runner of index i.
        std::atomic<std::size_t> mNextIndex{0};
        std::atomic<std::size_t> mFinishedCount{0};
        std::mutex mFinishedMutex;
        std::condition_variable mAllFinished;
    };

    /// CPUs this process may run on: its affinity mask where available, else the hardware thread count.
    [[nodiscard]] static std::size_t usableCpuCount() noexcept
    {
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
        {
            return static_cast<std::size_t>(CPU_COUNT(&set));
        }
#endif
        return std::thread::hardware_concurrency();
    }

    template <typename Body>
    [[nodiscard]] static std::shared_ptr<Job> tryMakeJob(std::size_t count, Body& body) noexcept
    {
        try
        {
            return std::make_shared<Job>(count, [&body](std::size_t i) { body(i); });
        }
        catch (std::bad_alloc const&)
        {
            return nullptr;
        }
    }

    /// Queues `job` for up to `helpers` workers (fewer if memory runs out); the caller's own run() claims
    /// whatever the helpers do not.
    void handOut(std::shared_ptr<Job> const& job, std::size_t helpers) noexcept
    {
        std::size_t queued = 0;
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            try
            {
                for (; queued < helpers; ++queued)
                {
                    mQueue.push_back(job);
                }
            }
            catch (std::bad_alloc const&)
            {
            }
        }
        for (std::size_t i = 0; i < queued; ++i)
        {
            mWorkAvailable.notify_one();
        }
    }

    /// The next queued job, or nullptr once the pool is stopping and the queue is drained.
    [[nodiscard]] std::shared_ptr<Job> waitForJob()
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        mWorkAvailable.wait(lock, [this] { return mStopping || !mQueue.empty(); });
        if (mQueue.empty())
        {
            return nullptr;
        }
        std::shared_ptr<Job> job = std::move(mQueue.front());
        mQueue.pop_front();
        return job;
    }

    void workerLoop()
    {
        while (std::shared_ptr<Job> const job = waitForJob())
        {
            job->run();
        }
    }

    void stopAndJoin() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            mStopping = true;
        }
        mWorkAvailable.notify_all();
        for (auto& worker : mWorkers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    std::mutex mQueueMutex;
    std::condition_variable mWorkAvailable;
    std::deque<std::shared_ptr<Job>> mQueue;
    bool mStopping{false};
    std::vector<std::thread> mWorkers;
};

/// Runs body(segment, begin, end) over `segmentCount` near-equal contiguous segments of [0, itemCount):
/// on `pool`, or one after another on the calling thread when `pool` is null. Rethrows like parallelFor.
template <typename Body>
void forEachSegment(HostWorkerPool* pool, std::size_t itemCount, std::size_t segmentCount, Body&& body)
{
    auto const runSegment = [&](std::size_t segment) {
        body(segment, segmentBegin(itemCount, segmentCount, segment),
            segmentBegin(itemCount, segmentCount, segment + 1));
    };
    if (pool != nullptr)
    {
        pool->parallelFor(segmentCount, runSegment);
        return;
    }
    for (std::size_t segment = 0; segment < segmentCount; ++segment)
    {
        runSegment(segment);
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
