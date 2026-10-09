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
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// Bulk O(n) host work on the submit path (the admission shape scan, the transfer-plan build) is
/// split into segments once there are at least this many items; below it one pass beats the hand-off.
inline constexpr std::size_t kBulkSegmentItems = 8192;
/// Upper bound on the number of segments of one bulk pass.
inline constexpr std::size_t kMaxBulkSegments = 16;

/// Segment count for a bulk pass over `n` items: 1 below kBulkSegmentItems, else about one segment
/// per kBulkSegmentItems items, clamped to [2, kMaxBulkSegments] (2 from the threshold up, so the
/// threshold itself still switches the pass to parallel).
[[nodiscard]] inline std::size_t bulkSegmentCount(std::size_t n) noexcept
{
    return n < kBulkSegmentItems ? 1 : std::clamp<std::size_t>(n / kBulkSegmentItems, 2, kMaxBulkSegments);
}

// ============================================================================
// HostWorkerPool — persistent host threads for bulk CPU work on the submit path
// ----------------------------------------------------------------------------
// Role
//   Runs the segments of a bulk pass (see bulkSegmentCount) on a few long-lived threads, so a pass
//   pays a condition-variable wake-up per helper instead of a thread start. Workers park on a
//   condition variable while idle.
//
// parallelFor(count, body)
//   Runs body(0) .. body(count - 1) and returns once every index has run. The CALLING thread takes
//   part: it and up to count - 1 woken workers claim indices from a shared counter. So a busy or
//   thread-less pool degrades to running the indices on the caller (never a deadlock), and
//   concurrent callers simply share the workers. If bodies throw, every index still runs and the
//   exception of the LOWEST index that threw is rethrown — the error a sequential loop over the
//   same indices would have hit first.
//
// Lifetime
//   The destructor wakes and joins the workers; no parallelFor may be running then. Thread
//   creation failure (resource limits) is not an error: the pool keeps the workers it got.
// ============================================================================
class HostWorkerPool
{
public:
    /// Start up to `threads` workers (0 is valid: parallelFor then runs everything on the caller).
    explicit HostWorkerPool(std::size_t threads)
    {
        mWorkers.reserve(threads);
        for (std::size_t i = 0; i < threads; ++i)
        {
            try
            {
                mWorkers.emplace_back([this] { workerLoop(); });
            }
            catch (std::system_error const&)
            {
                break; // out of threads: keep the workers already started
            }
            catch (...)
            {
                stopAndJoin(); // the destructor will not run for a throwing constructor
                throw;
            }
        }
    }

    ~HostWorkerPool()
    {
        stopAndJoin();
    }

    HostWorkerPool(HostWorkerPool const&) = delete;
    HostWorkerPool& operator=(HostWorkerPool const&) = delete;

    /// Worker count for one pool: a quarter of the CPUs this process may run on (its affinity mask,
    /// else the hardware thread count), between 1 and 16.
    [[nodiscard]] static std::size_t defaultThreadCount() noexcept
    {
        std::size_t cpus = std::thread::hardware_concurrency();
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0 && CPU_COUNT(&set) > 0)
        {
            cpus = static_cast<std::size_t>(CPU_COUNT(&set));
        }
#endif
        return std::clamp<std::size_t>(cpus / 4, 1, 16);
    }

    [[nodiscard]] std::size_t threadCount() const noexcept
    {
        return mWorkers.size();
    }

    template <typename Body>
    void parallelFor(std::size_t count, Body&& body)
    {
        if (count == 0)
        {
            return;
        }
        auto job = std::make_shared<Job>(count, [&body](std::size_t i) { body(i); });
        std::size_t const helpers = std::min(count - 1, mWorkers.size());
        if (helpers > 0)
        {
            {
                std::lock_guard<std::mutex> lk(mMu);
                try
                {
                    for (std::size_t h = 0; h < helpers; ++h)
                    {
                        mQueue.push_back(job);
                    }
                }
                catch (...)
                {
                    // No worker can have taken an entry yet (we hold mMu). Exhaust the index counter
                    // so the entries already queued never run `body` after this frame unwinds.
                    job->next.store(count);
                    throw;
                }
            }
            for (std::size_t h = 0; h < helpers; ++h)
            {
                mCv.notify_one();
            }
        }
        job->run();
        job->waitAll();
        for (auto const& error : job->errors)
        {
            if (error)
            {
                std::rethrow_exception(error);
            }
        }
    }

private:
    // One parallelFor call, shared with the workers that were handed it. A worker that dequeues it
    // after every index was claimed finds nothing to do and never calls `body`, which references the
    // caller's stack and is only valid until waitAll() returns — that is, until every CLAIMED index
    // finished. The shared_ptr keeps the counters alive for such late workers.
    struct Job
    {
        Job(std::size_t count, std::function<void(std::size_t)> body)
            : count(count)
            , body(std::move(body))
            , errors(count)
        {
        }

        void run() noexcept
        {
            for (std::size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
            {
                try
                {
                    body(i);
                }
                catch (...)
                {
                    errors[i] = std::current_exception();
                }
                if (done.fetch_add(1) + 1 == count)
                {
                    // Lock before notifying: the waiter checks `done` under `mu`, so the wake-up
                    // cannot slip in between its check and its wait.
                    std::lock_guard<std::mutex> lk(mu);
                    cv.notify_all();
                }
            }
        }

        void waitAll()
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this] { return done.load() == count; });
        }

        std::size_t const count;
        std::function<void(std::size_t)> const body;
        std::vector<std::exception_ptr> errors; // errors[i] written only by index i's runner, read after waitAll
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> done{0};
        std::mutex mu;
        std::condition_variable cv;
    };

    void stopAndJoin() noexcept
    {
        {
            std::lock_guard<std::mutex> lk(mMu);
            mStop = true;
        }
        mCv.notify_all();
        for (auto& worker : mWorkers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    void workerLoop()
    {
        while (true)
        {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(mMu);
                mCv.wait(lk, [this] { return mStop || !mQueue.empty(); });
                if (mQueue.empty())
                {
                    return; // stopping
                }
                job = std::move(mQueue.front());
                mQueue.pop_front();
            }
            job->run();
        }
    }

    std::mutex mMu;
    std::condition_variable mCv;
    std::deque<std::shared_ptr<Job>> mQueue;
    bool mStop{false};
    std::vector<std::thread> mWorkers; // joined (stopAndJoin) before the members above are destroyed
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
