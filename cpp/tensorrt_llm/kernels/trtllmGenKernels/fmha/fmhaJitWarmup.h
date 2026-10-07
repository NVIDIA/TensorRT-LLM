/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/logger.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

// Host-only building blocks for the asynchronous TRTLLM-Gen FMHA JIT warmup:
// a single-thread task worker with an explicitly ordered shutdown, and the
// registry that tracks each warmup sweep from scheduling through foreground
// verification. Neither depends on CUDA, so both are covered by host unit
// tests; the CUDA-specific parts (relaxed stream-capture mode on the worker
// thread, device selection, the sweeps themselves) live in fmhaKernels.h.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// A single background worker thread. A single thread is intentional: the
// warmup compiles serialize on the FMHA export library's interface mutex
// anyway (its kernel cache is not thread-safe), so a pool would add no
// parallelism.
//
// Lifetime: the process-wide instance is leaked and shut down through a
// std::atexit handler registered when the worker is first created, i.e. from
// the first warmup forward. atexit handlers run before the destruction of any
// static object whose initialization completed earlier, so a sweep still in
// flight at process exit finishes while everything it reads (the export
// library's globals, the kernel objects, the mutexes, the install-path string)
// is still alive. The one caveat is a static that the sweep itself constructs
// lazily on the worker thread: such a static would be destroyed before the
// handler runs. fmhaKernels.h therefore touches those on the registering
// thread before the first task is queued.
class TllmGenFmhaAsyncWarmupWorker
{
public:
    using Task = std::function<void()>;

    // threadInit runs once on the worker thread before any task (e.g. to set
    // the CUDA stream-capture mode for that thread).
    explicit TllmGenFmhaAsyncWarmupWorker(std::function<void()> threadInit = {})
        : mThreadInit(std::move(threadInit))
    {
    }

    ~TllmGenFmhaAsyncWarmupWorker()
    {
        shutdown();
    }

    TllmGenFmhaAsyncWarmupWorker(TllmGenFmhaAsyncWarmupWorker const&) = delete;
    TllmGenFmhaAsyncWarmupWorker& operator=(TllmGenFmhaAsyncWarmupWorker const&) = delete;

    // Queues a task for the worker thread. Returns false if the task was NOT
    // queued, in which case the caller must run it inline: either the worker
    // thread could not be started, or the worker has already been shut down.
    bool enqueue(Task task)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStopRequested)
            {
                return false;
            }
            if (!mThreadStarted)
            {
                // Only mark started after the thread object is successfully constructed:
                // if std::thread throws (resource exhaustion), the task must not be
                // stranded behind a worker that never runs (drain() would deadlock).
                try
                {
                    mThread = std::thread([this]() { workerLoop(); });
                    mThreadStarted = true;
                }
                catch (std::exception const& e)
                {
                    TLLM_LOG_WARNING(
                        "TRTLLM-Gen FMHA async JIT warmup: worker thread failed to start (%s); compiling inline.",
                        e.what());
                    return false;
                }
            }
            mQueue.push_back(std::move(task));
            mNumPendingTasks.fetch_add(1, std::memory_order_acq_rel);
        }
        mQueueCv.notify_one();
        return true;
    }

    int numPendingTasks() const
    {
        return mNumPendingTasks.load(std::memory_order_acquire);
    }

    bool isShutDown() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mStopRequested;
    }

    // Blocks until every queued task has finished. Returns immediately when
    // nothing is pending, including after shutdown().
    void drain()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mDrainCv.wait(lock, [this]() { return mNumPendingTasks.load(std::memory_order_acquire) == 0; });
    }

    // Runs every queued task to completion, then stops and joins the worker
    // thread. Idempotent. Later enqueue() calls return false so callers fall
    // back to inline execution.
    void shutdown()
    {
        std::thread thread;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            if (mStopRequested)
            {
                return;
            }
            mStopRequested = true;
            thread = std::move(mThread);
        }
        mQueueCv.notify_all();
        if (thread.joinable())
        {
            thread.join();
        }
    }

private:
    void workerLoop()
    {
        if (mThreadInit)
        {
            mThreadInit();
        }
        while (true)
        {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mQueueCv.wait(lock, [this]() { return !mQueue.empty() || mStopRequested; });
                if (mQueue.empty())
                {
                    return;
                }
                task = std::move(mQueue.front());
                mQueue.pop_front();
            }
            try
            {
                task();
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_WARNING("TRTLLM-Gen FMHA async JIT warmup task failed: %s", e.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("TRTLLM-Gen FMHA async JIT warmup task failed with an unknown error.");
            }
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mNumPendingTasks.fetch_sub(1, std::memory_order_acq_rel);
            }
            mDrainCv.notify_all();
        }
    }

    std::function<void()> mThreadInit;
    mutable std::mutex mMutex;
    std::condition_variable mQueueCv;
    std::condition_variable mDrainCv;
    std::deque<Task> mQueue;
    std::atomic<int> mNumPendingTasks{0};
    std::thread mThread;
    bool mThreadStarted{false};
    bool mStopRequested{false};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Per-sweep warmup progress.
enum class JITWarmupState : int
{
    kScheduled = 0,  // Background sweep queued.
    kBackgroundDone, // Background sweep finished (or failed); foreground verify pass pending.
    kVerified,       // Sweep ran, or re-ran, on a foreground thread.
};

// Tracks every distinct warmup sweep requested in the process, keyed by a
// fingerprint of the sweep parameters. The registry is shared by every kernel
// object on every device, so each entry records the device whose context the
// compiled modules were loaded into and the kernel object that owns the
// sweep: a verify pass must re-run a sweep through the same kernel object, on
// the same device, that scheduled it.
//
// TParams is the sweep parameter struct (copied by value) and TInstance the
// owning kernel type; the registry only stores a pointer to the latter.
template <typename TParams, typename TInstance>
class TllmGenFmhaJitWarmupRegistry
{
public:
    struct PendingSweep
    {
        uint64_t mFingerprint;
        TParams mParams;
        int mDeviceId;
        TInstance* mInstance;
    };

    // Registers a sweep. Returns true if an identical sweep is already
    // registered (scheduled, finished, or verified), in which case the caller
    // must not run it again.
    bool tryRegister(
        uint64_t fingerprint, TParams const& params, JITWarmupState initialState, int deviceId, TInstance* instance)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mEntries.find(fingerprint) != mEntries.end())
        {
            return true;
        }
        mEntries.emplace(fingerprint, Entry{params, initialState, deviceId, instance});
        return false;
    }

    // Moves a scheduled sweep to kBackgroundDone. Called when the background
    // task finishes, whether or not it succeeded, so a failed sweep is re-run
    // by the verify pass instead of being deduplicated away.
    void markBackgroundDone(uint64_t fingerprint)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mEntries.find(fingerprint);
        if (it != mEntries.end() && it->second.mState == JITWarmupState::kScheduled)
        {
            it->second.mState = JITWarmupState::kBackgroundDone;
            mNumUnverified.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    // Number of sweeps in kBackgroundDone state (cheap gate for the verify pass).
    int numUnverified() const
    {
        return mNumUnverified.load(std::memory_order_acquire);
    }

    // Claims one kBackgroundDone sweep owned by (deviceId, instance), marking
    // it verified. The caller runs the returned sweep.
    std::optional<PendingSweep> claimUnverified(int deviceId, TInstance const* instance)
    {
        return claim([deviceId, instance](Entry const& entry)
            { return entry.mDeviceId == deviceId && entry.mInstance == instance; });
    }

    // Claims one kBackgroundDone sweep regardless of owner, for a caller that
    // can switch devices and dispatch to the owning instance (the pre-capture
    // drain-and-verify barrier).
    std::optional<PendingSweep> claimAnyUnverified()
    {
        return claim([](Entry const&) { return true; });
    }

    std::optional<JITWarmupState> state(uint64_t fingerprint) const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mEntries.find(fingerprint);
        if (it == mEntries.end())
        {
            return std::nullopt;
        }
        return it->second.mState;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mEntries.size();
    }

private:
    struct Entry
    {
        TParams mParams;
        JITWarmupState mState;
        int mDeviceId;
        TInstance* mInstance;
    };

    template <typename Pred>
    std::optional<PendingSweep> claim(Pred const& pred)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        for (auto& [fingerprint, entry] : mEntries)
        {
            if (entry.mState == JITWarmupState::kBackgroundDone && pred(entry))
            {
                entry.mState = JITWarmupState::kVerified;
                mNumUnverified.fetch_sub(1, std::memory_order_acq_rel);
                return PendingSweep{fingerprint, entry.mParams, entry.mDeviceId, entry.mInstance};
            }
        }
        return std::nullopt;
    }

    mutable std::mutex mMutex;
    std::unordered_map<uint64_t, Entry> mEntries;
    std::atomic<int> mNumUnverified{0};
};

} // namespace kernels

TRTLLM_NAMESPACE_END
