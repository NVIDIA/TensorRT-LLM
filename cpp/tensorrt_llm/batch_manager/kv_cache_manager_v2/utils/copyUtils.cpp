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

#include "kv_cache_manager_v2/utils/copyUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <pthread.h>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename Func>
bool loopedReadWrite(Func&& func, ssize_t size) noexcept
{
    ssize_t count = 0;
    while (count < size)
    {
        ssize_t bytes = func(count);
        if (bytes <= 0)
        {
            if (errno == EINTR)
            {
                continue; // Retry on interrupt
            }
            int const err = errno;
            TLLM_LOG_ERROR("Disk read/write failed: %s\n", strerror(err));
            errno = err;
            return false;
        }
        count += bytes;
    }
    assert(count == size);
    return true;
}

bool writeAll(int fd, ssize_t pos, void const* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pwrite(fd, static_cast<std::byte const*>(data) + finished, size - finished, pos + finished); },
        size);
}

bool readAll(int fd, ssize_t pos, void* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pread(fd, static_cast<std::byte*>(data) + finished, size - finished, pos + finished); },
        size);
}

template <typename DstAddr, typename SrcAddr>
struct UserData
{
    std::vector<Task<DstAddr, SrcAddr>> tasks;
    ssize_t numBytes;
};

// Disk pages whose last write failed, keyed by (fd, page offset). Leaked on purpose: I/O workers may record a
// failure during static destruction.
struct FailedDiskWrites
{
    std::mutex mutex;
    std::set<std::pair<int, ssize_t>> pages;
    std::atomic<size_t> size{0};
};

FailedDiskWrites& failedDiskWrites()
{
    static auto* const instance = new FailedDiskWrites;
    return *instance;
}

void recordFailedDiskWrite(int fd, ssize_t pos) noexcept
{
    auto& failed = failedDiskWrites();
    try
    {
        std::lock_guard<std::mutex> lock(failed.mutex);
        failed.pages.emplace(fd, pos);
        failed.size.store(failed.pages.size(), std::memory_order_release);
    }
    catch (...)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] could not record a failed disk write; its page may be reused\n");
    }
}

// A new write to a page supersedes the failure of the previous one.
void forgetFailedDiskWrites(std::vector<Task<DiskAddress, MemAddress>> const& tasks) noexcept
{
    auto& failed = failedDiskWrites();
    if (failed.size.load(std::memory_order_acquire) == 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(failed.mutex);
    for (auto const& t : tasks)
    {
        failed.pages.erase({t.dst.fd, t.dst.pos});
    }
    failed.size.store(failed.pages.size(), std::memory_order_release);
}

std::atomic<uint64_t> gInjectedWriteFailures{0};

bool takeInjectedWriteFailure() noexcept
{
    uint64_t remaining = gInjectedWriteFailures.load(std::memory_order_relaxed);
    while (remaining > 0)
    {
        if (gInjectedWriteFailures.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed))
        {
            return true;
        }
    }
    return false;
}

// Write-behind (async disk I/O only): a host-to-disk page write is complete once its bytes are copied into staging
// memory, and an I/O worker writes the file afterwards. The host slot the page came from is then free at memory speed
// instead of after the disk write, which matters because a GPU -> host eviction into that slot (and so the restore
// that needs the GPU slot) waits for it. Until a page's flush has reached the file, every read of that disk page is
// served from staging. A newer write of the same page replaces the pending one, and a flush that has been superseded
// writes nothing. A failed flush keeps its page in staging, so the disk copy stays readable and no failure is
// recorded for it. Staging is bounded; a write that does not fit runs as before.
namespace
{

using DiskPageKey = std::pair<int, ssize_t>; // (fd, page offset)

struct StagedDiskPage;

class DiskWriteBehind
{
public:
    void configure(size_t capacity) noexcept
    {
        size_t const previous = mCapacity.exchange(capacity, std::memory_order_acq_rel);
        if (previous != capacity)
        {
            TLLM_LOG_INFO("[kvCacheManagerV2Utils] disk write-behind staging: %zu MiB", capacity >> 20);
        }
    }

    [[nodiscard]] size_t capacity() const noexcept
    {
        return mCapacity.load(std::memory_order_acquire);
    }

    //! Reserves \p bytes of staging for a write about to be issued; false when they do not fit.
    bool reserve(size_t bytes) noexcept
    {
        size_t const cap = capacity();
        size_t used = mReserved.load(std::memory_order_relaxed);
        do
        {
            if (cap == 0 || bytes > cap || used > cap - bytes)
            {
                return false;
            }
        } while (!mReserved.compare_exchange_weak(used, used + bytes, std::memory_order_relaxed));
        return true;
    }

    void unreserve(size_t bytes) noexcept
    {
        mReserved.fetch_sub(bytes, std::memory_order_relaxed);
    }

    [[nodiscard]] size_t reservedBytes() const noexcept
    {
        return mReserved.load(std::memory_order_relaxed);
    }

    //! A buffer of \p bytes from the free list, or a new one. Its bytes are already reserved.
    std::unique_ptr<std::byte[]> takeBuffer(size_t bytes)
    {
        {
            std::lock_guard<std::mutex> lock(mBufferMutex);
            auto it = mFreeBuffers.find(bytes);
            if (it != mFreeBuffers.end() && !it->second.empty())
            {
                auto buffer = std::move(it->second.back());
                it->second.pop_back();
                mFreeBytes -= bytes;
                return buffer;
            }
            // Free buffers of other sizes count against the bound too: drop them first.
            while (mFreeBytes > 0 && mFreeBytes + reservedBytes() > capacity())
            {
                for (auto& [size, list] : mFreeBuffers)
                {
                    if (!list.empty())
                    {
                        list.pop_back();
                        mFreeBytes -= size;
                        break;
                    }
                }
            }
        }
        return std::unique_ptr<std::byte[]>(new std::byte[bytes]);
    }

    void returnBuffer(std::unique_ptr<std::byte[]> buffer, size_t bytes) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(mBufferMutex);
            if (mFreeBytes + reservedBytes() + bytes <= capacity())
            {
                mFreeBuffers[bytes].push_back(std::move(buffer));
                mFreeBytes += bytes;
            }
        }
        catch (...)
        {
            // Out of memory for the free list: the buffer is released below.
        }
    }

    //! The pending page at \p key, or null.
    std::shared_ptr<StagedDiskPage> find(DiskPageKey key)
    {
        if (mNumPages.load(std::memory_order_acquire) == 0)
        {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mPages.find(key);
        return it == mPages.end() ? nullptr : it->second;
    }

    //! Makes \p page the pending content of \p key, replacing an older one.
    void publish(DiskPageKey key, std::shared_ptr<StagedDiskPage> page)
    {
        std::shared_ptr<StagedDiskPage> replaced; // destroyed outside the lock
        std::lock_guard<std::mutex> lock(mMutex);
        auto& slot = mPages[key];
        replaced = std::move(slot);
        slot = std::move(page);
        mNumPages.store(mPages.size(), std::memory_order_release);
    }

    enum class FlushStart
    {
        kStart,      // write it now
        kSuperseded, // a newer write replaced it (or it was dropped): nothing to write
        kBusy,       // an older flush of the same page is still writing: try again later
    };

    FlushStart beginFlush(DiskPageKey key, StagedDiskPage const* page)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto it = mPages.find(key);
        if (it == mPages.end() || it->second.get() != page)
        {
            return FlushStart::kSuperseded;
        }
        if (!mFlushing.insert(key).second)
        {
            return FlushStart::kBusy;
        }
        return FlushStart::kStart;
    }

    //! The flush of \p page has ended. Once written, the file holds the page, so it leaves staging unless a newer write
    //! replaced it meanwhile. A failed flush keeps it.
    void endFlush(DiskPageKey key, StagedDiskPage const* page, bool written)
    {
        std::shared_ptr<StagedDiskPage> released; // destroyed outside the lock
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mFlushing.erase(key);
            auto it = mPages.find(key);
            if (written && it != mPages.end() && it->second.get() == page)
            {
                released = std::move(it->second);
                mPages.erase(it);
                mNumPages.store(mPages.size(), std::memory_order_release);
            }
        }
        mFlushedCv.notify_all();
    }

    //! Drops the pending pages of \p fd at offsets in [\p minPos, \p maxPos], after any flush of them that is writing:
    //! their file is about to be closed or shrunk, or (one page) overwritten by a copy that bypasses staging.
    void drop(int fd, ssize_t minPos, ssize_t maxPos = std::numeric_limits<ssize_t>::max())
    {
        std::vector<std::shared_ptr<StagedDiskPage>> released; // destroyed outside the lock
        std::unique_lock<std::mutex> lock(mMutex);
        auto const inRange = [fd, minPos, maxPos](DiskPageKey const& key)
        { return key.first == fd && key.second >= minPos && key.second <= maxPos; };
        mFlushedCv.wait(lock, [&]() { return std::none_of(mFlushing.begin(), mFlushing.end(), inRange); });
        for (auto it = mPages.begin(); it != mPages.end();)
        {
            if (inRange(it->first))
            {
                released.push_back(std::move(it->second));
                it = mPages.erase(it);
            }
            else
            {
                ++it;
            }
        }
        mNumPages.store(mPages.size(), std::memory_order_release);
        lock.unlock();
    }

    [[nodiscard]] size_t numPages() const noexcept
    {
        return mNumPages.load(std::memory_order_acquire);
    }

private:
    std::atomic<size_t> mCapacity{0};
    std::atomic<size_t> mReserved{0}; // reserved for issued writes, plus staged pages not yet released
    std::mutex mMutex;
    std::condition_variable mFlushedCv;
    std::map<DiskPageKey, std::shared_ptr<StagedDiskPage>> mPages;
    std::set<DiskPageKey> mFlushing;
    std::atomic<size_t> mNumPages{0};
    std::mutex mBufferMutex;
    std::map<size_t, std::vector<std::unique_ptr<std::byte[]>>> mFreeBuffers;
    size_t mFreeBytes = 0;
};

// Leaked on purpose, like the I/O pool: detached workers may use it during static destruction.
DiskWriteBehind& diskWriteBehind()
{
    static auto* const instance = new DiskWriteBehind;
    return *instance;
}

struct StagedDiskPage
{
    StagedDiskPage(std::unique_ptr<std::byte[]> data, size_t bytes)
        : data(std::move(data))
        , bytes(bytes)
    {
    }

    StagedDiskPage(StagedDiskPage const&) = delete;
    StagedDiskPage& operator=(StagedDiskPage const&) = delete;

    ~StagedDiskPage()
    {
        auto& wb = diskWriteBehind();
        wb.unreserve(bytes);
        wb.returnBuffer(std::move(data), bytes);
    }

    std::unique_ptr<std::byte[]> data;
    size_t bytes;
};

//! Reads [pos + offset, + length) of a disk page: from staging while a write of the page is pending, else the file.
bool readDiskPage(int fd, ssize_t pos, ssize_t offset, void* dst, ssize_t length) noexcept
{
    try
    {
        if (auto const staged = diskWriteBehind().find({fd, pos});
            staged && static_cast<size_t>(offset + length) <= staged->bytes)
        {
            std::memcpy(dst, staged->data.get() + offset, static_cast<size_t>(length));
            return true;
        }
    }
    catch (std::exception const&)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] write-behind lookup failed; reading the file\n");
    }
    return readAll(fd, pos + offset, dst, length);
}

} // namespace

// Host copy threads. CUDA runs the host functions of a context one at a time (in practice on one driver thread), so a
// host-side copy callback that walks its page tasks one after another holds up every later host function on every
// stream, and with them the copies queued behind those host functions. With hostCopyThreads > 1, the page tasks of one
// callback (host <-> host, disk <-> disk, and host <-> disk when no async disk I/O pool runs) are spread over a
// process-wide pool. The callback thread takes tasks too and returns only once every task has finished, so stream order
// and the events recorded after the callback keep their meaning. Copies are unchanged.
namespace
{

// A thread costs more than it saves on a small batch: a callback uses one thread per this many bytes at most.
constexpr size_t kMinHostCopyBytesPerThread = size_t{4} << 20;

// Testing only: tasks run by pool workers rather than by callback threads.
std::atomic<uint64_t> gHostCopyHelperTasks{0};

struct HostCopyJob
{
    using Fn = bool (*)(void const* context, size_t task) noexcept;

    Fn fn = nullptr;
    void const* context = nullptr; // valid until every task has finished
    size_t numTasks = 0;
    size_t maxHelpers = 0;
    size_t joined = 0; // pool threads that took this job; guarded by the pool mutex
    std::atomic<size_t> next{0};
    std::atomic<size_t> finished{0};
    std::atomic<bool> success{true};

    //! Runs unclaimed tasks until none is left. Returns how many it ran.
    size_t work() noexcept
    {
        size_t ran = 0;
        for (size_t i = next.fetch_add(1, std::memory_order_relaxed); i < numTasks;
             i = next.fetch_add(1, std::memory_order_relaxed))
        {
            if (!fn(context, i))
            {
                success.store(false, std::memory_order_relaxed);
            }
            ++ran;
            // Release: the waiter that sees the last count also sees every task's bytes.
            if (finished.fetch_add(1, std::memory_order_acq_rel) + 1 == numTasks)
            {
                finished.notify_all();
            }
        }
        return ran;
    }

    void waitFinished() const noexcept
    {
        for (size_t done = finished.load(std::memory_order_acquire); done != numTasks;
             done = finished.load(std::memory_order_acquire))
        {
            finished.wait(done, std::memory_order_acquire);
        }
    }
};

// Workers are detached and the pool is leaked, like the disk I/O pool: a callback that runs during process exit still
// finds it, and no worker is ever joined.
class HostCopyPool
{
public:
    //! Starts workers until \p numWorkers run. Returns how many run.
    size_t grow(size_t numWorkers) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        try
        {
            while (mNumWorkers < numWorkers)
            {
                std::thread([this]() noexcept { run(); }).detach();
                ++mNumWorkers;
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] started %zu of %zu host copy threads: %s", mNumWorkers, numWorkers, e.what());
        }
        return mNumWorkers;
    }

    //! Queues \p job for up to job->maxHelpers workers. False if it could not be queued: the caller then runs it alone.
    bool post(std::shared_ptr<HostCopyJob> const& job) noexcept
    {
        try
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mJobs.push_back(job);
        }
        catch (std::exception const&)
        {
            return false;
        }
        mCv.notify_all();
        return true;
    }

    //! Takes \p job off the queue if no worker took its last helper place yet: its tasks are all claimed by now.
    void retract(HostCopyJob const* job) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        auto const it
            = std::find_if(mJobs.begin(), mJobs.end(), [job](auto const& queued) { return queued.get() == job; });
        if (it != mJobs.end())
        {
            mJobs.erase(it);
        }
    }

private:
    void run() noexcept
    {
        for (;;)
        {
            std::shared_ptr<HostCopyJob> job;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCv.wait(lock, [this] { return !mJobs.empty(); });
                job = mJobs.front();
                if (++job->joined >= job->maxHelpers)
                {
                    mJobs.pop_front();
                }
            }
            // The shared_ptr keeps next / finished alive after the caller returned; fn and context are only used for
            // a claimed task, which the caller waits for.
            size_t const ran = job->work();
            gHostCopyHelperTasks.fetch_add(ran, std::memory_order_relaxed);
        }
    }

    std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<std::shared_ptr<HostCopyJob>> mJobs;
    size_t mNumWorkers = 0;
};

// Threads a callback may use, the callback thread included: 1 runs every task on the callback thread.
std::atomic<size_t> gHostCopyThreads{1};
std::atomic<HostCopyPool*> gHostCopyPool{nullptr};
// The first value > 1 configureHostCopyThreads was given (0: none yet).
std::atomic<size_t> gConfiguredHostCopyThreads{0};

void forgetHostCopyPoolInChild() noexcept
{
    // A forked child has none of the workers, and the parent's pool mutex may have been held by one of them at the
    // fork. Leave that pool alone and run every task on the callback thread; a later configure call starts a new one.
    gHostCopyPool.store(nullptr, std::memory_order_relaxed);
    gHostCopyThreads.store(1, std::memory_order_relaxed);
    gConfiguredHostCopyThreads.store(0, std::memory_order_relaxed);
}

//! Sets the width of every later callback to \p numThreads (callback thread included), starting workers as needed.
//! Returns the width set, which is smaller when not every worker started.
size_t setHostCopyThreads(size_t numThreads) noexcept
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (numThreads <= 1)
    {
        gHostCopyThreads.store(1, std::memory_order_release);
        return 1;
    }
    auto* pool = gHostCopyPool.load(std::memory_order_acquire);
    if (pool == nullptr)
    {
        pool = new (std::nothrow) HostCopyPool;
        if (pool == nullptr)
        {
            return gHostCopyThreads.load();
        }
        static bool const kAtFork = ::pthread_atfork(nullptr, nullptr, forgetHostCopyPoolInChild) == 0;
        if (!kAtFork)
        {
            TLLM_LOG_WARNING("[kvCacheManagerV2Utils] pthread_atfork failed; a forked child may hang in a host copy");
        }
        gHostCopyPool.store(pool, std::memory_order_release);
    }
    size_t const width = pool->grow(numThreads - 1) + 1;
    gHostCopyThreads.store(width, std::memory_order_release);
    return width;
}

//! Runs fn(i) for every i in [0, numTasks), each exactly once, on the calling thread and up to hostCopyThreads - 1
//! pool workers, and returns once every call has returned: false if any of them returned false. The calls must be
//! independent of each other (no task reads what another one of the same callback writes). With one thread, or a
//! batch too small to share, it is a plain loop on the calling thread.
template <typename Fn>
bool forEachHostCopyTask(size_t numTasks, ssize_t numBytes, Fn const& fn) noexcept
{
    size_t const totalBytes = numTasks * static_cast<size_t>(std::max<ssize_t>(numBytes, 1));
    size_t const width = std::min(
        {gHostCopyThreads.load(std::memory_order_acquire), numTasks, totalBytes / kMinHostCopyBytesPerThread});
    auto* const pool = gHostCopyPool.load(std::memory_order_acquire);
    std::shared_ptr<HostCopyJob> job;
    if (width > 1 && pool != nullptr)
    {
        try
        {
            job = std::make_shared<HostCopyJob>();
        }
        catch (std::exception const&)
        {
            job = nullptr; // out of memory: run the tasks here
        }
    }
    if (job == nullptr)
    {
        bool success = true;
        for (size_t i = 0; i < numTasks; ++i)
        {
            success = fn(i) && success;
        }
        return success;
    }
    job->fn = [](void const* context, size_t task) noexcept { return (*static_cast<Fn const*>(context))(task); };
    job->context = &fn;
    job->numTasks = numTasks;
    job->maxHelpers = width - 1;
    bool const posted = pool->post(job);
    job->work();
    if (posted)
    {
        pool->retract(job.get());
    }
    job->waitFinished();
    return job->success.load(std::memory_order_relaxed);
}

//! Whether the callbacks are spread over threads: otherwise each one keeps its single-thread loop exactly.
bool hostCopyThreaded() noexcept
{
    return gHostCopyThreads.load(std::memory_order_acquire) > 1;
}

} // namespace

CUDA_CB void hostFnDiskToDiskCopy(void* userData) noexcept
{
    using Data = UserData<DiskAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    if (hostCopyThreaded())
    {
        // Destinations are distinct pages of another level than the sources, so the tasks are independent.
        ssize_t const numBytes = data->numBytes;
        bool const success = forEachHostCopyTask(data->tasks.size(), numBytes,
            [&tasks = data->tasks, numBytes](size_t i) noexcept
            {
                thread_local std::vector<std::byte> buffer;
                auto const& t = tasks[i];
                try
                {
                    if (buffer.size() < static_cast<size_t>(numBytes))
                    {
                        buffer.resize(static_cast<size_t>(numBytes));
                    }
                    if (!readDiskPage(t.src.fd, t.src.pos, 0, buffer.data(), numBytes))
                    {
                        return false;
                    }
                    if (diskWriteBehind().numPages() != 0)
                    {
                        // This write bypasses staging, so a pending write of the destination must not land after it.
                        diskWriteBehind().drop(t.dst.fd, t.dst.pos, t.dst.pos);
                    }
                }
                catch (std::exception const&)
                {
                    return false;
                }
                return writeAll(t.dst.fd, t.dst.pos, buffer.data(), numBytes);
            });
        if (!success)
        {
            TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToDiskCopy failed.\n");
        }
        return;
    }
    std::vector<std::byte> buffer(data->numBytes);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readDiskPage(t.src.fd, t.src.pos, 0, buffer.data(), data->numBytes);
        if (success && diskWriteBehind().numPages() != 0)
        {
            // This write bypasses staging, so a pending write of the destination must not land after it.
            try
            {
                diskWriteBehind().drop(t.dst.fd, t.dst.pos, t.dst.pos);
            }
            catch (std::exception const&)
            {
                success = false;
            }
        }
        success = success && writeAll(t.dst.fd, t.dst.pos, buffer.data(), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnDiskToHostCopy(void* userData) noexcept
{
    // Synchronous path, used when no disk I/O pool is running (disk_io_threads == 0).
    using Data = UserData<MemAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    if (hostCopyThreaded())
    {
        ssize_t const numBytes = data->numBytes;
        success = forEachHostCopyTask(data->tasks.size(), numBytes,
            [&tasks = data->tasks, numBytes](size_t i) noexcept
            {
                auto const& t = tasks[i];
                return readDiskPage(t.src.fd, t.src.pos, 0, reinterpret_cast<void*>(t.dst), numBytes);
            });
    }
    else
    {
        for (auto const& t : data->tasks)
        {
            success = success && readDiskPage(t.src.fd, t.src.pos, 0, reinterpret_cast<void*>(t.dst), data->numBytes);
        }
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToHostCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToDiskCopy(void* userData) noexcept
{
    // Synchronous path, used when no disk I/O pool is running (disk_io_threads == 0).
    using Data = UserData<DiskAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    if (hostCopyThreaded())
    {
        // Each page is written or recorded as failed on its own: a failure does not skip the other pages.
        ssize_t const numBytes = data->numBytes;
        success = forEachHostCopyTask(data->tasks.size(), numBytes,
            [&tasks = data->tasks, numBytes](size_t i) noexcept
            {
                auto const& t = tasks[i];
                bool const written = !takeInjectedWriteFailure()
                    && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), numBytes);
                if (!written)
                {
                    recordFailedDiskWrite(t.dst.fd, t.dst.pos);
                }
                return written;
            });
    }
    else
    {
        for (auto const& t : data->tasks)
        {
            success = success && !takeInjectedWriteFailure()
                && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), data->numBytes);
            if (!success)
            {
                recordFailedDiskWrite(t.dst.fd, t.dst.pos);
            }
        }
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnHostToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToHostCopy(void* userData) noexcept
{
    using Data = UserData<MemAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    if (hostCopyThreaded())
    {
        // Destinations are distinct slots, none of them another task's source.
        ssize_t const numBytes = data->numBytes;
        forEachHostCopyTask(data->tasks.size(), numBytes,
            [&tasks = data->tasks, numBytes](size_t i) noexcept
            {
                auto const& t = tasks[i];
                memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), numBytes);
                return true;
            });
        return;
    }
    for (auto const& t : data->tasks)
    {
        memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), data->numBytes);
    }
}

// Asynchronous disk I/O. CUDA may run host functions one at a time (in practice on one driver thread), so with the
// synchronous path every disk transfer can wait for every other one. Here the host function only hands the transfer to
// a worker pool, and the stream waits on a completion flag (cuStreamWaitValue64), so stream ordering is unchanged. Each
// transfer is split into kDiskIoChunkBytes pieces.
namespace
{

// Large enough that per-piece syscall and queueing costs are small next to the copy, small enough that one large
// transfer spreads over several workers and a read never waits long behind a write piece.
constexpr ssize_t kDiskIoChunkBytes = ssize_t{4} << 20;

// Background (write-through) bytes issued but not yet written are capped, so write-through cannot build up a backlog.
// An eviction can take back the slot of a copy whose write is still running; it then waits for that write instead of
// writing the page again, and promotes the write's remaining pieces to required writes
// (promoteBackgroundDiskWrites). The cap also bounds how much is ahead of a promoted piece: 1 GiB is well under a
// second of NVMe write bandwidth.
constexpr size_t kMaxBackgroundDiskWriteBytes = size_t{1} << 30;

// Pieces are served reads first, then required writes, then background writes. A piece that has waited longer than
// its limit is served ahead of that order (at most every other piece while reads wait), so a steady stream of reads
// cannot starve writes. Required writes free host slots that restores may be waiting for, so they age sooner.
constexpr auto kRequiredWriteMaxWait = std::chrono::milliseconds(10);
constexpr auto kBackgroundWriteMaxWait = std::chrono::milliseconds(100);

std::atomic<size_t> gAsyncDiskIoThreads{0};
std::atomic<size_t> gBackgroundDiskWriteBytes{0};
// Non-null while a BackgroundDiskWriteScope is alive on this thread: host-to-disk copies are background writes and are
// recorded here.
thread_local BackgroundDiskWrites* tBackgroundDiskWrites = nullptr;

// Completion flags: a ring of words in pinned, device-mapped host memory. Transfer t uses word t % kNumCompletionFlags
// and is complete once the word is >= t.
// 64 Ki words (512 KiB of pinned memory): far more than the transfers ever in flight, so launchDiskTransfer
// rarely has to wait before reusing a word.
constexpr uint64_t kNumCompletionFlags = uint64_t{1} << 16;

struct CompletionFlags
{
    uint64_t* host = nullptr; // read through flagWord(); cuStreamWaitValue64 polls the same 8-byte words
    CUdeviceptr device = 0;
    std::atomic<uint64_t> nextTicket{1};
};

static_assert(std::atomic_ref<uint64_t>::is_always_lock_free && sizeof(uint64_t) == 8,
    "host stores must land on the 8-byte word that cuStreamWaitValue64 polls");

//! The flag word of transfer \p ticket.
std::atomic_ref<uint64_t> flagWord(CompletionFlags const& flags, uint64_t ticket) noexcept
{
    return std::atomic_ref<uint64_t>(flags.host[ticket % kNumCompletionFlags]);
}

// nullptr when 64-bit stream memory operations are unavailable; transfers then run in the callback.
CompletionFlags* completionFlags() noexcept
{
    static CompletionFlags* const instance = []() -> CompletionFlags*
    {
        CUdevice device;
        int supported = 0;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS
            || cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device)
                != CUDA_SUCCESS
            || supported == 0)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] 64-bit stream memory operations unavailable; disk I/O completes in the "
                "stream callback");
            return nullptr;
        }
        void* host = nullptr;
        if (cuMemHostAlloc(
                &host, kNumCompletionFlags * sizeof(uint64_t), CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)
            != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] cuMemHostAlloc for disk I/O flags failed; disk I/O completes "
                "in the stream callback");
            return nullptr;
        }
        std::fill_n(static_cast<uint64_t*>(host), kNumCompletionFlags, uint64_t{0});
        CUdeviceptr devicePtr = 0;
        if (cuMemHostGetDevicePointer(&devicePtr, host, 0) != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] cuMemHostGetDevicePointer for disk I/O flags failed; disk I/O "
                "completes in the stream callback");
            cuMemFreeHost(host);
            return nullptr;
        }
        auto* flags = new CompletionFlags; // lives for the process; worker threads may outlive static destruction
        flags->host = static_cast<uint64_t*>(host);
        flags->device = devicePtr;
        return flags;
    }();
    return instance;
}

struct DiskTransfer : std::enable_shared_from_this<DiskTransfer>
{
    DiskTransfer() = default;
    DiskTransfer(DiskTransfer const&) = delete;
    DiskTransfer& operator=(DiskTransfer const&) = delete;

    ~DiskTransfer()
    {
        // Destroyed once its last piece has run, or when its launch failed.
        gBackgroundDiskWriteBytes.fetch_sub(backgroundBytes, std::memory_order_relaxed);
        diskWriteBehind().unreserve(unstagedReservedBytes.load());
    }

    bool isRead = false;
    size_t backgroundBytes = 0; // nonzero for a background write: its share of gBackgroundDiskWriteBytes
    bool promoted = false;      // a background write served as a required one (promoteBackgroundDiskWrites); pool lock
    std::vector<Task<MemAddress, DiskAddress>> reads;
    std::vector<Task<DiskAddress, MemAddress>> writes;
    ssize_t numBytes = 0;
    uint64_t ticket = 0; // 0: no completion flag (synchronous fallback)
    std::atomic<size_t> remaining{0};
    std::atomic<bool> ok{true};
    // Write-behind: the write completes once each page is copied into staging (stagePage); a worker flushes it later.
    bool staged = false;
    std::atomic<size_t> unstagedReservedBytes{0}; // reserved staging not yet handed to a staged page

    [[nodiscard]] size_t numTasks() const noexcept
    {
        return isRead ? reads.size() : writes.size();
    }

    bool runPiece(size_t task, ssize_t offset, ssize_t length) noexcept
    {
        if (isRead)
        {
            auto const& t = reads[task];
            return readDiskPage(t.src.fd, t.src.pos, offset, reinterpret_cast<std::byte*>(t.dst) + offset, length);
        }
        auto const& t = writes[task];
        bool const ok = !takeInjectedWriteFailure()
            && writeAll(t.dst.fd, t.dst.pos + offset, reinterpret_cast<std::byte const*>(t.src) + offset, length);
        if (!ok)
        {
            // Recorded before finish() publishes completion, so whoever waits on the transfer sees the failure.
            recordFailedDiskWrite(t.dst.fd, t.dst.pos);
        }
        return ok;
    }

    //! Runs the piece unless the transfer has already failed. A skipped write leaves its page incomplete, so it is
    //! recorded as failed like the piece that failed, as the synchronous path does for the tasks after a failure.
    bool runPieceUnlessFailed(size_t task, ssize_t offset, ssize_t length) noexcept
    {
        if (ok.load())
        {
            return runPiece(task, offset, length);
        }
        if (!isRead)
        {
            recordFailedDiskWrite(writes[task].dst.fd, writes[task].dst.pos);
        }
        return false;
    }

    void finish() noexcept
    {
        if (!ok.load())
        {
            TLLM_LOG_ERROR(isRead ? "[kvCacheManagerV2Utils] async disk-to-host copy failed.\n"
                                  : "[kvCacheManagerV2Utils] async host-to-disk copy failed.\n");
        }
        if (ticket != 0)
        {
            // The data written by pread must be visible before the device observes the flag. On aarch64 the C++
            // fence only orders stores for inner-shareable observers; a PCIe-attached GPU is outer-shareable, so
            // publish the data with an outer-shareable store barrier first, as Linux's dma_wmb() does.
            std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__aarch64__)
            asm volatile("dmb oshst" ::: "memory");
#endif
            auto word = flagWord(*completionFlags(), ticket);
            word.store(ticket, std::memory_order_release);
            word.notify_all(); // a launch may be waiting to reuse this word after the ring wrapped
        }
    }
};

class DiskIoPool
{
public:
    //! Returns how many of \p numThreads workers started.
    size_t start(size_t numThreads) noexcept
    {
        size_t started = 0;
        try
        {
            for (; started < numThreads; ++started)
            {
                std::thread([this] { run(); }).detach();
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] started %zu of %zu disk I/O threads: %s", started, numThreads, e.what());
        }
        return started;
    }

    //! At most \p maxWriters workers run write pieces (required writes, write-behind flushes, write-through copies)
    //! at once; the others are kept for reads. 0: no limit. Returns the previous limit (0: none).
    size_t setMaxWriters(size_t maxWriters)
    {
        size_t previous = 0;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            previous = mMaxWriters == std::numeric_limits<size_t>::max() ? 0 : mMaxWriters;
            mMaxWriters = maxWriters == 0 ? std::numeric_limits<size_t>::max() : maxWriters;
        }
        mCv.notify_all();
        return previous;
    }

    //! Testing only: the write cap and its counters (threads is filled in by the caller).
    DiskIoWriteCapStats writeCapStats(bool reset)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        DiskIoWriteCapStats stats;
        stats.maxWriters = mMaxWriters == std::numeric_limits<size_t>::max() ? 0 : mMaxWriters;
        stats.peakWriters = mPeakWriters;
        stats.readsPastCappedWrites = mReadsPastCappedWrites;
        stats.expiredWriteHolds = mExpiredWriteHolds;
        if (reset)
        {
            mPeakWriters = mWritersRunning;
            mReadsPastCappedWrites = 0;
            mExpiredWriteHolds = 0;
        }
        return stats;
    }

    //! Testing only: until \p until, a worker that took a write piece waits before running it; time_point::min()
    //! releases the waiting workers now.
    void holdWrites(std::chrono::steady_clock::time_point until)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mHoldWritesUntil = until;
        }
        mHoldCv.notify_all();
    }

    void delayBackground(std::chrono::steady_clock::duration delay)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mBackgroundDelay = delay;
        }
        mCv.notify_all();
    }

    void delayFlushes(std::chrono::steady_clock::duration delay)
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mFlushDelay = delay;
        }
        mCv.notify_all();
    }

    void waitIdle()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIdleCv.wait(lock, [this] { return mInFlight == 0; });
    }

    // Called from a stream callback: must not call CUDA APIs or block on I/O.
    void submit(std::shared_ptr<DiskTransfer> const& transfer)
    {
        if (transfer->staged)
        {
            // One stage piece per page; each queues the page's flush once it has run.
            size_t const numPages = transfer->numTasks();
            transfer->remaining.store(numPages);
            if (numPages == 0)
            {
                transfer->finish();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mInFlight += numPages;
                auto const queued = std::chrono::steady_clock::now();
                for (size_t task = 0; task < numPages; ++task)
                {
                    mStage.push_back(Piece{transfer, task, 0, transfer->numBytes, queued});
                }
            }
            mCv.notify_all();
            return;
        }
        auto const chunk = kDiskIoChunkBytes;
        size_t numPieces = 0;
        for (size_t task = 0; task < transfer->numTasks(); ++task)
        {
            numPieces += static_cast<size_t>((transfer->numBytes + chunk - 1) / chunk);
        }
        transfer->remaining.store(numPieces);
        if (numPieces == 0)
        {
            transfer->finish();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mInFlight += numPieces;
            auto& queue = transfer->isRead
                ? mReads
                : (transfer->backgroundBytes != 0 && !transfer->promoted ? mBackground : mWrites);
            auto const queued = std::chrono::steady_clock::now();
            for (size_t task = 0; task < transfer->numTasks(); ++task)
            {
                for (ssize_t offset = 0; offset < transfer->numBytes; offset += chunk)
                {
                    queue.push_back(
                        Piece{transfer, task, offset, std::min(chunk, transfer->numBytes - offset), queued});
                }
            }
        }
        mCv.notify_all();
    }

    //! Serves the queued pieces of a background write as required writes, and the rest of them once submitted.
    void promote(std::shared_ptr<DiskTransfer> const& transfer) noexcept
    {
        bool moved = false;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (transfer->promoted)
            {
                return;
            }
            transfer->promoted = true;
            // At most kMaxBackgroundDiskWriteBytes / kDiskIoChunkBytes (256) pieces are queued here, so a scan is
            // cheap.
            for (auto it = mBackground.begin(); it != mBackground.end();)
            {
                if (it->transfer != transfer)
                {
                    ++it;
                    continue;
                }
                try
                {
                    mWrites.push_back(std::move(*it));
                }
                catch (std::exception const&)
                {
                    break; // out of memory: the remaining pieces stay background writes, which still run
                }
                it = mBackground.erase(it);
                moved = true;
            }
        }
        // Workers may be asleep while the queued background pieces may not start yet, so wake them for the promoted
        // ones.
        if (moved)
        {
            mCv.notify_all();
        }
    }

private:
    struct Piece
    {
        std::shared_ptr<DiskTransfer> transfer; // null for a flush
        size_t task;
        ssize_t offset;
        ssize_t length;
        std::chrono::steady_clock::time_point queued;
        std::shared_ptr<StagedDiskPage> page = nullptr; // a flush: the staged page and where it goes
        DiskPageKey key{-1, 0};
    };

    // Testing only: a flush may start once it has been queued for mFlushDelay.
    [[nodiscard]] bool flushReady(std::chrono::steady_clock::time_point now) const noexcept
    {
        return !mFlush.empty() && now - mFlush.front().queued >= mFlushDelay;
    }

    // Testing only: a background piece may start once it has been queued for mBackgroundDelay.
    [[nodiscard]] bool backgroundReady(std::chrono::steady_clock::time_point now) const noexcept
    {
        return !mBackground.empty() && now - mBackground.front().queued >= mBackgroundDelay;
    }

    // A worker may start a write piece while fewer than mMaxWriters run one.
    [[nodiscard]] bool canStartWrite() const noexcept
    {
        return mWritersRunning < mMaxWriters;
    }

    [[nodiscard]] bool hasWork(std::chrono::steady_clock::time_point now) const noexcept
    {
        return !mStage.empty() || !mReads.empty()
            || (canStartWrite() && (!mWrites.empty() || flushReady(now) || backgroundReady(now)));
    }

    // Called with mMutex held and hasWork(now).
    std::deque<Piece>& nextQueue(std::chrono::steady_clock::time_point now)
    {
        // Staging a page is a memory copy that completes its write: always first.
        if (!mStage.empty())
        {
            return mStage;
        }
        if (!canStartWrite())
        {
            // hasWork(now): with no write allowed, reads are waiting.
            if (!mWrites.empty() || flushReady(now) || backgroundReady(now))
            {
                ++mReadsPastCappedWrites;
            }
            return mReads;
        }
        bool const readsWaiting = !mReads.empty();
        // Flushes free staging; they are served like required writes, the older of the two queues first.
        auto& writes = !flushReady(now) || (!mWrites.empty() && mWrites.front().queued <= mFlush.front().queued)
            ? mWrites
            : mFlush;
        if (!readsWaiting || !mServedAgedPiece)
        {
            auto const overdue = [now](std::deque<Piece> const& queue, auto maxWait)
            { return !queue.empty() && now - queue.front().queued >= maxWait; };
            if (overdue(writes, kRequiredWriteMaxWait))
            {
                mServedAgedPiece = readsWaiting;
                return writes;
            }
            if (backgroundReady(now) && overdue(mBackground, kBackgroundWriteMaxWait))
            {
                mServedAgedPiece = readsWaiting;
                return mBackground;
            }
        }
        mServedAgedPiece = false;
        if (readsWaiting)
        {
            return mReads;
        }
        return !writes.empty() || !backgroundReady(now) ? writes : mBackground;
    }

    void run()
    {
        for (;;)
        {
            Piece piece;
            bool isWrite = false;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                auto now = std::chrono::steady_clock::now();
                while (!hasWork(now))
                {
                    // Writes held back by mMaxWriters wait for a writer to finish (writerDone notifies).
                    if ((mBackground.empty() && mFlush.empty()) || !canStartWrite())
                    {
                        mCv.wait(lock);
                    }
                    else
                    {
                        // Only with a testing delay: wake when the oldest delayed piece may start.
                        auto wake = std::chrono::steady_clock::time_point::max();
                        if (!mBackground.empty())
                        {
                            wake = std::min(wake, mBackground.front().queued + mBackgroundDelay);
                        }
                        if (!mFlush.empty())
                        {
                            wake = std::min(wake, mFlush.front().queued + mFlushDelay);
                        }
                        mCv.wait_until(lock, wake);
                    }
                    now = std::chrono::steady_clock::now();
                }
                auto& queue = nextQueue(now);
                isWrite = &queue != &mStage && &queue != &mReads;
                if (isWrite)
                {
                    mPeakWriters = std::max(mPeakWriters, ++mWritersRunning);
                }
                piece = std::move(queue.front());
                queue.pop_front();
                if (isWrite && std::chrono::steady_clock::now() < mHoldWritesUntil)
                {
                    // Testing only (holdDiskWritePiecesForTesting): the worker counts as a writer while it waits.
                    auto const until = mHoldWritesUntil;
                    mHoldCv.wait_until(lock, until,
                        [this, until]
                        { return mHoldWritesUntil != until || std::chrono::steady_clock::now() >= until; });
                    if (mHoldWritesUntil == until)
                    {
                        ++mExpiredWriteHolds; // released by the deadline, not by the caller
                    }
                }
            }
            runOne(std::move(piece));
            if (isWrite)
            {
                writerDone();
            }
        }
    }

    void runOne(Piece piece) noexcept
    {
        if (piece.page)
        {
            runFlush(std::move(piece));
            return;
        }
        DiskTransfer& transfer = *piece.transfer;
        if (transfer.staged)
        {
            stagePage(transfer, piece.task);
            if (transfer.remaining.fetch_sub(1) == 1)
            {
                transfer.finish();
                transfer.remaining.notify_all();
            }
            pieceDone();
            return;
        }
        // Once a piece has failed, the transfer has failed: skip the syscalls of its remaining pieces.
        if (!transfer.runPieceUnlessFailed(piece.task, piece.offset, piece.length))
        {
            transfer.ok.store(false);
        }
        if (transfer.remaining.fetch_sub(1) == 1)
        {
            transfer.finish();
            transfer.remaining.notify_all();
        }
        pieceDone();
    }

    void writerDone()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            --mWritersRunning;
        }
        mCv.notify_one(); // a write held back by mMaxWriters may start now
    }

    void pieceDone()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (--mInFlight == 0)
        {
            mIdleCv.notify_all();
        }
    }

    //! Copies page \p task of a staged write into staging, makes it the page's pending content and queues its flush.
    //! Without memory for it, the page is written here instead, as an unstaged write would be.
    void stagePage(DiskTransfer& transfer, size_t task) noexcept
    {
        auto const& t = transfer.writes[task];
        size_t const bytes = static_cast<size_t>(transfer.numBytes);
        auto& wb = diskWriteBehind();
        try
        {
            auto page = std::make_shared<StagedDiskPage>(wb.takeBuffer(bytes), bytes);
            transfer.unstagedReservedBytes.fetch_sub(bytes); // now owned by the page
            std::memcpy(page->data.get(), reinterpret_cast<void const*>(t.src), bytes);
            DiskPageKey const key{t.dst.fd, t.dst.pos};
            wb.publish(key, page);
            {
                // A background write keeps its transfer, and so its share of the background write cap, until the
                // flush: write-through must not get ahead of the disk.
                std::lock_guard<std::mutex> lock(mMutex);
                mFlush.push_back(Piece{transfer.backgroundBytes != 0 ? transfer.shared_from_this() : nullptr, 0, 0,
                    transfer.numBytes, std::chrono::steady_clock::now(), page, key});
                ++mInFlight;
            }
            mCv.notify_one();
            return;
        }
        catch (std::exception const&)
        {
            // Out of memory: fall through to a direct write.
        }
        bool const ok = !takeInjectedWriteFailure()
            && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), transfer.numBytes);
        if (!ok)
        {
            recordFailedDiskWrite(t.dst.fd, t.dst.pos);
            transfer.ok.store(false);
        }
    }

    void runFlush(Piece piece) noexcept
    {
        auto& wb = diskWriteBehind();
        auto const start = wb.beginFlush(piece.key, piece.page.get());
        if (start == DiskWriteBehind::FlushStart::kBusy)
        {
            // An older flush of this page is still writing; this one must land after it.
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mFlush.push_back(std::move(piece));
            }
            std::this_thread::yield();
            return;
        }
        if (start == DiskWriteBehind::FlushStart::kStart)
        {
            bool const written = !takeInjectedWriteFailure()
                && writeAll(piece.key.first, piece.key.second, piece.page->data.get(), piece.length);
            if (!written)
            {
                TLLM_LOG_ERROR("[kvCacheManagerV2Utils] write-behind flush failed; the page stays in staging\n");
            }
            wb.endFlush(piece.key, piece.page.get(), written);
        }
        piece.page.reset();
        pieceDone();
    }

    std::mutex mMutex;
    std::condition_variable mCv;
    std::condition_variable mIdleCv;
    std::condition_variable mHoldCv;                         // testing only: holdDiskWritePiecesForTesting
    size_t mInFlight = 0;                                    // pieces queued or running
    bool mServedAgedPiece = false;                           // the last piece jumped the queue while reads were waiting
    size_t mMaxWriters = std::numeric_limits<size_t>::max(); // workers that may run write pieces at once
    size_t mWritersRunning = 0;                              // workers running a write piece
    size_t mPeakWriters = 0;                                 // testing only: most of mWritersRunning seen
    uint64_t mReadsPastCappedWrites = 0;                     // testing only: reads started while a write was capped
    uint64_t mExpiredWriteHolds = 0;                         // testing only: held write pieces released by the deadline
    std::chrono::steady_clock::time_point mHoldWritesUntil{}; // testing only: holdDiskWritePiecesForTesting
    std::chrono::steady_clock::duration mBackgroundDelay{0};  // testing only: delayBackgroundDiskWritesForTesting
    std::chrono::steady_clock::duration mFlushDelay{0};       // testing only: delayDiskWriteBehindFlushesForTesting
    std::deque<Piece> mStage;                                 // write-behind: copy a page into staging
    std::deque<Piece> mFlush;                                 // write-behind: write a staged page to its file
    std::deque<Piece> mReads;                                 // restores: on the critical path
    std::deque<Piece> mWrites;                                // required evictions to disk
    std::deque<Piece> mBackground;                            // write-through copies
};

// Set once by configureAsyncDiskIo, before gAsyncDiskIoThreads. Intentionally leaked: detached workers may
// still be waiting on it during static destruction.
std::atomic<DiskIoPool*> gDiskIoPool{nullptr};

DiskIoPool& diskIoPool()
{
    return *gDiskIoPool.load(std::memory_order_acquire);
}

} // namespace

struct BackgroundDiskWrites
{
    std::vector<std::weak_ptr<DiskTransfer>> transfers;
};

namespace
{

struct DiskLaunch
{
    std::shared_ptr<DiskTransfer> transfer;
};

CUDA_CB void hostFnSubmitDiskTransfer(void* userData) noexcept
{
    std::unique_ptr<DiskLaunch> const launch(static_cast<DiskLaunch*>(userData));
    diskIoPool().submit(launch->transfer);
}

void waitForTransfer(DiskTransfer const& transfer) noexcept
{
    // The worker that runs the last piece notifies.
    for (size_t left = transfer.remaining.load(); left != 0; left = transfer.remaining.load())
    {
        transfer.remaining.wait(left);
    }
}

// Fallback without stream memory operations: still parallel, but the callback waits for the transfer.
CUDA_CB void hostFnRunDiskTransfer(void* userData) noexcept
{
    std::unique_ptr<DiskLaunch> const launch(static_cast<DiskLaunch*>(userData));
    auto const transfer = launch->transfer;
    diskIoPool().submit(transfer);
    waitForTransfer(*transfer);
}

// Runs after hostFnSubmitDiskTransfer on the same stream, so the transfer's pieces are already counted.
CUDA_CB void hostFnWaitDiskTransfer(void* userData) noexcept
{
    std::unique_ptr<DiskLaunch> const launch(static_cast<DiskLaunch*>(userData));
    waitForTransfer(*launch->transfer);
}

CUresult launchDiskTransfer(std::shared_ptr<DiskTransfer> transfer, CUstream stream) noexcept
{
    CompletionFlags* const flags = completionFlags();
    if (flags == nullptr)
    {
        auto launch = std::make_unique<DiskLaunch>(DiskLaunch{std::move(transfer)});
        CUresult const result = cuLaunchHostFunc(stream, hostFnRunDiskTransfer, launch.get());
        if (result == CUDA_SUCCESS)
        {
            static_cast<void>(launch.release());
        }
        return result;
    }
    uint64_t const ticket = flags->nextTicket.fetch_add(1);
    if (ticket > kNumCompletionFlags)
    {
        // Never reuse a flag word whose previous transfer is still in flight. Whoever stores the word notifies.
        auto word = flagWord(*flags, ticket);
        for (uint64_t seen = word.load(std::memory_order_acquire); seen < ticket - kNumCompletionFlags;
             seen = word.load(std::memory_order_acquire))
        {
            word.wait(seen, std::memory_order_acquire);
        }
    }
    transfer->ticket = ticket;
    auto const submitted = transfer;
    auto launch = std::make_unique<DiskLaunch>(DiskLaunch{std::move(transfer)});
    if (CUresult const result = cuLaunchHostFunc(stream, hostFnSubmitDiskTransfer, launch.get());
        result != CUDA_SUCCESS)
    {
        // Retire the ticket, or the transfer that reuses its word after the ring wraps would wait forever.
        auto word = flagWord(*flags, ticket);
        word.store(ticket, std::memory_order_release);
        word.notify_all();
        return result;
    }
    static_cast<void>(launch.release()); // owned by the callback now
    CUresult const waitResult = cuStreamWaitValue64(
        stream, flags->device + (ticket % kNumCompletionFlags) * sizeof(uint64_t), ticket, CU_STREAM_WAIT_VALUE_GEQ);
    if (waitResult == CUDA_SUCCESS)
    {
        return CUDA_SUCCESS;
    }
    // The transfer will run regardless, so the stream must not move past it: wait in a host function instead.
    auto waitLaunch = std::make_unique<DiskLaunch>(DiskLaunch{submitted});
    if (cuLaunchHostFunc(stream, hostFnWaitDiskTransfer, waitLaunch.get()) != CUDA_SUCCESS)
    {
        return waitResult;
    }
    static_cast<void>(waitLaunch.release());
    TLLM_LOG_WARNING(
        "[kvCacheManagerV2Utils] cuStreamWaitValue64 failed (%d); the stream waits for the disk transfer in a host "
        "function",
        static_cast<int>(waitResult));
    return CUDA_SUCCESS;
}

bool asyncDiskIo() noexcept
{
    return gAsyncDiskIoThreads.load(std::memory_order_acquire) > 0;
}

} // namespace

bool asyncDiskIoEnabled() noexcept
{
    return asyncDiskIo();
}

BackgroundDiskWriteScope::BackgroundDiskWriteScope() noexcept
    : mPrevious(tBackgroundDiskWrites)
    , mWrites(new(std::nothrow) BackgroundDiskWrites)
{
    // Without the record (out of memory) the writes are still background writes; they just cannot be promoted.
    static thread_local BackgroundDiskWrites unrecorded;
    tBackgroundDiskWrites = mWrites ? mWrites.get() : &unrecorded;
}

BackgroundDiskWriteScope::~BackgroundDiskWriteScope()
{
    if (!mWrites)
    {
        tBackgroundDiskWrites->transfers.clear();
    }
    tBackgroundDiskWrites = mPrevious;
}

std::shared_ptr<BackgroundDiskWrites const> BackgroundDiskWriteScope::takeWrites()
{
    if (!mWrites || mWrites->transfers.empty())
    {
        return nullptr;
    }
    auto taken = std::make_shared<BackgroundDiskWrites>(std::move(*mWrites));
    mWrites->transfers.clear();
    return taken;
}

void promoteBackgroundDiskWrites(BackgroundDiskWrites const& writes) noexcept
{
    if (!asyncDiskIo())
    {
        return;
    }
    for (auto const& weak : writes.transfers)
    {
        if (auto transfer = weak.lock())
        {
            diskIoPool().promote(transfer);
        }
    }
}

size_t backgroundDiskWriteBudget() noexcept
{
    if (!asyncDiskIo())
    {
        return 0;
    }
    size_t const issued = gBackgroundDiskWriteBytes.load(std::memory_order_relaxed);
    return issued < kMaxBackgroundDiskWriteBytes ? kMaxBackgroundDiskWriteBytes - issued : 0;
}

bool consumeFailedDiskWrite(int fd, ssize_t pos) noexcept
{
    auto& failed = failedDiskWrites();
    if (failed.size.load(std::memory_order_acquire) == 0)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(failed.mutex);
    bool const found = failed.pages.erase({fd, pos}) > 0;
    failed.size.store(failed.pages.size(), std::memory_order_release);
    return found;
}

void injectDiskWriteFailuresForTesting(uint64_t count) noexcept
{
    gInjectedWriteFailures.store(count, std::memory_order_relaxed);
}

void delayBackgroundDiskWritesForTesting(double seconds) noexcept
{
    if (asyncDiskIo())
    {
        diskIoPool().delayBackground(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(seconds > 0 ? seconds : 0.0)));
    }
}

DiskIoWriteCapStats diskIoWriteCapStatsForTesting(bool reset) noexcept
{
    if (!asyncDiskIo())
    {
        return {};
    }
    auto stats = diskIoPool().writeCapStats(reset);
    stats.threads = gAsyncDiskIoThreads.load(std::memory_order_acquire);
    return stats;
}

size_t setDiskIoMaxWriteThreadsForTesting(size_t maxWriters) noexcept
{
    return asyncDiskIo() ? diskIoPool().setMaxWriters(maxWriters) : 0;
}

void holdDiskWritePiecesForTesting(double seconds) noexcept
{
    if (asyncDiskIo())
    {
        diskIoPool().holdWrites(seconds > 0 ? std::chrono::steady_clock::now()
                    + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(seconds))
                                            : std::chrono::steady_clock::time_point::min());
    }
}

void waitForDiskIo() noexcept
{
    if (asyncDiskIo())
    {
        diskIoPool().waitIdle();
    }
}

void configureDiskWriteBehind(size_t bytes) noexcept
{
    diskWriteBehind().configure(bytes);
}

void discardDiskWriteBehind(int fd, ssize_t minPos) noexcept
{
    if (diskWriteBehind().numPages() == 0)
    {
        return;
    }
    try
    {
        diskWriteBehind().drop(fd, minPos);
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] could not discard write-behind pages: %s\n", e.what());
    }
}

size_t diskWriteBehindPagesForTesting() noexcept
{
    return diskWriteBehind().numPages();
}

void delayDiskWriteBehindFlushesForTesting(double seconds) noexcept
{
    if (asyncDiskIo())
    {
        diskIoPool().delayFlushes(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(seconds > 0 ? seconds : 0.0)));
    }
}

void configureHostCopyThreads(size_t numThreads) noexcept
{
    if (numThreads <= 1)
    {
        return;
    }
    size_t expected = 0;
    if (!gConfiguredHostCopyThreads.compare_exchange_strong(expected, numThreads))
    {
        if (expected != numThreads)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] host copy callbacks already use %zu threads in this process; ignoring a "
                "request for %zu",
                expected, numThreads);
        }
        return;
    }
    size_t const width = setHostCopyThreads(numThreads);
    TLLM_LOG_INFO("[kvCacheManagerV2Utils] host copy callbacks: threads=%zu", width);
}

size_t hostCopyThreads() noexcept
{
    return gHostCopyThreads.load(std::memory_order_acquire);
}

size_t setHostCopyThreadsForTesting(size_t numThreads) noexcept
{
    return setHostCopyThreads(numThreads);
}

uint64_t hostCopyHelperTasksForTesting() noexcept
{
    return gHostCopyHelperTasks.load(std::memory_order_relaxed);
}

void configureAsyncDiskIo(size_t numThreads) noexcept
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // The pool is process-wide and lives until exit: a later request cannot resize it or turn it off.
    if (size_t const current = gAsyncDiskIoThreads.load(); current != 0)
    {
        if (current != numThreads)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] async disk I/O already uses %zu threads for this process; ignoring a request "
                "for %zu",
                current, numThreads);
        }
        return;
    }
    if (numThreads == 0)
    {
        return;
    }
    auto* pool = gDiskIoPool.load();
    if (pool == nullptr)
    {
        pool = new (std::nothrow) DiskIoPool;
        if (pool == nullptr)
        {
            return;
        }
        gDiskIoPool.store(pool, std::memory_order_release);
    }
    size_t const started = pool->start(numThreads);
    if (started == 0)
    {
        TLLM_LOG_WARNING("[kvCacheManagerV2Utils] no disk I/O thread started; using synchronous disk I/O");
        return;
    }
    // Write pieces may take at most half of the workers. A buffered write can block for milliseconds (the file's
    // inode lock, dirty-page throttling, page reclaim) while a read from the page cache takes tens of microseconds, so
    // with no limit a write backlog (write-behind, write-through) parks every worker in a write and restores read at
    // the rate writes complete. TLLM_KV_DISK_IO_MAX_WRITE_THREADS overrides the limit; 0 removes it.
    size_t maxWriters = std::max<size_t>(1, started / 2);
    if (char const* env = std::getenv("TLLM_KV_DISK_IO_MAX_WRITE_THREADS"); env != nullptr && *env != '\0')
    {
        char* end = nullptr;
        unsigned long long const value = std::strtoull(env, &end, 10);
        if (end != nullptr && *end == '\0' && *env != '-')
        {
            maxWriters = static_cast<size_t>(value);
        }
        else
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] ignoring TLLM_KV_DISK_IO_MAX_WRITE_THREADS=%s (not a non-negative integer)",
                env);
        }
    }
    pool->setMaxWriters(maxWriters);
    gAsyncDiskIoThreads.store(started, std::memory_order_release);
    TLLM_LOG_INFO("[kvCacheManagerV2Utils] async disk I/O: threads=%zu, write threads at most %s", started,
        maxWriters == 0 ? "unlimited" : std::to_string(maxWriters).c_str());
}

CUresult copyDiskToDisk(std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<DiskAddress, DiskAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnDiskToDiskCopy, data.release());
}

CUresult copyDiskToHost(std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    if (asyncDiskIo())
    {
        auto transfer = std::make_shared<DiskTransfer>();
        transfer->isRead = true;
        transfer->reads = std::move(tasks);
        transfer->numBytes = numBytes;
        return launchDiskTransfer(std::move(transfer), stream);
    }
    using Data = UserData<MemAddress, DiskAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnDiskToHostCopy, data.release());
}

CUresult copyHostToDisk(std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    forgetFailedDiskWrites(tasks);
    if (asyncDiskIo())
    {
        auto transfer = std::make_shared<DiskTransfer>();
        transfer->isRead = false;
        if (tBackgroundDiskWrites != nullptr)
        {
            transfer->backgroundBytes = static_cast<size_t>(numBytes) * tasks.size();
            gBackgroundDiskWriteBytes.fetch_add(transfer->backgroundBytes, std::memory_order_relaxed);
            try
            {
                tBackgroundDiskWrites->transfers.push_back(transfer);
            }
            catch (std::exception const&)
            {
                // Out of memory: the write cannot be promoted later, but it still runs as a background write.
            }
        }
        size_t const stagedBytes = static_cast<size_t>(numBytes) * tasks.size();
        // Background writes may take half of the staging at most, so evictions find room.
        auto& wb = diskWriteBehind();
        bool const fits = transfer->backgroundBytes == 0 || wb.reservedBytes() + stagedBytes <= wb.capacity() / 2;
        if (stagedBytes != 0 && fits && wb.reserve(stagedBytes))
        {
            transfer->staged = true;
            transfer->unstagedReservedBytes.store(stagedBytes);
        }
        transfer->writes = std::move(tasks);
        transfer->numBytes = numBytes;
        return launchDiskTransfer(std::move(transfer), stream);
    }
    using Data = UserData<DiskAddress, MemAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnHostToDiskCopy, data.release());
}

CUresult copyHostToHost(std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, CUstream stream) noexcept
{
    using Data = UserData<MemAddress, MemAddress>;
    auto data = std::make_unique<Data>(Data{std::move(tasks), numBytes});
    return cuLaunchHostFunc(stream, hostFnHostToHostCopy, data.release());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
