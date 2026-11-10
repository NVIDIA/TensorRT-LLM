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
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <set>
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

CUDA_CB void hostFnDiskToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<DiskAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    std::vector<std::byte> buffer(data->numBytes);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, buffer.data(), data->numBytes);
        success = success && writeAll(t.dst.fd, t.dst.pos, buffer.data(), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnDiskToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<MemAddress, DiskAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, reinterpret_cast<void*>(t.dst), data->numBytes);
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToHostCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<DiskAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && !takeInjectedWriteFailure()
            && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), data->numBytes);
        if (!success)
        {
            recordFailedDiskWrite(t.dst.fd, t.dst.pos);
        }
    }
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnHostToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    using Data = UserData<MemAddress, MemAddress>;
    auto const data = std::unique_ptr<Data>(static_cast<Data*>(userData));
    for (auto const& t : data->tasks)
    {
        memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), data->numBytes);
    }
}

// Asynchronous disk I/O. CUDA runs all host functions of a process on one thread, so with the synchronous path
// every disk transfer waits for every other one. Here the host function only hands the transfer to a worker pool,
// and the stream waits on a completion flag (cuStreamWaitValue64), so stream ordering is unchanged. Each transfer
// is split into kDiskIoChunkBytes pieces.
namespace
{

constexpr ssize_t kDiskIoChunkBytes = ssize_t{4} << 20;

// Background (write-through) bytes issued but not yet written are capped. A required write can end up waiting for a
// background one (an eviction that takes a written-through copy's slot back waits for that copy's write), so the cap
// bounds that wait: 1 GiB is well under a second of NVMe write bandwidth.
constexpr size_t kMaxBackgroundDiskWriteBytes = size_t{1} << 30;

// Pieces are served reads first, then required writes, then background writes. A piece that has waited longer than
// its limit is served ahead of that order (at most every other piece while reads wait), so a steady stream of reads
// cannot starve writes. Required writes free host slots that restores may be waiting for, so they age sooner.
constexpr auto kRequiredWriteMaxWait = std::chrono::milliseconds(10);
constexpr auto kBackgroundWriteMaxWait = std::chrono::milliseconds(100);

std::atomic<size_t> gAsyncDiskIoThreads{0};
std::atomic<size_t> gBackgroundDiskWriteBytes{0};
thread_local bool tBackgroundDiskWrites = false;

// Completion flags: a ring of words in pinned, device-mapped host memory. Transfer t uses word t % kNumCompletionFlags
// and is complete once the word is >= t.
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

struct DiskTransfer
{
    DiskTransfer() = default;
    DiskTransfer(DiskTransfer const&) = delete;
    DiskTransfer& operator=(DiskTransfer const&) = delete;

    ~DiskTransfer()
    {
        // Destroyed once its last piece has run, or when its launch failed.
        gBackgroundDiskWriteBytes.fetch_sub(backgroundBytes, std::memory_order_relaxed);
    }

    bool isRead = false;
    size_t backgroundBytes = 0; // nonzero for a background write: its share of gBackgroundDiskWriteBytes
    std::vector<Task<MemAddress, DiskAddress>> reads;
    std::vector<Task<DiskAddress, MemAddress>> writes;
    ssize_t numBytes = 0;
    uint64_t ticket = 0; // 0: no completion flag (synchronous fallback)
    std::atomic<size_t> remaining{0};
    std::atomic<bool> ok{true};

    [[nodiscard]] size_t numTasks() const noexcept
    {
        return isRead ? reads.size() : writes.size();
    }

    bool runPiece(size_t task, ssize_t offset, ssize_t length) noexcept
    {
        if (isRead)
        {
            auto const& t = reads[task];
            return readAll(t.src.fd, t.src.pos + offset, reinterpret_cast<std::byte*>(t.dst) + offset, length);
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
            flagWord(*completionFlags(), ticket).store(ticket, std::memory_order_release);
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

    void waitIdle()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mIdleCv.wait(lock, [this] { return mInFlight == 0; });
    }

    // Called from a stream callback: must not call CUDA APIs or block on I/O.
    void submit(std::shared_ptr<DiskTransfer> const& transfer)
    {
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
            auto& queue = transfer->isRead ? mReads : (transfer->backgroundBytes != 0 ? mBackground : mWrites);
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

private:
    struct Piece
    {
        std::shared_ptr<DiskTransfer> transfer;
        size_t task;
        ssize_t offset;
        ssize_t length;
        std::chrono::steady_clock::time_point queued;
    };

    // Called with mMutex held and at least one queue nonempty.
    std::deque<Piece>& nextQueue()
    {
        bool const readsWaiting = !mReads.empty();
        if (!readsWaiting || !mServedAgedPiece)
        {
            auto const now = std::chrono::steady_clock::now();
            auto const overdue = [now](std::deque<Piece> const& queue, auto maxWait)
            { return !queue.empty() && now - queue.front().queued >= maxWait; };
            if (overdue(mWrites, kRequiredWriteMaxWait))
            {
                mServedAgedPiece = readsWaiting;
                return mWrites;
            }
            if (overdue(mBackground, kBackgroundWriteMaxWait))
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
        return !mWrites.empty() ? mWrites : mBackground;
    }

    void run()
    {
        for (;;)
        {
            Piece piece;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCv.wait(lock, [this] { return !mReads.empty() || !mWrites.empty() || !mBackground.empty(); });
                auto& queue = nextQueue();
                piece = std::move(queue.front());
                queue.pop_front();
            }
            DiskTransfer& transfer = *piece.transfer;
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
            {
                std::lock_guard<std::mutex> lock(mMutex);
                if (--mInFlight == 0)
                {
                    mIdleCv.notify_all();
                }
            }
        }
    }

    std::mutex mMutex;
    std::condition_variable mCv;
    std::condition_variable mIdleCv;
    size_t mInFlight = 0;          // pieces queued or running
    bool mServedAgedPiece = false; // the last piece jumped the queue while reads were waiting
    std::deque<Piece> mReads;      // restores: on the critical path
    std::deque<Piece> mWrites;     // required evictions to disk
    std::deque<Piece> mBackground; // write-through copies
};

// Set once by configureAsyncDiskIo, before gAsyncDiskIoThreads. Intentionally leaked: detached workers may
// still be waiting on it during static destruction.
std::atomic<DiskIoPool*> gDiskIoPool{nullptr};

DiskIoPool& diskIoPool()
{
    return *gDiskIoPool.load(std::memory_order_acquire);
}

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
        // Never reuse a flag word whose previous transfer is still in flight.
        while (flagWord(*flags, ticket).load(std::memory_order_acquire) < ticket - kNumCompletionFlags)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    transfer->ticket = ticket;
    auto const submitted = transfer;
    auto launch = std::make_unique<DiskLaunch>(DiskLaunch{std::move(transfer)});
    if (CUresult const result = cuLaunchHostFunc(stream, hostFnSubmitDiskTransfer, launch.get());
        result != CUDA_SUCCESS)
    {
        // Retire the ticket, or the transfer that reuses its word after the ring wraps would wait forever.
        flagWord(*flags, ticket).store(ticket, std::memory_order_release);
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
{
    tBackgroundDiskWrites = true;
}

BackgroundDiskWriteScope::~BackgroundDiskWriteScope()
{
    tBackgroundDiskWrites = mPrevious;
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

void waitForDiskIo() noexcept
{
    if (asyncDiskIo())
    {
        diskIoPool().waitIdle();
    }
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
    gAsyncDiskIoThreads.store(started, std::memory_order_release);
    TLLM_LOG_INFO("[kvCacheManagerV2Utils] async disk I/O: threads=%zu", started);
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
        if (tBackgroundDiskWrites)
        {
            transfer->backgroundBytes = static_cast<size_t>(numBytes) * tasks.size();
            gBackgroundDiskWriteBytes.fetch_add(transfer->backgroundBytes, std::memory_order_relaxed);
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
