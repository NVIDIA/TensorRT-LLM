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
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <sys/stat.h>
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
// is split into kDiskIoChunkBytes pieces; aligned writes use O_DIRECT so eviction writes skip the page cache.
namespace
{

constexpr ssize_t kDiskIoChunkBytes = ssize_t{4} << 20;

std::atomic<size_t> gAsyncDiskIoThreads{0};

// O_DIRECT descriptors, reopened via /proc/self/fd and keyed by fd and inode, since a closed pool file's fd
// number can be reused for another file.
struct DirectFdEntry
{
    int direct = 0;        // direct fd + 1; 0 = not opened, -1 = open failed
    bool rejected = false; // the file system rejected an aligned O_DIRECT write
    dev_t dev = 0;
    ino_t ino = 0;
};

std::mutex& directFdMutex()
{
    static std::mutex mutex;
    return mutex;
}

std::vector<DirectFdEntry>& directFdTable() // indexed by the buffered fd
{
    static std::vector<DirectFdEntry> table;
    return table;
}

int directFd(int fd) noexcept
{
    auto& mutex = directFdMutex();
    auto& table = directFdTable();
    struct stat st = {};
    if (fd < 0 || ::fstat(fd, &st) != 0)
    {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (static_cast<size_t>(fd) >= table.size())
    {
        table.resize(static_cast<size_t>(fd) + 1);
    }
    DirectFdEntry& entry = table[fd];
    if (entry.direct != 0 && (entry.dev != st.st_dev || entry.ino != st.st_ino))
    {
        if (entry.direct > 0)
        {
            ::close(entry.direct - 1);
        }
        entry = DirectFdEntry{};
    }
    if (entry.direct == 0)
    {
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
        int const direct = ::open(path, O_RDWR | O_DIRECT | O_CLOEXEC);
        entry.direct = direct >= 0 ? direct + 1 : -1;
        entry.dev = st.st_dev;
        entry.ino = st.st_ino;
        if (direct < 0)
        {
            TLLM_LOG_WARNING("[kvCacheManagerV2Utils] O_DIRECT reopen of fd %d failed (%s); using buffered I/O", fd,
                strerror(errno));
        }
    }
    return entry.direct > 0 && !entry.rejected ? entry.direct - 1 : -1;
}

// Other workers may still be using the direct fd, so it stays open until releaseDiskFile().
void rejectDirectFd(int fd) noexcept
{
    std::lock_guard<std::mutex> lock(directFdMutex());
    auto& table = directFdTable();
    if (static_cast<size_t>(fd) < table.size() && !table[fd].rejected)
    {
        table[fd].rejected = true;
        TLLM_LOG_WARNING("[kvCacheManagerV2Utils] O_DIRECT write to fd %d rejected; using buffered I/O", fd);
    }
}

constexpr uintptr_t kDirectAlign = 4096;

bool isDirectAligned(ssize_t pos, ssize_t size, void const* data) noexcept
{
    return pos >= 0 && size >= 0 && static_cast<uintptr_t>(pos) % kDirectAlign == 0
        && static_cast<uintptr_t>(size) % kDirectAlign == 0 && reinterpret_cast<uintptr_t>(data) % kDirectAlign == 0;
}

bool writePiece(int fd, ssize_t pos, void const* data, ssize_t size) noexcept
{
    if (isDirectAligned(pos, size, data))
    {
        if (int const direct = directFd(fd); direct >= 0)
        {
            if (writeAll(direct, pos, data, size))
            {
                return true;
            }
            if (errno != EINVAL)
            {
                return false;
            }
            rejectDirectFd(fd);
        }
    }
    return writeAll(fd, pos, data, size);
}

// Completion flags: a ring of words in pinned, device-mapped host memory. Transfer t uses word t % kFlagSlots
// and is complete once the word is >= t.
constexpr uint64_t kFlagSlots = uint64_t{1} << 16;

struct CompletionFlags
{
    std::atomic<uint64_t>* host = nullptr;
    CUdeviceptr device = 0;
    std::atomic<uint64_t> nextTicket{1};
};

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
        if (cuMemHostAlloc(&host, kFlagSlots * sizeof(uint64_t), CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP)
            != CUDA_SUCCESS)
        {
            TLLM_LOG_WARNING(
                "[kvCacheManagerV2Utils] cuMemHostAlloc for disk I/O flags failed; disk I/O completes "
                "in the stream callback");
            return nullptr;
        }
        std::memset(host, 0, kFlagSlots * sizeof(uint64_t));
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
        flags->host = static_cast<std::atomic<uint64_t>*>(host);
        flags->device = devicePtr;
        return flags;
    }();
    return instance;
}

struct DiskTransfer
{
    bool isRead = false;
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
            && writePiece(t.dst.fd, t.dst.pos + offset, reinterpret_cast<std::byte const*>(t.src) + offset, length);
        if (!ok)
        {
            // Recorded before finish() publishes completion, so whoever waits on the transfer sees the failure.
            recordFailedDiskWrite(t.dst.fd, t.dst.pos);
        }
        return ok;
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
            completionFlags()->host[ticket % kFlagSlots].store(ticket, std::memory_order_release);
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
            auto& queue = transfer->isRead ? mReads : mWrites;
            for (size_t task = 0; task < transfer->numTasks(); ++task)
            {
                for (ssize_t offset = 0; offset < transfer->numBytes; offset += chunk)
                {
                    queue.push_back(Piece{transfer, task, offset, std::min(chunk, transfer->numBytes - offset)});
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
    };

    void run()
    {
        for (;;)
        {
            Piece piece;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCv.wait(lock, [this] { return !mReads.empty() || !mWrites.empty(); });
                // Reads (restores) first: they are on the critical path.
                auto& queue = !mReads.empty() ? mReads : mWrites;
                piece = std::move(queue.front());
                queue.pop_front();
            }
            DiskTransfer& transfer = *piece.transfer;
            if (!transfer.runPiece(piece.task, piece.offset, piece.length))
            {
                transfer.ok.store(false);
            }
            if (transfer.remaining.fetch_sub(1) == 1)
            {
                transfer.finish();
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
    size_t mInFlight = 0; // pieces queued or running
    std::deque<Piece> mReads;
    std::deque<Piece> mWrites;
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

// Fallback without stream memory operations: still parallel, but the callback waits for the transfer.
CUDA_CB void hostFnRunDiskTransfer(void* userData) noexcept
{
    std::unique_ptr<DiskLaunch> const launch(static_cast<DiskLaunch*>(userData));
    auto const transfer = launch->transfer;
    diskIoPool().submit(transfer);
    while (transfer->remaining.load() != 0)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
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
    if (ticket > kFlagSlots)
    {
        // Never reuse a flag word whose previous transfer is still in flight.
        while (flags->host[ticket % kFlagSlots].load(std::memory_order_acquire) < ticket - kFlagSlots)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    transfer->ticket = ticket;
    auto launch = std::make_unique<DiskLaunch>(DiskLaunch{std::move(transfer)});
    if (CUresult const result = cuLaunchHostFunc(stream, hostFnSubmitDiskTransfer, launch.get());
        result != CUDA_SUCCESS)
    {
        // Retire the ticket, or the transfer that reuses its word after the ring wraps would wait forever.
        flags->host[ticket % kFlagSlots].store(ticket, std::memory_order_release);
        return result;
    }
    static_cast<void>(launch.release()); // owned by the callback now
    return cuStreamWaitValue64(
        stream, flags->device + (ticket % kFlagSlots) * sizeof(uint64_t), ticket, CU_STREAM_WAIT_VALUE_GEQ);
}

bool asyncDiskIo() noexcept
{
    return gAsyncDiskIoThreads.load(std::memory_order_acquire) > 0;
}

} // namespace

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

void releaseDiskFile(int fd) noexcept
{
    if (fd < 0)
    {
        return;
    }
    if (asyncDiskIo())
    {
        // Workers may still hold pieces for this file; let them finish before its descriptors go away.
        diskIoPool().waitIdle();
    }
    std::lock_guard<std::mutex> lock(directFdMutex());
    auto& table = directFdTable();
    if (static_cast<size_t>(fd) < table.size())
    {
        DirectFdEntry& entry = table[fd];
        if (entry.direct > 0)
        {
            ::close(entry.direct - 1);
        }
        entry = DirectFdEntry{};
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
