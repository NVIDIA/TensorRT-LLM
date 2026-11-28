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

// Host copy threads and busy-stream skipping of KVCacheManagerV2's copy path.
//
// Both settings are process-wide, so every test sets them through the testing hooks and restores the defaults. The
// async disk I/O pool cannot be stopped once started, so the one test that starts it comes last, and the synchronous
// disk tests skip themselves when a repeated or shuffled run has already started it.

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/copyUtils.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/cudaEvent.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

// 4 MiB a thread at least, so 16 threads need 64 MiB in one callback.
constexpr ssize_t kPageBytes = ssize_t{1} << 20;
constexpr size_t kNumPages = 96;

std::byte patternByte(size_t page, size_t offset, uint32_t seed)
{
    uint64_t x = (static_cast<uint64_t>(page) << 32) ^ offset ^ (static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 29;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 32;
    return static_cast<std::byte>(x & 0xFF);
}

void fillPattern(std::vector<std::byte>& buffer, size_t numPages, uint32_t seed)
{
    for (size_t page = 0; page < numPages; ++page)
    {
        for (size_t offset = 0; offset < static_cast<size_t>(kPageBytes); ++offset)
        {
            buffer[page * kPageBytes + offset] = patternByte(page, offset, seed);
        }
    }
}

// A page permutation, so no destination page is the source page of the same index.
size_t permute(size_t page)
{
    return (page * 37 + 11) % kNumPages;
}

std::vector<Task<MemAddress, MemAddress>> hostTasks(std::vector<std::byte>& dst, std::vector<std::byte> const& src)
{
    std::vector<Task<MemAddress, MemAddress>> tasks;
    for (size_t page = 0; page < kNumPages; ++page)
    {
        tasks.push_back({reinterpret_cast<MemAddress>(dst.data() + permute(page) * kPageBytes),
            reinterpret_cast<MemAddress>(src.data() + page * kPageBytes)});
    }
    return tasks;
}

void expectPermuted(std::vector<std::byte> const& dst, std::vector<std::byte> const& src)
{
    for (size_t page = 0; page < kNumPages; ++page)
    {
        ASSERT_EQ(std::memcmp(dst.data() + permute(page) * kPageBytes, src.data() + page * kPageBytes, kPageBytes), 0)
            << "page " << page;
    }
}

class TempFile
{
public:
    explicit TempFile(size_t bytes)
    {
        char name[] = "/tmp/kvcm2_copy_utils_XXXXXX";
        mFd = ::mkstemp(name);
        EXPECT_GE(mFd, 0);
        ::unlink(name);
        EXPECT_EQ(::ftruncate(mFd, static_cast<off_t>(bytes)), 0);
    }

    ~TempFile()
    {
        if (mFd >= 0)
        {
            ::close(mFd);
        }
    }

    TempFile(TempFile const&) = delete;
    TempFile& operator=(TempFile const&) = delete;

    [[nodiscard]] int fd() const noexcept
    {
        return mFd;
    }

private:
    int mFd = -1;
};

//! Synchronizes its streams when it goes out of scope. Declared after the buffers a test's copies use, it runs before
//! they are destroyed, so a fatal assertion that returns early cannot leave a queued copy reading or writing them.
class SyncOnExit
{
public:
    explicit SyncOnExit(std::vector<CUstream> streams)
        : mStreams(std::move(streams))
    {
    }

    ~SyncOnExit()
    {
        for (CUstream stream : mStreams)
        {
            EXPECT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
        }
    }

    SyncOnExit(SyncOnExit const&) = delete;
    SyncOnExit& operator=(SyncOnExit const&) = delete;

private:
    std::vector<CUstream> mStreams;
};

class KvCacheManagerV2CopyUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess); // make the primary context current
        ASSERT_EQ(cuStreamCreate(&mStream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
    }

    void TearDown() override
    {
        setHostCopyThreadsForTesting(1);
        setBusyStreamSkipForTesting(0);
        if (mStream != nullptr)
        {
            EXPECT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
            EXPECT_EQ(cuStreamDestroy(mStream), CUDA_SUCCESS);
        }
    }

    CUstream mStream = nullptr;
};

TEST_F(KvCacheManagerV2CopyUtilsTest, HostToHostIsBitwiseTheSameAtOneAndManyThreads)
{
    std::vector<std::byte> src(kNumPages * kPageBytes);
    fillPattern(src, kNumPages, 1);
    std::vector<std::byte> one(src.size(), std::byte{0});
    std::vector<std::byte> many(src.size(), std::byte{0});
    SyncOnExit const sync({mStream});

    ASSERT_EQ(setHostCopyThreadsForTesting(1), 1U);
    uint64_t const helpersBefore = hostCopyHelperTasksForTesting();
    ASSERT_EQ(copyHostToHost(hostTasks(one, src), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    // One thread: every task ran on the callback thread.
    EXPECT_EQ(hostCopyHelperTasksForTesting(), helpersBefore);

    ASSERT_GT(setHostCopyThreadsForTesting(16), 1U);
    // A helper counts its tasks before it publishes their completion, so a count read after the stream synchronized
    // is final. Whether a helper wakes before the callback thread has claimed every task is up to the OS scheduler,
    // so the copy is repeated (each time checked bitwise) until one did, a bounded number of times.
    constexpr int kMaxRounds = 20;
    int round = 0;
    for (; round < kMaxRounds && hostCopyHelperTasksForTesting() == helpersBefore; ++round)
    {
        std::fill(many.begin(), many.end(), std::byte{0});
        ASSERT_EQ(copyHostToHost(hostTasks(many, src), kPageBytes, mStream), CUDA_SUCCESS);
        ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
        ASSERT_EQ(std::memcmp(one.data(), many.data(), one.size()), 0) << "round " << round;
    }
    EXPECT_GT(hostCopyHelperTasksForTesting(), helpersBefore) << "no helper took a task in " << round << " rounds";

    expectPermuted(one, src);
}

struct CheckData
{
    std::byte const* data;
    std::byte const* expected;
    size_t bytes;
    std::atomic<int>* result;
};

CUDA_CB void checkOnStream(void* userData) noexcept
{
    auto const* check = static_cast<CheckData const*>(userData);
    check->result->store(std::memcmp(check->data, check->expected, check->bytes) == 0 ? 1 : 0);
}

TEST_F(KvCacheManagerV2CopyUtilsTest, CallbacksKeepStreamOrder)
{
    // Three copies back to back on one stream, each reading what the one before wrote, then a host function that sees
    // the result: each callback returns only once all of its tasks finished.
    ASSERT_GT(setHostCopyThreadsForTesting(8), 1U);
    std::vector<std::byte> a(kNumPages * kPageBytes);
    fillPattern(a, kNumPages, 2);
    std::vector<std::byte> b(a.size(), std::byte{0});
    std::vector<std::byte> c(a.size(), std::byte{0});
    std::vector<std::byte> d(a.size(), std::byte{0});
    std::vector<std::byte> expected(a.size());
    std::atomic<int> result{-1};
    SyncOnExit const sync({mStream});
    ASSERT_EQ(copyHostToHost(hostTasks(b, a), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(copyHostToHost(hostTasks(c, b), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(copyHostToHost(hostTasks(d, c), kPageBytes, mStream), CUDA_SUCCESS);

    for (size_t page = 0; page < kNumPages; ++page)
    {
        size_t const dst = permute(permute(permute(page)));
        std::memcpy(expected.data() + dst * kPageBytes, a.data() + page * kPageBytes, kPageBytes);
    }
    CheckData check{d.data(), expected.data(), d.size(), &result};
    ASSERT_EQ(cuLaunchHostFunc(mStream, checkOnStream, &check), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    EXPECT_EQ(result.load(), 1);
}

TEST_F(KvCacheManagerV2CopyUtilsTest, CallbacksFromManyStreamsShareThePool)
{
    // Several streams issue large callbacks at once; CUDA runs them one at a time, and each one gets its own job.
    ASSERT_GT(setHostCopyThreadsForTesting(16), 1U);
    constexpr int kStreams = 4;
    std::vector<std::vector<std::byte>> srcs(kStreams);
    std::vector<std::vector<std::byte>> dsts(kStreams);
    std::vector<CUstream> streams;
    // Destroyed after their last copy and before the buffers, on every exit.
    auto const destroyStreams = FuncGuard(
        [&streams]()
        {
            for (CUstream stream : streams)
            {
                EXPECT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
                EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
            }
        });
    for (int s = 0; s < kStreams; ++s)
    {
        CUstream stream = nullptr;
        ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
        streams.push_back(stream);
        srcs[s].resize(kNumPages * kPageBytes);
        fillPattern(srcs[s], kNumPages, 10 + s);
        dsts[s].assign(srcs[s].size(), std::byte{0});
    }
    for (int round = 0; round < 3; ++round)
    {
        for (int s = 0; s < kStreams; ++s)
        {
            ASSERT_EQ(copyHostToHost(hostTasks(dsts[s], srcs[s]), kPageBytes, streams[s]), CUDA_SUCCESS);
        }
    }
    for (int s = 0; s < kStreams; ++s)
    {
        ASSERT_EQ(cuStreamSynchronize(streams[s]), CUDA_SUCCESS);
        expectPermuted(dsts[s], srcs[s]);
    }
}

TEST_F(KvCacheManagerV2CopyUtilsTest, SmallCallbacksStayOnTheCallbackThread)
{
    ASSERT_GT(setHostCopyThreadsForTesting(16), 1U);
    constexpr size_t kSmallPages = 4; // 4 MiB in all: below two threads' minimum
    std::vector<std::byte> src(kNumPages * kPageBytes);
    fillPattern(src, kNumPages, 3);
    std::vector<std::byte> dst(src.size(), std::byte{0});
    SyncOnExit const sync({mStream});
    auto tasks = hostTasks(dst, src);
    tasks.resize(kSmallPages);
    uint64_t const helpersBefore = hostCopyHelperTasksForTesting();
    ASSERT_EQ(copyHostToHost(tasks, kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    EXPECT_EQ(hostCopyHelperTasksForTesting(), helpersBefore);
    for (size_t page = 0; page < kSmallPages; ++page)
    {
        EXPECT_EQ(std::memcmp(dst.data() + permute(page) * kPageBytes, src.data() + page * kPageBytes, kPageBytes), 0);
    }
}

TEST_F(KvCacheManagerV2CopyUtilsTest, SynchronousDiskCopiesAtOneAndManyThreads)
{
    // Without the async disk I/O pool, host <-> disk copies run in the host function. The pool cannot be stopped, so
    // a repeated or shuffled run that already started it skips this test.
    if (asyncDiskIoEnabled())
    {
        GTEST_SKIP() << "the async disk I/O pool already runs in this process";
    }
    std::vector<std::byte> src(kNumPages * kPageBytes);
    fillPattern(src, kNumPages, 4);
    for (size_t const threads : {size_t{1}, size_t{16}})
    {
        SCOPED_TRACE("threads " + std::to_string(threads));
        setHostCopyThreadsForTesting(threads);
        TempFile first(kNumPages * kPageBytes);
        TempFile second(kNumPages * kPageBytes);
        std::vector<Task<DiskAddress, MemAddress>> writes;
        std::vector<Task<DiskAddress, DiskAddress>> moves;
        std::vector<Task<MemAddress, DiskAddress>> reads;
        std::vector<std::byte> back(src.size(), std::byte{0});
        SyncOnExit const sync({mStream});
        for (size_t page = 0; page < kNumPages; ++page)
        {
            writes.push_back({DiskAddress{first.fd(), static_cast<ssize_t>(permute(page) * kPageBytes)},
                reinterpret_cast<MemAddress>(src.data() + page * kPageBytes)});
            // first -> second keeps the page offsets, second -> host undoes the permutation.
            moves.push_back({DiskAddress{second.fd(), static_cast<ssize_t>(page * kPageBytes)},
                DiskAddress{first.fd(), static_cast<ssize_t>(page * kPageBytes)}});
            reads.push_back({reinterpret_cast<MemAddress>(back.data() + page * kPageBytes),
                DiskAddress{second.fd(), static_cast<ssize_t>(permute(page) * kPageBytes)}});
        }
        uint64_t const helpersBefore = hostCopyHelperTasksForTesting();
        ASSERT_EQ(copyHostToDisk(std::move(writes), kPageBytes, mStream), CUDA_SUCCESS);
        ASSERT_EQ(copyDiskToDisk(std::move(moves), kPageBytes, mStream), CUDA_SUCCESS);
        ASSERT_EQ(copyDiskToHost(std::move(reads), kPageBytes, mStream), CUDA_SUCCESS);
        ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
        EXPECT_EQ(std::memcmp(back.data(), src.data(), src.size()), 0);
        if (threads == 1)
        {
            EXPECT_EQ(hostCopyHelperTasksForTesting(), helpersBefore);
        }
        else
        {
            EXPECT_GT(hostCopyHelperTasksForTesting(), helpersBefore);
        }
        for (size_t page = 0; page < kNumPages; ++page)
        {
            EXPECT_FALSE(consumeFailedDiskWrite(first.fd(), static_cast<ssize_t>(page * kPageBytes)));
        }
    }
}

TEST_F(KvCacheManagerV2CopyUtilsTest, FailedSynchronousWritesAreRecordedPerPage)
{
    if (asyncDiskIoEnabled())
    {
        GTEST_SKIP() << "the async disk I/O pool already runs in this process";
    }
    ASSERT_GT(setHostCopyThreadsForTesting(16), 1U);
    std::vector<std::byte> src(kNumPages * kPageBytes);
    fillPattern(src, kNumPages, 5);
    TempFile file(kNumPages * kPageBytes);
    SyncOnExit const sync({mStream});
    std::vector<Task<DiskAddress, MemAddress>> writes;
    for (size_t page = 0; page < kNumPages; ++page)
    {
        writes.push_back({DiskAddress{file.fd(), static_cast<ssize_t>(page * kPageBytes)},
            reinterpret_cast<MemAddress>(src.data() + page * kPageBytes)});
    }
    constexpr uint64_t kFailures = 3;
    injectDiskWriteFailuresForTesting(kFailures);
    ASSERT_EQ(copyHostToDisk(std::move(writes), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    // With threads, a failed page does not stop the others: exactly the injected failures are recorded, and every
    // other page holds its bytes.
    uint64_t failed = 0;
    std::vector<std::byte> page(kPageBytes);
    for (size_t p = 0; p < kNumPages; ++p)
    {
        if (consumeFailedDiskWrite(file.fd(), static_cast<ssize_t>(p * kPageBytes)))
        {
            ++failed;
            continue;
        }
        ASSERT_EQ(::pread(file.fd(), page.data(), kPageBytes, static_cast<off_t>(p * kPageBytes)), kPageBytes);
        EXPECT_EQ(std::memcmp(page.data(), src.data() + p * kPageBytes, kPageBytes), 0) << "page " << p;
    }
    EXPECT_EQ(failed, kFailures);
}

std::atomic<bool> gReleaseStreams{false};
// A protective deadline, far beyond the test's own run time: a hold that is never released still ends.
constexpr auto kMaxStreamHold = std::chrono::seconds(120);

CUDA_CB void holdStream(void*) noexcept
{
    auto const deadline = std::chrono::steady_clock::now() + kMaxStreamHold;
    while (!gReleaseStreams.load() && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
    }
}

TEST_F(KvCacheManagerV2CopyUtilsTest, TemporaryStreamsSkipBusyPooledStreams)
{
    // Make every cached pooled stream busy: each one waits behind a host function that holds the callback thread.
    auto& pool = CudaStreamPool::instance();
    gReleaseStreams.store(false);
    std::vector<CUstream> busy;
    // Releases the holds on every exit, a failed assertion included, and only then waits for the busy streams.
    auto const release = FuncGuard(
        [&busy]()
        {
            gReleaseStreams.store(true);
            for (CUstream stream : busy)
            {
                EXPECT_EQ(cuStreamSynchronize(stream), CUDA_SUCCESS);
            }
        });
    {
        // The bounded check below needs two busy pooled streams at least: with one, the bounded acquisition would
        // take the only busy stream and then hand out the idle one.
        std::vector<CudaStreamPool::PoolItem> warm;
        warm.push_back(pool.get());
        warm.push_back(pool.get());
    }
    std::vector<CudaStreamPool::PoolItem> taken;
    int const cached = pool.cachedCount();
    ASSERT_GE(cached, 2);
    for (int i = 0; i < cached; ++i)
    {
        taken.push_back(pool.get());
        ASSERT_EQ(cuLaunchHostFunc(taken.back().get(), holdStream, nullptr), CUDA_SUCCESS);
        busy.push_back(taken.back().get());
    }
    taken.clear(); // back to the pool, still busy

    {
        // Off: the next pooled stream, busy as it is.
        TemporaryCudaStream plain({});
        EXPECT_EQ(cuStreamQuery(plain.get()), CUDA_ERROR_NOT_READY);
    }
    setBusyStreamSkipForTesting(4096);
    uint64_t const skippedBefore = busyStreamsSkippedForTesting();
    {
        TemporaryCudaStream idle({});
        EXPECT_EQ(cuStreamQuery(idle.get()), CUDA_SUCCESS);
        EXPECT_EQ(std::find(busy.begin(), busy.end(), idle.get()), busy.end());
    }
    EXPECT_GE(busyStreamsSkippedForTesting() - skippedBefore, static_cast<uint64_t>(cached));
    // A bound below the busy count hands out the last stream it took, busy or not.
    setBusyStreamSkipForTesting(1);
    {
        TemporaryCudaStream bounded({});
        EXPECT_EQ(cuStreamQuery(bounded.get()), CUDA_ERROR_NOT_READY);
    }
}

// Last: the async disk I/O pool runs for the rest of the process once started.
TEST_F(KvCacheManagerV2CopyUtilsTest, ZzHostCopiesComposeWithTheAsyncDiskPool)
{
    configureAsyncDiskIo(4);
    ASSERT_TRUE(asyncDiskIoEnabled());
    ASSERT_GT(setHostCopyThreadsForTesting(16), 1U);
    std::vector<std::byte> src(kNumPages * kPageBytes);
    fillPattern(src, kNumPages, 6);
    std::vector<std::byte> mid(src.size(), std::byte{0});
    std::vector<std::byte> back(src.size(), std::byte{0});
    TempFile file(kNumPages * kPageBytes);
    SyncOnExit const sync({mStream});
    std::vector<Task<DiskAddress, MemAddress>> writes;
    std::vector<Task<MemAddress, DiskAddress>> reads;
    for (size_t page = 0; page < kNumPages; ++page)
    {
        writes.push_back({DiskAddress{file.fd(), static_cast<ssize_t>(page * kPageBytes)},
            reinterpret_cast<MemAddress>(mid.data() + page * kPageBytes)});
        reads.push_back({reinterpret_cast<MemAddress>(back.data() + page * kPageBytes),
            DiskAddress{file.fd(), static_cast<ssize_t>(page * kPageBytes)}});
    }
    // host -> host on the host copy threads, then host -> disk and disk -> host on the I/O pool, on one stream: the
    // disk pieces are not run by the host copy threads, and each copy sees the one before it.
    ASSERT_EQ(copyHostToHost(hostTasks(mid, src), kPageBytes, mStream), CUDA_SUCCESS);
    uint64_t const helpersAfterHostCopy = [&]
    {
        EXPECT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
        return hostCopyHelperTasksForTesting();
    }();
    ASSERT_EQ(copyHostToDisk(std::move(writes), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(copyDiskToHost(std::move(reads), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    EXPECT_EQ(hostCopyHelperTasksForTesting(), helpersAfterHostCopy);
    EXPECT_EQ(std::memcmp(back.data(), mid.data(), mid.size()), 0);
    expectPermuted(back, src);
    waitForDiskIo();
}

// Also after the pool started. Page A is staged and its flush held back, so it fills the staging; page B is then
// written to the same offset straight to the file. B must be what a read returns, and A's flush, once released, must
// not land over it.
TEST_F(KvCacheManagerV2CopyUtilsTest, ZzDirectWriteRetiresAnOlderStagedCopy)
{
    configureAsyncDiskIo(4);
    ASSERT_TRUE(asyncDiskIoEnabled());
    std::vector<std::byte> a(kPageBytes);
    std::vector<std::byte> b(kPageBytes);
    fillPattern(a, 1, 7);
    fillPattern(b, 1, 8);
    std::vector<std::byte> back(kPageBytes, std::byte{0});
    TempFile file(kPageBytes);
    SyncOnExit const sync({mStream});
    waitForDiskIo();
    size_t const stagedBefore = diskWriteBehindPagesForTesting();
    auto const restore = FuncGuard(
        []()
        {
            delayDiskWriteBehindFlushesForTesting(0);
            waitForDiskIo();
            configureDiskWriteBehind(0);
        });
    configureDiskWriteBehind(kPageBytes); // room for one page: A takes all of it
    delayDiskWriteBehindFlushesForTesting(3600);
    auto const writePage = [&](std::vector<std::byte> const& src)
    {
        std::vector<Task<DiskAddress, MemAddress>> writes{
            {DiskAddress{file.fd(), 0}, reinterpret_cast<MemAddress>(src.data())}};
        return copyHostToDisk(std::move(writes), kPageBytes, mStream);
    };
    ASSERT_EQ(writePage(a), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    ASSERT_EQ(diskWriteBehindPagesForTesting(), stagedBefore + 1) << "A was not staged";
    ASSERT_EQ(writePage(b), CUDA_SUCCESS); // no staging left: B goes to the file
    std::vector<Task<MemAddress, DiskAddress>> reads{
        {reinterpret_cast<MemAddress>(back.data()), DiskAddress{file.fd(), 0}}};
    ASSERT_EQ(copyDiskToHost(std::move(reads), kPageBytes, mStream), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamSynchronize(mStream), CUDA_SUCCESS);
    EXPECT_EQ(std::memcmp(back.data(), b.data(), back.size()), 0) << "the read returned the older staged page";
    EXPECT_EQ(diskWriteBehindPagesForTesting(), stagedBefore);
    delayDiskWriteBehindFlushesForTesting(0);
    waitForDiskIo();
    ASSERT_EQ(::pread(file.fd(), back.data(), kPageBytes, 0), kPageBytes);
    EXPECT_EQ(std::memcmp(back.data(), b.data(), back.size()), 0) << "the older page's flush landed over the newer one";
}

} // namespace
