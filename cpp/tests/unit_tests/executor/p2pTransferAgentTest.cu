/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// P2P Transfer Agent Tests
// Covers BatchCopyWorkerPool, P2pTransferStatus (single- and multi-event modes),
// CudaEventPool, P2pTransferContext submit paths, and P2pRemoteMappingRegistry
// thread safety. Requires at least one CUDA GPU.

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/p2pTransferAgent.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

using namespace tensorrt_llm::executor::kv_cache;
namespace runtime = tensorrt_llm::runtime;

class P2pTransferAgentTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int deviceCount = 0;
        ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
        ASSERT_GT(deviceCount, 0) << "No CUDA device available";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    }

    void* gpuAllocPattern(size_t bytes, uint8_t pattern)
    {
        void* ptr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&ptr, bytes));
        TLLM_CUDA_CHECK(cudaMemset(ptr, pattern, bytes));
        mAllocations.push_back(ptr);
        return ptr;
    }

    void* gpuAllocZeroed(size_t bytes)
    {
        return gpuAllocPattern(bytes, 0);
    }

    void TearDown() override
    {
        for (auto* ptr : mAllocations)
        {
            cudaFree(ptr);
        }
        mAllocations.clear();
    }

private:
    std::vector<void*> mAllocations;
};

// ============================================================================
// BatchCopyWorkerPool
// ============================================================================

TEST_F(P2pTransferAgentTest, WorkerPoolBasicSubmitAndWait)
{
    constexpr int kNumWorkers = 2;
    constexpr size_t kSize = 4096;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    auto* src = gpuAllocPattern(kSize, 0xAB);
    auto* dst = gpuAllocZeroed(kSize);

    auto batchPending = std::make_shared<std::atomic<int>>(1);
    auto event = std::make_shared<runtime::CudaEvent>();
    auto stream = std::make_shared<runtime::CudaStream>();

    BatchCopyTask task;
    task.dst = {static_cast<void*>(dst)};
    task.src = {static_cast<void const*>(src)};
    task.sizes = {kSize};
    task.stream = stream->get();
    task.completionEvent = event;
    task.batchPending = batchPending;

    pool.submit(std::move(task));
    pool.waitAll();

    EXPECT_EQ(batchPending->load(), 0);

    event->synchronize();

    std::vector<uint8_t> hostBuf(kSize);
    TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dst, kSize, cudaMemcpyDeviceToHost));
    for (size_t ii = 0; ii < kSize; ++ii)
    {
        ASSERT_EQ(hostBuf[ii], 0xAB) << "Mismatch at byte " << ii;
    }
}

TEST_F(P2pTransferAgentTest, WorkerPoolMultipleTasks)
{
    constexpr int kNumWorkers = 2;
    constexpr int kNumTasks = 4;
    constexpr size_t kSize = 1024;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    auto batchPending = std::make_shared<std::atomic<int>>(kNumTasks);
    std::vector<std::shared_ptr<runtime::CudaEvent>> events;
    std::vector<std::shared_ptr<runtime::CudaStream>> streams;
    std::vector<void*> srcs, dsts;

    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        srcs.push_back(gpuAllocPattern(kSize, static_cast<uint8_t>(ii + 1)));
        dsts.push_back(gpuAllocZeroed(kSize));
        events.push_back(std::make_shared<runtime::CudaEvent>());
        streams.push_back(std::make_shared<runtime::CudaStream>());
    }

    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        BatchCopyTask task;
        task.dst = {dsts[ii]};
        task.src = {static_cast<void const*>(srcs[ii])};
        task.sizes = {kSize};
        task.stream = streams[ii]->get();
        task.completionEvent = events[ii];
        task.batchPending = batchPending;
        pool.submit(std::move(task));
    }

    pool.waitAll();
    EXPECT_EQ(batchPending->load(), 0);

    for (int ii = 0; ii < kNumTasks; ++ii)
    {
        events[ii]->synchronize();
        std::vector<uint8_t> hostBuf(kSize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dsts[ii], kSize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Task " << ii << " data mismatch";
    }
}

TEST_F(P2pTransferAgentTest, WorkerPoolPerBatchIsolation)
{
    constexpr int kNumWorkers = 2;
    constexpr size_t kSize = 1024;

    BatchCopyWorkerPool pool(kNumWorkers, 0);

    auto pendingA = std::make_shared<std::atomic<int>>(2);
    std::vector<std::shared_ptr<runtime::CudaEvent>> eventsA;
    std::vector<std::shared_ptr<runtime::CudaStream>> streamsA;

    for (int ii = 0; ii < 2; ++ii)
    {
        eventsA.push_back(std::make_shared<runtime::CudaEvent>());
        streamsA.push_back(std::make_shared<runtime::CudaStream>());

        BatchCopyTask task;
        auto* src = gpuAllocPattern(kSize, 0xAA);
        auto* dst = gpuAllocZeroed(kSize);
        task.dst = {dst};
        task.src = {static_cast<void const*>(src)};
        task.sizes = {kSize};
        task.stream = streamsA[ii]->get();
        task.completionEvent = eventsA[ii];
        task.batchPending = pendingA;
        pool.submit(std::move(task));
    }

    auto pendingB = std::make_shared<std::atomic<int>>(2);
    std::vector<std::shared_ptr<runtime::CudaEvent>> eventsB;
    std::vector<std::shared_ptr<runtime::CudaStream>> streamsB;

    for (int ii = 0; ii < 2; ++ii)
    {
        eventsB.push_back(std::make_shared<runtime::CudaEvent>());
        streamsB.push_back(std::make_shared<runtime::CudaStream>());

        BatchCopyTask task;
        auto* src = gpuAllocPattern(kSize, 0xBB);
        auto* dst = gpuAllocZeroed(kSize);
        task.dst = {dst};
        task.src = {static_cast<void const*>(src)};
        task.sizes = {kSize};
        task.stream = streamsB[ii]->get();
        task.completionEvent = eventsB[ii];
        task.batchPending = pendingB;
        pool.submit(std::move(task));
    }

    pool.waitAll();

    EXPECT_EQ(pendingA->load(), 0);
    EXPECT_EQ(pendingB->load(), 0);
}

// ============================================================================
// P2pTransferStatus
// ============================================================================

TEST_F(P2pTransferAgentTest, StatusSingleEventMode)
{
    auto stream = std::make_shared<runtime::CudaStream>();
    auto event = std::make_shared<runtime::CudaEvent>();
    stream->record(*event);

    P2pTransferStatus status(stream, event);
    auto result = status.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
    EXPECT_TRUE(status.isCompleted());
}

TEST_F(P2pTransferAgentTest, StatusMultiEventMode)
{
    constexpr int kNumEvents = 3;

    auto batchPending = std::make_shared<std::atomic<int>>(kNumEvents);
    std::vector<std::shared_ptr<runtime::CudaEvent>> events;
    std::vector<std::shared_ptr<runtime::CudaStream>> streams;

    for (int ii = 0; ii < kNumEvents; ++ii)
    {
        events.push_back(std::make_shared<runtime::CudaEvent>());
        streams.push_back(std::make_shared<runtime::CudaStream>());
    }

    P2pTransferStatus status(batchPending, events);

    // batchPending > 0 → events haven't been recorded yet
    EXPECT_FALSE(status.isCompleted());

    for (int ii = 0; ii < kNumEvents; ++ii)
    {
        streams[ii]->record(*events[ii]);
        batchPending->fetch_sub(1, std::memory_order_release);
    }

    auto result = status.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
    EXPECT_TRUE(status.isCompleted());
}

TEST_F(P2pTransferAgentTest, StatusMultiEventTimedWait)
{
    auto stream = std::make_shared<runtime::CudaStream>();

    constexpr size_t kLargeSize = 256 * 1024 * 1024; // 256MB
    void* buf = gpuAllocZeroed(kLargeSize);
    TLLM_CUDA_CHECK(cudaMemsetAsync(buf, 0xFF, kLargeSize, stream->get()));

    auto event = std::make_shared<runtime::CudaEvent>();
    stream->record(*event);

    auto batchPending = std::make_shared<std::atomic<int>>(0);
    P2pTransferStatus status(batchPending, {event});

    // Very short timeout: may be IN_PROGRESS or SUCCESS depending on GPU speed
    auto result = status.wait(0);
    EXPECT_TRUE(result == TransferState::kIN_PROGRESS || result == TransferState::kSUCCESS);

    // Infinite wait always succeeds
    result = status.wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
}

// ============================================================================
// P2pTransferContext::submitWithMemcpyBatch
// ============================================================================

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchEmptyInput)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;

    auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);
}

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchSingleEntry)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr size_t kSize = 8192;
    auto* src = gpuAllocPattern(kSize, 0xCD);
    auto* dst = gpuAllocZeroed(kSize);

    std::vector<void*> srcPtrs = {src};
    std::vector<void*> dstPtrs = {dst};
    std::vector<size_t> sizes = {kSize};

    auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    std::vector<uint8_t> hostBuf(kSize);
    TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dst, kSize, cudaMemcpyDeviceToHost));
    for (size_t ii = 0; ii < kSize; ++ii)
    {
        ASSERT_EQ(hostBuf[ii], 0xCD) << "Mismatch at byte " << ii;
    }
}

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchMultipleEntries)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 8;
    constexpr size_t kEntrySize = 16 * 1024;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(gpuAllocPattern(kEntrySize, static_cast<uint8_t>(ii + 1)));
        dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto result = status->wait(-1);
    EXPECT_EQ(result, TransferState::kSUCCESS);

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " first byte";
        EXPECT_EQ(hostBuf[kEntrySize - 1], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " last byte";
    }
}

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchLargeEntries)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 256 * 1024;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(gpuAllocPattern(kEntrySize, static_cast<uint8_t>(0x10 + ii)));
        dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS);

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(0x10 + ii)) << "Entry " << ii;
    }
}

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchConsecutiveCalls)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 4096;

    for (int batch = 0; batch < 2; ++batch)
    {
        std::vector<void*> srcPtrs, dstPtrs;
        std::vector<size_t> sizes;
        uint8_t pattern = static_cast<uint8_t>(0xA0 + batch);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            srcPtrs.push_back(gpuAllocPattern(kEntrySize, pattern));
            dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
            sizes.push_back(kEntrySize);
        }

        auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS) << "Batch " << batch;

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Batch " << batch << " entry " << ii;
        }
    }
}

// Two threads submit concurrently — each has its own P2pTransferContext
// (per-thread stream + prealloc buffers), so there is no global mBufferMutex serialization.
TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchConcurrentFromTwoThreads)
{
    P2pTransferAgent agent;

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 8192;

    auto runBatch = [&agent](uint8_t pattern, std::vector<void*> const& srcs, std::vector<void*> const& dsts)
    {
        auto& ctx = agent.contextForCurrentThread();
        std::vector<void*> srcPtrs(srcs.begin(), srcs.end());
        std::vector<void*> dstPtrs(dsts.begin(), dsts.end());
        std::vector<size_t> sizes(kNumEntries, kEntrySize);

        auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Pattern 0x" << std::hex << (int) pattern << " entry " << ii;
        }
    };

    std::vector<void*> srcsA, dstsA, srcsB, dstsB;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcsA.push_back(gpuAllocPattern(kEntrySize, 0xAA));
        dstsA.push_back(gpuAllocZeroed(kEntrySize));
        srcsB.push_back(gpuAllocPattern(kEntrySize, 0xBB));
        dstsB.push_back(gpuAllocZeroed(kEntrySize));
    }

    std::thread threadA(
        [&]()
        {
            TLLM_CUDA_CHECK(cudaSetDevice(0));
            runBatch(0xAA, srcsA, dstsA);
        });
    std::thread threadB(
        [&]()
        {
            TLLM_CUDA_CHECK(cudaSetDevice(0));
            runBatch(0xBB, srcsB, dstsB);
        });

    threadA.join();
    threadB.join();
}

TEST_F(P2pTransferAgentTest, SubmitMemcpyBatchIsCompletedPolling)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 4;
    constexpr size_t kEntrySize = 4096;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(gpuAllocPattern(kEntrySize, 0xEE));
        dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = ctx.submitWithMemcpyBatch(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!status->isCompleted())
    {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Timed out waiting for isCompleted()";
        std::this_thread::yield();
    }
    EXPECT_TRUE(status->isCompleted());
}

// ============================================================================
// CudaEventPool
// ============================================================================

TEST_F(P2pTransferAgentTest, EventPoolAcquireAndReuse)
{
    auto pool = std::make_shared<CudaEventPool>();

    cudaEvent_t firstHandle;
    {
        auto event = pool->acquire();
        firstHandle = event->get();
    }

    auto event2 = pool->acquire();
    EXPECT_EQ(event2->get(), firstHandle) << "Pool should reuse the same CUDA event";
}

TEST_F(P2pTransferAgentTest, EventPoolConcurrentAccess)
{
    auto pool = std::make_shared<CudaEventPool>();
    constexpr int kNumThreads = 8;
    constexpr int kNumCycles = 100;

    std::atomic<int> totalAcquires{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back(
            [&pool, &totalAcquires]()
            {
                for (int i = 0; i < kNumCycles; ++i)
                {
                    auto event = pool->acquire();
                    ASSERT_NE(event, nullptr);
                    totalAcquires.fetch_add(1, std::memory_order_relaxed);
                }
            });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(totalAcquires.load(), kNumThreads * kNumCycles);
}

TEST_F(P2pTransferAgentTest, EventPoolDestroyedBeforeEventsReturned)
{
    auto pool = std::make_shared<CudaEventPool>();
    auto event = pool->acquire();
    ASSERT_NE(event, nullptr);

    auto stream = std::make_shared<runtime::CudaStream>();

    // Destroy pool while event is still alive — weak_ptr in deleter expires gracefully
    pool.reset();

    stream->record(*event);
    event->synchronize();
    event.reset();
}

// ============================================================================
// P2pTransferContext::submitWithCubBatched (merged pinned H2D path)
// ============================================================================

TEST_F(P2pTransferAgentTest, SubmitCubBatchedMergedH2D)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 16;
    constexpr size_t kEntrySize = 4096;

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;
    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        srcPtrs.push_back(gpuAllocPattern(kEntrySize, static_cast<uint8_t>(ii + 1)));
        dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
        sizes.push_back(kEntrySize);
    }

    auto status = ctx.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS);

    for (int ii = 0; ii < kNumEntries; ++ii)
    {
        std::vector<uint8_t> hostBuf(kEntrySize);
        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
        EXPECT_EQ(hostBuf[0], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " first byte";
        EXPECT_EQ(hostBuf[kEntrySize - 1], static_cast<uint8_t>(ii + 1)) << "Entry " << ii << " last byte";
    }
}

TEST_F(P2pTransferAgentTest, SubmitCubBatchedEmptyInput)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    std::vector<void*> srcPtrs, dstPtrs;
    std::vector<size_t> sizes;
    auto status = ctx.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
    ASSERT_NE(status, nullptr);
    EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS);
}

TEST_F(P2pTransferAgentTest, SubmitCubBatchedConsecutiveCalls)
{
    P2pTransferAgent agent;
    auto& ctx = agent.contextForCurrentThread();

    constexpr int kNumEntries = 8;
    constexpr size_t kEntrySize = 2048;

    for (int batch = 0; batch < 2; ++batch)
    {
        std::vector<void*> srcPtrs, dstPtrs;
        std::vector<size_t> sizes;
        uint8_t pattern = static_cast<uint8_t>(0xC0 + batch);

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            srcPtrs.push_back(gpuAllocPattern(kEntrySize, pattern));
            dstPtrs.push_back(gpuAllocZeroed(kEntrySize));
            sizes.push_back(kEntrySize);
        }

        auto status = ctx.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
        ASSERT_NE(status, nullptr);
        EXPECT_EQ(status->wait(-1), TransferState::kSUCCESS) << "Batch " << batch;

        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            std::vector<uint8_t> hostBuf(kEntrySize);
            TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
            EXPECT_EQ(hostBuf[0], pattern) << "Batch " << batch << " entry " << ii;
        }
    }
}

// Verifies that N caller threads running submitWithCubBatched concurrently never contend
// on a shared buffer — each thread gets its own P2pTransferContext, independent stream
// and prealloc buffers.
TEST_F(P2pTransferAgentTest, SubmitCubBatchedConcurrentPerThreadContexts)
{
    P2pTransferAgent agent;

    constexpr int kNumThreads = 4;
    constexpr int kNumEntries = 8;
    constexpr size_t kEntrySize = 2048;
    constexpr int kIterations = 8;

    std::vector<std::thread> threads;
    std::atomic<int> failures{0};

    // Pre-allocate data on main thread (CUDA allocations don't need per-thread context)
    std::vector<std::vector<void*>> srcAll(kNumThreads), dstAll(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t)
    {
        for (int ii = 0; ii < kNumEntries; ++ii)
        {
            srcAll[t].push_back(gpuAllocPattern(kEntrySize, static_cast<uint8_t>(0xD0 + t)));
            dstAll[t].push_back(gpuAllocZeroed(kEntrySize));
        }
    }

    for (int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back(
            [&agent, t, &srcAll, &dstAll, &failures]()
            {
                TLLM_CUDA_CHECK(cudaSetDevice(0));
                auto& ctx = agent.contextForCurrentThread();
                uint8_t pattern = static_cast<uint8_t>(0xD0 + t);

                for (int iter = 0; iter < kIterations; ++iter)
                {
                    std::vector<void*> srcPtrs = srcAll[t];
                    std::vector<void*> dstPtrs = dstAll[t];
                    std::vector<size_t> sizes(kNumEntries, kEntrySize);

                    auto status = ctx.submitWithCubBatched(srcPtrs, dstPtrs, sizes);
                    if (!status || status->wait(-1) != TransferState::kSUCCESS)
                    {
                        failures.fetch_add(1);
                        return;
                    }

                    for (int ii = 0; ii < kNumEntries; ++ii)
                    {
                        std::vector<uint8_t> hostBuf(kEntrySize);
                        TLLM_CUDA_CHECK(cudaMemcpy(hostBuf.data(), dstPtrs[ii], kEntrySize, cudaMemcpyDeviceToHost));
                        if (hostBuf[0] != pattern || hostBuf[kEntrySize - 1] != pattern)
                        {
                            failures.fetch_add(1);
                            return;
                        }
                    }
                }
            });
    }

    for (auto& th : threads)
    {
        th.join();
    }

    EXPECT_EQ(failures.load(), 0);
}

// ============================================================================
// P2pRemoteMappingRegistry thread safety
// ============================================================================

// Reader threads (hasMapping / hasImportFailed / get) and writer threads
// (importAndMap with empty P2pMemInfo → failure path + cleanup) run concurrently.
// We only care about no crashes / data-races; the shared_mutex + shared_ptr design
// must allow concurrent readers while writers mutate under unique_lock.
TEST_F(P2pTransferAgentTest, RemoteMappingConcurrentReadWrite)
{
    P2pTransferAgent agent;
    auto& registry = agent.registry();

    constexpr int kNumReaders = 4;
    constexpr int kNumWriters = 2;
    constexpr int kIterations = 500;

    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;

    for (int r = 0; r < kNumReaders; ++r)
    {
        threads.emplace_back(
            [&registry, &stop, r]()
            {
                while (!stop.load(std::memory_order_acquire))
                {
                    std::string name = "remote_" + std::to_string(r % 4);
                    [[maybe_unused]] auto has = registry.hasMapping(name);
                    [[maybe_unused]] auto failed = registry.hasImportFailed(name);
                    [[maybe_unused]] auto mapping = registry.get(name);
                    std::this_thread::yield();
                }
            });
    }

    for (int w = 0; w < kNumWriters; ++w)
    {
        threads.emplace_back(
            [&registry, w]()
            {
                for (int i = 0; i < kIterations; ++i)
                {
                    std::string name = "remote_" + std::to_string(w);
                    // Empty P2pMemInfo → import marks name as failed (no pools to map)
                    P2pMemInfo emptyInfo;
                    emptyInfo.supported = true;
                    emptyInfo.handleType = VmmHandleType::kFabric;
                    registry.importAndMap(name, emptyInfo);
                    registry.cleanup(name);
                }
            });
    }

    for (int i = kNumReaders; i < static_cast<int>(threads.size()); ++i)
    {
        threads[i].join();
    }
    stop.store(true, std::memory_order_release);
    for (int i = 0; i < kNumReaders; ++i)
    {
        threads[i].join();
    }

    SUCCEED();
}
