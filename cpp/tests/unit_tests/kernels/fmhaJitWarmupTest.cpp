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

// Host-only tests for the asynchronous TRTLLM-Gen FMHA JIT warmup building
// blocks (fmhaJitWarmup.h): the background worker's lifecycle, the sweep
// registry's state machine, and the scheduling path the kernel runs them
// through. No GPU is involved; the "sweeps" here are plain callbacks.

#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaJitWarmup.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{

using tensorrt_llm::kernels::JITWarmupScheduleResult;
using tensorrt_llm::kernels::JITWarmupState;
using tensorrt_llm::kernels::scheduleJITWarmupSweep;
using tensorrt_llm::kernels::TllmGenFmhaAsyncWarmupWorker;
using tensorrt_llm::kernels::TllmGenFmhaJitCompileStats;
using tensorrt_llm::kernels::TllmGenFmhaJitWarmupRegistry;

// Stand-ins for the kernel object and the sweep parameters the registry stores.
struct FakeKernel
{
    int mId;
};

struct FakeParams
{
    int mMaxSeqLenQ;
    int mMaxSeqLenKv;
};

using Registry = TllmGenFmhaJitWarmupRegistry<FakeParams, FakeKernel>;

// A gate the test opens to let a queued task finish, so "pending" states can be
// observed deterministically.
class Gate
{
public:
    void open()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mOpen = true;
        }
        mCv.notify_all();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this]() { return mOpen; });
    }

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    bool mOpen{false};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Worker
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TllmGenFmhaAsyncWarmupWorker, RunsTasksInOrderAndDrainWaitsForAll)
{
    TllmGenFmhaAsyncWarmupWorker worker;
    std::mutex mutex;
    std::vector<int> order;
    Gate gate;

    ASSERT_TRUE(worker.enqueue(
        [&]()
        {
            gate.wait();
            std::lock_guard<std::mutex> lock(mutex);
            order.push_back(1);
        }));
    ASSERT_TRUE(worker.enqueue(
        [&]()
        {
            std::lock_guard<std::mutex> lock(mutex);
            order.push_back(2);
        }));
    EXPECT_EQ(worker.numPendingTasks(), 2);

    gate.open();
    worker.drain();
    EXPECT_EQ(worker.numPendingTasks(), 0);
    EXPECT_EQ(order, (std::vector<int>{1, 2}));
}

TEST(TllmGenFmhaAsyncWarmupWorker, ThreadInitRunsOnWorkerThreadBeforeTasks)
{
    std::thread::id initThread;
    std::thread::id taskThread;
    TllmGenFmhaAsyncWarmupWorker worker([&]() { initThread = std::this_thread::get_id(); });
    ASSERT_TRUE(worker.enqueue([&]() { taskThread = std::this_thread::get_id(); }));
    worker.drain();
    EXPECT_EQ(initThread, taskThread);
    EXPECT_NE(initThread, std::this_thread::get_id());
}

TEST(TllmGenFmhaAsyncWarmupWorker, FailingTaskDoesNotStrandTheQueue)
{
    TllmGenFmhaAsyncWarmupWorker worker;
    std::atomic<bool> secondRan{false};
    ASSERT_TRUE(worker.enqueue([]() { throw std::runtime_error("compile failed"); }));
    ASSERT_TRUE(worker.enqueue([&]() { secondRan = true; }));
    worker.drain();
    EXPECT_EQ(worker.numPendingTasks(), 0);
    EXPECT_TRUE(secondRan);
}

TEST(TllmGenFmhaAsyncWarmupWorker, ShutdownWithPendingWorkFinishesItThenRejectsNewWork)
{
    TllmGenFmhaAsyncWarmupWorker worker;
    std::atomic<int> numRan{0};
    Gate gate;
    ASSERT_TRUE(worker.enqueue(
        [&]()
        {
            gate.wait();
            ++numRan;
        }));
    ASSERT_TRUE(worker.enqueue([&]() { ++numRan; }));
    EXPECT_FALSE(worker.isShutDown());

    // shutdown() must wait for the queued work rather than drop it.
    std::thread shutdownThread([&]() { worker.shutdown(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(numRan, 0);
    gate.open();
    shutdownThread.join();

    EXPECT_EQ(numRan, 2);
    EXPECT_EQ(worker.numPendingTasks(), 0);
    EXPECT_TRUE(worker.isShutDown());

    // Later work is not queued; the caller runs it inline.
    EXPECT_FALSE(worker.enqueue([&]() { ++numRan; }));
    EXPECT_EQ(numRan, 2);
    // drain() after shutdown() returns immediately.
    worker.drain();
    // shutdown() is idempotent.
    worker.shutdown();
}

TEST(TllmGenFmhaAsyncWarmupWorker, ShutdownWithoutEverStartingIsSafe)
{
    TllmGenFmhaAsyncWarmupWorker worker;
    worker.shutdown();
    EXPECT_TRUE(worker.isShutDown());
    EXPECT_FALSE(worker.enqueue([]() {}));
    worker.drain();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Registry
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TllmGenFmhaJitWarmupRegistry, DeduplicatesIdenticalSweeps)
{
    Registry registry;
    FakeKernel kernel{1};
    FakeParams params{4, 1024};
    EXPECT_FALSE(registry.tryRegister(0xA, params, JITWarmupState::kScheduled, /*deviceId=*/0, &kernel));
    EXPECT_TRUE(registry.tryRegister(0xA, params, JITWarmupState::kScheduled, /*deviceId=*/0, &kernel));
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.state(0xA), JITWarmupState::kScheduled);
}

TEST(TllmGenFmhaJitWarmupRegistry, SynchronousSweepsAreVerifiedOnRegistration)
{
    Registry registry;
    FakeKernel kernel{1};
    EXPECT_FALSE(registry.tryRegister(0xA, FakeParams{4, 1024}, JITWarmupState::kVerified, 0, &kernel));
    EXPECT_EQ(registry.numUnverified(), 0);
    EXPECT_FALSE(registry.claimUnverified(0, &kernel).has_value());
    EXPECT_FALSE(registry.claimAnyUnverified().has_value());
}

TEST(TllmGenFmhaJitWarmupRegistry, BackgroundDoneIsClaimedExactlyOnceByItsOwner)
{
    Registry registry;
    FakeKernel kernel{1};
    FakeParams params{4, 1024};
    ASSERT_FALSE(registry.tryRegister(0xA, params, JITWarmupState::kScheduled, 0, &kernel));
    EXPECT_EQ(registry.numUnverified(), 0);
    EXPECT_FALSE(registry.claimUnverified(0, &kernel).has_value());

    registry.markBackgroundDone(0xA);
    EXPECT_EQ(registry.numUnverified(), 1);
    EXPECT_EQ(registry.state(0xA), JITWarmupState::kBackgroundDone);

    auto sweep = registry.claimUnverified(0, &kernel);
    ASSERT_TRUE(sweep.has_value());
    EXPECT_EQ(sweep->mFingerprint, 0xAu);
    EXPECT_EQ(sweep->mParams.mMaxSeqLenQ, 4);
    EXPECT_EQ(sweep->mParams.mMaxSeqLenKv, 1024);
    EXPECT_EQ(sweep->mDeviceId, 0);
    EXPECT_EQ(sweep->mInstance, &kernel);
    EXPECT_EQ(registry.numUnverified(), 0);
    EXPECT_EQ(registry.state(0xA), JITWarmupState::kVerified);

    EXPECT_FALSE(registry.claimUnverified(0, &kernel).has_value());
    // A second markBackgroundDone (e.g. from a retried task) must not reopen it.
    registry.markBackgroundDone(0xA);
    EXPECT_EQ(registry.numUnverified(), 0);
    EXPECT_EQ(registry.state(0xA), JITWarmupState::kVerified);
}

TEST(TllmGenFmhaJitWarmupRegistry, SameDeviceMultiConfigIsolation)
{
    // Two kernel objects on one device (e.g. fp8 and bf16 attention configs).
    // Each may only verify its own sweep.
    Registry registry;
    FakeKernel kernelA{1};
    FakeKernel kernelB{2};
    ASSERT_FALSE(registry.tryRegister(0xA, FakeParams{4, 1024}, JITWarmupState::kScheduled, 0, &kernelA));
    ASSERT_FALSE(registry.tryRegister(0xB, FakeParams{1, 2048}, JITWarmupState::kScheduled, 0, &kernelB));
    registry.markBackgroundDone(0xA);
    registry.markBackgroundDone(0xB);
    EXPECT_EQ(registry.numUnverified(), 2);

    // A reaches the verify pass first and must not take B's sweep.
    auto sweepA = registry.claimUnverified(0, &kernelA);
    ASSERT_TRUE(sweepA.has_value());
    EXPECT_EQ(sweepA->mInstance, &kernelA);
    EXPECT_EQ(sweepA->mFingerprint, 0xAu);
    EXPECT_FALSE(registry.claimUnverified(0, &kernelA).has_value());
    EXPECT_EQ(registry.state(0xB), JITWarmupState::kBackgroundDone);
    EXPECT_EQ(registry.numUnverified(), 1);

    auto sweepB = registry.claimUnverified(0, &kernelB);
    ASSERT_TRUE(sweepB.has_value());
    EXPECT_EQ(sweepB->mInstance, &kernelB);
    EXPECT_EQ(sweepB->mFingerprint, 0xBu);
    EXPECT_EQ(registry.numUnverified(), 0);
}

TEST(TllmGenFmhaJitWarmupRegistry, MultiDeviceSweepsAreDistinctAndClaimedPerDevice)
{
    // The same configuration on two devices is two sweeps (compiled modules are
    // per device context), and each device only verifies its own.
    Registry registry;
    FakeKernel kernelDev0{1};
    FakeKernel kernelDev1{2};
    FakeParams params{4, 1024};
    ASSERT_FALSE(registry.tryRegister(0xA0, params, JITWarmupState::kScheduled, 0, &kernelDev0));
    ASSERT_FALSE(registry.tryRegister(0xA1, params, JITWarmupState::kScheduled, 1, &kernelDev1));
    EXPECT_EQ(registry.size(), 2u);
    registry.markBackgroundDone(0xA0);
    registry.markBackgroundDone(0xA1);

    EXPECT_FALSE(registry.claimUnverified(1, &kernelDev0).has_value());
    EXPECT_FALSE(registry.claimUnverified(0, &kernelDev1).has_value());

    auto sweep0 = registry.claimUnverified(0, &kernelDev0);
    ASSERT_TRUE(sweep0.has_value());
    EXPECT_EQ(sweep0->mDeviceId, 0);
    auto sweep1 = registry.claimUnverified(1, &kernelDev1);
    ASSERT_TRUE(sweep1.has_value());
    EXPECT_EQ(sweep1->mDeviceId, 1);
    EXPECT_EQ(registry.numUnverified(), 0);
}

TEST(TllmGenFmhaJitWarmupRegistry, ClaimAnyUnverifiedServesTheDrainAndVerifyBarrier)
{
    Registry registry;
    FakeKernel kernelA{1};
    FakeKernel kernelB{2};
    ASSERT_FALSE(registry.tryRegister(0xA, FakeParams{4, 1024}, JITWarmupState::kScheduled, 0, &kernelA));
    ASSERT_FALSE(registry.tryRegister(0xB, FakeParams{1, 2048}, JITWarmupState::kScheduled, 1, &kernelB));
    registry.markBackgroundDone(0xA);
    registry.markBackgroundDone(0xB);

    int numClaimed = 0;
    while (auto sweep = registry.claimAnyUnverified())
    {
        // Each claimed sweep carries the owner and device the barrier must dispatch to.
        EXPECT_TRUE((sweep->mInstance == &kernelA && sweep->mDeviceId == 0)
            || (sweep->mInstance == &kernelB && sweep->mDeviceId == 1));
        ++numClaimed;
    }
    EXPECT_EQ(numClaimed, 2);
    EXPECT_EQ(registry.numUnverified(), 0);
    EXPECT_EQ(registry.state(0xA), JITWarmupState::kVerified);
    EXPECT_EQ(registry.state(0xB), JITWarmupState::kVerified);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Compile stats
////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(TllmGenFmhaJitCompileStats, CountsEveryMissIncludingRepeatsOfTheSameKey)
{
    using CacheResult = TllmGenFmhaJitCompileStats::CacheResult;
    TllmGenFmhaJitCompileStats stats;
    EXPECT_EQ(stats.numMisses(), 0);
    EXPECT_EQ(stats.numHits(), 0);
    EXPECT_EQ(stats.numUnknown(), 0);

    stats.record(CacheResult::kMiss); // first compile of a key
    stats.record(CacheResult::kHit);  // same key served from the cache
    stats.record(CacheResult::kMiss); // same key compiled again after an LRU eviction
    EXPECT_EQ(stats.numMisses(), 2);
    EXPECT_EQ(stats.numHits(), 1);
    EXPECT_EQ(stats.numUnknown(), 0);

    // An unreported result is neither a hit nor a miss; it must stay visible so a
    // test cannot read "no misses" as "no compiles".
    stats.record(CacheResult::kUnknown);
    EXPECT_EQ(stats.numMisses(), 2);
    EXPECT_EQ(stats.numUnknown(), 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Worker + registry together: the scheduling protocol used by fmhaKernels.h.
////////////////////////////////////////////////////////////////////////////////////////////////////

// Drives the kernel's scheduling path, scheduleJITWarmupSweep(), with the
// real worker and registry; only the sweep body is a stand-in. Sweeps are
// tallied by the thread they ran on, so a test can tell background runs from
// inline runs and verify-pass re-runs.
struct Scheduler
{
    // The registry is declared first so it outlives the worker's shutdown, which
    // runs any still-queued task against it.
    Registry mRegistry;
    TllmGenFmhaAsyncWarmupWorker mWorker;
    std::thread::id const mMainThread{std::this_thread::get_id()};
    std::atomic<int> mBackgroundSweeps{0};
    std::atomic<int> mForegroundSweeps{0};

    JITWarmupScheduleResult schedule(uint64_t fingerprint, FakeParams const& params, FakeKernel* kernel,
        bool failInBackground, bool async = true, int deviceId = 0)
    {
        return scheduleJITWarmupSweep(mRegistry, async ? &mWorker : nullptr, fingerprint, params, deviceId, kernel,
            [this, failInBackground](FakeParams const&, int)
            {
                if (std::this_thread::get_id() == mMainThread)
                {
                    ++mForegroundSweeps;
                    return;
                }
                if (failInBackground)
                {
                    throw std::runtime_error("nvrtc failed");
                }
                ++mBackgroundSweeps;
            });
    }

    // The pre-capture barrier: drain, then re-run every finished sweep once.
    int drainAndVerifyAll()
    {
        mWorker.drain();
        int numVerified = 0;
        while (auto sweep = mRegistry.claimAnyUnverified())
        {
            ++mForegroundSweeps;
            ++numVerified;
        }
        return numVerified;
    }
};

TEST(TllmGenFmhaJitWarmupProtocol, EveryScheduledSweepIsVerifiedExactlyOnceBeforeCapture)
{
    Scheduler scheduler;
    FakeKernel kernel{1};
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, /*failInBackground=*/false),
        JITWarmupScheduleResult::kQueued);
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, false), JITWarmupScheduleResult::kDuplicate);
    EXPECT_EQ(scheduler.schedule(0xB, FakeParams{1, 1024}, &kernel, false), JITWarmupScheduleResult::kQueued);

    EXPECT_EQ(scheduler.drainAndVerifyAll(), 2);
    EXPECT_EQ(scheduler.mBackgroundSweeps, 2);
    EXPECT_EQ(scheduler.mForegroundSweeps, 2);
    EXPECT_EQ(scheduler.mRegistry.numUnverified(), 0);
    // Nothing left for a later verify pass.
    EXPECT_EQ(scheduler.drainAndVerifyAll(), 0);
}

TEST(TllmGenFmhaJitWarmupProtocol, BackgroundFailureIsRetriedOnTheForeground)
{
    Scheduler scheduler;
    FakeKernel kernel{1};
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, /*failInBackground=*/true),
        JITWarmupScheduleResult::kQueued);
    scheduler.mWorker.drain();
    // The failed sweep is not stranded in kScheduled: it is handed to the verify pass.
    EXPECT_EQ(scheduler.mRegistry.state(0xA), JITWarmupState::kBackgroundDone);
    EXPECT_EQ(scheduler.mBackgroundSweeps, 0);

    EXPECT_EQ(scheduler.drainAndVerifyAll(), 1);
    EXPECT_EQ(scheduler.mForegroundSweeps, 1);
    EXPECT_EQ(scheduler.mRegistry.state(0xA), JITWarmupState::kVerified);
}

TEST(TllmGenFmhaJitWarmupProtocol, ShutDownWorkerFallsBackToInlineCompilation)
{
    Scheduler scheduler;
    FakeKernel kernel{1};
    scheduler.mWorker.shutdown();
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, false), JITWarmupScheduleResult::kRanInline);
    // Compiled inline by the caller, and still verified exactly once.
    EXPECT_EQ(scheduler.mForegroundSweeps, 1);
    EXPECT_EQ(scheduler.drainAndVerifyAll(), 1);
    EXPECT_EQ(scheduler.mForegroundSweeps, 2);
    EXPECT_EQ(scheduler.mBackgroundSweeps, 0);
}

TEST(TllmGenFmhaJitWarmupProtocol, SyncModeRunsInlineAndStillDeduplicates)
{
    // Async warmup disabled: the sweep runs on the caller, is registered as
    // verified, and an identical later request is dropped just as in async mode.
    Scheduler scheduler;
    FakeKernel kernel{1};
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, false, /*async=*/false),
        JITWarmupScheduleResult::kRanInline);
    EXPECT_EQ(scheduler.mForegroundSweeps, 1);
    EXPECT_EQ(scheduler.mRegistry.state(0xA), JITWarmupState::kVerified);
    EXPECT_EQ(scheduler.schedule(0xA, FakeParams{4, 1024}, &kernel, false, /*async=*/false),
        JITWarmupScheduleResult::kDuplicate);
    EXPECT_EQ(scheduler.mForegroundSweeps, 1);
    // Nothing for the barrier to verify.
    EXPECT_EQ(scheduler.drainAndVerifyAll(), 0);
    EXPECT_EQ(scheduler.mBackgroundSweeps, 0);
}

TEST(TllmGenFmhaJitWarmupProtocol, BackgroundSweepRunsOnItsOwnDevice)
{
    // The sweep body receives the device the request was registered on, not
    // whatever is current on the worker thread.
    Registry registry;
    TllmGenFmhaAsyncWarmupWorker worker;
    FakeKernel kernel{1};
    std::atomic<int> seenDevice{-1};
    EXPECT_EQ(scheduleJITWarmupSweep(registry, &worker, 0xA, FakeParams{4, 1024}, /*deviceId=*/3, &kernel,
                  [&seenDevice](FakeParams const&, int deviceId) { seenDevice = deviceId; }),
        JITWarmupScheduleResult::kQueued);
    worker.drain();
    EXPECT_EQ(seenDevice.load(), 3);
    auto const sweep = registry.claimUnverified(3, &kernel);
    ASSERT_TRUE(sweep.has_value());
    EXPECT_EQ(sweep->mDeviceId, 3);
}

} // namespace
