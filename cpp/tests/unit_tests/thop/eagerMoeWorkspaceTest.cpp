/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/thop/moe/cutlass/eagerMoeWorkspace.h"
#include "tensorrt_llm/thop/moe/cutlass/eagerMoeWorkspaceScope.h"

#include <thread>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>
#include <gtest/gtest.h>

using tensorrt_llm::torch_ext::EagerMoeWorkspace;

namespace
{
void forward(EagerMoeWorkspace& workspace, std::initializer_list<int64_t> sizes, bool warmup = false,
    bool completed = true, int64_t numTokens = 0)
{
    ASSERT_TRUE(workspace.beginForward(1, warmup, numTokens));
    for (auto const size : sizes)
    {
        workspace.get(size, false);
    }
    workspace.finishForward(completed);
}
} // namespace

TEST(EagerMoeWorkspace, CountsForwardsNotLayersAndCanGrowAgain)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true);
    forward(workspace, {32768});
    for (int i = 0; i < 2; ++i)
    {
        forward(workspace, {8192, 2048, 1024});
        EXPECT_EQ(workspace.capacity(), 32768);
    }
    forward(workspace, {4096, 1024});
    EXPECT_EQ(workspace.capacity(), 8192);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {1024});
    }
    EXPECT_EQ(workspace.capacity(), 4096);
    forward(workspace, {65536});
    EXPECT_EQ(workspace.capacity(), 65536);
}

TEST(EagerMoeWorkspace, FailureAndGrowthRestartWindow)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true);
    forward(workspace, {32768});
    forward(workspace, {4096});
    forward(workspace, {4096});
    forward(workspace, {4096}, false, false);
    forward(workspace, {4096});
    forward(workspace, {4096});
    EXPECT_EQ(workspace.capacity(), 32768);
    forward(workspace, {65536});
    forward(workspace, {4096});
    forward(workspace, {4096});
    EXPECT_EQ(workspace.capacity(), 65536);
    forward(workspace, {4096});
    EXPECT_EQ(workspace.capacity(), 4096);
}

TEST(EagerMoeWorkspace, MissingWarmupAndSharedOwnersOptOut)
{
    EagerMoeWorkspace workspace;
    workspace.get(4096, false);
    EXPECT_FALSE(workspace.beginForward(1, false));
    forward(workspace, {4096}, true);
    forward(workspace, {32768});
    EXPECT_FALSE(workspace.beginForward(2, false));
    EXPECT_FALSE(workspace.beginForward(1, false));
    EXPECT_EQ(workspace.capacity(), 32768);
}

TEST(EagerMoeWorkspace, OutOfScopeUseAndOverlapOptOut)
{
    for (bool const overlap : {false, true})
    {
        EagerMoeWorkspace workspace;
        forward(workspace, {4096}, true);
        if (overlap)
        {
            ASSERT_TRUE(workspace.beginForward(1, false));
            EXPECT_FALSE(workspace.beginForward(1, false));
            workspace.get(32768, false);
            workspace.finishForward(true);
        }
        else
        {
            workspace.get(32768, false);
        }
        EXPECT_FALSE(workspace.beginForward(1, false));
        EXPECT_EQ(workspace.capacity(), 32768);
    }
}

TEST(EagerMoeWorkspace, ReclamationPreservesQueuedConsumers)
{
    auto const stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true);
    forward(workspace, {32768});
    forward(workspace, {4096});
    forward(workspace, {4096});
    ASSERT_TRUE(workspace.beginForward(1, false));
    auto const& scratch = workspace.get(4096, false);
    scratch.fill_(37);
    auto result = scratch.clone();
    workspace.finishForward(true);
    EXPECT_EQ(workspace.capacity(), 4096);
    // Force reuse while the preceding fill/copy can still be queued.
    auto overwrite = torch::zeros({32768}, scratch.options());
    EXPECT_TRUE(result.eq(37).all().item<bool>());
}

TEST(EagerMoeWorkspace, GraphCapturePermanentlyPinsOwner)
{
    auto const stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    EagerMoeWorkspace workspace;
    forward(workspace, {65536}, true, true, 4096);
    stream.synchronize();
    // The allocation's capture flag is supplied by getWorkspaceInfo in the
    // runner. Graph-referenced storage must never enter the shrink policy.
    auto const& scratch = workspace.get(32768, true);
    auto const* pointer = scratch.data_ptr();
    auto const* identity = scratch.unsafeGetTensorImpl();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(scratch.data_ptr(), 19, 32768, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), cudaSuccess);
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FALSE(workspace.beginForward(1, false, 4096));
        workspace.get(1024, false);
    }
    EXPECT_EQ(workspace.capacity(), 32768);
    EXPECT_EQ(workspace.tensor().data_ptr(), pointer);
    EXPECT_EQ(workspace.tensor().unsafeGetTensorImpl(), identity);
    EXPECT_EQ(workspace.tensor().scalar_type(), torch::kInt8);
    ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
    EXPECT_TRUE(scratch.eq(19).all().item<bool>());
    ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST(EagerMoeWorkspace, ExactCapacityAndEmptyForwardRestartWindow)
{
    for (bool const exactCapacity : {false, true})
    {
        EagerMoeWorkspace workspace;
        forward(workspace, {4096}, true);
        forward(workspace, {32768});
        forward(workspace, {4096});
        forward(workspace, {4096});
        if (exactCapacity)
        {
            forward(workspace, {32768});
        }
        else
        {
            forward(workspace, {});
        }
        for (int i = 0; i < 2; ++i)
        {
            forward(workspace, {4096});
            EXPECT_EQ(workspace.capacity(), 32768);
        }
        forward(workspace, {4096});
        EXPECT_EQ(workspace.capacity(), 4096);
    }
}

TEST(EagerMoeWorkspace, FailedWarmupDoesNotEnableReclamation)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true, false);
    EXPECT_FALSE(workspace.beginForward(1, false));
    forward(workspace, {8192}, true);
    forward(workspace, {16384}, true);
    forward(workspace, {65536});
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {1024});
    }
    EXPECT_EQ(workspace.capacity(), 16384);
    EXPECT_EQ(workspace.tensor().scalar_type(), torch::kInt8);
    EXPECT_TRUE(workspace.tensor().is_cuda());
}

TEST(EagerMoeWorkspace, FiftyGrowthReclaimCyclesMatchDisabledOutputs)
{
    auto const stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    EagerMoeWorkspace enabled;
    EagerMoeWorkspace disabled;
    forward(enabled, {4096}, true);
    disabled.get(4096, false);
    for (int cycle = 0; cycle < 50; ++cycle)
    {
        auto const peak = (cycle % 2 + 2) * 32768;
        for (auto const required : {peak, 2048, 1024, 512})
        {
            ASSERT_TRUE(enabled.beginForward(1, false));
            // Vary the scratch payload, and copy before finishForward can
            // replace storage. Comparisons synchronize only after both arms.
            auto const value = cycle + 1;
            auto a = enabled.get(required, false).narrow(0, 0, required);
            a.fill_(value);
            auto result = a.clone();
            a = torch::Tensor{};
            enabled.finishForward(true);
            auto b = disabled.get(required, false).narrow(0, 0, required);
            b.fill_(value);
            EXPECT_TRUE(torch::equal(result, b));
        }
        EXPECT_EQ(enabled.capacity(), 4096);
        EXPECT_GE(disabled.capacity(), peak);
    }
}

TEST(EagerMoeWorkspace, ReleasesLiveAllocationBeforeSubsequentBufferGrowth)
{
    constexpr int64_t kMiB = 1024 * 1024;
    auto const allocated = []()
    {
        auto const stats = c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device());
        return stats.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
    };
    EagerMoeWorkspace workspace;
    forward(workspace, {4 * kMiB}, true);
    forward(workspace, {32 * kMiB});
    auto const before = allocated();
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {2 * kMiB});
    }
    EXPECT_EQ(workspace.capacity(), 4 * kMiB);
    EXPECT_EQ(before - allocated(), 28 * kMiB);
    // A subsequent allocation must not revive the released workspace storage.
    // Reserved/driver memory need not drop: the allocator may reuse its pool.
    auto subsequent = torch::empty({48 * kMiB}, workspace.tensor().options());
    subsequent.fill_(17);
    EXPECT_EQ(allocated() - before, 20 * kMiB);
    EXPECT_TRUE(subsequent.eq(17).all().item<bool>());
}

TEST(EagerMoeWorkspace, PreservesMaximumWarmupCapacityWithoutServingGrowth)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {65536}, true);
    forward(workspace, {4096}, true);
    auto const* pointer = workspace.tensor().data_ptr();
    for (int i = 0; i < 10; ++i)
    {
        forward(workspace, {4096, 2048});
        EXPECT_EQ(workspace.capacity(), 65536);
        EXPECT_EQ(workspace.tensor().data_ptr(), pointer);
    }
}

TEST(EagerMoeWorkspace, MarginalSavingsResetWindowAndDoNotChurn)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true);
    forward(workspace, {32768});
    forward(workspace, {4096});
    forward(workspace, {4096});
    forward(workspace, {16385});
    for (int i = 0; i < 2; ++i)
    {
        forward(workspace, {8192});
        EXPECT_EQ(workspace.capacity(), 32768);
    }
    forward(workspace, {4096});
    EXPECT_EQ(workspace.capacity(), 8192);
    auto const* pointer = workspace.tensor().data_ptr();
    for (int i = 0; i < 10; ++i)
    {
        forward(workspace, {6144});
        EXPECT_EQ(workspace.capacity(), 8192);
        EXPECT_EQ(workspace.tensor().data_ptr(), pointer);
    }
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {4096});
    }
    EXPECT_EQ(workspace.capacity(), 4096);
}

TEST(EagerMoeWorkspace, ReplacementDoesNotFragmentReleasedBurstBlock)
{
    if (c10::cuda::CUDACachingAllocator::name() != "native")
    {
        GTEST_SKIP() << "Checks native allocator segment reuse";
    }
    constexpr int64_t kMiB = 1024 * 1024;
    auto const reserved = []()
    {
        auto const stats = c10::cuda::CUDACachingAllocator::getDeviceStats(c10::cuda::current_device());
        return stats.reserved_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
    };
    // Isolate this allocator-layout regression from caches left by other tests.
    c10::cuda::CUDACachingAllocator::emptyCache();
    EagerMoeWorkspace workspace;
    forward(workspace, {8 * kMiB}, true);
    forward(workspace, {32 * kMiB});
    // Discard the released warmup block so the replacement needs fresh backing.
    c10::cuda::CUDACachingAllocator::emptyCache();
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {8 * kMiB});
    }
    auto const before = reserved();
    // Keep a downstream tensor alive across the next workspace growth. If the
    // replacement splits the old large block, this tensor pins that segment.
    auto downstream = torch::empty({4 * kMiB}, workspace.tensor().options());
    downstream.fill_(13);
    forward(workspace, {32 * kMiB});
    EXPECT_LE(reserved(), before);
    EXPECT_TRUE(downstream.eq(13).all().item<bool>());
}

TEST(EagerMoeWorkspace, ReclamationRetriesAfterReleasingScratchUnderMemoryPressure)
{
    if (c10::cuda::CUDACachingAllocator::name() != "native")
    {
        GTEST_SKIP() << "Uses native allocator quota and OOM counters";
    }
    constexpr int64_t kMiB = 1024 * 1024;
    auto const device = c10::cuda::current_device();
    auto const previousLimit = c10::cuda::CUDACachingAllocator::getMemoryFraction(device);
    auto restoreLimit
        = c10::make_scope_exit([&]() { c10::cuda::CUDACachingAllocator::setMemoryFraction(previousLimit, device); });
    c10::cuda::CUDACachingAllocator::emptyCache();
    EagerMoeWorkspace workspace;
    forward(workspace, {8 * kMiB}, true);
    forward(workspace, {32 * kMiB});
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto const before = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    auto const live = before.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
    size_t free = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
    // An allocator quota, not physical-memory exhaustion: at most 33 MiB live
    // here, and insufficient headroom for the 8 MiB replacement until release.
    c10::cuda::CUDACachingAllocator::setMemoryFraction(static_cast<double>(live + kMiB) / total, device);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {8 * kMiB});
    }
    EXPECT_EQ(workspace.capacity(), 8 * kMiB);
    auto const after = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    EXPECT_EQ(after.num_ooms, before.num_ooms + 1);
}

TEST(EagerMoeWorkspace, ObservedDemandIsReservedBeforeDownstreamAllocations)
{
    constexpr int64_t kMiB = 1024 * 1024;
    EagerMoeWorkspace workspace;
    forward(workspace, {8 * kMiB}, true, true, 1);
    forward(workspace, {32 * kMiB}, false, true, 4096);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {8 * kMiB}, false, true, 1);
    }
    EXPECT_EQ(workspace.capacity(), 8 * kMiB);
    ASSERT_TRUE(workspace.beginForward(1, false, 3840));
    EXPECT_EQ(workspace.capacity(), 32 * kMiB);
    auto downstream = torch::empty({4 * kMiB}, workspace.tensor().options());
    downstream.fill_(11);
    workspace.get(30 * kMiB, false);
    workspace.finishForward(true);
    EXPECT_TRUE(downstream.eq(11).all().item<bool>());
}

TEST(EagerMoeWorkspace, FailedForwardDoesNotPoisonDemandHint)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {8192}, true, true, 128);
    forward(workspace, {65536}, false, false, 4096);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {8192}, false, true, 128);
    }
    ASSERT_EQ(workspace.capacity(), 8192);
    ASSERT_TRUE(workspace.beginForward(1, false, 512));
    EXPECT_EQ(workspace.capacity(), 8192);
    workspace.finishForward(false);
}

TEST(EagerMoeWorkspace, ConservativeReservationOomStillAllowsSmallerActualDemand)
{
    if (c10::cuda::CUDACachingAllocator::name() != "native")
    {
        GTEST_SKIP() << "Uses native allocator quota and OOM counters";
    }
    constexpr int64_t kMiB = 1024 * 1024;
    auto const device = c10::cuda::current_device();
    auto const previousLimit = c10::cuda::CUDACachingAllocator::getMemoryFraction(device);
    auto restoreLimit
        = c10::make_scope_exit([&]() { c10::cuda::CUDACachingAllocator::setMemoryFraction(previousLimit, device); });
    c10::cuda::CUDACachingAllocator::emptyCache();
    EagerMoeWorkspace workspace;
    forward(workspace, {8 * kMiB}, true, true, 1);
    forward(workspace, {32 * kMiB}, false, true, 4096);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {8 * kMiB}, false, true, 1);
    }
    auto downstream = torch::empty({16 * kMiB}, workspace.tensor().options());
    auto const before = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    auto const live = before.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)].current;
    size_t free = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
    c10::cuda::CUDACachingAllocator::setMemoryFraction(static_cast<double>(live + kMiB) / total, device);
    ASSERT_TRUE(workspace.beginForward(1, false, 2048));
    workspace.get(4 * kMiB, false);
    workspace.finishForward(true);
    EXPECT_EQ(workspace.capacity(), 8 * kMiB);
    auto const after = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    EXPECT_EQ(after.num_ooms, before.num_ooms + 1);
}

TEST(EagerMoeWorkspace, FailedServingGrowthDoesNotRaiseWarmupFloor)
{
    EagerMoeWorkspace workspace;
    forward(workspace, {4096}, true);
    forward(workspace, {65536}, false, false);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {1024});
    }
    EXPECT_EQ(workspace.capacity(), 4096);
}

namespace
{
using tensorrt_llm::torch_ext::EagerMoeWorkspaceOwner;
using tensorrt_llm::torch_ext::EagerMoeWorkspaceScope;

class ScopeTestOwner : public EagerMoeWorkspaceOwner
{
public:
    bool beginWorkspaceForward(int64_t owner, bool warmup, int64_t numTokens) override
    {
        ++begins;
        lastOwner = owner;
        lastWarmup = warmup;
        lastTokens = numTokens;
        TORCH_CHECK(!failBegin, "test begin failure");
        return !reject;
    }

    void finishWorkspaceForward(bool completed) override
    {
        ++finishes;
        lastCompleted = completed;
        finishedStreams.push_back(c10::cuda::getCurrentCUDAStream());
        TORCH_CHECK(!failFinish, "test finish failure");
    }

    int begins = 0;
    int finishes = 0;
    int64_t lastOwner = 0;
    int64_t lastTokens = 0;
    bool lastWarmup = false;
    bool lastCompleted = false;
    bool failBegin = false;
    bool failFinish = false;
    bool reject = false;
    std::vector<c10::cuda::CUDAStream> finishedStreams;
};

void registerOwner(ScopeTestOwner* owner)
{
    EagerMoeWorkspaceScope::registerCurrent(owner, c10::cuda::current_device());
}
} // namespace

TEST(EagerMoeWorkspaceScope, RepeatedLayersAndEarlyReservation)
{
    EagerMoeWorkspaceScope scope(17);
    auto owner = c10::make_intrusive<ScopeTestOwner>();
    scope.begin(true, 4096);
    for (int i = 0; i < 30; ++i)
    {
        registerOwner(owner.get());
    }
    EXPECT_EQ(owner->begins, 1);
    EXPECT_EQ(owner->lastOwner, 17);
    EXPECT_TRUE(owner->lastWarmup);
    EXPECT_EQ(owner->lastTokens, 4096);
    EXPECT_TRUE(scope.finish(true));
    EXPECT_EQ(owner->finishes, 1);
    scope.begin(false, 128);
    EXPECT_EQ(owner->begins, 2);
    EXPECT_FALSE(owner->lastWarmup);
    EXPECT_EQ(owner->lastTokens, 128);
    registerOwner(owner.get());
    EXPECT_EQ(owner->begins, 2);
    scope.finish(true);
    EXPECT_EQ(owner->finishes, 2);
    registerOwner(owner.get());
    EXPECT_EQ(owner->begins, 2);
}

TEST(EagerMoeWorkspaceScope, StreamsAreIndependentAndCleanupRestoresCurrentStream)
{
    EagerMoeWorkspaceScope scope(1);
    auto owner = c10::make_intrusive<ScopeTestOwner>();
    auto const original = c10::cuda::getCurrentCUDAStream();
    auto const other = c10::cuda::getStreamFromPool();
    scope.begin(true, 0);
    registerOwner(owner.get());
    {
        c10::cuda::CUDAStreamGuard guard(other);
        registerOwner(owner.get());
    }
    registerOwner(owner.get());
    EXPECT_EQ(owner->begins, 2);
    scope.finish(true);
    ASSERT_EQ(owner->finishedStreams.size(), 2);
    EXPECT_EQ(owner->finishedStreams[0], original);
    EXPECT_EQ(owner->finishedStreams[1], other);
    EXPECT_EQ(c10::cuda::getCurrentCUDAStream(), original);
}

TEST(EagerMoeWorkspaceScope, FinishFailureDoesNotStrandOtherOwnersOrThreadContext)
{
    EagerMoeWorkspaceScope scope(1);
    auto first = c10::make_intrusive<ScopeTestOwner>();
    auto second = c10::make_intrusive<ScopeTestOwner>();
    first->failFinish = true;
    scope.begin(true, 0);
    registerOwner(first.get());
    registerOwner(second.get());
    EXPECT_THROW(scope.finish(true), c10::Error);
    EXPECT_EQ(first->finishes, 1);
    EXPECT_EQ(second->finishes, 1);
    first->failFinish = false;
    scope.begin(false, 0);
    registerOwner(first.get());
    scope.finish(false);
    EXPECT_EQ(first->finishes, 2);
    EXPECT_FALSE(first->lastCompleted);
}

TEST(EagerMoeWorkspaceScope, RejectedOwnerIsOnlyCheckedOnce)
{
    EagerMoeWorkspaceScope scope(1);
    auto owner = c10::make_intrusive<ScopeTestOwner>();
    owner->reject = true;
    scope.begin(false, 0);
    for (int i = 0; i < 8; ++i)
    {
        registerOwner(owner.get());
    }
    scope.finish(true);
    EXPECT_EQ(owner->begins, 1);
    EXPECT_EQ(owner->finishes, 0);
}

TEST(EagerMoeWorkspaceScope, LateOwnerAndRegistrationFailureAbortCleanly)
{
    EagerMoeWorkspaceScope scope(1);
    auto first = c10::make_intrusive<ScopeTestOwner>();
    auto failing = c10::make_intrusive<ScopeTestOwner>();
    scope.begin(true, 128);
    registerOwner(first.get());
    scope.finish(true);
    scope.begin(false, 128);
    failing->failBegin = true;
    EXPECT_THROW(registerOwner(failing.get()), c10::Error);
    scope.finish(false);
    EXPECT_EQ(first->finishes, 2);
    EXPECT_FALSE(first->lastCompleted);
    EXPECT_EQ(failing->finishes, 0);
    failing->failBegin = false;
    scope.begin(true, 128);
    registerOwner(failing.get());
    scope.finish(true);
    EXPECT_EQ(failing->finishes, 1);
}

TEST(EagerMoeWorkspaceScope, EarlyReservationFailureUnwindsAcceptedOwners)
{
    EagerMoeWorkspaceScope scope(1);
    auto first = c10::make_intrusive<ScopeTestOwner>();
    auto second = c10::make_intrusive<ScopeTestOwner>();
    scope.begin(true, 128);
    registerOwner(first.get());
    registerOwner(second.get());
    scope.finish(true);
    // Reservation order is the native address key order.
    auto& failing
        = reinterpret_cast<uintptr_t>(first.get()) < reinterpret_cast<uintptr_t>(second.get()) ? second : first;
    auto& accepted = failing == first ? second : first;
    failing->failBegin = true;
    EXPECT_THROW(scope.begin(false, 128), c10::Error);
    EXPECT_EQ(accepted->finishes, 2);
    EXPECT_FALSE(accepted->lastCompleted);
    EXPECT_EQ(failing->finishes, 1);
    failing->failBegin = false;
    scope.begin(false, 128);
    scope.finish(true);
}

TEST(EagerMoeWorkspaceScope, NestedScopesAndCrossThreadReuseAreRejected)
{
    EagerMoeWorkspaceScope first(1);
    EagerMoeWorkspaceScope second(2);
    first.begin(true, 0);
    EXPECT_THROW(first.begin(true, 0), c10::Error);
    EXPECT_THROW(second.begin(true, 0), c10::Error);
    std::thread other(
        [&]()
        {
            EXPECT_THROW(first.begin(true, 0), c10::Error);
            EXPECT_THROW(first.finish(false), c10::Error);
            second.begin(true, 0);
            second.finish(true);
        });
    other.join();
    first.finish(true);
    second.begin(true, 0);
    second.finish(true);
}

TEST(EagerMoeWorkspaceScope, KeepsOwnersAliveUntilScopeIsDestroyed)
{
    auto owner = c10::make_intrusive<ScopeTestOwner>();
    c10::weak_intrusive_ptr<ScopeTestOwner> weak(owner);
    {
        EagerMoeWorkspaceScope scope(1);
        scope.begin(true, 0);
        registerOwner(owner.get());
        owner.reset();
        EXPECT_FALSE(weak.expired());
        scope.finish(true);
        EXPECT_FALSE(weak.expired());
    }
    EXPECT_TRUE(weak.expired());
}

TEST(EagerMoeWorkspaceScope, EmptyForwardReportsNoOwners)
{
    EagerMoeWorkspaceScope scope(1);
    scope.begin(true, 128);
    EXPECT_FALSE(scope.finish(true));
    scope.begin(false, 128);
    EXPECT_FALSE(scope.finish(true));
}

TEST(EagerMoeWorkspaceScope, ExternalAliasOfStreamDoesNotStartSecondOwner)
{
    EagerMoeWorkspaceScope scope(1);
    auto owner = c10::make_intrusive<ScopeTestOwner>();
    auto const stream = c10::cuda::getStreamFromPool();
    auto const alias = c10::cuda::getStreamFromExternal(stream.stream(), stream.device_index());
    scope.begin(true, 0);
    {
        c10::cuda::CUDAStreamGuard guard(stream);
        registerOwner(owner.get());
    }
    {
        c10::cuda::CUDAStreamGuard guard(alias);
        registerOwner(owner.get());
    }
    EXPECT_EQ(owner->begins, 1);
    scope.finish(true);
    {
        c10::cuda::CUDAStreamGuard guard(alias);
        scope.begin(false, 128);
        EXPECT_EQ(owner->begins, 2);
        scope.finish(true);
    }
    EXPECT_EQ(owner->finishes, 2);
}
