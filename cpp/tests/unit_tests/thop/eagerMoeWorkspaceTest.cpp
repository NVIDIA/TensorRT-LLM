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
#include "tensorrt_llm/thop/moe/cutlass/eagerMoeWorkspace.h"
#include "tensorrt_llm/thop/moe/cutlass/eagerMoeWorkspaceScope.h"

#include <ATen/Context.h>
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
    forward(workspace, {1024}, true);
    for (int i = 0; i < 3; ++i)
    {
        forward(workspace, {1024});
    }
    EXPECT_EQ(workspace.capacity(), 4096);
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
    forward(workspace, {4096}, true, false);
    EXPECT_FALSE(workspace.beginForward(1, false));
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
    forward(workspace, {32769}); // More than half capacity restarts the window.
    for (int i = 0; i < 2; ++i)
    {
        forward(workspace, {4096});
        EXPECT_EQ(workspace.capacity(), 65536);
    }
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

TEST(EagerMoeWorkspace, ReclamationRetriesAfterReleasingScratchUnderMemoryPressure)
{
    if (c10::cuda::CUDACachingAllocator::name() != "native")
    {
        GTEST_SKIP() << "Uses native allocator quota and OOM counters";
    }
    constexpr int64_t kMiB = 1024 * 1024;
    // Allocator quota queries require PyTorch's CUDA initialization.
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
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

TEST(EagerMoeWorkspace, ConservativeReservationOomStillAllowsSmallerActualDemand)
{
    if (c10::cuda::CUDACachingAllocator::name() != "native")
    {
        GTEST_SKIP() << "Uses native allocator quota and OOM counters";
    }
    constexpr int64_t kMiB = 1024 * 1024;
    // Allocator quota queries require PyTorch's CUDA initialization.
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
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
    auto const previousData = workspace.tensor().data_ptr();
    ASSERT_EQ(workspace.capacity(), 8 * kMiB);
    ASSERT_TRUE(workspace.beginForward(1, false, 2048));
    ASSERT_EQ(workspace.capacity(), 8 * kMiB);
    EXPECT_EQ(workspace.tensor().data_ptr(), previousData);
    workspace.get(4 * kMiB, false);
    EXPECT_EQ(workspace.tensor().data_ptr(), previousData);
    workspace.finishForward(true);
    EXPECT_EQ(workspace.capacity(), 8 * kMiB);
    auto const after = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    EXPECT_EQ(after.num_ooms, before.num_ooms + 1);
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
        return true;
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
    for (int i = 0; i < 3; ++i)
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
    {
        auto const alias = c10::cuda::getStreamFromExternal(original.stream(), original.device_index());
        c10::cuda::CUDAStreamGuard guard(alias);
        registerOwner(owner.get());
    }
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
