/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/thop/moe/cutlass/eagerMoeWorkspace.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>

using tensorrt_llm::torch_ext::EagerMoeWorkspace;

namespace
{
void forward(
    EagerMoeWorkspace& workspace, std::initializer_list<int64_t> sizes, bool warmup = false, bool completed = true)
{
    ASSERT_TRUE(workspace.beginForward(1, warmup));
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
    forward(workspace, {4096}, true);
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
        EXPECT_FALSE(workspace.beginForward(1, false));
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

TEST(EagerMoeWorkspace, FailedWarmupDoesNotEstablishFloor)
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
