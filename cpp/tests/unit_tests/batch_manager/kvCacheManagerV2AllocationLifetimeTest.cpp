// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "kvCacheManagerV2TestUtils.h"
#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/introspection.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/funcGuard.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace
{
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

KVCacheManagerConfig makeLifetimeConfig()
{
    auto config = test::makeConfig();
    config.cacheTiers = {GpuCacheTierConfig{32UL << 20}};
    config.commitMinSnapshot = true;
    config.maxUtilForResume = 1.f;
    for (int layer = 1; layer <= 4; ++layer)
    {
        AllocationGroupId const group{layer * 10};
        config.allocationGroups.push_back(AllocationGroupConfig{.allocationGroupId = group,
            .allocationUnit = AllocationUnit::kFixed,
            .reusePolicy = ReusePolicy::PRIVATE,
            .initiallyActive = layer <= 2,
            .suspendPolicy = layer == 3 ? SuspendPolicy::kRelease : SuspendPolicy::kHold});
        config.layers.emplace_back(AttentionLayerConfig{.layerId = LayerId{layer},
            .buffers = {BufferConfig{.role = "state", .size = size_t{4096} << layer, .allocationGroupId = group}}});
    }
    config.constraints = {BatchDesc{{KVCacheDesc{
        .activeAllocationGroups = std::vector<AllocationGroupId>{AllocationGroupId{30}, AllocationGroupId{40}}}}}};
    return config;
}

class AllocationLifetimeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        ASSERT_EQ(cudaStreamCreate(&mOtherStream), cudaSuccess);
    }

    void TearDown() override
    {
        EXPECT_EQ(cudaStreamDestroy(mOtherStream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(mStream), cudaSuccess);
    }

    static LifeCycleId group(KvCacheManager const& manager, int layer)
    {
        return manager.getBufferGroupId(LayerId{layer}, "state");
    }

    static SlotCount freeSlots(KvCacheManager& manager, int layer)
    {
        auto& storage = manager.storage();
        return storage.getStatistics(kHotLevel, storage.getPoolGroupIndex(group(manager, layer))).free;
    }

    static void* address(KvCacheManager const& manager, KvCache const& cache, int layer)
    {
        int const slot = cache.getSsmBlockBaseIndex(group(manager, layer));
        return reinterpret_cast<void*>(manager.getMemPoolBaseAddress(LayerId{layer}, "state")
            + static_cast<size_t>(slot) * manager.getPageIndexScale(LayerId{layer}, "state")
                * manager.getPageStride(LayerId{layer}, "state"));
    }

    cudaStream_t mStream{};
    cudaStream_t mOtherStream{};
};

TEST_F(AllocationLifetimeTest, DeferredGroupHasNoAllocationUntilActivated)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto cache = manager->createKvCache();
    auto const before = freeSlots(*manager, 3);
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_EQ(cache->getSsmBlockBaseIndex(group(*manager, 3)), kBadPageIndex.value());
    EXPECT_EQ(freeSlots(*manager, 3), before);
    ASSERT_TRUE(cache->resize(64));
    EXPECT_EQ(freeSlots(*manager, 3), before);
    ASSERT_TRUE(cache->activateGroups({AllocationGroupId{30}}));
    int const slot = cache->getSsmBlockBaseIndex(group(*manager, 3));
    EXPECT_GE(slot, 0);
    EXPECT_EQ(freeSlots(*manager, 3), before - 1);
    ASSERT_TRUE(cache->activateGroups({AllocationGroupId{30}}));
    EXPECT_EQ(cache->getSsmBlockBaseIndex(group(*manager, 3)), slot);
    EXPECT_EQ(freeSlots(*manager, 3), before - 1);
    cache->close();
    EXPECT_EQ(freeSlots(*manager, 3), before);
    manager->shutdown();
}

TEST_F(AllocationLifetimeTest, SuspensionKeepsTwoGroupsAndReleasesReusableCapacity)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto& storage = manager->storage();
    auto const released = group(*manager, 3);
    auto const pool = storage.getPoolGroupIndex(released);
    auto reservations = storage.newSlotsForPoolGroup(kHotLevel, pool, freeSlots(*manager, 3) - 1);
    auto cleanup = FuncGuard(
        [&]()
        {
            for (auto& slot : reservations)
                storage.releaseSlot(released, kHotLevel, std::move(slot));
        });
    auto first = manager->createKvCache();
    ASSERT_TRUE(first->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{30}}));
    int const releasedSlot = first->getSsmBlockBaseIndex(released);
    std::array<int, 2> heldSlots{};
    for (int layer = 1; layer <= 2; ++layer)
    {
        heldSlots[layer - 1] = first->getSsmBlockBaseIndex(group(*manager, layer));
        ASSERT_EQ(cudaMemsetAsync(address(*manager, *first, layer), 10 + layer, 64, mStream), cudaSuccess);
    }
    first->suspend();
    EXPECT_EQ(freeSlots(*manager, 3), 1);
    auto second = manager->createKvCache();
    ASSERT_TRUE(second->resume(mOtherStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{30}}));
    EXPECT_EQ(second->getSsmBlockBaseIndex(released), releasedSlot);
    ASSERT_EQ(cudaMemsetAsync(address(*manager, *second, 3), 99, 64, mOtherStream), cudaSuccess);
    EXPECT_FALSE(first->resume(mStream));
    EXPECT_FALSE(first->isActive());
    second->close();
    ASSERT_TRUE(first->resume(mStream));
    for (int layer = 1; layer <= 2; ++layer)
    {
        EXPECT_EQ(first->getSsmBlockBaseIndex(group(*manager, layer)), heldSlots[layer - 1]);
        std::array<uint8_t, 64> data{};
        ASSERT_EQ(cudaMemcpyAsync(
                      data.data(), address(*manager, *first, layer), data.size(), cudaMemcpyDeviceToHost, mStream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
        EXPECT_TRUE(std::all_of(data.begin(), data.end(), [layer](auto value) { return value == 10 + layer; }));
    }
    first->suspend();
    ASSERT_TRUE(first->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{}));
    EXPECT_EQ(first->getSsmBlockBaseIndex(released), kBadPageIndex.value());
    EXPECT_EQ(freeSlots(*manager, 3), 1);
    first->close();
}

TEST_F(AllocationLifetimeTest, FailedActivationPreservesAllGroupsAndSelection)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(mStream));
    auto const releasedFree = freeSlots(*manager, 3);
    auto& storage = manager->storage();
    auto const blocked = group(*manager, 4);
    auto reservations
        = storage.newSlotsForPoolGroup(kHotLevel, storage.getPoolGroupIndex(blocked), freeSlots(*manager, 4));
    auto cleanup = FuncGuard(
        [&]()
        {
            for (auto& slot : reservations)
                storage.releaseSlot(blocked, kHotLevel, std::move(slot));
        });
    EXPECT_FALSE(cache->activateGroups({AllocationGroupId{30}, AllocationGroupId{40}}));
    EXPECT_EQ(freeSlots(*manager, 3), releasedFree);
    EXPECT_EQ(cache->getSsmBlockBaseIndex(group(*manager, 3)), kBadPageIndex.value());
    EXPECT_EQ(cache->getSsmBlockBaseIndex(blocked), kBadPageIndex.value());
    cache->suspend();
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_EQ(cache->getSsmBlockBaseIndex(group(*manager, 3)), kBadPageIndex.value());
    cache->close();
}

TEST_F(AllocationLifetimeTest, ReleasedSlotWaitsForTheLastCacheStreamRead)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto& storage = manager->storage();
    auto const released = group(*manager, 3);
    auto reservations
        = storage.newSlotsForPoolGroup(kHotLevel, storage.getPoolGroupIndex(released), freeSlots(*manager, 3) - 1);
    auto cleanup = FuncGuard(
        [&]()
        {
            for (auto& slot : reservations)
                storage.releaseSlot(released, kHotLevel, std::move(slot));
        });
    auto first = manager->createKvCache();
    ASSERT_TRUE(first->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{30}}));
    auto* source = address(*manager, *first, 3);
    auto* destination = address(*manager, *first, 1);
    ASSERT_EQ(cudaMemsetAsync(source, 17, 64, mStream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::atomic<bool> release{false};
    auto releaseStream = FuncGuard(
        [&]()
        {
            release.store(true, std::memory_order_release);
            EXPECT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
            EXPECT_EQ(cudaStreamSynchronize(mOtherStream), cudaSuccess);
        });
    ASSERT_EQ(cudaLaunchHostFunc(
                  mStream,
                  [](void* data)
                  {
                      auto& ready = *static_cast<std::atomic<bool>*>(data);
                      while (!ready.load(std::memory_order_acquire))
                          std::this_thread::yield();
                  },
                  &release),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(destination, source, 64, cudaMemcpyDeviceToDevice, mStream), cudaSuccess);
    first->suspend();
    auto second = manager->createKvCache();
    ASSERT_TRUE(second->resume(mOtherStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{30}}));
    EXPECT_EQ(address(*manager, *second, 3), source);
    ASSERT_EQ(cudaMemsetAsync(source, 99, 64, mOtherStream), cudaSuccess);
    EXPECT_EQ(cudaStreamQuery(mOtherStream), cudaErrorNotReady);
    release.store(true, std::memory_order_release);
    ASSERT_EQ(cudaStreamSynchronize(mOtherStream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(mStream), cudaSuccess);
    std::array<uint8_t, 64> data{};
    ASSERT_EQ(cudaMemcpy(data.data(), destination, data.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](auto value) { return value == 17; }));
    second->close();
    first->close();
}

TEST_F(AllocationLifetimeTest, FailedResumeDoesNotRememberAttemptedSelection)
{
    for (float limit : {0.9f, 1.f})
    {
        SCOPED_TRACE(limit);
        auto config = makeLifetimeConfig();
        config.maxUtilForResume = limit;
        auto manager = std::make_shared<KvCacheManager>(config);
        auto& storage = manager->storage();
        auto const blocked = group(*manager, 3);
        auto reservations
            = storage.newSlotsForPoolGroup(kHotLevel, storage.getPoolGroupIndex(blocked), freeSlots(*manager, 3));
        auto cleanup = FuncGuard(
            [&]()
            {
                for (auto& slot : reservations)
                    storage.releaseSlot(blocked, kHotLevel, std::move(slot));
            });
        auto cache = manager->createKvCache();
        EXPECT_FALSE(cache->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{30}}));
        ASSERT_TRUE(cache->resume(mStream));
        EXPECT_EQ(cache->getSsmBlockBaseIndex(blocked), kBadPageIndex.value());
        cache->close();
    }
}

TEST_F(AllocationLifetimeTest, HeldDeferredGroupsCannotBeDeselected)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{AllocationGroupId{40}}));
    int const slot = cache->getSsmBlockBaseIndex(group(*manager, 4));
    cache->suspend();
    EXPECT_THROW(cache->resume(mStream, std::nullopt, std::vector<AllocationGroupId>{}), std::runtime_error);
    EXPECT_FALSE(cache->isActive());
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_EQ(cache->getSsmBlockBaseIndex(group(*manager, 4)), slot);
    cache->close();
}

TEST_F(AllocationLifetimeTest, EveryBeamReallocatesReleasedFixedState)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(mStream));
    cache->setBeamWidth(BeamIndex{3});
    auto const before = freeSlots(*manager, 3);
    ASSERT_TRUE(cache->activateGroups({AllocationGroupId{30}}));
    EXPECT_EQ(freeSlots(*manager, 3), before - 3);
    cache->suspend();
    EXPECT_EQ(freeSlots(*manager, 3), before);
    ASSERT_TRUE(cache->resume(mStream));
    std::array<int, 3> slots;
    for (int beam = 0; beam < 3; ++beam)
    {
        slots[beam] = cache->getSsmBlockBaseIndex(group(*manager, 3), BeamIndex{beam});
        EXPECT_GE(slots[beam], 0);
    }
    std::sort(slots.begin(), slots.end());
    EXPECT_EQ(std::unique(slots.begin(), slots.end()), slots.end());
    EXPECT_EQ(freeSlots(*manager, 3), before - 3);
    cache->close();
}

TEST_F(AllocationLifetimeTest, SizingCountsOnlySelectedFixedGroups)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto const counts = KvCacheIntrospection::computeSlotsForBatch(*manager,
        BatchDesc{{KVCacheDesc{.capacity = 64},
            KVCacheDesc{.capacity = 128,
                .beamWidth = 3,
                .activeAllocationGroups = std::vector<AllocationGroupId>{AllocationGroupId{30}}}}},
        4, std::nullopt);
    auto const& storage = manager->storage();
    EXPECT_EQ(counts[storage.getPoolGroupIndex(group(*manager, 1))], 4);
    EXPECT_EQ(counts[storage.getPoolGroupIndex(group(*manager, 2))], 4);
    EXPECT_EQ(counts[storage.getPoolGroupIndex(group(*manager, 3))], 3);
    EXPECT_EQ(counts[storage.getPoolGroupIndex(group(*manager, 4))], 0);
    manager->shutdown();
}

TEST_F(AllocationLifetimeTest, BatchMembershipAndUnknownGroupsFailBeforeAllocation)
{
    auto manager = std::make_shared<KvCacheManager>(makeLifetimeConfig());
    auto cache = manager->createKvCache();
    ASSERT_TRUE(cache->resume(mStream));
    EXPECT_THROW(cache->activateGroups({AllocationGroupId{999}}), std::runtime_error);
    auto const before = freeSlots(*manager, 3);
    {
        Batch batch(*manager, mStream, {}, 1, 16);
        batch.addKvCache(*cache);
        EXPECT_THROW(cache->activateGroups({AllocationGroupId{30}}), LogicError);
        batch.removeKvCache(*cache);
    }
    EXPECT_EQ(freeSlots(*manager, 3), before);
    cache->close();
}
} // namespace
