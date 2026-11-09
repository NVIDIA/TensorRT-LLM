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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/batchedPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

std::vector<Slot> allocateSlots(SlotAllocator& allocator, std::size_t count)
{
    std::vector<Slot> slots;
    slots.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        slots.push_back(allocator.allocate());
    }
    return slots;
}

// Release slots[first, last). Slot's implicit move leaves the source's slot id
// intact (std::optional<SlotId> is trivially copyable), so hand ownership over
// with setSlot(), which resets the source and makes double release impossible.
void releaseSlots(
    SlotAllocator& allocator, std::vector<Slot>& slots, std::size_t first = 0, std::size_t last = SIZE_MAX)
{
    for (std::size_t i = first; i < std::min(last, slots.size()); ++i)
    {
        Slot slot;
        slot.setSlot(slots[i]);
        allocator.release(std::move(slot));
    }
}

// Pool rebalance shrinks a pool group whose new size is still above the slot-ID
// high-water mark. Regression for NVBug 6225866: finishShrink() used to compute
// the expected overflow count as numActiveSlots - targetCapacity, which goes
// negative here, so the count never matched and the shrink threw instead of
// completing. Mirrors TestSlotAllocatorShrink.test_shrink_underused_pool in
// tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py, which
// only covers the Python SlotAllocator.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkUnderusedPool)
{
    SlotAllocator allocator{SlotCount{184064}};
    auto slots = allocateSlots(allocator, 2048);
    releaseSlots(allocator, slots);
    EXPECT_EQ(allocator.numActiveSlots(), 2048);

    allocator.prepareForShrink(SlotCount{122624});
    EXPECT_EQ(allocator.numOverflowSlots(), 0);

    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.numSlots(), 122624);
    EXPECT_EQ(allocator.numActiveSlots(), 2048);
    EXPECT_FALSE(allocator.shrinkInProgress());
}

// The non-trivial migration path: every ID is issued, half are released, and the
// pool shrinks to half. The released overflow-range slots must be reclaimed by
// finishShrink(). Mirrors test_shrink_touched_pool.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkTouchedPool)
{
    SlotAllocator allocator{SlotCount{16}};
    auto slots = allocateSlots(allocator, 16);
    releaseSlots(allocator, slots, 8);
    EXPECT_EQ(allocator.numActiveSlots(), 16);

    allocator.prepareForShrink(SlotCount{8});
    EXPECT_EQ(allocator.numOverflowSlots(), 8);

    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.numSlots(), 8);
    EXPECT_EQ(allocator.numActiveSlots(), 8);
    EXPECT_FALSE(allocator.shrinkInProgress());

    // Leave the allocator quiescent so the debug-build destructor checks pass.
    releaseSlots(allocator, slots, 0, 8);
}

// SlotId is signed, so an upper-bound-only test admits negative ids, which convert to a huge
// size_t and index far outside the pool.
TEST(KvCacheManagerV2SlotAllocatorTest, RejectsSlotIdsOutsideThePool)
{
    SlotAllocator allocator{SlotCount{8}};

    EXPECT_FALSE(allocator.isValidSlotId(SlotId{-1}));
    EXPECT_FALSE(allocator.isValidSlotId(SlotId{std::numeric_limits<SlotCount>::min()}));
    EXPECT_TRUE(allocator.isValidSlotId(SlotId{0}));
    EXPECT_TRUE(allocator.isValidSlotId(SlotId{7}));
    EXPECT_FALSE(allocator.isValidSlotId(SlotId{8}));
}

// ---------------------------------------------------------------------------
// Windows (copy-based rebalancing): ids are arena-relative and the window can
// move at either end.
// ---------------------------------------------------------------------------

std::vector<SlotCount> idsOf(std::vector<Slot> const& slots)
{
    std::vector<SlotCount> ids;
    for (auto const& s : slots)
        ids.push_back(s.slotId().value());
    return ids;
}

SlotWindow window(SlotCount begin, SlotCount end)
{
    return SlotWindow{SlotId{begin}, SlotId{end}};
}

TEST(KvCacheManagerV2SlotAllocatorTest, WindowIssuesAwayFromTheMovingEdge)
{
    SlotAllocator up{window(100, 110), SlotAllocator::Growth::kAscending};
    auto upSlots = allocateSlots(up, 3);
    EXPECT_EQ(idsOf(upSlots), (std::vector<SlotCount>{100, 101, 102}));
    EXPECT_FALSE(up.isValidSlotId(SlotId{99}));
    EXPECT_TRUE(up.isValidSlotId(SlotId{109}));
    EXPECT_FALSE(up.isValidSlotId(SlotId{110}));

    SlotAllocator down{window(100, 110), SlotAllocator::Growth::kDescending};
    auto downSlots = allocateSlots(down, 3);
    EXPECT_EQ(idsOf(downSlots), (std::vector<SlotCount>{109, 108, 107}));
    EXPECT_EQ(down.numFreeSlots(), 7);

    releaseSlots(up, upSlots);
    releaseSlots(down, downSlots);
}

// The window's begin moves up past ids that were never issued: nothing to migrate.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkWindowBeginUnderused)
{
    SlotAllocator allocator{window(100, 116), SlotAllocator::Growth::kDescending};
    auto slots = allocateSlots(allocator, 4); // 115..112
    EXPECT_EQ(allocator.numIssuedOutside(window(104, 116)), 0);

    allocator.prepareForShrinkWindow(window(104, 116));
    EXPECT_EQ(allocator.numOverflowSlots(), 0);
    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.window(), window(104, 116));
    EXPECT_EQ(allocator.numSlots(), 12);
    EXPECT_EQ(allocator.numFreeSlots(), 8);
    // Ids that stayed inside the window keep their value.
    EXPECT_EQ(idsOf(slots), (std::vector<SlotCount>{115, 114, 113, 112}));
    releaseSlots(allocator, slots);
}

// The window's begin moves past occupied ids: they are overflow until released, and the shrink
// cannot finish before then.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkWindowBeginTouched)
{
    SlotAllocator allocator{window(0, 16), SlotAllocator::Growth::kAscending};
    auto slots = allocateSlots(allocator, 16);
    EXPECT_EQ(allocator.numIssuedOutside(window(8, 16)), 8);

    allocator.prepareForShrinkWindow(window(8, 16));
    EXPECT_TRUE(allocator.shrinkInProgress());
    EXPECT_EQ(allocator.numFreeSlots(), 0);
    EXPECT_EQ(allocator.getSlotsBlockingShrink().size(), 8u);

    releaseSlots(allocator, slots, 0, 7);
    EXPECT_THROW(allocator.finishShrink(), std::runtime_error); // id 7 still occupied
    releaseSlots(allocator, slots, 7, 8);
    EXPECT_EQ(allocator.numOverflowSlots(), 8);
    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.window(), window(8, 16));
    EXPECT_EQ(allocator.numActiveSlots(), 8);
    releaseSlots(allocator, slots, 8, 16);
}

// The target holds no issued id at all: fresh ids must come from inside the target while the
// old ones are still being vacated.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkWindowPastAllIssuedIds)
{
    SlotAllocator allocator{window(0, 16), SlotAllocator::Growth::kAscending};
    auto old = allocateSlots(allocator, 4); // 0..3
    allocator.prepareForShrinkWindow(window(8, 16));
    EXPECT_EQ(allocator.numIssuedOutsideTarget(), 4);
    EXPECT_EQ(allocator.numFreeSlots(), 8);

    // The migration destinations a shrink allocates while it is in progress.
    auto moved = allocateSlots(allocator, 4);
    EXPECT_EQ(idsOf(moved), (std::vector<SlotCount>{8, 9, 10, 11}));

    releaseSlots(allocator, old);
    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.numActiveSlots(), 4);
    EXPECT_EQ(allocator.numFreeSlots(), 4);
    releaseSlots(allocator, moved);
}

// Shrinking to an empty window at the begin (what PoolGroupBase::destroy does) while the issued
// run sits at the other end. Regression: the clipped run was anchored outside the target.
TEST(KvCacheManagerV2SlotAllocatorTest, ShrinkDescendingWindowToEmpty)
{
    SlotAllocator allocator{window(100, 116), SlotAllocator::Growth::kDescending};
    auto slots = allocateSlots(allocator, 4); // 115..112
    releaseSlots(allocator, slots);
    allocator.synchronize();
    allocator.prepareForShrink(SlotCount{0});
    EXPECT_EQ(allocator.numOverflowSlots(), 4);
    EXPECT_TRUE(allocator.finishShrink());
    EXPECT_EQ(allocator.window(), window(100, 100));
    EXPECT_EQ(allocator.numActiveSlots(), 0);
}

// Growing at the begin makes the new ids free; ascending growth falls back to them once the
// end is used up.
TEST(KvCacheManagerV2SlotAllocatorTest, ExpandWindowAtBegin)
{
    SlotAllocator allocator{window(8, 16), SlotAllocator::Growth::kAscending};
    auto slots = allocateSlots(allocator, 8);
    EXPECT_EQ(allocator.numFreeSlots(), 0);

    allocator.expandWindow(window(4, 16));
    EXPECT_EQ(allocator.numFreeSlots(), 4);
    auto more = allocateSlots(allocator, 4);
    EXPECT_EQ(idsOf(more), (std::vector<SlotCount>{7, 6, 5, 4}));
    EXPECT_THROW(allocator.allocate(), OutOfPagesError);

    releaseSlots(allocator, slots);
    releaseSlots(allocator, more);
}

// Copy-based rebalancing vacates slots by position, so TP ranks must reach identical slot layouts. With
// Recycling::kReleaseOrder, which slot comes back must not depend on whether a recycled slot's event has
// completed: a recycled slot whose event is still pending is handed out before a fresh id. kReadyFirst (the
// default) issues a fresh id instead. Arena pool groups are constructed with kReleaseOrder.
TEST(KvCacheManagerV2SlotAllocatorTest, ReleaseOrderRecyclingIgnoresEventReadiness)
{
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    // Hold the stream on a host function until the checks are done, so events recorded behind it stay pending.
    std::atomic<bool> go{false};
    ASSERT_EQ(cudaLaunchHostFunc(
                  stream,
                  [](void* flag)
                  {
                      while (!static_cast<std::atomic<bool>*>(flag)->load())
                      {
                      }
                  },
                  &go),
        cudaSuccess);

    SlotAllocator readyFirst{window(0, 8), SlotAllocator::Growth::kAscending};
    SlotAllocator releaseOrder{
        window(0, 8), SlotAllocator::Growth::kAscending, SlotAllocator::Recycling::kReleaseOrder};
    EXPECT_EQ(readyFirst.recycling(), SlotAllocator::Recycling::kReadyFirst);
    EXPECT_EQ(releaseOrder.recycling(), SlotAllocator::Recycling::kReleaseOrder);

    std::vector<Slot> held;
    for (SlotAllocator* allocator : {&readyFirst, &releaseOrder})
    {
        auto slots = allocateSlots(*allocator, 2); // ids 0, 1
        EXPECT_EQ(idsOf(slots), (std::vector<SlotCount>{0, 1}));
        slots[0].readyEvent = CachedCudaEvent(reinterpret_cast<CudaStream>(stream));
        ASSERT_FALSE(slots[0].queryReady());
        releaseSlots(*allocator, slots, 0, 1); // id 0 recycled with its event pending
        held.push_back(allocator->allocate());
        held.push_back(std::move(slots[1]));
    }
    EXPECT_EQ(held[0].slotId(), SlotId{2}); // kReadyFirst: a fresh id rather than wait
    EXPECT_EQ(held[2].slotId(), SlotId{0}); // kReleaseOrder: the pending recycled slot

    go.store(true);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<Slot> readyFirstSlots;
    std::vector<Slot> releaseOrderSlots;
    readyFirstSlots.push_back(std::move(held[0]));
    readyFirstSlots.push_back(std::move(held[1]));
    releaseOrderSlots.push_back(std::move(held[2]));
    releaseOrderSlots.push_back(std::move(held[3]));
    releaseSlots(readyFirst, readyFirstSlots);
    releaseSlots(releaseOrder, releaseOrderSlots);
    readyFirst.synchronize();
    releaseOrder.synchronize();
    // readyFirst still recycles slot 0 (its event has completed now) before issuing another fresh id.
    Slot again = readyFirst.allocate();
    EXPECT_EQ(again.slotId(), SlotId{0});
    std::vector<Slot> last;
    last.push_back(std::move(again));
    releaseSlots(readyFirst, last);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Arena layout.
// ---------------------------------------------------------------------------

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;

TypedVec<PoolGroupIndex, SlotWindow> arenaWindows(
    size_t arenaBytes, std::vector<size_t> sizes, std::vector<float> ratios, std::vector<SlotCount> minSlots)
{
    return ArenaGpuCacheLevelStorage::computeArenaWindows(arenaBytes, TypedVec<PoolGroupIndex, size_t>{sizes},
        TypedVec<PoolGroupIndex, float>{ratios}, TypedVec<PoolGroupIndex, SlotCount>{minSlots});
}

// Gemma-3-1B block sizes (full attention 128 KiB, sliding window 704 KiB).
TEST(KvCacheManagerV2ArenaLayoutTest, WindowsTileTheArena)
{
    size_t const arena = 1024 * kMiB;
    std::vector<size_t> const sizes{128 * kKiB, 704 * kKiB};
    auto const w = arenaWindows(arena, sizes, {0.5f, 0.5f}, {1, 1});

    size_t const end0 = toSizeT(w[PoolGroupIndex{0}].end) * sizes[0];
    size_t const begin1 = toSizeT(w[PoolGroupIndex{1}].begin) * sizes[1];
    EXPECT_EQ(w[PoolGroupIndex{0}].begin, SlotId{0});
    EXPECT_LE(end0, begin1);            // no overlap
    EXPECT_LT(begin1 - end0, sizes[0]); // under one (smaller) block wasted at the boundary
    EXPECT_EQ(begin1 % sizes[1], 0u);   // snapped to the larger block size
    EXPECT_EQ(toSizeT(w[PoolGroupIndex{1}].end), arena / sizes[1]);
    EXPECT_NEAR(static_cast<double>(begin1) / arena, 0.5, 0.01);
}

// A rebalance moves only the shared boundary; each window keeps its outer edge, so blocks away
// from the boundary keep their ids. The same ratio always gives the same windows.
TEST(KvCacheManagerV2ArenaLayoutTest, RebalanceMovesOnlyTheBoundary)
{
    size_t const arena = 1024 * kMiB;
    std::vector<size_t> const sizes{128 * kKiB, 704 * kKiB};
    auto const before = arenaWindows(arena, sizes, {0.5f, 0.5f}, {1, 1});
    auto const after = arenaWindows(arena, sizes, {0.3f, 0.7f}, {1, 1});
    EXPECT_EQ(after[PoolGroupIndex{0}].begin, before[PoolGroupIndex{0}].begin);
    EXPECT_EQ(after[PoolGroupIndex{1}].end, before[PoolGroupIndex{1}].end);
    EXPECT_LT(after[PoolGroupIndex{0}].end, before[PoolGroupIndex{0}].end);
    EXPECT_LT(after[PoolGroupIndex{1}].begin, before[PoolGroupIndex{1}].begin);
    EXPECT_EQ(arenaWindows(arena, sizes, {0.3f, 0.7f}, {1, 1})[PoolGroupIndex{1}], after[PoolGroupIndex{1}]);
}

TEST(KvCacheManagerV2ArenaLayoutTest, ThreeGroupsAndMinSlots)
{
    size_t const arena = 256 * kMiB;
    std::vector<size_t> const sizes{2 * kMiB, 10 * kMiB, 768 * kKiB};
    auto const w = arenaWindows(arena, sizes, {0.98f, 0.01f, 0.01f}, {1, 4, 50});
    size_t previousEnd = 0;
    for (PoolGroupIndex pg{0}; pg < PoolGroupIndex{3}; ++pg)
    {
        size_t const s = sizes[toSizeT(pg)];
        EXPECT_GE(toSizeT(w[pg].begin) * s, previousEnd);
        previousEnd = toSizeT(w[pg].end) * s;
    }
    EXPECT_LE(previousEnd, arena);
    EXPECT_GE(w[PoolGroupIndex{1}].size(), 4);
    EXPECT_GE(w[PoolGroupIndex{2}].size(), 50);
}

TEST(KvCacheManagerV2ArenaLayoutTest, RejectsArenaTooSmallForMinSlots)
{
    EXPECT_THROW(arenaWindows(32 * kMiB, {2 * kMiB, 10 * kMiB}, {0.5f, 0.5f}, {4, 4}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Arena layout with several pools per group (bound pools).
// ---------------------------------------------------------------------------

using PoolSizes = std::vector<std::vector<size_t>>;

TypedVec<PoolGroupIndex, TypedVec<PoolIndex, SlotWindow>> arenaPoolWindows(
    size_t arenaBytes, PoolSizes const& sizes, std::vector<float> ratios, std::vector<SlotCount> minSlots)
{
    TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> typed;
    for (auto const& group : sizes)
        typed.push_back(TypedVec<PoolIndex, size_t>{group});
    return ArenaGpuCacheLevelStorage::computeArenaPoolWindows(
        arenaBytes, typed, TypedVec<PoolGroupIndex, float>{ratios}, TypedVec<PoolGroupIndex, SlotCount>{minSlots});
}

// DeepSeek-V4-Flash per rank (fp8 KV, 256 tokens per block): full attention, sliding window, and
// compressor-state groups.
PoolSizes const kDsv4Sizes{{688128, 91392, 20480}, {20971520, 5636096}, {44040192, 11010048}};

// Every pool's window, in bytes, in layout order; checks they lie in order without overlap and that
// each group's pools have one slot count. Returns the bytes no pool uses.
size_t checkPoolLayout(
    size_t arena, PoolSizes const& sizes, TypedVec<PoolGroupIndex, TypedVec<PoolIndex, SlotWindow>> const& w)
{
    size_t previousEnd = 0;
    size_t used = 0;
    for (size_t g = 0; g < sizes.size(); ++g)
    {
        PoolGroupIndex const pg{static_cast<int>(g)};
        EXPECT_EQ(w[pg].size(), PoolIndex{static_cast<int>(sizes[g].size())});
        for (size_t p = 0; p < sizes[g].size(); ++p)
        {
            PoolIndex const pool{static_cast<int>(p)};
            size_t const s = sizes[g][p];
            EXPECT_EQ(w[pg][pool].size(), w[pg][PoolIndex{0}].size()) << "group " << g << " pool " << p;
            EXPECT_GE(toSizeT(w[pg][pool].begin) * s, previousEnd) << "group " << g << " pool " << p;
            previousEnd = toSizeT(w[pg][pool].end) * s;
            used += slotCountToSizeT(w[pg][pool].size()) * s;
        }
    }
    EXPECT_LE(previousEnd, arena);
    return arena - used;
}

// With one pool per group the general layout is exactly computeArenaWindows.
TEST(KvCacheManagerV2ArenaPoolLayoutTest, SinglePoolsMatchGroupLayout)
{
    size_t const arena = 1024 * kMiB;
    auto const pools = arenaPoolWindows(arena, {{128 * kKiB}, {704 * kKiB}}, {0.3f, 0.7f}, {1, 1});
    auto const groups = arenaWindows(arena, {128 * kKiB, 704 * kKiB}, {0.3f, 0.7f}, {1, 1});
    for (PoolGroupIndex pg{0}; pg < PoolGroupIndex{2}; ++pg)
    {
        ASSERT_EQ(pools[pg].size(), PoolIndex{1});
        EXPECT_EQ(pools[pg][PoolIndex{0}], groups[pg]);
    }
}

// Pools lie one after another, each group's pools share one slot count, the shares follow the
// ratio, and every pool wastes under two of its own blocks.
TEST(KvCacheManagerV2ArenaPoolLayoutTest, DeepSeekV4PoolsTileTheArena)
{
    size_t const arena = 100 * 1024 * kMiB;
    for (auto const& ratio :
        {std::vector<float>{0.95976f, 0.01311f, 0.02713f}, std::vector<float>{0.11645f, 0.30154f, 0.58200f}})
    {
        auto const w = arenaPoolWindows(arena, kDsv4Sizes, ratio, {1, 1, 1});
        size_t const unused = checkPoolLayout(arena, kDsv4Sizes, w);
        size_t bound = 0;
        for (auto const& group : kDsv4Sizes)
            for (size_t const s : group)
                bound += 2 * s;
        EXPECT_LT(unused, bound);
        for (size_t g = 0; g < kDsv4Sizes.size(); ++g)
        {
            PoolGroupIndex const pg{static_cast<int>(g)};
            size_t perSlot = 0;
            for (size_t const s : kDsv4Sizes[g])
                perSlot += s;
            double const share = static_cast<double>(slotCountToSizeT(w[pg][PoolIndex{0}].size()) * perSlot)
                / static_cast<double>(arena);
            EXPECT_NEAR(share, ratio[g], 0.01) << "group " << g;
        }
    }
}

TEST(KvCacheManagerV2ArenaPoolLayoutTest, SameRatioSameWindowsAndMinSlots)
{
    size_t const arena = 4 * 1024 * kMiB;
    auto const a = arenaPoolWindows(arena, kDsv4Sizes, {0.98f, 0.01f, 0.01f}, {1, 3, 5});
    auto const b = arenaPoolWindows(arena, kDsv4Sizes, {0.98f, 0.01f, 0.01f}, {1, 3, 5});
    checkPoolLayout(arena, kDsv4Sizes, a);
    for (PoolGroupIndex pg{0}; pg < PoolGroupIndex{3}; ++pg)
    {
        for (PoolIndex pool{0}; pool < a[pg].size(); ++pool)
            EXPECT_EQ(a[pg][pool], b[pg][pool]);
    }
    EXPECT_GE(a[PoolGroupIndex{1}][PoolIndex{0}].size(), 3);
    EXPECT_GE(a[PoolGroupIndex{2}][PoolIndex{0}].size(), 5);
    EXPECT_THROW(arenaPoolWindows(64 * kMiB, kDsv4Sizes, {0.5f, 0.25f, 0.25f}, {1, 2, 2}), std::invalid_argument);
}

// An arena of minArenaBytes lays out for any ratio with every group at its minimum; StorageManager raises
// a smaller quota (e.g. the estimation pass's, from free memory) to it, as the VMM path does.
TEST(KvCacheManagerV2ArenaPoolLayoutTest, MinArenaBytesFitsMinSlots)
{
    std::vector<std::vector<float>> const ratios{
        {0.98f, 0.01f, 0.01f}, {0.01f, 0.98f, 0.01f}, {0.01f, 0.01f, 0.98f}, {0.34f, 0.33f, 0.33f}};
    for (auto const& sizes : {kDsv4Sizes, PoolSizes{{2 * kMiB}, {10 * kMiB}, {768 * kKiB}}})
    {
        TypedVec<PoolGroupIndex, TypedVec<PoolIndex, size_t>> typed;
        for (auto const& group : sizes)
            typed.push_back(TypedVec<PoolIndex, size_t>{group});
        std::vector<SlotCount> const minSlots{40, 45, 45};
        size_t const arena
            = ArenaGpuCacheLevelStorage::minArenaBytes(typed, TypedVec<PoolGroupIndex, SlotCount>{minSlots});
        for (auto const& ratio : ratios)
        {
            auto const w = arenaPoolWindows(arena, sizes, ratio, minSlots);
            checkPoolLayout(arena, sizes, w);
            for (size_t g = 0; g < sizes.size(); ++g)
                EXPECT_GE(w[PoolGroupIndex{static_cast<int>(g)}][PoolIndex{0}].size(), minSlots[g]) << "group " << g;
        }
    }
}

// ---------------------------------------------------------------------------
// Bound pools: a group's slot ids map to each bound pool's own ids.
// ---------------------------------------------------------------------------

// No memory is touched: the arena base is only used for address arithmetic.
constexpr MemAddress kFakeArenaBase = MemAddress(1) << 40;

GpuPoolGroup makeBoundGroup(SlotWindow window0, SlotWindow window1, size_t arenaBytes)
{
    return GpuPoolGroup(TypedVec<PoolIndex, SlotWindow>{window0, window1}, SlotAllocator::Growth::kAscending,
        TypedVec<PoolIndex, size_t>{4 * kKiB, 1 * kKiB}, kFakeArenaBase, arenaBytes, size_t{2} << 20);
}

TEST(KvCacheManagerV2BoundPoolTest, BindsInOrderAndResolvesAddresses)
{
    auto group = makeBoundGroup(SlotWindow{SlotId{0}, SlotId{8}}, SlotWindow{SlotId{40}, SlotId{48}}, kMiB);
    EXPECT_FALSE(group.isBoundPool(PoolIndex{0}));
    EXPECT_TRUE(group.isBoundPool(PoolIndex{1}));
    auto const& pool = group.arenaPool(PoolIndex{1});
    for (SlotId s{0}; s < SlotId{8}; ++s)
    {
        EXPECT_EQ(pool.localId(s), SlotId{40 + s.value()});
        auto const addrs = group.slotAddress(s);
        EXPECT_EQ(std::get<MemAddress>(addrs[PoolIndex{0}]), kFakeArenaBase + toSizeT(s) * 4 * kKiB);
        EXPECT_EQ(std::get<MemAddress>(addrs[PoolIndex{1}]), kFakeArenaBase + (40 + toSizeT(s)) * kKiB);
    }
    // Pool 1 may address any id up to the arena end in its own (smaller) slots.
    EXPECT_EQ(group.slotIdUpperBound(PoolIndex{0}), SlotCount{256});
    EXPECT_EQ(group.slotIdUpperBound(PoolIndex{1}), SlotCount{1024});
}

// A move swaps bindings: the block keeps its bound-pool data, the vacated id takes the other one.
TEST(KvCacheManagerV2BoundPoolTest, MoveSwapsBindings)
{
    auto group = makeBoundGroup(SlotWindow{SlotId{0}, SlotId{8}}, SlotWindow{SlotId{40}, SlotId{48}}, kMiB);
    auto& pool = group.arenaPool(PoolIndex{1});
    group.swapBoundPoolBindings(SlotId{6}, SlotId{1});
    EXPECT_EQ(pool.localId(SlotId{6}), SlotId{41});
    EXPECT_EQ(pool.localId(SlotId{1}), SlotId{46});
    // An unbound destination takes the binding and leaves the source unbound.
    pool.unbind(SlotId{7});
    group.swapBoundPoolBindings(SlotId{7}, SlotId{2});
    EXPECT_EQ(pool.localId(SlotId{7}), SlotId{42});
    EXPECT_FALSE(pool.localId(SlotId{2}).has_value());
    EXPECT_FALSE(pool.groupId(SlotId{47}).has_value());
}

// After a rebalance every id in the group's window is bound again inside the pool's new window;
// ids already bound there keep their binding, ids outside the window lose theirs.
TEST(KvCacheManagerV2BoundPoolTest, CompleteBindingsAfterMove)
{
    auto group = makeBoundGroup(SlotWindow{SlotId{0}, SlotId{8}}, SlotWindow{SlotId{40}, SlotId{48}}, kMiB);
    auto& pool = group.arenaPool(PoolIndex{1});
    // The group's window becomes [2, 10), pool 1's [44, 52). A remap has already dropped the
    // bindings of free ids and moved the one block in use (id 3) into the new window.
    for (SlotId s{0}; s < SlotId{8}; ++s)
        pool.unbind(s);
    pool.setWindow(SlotWindow{SlotId{40}, SlotId{52}});
    pool.bind(SlotId{3}, SlotId{50}); // a block in use, already moved into the new window
    pool.bind(SlotId{0}, SlotId{41}); // stale binding of an id about to leave the group's window
    pool.setWindow(SlotWindow{SlotId{44}, SlotId{52}});
    group.slotAllocator().expandWindow(SlotWindow{SlotId{0}, SlotId{10}});
    group.slotAllocator().prepareForShrinkWindow(SlotWindow{SlotId{2}, SlotId{10}});
    group.slotAllocator().finishShrink();
    group.completeBindings();
    EXPECT_FALSE(pool.localId(SlotId{0}).has_value());
    EXPECT_EQ(pool.localId(SlotId{3}), SlotId{50});
    std::vector<int> seen;
    for (SlotId s{2}; s < SlotId{10}; ++s)
    {
        auto const local = pool.localId(s);
        ASSERT_TRUE(local.has_value());
        EXPECT_TRUE(pool.window().contains(*local));
        seen.push_back(local->value());
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_TRUE(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
}

// Host arena (copy-based rebalancing): a page that straddles two host registrations is copied by the copy engine
// in one piece per registration once its range is listed with addHostCopySplitRange. Without the listing the driver
// rejects the copy, which is what the second half checks where the copy engine is used.
TEST(HostArenaCopySplit, StraddlingPagesAreSplitAtRegistrationBoundaries)
{
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess); // a current context for the copier and the registrations
    constexpr size_t kUnit = 2ULL << 20;
    constexpr uint32_t kPageBytes = 80 << 10;  // 2 MiB is not a multiple: page 25 spans [2000 KiB, 2080 KiB)
    HostMem host(2 * kUnit, 2 * kUnit, HostMemBackingOptions{kUnit, true}, /*waitForFill=*/true);
    auto* bytes = reinterpret_cast<uint8_t*>(host.address());
    for (size_t i = 0; i < 2 * kUnit; ++i)
        bytes[i] = static_cast<uint8_t>(i * 7 + 3);
    std::vector<PageIndexPair> const onboard{{0, 24}, {1, 25}, {2, 26}}; // host pages 24..26 -> device 0..2
    ASSERT_LT(24 * size_t{kPageBytes}, kUnit);
    ASSERT_GT(26 * size_t{kPageBytes}, kUnit);

    void* device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, 3 * kPageBytes), cudaSuccess);
    void* devicePairs = nullptr;
    ASSERT_EQ(cudaMalloc(&devicePairs, sizeof(PageIndexPair) * 3), cudaSuccess);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    BatchedPageCopier copier;
    auto pairsFor = [&](std::vector<PageIndexPair> const& pairs) -> PageIndexPair const*
    {
        if (copier.pageIndexLocation() == PageIndexLocation::kHost)
            return pairs.data();
        EXPECT_EQ(cudaMemcpy(devicePairs, pairs.data(), sizeof(PageIndexPair) * pairs.size(), cudaMemcpyHostToDevice),
            cudaSuccess);
        return static_cast<PageIndexPair const*>(devicePairs);
    };
    auto args = [&](CUdeviceptr dstBase, CUdeviceptr srcBase, std::vector<PageIndexPair> const& pairs)
    {
        PoolCopyArgs a;
        a.dstBase = dstBase;
        a.srcBase = srcBase;
        a.dstStride = kPageBytes;
        a.srcStride = kPageBytes;
        a.bytesPerPage = kPageBytes;
        a.pairs = pairsFor(pairs);
        a.numPairs = static_cast<uint32_t>(pairs.size());
        return a;
    };
    auto const hostBase = static_cast<CUdeviceptr>(host.address());
    auto const deviceBase = reinterpret_cast<CUdeviceptr>(device);

    addHostCopySplitRange(host.address(), 2 * kUnit, kUnit);
    copier.launch(args(deviceBase, hostBase, onboard), CopyDirection::kH2D, reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<uint8_t> back(3 * kPageBytes);
    ASSERT_EQ(cudaMemcpy(back.data(), device, back.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(0, std::memcmp(back.data(), bytes + 24 * size_t{kPageBytes}, back.size()));

    // Offload back into the zeroed straddling page.
    std::memset(bytes + 25 * size_t{kPageBytes}, 0, kPageBytes);
    std::vector<PageIndexPair> const offload{{25, 1}};
    copier.launch(args(hostBase, deviceBase, offload), CopyDirection::kD2H, reinterpret_cast<CUstream>(stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(0, std::memcmp(bytes + 25 * size_t{kPageBytes}, back.data() + kPageBytes, kPageBytes));

    removeHostCopySplitRange(host.address());
    if (copier.pageIndexLocation() == PageIndexLocation::kHost)
    {
        // The copy engine: unsplit, the straddling page is rejected.
        EXPECT_ANY_THROW(copier.launch(
            args(deviceBase, hostBase, onboard), CopyDirection::kH2D, reinterpret_cast<CUstream>(stream)));
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(devicePairs);
    cudaFree(device);
    host.destroy();
}

} // namespace
