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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// ---------------------------------------------------------------------------
// Arena layout.
// ---------------------------------------------------------------------------

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;

TypedVec<PoolGroupIndex, SlotWindow> arenaWindows(size_t arenaBytes, std::vector<size_t> sizes,
    std::vector<float> ratios, std::vector<SlotCount> minSlots)
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
    EXPECT_LE(end0, begin1);                 // no overlap
    EXPECT_LT(begin1 - end0, sizes[0]);      // under one (smaller) block wasted at the boundary
    EXPECT_EQ(begin1 % sizes[1], 0u);        // snapped to the larger block size
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

} // namespace
