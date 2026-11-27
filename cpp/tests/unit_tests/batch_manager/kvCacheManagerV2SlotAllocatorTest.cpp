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
#include <cstdlib>
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

    // Leave the allocator quiescent so the debug-build destructor checks pass. release() does not
    // scan readiness, so synchronize() marks the recycled slots ready.
    releaseSlots(allocator, slots, 0, 8);
    allocator.synchronize();
}

std::vector<SlotCount> slotIds(std::vector<Slot> const& slots)
{
    std::vector<SlotCount> ids;
    for (auto const& slot : slots)
    {
        ids.push_back(slot.slotId().value());
    }
    return ids;
}

// Released slots come back in release order before fresh ids, so allocate() must scan readiness when
// it could still hand out a fresh id; once every id is issued it takes the front released slot. These
// slots carry no event, so a scanned slot is ready.
TEST(KvCacheManagerV2SlotAllocatorTest, ReusesReleasedSlotsInReleaseOrder)
{
    SlotAllocator allocator{SlotCount{4}};
    auto first = allocateSlots(allocator, 2);
    std::swap(first[0], first[1]);
    releaseSlots(allocator, first);
    auto second = allocateSlots(allocator, 4);
    EXPECT_EQ(slotIds(second), (std::vector<SlotCount>{1, 0, 2, 3}));

    std::vector<Slot> released;
    released.push_back(std::move(second[2]));
    released.push_back(std::move(second[0]));
    releaseSlots(allocator, released);
    auto third = allocateSlots(allocator, 2);
    EXPECT_EQ(slotIds(third), (std::vector<SlotCount>{2, 1}));

    releaseSlots(allocator, third);
    releaseSlots(allocator, second, 1, 2);
    releaseSlots(allocator, second, 3, 4);
    EXPECT_EQ(allocator.numOccupiedSlots(), 0);
    allocator.synchronize();
}

// The lazy scrub must keep the eager scrub's choice: allocate() hands out recycled slots from the
// front, in release order, and only then never-issued ids.
TEST(KvCacheManagerV2SlotAllocatorTest, LazyScrubReusesReleasedSlotsBeforeFreshIds)
{
    SlotAllocator allocator{SlotCount{8}};
    auto slots = allocateSlots(allocator, 4);
    std::vector<Slot> reordered;
    for (std::size_t const i : {std::size_t{2}, std::size_t{0}, std::size_t{3}, std::size_t{1}})
    {
        reordered.emplace_back();
        reordered.back().setSlot(slots[i]);
    }
    releaseSlots(allocator, reordered);

    auto again = allocateSlots(allocator, 5);
    std::vector<SlotId::ValueType> ids;
    for (auto const& slot : again)
    {
        ids.push_back(slot.slotId().value());
    }
    EXPECT_EQ(ids, (std::vector<SlotId::ValueType>{2, 0, 3, 1, 4}));

    releaseSlots(allocator, again);
}

// An allocator destroyed right after its slots come back, with no synchronize() in between, must pass
// the debug-build destructor checks: the released slots carry no pending event. The checks run only
// with TLLM_DEBUG_MODE=1, latched at library load, so the destructor runs in a child process that
// inherits it; without the release-time scrub the child aborts in the destructor.
TEST(KvCacheManagerV2SlotAllocatorDeathTest, DestroyedRightAfterRelease)
{
    {
        SlotAllocator allocator{SlotCount{4}};
        auto slots = allocateSlots(allocator, 4);
        releaseSlots(allocator, slots);
        EXPECT_EQ(allocator.numFreeSlots(), 4);
    }
    ::setenv("TLLM_DEBUG_MODE", "1", 1);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            {
                SlotAllocator allocator{SlotCount{4}};
                auto slots = allocateSlots(allocator, 4);
                releaseSlots(allocator, slots);
            }
            std::exit(0);
        },
        ::testing::ExitedWithCode(0), "");
    ::unsetenv("TLLM_DEBUG_MODE");
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

} // namespace
