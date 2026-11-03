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

std::vector<SlotId> slotIds(std::vector<Slot> const& slots)
{
    std::vector<SlotId> ids;
    ids.reserve(slots.size());
    for (auto const& slot : slots)
    {
        ids.push_back(slot.slotId());
    }
    return ids;
}

// Released slots come back first-in first-out ahead of never-issued ids, whether the next request
// takes them one at a time or in one allocateMultiple() call. Slots here carry no event (always
// ready), so release() alone makes them reusable.
TEST(KvCacheManagerV2SlotAllocatorTest, ReleaseHeavyThenAllocateHeavyReusesRecycledSlotsInOrder)
{
    SlotAllocator allocator{SlotCount{32}};
    auto slots = allocateSlots(allocator, 16);
    ASSERT_EQ(slotIds(slots).front(), SlotId{0});
    ASSERT_EQ(slotIds(slots).back(), SlotId{15});

    // Release every other slot, then the rest: FIFO recycle order 0, 2, ..., 14, 1, 3, ..., 15.
    std::vector<SlotId> expected;
    for (std::size_t first : {std::size_t{0}, std::size_t{1}})
    {
        for (std::size_t i = first; i < slots.size(); i += 2)
        {
            expected.push_back(slots[i].slotId());
            releaseSlots(allocator, slots, i, i + 1);
        }
    }
    EXPECT_EQ(allocator.numOccupiedSlots(), 0);

    auto batch = allocator.allocateMultiple(SlotCount{12});
    auto single = allocator.allocate();
    auto rest = allocator.allocateMultiple(SlotCount{5});
    std::vector<SlotId> got = slotIds(batch);
    got.push_back(single.slotId());
    for (SlotId id : slotIds(rest))
    {
        got.push_back(id);
    }
    // 16 recycled slots in release order, then never-issued ids 16 and 17.
    expected.push_back(SlotId{16});
    expected.push_back(SlotId{17});
    EXPECT_EQ(got, expected);
    EXPECT_EQ(allocator.numActiveSlots(), 18);

    releaseSlots(allocator, batch);
    std::vector<Slot> singleVec;
    singleVec.push_back(std::move(single));
    releaseSlots(allocator, singleVec);
    releaseSlots(allocator, rest);
    allocator.synchronize();
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
