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

} // namespace
