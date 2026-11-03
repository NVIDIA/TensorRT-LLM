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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/cudaEvent.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <utility>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

// Holds a stream on the device until open() is called, so an event recorded behind it stays
// not ready for as long as the test needs.
class StreamGate
{
public:
    StreamGate()
    {
        EXPECT_EQ(cuLaunchHostFunc(mStream.handle(), &StreamGate::wait, &mOpen), CUDA_SUCCESS);
    }

    ~StreamGate()
    {
        open();
        mStream.synchronize();
    }

    StreamGate(StreamGate const&) = delete;
    StreamGate& operator=(StreamGate const&) = delete;

    [[nodiscard]] CachedCudaEvent recordBehind()
    {
        return mStream.recordEvent();
    }

    void open() noexcept
    {
        mOpen.store(true, std::memory_order_release);
    }

    void drain()
    {
        open();
        mStream.synchronize();
    }

private:
    static void CUDA_CB wait(void* flag)
    {
        auto const* open = static_cast<std::atomic<bool> const*>(flag);
        while (!open->load(std::memory_order_acquire))
        {
        }
    }

    std::atomic<bool> mOpen{false};
    CachedCudaStream mStream;
};

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

void releaseWithEvent(SlotAllocator& allocator, std::vector<Slot>& slots, std::size_t first, std::size_t last,
    CachedCudaEvent const& event)
{
    for (std::size_t i = first; i < last; ++i)
    {
        Slot slot;
        slot.setSlot(slots[i]);
        slot.readyEvent = event;
        allocator.release(std::move(slot));
    }
}

void releaseAll(SlotAllocator& allocator, std::vector<Slot>& slots)
{
    for (auto& s : slots)
    {
        if (s.hasValidSlot())
        {
            Slot slot;
            slot.setSlot(s);
            allocator.release(std::move(slot));
        }
    }
}

class KvCacheManagerV2SlotAllocatorEventTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        {
            GTEST_SKIP() << "needs a CUDA device";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    }
};

// Slots released while their pages are still being read must not be handed out ahead of
// never-issued ids, whether the allocation is one slot or a batch; once their work completes
// they are reused before any new id, oldest first.
TEST_F(KvCacheManagerV2SlotAllocatorEventTest, InFlightSlotsWaitBehindFreshIdsThenComeBackInOrder)
{
    SlotAllocator allocator{SlotCount{16}};
    auto first = allocator.allocateMultiple(SlotCount{8});
    ASSERT_EQ(slotIds(first),
        (std::vector<SlotId>{SlotId{0}, SlotId{1}, SlotId{2}, SlotId{3}, SlotId{4}, SlotId{5}, SlotId{6}, SlotId{7}}));

    StreamGate gate;
    CachedCudaEvent const inFlight = gate.recordBehind();
    ASSERT_FALSE(inFlight.queryComplete());
    // Release-heavy: many releases against one in-flight event.
    releaseWithEvent(allocator, first, 0, 4, inFlight);

    auto fresh = allocator.allocateMultiple(SlotCount{4});
    EXPECT_EQ(slotIds(fresh), (std::vector<SlotId>{SlotId{8}, SlotId{9}, SlotId{10}, SlotId{11}}));
    Slot single = allocator.allocate();
    EXPECT_EQ(single.slotId(), SlotId{12});
    EXPECT_FALSE(inFlight.isClosed());

    gate.drain();

    // Allocate-heavy after completion: the recycled slots come back first, in release order,
    // with their events retired.
    auto recycled = allocator.allocateMultiple(SlotCount{3});
    EXPECT_EQ(slotIds(recycled), (std::vector<SlotId>{SlotId{0}, SlotId{1}, SlotId{2}}));
    for (auto const& slot : recycled)
    {
        EXPECT_TRUE(slot.readyEvent.isClosed());
    }
    Slot last = allocator.allocate();
    EXPECT_EQ(last.slotId(), SlotId{3});
    EXPECT_TRUE(last.readyEvent.isClosed());
    Slot next = allocator.allocate();
    EXPECT_EQ(next.slotId(), SlotId{13});

    std::vector<Slot> all;
    for (auto* group : {&first, &fresh, &recycled})
    {
        for (auto& s : *group)
        {
            all.push_back(std::move(s));
        }
    }
    all.push_back(std::move(single));
    all.push_back(std::move(last));
    all.push_back(std::move(next));
    releaseAll(allocator, all);
    allocator.synchronize();
}

// With every id issued, an in-flight recycled slot is the only one left: it is handed out with its
// event still open, so the caller orders its writes behind that work.
TEST_F(KvCacheManagerV2SlotAllocatorEventTest, ExhaustedPoolHandsOutInFlightSlotWithItsEvent)
{
    SlotAllocator allocator{SlotCount{4}};
    auto slots = allocator.allocateMultiple(SlotCount{4});

    StreamGate gate;
    CachedCudaEvent const inFlight = gate.recordBehind();
    releaseWithEvent(allocator, slots, 1, 3, inFlight);

    auto reused = allocator.allocateMultiple(SlotCount{2});
    EXPECT_EQ(slotIds(reused), (std::vector<SlotId>{SlotId{1}, SlotId{2}}));
    for (auto const& slot : reused)
    {
        EXPECT_FALSE(slot.readyEvent.isClosed());
        EXPECT_EQ(slot.readyEvent.handle(), inFlight.handle());
    }

    gate.drain();
    for (auto& slot : reused)
    {
        slot.readyEvent.synchronize();
    }
    std::vector<Slot> all;
    for (auto* group : {&slots, &reused})
    {
        for (auto& s : *group)
        {
            all.push_back(std::move(s));
        }
    }
    releaseAll(allocator, all);
    allocator.synchronize();
}

} // namespace
