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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/common.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/lifeCycleRegistry.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/config.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storageManager.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/typedIndex.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

template <typename Container, typename Index, typename = void>
struct HasIndexOperator : std::false_type
{
};

template <typename Container, typename Index>
struct HasIndexOperator<Container, Index, std::void_t<decltype(std::declval<Container&>()[std::declval<Index>()])>>
    : std::true_type
{
};

template <typename Range, typename Index, typename = void>
struct HasContains : std::false_type
{
};

template <typename Range, typename Index>
struct HasContains<Range, Index, std::void_t<decltype(std::declval<Range const&>().contains(std::declval<Index>()))>>
    : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasEqual : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasEqual<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() == std::declval<Rhs>())>> : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasNotEqual : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasNotEqual<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() != std::declval<Rhs>())>> : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasLess : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasLess<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() < std::declval<Rhs>())>> : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasGreater : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasGreater<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() > std::declval<Rhs>())>> : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasLessEqual : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasLessEqual<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() <= std::declval<Rhs>())>> : std::true_type
{
};

template <typename Lhs, typename Rhs, typename = void>
struct HasGreaterEqual : std::false_type
{
};

template <typename Lhs, typename Rhs>
struct HasGreaterEqual<Lhs, Rhs, std::void_t<decltype(std::declval<Lhs>() >= std::declval<Rhs>())>> : std::true_type
{
};

TEST(KvCacheManagerV2TypedIndexTest, StrongIndexSupportsIntegerArithmetic)
{
    LifeCycleId lc{3};

    EXPECT_EQ((lc + 2).value(), 5);
    EXPECT_EQ((2 + lc).value(), 5);
    EXPECT_EQ((lc - 2).value(), 1);
    EXPECT_EQ((lc + std::size_t{2}).value(), 5);
    EXPECT_EQ((std::int64_t{2} + lc).value(), 5);
    EXPECT_EQ((lc - std::int64_t{2}).value(), 1);

    LifeCycleId begin{4};
    LifeCycleId end{10};
    static_assert(std::is_same<decltype(end - begin), int>::value, "index difference must be an integer");
    EXPECT_EQ(end - begin, 6);

    lc += 4;
    EXPECT_EQ(lc.value(), 7);

    lc -= 2;
    EXPECT_EQ(lc.value(), 5);

    EXPECT_EQ((++lc).value(), 6);
    EXPECT_EQ((lc++).value(), 6);
    EXPECT_EQ(lc.value(), 7);
    EXPECT_EQ((--lc).value(), 6);
    EXPECT_EQ((lc--).value(), 6);
    EXPECT_EQ(lc.value(), 5);
}

TEST(KvCacheManagerV2TypedIndexTest, StrongIndexSupportsValueTypeUpperBoundComparison)
{
    static_assert(HasEqual<LifeCycleId, LifeCycleId>::value, "matching strong index equality must work");
    static_assert(HasNotEqual<LifeCycleId, LifeCycleId>::value, "matching strong index inequality must work");
    static_assert(HasLess<LifeCycleId, LifeCycleId>::value, "matching strong index ordering must work");
    static_assert(HasGreater<LifeCycleId, LifeCycleId>::value, "matching strong index ordering must work");
    static_assert(HasLessEqual<LifeCycleId, LifeCycleId>::value, "matching strong index ordering must work");
    static_assert(HasGreaterEqual<LifeCycleId, LifeCycleId>::value, "matching strong index ordering must work");

    static_assert(HasLess<LifeCycleId, LifeCycleId::ValueType>::value,
        "strong index must support upper-bound comparison against its value type");
    static_assert(HasLess<SlotId, SlotCount>::value, "slot id must support upper-bound comparison against slot counts");
    static_assert(HasGreaterEqual<LifeCycleId, LifeCycleId::ValueType>::value,
        "strong index must support upper-bound rejection against its value type");
    static_assert(
        HasGreaterEqual<SlotId, SlotCount>::value, "slot id must support upper-bound rejection against slot counts");

    static_assert(
        !HasEqual<LifeCycleId, LifeCycleId::ValueType>::value, "strong index must not compare equal to raw value type");
    static_assert(
        !HasEqual<LifeCycleId::ValueType, LifeCycleId>::value, "raw value type must not compare equal to strong index");
    static_assert(!HasNotEqual<LifeCycleId, LifeCycleId::ValueType>::value,
        "strong index must not compare unequal to raw value type");
    static_assert(!HasNotEqual<LifeCycleId::ValueType, LifeCycleId>::value,
        "raw value type must not compare unequal to strong index");
    static_assert(
        !HasLess<LifeCycleId::ValueType, LifeCycleId>::value, "raw value type must not order against strong index");
    static_assert(!HasGreater<LifeCycleId, LifeCycleId::ValueType>::value,
        "strong index must not use raw value type for greater-than comparison");
    static_assert(
        !HasGreater<LifeCycleId::ValueType, LifeCycleId>::value, "raw value type must not order against strong index");
    static_assert(!HasLessEqual<LifeCycleId, LifeCycleId::ValueType>::value,
        "strong index must not use raw value type for less-equal comparison");
    static_assert(!HasLessEqual<LifeCycleId::ValueType, LifeCycleId>::value,
        "raw value type must not order against strong index");
    static_assert(!HasGreaterEqual<LifeCycleId::ValueType, LifeCycleId>::value,
        "raw value type must not order against strong index");
}

TEST(KvCacheManagerV2TypedIndexTest, StrongIndexDefaultsMatchSentinels)
{
    static_assert(CacheLevel{}.value() == kGpuLevel.value(), "CacheLevel default should name the GPU level");
    static_assert(BeamIndex{}.value() == kDefaultBeamIndex.value(), "BeamIndex default should name the default beam");
    static_assert(BlockOrdinal{}.value() == kBadBlockOrdinal.value(), "BlockOrdinal default should be invalid");
    static_assert(PageIndex{}.value() == kBadPageIndex.value(), "PageIndex default should be invalid");
}

TEST(KvCacheManagerV2TypedIndexTest, SlotCapacityAccessorsUsePlainSlotCount)
{
    static_assert(std::is_same<decltype(std::declval<SlotAllocator const&>().numSlots()), SlotCount>::value,
        "SlotAllocator::numSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<PoolGroupBase const&>().numSlots()), SlotCount>::value,
        "PoolGroupBase::numSlots must return a plain slot count");
    static_assert(
        std::is_same<decltype(std::declval<CacheLevelStorage const&>().numSlots(PoolGroupIndex{0})), SlotCount>::value,
        "CacheLevelStorage::numSlots must return a plain slot count");
    static_assert(
        std::is_same<decltype(std::declval<StorageManager const&>().numSlots(PoolGroupIndex{0})), SlotCount>::value,
        "StorageManager::numSlots must return a plain slot count");
}

TEST(KvCacheManagerV2TypedIndexTest, SlotIdUsesInt64)
{
    static_assert(std::is_same<SlotId::ValueType, std::int64_t>::value, "SlotId must use int64_t");
    static_assert(std::is_same<SlotCount, std::int64_t>::value, "SlotCount must use int64_t");
}

TEST(KvCacheManagerV2TypedIndexTest, SlotCountsUsePlainSlotCount)
{
    static_assert(std::is_same<decltype(std::declval<SlotAllocator const&>().numFreeSlots()), SlotCount>::value,
        "SlotAllocator::numFreeSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<SlotAllocator const&>().numOccupiedSlots()), SlotCount>::value,
        "SlotAllocator::numOccupiedSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<SlotAllocator const&>().numOverflowSlots()), SlotCount>::value,
        "SlotAllocator::numOverflowSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<PoolGroupBase const&>().numFreeSlots()), SlotCount>::value,
        "PoolGroupBase::numFreeSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<CacheLevelStorage const&>().numFreeSlots(PoolGroupIndex{0})),
                      SlotCount>::value,
        "CacheLevelStorage::numFreeSlots must return a plain slot count");
    static_assert(std::is_same<decltype(std::declval<PrioritizedEvictionPolicy const&>().size()), SlotCount>::value,
        "PrioritizedEvictionPolicy::size must return a plain count");
    static_assert(std::is_same<decltype(std::declval<StorageStatistics const&>().total), SlotCount>::value,
        "StorageStatistics::total must be a plain count");
    static_assert(std::is_same<decltype(std::declval<StorageStatistics const&>().available()), SlotCount>::value,
        "StorageStatistics::available must return a plain count");
    static_assert(std::is_same<decltype(std::declval<StorageStatistics const&>().unavailable()), SlotCount>::value,
        "StorageStatistics::unavailable must return a plain count");
}

TEST(KvCacheManagerV2TypedIndexTest, AllocationRequestsUsePlainSlotCount)
{
    using SlotAllocatorAllocateMultiple = std::vector<Slot> (SlotAllocator::*)(SlotCount);
    static_assert(std::is_same<decltype(&SlotAllocator::allocateMultiple), SlotAllocatorAllocateMultiple>::value,
        "SlotAllocator::allocateMultiple must take a plain slot count");
    static_assert(!std::is_invocable<decltype(&SlotAllocator::allocateMultiple), SlotAllocator&, SlotId>::value,
        "SlotAllocator::allocateMultiple must not take a slot id");

    using StorageManagerNewGpuSlots = TypedVec<LifeCycleId, std::vector<Slot>> (StorageManager::*)(
        TypedVec<LifeCycleId, SlotCount> const&, MigrationRecorder const&, DropRecorder const&);
    static_assert(std::is_same<decltype(&StorageManager::newGpuSlots), StorageManagerNewGpuSlots>::value,
        "StorageManager::newGpuSlots must take plain slot counts");

    using StorageManagerNewSlotsForPoolGroup = std::vector<Slot> (StorageManager::*)(
        CacheLevel, PoolGroupIndex, SlotCount, MigrationRecorder const&, DropRecorder const&);
    static_assert(
        std::is_same<decltype(&StorageManager::newSlotsForPoolGroup), StorageManagerNewSlotsForPoolGroup>::value,
        "StorageManager::newSlotsForPoolGroup must take a plain slot count");
    static_assert(
        std::is_same<decltype(std::declval<PerLevelEvictionController const&>().numEvictablePages(PoolGroupIndex{0})),
            SlotCount>::value,
        "PerLevelEvictionController::numEvictablePages must return a plain count");
}

TEST(KvCacheManagerV2TypedIndexTest, SlotAllocatorRejectsOutOfRangeSlotId)
{
    SlotAllocator allocator(1);
    Slot invalidSlot;
    invalidSlot.setSlotId(SlotId{1});

    EXPECT_THROW(allocator.release(std::move(invalidSlot)), LogicError);
}

TEST(KvCacheManagerV2TypedIndexTest, HalfOpenRangeCarriesIndexType)
{
    using BlockRange = HalfOpenRange<BlockOrdinal>;

    static_assert(std::is_same<typename BlockRange::IndexType, BlockOrdinal>::value,
        "range endpoints must carry the requested index type");
    static_assert(std::is_same<decltype(std::declval<BlockRange>().length()), int>::value,
        "range length must remain an integer count");
    static_assert(HasContains<BlockRange, BlockOrdinal>::value, "matching index type must work");
    static_assert(!HasContains<BlockRange, LifeCycleId>::value, "different strong index must not work");
    static_assert(!HasContains<BlockRange, int>::value, "raw int must not be accepted by typed contains");

    BlockRange empty;
    EXPECT_EQ(empty.beg, BlockOrdinal{0});
    EXPECT_EQ(empty.end, BlockOrdinal{0});
    EXPECT_EQ(empty.length(), 0);
    EXPECT_FALSE(empty);

    BlockRange range{2, 5};
    EXPECT_EQ(range.beg, BlockOrdinal{2});
    EXPECT_EQ(range.end, BlockOrdinal{5});
    EXPECT_EQ(range.length(), 3);
    EXPECT_TRUE(range.contains(BlockOrdinal{2}));
    EXPECT_FALSE(range.contains(BlockOrdinal{5}));

    BlockRange overlap = intersect(range, BlockRange{4, 8});
    EXPECT_EQ(overlap.beg, BlockOrdinal{4});
    EXPECT_EQ(overlap.end, BlockOrdinal{5});
}

TEST(KvCacheManagerV2TypedIndexTest, TypedVecRequiresMatchingIndexType)
{
    using LifeCycleVector = TypedVec<LifeCycleId, int>;

    static_assert(HasIndexOperator<LifeCycleVector, LifeCycleId>::value, "matching index type must work");
    static_assert(!HasIndexOperator<LifeCycleVector, PoolGroupIndex>::value, "different strong index must not work");
    static_assert(!HasIndexOperator<LifeCycleVector, int>::value, "raw int must not index typed vector");

    LifeCycleVector values(LifeCycleId{2}, 0);
    values[LifeCycleId{0}] = 11;
    values[LifeCycleId{1}] = 17;

    EXPECT_EQ(values.size(), LifeCycleId{2});
    EXPECT_EQ(values[LifeCycleId{0}], 11);
    EXPECT_EQ(values.at(LifeCycleId{1}), 17);
}

TEST(KvCacheManagerV2TypedIndexTest, TypedVecSupportsVectorInterop)
{
    TypedVec<PoolGroupIndex, int> values;
    values.reserve(PoolGroupIndex{2});
    values.push_back(3);
    values.push_back(5);

    EXPECT_EQ(values.stdSize(), 2);
    EXPECT_EQ(values.size(), PoolGroupIndex{2});
    EXPECT_EQ(values.raw().at(0), 3);
    EXPECT_EQ(values[PoolGroupIndex{1}], 5);
}

} // namespace
