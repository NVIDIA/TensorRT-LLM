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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/sharedPtr.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/smallObjectPool.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
namespace pool = tensorrt_llm::batch_manager::kv_cache_manager_v2::detail;

bool isAligned(void const* ptr, std::size_t align)
{
    return reinterpret_cast<std::uintptr_t>(ptr) % align == 0;
}

struct alignas(32) DigestLike
{
    std::array<std::byte, 32> bytes{};
};

// Sizes 1..kPoolMaxSize at default new alignment map to 16-byte classes; a freed block is reused by
// the next allocation of its own class only.
TEST(KvCacheManagerV2SmallObjectPoolTest, ReusesFreedBlockWithinItsSizeClass)
{
    EXPECT_FALSE(pool::poolServes(0, 8));
    EXPECT_TRUE(pool::poolServes(pool::kPoolMaxSize, 16));
    EXPECT_FALSE(pool::poolServes(pool::kPoolMaxSize + 1, 8));
    EXPECT_FALSE(pool::poolServes(64, 32));
    EXPECT_EQ(pool::poolClass(16), 0u);
    EXPECT_EQ(pool::poolClass(17), 1u);
    EXPECT_EQ(pool::poolClass(pool::kPoolMaxSize), pool::kPoolNumClasses - 1);

    void* const block = pool::poolAllocate(24);
    EXPECT_TRUE(isAligned(block, 16));
    pool::poolDeallocate(block, 24);
    void* const otherClass = pool::poolAllocate(48);
    void* const sameClass = pool::poolAllocate(32);
    EXPECT_NE(otherClass, block);
    EXPECT_EQ(sameClass, block);
    pool::poolDeallocate(sameClass, 32);
    pool::poolDeallocate(otherClass, 48);
}

// A class caches at most kPoolMaxCached blocks per thread; further frees go back to the heap.
TEST(KvCacheManagerV2SmallObjectPoolTest, CapsCachedBlocksPerClass)
{
    constexpr std::size_t kSize = 80;
    std::size_t const cls = pool::poolClass(kSize);
    auto const& cache = pool::tPoolThreadCache;

    std::vector<void*> blocks(pool::kPoolMaxCached + 8);
    for (auto& block : blocks)
    {
        block = pool::poolAllocate(kSize);
    }
    EXPECT_EQ(cache.count[cls], 0u); // the allocations drained whatever the class held
    for (void* block : blocks)
    {
        pool::poolDeallocate(block, kSize);
    }
    EXPECT_EQ(cache.count[cls], pool::kPoolMaxCached);
}

// PoolAllocator serves small arrays from the pool; a larger or over-aligned one bypasses it.
TEST(KvCacheManagerV2SmallObjectPoolTest, AllocatorBypassesLargeAndOverAlignedArrays)
{
    PoolAllocator<std::uint64_t> words;
    std::uint64_t* const small = words.allocate(4);
    words.deallocate(small, 4);
    EXPECT_EQ(pool::tPoolThreadCache.head[pool::poolClass(4 * sizeof(std::uint64_t))], static_cast<void*>(small));

    auto const before = pool::tPoolThreadCache;
    constexpr std::size_t kLargeCount = pool::kPoolMaxSize / sizeof(std::uint64_t) + 1;
    words.deallocate(words.allocate(kLargeCount), kLargeCount);
    PoolAllocator<DigestLike> digests;
    DigestLike* const aligned = digests.allocate(2);
    EXPECT_TRUE(isAligned(aligned, alignof(DigestLike)));
    digests.deallocate(aligned, 2);
    for (std::size_t c = 0; c < pool::kPoolNumClasses; ++c)
    {
        EXPECT_EQ(pool::tPoolThreadCache.count[c], before.count[c]) << "class " << c;
    }
}

// makeShared pools a small object's control block and keeps an over-aligned object (a tree block's
// digest key) aligned through the class-scope aligned operator new.
TEST(KvCacheManagerV2SmallObjectPoolTest, MakeSharedKeepsObjectAlignment)
{
    std::vector<SharedPtr<DigestLike>> digests;
    std::vector<SharedPtr<int>> ints;
    for (int i = 0; i < 64; ++i)
    {
        digests.push_back(makeShared<DigestLike>());
        ints.push_back(makeShared<int>(i));
    }
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_TRUE(isAligned(digests[i].get(), alignof(DigestLike)));
        EXPECT_EQ(*ints[i], i);
    }
}

} // namespace
