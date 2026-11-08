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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BuddyAllocator.h"

#include "tensorrt_llm/common/assert.h"

namespace tensorrt_llm::executor::kv_cache::bounce
{

std::size_t BuddyAllocator::roundUpPow2(std::size_t v)
{
    std::size_t p = 1;
    while (p < v)
    {
        p <<= 1;
    }
    return p;
}

BuddyAllocator::BuddyAllocator(std::size_t capacity, std::size_t minBlock)
{
    TLLM_CHECK_WITH_INFO(capacity > 0 && minBlock > 0, "BuddyAllocator: capacity/minBlock must be > 0");
    mMinBlock = roundUpPow2(minBlock);
    TLLM_CHECK_WITH_INFO(capacity >= mMinBlock, "BuddyAllocator: capacity < minBlock");

    // Largest order L with minBlock<<L <= capacity. Usable space = minBlock<<L (the rest is unused).
    std::uint32_t order = 0;
    while ((mMinBlock << (order + 1)) <= capacity)
    {
        ++order;
    }
    mMaxOrder = order;
    mUsable = mMinBlock << mMaxOrder;

    mFree.resize(mMaxOrder + 1);
    mFree[mMaxOrder].insert(0); // one free block covering the whole usable space
}

std::size_t BuddyAllocator::orderForBytes(std::size_t bytes) const
{
    std::size_t const need = roundUpPow2(bytes < mMinBlock ? mMinBlock : bytes);
    std::uint32_t order = 0;
    while ((mMinBlock << order) < need)
    {
        ++order;
    }
    return order;
}

std::optional<std::uint64_t> BuddyAllocator::alloc(std::size_t bytes)
{
    // Reject empty and anything larger than the usable arena BEFORE orderForBytes/roundUpPow2 —
    // a `bytes` near SIZE_MAX would overflow roundUpPow2's doubling (p wraps to 0 -> infinite loop).
    // `mUsable` is a power of two well under 2^63, so this bound is also the over-large guard.
    if (bytes == 0 || bytes > mUsable)
    {
        return std::nullopt;
    }
    std::size_t const want = orderForBytes(bytes);
    if (want > mMaxOrder)
    {
        return std::nullopt; // larger than the whole arena
    }
    // Find the smallest order >= want that has a free block.
    std::uint32_t cur = static_cast<std::uint32_t>(want);
    while (cur <= mMaxOrder && mFree[cur].empty())
    {
        ++cur;
    }
    if (cur > mMaxOrder)
    {
        return std::nullopt; // no block big enough is free (fragmented / full)
    }
    // Take a block at `cur` and split down to `want`, returning the lower buddy each time.
    std::uint64_t block = *mFree[cur].begin();
    mFree[cur].erase(mFree[cur].begin());
    while (cur > want)
    {
        --cur;
        std::uint64_t const buddy = block + (mMinBlock << cur);
        mFree[cur].insert(buddy); // keep the upper half free, descend into the lower half
    }
    mAllocOrder.emplace(block, static_cast<std::uint32_t>(want));
    return block;
}

void BuddyAllocator::free(std::uint64_t offset)
{
    auto it = mAllocOrder.find(offset);
    if (it == mAllocOrder.end())
    {
        return; // not a live allocation (double free / bad offset) -> ignore, stay robust
    }
    std::uint32_t order = it->second;
    mAllocOrder.erase(it);

    // Coalesce with the buddy while it is free at the same order.
    while (order < mMaxOrder)
    {
        std::uint64_t const buddy = offset ^ (mMinBlock << order);
        auto bit = mFree[order].find(buddy);
        if (bit == mFree[order].end())
        {
            break; // buddy not free -> stop merging
        }
        mFree[order].erase(bit);
        offset = offset < buddy ? offset : buddy; // merged block starts at the lower address
        ++order;
    }
    mFree[order].insert(offset);
}

std::size_t BuddyAllocator::blockBytes(std::uint64_t offset) const noexcept
{
    auto it = mAllocOrder.find(offset);
    return it == mAllocOrder.end() ? 0 : (mMinBlock << it->second);
}

std::size_t BuddyAllocator::freeBytes() const noexcept
{
    std::size_t total = 0;
    for (std::uint32_t o = 0; o <= mMaxOrder; ++o)
    {
        total += mFree[o].size() * (mMinBlock << o);
    }
    return total;
}

std::size_t BuddyAllocator::maxAllocBytes() const noexcept
{
    for (std::uint32_t o = mMaxOrder + 1; o-- > 0;)
    {
        if (!mFree[o].empty())
        {
            return mMinBlock << o;
        }
    }
    return 0;
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
