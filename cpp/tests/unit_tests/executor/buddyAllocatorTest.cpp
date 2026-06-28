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

// Pure-logic tests for the bounce v2 buddy allocator: right-sizing, no overlap, coalescing on free,
// exhaustion/backpressure, and the "many small + one larger-than-buffer (recycled)" regimes.

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BuddyAllocator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;

namespace
{
// Assert no two live [offset, offset+size) ranges overlap.
void checkNoOverlap(std::vector<std::pair<std::uint64_t, std::size_t>> const& live)
{
    std::vector<std::pair<std::uint64_t, std::size_t>> s(live);
    std::sort(s.begin(), s.end());
    for (std::size_t i = 1; i < s.size(); ++i)
    {
        EXPECT_LE(s[i - 1].first + s[i - 1].second, s[i].first)
            << "overlap: [" << s[i - 1].first << ",+" << s[i - 1].second << ") vs [" << s[i].first << ",...)";
    }
}
} // namespace

TEST(BuddyAllocator, RoundsUpToPow2MultipleOfMinBlock)
{
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/64);
    EXPECT_EQ(a.capacity(), 1024u);
    EXPECT_EQ(a.minBlock(), 64u);
    EXPECT_EQ(a.maxAllocBytes(), 1024u); // whole arena free
    // 1 byte -> a minBlock (64). 65 bytes -> 128. 64 -> 64.
    auto o1 = a.alloc(1);
    auto o2 = a.alloc(65);
    auto o3 = a.alloc(64);
    ASSERT_TRUE(o1 && o2 && o3);
    EXPECT_EQ(*o1 % 64, 0u);
    // freeBytes accounts for the rounded-up block sizes (64 + 128 + 64 used).
    EXPECT_EQ(a.freeBytes(), 1024u - (64u + 128u + 64u));
    EXPECT_EQ(a.liveBlocks(), 3u);
}

TEST(BuddyAllocator, ManySmallRegionsDistinctNoOverlap)
{
    // capacity 16 * minBlock -> up to 16 concurrent minBlock allocations (high concurrency of small
    // requests, each right-sized to one block).
    b::BuddyAllocator a(/*capacity=*/16 * 256, /*minBlock=*/256);
    std::vector<std::pair<std::uint64_t, std::size_t>> live;
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 16; ++i)
    {
        auto o = a.alloc(200); // < 256 -> one minBlock
        ASSERT_TRUE(o.has_value()) << "small alloc " << i << " failed";
        EXPECT_EQ(seen.count(*o), 0u) << "offset reused while live";
        seen.insert(*o);
        live.emplace_back(*o, 256);
    }
    EXPECT_EQ(a.liveBlocks(), 16u);
    EXPECT_EQ(a.freeBytes(), 0u);
    EXPECT_FALSE(a.alloc(1).has_value()); // full -> backpressure (nullopt, no overlap/overcommit)
    checkNoOverlap(live);

    for (auto& [off, sz] : live)
    {
        a.free(off);
    }
    EXPECT_EQ(a.freeBytes(), a.capacity()); // all coalesced back
    EXPECT_EQ(a.maxAllocBytes(), a.capacity());
    EXPECT_EQ(a.liveBlocks(), 0u);
}

TEST(BuddyAllocator, FreeCoalescesBuddies)
{
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/256); // 4 minBlocks
    auto o0 = a.alloc(256);
    auto o1 = a.alloc(256);
    auto o2 = a.alloc(256);
    auto o3 = a.alloc(256);
    ASSERT_TRUE(o0 && o1 && o2 && o3);
    EXPECT_EQ(a.maxAllocBytes(), 0u); // fully split into minBlocks
    // Free all -> must coalesce all the way back to one 1024 block (maxAlloc == capacity).
    a.free(*o0);
    a.free(*o1);
    a.free(*o2);
    a.free(*o3);
    EXPECT_EQ(a.maxAllocBytes(), 1024u) << "buddies did not coalesce to the top order";
    // And a full-size alloc now succeeds.
    auto big = a.alloc(1024);
    ASSERT_TRUE(big.has_value());
    EXPECT_EQ(*big, 0u);
}

TEST(BuddyAllocator, TooLargeRejected)
{
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/256);
    EXPECT_FALSE(a.alloc(2048).has_value()); // larger than the whole arena
    EXPECT_TRUE(a.alloc(1024).has_value());
}

TEST(BuddyAllocator, ZeroAndOverflowSizeRejectedNoHang)
{
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/256);
    EXPECT_FALSE(a.alloc(0).has_value());
    // A near-SIZE_MAX request must be rejected WITHOUT hanging — roundUpPow2's doubling would
    // otherwise overflow to 0 and spin forever. The early `bytes > usable` bound prevents it.
    EXPECT_FALSE(a.alloc(std::numeric_limits<std::size_t>::max()).has_value());
    EXPECT_FALSE(a.alloc(std::numeric_limits<std::size_t>::max() - 1).has_value());
    EXPECT_EQ(a.freeBytes(), a.capacity()); // nothing consumed by the rejected requests
    EXPECT_TRUE(a.alloc(256).has_value());  // still usable afterwards
}

TEST(BuddyAllocator, LargeRecycledStreamLargerThanArena)
{
    // Models a single transfer whose TOTAL bytes >> arena: chunks of maxChunkBytes streamed through
    // a small arena with recycling (alloc chunk -> "send" -> free -> alloc next). The arena only
    // ever holds a few chunks; an unbounded total streams through. (R1 with variable regions.)
    std::size_t const minBlock = 1024;
    std::size_t const chunk = 4 * 1024;                    // maxChunkBytes
    b::BuddyAllocator a(/*capacity=*/4 * chunk, minBlock); // arena holds only 4 chunks at once
    std::size_t streamed = 0;
    std::vector<std::uint64_t> inflight;
    for (int i = 0; i < 1000; ++i) // 1000 chunks * 4KiB = 4MiB through a 16KiB arena
    {
        // keep a window of up to 4 chunks in flight
        if (inflight.size() == 4)
        {
            a.free(inflight.front());
            inflight.erase(inflight.begin());
        }
        auto o = a.alloc(chunk);
        ASSERT_TRUE(o.has_value()) << "chunk " << i << " did not fit despite recycling";
        inflight.push_back(*o);
        streamed += chunk;
    }
    EXPECT_GT(streamed, a.capacity() * 10u); // streamed far more than the arena holds
    for (auto o : inflight)
    {
        a.free(o);
    }
    EXPECT_EQ(a.freeBytes(), a.capacity());
    EXPECT_EQ(a.liveBlocks(), 0u);
}

TEST(BuddyAllocator, MixedSmallAndLargeShareArena)
{
    // Small regions coexist with near-max chunks; after smalls free + coalesce a large alloc fits.
    b::BuddyAllocator a(/*capacity=*/8 * 1024, /*minBlock=*/1024); // 8 blocks
    auto big = a.alloc(4 * 1024);                                  // 4 blocks
    ASSERT_TRUE(big.has_value());
    std::vector<std::uint64_t> smalls;
    for (int i = 0; i < 4; ++i)
    {
        auto o = a.alloc(1024);
        ASSERT_TRUE(o.has_value());
        smalls.push_back(*o);
    }
    EXPECT_EQ(a.freeBytes(), 0u);
    EXPECT_FALSE(a.alloc(1024).has_value()); // full
    // Free the smalls -> they coalesce -> a second 4*1024 chunk now fits.
    for (auto o : smalls)
    {
        a.free(o);
    }
    auto big2 = a.alloc(4 * 1024);
    ASSERT_TRUE(big2.has_value());
    a.free(*big);
    a.free(*big2);
    EXPECT_EQ(a.maxAllocBytes(), a.capacity());
}

TEST(BuddyAllocator, DoubleFreeIgnored)
{
    b::BuddyAllocator a(1024, 256);
    auto o = a.alloc(256);
    ASSERT_TRUE(o.has_value());
    a.free(*o);
    a.free(*o); // ignored, no corruption
    EXPECT_EQ(a.freeBytes(), a.capacity());
    EXPECT_EQ(a.liveBlocks(), 0u);
}

// ---- boundary cases ----

TEST(BuddyAllocator, CapacityRoundedDownToMinBlockMultiple)
{
    // capacity is NOT a power-of-two multiple of minBlock -> usable rounds DOWN to 512 (256<<1),
    // the trailing 1000-512 bytes are unusable. capacity() reports the usable size, not the arg.
    b::BuddyAllocator a(/*capacity=*/1000, /*minBlock=*/256);
    EXPECT_EQ(a.capacity(), 512u);
    EXPECT_EQ(a.minBlock(), 256u);
    EXPECT_EQ(a.maxAllocBytes(), 512u);
    auto whole = a.alloc(512);
    ASSERT_TRUE(whole.has_value());
    EXPECT_EQ(*whole, 0u);
    EXPECT_FALSE(a.alloc(1).has_value()); // nothing left (the rounded-off tail is not allocatable)
}

TEST(BuddyAllocator, MinBlockRoundedUpToPow2)
{
    // minBlock 100 -> rounded up to 128; capacity 512 -> orders 0..2 of 128/256/512.
    b::BuddyAllocator a(/*capacity=*/512, /*minBlock=*/100);
    EXPECT_EQ(a.minBlock(), 128u);
    EXPECT_EQ(a.capacity(), 512u);
    auto o1 = a.alloc(1);   // -> 128 (one minBlock)
    auto o2 = a.alloc(130); // -> 256 (next power of two)
    ASSERT_TRUE(o1 && o2);
    EXPECT_EQ(*o1 % 128, 0u);
    EXPECT_EQ(*o2 % 256, 0u);                       // 256-block is 256-aligned
    EXPECT_EQ(a.freeBytes(), 512u - (128u + 256u)); // 128 used + 256 used
}

TEST(BuddyAllocator, SingleBlockArena)
{
    // capacity == minBlock -> exactly one order-0 block (maxOrder 0).
    b::BuddyAllocator a(/*capacity=*/256, /*minBlock=*/256);
    EXPECT_EQ(a.capacity(), 256u);
    EXPECT_EQ(a.maxAllocBytes(), 256u);
    auto o = a.alloc(10);
    ASSERT_TRUE(o.has_value());
    EXPECT_EQ(*o, 0u);
    EXPECT_FALSE(a.alloc(1).has_value()); // only one block -> full
    EXPECT_EQ(a.maxAllocBytes(), 0u);
    a.free(*o);
    EXPECT_EQ(a.maxAllocBytes(), 256u);
    EXPECT_TRUE(a.alloc(256).has_value()); // reusable
}

TEST(BuddyAllocator, FreeUnknownOffsetIgnored)
{
    // Freeing an offset that was never allocated (out of range OR a valid-but-unallocated offset)
    // must be a no-op, not corrupt the free lists or fabricate free space.
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/256);
    auto o = a.alloc(256);
    ASSERT_TRUE(o.has_value());
    auto const freeBefore = a.freeBytes();
    a.free(999999);                // wildly out of range
    a.free(*o + 64);               // in range but not a block start / not allocated
    EXPECT_EQ(a.freeBytes(), freeBefore);
    EXPECT_EQ(a.liveBlocks(), 1u); // the one real allocation is untouched
    a.free(*o);                    // the genuine free still works exactly once
    EXPECT_EQ(a.freeBytes(), a.capacity());
    EXPECT_EQ(a.liveBlocks(), 0u);
}

TEST(BuddyAllocator, BlockBytesReportsRoundedUpSize)
{
    // alloc returns only the offset; blockBytes() exposes the actual (rounded-up) block size for
    // metrics. 0 for any non-live offset.
    b::BuddyAllocator a(/*capacity=*/1024, /*minBlock=*/64);
    auto o = a.alloc(65); // 65 -> rounded up to 128
    ASSERT_TRUE(o.has_value());
    EXPECT_EQ(a.blockBytes(*o), 128u);
    EXPECT_EQ(a.blockBytes(*o + 999), 0u); // not a live block start
    a.free(*o);
    EXPECT_EQ(a.blockBytes(*o), 0u);       // freed -> no longer live
}

TEST(BuddyAllocator, FragmentationBlocksLargeAllocDespiteFreeBytes)
{
    // The defining buddy property: freeing every OTHER minBlock leaves half the arena free in bytes,
    // but as scattered order-0 blocks with no free buddy pair -> a 2-block alloc must FAIL (no
    // premature coalescing), and freeing the rest must then coalesce so it succeeds.
    constexpr std::size_t kMin = 256;
    b::BuddyAllocator a(/*capacity=*/8 * kMin, kMin); // 8 order-0 blocks
    std::vector<std::uint64_t> all;
    for (int i = 0; i < 8; ++i)
    {
        auto o = a.alloc(kMin);
        ASSERT_TRUE(o.has_value());
        all.push_back(*o);
    }
    EXPECT_EQ(a.freeBytes(), 0u);
    // Free exactly the lower buddy of each pair (offset % 512 == 0) so no two freed blocks are
    // buddies -> they cannot coalesce; their live buddies block every order-1 merge.
    std::vector<std::uint64_t> held;
    for (auto o : all)
    {
        if (o % (2 * kMin) == 0)
            a.free(o);
        else
            held.push_back(o);
    }
    EXPECT_EQ(a.freeBytes(), 4 * kMin);          // half the arena is free...
    EXPECT_EQ(a.maxAllocBytes(), kMin);          // ...but only as isolated order-0 blocks
    EXPECT_FALSE(a.alloc(2 * kMin).has_value()); // a 2-block request can't be satisfied (fragmented)
    auto extra = a.alloc(kMin);                  // a 1-block request still works
    ASSERT_TRUE(extra.has_value());
    // Return everything (the extra 1-block + the held buddies) -> all 8 blocks coalesce back so a
    // full-capacity alloc fits again.
    a.free(*extra);
    for (auto o : held)
    {
        a.free(o);
    }
    EXPECT_EQ(a.liveBlocks(), 0u);
    EXPECT_EQ(a.maxAllocBytes(), a.capacity());
}
