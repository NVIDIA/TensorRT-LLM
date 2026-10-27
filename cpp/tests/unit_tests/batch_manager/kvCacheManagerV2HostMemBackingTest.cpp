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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMemBacking.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/poison.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

class HostMemBackingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        // Forces primary context creation; reserve() captures the current
        // context and refuses to run without one.
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
        mBacking = std::make_unique<HostMemBacking>(HostMemBackingOptions{kCommitUnit});
        // Deliberately larger than the allocation unit, so the span a case
        // commits in differs from the unit the backing allocates in.
        mChunkSize = mBacking->commitGranularity() * kUnitsPerChunk;
        mTotalSize = mChunkSize * kNumChunks;
    }

    static constexpr size_t kCommitUnit = size_t{2} << 20;
    static constexpr size_t kUnitsPerChunk = 4;
    static constexpr size_t kNumChunks = 4;

    std::unique_ptr<HostMemBacking> mBacking;
    size_t mChunkSize = 0;
    size_t mTotalSize = 0;
};

//! Writes a per-offset pattern so a mis-mapped chunk shows up as wrong data
//! rather than merely as a crash.
void stamp(MemAddress base, size_t offset, size_t size, uint8_t salt)
{
    auto* bytes = reinterpret_cast<uint8_t*>(base + offset);
    for (size_t i = 0; i < size; ++i)
    {
        bytes[i] = static_cast<uint8_t>((offset + i + salt) & 0xFF);
    }
}

bool verify(MemAddress base, size_t offset, size_t size, uint8_t salt)
{
    auto const* bytes = reinterpret_cast<uint8_t const*>(base + offset);
    for (size_t i = 0; i < size; ++i)
    {
        if (bytes[i] != static_cast<uint8_t>((offset + i + salt) & 0xFF))
        {
            return false;
        }
    }
    return true;
}

TEST_F(HostMemBackingTest, GranularityIsUsable)
{
    size_t const granularity = mBacking->commitGranularity();
    EXPECT_GT(granularity, 0U);
    EXPECT_EQ(granularity & (granularity - 1), 0U) << "granularity must be a power of two";
}

//! Whether the backing still holds the memory at `addr`.
//!
//! destroy() munmaps, and munmap really removes the VMA, so /proc/self/maps is
//! the answer: an address still inside a listed range is still held.
bool backingHoldsMemory(MemAddress addr)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
    {
        auto const dash = line.find('-');
        if (dash == std::string::npos)
        {
            continue;
        }
        auto const space = line.find(' ', dash);
        auto const begin = std::stoull(line.substr(0, dash), nullptr, 16);
        auto const end = std::stoull(line.substr(dash + 1, space - dash - 1), nullptr, 16);
        if (addr >= begin && addr < end)
        {
            return true;
        }
    }
    return false;
}

//! The ordinary path must give the memory back. Also the control for the case
//! below: without it, retention there would pass whether or not anything is ever
//! released.
TEST_F(HostMemBackingTest, DestroyReleasesTheMemory)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    mBacking->commit(0, mChunkSize);
    ASSERT_TRUE(backingHoldsMemory(base)) << "nothing was held to begin with";

    mBacking->destroy();
    EXPECT_FALSE(backingHoldsMemory(base)) << "the memory outlived an ordinary destroy()";
}

//! Teardown under poison skips the waits that let this range's CUDA consumers
//! finish, so the memory has to outlive destroy() rather than be released under
//! one. Deliberately strands it, which costs a tier only in a process whose cache
//! is already permanently unusable.
TEST_F(HostMemBackingTest, DestroyUnderPoisonKeepsTheMemory)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    mBacking->commit(0, mChunkSize);
    ASSERT_TRUE(backingHoldsMemory(base)) << "nothing was held to begin with";

    Poison::set("DestroyUnderPoisonKeepsTheMemory", "simulated invariant violation");
    ASSERT_TRUE(Poison::poisoned());
    mBacking->destroy();
    EXPECT_TRUE(backingHoldsMemory(base)) << "the memory was released while KVCM2 was poisoned";
    EXPECT_TRUE(takePoison().has_value());
}

TEST_F(HostMemBackingTest, CommittedRangeIsReadableAndWritable)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);

    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        mBacking->commit(chunk * mChunkSize, mChunkSize);
        stamp(base, chunk * mChunkSize, mChunkSize, 0);
    }
    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        EXPECT_TRUE(verify(base, chunk * mChunkSize, mChunkSize, 0)) << "chunk " << chunk;
    }
}

//! The base address must not move as chunks are committed: the allocator above
//! hands out pointers derived from it before the fill has finished.
TEST_F(HostMemBackingTest, BaseAddressIsStableAcrossCommits)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);

    mBacking->commit(0, mChunkSize);
    stamp(base, 0, mChunkSize, 0);

    for (size_t chunk = 1; chunk < kNumChunks; ++chunk)
    {
        mBacking->commit(chunk * mChunkSize, mChunkSize);
        // Data committed earlier survives later commits, and stays where it was.
        ASSERT_TRUE(verify(base, 0, mChunkSize, 0)) << "after committing chunk " << chunk;
    }
}

//! A range opened and populated ahead of time is committed by registration
//! alone: data written between populate() and commit() survives it.
TEST_F(HostMemBackingTest, CommitOfPopulatedRangeKeepsContents)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);

    mBacking->open(0, mTotalSize);
    mBacking->populate(0, mTotalSize);
    stamp(base, 0, mTotalSize, 3);
    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        mBacking->commit(chunk * mChunkSize, mChunkSize, /*populated=*/true);
    }
    EXPECT_TRUE(verify(base, 0, mTotalSize, 3));
    for (size_t offset = 0; offset < mTotalSize; offset += kCommitUnit)
    {
        unsigned int flags = 0;
        ASSERT_EQ(cudaHostGetFlags(&flags, reinterpret_cast<void*>(base + offset)), cudaSuccess)
            << "unit at " << offset << " not registered";
    }
}

TEST_F(HostMemBackingTest, DecommitFreesAndRecommitWorks)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);

    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        mBacking->commit(chunk * mChunkSize, mChunkSize);
    }
    stamp(base, 0, mTotalSize, 0);

    // Drop the tail, then bring it back. Contents need not survive, but the
    // range must be usable again and the base must not have moved.
    size_t const tailOffset = mChunkSize * (kNumChunks - 1);
    mBacking->decommit(tailOffset, mChunkSize);
    mBacking->commit(tailOffset, mChunkSize);

    stamp(base, tailOffset, mChunkSize, 7);
    EXPECT_TRUE(verify(base, tailOffset, mChunkSize, 7));
    EXPECT_TRUE(verify(base, 0, tailOffset, 0)) << "decommit disturbed the retained range";
}

//! The span a caller commits in is independent of the unit the backing
//! allocates in: a large commit must still be releasable piecewise, which is
//! what lets a resize be finer than the fill's progress step.
TEST_F(HostMemBackingTest, LargeCommitIsReleasablePiecewise)
{
    size_t const unit = mBacking->commitGranularity();
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);

    // One commit spanning many allocation units.
    mBacking->commit(0, mTotalSize);
    stamp(base, 0, mTotalSize, 4);

    // Release a single unit from the end, well inside what was one commit call.
    size_t const tail = mTotalSize - unit;
    mBacking->decommit(tail, unit);
    EXPECT_TRUE(verify(base, 0, tail, 4)) << "releasing one unit disturbed the rest";

    // And take it back.
    mBacking->commit(tail, unit);
    stamp(base, tail, unit, 8);
    EXPECT_TRUE(verify(base, tail, unit, 8));
    EXPECT_TRUE(verify(base, 0, tail, 4)) << "recommit clobbered the retained range";
}

TEST_F(HostMemBackingTest, DestroyIsIdempotent)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    mBacking->commit(0, mChunkSize);

    mBacking->destroy();
    mBacking->destroy();
    SUCCEED();
}

//! The driver rejects a copy whose host range spans two registrations. Nothing
//! reports this at registration time -- it surfaces only when the copy is made --
//! so it is pinned down here: it is the reason the commit unit is a whole number
//! of slots, and a regression would otherwise show up as a failed eviction.
TEST_F(HostMemBackingTest, CopyAcrossAUnitBoundaryIsRejected)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    mBacking->commit(0, mChunkSize);
    size_t const unit = mBacking->commitGranularity();
    ASSERT_GE(mChunkSize, 2 * unit) << "a chunk must hold at least two units for this to mean anything";

    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, 2 * unit), cudaSuccess);

    // Wholly inside the first unit: accepted.
    EXPECT_EQ(cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base), unit, cudaMemcpyHostToDevice), cudaSuccess);
    // Straddling the first boundary: rejected.
    EXPECT_NE(cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base + unit / 2), unit, cudaMemcpyHostToDevice),
        cudaSuccess)
        << "a copy spanning two registrations should not be accepted";
    cudaGetLastError(); // clear the sticky error the rejected copy leaves

    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);
}

//! The point of the whole exercise: the GPU must reach the committed range.
TEST_F(HostMemBackingTest, GpuRoundTrip)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        mBacking->commit(chunk * mChunkSize, mChunkSize);
    }
    stamp(base, 0, mTotalSize, 3);

    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, mTotalSize), cudaSuccess);

    // One copy per unit. Each unit is a separate registration and the driver
    // rejects a copy that straddles two, which is why the pools size their unit
    // to a whole number of slots: every copy they make is one slot.
    size_t const unit = mBacking->commitGranularity();
    for (size_t offset = 0; offset < mTotalSize; offset += unit)
    {
        ASSERT_EQ(cudaMemcpy(static_cast<uint8_t*>(deviceBuffer) + offset, reinterpret_cast<void const*>(base + offset),
                      unit, cudaMemcpyHostToDevice),
            cudaSuccess)
            << "unit at offset " << offset;
    }

    std::vector<uint8_t> readBack(mTotalSize, 0);
    ASSERT_EQ(cudaMemcpy(readBack.data(), deviceBuffer, mTotalSize, cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < mTotalSize; ++i)
    {
        ASSERT_EQ(readBack[i], static_cast<uint8_t>((i + 3) & 0xFF)) << "byte " << i;
    }
    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);
}

} // namespace
