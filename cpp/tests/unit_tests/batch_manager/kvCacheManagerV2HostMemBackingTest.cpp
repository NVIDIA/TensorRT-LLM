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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

//! Both backings must satisfy the same contract, so every case here runs
//! against each of them regardless of which one this platform would select.
class HostMemBackingTest : public ::testing::TestWithParam<HostMemBackingKind>
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        // Forces primary context creation; the VMM backing queries the current
        // device in its constructor.
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
        mBacking = createHostMemBacking(GetParam(), HostMemBackingOptions{kCommitUnit});
        // Deliberately larger than the allocation unit, so the span a case
        // commits in differs from the unit the backing allocates in.
        mChunkSize = mBacking->commitGranularity() * kUnitsPerChunk;
        mTotalSize = mChunkSize * kNumChunks;
    }

    static constexpr size_t kCommitUnit = size_t{2} << 20;
    static constexpr size_t kUnitsPerChunk = 4;
    static constexpr size_t kNumChunks = 4;

    std::unique_ptr<IHostMemBacking> mBacking;
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

TEST_P(HostMemBackingTest, GranularityIsUsable)
{
    size_t const granularity = mBacking->commitGranularity();
    EXPECT_GT(granularity, 0U);
    EXPECT_EQ(granularity & (granularity - 1), 0U) << "granularity must be a power of two";
}

//! The factory must hand back the backing that was asked for; every other case
//! here is parametrized on that assumption.
TEST_P(HostMemBackingTest, FactoryHonoursRequestedKind)
{
    EXPECT_STREQ(mBacking->name(), GetParam() == HostMemBackingKind::kMmap ? "mmap" : "vmm");
}

TEST_P(HostMemBackingTest, CommittedRangeIsReadableAndWritable)
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
TEST_P(HostMemBackingTest, BaseAddressIsStableAcrossCommits)
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

TEST_P(HostMemBackingTest, DecommitFreesAndRecommitWorks)
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
TEST_P(HostMemBackingTest, LargeCommitIsReleasablePiecewise)
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

TEST_P(HostMemBackingTest, DestroyIsIdempotent)
{
    MemAddress const base = mBacking->reserve(mTotalSize);
    ASSERT_NE(base, 0U);
    mBacking->commit(0, mChunkSize);

    mBacking->destroy();
    mBacking->destroy();
    SUCCEED();
}

//! The point of the whole exercise: the GPU must reach the committed range.
TEST_P(HostMemBackingTest, GpuRoundTrip)
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

    ASSERT_EQ(
        cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base), mTotalSize, cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<uint8_t> readBack(mTotalSize, 0);
    ASSERT_EQ(cudaMemcpy(readBack.data(), deviceBuffer, mTotalSize, cudaMemcpyDeviceToHost), cudaSuccess);

    for (size_t i = 0; i < mTotalSize; ++i)
    {
        ASSERT_EQ(readBack[i], static_cast<uint8_t>((i + 3) & 0xFF)) << "byte " << i;
    }
    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(AllBackings, HostMemBackingTest,
    ::testing::Values(HostMemBackingKind::kMmap, HostMemBackingKind::kVmm),
    [](::testing::TestParamInfo<HostMemBackingKind> const& info)
    { return info.param == HostMemBackingKind::kMmap ? "Mmap" : "Vmm"; });

} // namespace
