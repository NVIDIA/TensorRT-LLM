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
        // Forces primary context creation; the backing captures the current
        // context and refuses to be built without one.
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
        // Deliberately larger than the allocation unit, so the span a case moves
        // the target by differs from the unit the backing allocates in.
        mChunkSize = hostMemSizeGranularity() * kUnitsPerChunk;
        mTotalSize = mChunkSize * kNumChunks;
        mBacking = std::make_unique<HostMemBacking>(mTotalSize);
    }

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
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);
    mBacking->resizeTo(mChunkSize);
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
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);
    mBacking->resizeTo(mChunkSize);
    ASSERT_TRUE(backingHoldsMemory(base)) << "nothing was held to begin with";

    Poison::set("DestroyUnderPoisonKeepsTheMemory", "simulated invariant violation");
    ASSERT_TRUE(Poison::poisoned());
    mBacking->destroy();
    EXPECT_TRUE(backingHoldsMemory(base)) << "the memory was released while KVCM2 was poisoned";
    EXPECT_TRUE(takePoison().has_value());
}

TEST_F(HostMemBackingTest, CommittedRangeIsReadableAndWritable)
{
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);

    for (size_t chunk = 0; chunk < kNumChunks; ++chunk)
    {
        mBacking->resizeTo((chunk + 1) * mChunkSize);
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
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);

    mBacking->resizeTo(mChunkSize);
    stamp(base, 0, mChunkSize, 0);

    for (size_t chunk = 1; chunk < kNumChunks; ++chunk)
    {
        mBacking->resizeTo((chunk + 1) * mChunkSize);
        // Data committed earlier survives later commits, and stays where it was.
        ASSERT_TRUE(verify(base, 0, mChunkSize, 0)) << "after committing chunk " << chunk;
    }
}

//! A range opened and populated ahead of time is committed by registration
//! alone: data written between populate() and commit() survives it.
TEST_F(HostMemBackingTest, CommitOfPopulatedRangeKeepsContents)
{
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);

    mBacking->open(0, mTotalSize);
    mBacking->populate(0, mTotalSize);
    stamp(base, 0, mTotalSize, 3);
    mBacking->setTargetSize(mTotalSize);
    while (mBacking->committedSize() != mBacking->targetSize())
    {
        mBacking->resizeStep(/*populated=*/true);
    }
    EXPECT_TRUE(verify(base, 0, mTotalSize, 3));
    for (size_t offset = 0; offset < mTotalSize; offset += mBacking->registerUnit())
    {
        unsigned int flags = 0;
        ASSERT_EQ(cudaHostGetFlags(&flags, reinterpret_cast<void*>(base + offset)), cudaSuccess)
            << "unit at " << offset << " not registered";
    }
}

TEST_F(HostMemBackingTest, DecommitFreesAndRecommitWorks)
{
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);

    mBacking->resizeTo(mTotalSize);
    stamp(base, 0, mTotalSize, 0);

    // Drop the tail, then bring it back. Contents need not survive, but the
    // range must be usable again and the base must not have moved.
    size_t const tailOffset = mChunkSize * (kNumChunks - 1);
    mBacking->resizeTo(tailOffset);
    mBacking->resizeTo(mTotalSize);

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
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);

    // One target spanning many allocation units.
    mBacking->resizeTo(mTotalSize);
    stamp(base, 0, mTotalSize, 4);

    // Release a single unit from the end, well inside what one step populated.
    size_t const tail = mTotalSize - unit;
    mBacking->resizeTo(tail);
    EXPECT_TRUE(verify(base, 0, tail, 4)) << "releasing one unit disturbed the rest";

    // And take it back.
    mBacking->resizeTo(mTotalSize);
    stamp(base, tail, unit, 8);
    EXPECT_TRUE(verify(base, tail, unit, 8));
    EXPECT_TRUE(verify(base, 0, tail, 4)) << "recommit clobbered the retained range";
}

TEST_F(HostMemBackingTest, DestroyIsIdempotent)
{
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);
    mBacking->resizeTo(mChunkSize);

    mBacking->destroy();
    mBacking->destroy();
    SUCCEED();
}

//! The constraint the whole register-unit scheme exists for: the driver rejects a copy that
//! spans two registrations. Registrations are chunked well above a commit unit, so this
//! needs a register unit large enough to make the chunk small enough to straddle.
TEST_F(HostMemBackingTest, CopyAcrossARegistrationBoundaryIsRejected)
{
    // Above the chunk the backing aims for, so a registration is exactly one register unit and
    // the boundary is known here without naming the backing's own constant.
    constexpr size_t kRegisterUnit = size_t{40} << 20;
    size_t const total = 2 * kRegisterUnit;
    HostMemBacking backing(total, HostMemBackingOptions{.registerUnit = kRegisterUnit});
    MemAddress const base = backing.address();
    ASSERT_NE(base, 0U);
    backing.resizeTo(total);

    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, kRegisterUnit), cudaSuccess);

    // Wholly inside the first registration: accepted.
    EXPECT_EQ(cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base), kRegisterUnit, cudaMemcpyHostToDevice),
        cudaSuccess);
    // Straddling the boundary: rejected.
    EXPECT_NE(cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base + kRegisterUnit / 2), kRegisterUnit,
                  cudaMemcpyHostToDevice),
        cudaSuccess)
        << "a copy spanning two registrations should not be accepted";
    cudaGetLastError(); // clear the sticky error the rejected copy leaves

    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);
    backing.destroy();
}

//! A register unit that does not divide the commit unit puts registration bounds
//! somewhere other than the unit edges, so a decommit can land inside one. The
//! registration has to go -- it cannot be shortened -- and its surviving prefix
//! be registered again, or it pins pages madvise is about to discard.
TEST_F(HostMemBackingTest, DecommitSplitsTheRegistrationItLandsInside)
{
    constexpr size_t kRegisterUnit = 9216; // a whole number of neither pages nor commit units
    size_t const unit = hostMemSizeGranularity();
    ASSERT_NE(unit % kRegisterUnit, 0U)
        << "the register unit must not divide the commit unit for this to mean anything";

    size_t const total = unit * 4;
    HostMemBacking backing(total, HostMemBackingOptions{.registerUnit = kRegisterUnit});
    MemAddress const base = backing.address();
    ASSERT_NE(base, 0U);
    backing.resizeTo(total);
    stamp(base, 0, total, 7);

    backing.resizeTo(unit * 2);

    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, kRegisterUnit), cudaSuccess);
    // The last register unit that still fits below the new end: it sits in the
    // registration the decommit split, so reaching it proves the prefix came back.
    size_t const survivor = (unit * 2 / kRegisterUnit - 1) * kRegisterUnit;
    EXPECT_EQ(
        cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base + survivor), kRegisterUnit, cudaMemcpyHostToDevice),
        cudaSuccess)
        << "the prefix below the decommit should still be registered";
    cudaGetLastError();
    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);

    // The half that was dropped takes commits again, and the GPU must see what the
    // host writes there. A registration left over the discarded pages would still
    // name the physical pages madvise took, so this reads back stale bytes rather
    // than failing -- which is why the check is on the data and not on the status.
    backing.resizeTo(unit * 4);
    stamp(base, unit * 2, unit * 2, 9);

    size_t const recommitted = (unit * 2 / kRegisterUnit + 1) * kRegisterUnit;
    ASSERT_LT(recommitted + kRegisterUnit, unit * 4);
    void* checkBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&checkBuffer, kRegisterUnit), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(checkBuffer, reinterpret_cast<void const*>(base + recommitted), kRegisterUnit,
                  cudaMemcpyHostToDevice),
        cudaSuccess);
    std::vector<uint8_t> readBack(kRegisterUnit, 0);
    ASSERT_EQ(cudaMemcpy(readBack.data(), checkBuffer, kRegisterUnit, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < kRegisterUnit; ++i)
    {
        ASSERT_EQ(readBack[i], static_cast<uint8_t>((recommitted + i + 9) & 0xFF))
            << "byte " << i << " of the recommitted range";
    }
    EXPECT_EQ(cudaFree(checkBuffer), cudaSuccess);
    backing.destroy();
}

//! A step that knows the target is still ahead must stop registering on a chunk
//! boundary. Registering whatever it happened to populate would leave a remainder
//! pinned on its own, and the next step would start from there, so a long fill
//! would end up with twice the registrations it needs.
TEST_F(HostMemBackingTest, AStepShortOfTheTargetRegistersWholeChunksOnly)
{
    constexpr size_t kRegisterUnit = 9216; // divides neither a page nor a commit chunk
    // Several commit steps' worth, so most steps have a successor.
    size_t const total = hostMemSizeGranularity() * 64;
    HostMemBacking backing(total, HostMemBackingOptions{.registerUnit = kRegisterUnit});
    ASSERT_NE(backing.address(), 0U);

    backing.setTargetSize(total);
    size_t chunk = 0;
    while (backing.committedSize() != total)
    {
        size_t const registered = backing.resizeStep();
        if (registered == 0)
        {
            continue;
        }
        if (chunk == 0)
        {
            chunk = registered;
        }
        if (backing.committedSize() != total)
        {
            EXPECT_EQ(registered % chunk, 0U) << "a step left a partial registration behind";
        }
    }
    EXPECT_GT(chunk, 0U) << "nothing was registered at all";
    EXPECT_EQ(chunk % kRegisterUnit, 0U) << "a registration bound fell inside a register unit";
    // The final step registers the remainder, down to the last whole register unit.
    EXPECT_EQ(backing.registeredSize(), total / kRegisterUnit * kRegisterUnit);
}

//! A fill resumed after a target that left a partial tail starts its first
//! registration where that tail ended, which is the one start that does not sit
//! on a chunk boundary and so cannot be derived. A later shrink has to find it
//! anyway: unregistering at anything other than a registration's own base is
//! rejected by the driver, which poisons KVCM2.
TEST_F(HostMemBackingTest, ShrinkFindsARegistrationThatResumedOffAChunkBoundary)
{
    constexpr size_t kRegisterUnit = 9216;
    size_t const unit = hostMemSizeGranularity();
    size_t const total = unit * 64;
    // Not a whole number of register units, so the first fill stops short of its
    // target and the second resumes from there rather than from a boundary.
    size_t const first = unit * 3;
    ASSERT_NE(first % kRegisterUnit, 0U);

    HostMemBacking backing(total, HostMemBackingOptions{.registerUnit = kRegisterUnit});
    MemAddress const base = backing.address();
    ASSERT_NE(base, 0U);

    backing.resizeTo(first);
    size_t const resumeFrom = backing.registeredSize();
    ASSERT_LT(resumeFrom, first) << "the first target was meant to leave a partial tail";
    backing.resizeTo(total);

    // Down past the resumed registration's base and back up, twice, so a start
    // the object has mislaid shows up as a failed unregister rather than as
    // memory that merely happens to still work.
    backing.resizeTo(first);
    backing.resizeTo(total);
    backing.resizeTo(unit);
    backing.resizeTo(total);
    ASSERT_FALSE(Poison::poisoned()) << "a registration was dropped at the wrong address";
    EXPECT_EQ(backing.registeredSize(), total / kRegisterUnit * kRegisterUnit);

    // The register units either side of where the fill resumed must still hold
    // what the host writes, which a registration left over discarded pages would
    // not: it would name the physical pages madvise took.
    stamp(base, 0, total, 5);
    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, kRegisterUnit), cudaSuccess);
    std::vector<uint8_t> readBack(kRegisterUnit, 0);
    for (size_t index = resumeFrom / kRegisterUnit - 1; index <= resumeFrom / kRegisterUnit + 1; ++index)
    {
        size_t const offset = index * kRegisterUnit;
        ASSERT_EQ(cudaMemcpy(deviceBuffer, reinterpret_cast<void const*>(base + offset), kRegisterUnit,
                      cudaMemcpyHostToDevice),
            cudaSuccess)
            << "register unit at " << offset;
        ASSERT_EQ(cudaMemcpy(readBack.data(), deviceBuffer, kRegisterUnit, cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < kRegisterUnit; ++i)
        {
            ASSERT_EQ(readBack[i], static_cast<uint8_t>((offset + i + 5) & 0xFF)) << "byte " << i << " at " << offset;
        }
    }
    EXPECT_EQ(cudaFree(deviceBuffer), cudaSuccess);
    backing.destroy();
    EXPECT_FALSE(Poison::poisoned());
}

//! The point of the whole exercise: the GPU must reach the committed range.
TEST_F(HostMemBackingTest, GpuRoundTrip)
{
    MemAddress const base = mBacking->address();
    ASSERT_NE(base, 0U);
    mBacking->resizeTo(mTotalSize);
    stamp(base, 0, mTotalSize, 3);

    void* deviceBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceBuffer, mTotalSize), cudaSuccess);

    // One copy per unit, each well inside a registration. A caller whose copies are
    // larger declares them as the register unit, which keeps the bounds off them.
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
