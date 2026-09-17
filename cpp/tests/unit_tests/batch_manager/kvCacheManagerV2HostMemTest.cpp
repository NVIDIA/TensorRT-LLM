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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/exceptions.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

class ScopedEnv
{
public:
    ScopedEnv(char const* name, std::optional<std::string> value)
        : mName(name)
    {
        if (char const* oldValue = std::getenv(name); oldValue != nullptr)
        {
            mOldValue = oldValue;
        }
        if (value.has_value())
        {
            if (::setenv(name, value->c_str(), /*overwrite=*/1) != 0)
            {
                throw std::system_error(errno, std::generic_category(), "setenv failed");
            }
        }
        else if (::unsetenv(name) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "unsetenv failed");
        }
    }

    ~ScopedEnv()
    {
        if (mOldValue.has_value())
        {
            ::setenv(mName.c_str(), mOldValue->c_str(), /*overwrite=*/1);
        }
        else
        {
            ::unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::optional<std::string> mOldValue;
};

int gMadviseErrno = 0;
int gCapturedAdvice = 0;
int gMemsetCalls = 0;

int captureMadvise(void*, size_t, int advice)
{
    gCapturedAdvice = advice;
    return 0;
}

int failMadvise(void*, size_t, int)
{
    errno = gMadviseErrno;
    return -1;
}

void* countMemset(void* ptr, int value, size_t size)
{
    ++gMemsetCalls;
    return std::memset(ptr, value, size);
}

TEST(KvCacheManagerV2HostMemTest, SelectsConfiguredPageMode)
{
    hostMadvisePageMode(MemAddress{1}, HostMem::kAlignment, true, captureMadvise);
    EXPECT_EQ(gCapturedAdvice, MADV_HUGEPAGE);
    hostMadvisePageMode(MemAddress{1}, HostMem::kAlignment, false, captureMadvise);
    EXPECT_EQ(gCapturedAdvice, MADV_NOHUGEPAGE);

    ScopedEnv defaultThp("TLLM_KV_CACHE_MANAGER_V2_THP", std::nullopt);
    EXPECT_TRUE(hostUseThp());
    {
        ScopedEnv disableThp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");
        EXPECT_FALSE(hostUseThp());
    }
}

class PrefaultFallbackTest : public testing::TestWithParam<int>
{
};

TEST_P(PrefaultFallbackTest, TouchesMemory)
{
    std::vector<unsigned char> data(HostMem::kAlignment, 0xFF);
    gMadviseErrno = GetParam();
    gMemsetCalls = 0;
    hostPrefaultChunk(reinterpret_cast<MemAddress>(data.data()), data.size(), failMadvise, countMemset);
    EXPECT_EQ(gMemsetCalls, 1);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](unsigned char value) { return value == 0; }));
}

INSTANTIATE_TEST_SUITE_P(UnsupportedPopulateWrite, PrefaultFallbackTest, testing::Values(EINVAL, ENOSYS));

TEST(KvCacheManagerV2HostMemTest, ConvertsPrefaultEnomem)
{
    std::vector<unsigned char> data(HostMem::kAlignment);
    gMadviseErrno = ENOMEM;
    EXPECT_THROW(hostPrefaultChunk(reinterpret_cast<MemAddress>(data.data()), data.size(), failMadvise, countMemset),
        HostOOMError);
}

TEST(KvCacheManagerV2HostMemTest, PropagatesOtherPrefaultErrors)
{
    std::vector<unsigned char> data(HostMem::kAlignment);
    gMadviseErrno = EIO;
    try
    {
        hostPrefaultChunk(reinterpret_cast<MemAddress>(data.data()), data.size(), failMadvise, countMemset);
        FAIL() << "Expected std::system_error";
    }
    catch (std::system_error const& error)
    {
        EXPECT_EQ(error.code().value(), EIO);
    }
}

class HostMemPageModeTest : public testing::TestWithParam<char const*>
{
};

TEST_P(HostMemPageModeTest, AllocatesAndResizes)
{
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", std::string(GetParam()));
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    HostMem memory(4 * HostMem::kAlignment, HostMem::kAlignment);
    EXPECT_NE(memory.address(), 0);
    EXPECT_EQ(memory.size(), HostMem::kAlignment);
    memory.resize(2 * HostMem::kAlignment);
    EXPECT_EQ(memory.size(), 2 * HostMem::kAlignment);
}

INSTANTIATE_TEST_SUITE_P(ThpModes, HostMemPageModeTest, testing::Values("0", "1"));

TEST(KvCacheManagerV2HostMemTest, AllocationSupportsGpuRoundTrip)
{
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kSize = 4 << 20;
    HostMem memory(kSize);
    std::memset(reinterpret_cast<void*>(memory.address()), 0x5A, kSize);

    void* devicePtr = nullptr;
    ASSERT_EQ(cudaMalloc(&devicePtr, kSize), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(devicePtr, reinterpret_cast<void*>(memory.address()), kSize, cudaMemcpyHostToDevice), cudaSuccess);
    std::memset(reinterpret_cast<void*>(memory.address()), 0, kSize);
    ASSERT_EQ(
        cudaMemcpy(reinterpret_cast<void*>(memory.address()), devicePtr, kSize, cudaMemcpyDeviceToHost), cudaSuccess);

    auto const* bytes = reinterpret_cast<unsigned char const*>(memory.address());
    EXPECT_TRUE(std::all_of(bytes, bytes + kSize, [](unsigned char value) { return value == 0x5A; }));
    EXPECT_EQ(cudaFree(devicePtr), cudaSuccess);
}

//! Growing the host tier must not drop what it already holds, and must not move
//! it: the pool contains live KV pages and hands out pointers derived from the
//! base address.
TEST(KvCacheManagerV2HostMemTest, ResizeGrowPreservesContentsAndAddress)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kMax = 64 << 20;
    constexpr size_t kInitial = 8 << 20;
    constexpr size_t kGrown = 24 << 20;

    HostMem memory(kMax, kInitial);
    MemAddress const base = memory.address();
    auto* bytes = reinterpret_cast<unsigned char*>(base);
    for (size_t i = 0; i < kInitial; ++i)
    {
        bytes[i] = static_cast<unsigned char>(i & 0xFF);
    }

    memory.resize(kGrown);
    EXPECT_EQ(memory.size(), kGrown);
    ASSERT_EQ(memory.address(), base) << "resize moved the base address";

    for (size_t i = 0; i < kInitial; ++i)
    {
        ASSERT_EQ(bytes[i], static_cast<unsigned char>(i & 0xFF)) << "byte " << i << " lost by resize";
    }

    // The added tail is committed and writable.
    auto* tail = bytes + kInitial;
    std::memset(tail, 0xC3, kGrown - kInitial);
    EXPECT_TRUE(std::all_of(tail, tail + (kGrown - kInitial), [](unsigned char v) { return v == 0xC3; }));
}

//! Sizes are quantized to the unit memory is allocated in, which grows with the
//! tier. Nothing rounds to the fill chunk -- the last fill step is just short.
TEST(KvCacheManagerV2HostMemTest, GranularityGrowsWithTierSize)
{
    // Below the first threshold every tier shares the minimum unit.
    EXPECT_EQ(tierAllocationGranularity(CacheTier::HOST_MEM, 256 << 20), HostMem::kAlignment);
    // Above it the unit grows, and stops growing once the per-allocation cost
    // has been amortized away.
    EXPECT_GT(tierAllocationGranularity(CacheTier::HOST_MEM, size_t{4} << 30), HostMem::kAlignment);
    EXPECT_EQ(tierAllocationGranularity(CacheTier::HOST_MEM, size_t{1} << 40), size_t{32} << 20);
}

//! A commit unit larger than kAlignment must work end to end. Tiers of a few
//! GiB and up get one, so a size rounded to kAlignment instead would be
//! rejected by HostMem for every realistic host tier.
TEST(KvCacheManagerV2HostMemTest, HonoursCommitUnitLargerThanAlignment)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kUnit = size_t{32} << 20;
    ASSERT_GT(kUnit, HostMem::kAlignment);

    HostMem memory(4 * kUnit, 2 * kUnit, HostMemBackingOptions{kUnit});
    EXPECT_EQ(memory.size(), 2 * kUnit);
    auto* bytes = reinterpret_cast<unsigned char*>(memory.address());
    std::memset(bytes, 0x3C, 2 * kUnit);
    EXPECT_TRUE(std::all_of(bytes, bytes + 2 * kUnit, [](unsigned char v) { return v == 0x3C; }));

    // A size that is only kAlignment-aligned is not a valid size for this unit.
    EXPECT_THROW(memory.resize(2 * kUnit + HostMem::kAlignment), std::exception);
}

//! The pool must quantize its size to the same unit, or constructing any tier
//! whose granularity exceeds kAlignment throws.
TEST(KvCacheManagerV2HostMemTest, SlotPoolAlignsToCommitUnit)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    size_t const unit = tierAllocationGranularity(CacheTier::HOST_MEM, size_t{8} << 30);
    ASSERT_GT(unit, HostMem::kAlignment);

    // A slot size that divides kAlignment but not the larger unit, so a pool
    // rounding to the wrong one produces a size HostMem rejects.
    constexpr size_t kSlotSize = 1 << 20;
    HostSlotPool pool(kSlotSize, SlotCount{3}, /*vmSize=*/unit * 4, HostMemBackingOptions{unit});
    EXPECT_GE(pool.numSlots(), SlotCount{3});
    EXPECT_NE(std::get<MemAddress>(pool.slotAddress(SlotId{0})), MemAddress{0});
}

//! The reservation is a hard bound. Serving a larger resize would mean moving
//! the base address, which callers are entitled to assume never happens.
TEST(KvCacheManagerV2HostMemTest, ResizeBeyondReservationIsRejected)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kMax = 16 << 20;
    HostMem memory(kMax, size_t{4} << 20);
    MemAddress const base = memory.address();

    EXPECT_EQ(memory.vmSize(), kMax);
    EXPECT_NO_THROW(memory.resize(kMax));
    EXPECT_THROW(memory.resize(kMax + HostMem::kAlignment), std::exception);

    // The rejected resize left the object usable and unmoved.
    EXPECT_EQ(memory.address(), base);
    EXPECT_EQ(memory.size(), kMax);
}

//! Shrinking then regrowing stays inside the same reservation, so the address
//! survives a full cycle.
TEST(KvCacheManagerV2HostMemTest, AddressSurvivesShrinkAndRegrow)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kMax = 64 << 20;
    HostMem memory(kMax, size_t{32} << 20);
    MemAddress const base = memory.address();

    memory.resize(8 << 20);
    EXPECT_EQ(memory.address(), base);
    memory.resize(48 << 20);
    EXPECT_EQ(memory.address(), base);

    auto* bytes = reinterpret_cast<unsigned char*>(memory.address());
    std::memset(bytes, 0x2B, 48 << 20);
    EXPECT_TRUE(std::all_of(bytes, bytes + (48 << 20), [](unsigned char v) { return v == 0x2B; }));
}

TEST(KvCacheManagerV2HostMemTest, ResizeShrinkKeepsRetainedContents)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kInitial = 24 << 20;
    constexpr size_t kShrunk = 8 << 20;

    HostMem memory(kInitial);
    auto* bytes = reinterpret_cast<unsigned char*>(memory.address());
    std::memset(bytes, 0x7E, kInitial);

    memory.resize(kShrunk);
    EXPECT_EQ(memory.size(), kShrunk);

    auto const* retained = reinterpret_cast<unsigned char const*>(memory.address());
    EXPECT_TRUE(std::all_of(retained, retained + kShrunk, [](unsigned char v) { return v == 0x7E; }));
}

//! With waitForFill disabled the constructor returns before the range is
//! committed, and waitUsable is what makes a given offset safe to touch.
TEST(KvCacheManagerV2HostMemTest, DeferredFillIsUsableAfterWaitUsable)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // Large enough to span several fill chunks, so the wait is not trivially
    // satisfied by the first commit.
    constexpr size_t kSize = size_t{3} * (512 << 20);

    HostMem memory(kSize, kSize, HostMemBackingOptions{}, /*waitForFill=*/false);
    ASSERT_NE(memory.address(), 0U);

    // Walk the range in ascending order, exactly as the slot allocator does.
    auto* bytes = reinterpret_cast<unsigned char*>(memory.address());
    for (size_t offset = 0; offset < kSize; offset += (64 << 20))
    {
        memory.waitUsable(offset + (64 << 20));
        bytes[offset] = static_cast<unsigned char>((offset >> 20) & 0xFF);
    }

    memory.waitUsable(kSize);
    for (size_t offset = 0; offset < kSize; offset += (64 << 20))
    {
        ASSERT_EQ(bytes[offset], static_cast<unsigned char>((offset >> 20) & 0xFF)) << "offset " << offset;
    }
}

//! waitUsable must be safe to call from many threads at once, and must stay
//! correct when the fill has already finished before the first call.
TEST(KvCacheManagerV2HostMemTest, ConcurrentWaitUsableFromManyThreads)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t kSize = 64 << 20;
    constexpr int kThreads = 16;

    HostMem memory(kSize, kSize, HostMemBackingOptions{}, /*waitForFill=*/false);
    std::atomic<int> done{0};

    std::vector<std::thread> waiters;
    waiters.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        waiters.emplace_back(
            [&memory, &done, i]
            {
                memory.waitUsable(kSize);
                // Each waiter writes its own byte. waitUsable orders a waiter
                // against the fill worker but not against the other waiters, so
                // sharing one byte would be a data race even writing equal
                // values, and would say nothing about the rest of the range.
                *reinterpret_cast<unsigned char*>(memory.address() + kSize - 1 - i) = static_cast<unsigned char>(i);
                done.fetch_add(1, std::memory_order_relaxed);
            });
    }
    for (auto& waiter : waiters)
    {
        waiter.join();
    }
    EXPECT_EQ(done.load(), kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        EXPECT_EQ(
            *reinterpret_cast<unsigned char const*>(memory.address() + kSize - 1 - i), static_cast<unsigned char>(i))
            << "byte written by waiter " << i << " did not survive";
    }
}

} // namespace
