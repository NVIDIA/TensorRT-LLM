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
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <system_error>
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

TEST(KvCacheManagerV2HostMemTest, ReadsPrefaultThreadConfiguration)
{
    ScopedEnv disablePrefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    EXPECT_EQ(hostPrefaultThreads(), 0);
    {
        ScopedEnv threeThreads("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "3");
        EXPECT_EQ(hostPrefaultThreads(), 3);
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

TEST_P(HostMemPageModeTest, RegistersAndResizesWithPrefaultDisabled)
{
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", std::string(GetParam()));
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    HostMem memory(HostMem::kAlignment);
    EXPECT_NE(memory.address(), 0);
    EXPECT_EQ(memory.size(), HostMem::kAlignment);
    memory.resize(2 * HostMem::kAlignment);
    EXPECT_EQ(memory.size(), 2 * HostMem::kAlignment);
}

INSTANTIATE_TEST_SUITE_P(ThpModes, HostMemPageModeTest, testing::Values("0", "1"));

TEST(KvCacheManagerV2HostMemTest, PrefaultedAllocationSupportsGpuRoundTrip)
{
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "2");
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

bool isRegistered(MemAddress addr)
{
    unsigned int flags = 0;
    bool const registered = cudaHostGetFlags(&flags, reinterpret_cast<void*>(addr)) == cudaSuccess;
    cudaGetLastError();
    return registered;
}

void fillPattern(MemAddress addr, size_t begin, size_t end)
{
    auto* bytes = reinterpret_cast<unsigned char*>(addr);
    for (size_t i = begin; i < end; i += HostMem::kAlignment)
    {
        bytes[i] = static_cast<unsigned char>((i / HostMem::kAlignment) * 7 + 1);
    }
}

bool checkPattern(MemAddress addr, size_t begin, size_t end)
{
    auto const* bytes = reinterpret_cast<unsigned char const*>(addr);
    for (size_t i = begin; i < end; i += HostMem::kAlignment)
    {
        if (bytes[i] != static_cast<unsigned char>((i / HostMem::kAlignment) * 7 + 1))
        {
            return false;
        }
    }
    return true;
}

TEST(KvCacheManagerV2HostMemTest, ReservedResizeKeepsAddressDataAndRegistration)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "2");
    ScopedEnv delta("TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN", std::nullopt);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kMiB = 1 << 20;
    HostMem memory(4 * kMiB, /*alignHint=*/HostMem::kAlignment, /*maxSize=*/64 * kMiB);
    MemAddress const addr = memory.address();
    fillPattern(addr, 0, 4 * kMiB);

    memory.resize(40 * kMiB);
    EXPECT_EQ(memory.address(), addr);
    EXPECT_TRUE(checkPattern(addr, 0, 4 * kMiB));
    fillPattern(addr, 4 * kMiB, 40 * kMiB);
    EXPECT_TRUE(isRegistered(addr + 39 * kMiB));

    memory.resize(10 * kMiB);
    EXPECT_EQ(memory.address(), addr);
    EXPECT_TRUE(checkPattern(addr, 0, 10 * kMiB));
    EXPECT_TRUE(isRegistered(addr + 10 * kMiB - HostMem::kAlignment));
    EXPECT_FALSE(isRegistered(addr + 10 * kMiB));

    memory.resize(64 * kMiB);
    EXPECT_EQ(memory.address(), addr);
    EXPECT_TRUE(checkPattern(addr, 0, 10 * kMiB));
    EXPECT_TRUE(isRegistered(addr + 64 * kMiB - HostMem::kAlignment));

    // Past the reservation: falls back to moving the pages, data survives.
    fillPattern(addr, 0, 64 * kMiB);
    memory.resize(96 * kMiB);
    EXPECT_TRUE(checkPattern(memory.address(), 0, 64 * kMiB));
    EXPECT_TRUE(isRegistered(memory.address() + 96 * kMiB - HostMem::kAlignment));
}

TEST(KvCacheManagerV2HostMemTest, ReservedExtentsNeverSplitASlot)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv delta("TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN", std::nullopt);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kSlot = 3 * HostMem::kAlignment + 2048; // 14336, lcm with 4 KiB is 28672
    auto const bytesFor = [](size_t slots)
    { return (slots * kSlot + HostMem::kAlignment - 1) / HostMem::kAlignment * HostMem::kAlignment; };
    HostMem memory(bytesFor(100), kSlot, bytesFor(4000));
    for (size_t slots : {size_t{1003}, size_t{517}, size_t{2999}, size_t{9}, size_t{4000}})
    {
        memory.resize(bytesFor(slots));
        for (size_t slot = 0; slot < slots; slot += 37)
        {
            MemAddress const slotAddr = memory.address() + slot * kSlot;
            CUdeviceptr start = 0;
            ASSERT_EQ(cuPointerGetAttribute(&start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, slotAddr), CUDA_SUCCESS);
            EXPECT_EQ((start - memory.address()) % kSlot, 0u) << "slots=" << slots << " slot=" << slot;
            EXPECT_TRUE(isRegistered(slotAddr + kSlot - 1)) << "slots=" << slots << " slot=" << slot;
        }
    }
}

TEST(KvCacheManagerV2HostMemTest, OversizedAlignmentFallsBackToFullRegistration)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv delta("TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN", std::nullopt);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    // lcm(4 KiB, 1 MiB + 1) is ~4 GiB, beyond the 1 GiB extent limit, so this pool cannot use slot-aligned extents.
    constexpr size_t kMiB = 1 << 20;
    HostMem memory(4 * kMiB, kMiB + 1, 64 * kMiB);
    fillPattern(memory.address(), 0, 4 * kMiB);
    memory.resize(32 * kMiB);
    EXPECT_TRUE(checkPattern(memory.address(), 0, 4 * kMiB));
    EXPECT_TRUE(isRegistered(memory.address() + 32 * kMiB - HostMem::kAlignment));
    memory.resize(8 * kMiB);
    EXPECT_TRUE(checkPattern(memory.address(), 0, 4 * kMiB));
    EXPECT_FALSE(isRegistered(memory.address() + 8 * kMiB));
}

TEST(KvCacheManagerV2HostMemTest, DeltaPinCanBeDisabled)
{
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv delta("TLLM_KV_CACHE_MANAGER_V2_HOST_DELTA_PIN", "0");
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kMiB = 1 << 20;
    HostMem memory(2 * kMiB, HostMem::kAlignment, 64 * kMiB);
    fillPattern(memory.address(), 0, 2 * kMiB);
    memory.resize(8 * kMiB);
    EXPECT_TRUE(checkPattern(memory.address(), 0, 2 * kMiB));
    EXPECT_TRUE(isRegistered(memory.address() + 8 * kMiB - HostMem::kAlignment));
}

} // namespace
