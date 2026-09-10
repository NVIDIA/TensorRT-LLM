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

} // namespace
