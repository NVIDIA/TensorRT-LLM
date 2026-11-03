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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/cudaVirtMem.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/storage/core.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

constexpr size_t kMiB = size_t{1} << 20;

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
        int const rc = value.has_value() ? ::setenv(name, value->c_str(), /*overwrite=*/1) : ::unsetenv(name);
        if (rc != 0)
        {
            throw std::system_error(errno, std::generic_category(), "setenv/unsetenv failed");
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

    ScopedEnv(ScopedEnv const&) = delete;
    ScopedEnv& operator=(ScopedEnv const&) = delete;

private:
    std::string mName;
    std::optional<std::string> mOldValue;
};

//! Sizes of the physical allocations backing [address, address + bytes), in order.
std::vector<size_t> chunkSizes(MemAddress address, size_t bytes)
{
    std::vector<size_t> sizes;
    CUdeviceptr cur = static_cast<CUdeviceptr>(address);
    CUdeviceptr const end = cur + bytes;
    while (cur < end)
    {
        CUdeviceptr base = 0;
        size_t size = 0;
        EXPECT_EQ(cuMemGetAddressRange(&base, &size, cur), CUDA_SUCCESS);
        EXPECT_EQ(base, cur);
        if (size == 0)
        {
            break;
        }
        sizes.push_back(size);
        cur += size;
    }
    return sizes;
}

class KvCacheManagerV2VirtMemTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess); // make the primary context current
    }
};

//! The env var only raises the largest chunk; the unit the pools are sized in stays the policy's.
TEST(KvCacheManagerV2GpuChunkSizeTest, EnvRaisesOnlyTheLargestChunk)
{
    {
        ScopedEnv unset(kGpuChunkSizeEnvVar, std::nullopt);
        EXPECT_EQ(gpuMaxChunkSize(32 * kMiB), 32 * kMiB);
    }
    for (char const* value : {"", "0", "8", "32"})
    {
        ScopedEnv noEffect(kGpuChunkSizeEnvVar, std::string(value));
        EXPECT_EQ(gpuMaxChunkSize(32 * kMiB), 32 * kMiB) << "value '" << value << "'";
    }
    {
        ScopedEnv chunk(kGpuChunkSizeEnvVar, std::string("512"));
        EXPECT_EQ(gpuMaxChunkSize(32 * kMiB), 512 * kMiB);
        EXPECT_EQ(gpuMaxChunkSize(2 * kMiB), 512 * kMiB);
        // The sizing unit does not see the env var.
        EXPECT_EQ(tierAllocationGranularity(CacheTier::GPU_MEM, size_t{100} << 30), 32 * kMiB);
    }
    {
        ScopedEnv chunk(kGpuChunkSizeEnvVar, std::string("1000")); // not a power of two: rounded down
        EXPECT_EQ(gpuMaxChunkSize(32 * kMiB), 512 * kMiB);
    }
    for (char const* value : {"1g", "-512", "abc", "512 "})
    {
        ScopedEnv bad(kGpuChunkSizeEnvVar, std::string(value));
        EXPECT_THROW((void) gpuMaxChunkSize(32 * kMiB), std::invalid_argument) << "value '" << value << "'";
    }
}

TEST_F(KvCacheManagerV2VirtMemTest, ChunkSizeAtPicksTheLargestAlignedChunkThatFits)
{
    PooledPhysMemAllocator allocator(2 * kMiB, 20 * kMiB); // rounded down to 16 MiB
    EXPECT_EQ(allocator.maxChunkSize(), 16 * kMiB);
    EXPECT_EQ(allocator.chunkSizeAt(0, 64 * kMiB), 16 * kMiB);
    EXPECT_EQ(allocator.chunkSizeAt(0, 6 * kMiB), 4 * kMiB);
    EXPECT_EQ(allocator.chunkSizeAt(36 * kMiB, 28 * kMiB), 4 * kMiB);
    EXPECT_EQ(allocator.chunkSizeAt(2 * kMiB, 62 * kMiB), 2 * kMiB);

    PooledPhysMemAllocator uniform(2 * kMiB);
    EXPECT_EQ(uniform.maxChunkSize(), 2 * kMiB);
    EXPECT_EQ(uniform.chunkSizeAt(0, 64 * kMiB), 2 * kMiB);
}

//! A range is mapped from the largest chunks its offsets allow, a shrink unmaps only whole chunks
//! above the new end, and the usable size is always what was asked for.
TEST_F(KvCacheManagerV2VirtMemTest, MapsMixedChunksAndShrinksByWholeChunks)
{
    PooledPhysMemAllocator allocator(2 * kMiB, 16 * kMiB);
    VirtMem vm(64 * kMiB, allocator);

    vm.realloc(38 * kMiB);
    EXPECT_EQ(vm.usableBytes(), 38 * kMiB);
    EXPECT_EQ(vm.mappedBytes(), 38 * kMiB);
    EXPECT_EQ(
        chunkSizes(vm.address(), vm.mappedBytes()), (std::vector<size_t>{16 * kMiB, 16 * kMiB, 4 * kMiB, 2 * kMiB}));
    EXPECT_EQ(vm.numPhysMem(), 4U);

    // The written bytes survive a shrink that ends inside the second 16 MiB chunk: it stays mapped.
    ASSERT_EQ(cuMemsetD8(static_cast<CUdeviceptr>(vm.address()) + 17 * kMiB, 0x5A, kMiB), CUDA_SUCCESS);
    vm.realloc(20 * kMiB);
    EXPECT_EQ(vm.usableBytes(), 20 * kMiB);
    EXPECT_EQ(vm.mappedBytes(), 32 * kMiB);
    EXPECT_EQ(chunkSizes(vm.address(), vm.mappedBytes()), (std::vector<size_t>{16 * kMiB, 16 * kMiB}));
    unsigned char byte = 0;
    ASSERT_EQ(cuMemcpyDtoH(&byte, static_cast<CUdeviceptr>(vm.address()) + 17 * kMiB + 123, 1), CUDA_SUCCESS);
    EXPECT_EQ(byte, 0x5A);

    // Growing again first reuses what is still mapped.
    vm.realloc(30 * kMiB);
    EXPECT_EQ(vm.usableBytes(), 30 * kMiB);
    EXPECT_EQ(vm.mappedBytes(), 32 * kMiB);
    vm.realloc(48 * kMiB);
    EXPECT_EQ(chunkSizes(vm.address(), vm.mappedBytes()), (std::vector<size_t>{16 * kMiB, 16 * kMiB, 16 * kMiB}));

    // extend / shrink count physMemSize units of the usable range.
    vm.shrink(4);
    EXPECT_EQ(vm.usableBytes(), 40 * kMiB);
    EXPECT_EQ(vm.mappedBytes(), 48 * kMiB);
    vm.extend(1);
    EXPECT_EQ(vm.usableBytes(), 42 * kMiB);

    vm.realloc(0);
    EXPECT_EQ(vm.usableBytes(), 0U);
    EXPECT_EQ(vm.mappedBytes(), 0U);
    EXPECT_EQ(vm.numPhysMem(), 0U);
}

//! Without larger chunks the mapping is one unit per chunk, exactly as before.
TEST_F(KvCacheManagerV2VirtMemTest, UniformAllocatorMapsOneUnitPerChunk)
{
    PooledPhysMemAllocator allocator(2 * kMiB);
    VirtMem vm(16 * kMiB, allocator, /*initNumPhysMem=*/3);
    EXPECT_EQ(vm.mappedBytes(), 6 * kMiB);
    EXPECT_EQ(vm.usableBytes(), 6 * kMiB);
    EXPECT_EQ(vm.numPhysMem(), 3U);
    EXPECT_EQ(chunkSizes(vm.address(), vm.mappedBytes()), (std::vector<size_t>(3, 2 * kMiB)));
    vm.shrink(1);
    EXPECT_EQ(vm.mappedBytes(), 4 * kMiB);
    EXPECT_EQ(vm.numPhysMem(), 2U);
}

} // namespace
