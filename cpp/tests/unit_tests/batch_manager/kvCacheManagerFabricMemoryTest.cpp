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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

// ============================================================================
// Fabric memory pool allocation tests
// ============================================================================
// This test binary is separate from kvCacheManagerTest so that setting
// TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY before the static cache is
// initialized takes effect for all tests.

namespace
{

// Helper: check whether a device pointer was allocated via cuMemCreate with fabric handle.
bool isFabricVmmAllocation(void* ptr)
{
    CUmemGenericAllocationHandle handle{};
    auto ret = cuMemRetainAllocationHandle(&handle, ptr);
    if (ret != CUDA_SUCCESS)
    {
        return false; // Not a VMM allocation (e.g. cudaMalloc)
    }

    CUmemAllocationProp prop{};
    auto ret2 = cuMemGetAllocationPropertiesFromHandle(&prop, handle);
    cuMemRelease(handle);
    if (ret2 != CUDA_SUCCESS)
    {
        return false;
    }

    return (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) != 0;
}

} // namespace

class KVCacheManagerFabricMemoryTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        setenv("TRTLLM_KVCACHE_POOL_USE_FABRIC_MEMORY", "1", 1);
    }

    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(KVCacheManagerFabricMemoryTest, AllocatePoolsFallbackWhenFabricUnsupported)
{
    if (FabricMemory::supportFbaricMemory())
    {
        GTEST_SKIP() << "This test targets hardware without fabric memory support";
    }

    auto constexpr numLayers = 4;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Pool should use regular GPU allocation, not fabric VMM
    auto primaryPool = blockManager.getPrimaryPool(0);
    EXPECT_NE(primaryPool, nullptr);
    EXPECT_GT(primaryPool->getSize(), 0);
    EXPECT_FALSE(isFabricVmmAllocation(primaryPool->data()))
        << "Pool should use regular GPU allocation when fabric memory is unsupported";

    blockManager.releasePools();
}

TEST_F(KVCacheManagerFabricMemoryTest, AllocatePoolsWithFabricMemory)
{
    if (!FabricMemory::supportFbaricMemory())
    {
        GTEST_SKIP() << "Fabric memory not supported on this hardware";
    }

    auto constexpr numLayers = 4;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Pool should be a VMM allocation with fabric handle
    auto primaryPool = blockManager.getPrimaryPool(0);
    EXPECT_NE(primaryPool, nullptr);
    EXPECT_GT(primaryPool->getSize(), 0);
    EXPECT_TRUE(isFabricVmmAllocation(primaryPool->data()))
        << "Pool should be backed by fabric VMM when env var is set and hardware supports it";

    blockManager.releasePools();
}

TEST_F(KVCacheManagerFabricMemoryTest, ReleasePoolsClearsFabricMemory)
{
    auto constexpr numLayers = 2;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 2;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);

    size_t freeBefore = 0;
    size_t freeAfterAlloc = 0;
    size_t freeAfterRelease = 0;
    size_t total = 0;

    cudaMemGetInfo(&freeBefore, &total);

    blockManager.allocatePools(false);
    cudaMemGetInfo(&freeAfterAlloc, &total);
    EXPECT_LT(freeAfterAlloc, freeBefore) << "allocatePools should consume GPU memory";

    blockManager.releasePools();
    cudaMemGetInfo(&freeAfterRelease, &total);
    EXPECT_GE(freeAfterRelease, freeBefore - (1U << 20U))
        << "releasePools should return GPU memory (within 1MB tolerance)";

    // Second cycle: verify no accumulation
    cudaMemGetInfo(&freeBefore, &total);
    blockManager.allocatePools(false);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    blockManager.releasePools();
    cudaMemGetInfo(&freeAfterRelease, &total);
    EXPECT_GE(freeAfterRelease, freeBefore - (1U << 20U)) << "Second release should also return GPU memory";
}
