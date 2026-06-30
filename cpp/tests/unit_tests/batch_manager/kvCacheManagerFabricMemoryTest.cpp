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
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/testing/kvCacheManagerTestUtil.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using tensorrt_llm::batch_manager::LlmRequest;
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
    if (FabricMemory::supportFabricMemory())
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
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, 0);
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
    if (!FabricMemory::supportFabricMemory())
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
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, 0);
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

// Round-trip a fabric-backed primary block through the host-pinned secondary pool to validate
// that DRAM offload/onboard work correctly when the primary is fabric VMM and to make sure
// releasePools() syncs the transfer manager before unmapping the fabric allocation.
TEST_F(KVCacheManagerFabricMemoryTest, OffloadOnboardRoundTripWithFabricPrimary)
{
    if (!FabricMemory::supportFabricMemory())
    {
        GTEST_SKIP() << "Fabric memory not supported on this hardware";
    }

    using Element = std::uint16_t; // matches kHALF size
    auto constexpr numLayers = 2;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 32;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;
    auto constexpr maxNumSequences = 2;
    auto constexpr beamWidth = 1;
    auto constexpr beamIdx = 0;
    auto constexpr maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;
    auto constexpr transferMode = tensorrt_llm::executor::KvCacheTransferMode::DRAM;

    auto const stream = std::make_shared<tr::CudaStream>();
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksPerWindow, maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, 0);
    blockManager.allocatePools(false);

    auto primaryPoolPtr = blockManager.getPrimaryPool(0);
    auto secondaryPoolPtr = blockManager.getSecondaryPool(0);
    ASSERT_NE(primaryPoolPtr, nullptr);
    ASSERT_NE(secondaryPoolPtr, nullptr);
    EXPECT_TRUE(isFabricVmmAllocation(primaryPoolPtr->data()))
        << "Primary pool must be fabric VMM-backed for this test";

    // Block has shape [2, numLayers, numKvHeads, tokensPerBlock, sizePerHead].
    auto const elemsPerBlock = static_cast<size_t>(2) * numLayers * numKvHeads * tokensPerBlock * sizePerHead;
    auto const bytesPerBlock = elemsPerBlock * sizeof(Element);

    // Drive the primary block allocator through addSequenceBatch.
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType const requestId{0};
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    GenerationRequest seq{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    auto const promptLen = llmRequest->getNumTokens(beamIdx);
    auto const numContextBlocks = tc::ceilDiv(promptLen, blockManager.getTokensPerBlock());
    auto const seqStats = blockManager.addSequenceBatch({&seq}, {promptLen}, {numContextBlocks},
        {std::ref(*llmRequest)}, maxAttentionWindow,
        /*isEnableBlockReuse=*/false);
    ASSERT_FALSE(seqStats.empty());

    auto cacheBlockIds = seq.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    ASSERT_FALSE(cacheBlockIds.empty());
    auto const blockId = cacheBlockIds.front();
    auto block = blockManager.getBlockById(blockId, maxAttentionWindow);
    ASSERT_TRUE(block->isPrimary());

    // Write a known pattern into the fabric-backed primary block.
    std::vector<Element> hostBuffer(elemsPerBlock);
    for (size_t i = 0; i < elemsPerBlock; ++i)
    {
        hostBuffer[i] = static_cast<Element>(i & 0xFFFF);
    }
    auto const memoryPoolIndex = block->getMemoryPoolBlockIndex();
    auto primaryBlockPtr = tr::ITensor::slice(primaryPoolPtr, memoryPoolIndex, 1);
    ASSERT_EQ(
        cudaMemcpy(primaryBlockPtr->data(), hostBuffer.data(), bytesPerBlock, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Offload fabric primary -> host pinned secondary and verify pattern survived.
    blockManager.offloadBlock(block, maxAttentionWindow, transferMode, /*directory=*/"");
    EXPECT_FALSE(block->isPrimary());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto const secondaryIndex = block->getMemoryPoolBlockIndex();
    auto secondaryBlockPtr = tr::ITensor::slice(secondaryPoolPtr, secondaryIndex, 1);
    auto const* secondaryRaw = reinterpret_cast<Element const*>(secondaryBlockPtr->data());
    for (size_t i = 0; i < elemsPerBlock; ++i)
    {
        ASSERT_EQ(secondaryRaw[i], hostBuffer[i]) << "Mismatch at element " << i << " after fabric->host offload";
    }

    // Onboard host secondary -> fabric primary, then read back and verify integrity.
    blockManager.onboardBlock(seq, block, maxAttentionWindow, transferMode, /*directory=*/"");
    EXPECT_TRUE(block->isPrimary());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<Element> readback(elemsPerBlock, 0);
    auto const onboardIndex = block->getMemoryPoolBlockIndex();
    auto reboardBlockPtr = tr::ITensor::slice(primaryPoolPtr, onboardIndex, 1);
    ASSERT_EQ(cudaMemcpy(readback.data(), reboardBlockPtr->data(), bytesPerBlock, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < elemsPerBlock; ++i)
    {
        ASSERT_EQ(readback[i], hostBuffer[i]) << "Mismatch at element " << i << " after host->fabric onboard";
    }

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    blockManager.releaseBlocks(seq, llmRequest);

    // releasePools() must syncTransfers() + sync the buffer-manager stream before unmapping the
    // fabric allocation; if it does not, the transfer queued above can race with cuMemUnmap.
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
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, 0);

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
