/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <set>
#include <thread>
#include <variant>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tlk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;

using ParamType = bool;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const homogeneousLayers = info.param;
    std::string name = "KVCacheManagerTest";
    if (homogeneousLayers)
    {
        name += "Homogeneous";
    }
    else
    {
        name += "Heterogeneous";
    }
    return name;
}

class KVCacheManagerTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<ParamType> // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        auto const deviceCount = tc::getDeviceCount();
        if (deviceCount == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

namespace
{
void allocateBlocks(BlockManager& manager, GenerationRequest& sequence, std::size_t numBlocks, bool shareAmongBeams)
{
    for (std::size_t i = 0; i < numBlocks; ++i)
    {
        manager.allocateBlock(sequence, shareAmongBeams);
    }
}

// TODO: This is really ugly. Flushing the event queue is done in a separate thread, so if we want to check the value we
// need to wait for the thread to complete. It works, but it's technically not deterministic.
std::deque<tle::KVCacheEvent> getEvents(KVCacheManager& kvCacheManager)
{
    kvCacheManager.flushIterationEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return kvCacheManager.getLatestEvents(std::chrono::milliseconds(100));
}

} // namespace

TEST_F(KVCacheManagerTest, BlockManagerTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 64;
    auto constexpr blocksInPrimaryPool = 24;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    auto constexpr requestId = 42;
    auto constexpr beamWidth = 8;
    auto constexpr numBlocksPerBeam = blocksInPrimaryPool / beamWidth;
    auto constexpr numTokens = tokensPerBlock * numBlocksPerBeam;
    GenerationRequest seq0{requestId, numTokens, beamWidth, numBlocksPerBeam};
    blockManager.addSequence(seq0, numBlocksPerBeam, numBlocksPerBeam - 1);
    auto constexpr occupiedBlocks = (numBlocksPerBeam - 1) + beamWidth;
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - occupiedBlocks);
    auto const& ids = seq0.getCacheBlockIds();
    std::set<std::int32_t> idSet{};
    EXPECT_EQ(ids.size(), beamWidth);
    for (auto const& beam : ids)
    {
        EXPECT_EQ(beam.size(), blocksInPrimaryPool / beamWidth);
        idSet.insert(beam.begin(), beam.end());
    }
    EXPECT_EQ(idSet.size(), occupiedBlocks);
    blockManager.releaseBlocks(seq0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    blockManager.addSequence(seq0, numBlocksPerBeam, -1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocksPerBeam);
    EXPECT_EQ(ids.size(), beamWidth);
    for (std::size_t i = 0u; i < ids.front().size(); ++i)
    {
        for (std::size_t beam = 1u; beam < ids.size(); ++beam)
        {
            EXPECT_EQ(ids.at(beam).at(i), ids.at(0).at(i));
        }
    }
    blockManager.releaseBlocks(seq0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // occupy 22/24 blocks
    EXPECT_NO_THROW(blockManager.addSequence(seq0, numBlocksPerBeam, numBlocksPerBeam - 1));
    GenerationRequest seq1{requestId + 1, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_NO_THROW(blockManager.addSequence(seq1, numBlocksPerBeam, numBlocksPerBeam - 1));
    // same requestId not allowed
    GenerationRequest seq2{requestId, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_THROW(blockManager.addSequence(seq2, numBlocksPerBeam, numBlocksPerBeam - 1), std::runtime_error);
    // no more blocks
    GenerationRequest seq3{requestId + 2, numTokens, beamWidth, numBlocksPerBeam};
    EXPECT_THROW(blockManager.addSequence(seq3, numBlocksPerBeam, numBlocksPerBeam - 1), std::runtime_error);
}

template <typename T, nvinfer1::DataType type, int mask>
void runPartialCopyTest()
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    auto constexpr batchSize = 1;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr bytesPerToken = 4;
    auto constexpr maxAttentionWindow = 4096;
    auto constexpr maxAttentionWindowAllLayer = 4096;
    auto constexpr sinkTokenLen = 0;
    auto constexpr canUseOneMoreBlock = true;

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    auto constexpr beamIdx = 0;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(type, false);

    auto oneLayerBlockSize = blockManager.getBlockSize(0);
    EXPECT_EQ(oneLayerBlockSize, numKvHeads * sizePerHead * tokensPerBlock);

    auto primaryPoolPtr = blockManager.getPrimaryPool(0);
    auto secondaryPoolPtr = blockManager.getSecondaryPool(0);
    tk::KVBlockArray kvCacheBlockArray(batchSize, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, maxAttentionWindow,
        maxAttentionWindowAllLayer, sinkTokenLen, canUseOneMoreBlock, primaryPoolPtr->data(), secondaryPoolPtr->data(),
        nullptr);

    // Verify that shape of block for one layer of K or V is [numKvHeads, tokensPerBlock, sizePerHead] by comparing
    // against KVBlockArray::getKVLocalIdx method. We make this assumption in partialCopy kernel.
    auto constexpr localTokenIdx = 3;
    auto constexpr headIdx = 5;
    auto constexpr channelIdx = 7;
    auto localKIdx = kvCacheBlockArray.getKVLocalIdx(localTokenIdx, headIdx, sizePerHead, channelIdx);
    EXPECT_EQ(localKIdx, (headIdx * tokensPerBlock + localTokenIdx) * sizePerHead + channelIdx);
    // Pool block has shape [2, numLayers, numKvHeads, tokensPerBlock, sizePerHead]
    auto blockSize = 2 * numLayers * oneLayerBlockSize;
    auto primaryPoolSize = blocksInPrimaryPool * blockSize;
    auto secondaryPoolSize = blocksInSecondaryPool * blockSize;

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] (17 tokens, three blocks)
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    auto cacheBlockIds = seq0.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(cacheBlockIds, ::testing::ElementsAreArray({0, 1, 2}));

    // Offload all 3 blocks, fill with predictable pattern, onboard
    for (auto cacheBlockId : cacheBlockIds)
    {
        auto block = blockManager.getBlockById(cacheBlockId);
        EXPECT_TRUE(block->isPrimary());
        // offload so we can write to block in CPU code
        blockManager.offloadBlock(block);
        EXPECT_FALSE(block->isPrimary());
        // need to sync so D2H transfer is done before accessing blocks
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // fill with predictable pattern
        auto memoryPoolIndex = block->getMemoryPoolBlockIndex();
        auto blockPtr{tr::ITensor::slice(secondaryPoolPtr, memoryPoolIndex, 1)};
        auto rawBlockPtr = reinterpret_cast<T*>(blockPtr->data());
        for (int i = 0; i < blockSize; ++i)
        {
            rawBlockPtr[i] = i & mask;
        }
        // onboard
        blockManager.onboardBlock(block);
        EXPECT_TRUE(block->isPrimary());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(blockManager.verifyQueueIntegrity());
    }
    blockManager.releaseBlocks(seq0, llmRequest0);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    auto inputTokens1 = inputTokens;
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength1, beamWidth, maxBlocksPerSeq};
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 16);
    auto cacheBlockIds1 = seq1.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(cacheBlockIds1, ::testing::ElementsAreArray({0, 1, 6}));
    // store blocks 0, 1 ([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15])
    blockManager.storeContextBlocks(seq1, *llmRequest1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11] again.
    // Reuse blocks 0 and 1(pc). Block 1 is partially reused, but already referenced by seq1 so must be partial copied
    // into new block 2. Clear block 2 so we can see what was partial copied.
    auto block2 = blockManager.getBlockById(2);
    auto memoryPoolIndex2 = block2->getMemoryPoolBlockIndex();
    auto block2Ptr{tr::ITensor::slice(primaryPoolPtr, memoryPoolIndex2, 1)};
    EXPECT_EQ(cudaMemset(block2Ptr->data(), 0, blockSize * sizeof(T)), cudaSuccess);
    auto inputTokens2 = inputTokens;
    auto constexpr partiallyReusedTokens = 3;
    inputTokens2->resize(8 + partiallyReusedTokens + 1);
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    GenerationRequest seq2{requestId, inputLength2, beamWidth, maxBlocksPerSeq};
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 11);
    auto cacheBlockIds2 = seq2.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(cacheBlockIds2, ::testing::ElementsAreArray({0, 2}));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Verify partial copied block 2
    // Block has shape [2, numLayers, numKvHeads, tokensPerBlock, sizePerHead]
    blockManager.offloadBlock(block2);
    EXPECT_FALSE(block2->isPrimary());
    // need to sync so D2H transfer is done before accessing blocks
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    memoryPoolIndex2 = block2->getMemoryPoolBlockIndex();
    block2Ptr = tr::ITensor::slice(secondaryPoolPtr, memoryPoolIndex2, 1);
    T const* rawPtr2 = reinterpret_cast<T*>(block2Ptr->data());
    int numBad = 0;
    for (int i = 0; i < blockSize && numBad < 10; ++i)
    {
        T value = rawPtr2[i];
        int kOrV = i / (numLayers * numKvHeads * tokensPerBlock * sizePerHead);
        int j = i - kOrV * (numLayers * numKvHeads * tokensPerBlock * sizePerHead);
        int layer = j / (numKvHeads * tokensPerBlock * sizePerHead);
        j = j - layer * (numKvHeads * tokensPerBlock * sizePerHead);
        int head = j / (tokensPerBlock * sizePerHead);
        j = j - head * (tokensPerBlock * sizePerHead);
        int token = j / sizePerHead;
        j = j - token * sizePerHead;
        T expectedValue = (token < partiallyReusedTokens) ? i & mask : 0;
        if (value != expectedValue)
        {
            TLLM_LOG_WARNING(
                "block2[%d,%d,%d,%d,%d] - expected %d, actual %d", kOrV, layer, head, token, j, expectedValue, value);
            ++numBad;
        }
    }
    EXPECT_EQ(numBad, 0);
    blockManager.onboardBlock(block2);
    EXPECT_TRUE(block2->isPrimary());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseBlocks(seq2, llmRequest2);
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT64)
{
    runPartialCopyTest<std::uint64_t, nvinfer1::DataType::kINT64, -1>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT32)
{
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kINT32, -1>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyFLOAT)
{
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kFLOAT, -1>();
}

#ifdef ENABLE_BF16
TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyBF16)
{
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kBF16, 65535>();
}
#endif

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyHALF)
{
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kHALF, 65535>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyBOOL)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kBOOL, 255>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyUINT8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kUINT8, 255>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kINT8, 255>();
}

#ifdef ENABLE_FP8
TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyFP8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kFP8, 255>();
}
#endif

#ifdef ENABLE_FP4
TEST_F(KVCacheManagerTest, FP4BlockScaleManagementTest)
{
    auto constexpr numLayers = 6;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 16;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr beamWidth = 1;

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, false, stream, true,
        onboardBlocks);

    kvCacheManager.allocatePools(nvinfer1::DataType::kFP4, /*useUvm=*/false);

    // We should have one additional pool for the block scales.
    EXPECT_EQ(kvCacheManager.getBlockManager().getNumPools(), 2);
    EXPECT_EQ(kvCacheManager.getBlockManager().getNumPools(/*includeBlockScalePools=*/false), 1);
    EXPECT_NE(kvCacheManager.getBlockScalePoolPointers(), nullptr);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_TRUE(blockManager.containsBlockScales(1));
    EXPECT_EQ(blockManager.getBlockSize(0) / 16, blockManager.getBlockSize(1));
}
#endif

TEST_F(KVCacheManagerTest, BlockManagerReuseTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8]
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);  // block 2 contains [8]
    llmRequest0->addNewToken(10, beamIdx); // block 2 contains [8, 9]
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (blocks contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens [0, 1, 2, 3, 4, 5, 6, 7, 8] and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 ([0, 1, 2, 3], [4, 5, 6, 7]) and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);  // block 3 contains [8]
    llmRequest1->addNewToken(10, beamIdx); // block 3 contains [8, 9]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed (blocks contain [8, 9])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // reuse blocks 0, 1, 2(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8]) :: p = partial reuse
    auto inputTokens0 = std::make_shared<VecTokens>(*inputTokens);
    inputTokens0->emplace_back(9);
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), promptLen0 - 1);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // note that seq0 is holding blocks 0, 1 and 2 until releaseBlocks is called

    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // reuse blocks 0, 1 ([0, 1, 2, 3], [4, 5, 6, 7]) and get new block 4
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest1->addNewToken(10, beamIdx); // block 4 contains [8, 9, 10]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // block 2 is stored for reuse (block contains [8]). nb! Last token of last block is never stored
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block is never stored
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with less tokens
    // input tokens [0, 1, 2, 3, 4]
    auto inputLength2 = tokensPerBlock + 1;
    auto inputTokens2
        = std::make_shared<VecTokens>(VecTokens{inputTokens->begin(), inputTokens->begin() + inputLength2});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens2, samplingConfig, isStreaming);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0 ([0, 1, 2, 3]), get new block 5
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 5}));
    llmRequest2->addNewToken(5, beamIdx); // block 5 contains [4]
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with more tokens
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 0, 1, 4(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), numTokens - 1);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest3->addNewToken(11, beamIdx); // block 4 contains [8, 9, 11]
    numTokens = llmRequest3->getNumTokens(beamIdx);
    // one block used by both seq2 and seq3
    numBlocks += tc::ceilDiv(numTokens, tokensPerBlock) - 1;
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 5 is not stored since it is last block and has only one token
    blockManager.releaseBlocks(seq2, llmRequest2);
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block not stored
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with 11 tokens, then discard few tokens from request and release a shorter one
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12});
    auto inputTokens4Short = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4, samplingConfig, isStreaming);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1, 4(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8,9])
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), promptLen4 - 1);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    auto llmRequest4Short
        = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4Short, samplingConfig, isStreaming);

    // llmRequest4Short tokens [0, 1, 2, 3, 4, 5, 6, 7, 8]
    // blocks 0 and 1 ([0, 1, 2, 3], [4, 5, 6, 7]) are already stored,
    // block 4 is freed
    blockManager.releaseBlocks(seq4, llmRequest4Short);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with 11 tokens again and make sure no discarded tokens reuse happens
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    // reuse blocks 0, 1, 2(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8])
    // nb! LlmRequest retains state calculated during addSequence, this state affects result.
    // Calling addSequence a second time with same LlmRequest object will produce incorrect state.
    // Create new llmRequest4 instance to avoid this issue.
    llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4, samplingConfig, isStreaming);
    promptLen4 = llmRequest4->getNumTokens(beamIdx);
    numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), promptLen4 - 2);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with max size with incidental reuse of first token.
    // this happens because we initialize inputTokens5 with 0's.
    auto inputLength5 = blocksInPrimaryPool * tokensPerBlock - 1;
    auto inputTokens5 = std::make_shared<VecTokens>(VecTokens(inputLength5, 0));
    requestId = 5;
    auto llmRequest5 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens5, samplingConfig, isStreaming);

    numTokens = llmRequest5->getNumTokens(beamIdx);
    GenerationRequest seq5{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, all blocks need to be freed
    auto promptLen5 = llmRequest5->getNumTokens(beamIdx);
    auto numContextBlocks5 = tc::ceilDiv(promptLen5, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq5, promptLen5, numContextBlocks5, *llmRequest5);
    llmRequest5->addNewToken(0, beamIdx);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 1); // incidental reuse

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    blockManager.releaseBlocks(seq5, llmRequest5);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with min size that doesn't reuse blocks
    auto inputLength6 = 1;
    auto inputTokens6 = std::make_shared<VecTokens>(VecTokens(inputLength6, 0));
    requestId = 6;
    auto llmRequest6 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens6, samplingConfig, isStreaming);

    numTokens = llmRequest6->getNumTokens(beamIdx);
    GenerationRequest seq6{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, all blocks need to be freed
    auto promptLen6 = llmRequest6->getNumTokens(beamIdx);
    auto numContextBlocks6 = tc::ceilDiv(promptLen6, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq6, promptLen6, numContextBlocks6, *llmRequest6);
    llmRequest6->addNewToken(0, beamIdx);
    // no reuse occurs because we are unable to reuse last input token and inputLength6 == 1.
    EXPECT_EQ(llmRequest6->getContextCurrentPosition(), 0);

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - 1);

    blockManager.releaseBlocks(seq6, llmRequest6);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithExtraIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;
    auto constexpr numReturnSequences = 1;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto inputTokenExtraIds = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 3, 3, 0, 0, 0});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds, numReturnSequences);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds, numReturnSequences);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds, numReturnSequences);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds1, numReturnSequences);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [(2, 0), (3, 0), (4, 0)])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds2, numReturnSequences);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(3, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids
    auto inputTokenExtraIds3 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 4, 4, 0, 0, 0});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds3, numReturnSequences);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0, get new block 8, 9
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 8, 9}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithLoraTaskIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // loraTaskId is 0 for common cases
    LlmRequest::LoraTaskIdType loraTaskId{0};
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // get new blocks 0, 1, 2 ([0,1,2,3], [4,5,6,7], [8])
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);
    llmRequest0->addNewToken(10, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // store blocks 0, 1, 2 for reuse ([0,1,2,3], [4,5,6,7], [8,9])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and loraTaskId, then remove it
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // store block 3 for reuse ([8,9])
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    // nb! addNewToken adds new generated token, number of input tokens stay the same.
    // calling addNewToken before addSequence potentially triggers this error message:
    // Assertion failed: prepopulatedPromptLen < promptLen
    // because maximum value for prepopulatedPromptLen is number of input+output tokens - 1,
    // but promptLen is number of input tokens.
    llmRequest0->addNewToken(9, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // inputTokens1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    // reuse 0, 1, 2(p) ([0,1,2,3], [4,5,6,7], [8])
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // store block 4 for reuse ([8])
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [8, 9]). nb! Last token of last block is not stored
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(1);
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2);
    // no reuse expected. Input tokens match blocks 0 and 1, but lora task id differs.
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(9, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store blocks 5, 6, 7 for reuse ([0,1,2,3], [4,5,6,7], [8]) with loraTaskId 1
    blockManager.releaseBlocks(seq2, llmRequest2);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and more tokens
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 5, 6, 7(p) ([0,1,2,3], [4,5,6,7], [8])
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), promptLen3 - 2);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store block 7 for reuse ([8,9]) with loraTaskId 1
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 0 again but with less tokens
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(0);
    auto inputLength4 = 5;
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4});
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse blocks 0, get new block 8
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 8}));
    llmRequest4->addNewToken(5, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 8 is stored with [4] and loraTaskId 0
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // add request with same tokens as request0 but without loraTaskId
    requestId = 5;
    auto llmRequest5 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

    numTokens = llmRequest5->getNumTokens(beamIdx);
    GenerationRequest seq5{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 9, 10, 11
    auto promptLen5 = llmRequest5->getNumTokens(beamIdx);
    auto numContextBlocks5 = tc::ceilDiv(promptLen5, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq5, promptLen5, numContextBlocks5, *llmRequest5);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq5.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({9, 10, 11}));
    llmRequest5->addNewToken(9, beamIdx);
    numTokens = llmRequest5->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 9, 10, 11 are stored without loraTaskId
    blockManager.releaseBlocks(seq5, llmRequest5);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithExtraIdAndLoraTaskIdTest)
{
    // tc::Logger::getLogger()->setLevel(tc::Logger::Level::DEBUG);
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto inputTokenExtraIds = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 3, 3, 0, 0, 0});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::LoraTaskIdType loraTaskId1{1};
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds);

    GenerationRequest seq0{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)] with loraTaskId 1)
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens but different loraTaskId and then remove it
    requestId = 1;
    LlmRequest::LoraTaskIdType loraTaskId2 = static_cast<LlmRequest::LoraTaskIdType>(2);
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId2, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds);
    GenerationRequest seq1{requestId, inputLength, beamWidth, maxBlocksPerSeq};

    // no reuse, get new block 3, 4, 5
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 3, 4, 5 are stored for reuse (block 5 contains [(2, 0), (3, 0)] with loraTaskId 2)
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // reuse blocks 0, 1 and get new block 6
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0);
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 6}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 3, 4 and reuse block 5
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId2, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds1);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids and loraTaskId 1
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds2);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // no reuse, get new block 7, 8, 9
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({7, 8, 9}));
    llmRequest2->addNewToken(3, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids and loraTaskId 1
    auto inputTokenExtraIds3 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 4, 4, 0, 0, 0});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds3);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 0, get new block 10, 11
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 10, 11}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    ///////////////////////////////////////////////////////////////////////////
    // add request with partial different extra ids and loraTaskId 2
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, loraTaskId2, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false,
        false, std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5,
        std::nullopt, std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION,
        inputTokenExtraIds3);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, maxBlocksPerSeq};
    // reuse block 3, get new block 12, 13
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({3, 12, 13}));
    llmRequest4->addNewToken(3, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 3);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, KVCacheManagerPerRequestStatsTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    // Add the sequence to req0
    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest0));

    // After first addition, check allocations and reuses
    auto numBlocks = tc::ceilDiv(inputLength, tokensPerBlock);
    EXPECT_EQ(llmRequest0->getReusedBlocksPerRequest(), 0);
    EXPECT_EQ(llmRequest0->getAllocTotalBlocksPerRequest(), numBlocks);
    EXPECT_EQ(llmRequest0->getAllocNewBlocksPerRequest(), numBlocks);
    EXPECT_EQ(llmRequest0->getMissedBlocksPerRequest(), numBlocks);

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest0));

    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest1));

    auto const numSharedBlocks = inputLength / tokensPerBlock;
    EXPECT_EQ(llmRequest1->getReusedBlocksPerRequest(), numSharedBlocks);
    EXPECT_EQ(llmRequest1->getAllocTotalBlocksPerRequest(), numBlocks - numSharedBlocks);
    EXPECT_EQ(llmRequest1->getAllocNewBlocksPerRequest(), numBlocks - numSharedBlocks);
    EXPECT_EQ(llmRequest1->getMissedBlocksPerRequest(), numBlocks - numSharedBlocks);
    EXPECT_EQ(llmRequest1->getKVCacheHitRatePerRequest(),
        static_cast<float>(numSharedBlocks) / static_cast<float>(numBlocks));
}

TEST_F(KVCacheManagerTest, BlockManagerBlockPriorityTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksInPrimaryPool, blocksInSecondaryPool, maxNumSequences, stream, onboardBlocks);
    blockManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // Add a request at a high and very low priority
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, 90),
                                   KvCacheRetentionConfig::TokenRangeRetentionConfig(4, 8, 10)},
            20));
    GenerationRequest seq0{0, inputLength0, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks0 = tc::ceilDiv(inputLength0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, llmRequest0->getNumTokens(0), numContextBlocks0, *llmRequest0);

    // Add another sequence with different tokens, at a low priority
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{1, inputLength1, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks1 = tc::ceilDiv(inputLength1, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq1, llmRequest1->getNumTokens(0), numContextBlocks1, *llmRequest1);

    // Release both sequences
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseBlocks(seq1, llmRequest1);

    // Add and then release another sequence
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    llmRequest2->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 20)}, 20));
    GenerationRequest seq2{2, inputLength2, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks2 = tc::ceilDiv(inputLength2, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq2, llmRequest2->getNumTokens(0), numContextBlocks2, *llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);

    // Check that request 1 blocks were overwritten
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    GenerationRequest seq3{3, inputLength3, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks3 = tc::ceilDiv(inputLength3, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq3, llmRequest3->getNumTokens(0), numContextBlocks3, *llmRequest3);

    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 4);

    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 4);

    // Check that request 0 blocks weren't overwritten
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength4 = static_cast<SizeType32>(inputTokens4->size());
    auto llmRequest4 = std::make_shared<LlmRequest>(4, maxNewTokens, inputTokens4, samplingConfig, isStreaming);
    GenerationRequest seq4{4, inputLength3, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks4 = tc::ceilDiv(inputLength4, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq4, llmRequest4->getNumTokens(0), numContextBlocks4, *llmRequest4);

    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 4);

    // Check that request 2 block 0 has been evicted
    auto inputTokens5 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength5 = static_cast<SizeType32>(inputTokens5->size());
    auto llmRequest5 = std::make_shared<LlmRequest>(5, maxNewTokens, inputTokens5, samplingConfig, isStreaming);
    GenerationRequest seq5{5, inputLength5, beamWidth, maxBlocksPerSeq};
    auto numContextBlocks5 = tc::ceilDiv(inputLength5, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq5, llmRequest5->getNumTokens(0), numContextBlocks5, *llmRequest5);

    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);
}

TEST_F(KVCacheManagerTest, KVCacheManagerDecodeBlockPriorityTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 8;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 5)}, 90));
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);

    // 5 blocks available.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 5);

    llmRequest0->addNewToken(0, 0); // block 2 contains [8, 9, 10, 11]

    // Add a token to request 0, which occupies a new block 3.
    kvCacheManager.addToken(0);
    llmRequest0->addNewToken(0, 0); // block 3 contains [0]

    // 4 blocks left.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 4);

    // uses up 3 more blocks 4, 5, 6. [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22]
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    llmRequest1->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 90)}, 5));
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);

    // one block left.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 1);

    llmRequest1->addNewToken(0, 0); // block 6 contains [20, 21, 22, 23]
    // add another token, which occupies another new block
    kvCacheManager.addToken(1);
    llmRequest1->addNewToken(0, 0); // block 7 contains [0]

    // no block available.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 0);

    // remove both sequences, blocks get stored
    // leaf block 3 (priority 90), context blocks 2, 1, 0 (priority 5)
    kvCacheManager.removeSequence(0, llmRequest0);
    // leaf block 7 (priority 5), context blocks 6, 5, 4 (priority 90)
    kvCacheManager.removeSequence(1, llmRequest1);

    // all blocks are available again.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // no reuse, blocks are evicted by new request:
    // evict block 7 (lowest priority leaf), block 6 becomes leaf
    // evict block 3 (lowest and oldest priority leaf), block 2 becomes leaf
    // evict block 2 (lowest and oldest priority leaf), block 1 becomes leaf
    // uses up 3 blocks 7, 3, 2. [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34]
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    // leaf block 2 (priority 35), context blocks 3, 7 (priority 35)
    kvCacheManager.removeSequence(2, llmRequest2);

    // reuse blocks 0 and 1, new block 2 (lowest priority leaf)
    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);

    // Two blocks reused
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 8);
}

TEST_F(KVCacheManagerTest, KVCacheManagerTimedEvictionTest)
{

    using namespace tensorrt_llm::executor;
    using namespace std::chrono_literals;

    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 80, 10ms)}, 80));
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    llmRequest1->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 80));
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
    kvCacheManager.removeSequence(1, llmRequest1);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Manually trigger a refresh.
    kvCacheManager.refreshBlocks();

    // Clear out some of the blocks.
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    kvCacheManager.removeSequence(2, llmRequest2);

    // Check that the [12, 13, 14, 15] block is still in the cache
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);

    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);

    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 11);
}

TEST_F(KVCacheManagerTest, KVCacheManagerDecodeTimedEvictionTest)
{
    using namespace tensorrt_llm::executor;
    using namespace std::chrono_literals;

    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);
    {
        auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
        auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
        llmRequest0->setKvCacheRetentionConfig(
            KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)},
                50)); // Set all blocks to priority 50.
        kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
        kvCacheManager.storeContextBlocks(*llmRequest0);
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(0);
            llmRequest0->addNewToken(0, 0);
        }
        kvCacheManager.removeSequence(0, llmRequest0);
    }
    {
        auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
        auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
        llmRequest1->setKvCacheRetentionConfig(KvCacheRetentionConfig(
            {}, KvCacheRetentionConfig::kMaxRetentionPriority, 20ms)); // Set decode blocks to max priority for 20ms.
        kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
        kvCacheManager.storeContextBlocks(*llmRequest1);
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(1);
            llmRequest1->addNewToken(0, 0);
        }
        kvCacheManager.removeSequence(1, llmRequest1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    kvCacheManager.refreshBlocks();

    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 0, 0, 0, 0, 0, 0, 0});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    kvCacheManager.removeSequence(2, llmRequest2);

    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);

    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 8);
}

TEST_F(KVCacheManagerTest, KVCacheManagerSecondaryBlockPrimaryChildTest)
{
    // It's possible for a block in secondary memory to have a primary child. Make sure this case is handled.

    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, false, stream, true,
        onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    // get new blocks 0, 1, 2
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    kvCacheManager.removeSequence(0, llmRequest0);
    // store blocks 0, 1, 2 for reuse ([0,1,2,3], [4,5,6,7], [8,9,10])

    // Offload the last two blocks of llmRequest0 to secondary memory
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    // get blocks 3, 2, 1. This causes 2 and 1 to be offloaded to secondary
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
    kvCacheManager.removeSequence(1, llmRequest1);
    // store blocks 3, 2, 1 for reuse ([1,1,2,3], [4,5,6,7], [8,9,10])

    // Match the middle block of request 0
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    // reuse block 0
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 3);
    kvCacheManager.storeContextBlocks(*llmRequest2);

    // Add a decode block that matches the contents of seq 0 block 1, add a unique decode block
    for (int token = 4; token < 8; token++)
    {
        llmRequest2->addNewToken(token, 0);
        kvCacheManager.addToken(2);
    }
    llmRequest2->addNewToken(0, 0);
    kvCacheManager.addToken(2);

    llmRequest2->addNewToken(0, 0);
    kvCacheManager.addToken(2);

    // The middle block remains in secondary, but the third block is in primary
    kvCacheManager.removeSequence(2, llmRequest2);

    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 0, 0});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);
    // All blocks should be reused.
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 9);
}

TEST_F(KVCacheManagerTest, KVCacheManagerLeafBlockTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, false, stream, true,
        onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    llmRequest0->addNewToken(0, 0);
    kvCacheManager.addToken(0);

    // The second block allocated should be first in line for eviction.
    kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);

    GenerationRequest const& seq1 = kvCacheManager.getSequence(1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    // Block 1 should NOT be reused. It was not freed even if partial.
    EXPECT_THAT(seq1.getCacheBlockIds().at(0), ::testing::ElementsAreArray({2}));

    // Allocate the remaining 3 blocks in primary
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);

    kvCacheManager.removeSequence(1, llmRequest1);
    kvCacheManager.removeSequence(2, llmRequest2);

    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 11);

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 1);

    kvCacheManager.addToken(3);
    llmRequest3->addNewToken(0, 0);

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 0);

    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{42});
    auto const inputLength4 = static_cast<SizeType32>(inputTokens4->size());
    auto llmRequest4 = std::make_shared<LlmRequest>(4, maxNewTokens, inputTokens4, samplingConfig, isStreaming);

    EXPECT_THROW(kvCacheManager.addSequence(4, inputLength4, beamWidth, llmRequest4), std::exception);
}

TEST_P(KVCacheManagerTest, KVCacheManagerAllocationTest)
{
    using DType = half;

    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr useUvm = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    auto const granularity = tensorrt_llm::common::getAllocationGranularity();
    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow,
            temporaryAttentionWindow, sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks);

    auto const& bufferManager = kvCacheManager.getBlockManager().getBufferManager();
    auto const memoryPoolUsedBefore = bufferManager.memoryPoolUsed();
    kvCacheManager.allocatePools(dtype, useUvm);
    auto const memoryPoolUsedAfter = bufferManager.memoryPoolUsed();

    EXPECT_GT(memoryPoolUsedAfter, memoryPoolUsedBefore);
    auto const actualPoolMemoryUsageDiff = static_cast<std::size_t>(memoryPoolUsedAfter - memoryPoolUsedBefore);
    auto const expectedPoolMemoryUsageDiff = tensorrt_llm::common::roundUp(
        sizeof(DType) * static_cast<int64_t>(totalNumBlocks) * numLayers * 2 * numHeads * tokensPerBlock * sizePerHead,
        static_cast<SizeType32>(granularity));
    EXPECT_EQ(actualPoolMemoryUsageDiff, expectedPoolMemoryUsageDiff);
}

TEST_P(KVCacheManagerTest, KVCacheManagerTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    auto kvCacheBlockOffsetsRange = tensorrt_llm::runtime::BufferRange<tk::KVCacheIndex>(*kvCacheBlockOffsets);
    std::fill(kvCacheBlockOffsetsRange.begin(), kvCacheBlockOffsetsRange.end(),
        tk::KVCacheIndex{std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max()});

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice
                = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto* const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto block = 0; block < numSharedBlocks; ++block)
            {
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                }
                runningSum += offsetBetweenBlocks;
            }
            {
                auto const block = numSharedBlocks;
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    runningSum += offsetBetweenBlocks;
                }
            }
            {
                auto const block = numSharedBlocks + 1;
                auto const expected = std::numeric_limits<tk::KVCacheIndex::UnderlyingType>::max();
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), expected) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), expected) << "beam:" << beam << " block:" << block;
                }
            }
        }
    }

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        currentNumBlocks -= maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);
}

TEST_P(KVCacheManagerTest, KVCacheManagerRewindTokensTest)
{
    using DType = half;

    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow,
            temporaryAttentionWindow, sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq);
    EXPECT_NO_THROW(kvCacheManager.rewindKVCache(requestId, 4));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        currentNumBlocks -= maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.rewindKVCache(requestId, 2));
        currentNumBlocks += maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
}

TEST_P(KVCacheManagerTest, KVCacheManagerMaxAttentionWindowTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr blockLengthPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxNumTokens = tokensPerBlock * blockLengthPerSeq;

    auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
    // Enable cyclic kv cache for all new generated tokens.
    auto constexpr maxAttentionWindow = inputLength;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr numSharedBlocks = std::min(inputLength, maxAttentionWindow) / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (blockLengthPerSeq - numSharedBlocks) * maxBeamWidth;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow, tokensPerBlock);

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);

    tr::ITensor::SharedPtr const kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
    {
        auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
        auto const offsetBetweenBlocks = numLayersInPool * 2;
        tr::ITensor::SharedPtr const blockOffsetsSlice
            = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
        auto blockOffsetsShape = blockOffsetsSlice->getShape();
        auto* const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

        tk::KVCacheIndex::UnderlyingType runningSum{0};
        for (auto block = 0; block < numSharedBlocks; ++block)
        {
            for (auto beam = 0; beam < maxBeamWidth; ++beam)
            {
                auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
            }
            runningSum += offsetBetweenBlocks;
        }
        {
            auto const block = numSharedBlocks;
            for (auto beam = 0; beam < maxBeamWidth; ++beam)
            {
                auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                runningSum += offsetBetweenBlocks;
            }
        }
    }

    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq + 1);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numBlocksPerSeq + 1);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    EXPECT_EQ(blockManager.getNumFreeBlocks(), maxNumSequences);
}

TEST_F(KVCacheManagerTest, KVCacheManagerMaxAttentionWindowWithReuseTest)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    // Enable cyclic kv cache for long input tokens.
    auto constexpr maxAttentionWindow = 16;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow, tokensPerBlock);

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = true;
    auto constexpr onboardBlocks = true;

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow,
        sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    SizeType32 constexpr maxNewTokens = 4;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    SizeType32 requestId = 0;
    int inputLength = 16;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    ///////////////////////////////////////////////////////////////////////////
    // add a long request and then remove it
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3}));

    // add tokens to enable cyclic kv cache
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1017, beamIdx);
    kvCacheManager.addToken(requestId);
    auto numTokens = llmRequest->getNumTokens(beamIdx);
    auto numBlocks = seq0.getCacheBlockIds()[beamIdx].size();
    EXPECT_EQ(numBlocks, maxBlocksPerSeq);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    // no blocks stored because cyclic KV cache was enabled
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add a short request and then remove it
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5}));

    llmRequest->addNewToken(1007, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    numTokens = llmRequest->getNumTokens(beamIdx);
    numBlocks = seq1.getCacheBlockIds()[beamIdx].size();
    EXPECT_EQ(numBlocks, 3);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    // store blocks 4, 5 for reuse ([1000,1001,1002,1003], [1004,1005,1006,1007])

    ///////////////////////////////////////////////////////////////////////////
    // add a medium request and then remove it
    // reuse first 2 blocks {4, 5} in previous request, and get new block 7
    requestId = 2;
    inputLength = 10;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq2 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq2.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5, 7}));

    numTokens = llmRequest->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    // store block 7 for reuse ([1008])

    ///////////////////////////////////////////////////////////////////////////
    // add a longer request within attention window and try to reuse
    // reuse blocks 4, 5, 7(p) and get new block 8
    // upon reached attention window, shared block 4 is replaced with unshared block 9
    requestId = 3;
    inputLength = 15;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq3 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 9);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({4, 5, 7, 8}));

    llmRequest->addNewToken(1015, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    EXPECT_THAT(seq3.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({9, 5, 7, 8}));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a long request that exceeded attention window, no reuse
    requestId = 4;
    inputLength = 20;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    GenerationRequest const& seq4 = kvCacheManager.getSequence(requestId);
    EXPECT_THAT(seq4.getCacheBlockIds().at(beamIdx), ::testing::ElementsAreArray({10, 11, 12, 13}));
}

namespace
{
KVCacheManager setupKvCacheManagerForHashTest(bool enableBlockReuse)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr maxBlocksPerSeq = 8;
    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr temporaryAttentionWindow = 0;

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr onboardBlocks = true;

    return {std::vector<SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow,
        sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks, CacheType::kSELF, std::nullopt, nullptr,
        /*enableHashKey*/ true};
}

std::vector<size_t> getHashAndRetrieveBlocksByHashTest(
    BlockManager const& blockManager, std::vector<KVCacheBlock::IdType> const& blockIds)
{
    std::vector<size_t> blockHashes;
    for (auto blockId : blockIds)
    {
        blockHashes.emplace_back(blockManager.getBlockById(blockId)->getHash());
    }
    std::vector<BlockPtr> blockPtrs;
    for (auto hash : blockHashes)
    {
        auto range = blockManager.getBlocksByHash(hash);
        BlockPtr const prevBlock = blockPtrs.empty() ? nullptr : blockPtrs.back();
        BlockPtr thisBlock = nullptr;
        for (auto it = range.first; it != range.second; ++it)
        {
            if (it->second->getPrevBlockInSeq() == prevBlock)
            {
                thisBlock = it->second;
                break;
            }
        }
        EXPECT_NE(thisBlock, nullptr);
        blockPtrs.emplace_back(thisBlock);
    }
    EXPECT_EQ(blockHashes.size(), blockPtrs.size());
    for (size_t i = 0; i < blockHashes.size(); i++)
    {
        EXPECT_EQ(blockManager.getBlockById(blockIds[i]), blockPtrs[i]);
    }
    return blockHashes;
}
} // namespace

TEST_F(KVCacheManagerTest, KVCacheManagerHashKeyTest)
{
    auto kvCacheManager = setupKvCacheManagerForHashTest(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    SizeType32 constexpr maxNewTokens = 4;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    SizeType32 requestId = 0;
    int inputLength = 16;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    ///////////////////////////////////////////////////////////////////////////
    // add a request and then remove it without reuse
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    auto& blockIds = seq.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(blockIds, ::testing::ElementsAreArray({0, 1, 2, 3}));

    // get blocks by hash and try to retrieve them by hash
    auto blockHashes = getHashAndRetrieveBlocksByHashTest(blockManager, blockIds);

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    // blocks are all removed
    for (auto hash : blockHashes)
    {
        auto range = blockManager.getBlocksByHash(hash);
        EXPECT_EQ(range.first, range.second);
    }
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
}

TEST_F(KVCacheManagerTest, KVCacheManagerHashKeyWithReuseTest)
{
    auto kvCacheManager = setupKvCacheManagerForHashTest(true);

    auto const& blockManager = kvCacheManager.getBlockManager();

    SizeType32 constexpr maxNewTokens = 4;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = 1;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    SizeType32 requestId = 0;
    int inputLength = 16;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    ///////////////////////////////////////////////////////////////////////////
    // add a request and then remove it with reuse
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    auto& blockIds0 = seq0.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(blockIds0, ::testing::ElementsAreArray({0, 1, 2, 3}));

    // get blocks by hash and try to retrieve them by hash
    auto blockHashes = getHashAndRetrieveBlocksByHashTest(blockManager, blockIds0);

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    // store 4 blocks with total of 15 reusable tokens (last token is not stored).

    // TODO: Make reused blocks accessible by hash, after sequence removed. Test here.

    ///////////////////////////////////////////////////////////////////////////
    // add a new request with same prefix
    requestId = 1;
    inputLength = 20;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 15);
    auto& blockIds1 = seq1.getCacheBlockIds().at(beamIdx);
    EXPECT_THAT(blockIds1, ::testing::ElementsAreArray({0, 1, 2, 3, 4}));

    std::ignore = getHashAndRetrieveBlocksByHashTest(blockManager, blockIds1);

    // blocks are reused, so reused blocks are still accessible by previous hashes
    for (size_t i = 0; i < 4; i++)
    {
        auto range = blockManager.getBlocksByHash(blockHashes[i]);
        EXPECT_NE(range.first, range.second);
    }
    // evicted block is not accessible
    {
        size_t i = 4;
        auto range = blockManager.getBlocksByHash(blockHashes[i]);
        EXPECT_EQ(range.first, range.second);
    }
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 5);
}

TEST_F(KVCacheManagerTest, KVCacheManagerEventStream)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 2;
    auto constexpr onboardBlocks = true;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(dtype, false);

    auto events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheCreatedData>(events.front().data));
    EXPECT_THAT(std::get<tle::KVCacheCreatedData>(events.front().data).numBlocksPerCacheLevel,
        ::testing::ElementsAreArray({8, 2}));

    EXPECT_EQ(getEvents(kvCacheManager).size(), 0);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setLoraTaskId(42);
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
    EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).parentHash, std::nullopt);
    EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).blocks.size(), 2);
    EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).blocks[0].loraId, 42);
    EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).blocks[0].cacheLevel, 0);
    kvCacheManager.addToken(0);
    llmRequest0->addNewToken(0, 0);
    kvCacheManager.removeSequence(0, llmRequest0);

    auto newEvents = getEvents(kvCacheManager);
    EXPECT_EQ(newEvents.size(), 1);
    EXPECT_EQ(newEvents.front().eventId, events.front().eventId + 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(newEvents.front().data));
    // Check that it provides the link to the parent, and that it only stores the one block.
    auto storedEventData = std::get<tle::KVCacheStoredData>(newEvents.front().data);
    EXPECT_EQ(
        storedEventData.parentHash, std::get<tle::KVCacheStoredData>(events.front().data).blocks.back().blockHash);
    EXPECT_EQ(storedEventData.blocks.size(), 1);

    // Store with the same tokens
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto llmRequest1 = std::make_shared<LlmRequest>(1, 0, inputTokens1, samplingConfig, true);
    llmRequest1->setLoraTaskId(42);
    kvCacheManager.addSequence(1, inputTokens1->size(), beamWidth, llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    kvCacheManager.removeSequence(1, llmRequest1);

    events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 0);

    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto llmRequest2 = std::make_shared<LlmRequest>(2, 0, inputTokens2, samplingConfig, true);
    kvCacheManager.addSequence(2, inputTokens2->size(), beamWidth, llmRequest2);

    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto llmRequest3 = std::make_shared<LlmRequest>(3, 0, inputTokens3, samplingConfig, true);
    kvCacheManager.addSequence(3, inputTokens3->size(), beamWidth, llmRequest3);

    events = getEvents(kvCacheManager);
    size_t firstSwapped = std::get<tle::KVCacheUpdatedData>(events.front().data).blockHash;
    // 3 blocks swapped to secondary
    EXPECT_EQ(events.size(), 4);
    for (int i = 2; i >= 0; i--)
    {
        EXPECT_TRUE(std::holds_alternative<tle::KVCacheUpdatedData>(events.front().data));
        EXPECT_EQ(std::get<tle::KVCacheUpdatedData>(events.front().data).cacheLevel->oldValue, 0);
        EXPECT_EQ(std::get<tle::KVCacheUpdatedData>(events.front().data).cacheLevel->newValue, 1);
        events.pop_front();
    }
    // The first block swapped to secondary gets evicted.
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheRemovedData>(events.front().data));
    EXPECT_THAT(std::get<tle::KVCacheRemovedData>(events.front().data).blockHashes,
        ::testing::ElementsAreArray({firstSwapped}));

    kvCacheManager.removeSequence(2, llmRequest2);
    kvCacheManager.removeSequence(3, llmRequest3);

    events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 2);
    for (int i = 0; i < 2; i++)
    {
        EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
        EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).parentHash, std::nullopt);
        EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).blocks.size(), 3);
        events.pop_front();
    }

    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 1, 1, 1, 1, 0});
    auto llmRequest4 = std::make_shared<LlmRequest>(4, 0, inputTokens4, samplingConfig, true);
    llmRequest4->setLoraTaskId(42);
    kvCacheManager.addSequence(4, inputTokens4->size(), beamWidth, llmRequest4);

    events = getEvents(kvCacheManager);

    // Offload 1 block, onboard 1 block, remove 1 secondary block. The second new block allocated was never stored, so
    // it doesn't create a removed event
    auto onboardedBlocks = 0;
    auto offloadedBlocks = 0;
    auto removedBlocks = 0;

    EXPECT_EQ(events.size(), 3);

    for (int i = 0; i < 3; i++)
    {
        if (std::holds_alternative<tle::KVCacheUpdatedData>(events.front().data))
        {
            if (std::get<tle::KVCacheUpdatedData>(events.front().data).cacheLevel->oldValue == 0)
                offloadedBlocks++;
            else
                onboardedBlocks++;
        }
        else if (std::holds_alternative<tle::KVCacheRemovedData>(events.front().data))
            removedBlocks++;
        else
            FAIL();
        events.pop_front();
    }

    EXPECT_EQ(onboardedBlocks, 1);
    EXPECT_EQ(offloadedBlocks, 1);
    EXPECT_EQ(removedBlocks, 1);

    kvCacheManager.storeContextBlocks(*llmRequest4);

    events = getEvents(kvCacheManager);

    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
}

TEST_F(KVCacheManagerTest, KVCacheManagerEventStreamOverflow)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 2;
    auto constexpr onboardBlocks = true;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1));
    kvCacheManager.allocatePools(dtype, false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setLoraTaskId(42);
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    auto events = getEvents(kvCacheManager);

    // The 'created' event should be evicted.
    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
    EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).parentHash, std::nullopt);

    kvCacheManager.addToken(0);
    llmRequest0->addNewToken(0, 0);

    events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 0);

    kvCacheManager.removeSequence(0, llmRequest0);

    events = getEvents(kvCacheManager);

    // The decode block is stored, and linked to the parent.
    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
    EXPECT_NE(std::get<tle::KVCacheStoredData>(events.front().data).parentHash, std::nullopt);
}

TEST_F(KVCacheManagerTest, KVCacheManagerEventStreamPriority)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 2;
    auto constexpr onboardBlocks = true;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(dtype, false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setKvCacheRetentionConfig(tle::KvCacheRetentionConfig(
        std::vector{tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 35));
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    kvCacheManager.removeSequence(0, llmRequest0);
    auto events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 3);
    events.pop_front(); // The `CREATED` event

    for (int i = 0; i < 2; i++)
    {
        EXPECT_EQ(std::get<tle::KVCacheStoredData>(events.front().data).blocks.front().priority, 50);
        events.pop_front();
    }

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest1 = std::make_shared<LlmRequest>(1, 0, inputTokens1, samplingConfig, true);
    kvCacheManager.addSequence(1, inputTokens1->size(), beamWidth, llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    kvCacheManager.removeSequence(1, llmRequest1);

    events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 1); // The second partial block gets stored. No priorities updated.
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));

    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest2 = std::make_shared<LlmRequest>(2, 0, inputTokens2, samplingConfig, true);
    llmRequest2->setKvCacheRetentionConfig(tle::KvCacheRetentionConfig(
        std::vector{tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 10)},
        35)); // Update the context block priorities
    kvCacheManager.addSequence(2, inputTokens2->size(), beamWidth, llmRequest2);

    events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 2);
    for (int i = 0; i < 2; i++)
    {
        auto diff = std::get<tle::KVCacheUpdatedData>(events.front().data).priority;
        EXPECT_EQ(diff->oldValue, 50);
        EXPECT_EQ(diff->newValue, 10);
        events.pop_front();
    }
}

TEST_F(KVCacheManagerTest, KVCacheManagerEventStreamBlocking)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 2;
    auto constexpr onboardBlocks = true;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    KVCacheManager kvCacheManagerTest(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks, CacheType::kSELF, std::nullopt);

    EXPECT_EQ(getEvents(kvCacheManagerTest).size(), 0);

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, beamWidth, tokensPerBlock * maxBlocksPerSeq, 0, 0, stream, std::nullopt,
        true, onboardBlocks, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));

    kvCacheManager.allocatePools(dtype, false);
    kvCacheManager.flushIterationEvents();
    auto events = kvCacheManager.getLatestEvents(std::chrono::seconds(1));

    EXPECT_EQ(events.size(), 1);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    kvCacheManager.flushIterationEvents();
    events = kvCacheManager.getLatestEvents(std::chrono::seconds(1));

    EXPECT_EQ(events.size(), 1);

    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
}

TEST_F(KVCacheManagerTest, KVCacheTransferManagerConcurrencyTest)
{
    auto const blockSize = 16384;

    auto bufferManager = tensorrt_llm::runtime::BufferManager(std::make_shared<tr::CudaStream>());
    auto transferManager = KVCacheTransferManager(bufferManager);

    auto pool = KVCacheBlockPool(0, 0, 0, 0, 1);

    pool.primaryPtr = bufferManager.gpu(tr::ITensor::makeShape({1, blockSize}), nvinfer1::DataType::kFLOAT);
    bufferManager.setZero(*pool.primaryPtr);

    pool.secondaryPtr = tr::BufferManager::pinned(tr::ITensor::makeShape({1, blockSize}), nvinfer1::DataType::kFLOAT);

    // Write some specific data into the cpu blocks.
    for (int i = 0; i < blockSize; i++)
    {
        tr::bufferCast<float>(*pool.secondaryPtr)[i] = 1;
    }

    auto primaryBlock = std::make_shared<KVCacheBlock>(0, tensorrt_llm::kernels::KVCacheIndex(0, false));
    auto secondaryBlock = std::make_shared<KVCacheBlock>(1, tensorrt_llm::kernels::KVCacheIndex(0, true));

    transferManager.offload(primaryBlock, secondaryBlock, {pool});
    primaryBlock->swapMemoryPoolBlockOffset(secondaryBlock);
    transferManager.onboard(primaryBlock, secondaryBlock, {pool});
    transferManager.syncTransfers();

    transferManager.offload(primaryBlock, secondaryBlock, {pool});
    bufferManager.getStream().synchronize();

    for (int i = 0; i < blockSize; i++)
    {
        EXPECT_EQ(tr::bufferCast<float>(*pool.secondaryPtr)[i], 0);
    }
}

TEST_P(KVCacheManagerTest, KVCacheManagerSinkTokenLengthTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 4;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = static_cast<RequestIdType>(7);
    auto constexpr sinkTokensInLastBlock = sinkTokenLength % tokensPerBlock;
    auto constexpr bubbleLength = (sinkTokensInLastBlock) ? tokensPerBlock - sinkTokensInLastBlock : 0;
    auto constexpr inputLength = tokensPerBlock * maxBlocksPerSeq - bubbleLength - 1;
    auto constexpr maxAttentionWindow = inputLength - tokensPerBlock;
    auto constexpr temporaryAttentionWindow = 0;

    auto constexpr numSharedBlocks = (sinkTokenLength + bubbleLength) / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;
    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr numSharedBlocksCtx = (inputLength + bubbleLength) / tokensPerBlock;

    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            maxSequenceLength, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth);

    tr::ITensor::SharedPtr kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, requestId);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr blockOffsetsSlice
                = tr::ITensor::slice(tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto block = 0; block < numSharedBlocksCtx; ++block)
            {
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                }
                runningSum += offsetBetweenBlocks;
            }
            {
                auto const block = numSharedBlocksCtx;
                for (auto beam = 0; beam < maxBeamWidth; ++beam)
                {
                    auto const kOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 0, block);
                    auto const vOffsetIdx = tc::flat_index(blockOffsetsShape.d, beam, 1, block);
                    auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                    auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                    EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                    ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    runningSum += offsetBetweenBlocks;
                }
            }
        }
    }

    // replace the shared block with unshared blocks
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth * 2 + 1);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocksCtx - maxBeamWidth * 2 + 1);
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestCounter = 0; requestCounter < maxNumSequences; ++requestCounter)
    {
        auto const nextRequestId = static_cast<RequestIdType>(requestId + requestCounter);
        EXPECT_NO_THROW(kvCacheManager.addSequence(nextRequestId, inputLength, maxBeamWidth));
        currentNumBlocks -= numSharedBlocksCtx + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(nextRequestId));
        currentNumBlocks -= maxBeamWidth - 1;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    auto numUsedBlocks = maxNumSequences * (numSharedBlocksCtx + maxBeamWidth * 2 - 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numUsedBlocks);
}

TEST_P(KVCacheManagerTest, KVCacheManagerBatchTest)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    std::map<SizeType32, SizeType32> const expectedHeadsPerPool({{0, 1}, {1, 2}, {2, 3}});
    std::map<SizeType32, SizeType32> const expectedLayersPerPool({{0, 1}, {1, 2}, {2, 1}});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 64;
    auto constexpr maxBlocksPerSeq = 32;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 4;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxNumTokens;
    auto constexpr temporaryAttentionWindow = 0;
    auto constexpr inputLength = maxNumTokens - 2;
    auto constexpr numBlocksPerSeq = maxBlocksPerSeq - 1 + maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
            maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

    EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
        auto const currentNumBlocks = totalNumBlocks - (requestId + 1) * numBlocksPerSeq;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }

    tr::ITensor::SharedPtr const kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({expectedNumPools, maxNumSequences * maxBeamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);

    kvCacheManager.getBlockOffsetsOfBatch(*kvCacheBlockOffsets, 0, maxNumSequences, maxBeamWidth);

    EXPECT_EQ(blockManager.getNumPools(), expectedNumPools);
    if (homogeneousLayers)
    {
        EXPECT_EQ(blockManager.getBlockSize(0), numHeads * sizePerHead * tokensPerBlock);
    }
    else
    {
        for (auto layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {

            EXPECT_EQ(blockManager.getBlockSize(blockManager.getLayerPoolIdx(layerIdx)),
                numHeadsPerLayer.at(layerIdx) * sizePerHead * tokensPerBlock);
        }
    }

    {
        for (auto poolIdx = 0; poolIdx < blockManager.getNumPools(); poolIdx++)
        {
            auto const numLayersInPool = homogeneousLayers ? numLayers : expectedLayersPerPool.at(poolIdx);
            auto const offsetBetweenBlocks = numLayersInPool * 2;
            tr::ITensor::SharedPtr const blockOffsetsSlice = tr::ITensor::slice(
                tr::ITensor::at(kvCacheBlockOffsets, {poolIdx}), 0, maxNumSequences * maxBeamWidth);
            auto blockOffsetsShape = blockOffsetsSlice->getShape();
            auto* const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

            tk::KVCacheIndex::UnderlyingType runningSum{0};
            for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
            {
                for (auto block = 0; block < maxBlocksPerSeq - 1; ++block)
                {
                    for (auto beam = 0; beam < maxBeamWidth; ++beam)
                    {
                        auto const kOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 0, block);
                        auto const vOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 1, block);
                        auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                        auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                        EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                        ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                    }
                    runningSum += offsetBetweenBlocks;
                }
                auto const block = maxBlocksPerSeq - 1;
                {
                    for (auto beam = 0; beam < maxBeamWidth; ++beam)
                    {
                        auto const kOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 0, block);
                        auto const vOffsetIdx
                            = tc::flat_index(blockOffsetsShape.d, requestId * maxBeamWidth + beam, 1, block);
                        auto const kOffset = blockOffsetsPtr[kOffsetIdx];
                        auto const vOffset = blockOffsetsPtr[vOffsetIdx];
                        EXPECT_EQ(kOffset.get(), runningSum) << "beam:" << beam << " block:" << block;
                        ASSERT_EQ(vOffset.get(), runningSum + 1) << "beam:" << beam << " block:" << block;
                        runningSum += offsetBetweenBlocks;
                    }
                }
            }
        }
    }
}

namespace
{
void testNeededBlocksOneStep(bool kv_cache_block_reuse, int beamWidth, int draftLen, bool homogeneousLayers)
{
    using DType = half;
    using SizeType32 = KVCacheManager::SizeType32;

    auto constexpr numLayers = 4;
    auto constexpr numHeads = 2;
    // heterogeneous layers
    std::vector<SizeType32> const numHeadsPerLayer({3, 2, 1, 2});
    auto constexpr sizePerHead = 64;
    auto constexpr hiddenSize = numHeads * sizePerHead;
    auto constexpr tokensPerBlock = 8;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;

    TLLM_CHECK(draftLen == 0 || beamWidth == 1);

    // Deal with one sequence for now
    auto constexpr requestId = static_cast<RequestIdType>(7);
    SizeType32 maxNewTokens = 20;
    bool isStreaming = false;

    SizeType32 const maxInputLength{65};
    SizeType32 const maxMaxBeamWidth{beamWidth};

    for (int maxBeamWidth = 1; maxBeamWidth <= maxMaxBeamWidth; ++maxBeamWidth)
    {
        tr::SamplingConfig const samplingConfig{maxBeamWidth};
        for (int inputLength = 1; inputLength < maxInputLength; ++inputLength)
        {
            auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
            // auto constexpr maxAttentionWindow = maxNumTokens / 2;
            auto constexpr maxAttentionWindow = 46;
            auto constexpr temporaryAttentionWindow = 0;
            auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;
            auto constexpr blocksInSecondaryPool = 0;
            auto constexpr onboardBlocks = true;

            KVCacheManager kvCacheManager = homogeneousLayers
                ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
                    blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow,
                    sinkTokenLength, stream, std::nullopt, kv_cache_block_reuse, onboardBlocks)
                : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, totalNumBlocks, blocksInSecondaryPool,
                    maxNumSequences, maxBeamWidth, maxAttentionWindow, temporaryAttentionWindow, sinkTokenLength,
                    stream, std::nullopt, kv_cache_block_reuse, onboardBlocks);
            kvCacheManager.allocatePools(nvinfer1::DataType::kHALF, false);

            EXPECT_EQ(kvCacheManager.getMaxBlocksPerSeq(), tc::ceilDiv(maxAttentionWindow, tokensPerBlock));

            auto inputTokens = std::make_shared<VecTokens>(VecTokens(inputLength, 0));

            auto draftTokens = std::make_shared<std::vector<SizeType32>>(draftLen);
            auto llmRequest
                = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
            llmRequest->setDraftTokens(draftTokens);

            auto remainingBlocksToCompletion = kvCacheManager.getRemainingBlocksToCompletion(*llmRequest);
            auto neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false);

            EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth, llmRequest));
            for (int di = 0; di < draftLen && di < maxNewTokens && (inputLength + di) < maxAttentionWindow; ++di)
            {
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }
                EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
            }

            auto numUsedBlocksThisStep = kvCacheManager.getUsedNumBlocks();
            EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);

            // Simulate adding new tokens during generation
            llmRequest->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
            for (int i = draftLen; i < maxNewTokens && (inputLength + i) < maxAttentionWindow; i += (draftLen + 1))
            {
                auto numCurrentlyUsedBlocks = kvCacheManager.getUsedNumBlocks();
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }

                neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false);

                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    for (int di = 0;
                         di < draftLen && (i + di) < maxNewTokens && (inputLength + i + di) < maxAttentionWindow; ++di)
                    {
                        llmRequest->addNewToken(1, beam);
                    }
                }

                for (int di = 0;
                     di < draftLen + 1 && (i + di) < maxNewTokens && (inputLength + i + di) < maxAttentionWindow; ++di)
                {
                    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
                }
                numUsedBlocksThisStep = kvCacheManager.getUsedNumBlocks() - numCurrentlyUsedBlocks;

                EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);
            }

            // After adding all tokens, we should match remainingBlocksToCompletion
            EXPECT_EQ(remainingBlocksToCompletion, kvCacheManager.getUsedNumBlocks());
            EXPECT_EQ(kvCacheManager.getRemainingBlocksToCompletion(*llmRequest), 0);
        }
    }
}
} // namespace

TEST_P(KVCacheManagerTest, neededBlocksOneStepKvCacheBlockReuse)
{
    testNeededBlocksOneStep(true, 1, 0, GetParam()); // maxBeamWidth is 1 when kv cache reuse is enabled
}

TEST_P(KVCacheManagerTest, neededBlocksOneStep)
{
    testNeededBlocksOneStep(false, 4, 0, GetParam());
}

TEST_P(KVCacheManagerTest, neededBlocksOneStepKvCacheBlockReuseDraftTokens)
{
    testNeededBlocksOneStep(true, 1, 5, GetParam());
}

INSTANTIATE_TEST_SUITE_P(KVCacheManagerTest, KVCacheManagerTest, testing::Values(true, false), // homogeneousLayers
    generateTestName);

// calculateMaxBlockRequirementsPerBeam
using BlockRequirementsParamType = std::tuple<SizeType32, SizeType32, SizeType32, SizeType32, SizeType32>;

class BlockRequirementsParamTest : public ::testing::TestWithParam<BlockRequirementsParamType>
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_P(BlockRequirementsParamTest, TestCaculateMaxBlocksRequirement)
{
    BlockRequirementsParamType param = GetParam();
    auto const result = KVCacheManager::calculateMaxBlockRequirementsPerBeam(
        std::get<0>(param), std::get<1>(param), std::get<2>(param), std::get<3>(param));
    ASSERT_EQ(result, std::get<4>(param));
}

INSTANTIATE_TEST_SUITE_P(CalculateMaxBlockRequirementsPerBeam, BlockRequirementsParamTest,
    testing::Values(std::make_tuple(512, 0, 1024, 64, 8), std::make_tuple(513, 0, 1024, 64, 9),
        std::make_tuple(512, 0, 256, 64, 4), std::make_tuple(512, 0, 257, 64, 5), std::make_tuple(512, 64, 1024, 64, 8),
        std::make_tuple(513, 64, 1024, 64, 9), std::make_tuple(512, 64, 256, 64, 4),
        std::make_tuple(512, 64, 257, 64, 5), std::make_tuple(512, 65, 1024, 64, 9),
        std::make_tuple(513, 65, 1024, 64, 9), std::make_tuple(512, 65, 256, 64, 5),
        std::make_tuple(512, 65, 257, 64, 5)));

// calculateMaxBlockRequirements
TEST(CalculateMaxBlockRequirements, BeamWidthOneEqualRequirementsPerBeam)
{
    auto const perBeamResult = KVCacheManager::calculateMaxBlockRequirementsPerBeam(512, 0, 2048, 64);
    auto const result = KVCacheManager::calculateMaxBlockRequirements(257, 255, 0, 2048, 1, 64);
    ASSERT_EQ(result, perBeamResult);
}

TEST(CalculateMaxBlockRequirements, AttentionWindowFitsOutputEqualSingleBeamTimesBeamWidth)
{
    auto constexpr beamWidth = 4;
    auto constexpr attentionWindow = 128;
    auto constexpr outputLength = 255;
    auto const perBeamResult = KVCacheManager::calculateMaxBlockRequirementsPerBeam(512, 0, attentionWindow, 64);
    auto const result = KVCacheManager::calculateMaxBlockRequirements(257, 255, 0, attentionWindow, beamWidth, 64);
    ASSERT_EQ(result, perBeamResult * beamWidth);
}

TEST(CalculateMaxBlockRequirements, AttentionWindowOverlapsInputAndOutputReferenceResult)
{
    auto constexpr beamWidth = 4;
    auto constexpr attentionWindow = 412;
    auto constexpr outputLength = 255;
    auto const perBeamResult = KVCacheManager::calculateMaxBlockRequirementsPerBeam(512, 0, attentionWindow, 64);
    auto const result = KVCacheManager::calculateMaxBlockRequirements(257, 255, 0, attentionWindow, beamWidth, 64);
    auto const numContextBlocks = 2; // (412 - 255) / 64
    // There are 29 context tokens left over to be put in output blocks, so 284 tokens to fit in output blocks in total:
    // 5 blocks
    auto const numOutputBlocks = 5 * beamWidth;
    ASSERT_EQ(result, numContextBlocks + numOutputBlocks);
}

// calculateMaxAttentionWindow
TEST(CalculateMaxAttentionWindow, OutputTooLargeForCapacity)
{
    auto constexpr beamWidth = 4;
    auto constexpr blockCapacity = 12;
    auto constexpr outputLength = 255;
    auto const result
        = KVCacheManager::calculateMaxAttentionWindow(1024, outputLength, 0, blockCapacity, beamWidth, 64);
    ASSERT_EQ(result, 192);
}

TEST(CalculateMaxAttentionWindow, CapacityIsEnoughForWholeSequence)
{
    auto constexpr beamWidth = 4;
    auto constexpr blockCapacity = 256;
    auto constexpr outputLength = 1024;
    auto constexpr inputLength = 1024;
    auto const result
        = KVCacheManager::calculateMaxAttentionWindow(inputLength, outputLength, 0, blockCapacity, beamWidth, 64);
    ASSERT_EQ(result, inputLength + outputLength);
}

namespace
{
struct KvCacheManagerInstantiationParameters
{
    SizeType32 numLayers;
    std::variant<SizeType32, std::vector<SizeType32>> numHeadsPerLayer;
    SizeType32 sizePerHead;
    SizeType32 tokensPerBlock;
    SizeType32 numBlocksInPrimaryPool;
    SizeType32 sinkTokenLength;
    SizeType32 maxAttentionWindow;
    SizeType32 maxBeamWidth;
    SizeType32 maxNumTokens;
    bool kvCacheBlockReuse;
};

struct GetRemainingBlocksToCompletionOneRequestParameters
{
    KvCacheManagerInstantiationParameters kvCacheManagerInstantiationParameters;
    SizeType32 promptLength;
    SizeType32 maxOutputLength;
    SizeType32 expectedRemainingBlocksToCompletion;
};

struct FillKvCacheAndCompleteRequestsParameters
{
    KvCacheManagerInstantiationParameters kvCacheManagerInstantiationParameters;
    SizeType32 promptLength;
    SizeType32 maxOutputLength;
};

std::shared_ptr<KVCacheManager> createKvCacheManager(
    KvCacheManagerInstantiationParameters const& kvCacheInstantiationParameters, StreamPtr stream)
{
    auto const maxInputLength = kvCacheInstantiationParameters.maxNumTokens - 1;
    auto const temporaryKvCacheLength = std::min(kvCacheInstantiationParameters.maxNumTokens,
        maxInputLength - kvCacheInstantiationParameters.maxAttentionWindow);

    if (std::holds_alternative<SizeType32>(kvCacheInstantiationParameters.numHeadsPerLayer))
    {
        auto const numHeadsPerLayer = std::get<SizeType32>(kvCacheInstantiationParameters.numHeadsPerLayer);
        auto numHeadsPerLayerVec = std::vector<SizeType32>{kvCacheInstantiationParameters.numLayers};
        std::fill(numHeadsPerLayerVec.begin(), numHeadsPerLayerVec.end(), numHeadsPerLayer);
        return std::make_shared<KVCacheManager>(numHeadsPerLayerVec, kvCacheInstantiationParameters.sizePerHead,
            kvCacheInstantiationParameters.tokensPerBlock, kvCacheInstantiationParameters.numBlocksInPrimaryPool, 0,
            kvCacheInstantiationParameters.numBlocksInPrimaryPool, kvCacheInstantiationParameters.maxBeamWidth,
            kvCacheInstantiationParameters.maxAttentionWindow, temporaryKvCacheLength,
            kvCacheInstantiationParameters.sinkTokenLength, stream, std::nullopt,
            kvCacheInstantiationParameters.kvCacheBlockReuse, true);
    }
    if (std::holds_alternative<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer))
    {
        auto const numHeadsPerLayer
            = std::get<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer);
        return std::make_shared<KVCacheManager>(numHeadsPerLayer, kvCacheInstantiationParameters.sizePerHead,
            kvCacheInstantiationParameters.tokensPerBlock, kvCacheInstantiationParameters.numBlocksInPrimaryPool, 0,
            kvCacheInstantiationParameters.numBlocksInPrimaryPool, kvCacheInstantiationParameters.maxBeamWidth,
            kvCacheInstantiationParameters.maxAttentionWindow, temporaryKvCacheLength,
            kvCacheInstantiationParameters.sinkTokenLength, stream, std::nullopt,
            kvCacheInstantiationParameters.kvCacheBlockReuse, true);
    }
    TLLM_THROW("Unhandled type of num heads per layer provided.");
}

std::vector<LlmRequest> fillKvCacheManager(KVCacheManager& kvCacheManager, SizeType32 promptLength,
    SizeType32 maxOutputLength, SizeType32 maxBeamWidth, RequestIdType requestIdStart)
{
    auto inputTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    auto const llmRequest = LlmRequest{
        0,
        maxOutputLength,
        inputTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // Adding enough requests to fill the kv-cache.
    auto remainingFreeBlocks = kvCacheManager.getNumFreeBlocks();
    auto llmRequests = std::vector<LlmRequest>{};
    auto const remainingBlocksToCompletionFromStart = kvCacheManager.getRemainingBlocksToCompletion(llmRequest);
    do
    {
        ++requestIdStart;
        std::fill(inputTokens->begin(), inputTokens->end(), requestIdStart);
        llmRequests.emplace_back(
            requestIdStart, maxOutputLength, inputTokens, tensorrt_llm::runtime::SamplingConfig{maxBeamWidth}, true);
        remainingFreeBlocks -= remainingBlocksToCompletionFromStart;
    } while (remainingFreeBlocks >= remainingBlocksToCompletionFromStart);
    for (auto request : llmRequests)
    {
        kvCacheManager.addSequence(request.mRequestId, request.getPromptLen(), maxBeamWidth, request);
        request.mState = LlmRequestState::kGENERATION_IN_PROGRESS;
        kvCacheManager.storeContextBlocks(request);
    }

    kvCacheManager.refreshBlocks();
    return llmRequests;
}
} // namespace

class RemainingBlocksToCompletionTest
    : public ::testing::TestWithParam<GetRemainingBlocksToCompletionOneRequestParameters>
{
protected:
    void SetUp() override
    {
        auto const stream = std::make_shared<tr::CudaStream>();
        auto const params = GetParam();
        kvCacheManager = createKvCacheManager(params.kvCacheManagerInstantiationParameters, stream);
        kvCacheManager->allocatePools(nvinfer1::DataType::kFLOAT);
    }

    void TearDown() override {}

    std::shared_ptr<KVCacheManager> kvCacheManager;
};

TEST_P(RemainingBlocksToCompletionTest, RemainingBlocksToCompletionCorrectlyEstimated)
{
    auto const params = GetParam();
    auto const inputTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(params.promptLength));
    auto const llmRequest = LlmRequest{
        0,
        params.maxOutputLength,
        inputTokens,
        tensorrt_llm::runtime::SamplingConfig{params.kvCacheManagerInstantiationParameters.maxBeamWidth},
        true,
    };
    auto const result = kvCacheManager->getRemainingBlocksToCompletion(llmRequest);
    ASSERT_EQ(result, params.expectedRemainingBlocksToCompletion);
}

INSTANTIATE_TEST_SUITE_P(RemainingBlocksToCompletionCorrectlyEstimated, RemainingBlocksToCompletionTest,
    ::testing::Values(
        GetRemainingBlocksToCompletionOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                1,
                1,
                1,
                1,
                4096,
                0,
                4096,
                1,
                4096 * 4,
                false,
            },
            1024,
            128,
            1024 + 128,
        },
        GetRemainingBlocksToCompletionOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                1,
                1,
                1,
                64,
                4096,
                0,
                4096,
                1,
                4096 * 4,
                false,
            },
            1024,
            128,
            18,
        },
        GetRemainingBlocksToCompletionOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                1,
                1,
                1,
                64,
                4096,
                0,
                128,
                1,
                4096 * 4,
                false,
            },
            1024, 128,
            18, // See `temporaryAttentionWindow` concept.
        }));

class FillKvCacheAndCompleteRequestsTest : public ::testing::TestWithParam<FillKvCacheAndCompleteRequestsParameters>
{
protected:
    void SetUp() override
    {
        auto const stream = std::make_shared<tr::CudaStream>();
        auto const params = GetParam();
        kvCacheManager = createKvCacheManager(params.kvCacheManagerInstantiationParameters, stream);
        kvCacheManager->allocatePools(nvinfer1::DataType::kFLOAT);
    }

    void TearDown() override {}

    std::shared_ptr<KVCacheManager> kvCacheManager;
};

TEST_P(FillKvCacheAndCompleteRequestsTest, FillKvCacheWithRequestsAndCompleteOneByOne)
{
    auto const params = GetParam();
    auto llmRequests = fillKvCacheManager(*kvCacheManager, params.promptLength, params.maxOutputLength,
        params.kvCacheManagerInstantiationParameters.maxBeamWidth, 0);

    // Completing half.
    for (auto& llmRequest : llmRequests)
    {
        for (SizeType32 i = 0; i < params.maxOutputLength; i++)
        {
            llmRequest.addNewToken(0, 0);
            kvCacheManager->addToken(llmRequest.mRequestId);
        }
        kvCacheManager->removeSequence(llmRequest.mRequestId, llmRequest);
    }
    ASSERT_EQ(kvCacheManager->getNumFreeBlocks(), params.kvCacheManagerInstantiationParameters.numBlocksInPrimaryPool);
}

TEST_P(FillKvCacheAndCompleteRequestsTest, FillKvCacheAndCompleteInParallel)
{
    auto const params = GetParam();
    auto llmRequests = fillKvCacheManager(*kvCacheManager, params.promptLength, params.maxOutputLength,
        params.kvCacheManagerInstantiationParameters.maxBeamWidth, 0);
    for (SizeType32 i = 0; i < params.maxOutputLength; i++)
    {
        for (auto const& llmRequest : llmRequests)
        {
            kvCacheManager->addToken(llmRequest.mRequestId);
        }
    }
}

auto const paramValues = ::testing::Values(
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            1,
            4096,
            0,
            4096,
            1,
            4096 * 4,
            false,
        },
        128,
        128,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096,
            0,
            4096,
            1,
            4096 * 4,
            false,
        },
        256,
        128,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096,
            0,
            4096,
            1,
            4096 * 4,
            false,
        },
        256,
        128,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096,
            0,
            4096,
            1,
            4096 * 4,
            false,
        },
        500,
        100,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096,
            0,
            128,
            1,
            4096 * 4,
            false,
        },
        500,
        100,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 128,
            0,
            4096,
            1,
            4096 * 4,
            false,
        },
        5250,
        500,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096,
            0,
            2048,
            1,
            4096 * 4,
            false,
        },
        5000,
        500,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        5000,
        500,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        500,
        5000,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2048,
        2048,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2049,
        2048,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2047,
        2048,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        1024,
        1024,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        1025,
        1023,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        1023,
        1025,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2047,
        32,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2049,
        32,
    },
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            64,
            4096 * 16,
            0,
            2048,
            1,
            4096 * 4,
            true,
        },
        2048 - 60,
        32,
    });

INSTANTIATE_TEST_SUITE_P(FillKvCacheAndCompleteRequestsTest, FillKvCacheAndCompleteRequestsTest, paramValues);
