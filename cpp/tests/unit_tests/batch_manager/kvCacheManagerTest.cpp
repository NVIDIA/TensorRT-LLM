/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/testing/kvCacheManagerTestUtil.h"

#include "gtest/gtest.h"
#include <cstdint>
#include <gmock/gmock.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <set>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <variant>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tlk = tensorrt_llm::batch_manager::kv_cache_manager;
namespace tle = tensorrt_llm::executor;
namespace tr = tensorrt_llm::runtime;
namespace fs = std::filesystem;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

using ParamType = bool;

namespace
{
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

SizeType32 theOnlyWindowSize(KVCacheManager const& kvCacheManager)
{
    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getWindowSizesMetadata().size(), 1) << "Assuming a single window size";
    return blockManager.getPoolWindowSize(0);
}
} // namespace

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

    auto constexpr beamWidth = 8;
    auto constexpr numBlocksPerBeam = blocksInPrimaryPool / beamWidth;
    auto constexpr numTokens = tokensPerBlock * numBlocksPerBeam;
    auto constexpr maxAttentionWindow = numTokens;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto constexpr requestId = 42;

    // Test: last block NOT shared (inputLength not aligned to tokensPerBlock)
    auto constexpr numTokensNotAligned = numTokens - 1;
    auto inputTokensNotAligned = makeInputTokens(numTokensNotAligned);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokensNotAligned, samplingConfig, isStreaming);
    GenerationRequest seq0{requestId, numTokensNotAligned, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seq0}, {numTokensNotAligned}, {numBlocksPerBeam}, {std::ref(*llmReq0)},
        maxAttentionWindow, /*isEnableBlockReuse=*/false);
    auto constexpr occupiedBlocks = (numBlocksPerBeam - 1) + beamWidth;
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - occupiedBlocks);
    auto const& ids = seq0.getCacheBlockIds(maxAttentionWindow);
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

    // Test: last block shared (inputLength aligned to tokensPerBlock)
    auto inputTokensAligned = makeInputTokens(numTokens);
    auto llmReq1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokensAligned, samplingConfig, isStreaming);
    GenerationRequest seq0b{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seq0b}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq1)},
        maxAttentionWindow, /*isEnableBlockReuse=*/false);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocksPerBeam);
    auto const& idsShared = seq0b.getCacheBlockIds(maxAttentionWindow);
    EXPECT_EQ(idsShared.size(), beamWidth);
    for (std::size_t i = 0u; i < idsShared.front().size(); ++i)
    {
        for (std::size_t beam = 1u; beam < idsShared.size(); ++beam)
        {
            EXPECT_EQ(idsShared.at(beam).at(i), idsShared.at(0).at(i));
        }
    }
    blockManager.releaseBlocks(seq0b);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Occupy blocks with two sequences: test duplicate requestId and exhaustion
    auto llmReq2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokensNotAligned, samplingConfig, isStreaming);
    GenerationRequest seq0c{requestId, numTokensNotAligned, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_NO_THROW((void) blockManager.addSequenceBatch({&seq0c}, {numTokensNotAligned}, {numBlocksPerBeam},
        {std::ref(*llmReq2)}, maxAttentionWindow, /*isEnableBlockReuse=*/false));
    auto llmReq3 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId + 1}, maxNewTokens, inputTokensNotAligned, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId + 1, numTokensNotAligned, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_NO_THROW((void) blockManager.addSequenceBatch({&seq1}, {numTokensNotAligned}, {numBlocksPerBeam},
        {std::ref(*llmReq3)}, maxAttentionWindow,
        /*isEnableBlockReuse=*/false));
    // same requestId not allowed
    auto llmReq4 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokensNotAligned, samplingConfig, isStreaming);
    GenerationRequest seq2{requestId, numTokensNotAligned, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_THROW((void) blockManager.addSequenceBatch({&seq2}, {numTokensNotAligned}, {numBlocksPerBeam},
                     {std::ref(*llmReq4)}, maxAttentionWindow,
                     /*isEnableBlockReuse=*/false),
        std::runtime_error);
    // no more blocks
    auto llmReq5 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId + 2}, maxNewTokens, inputTokensNotAligned, samplingConfig, isStreaming);
    GenerationRequest seq3{requestId + 2, numTokensNotAligned, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_THROW((void) blockManager.addSequenceBatch({&seq3}, {numTokensNotAligned}, {numBlocksPerBeam},
                     {std::ref(*llmReq5)}, maxAttentionWindow,
                     /*isEnableBlockReuse=*/false),
        std::runtime_error);
}

template <typename T>
void writePatternToOffloadedBlocksDRAM(T* rawBlockPtr, int blockSize, int mask)
{
    for (int i = 0; i < blockSize; ++i)
    {
        rawBlockPtr[i] = i & mask;
    }
}

template <typename T>
void writePatternToOffloadedBlocksGDS(
    std::string const& directory, int blockId, SizeType32 numPools, int blockSize, int mask)
{
    for (size_t poolIdx = 0; poolIdx < numPools; ++poolIdx)
    {
        std::string filename
            = directory + "/block_" + std::to_string(blockId) + "_pool_" + std::to_string(poolIdx) + ".bin";
        int fd = ::open(filename.c_str(), O_WRONLY);
        if (fd >= 0)
        {
            auto poolBlockSize = blockSize / numPools;
            std::vector<T> buffer(poolBlockSize);
            for (int i = 0; i < poolBlockSize; ++i)
            {
                buffer[i] = i & mask;
            }
            auto const bytesToWrite = static_cast<size_t>(poolBlockSize) * sizeof(T);
            auto const written = ::write(fd, buffer.data(), bytesToWrite);
            EXPECT_EQ(written, static_cast<ssize_t>(bytesToWrite))
                << "Failed to write pattern to offloaded block file " << filename;
            ::close(fd);
        }
    }
}

template <typename T, nvinfer1::DataType type, int mask, KvCacheTransferMode transferMode>
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

    auto constexpr batchSize = 1;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr bytesPerToken = 4;
    auto constexpr maxAttentionWindow = 4096;
    auto constexpr maxAttentionWindowAllLayer = 4096;
    auto constexpr sinkTokenLen = 0;
    auto constexpr canUseOneMoreBlock = true;
    std::string directory = "";
    static int file_num = 0;

    if constexpr (transferMode == KvCacheTransferMode::GDS)
    {
        std::string filename = std::string("test_copy") + std::to_string(file_num++);
        auto dirPath = fs::absolute(filename);
        fs::create_directories(dirPath);
        directory = dirPath.string();
    }

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamWidth = 1;
    auto constexpr beamIdx = 0;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksPerWindow, maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, type, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

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
    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    auto cacheBlockIds = seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds, ::testing::ElementsAreArray({0, 1, 2}));

    // Offload all 3 blocks, fill with predictable pattern, onboard
    for (auto cacheBlockId : cacheBlockIds)
    {
        auto block = blockManager.getBlockById(cacheBlockId, maxAttentionWindow);
        EXPECT_TRUE(block->isPrimary());
        // offload so we can write to block in CPU code
        blockManager.offloadBlock(block, maxAttentionWindow, transferMode, directory);
        EXPECT_FALSE(block->isPrimary());
        // need to sync so D2H transfer is done before accessing blocks
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        // fill with predictable pattern
        auto memoryPoolIndex = block->getMemoryPoolBlockIndex();
        auto blockPtr{tr::ITensor::slice(secondaryPoolPtr, memoryPoolIndex, 1)};
        auto rawBlockPtr = reinterpret_cast<T*>(blockPtr->data());
        // Write value
        if constexpr (transferMode == KvCacheTransferMode::DRAM)
        {
            writePatternToOffloadedBlocksDRAM<T>(rawBlockPtr, blockSize, mask);
        }
        else if constexpr (transferMode == KvCacheTransferMode::GDS)
        {
            auto block_id = block->getBlockId();
            auto numPools = blockManager.getNumPools(false);
            writePatternToOffloadedBlocksGDS<T>(directory, block_id, numPools, blockSize, mask);
        }
        // onboard
        blockManager.onboardBlock(seq0, block, maxAttentionWindow, transferMode, directory);
        EXPECT_TRUE(block->isPrimary());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));
    }
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    auto inputTokens1 = inputTokens;
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength1, beamWidth, blockManager.getWindowSizesMetadata()};
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 16);
    auto cacheBlockIds1 = seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds1, ::testing::ElementsAreArray({0, 1, 6}));
    // store blocks 0, 1 ([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.storeContextBlocks(seq1, *llmRequest1);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11] again.
    // Reuse blocks 0 and 1(pc). Block 1 is partially reused, but already referenced by seq1 so must be partial copied
    // into new block 2. Clear block 2 so we can see what was partial copied.
    auto block2 = blockManager.getBlockById(2, maxAttentionWindow);
    auto memoryPoolIndex2 = block2->getMemoryPoolBlockIndex();
    auto block2Ptr{tr::ITensor::slice(primaryPoolPtr, memoryPoolIndex2, 1)};
    EXPECT_EQ(cudaMemset(block2Ptr->data(), 0, blockSize * sizeof(T)), cudaSuccess);
    auto inputTokens2 = inputTokens;
    auto constexpr partiallyReusedTokens = 3;
    inputTokens2->resize(8 + partiallyReusedTokens + 1);
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    GenerationRequest seq2{requestId, inputLength2, beamWidth, blockManager.getWindowSizesMetadata()};
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 11);
    auto cacheBlockIds2 = seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds2, ::testing::ElementsAreArray({0, 2}));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Verify partial copied block 2
    // Block has shape [2, numLayers, numKvHeads, tokensPerBlock, sizePerHead]
    blockManager.offloadBlock(block2, maxAttentionWindow);
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
    blockManager.onboardBlock(seq2, block2, maxAttentionWindow, transferMode, directory);
    EXPECT_TRUE(block2->isPrimary());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);

    if constexpr (transferMode == KvCacheTransferMode::GDS)
        fs::remove_all(directory);
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT64)
{
    runPartialCopyTest<std::uint64_t, nvinfer1::DataType::kINT64, -1, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint64_t, nvinfer1::DataType::kINT64, -1, KvCacheTransferMode::GDS>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT32)
{
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kINT32, -1, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kINT32, -1, KvCacheTransferMode::GDS>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyFLOAT)
{
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kFLOAT, -1, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint32_t, nvinfer1::DataType::kFLOAT, -1, KvCacheTransferMode::GDS>();
}

#ifdef ENABLE_BF16
TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyBF16)
{
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kBF16, 65535, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kBF16, 65535, KvCacheTransferMode::GDS>();
}
#endif

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyHALF)
{
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kHALF, 65535, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint16_t, nvinfer1::DataType::kHALF, 65535, KvCacheTransferMode::GDS>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyBOOL)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kBOOL, 255, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kBOOL, 255, KvCacheTransferMode::GDS>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyUINT8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kUINT8, 255, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kUINT8, 255, KvCacheTransferMode::GDS>();
}

TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyINT8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kINT8, 255, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kINT8, 255, KvCacheTransferMode::GDS>();
}

#ifdef ENABLE_FP8
TEST_F(KVCacheManagerTest, BlockManagerTestPartialCopyFP8)
{
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kFP8, 255, KvCacheTransferMode::DRAM>();
    runPartialCopyTest<std::uint8_t, nvinfer1::DataType::kFP8, 255, KvCacheTransferMode::GDS>();
}
#endif

TEST_F(KVCacheManagerTest, BlockManagerTestWindowSizeToShare)
{
    auto constexpr numPrimaryBlocks = 16384;
    // Single window size
    {
        std::map<SizeType32, std::vector<SizeType32>> windowSizeToLayers{{1024, {0, 1, 2}}};
        std::map<SizeType32, SizeType32> cacheSizePerTokenPerWindow{{1024, 1}}; // Uniform cache size per token.

        auto result = BlockManager::calculateWindowSizeToShare(windowSizeToLayers, cacheSizePerTokenPerWindow);
        EXPECT_EQ(result.size(), 1);
        EXPECT_NEAR(result.at(1024), 1.0f, 1e-6f);
        // With a single window size, the entire share should be allocated to it.
    }
    // Variable window size
    {
        std::map<SizeType32, std::vector<SizeType32>> windowSizeToLayers{
            {1024, {1}},       // contribution = 1024*1 = 1024
            {4096, {0, 4, 5}}, // contribution = 4096*1 = 4096
            {8192, {2, 3}},    // contribution = 8192*1 = 8192
        };
        // Use identical cache size per token across window sizes for simplicity.
        std::map<SizeType32, SizeType32> cacheSizePerTokenPerWindow{{1024, 1}, {4096, 1}, {8192, 1}};

        auto result = BlockManager::calculateWindowSizeToShare(windowSizeToLayers, cacheSizePerTokenPerWindow);
        EXPECT_EQ(result.size(), 3);

        // Ensure the shares sum to 1.
        auto const sumShares = std::accumulate(
            result.begin(), result.end(), 0.0f, [](float sum, auto const& kv) { return sum + kv.second; });
        EXPECT_NEAR(sumShares, 1.0f, 1e-6f);

        // Calculate expected shares based on contributions.
        std::map<SizeType32, float> expectedShares;
        std::map<SizeType32, SizeType32> contributions;
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            contributions[windowSize] = windowSize * 1.0f;
        }
        auto const totalContribution = std::accumulate(contributions.begin(), contributions.end(), 0.0f,
            [](float sum, auto const& kv) { return sum + kv.second; });

        for (auto const& [windowSize, contribution] : contributions)
        {
            expectedShares[windowSize] = static_cast<float>(contribution) / totalContribution;
            EXPECT_NEAR(result.at(windowSize), expectedShares[windowSize], 1e-6f);
        }

        // Verify the exact hard-coded values mentioned in the comment
        EXPECT_NEAR(result.at(1024), 0.0769f, 1e-4f);
        EXPECT_NEAR(result.at(4096), 0.3077f, 1e-4f);
        EXPECT_NEAR(result.at(8192), 0.6154f, 1e-4f);

        // Verify that when shares are converted to actual block counts, they match expected values.
        auto getRoundedBlocks
            = [&](float share) { return static_cast<SizeType32>(std::round(share * numPrimaryBlocks)); };
        EXPECT_EQ(getRoundedBlocks(result.at(1024)), 1260);
        EXPECT_EQ(getRoundedBlocks(result.at(4096)), 5041);
        EXPECT_EQ(getRoundedBlocks(result.at(8192)), 10082);
    }

    // Variable window size with different cache sizes per token per window
    {
        std::map<SizeType32, std::vector<SizeType32>> windowSizeToLayers{
            {1024, {1}},       // contribution = 1024*(1*2) = 2048 (cache size per token per layer = 2)
            {4096, {0, 4, 5}}, // contribution = 4096*(3*4) = 49152 (cache size per token per layer = 4)
            {8192, {2, 3}},    // contribution = 8192*(2*1) = 16384 (cache size per token per layer = 1)
        };
        // Different cache sizes per token per window.
        // cacheSizePerTokenPerWindow is accumulated across the layers of given window size.
        std::map<SizeType32, SizeType32> cacheSizePerTokenPerWindow{{1024, 2}, {4096, 12}, {8192, 2}};

        auto result = BlockManager::calculateWindowSizeToShare(windowSizeToLayers, cacheSizePerTokenPerWindow);
        EXPECT_EQ(result.size(), 3);

        // Ensure the shares sum to 1.
        auto const sumShares = std::accumulate(
            result.begin(), result.end(), 0.0f, [](float sum, auto const& kv) { return sum + kv.second; });
        EXPECT_NEAR(sumShares, 1.0f, 1e-6f);

        // Calculate expected shares based on contributions with different cache sizes per token.
        std::map<SizeType32, float> expectedShares;
        std::map<SizeType32, SizeType32> contributions;
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            auto const cacheSizePerToken = cacheSizePerTokenPerWindow.at(windowSize);
            contributions[windowSize] = windowSize * cacheSizePerToken;
        }
        auto const totalContribution = std::accumulate(contributions.begin(), contributions.end(), 0.0f,
            [](float sum, auto const& kv) { return sum + kv.second; });

        for (auto const& [windowSize, contribution] : contributions)
        {
            expectedShares[windowSize] = static_cast<float>(contribution) / totalContribution;
            EXPECT_NEAR(result.at(windowSize), expectedShares[windowSize], 1e-6f);
        }

        // Verify the calculated shares for different cache sizes per token
        EXPECT_NEAR(result.at(1024), 2048.0f / (2048.0f + 49152.0f + 16384.0f), 1e-6f);  // ~0.0303
        EXPECT_NEAR(result.at(4096), 49152.0f / (2048.0f + 49152.0f + 16384.0f), 1e-6f); // ~0.7273
        EXPECT_NEAR(result.at(8192), 16384.0f / (2048.0f + 49152.0f + 16384.0f), 1e-6f); // ~0.2424
    }

    // Edge case: Single layer per window with varying cache sizes
    {
        std::map<SizeType32, std::vector<SizeType32>> windowSizeToLayers{
            {1024, {0}}, // contribution = 1024*1*8 = 8192 (cache size per token = 8)
            {4096, {1}}, // contribution = 4096*1*2 = 8192 (cache size per token = 2)
            {8192, {2}}, // contribution = 8192*1*1 = 8192 (cache size per token = 1)
        };
        // Equal contributions but different cache sizes per token
        std::map<SizeType32, SizeType32> cacheSizePerTokenPerWindow{{1024, 8}, {4096, 2}, {8192, 1}};

        auto result = BlockManager::calculateWindowSizeToShare(windowSizeToLayers, cacheSizePerTokenPerWindow);
        EXPECT_EQ(result.size(), 3);

        // All should have equal shares since contributions are equal
        EXPECT_NEAR(result.at(1024), 1.0f / 3.0f, 1e-6f);
        EXPECT_NEAR(result.at(4096), 1.0f / 3.0f, 1e-6f);
        EXPECT_NEAR(result.at(8192), 1.0f / 3.0f, 1e-6f);

        // Ensure the shares sum to 1.
        auto const sumShares = std::accumulate(
            result.begin(), result.end(), 0.0f, [](float sum, auto const& kv) { return sum + kv.second; });
        EXPECT_NEAR(sumShares, 1.0f, 1e-6f);
    }
}

TEST_F(KVCacheManagerTest, FindBlocksInReuseTreeByBlockKeysTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

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

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, false, stream,
        maxAttentionWindow, maxAttentionWindow, true);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] (17 tokens, three blocks)
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest0)});
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    auto cacheBlockIds = kvCacheManager.getSequence(requestId).getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds, ::testing::ElementsAreArray({0, 1, 2}));

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(requestId, llmRequest0);

    inputTokens->pop_back();
    BlockKey fullKey{*inputTokens};
    auto const foundFull = kvCacheManager.findBlocksInReuseTreeByBlockKey(fullKey, maxAttentionWindow);
    ASSERT_NE(foundFull, nullptr);
    auto const& lastBlock = foundFull;

    // Check the chain back to previous blocks
    auto const prev2 = lastBlock->getPrevBlock();
    ASSERT_NE(prev2, nullptr);
    auto const prev1 = prev2->getPrevBlock();
    ASSERT_NE(prev1, nullptr);
    EXPECT_EQ(prev1->getPrevBlock(), nullptr);
}

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
    auto constexpr numFp4EltsPerContainer = 2;
    auto constexpr vectorSize = 16;

    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr beamWidth = 1;

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kFP4, false, stream,
        maxAttentionWindow, maxAttentionWindow, true);

    kvCacheManager.allocatePools(/*useUvm=*/false);

    // We should have one additional pool for the block scales.
    EXPECT_EQ(kvCacheManager.getBlockManager().getNumPools(), 2);
    EXPECT_EQ(kvCacheManager.getBlockManager().getNumPools(/*includeBlockScalePools=*/false), 1);
    EXPECT_NE(kvCacheManager.getBlockScalePoolPointers(), nullptr);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_TRUE(blockManager.containsBlockScales(1));
    // Block size of pool 0 reflects the number of container elements. It is number of FP4 elements / 2.
    // The expected block size of pool 1 should be the number of FP4 elements / vectorSize.
    EXPECT_EQ(blockManager.getBlockSize(0) * numFp4EltsPerContainer / vectorSize, blockManager.getBlockSize(1));
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

    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8]
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);  // block 2 contains [8]
    llmRequest0->addNewToken(10, beamIdx); // block 2 contains [8, 9]
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (blocks contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens [0, 1, 2, 3, 4, 5, 6, 7, 8] and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1 ([0, 1, 2, 3], [4, 5, 6, 7]) and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    // at this point, block 3 contains [8]
    llmRequest1->addNewToken(9, beamIdx);  // block 3 contains [8, 9]
    llmRequest1->addNewToken(10, beamIdx); // block 3 contains [8, 9, 10]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed (blocks contain [8, 9])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    // reuse blocks 0, 1, 2(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8]) :: p = partial reuse
    auto inputTokens0 = std::make_shared<VecTokens>(*inputTokens);
    inputTokens0->emplace_back(9);
    GenerationRequest seq0_dup{10, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest0 = std::make_shared<LlmRequest>(
        seq0_dup.getRequestId(), maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    prepopulatedPromptLen0 = blockManager
                                 .addSequenceBatch({&seq0_dup}, {promptLen0}, {numContextBlocks0},
                                     {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                 .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), promptLen0 - 1);
    EXPECT_THAT(seq0_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // note that seq0_dup is holding blocks 0, 1 and 2 until releaseBlocks is called

    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // reuse blocks 0, 1 ([0, 1, 2, 3], [4, 5, 6, 7]) and get new block 4
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    GenerationRequest seq1_dup{11, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest1 = std::make_shared<LlmRequest>(
        seq1_dup.getRequestId(), maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    prepopulatedPromptLen1 = blockManager
                                 .addSequenceBatch({&seq1_dup}, {promptLen1}, {numContextBlocks1},
                                     {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                 .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest1->addNewToken(10, beamIdx); // block 4 contains [8, 9, 10]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // block 2 is stored for reuse (block contains [8]). nb! Last token of last block is never stored
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0_dup, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block is never stored
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1_dup, llmRequest1);
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
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0 ([0, 1, 2, 3]), get new block 5
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 5}));
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
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse blocks 0, 1, 4(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), numTokens - 1);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest3->addNewToken(11, beamIdx); // block 4 contains [8, 9, 11]
    numTokens = llmRequest3->getNumTokens(beamIdx);
    // one block used by both seq2 and seq3
    numBlocks += tc::ceilDiv(numTokens, tokensPerBlock) - 1;
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 5 is not stored since it is last block and has only one token
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block not stored
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
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
    GenerationRequest seq4{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1, 4(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8,9])
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen4 = blockManager
                                      .addSequenceBatch({&seq4}, {promptLen4}, {numContextBlocks4},
                                          {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), promptLen4 - 1);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    auto llmRequest4Short
        = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens4Short, samplingConfig, isStreaming);

    // llmRequest4Short tokens [0, 1, 2, 3, 4, 5, 6, 7, 8]
    // blocks 0 and 1 ([0, 1, 2, 3], [4, 5, 6, 7]) are already stored,
    // block 4 is freed
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4Short);
    blockManager.releaseBlocks(seq4, llmRequest4Short);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with 11 tokens again and make sure no discarded tokens reuse happens
    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
    // reuse blocks 0, 1, 2(p) ([0, 1, 2, 3], [4, 5, 6, 7], [8])
    // nb! LlmRequest retains state calculated during addSequenceBatch, this state affects result.
    // Calling addSequenceBatch a second time with same LlmRequest object will produce incorrect state.
    // Create new llmRequest4 instance to avoid this issue.
    GenerationRequest seq4_dup{14, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest4 = std::make_shared<LlmRequest>(
        seq4_dup.getRequestId(), maxNewTokens, inputTokens4, samplingConfig, isStreaming);
    promptLen4 = llmRequest4->getNumTokens(beamIdx);
    numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    prepopulatedPromptLen4 = blockManager
                                 .addSequenceBatch({&seq4_dup}, {promptLen4}, {numContextBlocks4},
                                     {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                 .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), promptLen4 - 2);
    EXPECT_THAT(seq4_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4);
    blockManager.releaseBlocks(seq4_dup, llmRequest4);
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
    GenerationRequest seq5{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, all blocks need to be freed
    auto promptLen5 = llmRequest5->getNumTokens(beamIdx);
    auto numContextBlocks5 = tc::ceilDiv(promptLen5, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen5 = blockManager
                                      .addSequenceBatch({&seq5}, {promptLen5}, {numContextBlocks5},
                                          {std::ref(*llmRequest5)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest5->setPrepopulatedPromptLen(prepopulatedPromptLen5, blockManager.getTokensPerBlock());
    llmRequest5->addNewToken(0, beamIdx);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 1); // incidental reuse

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest5);
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
    GenerationRequest seq6{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, all blocks need to be freed
    auto promptLen6 = llmRequest6->getNumTokens(beamIdx);
    auto numContextBlocks6 = tc::ceilDiv(promptLen6, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen6 = blockManager
                                      .addSequenceBatch({&seq6}, {promptLen6}, {numContextBlocks6},
                                          {std::ref(*llmRequest6)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)[0]
                                      .prepopulatedLen;
    llmRequest6->setPrepopulatedPromptLen(prepopulatedPromptLen6, blockManager.getTokensPerBlock());
    llmRequest6->addNewToken(0, beamIdx);
    // no reuse occurs because we are unable to reuse last input token and inputLength6 == 1.
    EXPECT_EQ(llmRequest6->getContextCurrentPosition(), 0);

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest6);
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

    auto constexpr numReturnSequences = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto inputTokenExtraIds = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{1, 1, 2, 2, 3, 3, 0, 0, 0});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    GenerationRequest seq0_dup{10, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest0 = std::make_shared<LlmRequest>(seq0_dup.getRequestId(), maxNewTokens, inputTokens, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    prepopulatedPromptLen0 = blockManager
                                 .addSequenceBatch({&seq0_dup}, {promptLen0}, {numContextBlocks0},
                                     {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    GenerationRequest seq1_dup{11, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest1 = std::make_shared<LlmRequest>(seq1_dup.getRequestId(), maxNewTokens, inputTokens1, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1, numReturnSequences);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    prepopulatedPromptLen1 = blockManager
                                 .addSequenceBatch({&seq1_dup}, {promptLen1}, {numContextBlocks1},
                                     {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0_dup, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [(2, 0), (3, 0), (4, 0)])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1_dup, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2, numReturnSequences);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3, numReturnSequences);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new block 8, 9
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 8, 9}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithMultimodalHashTest)
{
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

    auto constexpr numReturnSequences = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // Create multimodal hash data (256-bit hash = 8 int32 values)
    auto multimodalHashes = std::make_shared<std::vector<std::vector<SizeType32>>>(std::vector<std::vector<SizeType32>>{
        {0x12345678, -0x6F543211, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666} // Hash 1
    });
    auto multimodalPositions
        = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{2});                    // Start at token 2
    auto multimodalLengths = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{4}); // Length 4 tokens
    // assume prompt id starts from 100
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 0, 1, 2});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        multimodalHashes, multimodalPositions, multimodalLengths, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false,
        std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt,
        std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt,
        numReturnSequences);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Input: [100, 101, 102, 103, 104, 105, 0, 1, 2] (9 tokens)
    // Multimodal: starts at token 2, length 4 → [102, 103, 104, 105]

    // Block 0: [100, 101, 102, 103] ← Contains multimodal (102, 103)
    // Block 1: [104, 105, 0, 1]     ← Contains multimodal (104, 105)
    // Block 2: [2, 3, 4]            ← No multimodal
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and same multimodal hash - should reuse
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        multimodalHashes, multimodalPositions, multimodalLengths, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false,
        std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt,
        std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt,
        numReturnSequences);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // should reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // block 3 matches block 2 and will be freed
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 2: Different multimodal hash
    requestId = 2;
    auto multimodalHashes2
        = std::make_shared<std::vector<std::vector<SizeType32>>>(std::vector<std::vector<SizeType32>>{
            {0x45678123, 0x23456789, 0x34567890, 0x12121212, 0x56565656, 0x78787878, 0x54545454, 0x67676767} // Hash 2
        });
    auto multimodalPositions2
        = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{2});                     // Start at token 2
    auto multimodalLengths2 = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{4}); // Length 4 tokens
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        multimodalHashes2, multimodalPositions2, multimodalLengths2, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false,
        std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt,
        std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt,
        numReturnSequences);

    GenerationRequest seq2{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new blocks 4, 5, 6
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({4, 5, 6}));
    llmRequest2->addNewToken(9, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 3: Multiple multimodal hashes and partial reuse
    requestId = 3;
    auto multimodalHashes3
        = std::make_shared<std::vector<std::vector<SizeType32>>>(std::vector<std::vector<SizeType32>>{
            {0x12345678, -0x6F543211, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666}, // Hash 1
            {0x45678123, 0x23456789, 0x34567890, 0x12121212, 0x56565656, 0x78787878, 0x54545454, 0x67676767}   // Hash 2
        });
    auto multimodalPositions3
        = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{2, 4}); // Start at token 2 and 4
    auto multimodalLengths3
        = std::make_shared<std::vector<SizeType32>>(std::vector<SizeType32>{2, 2}); // Length 2 tokens

    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        multimodalHashes3, multimodalPositions3, multimodalLengths3, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false,
        std::nullopt, std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt,
        std::nullopt, std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt,
        numReturnSequences);
    GenerationRequest seq3{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new blocks 7, 8
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(),
        tokensPerBlock); // only reuse block 0 [100, 101, 102, 103] with same hash/offset
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 7, 8}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    // clean up
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
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

    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // loraTaskId is 0 for common cases
    LlmRequest::LoraTaskIdType loraTaskId{0};
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // get new blocks 0, 1, 2 ([0,1,2,3], [4,5,6,7], [8])
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(9, beamIdx);
    llmRequest0->addNewToken(10, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // store blocks 0, 1, 2 for reuse ([0,1,2,3], [4,5,6,7], [8,9])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and loraTaskId, then remove it
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // store block 3 for reuse ([8,9])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    // reuse blocks 0, 1 and get new block 4
    GenerationRequest seq0_dup{10, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest0 = std::make_shared<LlmRequest>(seq0_dup.getRequestId(), maxNewTokens, inputTokens, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    prepopulatedPromptLen0 = blockManager
                                 .addSequenceBatch({&seq0_dup}, {promptLen0}, {numContextBlocks0},
                                     {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    // nb! addNewToken adds new generated token, number of input tokens stay the same.
    // calling addNewToken before addSequence potentially triggers this error message:
    // Assertion failed: prepopulatedPromptLen < promptLen
    // because maximum value for prepopulatedPromptLen is number of input+output tokens - 1,
    // but promptLen is number of input tokens.
    llmRequest0->addNewToken(9, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // inputTokens1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    GenerationRequest seq1_dup{11, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest1 = std::make_shared<LlmRequest>(seq1_dup.getRequestId(), maxNewTokens, inputTokens1, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    // reuse 0, 1, 2(p) ([0,1,2,3], [4,5,6,7], [8])
    prepopulatedPromptLen1 = blockManager
                                 .addSequenceBatch({&seq1_dup}, {promptLen1}, {numContextBlocks1},
                                     {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // store block 4 for reuse ([8])
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0_dup, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [8, 9]). nb! Last token of last block is not stored
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1_dup, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(1);
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    // no reuse expected. Input tokens match blocks 0 and 1, but lora task id differs.
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(9, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store blocks 5, 6, 7 for reuse ([0,1,2,3], [4,5,6,7], [8]) with loraTaskId 1
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and more tokens
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse blocks 5, 6, 7(p) ([0,1,2,3], [4,5,6,7], [8])
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), promptLen3 - 2);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store block 7 for reuse ([8,9]) with loraTaskId 1
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse blocks 0, get new block 8
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen4 = blockManager
                                      .addSequenceBatch({&seq4}, {promptLen4}, {numContextBlocks4},
                                          {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 8}));
    llmRequest4->addNewToken(5, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 8 is stored with [4] and loraTaskId 0
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4);
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // add request with same tokens as request0 but without loraTaskId
    requestId = 5;
    auto llmRequest5 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);

    numTokens = llmRequest5->getNumTokens(beamIdx);
    GenerationRequest seq5{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 9, 10, 11
    auto promptLen5 = llmRequest5->getNumTokens(beamIdx);
    auto numContextBlocks5 = tc::ceilDiv(promptLen5, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen5 = blockManager
                                      .addSequenceBatch({&seq5}, {promptLen5}, {numContextBlocks5},
                                          {std::ref(*llmRequest5)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest5->setPrepopulatedPromptLen(prepopulatedPromptLen5, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq5.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({9, 10, 11}));
    llmRequest5->addNewToken(9, beamIdx);
    numTokens = llmRequest5->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 9, 10, 11 are stored without loraTaskId
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest5);
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

    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 0, 1, 2 are stored for reuse (block 2 contains [(2, 0), (3, 0)] with loraTaskId 1)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens but different loraTaskId and then remove it
    requestId = 1;
    LlmRequest::LoraTaskIdType loraTaskId2 = static_cast<LlmRequest::LoraTaskIdType>(2);
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId2,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // no reuse, get new block 3, 4, 5
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 3, 4, 5 are stored for reuse (block 5 contains [(2, 0), (3, 0)] with loraTaskId 2)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    GenerationRequest seq0_dup{10, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest0 = std::make_shared<LlmRequest>(seq0_dup.getRequestId(), maxNewTokens, inputTokens, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId1, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // reuse blocks 0, 1 and get new block 6
    prepopulatedPromptLen0 = blockManager
                                 .addSequenceBatch({&seq0_dup}, {promptLen0}, {numContextBlocks0},
                                     {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 6}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 3, 4 and reuse block 5
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    GenerationRequest seq1_dup{11, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    llmRequest1 = std::make_shared<LlmRequest>(seq1_dup.getRequestId(), maxNewTokens, inputTokens1, samplingConfig,
        isStreaming, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        loraTaskId2, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    prepopulatedPromptLen1 = blockManager
                                 .addSequenceBatch({&seq1_dup}, {promptLen1}, {numContextBlocks1},
                                     {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                 .front()
                                 .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1_dup.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0_dup, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1_dup, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids and loraTaskId 1
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 7, 8, 9
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({7, 8, 9}));
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new block 10, 11
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 10, 11}));
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId2,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 3, get new block 12, 13
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen4 = blockManager
                                      .addSequenceBatch({&seq4}, {promptLen4}, {numContextBlocks4},
                                          {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 12, 13}));
    llmRequest4->addNewToken(3, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 3);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    blockManager.releaseBlocks(seq3, llmRequest3);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4);
    blockManager.releaseBlocks(seq4, llmRequest4);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

TEST_F(KVCacheManagerTest, BlockManagerReuseWithCacheSaltIdTest)
{
    // Test that cache_salt_id prevents KV cache reuse between requests with same tokens
    // but different cache_salt_id values.
    using VecTokenExtraIds = LlmRequest::VecTokenExtraIds;
    using CacheSaltIDType = LlmRequest::CacheSaltIDType;

    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr numReturnSequences = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // Create shared input tokens
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 1: Request without cache_salt_id
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt); // No cache_salt_id

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Add first request and get blocks 0, 1, 2
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {promptLen0}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));

    // Add generated tokens
    llmRequest0->addNewToken(3, beamIdx);
    llmRequest0->addNewToken(4, beamIdx);
    auto numTokens = llmRequest0->getNumTokens(beamIdx);
    auto numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(numBlocks, 3);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Release blocks to make them available for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 2: Request with same tokens but with cache_salt_id = 12345
    requestId = 1;
    CacheSaltIDType cacheSaltId1{12345};
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId1); // With cache_salt_id = 12345

    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should NOT reuse blocks despite same tokens, because cache_salt_id is different
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {promptLen1}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0); // No reuse, starts from scratch
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));

    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Release blocks
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 3: Request with same tokens and same cache_salt_id = 12345
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId1); // Same cache_salt_id = 12345

    GenerationRequest seq2{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // SHOULD reuse blocks because both tokens and cache_salt_id match
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {promptLen2}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 2 * tokensPerBlock); // Reuse blocks 3,4
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 6}));

    llmRequest2->addNewToken(3, beamIdx);
    llmRequest2->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Release blocks
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 4: Request with same tokens but different cache_salt_id = 67890
    requestId = 3;
    CacheSaltIDType cacheSaltId2{67890};
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId2); // Different cache_salt_id = 67890

    GenerationRequest seq3{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should NOT reuse blocks from any previous request because cache_salt_id is different
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {promptLen3}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 0); // No reuse
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({7, 8, 9}));

    llmRequest3->addNewToken(5, beamIdx);
    llmRequest3->addNewToken(6, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 5: Request without cache_salt_id again
    requestId = 4;
    auto llmRequest4 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false,
        std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt); // No cache_salt_id

    GenerationRequest seq4{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should reuse blocks from request0 (blocks 0,1) because both have no cache_salt_id
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen4 = blockManager
                                      .addSequenceBatch({&seq4}, {promptLen4}, {numContextBlocks4},
                                          {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 2 * tokensPerBlock); // Reuse blocks 0,1
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 10}));

    llmRequest4->addNewToken(7, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    // Clean up
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    blockManager.releaseBlocks(seq3, llmRequest3);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4);
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
    auto constexpr maxSequenceLength = maxBlocksPerSeq * tokensPerBlock;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    // Add the sequence to req0
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest0)}));

    // After first addition, check allocations and reuses
    auto numBlocks = tc::ceilDiv(inputLength, tokensPerBlock);
    EXPECT_EQ(llmRequest0->getReusedBlocksPerRequest(), 0);
    EXPECT_EQ(llmRequest0->getAllocTotalBlocksPerRequest(), numBlocks);
    EXPECT_EQ(llmRequest0->getAllocNewBlocksPerRequest(), numBlocks);
    EXPECT_EQ(llmRequest0->getMissedBlocksPerRequest(), numBlocks);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId, llmRequest0));

    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest1)}));

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

    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, maxAttentionWindow);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    SizeType32 constexpr maxNewTokens{0};
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
    GenerationRequest seq0{0, inputLength0, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks0 = tc::ceilDiv(inputLength0, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen0 = blockManager
                                      .addSequenceBatch({&seq0}, {llmRequest0->getNumTokens(0)}, {numContextBlocks0},
                                          {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest0->setPrepopulatedPromptLen(prepopulatedPromptLen0, blockManager.getTokensPerBlock());

    // Add another sequence with different tokens, at a low priority
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{1, inputLength1, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks1 = tc::ceilDiv(inputLength1, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen1 = blockManager
                                      .addSequenceBatch({&seq1}, {llmRequest1->getNumTokens(0)}, {numContextBlocks1},
                                          {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest1->setPrepopulatedPromptLen(prepopulatedPromptLen1, blockManager.getTokensPerBlock());

    // Release both sequences
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    blockManager.releaseBlocks(seq0, llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.releaseBlocks(seq1, llmRequest1);

    // Add and then release another sequence
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    llmRequest2->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 20)}, 20));
    GenerationRequest seq2{2, inputLength2, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks2 = tc::ceilDiv(inputLength2, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen2 = blockManager
                                      .addSequenceBatch({&seq2}, {llmRequest2->getNumTokens(0)}, {numContextBlocks2},
                                          {std::ref(*llmRequest2)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest2->setPrepopulatedPromptLen(prepopulatedPromptLen2, blockManager.getTokensPerBlock());
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    blockManager.releaseBlocks(seq2, llmRequest2);

    // Check that request 1 blocks were overwritten
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    GenerationRequest seq3{3, inputLength3, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks3 = tc::ceilDiv(inputLength3, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen3 = blockManager
                                      .addSequenceBatch({&seq3}, {llmRequest3->getNumTokens(0)}, {numContextBlocks3},
                                          {std::ref(*llmRequest3)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest3->setPrepopulatedPromptLen(prepopulatedPromptLen3, blockManager.getTokensPerBlock());

    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 4);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    blockManager.releaseBlocks(seq3, llmRequest3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 4);

    // Check that request 0 blocks weren't overwritten
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength4 = static_cast<SizeType32>(inputTokens4->size());
    auto llmRequest4 = std::make_shared<LlmRequest>(4, maxNewTokens, inputTokens4, samplingConfig, isStreaming);
    GenerationRequest seq4{4, inputLength3, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks4 = tc::ceilDiv(inputLength4, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen4 = blockManager
                                      .addSequenceBatch({&seq4}, {llmRequest4->getNumTokens(0)}, {numContextBlocks4},
                                          {std::ref(*llmRequest4)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest4->setPrepopulatedPromptLen(prepopulatedPromptLen4, blockManager.getTokensPerBlock());

    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 4);

    // Check that request 2 block 0 has been evicted
    auto inputTokens5 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength5 = static_cast<SizeType32>(inputTokens5->size());
    auto llmRequest5 = std::make_shared<LlmRequest>(5, maxNewTokens, inputTokens5, samplingConfig, isStreaming);
    GenerationRequest seq5{5, inputLength5, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks5 = tc::ceilDiv(inputLength5, blockManager.getTokensPerBlock());
    auto prepopulatedPromptLen5 = blockManager
                                      .addSequenceBatch({&seq5}, {llmRequest5->getNumTokens(0)}, {numContextBlocks5},
                                          {std::ref(*llmRequest5)}, maxAttentionWindow, /*isEnableBlockReuse=*/true)
                                      .front()
                                      .prepopulatedLen;
    llmRequest5->setPrepopulatedPromptLen(prepopulatedPromptLen5, blockManager.getTokensPerBlock());

    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);
}

TEST_F(KVCacheManagerTest, KVCacheManagerDecodeBlockPriorityTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 8;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 90)}, 90));
    kvCacheManager.addSequenceBatch({{{0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});

    // 5 blocks available.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 5);

    // Add a token to request 0, which occupies a new block 3.
    kvCacheManager.addToken(0);
    llmRequest0->addNewToken(0, 0); // block 3 contains [0]

    // 4 blocks left.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 4);

    // uses up 3 more blocks 4, 5, 6. [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    llmRequest1->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 5)}, 5));
    kvCacheManager.addSequenceBatch({{{1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});

    // one block left.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 1);

    // add another token, which occupies another new block
    kvCacheManager.addToken(1);
    llmRequest1->addNewToken(0, 0); // block 7 contains [0]

    // no block available.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 0);

    // remove both sequences, blocks get stored
    // leaf block 3 (priority 90), context blocks 2, 1, 0 (priority 5)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    // leaf block 7 (priority 5), context blocks 6, 5, 4 (priority 90)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    // all blocks are available again.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // no reuse, blocks are evicted by new request:
    // evict block 7 (lowest priority, first released block)
    // evict block 6 (lowest priority, second released block)
    // evict block 5 (lowest priority, third released block)
    // uses up 3 blocks 7, 6, 5. [24, 25, 26, 27], [28, 29, 30, 31], [32, 33, 34, 35]
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{2, inputLength2, beamWidth}}}, {std::ref(*llmRequest2)});
    // leaf block 2 (priority 35), context blocks 3, 7 (priority 35)
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{3, inputLength3, beamWidth}}}, {std::ref(*llmRequest3)});

    // Reuse block 0, 1, and partial reuse block 2. (maximum reuse is inputLength - 1)
    // Two blocks reused, the third block partially reused.
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 11);
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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 80, 10ms)}, 80));
    kvCacheManager.addSequenceBatch({{{0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    llmRequest1->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 80));
    kvCacheManager.addSequenceBatch({{{1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Manually trigger a refresh.
    kvCacheManager.refreshBlocks();

    // Clear out some of the blocks.
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{2, inputLength2, beamWidth}}}, {std::ref(*llmRequest2)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // Check that the [12, 13, 14, 15] block is still in the cache
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);

    kvCacheManager.addSequenceBatch({{{3, inputLength3, beamWidth}}}, {std::ref(*llmRequest3)});

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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 9;
    auto constexpr blocksInSecondaryPool = 0;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);
    {
        // 12 tokens, occupy 3 blocks 0, 1, 2.
        // [1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
        auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
        auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
        llmRequest0->setKvCacheRetentionConfig(
            KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)},
                50)); // Set all blocks to priority 50.
        kvCacheManager.addSequenceBatch({{{0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
        kvCacheManager.storeContextBlocks(*llmRequest0);
        // Occupy a new block, block 3, adding 3 tokens to block 3.
        // [1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 0, 0]
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(0);
            llmRequest0->addNewToken(0, 0);
        }
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
        (void) kvCacheManager.removeSequence(0, llmRequest0);
    }
    {
        // 12 tokens, occupy 3 blocks 4, 5, 6.
        // [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
        auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
        auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
        auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
        llmRequest1->setKvCacheRetentionConfig(KvCacheRetentionConfig(
            {}, KvCacheRetentionConfig::kMaxRetentionPriority, 20ms)); // Set decode blocks to max priority for 20ms.
        kvCacheManager.addSequenceBatch({{{1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
        kvCacheManager.storeContextBlocks(*llmRequest1);
        // Occupy a new block, block 3, adding 3 tokens to block 3.
        // [1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 0, 0]
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(1);
            llmRequest1->addNewToken(0, 0);
        }
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
        (void) kvCacheManager.removeSequence(1, llmRequest1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    kvCacheManager.refreshBlocks();

    // 8 tokens, occupying blocks 8, 6
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 0, 0, 0, 0, 0, 0, 0});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{2, inputLength2, beamWidth}}}, {std::ref(*llmRequest2)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // 12 tokens, reusing block 4, 5. Block 6 is overwritten so no reuse.
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{3, inputLength3, beamWidth}}}, {std::ref(*llmRequest3)});

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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    // 12 tokens, get block 0, 1, 2
    // [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    kvCacheManager.addSequenceBatch({{{0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    // store blocks 0, 1, 2 for reuse ([0,1,2,3], [4,5,6,7], [8,9,10])

    // Offload the last two blocks of llmRequest0 to secondary memory
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);

    // Uses blocks 3, 4, 5, block 2 and 1 to be offloaded to secondary
    // Block 4 is now in primary (replacing 2)
    // Block 5 is now in primary (replacing 1)
    kvCacheManager.addSequenceBatch({{{1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);
    // store blocks 3, 4, 5 for reuse ([1,1,2,3], [4,5,6,7], [8,9,10])

    // Match the middle block of request 0
    // Uses block 6, block 0 is offloaded to secondary
    // Block 6 copies content from block 0 to itselg.
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    // reuse block 0
    kvCacheManager.addSequenceBatch({{{2, inputLength2, beamWidth}}}, {std::ref(*llmRequest2)});
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 3);
    kvCacheManager.storeContextBlocks(*llmRequest2);

    // Add a decode block that matches the contents of seq 0 block 1, add a unique decode block
    // The 4 tokens added has the same content as block 1.
    for (int token = 4; token < 8; token++)
    {
        llmRequest2->addNewToken(token, 0);
        kvCacheManager.addToken(2);
    }
    // Add 2 more tokens, occupying another block
    llmRequest2->addNewToken(0, 0);
    kvCacheManager.addToken(2);

    llmRequest2->addNewToken(0, 0);
    kvCacheManager.addToken(2);
    // The middle block remains in secondary, but the third block is in primary
    // FIXME: When removing the sequence, we should observe whether released
    // blocks can replace itself as the block reused in the search tree if
    // the matching block is currently in secondary memory. We can release the
    // block in secondary if so.
    // If we do this, then the context current position at the bottom of this
    // unit test will be 9 because then the block content [4,5,6,7] can be
    // found and reused.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // 10 tokens, reusing the block 0 only because when we want to acquire
    // the second block, contents of block 3 will be offloaded to block 1.
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 0, 0});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{3, inputLength3, beamWidth}}}, {std::ref(*llmRequest3)});
    // The batch two-phase claim-then-onboard strategy protects matching blocks from eviction,
    // so all 9 matching tokens (2 full blocks + 1 partial) are recovered.
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 9);
}

TEST_F(KVCacheManagerTest, KVCacheManagerStoreContextBlocksUsesMaterializedContextExtent)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto constexpr kMaterializedContextLength = 5;
    auto constexpr kReusableContextLength = 4;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, /*chunkSize*/ 0, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{0, inputLength, beamWidth}}}, {std::ref(*llmRequest0)});
    llmRequest0->setContextCurrentPosition(kMaterializedContextLength);

    kvCacheManager.storeContextBlocks(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, std::nullopt);

    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{1, inputLength, beamWidth}}}, {std::ref(*llmRequest1)});

    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kReusableContextLength);
}

TEST_F(KVCacheManagerTest, KVCacheManagerReleaseBlocksUsesMaterializedContextExtent)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto constexpr kMaterializedContextLength = 5;
    auto constexpr kReusableContextLength = 4;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, /*chunkSize*/ 0, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());

    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{0, inputLength, beamWidth}}}, {std::ref(*llmRequest0)});
    llmRequest0->setContextCurrentPosition(kMaterializedContextLength);

    (void) kvCacheManager.removeSequence(0, llmRequest0);

    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{1, inputLength, beamWidth}}}, {std::ref(*llmRequest1)});

    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kReusableContextLength);
}

TEST_F(KVCacheManagerTest, KVCacheManagerLeafBlockTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    llmRequest0->addNewToken(0, 0);
    kvCacheManager.addToken(0);

    // The second block allocated should be first in line for eviction.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});

    GenerationRequest const& seq1 = kvCacheManager.getSequence(1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    // Block 1 should NOT be reused. It was not freed even if partial.
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(0), ::testing::ElementsAreArray({2}));

    // Allocate the remaining 3 blocks in primary
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{2, inputLength2, beamWidth}}}, {std::ref(*llmRequest2)});

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{3, inputLength3, beamWidth}}}, {std::ref(*llmRequest3)});
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 11);

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 1);

    kvCacheManager.addToken(3);
    llmRequest3->addNewToken(0, 0);

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 0);

    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{42});
    auto const inputLength4 = static_cast<SizeType32>(inputTokens4->size());
    auto llmRequest4 = std::make_shared<LlmRequest>(4, maxNewTokens, inputTokens4, samplingConfig, isStreaming);

    EXPECT_THROW(
        kvCacheManager.addSequenceBatch({{{4, inputLength4, beamWidth}}}, {std::ref(*llmRequest4)}), std::exception);
}

TEST_F(KVCacheManagerTest, KVCacheManagerLeafBlockWithDependentTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 1;

    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto constexpr beamIdx = 0;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true);
    kvCacheManager.allocatePools(false);

    // Create sequence with one block worth of context tokens
    int requestId0 = 0;
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0
        = std::make_shared<LlmRequest>(requestId0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId0, inputLength0, beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId0);

    // Add two more blocks of generated tokens
    for (int i = tokensPerBlock; i < 3 * tokensPerBlock; ++i)
    {
        llmRequest0->addNewToken(i, beamIdx);
        kvCacheManager.addToken(requestId0);
    }

    // Verify
    auto cacheBlockIds0 = seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds0, ::testing::ElementsAreArray({0, 1, 2}));

    // Raise priority of middle block to prevent offloading
    auto const& blockManager = kvCacheManager.getBlockManager();
    auto middleBlock = blockManager.getBlockById(cacheBlockIds0[1], maxAttentionWindow);
    middleBlock->setPriority(75);

    // Create another sequence with one block worth of context tokens (no reuse).
    // 4 tokens, occupying block 3
    int requestId1 = 1;
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1
        = std::make_shared<LlmRequest>(requestId1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId1, inputLength1, beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId1);

    // Verify that all primary blocks are in use
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    // Free first sequence
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(requestId0, llmRequest0);

    // Verify that 3 primary blocks are free.
    // Since block 1 has higher priority, block 2 and 0 will be used first.
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 3);

    // Write one generated token to second sequence. This will prompt block 2 to be offloaded.
    // Block 4 will be in primary (replacing block 2)
    llmRequest1->addNewToken(104, beamIdx);
    kvCacheManager.addToken(requestId1);

    // Verify that block 2 has block 1 as parent
    auto block2 = blockManager.getBlockById(2, maxAttentionWindow);
    EXPECT_TRUE(block2->getPrevBlock() != nullptr);
    EXPECT_EQ(block2->getPrevBlock()->getBlockId(), 1);
    EXPECT_FALSE(block2->isPrimary());

    // Fill block
    for (int i = 101 + tokensPerBlock; i < 100 + 2 * tokensPerBlock; ++i)
    {
        llmRequest1->addNewToken(i, beamIdx);
        kvCacheManager.addToken(requestId1);
    }

    // Verify
    auto cacheBlockIds1 = seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds1, ::testing::ElementsAreArray({3, 4}));

    // Write one generated token to second sequence. This will prompt block 0 to be offloaded,
    // replacing block 2.
    llmRequest1->addNewToken(100 + 2 * tokensPerBlock, beamIdx);
    kvCacheManager.addToken(requestId1);

    // Verify that block 2 is free, has no parent
    EXPECT_EQ(block2->getPrevBlock(), nullptr);
    // Verify that it is block 0 that is in secondary
    auto block0 = blockManager.getBlockById(0, maxAttentionWindow);
    EXPECT_FALSE(block0->isPrimary());

    // Cleanup
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(requestId1, llmRequest1);
}

TEST_P(KVCacheManagerTest, DISABLED_KVCacheManagerAllocationTest)
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

    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxSequenceLength;
    auto constexpr inputLength = maxSequenceLength - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr useUvm = false;

    auto const homogeneousLayers = GetParam();

    auto const granularity = tensorrt_llm::common::getAllocationGranularity();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);

    auto const& blockManager = kvCacheManager.getBlockManager();
    auto const& bufferManager = blockManager.getBufferManager(theOnlyWindowSize(kvCacheManager));
    auto const memoryPoolUsedBefore = bufferManager.memoryPoolUsed();
    kvCacheManager.allocatePools(useUvm);
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
    // Full attention
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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxSequenceLength;
    auto constexpr inputLength = maxSequenceLength - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{maxBeamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, sinkTokenLength,
            stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto inputTokens0 = makeInputTokens(inputLength);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReq0)}));
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
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth * 2);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        auto inputTokensLoop = makeInputTokens(inputLength);
        auto llmReqLoop = std::make_shared<LlmRequest>(static_cast<LlmRequest::RequestIdType>(requestId), maxNewTokens,
            inputTokensLoop, samplingConfig, isStreaming);
        EXPECT_NO_THROW(
            kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReqLoop)}));
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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxSequenceLength;
    auto constexpr inputLength = maxSequenceLength - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const homogeneousLayers = GetParam();

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{maxBeamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto inputTokens0 = makeInputTokens(inputLength);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReq0)}));
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
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        auto inputTokensLoop = makeInputTokens(inputLength);
        auto llmReqLoop = std::make_shared<LlmRequest>(static_cast<LlmRequest::RequestIdType>(requestId), maxNewTokens,
            inputTokensLoop, samplingConfig, isStreaming);
        EXPECT_NO_THROW(
            kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReqLoop)}));
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
    auto constexpr tokensPerBlock = 64;
    auto constexpr blockLengthPerSeq = 10;
    auto constexpr maxNumSequences = 8;
    // TODO: Support and add coverage for beamWidth > 1
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr requestId = 7;
    auto constexpr maxSequenceLength = tokensPerBlock * blockLengthPerSeq;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    auto constexpr inputLength = maxSequenceLength - tokensPerBlock - 1;
    auto constexpr maxAttentionWindow = inputLength; // sliding window attention
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (blockLengthPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{maxBeamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, sinkTokenLength,
            stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto inputTokens0 = makeInputTokens(inputLength);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReq0)}));
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
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth * 2);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        auto inputTokensLoop = makeInputTokens(inputLength);
        auto llmReqLoop = std::make_shared<LlmRequest>(static_cast<LlmRequest::RequestIdType>(requestId), maxNewTokens,
            inputTokensLoop, samplingConfig, isStreaming);
        EXPECT_NO_THROW(
            kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReqLoop)}));
        currentNumBlocks -= numSharedBlocks + maxBeamWidth;
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
        EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
        EXPECT_EQ(blockManager.getNumFreeBlocks(), currentNumBlocks);
    }
    EXPECT_EQ(blockManager.getNumFreeBlocks(), maxNumSequences);
}

TEST_F(KVCacheManagerTest, KVCacheManagerMaxAttentionWindowSmallerThanBlockSizeTest)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr maxSequenceLength = 128;

    // Enable sliding window kv cache for long input tokens.
    auto constexpr maxAttentionWindow = 3;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
        sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    auto const onlyWindowSize = theOnlyWindowSize(kvCacheManager);

    SizeType32 constexpr maxNewTokens = 40;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    ///////////////////////////////////////////////////////////////////////////
    // add a request that starts shorter and gets longer than the max attention window and then remove it
    SizeType32 requestId = 0;
    int inputLength = 2;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0}));

    // add tokens, reaching max attention window
    llmRequest->addNewToken(1002, beamIdx);
    kvCacheManager.addToken(requestId);
    auto numBlocks = seq0.getCacheBlockIds(onlyWindowSize)[beamIdx].size();
    EXPECT_EQ(numBlocks, 1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0}));

    // add new tokens exceeding max attention window, but not enough to allocate another block
    llmRequest->addNewToken(1003, beamIdx);
    kvCacheManager.addToken(requestId);
    numBlocks = seq0.getCacheBlockIds(onlyWindowSize)[beamIdx].size();
    EXPECT_EQ(numBlocks, 1);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0}));

    // add more new tokens, enough to allocate a new block but not enough to detach block
    llmRequest->addNewToken(1004, beamIdx);
    kvCacheManager.addToken(requestId);
    numBlocks = seq0.getCacheBlockIds(onlyWindowSize)[beamIdx].size();
    EXPECT_EQ(numBlocks, 2);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1}));

    // add more new tokens, enough to detach block without allocating a new one
    llmRequest->addNewToken(1005, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1006, beamIdx);
    kvCacheManager.addToken(requestId);
    numBlocks = seq0.getCacheBlockIds(onlyWindowSize)[beamIdx].size();
    EXPECT_EQ(numBlocks, 2);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1}));

    // add more new tokens, to allocate a new block
    llmRequest->addNewToken(1007, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    numBlocks = seq0.getCacheBlockIds(onlyWindowSize)[beamIdx].size();
    EXPECT_EQ(numBlocks, 3);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId, llmRequest));
    // no blocks stored because reuse is disabled
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
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

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxSequenceLength,
        maxSequenceLength, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);

    auto events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 1);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheCreatedData>(events.front().data));
    EXPECT_THAT(std::get<tle::KVCacheCreatedData>(events.front().data).numBlocksPerCacheLevel,
        ::testing::ElementsAreArray({8, 2}));

    EXPECT_EQ(getEvents(kvCacheManager).size(), 0);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setLoraTaskId(42);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
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
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

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
    kvCacheManager.addSequenceBatch({{{1, inputTokens1->size(), beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 0);

    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto llmRequest2 = std::make_shared<LlmRequest>(2, 0, inputTokens2, samplingConfig, true);
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto llmRequest3 = std::make_shared<LlmRequest>(3, 0, inputTokens3, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{2, static_cast<SizeType32>(inputTokens2->size()), beamWidth},
                                        {3, static_cast<SizeType32>(inputTokens3->size()), beamWidth}}},
        {std::ref(*llmRequest2), std::ref(*llmRequest3)});

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

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest3);
    (void) kvCacheManager.removeSequence(3, llmRequest3);

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
    kvCacheManager.addSequenceBatch({{{4, inputTokens4->size(), beamWidth}}}, {std::ref(*llmRequest4)});

    events = getEvents(kvCacheManager);

    // Onboard block 0, in replace, offload block 7
    // Offload block 6, and write content of [1,1,1,1] to block 1
    // Upon freeing up block 1, its child block 2, will be removed from the search tree,
    // which is a remove event.
    // Offload block 5, in replace onboard block 7, and write content of [0] to block 7.
    // In total, there are 2 offloads, 1 onboard, 1 removed, total of 4 events.
    // FIXME: For better improvement, when block 1 is overwritten, child blocks
    // are removed from the search tree and no longer reusable. Therefore these blocks
    // should be the first to be called upon when we want a new block.
    auto onboardedBlocks = 0;
    auto offloadedBlocks = 0;
    auto removedBlocks = 0;

    EXPECT_EQ(events.size(), 4);

    for (int i = 0; i < 4; i++)
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
    EXPECT_EQ(offloadedBlocks, 2);
    EXPECT_EQ(removedBlocks, 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest4);
    kvCacheManager.storeContextBlocks(*llmRequest4);

    events = getEvents(kvCacheManager);

    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
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
    auto constexpr maxSequenceLength = 128;

    // Enable sliding window kv cache for long input tokens.
    auto constexpr maxAttentionWindow = 16;
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxAttentionWindow + tokensPerBlock, tokensPerBlock);

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 16;

    auto constexpr enableBlockReuse = true;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
        sinkTokenLength, stream, maxSequenceLength, /*chunkSize=*/tokensPerBlock, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    auto const onlyWindowSize = theOnlyWindowSize(kvCacheManager);
    SizeType32 constexpr maxNewTokens = 40;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    ///////////////////////////////////////////////////////////////////////////
    // add a request just at the attention window and then remove it
    SizeType32 requestId = 0;
    int inputLength = 20;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3, 4}));

    // Simulate end of prefill and store context blocks in the reuse trie before any
    // addToken-triggered detachFrontBlock fires. Under the new SWA placeholder design,
    // OOW blocks must be in the trie before they are replaced with placeholders.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    kvCacheManager.storeContextBlocks(*llmRequest);

    // add tokens, making the window slide
    llmRequest->addNewToken(1020, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1021, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1022, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1023, beamIdx);
    kvCacheManager.addToken(requestId);
    auto numTokens = llmRequest->getNumTokens(beamIdx);
    auto numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPool;
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3, 4, 5}));

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));
    numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPool;
    EXPECT_EQ(numAllocatedPrimaryBlocks, 0);
    // store blocks 0, 1, 2, 3, 4, 5 for reuse ([1000,1001,1002,1003], [1004,1005,1006,1007], [1008,1009,1010,1011],
    // [1012,1013,1014,1015], [1016,1017,1018,1019], [1020,1021])

    ///////////////////////////////////////////////////////////////////////////
    // add a short request and then remove it
    // reuse first 2 blocks {0, 1(p)} in previous request, copying block 1 to a new block 6
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 6);
    EXPECT_THAT(seq1.getCacheBlockIds(onlyWindowSize).at(beamIdx),
        ::testing::ElementsAreArray(
            {0, 6})); // Can't use 5 since it's used to onboard block, so 6 is the next free block.

    llmRequest->addNewToken(1007, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    numTokens = llmRequest->getNumTokens(beamIdx);
    EXPECT_THAT(seq1.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 6, 7}));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));

    ///////////////////////////////////////////////////////////////////////////
    // add a medium request and then remove it
    // reuse first 3 blocks {0, 1, 2(p)} in first request, copying block 2 to a new block 8
    requestId = 2;
    inputLength = 10;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq2 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 9);
    EXPECT_THAT(seq2.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 8}));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));

    ///////////////////////////////////////////////////////////////////////////
    // add a longer request within attention window and try to reuse
    // reuse blocks {0, 1, 2, 3(p)}, copying block 3 to a new block 9
    // then upon reaching attention window, get new block 10
    requestId = 3;
    inputLength = 15;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq3 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 14);
    EXPECT_THAT(seq3.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 9}));

    // add new tokens to allocate another block, but not enough to detach block
    llmRequest->addNewToken(1015, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    EXPECT_THAT(seq3.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 9, 10}));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));
}

TEST_F(KVCacheManagerTest, KVCacheManagerSWAInvalidateReuseTest)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr maxSequenceLength = 128;
    SizeType32 constexpr maxNewTokens = 40;

    // Enable sliding window kv cache for long input tokens.
    auto constexpr attentionWindow = 8;
    auto constexpr numWindows = 1;
    auto const maxAttentionWindowVec = std::vector<SizeType32>{attentionWindow};
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    auto constexpr blocksInPrimaryPoolPerWindow = 4;
    auto constexpr blocksInSecondaryPoolPerWindow = 0;

    auto constexpr enableBlockReuse = true;

    auto const blocksPerWindow
        = BlocksPerWindow{{attentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, maxAttentionWindowVec, dtype, sinkTokenLength, stream, maxSequenceLength, maxSequenceLength,
        enableBlockReuse);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();
    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;
    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    auto constexpr beamIdx = 0;

    ///////////////////////////////////////////////////////////////////////////
    // add a request more than attention window size, and add token to trigger
    // detach.
    int inputLength = 11;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest0
        = std::make_shared<LlmRequest>(/*requestId=*/0, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    kvCacheManager.addSequenceBatch({{{/*requestId=*/0, inputLength, beamWidth}}}, {std::ref(*llmRequest0)});
    GenerationRequest const& seq0 = kvCacheManager.getSequence(/*requestId=*/0);

    // Block 0 goes out-of-window and is detached
    llmRequest0->addNewToken(1011, beamIdx);
    kvCacheManager.addToken(/*requestId=*/0);

    ///////////////////////////////////////////////////////////////////////////
    // add the second request that will take 2 blocks. Since we only have 4
    // blocks in the primary block pool and no secondary block pool, we will
    // need 1 block originally written by the previous sequence. Acquiring
    // the block means that the previous sequence is not valid to store for
    // reuse.
    inputLength = 8;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest1
        = std::make_shared<LlmRequest>(/*requestId=*/1, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{/*requestId=*/1, inputLength, beamWidth}}}, {std::ref(*llmRequest1)});
    GenerationRequest const& seq1 = kvCacheManager.getSequence(/*requestId=*/1);

    auto const onlyWindowSize = theOnlyWindowSize(kvCacheManager);
    (void) onlyWindowSize;
    // Note: isSequenceValidForStoreForReuse has been removed; the SWA placeholder
    // path now preserves reuse through missing OOW anchors once later matches make
    // those anchors fall outside the attention window. See
    // VSWAEvictedPlaceholderAnchorAllowsTrailingReuse for that invariant.

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(seq0.getRequestId(), llmRequest0)));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(seq1.getRequestId(), llmRequest1)));
}

TEST_F(KVCacheManagerTest, KVCacheManagerVariableWindowAttentionWithReuseTest)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 64;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxBeamWidth = 1;
    auto constexpr sinkTokenLength = 0;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr maxSequenceLength = 128;

    // Enable sliding window kv cache for long input tokens.
    auto constexpr minAttentionWindow = 8;
    auto constexpr maxAttentionWindow = 16;
    auto constexpr numWindows = 2;
    auto const maxAttentionWindowVec = std::vector<SizeType32>{maxAttentionWindow, minAttentionWindow};
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    auto constexpr blocksInPrimaryPoolPerWindow = 16;
    auto constexpr blocksInSecondaryPoolPerWindow = 16;

    auto constexpr enableBlockReuse = true;

    auto const blocksPerWindow
        = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}},
            {minAttentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, maxAttentionWindowVec, dtype, sinkTokenLength, stream, maxSequenceLength,
        /*chunkSize=*/tokensPerBlock, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    auto const allBlocksInPrimaryPools = blockManager.getNumPrimaryBlocks();
    EXPECT_THAT(allBlocksInPrimaryPools, blocksInPrimaryPoolPerWindow * numWindows);

    ASSERT_EQ(blockManager.isVariableWindow(), true);
    ASSERT_EQ(blockManager.isVariableGQA(), false);

    SizeType32 constexpr maxNewTokens = 11;

    // prepare tokens with token[i] = 1000 + i
    TokenIdType constexpr firstToken = 1000;

    auto constexpr beamWidth = maxBeamWidth;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    auto constexpr beamIdx = 0;

    auto const assertBlocks
        = [minAttentionWindow, maxAttentionWindow, beamIdx](GenerationRequest seq,
              std::initializer_list<int> expectedBlocksMin, std::initializer_list<int> expectedBlocksMax)
    {
        auto blocksMin = seq.getCacheBlockIds(minAttentionWindow).at(beamIdx);
        auto blocksMax = seq.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
        EXPECT_THAT(blocksMin, ::testing::ElementsAreArray(expectedBlocksMin));
        EXPECT_THAT(blocksMax, ::testing::ElementsAreArray(expectedBlocksMax));
    };

    ///////////////////////////////////////////////////////////////////////////
    // add a request just at the minimum attention window and then remove it
    SizeType32 requestId = 0;
    int inputLength = 11;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    assertBlocks(seq0, {0, 1, 2}, {0, 1, 2});

    // Simulate end of prefill and store context blocks in the reuse trie before any
    // addToken-triggered detachFrontBlock fires.  Under the new SWA placeholder design,
    // OOW blocks must be in the trie before they are replaced with placeholders;
    // otherwise their content is unrecoverable once the refcount drops to zero.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    kvCacheManager.storeContextBlocks(*llmRequest);

    // add tokens, making the minimum attention window slide (not reaching the max attention window)
    llmRequest->addNewToken(1011, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1012, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1013, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1014, beamIdx);
    kvCacheManager.addToken(requestId);
    assertBlocks(seq0, {0, 1, 2, 3}, {0, 1, 2, 3});
    auto numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPoolPerWindow * numWindows;

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));
    numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPoolPerWindow * numWindows;
    EXPECT_EQ(numAllocatedPrimaryBlocks, 0);
    // For both windows, store blocks 0, 1, 2, 3  for reuse ([1000,1001,1002,1003], [1004,1005,1006,1007],
    // [1008,1009,1010,1011], [1012,1013])

    ///////////////////////////////////////////////////////////////////////////
    // add a short request within both attention windows and try to reuse
    // reuse blocks {0, 1(p)} for both windows, copying block 1 to a new block 4 since it's not a leaf block and is
    // partially used. upon reached attention window, get new block 5
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 6);
    assertBlocks(seq1, {0, 4}, {0, 4});

    // add new tokens to allocate another block, but not enough to detach block
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1009, beamIdx);
    kvCacheManager.addToken(requestId);
    assertBlocks(seq1, {0, 4, 5}, {0, 4, 5});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(requestId, llmRequest)));
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

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxSequenceLength,
        maxSequenceLength, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1));
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setLoraTaskId(42);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
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

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

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

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxSequenceLength,
        maxSequenceLength, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setKvCacheRetentionConfig(tle::KvCacheRetentionConfig(
        std::vector{tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 35));
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
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
    kvCacheManager.addSequenceBatch({{{1, inputTokens1->size(), beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    events = getEvents(kvCacheManager);
    EXPECT_EQ(events.size(), 1); // The second partial block gets stored. No priorities updated.
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));

    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest2 = std::make_shared<LlmRequest>(2, 0, inputTokens2, samplingConfig, true);
    llmRequest2->setKvCacheRetentionConfig(tle::KvCacheRetentionConfig(
        std::vector{tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 10)},
        35)); // Update the context block priorities
    kvCacheManager.addSequenceBatch({{{2, inputTokens2->size(), beamWidth}}}, {std::ref(*llmRequest2)});

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

TEST_F(KVCacheManagerTest, GetPriorityByBlockId)
{
    auto constexpr numLayers = 2;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr numBlocks = 8;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr maxNumSequences = 4;
    auto constexpr beamWidth = 1;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    SizeType32 constexpr maxNewTokens = 4;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    tle::RetentionPriority constexpr highPriority = 80;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {numBlocks, 0}}};

    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxAttentionWindow,
        maxAttentionWindow, true);
    kvCacheManager.allocatePools(false);

    // Create a sequence and set a custom priority
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    auto llmRequest = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    // Set high priority for context blocks
    llmRequest->setKvCacheRetentionConfig(KvCacheRetentionConfig(
        {KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, highPriority)}, highPriority));

    kvCacheManager.addSequenceBatch({{{0, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    kvCacheManager.storeContextBlocks(*llmRequest);

    // Get block IDs for the sequence
    auto const& seq = kvCacheManager.getSequence(0);
    auto cacheBlockIds = seq.getCacheBlockIds(maxAttentionWindow).at(0);
    ASSERT_GE(cacheBlockIds.size(), 1);

    // Test 1: Valid block ID should return the set priority
    auto const validBlockId = cacheBlockIds[0];
    auto const retrievedPriority = kvCacheManager.getPriorityByBlockId(validBlockId, maxAttentionWindow);
    EXPECT_EQ(retrievedPriority, highPriority);

    // Test 2: Invalid block ID (negative) should return default priority
    auto const invalidNegative = kvCacheManager.getPriorityByBlockId(-1, maxAttentionWindow);
    EXPECT_EQ(invalidNegative, KvCacheRetentionConfig::kDefaultRetentionPriority);

    // Test 3: Invalid block ID (out of range) should return default priority
    auto const invalidOutOfRange = kvCacheManager.getPriorityByBlockId(9999, maxAttentionWindow);
    EXPECT_EQ(invalidOutOfRange, KvCacheRetentionConfig::kDefaultRetentionPriority);
}

TEST(KVCacheManagerHelpersTest, ChopVectorIntoBlocksBasicNoPartial)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto blocks = chopVectorIntoBlocks<int>(vec, 10, 4, false);
    std::vector<std::vector<int>> got(blocks.begin(), blocks.end());
    ASSERT_EQ(got.size(), 2);
    EXPECT_THAT(got[0], ::testing::ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(got[1], ::testing::ElementsAreArray({4, 5, 6, 7}));
}

TEST(KVCacheManagerHelpersTest, ChopVectorIntoBlocksBasicWithPartial)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto blocks = chopVectorIntoBlocks<int>(vec, 10, 4, true);
    std::vector<std::vector<int>> got(blocks.begin(), blocks.end());
    ASSERT_EQ(got.size(), 3);
    EXPECT_THAT(got[0], ::testing::ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(got[1], ::testing::ElementsAreArray({4, 5, 6, 7}));
    EXPECT_THAT(got[2], ::testing::ElementsAreArray({8, 9}));
}

TEST(KVCacheManagerHelpersTest, ChopVectorIntoBlocksWithUsableSize)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    std::vector<int> vec{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto blocks = chopVectorIntoBlocks<int>(vec, 7, 4, true);
    std::vector<std::vector<int>> got(blocks.begin(), blocks.end());
    ASSERT_EQ(got.size(), 2);
    EXPECT_THAT(got[0], ::testing::ElementsAreArray({0, 1, 2, 3}));
    EXPECT_THAT(got[1], ::testing::ElementsAreArray({4, 5, 6}));
}

TEST_F(KVCacheManagerTest, PinAndUnpinBlocksById)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    auto constexpr numLayers = 2;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;

    BlocksPerWindow const blocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxAttentionWindow, maxAttentionWindow, true);
    kvCacheManager.allocatePools(false);

    LlmRequest::RequestIdType requestId{0};
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    auto llmRequest = std::make_shared<LlmRequest>(requestId, 0, inputTokens, samplingConfig, isStreaming);

    kvCacheManager.addSequenceBatch(
        {{{requestId, static_cast<SizeType32>(inputTokens->size()), beamWidth}}}, {std::ref(*llmRequest)});
    auto const totalBlocks = kvCacheManager.getMaxNumBlocks();
    auto const freeAfterAlloc = kvCacheManager.getNumFreeBlocks();
    EXPECT_LT(freeAfterAlloc, totalBlocks);

    kvCacheManager.pinBlocks(requestId);
    auto lastBlockIdOpt = kvCacheManager.getLastBlockId(requestId);
    ASSERT_TRUE(lastBlockIdOpt.has_value());
    auto const& allBlockIds = kvCacheManager.getCacheBlockIds(requestId, maxAttentionWindow)[0];
    std::vector<SizeType32> pinnedBlockIds(allBlockIds.begin(), allBlockIds.end());
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest);
    (void) kvCacheManager.removeSequence(requestId, llmRequest);
    auto const freeAfterRemovePinned = kvCacheManager.getNumFreeBlocks();
    EXPECT_LT(freeAfterRemovePinned, totalBlocks);

    kvCacheManager.unpinBlocksById(pinnedBlockIds);
    auto const freeAfterUnpin = kvCacheManager.getNumFreeBlocks();
    EXPECT_EQ(freeAfterUnpin, totalBlocks);
}

// Regression test for NVBug 6018647: storeBlocks(pin=true) on a zero-ref block
// that sits in the eviction free queue must call claimBlock() before incRefCount().
// Without the fix, unpinBlocksById inserts the block into the free queue a second
// time, creating a ghost entry that inflates the free count and can cause hangs.
TEST_F(KVCacheManagerTest, StoreBlocksForReuseWithPinDoesNotCreateGhostFreeBlocks)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;
    auto constexpr numLayers = 2;
    auto constexpr numKvHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 6;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;

    BlocksPerWindow const blocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxAttentionWindow, maxAttentionWindow, true /* enableBlockReuse */);
    kvCacheManager.allocatePools(false);

    auto const totalBlocks = kvCacheManager.getMaxNumBlocks();

    // 8 tokens = 2 blocks (tokensPerBlock=4).
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    // Step 1: Add seq A (requestId=0). Tree is empty, no reuse.
    LlmRequest::RequestIdType requestIdA{0};
    auto llmRequestA = std::make_shared<LlmRequest>(requestIdA, 0, inputTokens, samplingConfig, isStreaming);
    // Step 2: Add seq A and B together. Tree is empty, no reuse for either.
    LlmRequest::RequestIdType requestIdB{1};
    auto llmRequestB = std::make_shared<LlmRequest>(requestIdB, 0, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequenceBatch({{{requestIdA, static_cast<SizeType32>(inputTokens->size()), beamWidth},
                                        {requestIdB, static_cast<SizeType32>(inputTokens->size()), beamWidth}}},
        {std::ref(*llmRequestA), std::ref(*llmRequestB)});

    // Both sequences allocated, 4 blocks consumed.
    auto const freeAfterBothAlloc = kvCacheManager.getNumFreeBlocks();
    EXPECT_EQ(freeAfterBothAlloc, totalBlocks - 4);

    // Step 3-4: Simulate prefill completion for both.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestA);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestB);

    // Step 5: Store A's blocks in the radix tree.
    kvCacheManager.storeContextBlocks(*llmRequestA);

    // Step 6: Remove seq A. Its blocks are stored in tree, refCount -> 0, released to free queue.
    (void) kvCacheManager.removeSequence(requestIdA, llmRequestA);
    auto const freeAfterRemoveA = kvCacheManager.getNumFreeBlocks();
    // A's 2 blocks + the 2 that were already free = totalBlocks - 2 (B's blocks).
    EXPECT_EQ(freeAfterRemoveA, totalBlocks - 2);

    // Step 7: storeBlocksForReuse with pin=true on seq B.
    // storeBlocks finds A's tree blocks (refCount=0, in free queue) as matches and pins them.
    // Without the fix: incRefCount alone, block stays in free queue -> ghost on unpin.
    // With the fix: claimBlock first, block removed from free queue -> correct lifecycle.
    auto pinnedBlockIds = kvCacheManager.storeBlocksForReuse(requestIdB, llmRequestB, /*pinBlocks=*/true);
    EXPECT_FALSE(pinnedBlockIds.empty());

    // Step 8: Unpin the blocks.
    kvCacheManager.unpinBlocksById(pinnedBlockIds);
    auto const freeAfterUnpin = kvCacheManager.getNumFreeBlocks();
    // A's blocks should be in the free queue exactly once. B's 2 blocks still allocated.
    // With the bug, ghost entries would inflate this beyond (totalBlocks - 2).
    EXPECT_EQ(freeAfterUnpin, totalBlocks - 2);
    EXPECT_LE(freeAfterUnpin, totalBlocks);

    // Step 9: Remove seq B. All blocks should now be free.
    (void) kvCacheManager.removeSequence(requestIdB, llmRequestB);
    auto const freeAfterAll = kvCacheManager.getNumFreeBlocks();
    EXPECT_EQ(freeAfterAll, totalBlocks);
    // Ghost entries would make free count exceed total blocks.
    EXPECT_LE(freeAfterAll, totalBlocks);
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

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManagerTest(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream,
        maxSequenceLength, maxSequenceLength, true, CacheType::kSELF, std::nullopt);

    EXPECT_EQ(getEvents(kvCacheManagerTest).size(), 0);

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxSequenceLength, maxSequenceLength, true, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1024));

    kvCacheManager.allocatePools(false);
    kvCacheManager.flushIterationEvents();
    auto events = kvCacheManager.getLatestEvents(std::chrono::seconds(1));

    EXPECT_EQ(events.size(), 1);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    kvCacheManager.flushIterationEvents();
    events = kvCacheManager.getLatestEvents(std::chrono::seconds(1));

    EXPECT_EQ(events.size(), 1);

    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));
}

TEST_F(KVCacheManagerTest, KVCacheManagerEventStreamWindowSize)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr maxNumSequences = 8;
    auto blocksInPool = std::vector<SizeType32>{8, 2};
    auto blocksInSlidingWindowPool = std::vector<SizeType32>{4, 2};

    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;
    auto const slidingWindow = tokensPerBlock * (maxBlocksPerSeq - 1);

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPool[0], blocksInPool[1]}},
        {slidingWindow, {blocksInSlidingWindowPool[0], blocksInSlidingWindowPool[1]}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow, slidingWindow}, dtype, 0, stream,
        maxSequenceLength, maxSequenceLength, true, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);

    auto events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 2);

    EXPECT_EQ(events.front().windowSize, slidingWindow);
    EXPECT_EQ(std::get<tle::KVCacheCreatedData>(events.front().data).numBlocksPerCacheLevel, blocksInSlidingWindowPool);

    EXPECT_EQ(events.back().windowSize, maxAttentionWindow);
    EXPECT_EQ(std::get<tle::KVCacheCreatedData>(events.back().data).numBlocksPerCacheLevel, blocksInPool);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    events = getEvents(kvCacheManager);

    EXPECT_EQ(events.size(), 2);

    // storeContextBlocks iterates windows in descending order (largest first) to preserve
    // per-window event ordering, so the full-attention window is emitted before the SWA window.
    EXPECT_EQ(events.front().windowSize, maxAttentionWindow);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.front().data));

    EXPECT_EQ(events.back().windowSize, slidingWindow);
    EXPECT_TRUE(std::holds_alternative<tle::KVCacheStoredData>(events.back().data));
}

TEST_F(KVCacheManagerTest, KVCacheTransferManagerConcurrencyTest)
{
    auto const blockSize = 16384;

    auto bufferManager = tensorrt_llm::runtime::BufferManager(std::make_shared<tr::CudaStream>());
    auto transferManager = KVCacheTransferManager(bufferManager);

    auto pool = KVCacheBlockPool(0, 2, 0, 0, 0);

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

TEST_P(KVCacheManagerTest, DISABLED_KVCacheManagerSinkTokenLengthTest)
{
    // TODO: Support sink attention and add coverage
    // TODO: Support and add coverage for beamWidth > 1
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

    auto constexpr numSharedBlocks = (sinkTokenLength + bubbleLength) / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;
    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr numSharedBlocksCtx = (inputLength + bubbleLength) / tokensPerBlock;

    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{maxBeamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, sinkTokenLength,
            stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto inputTokens0 = makeInputTokens(inputLength);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReq0)}));
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
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    auto currentNumBlocks = totalNumBlocks;
    for (auto requestCounter = 0; requestCounter < maxNumSequences; ++requestCounter)
    {
        auto const nextRequestId = static_cast<RequestIdType>(requestId + requestCounter);
        auto inputTokensLoop = makeInputTokens(inputLength);
        auto llmReqLoop = std::make_shared<LlmRequest>(
            LlmRequest::RequestIdType{nextRequestId}, maxNewTokens, inputTokensLoop, samplingConfig, isStreaming);
        EXPECT_NO_THROW(
            kvCacheManager.addSequenceBatch({{{nextRequestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReqLoop)}));
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
    // Full attention
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

    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxSequenceLength;

    auto constexpr inputLength = maxSequenceLength - 2;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;

    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{maxBeamWidth};
    bool constexpr isStreaming{false};

    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, maxSequenceLength, enableBlockReuse)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, sinkTokenLength,
            stream, maxSequenceLength, maxSequenceLength, enableBlockReuse);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        auto inputTokensLoop = makeInputTokens(inputLength);
        auto llmReqLoop = std::make_shared<LlmRequest>(static_cast<LlmRequest::RequestIdType>(requestId), maxNewTokens,
            inputTokensLoop, samplingConfig, isStreaming);
        EXPECT_NO_THROW(
            kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmReqLoop)}));
        auto const currentNumBlocks = totalNumBlocks - (requestId + 1) * (numSharedBlocks + maxBeamWidth);
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
                // Shared blocks
                for (auto block = 0; block < numSharedBlocks; ++block)
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
                // Unshared blocks
                auto const block = numSharedBlocks;
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
// beam search with SWA is not supported for now
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
    auto constexpr maxNumSequences = 8;
    auto constexpr sinkTokenLength = 0;
    auto const stream = std::make_shared<tr::CudaStream>();

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
        for (int inputLength = 44; inputLength < 45; ++inputLength)
        {
            auto constexpr maxAttentionWindow = 46;
            auto constexpr blocksInSecondaryPool = 0;

            auto constexpr maxSequenceLength = 256;
            auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);
            auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;
            auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

            KVCacheManager kvCacheManager = homogeneousLayers
                ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
                    maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
                    sinkTokenLength, stream, maxSequenceLength,
                    /*chunkSize=*/tokensPerBlock, kv_cache_block_reuse)
                : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
                    maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF,
                    sinkTokenLength, stream, maxSequenceLength,
                    /*chunkSize=*/tokensPerBlock, kv_cache_block_reuse);
            kvCacheManager.allocatePools(false);

            EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);

            auto inputTokens = std::make_shared<VecTokens>(VecTokens(inputLength, 0));

            auto draftTokens = std::make_shared<std::vector<SizeType32>>(draftLen);
            auto llmRequest
                = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
            llmRequest->setDraftTokens(draftTokens);

            auto const onlyWindowSize = theOnlyWindowSize(kvCacheManager);

            auto remainingBlocksToCompletion
                = kvCacheManager.getRemainingBlocksToCompletion(*llmRequest, onlyWindowSize);
            auto neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false, onlyWindowSize);
            auto currentNumAllocTotalBlocks = kvCacheManager.getNumAllocTotalBlocks();

            EXPECT_NO_THROW(
                kvCacheManager.addSequenceBatch({{{requestId, inputLength, maxBeamWidth}}}, {std::ref(*llmRequest)}));
            for (int di = 0; di < draftLen && di < maxNewTokens; ++di)
            {
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }
                EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
            }

            auto numUsedBlocksThisStep = kvCacheManager.getNumAllocTotalBlocks() - currentNumAllocTotalBlocks;
            EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);

            // Simulate adding new tokens during generation
            llmRequest->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
            for (int i = draftLen; i < maxNewTokens; i += (draftLen + 1))
            {
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }

                neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false, onlyWindowSize);
                currentNumAllocTotalBlocks = kvCacheManager.getNumAllocTotalBlocks();

                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    for (int di = 0; di < draftLen && (i + di) < maxNewTokens; ++di)
                    {
                        llmRequest->addNewToken(1, beam);
                    }
                }

                for (int di = 0; di < draftLen + 1 && (i + di) < maxNewTokens; ++di)
                {
                    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
                }
                numUsedBlocksThisStep = kvCacheManager.getNumAllocTotalBlocks() - currentNumAllocTotalBlocks;

                if (inputLength + i + draftLen + 1 < maxAttentionWindow)
                {
                    EXPECT_EQ(numUsedBlocksThisStep, neededBlocksOneStep);
                }
                else
                {
                    // This test calculates neededBlocksOneStep for the entire step (which may exceed
                    // maxAttentionWindow), but adds tokens only up to maxAttentionWindow. In this case,
                    // numUsedBlocksThisStep may be smaller than neededBlocksOneStep by 1 block.
                    ASSERT_THAT(numUsedBlocksThisStep,
                        testing::AnyOf(testing::Eq(neededBlocksOneStep), testing::Eq(neededBlocksOneStep - 1)));
                }
            }

            // After adding tokens, the currently used blocks plus remaining blocks to
            // completion should not exceed the initial estimate. With SWA, the initial
            // estimate includes a conservative numExtraBlocksPerBeam for transient sliding
            // window transitions, so actual consumption may be slightly less.
            // Note: getNumAllocTotalBlocks() is cumulative and includes blocks that were
            // detached and recycled during SWA, so we use getUsedNumBlocks() instead.
            EXPECT_LE(kvCacheManager.getUsedNumBlocks()
                    + kvCacheManager.getRemainingBlocksToCompletion(*llmRequest, onlyWindowSize),
                remainingBlocksToCompletion);
        }
    }
}
} // namespace

TEST_P(KVCacheManagerTest, neededBlocksOneStepKvCacheBlockReuse)
{
    testNeededBlocksOneStep(true, 1, 0, GetParam()); // maxBeamWidth is 1 when kv cache reuse is enabled
}

// Beam search with SWA is not supported yet
TEST_P(KVCacheManagerTest, DISABLED_neededBlocksOneStep)
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
        std::make_tuple(512, 0, 256, 64, 5), std::make_tuple(512, 0, 257, 64, 6), std::make_tuple(512, 64, 1024, 64, 8),
        std::make_tuple(513, 64, 1024, 64, 9), std::make_tuple(512, 64, 256, 64, 5),
        std::make_tuple(512, 64, 257, 64, 6), std::make_tuple(512, 65, 1024, 64, 9),
        std::make_tuple(513, 65, 1024, 64, 9), std::make_tuple(512, 65, 256, 64, 6),
        std::make_tuple(512, 65, 257, 64, 6)));

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
    // There are 29 context tokens left over to be put in output blocks, so 284 tokens to fit in output blocks in
    // total: 5 blocks
    auto const numOutputBlocks = (5 + kSWAExtraBlock) * beamWidth;
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
    BlocksPerWindow blocksPerWindow;
    SizeType32 sinkTokenLength;
    SizeType32 maxAttentionWindow;
    SizeType32 maxBeamWidth;
    SizeType32 maxNumTokens;
    bool kvCacheBlockReuse;
    std::vector<SizeType32> maxAttentionWindowVec = {maxAttentionWindow};
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
};

BlocksPerWindow blocksAndWindow(SizeType32 numPrimaryBlocks, SizeType32 windowSize)
{
    return {{windowSize, {numPrimaryBlocks, 0}}};
}

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
    auto const maxSequenceLength = kvCacheInstantiationParameters.maxNumTokens;
    auto const maxAttentionWindow = kvCacheInstantiationParameters.maxAttentionWindow;
    auto const [numBlocksInPrimaryPool, _] = kvCacheInstantiationParameters.blocksPerWindow.at(maxAttentionWindow);

    if (std::holds_alternative<SizeType32>(kvCacheInstantiationParameters.numHeadsPerLayer))
    {
        auto const numHeadsPerLayer = std::get<SizeType32>(kvCacheInstantiationParameters.numHeadsPerLayer);
        auto numHeadsPerLayerVec = std::vector<SizeType32>{kvCacheInstantiationParameters.numLayers};
        std::fill(numHeadsPerLayerVec.begin(), numHeadsPerLayerVec.end(), numHeadsPerLayer);
        return std::make_shared<KVCacheManager>(numHeadsPerLayerVec, kvCacheInstantiationParameters.sizePerHead,
            kvCacheInstantiationParameters.tokensPerBlock, kvCacheInstantiationParameters.blocksPerWindow,
            numBlocksInPrimaryPool, kvCacheInstantiationParameters.maxBeamWidth,
            std::vector<SizeType32>{kvCacheInstantiationParameters.maxAttentionWindow},
            kvCacheInstantiationParameters.dtype, kvCacheInstantiationParameters.sinkTokenLength, stream,
            kvCacheInstantiationParameters.maxNumTokens, kvCacheInstantiationParameters.maxNumTokens,
            kvCacheInstantiationParameters.kvCacheBlockReuse, CacheType::kSELF);
    }
    if (std::holds_alternative<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer))
    {
        auto const numHeadsPerLayer
            = std::get<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer);
        return std::make_shared<KVCacheManager>(numHeadsPerLayer, kvCacheInstantiationParameters.sizePerHead,
            kvCacheInstantiationParameters.tokensPerBlock, kvCacheInstantiationParameters.blocksPerWindow,
            numBlocksInPrimaryPool, kvCacheInstantiationParameters.maxBeamWidth,
            std::vector<SizeType32>{kvCacheInstantiationParameters.maxAttentionWindow},
            kvCacheInstantiationParameters.dtype, kvCacheInstantiationParameters.sinkTokenLength, stream,
            kvCacheInstantiationParameters.maxNumTokens, kvCacheInstantiationParameters.maxNumTokens,
            kvCacheInstantiationParameters.kvCacheBlockReuse, CacheType::kSELF);
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
    auto const remainingBlocksToCompletionFromStart
        = kvCacheManager.getRemainingBlocksToCompletion(llmRequest, theOnlyWindowSize(kvCacheManager));
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
        kvCacheManager.addSequenceBatch(
            {{{request.mRequestId, request.getPromptLen(), maxBeamWidth}}}, {std::ref(request)});
        request.mState = LlmRequestState::kGENERATION_IN_PROGRESS;
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(request);
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
        kvCacheManager->allocatePools(false);
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
    auto const result = kvCacheManager->getRemainingBlocksToCompletion(llmRequest, theOnlyWindowSize(*kvCacheManager));
    ASSERT_EQ(result, params.expectedRemainingBlocksToCompletion);
}

// TODO: Support and add coverage for beamWidth > 1
// TODO: Support and add coverage for sink attention
INSTANTIATE_TEST_SUITE_P(RemainingBlocksToCompletionCorrectlyEstimated, RemainingBlocksToCompletionTest,
    ::testing::Values(
        GetRemainingBlocksToCompletionOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                1,
                1,
                1,
                1,
                blocksAndWindow(4096, 4096),
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
                blocksAndWindow(4096, 4096),
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
                blocksAndWindow(4096, 128),
                0,
                128,
                1,
                4096 * 4,
                false,
            },
            1024,
            128,
            18,
        }));

class FillKvCacheAndCompleteRequestsTest : public ::testing::TestWithParam<FillKvCacheAndCompleteRequestsParameters>
{
protected:
    void SetUp() override
    {
        auto const stream = std::make_shared<tr::CudaStream>();
        auto const params = GetParam();
        kvCacheManager = createKvCacheManager(params.kvCacheManagerInstantiationParameters, stream);
        kvCacheManager->allocatePools(false);
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
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(llmRequest);
        (void) kvCacheManager->removeSequence(llmRequest.mRequestId, llmRequest);
    }
    auto const [expectedNumFreeBlocks, _] = params.kvCacheManagerInstantiationParameters.blocksPerWindow.at(
        params.kvCacheManagerInstantiationParameters.maxAttentionWindow);
    ASSERT_EQ(kvCacheManager->getNumFreeBlocks(), expectedNumFreeBlocks);
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

// TODO: Support and add coverage for beamWidth > 1
// TODO: Support and add coverage for sink attention
auto const paramValues = ::testing::Values(
    FillKvCacheAndCompleteRequestsParameters{
        KvCacheManagerInstantiationParameters{
            1,
            1,
            1,
            1,
            blocksAndWindow(4096, 4096),
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
            blocksAndWindow(4096, 4096),
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
            blocksAndWindow(4096, 4096),
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
            blocksAndWindow(4096, 4096),
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
            blocksAndWindow(4096, 128),
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
            blocksAndWindow(4096 * 128, 4096),
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
            blocksAndWindow(4096, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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
            blocksAndWindow(4096 * 16, 2048),
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

namespace
{
struct GetNeededBlocksOneStepOneRequestParameters
{
    KvCacheManagerInstantiationParameters kvCacheManagerInstantiationParameters;
    SizeType32 promptLength;
    SizeType32 draftLength;
    bool contextStep;
    SizeType32 previousGeneratedTokens;
    bool twoStepsLookAhead;
    SizeType32 expectedNeededBlocksOneStep;
};
} // namespace

class NeededBlocksOneStepTest : public ::testing::TestWithParam<GetNeededBlocksOneStepOneRequestParameters>
{
protected:
    void SetUp() override
    {
        auto const stream = std::make_shared<tr::CudaStream>();
        auto const params = GetParam();
        kvCacheManager = createKvCacheManager(params.kvCacheManagerInstantiationParameters, stream);
        kvCacheManager->allocatePools(/*useUvm=*/false);
    }

    void TearDown() override {}

    std::shared_ptr<KVCacheManager> kvCacheManager;
};

TEST_P(NeededBlocksOneStepTest, NeededBlocksOneStepTestCorrectlyEstimated)
{
    auto const params = GetParam();
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);
    auto const requestId = 0;
    auto const inputTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(params.promptLength));
    auto llmRequest = LlmRequest{
        requestId,
        params.kvCacheManagerInstantiationParameters.maxNumTokens,
        inputTokens,
        tensorrt_llm::runtime::SamplingConfig{params.kvCacheManagerInstantiationParameters.maxBeamWidth},
        true,
    };
    auto draftTokens = std::make_shared<std::vector<SizeType32>>(params.draftLength);
    llmRequest.setDraftTokens(draftTokens);
    if (params.contextStep)
    {
        auto neededBlocksOneStep = kvCacheManager->getNeededBlocksOneStep(llmRequest, false, onlyWindowSize);
        ASSERT_EQ(neededBlocksOneStep, params.expectedNeededBlocksOneStep);
    }
    else
    {
        kvCacheManager->addSequenceBatch(
            {{{requestId, params.promptLength, params.kvCacheManagerInstantiationParameters.maxBeamWidth}}},
            {std::ref(llmRequest)});
        llmRequest.setState(LlmRequestState::kGENERATION_IN_PROGRESS);
        for (int beam = 0; beam < params.kvCacheManagerInstantiationParameters.maxBeamWidth; beam++)
        {
            for (SizeType32 i = 0; i < params.previousGeneratedTokens; i++)
            {
                llmRequest.addNewToken(0, beam);
                kvCacheManager->addToken(llmRequest.mRequestId);
            }
        }

        auto neededBlocksOneStep
            = kvCacheManager->getNeededBlocksOneStep(llmRequest, params.twoStepsLookAhead, onlyWindowSize);
        ASSERT_EQ(neededBlocksOneStep, params.expectedNeededBlocksOneStep);
    }
}

INSTANTIATE_TEST_SUITE_P(NeededBlocksOneStepTestCorrectlyEstimated, NeededBlocksOneStepTest,
    ::testing::Values(
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 136,
            /* draftLength */ 0,
            /* contextStep */ true,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 9,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 512,
            /* draftLength */ 0,
            /* contextStep */ true,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 32,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 1024,
            /* draftLength */ 0,
            /* contextStep */ true,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 64,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 512,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 1,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 512,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 8,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 0,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 518,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 0,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 530,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 512,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 16,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 1,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 128,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 15,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 0,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 128,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 15,
            /* twoStepsLookAhead */ true,
            /* expectedNeededBlocksOneStep */ 1,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 128,
            /* draftLength */ 0,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 15,
            /* twoStepsLookAhead */ true,
            /* expectedNeededBlocksOneStep */ 1,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 302, // 14 tokens in last block
            /* draftLength */ 3,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 1,
        },
        GetNeededBlocksOneStepOneRequestParameters{
            KvCacheManagerInstantiationParameters{
                /* numLayers */ 1,
                /* numHeads */ 1,
                /* sizePerHead */ 1,
                /* tokensPerBlock */ 16,
                /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ 512),
                /* sinkTokenLength */ 0,
                /* maxAttentionWindow */ 512,
                /* maxBeamWidth */ 1,
                /* maxNumTokens */ 513,
                /* kvCacheBlockReuse */ false,
            },
            /* promptLength */ 298, // 10 tokens in last block
            /* draftLength */ 3,
            /* contextStep */ false,
            /* previousGeneratedTokens */ 0,
            /* twoStepsLookAhead */ false,
            /* expectedNeededBlocksOneStep */ 0,
        }));

TEST(KVCacheManagerReuseAccountingTest, ReuseAwareBlockEstimatesStayConsistentAfterContextAllocation)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 64; // 4 full context blocks
    auto constexpr maxNewTokens = 32; // 2 generation blocks
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    auto const baseTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength), 7);
    auto req0 = LlmRequest{
        0,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };
    kvCacheManager->addSequenceBatch({{{req0.mRequestId, req0.getPromptLen(), maxBeamWidth}}}, {std::ref(req0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->storeContextBlocks(req0);
    // Release the sequence to make blocks available in the radix tree for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->removeSequence(req0.mRequestId, req0);

    auto req1 = LlmRequest{
        1,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // Note: storeContextBlocks only stores (length - 1) tokens worth of blocks
    // For 64 tokens with 16 tokens/block, only 63/16 = 3 full blocks are stored
    auto const summary = kvCacheManager->analyzePrefixReuse(req1.getUniqueTokens(0), req1);
    auto const expectedReusableBlocks = (promptLength - 1) / tokensPerBlock; // 3 blocks
    EXPECT_EQ(summary.reusableBlocksAll, expectedReusableBlocks);

    // After removeSequence, reusable blocks have no active refs and are free in the eviction queue.
    // The scheduling functions must NOT subtract free reusable blocks to avoid double-counting
    // against the eviction policy's free count.
    auto const numContextBlocks = promptLength / tokensPerBlock; // 4 blocks
    auto const numGenBlocks = maxNewTokens / tokensPerBlock;     // 2 blocks

    // neededOneStep: all 4 context blocks (no subtraction for free reusable blocks)
    auto const neededOneStep
        = kvCacheManager->getNeededBlocksOneStep(req1, /*twoStepsLookAhead=*/false, onlyWindowSize);
    EXPECT_EQ(neededOneStep, numContextBlocks);

    // remainingBeforeAdd: 4 context + 2 generation = 6 (no subtraction)
    auto const remainingBeforeAdd = kvCacheManager->getRemainingBlocksToCompletion(req1, onlyWindowSize);
    EXPECT_EQ(remainingBeforeAdd, numContextBlocks + numGenBlocks);

    // Verify estimatedReusableTokens is still set after getRemainingBlocksToCompletion
    EXPECT_EQ(req1.getEstimatedReusableTokens(), expectedReusableBlocks * tokensPerBlock);

    // After addSequenceBatch, context blocks are allocated (reuse already applied during allocation)
    // Only generation blocks remain to be allocated
    kvCacheManager->addSequenceBatch({{{req1.mRequestId, req1.getPromptLen(), maxBeamWidth}}}, {std::ref(req1)});

    // Verify estimatedReusableTokens is cleared to 0 after addSequenceBatch
    EXPECT_EQ(req1.getEstimatedReusableTokens(), 0);

    auto const remainingAfterContextAlloc = kvCacheManager->getRemainingBlocksToCompletion(req1, onlyWindowSize);
    EXPECT_EQ(remainingAfterContextAlloc, maxNewTokens / tokensPerBlock);
}

TEST(KVCacheManagerReuseAccountingTest, NeededBlocksOneStepCapsAllocatedReuseAtExactBlockBoundary)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 48; // 3 full context blocks
    auto constexpr reusablePrefixLength = promptLength + 1;
    auto constexpr maxNewTokens = 32;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);
    auto const samplingConfig = tensorrt_llm::runtime::SamplingConfig{maxBeamWidth};
    auto constexpr isStreaming = true;
    auto makeRequest = [&](LlmRequest::RequestIdType requestId, std::vector<TokenIdType> const& tokens)
    {
        return LlmRequest(
            requestId, maxNewTokens, std::make_shared<std::vector<TokenIdType>>(tokens), samplingConfig, isStreaming);
    };

    auto reusableTokens = std::vector<TokenIdType>(static_cast<std::size_t>(reusablePrefixLength));
    std::iota(reusableTokens.begin(), reusableTokens.end(), 0);

    auto seedReq = makeRequest(0, reusableTokens);
    kvCacheManager->addSequenceBatch(
        {{{seedReq.mRequestId, seedReq.getPromptLen(), maxBeamWidth}}}, {std::ref(seedReq)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(seedReq);
    kvCacheManager->removeSequence(seedReq.mRequestId, seedReq);

    // Keep a request with 49 prompt tokens active so all 3 full prefix blocks remain
    // both reusable and allocated.
    auto holderReq = makeRequest(1, reusableTokens);
    kvCacheManager->addSequenceBatch(
        {{{holderReq.mRequestId, holderReq.getPromptLen(), maxBeamWidth}}}, {std::ref(holderReq)});
    EXPECT_EQ(holderReq.getContextCurrentPosition(), promptLength);

    auto promptTokens = std::vector<TokenIdType>(static_cast<std::size_t>(promptLength));
    std::iota(promptTokens.begin(), promptTokens.end(), 0);

    auto req1 = makeRequest(2, promptTokens);

    // Simulate a recompute-style context request: prompt length stays at the exact block
    // boundary, but one generated token already exists in the token history.
    req1.addNewToken(promptLength, 0);

    auto const summaryAlloc = kvCacheManager->analyzePrefixReuse(req1.getUniqueTokens(0), req1);
    EXPECT_EQ(summaryAlloc.reusableBlocksAllocated, promptLength / tokensPerBlock);

    auto const neededOneStep
        = kvCacheManager->getNeededBlocksOneStep(req1, /*twoStepsLookAhead=*/false, onlyWindowSize);
    EXPECT_EQ(neededOneStep, 1);

    auto const numAllocBlocksBeforeAdd = kvCacheManager->getNumAllocTotalBlocks();
    kvCacheManager->addSequenceBatch({{{req1.mRequestId, req1.getPromptLen(), maxBeamWidth}}}, {std::ref(req1)});
    auto const numAllocBlocksAfterAdd = kvCacheManager->getNumAllocTotalBlocks();
    EXPECT_EQ(numAllocBlocksAfterAdd - numAllocBlocksBeforeAdd, neededOneStep);
}

TEST(KVCacheManagerReuseAccountingTest, CountReusableBlocksNoMatchReturnsZero)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 64; // 4 full context blocks
    auto constexpr maxNewTokens = 32;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);

    // Create a request with unique tokens - nothing is cached yet
    auto const uniqueTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength), 42);
    auto req = LlmRequest{
        0,
        maxNewTokens,
        uniqueTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // No blocks should be reusable since nothing has been cached
    auto const summaryEmpty = kvCacheManager->analyzePrefixReuse(req.getUniqueTokens(0), req);
    EXPECT_EQ(summaryEmpty.reusableBlocksAll, 0);
}

TEST(KVCacheManagerReuseAccountingTest, CountReusableBlocksPartialMatch)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 64; // 4 full context blocks
    auto constexpr maxNewTokens = 32;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    // First request: cache tokens [0, 1, 2, ..., 63]
    auto const baseTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    std::iota(baseTokens->begin(), baseTokens->end(), 0);

    auto req0 = LlmRequest{
        0,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };
    kvCacheManager->addSequenceBatch({{{req0.mRequestId, req0.getPromptLen(), maxBeamWidth}}}, {std::ref(req0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->storeContextBlocks(req0);
    // Release the sequence to make blocks available in the radix tree for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->removeSequence(req0.mRequestId, req0);

    // Second request: shares first 2 blocks worth of tokens, then diverges
    auto partialMatchTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    // First 32 tokens match (2 blocks)
    std::iota(partialMatchTokens->begin(), partialMatchTokens->begin() + 32, 0);
    // Remaining tokens are different
    std::fill(partialMatchTokens->begin() + 32, partialMatchTokens->end(), 999);

    auto req1 = LlmRequest{
        1,
        maxNewTokens,
        partialMatchTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // Should find 2 reusable blocks (first 32 tokens match)
    auto const summaryShared = kvCacheManager->analyzePrefixReuse(req1.getUniqueTokens(0), req1);
    EXPECT_EQ(summaryShared.reusableBlocksAll, 2);

    // After removeSequence, reusable blocks are free (no active refs).
    // getNeededBlocksOneStep must NOT subtract free reusable blocks to avoid double-counting.
    auto const neededOneStep
        = kvCacheManager->getNeededBlocksOneStep(req1, /*twoStepsLookAhead=*/false, onlyWindowSize);
    EXPECT_EQ(neededOneStep, promptLength / tokensPerBlock); // All 4 context blocks

    // Blocks are free (released via removeSequence), so onlyAllocated=true yields 0 reusable blocks.
    EXPECT_EQ(req1.getEstimatedReusableTokens(), 0);
}

TEST(KVCacheManagerReuseAccountingTest, GetRemainingBlocksToCompletionWithPartialReuse)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 80; // 5 full context blocks
    auto constexpr maxNewTokens = 48; // 3 generation blocks
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    // First request: cache tokens
    auto const baseTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    std::iota(baseTokens->begin(), baseTokens->end(), 0);

    auto req0 = LlmRequest{
        0,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };
    kvCacheManager->addSequenceBatch({{{req0.mRequestId, req0.getPromptLen(), maxBeamWidth}}}, {std::ref(req0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->storeContextBlocks(req0);
    // Release the sequence to make blocks available in the radix tree for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->removeSequence(req0.mRequestId, req0);

    // Second request with identical tokens
    auto req1 = LlmRequest{
        1,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // After removeSequence, reusable blocks are free (no active refs).
    // getRemainingBlocksToCompletion must NOT subtract free reusable blocks from the BLOCK budget.
    // Needs all 5 context + 3 generation = 8 blocks.
    auto const remaining = kvCacheManager->getRemainingBlocksToCompletion(req1, onlyWindowSize);
    auto const numContextBlocks = promptLength / tokensPerBlock; // 5 blocks
    auto const numGenBlocks = maxNewTokens / tokensPerBlock;     // 3 blocks
    EXPECT_EQ(remaining, numContextBlocks + numGenBlocks);       // 5 context + 3 generation = 8

    // storeContextBlocks stores (promptLength - 1) / tokensPerBlock = 4 full blocks.
    // getRemainingBlocksToCompletion counts ALL reusable blocks (free or allocated) for the
    // TOKEN budget, so estimatedReusableTokens = min(4, 5) * tokensPerBlock = 64.
    auto const numStoredBlocks = (promptLength - 1) / tokensPerBlock; // 4
    EXPECT_EQ(req1.getEstimatedReusableTokens(), std::min(numStoredBlocks, numContextBlocks) * tokensPerBlock);
}

TEST(KVCacheManagerReuseAccountingTest, GetNeededBlocksOneStepWithFullReuse)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 48; // 3 full context blocks
    auto constexpr maxNewTokens = 16;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    // First request: cache tokens
    auto const baseTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    std::iota(baseTokens->begin(), baseTokens->end(), 0);

    auto req0 = LlmRequest{
        0,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };
    kvCacheManager->addSequenceBatch({{{req0.mRequestId, req0.getPromptLen(), maxBeamWidth}}}, {std::ref(req0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->storeContextBlocks(req0);
    // Release the sequence to make blocks available in the radix tree for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->removeSequence(req0.mRequestId, req0);

    // Second request with identical tokens - all context blocks should be reusable
    auto req1 = LlmRequest{
        1,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // After removeSequence, reusable blocks are free (no active refs).
    // getNeededBlocksOneStep must NOT subtract free reusable blocks.
    auto const neededOneStep
        = kvCacheManager->getNeededBlocksOneStep(req1, /*twoStepsLookAhead=*/false, onlyWindowSize);
    auto const numSharedBlocks = promptLength / tokensPerBlock; // 3 blocks
    EXPECT_EQ(neededOneStep, numSharedBlocks);                  // All 3 context blocks

    // Blocks are free (released via removeSequence), so onlyAllocated=true yields 0 reusable blocks.
    EXPECT_EQ(req1.getEstimatedReusableTokens(), 0);
}

TEST(KVCacheManagerReuseAccountingTest, ReuseDisabledReturnsFullBlockCount)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr promptLength = 64; // 4 full context blocks
    auto constexpr maxNewTokens = 32;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    // Create manager with reuse DISABLED
    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ false, // Reuse disabled
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    auto const baseTokens = std::make_shared<std::vector<TokenIdType>>(static_cast<std::size_t>(promptLength));
    std::iota(baseTokens->begin(), baseTokens->end(), 0);

    auto req = LlmRequest{
        0,
        maxNewTokens,
        baseTokens,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // With reuse disabled, should need all context blocks
    auto const neededOneStep = kvCacheManager->getNeededBlocksOneStep(req, /*twoStepsLookAhead=*/false, onlyWindowSize);
    EXPECT_EQ(neededOneStep, promptLength / tokensPerBlock); // All 4 context blocks

    // Verify estimatedReusableTokens stays 0 when reuse is disabled
    EXPECT_EQ(req.getEstimatedReusableTokens(), 0);

    // getRemainingBlocksToCompletion should include both context and generation blocks
    auto const remaining = kvCacheManager->getRemainingBlocksToCompletion(req, onlyWindowSize);
    auto const expectedContextBlocks = promptLength / tokensPerBlock;
    auto const expectedGenBlocks = maxNewTokens / tokensPerBlock;
    EXPECT_EQ(remaining, expectedContextBlocks + expectedGenBlocks); // 4 + 2 = 6 blocks

    // Verify estimatedReusableTokens still 0 after getRemainingBlocksToCompletion
    EXPECT_EQ(req.getEstimatedReusableTokens(), 0);
}

TEST(KVCacheManagerReuseAccountingTest, MultipleRequestsWithSharedPrefix)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr tokensPerBlock = 16;
    auto constexpr sharedPrefixLength = 32;                                // 2 blocks of shared prefix
    auto constexpr uniqueSuffixLength = 32;                                // 2 blocks of unique suffix
    auto constexpr promptLength = sharedPrefixLength + uniqueSuffixLength; // 4 total blocks
    auto constexpr maxNewTokens = 16;
    auto constexpr maxBeamWidth = 1;
    auto constexpr maxAttentionWindow = 512;
    auto constexpr maxNumTokens = 1024;

    auto kvCacheManager = createKvCacheManager(
        KvCacheManagerInstantiationParameters{
            /* numLayers */ 1,
            /* numHeads */ 1,
            /* sizePerHead */ 1,
            /* tokensPerBlock */ tokensPerBlock,
            /* blocksPerWindow */ blocksAndWindow(/* numPrimaryBlocks */ 256, /* windowSize */ maxAttentionWindow),
            /* sinkTokenLength */ 0,
            /* maxAttentionWindow */ maxAttentionWindow,
            /* maxBeamWidth */ maxBeamWidth,
            /* maxNumTokens */ maxNumTokens,
            /* kvCacheBlockReuse */ true,
        },
        stream);
    kvCacheManager->allocatePools(/*useUvm=*/false);
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);

    // Create shared prefix
    std::vector<TokenIdType> sharedPrefix(sharedPrefixLength);
    std::iota(sharedPrefix.begin(), sharedPrefix.end(), 0);

    // First request with shared prefix + unique suffix
    auto tokens0 = std::make_shared<std::vector<TokenIdType>>(sharedPrefix);
    for (int i = 0; i < uniqueSuffixLength; ++i)
    {
        tokens0->push_back(1000 + i);
    }

    auto req0 = LlmRequest{
        0,
        maxNewTokens,
        tokens0,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };
    kvCacheManager->addSequenceBatch({{{req0.mRequestId, req0.getPromptLen(), maxBeamWidth}}}, {std::ref(req0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->storeContextBlocks(req0);
    // Release the sequence to make blocks available in the radix tree for reuse
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(req0);
    kvCacheManager->removeSequence(req0.mRequestId, req0);

    // Second request with same shared prefix + different unique suffix
    auto tokens1 = std::make_shared<std::vector<TokenIdType>>(sharedPrefix);
    for (int i = 0; i < uniqueSuffixLength; ++i)
    {
        tokens1->push_back(2000 + i);
    }

    auto req1 = LlmRequest{
        1,
        maxNewTokens,
        tokens1,
        tensorrt_llm::runtime::SamplingConfig{maxBeamWidth},
        true,
    };

    // Should reuse 2 blocks (shared prefix) — analyzePrefixReuse counts all reusable regardless of ref state
    auto const summaryPrefix = kvCacheManager->analyzePrefixReuse(req1.getUniqueTokens(0), req1);
    EXPECT_EQ(summaryPrefix.reusableBlocksAll, sharedPrefixLength / tokensPerBlock);

    // After removeSequence, reusable blocks are free (no active refs).
    // getNeededBlocksOneStep must NOT subtract free reusable blocks.
    auto const neededOneStep
        = kvCacheManager->getNeededBlocksOneStep(req1, /*twoStepsLookAhead=*/false, onlyWindowSize);
    EXPECT_EQ(neededOneStep, promptLength / tokensPerBlock); // All 4 context blocks

    // Blocks are free (released via removeSequence), so onlyAllocated=true yields 0 reusable blocks.
    EXPECT_EQ(req1.getEstimatedReusableTokens(), 0);

    // getRemainingBlocksToCompletion: 4 context + 1 gen = 5 blocks (no subtraction; blocks are free)
    auto const remaining = kvCacheManager->getRemainingBlocksToCompletion(req1, onlyWindowSize);
    EXPECT_EQ(remaining, (promptLength / tokensPerBlock) + (maxNewTokens / tokensPerBlock));
}

// All remove events for the same window size during a single iteration must be consolidated
// into a single KVCacheRemovedData (not emitted as separate events).
TEST_F(KVCacheManagerTest, KVCacheManagerEventRemovedBatchedWithinWindow)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    // Tight pool of 4: seq0 and seq1 together use all 4 blocks, leaving none fresh for seq2.
    // seq2 therefore must evict tree blocks to obtain its 4 needed blocks.
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr beamWidth = 1;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxAttentionWindow,
        maxAttentionWindow, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);
    (void) getEvents(kvCacheManager);

    // Seq0: stores blockA([0,1,2,3]) as a leaf in the radix tree.
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

    // Seq1: stores blockB([10,11,12,13]) as a separate leaf in the radix tree.
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{10, 11, 12, 13, 14});
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{1, inputTokens1->size(), beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    (void) getEvents(kvCacheManager); // drain seq0/seq1 stored events

    // Seq2 needs 4 blocks (15 tokens) with no radix tree match. All 4 pool blocks are in
    // the free queue after seq0 and seq1 released them. Two of those 4 blocks (blockA and
    // blockB) are leaves in the radix tree, so each call to getFreeBlock (which detaches only this block now) emits a
    // remove event. Both removes accumulate into mLatestRemovedEvents[W] and are committed as one consolidated
    // KVCacheRemovedData when flush() is called.
    auto inputTokens2 = std::make_shared<VecTokens>(
        VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114});
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{2, inputTokens2->size(), beamWidth}}}, {std::ref(*llmRequest2)});

    auto events = getEvents(kvCacheManager);

    SizeType32 numRemovedEvents = 0;
    SizeType32 numTotalRemovedHashes = 0;
    for (auto const& event : events)
    {
        if (std::holds_alternative<tle::KVCacheRemovedData>(event.data))
        {
            ++numRemovedEvents;
            numTotalRemovedHashes
                += static_cast<SizeType32>(std::get<tle::KVCacheRemovedData>(event.data).blockHashes.size());
        }
    }

    // blockA and blockB were both evicted from the same window in the same iteration.
    // They must appear in exactly one consolidated Removed event, not two separate events.
    EXPECT_EQ(numRemovedEvents, 1) << "Expected 1 consolidated Removed event for same-window evictions, got "
                                   << numRemovedEvents;
    EXPECT_EQ(numTotalRemovedHashes, 2) << "Expected 2 hashes in the Removed event (blockA and blockB), got "
                                        << numTotalRemovedHashes;
}

// When evictions and a store happen for the same window in the same iteration, the Removed
// event must appear before the Stored event. This is the ordering guarantee provided by
// enqueueStoredEvent calling flushRemovedEvents before appending the Stored event.
TEST_F(KVCacheManagerTest, KVCacheManagerEventRemovedOrderedBeforeStore)
{
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr maxAttentionWindow = 32;
    auto constexpr beamWidth = 1;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};

    tle::RetentionPriority constexpr lowPriority = 0;
    tle::RetentionPriority constexpr highPriority = 80;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, dtype, 0, stream, maxAttentionWindow,
        maxAttentionWindow, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);
    (void) getEvents(kvCacheManager);

    // Seq0: store root → block0(lowPrio) → block1(highPrio) in the radix tree.
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, true);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, lowPriority),
                                   KvCacheRetentionConfig::TokenRangeRetentionConfig(4, std::nullopt, highPriority)},
            highPriority));
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    (void) getEvents(kvCacheManager); // drain

    // Seq1 with different tokens.
    // addSequenceBatch: evicts seq0's block0 (and its descendant block1) — removes buffered, not yet emitted.
    // storeContextBlocks: calls flushRemovedEvents(W) first, committing the buffered removes,
    //                     then appends the Stored event for seq1's new blocks.
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108});
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{1, inputTokens1->size(), beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);

    auto events = getEvents(kvCacheManager);

    // Find the positions of the first Removed and first Stored events.
    std::optional<SizeType32> removedPos;
    std::optional<SizeType32> storedPos;
    SizeType32 pos = 0;
    for (auto const& event : events)
    {
        if (!removedPos && std::holds_alternative<tle::KVCacheRemovedData>(event.data))
        {
            removedPos = pos;
        }
        if (!storedPos && std::holds_alternative<tle::KVCacheStoredData>(event.data))
        {
            storedPos = pos;
        }
        ++pos;
    }

    ASSERT_TRUE(removedPos.has_value()) << "Expected at least one Removed event";
    ASSERT_TRUE(storedPos.has_value()) << "Expected at least one Stored event";

    EXPECT_LT(*removedPos, *storedPos)
        << "Removed event (pos=" << *removedPos << ") must precede Stored event (pos=" << *storedPos
        << ") for the same window. enqueueStoredEvent must flush pending removes before appending the store.";
}

// A store event for window W2 must not flush pending remove events for a different window W1.
// Removes for W1 must only be committed when a store for W1 occurs or when flush() is called.
// This verifies per-window isolation in the lazy-batching remove event logic.
TEST_F(KVCacheManagerTest, KVCacheManagerEventStoreForDifferentWindowDoesNotFlushPendingRemoves)
{
    // Two windows: wFull (non-SWA, equal to maxSequenceLength) and wSWA (SWA, smaller).
    // storeContextBlocks skips SWA windows, so it only emits a Stored event for wFull.
    // This means wSWA removes are never flushed by the wFull store — they stay buffered
    // until flush() at end of iteration.
    //
    // Expected event order: [Removed(wFull), Stored(wFull), Removed(wSWA)]
    //   Removed(wFull) — flushed by wFull's own storeContextBlocks call
    //   Stored(wFull)  — emitted by storeContextBlocks for wFull
    //   Removed(wSWA)  — only flushed by the iteration-end flush(), AFTER storeContextBlocks
    //
    // If isolation were broken (wFull store flushes ALL windows' removes), the order
    // would be [Removed(wSWA), Removed(wFull), Stored(wFull)] — Stored(wFull) would
    // appear after Removed(wSWA), violating the per-window ordering guarantee.
    auto constexpr numLayers = 2;
    auto constexpr numHeads = 2;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    // Tight pool: seq0 uses 3 out of 4 blocks, leaving only 1 fresh block. seq1 therefore
    // has to evict seq0's cached tree blocks to obtain the 3 it needs.
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr beamWidth = 1;
    auto constexpr dtype = nvinfer1::DataType::kHALF;
    auto const stream = std::make_shared<tr::CudaStream>();
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};

    auto constexpr wSWA = tokensPerBlock * 2;  // 8 tokens — SWA (< maxSequenceLength)
    auto constexpr wFull = tokensPerBlock * 4; // 16 tokens — full attention = maxSequenceLength
    auto constexpr maxSequenceLength = wFull;

    auto const blocksPerWindow = BlocksPerWindow{
        {wSWA, {blocksInPrimaryPool, blocksInSecondaryPool}}, {wFull, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{wSWA, wFull}, dtype, 0, stream, maxSequenceLength,
        maxSequenceLength, true, CacheType::kSELF, std::nullopt, std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);
    (void) getEvents(kvCacheManager);

    // Seq0: 9 tokens → 3 blocks per window. storeContextBlocks stores 2 full blocks in wFull
    // (skips wSWA). removeSequence stores 2 full blocks in wSWA as well (releaseBlocks covers
    // all windows). After release, each window's free queue is [block3_fresh, block2, block1, block0],
    // with block0 and block1 in the respective radix trees.
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{0, inputTokens0->size(), beamWidth}}}, {std::ref(*llmRequest0)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    (void) getEvents(kvCacheManager); // drain

    // Seq1 with different tokens (9 tokens → 3 blocks per window).
    // addSequenceBatch for each window: gets block3 (fresh, no event), block2 (not in tree, no event),
    //   then block1 (in tree as leaf) → freeChildren(block1) → Removed(block1) buffered for that window.
    // storeContextBlocks:
    //   wSWA: skipped (SWA) — wSWA removes stay buffered
    //   wFull: flushRemovedEvents(wFull) → Removed(wFull) committed; Stored(wFull) committed
    // flush(): flushRemovedEvents(wSWA) → Removed(wSWA) committed
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108});
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, true);
    kvCacheManager.addSequenceBatch({{{1, inputTokens1->size(), beamWidth}}}, {std::ref(*llmRequest1)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);

    auto events = getEvents(kvCacheManager);

    // Find the position of the first Removed and Stored event for each window.
    std::optional<SizeType32> removedSWAPos, storedFullPos, removedFullPos;
    SizeType32 pos = 0;
    for (auto const& event : events)
    {
        if (std::holds_alternative<tle::KVCacheRemovedData>(event.data))
        {
            if (event.windowSize == wSWA && !removedSWAPos)
                removedSWAPos = pos;
            if (event.windowSize == wFull && !removedFullPos)
                removedFullPos = pos;
        }
        else if (std::holds_alternative<tle::KVCacheStoredData>(event.data))
        {
            if (event.windowSize == wFull && !storedFullPos)
            {
                storedFullPos = pos;
            }
        }
        ++pos;
    }

    ASSERT_TRUE(removedSWAPos.has_value()) << "Expected Removed event for wSWA";
    ASSERT_TRUE(removedFullPos.has_value()) << "Expected Removed event for wFull";
    ASSERT_TRUE(storedFullPos.has_value()) << "Expected Stored event for wFull";

    // Within wFull, removes must precede stores.
    EXPECT_LT(*removedFullPos, *storedFullPos) << "Removed(wFull) must precede Stored(wFull)";

    // The wFull store must NOT have flushed wSWA's pending removes prematurely.
    // Correct isolation: Stored(wFull) appears before Removed(wSWA).
    // Broken isolation: Removed(wSWA) appears before Stored(wFull).
    EXPECT_LT(*storedFullPos, *removedSWAPos)
        << "Stored(wFull) (pos=" << *storedFullPos << ") must precede Removed(wSWA) (pos=" << *removedSWAPos
        << "). The wFull store must not prematurely flush pending removes for wSWA.";
}

namespace
{
void testBlockManagerLinearAttention_ContextNoReuse(int beamWidth, int numTokens)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 32;
    auto constexpr blocksInPrimaryPool = 24;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto maxAttentionWindow = numTokens * 2;
    auto numBlocksPerBeam = tc::ceilDiv(numTokens, tokensPerBlock);
    SizeType32 constexpr linearWindowSizeCode = LinearAttentionMetadata::LinearCacheType::kRecurrentStates;

    LinearAttentionMetadata linearAttentionMetadata{
        // .linearLayerIndices = {2, 5, 8, 11},
        .cacheType = linearWindowSizeCode,
        .allRecurrentStatesBytes = 440 * 1024, // dummy value
        .statesSnapshotInterval = tokensPerBlock * 2,
        .saveLastSnapshot = true,
        .numPlaceholderBlocks = blocksInPrimaryPool * 100,
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}},
        {linearWindowSizeCode, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{linearWindowSizeCode, maxAttentionWindow}, nvinfer1::DataType::kHALF, 0,
        /*chunkSize*/ 0, CacheType::kSELF, std::nullopt, nullptr, false, true, nullptr, std::nullopt, false, 128, 0,
        false, linearAttentionMetadata);
    blockManager.allocatePools(false);

    ASSERT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    ASSERT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool * 2);
    ASSERT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool * 2);

    // Setup for LlmRequest construction
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    auto makeInputTokens = [](SizeType32 len)
    {
        auto tokens = std::make_shared<VecTokens>();
        for (SizeType32 i = 0; i < len; ++i)
        {
            tokens->push_back(i);
        }
        return tokens;
    };

    auto constexpr requestId = 42;

    // reuse disabled: basic allocation
    // use 1 + beamWidth blocks
    auto inputTokens0 = makeInputTokens(numTokens);
    auto llmReq0 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{requestId}, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    GenerationRequest seq0{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq0)},
        linearWindowSizeCode, /*isEnableBlockReuse=*/false);
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq0)},
        maxAttentionWindow, /*isEnableBlockReuse=*/false);
    // When block reuse is disabled, only the last context block has real memory.
    // Whether the last block is shared depends on whether inputLength is aligned to tokensPerBlock.
    bool isShareLastContextBlock = (beamWidth == 1) || (numTokens % tokensPerBlock == 0);
    auto occupiedBlocksLinear = isShareLastContextBlock ? 1 : beamWidth;
    TLLM_LOG_DEBUG("==========================================================");
    ASSERT_EQ(
        blocksInPrimaryPool - blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], occupiedBlocksLinear);

    auto const& ids1 = seq0.getCacheBlockIds(linearWindowSizeCode);
    std::set<std::int32_t> idSetPositive{};
    std::set<std::int32_t> idSetNegative{};
    ASSERT_EQ(ids1.size(), beamWidth);
    for (auto const& beam : ids1)
    {
        ASSERT_EQ(beam.size(), numBlocksPerBeam);
        for (auto id : beam)
        {
            if (id >= 0)
            {
                idSetPositive.insert(id);
            }
            else
            {
                idSetNegative.insert(id);
            }
        }
    }
    ASSERT_EQ(idSetPositive.size(), occupiedBlocksLinear);
    // All blocks except the last are placeholders (negative IDs), shared among beams
    ASSERT_EQ(idSetNegative.size(), numBlocksPerBeam - 1);

    blockManager.releaseBlocks(seq0);
    ASSERT_EQ(blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], blocksInPrimaryPool);

    TLLM_LOG_DEBUG("==========================================================");
    // reuse disabled: re-add after release, verify block sharing and count
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq0)},
        linearWindowSizeCode, /*isEnableBlockReuse=*/false);
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq0)},
        maxAttentionWindow, /*isEnableBlockReuse=*/false);
    ASSERT_EQ(
        blocksInPrimaryPool - blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], occupiedBlocksLinear);
    auto const& ids2 = seq0.getCacheBlockIds(linearWindowSizeCode);
    ASSERT_EQ(ids2.size(), beamWidth);
    if (isShareLastContextBlock)
    {
        // When last block is shared, all beams should have identical block IDs
        for (std::size_t i = 0u; i < ids2.front().size(); ++i)
        {
            for (std::size_t beam = 1u; beam < ids2.size(); ++beam)
            {
                ASSERT_EQ(ids2.at(beam).at(i), ids2.at(0).at(i));
            }
        }
    }
    blockManager.releaseBlocks(seq0);
    ASSERT_EQ(blocksInPrimaryPool - blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], 0);
    TLLM_LOG_DEBUG("==========================================================");

    // block burn out
    size_t i = 0;
    for (; i < blocksInPrimaryPool / occupiedBlocksLinear; ++i)
    {
        GenerationRequest seq{requestId + 1 + i, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
        auto llmReqLoop
            = std::make_shared<LlmRequest>(LlmRequest::RequestIdType{static_cast<uint64_t>(requestId + 1 + i)},
                maxNewTokens, inputTokens0, samplingConfig, isStreaming);
        ASSERT_NO_THROW((void) blockManager.addSequenceBatch({&seq}, {numTokens}, {numBlocksPerBeam},
            {std::ref(*llmReqLoop)}, linearWindowSizeCode,
            /*isEnableBlockReuse=*/false));
    }
    // no more blocks
    GenerationRequest seq3{requestId + 1 + i, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    auto llmReq3 = std::make_shared<LlmRequest>(LlmRequest::RequestIdType{static_cast<uint64_t>(requestId + 1 + i)},
        maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    ASSERT_THROW(blockManager.addSequenceBatch({&seq3}, {numTokens}, {numBlocksPerBeam}, {std::ref(*llmReq3)},
                     linearWindowSizeCode,
                     /*isEnableBlockReuse=*/false),
        std::runtime_error);
}

void testBlockManagerLinearAttention_ContextReuse(int beamWidth, int numTokens0, int numTokens1, int numReusedTokens)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 32;
    auto constexpr blocksInPrimaryPool = 48;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto maxAttentionWindow = numTokens0 * 2;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    SizeType32 constexpr linearWindowSizeCode = LinearAttentionMetadata::LinearCacheType::kRecurrentStates;

    LinearAttentionMetadata linearAttentionMetadata{
        // .linearLayerIndices = {2, 5, 8, 11},
        .cacheType = linearWindowSizeCode,
        .allRecurrentStatesBytes = 440 * 1024, // dummy value
        .statesSnapshotInterval = tokensPerBlock * 2,
        .saveLastSnapshot = true,
    };

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool * 2, blocksInSecondaryPool}},
        {linearWindowSizeCode, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{linearWindowSizeCode, maxAttentionWindow}, nvinfer1::DataType::kHALF, 0,
        /*chunkSize*/ 0, CacheType::kSELF, std::nullopt, nullptr, false, true, nullptr, std::nullopt, false, 128, 0,
        false, linearAttentionMetadata);
    blockManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>();
    for (int i = 0; i < numTokens0; ++i)
    {
        inputTokens0->push_back(i);
    }
    auto const inputLength = static_cast<SizeType32>(inputTokens0->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, numTokens0, inputTokens0, samplingConfig, isStreaming);

    // reuse enabled: basic allocation
    GenerationRequest seq0{requestId, numTokens0, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens0}, {tc::ceilDiv(numTokens0, tokensPerBlock)},
        {std::ref(*llmRequest0)}, linearWindowSizeCode, /*isEnableBlockReuse=*/true);
    (void) blockManager.addSequenceBatch({&seq0}, {numTokens0}, {tc::ceilDiv(numTokens0, tokensPerBlock)},
        {std::ref(*llmRequest0)}, maxAttentionWindow, /*isEnableBlockReuse=*/true);
    ASSERT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    int regularSnapshots = numTokens0 / linearAttentionMetadata.statesSnapshotInterval;
    int contextFinalState = (numTokens0 % tokensPerBlock != 0) ? beamWidth : 1;
    int lastSnapshot // only exists when: 1. the current block is not a full block. 2. the current-1 block is not
                     // multiple of statesSnapshotInterval.
        = (numTokens0 / linearAttentionMetadata.statesSnapshotInterval * linearAttentionMetadata.statesSnapshotInterval
              != numTokens0 / tokensPerBlock * tokensPerBlock)
            && (numTokens0 % tokensPerBlock != 0)
        ? 1
        : 0;
    auto occupiedBlocksLinear = regularSnapshots + contextFinalState + lastSnapshot;
    auto totalBlocks = tc::ceilDiv(numTokens0, tokensPerBlock) + contextFinalState - 1;
    auto placeholderBlocks = totalBlocks - occupiedBlocksLinear;
    TLLM_LOG_DEBUG("==========================================================");
    ASSERT_EQ(
        blocksInPrimaryPool - blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], occupiedBlocksLinear);

    auto ids0 = seq0.getCacheBlockIds(linearWindowSizeCode); // copy
    std::set<std::int32_t> idSetPositive{};
    std::set<std::int32_t> idSetNegative{};
    ASSERT_EQ(ids0.size(), beamWidth);
    for (auto const& beam : ids0)
    {
        ASSERT_EQ(beam.size(), tc::ceilDiv(numTokens0, tokensPerBlock));
        for (auto id : beam)
        {
            if (id >= 0)
            {
                idSetPositive.insert(id);
            }
            else
            {
                idSetNegative.insert(id);
            }
        }
    }
    ASSERT_EQ(idSetPositive.size(), occupiedBlocksLinear);
    ASSERT_EQ(idSetNegative.size(), placeholderBlocks);

    // pretend the prefill is done
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    llmRequest0->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
    blockManager.storeContextBlocks(seq0, *llmRequest0);
    blockManager.releaseBlocks(seq0);
    ASSERT_EQ(blockManager.getNumFreeBlocksPerWindowSize()[linearWindowSizeCode], blocksInPrimaryPool);

    auto inputTokensNoise = std::make_shared<VecTokens>();
    for (int i = 0; i < numTokens1; ++i)
    {
        inputTokensNoise->push_back(10000 + i);
    }
    auto llmRequestNoise
        = std::make_shared<LlmRequest>(9999, numTokens1, inputTokensNoise, samplingConfig, isStreaming);
    GenerationRequest seqNoise{9999, numTokens1, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seqNoise}, {numTokens1}, {tc::ceilDiv(numTokens1, tokensPerBlock)},
        {std::ref(*llmRequestNoise)}, linearWindowSizeCode, /*isEnableBlockReuse=*/true);
    (void) blockManager.addSequenceBatch({&seqNoise}, {numTokens1}, {tc::ceilDiv(numTokens1, tokensPerBlock)},
        {std::ref(*llmRequestNoise)}, maxAttentionWindow, /*isEnableBlockReuse=*/true);

    auto inputTokens1 = std::make_shared<VecTokens>();
    for (int i = 0; i < numReusedTokens; ++i)
    {
        inputTokens1->push_back(i);
    }
    for (int i = numReusedTokens; i < numTokens1; ++i)
    {
        inputTokens1->push_back(1000 + i);
    }

    auto llmRequest1 = std::make_shared<LlmRequest>(1, numTokens1, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{1, numTokens1, beamWidth, blockManager.getWindowSizesMetadata()};
    (void) blockManager.addSequenceBatch({&seq1}, {numTokens1}, {tc::ceilDiv(numTokens1, tokensPerBlock)},
        {std::ref(*llmRequest1)}, linearWindowSizeCode, /*isEnableBlockReuse=*/true);
    (void) blockManager.addSequenceBatch({&seq1}, {numTokens1}, {tc::ceilDiv(numTokens1, tokensPerBlock)},
        {std::ref(*llmRequest1)}, maxAttentionWindow, /*isEnableBlockReuse=*/true);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    blockManager.storeContextBlocks(seq1, *llmRequest1);
    int numReusedBlocks = numReusedTokens / tokensPerBlock;
    for (; numReusedBlocks > 0; --numReusedBlocks)
    {
        if ((numReusedBlocks % (linearAttentionMetadata.statesSnapshotInterval / tokensPerBlock)
                == 0)                                              // is a regular snapshot
            || (numReusedBlocks == (numTokens0 / tokensPerBlock))) // is the last snapshot
        {
            break;
        }
    }
    auto const& ids1 = seq1.getCacheBlockIds(linearWindowSizeCode);
    for (int i = 0; i < numReusedBlocks; ++i)
    {
        for (int beam = 0; beam < beamWidth; ++beam)
        {
            if (ids0.at(beam).at(i) < 0 || ids1.at(beam).at(i) < 0)
            {
                continue;
            }
            ASSERT_EQ(ids1.at(beam).at(i), ids0.at(beam).at(i))
                << "Block " << i << " should be reused for beam " << beam;
        }
    }

    for (int i = numReusedBlocks; i < tc::ceilDiv(numTokens1, tokensPerBlock); ++i)
    {
        for (int beam = 0; beam < beamWidth; ++beam)
        {
            if (i >= ids0.at(beam).size() || ids0.at(beam).at(i) < 0 || ids1.at(beam).at(i) < 0)
            {
                continue;
            }
            ASSERT_NE(ids1.at(beam).at(i), ids0.at(beam).at(i))
                << "Block " << i << " should NOT be reused for beam " << beam;
        }
    }

    auto matchedLen = seq1.getCurrentPrepopulatedPromptLen();
    ASSERT_EQ(matchedLen, numReusedBlocks * tokensPerBlock);
}

std::vector<std::vector<int>> getExpectedBlockIds(int beamWidth, int numTotalBlocks, int numContextBlocks,
    int tokensPerBlock, bool enableContextReuse, int numContextTokens, int statesSnapshotInterval)
{
    std::vector<std::vector<int>> expectedBlockIds(beamWidth, std::vector<int>(numTotalBlocks, -1));
    int blockId = -1;
    int placeholderId = -1;
    for (int blk = 0; blk < numTotalBlocks; ++blk)
    {
        bool shouldHaveMemory = false;
        if (blk == numTotalBlocks - 1)
        {
            shouldHaveMemory = true;
        }
        else if (enableContextReuse && blk < numContextBlocks)
        {
            int blockEndTokenCount = (blk + 1) * tokensPerBlock;
            shouldHaveMemory =
                // regular snapshot
                (blockEndTokenCount <= numContextTokens && blockEndTokenCount % statesSnapshotInterval == 0)
                // last snapshot
                || (blockEndTokenCount < numContextTokens && blockEndTokenCount + tokensPerBlock > numContextTokens);
        }
        else if (blk == numContextBlocks - 2 && beamWidth > 1)
        {
            // shouldHaveMemory = true;
        }
        bool sharedAmongBeams = (blk < numContextBlocks - 1) || (beamWidth == 1)
            || (numContextTokens % tokensPerBlock == 0 && blk == numContextBlocks - 1);
        if (!sharedAmongBeams && shouldHaveMemory)
        {
            for (int beam = 0; beam < beamWidth; ++beam)
            {
                expectedBlockIds[beam][blk] = ++blockId;
            }
        }
        else
        {
            int id = shouldHaveMemory ? ++blockId : --placeholderId;
            for (int beam = 0; beam < beamWidth; ++beam)
            {
                expectedBlockIds[beam][blk] = id;
            }
        }
    }
    return expectedBlockIds;
}

void testKVCacheManagerLinearAttention_DecodingBlockGrowth(
    int beamWidth, int numContextTokens, int numGenerateTokens, bool enableContextReuse)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 32;
    auto constexpr blocksInPrimaryPool = 24;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr batchSize = 1;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr bytesPerToken = 4;
    auto constexpr sinkTokenLen = 0;
    auto constexpr canUseOneMoreBlock = true;

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamIdx = 0;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto maxAttentionWindow = numContextTokens + numGenerateTokens + sinkTokenLen + 1;
    SizeType32 constexpr linearWindowSizeCode = LinearAttentionMetadata::LinearCacheType::kRecurrentStates;

    LinearAttentionMetadata linearAttentionMetadata{
        // .linearLayerIndices = {2, 5, 8, 11},
        .cacheType = linearWindowSizeCode,
        .allRecurrentStatesBytes = 440 * 1024, // dummy value
        .statesSnapshotInterval = tokensPerBlock * 2,
        .saveLastSnapshot = true,
        .numPlaceholderBlocks = blocksInPrimaryPool * 100,
    };
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}},
        {linearWindowSizeCode, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{linearWindowSizeCode},
        /*dtype*/ nvinfer1::DataType::kHALF,
        /*sinkTokenLen*/ sinkTokenLen,
        /*stream*/ stream,
        /*maxSequenceLength*/ maxAttentionWindow,
        /*chunkSize*/ 0,
        /*enableBlockReuse*/ enableContextReuse,
        /*cacheType*/ CacheType::kSELF,
        /*secondaryOffloadMinPriority*/ std::nullopt,
        /*eventManager*/ nullptr,
        /*enablePartialReuse*/ false,
        /*copyOnPartialReuse*/ true,
        /*kvCacheConnectorManager*/ nullptr,
        /*enableIndexerKCache*/ false,
        /*indexerKCacheQuantBlockSize*/ 128,
        /*indexerKCacheIndexHeadDim*/ 0,
        /*indexerKCacheUseFp4=*/false,
        /*linearAttentionMetadata*/ linearAttentionMetadata);

    auto inputTokens0 = std::make_shared<VecTokens>();
    for (int i = 0; i < numContextTokens; ++i)
    {
        inputTokens0->push_back(i);
    }
    auto const inputLength = static_cast<SizeType32>(inputTokens0->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0
        = std::make_shared<LlmRequest>(requestId, numContextTokens, inputTokens0, samplingConfig, isStreaming);

    // add context
    kvCacheManager.addSequenceBatch(
        {{{llmRequest0->mRequestId, numContextTokens, beamWidth}}}, {std::ref(*llmRequest0)});

    // check context blocks
    auto numContextBlocks = tc::ceilDiv(numContextTokens, tokensPerBlock);
    auto const blockIdsAfterContext = kvCacheManager.getCacheBlockIds(llmRequest0->mRequestId, linearWindowSizeCode);
    auto expectedBlockIdsAfterContext = getExpectedBlockIds(beamWidth, numContextBlocks, numContextBlocks,
        tokensPerBlock, enableContextReuse, numContextTokens, linearAttentionMetadata.statesSnapshotInterval);

    for (int beam = 0; beam < beamWidth; ++beam)
    {
        for (int blk = 0; blk < numContextBlocks; ++blk)
        {
            ASSERT_EQ(blockIdsAfterContext[beam][blk], expectedBlockIdsAfterContext[beam][blk]);
        }
    }

    // add generated tokens
    for (int i = 0; i < numGenerateTokens; ++i)
    {
        kvCacheManager.addToken(llmRequest0->mRequestId);
    }

    // check all blocks
    auto numTotalBlocks = tc::ceilDiv(numContextTokens + numGenerateTokens, tokensPerBlock);

    auto const blockIds = kvCacheManager.getCacheBlockIds(llmRequest0->mRequestId, linearWindowSizeCode);
    ASSERT_EQ(blockIds.size(), beamWidth);
    for (auto const& beam : blockIds)
    {
        ASSERT_EQ(beam.size(), numTotalBlocks);
    }

    auto expectedBlockIds = getExpectedBlockIds(beamWidth, numTotalBlocks, numContextBlocks, tokensPerBlock,
        enableContextReuse, numContextTokens, linearAttentionMetadata.statesSnapshotInterval);

    for (int beam = 0; beam < beamWidth; ++beam)
    {
        for (int blk = 0; blk < numTotalBlocks; ++blk)
        {
            ASSERT_EQ(blockIds[beam][blk], expectedBlockIds[beam][blk]);
        }
    }
}

void testKVCacheManagerLinearAttention_BlockCopying(
    int beamWidth, int numContextTokens, int numGenerateTokens, bool enableContextReuse)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 32;
    auto constexpr blocksInPrimaryPool = 30;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr batchSize = 1;
    auto constexpr maxBlocksPerSeq = 10;
    auto constexpr bytesPerToken = 4;
    auto constexpr sinkTokenLen = 0;
    auto constexpr canUseOneMoreBlock = true;

    SizeType32 constexpr maxNewTokens{0};
    auto constexpr beamIdx = 0;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto maxAttentionWindow = numContextTokens + numGenerateTokens + sinkTokenLen + 1;
    SizeType32 constexpr linearWindowSizeCode = LinearAttentionMetadata::LinearCacheType::kRecurrentStates;

    LinearAttentionMetadata linearAttentionMetadata{
        // .linearLayerIndices = {2, 5, 8, 11},
        .cacheType = linearWindowSizeCode,
        .allRecurrentStatesBytes = 440 * 1024, // dummy value
        .statesSnapshotInterval = tokensPerBlock * 2,
        .saveLastSnapshot = true,
        .numPlaceholderBlocks = blocksInPrimaryPool * 100,
    };
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}},
        {linearWindowSizeCode, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{linearWindowSizeCode, maxAttentionWindow},
        nvinfer1::DataType::kHALF, sinkTokenLen, stream, maxAttentionWindow, /*chunkSize*/ 0, enableContextReuse,
        CacheType::kSELF, std::nullopt, nullptr, false, true, nullptr, false, 128, 0, false, linearAttentionMetadata);
    kvCacheManager.allocatePools(false);

    char* poolBaseAddr
        = reinterpret_cast<char*>(kvCacheManager.getBlockManager().getRecurrentStatesPool().primaryPtr->data());
    // memory layout of the pool: [numLayers, blocksInPrimaryPool, 1 (kvFactor), sizePerBlock]
    size_t const strideBlockId = linearAttentionMetadata.allRecurrentStatesBytes;
    std::unique_ptr<char[]> hostBuffer(new char[strideBlockId]);

    auto inputTokens0 = std::make_shared<VecTokens>();
    for (int i = 0; i < numContextTokens; ++i)
    {
        inputTokens0->push_back(i);
    }
    auto llmRequest0
        = std::shared_ptr<LlmRequest>(new LlmRequest(0, numContextTokens, inputTokens0, samplingConfig, isStreaming));
    llmRequest0->setContextChunkSize(linearAttentionMetadata.statesSnapshotInterval);
    // add context
    kvCacheManager.addSequenceBatch(
        {{{llmRequest0->mRequestId, numContextTokens, beamWidth}}}, {std::ref(*llmRequest0)});

    auto const numContextBlocks = tc::ceilDiv(numContextTokens, tokensPerBlock);
    auto expectedBlockIds = getExpectedBlockIds(beamWidth, numContextBlocks, numContextBlocks, tokensPerBlock,
        enableContextReuse, numContextTokens, linearAttentionMetadata.statesSnapshotInterval);

    // verify block offsets
    // {numPools, maxNumSequences * beamWidth, 2(k&v), maxBlocksPerSeq}
    tr::ITensor::SharedPtr const kvCacheBlockOffsets = tr::BufferManager::cpu(
        tr::ITensor::makeShape({kvCacheManager.getNumPools(), maxNumSequences * beamWidth, 2, maxBlocksPerSeq}),
        tr::TRTDataType<tk::KVCacheIndex>::value);
    int const linearPoolIdx = kvCacheManager.getPoolLayerIdx(0); // layer 0 is the linear layer
    kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, llmRequest0->mRequestId);

    // slice since we only have 1 request
    auto blockOffsetsSlice = tr::ITensor::slice(
        tr::ITensor::at(kvCacheBlockOffsets, {linearPoolIdx}), 0, beamWidth); // {beamWidth, 2(k&v), maxBlocksPerSeq}

    auto blockOffsetsShape = blockOffsetsSlice->getShape();
    auto* const blockOffsetsPtr = tr::bufferCast<tk::KVCacheIndex>(*blockOffsetsSlice);

    auto blockIds = kvCacheManager.getCacheBlockIds(llmRequest0->mRequestId, linearWindowSizeCode);
    for (int beam = 0; beam < beamWidth; ++beam)
    {
        for (int blk = 0; blk < numContextBlocks; ++blk)
        {
            auto blockId = blockIds[beam][blk];
            auto blockOffsetK = blockOffsetsPtr[tc::flat_index(blockOffsetsShape.d, beam, 0, blk)].get();
            auto blockOffsetV = blockOffsetsPtr[tc::flat_index(blockOffsetsShape.d, beam, 1, blk)].get();
            void* addrK = poolBaseAddr + blockOffsetK * linearAttentionMetadata.allRecurrentStatesBytes;
            void* addrV = poolBaseAddr + blockOffsetV * linearAttentionMetadata.allRecurrentStatesBytes;
            ASSERT_EQ(blockId, expectedBlockIds[beam][blk]);
            ASSERT_EQ(blockOffsetK, blockOffsetV);
            if (blockId < 0)
            {
                ASSERT_EQ(blockOffsetK, tensorrt_llm::kernels::KVCacheIndex::nullIndex.get());
            }
            else
            {
                // blockId should equal to mempool index before any offloading/reusing happens
                ASSERT_EQ(blockOffsetK, blockId);
            }
        }
    }

    std::vector<int> contextPositionPerStep;
    for (int blk = 0; blk < numContextBlocks; ++blk)
    {
        if (expectedBlockIds[0][blk] >= 0)
        {
            contextPositionPerStep.push_back(std::min((blk + 1) * tokensPerBlock, numContextTokens));
        }
    }

    // initialize the pool with all zeros
    auto ret = cudaMemset(poolBaseAddr, 0,
        strideBlockId * numLayers / 2 * blocksInPrimaryPool); // half of the layers are linear attention
    ASSERT_EQ(ret, cudaSuccess);
    std::vector<int> expectedValuesAfterContext(beamWidth, 0);
    for (int step = 0; step < contextPositionPerStep.size(); ++step)
    {
        int contextPosition = contextPositionPerStep[step];
        // called before every forward step
        kvCacheManager.copyLinearAttentionBlock(*llmRequest0);
        cudaDeviceSynchronize();
        int blockIndex = tc::ceilDiv(contextPosition, tokensPerBlock) - 1;
        bool shareAmongBeams = beamWidth > 1 && expectedBlockIds[0][blockIndex] == expectedBlockIds[1][blockIndex];
        for (int beam = 0; beam < beamWidth; ++beam)
        {
            size_t byteOffset = blockOffsetsPtr[tc::flat_index(blockOffsetsShape.d, beam, 0, blockIndex)].get()
                * linearAttentionMetadata.allRecurrentStatesBytes;
            ret = cudaMemcpy(hostBuffer.get(), poolBaseAddr + byteOffset, strideBlockId, cudaMemcpyDeviceToHost);
            ASSERT_EQ(ret, cudaSuccess);
            uint64_t val = static_cast<uint64_t>(expectedValuesAfterContext[beam]);
            uint64_t expected
                = val | (val << 8) | (val << 16) | (val << 24) | (val << 32) | (val << 40) | (val << 48) | (val << 56);
            for (int i = 0; i < strideBlockId / sizeof(uint64_t); ++i)
            {
                ASSERT_EQ(reinterpret_cast<uint64_t*>(hostBuffer.get())[i], expected) << "i=" << i;
            }

            expectedValuesAfterContext[beam] = (shareAmongBeams ? 0 : beam) * 16 + step;
            if (shareAmongBeams)
            {
                for (int b = 0; b < beamWidth; ++b)
                {
                    expectedValuesAfterContext[b] = expectedValuesAfterContext[beam];
                }
            }
            ret = cudaMemset(poolBaseAddr + byteOffset, expectedValuesAfterContext[beam], strideBlockId);
            ASSERT_EQ(ret, cudaSuccess);
        }
        // call the api
        llmRequest0->setContextCurrentPosition(contextPosition);
    }

    kvCacheManager.storeContextBlocks(*llmRequest0);

    llmRequest0->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
    std::vector<size_t> byteOffsetsPerBeam(beamWidth);
    for (int genStep = 0; genStep < numGenerateTokens; ++genStep)
    {
        kvCacheManager.addToken(llmRequest0->mRequestId);
        llmRequest0->addNewTokens(std::vector<TokenIdType>(beamWidth, genStep + numContextTokens));
        kvCacheManager.copyLinearAttentionBlock(*llmRequest0);
        cudaDeviceSynchronize();
        // retrieve latest block info
        kvCacheManager.copyBlockOffsets(*kvCacheBlockOffsets, 0, llmRequest0->mRequestId);
        auto blockIds = kvCacheManager.getCacheBlockIds(llmRequest0->mRequestId, linearWindowSizeCode);
        for (int beam = 0; beam < beamWidth; ++beam)
        {
            auto const blockOffset
                = blockOffsetsPtr[tc::flat_index(blockOffsetsShape.d, beam, 0, blockIds[beam].size() - 1)].get();
            size_t byteOffset = blockOffset * linearAttentionMetadata.allRecurrentStatesBytes;
            if (genStep < 2)
            {
                ret = cudaMemcpy(hostBuffer.get(), poolBaseAddr + byteOffset, strideBlockId, cudaMemcpyDeviceToHost);
                ASSERT_EQ(ret, cudaSuccess);
                uint64_t val = static_cast<uint64_t>(expectedValuesAfterContext[beam]);
                uint64_t expected = val | (val << 8) | (val << 16) | (val << 24) | (val << 32) | (val << 40)
                    | (val << 48) | (val << 56);
                for (int i = 0; i < strideBlockId / sizeof(uint64_t); ++i)
                {
                    ASSERT_EQ(reinterpret_cast<uint64_t*>(hostBuffer.get())[i], expected);
                }
            }
            if (byteOffsetsPerBeam[beam] == 0)
            {
                byteOffsetsPerBeam[beam] = byteOffset;
            }
            else
            {
                // verify that the block address does not change
                ASSERT_EQ(byteOffset, byteOffsetsPerBeam[beam]);
            }
            if (genStep == 0)
            {
                expectedValuesAfterContext[beam] = beam * 16;
                ret = cudaMemset(poolBaseAddr + byteOffset, expectedValuesAfterContext[beam], strideBlockId);
                ASSERT_EQ(ret, cudaSuccess);
            }
        }
    }
}
} // namespace

class LinearAttentionContextNoReuseTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

TEST_P(LinearAttentionContextNoReuseTest, ContextNoReuse)
{
    auto const& [beamWidth, numTokens] = GetParam();
    testBlockManagerLinearAttention_ContextNoReuse(beamWidth, numTokens);
}

INSTANTIATE_TEST_SUITE_P(BlockManagerLinearAttention, LinearAttentionContextNoReuseTest,
    testing::Values(std::make_tuple(4, 10), // basic test
        std::make_tuple(8, 96),             // edge cases: numTokens % tokensPerBlock == 0
        std::make_tuple(1, 97)              // beamWidth = 1
        ));

class LinearAttentionContextReuseTest : public ::testing::TestWithParam<std::tuple<int, int, int, int>>
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

TEST_P(LinearAttentionContextReuseTest, ContextReuse)
{
    auto const& [beamWidth, numTokens0, numTokens1, numReusedTokens] = GetParam();
    testBlockManagerLinearAttention_ContextReuse(beamWidth, numTokens0, numTokens1, numReusedTokens);
}

INSTANTIATE_TEST_SUITE_P(BlockManagerLinearAttention, LinearAttentionContextReuseTest,
    testing::Values(std::make_tuple(4, 10, 135, 10), // no applicable reuse: seq0 is too short (< tokensPerBlock)
        std::make_tuple(4, 96, 135, 37),             // numTokens0 % tokensPerBlock == 0, seq1 is too short (< interval)
        std::make_tuple(4, 96, 135, 64),             // reuse on a regular snapshot
        std::make_tuple(4, 97, 135, 96),             // reuse on the last snapshot
        std::make_tuple(1, 97, 135, 97),             // beamWidth = 1, reuse on the last snapshot
        std::make_tuple(4, 130, 135, 101)            // normal case
        ));

class LinearAttentionDecodingBlockGrowthTest : public ::testing::TestWithParam<std::tuple<int, int, int, bool>>
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

TEST_P(LinearAttentionDecodingBlockGrowthTest, DecodingBlockGrowth)
{
    auto const& [beamWidth, numContextTokens, numGenerateTokens, enableContextReuse] = GetParam();
    testKVCacheManagerLinearAttention_DecodingBlockGrowth(
        beamWidth, numContextTokens, numGenerateTokens, enableContextReuse);
}

INSTANTIATE_TEST_SUITE_P(BlockManagerLinearAttention, LinearAttentionDecodingBlockGrowthTest,
    testing::Values(
        std::make_tuple(1, 100, 100, true), std::make_tuple(1, 100, 100, false), // normal case beamWidth = 1
        std::make_tuple(4, 100, 100, true), std::make_tuple(4, 100, 100, false), // normal case beamWidth > 1
        std::make_tuple(4, 96, 100, true),
        std::make_tuple(4, 96, 100, false) // edge cases: numContextTokens % tokensPerBlock == 0 and beamWidth > 1
        ));

class LinearAttentionBlockCopyingTest : public ::testing::TestWithParam<std::tuple<int, int, int>>
{
protected:
    void SetUp() override
    {
        if (tc::getDeviceCount() == 0)
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}
};

TEST_P(LinearAttentionBlockCopyingTest, BlockCopying)
{
    auto const& [beamWidth, numContextTokens, numGenerateTokens] = GetParam();
    testKVCacheManagerLinearAttention_BlockCopying(
        beamWidth, numContextTokens, numGenerateTokens, /*enableContextReuse=*/true);
}

INSTANTIATE_TEST_SUITE_P(BlockManagerLinearAttention, LinearAttentionBlockCopyingTest,
    testing::Values(std::make_tuple(1, 100, 35), // normal case beamWidth = 1
        std::make_tuple(4, 96, 35),              // edge cases: numContextTokens % tokensPerBlock == 0 and beamWidth > 1
        std::make_tuple(4, 97, 35)               // normal case beamWidth > 1
        ));

TEST_F(KVCacheManagerTest, StaticLinearHybridAllocationTest)
{
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 128;
    auto constexpr tokensPerBlock = 32;
    auto constexpr maxBatchSize = 20;
    auto constexpr kvFactor = 2;

    auto constexpr maxAttentionWindow = 1024;
    SizeType32 constexpr linearWindowSizeCode = LinearAttentionMetadata::LinearCacheType::kRecurrentStates;

    LinearAttentionMetadata const linearAttentionMetadata{
        .linearLayerIndices = {2, 5, 8, 11},
        .cacheType = linearWindowSizeCode,
        .allRecurrentStatesBytes = 1 * 1024, // dummy value
        .statesSnapshotInterval = tokensPerBlock * 2,
        .saveLastSnapshot = true,
        .numPlaceholderBlocks = std::nullopt,
    };

    // 12 layers total: layers 2, 5, 8, 11 are linear-attention layers; the rest use maxAttentionWindow.
    std::vector<SizeType32> const linearLayerIndices = linearAttentionMetadata.linearLayerIndices;
    std::set<SizeType32> const linearLayerSet(linearLayerIndices.begin(), linearLayerIndices.end());
    std::vector<SizeType32> regularLayerIndices;
    for (SizeType32 layer = 0; layer < numLayers; ++layer)
    {
        if (linearLayerSet.find(layer) == linearLayerSet.end())
        {
            regularLayerIndices.push_back(layer);
        }
    }
    std::map<SizeType32, std::vector<SizeType32>> const windowSizeToLayers{
        {maxAttentionWindow, regularLayerIndices},
        {linearWindowSizeCode, linearLayerIndices},
    };
    std::vector<SizeType32> const numKvHeadsPerLayer(numLayers, numKvHeads);

    // Sized to comfortably exceed the static reservation
    // (maxBatchSize * allRecurrentStatesBytes = 20 KiB) plus dynamic blocks for the regular window.
    uint64_t constexpr allottedPrimaryMemBytes = 64ULL * 1024ULL * 1024ULL;
    uint64_t constexpr allottedSecondaryMemBytes = 0;
    size_t constexpr extraCostMemory = 0;

    tensorrt_llm::runtime::WorldConfig const worldConfig{};

    // Static-hybrid path requires block reuse to be disabled.
    tle::KvCacheConfig const kvCacheConfigDisabledReuse{/*enableBlockReuse=*/false};
    auto const blocksPerWindow
        = KVCacheManager::calculateMaxNumBlocks(kvCacheConfigDisabledReuse, nvinfer1::DataType::kHALF,
            numKvHeadsPerLayer, sizePerHead, tokensPerBlock, worldConfig, windowSizeToLayers, allottedPrimaryMemBytes,
            allottedSecondaryMemBytes, extraCostMemory, kvFactor, maxBatchSize, linearAttentionMetadata);

    // Linear-attention pool gets exactly maxBatchSize blocks (statically reserved).
    ASSERT_EQ(blocksPerWindow.count(linearWindowSizeCode), 1);
    EXPECT_EQ(std::get<0>(blocksPerWindow.at(linearWindowSizeCode)), maxBatchSize);
    // No secondary memory was provided, so secondary block count is zero.
    EXPECT_EQ(std::get<1>(blocksPerWindow.at(linearWindowSizeCode)), 0);

    // The regular window must still receive a positive number of dynamic blocks
    // after the static reservation is subtracted from the primary memory budget.
    ASSERT_EQ(blocksPerWindow.count(maxAttentionWindow), 1);
    EXPECT_GT(std::get<0>(blocksPerWindow.at(maxAttentionWindow)), 0);

    // Sanity check: when block reuse is enabled, the static-hybrid path is not taken,
    // so the linear pool falls back to memory-budget-based sizing rather than maxBatchSize.
    tle::KvCacheConfig const kvCacheConfigEnabledReuse{/*enableBlockReuse=*/true};
    auto const dynamicBlocksPerWindow
        = KVCacheManager::calculateMaxNumBlocks(kvCacheConfigEnabledReuse, nvinfer1::DataType::kHALF,
            numKvHeadsPerLayer, sizePerHead, tokensPerBlock, worldConfig, windowSizeToLayers, allottedPrimaryMemBytes,
            allottedSecondaryMemBytes, extraCostMemory, kvFactor, maxBatchSize, linearAttentionMetadata);
    EXPECT_NE(std::get<0>(dynamicBlocksPerWindow.at(linearWindowSizeCode)), maxBatchSize);
}

///////////////////////////////////////////////////////////////////////////////
// addSequenceBatch corner-case tests
//
// These tests verify the two-phase claim-then-onboard strategy when multiple
// requests in a single addSequenceBatch call compete for the same radix tree
// blocks: partial vs full matches, leaf vs non-leaf, and shouldReleaseCopySource
// ownership tracking.
///////////////////////////////////////////////////////////////////////////////

// Helper: create a KVCacheManager for batch tests.
// tokensPerBlock=4, 16 primary blocks, block reuse enabled, partial reuse enabled.
static auto makeBatchTestKVCacheManager(std::shared_ptr<tensorrt_llm::runtime::CudaStream> const& stream)
{
    auto constexpr numLayers = 1;
    auto constexpr numKvHeads = 1;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 16;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * 8;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    auto mgr = std::make_unique<KVCacheManager>(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
        nvinfer1::DataType::kHALF, 0, stream, maxAttentionWindow, /*chunkSize=*/maxAttentionWindow,
        /*enableBlockReuse=*/true, CacheType::kSELF,
        /*secondaryOffloadMinPriority=*/std::nullopt,
        /*eventManager=*/nullptr,
        /*enablePartialReuse=*/true);
    mgr->allocatePools(false);
    return mgr;
}

// Helper: add a sequence, store its blocks for reuse, and remove it.
static void seedAndRelease(KVCacheManager& mgr, LlmRequest::RequestIdType reqId,
    std::shared_ptr<VecTokens> const& tokens, SizeType32 beamWidth = 1)
{
    auto const inputLength = static_cast<SizeType32>(tokens->size());
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto req = std::make_shared<LlmRequest>(reqId, maxNewTokens, tokens, samplingConfig, /*isStreaming=*/false);
    mgr.addSequenceBatch({{{reqId, inputLength, beamWidth}}}, {std::ref(*req)});
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
    (void) mgr.removeSequence(reqId, req);
}

// Test 1: Two requests in a batch, both partially match the same leaf block.
// The tracker should assign reuse to the last request and bump the first to copy.
TEST_F(KVCacheManagerTest, BatchAddSequence_LeafPartialThenPartial)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6] (7 tokens) → block0 [0,1,2,3] full, block1 [4,5] stored (last token excluded)
    // block1 is a partial leaf with 2 tokens.
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6}));

    // Batch: two requests with [0,1,2,3,4,X] → both match block0 fully,
    // both partially match block1 [4,5] (only token 4 matches in search key [4])
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 10});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 20});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)});

    // block0 fully matched (4 tokens) + block1 partial match (1 token) = 5 prepopulated
    EXPECT_EQ(req1->getContextCurrentPosition(), tokensPerBlock + 1); // 5
    EXPECT_EQ(req2->getContextCurrentPosition(), tokensPerBlock + 1); // 5

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    EXPECT_EQ(ids1.size(), 2);
    EXPECT_EQ(ids2.size(), 2);
    // Both share block0 (same physical block from radix tree)
    EXPECT_EQ(ids1[0], ids2[0]);
    // req2 is the reuser → gets the original block1. req1 was bumped to copy → gets a NEW block.
    EXPECT_EQ(ids2[1], 1); // reuser keeps original block ID 1
    EXPECT_NE(ids1[1], 1); // copier gets a different block

    // Clean up
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(1, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(2, req2);
}

// Test 2: Batch with leaf partial match followed by full match on same block.
// The full match should take priority, bumping the partial matcher to copy.
// Note: a "full match" requires isFull()=true on the stored block, meaning it has exactly tokensPerBlock tokens.
TEST_F(KVCacheManagerTest, BatchAddSequence_LeafPartialThenFull)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6,7,8] (9 tokens) → block0 [0,1,2,3] full, block1 [4,5,6,7] full (isFull=true)
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8}));

    // Request 1: [0,1,2,3,4,5,6,10] (8 tokens) → partial match on full block1 (key [4,5,6] vs stored [4,5,6,7])
    // Request 2: [0,1,2,3,4,5,6,7,8] (9 tokens) → full match on block1 (key [4,5,6,7] matches, isFull=true)
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 10});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)});

    // req1: partial match on full block1, bumped to copy by req2's full match → 4+3=7
    // req2: full match on both full blocks → 4+4=8
    EXPECT_EQ(req1->getContextCurrentPosition(), tokensPerBlock + 3); // 7
    EXPECT_EQ(req2->getContextCurrentPosition(), 2 * tokensPerBlock); // 8

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    // Both share block0 (ID 0)
    EXPECT_EQ(ids1[0], 0);
    EXPECT_EQ(ids2[0], 0);
    // req2 (full match) keeps the original block1 (ID 1)
    EXPECT_EQ(ids2[1], 1);
    // req1 (partial, bumped to copy) gets a new block
    EXPECT_NE(ids1[1], 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(1, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(2, req2);
}

// Test 3: Batch with leaf full match followed by partial match.
// The partial matcher sees fullyMatched=true and must copy.
TEST_F(KVCacheManagerTest, BatchAddSequence_LeafFullThenPartial)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6,7,8] → block0 [0,1,2,3] full, block1 [4,5,6,7] full
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8}));

    // Request 1: [0,1,2,3,4,5,6,7,8] → full match on both full blocks
    // Request 2: [0,1,2,3,4,5,6,20] → partial match on full block1, fullyMatched → must copy
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 20});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)});

    // req1: full match on both full blocks → 4+4=8
    // req2: partial match on full block1 (3/4 tokens), fullyMatched → must copy → 4+3=7
    EXPECT_EQ(req1->getContextCurrentPosition(), 2 * tokensPerBlock); // 8
    EXPECT_EQ(req2->getContextCurrentPosition(), tokensPerBlock + 3); // 7

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    // Both share block0 (ID 0)
    EXPECT_EQ(ids1[0], 0);
    EXPECT_EQ(ids2[0], 0);
    // req1 (full match) keeps the original block1 (ID 1)
    EXPECT_EQ(ids1[1], 1);
    // req2 (partial, fullyMatched → copy) gets a new block
    EXPECT_NE(ids2[1], 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(1, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(2, req2);
}

// Test 4: Batch with leaf partial → full → partial on same full block.
// First partial is bumped to copy by full, second partial also copies (fullyMatched).
TEST_F(KVCacheManagerTest, BatchAddSequence_LeafPartialFullPartial)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6,7,8] → block0 [0,1,2,3] full, block1 [4,5,6,7] full
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8}));

    // Request 1: partial [0,1,2,3,4,5,6,10] → partial on full block1 (3/4 match)
    // Request 2: full [0,1,2,3,4,5,6,7,8] → full match on block1
    // Request 3: partial [0,1,2,3,4,5,6,30] → partial on full block1, fullyMatched → copy
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 10});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto tokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 30});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);
    auto req3 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{3}, SizeType32{0}, tokens3, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth},
                              {3, static_cast<SizeType32>(tokens3->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2), std::ref(*req3)});

    // req1: partial on full block1, bumped to copy by full → 4+3=7
    // req2: full match on both full blocks → 4+4=8
    // req3: partial on full block1, fullyMatched → must copy → 4+3=7
    EXPECT_EQ(req1->getContextCurrentPosition(), tokensPerBlock + 3); // 7
    EXPECT_EQ(req2->getContextCurrentPosition(), 2 * tokensPerBlock); // 8
    EXPECT_EQ(req3->getContextCurrentPosition(), tokensPerBlock + 3); // 7

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    auto ids3 = mgr->getSequence(3).getCacheBlockIds(windowSize).at(0);
    // All share block0 (ID 0)
    EXPECT_EQ(ids1[0], 0);
    EXPECT_EQ(ids2[0], 0);
    EXPECT_EQ(ids3[0], 0);
    // req2 (full match) keeps the original block1 (ID 1)
    EXPECT_EQ(ids2[1], 1);
    // req1 and req3 (both copies) get new blocks, not block1
    EXPECT_NE(ids1[1], 1);
    EXPECT_NE(ids3[1], 1);
    // All three second blocks are unique
    EXPECT_NE(ids1[1], ids2[1]);
    EXPECT_NE(ids1[1], ids3[1]);
    EXPECT_NE(ids2[1], ids3[1]);

    for (int id = 1; id <= 3; ++id)
    {
        auto& req = (id == 1) ? req1 : (id == 2) ? req2 : req3;
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
        (void) mgr->removeSequence(id, req);
    }
}

// Test 5: Non-leaf partial match — shouldReleaseCopySource.
// Two requests partially match a non-leaf block. The last copier releases it.
TEST_F(KVCacheManagerTest, BatchAddSequence_NonLeafPartialThenPartial)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed two sequences that share a common prefix but diverge at the second block.
    // This creates a non-leaf block0 [0,1,2,3] with two leaf children.
    // Seq A: [0,1,2,3,4,5,6,7] → block0 [0,1,2,3], block1 [4,5,6,7]
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7}));
    // Seq B: [0,1,2,3,10,11,12,13] → block0 shared, block2 [10,11,12,13]
    seedAndRelease(*mgr, 1, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 10, 11, 12, 13}));

    auto freeBlocksBefore = mgr->getNumFreeBlocks();

    // Batch: two requests both partially match block0 (non-leaf).
    // Request 1: [0,1,2,50] → partial match on block0 [0,1,2,3], only 3 tokens match
    // Request 2: [0,1,60,70] → partial match on block0, only 2 tokens match
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 50});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 60, 70});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{10}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{11}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    // This should not throw — the shouldReleaseCopySource mechanism ensures the
    // claimed non-leaf copy source is released after the last copier's copy.
    EXPECT_NO_THROW(mgr->addSequenceBatch({{{10, static_cast<SizeType32>(tokens1->size()), beamWidth},
                                              {11, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)}));

    // Both should have 1 block allocated (partial match + fresh block)
    auto windowSize = theOnlyWindowSize(*mgr);
    EXPECT_EQ(mgr->getSequence(10).getCacheBlockIds(windowSize).at(0).size(), 1);
    EXPECT_EQ(mgr->getSequence(11).getCacheBlockIds(windowSize).at(0).size(), 1);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(10, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(11, req2);
}

// Test 6: Non-leaf full match then partial match.
// First request fully matches a non-leaf (continues to children), second partially matches it.
// Since fullyMatched=true in tracker, shouldReleaseCopySource=false for the copier.
TEST_F(KVCacheManagerTest, BatchAddSequence_NonLeafFullThenPartial)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed to create non-leaf block0 with two children (full blocks for true full match)
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8}));
    seedAndRelease(*mgr, 1, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 10, 11, 12, 13, 14}));

    // Request 1: [0,1,2,3,4,5,6,7,8] → full match block0, full match block1 (both isFull)
    // Request 2: [0,1,2,50] → partial match block0 (non-leaf, isFull)
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 50});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{10}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{11}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    EXPECT_NO_THROW(mgr->addSequenceBatch({{{10, static_cast<SizeType32>(tokens1->size()), beamWidth},
                                              {11, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)}));

    // req1 fully matches 2 full blocks → 4+4=8 prepopulated
    // req2 partially matches block0 → copies, fullyMatched prevents release
    EXPECT_EQ(req1->getContextCurrentPosition(), 2 * tokensPerBlock); // 8
    EXPECT_GT(req2->getContextCurrentPosition(), 0);                  // some partial match

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(10, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(11, req2);
}

// Test 7: Tight pool with non-leaf copy source — verifies the release is needed.
// Without shouldReleaseCopySource, this would fail with "No free block found".
TEST_F(KVCacheManagerTest, BatchAddSequence_NonLeafCopySourceTightPool)
{
    // Use a tight pool: 8 blocks, tokensPerBlock=4
    auto constexpr numLayers = 1;
    auto constexpr numKvHeads = 1;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * 8;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxAttentionWindow, /*chunkSize=*/maxAttentionWindow, /*enableBlockReuse=*/true, CacheType::kSELF,
        /*secondaryOffloadMinPriority=*/std::nullopt,
        /*eventManager=*/nullptr,
        /*enablePartialReuse=*/true);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // Seed to create a non-leaf block0 with children
    seedAndRelease(kvCacheManager, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7}));
    seedAndRelease(kvCacheManager, 1, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 10, 11, 12, 13}));

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // Request that partially matches block0 (non-leaf) and needs ALL remaining blocks.
    // Tokens: [0,1,50,...] → partial match on block0 (2 tokens), then needs many fresh blocks.
    // Total: 32 tokens = 8 blocks (32 / 4 = 8). All 8 pool blocks are needed.
    auto bigTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79});
    auto req = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{10}, SizeType32{0}, bigTokens, tr::SamplingConfig{beamWidth}, false);
    auto inputLen = static_cast<SizeType32>(bigTokens->size());
    auto numBlocks = (inputLen + tokensPerBlock - 1) / tokensPerBlock;

    // Without the shouldReleaseCopySource fix, this would throw "No free block found"
    // because the claimed non-leaf copy source would not be released.
    ASSERT_LE(numBlocks, blocksInPrimaryPool);
    EXPECT_NO_THROW(kvCacheManager.addSequenceBatch({{{10, inputLen, beamWidth}}}, {std::ref(*req)}));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
    (void) kvCacheManager.removeSequence(10, req);
}

// Test 8: Mixed batch — one request fully matches, another has no match at all.
// Verifies that batch handling works correctly for heterogeneous match patterns.
TEST_F(KVCacheManagerTest, BatchAddSequence_FullMatchAndNoMatch)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6,7,8] (9 tokens) → block0 [0,1,2,3] full, block1 [4,5,6,7] full
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8}));

    // Request 1: [0,1,2,3,4,5,6,7,8] → full match on both full blocks
    // Request 2: [100,101,102,103] → no match at all
    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2)});

    // Seed had 9 tokens → 2 full blocks stored (8 tokens), last excluded → 8 prepopulated
    EXPECT_EQ(req1->getContextCurrentPosition(), 2 * tokensPerBlock); // 8 reused
    EXPECT_EQ(req2->getContextCurrentPosition(), 0);                  // nothing reused

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    // req1 reuses the seeded blocks (IDs 0 and 1)
    EXPECT_EQ(ids1[0], 0);
    EXPECT_EQ(ids1[1], 1);
    // req2 has no match → all fresh blocks, none overlap with seeded blocks
    for (auto id : ids2)
    {
        EXPECT_NE(id, 0);
        EXPECT_NE(id, 1);
    }

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    (void) mgr->removeSequence(1, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req2);
    (void) mgr->removeSequence(2, req2);
}

// Test 9: Three requests partially match the same leaf — tracker bumps ownership twice.
TEST_F(KVCacheManagerTest, BatchAddSequence_LeafTriplePartialMatch)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto mgr = makeBatchTestKVCacheManager(stream);
    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Seed: [0,1,2,3,4,5,6] → block0 [0,1,2,3] full, block1 [4,5] partial leaf (2 tokens)
    seedAndRelease(*mgr, 0, std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6}));

    auto tokens1 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 10});
    auto tokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 20});
    auto tokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 30});
    auto req1 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, SizeType32{0}, tokens1, tr::SamplingConfig{beamWidth}, false);
    auto req2 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{2}, SizeType32{0}, tokens2, tr::SamplingConfig{beamWidth}, false);
    auto req3 = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{3}, SizeType32{0}, tokens3, tr::SamplingConfig{beamWidth}, false);

    mgr->addSequenceBatch({{{1, static_cast<SizeType32>(tokens1->size()), beamWidth},
                              {2, static_cast<SizeType32>(tokens2->size()), beamWidth},
                              {3, static_cast<SizeType32>(tokens3->size()), beamWidth}}},
        {std::ref(*req1), std::ref(*req2), std::ref(*req3)});

    // All three get block0 fully matched + partial block1 (1 token)
    // req1 & req2 bumped to copy, req3 is the final reuser
    EXPECT_EQ(req1->getContextCurrentPosition(), tokensPerBlock + 1);
    EXPECT_EQ(req2->getContextCurrentPosition(), tokensPerBlock + 1);
    EXPECT_EQ(req3->getContextCurrentPosition(), tokensPerBlock + 1);

    auto windowSize = theOnlyWindowSize(*mgr);
    auto ids1 = mgr->getSequence(1).getCacheBlockIds(windowSize).at(0);
    auto ids2 = mgr->getSequence(2).getCacheBlockIds(windowSize).at(0);
    auto ids3 = mgr->getSequence(3).getCacheBlockIds(windowSize).at(0);
    // All share block0
    EXPECT_EQ(ids1[0], ids2[0]);
    EXPECT_EQ(ids2[0], ids3[0]);
    // req3 is the final reuser → gets original block1
    EXPECT_EQ(ids3[1], 1);
    // req1 and req2 were bumped to copy → different blocks
    EXPECT_NE(ids1[1], 1);
    EXPECT_NE(ids2[1], 1);
    // All three second blocks should be unique
    EXPECT_NE(ids1[1], ids2[1]);
    EXPECT_NE(ids1[1], ids3[1]);
    EXPECT_NE(ids2[1], ids3[1]);

    for (int id = 1; id <= 3; ++id)
    {
        auto& req = (id == 1) ? req1 : (id == 2) ? req2 : req3;
        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
        (void) mgr->removeSequence(id, req);
    }
}

namespace
{
// Shared constants for all TruePriorityEviction tests.
auto constexpr kPE_NUM_LAYERS = 2;
auto constexpr kPE_NUM_HEADS = 2;
auto constexpr kPE_SIZE_PER_HEAD = 16;
auto constexpr kPE_TOKENS_PER_BLOCK = 4;
auto constexpr kPE_MAX_NUM_SEQUENCES = 8;
auto constexpr kPE_BEAM_WIDTH = 1;
SizeType32 constexpr kPE_MAX_NEW_TOKENS = 0;
bool constexpr kPE_IS_STREAMING = false;

// Factory: construct and allocate a KVCacheManager for TruePriorityEviction tests.
std::unique_ptr<KVCacheManager> makePriorityEvictionManager(
    SizeType32 blocksInPrimaryPool, SizeType32 maxAttentionWindow, std::shared_ptr<tr::CudaStream> const& stream)
{
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, 0}}};
    auto mgr = std::make_unique<KVCacheManager>(kPE_NUM_LAYERS, kPE_NUM_HEADS, kPE_SIZE_PER_HEAD, kPE_TOKENS_PER_BLOCK,
        blocksPerWindow, kPE_MAX_NUM_SEQUENCES, kPE_BEAM_WIDTH,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, nvinfer1::DataType::kHALF, 0, stream,
        maxAttentionWindow, /*chunkSize=*/maxAttentionWindow, /*enableBlockReuse=*/true);
    mgr->allocatePools(false);
    return mgr;
}

void addSequenceForTest(KVCacheManager& kvCacheManager, LlmRequest::RequestIdType requestId, SizeType32 inputLength,
    SizeType32 beamWidth, std::shared_ptr<LlmRequest> const& llmRequest)
{
    kvCacheManager.addSequenceBatch({{{requestId, inputLength, beamWidth}}}, {std::ref(*llmRequest)});
}
} // namespace

// Verifies that a low-priority interior block is evicted before its high-priority
// descendant leaf block. After evicting the interior block, the high-priority leaf
// is the last block to be evicted.
TEST_F(KVCacheManagerTest, TruePriorityEvictionInteriorBlockEvictedFirst)
{
    // 5 blocks total: B0 (MIN), B1 (HIGH), B2/B3/B4 (DEFAULT)
    auto constexpr blocksInPrimaryPool = 5;
    auto const maxAttentionWindow = kPE_TOKENS_PER_BLOCK * 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kPE_BEAM_WIDTH};
    auto kvCacheManager = makePriorityEvictionManager(blocksInPrimaryPool, maxAttentionWindow, stream);

    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq A: 8 tokens, B0=[0..3] at MIN priority (evict-first), B1=[4..7] at HIGH priority (evict-last).
    // B0 becomes an interior node in the trie (parent of B1).
    auto inputTokensA = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLengthA = static_cast<SizeType32>(inputTokensA->size());
    auto llmRequestA
        = std::make_shared<LlmRequest>(0, kPE_MAX_NEW_TOKENS, inputTokensA, samplingConfig, kPE_IS_STREAMING);
    llmRequestA->setKvCacheRetentionConfig(KvCacheRetentionConfig(
        {KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, KvCacheRetentionConfig::kMinRetentionPriority),
            KvCacheRetentionConfig::TokenRangeRetentionConfig(4, 8, 90)},
        KvCacheRetentionConfig::kDefaultRetentionPriority));
    addSequenceForTest(*kvCacheManager, 0, inputLengthA, kPE_BEAM_WIDTH, llmRequestA);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestA);
    kvCacheManager->storeContextBlocks(*llmRequestA);
    (void) kvCacheManager->removeSequence(0, llmRequestA);

    // All 5 blocks are now free:
    //   priority 0  (MIN):  [B0]             ← interior in trie, lowest priority
    //   priority 35 (DEFAULT): [B2, B3, B4]  ← never used, initialized to DEFAULT
    //   priority 90 (HIGH): [B1]             ← leaf in trie, highest priority
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq B: 16 new tokens (4 blocks), never overlaps with seq A.
    // With true priority eviction, blocks are claimed in priority order:
    //   B0 (prio 0) → B2, B3, B4 (prio 35, in queue order)
    // B1 (prio 90) must NOT be claimed — it has the highest priority.
    auto inputTokensB = std::make_shared<VecTokens>(
        VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115});
    auto const inputLengthB = static_cast<SizeType32>(inputTokensB->size());
    auto llmRequestB
        = std::make_shared<LlmRequest>(1, kPE_MAX_NEW_TOKENS, inputTokensB, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, inputLengthB, kPE_BEAM_WIDTH, llmRequestB);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestB);
    kvCacheManager->storeContextBlocks(*llmRequestB);

    // 4 blocks claimed by seq B; B1 (prio 90) is the surviving free block.
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), 1);

    // Queue integrity must be maintained after evicting the interior block B0.
    auto const& blockManager = kvCacheManager->getBlockManager();
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    (void) kvCacheManager->removeSequence(1, llmRequestB);
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Explicit reuse assertion: seq B stored its blocks [100..115] in the trie on
    // removeSequence.  A new request with the same prefix must reuse them, confirming
    // that the eviction path left the trie in a consistent state and didn't accidentally
    // evict the high-priority B1 block (which was the only remaining free block during
    // seq B's lifetime and is still in the trie under the orphaned [4..7] node).
    auto inputTokensC = std::make_shared<VecTokens>(inputTokensB->begin(), inputTokensB->end());
    auto const inputLengthC = static_cast<SizeType32>(inputTokensC->size());
    auto llmRequestC
        = std::make_shared<LlmRequest>(2, kPE_MAX_NEW_TOKENS, inputTokensC, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, inputLengthC, kPE_BEAM_WIDTH, llmRequestC);
    // At least the first kPE_TOKENS_PER_BLOCK * 3 tokens are reusable (3 full blocks).
    EXPECT_GE(llmRequestC->getContextCurrentPosition(), kPE_TOKENS_PER_BLOCK * 3);
    (void) kvCacheManager->removeSequence(2, llmRequestC);
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verifies that a HIGH-priority interior block is preserved while a LOW-priority
// leaf block (its descendant) is correctly evicted first.
TEST_F(KVCacheManagerTest, TruePriorityEvictionHighPriorityInteriorBlockPreserved)
{
    // 4 blocks total: B0 (HIGH=interior), B1 (MIN=leaf), B2/B3 (DEFAULT)
    auto constexpr blocksInPrimaryPool = 4;
    auto const maxAttentionWindow = kPE_TOKENS_PER_BLOCK * 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kPE_BEAM_WIDTH};
    auto kvCacheManager = makePriorityEvictionManager(blocksInPrimaryPool, maxAttentionWindow, stream);

    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq A: 8 tokens, B0=[0..3] at HIGH priority (90), B1=[4..7] at MIN priority (0).
    // B0 is interior (parent of B1); B1 is the leaf and has the LOWEST priority.
    auto inputTokensA = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLengthA = static_cast<SizeType32>(inputTokensA->size());
    auto llmRequestA
        = std::make_shared<LlmRequest>(0, kPE_MAX_NEW_TOKENS, inputTokensA, samplingConfig, kPE_IS_STREAMING);
    llmRequestA->setKvCacheRetentionConfig(KvCacheRetentionConfig(
        {KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, 90),
            KvCacheRetentionConfig::TokenRangeRetentionConfig(4, 8, KvCacheRetentionConfig::kMinRetentionPriority)},
        KvCacheRetentionConfig::kDefaultRetentionPriority));
    addSequenceForTest(*kvCacheManager, 0, inputLengthA, kPE_BEAM_WIDTH, llmRequestA);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestA);
    kvCacheManager->storeContextBlocks(*llmRequestA);
    (void) kvCacheManager->removeSequence(0, llmRequestA);

    // Free queue after release:
    //   priority 0  (MIN):     [B1]        ← leaf, lowest priority
    //   priority 35 (DEFAULT): [B2, B3]    ← never used
    //   priority 90 (HIGH):    [B0]        ← interior, highest priority
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq B: 4 new tokens (1 block). Should evict B1 (prio 0, lowest) — NOT the interior B0.
    auto inputTokensB = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103});
    auto const inputLengthB = static_cast<SizeType32>(inputTokensB->size());
    auto llmRequestB
        = std::make_shared<LlmRequest>(1, kPE_MAX_NEW_TOKENS, inputTokensB, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, inputLengthB, kPE_BEAM_WIDTH, llmRequestB);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestB);
    kvCacheManager->storeContextBlocks(*llmRequestB);

    // B1 (leaf, prio 0) claimed by seq B; B0, B2, B3 remain free.
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), 3);

    // Now B0 (interior, HIGH priority) still has its tokens in the trie.
    // A seq with the SAME prefix [0..3] should be able to reuse B0.
    (void) kvCacheManager->removeSequence(1, llmRequestB);

    auto inputTokensC = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 200, 201, 202, 203});
    auto const inputLengthC = static_cast<SizeType32>(inputTokensC->size());
    auto llmRequestC
        = std::make_shared<LlmRequest>(2, kPE_MAX_NEW_TOKENS, inputTokensC, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, inputLengthC, kPE_BEAM_WIDTH, llmRequestC);

    // B0 cached [0..3]; B1 was evicted (so [4..7] is no longer cached).
    // Seq C shares the first block [0..3] with seq A → B0 reused.
    // [200..203] is new, requires a fresh block.
    // contextCurrentPosition reflects how many tokens were prepopulated.
    EXPECT_EQ(llmRequestC->getContextCurrentPosition(), 4);

    auto const& blockManager = kvCacheManager->getBlockManager();
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    (void) kvCacheManager->removeSequence(2, llmRequestC);
}

// Verifies queue integrity is maintained through a sequence of interior block evictions
// in a 3-block chain (B0→B1→B2) with strictly ordered priorities.
TEST_F(KVCacheManagerTest, TruePriorityEvictionQueueIntegrityAfterChainEviction)
{
    // 6 blocks: B0 (prio MIN), B1 (prio DEFAULT), B2 (prio HIGH), B3/B4/B5 (DEFAULT)
    auto constexpr blocksInPrimaryPool = 6;
    auto const maxAttentionWindow = kPE_TOKENS_PER_BLOCK * 10;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kPE_BEAM_WIDTH};
    auto kvCacheManager = makePriorityEvictionManager(blocksInPrimaryPool, maxAttentionWindow, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq A: 12 tokens in 3 blocks with strictly ordered priorities.
    // B0=[0..3]: MIN priority (0)   — will be evicted first (interior node, parent of B1)
    // B1=[4..7]: DEFAULT priority   — will be evicted second (interior node, parent of B2)
    // B2=[8..11]: HIGH priority (90) — will be evicted last (leaf node)
    // Trie chain: root → B0 → B1 → B2
    auto inputTokensA = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLengthA = static_cast<SizeType32>(inputTokensA->size());
    auto llmRequestA
        = std::make_shared<LlmRequest>(0, kPE_MAX_NEW_TOKENS, inputTokensA, samplingConfig, kPE_IS_STREAMING);
    llmRequestA->setKvCacheRetentionConfig(KvCacheRetentionConfig(
        {KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, KvCacheRetentionConfig::kMinRetentionPriority),
            KvCacheRetentionConfig::TokenRangeRetentionConfig(4, 8, KvCacheRetentionConfig::kDefaultRetentionPriority),
            KvCacheRetentionConfig::TokenRangeRetentionConfig(8, 12, 90)},
        KvCacheRetentionConfig::kDefaultRetentionPriority));
    addSequenceForTest(*kvCacheManager, 0, inputLengthA, kPE_BEAM_WIDTH, llmRequestA);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequestA);
    kvCacheManager->storeContextBlocks(*llmRequestA);
    (void) kvCacheManager->removeSequence(0, llmRequestA);

    // Free queue after release (6 blocks):
    //   prio 0  (MIN):     [B0]          ← interior, lowest priority
    //   prio 35 (DEFAULT): [B3, B4, B5, B1]  ← B3/B4/B5 never used; B1 interior
    //   prio 90 (HIGH):    [B2]          ← leaf, highest priority
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    // Step 1: claim 1 block — must take B0 (prio 0, lowest).
    // B0 is an interior node (parent of B1→B2 in the trie).
    // True priority eviction detaches ONLY B0; B1 and B2 remain in trie.
    auto inputTokensX = std::make_shared<VecTokens>(VecTokens{200, 201, 202, 203});
    auto const inputLengthX = static_cast<SizeType32>(inputTokensX->size());
    auto llmRequestX
        = std::make_shared<LlmRequest>(1, kPE_MAX_NEW_TOKENS, inputTokensX, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, inputLengthX, kPE_BEAM_WIDTH, llmRequestX);

    // 5 blocks remain after B0 is claimed.
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), 5);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    (void) kvCacheManager->removeSequence(1, llmRequestX);

    // Step 2: claim 3 more blocks (all DEFAULT-priority: B3, B4, B5 or B1 depending on queue).
    // With true priority eviction, B3, B4, B5 (initialized at DEFAULT ahead of B1 in the queue)
    // and B1 (also DEFAULT) are all candidates; B2 (HIGH=90) is still protected.
    auto inputTokensY
        = std::make_shared<VecTokens>(VecTokens{300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311});
    auto const inputLengthY = static_cast<SizeType32>(inputTokensY->size());
    auto llmRequestY
        = std::make_shared<LlmRequest>(2, kPE_MAX_NEW_TOKENS, inputTokensY, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, inputLengthY, kPE_BEAM_WIDTH, llmRequestY);

    // After seq X is released (returns 1 block) and seq Y claims 3:
    // free = 6 (all released by X) - 3 (claimed by Y) = 3
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), 3);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    (void) kvCacheManager->removeSequence(2, llmRequestY);
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    // Explicit reuse assertion: seq Y stored its 3 blocks ([300..311]) in the trie
    // on removeSequence.  A new request with the same prefix must be able to reuse at
    // least one of those blocks, confirming the trie is consistent after interior-block
    // eviction.
    auto inputTokensZ = std::make_shared<VecTokens>(*inputTokensY);
    auto llmRequestZ
        = std::make_shared<LlmRequest>(3, kPE_MAX_NEW_TOKENS, inputTokensZ, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 3, static_cast<SizeType32>(inputTokensZ->size()), kPE_BEAM_WIDTH, llmRequestZ);
    EXPECT_GE(llmRequestZ->getContextCurrentPosition(), kPE_TOKENS_PER_BLOCK);
    (void) kvCacheManager->removeSequence(3, llmRequestZ);
    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));
}

// Verifies that after a sequence stores blocks in the trie and those blocks are evicted
// via interior-block eviction, subsequent sequences can still allocate and store blocks
// correctly (no trie corruption or assertion failures).
TEST_F(KVCacheManagerTest, TruePriorityEvictionNoCrashAfterInteriorEviction)
{
    auto constexpr blocksInPrimaryPool = 8;
    auto const maxAttentionWindow = kPE_TOKENS_PER_BLOCK * 10;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kPE_BEAM_WIDTH};
    auto kvCacheManager = makePriorityEvictionManager(blocksInPrimaryPool, maxAttentionWindow, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 3 blocks — MIN priority for first block (interior), DEFAULT for the rest.
    // Trie: root → B0(MIN) → B1(DEFAULT) → B2(DEFAULT)
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kPE_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kPE_IS_STREAMING);
    llmRequest0->setKvCacheRetentionConfig(KvCacheRetentionConfig(
        {KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 4, KvCacheRetentionConfig::kMinRetentionPriority)},
        KvCacheRetentionConfig::kDefaultRetentionPriority));
    addSequenceForTest(*kvCacheManager, 0, static_cast<SizeType32>(inputTokens0->size()), kPE_BEAM_WIDTH, llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);
    (void) kvCacheManager->removeSequence(0, llmRequest0);

    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    // Seq 1: 8 completely new tokens — forces eviction of B0 (MIN priority, interior node).
    // True priority eviction detaches only B0; B1, B2 remain in trie.
    auto inputTokens1
        = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111});
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kPE_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kPE_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, static_cast<SizeType32>(inputTokens1->size()), kPE_BEAM_WIDTH, llmRequest1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    kvCacheManager->storeContextBlocks(*llmRequest1);
    (void) kvCacheManager->removeSequence(1, llmRequest1);

    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    // Seq 2: 4 new tokens (fresh; no overlap with any prior sequence).
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{200, 201, 202, 203});
    auto llmRequest2
        = std::make_shared<LlmRequest>(2, kPE_MAX_NEW_TOKENS, inputTokens2, samplingConfig, kPE_IS_STREAMING);
    EXPECT_NO_THROW(addSequenceForTest(
        *kvCacheManager, 2, static_cast<SizeType32>(inputTokens2->size()), kPE_BEAM_WIDTH, llmRequest2));
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    kvCacheManager->storeContextBlocks(*llmRequest2);
    (void) kvCacheManager->removeSequence(2, llmRequest2);

    EXPECT_EQ(kvCacheManager->getNumFreeBlocks(), blocksInPrimaryPool);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));

    // Seq 3: reuses the same tokens as seq 1 — verifies that the interior-eviction
    // path left the trie in a consistent state for subsequent insertions/lookups.
    auto inputTokens3
        = std::make_shared<VecTokens>(VecTokens{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111});
    auto llmRequest3
        = std::make_shared<LlmRequest>(3, kPE_MAX_NEW_TOKENS, inputTokens3, samplingConfig, kPE_IS_STREAMING);
    EXPECT_NO_THROW(addSequenceForTest(
        *kvCacheManager, 3, static_cast<SizeType32>(inputTokens3->size()), kPE_BEAM_WIDTH, llmRequest3));

    // Seq 1's blocks are in the trie and should be reused.
    EXPECT_GT(llmRequest3->getContextCurrentPosition(), 0);

    (void) kvCacheManager->removeSequence(3, llmRequest3);
    EXPECT_TRUE(blockManager.verifyQueueIntegrity(maxAttentionWindow));
}

namespace
{
// Shared constants for all VSWA tests.
auto constexpr kVSWA_TOKENS_PER_BLOCK = 4;
auto constexpr kVSWA_ATTENTION_WINDOW = 8;
auto constexpr kVSWA_MAX_SEQUENCE_LENGTH = 128;
SizeType32 constexpr kVSWA_MAX_NEW_TOKENS = 40;
auto constexpr kVSWA_BEAM_WIDTH = 1;
auto constexpr kVSWA_BEAM_IDX = 0;
bool constexpr kVSWA_IS_STREAMING = false;
TokenIdType constexpr kVSWA_FIRST_TOKEN = 1000;

// Factory: construct and allocate a KVCacheManager for VSWA tests.
// numLayers=2, numHeads=2, sizePerHead=64, tokensPerBlock=4, attentionWindow=8,
// maxNumSequences=8, beamWidth=1, sinkTokenLength=0, maxSequenceLength=128.
std::unique_ptr<KVCacheManager> makeVSWAManager(
    SizeType32 blocksInPrimaryPool, bool enableBlockReuse, std::shared_ptr<tr::CudaStream> const& stream)
{
    auto const blocksPerWindow = BlocksPerWindow{{kVSWA_ATTENTION_WINDOW, {blocksInPrimaryPool, 0}}};
    auto mgr = std::make_unique<KVCacheManager>(2, 2, 64, kVSWA_TOKENS_PER_BLOCK, blocksPerWindow, 8, kVSWA_BEAM_WIDTH,
        std::vector<SizeType32>{kVSWA_ATTENTION_WINDOW}, nvinfer1::DataType::kHALF, 0, stream,
        kVSWA_MAX_SEQUENCE_LENGTH, /*chunkSize=*/kVSWA_MAX_SEQUENCE_LENGTH, enableBlockReuse);
    mgr->allocatePools(false);
    return mgr;
}

// Factory: construct and allocate a KVCacheManager with window==tokensPerBlock for
// multi-OOW tests.  With window=4 and tpb=4 the OOW condition fires at numTokens=8,
// so two consecutive addToken calls (after 11 context tokens) cause two OOW events
// before the next block boundary — exercising the prevBlock->isPlaceholder() path in
// storeNewBlock.
std::unique_ptr<KVCacheManager> makeSmallWindowManager(
    SizeType32 blocksInPrimaryPool, std::shared_ptr<tr::CudaStream> const& stream)
{
    SizeType32 constexpr kSmallWindow = 4;
    SizeType32 constexpr kSmallTpb = 4;
    SizeType32 constexpr kSmallMaxSeqLen = 128;
    auto const blocksPerWindow = BlocksPerWindow{{kSmallWindow, {blocksInPrimaryPool, 0}}};
    auto mgr = std::make_unique<KVCacheManager>(2, 2, 64, kSmallTpb, blocksPerWindow, 8, kVSWA_BEAM_WIDTH,
        std::vector<SizeType32>{kSmallWindow}, nvinfer1::DataType::kHALF, 0, stream, kSmallMaxSeqLen,
        /*chunkSize=*/kSmallMaxSeqLen, /*enableBlockReuse=*/true);
    mgr->allocatePools(false);
    return mgr;
}
} // namespace

// Verify that a non-stolen OOW block (hasRefs() == 0 at releaseBlocks time) is
// stored in the reuse trie and can be reused by a subsequent sequence.
TEST_F(KVCacheManagerTest, VSWANonStolenOOWBlockStoredForReuse)
{
    // SWA with a generous pool so the OOW block is never stolen.
    auto constexpr blocksInPrimaryPool = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();
    TokenIdType constexpr firstToken = kVSWA_FIRST_TOKEN;

    // Seq 0: 11 input tokens → allocates 3 blocks covering tokens [1000..1010].
    // After addToken (token 1011), numTokens==12 triggers OOW for block 0 (tokens
    // [1000..1003]) which enters the free queue at MIN priority with hasRefs()==0.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), firstToken);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the reuse trie so they are there before B0 goes OOW.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    llmRequest0->addNewToken(firstToken + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);

    // Release seq 0: the placeholder at position 0 → storeBlocks sees node K0 still has value B0
    // (not stolen) → advances prevBlock → B0's chain stored for reuse.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, llmRequest0)));

    // Seq 1: exactly the same 4-token prefix as the OOW block → must reuse it.
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), firstToken);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);

    // The OOW block was stored with 4 tokens, but S1's usableSize=4-1=3 so the
    // search key has 3 tokens.  3/4 tokens match → contextCurrentPosition == 3.
    // Any non-zero value confirms the OOW block was stored and is being reused.
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kVSWA_TOKENS_PER_BLOCK - 1);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    // All blocks must be free — no leaks.
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that storeNewBlock stores SWA blocks (including OOW blocks) into the reuse
// trie during generation — i.e., without waiting for removeSequence.
// After two generation steps (reaching a block boundary at usableSize=12), blocks
// B0 (OOW), B1, and B2 are stored.  A subsequent sequence can then reuse B0 while
// seq0 is still alive.
TEST_F(KVCacheManagerTest, VSWABlockStoredDuringGeneration)
{
    // Generous pool so no blocks are stolen.
    auto constexpr blocksInPrimaryPool = 10;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 input tokens covering blocks B0=[1000..1003], B1=[1004..1007], B2=[1008..1010] (partial).
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the reuse trie during context (invariant: stored before OOW).
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // Generation step 1: token 1011.
    // numTokens becomes 12; usableSize=11, 11%4!=0 → storeNewBlock is a no-op.
    // adjustBlocksIfNeeded: 12-0*4=12 >= 8+4=12 → B0 goes OOW (detachFrontBlock).
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // no-op (usableSize=11)

    // Generation step 2: token 1012.
    // numTokens becomes 13; usableSize=12, 12%4==0 → storeNewBlock fires.
    // storeNewBlock processes [P0, B1, B2]: P0→node K0 has value B0→advance;
    // B1→node K1 has value B1 (from context)→advance; B2→node K2 empty→insert.
    // adjustBlocksIfNeeded: 13-1*4=9 < 12 → no additional OOW detach.
    // (13-1)%4==0 → a new block B3 is allocated for position 3.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 12, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // stores B2 (B0+B1 already in trie from context)

    // Seq 1: same 4-token prefix as B0 → should reuse it without seq0 being released.
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);

    // B0 was stored during generation (not just at release time).
    // usableSize for seq1 context = 4-1=3 tokens → partial match of 3 tokens.
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kVSWA_TOKENS_PER_BLOCK - 1);

    // Clean up both sequences.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that when an OOW block is stolen by another sequence, storeBlocks does
// not restore that missing anchor under the original sequence's key or corrupt
// the acquiring sequence's trie, and all blocks are properly released.
TEST_F(KVCacheManagerTest, VSWAStolenOOWBlockNoCorruption)
{
    // Tight pool: seq0 needs 3 context blocks + 1 for addToken = 4 total.
    // Seq1 needs 2 blocks.  The one block in the free queue after seq0's addToken
    // goes to seq1, which steals the OOW block.
    auto constexpr blocksInPrimaryPool = 4;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 tokens, triggering 1 OOW block after addToken.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the trie before they can go OOW.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);

    // After addToken: B0 goes OOW (detachFrontBlock); (12-1)%4 != 0 so no new
    // block is allocated.  S0 holds B1, B2 in-window.  Pool=4: B0(DEFAULT) + B3(DEFAULT)
    // = 2 free blocks.  B0 is still in the trie (stored by storeContextBlocks).
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 2);

    // Seq 1: 8 tokens → needs 2 blocks.  It acquires B3 (DEFAULT, oldest) and B0 (DEFAULT),
    // stealing the OOW block away from seq 0.  getFreeBlock(B0) calls detachFromLookupNode,
    // removing B0 from the trie.
    auto inputTokens1 = std::make_shared<VecTokens>(8);
    std::iota(inputTokens1->begin(), inputTokens1->end(), kVSWA_FIRST_TOKEN + 100);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, 8, kVSWA_BEAM_WIDTH, llmRequest1);

    // Seq 0's removeSequence: storeBlocks sees placeholder P0 with no value
    // because B0 was detached from trie when seq1's getFreeBlock claimed it.
    // The missing anchor is not restored under seq0's key.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, llmRequest0)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));

    // All blocks must be free after both sequences are released.
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Reuse assertion: seq1 stored its 2 blocks ([kVSWA_FIRST_TOKEN+100 .. +107]) in the
    // trie during removeSequence.  A follow-up request with seq1's prefix must be able to
    // reuse at least one of those blocks, confirming that seq0's storeBlocks correctly
    // handled the stolen OOW block without corrupting the trie with seq0's stale prefix.
    auto inputTokensReuse = std::make_shared<VecTokens>(*inputTokens1);
    auto llmRequestReuse
        = std::make_shared<LlmRequest>(2, kVSWA_MAX_NEW_TOKENS, inputTokensReuse, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(
        *kvCacheManager, 2, static_cast<SizeType32>(inputTokensReuse->size()), kVSWA_BEAM_WIDTH, llmRequestReuse);
    EXPECT_GT(llmRequestReuse->getContextCurrentPosition(), 0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(2, llmRequestReuse)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify the placeholder path when the acquiring sequence finishes before the
// original sequence: the OOW block has hasRefs()==false but is stored in the trie
// under the acquirer's key. storeBlocks for the original sequence encounters a
// placeholder at the OOW position; the trie node for K_seq0_block0 has no value
// because the block is stored at seq1's key, not seq0's. The missing anchor is
// not restored under seq0's key, preserving the acquirer's trie entry for reuse.
TEST_F(KVCacheManagerTest, VSWAStolenAndReleasedOOWBlockIsInLookupTreeProtection)
{
    // Pool=3: seq0 uses all 3 blocks (B0..B2) for context. After addToken, only B0 is
    // in the free queue (no B3 exists), so seq1 must take B0 — the stolen OOW block.
    auto constexpr blocksInPrimaryPool = 3;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();
    TokenIdType constexpr seq1FirstToken = 1100;

    // Seq 0: 11 tokens → allocates B0 (tokens 1000..1003), B1 (1004..1007), B2 (1008..1010).
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the trie before B0 goes OOW.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // addToken: B0 goes OOW at DEFAULT priority; no new block allocated ((12-1)%4 != 0).
    // Free queue: [B0] — the only free block in the pool.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 1);

    // Seq 1: 4 tokens (distinct prefix) → steals B0 (the only free block).
    // getFreeBlock(B0) calls detachFromLookupNode, removing B0 from the trie.
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), seq1FirstToken);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0); // pool exhausted

    // removeSequence(1) FIRST: seq1's storeBlocks stores B0 (now holding seq1's tokens) in
    // the trie under seq1's key.  B0 is no longer in the trie at seq0's key.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 1); // B0 freed into free queue

    // removeSequence(0): seq0's storeBlocks encounters P0 (placeholder) at position 0.
    // The trie node for K_seq0_block0 has no value because B0 is stored at seq1's key.
    // The missing anchor is not restored. No crash, no trie corruption.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, llmRequest0)));

    // All 3 blocks must be free (no leaks).
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq 2: same prefix as seq1 → must reuse B0 stored at seq1's key.
    auto inputTokens2 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens2->begin(), inputTokens2->end(), seq1FirstToken);
    auto llmRequest2
        = std::make_shared<LlmRequest>(2, kVSWA_MAX_NEW_TOKENS, inputTokens2, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest2);
    // 4 tokens stored, usable key has 3 tokens → 3/4 match → contextCurrentPosition==3.
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), kVSWA_TOKENS_PER_BLOCK - 1);

    // Seq 3: same prefix as seq0's first block must NOT find B0. The missing
    // anchor was not restored, and B0 is only in the trie at seq1's key.
    auto inputTokens3 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens3->begin(), inputTokens3->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest3
        = std::make_shared<LlmRequest>(3, kVSWA_MAX_NEW_TOKENS, inputTokens3, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 3, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 0);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(2, llmRequest2)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(3, llmRequest3)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that OOW blocks are released at their original DEFAULT priority in detachFrontBlock,
// making them the first candidates for eviction over untouched DEFAULT-priority blocks.
TEST_F(KVCacheManagerTest, VSWAOOWBlockReleasedAtOriginalPriority)
{
    // Pool of 5 blocks: seq0 uses B0,B1,B2 (context) + allocates B3 on the block
    // boundary after addToken.  B4 is the single untouched DEFAULT-priority free block.
    // The OOW block B0 enters the free queue at its original DEFAULT priority — it is
    // NOT forced to MIN priority.  The next allocation follows normal LRU order among
    // equal-priority blocks and does NOT preferentially claim B0.
    auto constexpr blocksInPrimaryPool = 5;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    // Use reuse=false so seq1's addSequenceBatch does a plain allocation (no trie lookup).
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/false, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 input tokens → allocates B0, B1, B2.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);

    // Capture B0's ID before it goes OOW.
    auto const onlyWindowSize = theOnlyWindowSize(*kvCacheManager);
    auto const& seq0 = kvCacheManager->getSequence(0);
    auto const oowBlockId = seq0.getCacheBlockIds(onlyWindowSize)[kVSWA_BEAM_IDX][0];

    // addToken: B0 → free queue at DEFAULT (original) priority; B3 allocated (block boundary).
    // Free queue now: B0 (DEFAULT), B4 (DEFAULT) — same priority, LRU order applies.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);

    // After addToken: B0 OOW (DEFAULT priority). (12-1)%4 != 0 → no new block allocated.
    // S0 holds B1, B2.  Pool=5: B0(DEFAULT) + B3(DEFAULT) + B4(DEFAULT) = 3 free blocks.
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 3);

    // Seq 1: 4 tokens → needs 1 block.  Both B0 and B4 have DEFAULT priority; LRU picks
    // the block that has been free the longest (B4 was never used), NOT B0 (just released).
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), 2000);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);

    auto const& seq1 = kvCacheManager->getSequence(1);
    auto const seq1BlockId = seq1.getCacheBlockIds(onlyWindowSize)[kVSWA_BEAM_IDX][0];

    // OOW block (B0, DEFAULT priority) must NOT have been preferentially chosen over
    // the other free blocks — its priority is unchanged from context time.
    EXPECT_NE(seq1BlockId, oowBlockId);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, llmRequest0)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify the placeholder approach: when a stolen OOW block is freed by its new
// owner without being stored in the reuse trie, storeBlocks for the original
// sequence encounters a placeholder at the OOW position and finds no trie value
// for the anchor. The missing anchor is not restored under the original key.
TEST_F(KVCacheManagerTest, VSWAStolenOOWBlockPlaceholderDoesNotRestoreAnchor)
{
    // Pool=3: seq0 uses all 3 blocks (B0..B2) for context.  After addToken, only B0 is
    // in the free queue, so seq1 (1 block) must take B0 — the stolen OOW block.
    auto constexpr blocksInPrimaryPool = 3;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);

    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 tokens → allocates B0 (tokens 1000..1003), B1 (1004..1007), B2 (1008..1010).
    // storeContextBlocks stores B0 and B1 in the trie (B2 is partial, not stored).
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the trie before B0 goes OOW.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // addToken: B0 goes OOW at DEFAULT priority → placeholder P0 replaces B0 in seq0's block list.
    // (12-1)%4 != 0 → no new block.  Free queue: [B0] — the only free block in the pool.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 1);

    // Seq 1: 4 tokens (distinct prefix, starting at 2000) → steals B0 (the only free block).
    // getFreeBlock calls detachFromLookupNode(B0), removing B0 from the trie.
    TokenIdType constexpr seq1FirstToken = 2000;
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), seq1FirstToken);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0); // pool exhausted

    // Release seq1 with std::nullopt: simulates a failed/cancelled request.
    // B0 is freed back to the pool without being stored — it is no longer in the trie.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 1); // B0 freed

    // Release seq0: storeBlocks encounters P0 (placeholder) at position 0.
    // The trie node for B0's original key has no value because B0 was removed
    // by seq1's getFreeBlock. The missing anchor is not restored under seq0's key.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, llmRequest0)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Negative reuse check for B0: seq2 with seq0's OOW block prefix must NOT
    // find it because the missing anchor was not restored under seq0's key.
    auto inputTokens2 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens2->begin(), inputTokens2->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest2
        = std::make_shared<LlmRequest>(2, kVSWA_MAX_NEW_TOKENS, inputTokens2, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest2);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(2, llmRequest2)));

    // B1 is not reusable as a standalone first block because it remains below
    // seq0's missing B0 trie node.
    auto inputTokens3 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens3->begin(), inputTokens3->end(), kVSWA_FIRST_TOKEN + kVSWA_TOKENS_PER_BLOCK);
    auto llmRequest3
        = std::make_shared<LlmRequest>(3, kVSWA_MAX_NEW_TOKENS, inputTokens3, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 3, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest3);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(3, llmRequest3)));

    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that detachFrontBlock replaces the OOW block slot with a placeholder, and that
// when the OOW block is still in the trie, storeBlocks advances the search root past the
// placeholder (chain preserved) so subsequent in-window blocks are stored correctly.
TEST_F(KVCacheManagerTest, VSWAPlaceholderAdvancesSearchRootWhenOOWBlockInTrie)
{
    // Generous pool: no blocks stolen.
    auto constexpr blocksInPrimaryPool = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 tokens → B0=[1000..1003], B1=[1004..1007], B2=[1008..1010].
    // storeContextBlocks stores B0 and B1 during context.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    // Store B0 and B1 in the reuse trie during context (invariant: stored before OOW).
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // addToken step 1 (token 1011): B0 goes OOW → P0 placeholder replaces B0 in block list.
    // B0 remains in the trie at DEFAULT priority (it was stored by storeContextBlocks).
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // usableSize=11, no-op

    // addToken step 2 (token 1012): (13-1)%4=0 → block boundary, B3 allocated.
    // storeNewBlock fires with usableSize=12: processes [P0, B1, B2].
    //   insertNodes([K0,K1,K2]) finds/creates all nodes.
    //   P0 (placeholder) → node K0 has value B0 (still in trie) → advance prevBlock.
    //   B1 → node K1 has value B1 (from context) → slot occupied → advance prevBlock.
    //   B2 → node K2 is empty → insert B2 into trie.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 12, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // stores B2

    // Seq 1: same prefix as B0 ([1000..1003]) → must reuse B0 from the trie.
    auto inputTokens1 = std::make_shared<VecTokens>(kVSWA_TOKENS_PER_BLOCK);
    std::iota(inputTokens1->begin(), inputTokens1->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kVSWA_TOKENS_PER_BLOCK, kVSWA_BEAM_WIDTH, llmRequest1);
    // B0 stored with 4 tokens; seq1's usable key has 3 tokens → 3/4 match.
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kVSWA_TOKENS_PER_BLOCK - 1);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that schedulingRemoveSequence correctly skips placeholder blocks and does NOT
// call decSchedulingRefCount() on them (which would trigger a TLLM_CHECK since the
// scheduling ref count of a freshly-created placeholder is 0).
//
// After storeContextBlocks + addToken:
//   seq0.mAllocatedBlocksPerSeq = [P0 (placeholder), B1, B2]
// startScheduling() copies mRefCount → mSchedulingRefCount for every slot, including P0
// (mSchedulingRefCount = 0 for a placeholder).
// schedulingRemoveSequence must skip P0 and only decrement B1 and B2, making all
// blocksInPrimaryPool available from the scheduler's perspective.
TEST_F(KVCacheManagerTest, VSWASchedulingRemoveSequenceSkipsPlaceholders)
{
    // Pool=5: seq0 uses B0, B1, B2 for context; B3, B4 are free throughout.
    auto constexpr blocksInPrimaryPool = 5;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 input tokens → B0=[1000..1003], B1=[1004..1007], B2=[1008..1010].
    // storeContextBlocks stores B0 and B1 before they can go OOW.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // addToken: B0 goes OOW → detachFrontBlock replaces B0 with placeholder P0.
    // P0 has mRefCount=0 and isPlaceholder()==true.
    // (12-1)%4 != 0 → no new block.  Free pool: B0(DEFAULT), B3(DEFAULT), B4(DEFAULT) = 3 free.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 3);

    // startScheduling() snapshots free blocks and copies mRefCount → mSchedulingRefCount
    // for every allocated block, including the placeholder P0 (mSchedulingRefCount = 0).
    kvCacheManager->startScheduling();

    // schedulingRemoveSequence must skip P0 (isPlaceholder()==true) and only decrement
    // the scheduling ref counts of B1 and B2.  Calling decSchedulingRefCount() on P0
    // (with mSchedulingRefCount=0) would fire TLLM_CHECK_WITH_INFO and abort the test.
    EXPECT_NO_THROW(kvCacheManager->schedulingRemoveSequence(0));

    // After skipping P0 and releasing B1+B2, all 5 blocks are available for scheduling.
    EXPECT_TRUE(blockManager.schedulingHasFreeBlocks(blocksInPrimaryPool, kVSWA_ATTENTION_WINDOW));

    // Actual release: verify no leaks.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify multiple consecutive OOW placeholders are handled correctly by storeNewBlock.
//
// With window=4 and tpb=4, the OOW condition (numTokens - removed*4 >= 8) fires twice
// in a single addToken call when numTokens reaches 12 after 11 context tokens:
//   1st OOW: 12 - 0*4 = 12 >= 8 → B0 OOW → P0 inserted
//   2nd OOW: 12 - 1*4 = 8  >= 8 → B1 OOW → P1 inserted
//   seq: [P0, P1, B2]
// storeNewBlock fires on the *next* addToken (numTokens=13, usableSize=12):
//   blockKeys.size()=3, beam0Blocks=[P0, P1, B2, B3]
//   lastBlock=B2(idx 2), prevBlock=P1(idx 1)
//   prevBlock->isPlaceholder()==true → "store all blocks" path
//   storeBlocks([K0,K1,K2], [P0,P1,B2,B3]):
//     insertNodes([K0,K1,K2]) finds/creates all nodes.
//     P0 → node K0 has value B0 (storeContextBlocks) → advance prevBlock
//     P1 → node K1 has value B1 (storeContextBlocks) → advance prevBlock
//     B2 → node K2 is empty → insert
// A subsequent sequence with B0's prefix must reuse B0 (contextCurrentPosition==3).
TEST_F(KVCacheManagerTest, VSWAStoreNewBlockWithMultipleOOWPlaceholders)
{
    auto constexpr blocksInPrimaryPool = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    // window=4 == tpb=4: two OOW events occur before the next storeNewBlock boundary.
    auto kvCacheManager = makeSmallWindowManager(blocksInPrimaryPool, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();
    SizeType32 constexpr kSmallWindow = 4;
    SizeType32 constexpr kSmallTpb = 4;

    // Seq 0: 11 input tokens → B0=[1000..1003], B1=[1004..1007], B2=[1008..1010] (partial).
    // storeContextBlocks stores B0 and B1 (both full) so they are in the trie before OOW.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);

    // addToken step 1 (token 1011): numTokens=12.
    //   12 - 0*4 = 12 >= 4+4=8 → B0 OOW (P0 inserted, numFront=1)
    //   12 - 1*4 = 8  >= 8     → B1 OOW (P1 inserted, numFront=2)
    //   12 - 2*4 = 4  <  8     → stop
    //   (12-1)%4=3 != 0 → no new block.  seq: [P0, P1, B2]
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 11, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // usableSize=11, 11%4!=0 → no-op

    // addToken step 2 (token 1012): numTokens=13.
    //   13 - 2*4 = 5 < 8 → no OOW.
    //   (13-1)%4=0 → B3 allocated.  seq: [P0, P1, B2, B3]
    // storeNewBlock(usableSize=12): 12%4=0 → fires.
    //   blockKeys=[K0,K1,K2], beam0Blocks=[P0,P1,B2,B3]
    //   prevBlock = P1 (index 1) → isPlaceholder()==true → "store all blocks" path
    //   storeBlocks: P0→advance(B0 in trie); P1→advance(B1 in trie); B2→insert.
    llmRequest0->addNewToken(kVSWA_FIRST_TOKEN + 12, kVSWA_BEAM_IDX);
    kvCacheManager->addToken(0);
    kvCacheManager->storeNewBlock(*llmRequest0); // stores B2

    // Seq 1: same 4-token prefix as B0 → must reuse B0.
    // usableSize for seq1 context = kSmallTpb-1=3 tokens → partial match of 3 tokens.
    auto inputTokens1 = std::make_shared<VecTokens>(kSmallTpb);
    std::iota(inputTokens1->begin(), inputTokens1->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, kSmallTpb, kVSWA_BEAM_WIDTH, llmRequest1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), kSmallTpb - 1);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, llmRequest1)));
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that storeBlocks continues past an already-occupied non-placeholder trie slot
// and stores subsequent blocks, leaving the trie intact for later sequences to reuse.
//
// Scenario:
//   Seq 0: tokens [K0, K1_A] → B0 at K0, B1_A at K1_A stored in trie on release.
//   Seq 1: tokens [K0, K1_B] → reuses B0, allocates B1_B fresh; B1_B stored at K1_B.
//   Seq 2: tokens [K0, K1_A] → reuses B0 and B1_A from trie (contextCurrentPosition > 0).
//   Seq 2 release storeBlocks sees K0 and K1_A already occupied — skips both cleanly.
//   Seq 3: same tokens as seq 2 → must still reuse B0 and B1_A (trie not corrupted).
TEST_F(KVCacheManagerTest, VSWAStoreBlocksSkipsOccupiedSlotsAndContinues)
{
    auto constexpr blocksInPrimaryPool = 8;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();

    TokenIdType constexpr kSharedFirst = kVSWA_FIRST_TOKEN;      // shared B0 prefix
    TokenIdType constexpr kSeqASecond = kVSWA_FIRST_TOKEN + 100; // B1_A suffix
    TokenIdType constexpr kSeqBSecond = kVSWA_FIRST_TOKEN + 200; // B1_B suffix

    // Seq 0: 9 tokens → B0=[1000..1003], B1_A=[1100..1103], partial B2.
    auto tokens0 = std::make_shared<VecTokens>(9);
    std::iota(tokens0->begin(), tokens0->begin() + kVSWA_TOKENS_PER_BLOCK, kSharedFirst);
    std::iota(tokens0->begin() + kVSWA_TOKENS_PER_BLOCK, tokens0->end(), kSeqASecond);
    auto req0 = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, tokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 9, kVSWA_BEAM_WIDTH, req0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req0);
    kvCacheManager->storeContextBlocks(*req0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, req0)));
    // Trie: B0 at K0, B1_A at K1_A under B0.

    // Seq 1: 9 tokens → reuses B0, allocates B1_B for the distinct K1_B suffix.
    auto tokens1 = std::make_shared<VecTokens>(9);
    std::iota(tokens1->begin(), tokens1->begin() + kVSWA_TOKENS_PER_BLOCK, kSharedFirst);
    std::iota(tokens1->begin() + kVSWA_TOKENS_PER_BLOCK, tokens1->end(), kSeqBSecond);
    auto req1 = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, tokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, 9, kVSWA_BEAM_WIDTH, req1);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req1);
    kvCacheManager->storeContextBlocks(*req1);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, req1)));
    // Trie: B0 at K0, B1_A at K1_A and B1_B at K1_B (both children of B0).

    // Seq 2: same prefix as seq 0 ([K0, K1_A, ...]) → reuses B0 and B1_A.
    auto tokens2 = std::make_shared<VecTokens>(*tokens0);
    auto req2 = std::make_shared<LlmRequest>(2, kVSWA_MAX_NEW_TOKENS, tokens2, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 2, 9, kVSWA_BEAM_WIDTH, req2);
    EXPECT_GT(req2->getContextCurrentPosition(), 0);

    // storeBlocks for seq 2: K0 and K1_A are both occupied → skips both without crash.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(2, req2)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq 3: same tokens as seq 2 → B0 and B1_A must still be reusable (trie intact).
    auto tokens3 = std::make_shared<VecTokens>(*tokens0);
    auto req3 = std::make_shared<LlmRequest>(3, kVSWA_MAX_NEW_TOKENS, tokens3, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 3, 9, kVSWA_BEAM_WIDTH, req3);
    EXPECT_GT(req3->getContextCurrentPosition(), 0);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(3, req3)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Verify that storeBlocksForReuse with pinBlocks=true pins both already-in-trie blocks
// (occupied slots, advanced via prevBlock) and newly stored blocks (empty slots).
// After removeSequence the pinned blocks must NOT be in the free pool, and unpinning
// them restores the full pool.
//
// Setup:
//   Seq 0: 11 tokens → storeContextBlocks stores B0=[1000..1003] and B1=[1004..1007]
//          (both full). B2=[1008..1010] is partial and NOT stored.
//          removeSequence with std::nullopt: no storeBlocks; B0 and B1 remain in the
//          trie (cached), B2 is freed.
//   Seq 1: same 11 tokens → reuses B0 and B1; allocates fresh B2'.
//   storeBlocksForReuse(pinBlocks=true):
//     usableSize = 10 → blockKeys = [K0_full, K1_full, K2_partial]
//     K0 → occupied by B0  → pin B0  (occupied-slot path)
//     K1 → occupied by B1  → pin B1  (occupied-slot path)
//     K2 → empty           → store B2' + pin B2'  (empty-slot path)
//   pinnedIds.size() == 3.
TEST_F(KVCacheManagerTest, VSWAStoreBlocksForReuseWithPinBlocksPinsAllChainBlocks)
{
    auto constexpr blocksInPrimaryPool = 6;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};
    auto kvCacheManager = makeVSWAManager(blocksInPrimaryPool, /*enableBlockReuse=*/true, stream);
    auto const& blockManager = kvCacheManager->getBlockManager();

    // Seq 0: 11 tokens → B0 (full), B1 (full) stored by storeContextBlocks; B2 partial.
    // Release with std::nullopt so storeBlocks is NOT called → B2 slot stays empty in trie.
    auto inputTokens0 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 0, 11, kVSWA_BEAM_WIDTH, llmRequest0);
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager->storeContextBlocks(*llmRequest0);
    // Release without storing: B0 and B1 remain in the trie; B2 freed.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(0, std::nullopt)));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Seq 1: same 11 tokens → reuses B0 and B1 (contextCurrentPosition > 0); allocates B2'.
    auto inputTokens1 = std::make_shared<VecTokens>(11);
    std::iota(inputTokens1->begin(), inputTokens1->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest1
        = std::make_shared<LlmRequest>(1, kVSWA_MAX_NEW_TOKENS, inputTokens1, samplingConfig, kVSWA_IS_STREAMING);
    addSequenceForTest(*kvCacheManager, 1, 11, kVSWA_BEAM_WIDTH, llmRequest1);
    EXPECT_GT(llmRequest1->getContextCurrentPosition(), 0);

    // storeBlocksForReuse with pinBlocks=true:
    //   usableSize=10 → blockKeys=[K0_full, K1_full, K2_partial], beam0Blocks=[B0,B1,B2']
    //   K0 occupied by B0 → skip + pin B0.  K1 occupied by B1 → skip + pin B1.
    //   K2 empty → store B2' + pin B2'.  Total: 3 pinned blocks.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest1);
    auto pinnedIds = kvCacheManager->storeBlocksForReuse(1, llmRequest1, /*pinBlocks=*/true);
    EXPECT_EQ(static_cast<SizeType32>(pinnedIds.size()), 3);

    // removeSequence releases the sequence's ref; pinned blocks keep their extra ref.
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager->removeSequence(1, std::nullopt)));
    EXPECT_LT(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    // Unpinning restores the full pool.
    kvCacheManager->unpinBlocksById(pinnedIds);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);
}

// Regression test for thorjohnsen review comment #2934049162.
//
// Scenario: a sequence produces N > window blocks, so early blocks go OOW and are
// represented as SWA placeholders in the sequence. If one OOW anchor is later
// evicted from the lookup tree, a new sequence with a sufficiently long shared
// prefix must still traverse the value-less anchor and reuse/store later blocks
// once that missing anchor is outside the sliding attention window.
//
// Construction (tpb = 4, window = 12 = 3 blocks):
//  - Seq 0: 28 tokens = 7 blocks [b0..b6]. After context and sliding,
//    blocks b0..b3 go OOW; storeContextBlocks stores b0..b5 in the trie.
//  - Detach b0's trie value directly to model an evicted OOW anchor while
//    leaving b1..b5 nodes below it.
//  - Seq 2: same 28-token prompt. addSequenceBatch can safely prepopulate b1..b5
//    through the missing b0 anchor because b0 is outside the 12-token window by
//    then. removeSequence must then attach the trailing partial b6 behind the
//    reused descendants instead of stopping at the value-less b0 node.
TEST_F(KVCacheManagerTest, VSWAEvictedPlaceholderAnchorAllowsTrailingReuse)
{
    auto constexpr tpb = 4;
    auto constexpr window = 3 * tpb;  // 12 tokens = 3 blocks
    auto constexpr numBlocksSeq0 = 7; // 7 blocks = 28 tokens in seq0
    auto constexpr blocksInPrimaryPool = 16;
    auto const stream = std::make_shared<tr::CudaStream>();
    tr::SamplingConfig const samplingConfig{kVSWA_BEAM_WIDTH};

    auto const blocksPerWindow = BlocksPerWindow{{window, {blocksInPrimaryPool, 0}}};
    KVCacheManager kvCacheManager(2, 2, 64, tpb, blocksPerWindow, 8, kVSWA_BEAM_WIDTH, std::vector<SizeType32>{window},
        nvinfer1::DataType::kHALF, 0, stream,
        /*maxSequenceLength=*/128, /*chunkSize=*/128, /*enableBlockReuse=*/true);
    kvCacheManager.allocatePools(false);
    auto const& blockManager = kvCacheManager.getBlockManager();

    auto const makeBlockKeys = [&](SizeType32 usableTokens, bool allowPartial)
    {
        auto prefixTokens = std::make_shared<VecTokens>(usableTokens);
        std::iota(prefixTokens->begin(), prefixTokens->end(), kVSWA_FIRST_TOKEN);
        auto prefixRequest
            = std::make_shared<LlmRequest>(99, kVSWA_MAX_NEW_TOKENS, prefixTokens, samplingConfig, kVSWA_IS_STREAMING);
        auto const& uniqueTokens = prefixRequest->getUniqueTokens(kVSWA_BEAM_IDX);
        auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableTokens, tpb, allowPartial);
        return buildBlockKeys(blockedUniqueTokens, *prefixRequest);
    };

    // Seq 0: 28 tokens covering 7 blocks.
    auto inputTokens0 = std::make_shared<VecTokens>(numBlocksSeq0 * tpb);
    std::iota(inputTokens0->begin(), inputTokens0->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest0
        = std::make_shared<LlmRequest>(0, kVSWA_MAX_NEW_TOKENS, inputTokens0, samplingConfig, kVSWA_IS_STREAMING);
    kvCacheManager.addSequenceBatch({{{0, numBlocksSeq0 * tpb, kVSWA_BEAM_WIDTH}}}, {std::ref(*llmRequest0)});

    // Simulate prefill completion so storeContextBlocks honors the full context extent.
    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(0, std::nullopt)));

    // Sanity: storeContextBlocks uses totalTokens - 1 usable tokens once prefill
    // completes, so the first six full blocks should be in the reuse trie.
    auto const freeBlocksBaseline = blockManager.getNumFreeBlocks();
    EXPECT_EQ(freeBlocksBaseline, blocksInPrimaryPool);

    auto const b0Keys = makeBlockKeys(tpb, /*allowPartial=*/false);
    auto const b4Keys = makeBlockKeys(5 * tpb, /*allowPartial=*/false);
    auto const b6PartialKeys = makeBlockKeys(numBlocksSeq0 * tpb - 1, /*allowPartial=*/true);
    auto b0Block = kvCacheManager.findBlocksInReuseTreeByBlockKeys(b0Keys, window);
    ASSERT_NE(b0Block, nullptr);
    ASSERT_NE(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b4Keys, window), nullptr);
    ASSERT_EQ(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b6PartialKeys, window), nullptr);

    // Model eviction of the first OOW anchor. Descendant trie nodes are still
    // present, and SWA lookup may traverse through b0 once the matched suffix
    // makes b0 fall outside the active window.
    b0Block->detachFromLookupNode();
    ASSERT_EQ(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b0Keys, window), nullptr);
    ASSERT_NE(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b4Keys, window), nullptr);
    ASSERT_EQ(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b6PartialKeys, window), nullptr);

    // Seq 2: same 28-token prompt. addSequenceBatch sees 27 reusable token states:
    // b0 is a traversal-only missing anchor, b1..b5 are full matches, and b6 is
    // not present yet. The safe prepopulated length is therefore 24 tokens.
    auto inputTokens2 = std::make_shared<VecTokens>(numBlocksSeq0 * tpb);
    std::iota(inputTokens2->begin(), inputTokens2->end(), kVSWA_FIRST_TOKEN);
    auto llmRequest2
        = std::make_shared<LlmRequest>(2, kVSWA_MAX_NEW_TOKENS, inputTokens2, samplingConfig, kVSWA_IS_STREAMING);
    kvCacheManager.addSequenceBatch({{{2, numBlocksSeq0 * tpb, kVSWA_BEAM_WIDTH}}}, {std::ref(*llmRequest2)});

    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 6 * tpb)
        << "safe SWA reuse should continue past the missing OOW anchor";

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*llmRequest2);
    EXPECT_NO_THROW(static_cast<void>(kvCacheManager.removeSequence(2, llmRequest2)));
    ASSERT_NE(kvCacheManager.findBlocksInReuseTreeByBlockKeys(b6PartialKeys, window), nullptr);
}

///////////////////////////////////////////////////////////////////////////////
// KvCacheConnector decode-time block allocation tests
//
// Repro and coverage for https://github.com/NVIDIA/TensorRT-LLM/issues/13320:
// when a kv_connector::KvCacheConnectorManager is attached, decode-time
// addToken() must still grow GenerationRequest::mCacheBlockIds at every
// tokens_per_block boundary. Failing to do so causes decode KV writes to
// overwrite the prefill block, silently corrupting attention outputs.
//
// The C++ source (KVCacheManager::addToken -> BlockManager::adjustBlocksIfNeeded
// -> WindowBlockManager::allocateBlock) does not branch on
// mKvCacheConnectorManager on the decode path, so on paper these tests should
// pass with a connector attached. They are intentionally written to exercise
// both the "attached but reports zero matches" path and the "attached and
// reports >0 matches" path so a regression in either direction surfaces.
///////////////////////////////////////////////////////////////////////////////

namespace
{
// Minimal mock matching the KvCacheConnectorManager interface declared in
// cpp/include/tensorrt_llm/batch_manager/kvCacheConnector.h. Returns a
// configurable number of "externally matched" tokens, the same hook the
// real Dynamo / KVBM connector uses.
class MockKvCacheConnectorManager : public kv_connector::KvCacheConnectorManager
{
public:
    explicit MockKvCacheConnectorManager(SizeType32 numNewMatchedTokens)
        : mNumNewMatchedTokens(numNewMatchedTokens)
    {
    }

    SizeType32 getNumNewMatchedTokens(LlmRequest const& /*request*/, SizeType32 /*numComputedTokens*/) override
    {
        ++mCallCount;
        return mNumNewMatchedTokens;
    }

    SizeType32 getCallCount() const
    {
        return mCallCount;
    }

private:
    SizeType32 mNumNewMatchedTokens{0};
    SizeType32 mCallCount{0};
};

// Build a small KVCacheManager wired to the supplied connector. tokensPerBlock=4
// keeps the boundary math obvious; 16 primary blocks is plenty for the
// scenarios below.
std::unique_ptr<KVCacheManager> makeConnectorTestKVCacheManager(
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> const& stream,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> connector)
{
    auto constexpr numLayers = 1;
    auto constexpr numKvHeads = 1;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 4;
    auto constexpr beamWidth = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * 8;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    auto mgr
        = std::make_unique<KVCacheManager>(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            /*dtype*/ nvinfer1::DataType::kHALF,
            /*sinkTokenLength*/ 0, stream,
            /*maxSequenceLength*/ maxAttentionWindow,
            /*chunkSize*/ maxAttentionWindow,
            /*enableBlockReuse*/ true,
            /*cacheType*/ CacheType::kSELF,
            /*secondaryOffloadMinPriority*/ std::nullopt,
            /*eventManager*/ nullptr,
            /*enablePartialReuse*/ true,
            /*copyOnPartialReuse*/ true,
            /*kvCacheConnectorManager*/ std::move(connector));
    mgr->allocatePools(false);
    return mgr;
}

// Drive a single-request decode loop until mNumTokens reaches `targetNumTokens`,
// returning the cache block ids observed at every step. The first entry in the
// returned vector corresponds to the post-prefill state; each subsequent entry
// is the state after one addToken call.
std::vector<std::vector<SizeType32>> driveDecode(
    KVCacheManager& mgr, LlmRequest::RequestIdType reqId, SizeType32 targetNumTokens)
{
    auto const windowSize = theOnlyWindowSize(mgr);
    std::vector<std::vector<SizeType32>> trace;
    trace.push_back(mgr.getCacheBlockIds(reqId, windowSize).at(0));

    while (mgr.getSequence(reqId).getNumTokens() < targetNumTokens)
    {
        mgr.addToken(reqId);
        trace.push_back(mgr.getCacheBlockIds(reqId, windowSize).at(0));
    }
    return trace;
}
} // namespace

// Test 1: connector attached but reports zero external matches. Decode-time
// boundary allocation must still fire; this is the simplest possible
// invariant a KvCacheConnector must not break.
TEST_F(KVCacheManagerTest, KvCacheConnector_DecodeBlockBoundary_NoExternalMatches)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    auto connector = std::make_shared<MockKvCacheConnectorManager>(/*numNewMatchedTokens=*/0);
    auto mgr = makeConnectorTestKVCacheManager(stream, connector);

    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Prompt with 2 tokens -> exactly one block at prefill.
    auto tokens = std::make_shared<VecTokens>(VecTokens{0, 1});
    auto const promptLen = static_cast<SizeType32>(tokens->size());
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto req = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{0}, maxNewTokens, tokens, samplingConfig, /*isStreaming=*/false);

    mgr->addSequenceBatch({{{req->mRequestId, promptLen, beamWidth}}}, {std::ref(*req)});

    auto const windowSize = theOnlyWindowSize(*mgr);
    EXPECT_EQ(mgr->getSequence(req->mRequestId).getNumTokens(), promptLen);
    ASSERT_EQ(mgr->getCacheBlockIds(req->mRequestId, windowSize).at(0).size(), 1U);

    // Drive decode past the first block boundary. Boundary fires when
    // (mNumTokens - 1) % tokensPerBlock == 0, i.e. at mNumTokens == 5 with
    // tokensPerBlock == 4. Stop at 7 so we are firmly past the boundary.
    auto const targetNumTokens = static_cast<SizeType32>(2 * tokensPerBlock - 1); // 7
    auto trace = driveDecode(*mgr, req->mRequestId, targetNumTokens);

    EXPECT_EQ(mgr->getSequence(req->mRequestId).getNumTokens(), targetNumTokens);

    auto const finalIds = mgr->getCacheBlockIds(req->mRequestId, windowSize).at(0);
    EXPECT_GT(finalIds.size(), 1U) << "Decode never allocated a second block despite crossing tokensPerBlock boundary; "
                                      "trace size="
                                   << trace.size() << ", final block count=" << finalIds.size();
    EXPECT_EQ(finalIds.size(), 2U);

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
    (void) mgr->removeSequence(req->mRequestId, req);
}

// Test 2: connector attached AND reports a non-zero external match count
// during prefill (the production path used by the Dynamo / KVBM connector).
// Decode-time boundary allocation must still fire correctly.
TEST_F(KVCacheManagerTest, KvCacheConnector_DecodeBlockBoundary_WithExternalMatches)
{
    auto const stream = std::make_shared<tr::CudaStream>();
    // Report 4 matched tokens (exactly one block). The connector contract is
    // that the worker has already loaded these into the allocated block.
    auto constexpr numConnectorMatched = 4;
    auto connector = std::make_shared<MockKvCacheConnectorManager>(numConnectorMatched);
    auto mgr = makeConnectorTestKVCacheManager(stream, connector);

    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    // Prompt with 6 tokens -> two blocks at prefill (block0 full, block1 with 2 tokens).
    auto tokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5});
    auto const promptLen = static_cast<SizeType32>(tokens->size());
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    auto req = std::make_shared<LlmRequest>(
        LlmRequest::RequestIdType{1}, maxNewTokens, tokens, samplingConfig, /*isStreaming=*/false);

    mgr->addSequenceBatch({{{req->mRequestId, promptLen, beamWidth}}}, {std::ref(*req)});

    auto const windowSize = theOnlyWindowSize(*mgr);
    EXPECT_EQ(mgr->getSequence(req->mRequestId).getNumTokens(), promptLen);
    ASSERT_EQ(mgr->getCacheBlockIds(req->mRequestId, windowSize).at(0).size(), 2U)
        << "Prefill must allocate ceil(promptLen / tokensPerBlock) blocks regardless of connector matches";

    EXPECT_GT(connector->getCallCount(), 0);
    EXPECT_EQ(req->getPrepopulatedPromptLen(), numConnectorMatched);

    // Drive decode well past two more block boundaries (mNumTokens == 9 and 13).
    auto const targetNumTokens = static_cast<SizeType32>(promptLen + 2 * tokensPerBlock + 1); // 15
    auto trace = driveDecode(*mgr, req->mRequestId, targetNumTokens);

    EXPECT_EQ(mgr->getSequence(req->mRequestId).getNumTokens(), targetNumTokens);

    auto const finalIds = mgr->getCacheBlockIds(req->mRequestId, windowSize).at(0);
    auto const expectedNumBlocks = tc::ceilDiv(targetNumTokens, tokensPerBlock);
    EXPECT_EQ(finalIds.size(), static_cast<size_t>(expectedNumBlocks))
        << "With connector reporting " << numConnectorMatched << " matched tokens, expected " << expectedNumBlocks
        << " blocks after " << targetNumTokens << " total tokens, got " << finalIds.size() << " (issue #13320)";

    tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
    (void) mgr->removeSequence(req->mRequestId, req);
}

// Test 3: parity check. Run the same decode workload with and without a
// connector attached and assert mCacheBlockIds growth is identical. Any
// future regression that gates allocateBlock on mKvCacheConnectorManager
// will diverge the two traces.
TEST_F(KVCacheManagerTest, KvCacheConnector_DecodeBlockBoundary_ParityWithBaseline)
{
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto constexpr tokensPerBlock = 4;

    auto runScenario = [&](std::shared_ptr<kv_connector::KvCacheConnectorManager> connector)
    {
        auto mgr = makeConnectorTestKVCacheManager(stream, std::move(connector));
        auto tokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4});
        auto const promptLen = static_cast<SizeType32>(tokens->size());
        SizeType32 constexpr maxNewTokens{0};
        tr::SamplingConfig const samplingConfig{beamWidth};
        auto req = std::make_shared<LlmRequest>(
            LlmRequest::RequestIdType{0}, maxNewTokens, tokens, samplingConfig, /*isStreaming=*/false);
        mgr->addSequenceBatch({{{req->mRequestId, promptLen, beamWidth}}}, {std::ref(*req)});

        auto const targetNumTokens = static_cast<SizeType32>(3 * tokensPerBlock + 2); // 14
        auto trace = driveDecode(*mgr, req->mRequestId, targetNumTokens);

        std::vector<size_t> sizes;
        sizes.reserve(trace.size());
        for (auto const& step : trace)
        {
            sizes.push_back(step.size());
        }

        tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion(*req);
        (void) mgr->removeSequence(req->mRequestId, req);
        return sizes;
    };

    auto baseline = runScenario(/*connector=*/nullptr);
    auto withConnector = runScenario(std::make_shared<MockKvCacheConnectorManager>(/*numNewMatchedTokens=*/0));

    ASSERT_EQ(baseline.size(), withConnector.size());
    for (size_t i = 0; i < baseline.size(); ++i)
    {
        EXPECT_EQ(baseline[i], withConnector[i])
            << "Block count diverged at decode step " << i << " (baseline=" << baseline[i]
            << ", withConnector=" << withConnector[i] << "); see issue #13320";
    }
}
