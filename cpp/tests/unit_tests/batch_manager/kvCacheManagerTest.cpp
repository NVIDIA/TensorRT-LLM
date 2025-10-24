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

#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
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
    auto constexpr onboardBlocks = true;

    auto constexpr beamWidth = 8;
    auto constexpr numBlocksPerBeam = blocksInPrimaryPool / beamWidth;
    auto constexpr numTokens = tokensPerBlock * numBlocksPerBeam;
    auto constexpr maxAttentionWindow = numTokens;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
    blockManager.allocatePools(false);

    EXPECT_EQ(blockManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(blockManager.getMaxNumBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    auto constexpr requestId = 42;
    GenerationRequest seq0{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    blockManager.addSequence(seq0, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/false);
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

    blockManager.addSequence(seq0, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/true);
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
    EXPECT_NO_THROW(
        blockManager.addSequence(seq0, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/false));
    GenerationRequest seq1{requestId + 1, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_NO_THROW(
        blockManager.addSequence(seq1, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/false));
    // same requestId not allowed
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_THROW(
        blockManager.addSequence(seq2, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/false),
        std::runtime_error);
    // no more blocks
    GenerationRequest seq3{requestId + 2, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    EXPECT_THROW(
        blockManager.addSequence(seq3, numBlocksPerBeam, maxAttentionWindow, /*isShareLastContextBlock=*/false),
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
            ::write(fd, buffer.data(), poolBlockSize * sizeof(T));
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
    auto constexpr onboardBlocks = true;

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
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, type, 0, onboardBlocks);
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
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    auto inputTokens1 = inputTokens;
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{requestId, inputLength1, beamWidth, blockManager.getWindowSizesMetadata()};
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 16);
    auto cacheBlockIds1 = seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds1, ::testing::ElementsAreArray({0, 1, 6}));
    // store blocks 0, 1 ([0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15])
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
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
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

    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseSequence(seq1.getRequestId());
    blockManager.releaseSequence(seq2.getRequestId());

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

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        false, stream, true, onboardBlocks);

    // Add sequence [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] (17 tokens, three blocks)
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto const inputLength = static_cast<SizeType32>(inputTokens->size());
    LlmRequest::RequestIdType requestId{0};
    auto llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest0);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 0);
    auto cacheBlockIds = kvCacheManager.getSequence(requestId).getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(cacheBlockIds, ::testing::ElementsAreArray({0, 1, 2}));

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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr beamWidth = 1;

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kFP4,
        false, stream, maxAttentionWindow, true, onboardBlocks);

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
    auto constexpr onboardBlocks = true;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
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
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    // at this point, block 3 contains [8]
    llmRequest1->addNewToken(9, beamIdx);  // block 3 contains [8, 9]
    llmRequest1->addNewToken(10, beamIdx); // block 3 contains [8, 9, 10]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed (blocks contain [8, 9])
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
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
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), promptLen0 - 1);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // note that seq0 is holding blocks 0, 1 and 2 until releaseBlocks is called

    // input tokens [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    // reuse blocks 0, 1 ([0, 1, 2, 3], [4, 5, 6, 7]) and get new block 4
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest1->addNewToken(10, beamIdx); // block 4 contains [8, 9, 10]
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // block 2 is stored for reuse (block contains [8]). nb! Last token of last block is never stored
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block is never stored
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
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
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
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
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), numTokens - 1);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    llmRequest3->addNewToken(11, beamIdx); // block 4 contains [8, 9, 11]
    numTokens = llmRequest3->getNumTokens(beamIdx);
    // one block used by both seq2 and seq3
    numBlocks += tc::ceilDiv(numTokens, tokensPerBlock) - 1;
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 5 is not stored since it is last block and has only one token
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseSequence(seq2.getRequestId());
    // block 4 is stored for reuse (block contains [8, 9]). nb! Last token of last block not stored
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseSequence(seq3.getRequestId());
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
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq4, llmRequest4Short);
    blockManager.releaseSequence(seq4.getRequestId());
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
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4, maxAttentionWindow);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), promptLen4 - 2);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    blockManager.releaseBlocks(seq4, llmRequest4);
    blockManager.releaseSequence(seq4.getRequestId());
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
    blockManager.holdSequence(seq5.getRequestId());
    blockManager.addSequence(seq5, promptLen5, numContextBlocks5, *llmRequest5, maxAttentionWindow);
    llmRequest5->addNewToken(0, beamIdx);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 1); // incidental reuse

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), blocksInPrimaryPool);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    blockManager.releaseBlocks(seq5, llmRequest5);
    blockManager.releaseSequence(seq5.getRequestId());
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
    blockManager.holdSequence(seq6.getRequestId());
    blockManager.addSequence(seq6, promptLen6, numContextBlocks6, *llmRequest6, maxAttentionWindow);
    llmRequest6->addNewToken(0, beamIdx);
    // no reuse occurs because we are unable to reuse last input token and inputLength6 == 1.
    EXPECT_EQ(llmRequest6->getContextCurrentPosition(), 0);

    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - 1);

    blockManager.releaseBlocks(seq6, llmRequest6);
    blockManager.releaseSequence(seq6.getRequestId());
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
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and then remove it
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds, numReturnSequences);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 0, 1 and reuse block 2
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1, numReturnSequences);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [(2, 0), (3, 0), (4, 0)])
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2, numReturnSequences);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
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
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3, numReturnSequences);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new block 8, 9
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 8, 9}));
    llmRequest3->addNewToken(3, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseSequence(seq2.getRequestId());
    blockManager.releaseSequence(seq3.getRequestId());
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
    auto constexpr onboardBlocks = true;
    auto constexpr numReturnSequences = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and same multimodal hash - should reuse
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        multimodalHashes, multimodalPositions, multimodalLengths, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // should reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // block 3 matches block 2 and will be freed
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences);

    GenerationRequest seq2{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new blocks 4, 5, 6
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt,
        std::nullopt, false, std::nullopt, false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt,
        std::nullopt, LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences);
    GenerationRequest seq3{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new blocks 7, 8
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(),
        tokensPerBlock); // only reuse block 0 [100, 101, 102, 103] with same hash/offset
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 7, 8}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    // clean up
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseSequence(seq2.getRequestId());
    blockManager.releaseSequence(seq3.getRequestId());
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
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // get new blocks 0, 1, 2 ([0,1,2,3], [4,5,6,7], [8])
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens and loraTaskId, then remove it
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    requestId = 1;
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // reuse blocks 0, 1 and get new block 3
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 3}));
    llmRequest1->addNewToken(9, beamIdx);
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // store block 3 for reuse ([8,9])
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    // inputTokens = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    // reuse blocks 0, 1 and get new block 4
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
    // nb! addNewToken adds new generated token, number of input tokens stay the same.
    // calling addNewToken before addSequence potentially triggers this error message:
    // Assertion failed: prepopulatedPromptLen < promptLen
    // because maximum value for prepopulatedPromptLen is number of input+output tokens - 1,
    // but promptLen is number of input tokens.
    llmRequest0->addNewToken(9, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 4}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // inputTokens1 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    // reuse 0, 1, 2(p) ([0,1,2,3], [4,5,6,7], [8])
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2}));
    llmRequest1->addNewToken(10, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks + 1);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks - 1);

    // store block 4 for reuse ([8])
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 2 is stored for reuse (block contains [8, 9]). nb! Last token of last block is not stored
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1
    loraTaskId = static_cast<LlmRequest::LoraTaskIdType>(1);
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 5, 6, 7
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
    // no reuse expected. Input tokens match blocks 0 and 1, but lora task id differs.
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest2->addNewToken(9, beamIdx);
    numTokens = llmRequest2->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store blocks 5, 6, 7 for reuse ([0,1,2,3], [4,5,6,7], [8]) with loraTaskId 1
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseSequence(seq2.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and more tokens
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11});
    requestId = 3;
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens3, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse blocks 5, 6, 7(p) ([0,1,2,3], [4,5,6,7], [8])
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), promptLen3 - 2);
    EXPECT_THAT(seq3.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({5, 6, 7}));
    llmRequest3->addNewToken(11, beamIdx);
    numTokens = llmRequest3->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // store block 7 for reuse ([8,9]) with loraTaskId 1
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseSequence(seq3.getRequestId());
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse blocks 0, get new block 8
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4, maxAttentionWindow);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 8}));
    llmRequest4->addNewToken(5, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 8 is stored with [4] and loraTaskId 0
    blockManager.releaseBlocks(seq4, llmRequest4);
    blockManager.releaseSequence(seq4.getRequestId());
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
    blockManager.holdSequence(seq5.getRequestId());
    blockManager.addSequence(seq5, promptLen5, numContextBlocks5, *llmRequest5, maxAttentionWindow);
    EXPECT_EQ(llmRequest5->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq5.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({9, 10, 11}));
    llmRequest5->addNewToken(9, beamIdx);
    numTokens = llmRequest5->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    // blocks 9, 10, 11 are stored without loraTaskId
    blockManager.releaseBlocks(seq5, llmRequest5);
    blockManager.releaseSequence(seq5.getRequestId());
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
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    ///////////////////////////////////////////////////////////////////////////
    // add request with loraTaskId 1 and then remove it
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // new request with same tokens but different loraTaskId and then remove it
    requestId = 1;
    LlmRequest::LoraTaskIdType loraTaskId2 = static_cast<LlmRequest::LoraTaskIdType>(2);
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId2, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // no reuse, get new block 3, 4, 5
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // blocks 3, 4, 5 are stored for reuse (block 5 contains [(2, 0), (3, 0)] with loraTaskId 2)
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add both requests again and then remove them
    llmRequest0 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds);
    promptLen0 = llmRequest0->getNumTokens(beamIdx);
    numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    // reuse blocks 0, 1 and get new block 6
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
    llmRequest0->addNewToken(3, beamIdx);
    EXPECT_EQ(llmRequest0->getContextCurrentPosition(), 2 * tokensPerBlock);
    EXPECT_THAT(seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 6}));
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // reuse blocks 3, 4 and reuse block 5
    auto inputTokens1 = std::make_shared<VecTokens>(llmRequest1->getTokens(0));
    auto inputTokenExtraIds1 = std::make_shared<VecTokenExtraIds>(*inputTokenExtraIds);
    inputTokenExtraIds1->push_back(0);
    inputTokenExtraIds1->push_back(0);
    llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens1, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId2, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds1);
    promptLen1 = llmRequest1->getNumTokens(beamIdx);
    numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), llmRequest1->getNumTokens(beamIdx) - 1);
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));
    llmRequest1->addNewToken(5, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // add request with totally different extra ids and loraTaskId 1
    auto inputTokenExtraIds2 = std::make_shared<VecTokenExtraIds>(VecTokenExtraIds{4, 4, 5, 5, 6, 6, 0, 0, 0});
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds2);

    numTokens = llmRequest2->getNumTokens(beamIdx);
    GenerationRequest seq2{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // no reuse, get new block 7, 8, 9
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId1, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest3->getNumTokens(beamIdx);
    GenerationRequest seq3{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 0, get new block 10, 11
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
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
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, loraTaskId2, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, inputTokenExtraIds3);

    numTokens = llmRequest4->getNumTokens(beamIdx);
    GenerationRequest seq4{requestId, numTokens, beamWidth, blockManager.getWindowSizesMetadata()};
    // reuse block 3, get new block 12, 13
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4, maxAttentionWindow);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), tokensPerBlock);
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 12, 13}));
    llmRequest4->addNewToken(3, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 3);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 3);

    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseBlocks(seq4, llmRequest4);
    blockManager.releaseSequence(seq2.getRequestId());
    blockManager.releaseSequence(seq3.getRequestId());
    blockManager.releaseSequence(seq4.getRequestId());
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
    auto constexpr onboardBlocks = true;
    auto constexpr numReturnSequences = 1;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt); // No cache_salt_id

    GenerationRequest seq0{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Add first request and get blocks 0, 1, 2
    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);
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
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseSequence(seq0.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 2: Request with same tokens but with cache_salt_id = 12345
    requestId = 1;
    CacheSaltIDType cacheSaltId1{12345};
    auto llmRequest1 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId1); // With cache_salt_id = 12345

    GenerationRequest seq1{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should NOT reuse blocks despite same tokens, because cache_salt_id is different
    auto promptLen1 = llmRequest1->getNumTokens(beamIdx);
    auto numContextBlocks1 = tc::ceilDiv(promptLen1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, promptLen1, numContextBlocks1, *llmRequest1, maxAttentionWindow);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0); // No reuse, starts from scratch
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 5}));

    llmRequest1->addNewToken(3, beamIdx);
    llmRequest1->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Release blocks
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq1.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 3: Request with same tokens and same cache_salt_id = 12345
    requestId = 2;
    auto llmRequest2 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId1); // Same cache_salt_id = 12345

    GenerationRequest seq2{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // SHOULD reuse blocks because both tokens and cache_salt_id match
    auto promptLen2 = llmRequest2->getNumTokens(beamIdx);
    auto numContextBlocks2 = tc::ceilDiv(promptLen2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, promptLen2, numContextBlocks2, *llmRequest2, maxAttentionWindow);
    EXPECT_EQ(llmRequest2->getContextCurrentPosition(), 2 * tokensPerBlock); // Reuse blocks 3,4
    EXPECT_THAT(seq2.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({3, 4, 6}));

    llmRequest2->addNewToken(3, beamIdx);
    llmRequest2->addNewToken(4, beamIdx);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks);

    // Release blocks
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseSequence(seq2.getRequestId());
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), 0);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool);

    ///////////////////////////////////////////////////////////////////////////
    // Test Case 4: Request with same tokens but different cache_salt_id = 67890
    requestId = 3;
    CacheSaltIDType cacheSaltId2{67890};
    auto llmRequest3 = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        cacheSaltId2); // Different cache_salt_id = 67890

    GenerationRequest seq3{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should NOT reuse blocks from any previous request because cache_salt_id is different
    auto promptLen3 = llmRequest3->getNumTokens(beamIdx);
    auto numContextBlocks3 = tc::ceilDiv(promptLen3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, promptLen3, numContextBlocks3, *llmRequest3, maxAttentionWindow);
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
        std::nullopt, std::nullopt, std::nullopt, false, false, false, std::nullopt, std::nullopt, false, std::nullopt,
        false, std::nullopt, false, std::nullopt, 0.5, std::nullopt, std::nullopt, std::nullopt,
        LlmRequestType::LLMREQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, numReturnSequences, std::nullopt,
        std::nullopt, false, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::nullopt); // No cache_salt_id

    GenerationRequest seq4{requestId, inputLength, beamWidth, blockManager.getWindowSizesMetadata()};

    // Should reuse blocks from request0 (blocks 0,1) because both have no cache_salt_id
    auto promptLen4 = llmRequest4->getNumTokens(beamIdx);
    auto numContextBlocks4 = tc::ceilDiv(promptLen4, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, promptLen4, numContextBlocks4, *llmRequest4, maxAttentionWindow);
    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 2 * tokensPerBlock); // Reuse blocks 0,1
    EXPECT_THAT(seq4.getCacheBlockIds(maxAttentionWindow).at(beamIdx), ::testing::ElementsAreArray({0, 1, 10}));

    llmRequest4->addNewToken(7, beamIdx);
    numTokens = llmRequest4->getNumTokens(beamIdx);
    numBlocks = tc::ceilDiv(numTokens, tokensPerBlock);
    EXPECT_EQ(blockManager.getNumAllocatedBlocks(), numBlocks * 2);
    EXPECT_EQ(blockManager.getNumFreeBlocks(), blocksInPrimaryPool - numBlocks * 2);

    // Clean up
    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseBlocks(seq4, llmRequest4);
    blockManager.releaseSequence(seq3.getRequestId());
    blockManager.releaseSequence(seq4.getRequestId());
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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens{0};
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

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

    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId, llmRequest0));

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
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto constexpr beamWidth = 1;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    BlockManager blockManager(std::vector(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, stream, maxAttentionWindow, beamWidth,
        std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF, 0,
        onboardBlocks);
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
    blockManager.holdSequence(seq0.getRequestId());
    blockManager.addSequence(seq0, llmRequest0->getNumTokens(0), numContextBlocks0, *llmRequest0, maxAttentionWindow);

    // Add another sequence with different tokens, at a low priority
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    GenerationRequest seq1{1, inputLength1, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks1 = tc::ceilDiv(inputLength1, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq1.getRequestId());
    blockManager.addSequence(seq1, llmRequest1->getNumTokens(0), numContextBlocks1, *llmRequest1, maxAttentionWindow);

    // Release both sequences
    blockManager.releaseBlocks(seq0, llmRequest0);
    blockManager.releaseBlocks(seq1, llmRequest1);
    blockManager.releaseSequence(seq0.getRequestId());
    blockManager.releaseSequence(seq1.getRequestId());

    // Add and then release another sequence
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    llmRequest2->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 20)}, 20));
    GenerationRequest seq2{2, inputLength2, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks2 = tc::ceilDiv(inputLength2, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq2.getRequestId());
    blockManager.addSequence(seq2, llmRequest2->getNumTokens(0), numContextBlocks2, *llmRequest2, maxAttentionWindow);
    blockManager.releaseBlocks(seq2, llmRequest2);
    blockManager.releaseSequence(seq2.getRequestId());

    // Check that request 1 blocks were overwritten
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{8, 9, 10, 11, 12, 13, 14, 15});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    GenerationRequest seq3{3, inputLength3, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks3 = tc::ceilDiv(inputLength3, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq3.getRequestId());
    blockManager.addSequence(seq3, llmRequest3->getNumTokens(0), numContextBlocks3, *llmRequest3, maxAttentionWindow);

    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 4);

    blockManager.releaseBlocks(seq3, llmRequest3);
    blockManager.releaseSequence(seq3.getRequestId());
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 4);

    // Check that request 0 blocks weren't overwritten
    auto inputTokens4 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto const inputLength4 = static_cast<SizeType32>(inputTokens4->size());
    auto llmRequest4 = std::make_shared<LlmRequest>(4, maxNewTokens, inputTokens4, samplingConfig, isStreaming);
    GenerationRequest seq4{4, inputLength3, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks4 = tc::ceilDiv(inputLength4, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq4.getRequestId());
    blockManager.addSequence(seq4, llmRequest4->getNumTokens(0), numContextBlocks4, *llmRequest4, maxAttentionWindow);

    EXPECT_EQ(llmRequest4->getContextCurrentPosition(), 4);

    // Check that request 2 block 0 has been evicted
    auto inputTokens5 = std::make_shared<VecTokens>(VecTokens{16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength5 = static_cast<SizeType32>(inputTokens5->size());
    auto llmRequest5 = std::make_shared<LlmRequest>(5, maxNewTokens, inputTokens5, samplingConfig, isStreaming);
    GenerationRequest seq5{5, inputLength5, beamWidth, blockManager.getWindowSizesMetadata()};
    auto numContextBlocks5 = tc::ceilDiv(inputLength5, blockManager.getTokensPerBlock());
    blockManager.holdSequence(seq5.getRequestId());
    blockManager.addSequence(seq5, llmRequest5->getNumTokens(0), numContextBlocks5, *llmRequest5, maxAttentionWindow);

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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 8);

    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 90)}, 90));
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);

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
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);

    // one block left.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 1);

    // add another token, which occupies another new block
    kvCacheManager.addToken(1);
    llmRequest1->addNewToken(0, 0); // block 7 contains [0]

    // no block available.
    EXPECT_EQ(kvCacheManager.getNumFreeBlocks(), 0);

    // remove both sequences, blocks get stored
    // leaf block 3 (priority 90), context blocks 2, 1, 0 (priority 5)
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    // leaf block 7 (priority 5), context blocks 6, 5, 4 (priority 90)
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
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    // leaf block 2 (priority 35), context blocks 3, 7 (priority 35)
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // Uses 3 blocks 0, 1, 2 which contain [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);

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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    llmRequest0->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 80, 10ms)}, 80));
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    llmRequest1->setKvCacheRetentionConfig(
        KvCacheRetentionConfig({KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 80));
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Manually trigger a refresh.
    kvCacheManager.refreshBlocks();

    // Clear out some of the blocks.
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxNumSequences = 8;
    auto constexpr blocksInPrimaryPool = 9;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
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
        kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
        kvCacheManager.storeContextBlocks(*llmRequest0);
        // Occupy a new block, block 3, adding 3 tokens to block 3.
        // [1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 0, 0]
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(0);
            llmRequest0->addNewToken(0, 0);
        }
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
        kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
        kvCacheManager.storeContextBlocks(*llmRequest1);
        // Occupy a new block, block 3, adding 3 tokens to block 3.
        // [1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [0, 0, 0]
        for (int i = 0; i < 3; i++)
        {
            kvCacheManager.addToken(1);
            llmRequest1->addNewToken(0, 0);
        }
        (void) kvCacheManager.removeSequence(1, llmRequest1);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    kvCacheManager.refreshBlocks();

    // 8 tokens, occupying blocks 8, 6
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 0, 0, 0, 0, 0, 0, 0});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // 12 tokens, reusing block 4, 5. Block 6 is overwritten so no reuse.
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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr blocksInPrimaryPool = 4;
    auto constexpr blocksInSecondaryPool = 4;
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    // 12 tokens, get block 0, 1, 2
    // [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    (void) kvCacheManager.removeSequence(0, llmRequest0);
    // store blocks 0, 1, 2 for reuse ([0,1,2,3], [4,5,6,7], [8,9,10])

    // Offload the last two blocks of llmRequest0 to secondary memory
    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);

    // Uses blocks 3, 4, 5, block 2 and 1 to be offloaded to secondary
    // Block 4 is now in primary (replacing 2)
    // Block 5 is now in primary (replacing 1)
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);
    // store blocks 3, 4, 5 for reuse ([1,1,2,3], [4,5,6,7], [8,9,10])

    // Match the middle block of request 0
    // Uses block 6, block 0 is offloaded to secondary
    // Block 6 copies content from block 0 to itselg.
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    // reuse block 0
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);
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
    (void) kvCacheManager.removeSequence(2, llmRequest2);

    // 10 tokens, reusing the block 0 only because when we want to acquire
    // the second block, contents of block 3 will be offloaded to block 1.
    auto inputTokens3 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7, 0, 0});
    auto const inputLength3 = static_cast<SizeType32>(inputTokens3->size());
    auto llmRequest3 = std::make_shared<LlmRequest>(3, maxNewTokens, inputTokens3, samplingConfig, isStreaming);
    kvCacheManager.addSequence(3, inputLength3, beamWidth, llmRequest3);
    // Check out FIXME note above. If addressed, this should be 9.
    EXPECT_EQ(llmRequest3->getContextCurrentPosition(), 4);
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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;
    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0 = std::make_shared<LlmRequest>(0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    kvCacheManager.addSequence(0, inputLength0, beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    llmRequest0->addNewToken(0, 0);
    kvCacheManager.addToken(0);

    // The second block allocated should be first in line for eviction.
    (void) kvCacheManager.removeSequence(0, llmRequest0);

    auto inputTokens1 = std::make_shared<VecTokens>(VecTokens{1, 1, 2, 3});
    auto const inputLength1 = static_cast<SizeType32>(inputTokens1->size());
    auto llmRequest1 = std::make_shared<LlmRequest>(1, maxNewTokens, inputTokens1, samplingConfig, isStreaming);
    kvCacheManager.addSequence(1, inputLength1, beamWidth, llmRequest1);

    GenerationRequest const& seq1 = kvCacheManager.getSequence(1);
    EXPECT_EQ(llmRequest1->getContextCurrentPosition(), 0);
    // Block 1 should NOT be reused. It was not freed even if partial.
    EXPECT_THAT(seq1.getCacheBlockIds(maxAttentionWindow).at(0), ::testing::ElementsAreArray({2}));

    // Allocate the remaining 3 blocks in primary
    auto inputTokens2 = std::make_shared<VecTokens>(VecTokens{2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto const inputLength2 = static_cast<SizeType32>(inputTokens2->size());
    auto llmRequest2 = std::make_shared<LlmRequest>(2, maxNewTokens, inputTokens2, samplingConfig, isStreaming);
    kvCacheManager.addSequence(2, inputLength2, beamWidth, llmRequest2);

    (void) kvCacheManager.removeSequence(1, llmRequest1);
    (void) kvCacheManager.removeSequence(2, llmRequest2);

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
    auto constexpr onboardBlocks = true;
    auto const stream = std::make_shared<tr::CudaStream>();

    auto constexpr beamWidth = 1;
    auto constexpr beamIdx = 0;
    SizeType32 constexpr maxNewTokens = 8;
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};

    auto const maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    // Create sequence with one block worth of context tokens
    int requestId0 = 0;
    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3});
    auto const inputLength0 = static_cast<SizeType32>(inputTokens0->size());
    auto llmRequest0
        = std::make_shared<LlmRequest>(requestId0, maxNewTokens, inputTokens0, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId0, inputLength0, beamWidth, llmRequest0);
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
    kvCacheManager.addSequence(requestId1, inputLength1, beamWidth, llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId1);

    // Verify that all primary blocks are in use
    EXPECT_EQ(blockManager.getNumFreeBlocks(), 0);

    // Free first sequence
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
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    auto const granularity = tensorrt_llm::common::getAllocationGranularity();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            std::nullopt, nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse,
            onboardBlocks);

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
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
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
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth * 2);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
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
    auto constexpr maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto constexpr maxAttentionWindow = maxSequenceLength;
    auto constexpr inputLength = maxSequenceLength - tokensPerBlock - 1;
    auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
    auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

    auto constexpr totalNumBlocks = maxNumSequences * numBlocksPerSeq;
    auto constexpr blocksInSecondaryPool = 0;

    auto constexpr enableBlockReuse = false;
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(std::vector<KVCacheManager::SizeType32>(numLayers, numHeads), sizePerHead, tokensPerBlock,
            blocksPerWindow, maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow},
            std::nullopt, nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse,
            onboardBlocks);
    kvCacheManager.allocatePools(false);

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
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
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
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
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
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth);
    EXPECT_NO_THROW(kvCacheManager.addToken(requestId));
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks - numSharedBlocks - maxBeamWidth * 2);
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
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
    auto constexpr onboardBlocks = true;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
        nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
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

    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
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
    auto constexpr onboardBlocks = true;
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
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, dtype, 0, stream,
        maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1024));
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
    kvCacheManager.addSequence(1, inputTokens1->size(), beamWidth, llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

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

    (void) kvCacheManager.removeSequence(2, llmRequest2);
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
    kvCacheManager.addSequence(4, inputTokens4->size(), beamWidth, llmRequest4);

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
    auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);

    auto constexpr blocksInPrimaryPool = 16;
    auto constexpr blocksInSecondaryPool = 16;

    auto constexpr enableBlockReuse = true;
    auto constexpr onboardBlocks = true;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
        nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
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
    int inputLength = 16;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    auto constexpr beamIdx = 0;

    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3}));

    // add tokens, making the window slide
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1017, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1018, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1019, beamIdx);
    kvCacheManager.addToken(requestId);
    auto numTokens = llmRequest->getNumTokens(beamIdx);
    auto numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPool;
    EXPECT_THAT(seq0.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 3, 4}));

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPool;
    EXPECT_EQ(numAllocatedPrimaryBlocks, 0);
    // store blocks 0, 1, 2, 3, 4  for reuse ([1000,1001,1002,1003], [1004,1005,1006,1007], [1008,1009,1010,1011],
    // [1012,1013,1014,1015], [1016,1017])

    ///////////////////////////////////////////////////////////////////////////
    // add a short request and then remove it
    // reuse first 2 blocks {0, 1(p)} in previous request, copying block 1 to a new block 5
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 6);
    EXPECT_THAT(seq1.getCacheBlockIds(onlyWindowSize).at(beamIdx),
        ::testing::ElementsAreArray(
            {0, 5})); // Can't use 5 since it's used to onboard block, so 6 is the next free block.

    llmRequest->addNewToken(1007, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    numTokens = llmRequest->getNumTokens(beamIdx);
    EXPECT_THAT(seq1.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 5, 6}));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a medium request and then remove it
    // reuse first 3 blocks {0, 1, 2(p)} in first request, copying block 2 to a new block 7
    requestId = 2;
    inputLength = 10;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq2 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 9);
    EXPECT_THAT(seq2.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 7}));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));

    ///////////////////////////////////////////////////////////////////////////
    // add a longer request within attention window and try to reuse
    // reuse blocks {0, 1, 2, 3(p)}, copying block 3 to a new block 8
    // then upon reaching attention window, get new block 9
    requestId = 3;
    inputLength = 15;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq3 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 14);
    EXPECT_THAT(seq3.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 8}));

    // add new tokens to allocate another block, but not enough to detach block
    llmRequest->addNewToken(1015, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1016, beamIdx);
    kvCacheManager.addToken(requestId);
    EXPECT_THAT(seq3.getCacheBlockIds(onlyWindowSize).at(beamIdx), ::testing::ElementsAreArray({0, 1, 2, 8, 9}));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
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
    auto constexpr onboardBlocks = true;

    auto const blocksPerWindow
        = BlocksPerWindow{{attentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, maxAttentionWindowVec, std::nullopt, dtype, sinkTokenLength, stream, maxSequenceLength,
        enableBlockReuse, onboardBlocks);
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

    kvCacheManager.addSequence(/*requestId=*/0, inputLength, beamWidth, llmRequest0);
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
    kvCacheManager.addSequence(/*requestId=*/1, inputLength, beamWidth, llmRequest1);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(/*requestId=*/1);

    auto const onlyWindowSize = theOnlyWindowSize(kvCacheManager);
    EXPECT_FALSE(blockManager.isSequenceValidForStoreForReuse(seq0.getRequestId(), onlyWindowSize));
    EXPECT_TRUE(blockManager.isSequenceValidForStoreForReuse(seq1.getRequestId(), onlyWindowSize));

    EXPECT_NO_THROW(kvCacheManager.removeSequence(seq0.getRequestId(), llmRequest0));
    EXPECT_NO_THROW(kvCacheManager.removeSequence(seq1.getRequestId(), llmRequest1));
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
    auto constexpr onboardBlocks = true;

    auto const blocksPerWindow
        = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}},
            {minAttentionWindow, {blocksInPrimaryPoolPerWindow, blocksInSecondaryPoolPerWindow}}};
    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        maxBeamWidth, maxAttentionWindowVec, std::nullopt, dtype, sinkTokenLength, stream, maxSequenceLength,
        enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(false);

    auto const& blockManager = kvCacheManager.getBlockManager();

    auto const allBlocksInPrimaryPools = blockManager.getNumPrimaryBlocks();
    EXPECT_THAT(allBlocksInPrimaryPools, blocksInPrimaryPoolPerWindow * numWindows);

    ASSERT_EQ(blockManager.isVariableWindow(), true);
    ASSERT_EQ(blockManager.isVariableGQA(), false);

    SizeType32 constexpr maxNewTokens = 40;

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
    int inputLength = 8;
    auto inputTokens = std::make_shared<VecTokens>(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    auto llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);

    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq0 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 0);
    assertBlocks(seq0, {0, 1}, {0, 1});

    // add tokens, making the minimum attention window slide (not reaching the max attention window)
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1009, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1010, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1011, beamIdx);
    kvCacheManager.addToken(requestId);
    assertBlocks(seq0, {0, 1, 2}, {0, 1, 2});
    auto numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPoolPerWindow * numWindows;

    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
    numAllocatedPrimaryBlocks = blockManager.getNumAllocatedBlocks() - blocksInSecondaryPoolPerWindow * numWindows;
    EXPECT_EQ(numAllocatedPrimaryBlocks, 0);
    // For both windows, store blocks 0, 1, 2  for reuse ([1000,1001,1002,1003], [1004,1005,1006,1007],
    // [1008,1009,1010,1011])

    ///////////////////////////////////////////////////////////////////////////
    // add a short request within both attention windows and try to reuse
    // reuse blocks {0, 1(p)} for both windows, copying block 1 to a new block 4 since it's not a leaf block and is
    // partially used. upon reached attention window, get new block 5
    requestId = 1;
    inputLength = 7;
    inputTokens->resize(inputLength);
    std::iota(inputTokens->begin(), inputTokens->end(), firstToken);
    llmRequest = std::make_shared<LlmRequest>(requestId, maxNewTokens, inputTokens, samplingConfig, isStreaming);
    kvCacheManager.addSequence(requestId, inputLength, beamWidth, llmRequest);
    GenerationRequest const& seq1 = kvCacheManager.getSequence(requestId);
    EXPECT_EQ(llmRequest->getContextCurrentPosition(), 6);
    assertBlocks(seq1, {0, 3}, {0, 3});

    // add new tokens to allocate another block, but not enough to detach block
    llmRequest->addNewToken(1008, beamIdx);
    kvCacheManager.addToken(requestId);
    llmRequest->addNewToken(1009, beamIdx);
    kvCacheManager.addToken(requestId);
    assertBlocks(seq1, {0, 3, 4}, {0, 3, 4});
    EXPECT_NO_THROW(kvCacheManager.removeSequence(requestId, llmRequest));
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

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, dtype, 0, stream,
        maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1));
    kvCacheManager.allocatePools(false);

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
    auto constexpr onboardBlocks = true;
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
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, dtype, 0, stream,
        maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1024));
    kvCacheManager.allocatePools(false);

    auto inputTokens0 = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    auto llmRequest0 = std::make_shared<LlmRequest>(0, 0, inputTokens0, samplingConfig, true);
    llmRequest0->setKvCacheRetentionConfig(tle::KvCacheRetentionConfig(
        std::vector{tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, std::nullopt, 50)}, 35));
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
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
    kvCacheManager.addSequence(1, inputTokens1->size(), beamWidth, llmRequest1);
    kvCacheManager.storeContextBlocks(*llmRequest1);
    (void) kvCacheManager.removeSequence(1, llmRequest1);

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
    auto constexpr onboardBlocks = true;
    auto constexpr beamWidth = 1;
    auto const maxAttentionWindow = tokensPerBlock * blocksInPrimaryPool;

    BlocksPerWindow const blocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager(numLayers, numKvHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxAttentionWindow, true, onboardBlocks);
    kvCacheManager.allocatePools(false);

    LlmRequest::RequestIdType requestId{0};
    auto inputTokens = std::make_shared<VecTokens>(VecTokens{0, 1, 2, 3, 4, 5, 6, 7});
    tr::SamplingConfig const samplingConfig{beamWidth};
    bool constexpr isStreaming{false};
    auto llmRequest = std::make_shared<LlmRequest>(requestId, 0, inputTokens, samplingConfig, isStreaming);

    kvCacheManager.addSequence(requestId, static_cast<SizeType32>(inputTokens->size()), beamWidth, llmRequest);
    auto const totalBlocks = kvCacheManager.getMaxNumBlocks();
    auto const freeAfterAlloc = kvCacheManager.getNumFreeBlocks();
    EXPECT_LT(freeAfterAlloc, totalBlocks);

    kvCacheManager.pinBlocks(requestId);
    auto lastBlockIdOpt = kvCacheManager.getLastBlockId(requestId);
    ASSERT_TRUE(lastBlockIdOpt.has_value());
    (void) kvCacheManager.removeSequence(requestId, llmRequest);
    auto const freeAfterRemovePinned = kvCacheManager.getNumFreeBlocks();
    EXPECT_LT(freeAfterRemovePinned, totalBlocks);

    kvCacheManager.unpinBlocksById(lastBlockIdOpt.value());
    auto const freeAfterUnpin = kvCacheManager.getNumFreeBlocks();
    EXPECT_EQ(freeAfterUnpin, totalBlocks);
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

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    auto const maxAttentionWindow = maxSequenceLength;

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {blocksInPrimaryPool, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManagerTest(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, dtype, 0,
        stream, maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt);

    EXPECT_EQ(getEvents(kvCacheManagerTest).size(), 0);

    KVCacheManager kvCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
        0, stream, maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt,
        std::make_unique<tlk::KVCacheEventManager>(1024));

    kvCacheManager.allocatePools(false);
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
    auto constexpr onboardBlocks = true;
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
        beamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow, slidingWindow}, std::nullopt, dtype, 0,
        stream, maxSequenceLength, true, onboardBlocks, CacheType::kSELF, std::nullopt,
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
    kvCacheManager.addSequence(0, inputTokens0->size(), beamWidth, llmRequest0);
    kvCacheManager.storeContextBlocks(*llmRequest0);

    events = getEvents(kvCacheManager);

    // Expecting only 1 event, storeContextBlock is not called for sliding window.
    EXPECT_EQ(events.size(), 1);

    EXPECT_EQ(events.back().windowSize, maxAttentionWindow);
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
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    auto const maxSequenceLength = tokensPerBlock * maxBlocksPerSeq;
    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
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
    EXPECT_NO_THROW((void) kvCacheManager.removeSequence(requestId));
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
    auto constexpr onboardBlocks = true;
    auto const homogeneousLayers = GetParam();
    auto const expectedNumPools = homogeneousLayers ? 1 : static_cast<SizeType32>(expectedHeadsPerPool.size());

    auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

    KVCacheManager kvCacheManager = homogeneousLayers
        ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
            maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks)
        : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
            std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt, nvinfer1::DataType::kHALF,
            sinkTokenLength, stream, maxSequenceLength, enableBlockReuse, onboardBlocks);
    kvCacheManager.allocatePools(false);

    EXPECT_EQ(kvCacheManager.getOffsetTableDimensions().maxBlocksPerSeq, maxBlocksPerSeq);
    EXPECT_EQ(kvCacheManager.getTokensPerBlock(), tokensPerBlock);
    EXPECT_EQ(kvCacheManager.getMaxNumBlocks(), totalNumBlocks);

    auto const& blockManager = kvCacheManager.getBlockManager();
    EXPECT_EQ(blockManager.getNumFreeBlocks(), totalNumBlocks);

    for (auto requestId = 0; requestId < maxNumSequences; ++requestId)
    {
        EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth));
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
            auto constexpr onboardBlocks = true;
            auto constexpr maxSequenceLength = 256;
            auto constexpr maxBlocksPerSeq = tc::ceilDiv(maxSequenceLength, tokensPerBlock);
            auto constexpr totalNumBlocks = maxNumSequences * maxBlocksPerSeq;
            auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

            KVCacheManager kvCacheManager = homogeneousLayers
                ? KVCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
                    maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
                    nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, kv_cache_block_reuse,
                    onboardBlocks)
                : KVCacheManager(numHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
                    maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
                    nvinfer1::DataType::kHALF, sinkTokenLength, stream, maxSequenceLength, kv_cache_block_reuse,
                    onboardBlocks);
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

            EXPECT_NO_THROW(kvCacheManager.addSequence(requestId, inputLength, maxBeamWidth, llmRequest));
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
            for (int i = draftLen; i < maxNewTokens && (inputLength + i) < maxAttentionWindow; i += (draftLen + 1))
            {
                for (int beam = 0; beam < maxBeamWidth; beam++)
                {
                    llmRequest->addNewToken(1, beam);
                }

                neededBlocksOneStep = kvCacheManager.getNeededBlocksOneStep(*llmRequest, false, onlyWindowSize);
                currentNumAllocTotalBlocks = kvCacheManager.getNumAllocTotalBlocks();

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

            // After adding tokens, initial remainingBlocksToCompletion should match current state + new
            // remainingBlocksToCompletion
            EXPECT_EQ(remainingBlocksToCompletion,
                kvCacheManager.getNumAllocTotalBlocks()
                    + kvCacheManager.getRemainingBlocksToCompletion(*llmRequest, onlyWindowSize));
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
    auto const maxInputLength = kvCacheInstantiationParameters.maxNumTokens - 1;
    auto const temporaryKvCacheInputs
        = TempAttentionWindowInputs{true, maxInputLength, kvCacheInstantiationParameters.maxNumTokens};

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
            std::vector<SizeType32>{kvCacheInstantiationParameters.maxAttentionWindow}, temporaryKvCacheInputs,
            kvCacheInstantiationParameters.dtype, kvCacheInstantiationParameters.sinkTokenLength, stream,
            maxSequenceLength, kvCacheInstantiationParameters.kvCacheBlockReuse, true, CacheType::kSELF);
    }
    if (std::holds_alternative<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer))
    {
        auto const numHeadsPerLayer
            = std::get<std::vector<SizeType32>>(kvCacheInstantiationParameters.numHeadsPerLayer);
        return std::make_shared<KVCacheManager>(numHeadsPerLayer, kvCacheInstantiationParameters.sizePerHead,
            kvCacheInstantiationParameters.tokensPerBlock, kvCacheInstantiationParameters.blocksPerWindow,
            numBlocksInPrimaryPool, kvCacheInstantiationParameters.maxBeamWidth,
            std::vector<SizeType32>{kvCacheInstantiationParameters.maxAttentionWindow}, temporaryKvCacheInputs,
            kvCacheInstantiationParameters.dtype, kvCacheInstantiationParameters.sinkTokenLength, stream,
            maxSequenceLength, kvCacheInstantiationParameters.kvCacheBlockReuse, true, CacheType::kSELF);
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
        kvCacheManager->addSequence(
            requestId, params.promptLength, params.kvCacheManagerInstantiationParameters.maxBeamWidth, llmRequest);
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
