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

#include "tensorrt_llm/batch_manager/kvCacheUtils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;

// ---------------------------------------
//            BlockIteratorTest
// ---------------------------------------

class BlockIteratorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(BlockIteratorTest, BasicTest)
{
    using DataType = int32_t;
    auto constexpr mNumPrimaryBlocks = 10;
    auto constexpr mNumLayers = 5;
    auto constexpr mBlockSize = 32;
    auto const cacheShape = tr::ITensor::makeShape({mNumPrimaryBlocks, mNumLayers, 2, mBlockSize});
    constexpr nvinfer1::DataType dtype{tr::TRTDataType<DataType>::value};
    tr::ITensor::SharedPtr pool = tr::BufferManager::cpu(cacheShape, dtype);
    std::vector<SizeType32> blockIds(mNumPrimaryBlocks);
    std::iota(blockIds.begin(), blockIds.end(), 0);
    for (auto idx : blockIds)
    {
        auto blockTensor = tr::ITensor::slice(pool, blockIds.at(idx), 1);
        std::fill_n(tr::bufferCast<DataType>(*blockTensor), blockTensor->getSize(), idx);
    }
    auto range = BlockRangeForWindow(nullptr, 0, std::move(blockIds), std::move(pool));
    auto begin = range.begin();
    auto end = range.end();
    auto allEqualTo = [](tr::ITensor const& tensor, auto x) -> bool
    {
        const auto* begin = tr::bufferCast<decltype(x)>(tensor);
        const auto* end = begin + tensor.getSize();
        return std::all_of(begin, end, [x](auto n) { return n == x; });
    };
    DataType cnt{0};
    for (auto const& tensor : range)
    {
        EXPECT_TRUE(allEqualTo(tensor, cnt++));
    }
}

TEST_F(BlockIteratorTest, CacheManagerTest)
{
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    auto constexpr numLayers = 12;
    auto constexpr numKvHeads = 6;
    auto constexpr sizePerHead = 16;
    auto constexpr tokensPerBlock = 4;
    auto constexpr maxBlocksPerSeq = 4;
    auto constexpr blocksInPrimaryPool = 8;
    auto constexpr blocksInSecondaryPool = 0;
    auto constexpr maxNumSequences = 8;
    auto constexpr maxAttentionWindow = tokensPerBlock * maxBlocksPerSeq;

    auto const stream = std::make_shared<tr::CudaStream>();
    auto constexpr onboardBlocks = true;

    // TODO: Support and add coverage for beamWidth > 1
    auto constexpr beamWidth = 1;
    auto constexpr numBlocksPerBeam = blocksInPrimaryPool / beamWidth;
    auto constexpr maxSequenceLength = tokensPerBlock * numBlocksPerBeam;
    auto const maxAttentionWindowVec = std::vector<BlockManager::SizeType32>{maxAttentionWindow};

    using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
    const BlocksPerWindow blocksPerWindow
        = {{maxAttentionWindow, std::make_tuple(blocksInPrimaryPool, blocksInSecondaryPool)}};

    BlockManager blockManager(std::vector<BlockManager::SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock,
        blocksPerWindow, maxNumSequences, stream, maxSequenceLength, beamWidth, maxAttentionWindowVec, std::nullopt,
        dataType, 0, onboardBlocks);
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

    auto constexpr beamIdx = 0;
    auto promptLen0 = llmRequest0->getNumTokens(beamIdx);
    auto numContextBlocks0 = tc::ceilDiv(promptLen0, blockManager.getTokensPerBlock());
    blockManager.addSequence(seq0, promptLen0, numContextBlocks0, *llmRequest0, maxAttentionWindow);

    auto const blockIds = seq0.getCacheBlockIds(maxAttentionWindow).at(beamIdx);
    EXPECT_THAT(blockIds, ::testing::ElementsAreArray({0, 1, 2}));

    auto const pool = blockManager.getPrimaryPool(0);
    TLLM_CHECK(pool);
    auto blockIdsVec = std::vector<SizeType32>(blockIds.begin(), blockIds.end());
    auto poolCopy = pool;
    auto range = BlockRangeForWindow(nullptr, maxAttentionWindow, std::move(blockIdsVec), std::move(poolCopy));
    size_t cnt{0};
    for (auto iter = range.begin(); iter != range.end(); ++iter, ++cnt)
    {
        EXPECT_EQ(iter->getSize(), numLayers * numKvHeads * sizePerHead * tokensPerBlock * 2);
    }
    EXPECT_EQ(cnt, blockIds.size());
}
