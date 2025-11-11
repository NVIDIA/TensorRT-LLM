/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class BlockIterator;

class BlockRangeForWindow
{
public:
    BlockRangeForWindow(BaseKVCacheManager const* cacheManager, SizeType32 windowSize, std::vector<SizeType32> blockIds,
        runtime::ITensor::SharedPtr pool)
        : mCacheManager(cacheManager)
        , mWindowSize(windowSize)
        , mBlockIds(std::move(blockIds))
        , mPool(std::move(pool))
    {
    }

    struct Sentinel
    {
    };

    friend class BlockIterator;
    BlockIterator begin() const;

    [[nodiscard]] Sentinel end() const
    {
        return {};
    }

    [[nodiscard]] size_t size() const
    {
        return mBlockIds.size();
    }

private:
    BaseKVCacheManager const* mCacheManager;
    SizeType32 mWindowSize;
    std::vector<SizeType32> mBlockIds;
    runtime::ITensor::SharedPtr mPool;
};

class BlockRange
{
public:
    static BlockRange fromAllBlockIds(BaseKVCacheManager const& cacheManager, LlmRequest::RequestIdType requestId)
    {

        return BlockRange(cacheManager, requestId);
    }

    static BlockRange fromReuseTree(
        BaseKVCacheManager& cacheManager, BlockKey const& lastBlockKey, int32_t indexFromEnd)
    {

        auto poolNum = cacheManager.getNumPools();
        TLLM_CHECK_WITH_INFO(poolNum == 1, "Reuse tree is not supported for multiple pools or variable window size");

        auto windowSize = cacheManager.getBlockManager().getWindowSizesMetadata().begin()->first;
        // Find the last block in the reuse tree for the provided full sequence of block keys
        auto lastBlock = cacheManager.findBlocksInReuseTreeByBlockKey(lastBlockKey, windowSize);
        // TODO: handle the case where the last block is not found
        TLLM_CHECK_WITH_INFO(lastBlock, "Couldn't find the requested block in the reuse tree");
        int32_t const numBlocksToCollect = indexFromEnd + 1;

        std::vector<SizeType32> blockIds;
        blockIds.reserve(numBlocksToCollect);
        for (int32_t i = 0; i < numBlocksToCollect; ++i)
        {
            TLLM_CHECK_WITH_INFO(
                lastBlock->getBlockId() != KVCacheBlock::kCachedBlocksRootId, "last block has no block id");
            blockIds.push_back(lastBlock->getBlockId());
            if (i + 1 < numBlocksToCollect)
            {
                TLLM_CHECK_WITH_INFO(lastBlock->getPrevBlock(), "last block has no prev block");
                lastBlock = lastBlock->getPrevBlock();
            }
        }
        // Reverse to chronological order: oldest to newest
        std::reverse(blockIds.begin(), blockIds.end());
        std::unordered_map<SizeType32, std::vector<SizeType32>> blockIdsPerWindow;
        blockIdsPerWindow[windowSize] = blockIds;
        return BlockRange(cacheManager, blockIdsPerWindow, 0);
    }

    void setBlockIdsForWindow(SizeType32 windowSize, std::vector<SizeType32> blockIds)
    {
        TLLM_CHECK_WITH_INFO(mBlockIdsPerWindow.find(windowSize) != mBlockIdsPerWindow.end(),
            "Window size %d should exists", windowSize);
        mBlockIdsPerWindow[windowSize] = std::move(blockIds);
    }

    void setBlockIdsForAllWindows(std::unordered_map<SizeType32, std::vector<SizeType32>> blockIdsPerWindow)
    {
        for (auto const& [windowSize, blockIds] : blockIdsPerWindow)
        {
            TLLM_CHECK_WITH_INFO(
                mPoolsPerWindow.find(windowSize) != mPoolsPerWindow.end(), "Window size %d should exists", windowSize);
        }
        mBlockIdsPerWindow = std::move(blockIdsPerWindow);
    }

    [[nodiscard]] std::unordered_map<SizeType32, std::vector<size_t>> getBlockHashesPerWindow() const
    {
        TLLM_CHECK(mManager);
        std::unordered_map<SizeType32, std::vector<size_t>> blockHashesPerWindow;
        auto& blockManager = mManager->getBlockManager();
        for (auto const& [windowSize, blockIds] : mBlockIdsPerWindow)
        {
            for (auto const& blockId : blockIds)
            {
                blockHashesPerWindow[windowSize].emplace_back(
                    blockManager.getBlockById(blockId, windowSize)->getHash());
            }
        }
        return blockHashesPerWindow;
    }

    BlockRangeForWindow getBlockRangeForWindow(SizeType32 windowSize) const
    {
        TLLM_CHECK_WITH_INFO(
            mPoolsPerWindow.find(windowSize) != mPoolsPerWindow.end(), "Window size %d not found", windowSize);
        auto pool = mPoolsPerWindow.at(windowSize).front();
        auto blockIds = mBlockIdsPerWindow.at(windowSize);
        return BlockRangeForWindow(mManager, windowSize, std::move(blockIds), std::move(pool));
    }

    std::vector<SizeType32> getWindowSizes() const
    {
        std::vector<SizeType32> windowSizes;
        for (auto const& [windowSize, _] : mPoolsPerWindow)
        {
            windowSizes.push_back(windowSize);
        }
        return windowSizes;
    }

    std::unordered_map<SizeType32, std::vector<SizeType32>> const& getBlockIdsPerWindow() const
    {
        return mBlockIdsPerWindow;
    }

private:
    BlockRange(BaseKVCacheManager const& cacheManager,
        std::unordered_map<SizeType32, std::vector<SizeType32>> blockIdsPerWindow, LlmRequest::RequestIdType requestId)
        : mManager(&cacheManager)
        , mRequestId(requestId)
        , mBlockIdsPerWindow(std::move(blockIdsPerWindow))
    {

        // cacheManager.getBlockManager.getPrimaryPool(0);
        auto poolNum = mManager->getNumPools();
        for (SizeType32 poolIdx = 0; poolIdx < poolNum; ++poolIdx)
        {
            auto windowSize = cacheManager.getBlockManager().getPoolWindowSize(poolIdx);
            mPoolsPerWindow[windowSize].push_back(cacheManager.getBlockManager().getPrimaryPool(poolIdx));
        }
    }

    BlockRange(BaseKVCacheManager const& cacheManager, LlmRequest::RequestIdType requestId)
        : mManager(&cacheManager)
        , mRequestId(requestId)
    {
        auto poolNum = mManager->getNumPools();
        for (SizeType32 poolIdx = 0; poolIdx < poolNum; ++poolIdx)
        {
            auto windowSize = cacheManager.getBlockManager().getPoolWindowSize(poolIdx);
            mPoolsPerWindow[windowSize].push_back(cacheManager.getBlockManager().getPrimaryPool(poolIdx));
            mBlockIdsPerWindow[windowSize]
                = cacheManager.getSequence(mRequestId).getCacheBlockIds(windowSize).at(kFIRST_AND_ONLY_BEAM);
        }
    }

private:
    BaseKVCacheManager const* mManager;
    LlmRequest::RequestIdType const mRequestId;
    std::unordered_map<SizeType32, std::vector<SizeType32>> mBlockIdsPerWindow;
    std::unordered_map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> mPoolsPerWindow;

    static constexpr SizeType32 kFIRST_AND_ONLY_BEAM = 0;
    static constexpr SizeType32 kFIRST_POOL_INDEX = 0;
};

class BlockIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = runtime::ITensor;
    using pointer = runtime::ITensor::SharedPtr;
    using reference = value_type&;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    BlockIterator(BlockRangeForWindow const* range, size_t idx)
        : mRange{range}
        , mIdx{idx}
    {
        TLLM_CHECK(mIdx == 0 || mIdx < mRange->mBlockIds.size());
        update();
    }

    [[nodiscard]] pointer operator->()
    {
        return mCurrent;
    }

    [[nodiscard]] reference operator*()
    {
        return *mCurrent;
    }

    BlockIterator& operator++()
    {
        mIdx++;
        update();
        return *this;
    }

    BlockIterator operator++(int)
    {
        auto ret = *this;
        mIdx++;
        update();
        return ret;
    }

    operator runtime::ITensor::SharedPtr()
    {
        return mCurrent;
    }

    [[nodiscard]] bool operator==(BlockIterator const& other) const
    {
        return mIdx == other.mIdx && mRange == other.mRange;
    }

    [[nodiscard]] bool operator==(BlockRangeForWindow::Sentinel other) const
    {
        return mIdx == mRange->mBlockIds.size();
    }

    template <class T>
    [[nodiscard]] bool operator!=(T const& other) const
    {
        return !(*this == other);
    }

private:
    void update()
    {
        if (mIdx < mRange->mBlockIds.size())
        {
            if (mRange->mCacheManager != nullptr)
            {
                BlockPtr const& block = mRange->mCacheManager->getBlockManager().getBlockById(
                    mRange->mBlockIds.at(mIdx), mRange->mWindowSize);
                TLLM_CHECK_WITH_INFO(block->isPrimary(), "cache transceiver only supports primary blocks");
                auto const blockOffset = block->getMemoryPoolBlockIndex();
                mCurrent = runtime::ITensor::slice(mRange->mPool, blockOffset, 1);
            }
            else
            {
                mCurrent = runtime::ITensor::slice(mRange->mPool, mRange->mBlockIds.at(mIdx), 1);
            }
        }
    }

    BlockRangeForWindow const* mRange;
    runtime::ITensor::SharedPtr mCurrent;
    size_t mIdx;
};

inline BlockIterator BlockRangeForWindow::begin() const
{
    return {this, 0};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
