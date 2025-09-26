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

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class BlockIterator;

class BlockRange
{
public:
    // C++20 std::default_sentinel_t equivalent
    struct Sentinel
    {
    };

    static BlockRange fromAllBlockIds(BaseKVCacheManager const& cacheManager, LlmRequest::RequestIdType requestId,
        SizeType32 beam = kFIRST_AND_ONLY_BEAM)
    {
        assert(kFIRST_AND_ONLY_BEAM == beam);
        auto const windowSize = firstWindowSize(cacheManager);
        auto const blockIds = cacheManager.getSequence(requestId).getCacheBlockIds(windowSize).at(kFIRST_AND_ONLY_BEAM);
        return BlockRange(cacheManager, blockIds, requestId);
    }

    static BlockRange fromReuseTree(
        BaseKVCacheManager& cacheManager, BlockKey const& lastBlockKey, int32_t indexFromEnd)
    {
        auto const windowSize = firstWindowSize(cacheManager);
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
        return BlockRange(cacheManager, blockIds, 0);
    }

    BlockRange(runtime::ITensor::SharedPtr pool, std::vector<SizeType32> const& blockIds) // Only used in tests
        : mManager{nullptr}
        , mPool{std::move(pool)}
        , mWindowSize{0}
        , mRequestId{0}
        , mBlockIds{blockIds}
    {
        TLLM_CHECK(mPool);
    }

    [[nodiscard]] BlockIterator begin() const;

    [[nodiscard]] Sentinel end() const
    {
        return {};
    }

    [[nodiscard]] size_t size() const
    {
        return mBlockIds.size();
    }

    [[nodiscard]] std::vector<SizeType32> const& getBlockIds() const
    {
        return mBlockIds;
    }

    void setBlockIds(std::vector<SizeType32> blockIds)
    {
        mBlockIds = std::move(blockIds);
    }

    void updatePoolIdx(SizeType32 poolIdx)
    {
        TLLM_CHECK(mManager);
        mPool = mManager->getBlockManager().getPrimaryPool(poolIdx);
        auto const newWindowSize = mManager->getBlockManager().getPoolWindowSize(poolIdx);
        if (newWindowSize != mWindowSize)
        {
            mWindowSize = newWindowSize;
            mBlockIds = mManager->getSequence(mRequestId).getCacheBlockIds(mWindowSize).at(kFIRST_AND_ONLY_BEAM);
        }
    }

    friend class BlockIterator;

private:
    BlockRange(
        BaseKVCacheManager const& cacheManager, std::vector<SizeType32> blockIds, LlmRequest::RequestIdType requestId)
        : mManager(&cacheManager)
        , mPool(cacheManager.getBlockManager().getPrimaryPool(kFIRST_POOL_INDEX))
        , mWindowSize(firstWindowSize(cacheManager))
        , mRequestId(requestId)
        , mBlockIds(std::move(blockIds))
    {
    }

    static SizeType32 firstWindowSize(BaseKVCacheManager const& cacheManager)
    {
        constexpr SizeType32 FIRST_POOL_IDX = 0;
        return cacheManager.getBlockManager().getPoolWindowSize(FIRST_POOL_IDX);
    }

private:
    BaseKVCacheManager const* mManager;
    runtime::ITensor::SharedPtr mPool;
    SizeType32 mWindowSize;
    const LlmRequest::RequestIdType mRequestId;
    std::vector<SizeType32> mBlockIds;

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

    BlockIterator(BlockRange const* range, size_t idx)
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

    [[nodiscard]] bool operator==(BlockRange::Sentinel other) const
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
            mCurrent = runtime::ITensor::slice(mRange->mPool, mRange->mBlockIds.at(mIdx), 1);
        }
    }

    BlockRange const* mRange;
    runtime::ITensor::SharedPtr mCurrent;
    size_t mIdx;
};

inline BlockIterator BlockRange::begin() const
{
    return {this, 0};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
