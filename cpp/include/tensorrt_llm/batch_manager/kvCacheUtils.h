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

    BlockRange(BaseKVCacheManager const& cacheManager, LlmRequest::RequestIdType requestId, SizeType32 beam,
        SizeType32 poolIdx = 0)
        : mManager(&cacheManager)
        , mPool(cacheManager.getBlockManager().getPrimaryPool(poolIdx))
        , mBlockIds(cacheManager.getSequence(requestId).getCacheBlockIds().at(beam))
    {
    }

    BlockRange(BaseKVCacheManager const& cacheManager, std::vector<SizeType32> blockIds, SizeType32 poolIdx = 0)
        : mManager(&cacheManager)
        , mPool(cacheManager.getBlockManager().getPrimaryPool(poolIdx))
        , mBlockIds(std::move(blockIds))
    {
    }

    BlockRange(runtime::ITensor::SharedPtr pool, std::vector<SizeType32> const& blockIds)
        : mManager{nullptr}
        , mPool{std::move(pool)}
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

    [[nodiscard]] std::vector<size_t> getBlockHashes() const
    {
        TLLM_CHECK(mManager);
        std::vector<size_t> blockHashes;
        blockHashes.reserve(mBlockIds.size());
        auto& blockManager = mManager->getBlockManager();
        for (auto id : mBlockIds)
        {
            blockHashes.emplace_back(blockManager.getBlockById(id)->getHash());
        }
        return blockHashes;
    }

    void updatePoolIdx(SizeType32 poolIdx)
    {
        if (mManager)
        {
            mPool = mManager->getBlockManager().getPrimaryPool(poolIdx);
        }
    }

    friend class BlockIterator;

private:
    BaseKVCacheManager const* mManager;
    runtime::ITensor::SharedPtr mPool;
    std::vector<SizeType32> mBlockIds;
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
