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

class BlockIterator
{
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = runtime::ITensor;
    using pointer = runtime::ITensor::SharedPtr;
    using reference = value_type&;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    BlockIterator(runtime::ITensor::SharedPtr blockPoolPtr, std::vector<SizeType32> blockIds, size_t idx)
        : mPool{std::move(blockPoolPtr)}
        , mBlockIds{std::move(blockIds)}
        , mIdx{idx}
    {
        TLLM_CHECK(mPool);
        TLLM_CHECK(mIdx <= mBlockIds.size());
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
        ret.update();
        mIdx++;
        return ret;
    }

    [[nodiscard]] bool operator==(BlockIterator const& other) const
    {
        return mIdx == other.mIdx && mPool.get() == other.mPool.get();
    }

    [[nodiscard]] bool operator!=(BlockIterator const& other) const
    {
        return !(*this == other);
    }

private:
    void update()
    {
        if (mIdx < mBlockIds.size())
        {
            mCurrent = runtime::ITensor::slice(mPool, mBlockIds.at(mIdx), 1);
        }
    }

    runtime::ITensor::SharedPtr mPool;
    runtime::ITensor::SharedPtr mCurrent;
    const std::vector<SizeType32> mBlockIds;
    size_t mIdx;
};

[[nodiscard]] BlockIterator getBlockBeginIt(
    KVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam);

[[nodiscard]] BlockIterator getBlockEndIt(
    KVCacheManager const& cacheManager, LlmRequest const& request, SizeType32 beam);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
