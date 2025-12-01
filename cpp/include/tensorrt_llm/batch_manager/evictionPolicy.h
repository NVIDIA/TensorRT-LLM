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

#include <chrono>
#include <vector>

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

namespace tensorrt_llm::batch_manager::eviction_policy
{

class BaseEvictionPolicy
{
public:
    virtual ~BaseEvictionPolicy() = default;

    // TODO(TRTLLM-1564): Don't use a separate `initialize` function. Ensure eviction policies can't be in-between a
    // state of construction and initialization.
    virtual void initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
        = 0;

    /// @brief Get a free block from the specified cache level
    /// @returns The pointer to the free block, along with whether it can be offloaded
    virtual std::tuple<BlockPtr, bool> getFreeBlock(SizeType32 cacheLevel) = 0;
    /// @brief Release a block. Prioritize the block for eviction if toFront=true
    virtual void releaseBlock(BlockPtr block) = 0;
    virtual void releaseBlock(BlockPtr block, bool toFront) = 0;
    /// @brief Get the amount of free blocks in the primary memory pool
    virtual SizeType32 getNumFreeBlocks(SizeType32 cacheLevel) = 0;
    /// @brief Claim a free block. Called when the cache manager allocates or reuses a new block
    virtual void claimBlock(BlockPtr block) = 0;
    virtual void claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
        std::optional<std::chrono::milliseconds> durationMs)
        = 0;
    /// @brief Perform any per-iteration bookkeeping
    virtual void refresh() = 0;

    virtual bool verifyQueueIntegrity() = 0;
};

struct ExpiringBlockComparator
{
    bool operator()(BlockPtr const& a, BlockPtr const& b) const
    {
        // If two blocks expire in the same millisecond, their expiration times will be equal. As a fallback, check the
        // raw pointer values.
        return a->getExpirationTime() != b->getExpirationTime() ? a->getExpirationTime() < b->getExpirationTime()
                                                                : a.get() < b.get();
    }
};

class LRUEvictionPolicy : public BaseEvictionPolicy
{
public:
    void initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority) override;
    std::tuple<BlockPtr, bool> getFreeBlock(SizeType32 cacheLevel) override;

    void releaseBlock(BlockPtr block) override;
    void releaseBlock(BlockPtr block, bool toFront) override;

    SizeType32 getNumFreeBlocks(SizeType32 cacheLevel) override;

    void claimBlock(BlockPtr block) override;
    void claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
        std::optional<std::chrono::milliseconds> durationMs) override;

    // Check the expiring blocks heap, and move expired blocks back to the default queue.
    void refresh() override;

    // Making this public and virtual makes it possible to test.
    [[nodiscard]] virtual std::chrono::steady_clock::time_point::duration getTime() const;

    bool verifyQueueIntegrity() override;

private:
    // Queues of available leaf blocks, split by cache level and priority level
    std::vector<std::vector<FreeBlocksQueue>> mFreeQueues;
    // Iterators to block entries in mFreeQueues
    std::vector<std::optional<FreeBlocksQueue::iterator>> mFreeBlockIterators;
    // Amount of free blocks at each cache level
    std::vector<SizeType32> mNumFreeBlocksPerLevel;
    // Secondary offload threshold. Blocks below this priority won't be evicted.
    executor::RetentionPriority mSecondaryOffloadMinPriority;
    // Heap of block times
    std::set<BlockPtr, ExpiringBlockComparator> mExpiringBlockHeap;
};

} // namespace tensorrt_llm::batch_manager::eviction_policy
