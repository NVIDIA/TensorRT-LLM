/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/evictionPolicy.h"

using namespace tensorrt_llm::batch_manager::kv_cache_manager;

// This implements priority-based eviction.
// Blocks are assigned priority levels, with blocks at a lower priority evicted before blocks at a higher priority.
// New priority values always override the previous value.

namespace tensorrt_llm::batch_manager::eviction_policy
{

auto const kMinPriority = executor::KvCacheRetentionConfig::kMinRetentionPriority;
auto const kMaxPriority = executor::KvCacheRetentionConfig::kMaxRetentionPriority;
auto const kNumPriorities = kMaxPriority - kMinPriority + 1;

auto const kDefaultPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority;
executor::RetentionPriority const kDefaultSecondaryOffloadMinPriority = 30;

int const kNumCacheLevels = 2;
int const kPlaceholderLevel = kNumCacheLevels; // placeholder blocks live at level 2

namespace
{
SizeType32 getCacheLevel(BlockPtr const& block)
{
    if (block->isPlaceholder())
    {
        return kPlaceholderLevel;
    }
    return block->isPrimary() ? 0 : 1;
}

constexpr SizeType32 getPriorityIdx(executor::RetentionPriority priority)
{
    return priority - kMinPriority;
}

constexpr auto defaultPriorityIdx = getPriorityIdx(kDefaultPriority);
} // namespace

void LRUEvictionPolicy::initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
{
    SizeType32 startIdx = 0;

    // Create queues for all levels: primary, secondary, and placeholder (initially empty).
    mFreeQueues.resize(kPlaceholderLevel + 1, std::vector<FreeBlocksQueue>(kNumPriorities));
    mFreeBlockIterators.positive.resize(mAllBlocksById.size());
    mNumFreeBlocksPerLevel.resize(kPlaceholderLevel + 1, 0);

    for (SizeType32 cacheLevel = 0; cacheLevel < kNumCacheLevels; cacheLevel++)
    {
        auto& freeQueue = mFreeQueues[cacheLevel][defaultPriorityIdx];

        for (SizeType32 blockId = 0; blockId < sizes[cacheLevel]; blockId++)
        {
            // Initialize all blocks to be the default priority level
            mFreeBlockIterators[startIdx + blockId] = std::make_tuple(
                cacheLevel, defaultPriorityIdx, freeQueue.insert(freeQueue.end(), mAllBlocksById[startIdx + blockId]));
        }

        mNumFreeBlocksPerLevel[cacheLevel] = sizes[cacheLevel];
        startIdx += sizes[cacheLevel];
    }

    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority.value_or(kDefaultSecondaryOffloadMinPriority);
}

void LRUEvictionPolicy::initializePlaceholders(std::vector<BlockPtr>& allPlaceholderBlocksById)
{
    auto const len = static_cast<SizeType32>(allPlaceholderBlocksById.size());

    // Placeholder IDs -2, -3, ... map to indices 2, 3, ... via abs(id).
    // Indices 0 and 1 are unused (0 is invalid, 1 corresponds to kCachedBlocksRootId).
    mFreeBlockIterators.negative.resize(len);

    auto& freeQueue = mFreeQueues[kPlaceholderLevel][defaultPriorityIdx];

    for (auto const& block : allPlaceholderBlocksById)
    {
        if (block)
        {
            mFreeBlockIterators[block->getBlockId()]
                = std::make_tuple(kPlaceholderLevel, defaultPriorityIdx, freeQueue.insert(freeQueue.end(), block));
            mNumFreeBlocksPerLevel[kPlaceholderLevel]++;
        }
    }
}

bool LRUEvictionPolicy::verifyQueueIntegrity()
{
    static char const* const levelToStr[] = {"primary", "secondary", "placeholder"};
    static const std::function<bool(BlockPtr const&)> levelValidators[]
        = {[](BlockPtr const& block) { return block->isPrimary(); },
            [](BlockPtr const& block) { return !block->isPrimary(); },
            [](BlockPtr const& block) { return block->isPlaceholder(); }};
    bool queueCompromised = false;
    for (SizeType32 queueLevel = 0; queueLevel < kNumCacheLevels + 1; queueLevel++)
    {
        for (SizeType32 pri = 0; pri < kNumPriorities; pri++)
        {
            for (auto const& block : mFreeQueues[queueLevel][pri])
            {
                bool const valid = levelValidators[queueLevel](block);
                if (!valid)
                {
                    TLLM_LOG_WARNING("Block (id %d) has level=%s, but misplaced at queueLevel %s", block->getBlockId(),
                        levelToStr[queueLevel], levelToStr[queueLevel]);
                    queueCompromised = true;
                }
                if (block->hasRefs())
                {
                    TLLM_LOG_WARNING("Found block (id %d) with references at queueLevel %s", block->getBlockId(),
                        levelToStr[queueLevel]);
                    queueCompromised = true;
                }
            }
        }
    }
    TLLM_LOG_DEBUG("LRUEvictionPolicy queues are %s", queueCompromised ? "compromised" : "not compromised");
    return !queueCompromised;
}

std::tuple<BlockPtr, bool> LRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel, bool wantPlaceholder)
{
    SizeType32 const level = wantPlaceholder ? kPlaceholderLevel : cacheLevel;

    for (SizeType32 pri = 0; pri < kNumPriorities; pri++)
    {
        // Find the first non-empty queue, and return the first block.
        if (!mFreeQueues[level][pri].empty())
        {
            auto block = mFreeQueues[level][pri].front();

            // mFreeQueues only contains leaf blocks, so no need to iterate through the next block pointers.
            // It's possible to have a primary block with children in secondary memory. We handle this
            // by freeing all descendants in WindowBlockManager::getFreeBlock. This is done either by
            // offloading (preferred method) or explicitly.
            bool const canOffload
                = !wantPlaceholder && cacheLevel == 0 && pri >= getPriorityIdx(mSecondaryOffloadMinPriority);
            return std::make_tuple(block, canOffload);
        }
    }
    TLLM_THROW("No free block found. This shouldn't happen!");
}

void LRUEvictionPolicy::addToFreeBlockQueue(BlockPtr block, bool toFront)
{
    // Info needed to add (and later remove) block to queue.
    SizeType32 const cacheLevel = getCacheLevel(block);
    SizeType32 const id = block->getBlockId();
    SizeType32 const priority = getPriorityIdx(block->getPriority());
    // Add block to queue along with all info required to later remove it
    auto& q = mFreeQueues[cacheLevel][priority];
    auto insertItr = toFront ? q.begin() : q.end();
    mFreeBlockIterators[id] = std::make_tuple(cacheLevel, priority, q.insert(insertItr, block));
    mNumFreeBlocksPerLevel[cacheLevel]++;
}

bool LRUEvictionPolicy::removeFromFreeBlockQueue(BlockPtr block)
{
    SizeType32 const id = block->getBlockId();
    if (mFreeBlockIterators[id].has_value())
    {
        // Remove block using stored values, not current values
        auto [cacheLevel, priority, it] = mFreeBlockIterators[id].value();
        mFreeQueues[cacheLevel][priority].erase(it);
        mNumFreeBlocksPerLevel[cacheLevel] -= 1;
        mFreeBlockIterators[id] = std::nullopt;
        return true;
    }
    else
    {
        return false;
    }
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block)
{
    releaseBlock(block, false);
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block, bool toFront)
{
    // The dummy root block (kCachedBlocksRootId) is permanently attached to the lookup tree
    // via setAsRoot() and must never enter the eviction queue — it is not a real cache block.
    TLLM_CHECK_WITH_INFO(
        block->getBlockId() != tensorrt_llm::batch_manager::kv_cache_manager::KVCacheBlock::kCachedBlocksRootId,
        "Attempted to release the cached-blocks root into the eviction queue");

    addToFreeBlockQueue(block, toFront);

    if (block->getDurationMs().has_value()
        && block->getPriority() != executor::KvCacheRetentionConfig::kDefaultRetentionPriority)
    {
        auto expirationTime = getTime() + *block->getDurationMs();
        block->setExpirationTime(expirationTime);
        mExpiringBlockHeap.emplace(block);
    }
}

SizeType32 LRUEvictionPolicy::getNumFreeBlocks(SizeType32 cacheLevel)
{
    return mNumFreeBlocksPerLevel[cacheLevel];
}

void LRUEvictionPolicy::claimBlock(BlockPtr block)
{
    claimBlock(block, std::nullopt, std::nullopt);
}

void LRUEvictionPolicy::claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
    std::optional<std::chrono::milliseconds> durationMs)
{
    if (removeFromFreeBlockQueue(block))
    {
        // Only need to remove block from expiring block heap if block was removed from free blocks queue
        mExpiringBlockHeap.erase(block);
    }

    if (priority.has_value())
    {
        block->setPriority(*priority);
    }
    block->setDurationMs(durationMs);
}

std::chrono::steady_clock::time_point::duration LRUEvictionPolicy::getTime() const
{
    return std::chrono::steady_clock::now().time_since_epoch();
}

void LRUEvictionPolicy::refresh()
{
    while (!mExpiringBlockHeap.empty())
    {
        auto const block = *mExpiringBlockHeap.begin();
        if (block->getExpirationTime() > getTime())
        {
            break;
        }

        mExpiringBlockHeap.erase(mExpiringBlockHeap.begin());

        // Add block to free blocks queue with default priority if it was removed from another priority free blocks
        // queue
        if (removeFromFreeBlockQueue(block))
        {
            block->setPriority(kDefaultPriority);
            addToFreeBlockQueue(block, /*toFront*/ false);
        }
    }
}

} // namespace tensorrt_llm::batch_manager::eviction_policy
