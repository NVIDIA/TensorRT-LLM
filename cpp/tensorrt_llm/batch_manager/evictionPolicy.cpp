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

auto const kDefaultPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority;
executor::RetentionPriority const kDefaultSecondaryOffloadMinPriority = 30;

int const kNumCacheLevels = 2;

namespace
{
SizeType32 getCacheLevel(BlockPtr const& block)
{
    return block->isPrimary() ? 0 : 1;
}

SizeType32 getPriorityIdx(executor::RetentionPriority priority)
{
    return priority - kMinPriority;
}
} // namespace

void LRUEvictionPolicy::initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> sizes,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
{
    SizeType32 startIdx = 0;

    auto const defaultPriorityIdx = getPriorityIdx(kDefaultPriority);

    // For each cache level, create a separate list of queues.
    for (SizeType32 cacheLevel = 0; cacheLevel < kNumCacheLevels; cacheLevel++)
    {
        mFreeBlockIterators.reserve(mFreeBlockIterators.size() + sizes[cacheLevel]);
        mFreeQueues.emplace_back(std::vector<FreeBlocksQueue>(kMaxPriority - kMinPriority + 1));

        auto& freeQueue = mFreeQueues[cacheLevel][defaultPriorityIdx];

        for (SizeType32 blockId = 0; blockId < sizes[cacheLevel]; blockId++)
        {
            // Initialize all blocks to be the default priority level
            mFreeBlockIterators.emplace_back(freeQueue.insert(freeQueue.end(), mAllBlocksById[startIdx + blockId]));
        }

        startIdx += sizes[cacheLevel];
    }
    mNumFreeBlocksPerLevel = sizes;

    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority.value_or(kDefaultSecondaryOffloadMinPriority);
}

bool LRUEvictionPolicy::verifyQueueIntegrity()
{
    bool queueCompromised = false;
    for (SizeType32 cacheLevel = 0; cacheLevel < 2; cacheLevel++)
    {
        for (SizeType32 level = 0; level < kMaxPriority - kMinPriority + 1; level++)
        {
            for (auto const& block : mFreeQueues[cacheLevel][level])
            {
                if ((cacheLevel == 0 && !block->isPrimary()) || (cacheLevel == 1 && block->isPrimary()))
                {
                    TLLM_LOG_WARNING("Found %s block (id %d) at cacheLevel %d",
                        block->isPrimary() ? "primary" : "secondary", block->getBlockId(), cacheLevel);
                    queueCompromised = true;
                }
                if (block->hasRefs())
                {
                    TLLM_LOG_WARNING(
                        "Found block (id %d) with references at cacheLevel %d", block->getBlockId(), cacheLevel);
                    queueCompromised = true;
                }
            }
        }
    }
    TLLM_LOG_DEBUG("LRUEvictionPolicy queues are %s", queueCompromised ? "compromised" : "not compromised");
    return !queueCompromised;
}

std::string LRUEvictionPolicy::printFreeQueues() const
{
    std::stringstream os;
    os << "Free queues:" << std::endl;
    for (SizeType32 cacheLevel = 0; cacheLevel < 2; cacheLevel++)
    {
        switch (cacheLevel)
        {
        case 0: os << "  Primary" << std::endl; break;
        case 1: os << "  Secondary" << std::endl; break;
        default: throw "Unknown cacheLevel";
        }
        for (SizeType32 level = 0; level < kMaxPriority - kMinPriority + 1; level++)
        {
            auto const& blocks = mFreeQueues[cacheLevel][level];
            if (!blocks.empty())
            {
                os << "    " << level << std::setw(3) << " : ";
                for (auto const& block : blocks)
                {
                    os << block->getBlockId() << " ";
                }
                os << std::endl;
            }
        }
    }
    return os.str();
}

std::tuple<BlockPtr, bool> LRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel)
{
    for (SizeType32 level = 0; level < kMaxPriority - kMinPriority + 1; level++)
    {
        // Find the first non-empty queue, and return the first block.
        if (!mFreeQueues[cacheLevel][level].empty())
        {
            auto block = mFreeQueues[cacheLevel][level].front();

            // mFreeQueues only contains leaf blocks, so no need to iterate through the next block pointers.
            // It's possible to have a primary block with children in secondary memory. We handle this
            // by freeing all descendants in WindowBlockManager::getFreeBlock. This is done either by
            // offloading (preferred method) or explicitly.
            return std::make_tuple(block, cacheLevel == 0 && level >= mSecondaryOffloadMinPriority);
        }
    }
    TLLM_THROW("No free block found. This shouldn't happen!");
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block)
{
    releaseBlock(block, false);
}

void LRUEvictionPolicy::releaseBlock(BlockPtr block, bool toFront)
{
    SizeType32 const cacheLevel = getCacheLevel(block);
    SizeType32 const id = block->getBlockId();

    auto priority = block->getPriority();
    auto priorityIdx = getPriorityIdx(priority);
    TLLM_LOG_DEBUG("%s;%d - LRUEvictionPolicy::releaseBlock :: blockId=%d, cacheLevel=%d, priority=%d, priorityIdx=%d",
        __FILE__, __LINE__, id, cacheLevel, priority, priorityIdx);
    auto& q = mFreeQueues[cacheLevel][priorityIdx];
    if (toFront)
    {
        mFreeBlockIterators[id] = q.insert(q.begin(), block);
    }
    else
    {
        mFreeBlockIterators[id] = q.insert(q.end(), block);
    }

    mNumFreeBlocksPerLevel[cacheLevel]++;
    TLLM_LOG_DEBUG("Increased mNumFreeBlocksPerLevel[%d] from %d to %d", cacheLevel,
        mNumFreeBlocksPerLevel[cacheLevel] - 1, mNumFreeBlocksPerLevel[cacheLevel]);

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
    SizeType32 const id = block->getBlockId();
    SizeType32 const cacheLevel = getCacheLevel(block);
    auto priorityIdx = getPriorityIdx(getPriorityIdx(block->getPriority()));
    TLLM_LOG_DEBUG("%s;%d - LRUEvictionPolicy::claimBlock :: blockId=%d, cacheLevel=%d, priority=%d, priorityIdx=%d",
        __FILE__, __LINE__, id, cacheLevel, priority, priorityIdx);

    // Detach block from free queue
    if (mFreeBlockIterators[id] != std::nullopt)
    {
        mFreeQueues[cacheLevel][priorityIdx].erase(*mFreeBlockIterators[id]);
        mNumFreeBlocksPerLevel[cacheLevel] -= 1;
        TLLM_LOG_DEBUG("Decreased mNumFreeBlocksPerLevel[%d] from %d to %d", cacheLevel,
            mNumFreeBlocksPerLevel[cacheLevel] + 1, mNumFreeBlocksPerLevel[cacheLevel]);
        mFreeBlockIterators[id] = std::nullopt;
    }

    // Explicitly set priority, if provided
    if (priority.has_value())
    {
        block->setPriority(*priority);
    }

    // Detach block from expiring heap (processing of time limited retention priority)
    mExpiringBlockHeap.erase(block);
    if (durationMs.has_value())
    {
        block->setDurationMs(durationMs);
    }
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

        auto const id = block->getBlockId();
        auto const level = getCacheLevel(block);

        mExpiringBlockHeap.erase(mExpiringBlockHeap.begin());

        if (mFreeBlockIterators[id] != std::nullopt)
        {
            // This is already in another queue. Delete it, and bring it down to the default queue
            mFreeQueues[level][getPriorityIdx(block->getPriority())].erase(*mFreeBlockIterators[id]);
            auto& q = mFreeQueues[level][getPriorityIdx(kDefaultPriority)];
            mFreeBlockIterators[id] = q.insert(q.end(), block);
        }
        block->setPriority(kDefaultPriority);
    }
}

} // namespace tensorrt_llm::batch_manager::eviction_policy
