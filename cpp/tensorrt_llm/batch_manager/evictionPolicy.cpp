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
        mReleasedBlocks.emplace_back(std::unordered_set<SizeType32>());

        auto& freeQueue = mFreeQueues[cacheLevel][defaultPriorityIdx];

        for (SizeType32 blockId = 0; blockId < sizes[cacheLevel]; blockId++)
        {
            // Initialize all blocks to be the default priority level
            mFreeBlockIterators.emplace_back(freeQueue.insert(freeQueue.end(), mAllBlocksById[startIdx + blockId]));
            mReleasedBlocks[cacheLevel].insert(startIdx + blockId);
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

std::tuple<BlockPtr, bool> LRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel)
{
    for (SizeType32 level = 0; level < kMaxPriority - kMinPriority + 1; level++)
    {
        // Find the first non-empty queue, and return the first block.
        if (!mFreeQueues[cacheLevel][level].empty())
        {
            auto block = mFreeQueues[cacheLevel][level].front();

            // mFreeQueues only contains leaf blocks, so no need to iterate through the next block pointers.
            // It's theoretically possible for a primary block below the offload threshold to have a child in secondary
            // memory. This would only happen if block priorities are higher at the end of the sequence than the
            // beginning. In this case, we still need to offload, even if it's below the threshold.
            return std::make_tuple(
                block, cacheLevel == 0 && (level > mSecondaryOffloadMinPriority || !block->getNextBlocks().empty()));
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

    mReleasedBlocks[cacheLevel].insert(id);

    // It's possible that this block is the child of a matched block that's in mFreeQueues. If this happens, we need to
    // remove the parent from mFreeQueues, since it's no longer a released leaf block.
    auto parent = block->getPrevBlock();
    if (parent != nullptr)
    {
        auto const parentId = parent->getBlockId();
        if (parentId != KVCacheBlock::kCachedBlocksRootId && mFreeBlockIterators[parent->getBlockId()] != std::nullopt
            && !isReleasedLeafBlock(parent))
        {
            mFreeQueues[getCacheLevel(parent)][getPriorityIdx(parent->getPriority())].erase(
                *mFreeBlockIterators[parentId]);
            mFreeBlockIterators[parentId] = std::nullopt;
        }
    }

    if (mFreeBlockIterators[block->getBlockId()] == std::nullopt && isReleasedLeafBlock(block))
    {
        // If there are no children, this is a leaf block. Insert into a queue.
        auto& q = mFreeQueues[cacheLevel][getPriorityIdx(block->getPriority())];
        if (toFront)
        {
            mFreeBlockIterators[id] = q.insert(q.begin(), block);
        }
        else
        {
            mFreeBlockIterators[id] = q.insert(q.end(), block);
        }
    }

    mNumFreeBlocksPerLevel[cacheLevel]++;

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

    if (mReleasedBlocks[cacheLevel].find(id) != mReleasedBlocks[cacheLevel].end())
    {
        mNumFreeBlocksPerLevel[cacheLevel] -= 1;
        mReleasedBlocks[cacheLevel].erase(id);
    }

    if (mFreeBlockIterators[id] != std::nullopt)
    {
        mFreeQueues[cacheLevel][getPriorityIdx(block->getPriority())].erase(*mFreeBlockIterators[id]);

        BlockPtr const parent = block->getPrevBlock();

        if (parent.get() != nullptr && parent->getBlockId() != KVCacheBlock::kCachedBlocksRootId
            && mFreeBlockIterators[parent->getBlockId()] == std::nullopt && isReleasedLeafBlock(parent))
        {
            auto& q = mFreeQueues[getCacheLevel(parent)][getPriorityIdx(parent->getPriority())];
            mFreeBlockIterators[parent->getBlockId()] = q.insert(q.end(), parent);
        }
    }

    mFreeBlockIterators[id] = std::nullopt;

    if (priority.has_value())
    {
        block->setPriority(*priority);
    }

    mExpiringBlockHeap.erase(block);
    block->setDurationMs(durationMs);
}

bool LRUEvictionPolicy::isReleasedLeafBlock(BlockPtr const& block)
{
    SizeType32 const blockCacheLevel = getCacheLevel(block);

    if (mReleasedBlocks[blockCacheLevel].find(block->getBlockId()) == mReleasedBlocks[blockCacheLevel].end())
    {
        return false;
    }

    for (auto const& p : block->getNextBlocks())
    {
        SizeType32 const childCacheLevel = getCacheLevel(p.second);
        if (mReleasedBlocks[childCacheLevel].find(p.second->getBlockId()) != mReleasedBlocks[childCacheLevel].end()
            && childCacheLevel <= blockCacheLevel)
        {
            return false;
        }
    }

    return true;
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
