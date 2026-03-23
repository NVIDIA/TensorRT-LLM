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
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"

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
    if (block->isPlaceholder())
    {
        return 0;
    }
    return block->isPrimary() ? 0 : 1;
}

SizeType32 getPriorityIdx(executor::RetentionPriority priority)
{
    return priority - kMinPriority;
}
} // namespace

void LRUEvictionPolicy::initialize(std::vector<BlockPtr>& mAllBlocksById, std::vector<SizeType32> blocksPerCacheLevel,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
{
    SizeType32 startIdx = 0;

    auto const defaultPriorityIdx = getPriorityIdx(kDefaultPriority);

    // For each cache level, create a separate list of queues.
    for (SizeType32 cacheLevel = 0; cacheLevel < kNumCacheLevels; cacheLevel++)
    {
        mFreeBlockIterators.reserve(mFreeBlockIterators.size() + blocksPerCacheLevel[cacheLevel]);
        mFreeQueues.emplace_back(std::vector<FreeBlocksQueue>(kMaxPriority - kMinPriority + 1));

        auto& freeQueue = mFreeQueues[cacheLevel][defaultPriorityIdx];

        for (SizeType32 blockId = 0; blockId < blocksPerCacheLevel[cacheLevel]; blockId++)
        {
            // Initialize all blocks to be the default priority level
            mFreeBlockIterators.emplace_back(freeQueue.insert(freeQueue.end(), mAllBlocksById[startIdx + blockId]));
        }

        startIdx += blocksPerCacheLevel[cacheLevel];
    }
    mNumFreeBlocksPerLevel = blocksPerCacheLevel;

    mSecondaryOffloadMinPriority = secondaryOffloadMinPriority.value_or(kDefaultSecondaryOffloadMinPriority);
}

bool LRUEvictionPolicy::verifyQueueIntegrity()
{
    bool queueCompromised = false;
    for (SizeType32 cacheLevel = 0; cacheLevel < kNumCacheLevels; cacheLevel++)
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

std::tuple<BlockPtr, bool> LRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel, bool wantPlaceholder)
{
    TLLM_CHECK_WITH_INFO(!wantPlaceholder,
        "LRUEvictionPolicy does not manage placeholder blocks. Use MaybePlaceholderLRUEvictionPolicy.");

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
    // The dummy root block (kCachedBlocksRootId) is permanently attached to the lookup tree
    // via setAsRoot() and must never enter the eviction queue — it is not a real cache block.
    TLLM_CHECK_WITH_INFO(
        block->getBlockId() != tensorrt_llm::batch_manager::kv_cache_manager::KVCacheBlock::kCachedBlocksRootId,
        "Attempted to release the cached-blocks root into the eviction queue");
    SizeType32 const cacheLevel = getCacheLevel(block);
    SizeType32 const idx = blockIdx(block->getBlockId());

    // If there are no children, this is a leaf block. Insert into a queue.
    auto& q = mFreeQueues[cacheLevel][getPriorityIdx(block->getPriority())];
    if (toFront)
    {
        mFreeBlockIterators[idx] = q.insert(q.begin(), block);
    }
    else
    {
        mFreeBlockIterators[idx] = q.insert(q.end(), block);
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
    SizeType32 const idx = blockIdx(block->getBlockId());
    SizeType32 const cacheLevel = getCacheLevel(block);

    if (mFreeBlockIterators[idx] != std::nullopt)
    {
        mFreeQueues[cacheLevel][getPriorityIdx(block->getPriority())].erase(*mFreeBlockIterators[idx]);
        mNumFreeBlocksPerLevel[cacheLevel] -= 1;
    }

    mFreeBlockIterators[idx] = std::nullopt;

    if (priority.has_value())
    {
        block->setPriority(*priority);
    }

    mExpiringBlockHeap.erase(block);
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

        auto const idx = blockIdx(block->getBlockId());
        auto const level = getCacheLevel(block);

        mExpiringBlockHeap.erase(mExpiringBlockHeap.begin());

        if (mFreeBlockIterators[idx] != std::nullopt)
        {
            // This is already in another queue. Delete it, and bring it down to the default queue
            mFreeQueues[level][getPriorityIdx(block->getPriority())].erase(*mFreeBlockIterators[idx]);
            auto& q = mFreeQueues[level][getPriorityIdx(kDefaultPriority)];
            mFreeBlockIterators[idx] = q.insert(q.end(), block);
        }
        block->setPriority(kDefaultPriority);
    }
}

// ---- PlaceholderInnerLRUEvictionPolicy ----
// Manages pre-allocated placeholder blocks (with negative IDs starting at -2) via the standard queue
// system. Overrides blockIdx() to map negative IDs to 0-based queue indices.
namespace
{
class PlaceholderInnerLRUEvictionPolicy : public LRUEvictionPolicy
{
protected:
    SizeType32 blockIdx(KVCacheBlock::IdType blockId) const override
    {
        // blockId is negative: -2 → 0, -3 → 1, ...
        TLLM_CHECK_WITH_INFO(blockId < -1, "PlaceholderInnerLRUEvictionPolicy expects blockId < -1, got %d", blockId);
        return -blockId - 2;
    }

public:
    bool verifyQueueIntegrity() override
    {
        bool queueCompromised = false;
        for (SizeType32 level = 0; level < kMaxPriority - kMinPriority + 1; level++)
        {
            for (auto const& block : mFreeQueues[kPrimaryLevel][level])
            {
                if (!block->isPlaceholder())
                {
                    TLLM_LOG_WARNING("Found non-placeholder block (id %d) in PlaceholderInnerLRUEvictionPolicy",
                        block->getBlockId());
                    queueCompromised = true;
                }
                if (block->hasRefs())
                {
                    TLLM_LOG_WARNING(
                        "Found placeholder block (id %d) with references in placeholder policy", block->getBlockId());
                    queueCompromised = true;
                }
            }
        }
        return !queueCompromised;
    }
};
} // anonymous namespace

// ---- MaybePlaceholderLRUEvictionPolicy ----

void MaybePlaceholderLRUEvictionPolicy::initializePlaceholders(std::vector<BlockPtr>& allPlaceholderBlocksById,
    SizeType32 numPlaceholderBlocks, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority)
{
    mPlaceholderEvictionPolicy = std::make_shared<PlaceholderInnerLRUEvictionPolicy>();

    // Extract the actual placeholder blocks from allPlaceholderBlocksById[2..numPlaceholderBlocks+1]
    // so the inner policy's mFreeBlockIterators[i] corresponds to blockId = -(i+2).
    std::vector<BlockPtr> placeholderBlocks(
        allPlaceholderBlocksById.begin() + 2, allPlaceholderBlocksById.begin() + numPlaceholderBlocks + 2);

    mPlaceholderEvictionPolicy->initialize(placeholderBlocks, {numPlaceholderBlocks, 0}, secondaryOffloadMinPriority);
}

std::tuple<BlockPtr, bool> MaybePlaceholderLRUEvictionPolicy::getFreeBlock(SizeType32 cacheLevel, bool wantPlaceholder)
{
    if (wantPlaceholder)
    {
        TLLM_CHECK_WITH_INFO(mPlaceholderEvictionPolicy != nullptr,
            "Placeholder eviction policy not initialized. Call initializePlaceholders() first.");
        return mPlaceholderEvictionPolicy->getFreeBlock(kPrimaryLevel);
    }
    return LRUEvictionPolicy::getFreeBlock(cacheLevel);
}

void MaybePlaceholderLRUEvictionPolicy::releaseBlock(BlockPtr block, bool toFront)
{
    if (block->isPlaceholder())
    {
        TLLM_CHECK_WITH_INFO(mPlaceholderEvictionPolicy != nullptr,
            "Placeholder eviction policy not initialized. Call initializePlaceholders() first.");
        mPlaceholderEvictionPolicy->releaseBlock(block, toFront);
        return;
    }
    LRUEvictionPolicy::releaseBlock(block, toFront);
}

void MaybePlaceholderLRUEvictionPolicy::claimBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
    std::optional<std::chrono::milliseconds> durationMs)
{
    if (block->isPlaceholder())
    {
        TLLM_CHECK_WITH_INFO(mPlaceholderEvictionPolicy != nullptr,
            "Placeholder eviction policy not initialized. Call initializePlaceholders() first.");
        mPlaceholderEvictionPolicy->claimBlock(block, priority, durationMs);
        return;
    }
    LRUEvictionPolicy::claimBlock(block, priority, durationMs);
}

void MaybePlaceholderLRUEvictionPolicy::refresh()
{
    LRUEvictionPolicy::refresh();
    if (mPlaceholderEvictionPolicy)
    {
        mPlaceholderEvictionPolicy->refresh();
    }
}

bool MaybePlaceholderLRUEvictionPolicy::verifyQueueIntegrity()
{
    bool ok = LRUEvictionPolicy::verifyQueueIntegrity();
    if (mPlaceholderEvictionPolicy)
    {
        ok = mPlaceholderEvictionPolicy->verifyQueueIntegrity() && ok;
    }
    return ok;
}

} // namespace tensorrt_llm::batch_manager::eviction_policy
