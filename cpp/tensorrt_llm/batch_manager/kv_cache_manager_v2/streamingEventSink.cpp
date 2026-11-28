/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/streamingEventSink.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/page.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

StreamingEventSink::StreamingEventSink(int maxEntries, std::optional<int> mmTokenIdOffset)
    : mMaxEntries(maxEntries)
    , mMmTokenIdOffset(mmTokenIdOffset)
{
    if (mMaxEntries <= 0)
    {
        throw std::invalid_argument("maxEntries must be positive");
    }
    if (mMmTokenIdOffset.has_value() && *mMmTokenIdOffset < 0)
    {
        throw std::invalid_argument("mmTokenIdOffset must be non-negative");
    }
}

void StreamingEventSink::setTargetLifeCycle(LifeCycleId lifeCycle)
{
    if (lifeCycle.value() < 0)
    {
        throw std::invalid_argument("lifeCycle must be non-negative");
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mTargetLifeCycle = lifeCycle;
}

void StreamingEventSink::setRecurrentLifeCycles(std::vector<LifeCycleId> lifeCycles)
{
    for (auto const lifeCycle : lifeCycles)
    {
        if (lifeCycle.value() < 0)
        {
            throw std::invalid_argument("lifeCycle must be non-negative");
        }
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mTargetLifeCycle.has_value())
    {
        for (auto const lifeCycle : lifeCycles)
        {
            if (lifeCycle == *mTargetLifeCycle)
            {
                throw std::invalid_argument("a recurrent life cycle cannot also be the attention target");
            }
        }
    }
    mRecurrentLifeCycles = std::move(lifeCycles);
    mPublishRecurrentCheckpoints = true;
}

void StreamingEventSink::setPublishRecurrentCheckpoints(bool enabled)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mPublishRecurrentCheckpoints = enabled;
}

void StreamingEventSink::declareLifeCycles(LifeCycleRegistry const& lifeCycles)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mRecurrentLifeCycles.clear();
    for (auto const id : lifeCycles.ssmLifeCycleIds())
    {
        // PRIVATE SSM life cycles (speculative-decoding replay regions) never store a block.
        if (!isPrivateLifeCycle(lifeCycles[id]))
        {
            mRecurrentLifeCycles.push_back(id);
        }
    }
}

bool StreamingEventSink::publishesRecurrentCheckpoints() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return publishesRecurrentCheckpointsUnlocked();
}

bool StreamingEventSink::publishesRecurrentCheckpointsUnlocked() const
{
    return mPublishRecurrentCheckpoints && !mRecurrentLifeCycles.empty();
}

bool StreamingEventSink::isRecurrentLifeCycle(LifeCycleId lifeCycle) const
{
    return publishesRecurrentCheckpointsUnlocked()
        && std::find(mRecurrentLifeCycles.begin(), mRecurrentLifeCycles.end(), lifeCycle) != mRecurrentLifeCycles.end();
}

std::vector<StreamingEventData> StreamingEventSink::drainIterationEvents()
{
    std::lock_guard<std::mutex> lock(mMutex);
    auto events = std::move(mPendingEvents);
    mPendingEvents.clear();
    mPendingEntries = 0;
    return events;
}

StreamingEventStats StreamingEventSink::getStats() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mStats;
}

void StreamingEventSink::addStoredBlock(Block const& block)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addStoredBlockUnlocked(block);
    // A block announced with snapshots already attached (an orphan re-attached after its
    // snapshot was taken, or a block whose snapshot survived a target-page eviction) carries
    // its checkpoints in the same announcement.
    addResidentCheckpointsUnlocked(block);
}

void StreamingEventSink::addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle == *mTargetLifeCycle)
    {
        addStoredBlockUnlocked(block);
        // Re-storing the attention page republishes the block; its surviving snapshots must
        // come back with it, since the whole-block removal withdrew them.
        addResidentCheckpointsUnlocked(block);
        return;
    }
    if (isRecurrentLifeCycle(lifeCycle))
    {
        addRecurrentCheckpointUnlocked(block, lifeCycle);
        return;
    }
    ++mStats.nonTargetLifeCyclesIgnored;
}

void StreamingEventSink::addRemovedBlock(Digest const& blockKey)
{
    std::lock_guard<std::mutex> lock(mMutex);
    addRemovedBlockUnlocked(blockKey);
}

void StreamingEventSink::addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mTargetLifeCycle.has_value())
    {
        return;
    }
    if (lifeCycle == *mTargetLifeCycle)
    {
        addRemovedBlockUnlocked(blockKey);
        return;
    }
    if (isRecurrentLifeCycle(lifeCycle))
    {
        // The snapshot page was evicted on its own; the attention block stays published.
        auto const stored = mStoredBlocks.find(blockKey);
        if (stored == mStoredBlocks.end())
        {
            return;
        }
        auto& groups = stored->second.publishedCheckpointGroups;
        auto const it = std::find(groups.begin(), groups.end(), lifeCycle.value());
        if (it == groups.end())
        {
            return;
        }
        groups.erase(it);
        addRemovedHashUnlocked(stored->second.blockHash, lifeCycle.value());
        ++mStats.recurrentCheckpointsRemoved;
        return;
    }
    ++mStats.nonTargetLifeCyclesIgnored;
}

void StreamingEventSink::addAllBlocksCleared()
{
    std::lock_guard<std::mutex> lock(mMutex);
    // Reset markers must survive the entry cap, like removals. Keep queued
    // events in order and discard all residency remembered before the reset.
    mPendingEvents.emplace_back(StreamingAllBlocksClearedData{});
    mStoredBlocks.clear();
}

void StreamingEventSink::addCacheLevelUpdated(Digest const&, CacheLevel, CacheLevel, LifeCycleId)
{
    // The streaming protocol currently tracks radix-tree residency, not cache-tier movement.
}

void StreamingEventSink::addStoredBlockUnlocked(Block const& block)
{
    if (!mTargetLifeCycle.has_value() || *mTargetLifeCycle >= block.storage.size())
    {
        return;
    }
    auto const* page = block.getPage(*mTargetLifeCycle);
    if (page == nullptr)
    {
        return;
    }
    if (!block.isFull() || page->numTokensInBlock < static_cast<int>(block.tokens.size()))
    {
        ++mStats.partialBlocksSuppressed;
        return;
    }
    if (mStoredBlocks.count(block.key) != 0)
    {
        return;
    }
    if (block.prev == nullptr)
    {
        throw std::logic_error("Cannot publish an orphan KV cache block");
    }

    int64_t const blockHash = wireHash(block.key);
    std::optional<int64_t> parentHash;
    std::optional<LoraTaskIdType> loraId;
    if (block.prev->type() == NodeBase::Type::kBLOCK)
    {
        auto const& parent = *static_cast<Block const*>(block.prev);
        auto const storedParent = mStoredBlocks.find(parent.key);
        if (storedParent == mStoredBlocks.end())
        {
            recordDroppedEventUnlocked("parent block has not been published");
            return;
        }
        parentHash = storedParent->second.blockHash;
        // Inherit scope metadata from the published parent without walking the ancestor chain.
        loraId = storedParent->second.loraId;
    }
    else
    {
        loraId = static_cast<RootBlock const*>(block.prev)->reuseScope.loraId;
    }

    auto decoded = decodeEventBlock(block, mMmTokenIdOffset);
    if (!reserveEntryUnlocked())
    {
        return;
    }

    mStoredBlocks.emplace(block.key, StoredBlock{blockHash, loraId});
    if (!mPendingEvents.empty())
    {
        auto* stored = std::get_if<StreamingBlockStoredData>(&mPendingEvents.back());
        // Never merge into a tagged checkpoint event: it shares hashes with attention blocks.
        if (stored != nullptr && !stored->groupIdx.has_value() && !stored->blockHashes.empty() && parentHash.has_value()
            && stored->blockHashes.back() == *parentHash && stored->loraId == loraId)
        {
            stored->blockHashes.push_back(blockHash);
            stored->tokenIds.insert(stored->tokenIds.end(), std::make_move_iterator(decoded.tokenIds.begin()),
                std::make_move_iterator(decoded.tokenIds.end()));
            stored->mmKeys.push_back(std::move(decoded.mmKeys));
            ++mStats.storedBlocks;
            return;
        }
    }
    std::vector<std::vector<MmKey>> mmKeys;
    mmKeys.push_back(std::move(decoded.mmKeys));
    mPendingEvents.emplace_back(
        StreamingBlockStoredData{{blockHash}, parentHash, std::move(decoded.tokenIds), std::move(mmKeys), loraId});
    ++mStats.storedBlocks;
}

void StreamingEventSink::addResidentCheckpointsUnlocked(Block const& block)
{
    if (!publishesRecurrentCheckpointsUnlocked())
    {
        return;
    }
    for (auto const lifeCycle : mRecurrentLifeCycles)
    {
        if (lifeCycle < block.storage.size() && block.getPage(lifeCycle) != nullptr)
        {
            addRecurrentCheckpointUnlocked(block, lifeCycle);
        }
    }
}

void StreamingEventSink::addRecurrentCheckpointUnlocked(Block const& block, LifeCycleId lifeCycle)
{
    // The checkpoint is keyed by the attention block's hash, so the block must already be
    // published; a partial block or an unpublished parent chain publishes no checkpoint either.
    auto const stored = mStoredBlocks.find(block.key);
    if (stored == mStoredBlocks.end())
    {
        return;
    }
    if (lifeCycle >= block.storage.size())
    {
        return;
    }
    auto const* page = block.getPage(lifeCycle);
    if (page == nullptr)
    {
        return;
    }
    // A snapshot taken inside the block (unaligned snapshot point) is a valid resume point only
    // at that token depth, which the wire format cannot express: publish full-block snapshots only.
    if (!block.isFull() || page->numTokensInBlock < static_cast<int>(block.tokens.size()))
    {
        ++mStats.recurrentCheckpointsSuppressed;
        return;
    }
    int const groupIdx = lifeCycle.value();
    auto& groups = stored->second.publishedCheckpointGroups;
    if (std::find(groups.begin(), groups.end(), groupIdx) != groups.end())
    {
        return;
    }
    std::optional<int64_t> parentHash;
    if (block.prev != nullptr && block.prev->type() == NodeBase::Type::kBLOCK)
    {
        auto const& parent = *static_cast<Block const*>(block.prev);
        auto const storedParent = mStoredBlocks.find(parent.key);
        if (storedParent != mStoredBlocks.end())
        {
            parentHash = storedParent->second.blockHash;
        }
    }
    if (!reserveEntryUnlocked())
    {
        return;
    }
    int64_t const blockHash = stored->second.blockHash;
    auto const loraId = stored->second.loraId;
    groups.push_back(groupIdx);
    ++mStats.recurrentCheckpointsStored;
    if (!mPendingEvents.empty())
    {
        auto* pending = std::get_if<StreamingBlockStoredData>(&mPendingEvents.back());
        if (pending != nullptr && pending->groupIdx == groupIdx && !pending->blockHashes.empty()
            && parentHash.has_value() && pending->blockHashes.back() == *parentHash && pending->loraId == loraId)
        {
            pending->blockHashes.push_back(blockHash);
            pending->mmKeys.emplace_back();
            return;
        }
    }
    std::vector<std::vector<MmKey>> mmKeys(1);
    mPendingEvents.emplace_back(StreamingBlockStoredData{
        {blockHash}, parentHash, {}, std::move(mmKeys), loraId, groupIdx, std::string("mamba")});
}

void StreamingEventSink::addRemovedBlockUnlocked(Digest const& blockKey)
{
    auto const stored = mStoredBlocks.find(blockKey);
    if (stored == mStoredBlocks.end())
    {
        return;
    }
    int64_t const blockHash = stored->second.blockHash;
    auto const groups = std::move(stored->second.publishedCheckpointGroups);
    mStoredBlocks.erase(stored);
    for (int const groupIdx : groups)
    {
        addRemovedHashUnlocked(blockHash, groupIdx);
        ++mStats.recurrentCheckpointsRemoved;
    }
    addRemovedHashUnlocked(blockHash);
}

void StreamingEventSink::addRemovedHashUnlocked(int64_t blockHash, std::optional<int> groupIdx)
{
    if (!mPendingEvents.empty())
    {
        auto* removed = std::get_if<StreamingBlockRemovedData>(&mPendingEvents.back());
        if (removed != nullptr && removed->groupIdx == groupIdx)
        {
            removed->blockHashes.push_back(blockHash);
            if (!groupIdx.has_value())
            {
                ++mStats.removedBlocks;
            }
            return;
        }
    }
    if (groupIdx.has_value())
    {
        mPendingEvents.emplace_back(StreamingBlockRemovedData{{blockHash}, groupIdx, std::string("mamba")});
        return;
    }
    mPendingEvents.emplace_back(StreamingBlockRemovedData{{blockHash}});
    ++mStats.removedBlocks;
}

bool StreamingEventSink::reserveEntryUnlocked()
{
    if (mPendingEntries < mMaxEntries)
    {
        ++mPendingEntries;
        return true;
    }
    recordDroppedEventUnlocked("per-iteration safety cap was exceeded");
    return false;
}

void StreamingEventSink::recordDroppedEventUnlocked(char const* reason)
{
    ++mStats.droppedEvents;
    int64_t const dropped = mStats.droppedEvents;
    if (dropped == 1 || (dropped & (dropped - 1)) == 0)
    {
        TLLM_LOG_WARNING("Dropping streaming KV store event because %s; dropped_events=%" PRId64, reason, dropped);
    }
}

int64_t StreamingEventSink::wireHash(Digest const& digest)
{
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(value); ++i)
    {
        value = (value << 8U) | std::to_integer<uint8_t>(digest[i]);
    }
    uint64_t constexpr kSignedMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (value <= kSignedMax)
    {
        return static_cast<int64_t>(value);
    }
    return -static_cast<int64_t>(~value) - 1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
