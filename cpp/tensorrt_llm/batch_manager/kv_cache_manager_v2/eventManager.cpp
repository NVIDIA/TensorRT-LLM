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

#include "kv_cache_manager_v2/eventManager.h"

#include "kv_cache_manager_v2/blockRadixTree.h"
#include "kv_cache_manager_v2/page.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

constexpr std::size_t kMaxRoutingBlocksPerEvent = 256;

// A page whose recorded token count is short of the block's span cannot be announced: the
// event payload carries the block's full token list and cannot express a shorter valid
// prefix.
bool pageCoversBlock(CommittedPage const* page, Block const& block)
{
    return page != nullptr && page->numTokensInBlock >= static_cast<int>(block.tokens.size());
}

} // namespace

EventManager::EventManager(int maxKvEventEntries, int windowSize, std::optional<int> attentionDpRank,
    AttentionDpGatherFn attentionDpGather, std::string hashAlgo, std::map<int, int> windowSizeByLayerGroup)
    : mMaxKvEventEntries(maxKvEventEntries)
    , mWindowSize(windowSize)
    , mWindowSizeByLayerGroup(std::move(windowSizeByLayerGroup))
    , mAttentionDpRank(attentionDpRank)
    , mAttentionDpGather(std::move(attentionDpGather))
{
    std::tie(mHashAlgo, mHashAlgoName) = parseHashAlgorithm(hashAlgo);
}

std::pair<EventManager::HashAlgorithm, std::string> EventManager::parseHashAlgorithm(std::string const& hashAlgo)
{
    if (hashAlgo == "auto" || hashAlgo == "v1_block_key")
    {
        return {HashAlgorithm::kV1, "v1_block_key"};
    }
    if (hashAlgo == "v2_sha256")
    {
        return {HashAlgorithm::kV2Sha256, "v2_sha256"};
    }
    if (hashAlgo == "v2_sha256_64")
    {
        return {HashAlgorithm::kV2Sha256_64, "v2_sha256_64"};
    }
    throw std::invalid_argument("Unsupported V2 KV cache event hash algorithm: " + hashAlgo);
}

void EventManager::addCreatedEvent(
    std::vector<int> numBlocksPerCacheLevel, std::optional<std::vector<int>> layerGroupIds)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed)
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        // Direct routing only consumes stored/removed state. Avoid allocating a
        // created event that the wire translator would immediately discard.
        return;
    }
    KVCacheCreatedData data{std::move(numBlocksPerCacheLevel)};
    if (!layerGroupIds.has_value())
    {
        if (acceptsRoutingLayerGroup(std::nullopt))
        {
            addEventUnlocked(std::move(data), std::nullopt);
        }
        return;
    }
    for (int layerGroupId : *layerGroupIds)
    {
        if (acceptsRoutingLayerGroup(layerGroupId))
        {
            addEventUnlocked(data, layerGroupId);
        }
    }
}

void EventManager::setLayerGroupWindowSizes(std::map<int, int> windowSizes)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mWindowSizeByLayerGroup = std::move(windowSizes);
}

void EventManager::setRoutingLayerGroup(int layerGroupId)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (layerGroupId < 0)
    {
        throw std::invalid_argument("Routing layer group must be non-negative");
    }
    if (mHashAlgo != HashAlgorithm::kV1)
    {
        throw std::logic_error("Routing events require v1_block_key hashes");
    }
    if (mClosing || mClosed || mNextEventId != 0 || !mPendingEvents.empty() || !mEvents.empty()
        || !mLatestRemovedBlockHashes.empty() || !mStoredBlocks.empty() || !mSuppressedRoutingBlockKeys.empty()
        || !mV1HashStates.empty())
    {
        throw std::logic_error("Routing event selection must be configured before use");
    }
    mRoutingLayerGroupId = layerGroupId;
}

void EventManager::addStoredEvent(KVCacheStoredData data, EventLayerGroupId layerGroupId)
{
    if (data.blocks.empty() || mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(layerGroupId))
    {
        return;
    }
    flushRemovedEventUnlocked(layerGroupId);
    addStoredEventUnlocked(std::move(data), layerGroupId);
}

void EventManager::addRemovedEvent(std::vector<EventBlockHash> blockHashes, EventLayerGroupId layerGroupId)
{
    if (blockHashes.empty() || mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(layerGroupId))
    {
        return;
    }
    enqueueRemovedEventUnlocked(std::move(blockHashes), layerGroupId);
}

void EventManager::addUpdatedEvent(EventBlockHash blockHash, std::optional<KVCacheEventDiff> cacheLevel,
    std::optional<KVCacheEventDiff> priority, EventLayerGroupId layerGroupId)
{
    if ((!cacheLevel.has_value() && !priority.has_value()) || mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(layerGroupId))
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value() && mPendingRoutingEntries >= mMaxKvEventEntries)
    {
        ++mDroppedEventCount;
        return;
    }
    mPendingRoutingEntries += mRoutingLayerGroupId.has_value() ? 1 : 0;
    addEventUnlocked(KVCacheUpdatedData{std::move(blockHash), cacheLevel, priority}, layerGroupId);
}

void EventManager::addUpdatedEvent(Digest const& blockKey, std::optional<KVCacheEventDiff> cacheLevel,
    std::optional<KVCacheEventDiff> priority, EventLayerGroupId layerGroupId)
{
    if ((!cacheLevel.has_value() && !priority.has_value()) || mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(layerGroupId))
    {
        return;
    }
    auto const state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value() && mPendingRoutingEntries >= mMaxKvEventEntries)
    {
        ++mDroppedEventCount;
        return;
    }
    mPendingRoutingEntries += mRoutingLayerGroupId.has_value() ? 1 : 0;
    addEventUnlocked(KVCacheUpdatedData{state->second.blockHash, cacheLevel, priority}, layerGroupId);
}

void EventManager::addStoredBlock(Block const& block)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed)
    {
        return;
    }
    addStoredBlockUnlocked(block);
}

void EventManager::addStoredBlockUnlocked(Block const& block)
{
    if (mSuppressedRoutingBlockKeys.count(block.key) != 0)
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        int const lifeCycleId = *mRoutingLayerGroupId;
        auto const* page = lifeCycleId >= 0 && lifeCycleId < block.storage.size().value()
            ? block.getPage(LifeCycleId{lifeCycleId})
            : nullptr;
        if (!pageCoversBlock(page, block))
        {
            return;
        }
        std::vector<UniqueToken> eventTokens;
        EventBlockHash blockHash = v1HashFromBlock(block, &eventTokens);
        if (!isRoutingBlockSupported(block))
        {
            mSuppressedRoutingBlockKeys.insert(block.key);
            return;
        }
        mStoredBlocks.insert_or_assign(block.key, StoredBlockState{blockHash, {}});
        flushRemovedEventUnlocked(lifeCycleId);
        addStoredEventUnlocked(KVCacheStoredData{parentHashFromBlock(block),
                                   {KVCacheStoredBlockData{std::move(blockHash), std::move(eventTokens),
                                       page->cacheLevel.value(), page->priority, {}, std::nullopt}}},
            lifeCycleId);
        return;
    }

    std::set<int> lifeCycleIds;
    for (LifeCycleId lifeCycle{0}; lifeCycle < block.storage.size(); ++lifeCycle)
    {
        if (pageCoversBlock(block.getPage(lifeCycle), block))
        {
            lifeCycleIds.insert(lifeCycle.value());
        }
    }
    if (lifeCycleIds.empty())
    {
        return;
    }

    EventBlockHash blockHash = hashFromBlock(block);
    if (!isRoutingBlockSupported(block))
    {
        mSuppressedRoutingBlockKeys.insert(block.key);
        return;
    }
    auto const storedState
        = mStoredBlocks.insert_or_assign(block.key, StoredBlockState{blockHash, std::move(lifeCycleIds)}).first;
    auto parentHash = parentHashFromBlock(block);
    for (int lifeCycleId : storedState->second.lifeCycleIds)
    {
        auto blockData = storedBlockFromBlock(block, lifeCycleId);
        if (blockData.has_value())
        {
            flushRemovedEventUnlocked(lifeCycleId);
            addStoredEventUnlocked(KVCacheStoredData{parentHash, {std::move(*blockData)}}, lifeCycleId);
        }
    }
}

void EventManager::addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(lifeCycle.value()))
    {
        return;
    }
    auto state = mStoredBlocks.find(block.key);
    if (state == mStoredBlocks.end())
    {
        addStoredBlockUnlocked(block);
        return;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        return;
    }
    int const lifeCycleId = lifeCycle.value();
    if (state->second.lifeCycleIds.count(lifeCycleId) != 0)
    {
        return;
    }
    auto blockData = storedBlockFromBlock(block, lifeCycleId);
    if (!blockData.has_value())
    {
        return;
    }
    state->second.lifeCycleIds.insert(lifeCycleId);
    flushRemovedEventUnlocked(lifeCycleId);
    addStoredEventUnlocked(KVCacheStoredData{parentHashFromBlock(block), {std::move(*blockData)}}, lifeCycleId);
}

void EventManager::addRemovedBlock(Digest const& blockKey)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed)
    {
        return;
    }
    auto state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        mSuppressedRoutingBlockKeys.erase(blockKey);
        dropHashCache(blockKey);
        return;
    }
    EventBlockHash blockHash = state->second.blockHash;
    if (mRoutingLayerGroupId.has_value())
    {
        int const lifeCycleId = *mRoutingLayerGroupId;
        mStoredBlocks.erase(state);
        dropHashCache(blockKey);
        enqueueRemovedEventUnlocked({std::move(blockHash)}, lifeCycleId);
        return;
    }
    auto lifeCycleIds = std::move(state->second.lifeCycleIds);
    mStoredBlocks.erase(state);
    dropHashCache(blockKey);

    if (lifeCycleIds.empty())
    {
        enqueueRemovedEventUnlocked({std::move(blockHash)}, std::nullopt);
        return;
    }
    for (int lifeCycleId : lifeCycleIds)
    {
        enqueueRemovedEventUnlocked({blockHash}, lifeCycleId);
    }
}

void EventManager::addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(lifeCycle.value()))
    {
        return;
    }
    auto state = mStoredBlocks.find(blockKey);
    int const lifeCycleId = lifeCycle.value();
    if (state == mStoredBlocks.end())
    {
        if (mSuppressedRoutingBlockKeys.erase(blockKey) != 0)
        {
            dropHashCache(blockKey);
        }
        return;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        EventBlockHash blockHash = std::move(state->second.blockHash);
        mStoredBlocks.erase(state);
        dropHashCache(blockKey);
        enqueueRemovedEventUnlocked({std::move(blockHash)}, lifeCycleId);
        return;
    }
    if (state->second.lifeCycleIds.erase(lifeCycleId) == 0)
    {
        return;
    }
    EventBlockHash blockHash = state->second.blockHash;
    if (state->second.lifeCycleIds.empty())
    {
        mStoredBlocks.erase(state);
        dropHashCache(blockKey);
    }
    enqueueRemovedEventUnlocked({std::move(blockHash)}, lifeCycleId);
}

void EventManager::addCacheLevelUpdated(
    Digest const& blockKey, CacheLevel oldLevel, CacheLevel newLevel, LifeCycleId lifeCycle)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    if (mClosing || mClosed || !acceptsRoutingLayerGroup(lifeCycle.value()))
    {
        return;
    }
    auto state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value() && mPendingRoutingEntries >= mMaxKvEventEntries)
    {
        ++mDroppedEventCount;
        return;
    }
    mPendingRoutingEntries += mRoutingLayerGroupId.has_value() ? 1 : 0;
    addEventUnlocked(
        KVCacheUpdatedData{state->second.blockHash, KVCacheEventDiff{oldLevel.value(), newLevel.value()}, std::nullopt},
        lifeCycle.value());
}

void EventManager::addStoredEventUnlocked(KVCacheStoredData data, EventLayerGroupId layerGroupId)
{
    if (mRoutingLayerGroupId.has_value())
    {
        int const available
            = std::min<int>(kMaxRoutingBlocksPerEvent, std::max(0, mMaxKvEventEntries - mPendingRoutingEntries));
        if (static_cast<int>(data.blocks.size()) > available)
        {
            mDroppedEventCount += static_cast<int64_t>(data.blocks.size()) - available;
            data.blocks.resize(static_cast<std::size_t>(available));
        }
        if (data.blocks.empty())
        {
            return;
        }
        mPendingRoutingEntries += static_cast<int>(data.blocks.size());
    }
    bool const hasPendingRemovedEvents = !mLatestRemovedBlockHashes.empty();
    auto latest = mLatestStoredEventIds.find(layerGroupId);
    if (!hasPendingRemovedEvents && latest != mLatestStoredEventIds.end())
    {
        auto pending = std::find_if(mPendingEvents.rbegin(), mPendingEvents.rend(),
            [&](KVCacheEvent const& event) { return event.eventId == latest->second; });
        if (pending == mPendingEvents.rend())
        {
            throw std::logic_error("Stored event coalescing lost the pending event");
        }
        if (auto* stored = std::get_if<KVCacheStoredData>(&pending->data); stored != nullptr && !stored->blocks.empty()
            && (!mRoutingLayerGroupId.has_value()
                || stored->blocks.size() + data.blocks.size() <= kMaxRoutingBlocksPerEvent)
            && data.parentHash.has_value() && stored->blocks.back().blockHash == *data.parentHash)
        {
            std::move(data.blocks.begin(), data.blocks.end(), std::back_inserter(stored->blocks));
            return;
        }
    }

    auto& event = addEventUnlocked(std::move(data), layerGroupId);
    mLatestStoredEventIds.insert_or_assign(layerGroupId, event.eventId);
}

void EventManager::enqueueRemovedEventUnlocked(std::vector<EventBlockHash> blockHashes, EventLayerGroupId layerGroupId)
{
    if (blockHashes.empty())
    {
        return;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        int const available = std::max(0, mMaxKvEventEntries - mPendingRoutingEntries);
        if (static_cast<int>(blockHashes.size()) > available)
        {
            mDroppedEventCount += static_cast<int64_t>(blockHashes.size()) - available;
            blockHashes.resize(static_cast<std::size_t>(available));
        }
        if (blockHashes.empty())
        {
            return;
        }
        mPendingRoutingEntries += static_cast<int>(blockHashes.size());
    }
    auto& pending = mLatestRemovedBlockHashes[layerGroupId];
    std::move(blockHashes.begin(), blockHashes.end(), std::back_inserter(pending));
    mLatestStoredEventIds.erase(layerGroupId);
}

void EventManager::flushRemovedEventUnlocked(EventLayerGroupId layerGroupId)
{
    auto removed = mLatestRemovedBlockHashes.find(layerGroupId);
    if (removed == mLatestRemovedBlockHashes.end() || removed->second.empty())
    {
        return;
    }
    auto blockHashes = std::move(removed->second);
    mLatestRemovedBlockHashes.erase(removed);
    addEventUnlocked(KVCacheRemovedData{std::move(blockHashes)}, layerGroupId);
}

void EventManager::flushAllRemovedEventsUnlocked()
{
    while (!mLatestRemovedBlockHashes.empty())
    {
        flushRemovedEventUnlocked(mLatestRemovedBlockHashes.begin()->first);
    }
}

KVCacheEvent& EventManager::addEventUnlocked(KVCacheEventData data, EventLayerGroupId layerGroupId)
{
    if (mMaxKvEventEntries <= 0)
    {
        throw std::logic_error("Cannot add an event when the event queue is disabled");
    }
    if (!std::holds_alternative<KVCacheRemovedData>(data))
    {
        flushAllRemovedEventsUnlocked();
    }
    mPendingEvents.push_back(KVCacheEvent{
        mNextEventId++, std::move(data), getWindowSize(layerGroupId), mHashAlgoName, mAttentionDpRank, layerGroupId});
    if (!std::holds_alternative<KVCacheStoredData>(mPendingEvents.back().data))
    {
        mLatestStoredEventIds.erase(layerGroupId);
    }
    return mPendingEvents.back();
}

std::vector<KVCacheEvent> EventManager::drainPendingEventsUnlocked()
{
    flushAllRemovedEventsUnlocked();
    auto events = std::move(mPendingEvents);
    mPendingEvents.clear();
    mLatestStoredEventIds.clear();
    mPendingRoutingEntries = 0;
    return events;
}

bool EventManager::publishEventsUnlocked(std::vector<KVCacheEvent> events, std::optional<int> maxKvEventEntries)
{
    if (events.empty() || mClosed)
    {
        return false;
    }
    int const capacity = maxKvEventEntries.value_or(mMaxKvEventEntries);
    if (capacity <= 0)
    {
        mDroppedEventCount += static_cast<int64_t>(events.size());
        return false;
    }
    if (mRoutingLayerGroupId.has_value())
    {
        int incomingEntries = 0;
        for (auto const& event : events)
        {
            incomingEntries += routingEventWeight(event);
        }
        while (!events.empty() && incomingEntries > capacity)
        {
            int const dropped = routingEventWeight(events.front());
            incomingEntries -= dropped;
            mDroppedEventCount += dropped;
            events.erase(events.begin());
        }
        while (!mEvents.empty() && mQueuedRoutingEntries + incomingEntries > capacity)
        {
            int const dropped = routingEventWeight(mEvents.front());
            mQueuedRoutingEntries -= dropped;
            mDroppedEventCount += dropped;
            mEvents.pop_front();
        }
        if (events.empty())
        {
            return false;
        }
        std::move(events.begin(), events.end(), std::back_inserter(mEvents));
        mQueuedRoutingEntries += incomingEntries;
        mQueueHighWatermark = std::max(mQueueHighWatermark, mQueuedRoutingEntries);
        return true;
    }
    auto const keepIncoming = std::min<std::size_t>(events.size(), static_cast<std::size_t>(capacity));
    auto const maxExisting = static_cast<std::size_t>(capacity) - keepIncoming;
    auto const droppedExisting = mEvents.size() > maxExisting ? mEvents.size() - maxExisting : 0;
    while (mEvents.size() > maxExisting)
    {
        mEvents.pop_front();
    }
    mDroppedEventCount += static_cast<int64_t>(droppedExisting + events.size() - keepIncoming);
    auto first = events.end() - static_cast<std::ptrdiff_t>(keepIncoming);
    std::move(first, events.end(), std::back_inserter(mEvents));
    mQueueHighWatermark = std::max(mQueueHighWatermark, static_cast<int>(mEvents.size()));
    return keepIncoming > 0;
}

std::vector<KVCacheEvent> EventManager::trimEvents(std::vector<KVCacheEvent> events, int maxKvEventEntries)
{
    if (maxKvEventEntries <= 0)
    {
        return {};
    }
    if (static_cast<int>(events.size()) > maxKvEventEntries)
    {
        events.erase(events.begin(), events.end() - maxKvEventEntries);
    }
    return events;
}

void EventManager::flushIterationEvents()
{
    if (mAttentionDpGather)
    {
        std::vector<KVCacheEvent> localEvents;
        int64_t trimmedEventCount = 0;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mClosing || mClosed)
            {
                return;
            }
            ++mActiveGatherCount;
            auto pendingEvents = drainPendingEventsUnlocked();
            trimmedEventCount += std::max<int64_t>(0, static_cast<int64_t>(pendingEvents.size()) - mMaxKvEventEntries);
            localEvents = trimEvents(std::move(pendingEvents), mMaxKvEventEntries);
        }
        std::vector<std::vector<KVCacheEvent>> gatheredEvents;
        try
        {
            gatheredEvents = mAttentionDpGather(localEvents);
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> lock(mMutex);
                --mActiveGatherCount;
            }
            mCondition.notify_all();
            throw;
        }

        std::vector<KVCacheEvent> events;
        if (mAttentionDpRank == std::optional<int>{0})
        {
            for (auto& rankEvents : gatheredEvents)
            {
                trimmedEventCount += std::max<int64_t>(0, static_cast<int64_t>(rankEvents.size()) - mMaxKvEventEntries);
                auto trimmed = trimEvents(std::move(rankEvents), mMaxKvEventEntries);
                std::move(trimmed.begin(), trimmed.end(), std::back_inserter(events));
            }
        }
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mDroppedEventCount += trimmedEventCount;
            if (!mClosed && mAttentionDpRank == std::optional<int>{0})
            {
                publishEventsUnlocked(std::move(events), mMaxKvEventEntries * std::max<int>(1, gatheredEvents.size()));
            }
            --mActiveGatherCount;
        }
        // Also wakes close(), which may be waiting for this gather to finish.
        mCondition.notify_all();
        return;
    }

    bool published = false;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mClosing && !mClosed)
        {
            published = publishEventsUnlocked(drainPendingEventsUnlocked());
        }
    }
    if (published)
    {
        mCondition.notify_all();
    }
}

std::vector<KVCacheEvent> EventManager::getLatestEvents(std::optional<double> timeoutMs, std::optional<int> maxEvents)
{
    std::unique_lock<std::mutex> lock(mMutex);
    if (mEvents.empty() && !timeoutMs.has_value())
    {
        mCondition.wait(lock, [&] { return mClosed || !mEvents.empty(); });
    }
    else if (mEvents.empty() && *timeoutMs > 0)
    {
        mCondition.wait_for(
            lock, std::chrono::duration<double, std::milli>(*timeoutMs), [&] { return mClosed || !mEvents.empty(); });
    }
    std::size_t count = mEvents.size();
    if (maxEvents.has_value())
    {
        count = std::min(count, static_cast<std::size_t>(std::max(0, *maxEvents)));
    }
    std::deque<KVCacheEvent> queued;
    for (std::size_t index = 0; index < count; ++index)
    {
        if (mRoutingLayerGroupId.has_value())
        {
            mQueuedRoutingEntries -= routingEventWeight(mEvents.front());
        }
        queued.push_back(std::move(mEvents.front()));
        mEvents.pop_front();
    }
    lock.unlock();
    std::vector<KVCacheEvent> events;
    events.reserve(queued.size());
    std::move(queued.begin(), queued.end(), std::back_inserter(events));
    return events;
}

int64_t EventManager::discardEvents()
{
    std::lock_guard<std::mutex> lock(mMutex);
    mPendingEvents.clear();
    mLatestStoredEventIds.clear();
    mLatestRemovedBlockHashes.clear();
    mPendingRoutingEntries = 0;
    mEvents.clear();
    mQueuedRoutingEntries = 0;
    return mDroppedEventCount;
}

void EventManager::close()
{
    std::unique_lock<std::mutex> lock(mMutex);
    if (mClosed)
    {
        return;
    }
    if (mClosing)
    {
        mCondition.wait(lock, [&] { return mClosed; });
        return;
    }
    mClosing = true;
    mCondition.wait(lock, [&] { return mActiveGatherCount == 0; });
    auto pendingEvents = drainPendingEventsUnlocked();
    if (mAttentionDpGather && mAttentionDpRank != std::optional<int>{0})
    {
        mDroppedEventCount += static_cast<int64_t>(pendingEvents.size());
    }
    else
    {
        publishEventsUnlocked(std::move(pendingEvents));
    }
    mClosed = true;
    mClosing = false;
    lock.unlock();
    mCondition.notify_all();
}

bool EventManager::acceptsRoutingLayerGroup(EventLayerGroupId layerGroupId) const
{
    return !mRoutingLayerGroupId.has_value() || layerGroupId == mRoutingLayerGroupId;
}

bool EventManager::isRoutingBlockSupported(Block const& block) const
{
    if (!mRoutingLayerGroupId.has_value())
    {
        return true;
    }
    auto const state = mV1HashStates.find(block.key);
    return mHashAlgo == HashAlgorithm::kV1 && state != mV1HashStates.end() && state->second.compatible
        && !state->second.rootAttrs.first.has_value() && !state->second.rootAttrs.second.has_value();
}

int64_t EventManager::getDroppedEventCount() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mDroppedEventCount;
}

int EventManager::getQueueHighWatermark() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mQueueHighWatermark;
}

bool EventManager::isClosedAndEmpty() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mClosed && mEvents.empty();
}

int EventManager::getWindowSize(EventLayerGroupId layerGroupId) const
{
    if (!layerGroupId.has_value())
    {
        return mWindowSize;
    }
    auto const windowSize = mWindowSizeByLayerGroup.find(*layerGroupId);
    return windowSize == mWindowSizeByLayerGroup.end() ? mWindowSize : windowSize->second;
}

int EventManager::routingEventWeight(KVCacheEvent const& event) const
{
    if (auto const* stored = std::get_if<KVCacheStoredData>(&event.data))
    {
        return std::max(1, static_cast<int>(stored->blocks.size()));
    }
    if (auto const* removed = std::get_if<KVCacheRemovedData>(&event.data))
    {
        return std::max(1, static_cast<int>(removed->blockHashes.size()));
    }
    return 1;
}

std::string EventManager::digestToHex(Digest const& digest)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.resize(digest.size() * 2);
    for (size_t i = 0; i < digest.size(); ++i)
    {
        auto const value = std::to_integer<uint8_t>(digest[i]);
        result[2 * i] = kHex[value >> 4U];
        result[2 * i + 1] = kHex[value & 0x0FU];
    }
    return result;
}

uint64_t EventManager::truncateDigestToInt64(Digest const& digest)
{
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i)
    {
        result = (result << 8U) | std::to_integer<uint8_t>(digest[static_cast<size_t>(i)]);
    }
    return result;
}

EventBlockHash EventManager::normalizeDigest(Digest const& digest) const
{
    if (mHashAlgo == HashAlgorithm::kV2Sha256_64)
    {
        return truncateDigestToInt64(digest);
    }
    return digestToHex(digest);
}

EventBlockHash EventManager::hashFromBlock(Block const& block)
{
    if (mHashAlgo == HashAlgorithm::kV1)
    {
        return v1HashFromBlock(block);
    }
    return normalizeDigest(block.key);
}

std::optional<EventBlockHash> EventManager::parentHashFromBlock(Block const& block)
{
    if (block.prev == nullptr)
    {
        throw std::logic_error("Cannot hash an orphan KV cache block");
    }
    if (block.prev->type() == NodeBase::Type::kROOT_BLOCK)
    {
        return std::nullopt;
    }
    return hashFromBlock(*static_cast<Block const*>(block.prev));
}

std::optional<KVCacheStoredBlockData> EventManager::storedBlockFromBlock(
    Block const& block, std::optional<int> lifeCycleId)
{
    CacheLevel cacheLevel = kHotLevel;
    Priority priority = kPriorityDefault;
    bool foundPage = false;
    for (LifeCycleId lifeCycle{0}; lifeCycle < block.storage.size(); ++lifeCycle)
    {
        if (lifeCycleId.has_value() && lifeCycle.value() != *lifeCycleId)
        {
            continue;
        }
        auto const* page = block.getPage(lifeCycle);
        if (pageCoversBlock(page, block))
        {
            cacheLevel = page->cacheLevel;
            priority = page->priority;
            foundPage = true;
            break;
        }
    }
    if (lifeCycleId.has_value() && !foundPage)
    {
        return std::nullopt;
    }

    std::vector<UniqueToken> tokens;
    tokens.reserve(block.tokens.size());
    for (auto const& token : block.tokens)
    {
        if (!token.isDigest())
        {
            UniqueToken uniqueToken;
            uniqueToken.tokenId = EventTokenId{std::in_place_index<0>, token.tokenId()};
            tokens.push_back(std::move(uniqueToken));
        }
        else
        {
            UniqueToken uniqueToken;
            uniqueToken.tokenId = EventTokenId{std::in_place_index<1>, digestToHex(token.digest())};
            tokens.push_back(std::move(uniqueToken));
        }
    }
    return KVCacheStoredBlockData{
        hashFromBlock(block), std::move(tokens), cacheLevel.value(), priority, {}, std::nullopt};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
