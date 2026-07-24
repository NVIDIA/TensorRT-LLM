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
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

constexpr uint32_t kUint32HashConst = 0x045D9F3BU;
constexpr uint64_t kUint64HashConst1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kUint64HashConst2 = 0x94D049BB133111EBULL;
constexpr uint32_t kHashCombineConst = 0x9E3779B9U;
constexpr uint64_t kParentHashConst = 0xBF58476D1CE4E5B9ULL;

uint64_t hash32Mix(int64_t input, uint64_t seed)
{
    uint32_t value = static_cast<uint32_t>(input);
    value = ((value >> 16U) ^ value) * kUint32HashConst;
    value = ((value >> 16U) ^ value) * kUint32HashConst;
    value = (value >> 16U) ^ value;
    value += kHashCombineConst;
    return seed ^ (static_cast<uint64_t>(value) + (seed << 6U) + (seed >> 2U));
}

uint64_t hash64Mix(int64_t input, uint64_t seed)
{
    uint64_t value = static_cast<uint64_t>(input);
    value = (value ^ (value >> 30U)) * kUint64HashConst1;
    value = (value ^ (value >> 27U)) * kUint64HashConst2;
    value ^= value >> 31U;
    return seed ^ (value + static_cast<uint64_t>(kHashCombineConst) + (seed << 6U) + (seed >> 2U));
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
    KVCacheCreatedData data{std::move(numBlocksPerCacheLevel)};
    if (!layerGroupIds.has_value())
    {
        addEventUnlocked(std::move(data), std::nullopt);
        return;
    }
    for (int layerGroupId : *layerGroupIds)
    {
        addEventUnlocked(data, layerGroupId);
    }
}

void EventManager::setLayerGroupWindowSizes(std::map<int, int> windowSizes)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mWindowSizeByLayerGroup = std::move(windowSizes);
}

void EventManager::addStoredEvent(KVCacheStoredData data, EventLayerGroupId layerGroupId)
{
    if (data.blocks.empty() || mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
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
    auto const state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        return;
    }
    addEventUnlocked(KVCacheUpdatedData{state->second.blockHash, cacheLevel, priority}, layerGroupId);
}

void EventManager::addStoredBlock(Block const& block)
{
    if (mMaxKvEventEntries <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    addStoredBlockUnlocked(block);
}

void EventManager::addStoredBlockUnlocked(Block const& block)
{
    std::set<int> lifeCycleIds;
    for (LifeCycleId lifeCycle{0}; lifeCycle < block.storage.size(); ++lifeCycle)
    {
        if (block.storage[lifeCycle] != nullptr)
        {
            lifeCycleIds.insert(lifeCycle.value());
        }
    }
    if (lifeCycleIds.empty())
    {
        return;
    }

    EventBlockHash blockHash = hashFromBlock(block);
    mStoredBlocks.insert_or_assign(block.key, StoredBlockState{blockHash, lifeCycleIds});
    auto parentHash = parentHashFromBlock(block);
    for (int lifeCycleId : lifeCycleIds)
    {
        auto blockData = storedBlockFromBlock(block, std::set<int>{lifeCycleId});
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
    auto state = mStoredBlocks.find(block.key);
    if (state == mStoredBlocks.end())
    {
        addStoredBlockUnlocked(block);
        return;
    }
    int const lifeCycleId = lifeCycle.value();
    if (state->second.lifeCycleIds.count(lifeCycleId) != 0)
    {
        return;
    }
    auto blockData = storedBlockFromBlock(block, std::set<int>{lifeCycleId});
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
    auto state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        return;
    }
    EventBlockHash blockHash = state->second.blockHash;
    auto lifeCycleIds = state->second.lifeCycleIds;
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
    auto state = mStoredBlocks.find(blockKey);
    int const lifeCycleId = lifeCycle.value();
    if (state == mStoredBlocks.end() || state->second.lifeCycleIds.erase(lifeCycleId) == 0)
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
    auto state = mStoredBlocks.find(blockKey);
    if (state == mStoredBlocks.end())
    {
        return;
    }
    addEventUnlocked(
        KVCacheUpdatedData{state->second.blockHash, KVCacheEventDiff{oldLevel.value(), newLevel.value()}, std::nullopt},
        lifeCycle.value());
}

void EventManager::addStoredEventUnlocked(KVCacheStoredData data, EventLayerGroupId layerGroupId)
{
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
    return events;
}

void EventManager::publishEventsUnlocked(std::vector<KVCacheEvent> events, std::optional<int> maxKvEventEntries)
{
    if (events.empty())
    {
        return;
    }
    int const capacity = maxKvEventEntries.value_or(mMaxKvEventEntries);
    std::move(events.begin(), events.end(), std::back_inserter(mEvents));
    while (static_cast<int>(mEvents.size()) > capacity)
    {
        mEvents.pop_front();
    }
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
        {
            std::lock_guard<std::mutex> lock(mMutex);
            localEvents = trimEvents(drainPendingEventsUnlocked(), mMaxKvEventEntries);
        }
        auto gatheredEvents = mAttentionDpGather(localEvents);
        if (mAttentionDpRank != std::optional<int>{0})
        {
            return;
        }

        std::vector<KVCacheEvent> events;
        for (auto& rankEvents : gatheredEvents)
        {
            auto trimmed = trimEvents(std::move(rankEvents), mMaxKvEventEntries);
            std::move(trimmed.begin(), trimmed.end(), std::back_inserter(events));
        }
        {
            std::lock_guard<std::mutex> lock(mMutex);
            publishEventsUnlocked(std::move(events), mMaxKvEventEntries * std::max<int>(1, gatheredEvents.size()));
        }
        mCondition.notify_all();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mMutex);
        publishEventsUnlocked(drainPendingEventsUnlocked());
    }
    mCondition.notify_all();
}

std::vector<KVCacheEvent> EventManager::getLatestEvents(std::optional<double> timeoutMs)
{
    std::unique_lock<std::mutex> lock(mMutex);
    if (mEvents.empty() && !timeoutMs.has_value())
    {
        mCondition.wait(lock, [&] { return !mEvents.empty(); });
    }
    else if (mEvents.empty() && *timeoutMs > 0)
    {
        mCondition.wait_for(
            lock, std::chrono::duration<double, std::milli>(*timeoutMs), [&] { return !mEvents.empty(); });
    }
    std::vector<KVCacheEvent> events;
    events.reserve(mEvents.size());
    std::move(mEvents.begin(), mEvents.end(), std::back_inserter(events));
    mEvents.clear();
    return events;
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
    Block const& block, std::optional<std::set<int>> const& lifeCycleIds)
{
    CacheLevel cacheLevel = kGpuLevel;
    Priority priority = kPriorityDefault;
    bool foundPage = false;
    for (LifeCycleId lifeCycle{0}; lifeCycle < block.storage.size(); ++lifeCycle)
    {
        if (lifeCycleIds.has_value() && lifeCycleIds->count(lifeCycle.value()) == 0)
        {
            continue;
        }
        auto const* page = block.storage[lifeCycle];
        if (page != nullptr)
        {
            cacheLevel = page->cacheLevel;
            priority = page->priority;
            foundPage = true;
            break;
        }
    }
    if (lifeCycleIds.has_value() && !foundPage)
    {
        return std::nullopt;
    }

    std::vector<UniqueToken> tokens;
    tokens.reserve(block.tokens.size());
    for (auto const& token : block.tokens)
    {
        if (auto const* tokenId = std::get_if<TokenId>(&token))
        {
            UniqueToken uniqueToken;
            uniqueToken.tokenId = EventTokenId{std::in_place_index<0>, *tokenId};
            tokens.push_back(std::move(uniqueToken));
        }
        else
        {
            UniqueToken uniqueToken;
            uniqueToken.tokenId
                = EventTokenId{std::in_place_index<1>, digestToHex(std::get<DigestToken>(token).digest())};
            tokens.push_back(std::move(uniqueToken));
        }
    }
    return KVCacheStoredBlockData{
        hashFromBlock(block), std::move(tokens), cacheLevel.value(), priority, {}, std::nullopt};
}

uint64_t EventManager::hashV1BlockKey(std::vector<TokenId> const& tokens, uint64_t parentHash,
    std::optional<LoraTaskIdType> loraTaskId, std::optional<std::uint64_t> cacheSaltId)
{
    uint64_t seed = static_cast<uint64_t>(tokens.size()) ^ (parentHash * kParentHashConst);
    if (parentHash == 0 && cacheSaltId.has_value())
    {
        seed = hash64Mix(*cacheSaltId, seed);
    }
    for (TokenId token : tokens)
    {
        seed = hash32Mix(token, seed);
    }
    if (loraTaskId.has_value())
    {
        seed = hash64Mix(*loraTaskId, seed);
    }
    return seed;
}

uint64_t EventManager::v1HashFromBlock(Block const& block)
{
    if (auto const cached = mV1HashByBlockKey.find(block.key); cached != mV1HashByBlockKey.end())
    {
        return cached->second;
    }

    std::vector<Block const*> chain;
    NodeBase const* current = &block;
    uint64_t parentHash = 0;
    bool parentIsV1Compatible = true;
    V1RootAttrs rootAttrs;
    while (current->type() == NodeBase::Type::kBLOCK)
    {
        auto const* currentBlock = static_cast<Block const*>(current);
        if (auto const cached = mV1HashByBlockKey.find(currentBlock->key); cached != mV1HashByBlockKey.end())
        {
            parentHash = cached->second;
            parentIsV1Compatible = mV1HashCompatibleKeys.count(currentBlock->key) != 0;
            rootAttrs = mV1RootAttrsByBlockKey.at(currentBlock->key);
            break;
        }
        chain.push_back(currentBlock);
        current = currentBlock->prev;
        if (current == nullptr)
        {
            throw std::logic_error("Cannot hash an orphan KV cache block");
        }
    }
    if (current->type() == NodeBase::Type::kROOT_BLOCK)
    {
        auto const& reuseScope = static_cast<RootBlock const*>(current)->reuseScope;
        rootAttrs = {reuseScope.loraId, reuseScope.salt};
    }

    for (auto chainIter = chain.rbegin(); chainIter != chain.rend(); ++chainIter)
    {
        Block const& currentBlock = **chainIter;
        std::vector<TokenId> textTokens;
        textTokens.reserve(currentBlock.tokens.size());
        if (parentIsV1Compatible)
        {
            for (auto const& token : currentBlock.tokens)
            {
                auto const* tokenId = std::get_if<TokenId>(&token);
                if (tokenId == nullptr)
                {
                    parentIsV1Compatible = false;
                    break;
                }
                textTokens.push_back(*tokenId);
            }
        }
        if (parentIsV1Compatible)
        {
            parentHash = hashV1BlockKey(textTokens, parentHash, rootAttrs.first, rootAttrs.second);
            mV1HashCompatibleKeys.insert(currentBlock.key);
        }
        else
        {
            parentHash = fallbackV1Hash(currentBlock.key);
        }
        mV1HashByBlockKey.insert_or_assign(currentBlock.key, parentHash);
        mV1RootAttrsByBlockKey.insert_or_assign(currentBlock.key, rootAttrs);
    }
    return parentHash;
}

uint64_t EventManager::fallbackV1Hash(Digest const& blockKey)
{
    if (!mWarnedV1HashFallback)
    {
        TLLM_LOG_WARNING(
            "V2 KV cache event hash algorithm v1_block_key only matches v1 for text-token radix blocks. "
            "Falling back to truncated SHA-256 block hash for unsupported blocks.");
        mWarnedV1HashFallback = true;
    }
    return truncateDigestToInt64(blockKey);
}

void EventManager::dropHashCache(Digest const& blockKey)
{
    mV1HashByBlockKey.erase(blockKey);
    mV1HashCompatibleKeys.erase(blockKey);
    mV1RootAttrsByBlockKey.erase(blockKey);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
