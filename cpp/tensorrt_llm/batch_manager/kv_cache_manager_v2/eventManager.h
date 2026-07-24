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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/eventSink.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

using EventBlockHash = std::variant<uint64_t, std::string>;
using EventTokenId = std::variant<int64_t, std::string>;
using EventLayerGroupId = std::optional<int>;

struct UniqueToken
{
    EventTokenId tokenId;
    int64_t tokenExtraId = 0;

    bool operator==(UniqueToken const& other) const
    {
        return tokenId == other.tokenId && tokenExtraId == other.tokenExtraId;
    }
};

struct KVCacheCreatedData
{
    std::vector<int> numBlocksPerCacheLevel;

    bool operator==(KVCacheCreatedData const& other) const
    {
        return numBlocksPerCacheLevel == other.numBlocksPerCacheLevel;
    }
};

struct MmKey
{
    std::string hash;
    int startOffset = 0;
    std::optional<std::string> uuid;
    bool hasUuidField = false;

    bool operator==(MmKey const& other) const
    {
        return hash == other.hash && startOffset == other.startOffset && uuid == other.uuid
            && hasUuidField == other.hasUuidField;
    }
};

struct KVCacheStoredBlockData
{
    EventBlockHash blockHash;
    std::vector<UniqueToken> tokens;
    int cacheLevel = kGpuLevel.value();
    int priority = kPriorityDefault;
    std::vector<MmKey> mmKeys;
    std::optional<std::string> cacheSalt;

    bool operator==(KVCacheStoredBlockData const& other) const
    {
        return blockHash == other.blockHash && tokens == other.tokens && cacheLevel == other.cacheLevel
            && priority == other.priority && mmKeys == other.mmKeys && cacheSalt == other.cacheSalt;
    }
};

struct KVCacheStoredData
{
    std::optional<EventBlockHash> parentHash;
    std::vector<KVCacheStoredBlockData> blocks;

    bool operator==(KVCacheStoredData const& other) const
    {
        return parentHash == other.parentHash && blocks == other.blocks;
    }
};

struct KVCacheRemovedData
{
    std::vector<EventBlockHash> blockHashes;

    bool operator==(KVCacheRemovedData const& other) const
    {
        return blockHashes == other.blockHashes;
    }
};

struct KVCacheEventDiff
{
    int oldValue = 0;
    int newValue = 0;

    bool operator==(KVCacheEventDiff const& other) const
    {
        return oldValue == other.oldValue && newValue == other.newValue;
    }
};

struct KVCacheUpdatedData
{
    EventBlockHash blockHash;
    std::optional<KVCacheEventDiff> cacheLevel;
    std::optional<KVCacheEventDiff> priority;

    bool operator==(KVCacheUpdatedData const& other) const
    {
        return blockHash == other.blockHash && cacheLevel == other.cacheLevel && priority == other.priority;
    }
};

using KVCacheEventData = std::variant<KVCacheCreatedData, KVCacheStoredData, KVCacheRemovedData, KVCacheUpdatedData>;

struct KVCacheEvent
{
    int64_t eventId = 0;
    KVCacheEventData data;
    int windowSize = 0;
    std::optional<std::string> hashAlgo;
    std::optional<int> attentionDpRank;
    EventLayerGroupId layerGroupId;

    bool operator==(KVCacheEvent const& other) const
    {
        return eventId == other.eventId && data == other.data && windowSize == other.windowSize
            && hashAlgo == other.hashAlgo && attentionDpRank == other.attentionDpRank
            && layerGroupId == other.layerGroupId;
    }
};

class EventManager final : public EventSink
{
public:
    using AttentionDpGatherFn = std::function<std::vector<std::vector<KVCacheEvent>>(std::vector<KVCacheEvent> const&)>;

    EventManager(int maxKvEventEntries, int windowSize = 0, std::optional<int> attentionDpRank = std::nullopt,
        AttentionDpGatherFn attentionDpGather = {}, std::string hashAlgo = "v2_sha256",
        std::map<int, int> windowSizeByLayerGroup = {});

    void addCreatedEvent(
        std::vector<int> numBlocksPerCacheLevel, std::optional<std::vector<int>> layerGroupIds = std::nullopt);
    void setLayerGroupWindowSizes(std::map<int, int> windowSizes);
    void addStoredEvent(KVCacheStoredData data, EventLayerGroupId layerGroupId = std::nullopt);
    void addRemovedEvent(std::vector<EventBlockHash> blockHashes, EventLayerGroupId layerGroupId = std::nullopt);
    void addUpdatedEvent(EventBlockHash blockHash, std::optional<KVCacheEventDiff> cacheLevel = std::nullopt,
        std::optional<KVCacheEventDiff> priority = std::nullopt, EventLayerGroupId layerGroupId = std::nullopt);
    void addUpdatedEvent(Digest const& blockKey, std::optional<KVCacheEventDiff> cacheLevel = std::nullopt,
        std::optional<KVCacheEventDiff> priority = std::nullopt, EventLayerGroupId layerGroupId = std::nullopt);

    void flushIterationEvents();
    std::vector<KVCacheEvent> getLatestEvents(std::optional<double> timeoutMs = std::nullopt);

    std::string const& hashAlgorithm() const noexcept
    {
        return mHashAlgoName;
    }

    static uint64_t hashV1BlockKey(std::vector<TokenId> const& tokens, uint64_t parentHash = 0,
        std::optional<LoraTaskIdType> loraTaskId = std::nullopt,
        std::optional<std::uint64_t> cacheSaltId = std::nullopt);

    void addStoredBlock(Block const& block) override;
    void addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle) override;
    void addRemovedBlock(Digest const& blockKey) override;
    void addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle) override;
    void addCacheLevelUpdated(
        Digest const& blockKey, CacheLevel oldLevel, CacheLevel newLevel, LifeCycleId lifeCycle) override;

private:
    enum class HashAlgorithm
    {
        kV1,
        kV2Sha256,
        kV2Sha256_64,
    };

    struct StoredBlockState
    {
        EventBlockHash blockHash;
        std::set<int> lifeCycleIds;
    };

    using V1RootAttrs = std::pair<std::optional<LoraTaskIdType>, std::optional<std::uint64_t>>;

    static std::pair<HashAlgorithm, std::string> parseHashAlgorithm(std::string const& hashAlgo);
    static std::string digestToHex(Digest const& digest);
    static uint64_t truncateDigestToInt64(Digest const& digest);
    static std::vector<KVCacheEvent> trimEvents(std::vector<KVCacheEvent> events, int maxKvEventEntries);

    EventBlockHash normalizeDigest(Digest const& digest) const;
    EventBlockHash hashFromBlock(Block const& block);
    uint64_t v1HashFromBlock(Block const& block);
    uint64_t fallbackV1Hash(Digest const& blockKey);
    std::optional<EventBlockHash> parentHashFromBlock(Block const& block);
    std::optional<KVCacheStoredBlockData> storedBlockFromBlock(
        Block const& block, std::optional<std::set<int>> const& lifeCycleIds = std::nullopt);

    void addStoredBlockUnlocked(Block const& block);
    void addStoredEventUnlocked(KVCacheStoredData data, EventLayerGroupId layerGroupId);
    void enqueueRemovedEventUnlocked(std::vector<EventBlockHash> blockHashes, EventLayerGroupId layerGroupId);
    void flushRemovedEventUnlocked(EventLayerGroupId layerGroupId);
    void flushAllRemovedEventsUnlocked();
    KVCacheEvent& addEventUnlocked(KVCacheEventData data, EventLayerGroupId layerGroupId);
    std::vector<KVCacheEvent> drainPendingEventsUnlocked();
    void publishEventsUnlocked(std::vector<KVCacheEvent> events, std::optional<int> maxKvEventEntries = std::nullopt);
    int getWindowSize(EventLayerGroupId layerGroupId) const;
    void dropHashCache(Digest const& blockKey);

    int mMaxKvEventEntries;
    int mWindowSize;
    std::map<int, int> mWindowSizeByLayerGroup;
    std::optional<int> mAttentionDpRank;
    AttentionDpGatherFn mAttentionDpGather;
    HashAlgorithm mHashAlgo;
    std::string mHashAlgoName;
    int64_t mNextEventId = 0;

    std::unordered_map<Digest, StoredBlockState> mStoredBlocks;
    std::map<EventLayerGroupId, int64_t> mLatestStoredEventIds;
    std::map<EventLayerGroupId, std::vector<EventBlockHash>> mLatestRemovedBlockHashes;
    std::vector<KVCacheEvent> mPendingEvents;
    std::deque<KVCacheEvent> mEvents;

    std::unordered_map<Digest, uint64_t> mV1HashByBlockKey;
    std::unordered_set<Digest> mV1HashCompatibleKeys;
    std::unordered_map<Digest, V1RootAttrs> mV1RootAttrsByBlockKey;
    bool mWarnedV1HashFallback = false;

    mutable std::mutex mMutex;
    std::condition_variable mCondition;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
