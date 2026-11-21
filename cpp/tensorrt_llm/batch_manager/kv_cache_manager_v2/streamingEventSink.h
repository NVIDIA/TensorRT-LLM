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

#include "kv_cache_manager_v2/eventData.h"
#include "kv_cache_manager_v2/eventSink.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Semantic data for one wire-level BlockStored event.
struct StreamingBlockStoredData
{
    std::vector<int64_t> blockHashes;
    std::optional<int64_t> parentBlockHash;
    std::vector<EventTokenId> tokenIds;
    //! One entry per blockHash. Empty block entries represent text-only blocks.
    std::vector<std::vector<MmKey>> mmKeys;
    std::optional<LoraTaskIdType> loraId;
    //! Set for recurrent-state checkpoints: the group index (1-based over the declared recurrent
    //! life cycles) and the kind "mamba". Attention events leave both unset.
    std::optional<int> groupIdx;
    std::optional<std::string> kvCacheSpecKind;
};

//! Semantic data for one wire-level BlockRemoved event.
struct StreamingBlockRemovedData
{
    std::vector<int64_t> blockHashes;
    //! Set for recurrent-state checkpoint removals, as on StreamingBlockStoredData.
    std::optional<int> groupIdx;
    std::optional<std::string> kvCacheSpecKind;
};

//! Semantic data for one wire-level AllBlocksCleared event.
struct StreamingAllBlocksClearedData
{
};

using StreamingEventData
    = std::variant<StreamingBlockStoredData, StreamingBlockRemovedData, StreamingAllBlocksClearedData>;

//! Counters accumulated over the lifetime of a streaming event sink.
struct StreamingEventStats
{
    int64_t storedBlocks = 0;
    int64_t removedBlocks = 0;
    int64_t partialBlocksSuppressed = 0;
    int64_t nonTargetLifeCyclesIgnored = 0;
    int64_t droppedEvents = 0;
    //! Recurrent-state checkpoints announced and withdrawn, when checkpoint publishing is on.
    int64_t recurrentCheckpointsStored = 0;
    int64_t recurrentCheckpointsRemoved = 0;
};

//! Captures streaming KV-cache lifecycle events without depending on Python or a transport.
class StreamingEventSink final : public EventSink
{
public:
    StreamingEventSink(int maxEntries, std::optional<int> mmTokenIdOffset = std::nullopt);

    bool needsTokenDigestContext() const override
    {
        return mMmTokenIdOffset.has_value();
    }

    void setTargetLifeCycle(LifeCycleId lifeCycle);
    //! Publish the recurrent-state (SSM) snapshots of these life cycles as tagged checkpoint events
    //! (`groupIdx` = 1 + position in `lifeCycles`, `kvCacheSpecKind` = "mamba"). A checkpoint is
    //! published for a block only after the block's attention page was published, under the same
    //! block hash, so consumers can align it with the attention prefix. Empty disables publishing.
    void setRecurrentLifeCycles(std::vector<LifeCycleId> lifeCycles);
    [[nodiscard]] std::vector<StreamingEventData> drainIterationEvents();
    [[nodiscard]] StreamingEventStats getStats() const;

    void addStoredBlock(Block const& block) override;
    void addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle) override;
    void addRemovedBlock(Digest const& blockKey) override;
    void addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle) override;
    void addAllBlocksCleared() override;
    void addCacheLevelUpdated(
        Digest const& blockKey, CacheLevel oldLevel, CacheLevel newLevel, LifeCycleId lifeCycle) override;

private:
    struct StoredBlock
    {
        int64_t blockHash;
        std::optional<LoraTaskIdType> loraId;
        //! Recurrent group indices whose checkpoint was published for this block.
        std::vector<int> recurrentGroups;
    };

    void addStoredBlockUnlocked(Block const& block);
    void addRecurrentCheckpointUnlocked(Block const& block, LifeCycleId lifeCycle, int groupIdx);
    void addRemovedBlockUnlocked(Digest const& blockKey);
    void addRemovedHashUnlocked(int64_t blockHash, std::optional<int> groupIdx = std::nullopt);
    [[nodiscard]] std::optional<int> recurrentGroupIdx(LifeCycleId lifeCycle) const;
    void recordDroppedEventUnlocked(char const* reason);
    [[nodiscard]] bool reserveEntryUnlocked();
    [[nodiscard]] static int64_t wireHash(Digest const& digest);

    int mMaxEntries;
    std::optional<int> mMmTokenIdOffset;
    std::optional<LifeCycleId> mTargetLifeCycle;
    std::vector<LifeCycleId> mRecurrentLifeCycles;
    int mPendingEntries = 0;
    std::unordered_map<Digest, StoredBlock> mStoredBlocks;
    std::vector<StreamingEventData> mPendingEvents;
    StreamingEventStats mStats;
    mutable std::mutex mMutex;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
