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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"

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
    //! Set for recurrent-state checkpoints: the native layer-group (life cycle) id of the
    //! recurrent group, the same id buffered events report as layer_group_id, and the kind
    //! "mamba". Attention events leave both unset.
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
    //! Snapshots that cover only part of their block (unaligned snapshot points) and were
    //! therefore not published as checkpoints.
    int64_t recurrentCheckpointsSuppressed = 0;
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
    //! Opt in to publishing recurrent-state (SSM) snapshots as tagged checkpoint events
    //! (`groupIdx` = the recurrent life cycle's id, `kvCacheSpecKind` = "mamba"). The recurrent
    //! life cycles come from declareLifeCycles(): every non-PRIVATE SSM life cycle. A checkpoint is
    //! published for a block only after the block's attention page was published, under the same
    //! block hash, so consumers can align it with the attention prefix.
    void setPublishRecurrentCheckpoints(bool enabled);
    //! Override the recurrent life cycles derived by declareLifeCycles(); also opts in. For tests
    //! and for managers whose registry is not declared through the sink.
    void setRecurrentLifeCycles(std::vector<LifeCycleId> lifeCycles);
    void declareLifeCycles(LifeCycleRegistry const& lifeCycles) override;
    bool publishesRecurrentCheckpoints() const override;
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
        //! Life cycle ids of the recurrent groups whose checkpoint is published for this block.
        std::vector<int> publishedCheckpointGroups;
    };

    void addStoredBlockUnlocked(Block const& block);
    void addResidentCheckpointsUnlocked(Block const& block);
    void addRecurrentCheckpointUnlocked(Block const& block, LifeCycleId lifeCycle);
    void addRemovedBlockUnlocked(Digest const& blockKey);
    void addRemovedHashUnlocked(int64_t blockHash, std::optional<int> groupIdx = std::nullopt);
    [[nodiscard]] bool isRecurrentLifeCycle(LifeCycleId lifeCycle) const;
    [[nodiscard]] bool publishesRecurrentCheckpointsUnlocked() const;
    void recordDroppedEventUnlocked(char const* reason);
    [[nodiscard]] bool reserveEntryUnlocked();
    [[nodiscard]] static int64_t wireHash(Digest const& digest);

    int mMaxEntries;
    std::optional<int> mMmTokenIdOffset;
    std::optional<LifeCycleId> mTargetLifeCycle;
    //! Non-PRIVATE SSM life cycles (from declareLifeCycles(), or an explicit override).
    std::vector<LifeCycleId> mRecurrentLifeCycles;
    bool mPublishRecurrentCheckpoints = false;
    int mPendingEntries = 0;
    std::unordered_map<Digest, StoredBlock> mStoredBlocks;
    std::vector<StreamingEventData> mPendingEvents;
    StreamingEventStats mStats;
    mutable std::mutex mMutex;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
