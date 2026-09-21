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
};

//! Semantic data for one wire-level BlockRemoved event.
struct StreamingBlockRemovedData
{
    std::vector<int64_t> blockHashes;
};

using StreamingEventData = std::variant<StreamingBlockStoredData, StreamingBlockRemovedData>;

//! Counters accumulated over the lifetime of a streaming event sink.
struct StreamingEventStats
{
    int64_t storedBlocks = 0;
    int64_t removedBlocks = 0;
    int64_t partialBlocksSuppressed = 0;
    int64_t nonTargetLifeCyclesIgnored = 0;
    int64_t droppedEvents = 0;
};

//! Captures streaming KV-cache lifecycle events without depending on Python or a transport.
class StreamingEventSink final : public EventSink
{
public:
    StreamingEventSink(int tokensPerBlock, int maxEntries, std::optional<int> mmTokenIdOffset = std::nullopt);

    bool needsTokenDigestContext() const override
    {
        return mMmTokenIdOffset.has_value();
    }

    void setTargetLifeCycle(LifeCycleId lifeCycle);
    [[nodiscard]] std::vector<StreamingEventData> drainIterationEvents();
    [[nodiscard]] StreamingEventStats getStats() const;

    void addStoredBlock(Block const& block) override;
    void addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle) override;
    void addRemovedBlock(Digest const& blockKey) override;
    void addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle) override;
    void addCacheLevelUpdated(
        Digest const& blockKey, CacheLevel oldLevel, CacheLevel newLevel, LifeCycleId lifeCycle) override;

private:
    void addStoredBlockUnlocked(Block const& block);
    void addRemovedBlockUnlocked(Digest const& blockKey);
    void addRemovedHashUnlocked(int64_t blockHash);
    [[nodiscard]] bool reserveEntryUnlocked();
    [[nodiscard]] static int64_t wireHash(Digest const& digest);

    int mTokensPerBlock;
    int mMaxEntries;
    std::optional<int> mMmTokenIdOffset;
    std::optional<LifeCycleId> mTargetLifeCycle;
    int mPendingEntries = 0;
    std::unordered_map<Digest, int64_t> mStoredBlocks;
    std::vector<StreamingEventData> mPendingEvents;
    StreamingEventStats mStats;
    mutable std::mutex mMutex;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
