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

#include "kv_cache_manager_v2/stats.h"

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct PendingAllocationSegment
{
    LifeCycleId lifeCycle;
    BlockOrdinal blockBegin;
    BlockOrdinal blockEnd;
    int beamWidth;
    bool countAsMissed;
    bool countAsGeneration;
    bool recordManagerStats;
    bool recordRequestStats;
    // Allocated for content a KV connector loaded from an external store: booked as reused rather
    // than missed.
    bool countAsConnectorReused{false};
};

struct PendingStatsDelta
{
    KVCacheStatsDelta globalStats;
    KVCacheStatsDelta requestStats;
    KVCacheIterationStatsDelta iterationStats;
    std::optional<LifeCycleId> lifeCycle;
    int64_t connectorReusedBlocks{0};

    [[nodiscard]] bool empty() const noexcept
    {
        return globalStats.empty() && requestStats.empty() && iterationStats.empty() && connectorReusedBlocks == 0;
    }
};

class PendingStats
{
public:
    [[nodiscard]] bool empty() const noexcept
    {
        return mRequestStats.empty() && mGlobalStats.empty() && mIterationStatsByLifeCycle.empty()
            && mSsmSnapshotIterationStatsByLifeCycle.empty() && countsByLevelEmpty(mCachedTokensByLevel)
            && mConnectorReusedBlocksByLifeCycle.empty() && mConnectorCachedTokens == 0;
    }

    void clear() noexcept
    {
        mRequestStats.clear();
        mGlobalStats.clear();
        mIterationStatsByLifeCycle.clear();
        mSsmSnapshotIterationStatsByLifeCycle.clear();
        mReusedBlocksByLevelByLifeCycle.clear();
        mConnectorReusedBlocksByLifeCycle.clear();
        mCachedTokensByLevel.clear();
        mConnectorCachedTokens = 0;
        mAllocationSegments.clear();
    }

    bool recordAllocationRange(LifeCycleId lifeCycle, BlockOrdinal blockBegin, BlockOrdinal blockEnd, int beamWidth,
        bool countAsMissed, bool countAsGeneration = false, bool recordManagerStats = true,
        bool recordRequestStats = true)
    {
        if (blockBegin >= blockEnd)
        {
            return false;
        }
        PendingAllocationSegment segment{lifeCycle, blockBegin, blockEnd, beamWidth, countAsMissed, countAsGeneration,
            recordManagerStats, recordRequestStats};
        if (!add(allocationDelta(segment, blockBegin, blockEnd)))
        {
            return false;
        }
        mAllocationSegments.push_back(segment);
        return true;
    }

    // `byLevel` splits the same full/partial counts across the cache levels the reused pages were
    // resident on. It rides along with the scalar counters so both are committed or discarded
    // together; reuse is never rolled back (only allocation ranges are), so add-only is enough.
    bool recordReuse(LifeCycleId lifeCycle, int fullReusedBlocks, int partialReusedBlocks,
        ReusedBlocksByLevel const& byLevel = {}, bool recordManagerStats = true, bool recordRequestStats = true)
    {
        int const reusedBlocks = fullReusedBlocks + partialReusedBlocks;
        if (reusedBlocks == 0 || (!recordManagerStats && !recordRequestStats))
        {
            return false;
        }

        PendingStatsDelta delta;
        if (recordManagerStats)
        {
            delta.globalStats.reusedBlocks = reusedBlocks;
            delta.iterationStats.iterReusedBlocks = reusedBlocks;
            delta.iterationStats.iterFullReusedBlocks = fullReusedBlocks;
            delta.iterationStats.iterPartialReusedBlocks = partialReusedBlocks;
            delta.lifeCycle = lifeCycle;
            mReusedBlocksByLevelByLifeCycle[lifeCycle].add(byLevel);
        }
        if (recordRequestStats)
        {
            delta.requestStats.reusedBlocks = reusedBlocks;
        }
        return add(delta);
    }

    // Record one SSM snapshot lookup for a lifecycle. Mirrors Python's
    // _PendingStats.record_ssm_snapshot_lookup(). Returns false if lookupTokens == 0.
    bool recordSsmSnapshotLookup(LifeCycleId lifeCycle, int lookupTokens, int reusedTokens, int tokensPerBlock)
    {
        if (lookupTokens == 0)
        {
            return false;
        }
        TLLM_CHECK_DEBUG(lookupTokens > 0);
        TLLM_CHECK_DEBUG(0 <= reusedTokens && reusedTokens <= lookupTokens);
        TLLM_CHECK_DEBUG(tokensPerBlock > 0);

        bool const isHit = reusedTokens > 0;
        SsmSnapshotIterationStatsDelta delta;
        delta.iterSnapshotLookups = 1;
        delta.iterSnapshotHits = isHit ? 1 : 0;
        delta.iterSnapshotMisses = isHit ? 0 : 1;
        delta.iterReusedTokens = reusedTokens;
        delta.iterUnreusedTokens = lookupTokens - reusedTokens;
        delta.iterAlignedSnapshotHits = (isHit && reusedTokens % tokensPerBlock == 0) ? 1 : 0;
        delta.iterUnalignedSnapshotHits = (isHit && reusedTokens % tokensPerBlock != 0) ? 1 : 0;
        mSsmSnapshotIterationStatsByLifeCycle[lifeCycle].add(delta);
        return true;
    }

    //! Rebook the pending missed allocations in [blockBegin, blockEnd) as reused, for blocks whose
    //! content a KV connector loads instead of the forward pass computing it. The blocks stay
    //! counted as allocated. Allocations already committed are not pending and stay missed.
    bool recordConnectorReuse(BlockOrdinal blockBegin, BlockOrdinal blockEnd)
    {
        if (blockBegin >= blockEnd || mAllocationSegments.empty())
        {
            return false;
        }

        bool changed = false;
        std::vector<PendingAllocationSegment> segments;
        segments.reserve(mAllocationSegments.size() + 2);
        for (auto const& segment : mAllocationSegments)
        {
            BlockOrdinal const servedBegin = std::max(blockBegin, segment.blockBegin);
            BlockOrdinal const servedEnd = std::min(blockEnd, segment.blockEnd);
            if (servedBegin >= servedEnd || !segment.countAsMissed)
            {
                segments.push_back(segment);
                continue;
            }

            changed = true;
            subtract(allocationDelta(segment, servedBegin, servedEnd));
            if (segment.blockBegin < servedBegin)
            {
                auto head = segment;
                head.blockEnd = servedBegin;
                segments.push_back(head);
            }
            auto served = segment;
            served.blockBegin = servedBegin;
            served.blockEnd = servedEnd;
            served.countAsMissed = false;
            served.countAsConnectorReused = true;
            add(allocationDelta(served, servedBegin, servedEnd));
            segments.push_back(served);
            if (servedEnd < segment.blockEnd)
            {
                auto tail = segment;
                tail.blockBegin = servedEnd;
                segments.push_back(tail);
            }
        }
        mAllocationSegments = std::move(segments);
        return changed;
    }

    bool subtractAllocationRange(BlockOrdinal blockBegin, BlockOrdinal blockEnd)
    {
        if (blockBegin >= blockEnd || mAllocationSegments.empty())
        {
            return false;
        }

        // Every segment is examined: once recordConnectorReuse() has split segments they are no
        // longer ordered by position across life cycles. The list is cleared at each commit.
        bool changed = false;
        int index = static_cast<int>(mAllocationSegments.size()) - 1;
        while (index >= 0)
        {
            auto& segment = mAllocationSegments[static_cast<size_t>(index)];
            BlockOrdinal const removedBegin = std::max(blockBegin, segment.blockBegin);
            BlockOrdinal const removedEnd = std::min(blockEnd, segment.blockEnd);
            if (removedBegin >= removedEnd)
            {
                --index;
                continue;
            }

            changed = true;
            subtract(allocationDelta(segment, removedBegin, removedEnd));
            if (removedBegin <= segment.blockBegin)
            {
                mAllocationSegments.erase(mAllocationSegments.begin() + index);
            }
            else
            {
                TLLM_CHECK_DEBUG(removedEnd == segment.blockEnd);
                segment.blockEnd = removedBegin;
            }
            --index;
        }
        return changed;
    }

    KVCacheStatsDelta const& globalStats() const noexcept
    {
        return mGlobalStats;
    }

    KVCacheStatsDelta const& requestStats() const noexcept
    {
        return mRequestStats;
    }

    IterationStatsByLifeCycle const& iterationStatsByLifeCycle() const noexcept
    {
        return mIterationStatsByLifeCycle;
    }

    SsmSnapshotIterationStatsByLifeCycle const& ssmSnapshotIterationStatsByLifeCycle() const noexcept
    {
        return mSsmSnapshotIterationStatsByLifeCycle;
    }

    ReusedBlocksByLevelByLifeCycle const& reusedBlocksByLevelByLifeCycle() const noexcept
    {
        return mReusedBlocksByLevelByLifeCycle;
    }

    ConnectorReusedBlocksByLifeCycle const& connectorReusedBlocksByLifeCycle() const noexcept
    {
        return mConnectorReusedBlocksByLifeCycle;
    }

    // Cached-token attribution for the sequence's reuse match, indexed by cache level.
    //
    // Unlike the reuse counters this is a manager-global quantity rather than a per-lifecycle one:
    // a match spans every lifecycle at once (the final SSM checkpoint summarizes the whole
    // recurrent prefix, so its tier applies to every matched token), leaving no single lifecycle to
    // attribute it to. It still rides the pending-stats lifecycle so it is committed or discarded
    // together with the counters it was derived from -- in particular, a dummy sequence's
    // attribution is dropped by the same discardPendingStats() that drops its reuse counters.
    bool recordCachedTokensByLevel(CountsByLevel const& counts)
    {
        if (countsByLevelEmpty(counts))
        {
            return false;
        }
        addCountsByLevel(mCachedTokensByLevel, counts);
        return true;
    }

    void clearCachedTokensByLevel() noexcept
    {
        mCachedTokensByLevel.clear();
    }

    //! Limit a level's staged attribution without restoring counts already dropped or committed.
    void limitCachedTokensByLevel(CacheLevel level, int64_t maxTokens)
    {
        // No attribution is staged when stats are disabled, discarded, or already committed.
        if (mCachedTokensByLevel.empty())
        {
            return;
        }
        TLLM_CHECK_DEBUG(level < mCachedTokensByLevel.size());
        if (level >= mCachedTokensByLevel.size())
        {
            return;
        }
        auto& count = mCachedTokensByLevel.at(level);
        TLLM_CHECK_DEBUG(maxTokens >= 0);
        count = std::min(count, maxTokens);
    }

    [[nodiscard]] CountsByLevel const& cachedTokensByLevel() const noexcept
    {
        return mCachedTokensByLevel;
    }

    // Prompt tokens a KV connector loaded from an external store, the cached tokens that sit on no
    // cache level. Manager-global like the attribution above, for the same reason.
    bool recordConnectorCachedTokens(int64_t numTokens)
    {
        TLLM_CHECK_DEBUG(numTokens >= 0);
        if (numTokens <= 0)
        {
            return false;
        }
        mConnectorCachedTokens += numTokens;
        return true;
    }

    [[nodiscard]] int64_t connectorCachedTokens() const noexcept
    {
        return mConnectorCachedTokens;
    }

private:
    static PendingStatsDelta allocationDelta(
        PendingAllocationSegment const& segment, BlockOrdinal blockBegin, BlockOrdinal blockEnd)
    {
        int64_t const numBlocks = static_cast<int64_t>(blockEnd - blockBegin) * segment.beamWidth;
        int64_t const missedBlocks = segment.countAsMissed ? numBlocks : 0;
        int64_t const connectorReusedBlocks = segment.countAsConnectorReused ? numBlocks : 0;
        PendingStatsDelta delta;
        if (segment.recordManagerStats)
        {
            delta.globalStats.allocTotalBlocks = numBlocks;
            delta.globalStats.allocNewBlocks = numBlocks;
            delta.globalStats.reusedBlocks = connectorReusedBlocks;
            delta.globalStats.missedBlocks = missedBlocks;
            delta.iterationStats.iterAllocTotalBlocks = numBlocks;
            delta.iterationStats.iterAllocNewBlocks = numBlocks;
            delta.iterationStats.iterReusedBlocks = connectorReusedBlocks;
            delta.iterationStats.iterFullReusedBlocks = connectorReusedBlocks;
            delta.iterationStats.iterMissedBlocks = missedBlocks;
            delta.iterationStats.iterGenAllocBlocks = segment.countAsGeneration ? numBlocks : 0;
            delta.lifeCycle = segment.lifeCycle;
            delta.connectorReusedBlocks = connectorReusedBlocks;
        }
        if (segment.recordRequestStats)
        {
            delta.requestStats.allocTotalBlocks = numBlocks;
            delta.requestStats.allocNewBlocks = numBlocks;
            delta.requestStats.reusedBlocks = connectorReusedBlocks;
            delta.requestStats.missedBlocks = missedBlocks;
        }
        return delta;
    }

    void addConnectorReusedBlocks(std::optional<LifeCycleId> lifeCycle, int64_t numBlocks)
    {
        if (numBlocks == 0)
        {
            return;
        }
        TLLM_CHECK_DEBUG(lifeCycle.has_value());
        auto const it = mConnectorReusedBlocksByLifeCycle.try_emplace(*lifeCycle, 0).first;
        it->second += numBlocks;
        if (it->second == 0)
        {
            mConnectorReusedBlocksByLifeCycle.erase(it);
        }
    }

    bool add(PendingStatsDelta const& delta)
    {
        if (delta.empty())
        {
            return false;
        }
        mGlobalStats.add(delta.globalStats);
        mRequestStats.add(delta.requestStats);
        if (!delta.iterationStats.empty())
        {
            TLLM_CHECK_DEBUG(delta.lifeCycle.has_value());
            mIterationStatsByLifeCycle[*delta.lifeCycle].add(delta.iterationStats);
        }
        addConnectorReusedBlocks(delta.lifeCycle, delta.connectorReusedBlocks);
        return true;
    }

    bool subtract(PendingStatsDelta const& delta)
    {
        if (delta.empty())
        {
            return false;
        }
        mGlobalStats.subtract(delta.globalStats);
        mRequestStats.subtract(delta.requestStats);
        if (!delta.iterationStats.empty())
        {
            TLLM_CHECK_DEBUG(delta.lifeCycle.has_value());
            auto const it = mIterationStatsByLifeCycle.find(*delta.lifeCycle);
            if (it != mIterationStatsByLifeCycle.end())
            {
                it->second.subtract(delta.iterationStats);
                if (it->second.empty())
                {
                    mIterationStatsByLifeCycle.erase(it);
                }
            }
        }
        addConnectorReusedBlocks(delta.lifeCycle, -delta.connectorReusedBlocks);
        return true;
    }

    KVCacheStatsDelta mRequestStats;
    KVCacheStatsDelta mGlobalStats;
    IterationStatsByLifeCycle mIterationStatsByLifeCycle;
    SsmSnapshotIterationStatsByLifeCycle mSsmSnapshotIterationStatsByLifeCycle;
    ReusedBlocksByLevelByLifeCycle mReusedBlocksByLevelByLifeCycle;
    ConnectorReusedBlocksByLifeCycle mConnectorReusedBlocksByLifeCycle;
    CountsByLevel mCachedTokensByLevel;
    int64_t mConnectorCachedTokens{0};
    std::vector<PendingAllocationSegment> mAllocationSegments;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
