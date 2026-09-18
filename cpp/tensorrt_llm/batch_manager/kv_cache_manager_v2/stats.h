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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct KVCacheStatsDelta
{
    int64_t allocTotalBlocks = 0;
    int64_t allocNewBlocks = 0;
    int64_t reusedBlocks = 0;
    int64_t missedBlocks = 0;

    void add(KVCacheStatsDelta const& other) noexcept
    {
        allocTotalBlocks += other.allocTotalBlocks;
        allocNewBlocks += other.allocNewBlocks;
        reusedBlocks += other.reusedBlocks;
        missedBlocks += other.missedBlocks;
    }

    void subtract(KVCacheStatsDelta const& other) noexcept
    {
        allocTotalBlocks -= other.allocTotalBlocks;
        allocNewBlocks -= other.allocNewBlocks;
        reusedBlocks -= other.reusedBlocks;
        missedBlocks -= other.missedBlocks;
    }

    void clear() noexcept
    {
        *this = {};
    }

    [[nodiscard]] KVCacheStatsDelta copy() const noexcept
    {
        return *this;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return allocTotalBlocks == 0 && allocNewBlocks == 0 && reusedBlocks == 0 && missedBlocks == 0;
    }

    [[nodiscard]] bool operator==(KVCacheStatsDelta const& other) const noexcept
    {
        return allocTotalBlocks == other.allocTotalBlocks && allocNewBlocks == other.allocNewBlocks
            && reusedBlocks == other.reusedBlocks && missedBlocks == other.missedBlocks;
    }
};

struct KVCacheIterationStatsDelta
{
    int64_t iterAllocTotalBlocks = 0;
    int64_t iterAllocNewBlocks = 0;
    int64_t iterReusedBlocks = 0;
    int64_t iterFullReusedBlocks = 0;
    int64_t iterPartialReusedBlocks = 0;
    int64_t iterMissedBlocks = 0;
    int64_t iterGenAllocBlocks = 0;
    int64_t iterOnboardBlocks = 0;
    int64_t iterOnboardBytes = 0;
    int64_t iterOffloadBlocks = 0;
    int64_t iterOffloadBytes = 0;
    int64_t iterIntraDeviceCopyBlocks = 0;
    int64_t iterIntraDeviceCopyBytes = 0;
    int64_t iterHostDroppedBlocks = 0;
    int64_t iterHostDroppedBytes = 0;

    void add(KVCacheIterationStatsDelta const& other) noexcept
    {
        iterAllocTotalBlocks += other.iterAllocTotalBlocks;
        iterAllocNewBlocks += other.iterAllocNewBlocks;
        iterReusedBlocks += other.iterReusedBlocks;
        iterFullReusedBlocks += other.iterFullReusedBlocks;
        iterPartialReusedBlocks += other.iterPartialReusedBlocks;
        iterMissedBlocks += other.iterMissedBlocks;
        iterGenAllocBlocks += other.iterGenAllocBlocks;
        iterOnboardBlocks += other.iterOnboardBlocks;
        iterOnboardBytes += other.iterOnboardBytes;
        iterOffloadBlocks += other.iterOffloadBlocks;
        iterOffloadBytes += other.iterOffloadBytes;
        iterIntraDeviceCopyBlocks += other.iterIntraDeviceCopyBlocks;
        iterIntraDeviceCopyBytes += other.iterIntraDeviceCopyBytes;
        iterHostDroppedBlocks += other.iterHostDroppedBlocks;
        iterHostDroppedBytes += other.iterHostDroppedBytes;
    }

    void subtract(KVCacheIterationStatsDelta const& other) noexcept
    {
        iterAllocTotalBlocks -= other.iterAllocTotalBlocks;
        iterAllocNewBlocks -= other.iterAllocNewBlocks;
        iterReusedBlocks -= other.iterReusedBlocks;
        iterFullReusedBlocks -= other.iterFullReusedBlocks;
        iterPartialReusedBlocks -= other.iterPartialReusedBlocks;
        iterMissedBlocks -= other.iterMissedBlocks;
        iterGenAllocBlocks -= other.iterGenAllocBlocks;
        iterOnboardBlocks -= other.iterOnboardBlocks;
        iterOnboardBytes -= other.iterOnboardBytes;
        iterOffloadBlocks -= other.iterOffloadBlocks;
        iterOffloadBytes -= other.iterOffloadBytes;
        iterIntraDeviceCopyBlocks -= other.iterIntraDeviceCopyBlocks;
        iterIntraDeviceCopyBytes -= other.iterIntraDeviceCopyBytes;
        iterHostDroppedBlocks -= other.iterHostDroppedBlocks;
        iterHostDroppedBytes -= other.iterHostDroppedBytes;
    }

    void clear() noexcept
    {
        *this = {};
    }

    [[nodiscard]] KVCacheIterationStatsDelta copy() const noexcept
    {
        return *this;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return iterAllocTotalBlocks == 0 && iterAllocNewBlocks == 0 && iterReusedBlocks == 0
            && iterFullReusedBlocks == 0 && iterPartialReusedBlocks == 0 && iterMissedBlocks == 0
            && iterGenAllocBlocks == 0 && iterOnboardBlocks == 0 && iterOnboardBytes == 0 && iterOffloadBlocks == 0
            && iterOffloadBytes == 0 && iterIntraDeviceCopyBlocks == 0 && iterIntraDeviceCopyBytes == 0
            && iterHostDroppedBlocks == 0 && iterHostDroppedBytes == 0;
    }

    [[nodiscard]] double iterCacheHitRate() const noexcept
    {
        int64_t const total = iterReusedBlocks + iterMissedBlocks;
        if (iterReusedBlocks == 0 || total == 0)
        {
            return 0.0;
        }
        return static_cast<double>(iterReusedBlocks) / static_cast<double>(total);
    }

    [[nodiscard]] bool operator==(KVCacheIterationStatsDelta const& other) const noexcept
    {
        return iterAllocTotalBlocks == other.iterAllocTotalBlocks && iterAllocNewBlocks == other.iterAllocNewBlocks
            && iterReusedBlocks == other.iterReusedBlocks && iterFullReusedBlocks == other.iterFullReusedBlocks
            && iterPartialReusedBlocks == other.iterPartialReusedBlocks && iterMissedBlocks == other.iterMissedBlocks
            && iterGenAllocBlocks == other.iterGenAllocBlocks && iterOnboardBlocks == other.iterOnboardBlocks
            && iterOnboardBytes == other.iterOnboardBytes && iterOffloadBlocks == other.iterOffloadBlocks
            && iterOffloadBytes == other.iterOffloadBytes
            && iterIntraDeviceCopyBlocks == other.iterIntraDeviceCopyBlocks
            && iterIntraDeviceCopyBytes == other.iterIntraDeviceCopyBytes
            && iterHostDroppedBlocks == other.iterHostDroppedBlocks
            && iterHostDroppedBytes == other.iterHostDroppedBytes;
    }
};

using IterationStatsByLifeCycle = std::unordered_map<LifeCycleId, KVCacheIterationStatsDelta>;

// ---------------------------------------------------------------------------
// CountsByLevel — counters indexed by CacheLevel, so entry i always belongs to
// the i-th configured cache tier. The length follows the configured tier list
// instead of a hard-coded GPU/host/disk split, which is what lets a deployment
// with a hot and a cold GPU level report them as two distinct entries.
//
// Kept outside KVCacheIterationStatsDelta on purpose: the level count is a
// runtime quantity, while that struct is a fixed-field record whose field-wise
// add/subtract helpers assume scalar members.
// ---------------------------------------------------------------------------

using CountsByLevel = TypedVec<CacheLevel, int64_t>;

//! Element-wise accumulate, widening `dst` when `src` covers more levels.
inline void addCountsByLevel(CountsByLevel& dst, CountsByLevel const& src)
{
    if (dst.size() < src.size())
    {
        dst.resize(src.size(), 0);
    }
    for (CacheLevel level{0}; level < src.size(); ++level)
    {
        dst.at(level) += src.at(level);
    }
}

inline bool countsByLevelEmpty(CountsByLevel const& counts) noexcept
{
    return std::all_of(counts.begin(), counts.end(), [](int64_t count) { return count == 0; });
}

inline int64_t countsByLevelTotal(CountsByLevel const& counts) noexcept
{
    return std::accumulate(counts.begin(), counts.end(), int64_t{0});
}

//! Reuse block counts split by the cache level the reused pages were resident on when the match
//! was taken.
struct ReusedBlocksByLevel
{
    CountsByLevel full;
    CountsByLevel partial;

    void add(ReusedBlocksByLevel const& other)
    {
        addCountsByLevel(full, other.full);
        addCountsByLevel(partial, other.partial);
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return countsByLevelEmpty(full) && countsByLevelEmpty(partial);
    }
};

using ReusedBlocksByLevelByLifeCycle = std::unordered_map<LifeCycleId, ReusedBlocksByLevel>;

// ---------------------------------------------------------------------------
// SsmSnapshotIterationStatsDelta — per-lifecycle counters for SSM snapshot
// reuse in one iteration. Mirrors Python's SsmSnapshotIterationStatsDelta.
// ---------------------------------------------------------------------------
struct SsmSnapshotIterationStatsDelta
{
    int64_t iterSnapshotLookups = 0;
    int64_t iterSnapshotHits = 0;
    int64_t iterSnapshotMisses = 0;
    int64_t iterReusedTokens = 0;
    int64_t iterUnreusedTokens = 0;
    int64_t iterAlignedSnapshotHits = 0;
    int64_t iterUnalignedSnapshotHits = 0;

    void add(SsmSnapshotIterationStatsDelta const& other) noexcept
    {
        iterSnapshotLookups += other.iterSnapshotLookups;
        iterSnapshotHits += other.iterSnapshotHits;
        iterSnapshotMisses += other.iterSnapshotMisses;
        iterReusedTokens += other.iterReusedTokens;
        iterUnreusedTokens += other.iterUnreusedTokens;
        iterAlignedSnapshotHits += other.iterAlignedSnapshotHits;
        iterUnalignedSnapshotHits += other.iterUnalignedSnapshotHits;
    }

    void subtract(SsmSnapshotIterationStatsDelta const& other) noexcept
    {
        iterSnapshotLookups -= other.iterSnapshotLookups;
        iterSnapshotHits -= other.iterSnapshotHits;
        iterSnapshotMisses -= other.iterSnapshotMisses;
        iterReusedTokens -= other.iterReusedTokens;
        iterUnreusedTokens -= other.iterUnreusedTokens;
        iterAlignedSnapshotHits -= other.iterAlignedSnapshotHits;
        iterUnalignedSnapshotHits -= other.iterUnalignedSnapshotHits;
    }

    void clear() noexcept
    {
        *this = {};
    }

    [[nodiscard]] SsmSnapshotIterationStatsDelta copy() const noexcept
    {
        return *this;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return iterSnapshotLookups == 0 && iterSnapshotHits == 0 && iterSnapshotMisses == 0 && iterReusedTokens == 0
            && iterUnreusedTokens == 0 && iterAlignedSnapshotHits == 0 && iterUnalignedSnapshotHits == 0;
    }

    [[nodiscard]] double iterSnapshotHitRate() const noexcept
    {
        if (iterSnapshotHits == 0 || iterSnapshotLookups == 0)
        {
            return 0.0;
        }
        return static_cast<double>(iterSnapshotHits) / static_cast<double>(iterSnapshotLookups);
    }

    [[nodiscard]] bool operator==(SsmSnapshotIterationStatsDelta const& other) const noexcept
    {
        return iterSnapshotLookups == other.iterSnapshotLookups && iterSnapshotHits == other.iterSnapshotHits
            && iterSnapshotMisses == other.iterSnapshotMisses && iterReusedTokens == other.iterReusedTokens
            && iterUnreusedTokens == other.iterUnreusedTokens
            && iterAlignedSnapshotHits == other.iterAlignedSnapshotHits
            && iterUnalignedSnapshotHits == other.iterUnalignedSnapshotHits;
    }
};

using SsmSnapshotIterationStatsByLifeCycle = std::unordered_map<LifeCycleId, SsmSnapshotIterationStatsDelta>;

struct PoolGroupPeakBlockStats
{
    SlotCount available = 0;
    SlotCount unavailable = 0;
    SlotCount evictable = 0;

    [[nodiscard]] bool operator==(PoolGroupPeakBlockStats const& other) const noexcept
    {
        return available == other.available && unavailable == other.unavailable && evictable == other.evictable;
    }
};

using PeakBlockStatsByPoolGroup = TypedVec<PoolGroupIndex, PoolGroupPeakBlockStats>;
using PeakBlockStatsByCacheLevel = TypedVec<CacheLevel, PeakBlockStatsByPoolGroup>;

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
