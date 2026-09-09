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

#include <cstdint>
#include <unordered_map>

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
    // SWA scratch reuse attribution. Scratch blocks are excluded from iterAlloc* by design, so
    // these are the only accounting for the saving: iterAllocNewBlocks drops by exactly
    // iterScratchBlocks.
    //
    // The two fields have DIFFERENT semantics, which the names carry deliberately:
    //   iterScratchBlocks       - a COUNT. Blocks served from shared scratch sub-pages during
    //                             this iteration. Additive: summing over iterations gives the
    //                             total number of blocks ever served from scratch.
    //   iterScratchSlotsInUse   - a GAUGE, sampled per lifecycle as slots are recorded and
    //                             therefore summed across lifecycles WITHIN one iteration to
    //                             give the slots concurrently in use. It is reset every
    //                             iteration along with the rest of the delta. Accumulating it
    //                             ACROSS iterations is meaningless - it is an occupancy, not a
    //                             flow. Consumers that sum every int field indiscriminately
    //                             will produce a nonsense number here; the "InUse" suffix is
    //                             the contract.
    int64_t iterScratchBlocks = 0;
    int64_t iterScratchSlotsInUse = 0;

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
        iterScratchBlocks += other.iterScratchBlocks;
        iterScratchSlotsInUse += other.iterScratchSlotsInUse;
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
        iterScratchBlocks -= other.iterScratchBlocks;
        iterScratchSlotsInUse -= other.iterScratchSlotsInUse;
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
            && iterHostDroppedBlocks == 0 && iterHostDroppedBytes == 0 && iterScratchBlocks == 0
            && iterScratchSlotsInUse == 0;
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
            && iterHostDroppedBytes == other.iterHostDroppedBytes && iterScratchBlocks == other.iterScratchBlocks
            && iterScratchSlotsInUse == other.iterScratchSlotsInUse;
    }
};

using IterationStatsByLifeCycle = std::unordered_map<LifeCycleId, KVCacheIterationStatsDelta>;

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
