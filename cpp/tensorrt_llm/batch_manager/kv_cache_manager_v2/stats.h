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
