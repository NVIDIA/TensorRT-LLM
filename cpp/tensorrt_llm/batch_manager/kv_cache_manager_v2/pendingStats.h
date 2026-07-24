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
};

struct PendingStatsDelta
{
    KVCacheStatsDelta globalStats;
    KVCacheStatsDelta requestStats;
    KVCacheIterationStatsDelta iterationStats;
    std::optional<LifeCycleId> lifeCycle;

    [[nodiscard]] bool empty() const noexcept
    {
        return globalStats.empty() && requestStats.empty() && iterationStats.empty();
    }
};

class PendingStats
{
public:
    [[nodiscard]] bool empty() const noexcept
    {
        return mRequestStats.empty() && mGlobalStats.empty() && mIterationStatsByLifeCycle.empty();
    }

    void clear() noexcept
    {
        mRequestStats.clear();
        mGlobalStats.clear();
        mIterationStatsByLifeCycle.clear();
        mAllocationSegments.clear();
    }

    bool recordAllocationRange(LifeCycleId lifeCycle, BlockOrdinal blockBegin, BlockOrdinal blockEnd, int beamWidth,
        bool countAsMissed, bool countAsGeneration = false)
    {
        if (blockBegin >= blockEnd)
        {
            return false;
        }
        PendingAllocationSegment segment{lifeCycle, blockBegin, blockEnd, beamWidth, countAsMissed, countAsGeneration};
        if (!add(allocationDelta(segment, blockBegin, blockEnd)))
        {
            return false;
        }
        mAllocationSegments.push_back(segment);
        return true;
    }

    bool recordReuse(LifeCycleId lifeCycle, int fullReusedBlocks, int partialReusedBlocks)
    {
        int const reusedBlocks = fullReusedBlocks + partialReusedBlocks;
        if (reusedBlocks == 0)
        {
            return false;
        }

        PendingStatsDelta delta;
        delta.globalStats.reusedBlocks = reusedBlocks;
        delta.requestStats.reusedBlocks = reusedBlocks;
        delta.iterationStats.iterReusedBlocks = reusedBlocks;
        delta.iterationStats.iterFullReusedBlocks = fullReusedBlocks;
        delta.iterationStats.iterPartialReusedBlocks = partialReusedBlocks;
        delta.lifeCycle = lifeCycle;
        return add(delta);
    }

    bool subtractAllocationRange(BlockOrdinal blockBegin, BlockOrdinal blockEnd)
    {
        if (blockBegin >= blockEnd || mAllocationSegments.empty())
        {
            return false;
        }

        bool changed = false;
        int index = static_cast<int>(mAllocationSegments.size()) - 1;
        while (index >= 0)
        {
            auto& segment = mAllocationSegments[static_cast<size_t>(index)];
            if (segment.blockEnd <= blockBegin)
            {
                break;
            }
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

private:
    static PendingStatsDelta allocationDelta(
        PendingAllocationSegment const& segment, BlockOrdinal blockBegin, BlockOrdinal blockEnd)
    {
        int64_t const numBlocks = static_cast<int64_t>(std::max(0, blockEnd - blockBegin)) * segment.beamWidth;
        PendingStatsDelta delta;
        delta.globalStats.allocTotalBlocks = numBlocks;
        delta.globalStats.allocNewBlocks = numBlocks;
        delta.globalStats.missedBlocks = segment.countAsMissed ? numBlocks : 0;
        delta.requestStats = delta.globalStats.copy();
        delta.iterationStats.iterAllocTotalBlocks = numBlocks;
        delta.iterationStats.iterAllocNewBlocks = numBlocks;
        delta.iterationStats.iterMissedBlocks = segment.countAsMissed ? numBlocks : 0;
        delta.iterationStats.iterGenAllocBlocks = segment.countAsGeneration ? numBlocks : 0;
        delta.lifeCycle = segment.lifeCycle;
        return delta;
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
        return true;
    }

    KVCacheStatsDelta mRequestStats;
    KVCacheStatsDelta mGlobalStats;
    IterationStatsByLifeCycle mIterationStatsByLifeCycle;
    std::vector<PendingAllocationSegment> mAllocationSegments;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
