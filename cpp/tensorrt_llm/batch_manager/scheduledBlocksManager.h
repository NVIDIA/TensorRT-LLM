/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <algorithm>
#include <map>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class NoEvictScheduledBlocksManager
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

public:
    explicit NoEvictScheduledBlocksManager(BaseKVCacheManager const& kvCacheManager)
        : mKvCacheManager(kvCacheManager)
        , mAvailableBlocks(mKvCacheManager.getBlockManager().getNumFreeBlocksPerWindowSize())
    {
    }

    /// @brief  Check whether enough blocks are available for the request.
    /// Caches the per-window block counts for a subsequent commitBlocks() call.
    bool enoughAvailableBlocks(
        LlmRequest const& req, std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt)
    {
        mCachedNeededBlocks.clear();
        bool enough = true;
        for (auto const& [windowSize, availableBlocks] : mAvailableBlocks)
        {
            auto const needed = mKvCacheManager.getRemainingBlocksToCompletion(req, windowSize, cachedSummary);
            mCachedNeededBlocks.emplace_back(windowSize, needed);
            if (needed > availableBlocks)
            {
                enough = false;
            }
        }
        return enough;
    }

    /// @brief  Subtract the block counts cached by the last enoughAvailableBlocks() call.
    void commitBlocks()
    {
        for (auto const& [windowSize, needed] : mCachedNeededBlocks)
        {
            mAvailableBlocks[windowSize] -= needed;
        }
        mCachedNeededBlocks.clear();
    }

private:
    BaseKVCacheManager const& mKvCacheManager;
    std::map<SizeType32, SizeType32> mAvailableBlocks;
    // Cache from last enoughAvailableBlocks call, consumed by commitBlocks
    std::vector<std::pair<SizeType32, SizeType32>> mCachedNeededBlocks;
};

class MaxUtilizationScheduledBlocksManager
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

public:
    MaxUtilizationScheduledBlocksManager(BaseKVCacheManager const& kvCacheManager, bool twoStepsLookAhead)
        : mKvCacheManager(kvCacheManager)
        , mTwoStepsLookAhead(twoStepsLookAhead)
    {
        auto const& windowSizes = mKvCacheManager.getBlockManager().getWindowSizesMetadata();
        for (auto const& [windowSize, _] : windowSizes)
        {
            mNumScheduledBlocks[windowSize] = 0;
        }
    }

    std::optional<std::map<SizeType32, SizeType32>> prepareNewNumberOfBlocksIfWeEndUpScheduling(
        LlmRequest const& req, std::optional<PrefixReuseSummary> const& cachedSummary = std::nullopt)
    {
        std::map<SizeType32, SizeType32> blocksIfScheduled;
        for (auto const& [windowSize, numScheduled] : mNumScheduledBlocks)
        {
            auto const required
                = mKvCacheManager.getNeededBlocksOneStep(req, mTwoStepsLookAhead, windowSize, cachedSummary);

            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu required blocks %i for %i window size",
                req.mRequestId, required, windowSize);

            auto const scheduledTotal = numScheduled + required;
            bool const hasFreeBlocks
                = mKvCacheManager.getBlockManager().schedulingHasFreeBlocks(scheduledTotal, windowSize);
            if (!hasFreeBlocks)
            {
                return std::nullopt;
            }
            blocksIfScheduled[windowSize] = scheduledTotal;
        }
        return blocksIfScheduled;
    }

    void updateScheduledBlocks(std::map<SizeType32, SizeType32> const& numBlocksIfScheduled)
    {
        assert(numBlocksIfScheduled.size() == mNumScheduledBlocks.size());
        for (auto const& [windowSize, blocksIfScheduled] : numBlocksIfScheduled)
        {
            TLLM_LOG_DEBUG(
                "MaxUtilizationScheduler: scheduled blocks %i for window size %i", blocksIfScheduled, windowSize);
            mNumScheduledBlocks.at(windowSize) = blocksIfScheduled;
        }
    }

private:
    BaseKVCacheManager const& mKvCacheManager;
    std::map<SizeType32, SizeType32> mNumScheduledBlocks;
    bool const mTwoStepsLookAhead;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
