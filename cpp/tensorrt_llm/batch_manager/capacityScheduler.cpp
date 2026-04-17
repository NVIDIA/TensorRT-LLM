/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/scheduledBlocksManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

namespace tensorrt_llm::batch_manager
{
using kv_cache_manager::VecUniqueTokens;
using kv_cache_manager::BlockKey;
using kv_cache_manager::BlockKeyHasher;

namespace
{

std::tuple<std::unordered_set<BlockKey, BlockKeyHasher>, std::unordered_set<BlockKey, BlockKeyHasher>>
prefillWithChunkedContextsAlreadyExecuting(RequestList const& activeRequests,
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager = std::nullopt)
{
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedContextBlocks;
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedCrossContextBlocks;
    for (auto const& req : activeRequests)
    {
        if (req->isContextInitState() && !req->isFirstContextChunk())
        {
            // Chunked context request already executing, but haven't completed all chunks yet.
            // Skipping is not an option, register it's contributed blocks
            if (kvCacheManager.isEnableBlockReuse())
            {
                auto uniqueTokens = req->getUniqueTokens(0);
                auto summary = kvCacheManager.analyzePrefixReuse(uniqueTokens, *req);
                if (summary.firstNewBlock.has_value())
                {
                    newlyContributedContextBlocks.insert(summary.firstNewBlock.value());
                }
            }
            if (crossKvCacheManager && crossKvCacheManager->isEnableBlockReuse())
            {
                auto uniqueTokens = *(req->getEncoderUniqueTokens().value());
                auto summary = crossKvCacheManager->analyzePrefixReuse(uniqueTokens, *req);
                if (summary.firstNewBlock.has_value())
                {
                    newlyContributedCrossContextBlocks.insert(summary.firstNewBlock.value());
                }
            }
        }
    }
    return {std::move(newlyContributedContextBlocks), std::move(newlyContributedCrossContextBlocks)};
}

/// @brief Check if a single manager's summary indicates we should skip this request.
/// @details Returns true if the request's first new context block was already contributed
/// by an earlier scheduled request (so waiting would let us reuse it). Does NOT mutate
/// the set — registration is deferred to beneficialToSkip after both KV checks pass.
bool oneManagerBeneficialToSkip(std::optional<kv_cache_manager::PrefixReuseSummary> const& summary,
    std::unordered_set<BlockKey, BlockKeyHasher> const& newlyContributedContextBlocks)
{
    return summary.has_value() && summary->firstNewBlock.has_value()
        && newlyContributedContextBlocks.count(summary->firstNewBlock.value()) > 0;
}

/// @brief Check if it is beneficial to skip this request rather than schedule it.
/// @details Returns true if this request can reuse KV cache block(s) that will be contributed
/// by already-scheduled context requests. Uses pre-computed PrefixReuseSummary values.
/// When the request is NOT skipped, registers its firstNewBlock contributions so that
/// subsequent duplicate requests can be deferred.
bool beneficialToSkip(std::optional<kv_cache_manager::PrefixReuseSummary> const& summary,
    std::optional<kv_cache_manager::PrefixReuseSummary> const& crossSummary,
    std::unordered_set<BlockKey, BlockKeyHasher>& newlyContributedContextBlocks,
    std::unordered_set<BlockKey, BlockKeyHasher>& newlyContributedCrossContextBlocks)
{
    if (oneManagerBeneficialToSkip(summary, newlyContributedContextBlocks))
    {
        return true;
    }
    if (oneManagerBeneficialToSkip(crossSummary, newlyContributedCrossContextBlocks))
    {
        return true;
    }
    // Request is NOT skipped — register its contributions so subsequent duplicate
    // requests can be deferred correctly.
    if (summary.has_value() && summary->firstNewBlock.has_value())
    {
        newlyContributedContextBlocks.insert(summary->firstNewBlock.value());
    }
    if (crossSummary.has_value() && crossSummary->firstNewBlock.has_value())
    {
        newlyContributedCrossContextBlocks.insert(crossSummary->firstNewBlock.value());
    }
    return false;
}

} // namespace

MaxRequestsScheduler::MaxRequestsScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
{
}

MaxUtilizationScheduler::MaxUtilizationScheduler(SizeType32 maxNumRequests, bool twoStepsLookAhead,
    LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
    , mTwoStepsLookAhead{twoStepsLookAhead}
{
}

GuaranteedNoEvictScheduler::GuaranteedNoEvictScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
{
}

StaticBatchScheduler::StaticBatchScheduler(
    SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
    : GuaranteedNoEvictScheduler(maxNumRequests, noScheduleUntilState, noScheduleAfterState)
{
}

std::tuple<RequestVector, RequestVector> MaxRequestsScheduler::operator()(RequestList const& activeRequests) const
{
    RequestVector scheduledRequests;
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState()))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }

        if (req->isEncoderInitState() || req->isContextInitState() || req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

std::tuple<RequestVector, RequestVector> StaticBatchScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    return this->impl<true>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
}

std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    return impl<false>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
}

template <bool StaticBatchScheduling>
std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::impl(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    RequestVector scheduledRequests;

    // Now check if we can add pending requests
    auto const maxPeftCachePages
        = peftCacheManager ? peftCacheManager->getMaxDevicePages() : std::numeric_limits<SizeType32>::max();

    // The optimization of delaying requests won't work for variable window attention
    bool skippingIsRelevant = (!kvCacheManager.getBlockManager().isVariableWindow())
        && (!crossKvCacheManager || !crossKvCacheManager->getBlockManager().isVariableWindow());

    // Keep track of blocks contributed by requests in context phase
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedContextBlocks;
    std::unordered_set<BlockKey, BlockKeyHasher> newlyContributedCrossContextBlocks;
    if constexpr (!StaticBatchScheduling)
    {
        if (skippingIsRelevant)
        {
            std::tie(newlyContributedContextBlocks, newlyContributedCrossContextBlocks)
                = prefillWithChunkedContextsAlreadyExecuting(activeRequests, kvCacheManager, crossKvCacheManager);
        }
    }

    // If a request is already in progress, include it
    // If it's been allocated, it had resource to run to completion
    // Also keep track of blocks needed to drive all in-progress requests to completion
    auto reservedBlocks = kv_cache_manager::NoEvictScheduledBlocksManager(kvCacheManager);
    auto reservedCrossBlocks = crossKvCacheManager
        ? std::optional(kv_cache_manager::NoEvictScheduledBlocksManager(*crossKvCacheManager))
        : std::nullopt;
    SizeType32 claimedPeftPages{0};
    std::unordered_set<uint64_t> uniqTaskIds{};
    RequestVector pendingRequests;
    RequestVector pendingDisGenInitRequests;
    pendingRequests.reserve(activeRequests.size());
    pendingDisGenInitRequests.reserve(activeRequests.size());
    for (auto const& req : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            continue;
        }

        if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
        {
            break;
        }

        if (req->isGenerationInProgressState())
        {
            scheduledRequests.emplace_back(req);
            reservedBlocks.enoughAvailableBlocks(*req);
            reservedBlocks.commitBlocks();
            if (reservedCrossBlocks)
            {
                reservedCrossBlocks->enoughAvailableBlocks(*req);
                reservedCrossBlocks->commitBlocks();
            }
            bool const reqHasLora = req->getLoraTaskId().has_value();
            bool const isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
            if (isNewTask)
            {
                claimedPeftPages += peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;
                uniqTaskIds.insert(req->getLoraTaskId().value());
            }
        }
        else if (req->isDisaggGenerationInitState())
        {
            pendingDisGenInitRequests.emplace_back(req);
        }
        else
        {
            pendingRequests.emplace_back(req);
        }
    }

    // If StaticBatchScheduling == true check if we can add pending requests only when no requests are active.
    // Otherwise, add just check that we can add pending requests.
    if (!StaticBatchScheduling || scheduledRequests.size() == 0)
    {
        // Now check if we can add pending requests
        auto availablePeftPages = maxPeftCachePages - claimedPeftPages;

        // Loop over pending requests and add them if they can be scheduled
        // Start by trying to include disagg generation init requests
        for (auto const& requests : {pendingDisGenInitRequests, pendingRequests})
        {
            for (auto const& req : requests)
            {
                // For first-chunk context requests with block reuse, compute the prefix reuse
                // summary once. This single radix tree walk serves both the beneficial-to-skip
                // check and the block budget estimation in getRemainingBlocksToCompletion,
                // eliminating 2 redundant walks per request.
                bool const isFirstChunkContext
                    = req->isContextInitState() && req->isFirstContextChunk() && !req->isDisaggGenerationInitState();
                std::optional<kv_cache_manager::PrefixReuseSummary> summary;
                std::optional<kv_cache_manager::PrefixReuseSummary> crossSummary;
                if (isFirstChunkContext)
                {
                    // analyzePrefixReuse asserts on variable-window managers; skip the walk there
                    // and let downstream callers fall back to their fresh tree-walk path.
                    if (kvCacheManager.isEnableBlockReuse() && !kvCacheManager.getBlockManager().isVariableWindow())
                    {
                        auto uniqueTokens = req->getUniqueTokens(0);
                        summary = kvCacheManager.analyzePrefixReuse(uniqueTokens, *req);
                    }
                    if (crossKvCacheManager && crossKvCacheManager->isEnableBlockReuse()
                        && !crossKvCacheManager->getBlockManager().isVariableWindow())
                    {
                        auto uniqueTokens = *(req->getEncoderUniqueTokens().value());
                        crossSummary = crossKvCacheManager->analyzePrefixReuse(uniqueTokens, *req);
                    }
                }

                // Beneficial-to-skip check using the cached summary
                if (!StaticBatchScheduling && skippingIsRelevant && isFirstChunkContext
                    && beneficialToSkip(
                        summary, crossSummary, newlyContributedContextBlocks, newlyContributedCrossContextBlocks))
                {
                    continue;
                }

                if (scheduledRequests.size() >= static_cast<std::size_t>(mMaxNumRequests))
                {
                    break;
                }

                if (req->isContextInitState() || req->isDisaggGenerationInitState())
                {
                    // Check block availability using the cached summary when available.
                    // enoughAvailableBlocks is check-only (no decrement) — safe if cross check fails.
                    bool enoughBlocks = reservedBlocks.enoughAvailableBlocks(*req, summary);
                    bool enoughCrossBlocks = true;
                    if (reservedCrossBlocks)
                    {
                        enoughCrossBlocks = reservedCrossBlocks->enoughAvailableBlocks(*req, crossSummary);
                    }
                    bool reqHasLora = req->getLoraTaskId().has_value();
                    bool isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
                    auto neededPeftPages = isNewTask && peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;

                    if (enoughBlocks && enoughCrossBlocks && neededPeftPages <= availablePeftPages)
                    {
                        scheduledRequests.emplace_back(req);
                        // Decrement using the cached values computed by enoughAvailableBlocks.
                        reservedBlocks.commitBlocks();
                        if (reservedCrossBlocks)
                        {
                            reservedCrossBlocks->commitBlocks();
                        }
                        availablePeftPages -= neededPeftPages;
                        if (isNewTask)
                        {
                            uniqTaskIds.insert(req->getLoraTaskId().value());
                        }
                    }
                    else if (!enoughBlocks || !enoughCrossBlocks)
                    {
                        // If one requests fails to be scheduled, break
                        break;
                    }
                }
            }
        }
    }
    return {std::move(scheduledRequests), RequestVector{}};
}

// TODO(nhaber): remove forward declare and just keep the function here, right before the merge. I put it below just so
// the remote diff is easier to look at/rebase conflicts
bool trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req, SizeType32 maxNumRequests,
    RequestVector& scheduledRequests, kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds,
    std::optional<kv_cache_manager::PrefixReuseSummary> const& cachedSummary);

std::tuple<RequestVector, RequestVector> MaxUtilizationScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager& kvCacheManager, OptionalRef<BasePeftCacheManager const> peftCacheManager,
    RequestList const& activeRequests) const
{
    kvCacheManager.startScheduling();

    // The optimization of delaying requests won't work for variable window attention
    bool skippingIsRelevant = !kvCacheManager.getBlockManager().isVariableWindow();

    // Keep track of number of requests and block needed for the scheduled requests
    auto scheduledBlocksManager
        = kv_cache_manager::MaxUtilizationScheduledBlocksManager(kvCacheManager, mTwoStepsLookAhead);
    SizeType32 numScheduledPeftPages{0};
    std::unordered_set<uint64_t> seenTaskIds;

    // Keep track of blocks contributed by requests in context phase
    auto [newlyContributedContextBlocks, newlyContributedCrossContextBlocks]
        = prefillWithChunkedContextsAlreadyExecuting(activeRequests, kvCacheManager);

    // Find last active in case we need to evict
    auto startedReqLambda = [this](std::shared_ptr<LlmRequest> const& req)
    {
        return (req->hasReachedState(getNoScheduleUntilState()) && !req->hasReachedState(getNoScheduleAfterState())
            && ((req->isContextInitState() && !req->isFirstContextChunk()) || req->isGenerationInProgressState()));
    };

    RequestVector scheduledRequests;
    RequestVector pausedRequests;
    auto reqItEnd = std::end(activeRequests);
    for (auto reqIt = std::begin(activeRequests); reqIt != reqItEnd;)
    {
        auto const& req = *reqIt;
        TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduling request ID %lu", req->mRequestId);

        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu cannot / should not be scheduled", req->mRequestId);
            reqIt++;
            continue;
        }

        // For first-chunk context requests with block reuse, compute the prefix reuse
        // summary once. This single radix tree walk serves both the beneficial-to-skip
        // check and the block budget estimation in getNeededBlocksOneStep.
        bool const isFirstChunkContext
            = req->isContextInitState() && req->isFirstContextChunk() && !req->isDisaggGenerationInitState();
        std::optional<kv_cache_manager::PrefixReuseSummary> summary;
        // analyzePrefixReuse asserts on variable-window managers; skip the walk there
        // and let downstream callers fall back to their fresh tree-walk path.
        if (isFirstChunkContext && kvCacheManager.isEnableBlockReuse()
            && !kvCacheManager.getBlockManager().isVariableWindow())
        {
            auto uniqueTokens = req->getUniqueTokens(0);
            summary = kvCacheManager.analyzePrefixReuse(uniqueTokens, *req);
        }

        // Beneficial-to-skip check using the cached summary (no cross KV cache for MaxUtil)
        if (skippingIsRelevant && isFirstChunkContext
            && beneficialToSkip(
                summary, std::nullopt, newlyContributedContextBlocks, newlyContributedCrossContextBlocks))
        {
            reqIt++;
            continue;
        }

        bool const wasScheduled = trySchedulingRequestMaxUtilization(req, mMaxNumRequests, scheduledRequests,
            scheduledBlocksManager, peftCacheManager, numScheduledPeftPages, seenTaskIds, summary);
        if (wasScheduled)
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu -> start", req->mRequestId);
            reqIt++;
        }
        else
        {
            auto const rbegin = std::reverse_iterator(reqItEnd);
            auto const rend = std::reverse_iterator(reqIt);
            auto const lastStartedReqIt = std::find_if(rbegin, rend, startedReqLambda);
            if (lastStartedReqIt != rend)
            {
                // If we can't allocate a started request, we need to start freeing started requests
                // from the end of the vector and try again
                // Here we simulate freeing the kvCache blocks associated with that sequence
                kvCacheManager.schedulingRemoveSequence((*lastStartedReqIt)->mRequestId);
                pausedRequests.emplace_back(*lastStartedReqIt);
                TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu -> pause", (*lastStartedReqIt)->mRequestId);
                reqItEnd = std::next(lastStartedReqIt).base();
            }
            else
            {
                break;
            }
        }
    }

    return {std::move(scheduledRequests), std::move(pausedRequests)};
}

bool trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req, SizeType32 maxNumRequests,
    RequestVector& scheduledRequests, kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds, std::optional<kv_cache_manager::PrefixReuseSummary> const& cachedSummary)
{
    if (scheduledRequests.size() < static_cast<std::size_t>(maxNumRequests))
    {
        bool reqHasLora = req->getLoraTaskId().has_value();
        bool isNewTask = reqHasLora && !seenTaskIds.count(req->getLoraTaskId().value());
        SizeType32 numRequiredPeftPages
            = (isNewTask && peftCacheManager) ? peftCacheManager->determineNumPages(req) : 0;
        TLLM_LOG_DEBUG(
            "MaxUtilizationScheduler: request ID %lu required peft pages: %i", req->mRequestId, numRequiredPeftPages);
        // Use the cached summary when available to avoid a redundant tree walk
        auto const scheduledBlocksIfFitsKvCache
            = blocksManager.prepareNewNumberOfBlocksIfWeEndUpScheduling(*req, cachedSummary);
        bool fitsPeft
            = (peftCacheManager ? numRequiredPeftPages + numScheduledPeftPages <= peftCacheManager->getMaxDevicePages()
                                : true);

        if (scheduledBlocksIfFitsKvCache && fitsPeft)
        {
            blocksManager.updateScheduledBlocks(scheduledBlocksIfFitsKvCache.value());
            numScheduledPeftPages += numRequiredPeftPages;
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduled peft pages: %i", numRequiredPeftPages);
            scheduledRequests.emplace_back(req);
            if (isNewTask)
            {
                seenTaskIds.insert(req->getLoraTaskId().value());
            }
            return true;
        }
    }
    return false;
}

CapacityScheduler::CapacityScheduler(SizeType32 maxNumRequests,
    executor::CapacitySchedulerPolicy capacitySchedulerPolicy, bool hasKvCacheManager, bool twoStepsLookAhead,
    LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
{
    if (!hasKvCacheManager)
    {
        mScheduler = MaxRequestsScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kMAX_UTILIZATION)
    {
        mScheduler
            = MaxUtilizationScheduler{maxNumRequests, twoStepsLookAhead, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT)
    {
        mScheduler = GuaranteedNoEvictScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else if (capacitySchedulerPolicy == executor::CapacitySchedulerPolicy::kSTATIC_BATCH)
    {
        mScheduler = StaticBatchScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState};
    }
    else
    {
        throw std::runtime_error("Unsupported capacity scheduler policy");
    }
}

std::tuple<RequestVector, RequestVector, RequestVector> CapacityScheduler::operator()(RequestList const& activeRequests,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);
    return std::visit(
        [&activeRequests, &kvCacheManager, &crossKvCacheManager, &peftCacheManager](
            auto const& scheduler) -> std::tuple<RequestVector, RequestVector, RequestVector>
        {
            RequestVector tmpFittingRequests;
            RequestVector pausedRequests;
            if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxRequestsScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests) = scheduler(activeRequests);
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxUtilizationScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, peftCacheManager, activeRequests);
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, GuaranteedNoEvictScheduler>
                || std::is_same_v<std::decay_t<decltype(scheduler)>, StaticBatchScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
            }
            else
            {
                throw std::runtime_error("Unsupported capacity scheduler policy");
            }
            TLLM_LOG_DEBUG("[Summary] Capacity scheduler allows %d requests, pauses %d requests",
                tmpFittingRequests.size(), pausedRequests.size());

            RequestVector fittingRequests;
            RequestVector fittingDisaggGenInitRequests;
            for (auto const& llmReq : tmpFittingRequests)
            {
                if (llmReq->isDisaggGenerationInitState())
                {
                    fittingDisaggGenInitRequests.push_back(llmReq);
                }
                else
                {
                    fittingRequests.push_back(llmReq);
                }
            }

            return {std::move(fittingRequests), std::move(fittingDisaggGenInitRequests), std::move(pausedRequests)};
        },
        mScheduler);
}

} // namespace tensorrt_llm::batch_manager
