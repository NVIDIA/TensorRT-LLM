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
#include "tensorrt_llm/batch_manager/agentTree.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/scheduledBlocksManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{
using kv_cache_manager::VecUniqueTokens;
using kv_cache_manager::BlockKey;
using kv_cache_manager::BlockKeyHasher;

struct GuaranteedNoEvictDecisionTrace
{
    std::uint64_t transferActive{0};
    bool logicalCapacityReached{false};
    std::unordered_map<RequestIdType, std::uint8_t> rejectionReasons;
};

struct GuaranteedNoEvictReplayStats
{
    ~GuaranteedNoEvictReplayStats()
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=summary mode=complete_first_gate_replay "
            "iterations=%llu shadow_iterations=%llu treatment_only_iterations=%llu shadow_stopped=%d "
            "transfer_active_iterations=%llu transfer_active_total=%llu transfer_active_max=%llu "
            "divergence_iterations=%llu post_candidates_total=%llu pre_candidates_total=%llu "
            "pre_only_total=%llu pre_only_disagg_init_total=%llu pre_only_ready_generation_total=%llu "
            "pre_only_other_total=%llu post_only_total=%llu reason_logical_total=%llu reason_self_kv_total=%llu "
            "reason_cross_kv_total=%llu reason_peft_total=%llu reason_unknown_total=%llu",
            static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(shadowIterations),
            static_cast<unsigned long long>(treatmentOnlyIterations), static_cast<int>(shadowStopped),
            static_cast<unsigned long long>(transferActiveIterations),
            static_cast<unsigned long long>(transferActiveTotal), static_cast<unsigned long long>(transferActiveMax),
            static_cast<unsigned long long>(divergenceIterations), static_cast<unsigned long long>(postCandidatesTotal),
            static_cast<unsigned long long>(preCandidatesTotal), static_cast<unsigned long long>(preOnlyTotal),
            static_cast<unsigned long long>(preOnlyDisaggInitTotal),
            static_cast<unsigned long long>(preOnlyReadyGenerationTotal),
            static_cast<unsigned long long>(preOnlyOtherTotal), static_cast<unsigned long long>(postOnlyTotal),
            static_cast<unsigned long long>(reasonLogicalTotal), static_cast<unsigned long long>(reasonSelfKvTotal),
            static_cast<unsigned long long>(reasonCrossKvTotal), static_cast<unsigned long long>(reasonPeftTotal),
            static_cast<unsigned long long>(reasonUnknownTotal));
    }

    std::uint64_t iterations{0};
    std::uint64_t shadowIterations{0};
    std::uint64_t treatmentOnlyIterations{0};
    std::uint64_t transferActiveIterations{0};
    std::uint64_t transferActiveTotal{0};
    std::uint64_t transferActiveMax{0};
    std::uint64_t divergenceIterations{0};
    std::uint64_t postCandidatesTotal{0};
    std::uint64_t preCandidatesTotal{0};
    std::uint64_t preOnlyTotal{0};
    std::uint64_t preOnlyDisaggInitTotal{0};
    std::uint64_t preOnlyReadyGenerationTotal{0};
    std::uint64_t preOnlyOtherTotal{0};
    std::uint64_t postOnlyTotal{0};
    std::uint64_t reasonLogicalTotal{0};
    std::uint64_t reasonSelfKvTotal{0};
    std::uint64_t reasonCrossKvTotal{0};
    std::uint64_t reasonPeftTotal{0};
    std::uint64_t reasonUnknownTotal{0};
    bool transferActiveLogged{false};
    bool divergenceLogged{false};
    bool noTransferShadowSkipLogged{false};
    bool shadowStopped{false};
};

namespace
{

constexpr char kReplayPrePr15356GneFirstGateEnv[] = "TRTLLM_NVBUG_6448152_REPLAY_PRE15356_GNE_FIRST_GATE";

constexpr std::uint8_t kAdmissionReasonLogical = 1U << 0U;
constexpr std::uint8_t kAdmissionReasonSelfKv = 1U << 1U;
constexpr std::uint8_t kAdmissionReasonCrossKv = 1U << 2U;
constexpr std::uint8_t kAdmissionReasonPeft = 1U << 3U;

void addAdmissionReason(
    GuaranteedNoEvictDecisionTrace* trace, std::shared_ptr<LlmRequest> const& req, std::uint8_t reason)
{
    if (trace != nullptr)
    {
        trace->rejectionReasons[req->mRequestId] |= reason;
    }
}

bool hasRequest(RequestVector const& requests, RequestIdType requestId)
{
    return std::any_of(
        requests.begin(), requests.end(), [requestId](auto const& req) { return req->mRequestId == requestId; });
}

void recordReplayComparison(GuaranteedNoEvictReplayStats& stats, RequestVector const& postPrCandidates,
    RequestVector const& prePrCandidates, GuaranteedNoEvictDecisionTrace const& postPrTrace)
{
    std::uint64_t preOnly{0};
    std::uint64_t preOnlyDisaggInit{0};
    std::uint64_t preOnlyReadyGeneration{0};
    std::uint64_t preOnlyOther{0};
    std::uint64_t postOnly{0};
    std::uint64_t reasonLogical{0};
    std::uint64_t reasonSelfKv{0};
    std::uint64_t reasonCrossKv{0};
    std::uint64_t reasonPeft{0};
    std::uint64_t reasonUnknown{0};

    for (auto const& req : prePrCandidates)
    {
        if (hasRequest(postPrCandidates, req->mRequestId))
        {
            continue;
        }
        ++preOnly;
        if (req->isDisaggGenerationInitState())
        {
            ++preOnlyDisaggInit;
        }
        else if (req->isGenerationInProgressState())
        {
            ++preOnlyReadyGeneration;
        }
        else
        {
            ++preOnlyOther;
        }
        auto reason = std::uint8_t{0};
        auto const reasonIt = postPrTrace.rejectionReasons.find(req->mRequestId);
        if (reasonIt != postPrTrace.rejectionReasons.end())
        {
            reason = reasonIt->second;
        }
        if (reason == 0 && postPrTrace.logicalCapacityReached)
        {
            reason = kAdmissionReasonLogical;
        }
        reasonLogical += static_cast<std::uint64_t>((reason & kAdmissionReasonLogical) != 0);
        reasonSelfKv += static_cast<std::uint64_t>((reason & kAdmissionReasonSelfKv) != 0);
        reasonCrossKv += static_cast<std::uint64_t>((reason & kAdmissionReasonCrossKv) != 0);
        reasonPeft += static_cast<std::uint64_t>((reason & kAdmissionReasonPeft) != 0);
        reasonUnknown += static_cast<std::uint64_t>(reason == 0);
    }
    for (auto const& req : postPrCandidates)
    {
        postOnly += static_cast<std::uint64_t>(!hasRequest(prePrCandidates, req->mRequestId));
    }

    auto postIt = postPrCandidates.begin();
    for (auto const& req : prePrCandidates)
    {
        if (postIt != postPrCandidates.end() && (*postIt)->mRequestId == req->mRequestId)
        {
            ++postIt;
        }
    }
    bool const postIsOrderedSubset = postIt == postPrCandidates.end();
    TLLM_CHECK_WITH_INFO(postOnly == 0 && postIsOrderedSubset,
        "Complete pre-PR15356 first-gate replay must be an ordered relaxation of post-PR15356 admission");

    ++stats.iterations;
    ++stats.shadowIterations;
    stats.transferActiveTotal += postPrTrace.transferActive;
    stats.transferActiveMax = std::max(stats.transferActiveMax, postPrTrace.transferActive);
    stats.transferActiveIterations += static_cast<std::uint64_t>(postPrTrace.transferActive > 0);
    stats.divergenceIterations += static_cast<std::uint64_t>(preOnly > 0);
    stats.postCandidatesTotal += postPrCandidates.size();
    stats.preCandidatesTotal += prePrCandidates.size();
    stats.preOnlyTotal += preOnly;
    stats.preOnlyDisaggInitTotal += preOnlyDisaggInit;
    stats.preOnlyReadyGenerationTotal += preOnlyReadyGeneration;
    stats.preOnlyOtherTotal += preOnlyOther;
    stats.postOnlyTotal += postOnly;
    stats.reasonLogicalTotal += reasonLogical;
    stats.reasonSelfKvTotal += reasonSelfKv;
    stats.reasonCrossKvTotal += reasonCrossKv;
    stats.reasonPeftTotal += reasonPeft;
    stats.reasonUnknownTotal += reasonUnknown;

    if (postPrTrace.transferActive > 0 && !stats.transferActiveLogged)
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=transfer_active mode=complete_first_gate_replay "
            "transfer_active=%llu post_candidates=%llu pre_candidates=%llu",
            static_cast<unsigned long long>(postPrTrace.transferActive),
            static_cast<unsigned long long>(postPrCandidates.size()),
            static_cast<unsigned long long>(prePrCandidates.size()));
        stats.transferActiveLogged = true;
    }
    if (preOnly > 0 && !stats.divergenceLogged)
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=exercised mode=complete_first_gate_replay "
            "transfer_active=%llu post_candidates=%llu pre_candidates=%llu pre_only=%llu post_only=%llu "
            "pre_only_disagg_init=%llu pre_only_ready_generation=%llu pre_only_other=%llu "
            "post_is_ordered_subset=%d reason_logical=%llu reason_self_kv=%llu reason_cross_kv=%llu "
            "reason_peft=%llu reason_unknown=%llu",
            static_cast<unsigned long long>(postPrTrace.transferActive),
            static_cast<unsigned long long>(postPrCandidates.size()),
            static_cast<unsigned long long>(prePrCandidates.size()), static_cast<unsigned long long>(preOnly),
            static_cast<unsigned long long>(postOnly), static_cast<unsigned long long>(preOnlyDisaggInit),
            static_cast<unsigned long long>(preOnlyReadyGeneration), static_cast<unsigned long long>(preOnlyOther),
            static_cast<int>(postIsOrderedSubset), static_cast<unsigned long long>(reasonLogical),
            static_cast<unsigned long long>(reasonSelfKv), static_cast<unsigned long long>(reasonCrossKv),
            static_cast<unsigned long long>(reasonPeft), static_cast<unsigned long long>(reasonUnknown));
        stats.divergenceLogged = true;
    }
    if (postPrTrace.transferActive > 0 && preOnly > 0 && !stats.shadowStopped)
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=shadow_stopped mode=complete_first_gate_replay "
            "shadow_iterations=%llu treatment_continues=1",
            static_cast<unsigned long long>(stats.shadowIterations));
        stats.shadowStopped = true;
    }
}

void recordTreatmentOnlyIteration(GuaranteedNoEvictReplayStats& stats, std::uint64_t transferActive)
{
    ++stats.iterations;
    ++stats.treatmentOnlyIterations;
    stats.transferActiveTotal += transferActive;
    stats.transferActiveMax = std::max(stats.transferActiveMax, transferActive);
    stats.transferActiveIterations += static_cast<std::uint64_t>(transferActive > 0);
    if (transferActive == 0 && !stats.noTransferShadowSkipLogged)
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=shadow_skipped mode=complete_first_gate_replay "
            "reason=no_transfer shadow_iterations=%llu treatment_continues=1",
            static_cast<unsigned long long>(stats.shadowIterations));
        stats.noTransferShadowSkipLogged = true;
    }
}

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

template <typename KVCacheManagerT>
void checkRequiredCrossKvCacheManager(
    LlmRequestState noScheduleUntilState, OptionalRef<KVCacheManagerT> crossKvCacheManager)
{
    if (noScheduleUntilState != LlmRequestState::kENCODER_INIT)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(
        static_cast<bool>(crossKvCacheManager), "Encoder-decoder scheduling requires a cross_kv_cache_manager.");
}

void claimPeftPagesForRequest(std::shared_ptr<LlmRequest> const& req,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& claimedPeftPages,
    std::unordered_set<uint64_t>& uniqTaskIds)
{
    bool const reqHasLora = req->getLoraTaskId().has_value();
    bool const isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
    if (isNewTask)
    {
        claimedPeftPages += peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;
        uniqTaskIds.insert(req->getLoraTaskId().value());
    }
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

GuaranteedNoEvictScheduler::GuaranteedNoEvictScheduler(SizeType32 maxNumRequests, LlmRequestState noScheduleUntilState,
    LlmRequestState noScheduleAfterState, bool replayPrePr15356FirstGate)
    : BaseCapacityScheduler(noScheduleUntilState, noScheduleAfterState)
    , mMaxNumRequests(maxNumRequests)
    , mReplayPrePr15356FirstGate(replayPrePr15356FirstGate)
    , mReplayStats(replayPrePr15356FirstGate ? std::make_shared<GuaranteedNoEvictReplayStats>() : nullptr)
{
    if (mReplayPrePr15356FirstGate)
    {
        TLLM_LOG_INFO(
            "NVBUG6448152_PRE15356_GNE event=effective_config policy=guaranteed_no_evict "
            "mode=complete_first_gate_replay post_shadow=transfer_active_until_first_exercised_divergence "
            "ignore_transfer_logical=1 "
            "ignore_transfer_self_kv=1 ignore_transfer_cross_kv=1 ignore_transfer_peft=1 "
            "max_num_requests=%d",
            static_cast<int>(mMaxNumRequests));
    }
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
    if (!mReplayPrePr15356FirstGate)
    {
        return impl<false>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests);
    }
    auto prePrResult = impl<false>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests, true);
    auto const transferActive = static_cast<std::uint64_t>(std::count_if(activeRequests.begin(), activeRequests.end(),
        [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); }));
    if (transferActive == 0 || mReplayStats->shadowStopped)
    {
        recordTreatmentOnlyIteration(*mReplayStats, transferActive);
        return prePrResult;
    }

    GuaranteedNoEvictDecisionTrace postPrTrace;
    auto postPrResult
        = impl<false>(kvCacheManager, crossKvCacheManager, peftCacheManager, activeRequests, false, &postPrTrace);
    TLLM_CHECK_WITH_INFO(postPrTrace.transferActive == transferActive,
        "Transfer-active precondition count must match the post-PR15356 decision trace");
    recordReplayComparison(*mReplayStats, std::get<0>(postPrResult), std::get<0>(prePrResult), postPrTrace);
    return prePrResult;
}

template <bool StaticBatchScheduling>
std::tuple<RequestVector, RequestVector> GuaranteedNoEvictScheduler::impl(
    kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests,
    bool replayPrePr15356FirstGate, GuaranteedNoEvictDecisionTrace* trace) const
{
    RequestVector scheduledRequests;

    checkRequiredCrossKvCacheManager(getNoScheduleUntilState(), crossKvCacheManager);

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
    std::size_t numAdmittedRequests{0};
    RequestVector pendingRequests;
    RequestVector pendingDisGenInitRequests;
    pendingRequests.reserve(activeRequests.size());
    pendingDisGenInitRequests.reserve(activeRequests.size());
    for (auto const& req : activeRequests)
    {
        bool const isDisaggGenerationTransfer = req->isDisaggGenerationTransmissionInProgress();
        if (trace != nullptr && isDisaggGenerationTransfer)
        {
            ++trace->transferActive;
        }
        if (replayPrePr15356FirstGate && isDisaggGenerationTransfer)
        {
            // Before PR #15356, transfer-only requests were invisible to the
            // capacity scheduler: they consumed neither logical request slots
            // nor self-KV, cross-KV, or PEFT capacity.
            continue;
        }

        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState() && !req->isDisaggGenerationTransmissionInProgress()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            continue;
        }

        if (numAdmittedRequests >= static_cast<std::size_t>(mMaxNumRequests))
        {
            if (trace != nullptr)
            {
                trace->logicalCapacityReached = true;
                addAdmissionReason(trace, req, kAdmissionReasonLogical);
            }
            break;
        }

        if (req->isDisaggGenerationTransmissionInProgress() || req->isGenerationInProgressState())
        {
            ++numAdmittedRequests;
            if (req->isGenerationInProgressState())
            {
                scheduledRequests.emplace_back(req);
            }
            reservedBlocks.enoughAvailableBlocks(*req);
            reservedBlocks.commitBlocks();
            if (reservedCrossBlocks)
            {
                reservedCrossBlocks->enoughAvailableBlocks(*req);
                reservedCrossBlocks->commitBlocks();
            }
            claimPeftPagesForRequest(req, peftCacheManager, claimedPeftPages, uniqTaskIds);
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
    if (!StaticBatchScheduling || numAdmittedRequests == 0)
    {
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
                // Encoder-init requests do not consume self- or cross-KV
                // blocks. We still keep the cross reuse summary available for
                // beneficial-to-skip so duplicate encoder inputs can be ordered
                // consistently before their decoder-context admission budgets
                // the cross pool.
                bool const isEncoderInit = req->isEncoderInitState();
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
                else if (isEncoderInit && crossKvCacheManager && crossKvCacheManager->isEnableBlockReuse()
                    && !crossKvCacheManager->getBlockManager().isVariableWindow())
                {
                    // Encoder admission only needs the cross summary for reuse ordering.
                    auto uniqueTokens = *(req->getEncoderUniqueTokens().value());
                    crossSummary = crossKvCacheManager->analyzePrefixReuse(uniqueTokens, *req);
                }
                // Beneficial-to-skip check using the cached summary
                if (!StaticBatchScheduling && skippingIsRelevant && (isFirstChunkContext || isEncoderInit)
                    && beneficialToSkip(
                        summary, crossSummary, newlyContributedContextBlocks, newlyContributedCrossContextBlocks))
                {
                    continue;
                }

                if (numAdmittedRequests >= static_cast<std::size_t>(mMaxNumRequests))
                {
                    if (trace != nullptr)
                    {
                        trace->logicalCapacityReached = true;
                        addAdmissionReason(trace, req, kAdmissionReasonLogical);
                    }
                    break;
                }

                if (isEncoderInit)
                {
                    bool reqHasLora = req->getLoraTaskId().has_value();
                    bool isNewTask = reqHasLora && !uniqTaskIds.count(req->getLoraTaskId().value());
                    auto neededPeftPages = isNewTask && peftCacheManager ? peftCacheManager->determineNumPages(req) : 0;

                    if (neededPeftPages <= availablePeftPages)
                    {
                        scheduledRequests.emplace_back(req);
                        ++numAdmittedRequests;
                        availablePeftPages -= neededPeftPages;
                        if (isNewTask)
                        {
                            uniqTaskIds.insert(req->getLoraTaskId().value());
                        }
                    }
                    else
                    {
                        addAdmissionReason(trace, req, kAdmissionReasonPeft);
                    }
                }
                else if (req->isContextInitState() || req->isDisaggGenerationInitState())
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

                    if (!enoughBlocks)
                    {
                        addAdmissionReason(trace, req, kAdmissionReasonSelfKv);
                    }
                    if (!enoughCrossBlocks)
                    {
                        addAdmissionReason(trace, req, kAdmissionReasonCrossKv);
                    }
                    if (neededPeftPages > availablePeftPages)
                    {
                        addAdmissionReason(trace, req, kAdmissionReasonPeft);
                    }

                    if (enoughBlocks && enoughCrossBlocks && neededPeftPages <= availablePeftPages)
                    {
                        scheduledRequests.emplace_back(req);
                        ++numAdmittedRequests;
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
    std::size_t& numAdmittedRequests, RequestVector& scheduledRequests,
    kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    std::optional<kv_cache_manager::MaxUtilizationScheduledBlocksManager>& crossBlocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds,
    std::optional<kv_cache_manager::PrefixReuseSummary> const& cachedSummary);

std::tuple<RequestVector, RequestVector> MaxUtilizationScheduler::operator()(
    kv_cache_manager::BaseKVCacheManager& kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const
{
    checkRequiredCrossKvCacheManager(getNoScheduleUntilState(), crossKvCacheManager);

    kvCacheManager.startScheduling();
    if (crossKvCacheManager)
    {
        crossKvCacheManager->startScheduling();
    }

    // The optimization of delaying requests won't work for variable window attention
    bool skippingIsRelevant = !kvCacheManager.getBlockManager().isVariableWindow();

    // Keep track of number of requests and block needed for the scheduled requests
    auto scheduledBlocksManager
        = kv_cache_manager::MaxUtilizationScheduledBlocksManager(kvCacheManager, mTwoStepsLookAhead);
    // Mirror the budget tracker for the cross pool when present.
    // Encoder-init requests do not consume either tracker; decoder
    // context/generation requests update both trackers in lockstep.
    std::optional<kv_cache_manager::MaxUtilizationScheduledBlocksManager> scheduledCrossBlocksManager;
    if (crossKvCacheManager)
    {
        scheduledCrossBlocksManager.emplace(*crossKvCacheManager, mTwoStepsLookAhead);
    }
    SizeType32 numScheduledPeftPages{0};
    std::unordered_set<uint64_t> seenTaskIds;

    // Keep track of blocks contributed by requests in context phase
    auto [newlyContributedContextBlocks, newlyContributedCrossContextBlocks]
        = prefillWithChunkedContextsAlreadyExecuting(activeRequests, kvCacheManager);

    // Find last active in case we need to evict.  Encoder-init requests are
    // intentionally excluded here: they hold no started self- or cross-pool
    // blocks, so pausing them would not free any KV budget.
    auto startedReqLambda = [this](std::shared_ptr<LlmRequest> const& req)
    {
        return (req->hasReachedState(getNoScheduleUntilState()) && !req->hasReachedState(getNoScheduleAfterState())
            && ((req->isContextInitState() && !req->isFirstContextChunk()) || req->isGenerationInProgressState()));
    };

    RequestVector scheduledRequests;
    RequestVector pausedRequests;
    std::size_t numAdmittedRequests{0};
    auto reqItEnd = std::end(activeRequests);
    for (auto reqIt = std::begin(activeRequests); reqIt != reqItEnd;)
    {
        auto const& req = *reqIt;
        TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduling request ID %lu", req->mRequestId);

        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (
            // Allow disagg_generation_init requests to be scheduled, so that we'll allocate their KV cache
            !req->isDisaggGenerationInitState() && !req->isDisaggGenerationTransmissionInProgress()
            && (!req->hasReachedState(getNoScheduleUntilState()) || req->hasReachedState(getNoScheduleAfterState())))
        {
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: request ID %lu cannot / should not be scheduled", req->mRequestId);
            reqIt++;
            continue;
        }

        if (req->isDisaggGenerationTransmissionInProgress())
        {
            if (numAdmittedRequests >= static_cast<std::size_t>(mMaxNumRequests))
            {
                break;
            }
            claimPeftPagesForRequest(req, peftCacheManager, numScheduledPeftPages, seenTaskIds);
            ++numAdmittedRequests;
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

        bool const wasScheduled = trySchedulingRequestMaxUtilization(req, mMaxNumRequests, numAdmittedRequests,
            scheduledRequests, scheduledBlocksManager, scheduledCrossBlocksManager, peftCacheManager,
            numScheduledPeftPages, seenTaskIds, summary);
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
                if (crossKvCacheManager)
                {
                    // Mirror self-pool eviction on the cross pool so any cross
                    // blocks held by the paused request are released for reuse
                    // by other admissions in this iteration.
                    crossKvCacheManager->schedulingRemoveSequence((*lastStartedReqIt)->mRequestId);
                }
                pausedRequests.emplace_back(*lastStartedReqIt);
                TLLM_LOG_INFO("MaxUtilizationScheduler: request ID %lu -> pause", (*lastStartedReqIt)->mRequestId);
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
    std::size_t& numAdmittedRequests, RequestVector& scheduledRequests,
    kv_cache_manager::MaxUtilizationScheduledBlocksManager& blocksManager,
    std::optional<kv_cache_manager::MaxUtilizationScheduledBlocksManager>& crossBlocksManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager, SizeType32& numScheduledPeftPages,
    std::unordered_set<uint64_t>& seenTaskIds, std::optional<kv_cache_manager::PrefixReuseSummary> const& cachedSummary)
{
    if (numAdmittedRequests < static_cast<std::size_t>(maxNumRequests))
    {
        bool reqHasLora = req->getLoraTaskId().has_value();
        bool isNewTask = reqHasLora && !seenTaskIds.count(req->getLoraTaskId().value());
        SizeType32 numRequiredPeftPages
            = (isNewTask && peftCacheManager) ? peftCacheManager->determineNumPages(req) : 0;
        TLLM_LOG_DEBUG(
            "MaxUtilizationScheduler: request ID %lu required peft pages: %i", req->mRequestId, numRequiredPeftPages);
        bool fitsPeft
            = (peftCacheManager ? numRequiredPeftPages + numScheduledPeftPages <= peftCacheManager->getMaxDevicePages()
                                : true);

        if (req->isEncoderInitState())
        {
            // Encoder admission does not reserve KV blocks. The scheduler
            // entry point verifies the cross manager globally before encoder
            // work can be admitted.
            if (fitsPeft)
            {
                numScheduledPeftPages += numRequiredPeftPages;
                scheduledRequests.emplace_back(req);
                ++numAdmittedRequests;
                if (isNewTask)
                {
                    seenTaskIds.insert(req->getLoraTaskId().value());
                }
                return true;
            }
            return false;
        }

        // Use the cached summary when available to avoid a redundant tree walk
        auto const scheduledBlocksIfFitsKvCache
            = blocksManager.prepareNewNumberOfBlocksIfWeEndUpScheduling(*req, cachedSummary);
        // Context/generation requests must fit in both pools when a cross
        // manager is present.  Self-pool fit is checked first so that the
        // budget probe is cheap when self is already saturated.
        std::optional<std::map<SizeType32, SizeType32>> crossScheduledIfFits;
        if (crossBlocksManager)
        {
            crossScheduledIfFits = crossBlocksManager->prepareNewNumberOfBlocksIfWeEndUpScheduling(*req);
            if (!crossScheduledIfFits)
            {
                return false;
            }
        }

        if (scheduledBlocksIfFitsKvCache && fitsPeft)
        {
            blocksManager.updateScheduledBlocks(scheduledBlocksIfFitsKvCache.value());
            if (crossScheduledIfFits)
            {
                crossBlocksManager->updateScheduledBlocks(crossScheduledIfFits.value());
            }
            numScheduledPeftPages += numRequiredPeftPages;
            TLLM_LOG_DEBUG("MaxUtilizationScheduler: scheduled peft pages: %i", numRequiredPeftPages);
            scheduledRequests.emplace_back(req);
            ++numAdmittedRequests;
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
        mScheduler = GuaranteedNoEvictScheduler{maxNumRequests, noScheduleUntilState, noScheduleAfterState,
            common::getBoolEnv(kReplayPrePr15356GneFirstGateEnv)};
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

void CapacityScheduler::setAgentTreeReorderPolicy(
    float agentPercentage, std::optional<std::vector<std::string>> agentTypes, SizeType32 agentInflightSeqNum)
{
    batch_scheduler::AgentTreeConfig config;
    config.agentPercentage = agentPercentage;
    config.agentTypes = std::move(agentTypes);
    config.agentInflightSeqNum = agentInflightSeqNum;

    mReorderPolicy = std::make_unique<agent_tree::AgentTreePolicy>(std::move(config));
}

std::tuple<RequestVector, RequestVector, RequestVector> CapacityScheduler::operator()(RequestList const& activeRequests,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    OptionalRef<BasePeftCacheManager const> peftCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager) const
{
    NVTX3_SCOPED_RANGE(capacitySchedulerScheduling);

    // Apply reorder policy if set
    RequestList requestsToSchedule = mReorderPolicy ? mReorderPolicy->reorderRequests(activeRequests) : activeRequests;

    return std::visit(
        [&requestsToSchedule, &kvCacheManager, &crossKvCacheManager, &peftCacheManager](
            auto const& scheduler) -> std::tuple<RequestVector, RequestVector, RequestVector>
        {
            RequestVector tmpFittingRequests;
            RequestVector pausedRequests;
            if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxRequestsScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests) = scheduler(requestsToSchedule);
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, MaxUtilizationScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, crossKvCacheManager, peftCacheManager, requestsToSchedule);
            }
            else if constexpr (std::is_same_v<std::decay_t<decltype(scheduler)>, GuaranteedNoEvictScheduler>
                || std::is_same_v<std::decay_t<decltype(scheduler)>, StaticBatchScheduler>)
            {
                std::tie(tmpFittingRequests, pausedRequests)
                    = scheduler(*kvCacheManager, crossKvCacheManager, peftCacheManager, requestsToSchedule);
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
