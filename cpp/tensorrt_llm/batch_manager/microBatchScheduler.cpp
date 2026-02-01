/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager
{

using SizeType32 = MicroBatchScheduler::SizeType32;

MicroBatchScheduler::MicroBatchScheduler(std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig,
    std::optional<SizeType32> maxContextLength, LlmRequestState noScheduleUntilState,
    LlmRequestState noScheduleAfterState)
    : mMaxContextLength(maxContextLength)
    , mCtxChunkConfig(ctxChunkConfig)
    , mNoScheduleUntilState(noScheduleUntilState)
    , mNoScheduleAfterState(noScheduleAfterState)
{
}

void MicroBatchScheduler::fitDraftTokens(RequestVector& contextsToBeChunked,
    std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    // How many compute tokens (chunk - reusable) are in this batch already?
    SizeType32 numCtxTokens{0};
    for (auto const& llmReq : contextsToBeChunked)
    {
        SizeType32 const chunkSize = llmReq->getContextChunkSize();
        // contextRemaining = P for first chunk; used to compute actual model token count.
        SizeType32 const contextRemaining = llmReq->getContextRemainingLength();
        SizeType32 const reusable
            = llmReq->isFirstContextChunk() ? std::min(llmReq->getEstimatedReusableTokens(), contextRemaining) : 0;
        numCtxTokens += std::min(chunkSize, std::max<SizeType32>(0, contextRemaining - reusable));
    }

    // Discard draft tokens that won't fit into the existing chunk unit, max
    // context length, or token capacity.
    for (auto& llmReq : contextsToBeChunked)
    {
        if (llmReq->isLastContextChunk() && llmReq->hasDraftTokens())
        {
            // How many more tokens could fit into this chunkUnit? (Round up to next multiple of chunkUnitSize)
            // Each chunkUnit requires an additional kvcache block, so we don't want to use an extra one just for draft
            // tokens.
            SizeType32 remainder = llmReq->getContextChunkSize() % chunkUnitSize;
            SizeType32 remainingSpaceForDraftTokens = remainder == 0 ? 0 : chunkUnitSize - remainder;

            if (maxContextLength)
            {
                // How much space is remaining before reaching maxContextLength?
                SizeType32 remainingContextLength = maxContextLength.value() - llmReq->getContextChunkSize();
                remainingSpaceForDraftTokens = std::min(remainingSpaceForDraftTokens, remainingContextLength);
            }
            if (ctxTokensCapacity)
            {
                // How much space is remaining before reaching ctxTokensCapacity?
                remainingSpaceForDraftTokens
                    = std::min(remainingSpaceForDraftTokens, ctxTokensCapacity.value() - numCtxTokens);
                numCtxTokens += remainingSpaceForDraftTokens;
            }
            // Discard draft tokens.
            SizeType32 const draftTokensToDiscard = llmReq->getNumDraftTokens() - remainingSpaceForDraftTokens;
            if (draftTokensToDiscard > 0)
            {
                TLLM_LOG_DEBUG("Discarding %d draft tokens", draftTokensToDiscard);
                llmReq->discardDraftTokens(draftTokensToDiscard);
            }
        }
    }
}

// Assigns chunk sizes to context requests under the kEQUAL_PROGRESS policy.
//
// All requests advance together in lock-step: each iteration of the outer while-loop
// offers every request one additional chunkUnitSize of tokens. This continues until
// the compute budget (ctxTokensCapacity) is exhausted or no request can advance further.
//
// Budget accounting is compute-aware: tokens covered by the reusable KV-cache prefix
// (getEstimatedReusableTokens) are served from cache and do not consume forward-pass
// capacity. Only tokens beyond that prefix count against ctxTokensCapacity.
// The reusable prefix is only considered on the very first chunk of a request
// (isFirstContextChunk), since subsequent chunks start past the cached prefix.
//
// A request is skipped for this iteration if adding chunkUnitSize would exceed
// ctxTokensCapacity or maxContextLength; its chunk size is left unchanged.
//
// Loop-termination uses the raw token increment (actualIncrement), not the
// compute-adjusted one, so a request whose entire new chunk is reusable still
// counts as progress and does not cause a premature exit.
template <>
void MicroBatchScheduler::setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kEQUAL_PROGRESS>(
    RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    SizeType32 numCtxTokens{0};
    SizeType32 numTokensSingleLoop{1};

    while ((!ctxTokensCapacity || numCtxTokens < ctxTokensCapacity.value()) && numTokensSingleLoop)
    {
        numTokensSingleLoop = 0;
        for (auto& llmReq : contextsToBeChunked)
        {
            SizeType32 pastChunkSize = llmReq->getContextChunkSize();

            SizeType32 suggestedChunkSize = pastChunkSize + chunkUnitSize;
            llmReq->setContextChunkSize(suggestedChunkSize);

            SizeType32 actualChunkSize = llmReq->getContextChunkSize();
            SizeType32 actualIncrement = actualChunkSize - pastChunkSize;

            // Compute-aware budget: reusable tokens are served from cache and do not
            // consume forward-pass capacity. Only the tokens beyond the reusable prefix count.
            SizeType32 const reusable = llmReq->isFirstContextChunk()
                ? std::min(llmReq->getEstimatedReusableTokens(), llmReq->getContextRemainingLength())
                : 0;
            SizeType32 const pastCompute = std::max<SizeType32>(0, pastChunkSize - std::min(reusable, pastChunkSize));
            SizeType32 const actualCompute
                = std::max<SizeType32>(0, actualChunkSize - std::min(reusable, actualChunkSize));
            SizeType32 const computeIncrement = actualCompute - pastCompute;

            if ((ctxTokensCapacity && numCtxTokens + computeIncrement > ctxTokensCapacity.value())
                || (maxContextLength && actualChunkSize > maxContextLength.value()))
            {
                llmReq->setContextChunkSize(pastChunkSize);
                continue;
            }
            numCtxTokens += computeIncrement;
            // Keep raw actualIncrement for loop-termination detection (not compute-aware).
            numTokensSingleLoop += actualIncrement;
        }
    }
}

// Assigns chunk sizes to context requests under the kFIRST_COME_FIRST_SERVED policy.
//
// Requests are processed in order. Each request greedily claims as many tokens as
// possible from the remaining compute budget (ctxTokensCapacity) before the next
// request is considered — hence "first come, first served".
//
// For each request the desired chunk is its full remaining context length. The actual
// chunk size is then reduced by two independent constraints (applied in order):
//
//   1. ctxTokensCapacity: the available forward-pass compute budget.
//      Reusable tokens (cached KV prefix, first chunk only) are free; only
//      non-reusable tokens count. If the non-reusable portion exceeds the budget,
//      the chunk is capped at ctxTokensCapacity (not reusable + ctxTokensCapacity,
//      because the model processes tokens starting from position 0 of the chunk).
//
//   2. maxContextLength: an upper bound on the number of compute tokens per chunk.
//      If the non-reusable portion exceeds this, the chunk is capped at
//      reusable + maxContextLength (clamped to suggestedChunkSize).
//
// When either constraint trims the chunk, the result is aligned down to the nearest
// chunkUnitSize boundary to avoid KV-cache fragmentation.
//
// After assigning the chunk, ctxTokensCapacity is decremented by the actual model
// cost: min(actualChunkSize, non-reusable tokens), so the budget available to
// subsequent requests reflects only the compute consumed by this one.
template <>
void MicroBatchScheduler::setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED>(
    RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    for (auto& llmReq : contextsToBeChunked)
    {
        SizeType32 const suggestedChunkSize = llmReq->getContextRemainingLength();
        // Reusable tokens are "free" — they don't consume forward-pass compute budget.
        SizeType32 const reusable
            = llmReq->isFirstContextChunk() ? std::min(llmReq->getEstimatedReusableTokens(), suggestedChunkSize) : 0;
        SizeType32 const computeCost = suggestedChunkSize - reusable;
        SizeType32 actualChunkSize = suggestedChunkSize;
        if (ctxTokensCapacity && computeCost > ctxTokensCapacity.value())
        {
            // Model processes min(chunk_size, P - reusable) tokens starting from position reusable.
            // To keep model tokens within budget: chunk_size <= capacity (not reusable + capacity).
            actualChunkSize = ctxTokensCapacity.value();
        }
        if (maxContextLength)
        {
            // maxContextLength limits compute tokens, not total tokens.
            SizeType32 const actualCompute = std::max<SizeType32>(0, actualChunkSize - reusable);
            if (actualCompute > maxContextLength.value())
            {
                actualChunkSize = std::min<SizeType32>(reusable + maxContextLength.value(), suggestedChunkSize);
            }
        }
        if (actualChunkSize != suggestedChunkSize)
        {
            actualChunkSize = actualChunkSize / chunkUnitSize * chunkUnitSize;
        }
        llmReq->setContextChunkSize(actualChunkSize);
        if (ctxTokensCapacity)
        {
            // Decrement by actual model token count: min(chunk_size, P - reusable).
            // This equals min(actualChunkSize, computeCost) since computeCost = suggestedChunkSize - reusable.
            SizeType32 const modelCost
                = std::min(actualChunkSize, std::max<SizeType32>(0, suggestedChunkSize - reusable));
            ctxTokensCapacity = ctxTokensCapacity.value() - modelCost;
        }
    }
}

// Assigns chunk sizes to context requests under the kFORCE_CHUNK policy.
//
// Every request is assigned exactly min(contextRemainingLength, chunkUnitSize) tokens.
// Requests whose chunk would push the running total past ctxTokensCapacity are zeroed.
//
// This policy is designed for linear attention state caching, so reusable KV-cache tokens are NOT
// calculated because it's not supported yet.
template <>
void MicroBatchScheduler::setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFORCE_CHUNK>(
    RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity, SizeType32 const chunkUnitSize,
    std::optional<SizeType32> const& maxContextLength)
{
    if (maxContextLength && maxContextLength.value() < chunkUnitSize)
    {
        TLLM_THROW(
            "The forced chunk size (%d) exceeds the max context length (%d)", chunkUnitSize, maxContextLength.value());
    }
    SizeType32 totalTokens{0};
    for (auto& llmReq : contextsToBeChunked)
    {
        SizeType32 const chunkSize = std::min(llmReq->getContextRemainingLength(), chunkUnitSize);
        if (ctxTokensCapacity && totalTokens + chunkSize > ctxTokensCapacity.value())
        {
            llmReq->setContextChunkSize(0);
        }
        else
        {
            llmReq->setContextChunkSize(chunkSize);
            totalTokens += llmReq->getContextChunkSize();
        }
    }
}

// Entry point for chunk-size assignment. Resets all chunk sizes to zero, then
// dispatches to the appropriate policy-specific implementation:
//
//   kEQUAL_PROGRESS        — all requests advance together one chunkUnitSize at a time.
//   kFIRST_COME_FIRST_SERVED — requests are served greedily in order until the budget
//                              is exhausted.
//   kFORCE_CHUNK           — every request gets exactly min(remaining, chunkUnitSize)
//                              tokens; budget is charged at face value (no reuse discount).
//
// EQUAL_PROGRESS and FIRST_COME_FIRST_SERVED are compute-aware: tokens covered by the
// reusable KV-cache prefix are not charged against ctxTokensCapacity.
// FORCE_CHUNK intentionally skips reuse accounting.
// See the individual template specialisations above for full details.
void MicroBatchScheduler::setCtxRequestsChunkSize(RequestVector& contextsToBeChunked,
    ContextChunkingPolicy const ctxChunkPolicy, std::optional<SizeType32> ctxTokensCapacity,
    SizeType32 const chunkUnitSize, std::optional<SizeType32> const& maxContextLength)
{
    for (auto& llmReq : contextsToBeChunked)
    {
        llmReq->setContextChunkSize(0);
    }
    switch (ctxChunkPolicy)
    {
    case ContextChunkingPolicy::kEQUAL_PROGRESS:
        setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kEQUAL_PROGRESS>(
            contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
        break;
    case ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED:
        setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED>(
            contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
        break;
    case ContextChunkingPolicy::kFORCE_CHUNK:
        setCtxRequestsChunkSize<MicroBatchScheduler::ContextChunkingPolicy::kFORCE_CHUNK>(
            contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
        break;
    default: TLLM_THROW("The chunked scheduling type `NO_CHUNKING` cannot be performed.");
    }

    // After scheduling chunk sizes, discard draft tokens that won't fit.
    fitDraftTokens(contextsToBeChunked, ctxTokensCapacity, chunkUnitSize, maxContextLength);
}

std::tuple<RequestVector, RequestVector> MicroBatchScheduler::operator()(RequestVector& activeRequests,
    ReqIdsSet const& inflightReqIds, SizeType32 maxBatchSizeRuntime,
    std::optional<SizeType32> maxNumTokensRuntime) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(microBatcherScheduleRequests);

    RequestVector contextRequests, generationRequests;
    // batchNumTokens tracks COMPUTE tokens only (excluding reusable cached tokens)
    SizeType32 batchNumTokens{0};
    SizeType32 scheduledReqSize{0};
    SizeType32 scheduledBeamWidth{0}; // 0 means no request is scheduled

    RequestVector contextsToBeChunked;
    SizeType32 numChunkedComputeTokens{0};
    bool allContextRequestsFit{true};

    // 1. Select the generation phase requests that meet the criteria of total token size.
    //    If there is any remaining space, include the context requests and divide them into chunks.
    for (auto& llmReq : activeRequests)
    {
        // if request cannot be scheduled yet or request should no longer be scheduled, skip
        if (!llmReq->hasReachedState(mNoScheduleUntilState) || llmReq->hasReachedState(mNoScheduleAfterState))
        {
            continue;
        }

        // if already in execution, skip
        if (inflightReqIds.find(llmReq->mRequestId) != inflightReqIds.end())
        {
            continue;
        }

        SizeType32 reqNumTokens = 0;
        if (llmReq->isEncoderInitState())
        {
            reqNumTokens = llmReq->getEncoderOutputLen();
            TLLM_CHECK_WITH_INFO(!mMaxContextLength || reqNumTokens <= mMaxContextLength.value(),
                "The number of encoder tokens (%d) exceeds the limit value (%d)", reqNumTokens,
                mMaxContextLength.value());
            if (maxNumTokensRuntime && batchNumTokens + reqNumTokens > maxNumTokensRuntime.value())
            {
                break;
            }
            TLLM_LOG_DEBUG("encoder request scheduled: ID %u", llmReq->mRequestId);
            contextRequests.emplace_back(llmReq);
            batchNumTokens += reqNumTokens;
        }
        else if (llmReq->isContextInitState())
        {
            // Reusable tokens set by capacity scheduler (from radix tree lookup).
            // Only valid for the first context chunk; subsequent chunks must compute all remaining tokens.
            SizeType32 const reusable = llmReq->isFirstContextChunk() ? llmReq->getEstimatedReusableTokens() : 0;

            if (!mCtxChunkConfig) // skip chunking
            {
                constexpr SizeType32 beam{0};
                reqNumTokens
                    = llmReq->getNumTokens(beam) + (llmReq->hasDraftTokens() ? llmReq->getNumDraftTokens() : 0);
                // Compute tokens = total - reusable (at least 1 to make progress)
                SizeType32 const computeTokens = std::max(1, reqNumTokens - reusable);
                TLLM_CHECK_WITH_INFO(!mMaxContextLength || computeTokens <= mMaxContextLength.value(),
                    "Context compute tokens (%d) exceeds the limit value (%d)", computeTokens,
                    mMaxContextLength.value());
                if (maxNumTokensRuntime && batchNumTokens + computeTokens > maxNumTokensRuntime.value())
                {
                    break;
                }
                TLLM_LOG_DEBUG("context request scheduled: ID %u (reusable %d)", llmReq->mRequestId, reusable);
                contextRequests.emplace_back(llmReq);
                batchNumTokens += computeTokens;
            }
            else
            {
                llmReq->setContextChunkSize(llmReq->getContextRemainingLength());
                auto const draftTokens
                    = (llmReq->isLastContextChunk() && llmReq->hasDraftTokens()) ? llmReq->getNumDraftTokens() : 0;
                // Compute cost: context compute + draft tokens
                // (reusable tokens only offset context tokens, not draft tokens)
                SizeType32 const contextCompute = std::max(0, llmReq->getContextChunkSize() - reusable);
                SizeType32 computeTokens = contextCompute + draftTokens;

                if (mMaxContextLength)
                {
                    if (mMaxContextLength.value() < computeTokens)
                    {
                        // The context exceeds the length limit, we need to try chunking later.
                        computeTokens = mMaxContextLength.value();
                        allContextRequestsFit = false;
                    }
                }
                contextsToBeChunked.emplace_back(llmReq);
                numChunkedComputeTokens += computeTokens;
                TLLM_LOG_DEBUG(
                    "contexts-to-be-chunked request scheduled: ID %u (reusable %d)", llmReq->mRequestId, reusable);
            }
        }
        else // (llmReq->isGenerationInProgressState())
        {
            auto const reqBeamWidth = llmReq->getBeamWidthByIter();
            reqNumTokens = reqBeamWidth + llmReq->getNumDraftTokens();
            if (maxNumTokensRuntime && batchNumTokens + reqNumTokens > maxNumTokensRuntime.value())
            {
                break;
            }
            if (scheduledBeamWidth == 0) // set `scheduledBeamWidth` when the first request is scheduled
            {
                scheduledBeamWidth = reqBeamWidth;
            }
            else if (scheduledBeamWidth != reqBeamWidth) // Skip request with different beam width
            {
                TLLM_LOG_DEBUG(
                    "generation request skipped: ID %u since its beam width (%d) is different from scheduled ones (%d)",
                    llmReq->mRequestId, reqBeamWidth, scheduledBeamWidth);
                continue;
            }
            TLLM_LOG_DEBUG("generation request scheduled: ID %u with beam width %d", llmReq->mRequestId, reqBeamWidth);
            generationRequests.emplace_back(llmReq);
            batchNumTokens += reqNumTokens;
        }

        if (++scheduledReqSize >= maxBatchSizeRuntime)
        {
            break;
        }
    }

    if (maxNumTokensRuntime && numChunkedComputeTokens > maxNumTokensRuntime.value() - batchNumTokens)
    {
        allContextRequestsFit = false;
    }

    // For FORCE_CHUNK policy, always re-chunk regardless of whether all contexts fit.
    if (mCtxChunkConfig && mCtxChunkConfig.value().chunkingPolicy == ContextChunkingPolicy::kFORCE_CHUNK)
    {
        allContextRequestsFit = false;
    }

    // 2. If not all contexts fit into the batch, the chunk size should be adjusted accordingly.
    if (!allContextRequestsFit)
    {
        TLLM_CHECK_WITH_INFO(mCtxChunkConfig, "If chunking is not enabled, context scheduling should be completed.");
        auto const ctxTokensCapacity
            = maxNumTokensRuntime ? std::make_optional(maxNumTokensRuntime.value() - batchNumTokens) : std::nullopt;
        setCtxRequestsChunkSize(contextsToBeChunked, mCtxChunkConfig.value().chunkingPolicy, ctxTokensCapacity,
            mCtxChunkConfig.value().chunkUnitSize, mMaxContextLength);
    }
    for (auto const& llmReq : contextsToBeChunked)
    {
        if (llmReq->getContextChunkSize() > 0)
        {
            contextRequests.emplace_back(llmReq);
            // Only count compute tokens (total - reusable).
            // Reusable credit only applies to the first context chunk.
            SizeType32 const reusable = llmReq->isFirstContextChunk() ? llmReq->getEstimatedReusableTokens() : 0;
            SizeType32 const computeTokens = std::max(0, llmReq->getContextChunkSize() - reusable);
            batchNumTokens += computeTokens;
            TLLM_LOG_DEBUG("context request scheduled: ID %lu, chunk size %d%s", llmReq->mRequestId,
                llmReq->getContextChunkSize(), reusable > 0 ? (", reusable " + std::to_string(reusable)).c_str() : "");
        }
    }

    utils::sortRequests(contextRequests, generationRequests, !allContextRequestsFit);

    TLLM_LOG_DEBUG(
        "batchSize (num ctx/enc requests + num gen requests): %u", contextRequests.size() + generationRequests.size());
    TLLM_LOG_DEBUG("batchNumTokens (num ctx/enc input tokens + num gen input tokens) / maxNumTokens: %d / %d",
        batchNumTokens, maxNumTokensRuntime.value_or(0));
    TLLM_LOG_DEBUG(
        "[Summary] Micro Batch scheduler schedules %d context/encoder requests, %d generation requests. "
        "%d requests inflight with the model already",
        contextRequests.size(), generationRequests.size(), inflightReqIds.size());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(contextRequests), std::move(generationRequests)};
}

} // namespace tensorrt_llm::batch_manager
