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

#include "inflightBatchingUtils.h"

namespace tensorrt_llm::batch_manager::utils
{
using ITensor = runtime::ITensor;

TensorPtr collectRequestIds(RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    auto const numRequests = static_cast<ITensor::DimType64>(contextRequests.size() + generationRequests.size());
    auto requestIds
        = runtime::BufferManager::cpu(ITensor::makeShape({numRequests}), runtime::TRTDataType<RequestIdType>::value);
    auto requestIdsRange = runtime::BufferRange<RequestIdType>(*requestIds);
    auto batchIdx{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& request : requests)
        {
            requestIdsRange[batchIdx++] = request->mRequestId;
        }
    }
    return requestIds;
}

void sortRequests(RequestVector& contextRequests, RequestVector& generationRequests, bool chunksPresent)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto sortByLoraId = [](RequestVector::iterator begin, RequestVector::iterator end)
    {
        std::sort(
            begin, end, [](auto const& lhs, auto const& rhs) { return lhs->getLoraTaskId() < rhs->getLoraTaskId(); });
    };

    if (chunksPresent)
    {
        // Move context requests that reached the last context chunk to the end of the vector.
        // This order is required for moveFinishedContextRequestsToGeneration.
        auto firstFinished = std::partition(contextRequests.begin(), contextRequests.end(),
            [](auto const& llmReq) { return !llmReq->isLastContextChunk(); });

        // Sort context requests by lora task id, but keep finished requests separate.
        sortByLoraId(contextRequests.begin(), firstFinished);
        sortByLoraId(firstFinished, contextRequests.end());
    }
    else
    {
        sortByLoraId(contextRequests.begin(), contextRequests.end());
    }
    sortByLoraId(generationRequests.begin(), generationRequests.end());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void moveFinishedContextRequestsToGeneration(ScheduledRequests& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto& contextRequests = scheduledRequests.contextRequests;
    auto& generationRequests = scheduledRequests.generationRequests;
    auto firstFinished = std::find_if(
        contextRequests.begin(), contextRequests.end(), [](auto const& llmReq) { return llmReq->isContextFinished(); });
    TLLM_LOG_DEBUG(
        "Found %ld unfinished chunked context requests. Found %ld finished context requests, moving them to "
        "generation.",
        std::distance(contextRequests.begin(), firstFinished), std::distance(firstFinished, contextRequests.end()));
    generationRequests.insert(generationRequests.begin(), std::make_move_iterator(firstFinished),
        std::make_move_iterator(contextRequests.end()));
    contextRequests.erase(firstFinished, contextRequests.end());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void terminateRequest(SequenceSlotManager& seqSlotManager, LlmRequest& llmReq, SizeType32 maxInputLen,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    OptionalRef<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager> peftCacheManager, bool pause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // If a sequence slot is associated with this request id, free it
    seqSlotManager.freeSequenceSlot(llmReq.mRequestId);
    // Remove the sequence from kvCacheManager
    auto const requestId = llmReq.mRequestId;
    if (kvCacheManager)
    {
        (void) kvCacheManager->removeSequence(requestId, llmReq);
    }
    if (crossKvCacheManager)
    {
        (void) crossKvCacheManager->removeSequence(requestId, llmReq);
    }
    if (pause && !llmReq.isGenerationCompleteState())
    {
        llmReq.pause(maxInputLen);
    }
    else
    {
        TLLM_LOG_DEBUG("terminated: request ID %lu, paused: %d", requestId, pause);
    }

    if (peftCacheManager)
    {
        peftCacheManager->markRequestDone(llmReq, pause);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<SizeType32> getRequestBeamWidths(
    RequestVector const& contextRequests, RequestVector const& generationRequests)
{
    std::vector<SizeType32> beamWidths{};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            beamWidths.push_back(llmReq->getBeamWidthByIter());
        }
    }
    return beamWidths;
}

} // namespace tensorrt_llm::batch_manager::utils
