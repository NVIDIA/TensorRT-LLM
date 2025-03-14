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

#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

void tensorrt_llm::batch_manager::PauseRequests::operator()(RequestVector& requestsToPause, ReqIdsSet& inflightReqIds,
    ReqIdsSet& reqIdsToPause, bool pauseFlagged, SequenceSlotManager& seqSlotManager,
    OptionalRef<BaseKVCacheManager> kvCacheManager, OptionalRef<BaseKVCacheManager> crossKvCacheManager,
    OptionalRef<BasePeftCacheManager> peftCacheManager) const
{
    NVTX3_SCOPED_RANGE(PauseRequests);
    if (!pauseFlagged)
    {
        // Loop over requests flagged to be paused, and if not in flight pause it right away
        for (auto& llmReq : requestsToPause)
        {
            auto const reqId = llmReq->mRequestId;
            if (inflightReqIds.find(reqId) == inflightReqIds.end())
            {
                // Not in flight, can terminate right away
                utils::terminateRequest(
                    seqSlotManager, *llmReq, mMaxInputLen, kvCacheManager, crossKvCacheManager, peftCacheManager, true);
            }
            else
            {
                // In flight, add to set for pausing later
                reqIdsToPause.insert(reqId);
            }
        }
    }
    else
    {
        for (auto& llmReq : requestsToPause)
        {
            auto const reqId = llmReq->mRequestId;
            inflightReqIds.erase(reqId);
            TLLM_LOG_DEBUG("request with ID %lu removed from DECODER model inflight set", reqId);

            // If a request in this context had been flagged to be paused, pause it right away
            if (reqIdsToPause.find(reqId) != reqIdsToPause.end())
            {
                utils::terminateRequest(
                    seqSlotManager, *llmReq, mMaxInputLen, kvCacheManager, crossKvCacheManager, peftCacheManager, true);
                reqIdsToPause.erase(reqId);
            }
        }
    }
}
