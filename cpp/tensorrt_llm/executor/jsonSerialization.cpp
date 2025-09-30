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

#include "tensorrt_llm/common/jsonSerializeOptional.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tensorrt_llm::executor
{

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(KvCacheStats, maxNumBlocks, freeNumBlocks, usedNumBlocks, tokensPerBlock,
    allocTotalBlocks, allocNewBlocks, reusedBlocks, missedBlocks, cacheHitRate);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    StaticBatchingStats, numScheduledRequests, numContextRequests, numCtxTokens, numGenTokens, emptyGenSlots);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InflightBatchingStats, numScheduledRequests, numContextRequests, numGenRequests,
    numPausedRequests, numCtxTokens, microBatchId, avgNumDecodedTokensPerIter);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SpecDecodingStats, numDraftTokens, numAcceptedTokens, numRequestsWithDraftTokens,
    acceptanceLength, iterLatencyMS, draftOverhead);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IterationStats, timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
    numNewActiveRequests, numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests,
    maxBatchSizeStatic, maxBatchSizeTunerRecommended, maxBatchSizeRuntime, maxNumTokensStatic,
    maxNumTokensTunerRecommended, maxNumTokensRuntime, gpuMemUsage, cpuMemUsage, pinnedMemUsage, kvCacheStats,
    staticBatchingStats, inflightBatchingStats, specDecodingStats);
NLOHMANN_JSON_SERIALIZE_ENUM(RequestStage,
    {{RequestStage::kQUEUED, "QUEUED"}, {RequestStage::kCONTEXT_IN_PROGRESS, "CONTEXT_IN_PROGRESS"},
        {RequestStage::kGENERATION_IN_PROGRESS, "GENERATION_IN_PROGRESS"},
        {RequestStage::kGENERATION_COMPLETE, "GENERATION_COMPLETE"}});
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DisServingRequestStats, kvCacheTransferMS, kvCacheSize);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestStats, id, stage, contextPrefillPosition, numGeneratedTokens,
    avgNumDecodedTokensPerIter, scheduled, paused, disServingStats, allocTotalBlocksPerRequest,
    allocNewBlocksPerRequest, reusedBlocksPerRequest, missedBlocksPerRequest, kvCacheHitRatePerRequest);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(RequestStatsPerIteration, iter, requestStats);

std::string JsonSerialization::toJsonStr(IterationStats const& iterationStats)
{
    json j = iterationStats;
    return j.dump();
}

std::string JsonSerialization::toJsonStr(RequestStatsPerIteration const& requestStatsPerIter)
{
    json j = requestStatsPerIter;
    return j.dump();
}

std::string JsonSerialization::toJsonStr(RequestStats const& requestStats)
{
    json j = requestStats;
    return j.dump();
}

} // namespace tensorrt_llm::executor
