/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/common.h"
#include <list>

namespace tensorrt_llm::batch_manager::batch_scheduler
{

enum class SchedulerPolicy
{
    MAX_UTILIZATION,
    GUARANTEED_COMPLETION,
};

class BatchScheduler
{
public:
    using RequestTable = std::map<uint64_t, std::shared_ptr<LlmRequest>>;
    using SizeType = tensorrt_llm::runtime::SizeType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;

    BatchScheduler(int32_t maxNumRequests, int32_t maxInputLen,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager, SchedulerPolicy schedulerPolicy)
        : mMaxNumRequests(maxNumRequests)
        , mMaxInputLen(maxInputLen)
        , mKvCacheManager(kvCacheManager)
        , mSchedulerPolicy(schedulerPolicy)
    {
    }

    /// @brief Takes as input a sorted list of requets and outputs a map of requests
    ///        to update for this current iteration
    RequestTable scheduleRequests(const RequestList& requestList);

private:
    /// @brief Schedule request using the MAX_UTILIZATION policy
    RequestTable scheduleRequestsMaxUtilization(const RequestList& requestList);

    /// @brief Try reserving resources to advance this req by one step, using MAX_UTILIZATION policy
    bool trySchedulingRequestMaxUtilization(
        const LlmRequest& req, SizeType& numScheduledRequests, SizeType& numScheduledBlocks);

    /// @brief Schedule request using the GUARANTEED_COMPLETION policy
    RequestTable scheduleRequestsGuaranteedCompletion(const RequestList& requestList);

    /// @brief Schedule up to mMaxNumReuests requests
    RequestTable scheduleMaxNumRequests(const RequestList& requestList);

    /// The maximum number of requests to schedule
    int32_t mMaxNumRequests;

    /// The maximum prompt length
    int32_t mMaxInputLen;

    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager;

    /// The scheduling policy to use
    SchedulerPolicy mSchedulerPolicy;
};

} // namespace tensorrt_llm::batch_manager::batch_scheduler
