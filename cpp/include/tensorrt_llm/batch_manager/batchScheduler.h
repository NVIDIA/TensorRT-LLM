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
#include "tensorrt_llm/runtime/common.h"
#include <list>

namespace tensorrt_llm::batch_manager::batch_scheduler
{

enum class SchedulerPolicy
{
    MAX_UTILIZATION,
    GUARANTEED_NO_EVICT,
};

class BatchScheduler
{
public:
    using RequestTable = std::map<uint64_t, std::shared_ptr<LlmRequest>>;
    using SizeType = tensorrt_llm::runtime::SizeType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;

    BatchScheduler(SizeType targetBatchSize, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        SchedulerPolicy schedulerPolicy, bool manyMicroBatches = false)
        : mTargetBatchSize(targetBatchSize)
        , mKvCacheManager(kvCacheManager)
        , mSchedulerPolicy(schedulerPolicy)
        , mManyMicroBatches(manyMicroBatches)
    {
    }

    /// @brief Takes as input a sorted list of requests and outputs a sorted lists of requests
    ///        to update for this current iteration, and a map of requests to terminate
    std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& requestList);

private:
    /// @brief Schedule request using the MAX_UTILIZATION policy
    std::tuple<RequestList, RequestTable> scheduleRequestsMaxUtilization(const RequestList& requestList);

    /// @brief Try reserving resources to advance this req by one step, using MAX_UTILIZATION policy
    bool trySchedulingRequestMaxUtilization(
        const LlmRequest& req, SizeType& numScheduledRequests, SizeType& numScheduledBlocks);

    /// @brief Schedule request using the GUARANTEED_NO_EVICT policy
    std::tuple<RequestList, RequestTable> scheduleRequestsGuaranteedNoEvict(const RequestList& requestList);

    /// @brief Schedule up to mMaxNumReuests requests
    std::tuple<RequestList, RequestTable> scheduleTargetBatchSize(const RequestList& requestList);

    /// The target number of requests to include in a batch
    SizeType mTargetBatchSize;

    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager;

    /// The scheduling policy to use
    SchedulerPolicy mSchedulerPolicy;

    /// @brief Boolean that indicates if multiple micro batches might be in flight
    bool mManyMicroBatches;
};

} // namespace tensorrt_llm::batch_manager::batch_scheduler
