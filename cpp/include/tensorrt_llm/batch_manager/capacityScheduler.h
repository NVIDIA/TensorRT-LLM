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
#include "tensorrt_llm/batch_manager/schedulerPolicy.h"
#include "tensorrt_llm/runtime/common.h"

#include <list>
#include <utility>

namespace tensorrt_llm::batch_manager::batch_scheduler
{

/// @brief This scheduler takes into account the given request capacity and the KV cache capacity.
///        Depending on the SchedulerPolicy it will schedule already started and new requests,
///        or even terminate previously started requests.
class CapacityScheduler
{
public:
    virtual ~CapacityScheduler() = default;

    using RequestTable = std::map<uint64_t, std::shared_ptr<LlmRequest>>;
    using SizeType = tensorrt_llm::runtime::SizeType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;

    /// @brief Takes as input a sorted list of requests and outputs a sorted lists of requests
    ///        to update for this current iteration, and a map of requests to terminate
    virtual std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& activeRequests) = 0;
};

/// @brief Schedule up to maxNumRequests requests
class MaxRequestsScheduler : public CapacityScheduler
{
public:
    explicit MaxRequestsScheduler(SizeType maxNumRequests);

    std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& activeRequests) override;

private:
    SizeType mMaxNumRequests;
};

/// @brief   Schedule requests using the MAX_UTILIZATION policy
/// @details Try reserving resources to advance requests by one step,
///          may terminate previously started requests.
class MaxUtilizationScheduler : public CapacityScheduler
{
public:
    MaxUtilizationScheduler(
        SizeType maxNumRequests, kv_cache_manager::KVCacheManager* const kvCacheManager, bool manyMicroBatches);

    std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& activeRequests) override;

private:
    bool trySchedulingRequestMaxUtilization(
        const LlmRequest& req, SizeType& numScheduledRequests, SizeType& numScheduledBlocks);

    SizeType mMaxNumRequests;
    kv_cache_manager::KVCacheManager* mKvCacheManager{nullptr};
    /// @brief Boolean that indicates if multiple micro batches might be in flight
    bool mManyMicroBatches;
};

/// @brief Schedule requests using the GUARANTEED_NO_EVICT policy
class GuaranteedNoEvictScheduler : public CapacityScheduler
{
public:
    GuaranteedNoEvictScheduler(SizeType maxNumRequests, kv_cache_manager::KVCacheManager* const kvCacheManager);

    std::tuple<RequestList, RequestTable> scheduleRequests(const RequestList& activeRequests) override;

private:
    SizeType mMaxNumRequests;
    kv_cache_manager::KVCacheManager* mKvCacheManager{nullptr};
};

std::unique_ptr<CapacityScheduler> makeCapacityScheduler(tensorrt_llm::runtime::SizeType maxNumRequests,
    kv_cache_manager::KVCacheManager* const kvCacheManager, SchedulerPolicy schedulerPolicy,
    bool manyMicroBatches = false);

} // namespace tensorrt_llm::batch_manager::batch_scheduler
