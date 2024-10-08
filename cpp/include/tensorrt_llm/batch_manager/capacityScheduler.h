/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/runtime/common.h"
#include <variant>

namespace tensorrt_llm::batch_manager
{
namespace kv_cache_manager
{
class KVCacheManager;
}
class BasePeftCacheManager;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager
{

using tensorrt_llm::runtime::SizeType32;

/// @brief This scheduler takes into account the given request capacity and the KV cache capacity.
///        Depending on the CapacitySchedulerPolicy it will schedule already started and new requests,
///        or even pause previously started requests.
class BaseCapacityScheduler
{
public:
    explicit BaseCapacityScheduler(LlmRequestState noScheduleUntilState, LlmRequestState noScheduleAfterState)
        : mNoScheduleUntilState(noScheduleUntilState)
        , mNoScheduleAfterState(noScheduleAfterState)
    {
    }

    [[nodiscard]] LlmRequestState constexpr getNoScheduleUntilState() const noexcept
    {
        return mNoScheduleUntilState;
    }

    [[nodiscard]] LlmRequestState constexpr getNoScheduleAfterState() const noexcept
    {
        return mNoScheduleAfterState;
    }

private:
    /// The state until/after which the scheduler should not schedule requests
    LlmRequestState mNoScheduleUntilState;
    LlmRequestState mNoScheduleAfterState;
};

/// @brief Schedule up to maxNumRequests requests
class MaxRequestsScheduler : public BaseCapacityScheduler
{
public:
    explicit MaxRequestsScheduler(SizeType32 maxNumRequests,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    /// @brief Takes as input a sorted list of requests and outputs a sorted lists of requests
    ///        to update for this current iteration, and a map of requests to pause
    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
};

/// @brief   Schedule requests using the MAX_UTILIZATION policy
/// @details Try reserving resources to advance requests by one step,
///          may pause previously started requests.
class MaxUtilizationScheduler : public BaseCapacityScheduler
{
public:
    MaxUtilizationScheduler(SizeType32 maxNumRequests, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager, bool manyMicroBatches,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

private:
    /// @return {fitsKvCache, fitsPeft}
    std::pair<bool, bool> trySchedulingRequestMaxUtilization(std::shared_ptr<LlmRequest> const& req,
        RequestVector& scheduledRequests, SizeType32& numScheduledBlocks, SizeType32& numScheduledPeftPages,
        std::unordered_set<uint64_t>& seenTaskIds) const;

    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager{nullptr};
    /// @brief Boolean that indicates if multiple micro batches might be in flight
    bool mManyMicroBatches;
};

/// @brief Schedule requests using the GUARANTEED_NO_EVICT policy
class GuaranteedNoEvictScheduler : public BaseCapacityScheduler
{
public:
    GuaranteedNoEvictScheduler(SizeType32 maxNumRequests,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

protected:
    [[nodiscard]] std::tuple<RequestVector, RequestVector> forwardImpl(
        RequestList const& activeRequests, bool staticBatchScheduling) const;

private:
    SizeType32 mMaxNumRequests;
    std::shared_ptr<kv_cache_manager::KVCacheManager> mKvCacheManager{nullptr};
    std::shared_ptr<kv_cache_manager::KVCacheManager> mCrossKvCacheManager{nullptr};
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager{nullptr};
};

/// @brief Schedule requests using the STATIC_BATCH policy
class StaticBatchScheduler : public GuaranteedNoEvictScheduler
{
public:
    StaticBatchScheduler(SizeType32 maxNumRequests, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;
};

class CapacityScheduler : public Algorithm
{
public:
    constexpr static auto name{"CapacityScheduler"};

    CapacityScheduler() = default;

    CapacityScheduler(SizeType32 maxNumRequests, std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager,
        executor::CapacitySchedulerPolicy capacitySchedulerPolicy, bool manyMicroBatches = false,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    static CapacityScheduler make(SizeType32 maxNumRequests,
        std::shared_ptr<kv_cache_manager::KVCacheManager> kvCacheManager,
        std::shared_ptr<kv_cache_manager::KVCacheManager> crossKvCacheManager,
        std::shared_ptr<BasePeftCacheManager> peftCacheManager,
        executor::CapacitySchedulerPolicy capacitySchedulerPolicy, bool manyMicroBatches = false,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE)
    {
        return CapacityScheduler{maxNumRequests, std::move(kvCacheManager), std::move(crossKvCacheManager),
            std::move(peftCacheManager), capacitySchedulerPolicy, manyMicroBatches, noScheduleUntilState,
            noScheduleAfterState};
    }

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

private:
    std::variant<std::monostate, MaxRequestsScheduler, MaxUtilizationScheduler, GuaranteedNoEvictScheduler,
        StaticBatchScheduler>
        mScheduler;
};

} // namespace tensorrt_llm::batch_manager
