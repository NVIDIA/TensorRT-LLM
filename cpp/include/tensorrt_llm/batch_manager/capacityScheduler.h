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
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/common.h"
#include <variant>

namespace tensorrt_llm::batch_manager
{
namespace kv_cache_manager
{
class BaseKVCacheManager;
}
class BasePeftCacheManager;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager
{

using tensorrt_llm::runtime::SizeType32;
using common::OptionalRef;

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
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    /// @brief Takes as input a sorted list of requests and outputs a sorted lists of requests
    ///        to update for this current iteration, and a map of requests to pause
    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
};

/// @brief   Schedule requests using the MAX_UTILIZATION policy
/// @details Try reserving resources to advance requests by one step,
///          may pause previously started requests.
class MaxUtilizationScheduler : public BaseCapacityScheduler
{
public:
    MaxUtilizationScheduler(SizeType32 maxNumRequests, bool twoStepsLookAhead,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager& kvCacheManager, OptionalRef<BasePeftCacheManager const> peftCacheManager,
        RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
    /// @brief Boolean that indicates if two step lookahead is enabled
    bool mTwoStepsLookAhead;
};

/// @brief Schedule requests using the GUARANTEED_NO_EVICT policy
class GuaranteedNoEvictScheduler : public BaseCapacityScheduler
{
public:
    GuaranteedNoEvictScheduler(SizeType32 maxNumRequests,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;

protected:
    template <bool StaticBatchScheduling>
    [[nodiscard]] std::tuple<RequestVector, RequestVector> impl(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;

private:
    SizeType32 mMaxNumRequests;
};

/// @brief Schedule requests using the STATIC_BATCH policy
class StaticBatchScheduler : public GuaranteedNoEvictScheduler
{
public:
    StaticBatchScheduler(SizeType32 maxNumRequests,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    [[nodiscard]] std::tuple<RequestVector, RequestVector> operator()(
        kv_cache_manager::BaseKVCacheManager const& kvCacheManager,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager,
        OptionalRef<BasePeftCacheManager const> peftCacheManager, RequestList const& activeRequests) const;
};

class CapacityScheduler : public Algorithm
{
public:
    constexpr static auto name{"CapacityScheduler"};

    explicit CapacityScheduler(SizeType32 maxNumRequests, executor::CapacitySchedulerPolicy capacitySchedulerPolicy,
        bool hasKvCacheManager, bool twoStepsLookAhead = false,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_COMPLETE);

    /**
     * @brief Schedules requests following the selected policy.
     *
     * @param kvCacheManager Required in MaxUtilizationScheduler (as a ref) and in GuaranteedNoEvictScheduler and
     * StaticBatchScheduler (as a const ref).
     * @param crossKvCacheManager Optional used in GuaranteedNoEvictScheduler and StaticBatchScheduler.
     * @param peftCacheManager Optional used in MaxUtilizationScheduler, GuaranteedNoEvictScheduler and
     * StaticBatchScheduler.
     * @param activeRequests
     * @return std::tuple<RequestVector, RequestVector, RequestVector>, fittingRequests, fittingDisaggInitRequests and
     * pausedRequests respectively.
     */
    [[nodiscard]] std::tuple<RequestVector, RequestVector, RequestVector> operator()(RequestList const& activeRequests,
        OptionalRef<kv_cache_manager::BaseKVCacheManager> kvCacheManager = std::nullopt,
        OptionalRef<BasePeftCacheManager const> peftCacheManager = std::nullopt,
        OptionalRef<kv_cache_manager::BaseKVCacheManager const> crossKvCacheManager = std::nullopt) const;

private:
    std::variant<std::monostate, MaxRequestsScheduler, MaxUtilizationScheduler, GuaranteedNoEvictScheduler,
        StaticBatchScheduler>
        mScheduler;
};

} // namespace tensorrt_llm::batch_manager
