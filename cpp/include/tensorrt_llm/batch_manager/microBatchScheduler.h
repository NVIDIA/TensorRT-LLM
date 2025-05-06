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

namespace tensorrt_llm::batch_manager
{

namespace batch_scheduler
{

struct ContextChunkingConfig
{
    ContextChunkingConfig() = default;

    executor::ContextChunkingPolicy chunkingPolicy;
    /// The minimum size, also known as the chunk unit size. It generally
    /// needs to be equal to the size of the kv cache block or its integer
    /// multiples (except for the last context chunk) to avoid fragmentation.
    /// When set to null, it indicates that the context chunk is disabled.
    tensorrt_llm::runtime::SizeType32 chunkUnitSize;
};

} // namespace batch_scheduler

/// @brief This scheduler takes into account the desired batch size and limits of the TRT engine to schedule requests.
class MicroBatchScheduler : Algorithm
{
public:
    constexpr static auto name{"MicroBatchScheduler"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using ContextChunkingPolicy = tensorrt_llm::executor::ContextChunkingPolicy;

    explicit MicroBatchScheduler(std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig = std::nullopt,
        std::optional<SizeType32> maxContextLength = std::nullopt,
        LlmRequestState noScheduleUntilState = LlmRequestState::kCONTEXT_INIT,
        LlmRequestState noScheduleAfterState = LlmRequestState::kGENERATION_TO_COMPLETE);

    std::tuple<RequestVector, RequestVector> operator()(RequestVector& activeRequests, ReqIdsSet const& inflightReqIds,
        SizeType32 maxBatchSizeRuntime, std::optional<SizeType32> maxNumTokensRuntime) const;

    static void setCtxRequestsChunkSize(RequestVector& contextsToBeChunked, ContextChunkingPolicy ctxChunkPolicy,
        std::optional<SizeType32> ctxTokensCapacity, SizeType32 chunkUnitSize,
        std::optional<SizeType32> const& maxContextLength);

private:
    template <ContextChunkingPolicy tPolicy>
    static void setCtxRequestsChunkSize(RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity,
        SizeType32 chunkUnitSize, std::optional<SizeType32> const& maxContextLength);

    /// After the chunk sizes have been determined, this function will discard
    /// any draft tokens that don't fit.
    static void fitDraftTokens(RequestVector& contextsToBeChunked, std::optional<SizeType32> ctxTokensCapacity,
        SizeType32 chunkUnitSize, std::optional<SizeType32> const& maxContextLength);

    /// The maximum length of the context. If the context exceeds this length,
    /// it must be chunked, otherwise it cannot be processed. Therefore, it
    /// needs to be set together with the chunk unit size to make sense.
    /// When set to null, it indicates that context length is unlimited.
    std::optional<SizeType32> mMaxContextLength;

    std::optional<batch_scheduler::ContextChunkingConfig> mCtxChunkConfig;

    /// The state until/after which the scheduler should not schedule requests
    LlmRequestState mNoScheduleUntilState;
    LlmRequestState mNoScheduleAfterState;
};

} // namespace tensorrt_llm::batch_manager
