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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/types.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor
{

enum class MpiId : uint64_t
{
    PENDING_REQUEST = 1,
    RESPONSE = 2,
    CANCEL_REQUEST = 3,
    TERMINATION = 4,
    ITER_STATS = 5,
    REQUEST_ITER_STATS = 6,
};

struct PendingRequestData
{
    std::vector<RequestWithId> requests;
};

struct RequestIdsData
{
    std::vector<IdType> ids;
};

struct ResponseData
{
    std::vector<Response> responses;
};

struct IterStatsData
{
    std::vector<IterationStats> iterStatsVec;
};

struct RequestStatsPerIterationData
{
    std::vector<RequestStatsPerIteration> requestStatsPerIterationVec;
};

using MpiMessageData
    = std::variant<PendingRequestData, RequestIdsData, ResponseData, IterStatsData, RequestStatsPerIterationData>;

struct MpiMessage
{
    MpiMessage(MpiId _id)
        : id(_id)
    {
    }

    MpiId id;

    MpiMessageData data;
};

} // namespace tensorrt_llm::executor
