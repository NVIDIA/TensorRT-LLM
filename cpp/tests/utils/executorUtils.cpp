/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "executorUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <chrono>
#include <unordered_set>

std::unordered_map<tensorrt_llm::batch_manager::RequestIdType, std::vector<tensorrt_llm::executor::Response>>
tensorrt_llm::testing::runThroughRequests(executor::Executor& executor, std::vector<executor::Request> const& requests,
    std::chrono::duration<float, std::milli> timeout)
{
    std::unordered_map<batch_manager::RequestIdType, std::vector<executor::Response>> accumulatedResponses;
    auto const requestIds = executor.enqueueRequests(requests);
    TLLM_CHECK_WITH_INFO(requestIds.size() == requests.size(),
        "Expected %zu request IDs, got %zu", requests.size(), requestIds.size());

    auto pendingRequestIds = std::unordered_set<batch_manager::RequestIdType>(requestIds.begin(), requestIds.end());
    TLLM_CHECK_WITH_INFO(
        pendingRequestIds.size() == requestIds.size(), "Executor returned duplicate request IDs");

    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!pendingRequestIds.empty())
    {
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline)
        {
            TLLM_THROW("Timed out waiting for executor responses. Remaining requests: %zu", pendingRequestIds.size());
        }

        auto const remainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        auto const responses = executor.awaitResponses(remainingTime);
        if (responses.empty())
        {
            TLLM_THROW("Timed out waiting for executor responses. Remaining requests: %zu", pendingRequestIds.size());
        }

        for (auto const& response : responses)
        {
            auto const requestId = response.getRequestId();
            auto const requestIt = pendingRequestIds.find(requestId);
            if (requestIt == pendingRequestIds.end())
            {
                TLLM_THROW("Received response for unexpected or completed request: %lu", requestId);
            }
            if (response.hasError())
            {
                TLLM_LOG_ERROR("Error response received for request: %lu", requestId);
                TLLM_THROW(response.getErrorMsg());
            }
            auto const isFinal = response.getResult().isFinal;
            accumulatedResponses[requestId].emplace_back(response);
            if (isFinal)
            {
                TLLM_LOG_DEBUG("Final response received for request: %lu", requestId);
                pendingRequestIds.erase(requestIt);
            }
        }
    }
    return accumulatedResponses;
}
