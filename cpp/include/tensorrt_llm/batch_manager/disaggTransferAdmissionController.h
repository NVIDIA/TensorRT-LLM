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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <cstddef>
#include <optional>

namespace tensorrt_llm::batch_manager
{

class DisaggTransferAdmissionController
{
public:
    enum class Policy
    {
        kFcfsEstimatedBlockBudget
    };

    struct Result
    {
        RequestVector admittedRequests;
        std::size_t activeTransferBlocks{};
        std::size_t admittedTransferBlocks{};
        std::size_t deferredRequestCount{};
        bool limitedByBudget{};

        [[nodiscard]] bool isBlockedByActiveTransfers() const
        {
            return limitedByBudget && admittedRequests.empty() && activeTransferBlocks > 0;
        }
    };

    explicit DisaggTransferAdmissionController(std::optional<std::size_t> maxTokensInBuffer, SizeType32 tokensPerBlock,
        Policy policy = Policy::kFcfsEstimatedBlockBudget)
        : mMaxTransferBlocks(toBlockBudget(maxTokensInBuffer, tokensPerBlock))
        , mTokensPerBlock(tokensPerBlock)
        , mPolicy(policy)
    {
    }

    [[nodiscard]] bool enabled() const
    {
        return mMaxTransferBlocks.has_value();
    }

    [[nodiscard]] std::optional<std::size_t> getMaxTransferBlocks() const
    {
        return mMaxTransferBlocks;
    }

    [[nodiscard]] Result select(RequestList const& activeRequests, RequestVector const& candidates) const
    {
        if (!enabled())
        {
            return Result{
                candidates, estimateActiveTransferBlocks(activeRequests), estimateRequestsBlocks(candidates), 0, false};
        }

        switch (mPolicy)
        {
        case Policy::kFcfsEstimatedBlockBudget: return selectFcfsEstimatedBlockBudget(activeRequests, candidates);
        }

        return Result{};
    }

private:
    [[nodiscard]] static std::optional<std::size_t> toBlockBudget(
        std::optional<std::size_t> maxTokensInBuffer, SizeType32 tokensPerBlock)
    {
        if (!maxTokensInBuffer.has_value() || maxTokensInBuffer.value() == 0 || tokensPerBlock <= 0)
        {
            return std::nullopt;
        }
        auto const blockSize = static_cast<std::size_t>(tokensPerBlock);
        return (maxTokensInBuffer.value() + blockSize - 1) / blockSize;
    }

    [[nodiscard]] std::size_t estimateRequestBlocks(LlmRequest const& request) const
    {
        if (mTokensPerBlock <= 0)
        {
            return 0;
        }
        auto const promptLen = static_cast<std::size_t>(request.getPromptLen());
        auto const blockSize = static_cast<std::size_t>(mTokensPerBlock);
        return (promptLen + blockSize - 1) / blockSize;
    }

    [[nodiscard]] std::size_t estimateRequestsBlocks(RequestVector const& requests) const
    {
        std::size_t blocks{};
        for (auto const& request : requests)
        {
            blocks += estimateRequestBlocks(*request);
        }
        return blocks;
    }

    [[nodiscard]] std::size_t estimateActiveTransferBlocks(RequestList const& activeRequests) const
    {
        std::size_t blocks{};
        for (auto const& request : activeRequests)
        {
            if (request->isDisaggGenerationTransmissionInProgress())
            {
                blocks += estimateRequestBlocks(*request);
            }
        }
        return blocks;
    }

    [[nodiscard]] Result selectFcfsEstimatedBlockBudget(
        RequestList const& activeRequests, RequestVector const& candidates) const
    {
        Result result;
        result.activeTransferBlocks = estimateActiveTransferBlocks(activeRequests);

        auto const maxTransferBlocks = mMaxTransferBlocks.value();
        auto usedBlocks = result.activeTransferBlocks;
        for (auto const& request : candidates)
        {
            auto const requestBlocks = estimateRequestBlocks(*request);
            bool const fitsBudget = usedBlocks + requestBlocks <= maxTransferBlocks;
            bool const admitOversizedHead = result.admittedRequests.empty() && result.activeTransferBlocks == 0
                && requestBlocks > maxTransferBlocks;
            if (!fitsBudget && !admitOversizedHead)
            {
                result.limitedByBudget = true;
                break;
            }

            result.admittedRequests.push_back(request);
            usedBlocks += requestBlocks;
            result.admittedTransferBlocks += requestBlocks;
        }

        result.deferredRequestCount = candidates.size() - result.admittedRequests.size();
        return result;
    }

    std::optional<std::size_t> mMaxTransferBlocks;
    SizeType32 mTokensPerBlock;
    Policy mPolicy;
};

} // namespace tensorrt_llm::batch_manager
