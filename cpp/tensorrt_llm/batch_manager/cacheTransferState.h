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

#include "tensorrt_llm/common/assert.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager::detail
{

using TransferRequestId = std::uint64_t;

enum TransferStateFlag : std::uint64_t
{
    kTransferCompleted = 1U << 0,
    kTransferFailed = 1U << 1,
    kTransferTimedOut = 1U << 2,
};

enum TransferGlobalFlag : std::uint64_t
{
    kTransferBufferPoisoned = 1U << 0,
};

struct TransferTopologyOutcome
{
    // Logical failure is a union: one failed or timed-out rank fails the request everywhere.
    std::unordered_set<TransferRequestId> failedRequestIds;
    std::unordered_set<TransferRequestId> timedOutRequestIds;
    // Quiescence is an intersection: every rank has reached a local terminal state.
    std::unordered_set<TransferRequestId> quiescedRequestIds;
    std::unordered_set<TransferRequestId> completedRequestIds;
    bool transferBufferPoisoned{false};
};

enum class ReadySignalSummary
{
    kReady,
    kNotReady,
    kPartiallyReady,
};

inline ReadySignalSummary summarizeReadySignals(bool const allPeersReady, bool const anyPeerReady)
{
    TLLM_CHECK_WITH_INFO(!allPeersReady || anyPeerReady, "An all-ready response must contain at least one ready peer.");
    if (allPeersReady)
    {
        return ReadySignalSummary::kReady;
    }
    return anyPeerReady ? ReadySignalSummary::kPartiallyReady : ReadySignalSummary::kNotReady;
}

// Each rank payload starts with global flags, followed by request-id/state-flag pairs.
inline TransferTopologyOutcome summarizePackedTransferStates(
    std::vector<std::vector<std::uint64_t>> const& rankPayloads)
{
    TLLM_CHECK_WITH_INFO(!rankPayloads.empty(), "Packed transfer state summary requires at least one rank payload.");

    struct Counts
    {
        int completed{0};
        int failed{0};
        int timedOut{0};
    };

    std::unordered_map<TransferRequestId, Counts> countsByRequest;
    TransferTopologyOutcome outcome;
    for (auto const& payload : rankPayloads)
    {
        TLLM_CHECK_WITH_INFO(!payload.empty() && ((payload.size() - 1) % 2 == 0),
            "Packed transfer state payload must contain global flags followed by request/state pairs.");
        outcome.transferBufferPoisoned
            = outcome.transferBufferPoisoned || ((payload.front() & kTransferBufferPoisoned) != 0);

        std::unordered_set<TransferRequestId> seenRequestIds;
        for (size_t index = 1; index < payload.size(); index += 2)
        {
            auto const requestId = payload.at(index);
            auto const flags = payload.at(index + 1);
            TLLM_CHECK_WITH_INFO(seenRequestIds.insert(requestId).second,
                "Packed transfer state payload contains duplicate request ID %lu.", requestId);
            TLLM_CHECK_WITH_INFO(!((flags & kTransferCompleted) != 0 && (flags & kTransferFailed) != 0),
                "Request %lu cannot be both locally completed and failed.", requestId);

            auto& counts = countsByRequest[requestId];
            counts.completed += (flags & kTransferCompleted) != 0;
            counts.failed += (flags & kTransferFailed) != 0;
            counts.timedOut += (flags & kTransferTimedOut) != 0;
        }
    }

    auto const rankCount = static_cast<int>(rankPayloads.size());
    for (auto const& [requestId, counts] : countsByRequest)
    {
        auto const terminalCount = counts.completed + counts.failed;
        bool const logicallyFailed = counts.failed > 0 || counts.timedOut > 0;
        if (logicallyFailed)
        {
            outcome.failedRequestIds.insert(requestId);
        }
        if (counts.timedOut > 0)
        {
            outcome.timedOutRequestIds.insert(requestId);
        }
        if (terminalCount == rankCount)
        {
            outcome.quiescedRequestIds.insert(requestId);
            if (!logicallyFailed && counts.completed == rankCount)
            {
                outcome.completedRequestIds.insert(requestId);
            }
        }
    }
    return outcome;
}

} // namespace tensorrt_llm::batch_manager::detail
