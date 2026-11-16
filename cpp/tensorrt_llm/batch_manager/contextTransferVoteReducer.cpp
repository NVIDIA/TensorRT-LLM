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

#include "tensorrt_llm/batch_manager/contextTransferCoordinator.h"

#include "tensorrt_llm/common/assert.h"

namespace tensorrt_llm::batch_manager
{

ContextTransferVoteReducer::ContextTransferVoteReducer(int const participantCount)
    : mParticipantCount(participantCount)
{
    TLLM_CHECK_WITH_INFO(participantCount > 0, "Context-transfer consensus requires at least one participant.");
}

void ContextTransferVoteReducer::recordVote(
    int const participantRank, std::uint64_t const requestId, ContextTransferVote const vote)
{
    TLLM_CHECK_WITH_INFO(participantRank >= 0 && participantRank < mParticipantCount,
        "Context-transfer consensus participant rank is out of range.");
    TLLM_CHECK_WITH_INFO(vote == ContextTransferVote::kCompleted || vote == ContextTransferVote::kFailed,
        "Context-transfer consensus received an invalid vote.");

    auto [requestIt, inserted] = mRequestVotes.try_emplace(requestId, mParticipantCount);
    static_cast<void>(inserted);
    auto& requestVotes = requestIt->second;
    auto& recordedVote = requestVotes.votes.at(static_cast<std::size_t>(participantRank));
    auto const packedVote = static_cast<std::uint64_t>(vote);
    if (recordedVote != 0)
    {
        TLLM_CHECK_WITH_INFO(
            recordedVote == packedVote, "Context-transfer participant changed its terminal vote for a request.");
        return;
    }

    recordedVote = packedVote;
    ++requestVotes.terminalCount;
    requestVotes.failed = requestVotes.failed || vote == ContextTransferVote::kFailed;
}

void ContextTransferVoteReducer::recordTimeout(std::uint64_t const requestId)
{
    auto [requestIt, inserted] = mRequestVotes.try_emplace(requestId, mParticipantCount);
    static_cast<void>(inserted);
    auto& requestVotes = requestIt->second;
    if (!requestVotes.timedOut)
    {
        requestVotes.timedOut = true;
        requestVotes.timeoutPending = true;
    }
}

ContextTransferConsensusResult ContextTransferVoteReducer::takeReady()
{
    ContextTransferConsensusResult result;
    for (auto requestIt = mRequestVotes.begin(); requestIt != mRequestVotes.end();)
    {
        auto& requestVotes = requestIt->second;
        if (requestVotes.timeoutPending)
        {
            result.timedOutRequestIds.insert(requestIt->first);
            requestVotes.timeoutPending = false;
        }
        if (requestVotes.terminalCount != mParticipantCount)
        {
            ++requestIt;
            continue;
        }

        auto& terminalRequestIds
            = (requestVotes.failed || requestVotes.timedOut) ? result.failedRequestIds : result.completedRequestIds;
        terminalRequestIds.insert(requestIt->first);
        requestIt = mRequestVotes.erase(requestIt);
    }
    return result;
}

void ContextTransferVoteReducer::clear() noexcept
{
    mRequestVotes.clear();
}

} // namespace tensorrt_llm::batch_manager
