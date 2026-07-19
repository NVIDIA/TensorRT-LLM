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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class CacheTransceiverComm;

enum class ContextTransferVote : std::uint64_t
{
    kCompleted = 1,
    kFailed = 2,
};

struct ContextTransferConsensusResult
{
    std::unordered_set<std::uint64_t> completedRequestIds;
    std::unordered_set<std::uint64_t> failedRequestIds;
};

//! Accumulates one immutable terminal vote per participant and request.
class ContextTransferVoteReducer
{
public:
    explicit ContextTransferVoteReducer(int participantCount);

    void recordVote(int participantRank, std::uint64_t requestId, ContextTransferVote vote);

    [[nodiscard]] ContextTransferConsensusResult takeCompleted();

    [[nodiscard]] std::size_t size() const noexcept;

    void clear() noexcept;

private:
    struct RequestVotes
    {
        explicit RequestVotes(int participantCount)
            : votes(static_cast<std::size_t>(participantCount), 0)
        {
        }

        std::vector<std::uint64_t> votes;
        int terminalCount{0};
        bool failed{false};
    };

    int mParticipantCount;
    std::unordered_map<std::uint64_t, RequestVotes> mRequestVotes;
};

//! Diagnostic CTX protocol that asynchronously gathers PP votes at the last stage and broadcasts the decision.
class ContextTransferCoordinator
{
public:
    explicit ContextTransferCoordinator(std::shared_ptr<CacheTransceiverComm> comm);
    ~ContextTransferCoordinator();

    ContextTransferCoordinator(ContextTransferCoordinator const&) = delete;
    ContextTransferCoordinator& operator=(ContextTransferCoordinator const&) = delete;

    //! Publish this rank's immutable local terminal outcome without waiting for peers.
    void publishLocalOutcome(std::uint64_t requestId, bool failed);

    //! Make nonblocking protocol progress and return newly committed global outcomes.
    [[nodiscard]] ContextTransferConsensusResult poll();

    //! Exchange ordered close markers so no active MPI request outlives its backing buffer.
    void shutdown() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace tensorrt_llm::batch_manager
