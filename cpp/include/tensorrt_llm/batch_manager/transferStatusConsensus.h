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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class CacheTransceiverComm;

struct WorkerPublishedConsensusConfig
{
    bool enabledForCppBinding{false};
    bool mpiControlPlane{false};
    bool nixlUcxBackend{false};
    int tensorParallelism{1};
    int contextParallelism{1};
    int pipelineParallelism{1};
    bool attentionDp{false};
    bool inflightCancellation{false};
    bool transferOverlap{true};
    bool layerwiseTransfer{false};
    bool mpiThreadMultiple{false};
};

//! Return whether the worker-published protocol is supported by one rank's effective configuration.
[[nodiscard]] bool supportsWorkerPublishedConsensus(WorkerPublishedConsensusConfig const& config);

//! Return whether every rank advertised the same nonzero protocol version.
[[nodiscard]] bool selectWorkerPublishedConsensus(std::vector<std::uint64_t> const& protocolVersions);

enum class TransferStatusVote : std::uint64_t
{
    kCompleted = 1,
    kFailed = 2,
};

struct TransferStatusConsensusResult
{
    std::unordered_set<std::uint64_t> completedRequestIds;
    std::unordered_set<std::uint64_t> failedRequestIds;
};

//! Reduces one immutable terminal vote per participant and request.
class TransferStatusConsensus
{
public:
    explicit TransferStatusConsensus(int participantCount);

    //! Record a participant's terminal vote. Identical duplicates are ignored; conflicting votes are rejected.
    void recordVote(int participantRank, std::uint64_t requestId, TransferStatusVote vote);

    //! Return and remove every request for which all participants have cast a terminal vote.
    [[nodiscard]] TransferStatusConsensusResult takeCompleted();

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

//! Exchanges immutable worker-published terminal votes without requiring peer scheduler participation.
class ContextTransferVoteMailbox
{
public:
    explicit ContextTransferVoteMailbox(std::shared_ptr<CacheTransceiverComm> comm);
    ~ContextTransferVoteMailbox();

    ContextTransferVoteMailbox(ContextTransferVoteMailbox const&) = delete;
    ContextTransferVoteMailbox& operator=(ContextTransferVoteMailbox const&) = delete;

    //! Publish one terminal outcome to every peer. The first internal error is reported by a later scheduler poll.
    void publishOutcomeToPeers(std::uint64_t requestId, bool failed) noexcept;

    //! Record the already-published local worker outcome in this rank's reducer.
    [[nodiscard]] bool recordLocalOutcome(std::uint64_t requestId);

    //! Drain peer votes and return newly committed global outcomes.
    [[nodiscard]] TransferStatusConsensusResult poll();

    //! Stop accepting publications and exchange an ordered close marker with every peer.
    void shutdown() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace tensorrt_llm::batch_manager
