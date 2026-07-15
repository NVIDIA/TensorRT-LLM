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

#include "tensorrt_llm/batch_manager/transferStatusConsensus.h"

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <list>
#include <mutex>
#include <thread>
#include <utility>

namespace tensorrt_llm::batch_manager
{

bool supportsWorkerPublishedConsensus(WorkerPublishedConsensusConfig const& config)
{
    return config.enabledForCppBinding && config.mpiControlPlane && config.nixlUcxBackend
        && config.tensorParallelism == 1 && config.contextParallelism == 1 && config.pipelineParallelism > 1
        && !config.attentionDp && !config.inflightCancellation && config.transferOverlap && !config.layerwiseTransfer
        && config.mpiThreadMultiple;
}

bool selectWorkerPublishedConsensus(std::vector<std::uint64_t> const& protocolVersions)
{
    if (protocolVersions.empty() || protocolVersions.front() == 0)
    {
        return false;
    }
    return std::all_of(protocolVersions.begin(), protocolVersions.end(),
        [&](std::uint64_t const version) { return version == protocolVersions.front(); });
}

TransferStatusConsensus::TransferStatusConsensus(int const participantCount)
    : mParticipantCount(participantCount)
{
    TLLM_CHECK_WITH_INFO(participantCount > 0, "Transfer-status consensus requires at least one participant.");
}

void TransferStatusConsensus::recordVote(
    int const participantRank, std::uint64_t const requestId, TransferStatusVote const vote)
{
    TLLM_CHECK_WITH_INFO(participantRank >= 0 && participantRank < mParticipantCount,
        "Transfer-status consensus participant rank is out of range.");
    TLLM_CHECK_WITH_INFO(vote == TransferStatusVote::kCompleted || vote == TransferStatusVote::kFailed,
        "Transfer-status consensus received an invalid vote.");

    auto [requestIt, inserted] = mRequestVotes.try_emplace(requestId, mParticipantCount);
    static_cast<void>(inserted);
    auto& requestVotes = requestIt->second;
    auto& recordedVote = requestVotes.votes.at(static_cast<std::size_t>(participantRank));
    auto const packedVote = static_cast<std::uint64_t>(vote);
    if (recordedVote != 0)
    {
        TLLM_CHECK_WITH_INFO(recordedVote == packedVote,
            "Transfer-status consensus participant changed its terminal vote for a request.");
        return;
    }

    recordedVote = packedVote;
    ++requestVotes.terminalCount;
    requestVotes.failed = requestVotes.failed || vote == TransferStatusVote::kFailed;
}

TransferStatusConsensusResult TransferStatusConsensus::takeCompleted()
{
    TransferStatusConsensusResult result;
    for (auto requestIt = mRequestVotes.begin(); requestIt != mRequestVotes.end();)
    {
        auto const& requestVotes = requestIt->second;
        if (requestVotes.terminalCount != mParticipantCount)
        {
            ++requestIt;
            continue;
        }

        auto& terminalRequestIds = requestVotes.failed ? result.failedRequestIds : result.completedRequestIds;
        terminalRequestIds.insert(requestIt->first);
        requestIt = mRequestVotes.erase(requestIt);
    }
    return result;
}

class ContextTransferVoteMailbox::Impl
{
public:
    explicit Impl(std::shared_ptr<CacheTransceiverComm> comm)
        : mComm(std::move(comm))
        , mConsensus(mComm->getSize())
    {
        TLLM_CHECK_WITH_INFO(mComm->isMpi(), "Worker-published context-transfer consensus requires MPI.");
        TLLM_CHECK_WITH_INFO(mComm->getSize() > 1,
            "Worker-published context-transfer consensus requires multiple pipeline-parallel participants.");
    }

    void publishOutcomeToPeers(std::uint64_t const requestId, bool const failed) noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mShutdown || mError)
        {
            return;
        }

        try
        {
            auto const vote = failed ? TransferStatusVote::kFailed : TransferStatusVote::kCompleted;
            auto const [voteIt, inserted] = mPublishedLocalVotes.emplace(requestId, vote);
            TLLM_CHECK_WITH_INFO(inserted || voteIt->second == vote,
                "Context-transfer worker changed its published terminal vote for a request.");
            if (inserted)
            {
                queuePacketForPeers(requestId, static_cast<std::uint64_t>(vote));
            }
        }
        catch (...)
        {
            mError = std::current_exception();
        }
    }

    [[nodiscard]] bool recordLocalOutcome(std::uint64_t const requestId)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        rethrowError();
        auto const voteIt = mPublishedLocalVotes.find(requestId);
        TLLM_CHECK_WITH_INFO(voteIt != mPublishedLocalVotes.end(),
            "Context-transfer future became ready before its worker published a terminal vote.");
        auto const vote = voteIt->second;
        mConsensus.recordVote(mComm->getRank(), requestId, vote);
        mPublishedLocalVotes.erase(voteIt);
        return vote == TransferStatusVote::kFailed;
    }

    [[nodiscard]] TransferStatusConsensusResult poll()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        rethrowError();
        reapCompletedSends();
        drainIncomingPackets();
        return mConsensus.takeCompleted();
    }

    void shutdown() noexcept
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (mShutdown)
        {
            return;
        }
        mShutdown = true;

        try
        {
            // Sender workers have stopped before shutdown. MPI preserves the order of messages from one source with
            // the same tag, so receiving a close marker proves that every earlier vote from that peer was drained.
            queuePacketForPeers(/*requestId=*/0, kCloseMarker);
            auto const peerCount = static_cast<std::size_t>(mComm->getSize() - 1);
            auto const shutdownDeadline = std::chrono::steady_clock::now() + kShutdownTimeout;
            while (!mPendingSends.empty() || mClosedPeers.size() != peerCount)
            {
                reapCompletedSends();
                drainIncomingPackets();
                if (!mPendingSends.empty() || mClosedPeers.size() != peerCount)
                {
                    if (std::chrono::steady_clock::now() >= shutdownDeadline)
                    {
                        TLLM_LOG_ERROR(
                            "Timed out shutting down context-transfer vote mailbox; received %zu of %zu peer close "
                            "markers with %zu sends still pending. Aborting to avoid freeing active MPI requests.",
                            mClosedPeers.size(), peerCount, mPendingSends.size());
                        std::abort();
                    }
                    lock.unlock();
                    std::this_thread::yield();
                    lock.lock();
                }
            }
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Failed to shut down context-transfer vote mailbox: %s", error.what());
            std::abort();
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Failed to shut down context-transfer vote mailbox with an unknown error");
            std::abort();
        }
    }

private:
    static constexpr std::size_t kPacketFieldCount = 2;
    static constexpr std::uint64_t kCloseMarker = 3;
    static constexpr auto kShutdownTimeout = std::chrono::seconds(30);

    struct PendingSend
    {
        std::array<std::uint64_t, kPacketFieldCount> packet{};
        std::unique_ptr<mpi::MpiRequest> request;
    };

    void rethrowError() const
    {
        if (mError)
        {
            std::rethrow_exception(mError);
        }
    }

    void queuePacketForPeers(std::uint64_t const requestId, std::uint64_t const value)
    {
        for (int peer = 0; peer < mComm->getSize(); ++peer)
        {
            if (peer == mComm->getRank())
            {
                continue;
            }

            mPendingSends.emplace_back();
            auto& pendingSend = mPendingSends.back();
            pendingSend.packet = {requestId, value};
            try
            {
                pendingSend.request = mComm->sendAsync(pendingSend.packet.data(), pendingSend.packet.size(),
                    mpi::MpiType::kUINT64, peer, mpi::MpiTag::kContextTransferStatus);
            }
            catch (...)
            {
                mPendingSends.pop_back();
                throw;
            }
        }
    }

    void reapCompletedSends()
    {
        for (auto sendIt = mPendingSends.begin(); sendIt != mPendingSends.end();)
        {
            TLLM_CHECK(sendIt->request);
            if (sendIt->request->isCompleted())
            {
                sendIt = mPendingSends.erase(sendIt);
            }
            else
            {
                ++sendIt;
            }
        }
    }

    void drainIncomingPackets()
    {
        for (int peer = 0; peer < mComm->getSize(); ++peer)
        {
            if (peer == mComm->getRank())
            {
                continue;
            }

            MPI_Status status{};
            while (mComm->iprobe(peer, mpi::MpiTag::kContextTransferStatus, &status))
            {
                std::array<std::uint64_t, kPacketFieldCount> packet{};
                mComm->recv(
                    packet.data(), packet.size(), mpi::MpiType::kUINT64, peer, mpi::MpiTag::kContextTransferStatus);
                if (packet.back() == kCloseMarker)
                {
                    TLLM_CHECK_WITH_INFO(mClosedPeers.insert(peer).second,
                        "Context-transfer vote mailbox received a duplicate close marker.");
                    continue;
                }
                TLLM_CHECK_WITH_INFO(mClosedPeers.find(peer) == mClosedPeers.end(),
                    "Context-transfer vote mailbox received a vote after its peer close marker.");
                mConsensus.recordVote(peer, packet.front(), static_cast<TransferStatusVote>(packet.back()));
            }
        }
    }

    std::shared_ptr<CacheTransceiverComm> mComm;
    TransferStatusConsensus mConsensus;
    std::unordered_map<std::uint64_t, TransferStatusVote> mPublishedLocalVotes;
    std::unordered_set<int> mClosedPeers;
    std::list<PendingSend> mPendingSends;
    std::mutex mMutex;
    std::exception_ptr mError;
    bool mShutdown{false};
};

ContextTransferVoteMailbox::ContextTransferVoteMailbox(std::shared_ptr<CacheTransceiverComm> comm)
    : mImpl(std::make_unique<Impl>(std::move(comm)))
{
}

ContextTransferVoteMailbox::~ContextTransferVoteMailbox()
{
    shutdown();
}

void ContextTransferVoteMailbox::publishOutcomeToPeers(std::uint64_t const requestId, bool const failed) noexcept
{
    mImpl->publishOutcomeToPeers(requestId, failed);
}

bool ContextTransferVoteMailbox::recordLocalOutcome(std::uint64_t const requestId)
{
    return mImpl->recordLocalOutcome(requestId);
}

TransferStatusConsensusResult ContextTransferVoteMailbox::poll()
{
    return mImpl->poll();
}

void ContextTransferVoteMailbox::shutdown() noexcept
{
    if (mImpl)
    {
        mImpl->shutdown();
    }
}

} // namespace tensorrt_llm::batch_manager
