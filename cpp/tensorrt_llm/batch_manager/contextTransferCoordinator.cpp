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

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <list>
#include <thread>
#include <utility>

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

class ContextTransferCoordinator::Impl
{
public:
    explicit Impl(std::shared_ptr<CacheTransceiverComm> comm)
        : mComm(std::move(comm))
        , mCoordinatorRank(0)
        , mReducer(mComm ? mComm->getSize() : 1)
    {
        TLLM_CHECK_WITH_INFO(mComm != nullptr, "Context-transfer coordination requires a communicator.");
        TLLM_CHECK_WITH_INFO(mComm->isMpi(), "Asynchronous context-transfer coordination requires MPI.");
        TLLM_CHECK_WITH_INFO(
            mComm->getSize() > 1, "Asynchronous context-transfer coordination requires multiple participants.");
        mCoordinatorRank = mComm->getSize() - 1;
    }

    void publishLocalOutcome(std::uint64_t const requestId, bool const failed)
    {
        TLLM_CHECK_WITH_INFO(!mShutdown, "Cannot publish a context-transfer vote after coordinator shutdown.");
        auto const vote = failed ? ContextTransferVote::kFailed : ContextTransferVote::kCompleted;
        auto const [voteIt, inserted] = mPublishedLocalVotes.emplace(requestId, vote);
        TLLM_CHECK_WITH_INFO(
            inserted || voteIt->second == vote, "This rank changed its terminal vote for a context transfer.");
        if (!inserted)
        {
            return;
        }

        try
        {
            if (isCoordinator())
            {
                mReducer.recordVote(mComm->getRank(), requestId, vote);
            }
            else
            {
                queuePacket(
                    requestId, static_cast<std::uint64_t>(vote), mCoordinatorRank, mpi::MpiTag::kContextTransferEvent);
            }
        }
        catch (...)
        {
            mPublishedLocalVotes.erase(voteIt);
            throw;
        }
    }

    void publishTimeout(std::uint64_t const requestId)
    {
        TLLM_CHECK_WITH_INFO(!mShutdown, "Cannot publish a context-transfer timeout after coordinator shutdown.");
        TLLM_CHECK_WITH_INFO(mPublishedLocalVotes.find(requestId) == mPublishedLocalVotes.end(),
            "A context-transfer timeout must be published before its immutable terminal vote.");
        auto const [timeoutIt, inserted] = mPublishedTimeouts.insert(requestId);
        if (!inserted)
        {
            return;
        }

        try
        {
            if (isCoordinator())
            {
                mReducer.recordTimeout(requestId);
            }
            else
            {
                queuePacket(requestId, kTimedOutSignal, mCoordinatorRank, mpi::MpiTag::kContextTransferEvent);
            }
        }
        catch (...)
        {
            mPublishedTimeouts.erase(timeoutIt);
            throw;
        }
    }

    [[nodiscard]] ContextTransferConsensusResult poll()
    {
        TLLM_CHECK_WITH_INFO(!mShutdown, "Cannot poll context-transfer coordination after shutdown.");
        return progress();
    }

    void shutdown() noexcept
    {
        if (mShutdown)
        {
            return;
        }
        mShutdown = true;

        try
        {
            if (!isCoordinator())
            {
                queuePacket(/*requestId=*/0, kCloseMarker, mCoordinatorRank, mpi::MpiTag::kContextTransferEvent);
            }

            auto const deadline = std::chrono::steady_clock::now() + kShutdownTimeout;
            while (!shutdownComplete())
            {
                static_cast<void>(progress());
                if (!shutdownComplete())
                {
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        TLLM_LOG_ERROR(
                            "Timed out shutting down asynchronous context-transfer coordinator; rank=%d "
                            "peer_closes=%zu/%d ack=%d pending_sends=%zu. Aborting to avoid freeing active MPI "
                            "requests.",
                            mComm->getRank(), mClosedPeers.size(), mComm->getSize() - 1, mCloseAcknowledged,
                            mPendingSends.size());
                        std::abort();
                    }
                    std::this_thread::yield();
                }
            }
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Failed to shut down asynchronous context-transfer coordinator: %s", error.what());
            std::abort();
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Failed to shut down asynchronous context-transfer coordinator with an unknown error.");
            std::abort();
        }
    }

private:
    static constexpr std::size_t kPacketFieldCount = 2;
    static constexpr std::uint64_t kTimedOutSignal = 3;
    static constexpr std::uint64_t kCloseMarker = 4;
    static constexpr auto kShutdownTimeout = std::chrono::seconds(30);

    struct PendingSend
    {
        std::array<std::uint64_t, kPacketFieldCount> packet{};
        std::unique_ptr<mpi::MpiRequest> request;
    };

    [[nodiscard]] bool isCoordinator() const
    {
        return mComm->getRank() == mCoordinatorRank;
    }

    void queuePacket(std::uint64_t const requestId, std::uint64_t const value, int const peer, mpi::MpiTag const tag)
    {
        mPendingSends.emplace_back();
        auto& pendingSend = mPendingSends.back();
        pendingSend.packet = {requestId, value};
        try
        {
            pendingSend.request = mComm->sendAsync(
                pendingSend.packet.data(), pendingSend.packet.size(), mpi::MpiType::kUINT64, peer, tag);
        }
        catch (...)
        {
            mPendingSends.pop_back();
            throw;
        }
    }

    void queueUpdateForPeers(std::uint64_t const requestId, std::uint64_t const update)
    {
        for (int peer = 0; peer < mComm->getSize(); ++peer)
        {
            if (peer != mCoordinatorRank)
            {
                queuePacket(requestId, update, peer, mpi::MpiTag::kContextTransferUpdate);
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

    void drainVotes()
    {
        for (int peer = 0; peer < mComm->getSize(); ++peer)
        {
            if (peer == mCoordinatorRank)
            {
                continue;
            }

            MPI_Status status{};
            while (mComm->iprobe(peer, mpi::MpiTag::kContextTransferEvent, &status))
            {
                std::array<std::uint64_t, kPacketFieldCount> packet{};
                mComm->recv(
                    packet.data(), packet.size(), mpi::MpiType::kUINT64, peer, mpi::MpiTag::kContextTransferEvent);
                if (packet.back() == kCloseMarker)
                {
                    TLLM_CHECK_WITH_INFO(
                        mClosedPeers.insert(peer).second, "Received a duplicate context-transfer close marker.");
                    continue;
                }
                TLLM_CHECK_WITH_INFO(mClosedPeers.find(peer) == mClosedPeers.end(),
                    "Received a context-transfer vote after its peer close marker.");
                if (packet.back() == kTimedOutSignal)
                {
                    mReducer.recordTimeout(packet.front());
                    continue;
                }
                mReducer.recordVote(peer, packet.front(), static_cast<ContextTransferVote>(packet.back()));
            }
        }
    }

    ContextTransferConsensusResult completeCoordinatorUpdates()
    {
        auto result = mReducer.takeReady();
        for (auto const requestId : result.timedOutRequestIds)
        {
            queueUpdateForPeers(requestId, kTimedOutSignal);
        }
        for (auto const requestId : result.failedRequestIds)
        {
            queueUpdateForPeers(requestId, static_cast<std::uint64_t>(ContextTransferVote::kFailed));
            mPublishedLocalVotes.erase(requestId);
            mPublishedTimeouts.erase(requestId);
        }
        for (auto const requestId : result.completedRequestIds)
        {
            queueUpdateForPeers(requestId, static_cast<std::uint64_t>(ContextTransferVote::kCompleted));
            mPublishedLocalVotes.erase(requestId);
            mPublishedTimeouts.erase(requestId);
        }
        return result;
    }

    ContextTransferConsensusResult drainUpdates()
    {
        ContextTransferConsensusResult result;
        MPI_Status status{};
        while (mComm->iprobe(mCoordinatorRank, mpi::MpiTag::kContextTransferUpdate, &status))
        {
            std::array<std::uint64_t, kPacketFieldCount> packet{};
            mComm->recv(packet.data(), packet.size(), mpi::MpiType::kUINT64, mCoordinatorRank,
                mpi::MpiTag::kContextTransferUpdate);
            if (packet.back() == kCloseMarker)
            {
                TLLM_CHECK_WITH_INFO(!mCloseAcknowledged, "Received a duplicate coordinator close marker.");
                mCloseAcknowledged = true;
                mPublishedLocalVotes.clear();
                mPublishedTimeouts.clear();
                continue;
            }

            if (packet.back() == kTimedOutSignal)
            {
                result.timedOutRequestIds.insert(packet.front());
                continue;
            }

            auto const outcome = static_cast<ContextTransferVote>(packet.back());
            TLLM_CHECK_WITH_INFO(outcome == ContextTransferVote::kCompleted || outcome == ContextTransferVote::kFailed,
                "Received an invalid context-transfer commit outcome.");
            auto const localVoteIt = mPublishedLocalVotes.find(packet.front());
            TLLM_CHECK_WITH_INFO(localVoteIt != mPublishedLocalVotes.end(),
                "Received a context-transfer commit before publishing the local terminal vote.");
            if (outcome == ContextTransferVote::kFailed)
            {
                result.failedRequestIds.insert(packet.front());
            }
            else
            {
                result.completedRequestIds.insert(packet.front());
            }
            mPublishedLocalVotes.erase(localVoteIt);
            mPublishedTimeouts.erase(packet.front());
        }
        return result;
    }

    ContextTransferConsensusResult progress()
    {
        reapCompletedSends();
        if (isCoordinator())
        {
            drainVotes();
            auto result = completeCoordinatorUpdates();
            if (mShutdown && !mCloseSent && mClosedPeers.size() == static_cast<std::size_t>(mComm->getSize() - 1))
            {
                // Shutdown is an explicit abort epoch. A peer close proves that no more votes will arrive from that
                // peer, so incomplete requests cannot reach a global decision and are intentionally abandoned.
                mReducer.clear();
                mPublishedLocalVotes.clear();
                mPublishedTimeouts.clear();
                for (int peer = 0; peer < mCoordinatorRank; ++peer)
                {
                    queuePacket(/*requestId=*/0, kCloseMarker, peer, mpi::MpiTag::kContextTransferUpdate);
                }
                mCloseSent = true;
            }
            return result;
        }
        return drainUpdates();
    }

    [[nodiscard]] bool shutdownComplete() const
    {
        if (isCoordinator())
        {
            return mCloseSent && mPendingSends.empty();
        }
        return mCloseAcknowledged && mPendingSends.empty();
    }

    std::shared_ptr<CacheTransceiverComm> mComm;
    int mCoordinatorRank;
    ContextTransferVoteReducer mReducer;
    std::unordered_map<std::uint64_t, ContextTransferVote> mPublishedLocalVotes;
    std::unordered_set<std::uint64_t> mPublishedTimeouts;
    std::unordered_set<int> mClosedPeers;
    std::list<PendingSend> mPendingSends;
    bool mShutdown{false};
    bool mCloseSent{false};
    bool mCloseAcknowledged{false};
};

ContextTransferCoordinator::ContextTransferCoordinator(std::shared_ptr<CacheTransceiverComm> comm)
    : mImpl(std::make_unique<Impl>(std::move(comm)))
{
}

ContextTransferCoordinator::~ContextTransferCoordinator()
{
    shutdown();
}

void ContextTransferCoordinator::publishLocalOutcome(std::uint64_t const requestId, bool const failed)
{
    mImpl->publishLocalOutcome(requestId, failed);
}

void ContextTransferCoordinator::publishTimeout(std::uint64_t const requestId)
{
    mImpl->publishTimeout(requestId);
}

ContextTransferConsensusResult ContextTransferCoordinator::poll()
{
    return mImpl->poll();
}

void ContextTransferCoordinator::shutdown() noexcept
{
    if (mImpl)
    {
        mImpl->shutdown();
    }
}

} // namespace tensorrt_llm::batch_manager
