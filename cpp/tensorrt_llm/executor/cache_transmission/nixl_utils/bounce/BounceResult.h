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

#include "tensorrt_llm/executor/transferAgent.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace tensorrt_llm::executor::kv_cache::bounce
{

/// Why a bounce request resolved kFAILURE — carried inside the request's result future and
/// surfaced to the upper layer through TransferStatus::getLastStatusStr(); the bare kFAILURE
/// enum alone tells an operator nothing.
enum class BounceFailReason : std::uint8_t
{
    kNone = 0,
    kPlanRejected,      // request did not fit a transfer plan (submit-time TLLM_CHECK)
    kNoProgressTimeout, // no GRANT/ACK progress within requestTimeoutMs (peer unreachable/stalled)
    kPeerDropped,       // forgetPeer()/invalidateRemoteAgent while the request was in flight
    kGatherFailed,      // gather kernel launch / event-record / async poll failure (CUDA error)
    kWriteFailed,       // the RDMA write reported failure (getXferStatus == failed)
    kProtocolError,     // GRANT mispair or plan overflow — the flow was abandoned
    kShutdown,          // transport shut down while the request was still pending
    kPeerNack,          // receiver reported it cannot complete the chunk/request (NACK)
};

[[nodiscard]] char const* toString(BounceFailReason reason);

/// How long a request failed by `trigger` holds its result back for the receiver's answers, restarted by
/// each answer. None after kNoProgressTimeout, which already saw no progress for requestTimeoutMs: the
/// result resolves at once, in-doubt if a sent chunk is unanswered (its DATA may be recent; a late answer
/// still ends it).
[[nodiscard]] std::chrono::milliseconds answerWaitFor(
    BounceFailReason trigger, std::chrono::milliseconds requestTimeout);

/// What a failed request may still do to caller memory after its result is set: gathers that may still
/// read its source, and sent chunks the receiver has not answered — it answers a chunk only after the
/// chunk's scatter has last written the destination. Once the result is set this only shrinks. Updated by
/// the sender's IO thread, read from any thread.
class OutstandingAccess
{
public:
    /// A chunk whose DATA is out, named the way the receiver's ACK/NACK names it.
    struct SentChunk
    {
        std::uint32_t chunkIdx{};
        std::uint64_t regionHandle{};
    };

    /// Recorded once, before the result is set: afterwards answers and SourceReads can only shrink it.
    explicit OutstandingAccess(std::vector<SentChunk> unansweredChunks = {});

    /// One gather that may still read the source, until this is destroyed. It keeps the record alive.
    class SourceRead
    {
    public:
        explicit SourceRead(std::shared_ptr<OutstandingAccess> const& access);
        SourceRead(SourceRead&& other) noexcept = default;
        SourceRead(SourceRead const&) = delete;
        SourceRead& operator=(SourceRead const&) = delete;
        SourceRead& operator=(SourceRead&&) = delete;
        ~SourceRead();

    private:
        std::shared_ptr<OutstandingAccess> mAccess;
    };

    /// The receiver answered (ACK or NACK) the chunk. False if it is not an unanswered chunk.
    bool answer(std::uint32_t chunkIdx, std::uint64_t regionHandle);

    [[nodiscard]] std::size_t unansweredChunks() const;
    [[nodiscard]] bool quiesced() const;

private:
    mutable std::mutex mMu;
    std::uint32_t mSourceReads{0};
    std::vector<SentChunk> mUnansweredChunks;
};

/// A bounce request's single result: the terminal state, on kFAILURE its cause, and what the request may
/// still access. The state and cause never change; the outstanding accesses only end.
struct BounceResult
{
    TransferState state{TransferState::kFAILURE};
    BounceFailReason reason{BounceFailReason::kNone};
    std::shared_ptr<OutstandingAccess const> outstanding; // null: nothing outstanding

    [[nodiscard]] bool destinationQuiesced() const;
    [[nodiscard]] bool quiesced() const;
};

/// A failed request's result, held back until every sent chunk is answered or the answer wait is over (a
/// caller that trusts wait() would otherwise reuse the destination while the receiver may still scatter into
/// it). Answers keep shrinking the record after that, so isCompleted() can turn true after wait() returns.
class FailedRequest
{
public:
    using Clock = std::chrono::steady_clock;

    /// `answerWait` is how long the receiver gets to answer, from now and again after each answer.
    FailedRequest(BounceFailReason reason, std::shared_ptr<std::promise<BounceResult>> promise,
        std::shared_ptr<OutstandingAccess> outstanding, Clock::duration answerWait, Clock::time_point now);

    /// The receiver answered (ACK or NACK) a sent chunk. False if it is not an unanswered chunk.
    bool answer(std::uint32_t chunkIdx, std::uint64_t regionHandle, Clock::time_point now);
    /// Resolve the result once every sent chunk is answered or the answer wait is over. If this call resolved
    /// it: the number of sent chunks still unanswered then.
    std::optional<std::size_t> resolveIfDue(Clock::time_point now);
    /// Resolve the result whatever is still unanswered (shutdown); returns like resolveIfDue.
    std::optional<std::size_t> resolve();

    [[nodiscard]] BounceFailReason reason() const;
    [[nodiscard]] bool resolved() const;
    /// The result is held back, or an answer can still change it (a sent chunk is unanswered and the record
    /// is alive).
    [[nodiscard]] bool worthKeeping() const;

private:
    struct AwaitingAnswers
    {
        std::shared_ptr<std::promise<BounceResult>> promise;
        std::shared_ptr<OutstandingAccess> outstanding; // keeps mRecord alive until the result is set
        Clock::time_point deadline;
    };

    /// 0 once the record has expired.
    [[nodiscard]] std::size_t unansweredChunks() const;

    BounceFailReason mReason;
    Clock::duration mAnswerWait;
    std::optional<AwaitingAnswers> mAwaiting; // empty once resolved
    // Once resolved, the result's holders and running orphaned gathers keep the record alive.
    std::weak_ptr<OutstandingAccess> mRecord;
};

} // namespace tensorrt_llm::executor::kv_cache::bounce
