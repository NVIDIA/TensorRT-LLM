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

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceResult.h"

#include <algorithm>
#include <utility>

namespace tensorrt_llm::executor::kv_cache::bounce
{

char const* toString(BounceFailReason reason)
{
    // Prefixed with the module name: these strings travel up through TransferStatus::
    // getLastStatusStr() into upper-layer error logs, where "RDMA write failed" alone would not
    // say WHICH transport failed.
    switch (reason)
    {
    case BounceFailReason::kNone: return "bounce: none";
    case BounceFailReason::kPlanRejected: return "bounce: plan rejected (request did not fit a transfer plan)";
    case BounceFailReason::kNoProgressTimeout: return "bounce: no GRANT/ACK progress within requestTimeoutMs";
    case BounceFailReason::kPeerDropped: return "bounce: peer dropped (forgetPeer/invalidateRemoteAgent)";
    case BounceFailReason::kGatherFailed: return "bounce: gather kernel failed (CUDA error)";
    case BounceFailReason::kWriteFailed: return "bounce: RDMA write failed";
    case BounceFailReason::kProtocolError: return "bounce: protocol error (GRANT mispair/plan overflow)";
    case BounceFailReason::kShutdown: return "bounce: transport shut down while pending";
    case BounceFailReason::kPeerNack: return "bounce: receiver reported it cannot complete the chunk/request (NACK)";
    }
    return "bounce: unknown";
}

std::chrono::milliseconds answerWaitFor(BounceFailReason trigger, std::chrono::milliseconds requestTimeout)
{
    return trigger == BounceFailReason::kNoProgressTimeout ? std::chrono::milliseconds{0} : requestTimeout;
}

OutstandingAccess::OutstandingAccess(std::vector<SentChunk> unansweredChunks)
    : mUnansweredChunks(std::move(unansweredChunks))
{
}

OutstandingAccess::SourceRead::SourceRead(std::shared_ptr<OutstandingAccess> const& access)
    : mAccess(access)
{
    std::lock_guard<std::mutex> lk(access->mMu);
    ++access->mSourceReads;
}

OutstandingAccess::SourceRead::~SourceRead()
{
    if (mAccess != nullptr) // null once moved from
    {
        std::lock_guard<std::mutex> lk(mAccess->mMu);
        --mAccess->mSourceReads;
    }
}

bool OutstandingAccess::answer(std::uint32_t chunkIdx, std::uint64_t regionHandle)
{
    std::lock_guard<std::mutex> lk(mMu);
    auto const chunk = std::find_if(mUnansweredChunks.begin(), mUnansweredChunks.end(),
        [&](SentChunk const& c) { return c.chunkIdx == chunkIdx && c.regionHandle == regionHandle; });
    if (chunk == mUnansweredChunks.end())
    {
        return false;
    }
    mUnansweredChunks.erase(chunk);
    return true;
}

std::size_t OutstandingAccess::unansweredChunks() const
{
    std::lock_guard<std::mutex> lk(mMu);
    return mUnansweredChunks.size();
}

bool OutstandingAccess::quiesced() const
{
    std::lock_guard<std::mutex> lk(mMu);
    return mSourceReads == 0 && mUnansweredChunks.empty();
}

bool BounceResult::destinationQuiesced() const
{
    return outstanding == nullptr || outstanding->unansweredChunks() == 0;
}

bool BounceResult::quiesced() const
{
    return outstanding == nullptr || outstanding->quiesced();
}

FailedRequest::FailedRequest(BounceFailReason reason, std::shared_ptr<std::promise<BounceResult>> promise,
    std::shared_ptr<OutstandingAccess> outstanding, Clock::duration answerWait, Clock::time_point now)
    : mReason(reason)
    , mAnswerWait(answerWait)
    , mRecord(outstanding)
{
    mAwaiting.emplace(AwaitingAnswers{std::move(promise), std::move(outstanding), now + answerWait});
}

bool FailedRequest::answer(std::uint32_t chunkIdx, std::uint64_t regionHandle, Clock::time_point now)
{
    auto const record = mRecord.lock();
    if (record == nullptr || !record->answer(chunkIdx, regionHandle))
    {
        return false;
    }
    if (mAwaiting)
    {
        mAwaiting->deadline = now + mAnswerWait; // a live receiver gets a fresh wait for each remaining answer
    }
    return true;
}

std::optional<std::size_t> FailedRequest::resolveIfDue(Clock::time_point now)
{
    bool const due = mAwaiting && (mAwaiting->outstanding->unansweredChunks() == 0 || now >= mAwaiting->deadline);
    return due ? resolve() : std::nullopt;
}

std::optional<std::size_t> FailedRequest::resolve()
{
    if (!mAwaiting)
    {
        return std::nullopt;
    }
    AwaitingAnswers awaiting = std::move(*mAwaiting);
    mAwaiting.reset();
    // Count before the record moves into the result.
    auto const unanswered = awaiting.outstanding->unansweredChunks();
    awaiting.promise->set_value({TransferState::kFAILURE, mReason, std::move(awaiting.outstanding)});
    return unanswered;
}

BounceFailReason FailedRequest::reason() const
{
    return mReason;
}

bool FailedRequest::resolved() const
{
    return !mAwaiting;
}

bool FailedRequest::worthKeeping() const
{
    return !resolved() || unansweredChunks() > 0;
}

std::size_t FailedRequest::unansweredChunks() const
{
    auto const record = mRecord.lock();
    return record == nullptr ? 0 : record->unansweredChunks();
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
