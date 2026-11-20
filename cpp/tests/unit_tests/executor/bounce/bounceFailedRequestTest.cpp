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
#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceTransferStatus.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace b = tensorrt_llm::executor::kv_cache::bounce;
namespace kvc = tensorrt_llm::executor::kv_cache;

namespace
{

using Clock = b::FailedRequest::Clock;
constexpr auto kAnswerWait = std::chrono::seconds(10);
constexpr std::uint64_t kFirstRegion = 100; // chunk i was granted region kFirstRegion + i
constexpr auto kReason = b::BounceFailReason::kWriteFailed;

b::BounceTransferStatus resolvedStatus(b::BounceResult const& result)
{
    std::promise<b::BounceResult> promise;
    promise.set_value(result);
    return b::BounceTransferStatus{promise.get_future().share()};
}

// Everything that can happen to a failed request: the receiver's answers, time passing, its orphaned
// gather ending, the result's last holder letting go, and shutdown.
enum class Event
{
    kAnswerChunk0,
    kAnswerChunk1,
    kCrossedAnswer,  // names chunk 1 with chunk 0's region
    kTimePasses,     // to just before the answer deadline
    kAnswerWaitEnds, // to the answer deadline
    kGatherEnds,
    kHolderLetsGo,
    kShutdown, // last
};

// Indexed by Event.
constexpr std::array<char const*, 8> kEventNames{
    "answer0", "answer1", "crossedAnswer", "timePasses", "answerWaitEnds", "gatherEnds", "holderLetsGo", "shutdown"};
static_assert(kEventNames.size() == static_cast<std::size_t>(Event::kShutdown) + 1, "one name per Event");

// One failed request driven the way BounceSender drives it — failRequest and answerFailedRequest feed it,
// the per-tick settleFailedRequests (sweep) resolves and drops it, failAll resolves it — next to a model of
// what must hold: which sent chunks are still unanswered, whether its gather still runs, whether anyone
// holds the result, and when the answer wait ends.
class FailedRequestModel
{
public:
    FailedRequestModel(std::uint32_t sentChunks, bool gatherRunning, Clock::duration answerWait)
        : mAnswerWait(answerWait)
    {
        std::vector<b::OutstandingAccess::SentChunk> sent;
        for (std::uint32_t chunk = 0; chunk < sentChunks; ++chunk)
        {
            sent.push_back({.chunkIdx = chunk, .regionHandle = kFirstRegion + chunk});
            mUnanswered.insert(chunk);
        }
        auto outstanding = std::make_shared<b::OutstandingAccess>(std::move(sent));
        if (gatherRunning)
        {
            mGather.emplace(outstanding);
        }
        auto promise = std::make_shared<std::promise<b::BounceResult>>();
        mHolder.emplace(promise->get_future().share());
        mDeadline = mNow + mAnswerWait;
        mFailedRequest.emplace(kReason, std::move(promise), std::move(outstanding), mAnswerWait, mNow);
        sweep();
    }

    void apply(Event event)
    {
        switch (event)
        {
        case Event::kAnswerChunk0: answer(0, kFirstRegion); break;
        case Event::kAnswerChunk1: answer(1, kFirstRegion + 1); break;
        case Event::kCrossedAnswer: answer(1, kFirstRegion); break;
        case Event::kTimePasses: mNow = std::max(mNow, mDeadline - std::chrono::milliseconds(1)); break;
        case Event::kAnswerWaitEnds: mNow = std::max(mNow, mDeadline); break;
        case Event::kGatherEnds: mGather.reset(); break;
        case Event::kHolderLetsGo: mHolder.reset(); break;
        case Event::kShutdown: shutdown(); break;
        }
        sweep(); // every tick ends with settleFailedRequests
    }

    void check()
    {
        if (mFailedRequest)
        {
            EXPECT_EQ(mFailedRequest->resolved(), mResolved);
            bool const worthKeeping = !mResolved || (!mUnanswered.empty() && outstandingAlive());
            EXPECT_EQ(mFailedRequest->worthKeeping(), worthKeeping);
        }
        if (!mHolder)
        {
            return;
        }
        EXPECT_EQ(mHolder->wait(0), mResolved ? kvc::TransferState::kFAILURE : kvc::TransferState::kIN_PROGRESS);
        // Completed exactly when resolved and nothing can still read the source or write the destination.
        bool const completed = mHolder->isCompleted();
        EXPECT_EQ(completed, mResolved && mUnanswered.empty() && !mGather.has_value());
        EXPECT_TRUE(completed || !mWasCompleted) << "isCompleted() went back to false";
        mWasCompleted = completed;
        if (mResolved)
        {
            std::string expected = b::toString(kReason);
            if (!mUnanswered.empty())
            {
                expected += "; destination not proven quiescent";
            }
            EXPECT_EQ(mHolder->getLastStatusStr(), expected);
        }
    }

private:
    // The resolved record lives on through the result's holder and the running gather's source read.
    [[nodiscard]] bool outstandingAlive() const
    {
        return !mResolved || mHolder.has_value() || mGather.has_value();
    }

    // answerFailedRequest: a matching answer ends its chunk and restarts the answer wait.
    void answer(std::uint32_t chunk, std::uint64_t region)
    {
        if (!mFailedRequest)
        {
            return;
        }
        bool const matches = outstandingAlive() && region == kFirstRegion + chunk && mUnanswered.count(chunk) > 0;
        EXPECT_EQ(mFailedRequest->answer(chunk, region, mNow), matches);
        if (matches)
        {
            mUnanswered.erase(chunk);
            if (!mResolved)
            {
                mDeadline = mNow + mAnswerWait;
            }
        }
    }

    // settleFailedRequests: resolve when due, drop once not worth keeping.
    void sweep()
    {
        if (!mFailedRequest)
        {
            return;
        }
        bool const due = !mResolved && (mUnanswered.empty() || mNow >= mDeadline);
        auto const unanswered = mFailedRequest->resolveIfDue(mNow);
        EXPECT_EQ(unanswered.has_value(), due) << "resolved at the wrong moment";
        if (unanswered)
        {
            EXPECT_EQ(*unanswered, mUnanswered.size());
        }
        mResolved = mResolved || due;
        if (!mFailedRequest->worthKeeping())
        {
            mFailedRequest.reset();
        }
    }

    // failAll: whatever is unanswered stays so, since no answer can arrive any more.
    void shutdown()
    {
        if (!mFailedRequest)
        {
            return;
        }
        auto const unanswered = mFailedRequest->resolve();
        EXPECT_EQ(unanswered.has_value(), !mResolved);
        if (unanswered)
        {
            EXPECT_EQ(*unanswered, mUnanswered.size());
        }
        mResolved = true;
        mFailedRequest.reset();
    }

    Clock::duration mAnswerWait;
    Clock::time_point mNow{Clock::time_point{} + std::chrono::hours(1)};
    Clock::time_point mDeadline;
    std::set<std::uint32_t> mUnanswered;
    std::optional<b::OutstandingAccess::SourceRead> mGather;
    std::optional<b::BounceTransferStatus> mHolder;
    std::optional<b::FailedRequest> mFailedRequest;
    bool mResolved{false};
    bool mWasCompleted{false};
};

std::string describe(
    std::uint32_t sentChunks, bool gatherRunning, Clock::duration answerWait, std::vector<Event> const& order)
{
    std::ostringstream out;
    out << "sentChunks=" << sentChunks << " gatherRunning=" << gatherRunning
        << " answerWaitMs=" << std::chrono::duration_cast<std::chrono::milliseconds>(answerWait).count() << " order:";
    for (auto const event : order)
    {
        out << ' ' << kEventNames.at(static_cast<std::size_t>(event));
    }
    return out.str();
}

} // namespace

// Every order of every event that can reach a failed request with up to two sent chunks and one running
// gather, with the usual answer wait and with none (kNoProgressTimeout); the model's checks run after each
// event. Time passes twice, so an answer can fall between the two. Stops at the first order that breaks one.
TEST(BounceFailedRequest, EveryEventOrderKeepsTheContract)
{
    std::array<Clock::duration, 2> const answerWaits{kAnswerWait, Clock::duration::zero()};
    for (std::uint32_t sentChunks = 0; sentChunks <= 2; ++sentChunks)
    {
        for (auto const answerWait : answerWaits)
        {
            for (bool const gatherRunning : {false, true})
            {
                std::vector<Event> order{Event::kCrossedAnswer, Event::kTimePasses, Event::kTimePasses,
                    Event::kAnswerWaitEnds, Event::kHolderLetsGo, Event::kShutdown};
                if (sentChunks >= 1)
                {
                    order.push_back(Event::kAnswerChunk0);
                }
                if (sentChunks >= 2)
                {
                    order.push_back(Event::kAnswerChunk1);
                }
                if (gatherRunning)
                {
                    order.push_back(Event::kGatherEnds);
                }
                std::sort(order.begin(), order.end());
                do
                {
                    FailedRequestModel model(sentChunks, gatherRunning, answerWait);
                    model.check();
                    for (auto const event : order)
                    {
                        model.apply(event);
                        model.check();
                    }
                    if (::testing::Test::HasFailure())
                    {
                        FAIL() << describe(sentChunks, gatherRunning, answerWait, order);
                    }
                } while (std::next_permutation(order.begin(), order.end()));
            }
        }
    }
}

// The status contract over hand-built results that left nothing outstanding (the model test covers in-flight
// and outstanding ones): completed, with the failure cause as the status string.
TEST(BounceTransferStatus, IsCompletedOnlyForAResolvedQuiescedResult)
{
    auto const success = resolvedStatus({.state = kvc::TransferState::kSUCCESS});
    EXPECT_TRUE(success.isCompleted());
    EXPECT_EQ(success.getLastStatusStr(), "");

    auto const quiescedFailure
        = resolvedStatus({.state = kvc::TransferState::kFAILURE, .reason = b::BounceFailReason::kPlanRejected});
    EXPECT_TRUE(quiescedFailure.isCompleted());
    EXPECT_EQ(quiescedFailure.getLastStatusStr(), b::toString(b::BounceFailReason::kPlanRejected));
}

// A gather's read of the source ends when its SourceRead is destroyed, and only once: a moved-from one
// ends nothing.
TEST(BounceOutstandingAccess, SourceReadEndsWhenDestroyed)
{
    auto access = std::make_shared<b::OutstandingAccess>();
    std::optional<b::OutstandingAccess::SourceRead> read{std::in_place, access};
    EXPECT_FALSE(access->quiesced());

    std::optional<b::OutstandingAccess::SourceRead> moved{std::move(*read)};
    read.reset();
    EXPECT_FALSE(access->quiesced()) << "the moved-from SourceRead ended the read";
    moved.reset();
    EXPECT_TRUE(access->quiesced());
}

// Every trigger waits the full request timeout for the receiver's answers, except kNoProgressTimeout, which
// resolves at once.
TEST(BounceFailedRequest, AnswerWaitFollowsTheTrigger)
{
    constexpr std::chrono::milliseconds kRequestTimeout{30000};
    for (auto const trigger :
        {b::BounceFailReason::kNone, b::BounceFailReason::kPlanRejected, b::BounceFailReason::kNoProgressTimeout,
            b::BounceFailReason::kPeerDropped, b::BounceFailReason::kGatherFailed, b::BounceFailReason::kWriteFailed,
            b::BounceFailReason::kProtocolError, b::BounceFailReason::kShutdown, b::BounceFailReason::kPeerNack})
    {
        auto const expected
            = trigger == b::BounceFailReason::kNoProgressTimeout ? std::chrono::milliseconds{0} : kRequestTimeout;
        EXPECT_EQ(b::answerWaitFor(trigger, kRequestTimeout), expected) << b::toString(trigger);
    }
}
