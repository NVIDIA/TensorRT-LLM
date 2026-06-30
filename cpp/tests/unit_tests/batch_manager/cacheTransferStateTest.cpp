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

#include "tensorrt_llm/batch_manager/cacheTransferState.h"

#include <gtest/gtest.h>

#include <exception>
#include <limits>

namespace tensorrt_llm::batch_manager::detail
{
namespace
{

TEST(CacheTransferStateTest, TerminalMarkerRemainsUsefulAfterFutureIsConsumed)
{
    auto const firstPoll = summarizePackedTransferStates({{0, 17, kTransferCompleted}, {0}});
    EXPECT_TRUE(firstPoll.completedRequestIds.empty());
    EXPECT_TRUE(firstPoll.quiescedRequestIds.empty());

    auto const secondPoll = summarizePackedTransferStates({{0, 17, kTransferCompleted}, {0, 17, kTransferCompleted}});
    EXPECT_EQ(secondPoll.completedRequestIds, std::unordered_set<TransferRequestId>{17});
    EXPECT_EQ(secondPoll.quiescedRequestIds, std::unordered_set<TransferRequestId>{17});
}

TEST(CacheTransferStateTest, FailureUnionPrecedesQuiescenceIntersection)
{
    auto const firstPoll = summarizePackedTransferStates({{0, 29, kTransferFailed}, {0}});
    EXPECT_EQ(firstPoll.failedRequestIds, std::unordered_set<TransferRequestId>{29});
    EXPECT_TRUE(firstPoll.quiescedRequestIds.empty());

    auto const secondPoll = summarizePackedTransferStates({{0, 29, kTransferFailed}, {0, 29, kTransferCompleted}});
    EXPECT_EQ(secondPoll.failedRequestIds, std::unordered_set<TransferRequestId>{29});
    EXPECT_EQ(secondPoll.quiescedRequestIds, std::unordered_set<TransferRequestId>{29});
    EXPECT_TRUE(secondPoll.completedRequestIds.empty());
}

TEST(CacheTransferStateTest, TimeoutVoteWinsOverLateSuccess)
{
    auto const outcome
        = summarizePackedTransferStates({{0, 41, kTransferCompleted | kTransferTimedOut}, {0, 41, kTransferCompleted}});
    EXPECT_EQ(outcome.timedOutRequestIds, std::unordered_set<TransferRequestId>{41});
    EXPECT_EQ(outcome.failedRequestIds, std::unordered_set<TransferRequestId>{41});
    EXPECT_EQ(outcome.quiescedRequestIds, std::unordered_set<TransferRequestId>{41});
    EXPECT_TRUE(outcome.completedRequestIds.empty());
}

TEST(CacheTransferStateTest, LargeRequestIdIsPreserved)
{
    auto const requestId = std::numeric_limits<TransferRequestId>::max();
    auto const outcome
        = summarizePackedTransferStates({{0, requestId, kTransferCompleted}, {0, requestId, kTransferCompleted}});
    EXPECT_EQ(outcome.completedRequestIds, std::unordered_set<TransferRequestId>{requestId});
    EXPECT_EQ(outcome.quiescedRequestIds, std::unordered_set<TransferRequestId>{requestId});
}

TEST(CacheTransferStateTest, MalformedPayloadIsRejected)
{
    EXPECT_THROW((void) summarizePackedTransferStates({{}}), std::exception);
    EXPECT_THROW((void) summarizePackedTransferStates({{0, 17}}), std::exception);
}

TEST(CacheTransferStateTest, DuplicateRequestIdOnOneRankIsRejected)
{
    EXPECT_THROW(
        (void) summarizePackedTransferStates({{0, 17, kTransferCompleted, 17, kTransferFailed}}), std::exception);
}

TEST(CacheTransferStateTest, PoisonIsTopologyWideIndependentOfRequestState)
{
    auto const outcome = summarizePackedTransferStates({{0}, {kTransferBufferPoisoned}});
    EXPECT_TRUE(outcome.transferBufferPoisoned);
    EXPECT_TRUE(outcome.failedRequestIds.empty());
    EXPECT_TRUE(outcome.quiescedRequestIds.empty());
}

TEST(CacheTransferStateTest, ReadySignalsDistinguishNoneAllAndPartial)
{
    EXPECT_EQ(summarizeReadySignals(/*allPeersReady=*/true, /*anyPeerReady=*/true), ReadySignalSummary::kReady);
    EXPECT_EQ(summarizeReadySignals(/*allPeersReady=*/false, /*anyPeerReady=*/false), ReadySignalSummary::kNotReady);
    EXPECT_EQ(
        summarizeReadySignals(/*allPeersReady=*/false, /*anyPeerReady=*/true), ReadySignalSummary::kPartiallyReady);
    EXPECT_THROW((void) summarizeReadySignals(/*allPeersReady=*/true, /*anyPeerReady=*/false), std::exception);
}

} // namespace
} // namespace tensorrt_llm::batch_manager::detail
