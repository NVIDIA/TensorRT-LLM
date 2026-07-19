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

#include <gtest/gtest.h>

namespace tensorrt_llm::batch_manager
{
namespace
{

TEST(ContextTransferVoteReducerTest, WaitsForEveryParticipant)
{
    ContextTransferVoteReducer reducer(4);
    reducer.recordVote(0, 17, ContextTransferVote::kCompleted);
    reducer.recordVote(1, 17, ContextTransferVote::kCompleted);
    reducer.recordVote(2, 17, ContextTransferVote::kCompleted);
    EXPECT_TRUE(reducer.takeCompleted().completedRequestIds.empty());

    reducer.recordVote(3, 17, ContextTransferVote::kCompleted);
    auto const result = reducer.takeCompleted();
    EXPECT_EQ(result.completedRequestIds, std::unordered_set<std::uint64_t>{17});
    EXPECT_TRUE(result.failedRequestIds.empty());
}

TEST(ContextTransferVoteReducerTest, FailureWinsAfterEveryParticipantVotes)
{
    ContextTransferVoteReducer reducer(4);
    reducer.recordVote(0, 23, ContextTransferVote::kCompleted);
    reducer.recordVote(1, 23, ContextTransferVote::kFailed);
    reducer.recordVote(2, 23, ContextTransferVote::kCompleted);
    reducer.recordVote(3, 23, ContextTransferVote::kCompleted);

    auto const result = reducer.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_EQ(result.failedRequestIds, std::unordered_set<std::uint64_t>{23});
}

TEST(ContextTransferVoteReducerTest, RejectsChangedTerminalVote)
{
    ContextTransferVoteReducer reducer(2);
    reducer.recordVote(0, 29, ContextTransferVote::kCompleted);
    reducer.recordVote(0, 29, ContextTransferVote::kCompleted);
    EXPECT_ANY_THROW(reducer.recordVote(0, 29, ContextTransferVote::kFailed));
}

TEST(ContextTransferVoteReducerTest, AccumulatesInterleavedRequestsIndependently)
{
    ContextTransferVoteReducer reducer(2);
    reducer.recordVote(0, 31, ContextTransferVote::kCompleted);
    reducer.recordVote(1, 37, ContextTransferVote::kFailed);
    reducer.recordVote(1, 31, ContextTransferVote::kCompleted);

    auto result = reducer.takeCompleted();
    EXPECT_EQ(result.completedRequestIds, std::unordered_set<std::uint64_t>{31});
    EXPECT_TRUE(result.failedRequestIds.empty());

    reducer.recordVote(0, 37, ContextTransferVote::kCompleted);
    result = reducer.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_EQ(result.failedRequestIds, std::unordered_set<std::uint64_t>{37});
}

TEST(ContextTransferVoteReducerTest, RejectsInvalidInput)
{
    EXPECT_ANY_THROW(ContextTransferVoteReducer(0));
    ContextTransferVoteReducer reducer(2);
    EXPECT_ANY_THROW(reducer.recordVote(-1, 41, ContextTransferVote::kCompleted));
    EXPECT_ANY_THROW(reducer.recordVote(2, 41, ContextTransferVote::kCompleted));
    EXPECT_ANY_THROW(reducer.recordVote(0, 41, static_cast<ContextTransferVote>(99)));
}

} // namespace
} // namespace tensorrt_llm::batch_manager
