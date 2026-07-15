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

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <unordered_set>

namespace tensorrt_llm::batch_manager
{
namespace
{

TEST(TransferStatusConsensusTest, CompletesOnlyAfterEveryParticipantVotes)
{
    TransferStatusConsensus consensus(3);

    consensus.recordVote(0, 11, TransferStatusVote::kCompleted);
    consensus.recordVote(2, 11, TransferStatusVote::kCompleted);
    auto result = consensus.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_TRUE(result.failedRequestIds.empty());

    consensus.recordVote(1, 11, TransferStatusVote::kCompleted);
    result = consensus.takeCompleted();
    EXPECT_EQ(result.completedRequestIds, std::unordered_set<std::uint64_t>{11});
    EXPECT_TRUE(result.failedRequestIds.empty());
}

TEST(TransferStatusConsensusTest, FailureWinsAfterEveryParticipantIsTerminal)
{
    TransferStatusConsensus consensus(3);

    consensus.recordVote(2, 7, TransferStatusVote::kCompleted);
    consensus.recordVote(0, 7, TransferStatusVote::kCompleted);
    consensus.recordVote(1, 7, TransferStatusVote::kFailed);
    auto const result = consensus.takeCompleted();

    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_EQ(result.failedRequestIds, std::unordered_set<std::uint64_t>{7});
}

TEST(TransferStatusConsensusTest, AccumulatesInterleavedRequestsIndependently)
{
    TransferStatusConsensus consensus(2);

    consensus.recordVote(0, 3, TransferStatusVote::kCompleted);
    consensus.recordVote(1, 5, TransferStatusVote::kFailed);
    consensus.recordVote(1, 3, TransferStatusVote::kCompleted);
    auto result = consensus.takeCompleted();
    EXPECT_EQ(result.completedRequestIds, std::unordered_set<std::uint64_t>{3});
    EXPECT_TRUE(result.failedRequestIds.empty());

    consensus.recordVote(0, 5, TransferStatusVote::kCompleted);
    result = consensus.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_EQ(result.failedRequestIds, std::unordered_set<std::uint64_t>{5});
}

TEST(TransferStatusConsensusTest, AcceptsIdenticalDuplicateAndRejectsConflictingDuplicate)
{
    TransferStatusConsensus consensus(2);

    consensus.recordVote(0, 17, TransferStatusVote::kCompleted);
    EXPECT_NO_THROW(consensus.recordVote(0, 17, TransferStatusVote::kCompleted));
    EXPECT_ANY_THROW(consensus.recordVote(0, 17, TransferStatusVote::kFailed));
    EXPECT_TRUE(consensus.takeCompleted().completedRequestIds.empty());
}

TEST(TransferStatusConsensusTest, RejectsInvalidParticipant)
{
    TransferStatusConsensus consensus(2);

    EXPECT_ANY_THROW(consensus.recordVote(-1, 9, TransferStatusVote::kCompleted));
    EXPECT_ANY_THROW(consensus.recordVote(2, 9, TransferStatusVote::kCompleted));
}

TEST(TransferStatusConsensusTest, RejectsInvalidVote)
{
    TransferStatusConsensus consensus(1);

    EXPECT_ANY_THROW(consensus.recordVote(0, 9, static_cast<TransferStatusVote>(99)));
}

TEST(TransferStatusConsensusTest, RejectsEmptyParticipantSet)
{
    EXPECT_ANY_THROW(TransferStatusConsensus(0));
}

TEST(TransferStatusConsensusTest, DoesNotCommitFailureBeforeEveryParticipantIsTerminal)
{
    TransferStatusConsensus consensus(4);

    consensus.recordVote(0, 23, TransferStatusVote::kFailed);
    consensus.recordVote(1, 23, TransferStatusVote::kCompleted);
    consensus.recordVote(2, 23, TransferStatusVote::kCompleted);
    auto result = consensus.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_TRUE(result.failedRequestIds.empty());

    consensus.recordVote(3, 23, TransferStatusVote::kCompleted);
    result = consensus.takeCompleted();
    EXPECT_TRUE(result.completedRequestIds.empty());
    EXPECT_EQ(result.failedRequestIds, std::unordered_set<std::uint64_t>{23});
}

TEST(WorkerPublishedConsensusConfigTest, RequiresEveryQualifiedCondition)
{
    WorkerPublishedConsensusConfig supported;
    supported.enabledForCppBinding = true;
    supported.mpiControlPlane = true;
    supported.nixlUcxBackend = true;
    supported.pipelineParallelism = 4;
    supported.transferOverlap = true;
    supported.mpiThreadMultiple = true;
    ASSERT_TRUE(supportsWorkerPublishedConsensus(supported));

    auto expectUnsupported = [&](std::function<void(WorkerPublishedConsensusConfig&)> const& mutate)
    {
        auto config = supported;
        mutate(config);
        EXPECT_FALSE(supportsWorkerPublishedConsensus(config));
    };
    expectUnsupported([](auto& config) { config.enabledForCppBinding = false; });
    expectUnsupported([](auto& config) { config.mpiControlPlane = false; });
    expectUnsupported([](auto& config) { config.nixlUcxBackend = false; });
    expectUnsupported([](auto& config) { config.tensorParallelism = 2; });
    expectUnsupported([](auto& config) { config.contextParallelism = 2; });
    expectUnsupported([](auto& config) { config.pipelineParallelism = 1; });
    expectUnsupported([](auto& config) { config.attentionDp = true; });
    expectUnsupported([](auto& config) { config.inflightCancellation = true; });
    expectUnsupported([](auto& config) { config.transferOverlap = false; });
    expectUnsupported([](auto& config) { config.layerwiseTransfer = true; });
    expectUnsupported([](auto& config) { config.mpiThreadMultiple = false; });
}

TEST(WorkerPublishedConsensusConfigTest, SelectsOnlyOneNonzeroVersionOnEveryRank)
{
    EXPECT_FALSE(selectWorkerPublishedConsensus({}));
    EXPECT_FALSE(selectWorkerPublishedConsensus({0, 0, 0, 0}));
    EXPECT_FALSE(selectWorkerPublishedConsensus({1, 0, 1, 1}));
    EXPECT_FALSE(selectWorkerPublishedConsensus({1, 2, 1, 1}));
    EXPECT_TRUE(selectWorkerPublishedConsensus({1, 1, 1, 1}));
}

} // namespace
} // namespace tensorrt_llm::batch_manager
