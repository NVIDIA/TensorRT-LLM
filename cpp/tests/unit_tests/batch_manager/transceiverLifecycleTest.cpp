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

#include "tensorrt_llm/batch_manager/transceiverLifecycle.h"
#include "tensorrt_llm/batch_manager/transceiverLifecycleUtils.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::batch_manager
{
namespace
{

TEST(TransceiverLifecycleTest, ReuseRequiresPositiveQuiescenceEvidence)
{
    EXPECT_TRUE(isReusable(PhysicalDisposition::kNOT_EXPOSED));
    EXPECT_TRUE(isReusable(PhysicalDisposition::kQUIESCED_SUCCESS));
    EXPECT_TRUE(isReusable(PhysicalDisposition::kQUIESCED_FAILURE));
    EXPECT_FALSE(isReusable(PhysicalDisposition::kACTIVE));
    EXPECT_FALSE(isReusable(PhysicalDisposition::kQUIESCING));
    EXPECT_FALSE(isReusable(PhysicalDisposition::kIN_DOUBT));
}

TEST(TransceiverLifecycleTest, LogicalAcceptanceDoesNotImplyPhysicalReuse)
{
    CancelResult const result{
        LogicalDisposition::kACCEPTED, PhysicalDisposition::kQUIESCING, true, "cancellation requested"};

    EXPECT_FALSE(result.safeToReuse());
}

TEST(TransceiverLifecycleTest, LegacyModeRequiresExplicitQualification)
{
    TransceiverCapabilities const capabilities;

    EXPECT_FALSE(capabilities.qualifiedLegacyMode);
}

TEST(TransceiverLifecycleTest, InDoubtShutdownDoesNotReleaseManagers)
{
    ShutdownResult const defaultResult;
    ShutdownResult const result{PhysicalDisposition::kIN_DOUBT, 2, true, "endpoint replacement required"};

    EXPECT_FALSE(defaultResult.inDoubtContextCount.has_value());
    EXPECT_FALSE(defaultResult.safeToReleaseManagers());
    EXPECT_FALSE(result.safeToReleaseManagers());
}

struct LegacyCancelCase
{
    lifecycle_detail::LegacyCancelObservation observation;
    LogicalDisposition logical;
    PhysicalDisposition physical;
    bool retryable;
};

class LegacyCancelMappingTest : public ::testing::TestWithParam<LegacyCancelCase>
{
};

struct LegacyCancelClassificationCase
{
    bool cancellationRequested;
    bool cancelledBeforeLocalWork;
    bool requestWasFound;
    lifecycle_detail::LegacyCancelObservation expected;
};

class LegacyCancelClassificationTest : public ::testing::TestWithParam<LegacyCancelClassificationCase>
{
};

TEST_P(LegacyCancelClassificationTest, ClassifiesRuntimeState)
{
    auto const& testCase = GetParam();

    EXPECT_EQ(lifecycle_detail::classifyLegacyCancelObservation(
                  testCase.cancellationRequested, testCase.cancelledBeforeLocalWork, testCase.requestWasFound),
        testCase.expected);
}

INSTANTIATE_TEST_SUITE_P(RuntimeStates, LegacyCancelClassificationTest,
    ::testing::Values(LegacyCancelClassificationCase{true, true, true,
                          lifecycle_detail::LegacyCancelObservation::kCANCELLED_WITHOUT_REUSE_PROOF},
        LegacyCancelClassificationCase{
            true, false, true, lifecycle_detail::LegacyCancelObservation::kCANCELLATION_REQUESTED},
        LegacyCancelClassificationCase{
            false, false, true, lifecycle_detail::LegacyCancelObservation::kACTIVE_UNSUPPORTED},
        LegacyCancelClassificationCase{
            false, false, false, lifecycle_detail::LegacyCancelObservation::kOWNER_NOT_FOUND}));

TEST_P(LegacyCancelMappingTest, FailsClosedWithoutPositiveReuseProof)
{
    auto const& testCase = GetParam();
    auto const result = lifecycle_detail::makeLegacyCancelResult(testCase.observation, "test");

    EXPECT_EQ(result.logical, testCase.logical);
    EXPECT_EQ(result.physical, testCase.physical);
    EXPECT_EQ(result.retryable, testCase.retryable);
    EXPECT_FALSE(result.safeToReuse());
    EXPECT_EQ(lifecycle_detail::legacyCancellationAccepted(result), testCase.logical == LogicalDisposition::kACCEPTED);
}

INSTANTIATE_TEST_SUITE_P(Observations, LegacyCancelMappingTest,
    ::testing::Values(LegacyCancelCase{lifecycle_detail::LegacyCancelObservation::kCANCELLED_WITHOUT_REUSE_PROOF,
                          LogicalDisposition::kACCEPTED, PhysicalDisposition::kIN_DOUBT, false},
        LegacyCancelCase{lifecycle_detail::LegacyCancelObservation::kCANCELLATION_REQUESTED,
            LogicalDisposition::kACCEPTED, PhysicalDisposition::kQUIESCING, true},
        LegacyCancelCase{lifecycle_detail::LegacyCancelObservation::kACTIVE_UNSUPPORTED, LogicalDisposition::kREJECTED,
            PhysicalDisposition::kACTIVE, true},
        LegacyCancelCase{lifecycle_detail::LegacyCancelObservation::kOWNER_NOT_FOUND, LogicalDisposition::kNOT_FOUND,
            PhysicalDisposition::kIN_DOUBT, true}));

TEST(TransceiverLifecycleTest, PoisonedStorageDowngradesAnyPhysicalResult)
{
    CancelResult const initial{LogicalDisposition::kACCEPTED, PhysicalDisposition::kQUIESCED_SUCCESS, true, "complete"};

    auto const result = lifecycle_detail::failClosedForPoisonedStorage(initial);

    EXPECT_EQ(result.physical, PhysicalDisposition::kIN_DOUBT);
    EXPECT_FALSE(result.retryable);
    EXPECT_FALSE(result.safeToReuse());
}

TEST(TransceiverLifecycleTest, LegacyShutdownAlwaysFailsClosed)
{
    auto const clean = lifecycle_detail::makeLegacyShutdownResult(false);
    auto const poisoned = lifecycle_detail::makeLegacyShutdownResult(true);

    EXPECT_EQ(clean.physical, PhysicalDisposition::kIN_DOUBT);
    EXPECT_FALSE(clean.inDoubtContextCount.has_value());
    EXPECT_FALSE(clean.fatal);
    EXPECT_FALSE(clean.safeToReleaseManagers());
    EXPECT_EQ(poisoned.physical, PhysicalDisposition::kIN_DOUBT);
    EXPECT_FALSE(poisoned.inDoubtContextCount.has_value());
    EXPECT_TRUE(poisoned.fatal);
    EXPECT_FALSE(poisoned.safeToReleaseManagers());
}

} // namespace
} // namespace tensorrt_llm::batch_manager
