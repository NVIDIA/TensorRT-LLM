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

#include "tensorrt_llm/runtime/locality_domain/localityDomainResourceConfig.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::locality_domain::detail
{

#if CUDA_VERSION >= 13040

TEST(LocalityDomainPublicConfigTest, StrictSplitUsesDiscoveryWithoutCoschedulingOverride)
{
    SmResourceGroupParams const groupParams = makeStrictSmResourceGroupParams();
    for (int localityDomainId = 0; localityDomainId < kLocalityDomainCount; ++localityDomainId)
    {
        auto const& params = groupParams[localityDomainId];
        EXPECT_EQ(params.smCount, 0);
        EXPECT_EQ(params.coscheduledSmCount, 0);
        EXPECT_EQ(params.preferredCoscheduledSmCount, 0);
        EXPECT_EQ(params.flags, CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID);
        EXPECT_EQ(params.localityDomainId, localityDomainId);
        for (unsigned int const reserved : params.reserved)
        {
            EXPECT_EQ(reserved, 0);
        }
    }
}

TEST(LocalityDomainPublicConfigTest, BalancedSplitUsesHalfDeviceWithBackfill)
{
    // Any total divisible by kSmCountAlignment * kLocalityDomainCount works here; the
    // helper is pure arithmetic, so the value is deliberately not tied to a specific device.
    constexpr unsigned int kTestSmCount = 128;
    constexpr unsigned int kExpectedSmCount = 64; // 128 / 2, computed independently of the helper
    SmResourceGroupParams const groupParams = makeBalancedSmResourceGroupParams(kTestSmCount);
    for (int localityDomainId = 0; localityDomainId < kLocalityDomainCount; ++localityDomainId)
    {
        auto const& params = groupParams[localityDomainId];
        EXPECT_EQ(params.smCount, kExpectedSmCount);
        EXPECT_EQ(params.coscheduledSmCount, 0);
        EXPECT_EQ(params.preferredCoscheduledSmCount, 0);
        EXPECT_EQ(params.flags, CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID | CU_DEV_SM_RESOURCE_GROUP_BACKFILL);
        EXPECT_EQ(params.localityDomainId, localityDomainId);
        for (unsigned int const reserved : params.reserved)
        {
            EXPECT_EQ(reserved, 0);
        }
    }
}

TEST(LocalityDomainPublicConfigTest, BalancedSplitRejectsOddPerGroupSmCount)
{
    EXPECT_TRUE(isBalancedSmCountValid(128));  // 64 per group, even
    EXPECT_FALSE(isBalancedSmCountValid(126)); // 63 per group, odd
    EXPECT_FALSE(isBalancedSmCountValid(2));   // below the per-group alignment minimum
    EXPECT_FALSE(isBalancedSmCountValid(0));
}

TEST(LocalityDomainPublicConfigTest, StrictSplitAcceptsRemainderAndExactCover)
{
    EXPECT_TRUE(isStrictSplitCountValid(128, 60, 8));   // split with a remainder
    EXPECT_TRUE(isStrictSplitCountValid(120, 60, 0));   // exact cover, no remainder
    EXPECT_FALSE(isStrictSplitCountValid(128, 60, 6));  // remainder does not account for every SM
    EXPECT_FALSE(isStrictSplitCountValid(118, 60, 0));  // per-group count exceeds half the device
    EXPECT_FALSE(isStrictSplitCountValid(128, 0, 128)); // empty locality domain
}

#else

TEST(LocalityDomainPublicConfigTest, CUDAHeadersOlderThan134CompileWithoutPublicTypes)
{
    SUCCEED();
}

#endif

} // namespace tensorrt_llm::locality_domain::detail
