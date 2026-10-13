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

#include "tensorrt_llm/runtime/ugpu/ugpuResourceConfig.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::ugpu::detail
{

#if CUDA_VERSION >= 13040

TEST(UgpuPublicConfigTest, StrictSplitUsesDiscoveryWithoutCoschedulingOverride)
{
    SmResourceGroupParams const groupParams = makeStrictSmResourceGroupParams();
    for (int ugpuId = 0; ugpuId < kUgpuCount; ++ugpuId)
    {
        auto const& params = groupParams[ugpuId];
        EXPECT_EQ(params.smCount, 0);
        EXPECT_EQ(params.coscheduledSmCount, 0);
        EXPECT_EQ(params.preferredCoscheduledSmCount, 0);
        EXPECT_EQ(params.flags, CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID);
        EXPECT_EQ(params.localityDomainId, ugpuId);
        for (unsigned int const reserved : params.reserved)
        {
            EXPECT_EQ(reserved, 0);
        }
    }
}

TEST(UgpuPublicConfigTest, BalancedSplitUsesHalfDeviceWithBackfill)
{
    constexpr unsigned int kRubinSmCount = 212;
    constexpr unsigned int kExpectedSmCount = 106;
    SmResourceGroupParams const groupParams = makeBalancedSmResourceGroupParams(kRubinSmCount);
    for (int ugpuId = 0; ugpuId < kUgpuCount; ++ugpuId)
    {
        auto const& params = groupParams[ugpuId];
        EXPECT_EQ(params.smCount, kExpectedSmCount);
        EXPECT_EQ(params.coscheduledSmCount, 0);
        EXPECT_EQ(params.preferredCoscheduledSmCount, 0);
        EXPECT_EQ(params.flags, CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID | CU_DEV_SM_RESOURCE_GROUP_BACKFILL);
        EXPECT_EQ(params.localityDomainId, ugpuId);
        for (unsigned int const reserved : params.reserved)
        {
            EXPECT_EQ(reserved, 0);
        }
    }
}

TEST(UgpuPublicConfigTest, BalancedSplitRejectsOddPerGroupSmCount)
{
    EXPECT_TRUE(isBalancedSmCountValid(212));
    EXPECT_FALSE(isBalancedSmCountValid(210));
    EXPECT_FALSE(isBalancedSmCountValid(2));
    EXPECT_FALSE(isBalancedSmCountValid(0));
}

TEST(UgpuPublicConfigTest, StrictSplitAcceptsRubinRemainderAndExactCover)
{
    EXPECT_TRUE(isStrictSplitCountValid(212, 100, 12));
    EXPECT_TRUE(isStrictSplitCountValid(200, 100, 0));
    EXPECT_FALSE(isStrictSplitCountValid(212, 100, 10));
    EXPECT_FALSE(isStrictSplitCountValid(198, 100, 0));
    EXPECT_FALSE(isStrictSplitCountValid(212, 0, 212));
}

#else

TEST(UgpuPublicConfigTest, CUDAHeadersOlderThan134CompileWithoutPublicTypes)
{
    SUCCEED();
}

#endif

} // namespace tensorrt_llm::ugpu::detail
