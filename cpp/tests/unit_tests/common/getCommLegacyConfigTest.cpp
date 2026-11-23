/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nccl.h>

#undef NCCL_VERSION_CODE
#define NCCL_VERSION_CODE NCCL_VERSION(2, 28, 0)

#include "tensorrt_llm/common/opUtils.cpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace
{

class GetCommLegacyConfigTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto const& session = tensorrt_llm::mpi::MpiComm::session();
        int deviceCount = 0;
        ASSERT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
        ASSERT_GT(deviceCount, 0);
        ASSERT_EQ(cudaSetDevice(session.getRank() % deviceCount), cudaSuccess);
        for (int groupRank = 0; groupRank < session.getSize(); ++groupRank)
        {
            mGroup.insert(groupRank);
        }
    }

    std::set<int> mGroup;
};

TEST_F(GetCommLegacyConfigTest, DefaultAndZeroHintShareCommunicator)
{
    auto const defaultComm = tensorrt_llm::getComm(mGroup);
    ASSERT_NE(defaultComm, nullptr);
    ASSERT_NE(*defaultComm, nullptr);
    EXPECT_EQ(defaultComm, tensorrt_llm::getComm(mGroup, 0));
}

TEST_F(GetCommLegacyConfigTest, RejectsUnsupportedHint)
{
    constexpr int kMinCtas = 64;
    EXPECT_THAT([this]() { tensorrt_llm::getComm(mGroup, kMinCtas); },
        ::testing::ThrowsMessage<tensorrt_llm::common::TllmException>(
            ::testing::HasSubstr("minCtas requires NCCL >= 2.29")));
}

} // namespace
