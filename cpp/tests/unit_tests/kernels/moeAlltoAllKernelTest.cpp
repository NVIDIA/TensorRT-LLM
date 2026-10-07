/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"

#include <gtest/gtest.h>

#include <string>

namespace
{

using tensorrt_llm::common::TllmException;
using namespace tensorrt_llm::kernels::moe_comm;

template <typename LaunchParams, typename LaunchFunction>
void expectEmptyExecutionControlRejected(LaunchParams const& params, LaunchFunction launch)
{
    try
    {
        launch(params, MoeA2AExecutionControl{});
        FAIL() << "an empty execution control must fail closed";
    }
    catch (TllmException const& error)
    {
        EXPECT_NE(std::string{error.what()}.find("live_epoch"), std::string::npos);
    }
}

template <typename LaunchParams, typename LaunchFunction>
void expectRankMaskWithoutControlRejected(LaunchParams const& params, LaunchFunction launch)
{
    try
    {
        launch(params);
        FAIL() << "rank-mask launch without execution control must fail before mutation";
    }
    catch (TllmException const& error)
    {
        EXPECT_NE(std::string{error.what()}.find("requires a registered MoeA2AExecutionControl"), std::string::npos);
    }
}

TEST(MoeAlltoAllKernelTest, RejectsEmptyExecutionControl)
{
    MoeA2ADispatchParams dispatchParams{};
    dispatchParams.top_k = 1;
    dispatchParams.ep_size = 1;
    dispatchParams.ep_rank = 0;
    dispatchParams.num_experts = 1;
    dispatchParams.num_payloads = 1;
    dispatchParams.enable_rank_mask = true;
    expectEmptyExecutionControlRejected(dispatchParams,
        static_cast<void (*)(MoeA2ADispatchParams const&, MoeA2AExecutionControl const&)>(moe_a2a_dispatch_launch));
    expectEmptyExecutionControlRejected(dispatchParams,
        static_cast<void (*)(MoeA2ADispatchParams const&, MoeA2AExecutionControl const&)>(
            moe_a2a_prepare_dispatch_launch));
    expectRankMaskWithoutControlRejected(
        dispatchParams, static_cast<void (*)(MoeA2ADispatchParams const&)>(moe_a2a_dispatch_launch));
    expectRankMaskWithoutControlRejected(
        dispatchParams, static_cast<void (*)(MoeA2ADispatchParams const&)>(moe_a2a_prepare_dispatch_launch));

    MoeA2ACombineParams combineParams{};
    combineParams.top_k = 1;
    combineParams.ep_size = 1;
    combineParams.ep_rank = 0;
    combineParams.elements_per_token = 1;
    combineParams.enable_rank_mask = true;
    expectEmptyExecutionControlRejected(combineParams,
        static_cast<void (*)(MoeA2ACombineParams const&, MoeA2AExecutionControl const&)>(moe_a2a_combine_launch));
    expectEmptyExecutionControlRejected(combineParams,
        static_cast<void (*)(MoeA2ACombineParams const&, MoeA2AExecutionControl const&)>(
            moe_a2a_prepare_combine_launch));
    expectRankMaskWithoutControlRejected(
        combineParams, static_cast<void (*)(MoeA2ACombineParams const&)>(moe_a2a_combine_launch));
    expectRankMaskWithoutControlRejected(
        combineParams, static_cast<void (*)(MoeA2ACombineParams const&)>(moe_a2a_prepare_combine_launch));
}

TEST(MoeAlltoAllKernelTest, StatusEncodesEverySupportedPeerWithoutCorruptingEpoch)
{
    for (auto const epoch : {uint64_t{0}, uint64_t{1}, kMoeA2AExecutionEpochMask})
    {
        for (int peer = -1; peer < kMaxRanks; ++peer)
        {
            auto const status
                = packMoeA2AExecutionStatus(epoch, MoeA2AExecutionPhase::kCombine, MoeA2AAbortReason::kTimeout, peer);
            EXPECT_EQ(status >> 63, 0);
            EXPECT_EQ((status >> 25) & kMoeA2AExecutionEpochMask, epoch);
            EXPECT_EQ((status >> 16) & 0x1ff, static_cast<uint64_t>(peer + 1));
            EXPECT_EQ((status >> 8) & 0xff, static_cast<uint64_t>(MoeA2AExecutionPhase::kCombine));
            EXPECT_EQ(status & 0xff, static_cast<uint64_t>(MoeA2AAbortReason::kTimeout));
        }
    }
}

TEST(MoeAlltoAllKernelTest, RejectsCftExecutionAbortBeforeAnyLaunch)
{
    auto const expectCftRejected = [](auto launch)
    {
        try
        {
            launch();
            FAIL() << "unqualified CFT execution abort must fail before a kernel launch";
        }
        catch (TllmException const& error)
        {
            EXPECT_NE(std::string{error.what()}.find("supports only non-CFT transport"), std::string::npos);
        }
    };
    MoeA2ADispatchParams dispatch{};
    dispatch.enable_rank_mask = true;
    dispatch.use_cft_counted_writes = true;
    expectCftRejected([&] { moe_a2a_prepare_dispatch_launch(dispatch, MoeA2AExecutionControl{}); });
    expectCftRejected([&] { moe_a2a_dispatch_launch(dispatch, MoeA2AExecutionControl{}); });

    MoeA2ACombineParams combine{};
    combine.enable_rank_mask = true;
    combine.use_cft_for_combine = true;
    expectCftRejected([&] { moe_a2a_prepare_combine_launch(combine, MoeA2AExecutionControl{}); });
    expectCftRejected([&] { moe_a2a_combine_launch(combine, MoeA2AExecutionControl{}); });
    expectCftRejected([&] { moe_a2a_cft_combine_push_launch(combine); });
}

} // namespace
