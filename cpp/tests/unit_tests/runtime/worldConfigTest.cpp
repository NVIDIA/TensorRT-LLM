/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

TEST(WorldConfig, DeviceIds)
{
    auto constexpr tensorParallelism = 2;
    auto constexpr pipelineParallelism = 3;
    auto constexpr contextParallelism = 2;
    auto constexpr rank = 1;
    auto constexpr gpusPerNode = 16; // only to test
    EXPECT_NO_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode));

    EXPECT_NO_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
        std::vector{0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11}));

    // Too many GPUs
    EXPECT_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
                     std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11}),
        tc::TllmException);

    // GPUs out of range
    EXPECT_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
                     std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1}),
        tc::TllmException);
    EXPECT_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
                     std::vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16}),
        tc::TllmException);

    // duplicated GPUs
    EXPECT_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
                     std::vector{0, 1, 5, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
        tc::TllmException);

    // non-contiguous GPUs generate just a warning
    EXPECT_NO_THROW(tr::WorldConfig(tensorParallelism, pipelineParallelism, contextParallelism, rank, gpusPerNode,
        std::vector{0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12}));
}

// Test for parallel rank calculations and group membership.
// Layout: pp is outermost, then tp, then cp is innermost (consecutive).
// rank = ppRank * (tp * cp) + tpRank * cp + cpRank
TEST(WorldConfig, ParallelRanks)
{
    auto constexpr tp = 2;
    auto constexpr pp = 2;
    auto constexpr cp = 2;
    auto constexpr gpusPerNode = 16;

    // Test all 8 ranks in a tp=2, pp=2, cp=2 configuration.
    // Rank 0: ppRank=0, tpRank=0, cpRank=0
    {
        tr::WorldConfig config(tp, pp, cp, 0, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 0);
        EXPECT_EQ(config.getTensorParallelRank(), 0);
        EXPECT_EQ(config.getContextParallelRank(), 0);
    }
    // Rank 1: ppRank=0, tpRank=0, cpRank=1
    {
        tr::WorldConfig config(tp, pp, cp, 1, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 0);
        EXPECT_EQ(config.getTensorParallelRank(), 0);
        EXPECT_EQ(config.getContextParallelRank(), 1);
    }
    // Rank 2: ppRank=0, tpRank=1, cpRank=0
    {
        tr::WorldConfig config(tp, pp, cp, 2, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 0);
        EXPECT_EQ(config.getTensorParallelRank(), 1);
        EXPECT_EQ(config.getContextParallelRank(), 0);
    }
    // Rank 3: ppRank=0, tpRank=1, cpRank=1
    {
        tr::WorldConfig config(tp, pp, cp, 3, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 0);
        EXPECT_EQ(config.getTensorParallelRank(), 1);
        EXPECT_EQ(config.getContextParallelRank(), 1);
    }
    // Rank 4: ppRank=1, tpRank=0, cpRank=0
    {
        tr::WorldConfig config(tp, pp, cp, 4, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 1);
        EXPECT_EQ(config.getTensorParallelRank(), 0);
        EXPECT_EQ(config.getContextParallelRank(), 0);
    }
    // Rank 5: ppRank=1, tpRank=0, cpRank=1
    {
        tr::WorldConfig config(tp, pp, cp, 5, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 1);
        EXPECT_EQ(config.getTensorParallelRank(), 0);
        EXPECT_EQ(config.getContextParallelRank(), 1);
    }
    // Rank 6: ppRank=1, tpRank=1, cpRank=0
    {
        tr::WorldConfig config(tp, pp, cp, 6, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 1);
        EXPECT_EQ(config.getTensorParallelRank(), 1);
        EXPECT_EQ(config.getContextParallelRank(), 0);
    }
    // Rank 7: ppRank=1, tpRank=1, cpRank=1
    {
        tr::WorldConfig config(tp, pp, cp, 7, gpusPerNode);
        EXPECT_EQ(config.getPipelineParallelRank(), 1);
        EXPECT_EQ(config.getTensorParallelRank(), 1);
        EXPECT_EQ(config.getContextParallelRank(), 1);
    }
}

TEST(WorldConfig, ParallelGroups)
{
    auto constexpr tp = 2;
    auto constexpr pp = 2;
    auto constexpr cp = 2;
    auto constexpr gpusPerNode = 16;

    // Test group membership for rank 3 (ppRank=0, tpRank=1, cpRank=1).
    // CP group: all ranks with same (ppRank=0, tpRank=1) = [2, 3].
    // TP group: all ranks with same (ppRank=0, cpRank=1) = [1, 3].
    // PP group: all ranks with same (tpRank=1, cpRank=1) = [3, 7].
    {
        tr::WorldConfig config(tp, pp, cp, 3, gpusPerNode);
        auto cpGroup = config.getContextParallelGroup();
        auto tpGroup = config.getTensorParallelGroup();
        auto ppGroup = config.getPipelineParallelGroup();

        EXPECT_EQ(cpGroup, (std::vector<tr::SizeType32>{2, 3}));
        EXPECT_EQ(tpGroup, (std::vector<tr::SizeType32>{1, 3}));
        EXPECT_EQ(ppGroup, (std::vector<tr::SizeType32>{3, 7}));
    }

    // Test group membership for rank 5 (ppRank=1, tpRank=0, cpRank=1).
    // CP group: all ranks with same (ppRank=1, tpRank=0) = [4, 5].
    // TP group: all ranks with same (ppRank=1, cpRank=1) = [5, 7].
    // PP group: all ranks with same (tpRank=0, cpRank=1) = [1, 5].
    {
        tr::WorldConfig config(tp, pp, cp, 5, gpusPerNode);
        auto cpGroup = config.getContextParallelGroup();
        auto tpGroup = config.getTensorParallelGroup();
        auto ppGroup = config.getPipelineParallelGroup();

        EXPECT_EQ(cpGroup, (std::vector<tr::SizeType32>{4, 5}));
        EXPECT_EQ(tpGroup, (std::vector<tr::SizeType32>{5, 7}));
        EXPECT_EQ(ppGroup, (std::vector<tr::SizeType32>{1, 5}));
    }
}

TEST(WorldConfig, ParallelGroupsLargerConfig)
{
    // Test with tp=2, pp=2, cp=4, worldSize=16.
    auto constexpr tp = 2;
    auto constexpr pp = 2;
    auto constexpr cp = 4;
    auto constexpr gpusPerNode = 16;

    // Rank 9: ppRank = 9 / (2*4) = 1, tpRank = (9 % 8) / 4 = 0, cpRank = 9 % 4 = 1.
    // CP group: ranks with same (ppRank=1, tpRank=0) = [8, 9, 10, 11].
    // TP group: ranks with same (ppRank=1, cpRank=1) = [9, 13].
    // PP group: ranks with same (tpRank=0, cpRank=1) = [1, 9].
    {
        tr::WorldConfig config(tp, pp, cp, 9, gpusPerNode);

        EXPECT_EQ(config.getPipelineParallelRank(), 1);
        EXPECT_EQ(config.getTensorParallelRank(), 0);
        EXPECT_EQ(config.getContextParallelRank(), 1);

        auto cpGroup = config.getContextParallelGroup();
        auto tpGroup = config.getTensorParallelGroup();
        auto ppGroup = config.getPipelineParallelGroup();

        EXPECT_EQ(cpGroup, (std::vector<tr::SizeType32>{8, 9, 10, 11}));
        EXPECT_EQ(tpGroup, (std::vector<tr::SizeType32>{9, 13}));
        EXPECT_EQ(ppGroup, (std::vector<tr::SizeType32>{1, 9}));
    }
}
