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
