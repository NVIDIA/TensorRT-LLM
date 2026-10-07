/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

TEST(CacheTransceiverConfigTest, validatesKvTransferPollInterval)
{
    auto makeConfig = [](std::optional<int> pollIntervalMs)
    { return CacheTransceiverConfig(std::nullopt, std::nullopt, std::nullopt, std::nullopt, pollIntervalMs); };

    EXPECT_EQ(makeConfig(std::nullopt).getKvTransferPollIntervalMs(), std::nullopt);
    EXPECT_EQ(makeConfig(1).getKvTransferPollIntervalMs(), 1);
    EXPECT_THROW(makeConfig(0), TllmException);
    EXPECT_THROW(makeConfig(-1), TllmException);
}
