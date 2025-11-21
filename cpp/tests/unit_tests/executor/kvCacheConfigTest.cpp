/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST(KvCacheConfigTest, validInputs)
{
    {
        {
            auto kvCacheConfig = KvCacheConfig();
        }
        {
            auto kvCacheConfig = KvCacheConfig(true);
        }
        {
            auto kvCacheConfig = KvCacheConfig(true, 1);
        }
        {
            auto kvCacheConfig = KvCacheConfig(true, 100, std::vector(1, 1000));
        }
        {
            auto kvCacheConfig = KvCacheConfig(true, 100, std::vector(1, 1000), 1000);
        }
        {
            auto kvCacheConfig = KvCacheConfig(true, 100, std::vector(1, 1000), 1000, 0.1);
        }
    }
}

void testInvalid(bool enableBlockReuse = false, std::optional<SizeType32> maxTokens = std::nullopt,
    std::optional<std::vector<SizeType32>> maxAttentionWindowVec = std::nullopt,
    std::optional<SizeType32> sinkTokenLength = std::nullopt,
    std::optional<FloatType> freeGpuMemoryFraction = std::nullopt)
{
    try
    {
        auto kvCacheConfig
            = KvCacheConfig(enableBlockReuse, maxTokens, maxAttentionWindowVec, sinkTokenLength, freeGpuMemoryFraction);
        FAIL() << "Expected TllmException";
    }
    catch (TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr("Assertion failed"));
    }
    catch (std::exception const& e)
    {
        FAIL() << "Expected TllmException";
    }
}

TEST(KvCacheConfigTest, invalidInputs)
{
    // Negative maxTokens
    testInvalid(true, 0);
    testInvalid(true, -1);

    // Negative maxAttentionWindow
    testInvalid(true, std::nullopt, std::vector(1, 0));
    testInvalid(true, std::nullopt, std::vector(1, -1));

    // Negative sink token
    testInvalid(true, std::nullopt, std::nullopt, 0);
    testInvalid(true, std::nullopt, std::nullopt, -1);

    // free gpu memory fraction
    testInvalid(true, std::nullopt, std::nullopt, std::nullopt, -0.1);
    testInvalid(true, std::nullopt, std::nullopt, std::nullopt, 1.1);
}
