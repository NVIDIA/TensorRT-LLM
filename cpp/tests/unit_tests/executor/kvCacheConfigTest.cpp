/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
