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

#include "tensorrt_llm/executor/dynamicBatchTuner.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::common;

TEST(DynamicBatchTunerTest, Stats)
{
    // moving average window size is 3
    DynamicBatchConfig dynamicBatchConfig(true, true, 3);
    DynamicBatchTuner dynamicBatchTuner(dynamicBatchConfig);

    // check no division by zero issue
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 0);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 0);

    dynamicBatchTuner.updateStats(1, 2);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 1);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 2);

    dynamicBatchTuner.updateStats(2, 3);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 1.5);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 2.5);

    dynamicBatchTuner.updateStats(3, 4);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 2);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 3);

    // check that the first element is removed from the moving average window
    dynamicBatchTuner.updateStats(4, 5);
    EXPECT_EQ(dynamicBatchTuner.getAverageInputLength(), 3);
    EXPECT_EQ(dynamicBatchTuner.getAverageOutputLength(), 4);
}

TEST(DynamicBatchConfig, RuntimeBatchSize)
{
    // moving average window size is 3
    DynamicBatchConfig dynamicBatchConfig(true, true, 3);
    DynamicBatchTuner dynamicBatchTuner(dynamicBatchConfig);
    // check runtime batch size computation
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(143), 128);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(335), 256);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(671), 512);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(831), 768);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(1279), 1024);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(1663), 1536);
    // fall back
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(2049), 2048);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeBatchSize(1665), 1665);
}

TEST(DynamicBatchConfig, RuntimeMaxNumTokens)
{
    // moving average window size is 1
    DynamicBatchConfig dynamicBatchConfig(true, true, 1);
    DynamicBatchTuner dynamicBatchTuner(dynamicBatchConfig);

    // context heavy
    dynamicBatchTuner.updateStats(100, 2);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(1), 8192);
    // context heavy fall back
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(256), 16384);

    // balanced
    dynamicBatchTuner.updateStats(100, 100);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(1), 4096);
    // balanced fall back
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(4000), 8192);

    // gen heavy
    dynamicBatchTuner.updateStats(2, 256);
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(1), 2048);
    // gen heavy fall back
    EXPECT_EQ(dynamicBatchTuner.getRuntimeMaxNumTokens(4000), 4096);
}
