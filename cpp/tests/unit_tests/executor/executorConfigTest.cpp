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

TEST(ExecutorConfigTest, ctorValidInputs)
{
    SchedulerConfig schedulerConfig;
    KvCacheConfig kvCacheConfig;
    {
        auto executorConfig = ExecutorConfig(1);
    }
    {
        auto executorConfig = ExecutorConfig(1, schedulerConfig, kvCacheConfig, true, true, 0);
    }
    {
        auto executorConfig = ExecutorConfig(1, schedulerConfig, kvCacheConfig, true, true, 1000);
    }
}

void testInvalid(SizeType32 maxBeamWidth = 1, SchedulerConfig schedulerConfig = SchedulerConfig(),
    KvCacheConfig kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = false, bool normalizeLogProbs = true,
    SizeType32 iterStatsMaxIterations = 1000, BatchingType batchingType = BatchingType::kINFLIGHT,
    std::optional<ParallelConfig> parallelConfig = std::nullopt)
{
    try
    {
        auto executorConfig = ExecutorConfig(maxBeamWidth, schedulerConfig, kvCacheConfig, enableChunkedContext,
            normalizeLogProbs, iterStatsMaxIterations);
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

TEST(ExecutorConfigTest, ctorInvalidInputs)
{
    testInvalid(0);
    testInvalid(-1);

    SchedulerConfig schedulerConfig;
    KvCacheConfig kvCacheConfig;

    // Empty device ids
    try
    {
        ParallelConfig parallelConfigInvalid;
        parallelConfigInvalid.setDeviceIds({});
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

    // iter stats negative
    ParallelConfig parallelConfigValid;
    testInvalid(1, schedulerConfig, kvCacheConfig, true, true, -1, BatchingType::kINFLIGHT, parallelConfigValid);
}

TEST(ExecutorConfigTest, extendedRuntimePerfKnobConfigTest)
{
    ExtendedRuntimePerfKnobConfig extendedRuntimePerfKnobConfig;
    {
        auto executorConfig = ExecutorConfig(1);
        executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);
    }
    {
        auto executorConfig = ExecutorConfig(1);
        extendedRuntimePerfKnobConfig.setMultiBlockMode(true);
        extendedRuntimePerfKnobConfig.setEnableContextFMHAFP32Acc(true);
        executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);
    }
    {
        auto executorConfig = ExecutorConfig(1);
        extendedRuntimePerfKnobConfig.setMultiBlockMode(true);
        extendedRuntimePerfKnobConfig.setMultiBlockMode(false);
        extendedRuntimePerfKnobConfig.setEnableContextFMHAFP32Acc(true);
        extendedRuntimePerfKnobConfig.setEnableContextFMHAFP32Acc(false);
        executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);
    }
    {
        ExtendedRuntimePerfKnobConfig newExtendedRuntimePerfKnobConfig(false, false);
        auto executorConfig = ExecutorConfig(1);
        newExtendedRuntimePerfKnobConfig.setMultiBlockMode(true);
        newExtendedRuntimePerfKnobConfig.setEnableContextFMHAFP32Acc(true);
        executorConfig.setExtendedRuntimePerfKnobConfig(newExtendedRuntimePerfKnobConfig);
    }
}
