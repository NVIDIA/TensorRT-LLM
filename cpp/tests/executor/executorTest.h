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

#pragma once

#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tests/utils/common.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

namespace tensorrt_llm::testing
{

class GptExecutorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    using SizeType32 = tensorrt_llm::testing::SizeType32;

protected:
    void SetUp() override
    {
        mDeviceCount = tensorrt_llm::common::getDeviceCount();
        if (mDeviceCount == 0)
        {
            GTEST_SKIP() << "No GPUs found";
        }

        mLogger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
        initTrtLlmPlugins(mLogger.get());
    }

    void TearDown() override {}

    int mDeviceCount{};
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
    SizeType32 mMaxWaitMs = 300000;
    SizeType32 mTrigWarnMs = 10000;
};

} // namespace tensorrt_llm::testing
