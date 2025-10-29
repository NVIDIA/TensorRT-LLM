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
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace tensorrt_llm::executor;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

void testInvalid(
    IdType taskId, std::optional<Tensor> weights, std::optional<Tensor> config, std::string const& expectedErrStr)
{
    try
    {
        auto loraConfig = LoraConfig(taskId, weights, config);
        FAIL() << "Expected TllmException";
    }
    catch (tc::TllmException& e)
    {
        EXPECT_THAT(e.what(), testing::HasSubstr(expectedErrStr));
    }
    catch (std::exception const& e)
    {
        FAIL() << "Expected TllmException";
    }
}

TEST(LoraConfigTest, invalidInputs)
{
    SizeType32 weightsDim0 = 32;
    SizeType32 weightsDim1 = 64;
    SizeType32 configDim1 = 3;

    auto weights = Tensor::cpu(DataType::kFP16, {weightsDim0, weightsDim1});
    auto config = Tensor::cpu(DataType::kINT32, {weightsDim0, configDim1});
    auto stream = std::make_shared<tr::CudaStream>();

    // This should work
    auto loraConfig = LoraConfig(1, weights, config);
    // Having config only without weights is allowed
    loraConfig = LoraConfig(1, std::nullopt, config);

    {
        // Only weights specified without config - not allowed
        testInvalid(1, weights, std::nullopt, "lora weights must also have lora config");
    }

    {
        auto invalidWeights = Tensor::cpu(DataType::kFP16, {1});
        // Wrong shape
        testInvalid(1, invalidWeights, config, "Expected weights tensor to have 2 dimensions");
    }
    {
        auto invalidWeights = Tensor::gpu(DataType::kFP16, stream, {weightsDim0, weightsDim1});
        // Wrong memory type
        testInvalid(1, invalidWeights, config, "to be in CPU memory");
    }
    {
        auto invalidConfig = Tensor::cpu(DataType::kFP16, {weightsDim0, 3});
        // Wrong type
        testInvalid(1, weights, invalidConfig, "to have type kINT32");
    }
    {
        auto invalidConfig = Tensor::gpu(DataType::kINT32, stream, {weightsDim0, configDim1});
        // Wrong memory type
        testInvalid(1, weights, invalidConfig, "to be in CPU memory");
    }

    {
        // Shapes not matching
        auto invalidConfig = Tensor::cpu(DataType::kINT32, {16, configDim1});
        // Wrong memory type
        testInvalid(1, weights, invalidConfig, "dim 0 of lora weights and lora config to have the same size");
    }
}

TEST(LoraConfigTest, serializeDeserialize)
{
    IdType taskId = 1000;

    SizeType32 weightsDim0 = 32;
    SizeType32 weightsDim1 = 64;
    SizeType32 configDim1 = 3;

    auto weights = Tensor::cpu(DataType::kFP32, {weightsDim0, weightsDim1});
    float* weightsData = reinterpret_cast<float*>(weights.getData());
    for (int i = 0; i < weightsDim0; ++i)
    {
        for (int j = 0; j < weightsDim1; ++j)
        {
            weightsData[i * weightsDim1 + j] = (i * weightsDim1 + j) * 1.0f;
        }
    }

    auto config = Tensor::cpu(DataType::kINT32, {weightsDim0, configDim1});
    int32_t* configData = reinterpret_cast<int32_t*>(config.getData());
    for (int i = 0; i < weightsDim0; ++i)
    {
        for (int j = 0; j < configDim1; ++j)
        {
            weightsData[i * configDim1 + j] = 3 * (i * configDim1 + j);
        }
    }

    auto loraConfig = LoraConfig(taskId, weights, config);
    auto serializedSize = Serialization::serializedSize(loraConfig);

    std::ostringstream os;
    Serialization::serialize(loraConfig, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newLoraConfig = Serialization::deserializeLoraConfig(is);

    EXPECT_EQ(newLoraConfig.getTaskId(), loraConfig.getTaskId());
    EXPECT_EQ(newLoraConfig.getWeights().value().getShape().size(), loraConfig.getWeights().value().getShape().size());
    EXPECT_EQ(newLoraConfig.getWeights().value().getShape()[0], loraConfig.getWeights().value().getShape()[0]);
    EXPECT_EQ(newLoraConfig.getWeights().value().getShape()[1], loraConfig.getWeights().value().getShape()[1]);
    float* newWeightsData = reinterpret_cast<float*>(newLoraConfig.getWeights().value().getData());
    for (int i = 0; i < weightsDim0; ++i)
    {
        for (int j = 0; j < weightsDim1; ++j)
        {
            EXPECT_FLOAT_EQ(weightsData[i * weightsDim1 + j], newWeightsData[i * weightsDim1 + j]);
        }
    }
    int32_t* newConfigData = reinterpret_cast<int32_t*>(newLoraConfig.getConfig().value().getData());
    for (int i = 0; i < weightsDim0; ++i)
    {
        for (int j = 0; j < configDim1; ++j)
        {
            EXPECT_EQ(configData[i * configDim1 + j], newConfigData[i * configDim1 + j]);
        }
    }
}
