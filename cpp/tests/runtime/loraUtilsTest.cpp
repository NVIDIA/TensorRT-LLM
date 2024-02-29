/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntimeBase.h>
#include <algorithm>
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>

namespace tensorrt_llm::runtime::lora
{
using TensorPtr = ITensor::SharedPtr;

class LoraUtilsTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    LoraUtilsTest() {}

    void SetUp() override
    {
        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
};

TEST_F(LoraUtilsTest, null_values)
{
    std::optional<TensorPtr> optReqLoraWeights = std::nullopt;
    std::optional<TensorPtr> optReqLoraConfig = mManager->emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kHALF);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    optReqLoraConfig = std::nullopt;

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());
}

TEST_F(LoraUtilsTest, dims_mem_type)
{
    std::optional<TensorPtr> optReqLoraWeights = mManager->cpu(ITensor::makeShape({1, 2}), nvinfer1::DataType::kHALF);
    std::optional<TensorPtr> optReqLoraConfig
        = mManager->cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    std::optional<TensorPtr> optGpuWeights = mManager->gpu(ITensor::makeShape({1, 2, 50}), nvinfer1::DataType::kHALF);

    EXPECT_THAT([&]() { loraValidateRequestTensorDims(optGpuWeights, optReqLoraConfig); },
        testing::Throws<std::runtime_error>());

    optReqLoraWeights = mManager->cpu(ITensor::makeShape({1, 2, 50}), nvinfer1::DataType::kHALF);
    optReqLoraConfig = mManager->cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);

    loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig);
}

TEST_F(LoraUtilsTest, loraValidateRequestTensors)
{
    auto modelConfig = GptModelConfig(0, 2, 1, 4, nvinfer1::DataType::kFLOAT);
    auto worldConfig = WorldConfig();

    std::optional<TensorPtr> optReqLoraWeights
        = mManager->cpu(ITensor::makeShape({1, 2, 32}), nvinfer1::DataType::kFLOAT);
    std::optional<TensorPtr> optReqLoraConfig
        = mManager->cpu(ITensor::makeShape({1, 2, 3}), nvinfer1::DataType::kINT32);

    std::vector<int32_t> config{
        1,
        0,
        4,
        1,
        1,
        4,
    };

    auto configPtr = bufferCast<int32_t>(*optReqLoraConfig.value());
    std::copy_n(config.data(), config.size(), configPtr);

    EXPECT_THAT([&]() { loraValidateRequestTensors(optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig); },
        testing::Throws<std::runtime_error>());

    std::vector<LoraModule> modules{
        LoraModule(LoraModule::ModuleType::kATTN_Q, 4, 4, false, true, -1, 0),
    };
    modelConfig.setLoraModules(modules);

    loraValidateRequestTensors(optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig);
}

} // namespace tensorrt_llm::runtime::lora
