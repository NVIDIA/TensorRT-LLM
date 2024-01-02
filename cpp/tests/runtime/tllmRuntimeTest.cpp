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

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace fs = std::filesystem;
namespace trt = nvinfer1;
namespace onnx = nvonnxparser;

namespace
{
auto const TEST_RESOURCE_DIR = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const MNIST_MODEL_PATH = TEST_RESOURCE_DIR / "models/mnist.onnx";

template <typename T>
std::unique_ptr<T> makeUnique(T* ptr)
{
    EXPECT_NE(ptr, nullptr);
    return std::unique_ptr<T>(ptr);
}

std::unique_ptr<trt::IHostMemory> buildMnistEngine(trt::ILogger& logger)
{
    EXPECT_TRUE(fs::exists(MNIST_MODEL_PATH));
    auto builder = makeUnique(trt::createInferBuilder(logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(trt::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = makeUnique(builder->createNetworkV2(explicitBatch));
    auto parser = makeUnique(nvonnxparser::createParser(*network, logger));
    auto const parsingSuccess = parser->parseFromFile(
        MNIST_MODEL_PATH.string().c_str(), static_cast<int32_t>(trt::ILogger::Severity::kWARNING));
    EXPECT_TRUE(parsingSuccess);
    auto config = makeUnique(builder->createBuilderConfig());
    return makeUnique(builder->buildSerializedNetwork(*network, *config));
}
} // namespace

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class TllmRuntimeTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();

        if (mDeviceCount == 0)
            GTEST_SKIP();

        mLogger.setLevel(trt::ILogger::Severity::kINFO);
        mSerializedEngine = buildMnistEngine(mLogger);
        ASSERT_NE(mSerializedEngine, nullptr);
    }

    void TearDown() override {}

    int mDeviceCount;
    TllmLogger mLogger{};
    std::unique_ptr<trt::IHostMemory> mSerializedEngine;
};

TEST_F(TllmRuntimeTest, SinglePass)
{
    EXPECT_TRUE(mSerializedEngine);
    TllmRuntime rt{*mSerializedEngine, mLogger};
    auto& engine = rt.getEngine();
    EXPECT_FALSE(engine.hasImplicitBatchDimension());
    EXPECT_EQ(rt.getNbProfiles(), engine.getNbOptimizationProfiles());
    EXPECT_EQ(rt.getNbContexts(), 0);
    auto const nbIoTensors = engine.getNbIOTensors();
    EXPECT_EQ(nbIoTensors, 2);
    rt.addContext(0);
    EXPECT_EQ(rt.getNbContexts(), 1);

    auto constexpr dataType = trt::DataType::kFLOAT;

    auto const inputName = engine.getIOTensorName(0);
    EXPECT_EQ(engine.getTensorIOMode(inputName), trt::TensorIOMode::kINPUT);
    auto const inputDims = engine.getTensorShape(inputName);
    std::array constexpr inputDimsExpected = {1, 1, 28, 28};
    EXPECT_EQ(inputDims.nbDims, inputDimsExpected.size());
    for (int i = 0; i < inputDims.nbDims; ++i)
    {
        EXPECT_EQ(inputDims.d[i], inputDimsExpected[i]);
    }
    EXPECT_EQ(engine.getTensorDataType(inputName), dataType);

    auto const outputName = engine.getIOTensorName(1);
    EXPECT_EQ(engine.getTensorIOMode(outputName), trt::TensorIOMode::kOUTPUT);
    auto const outputDims = engine.getTensorShape(outputName);
    std::array constexpr outputDimsExpected = {1, 10};
    EXPECT_EQ(outputDims.nbDims, outputDimsExpected.size());
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        EXPECT_EQ(outputDims.d[i], outputDimsExpected[i]);
    }
    EXPECT_EQ(engine.getTensorDataType(outputName), dataType);

    auto& allocator = rt.getBufferManager();
    TllmRuntime::TensorMap tensorMap{};
    auto inputBuffer = std::shared_ptr<ITensor>{allocator.gpu(inputDims, dataType)};
    allocator.setZero(*inputBuffer);
    tensorMap.insert(std::make_pair(inputName, inputBuffer));
    rt.setInputTensors(0, tensorMap);
    rt.setOutputTensors(0, tensorMap);
    ASSERT_NE(tensorMap.find(outputName), tensorMap.end());
    auto outputBuffer = tensorMap.at(outputName);
    allocator.setZero(*outputBuffer);
    rt.executeContext(0);

    std::vector<float> output(outputBuffer->getSize());
    allocator.copy(*outputBuffer, output.data());
    rt.getStream().synchronize();
    auto min = std::min_element(output.begin(), output.end());
    EXPECT_NEAR(*min, -0.126409f, 1e-5f);
    auto max = std::max_element(output.begin(), output.end());
    EXPECT_NEAR(*max, 0.140218f, 1e-5f);
}
