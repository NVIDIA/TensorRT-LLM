/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <numeric>
#include <string>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace fs = std::filesystem;

class UtilsTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount == 0)
            GTEST_SKIP();

        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    void TearDown() override {}

    int mDeviceCount;
    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
};

TEST_F(UtilsTest, LoadNpy)
{
    auto const testResourcePath = fs::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
    auto const inputFile = testResourcePath / "data/input_tokens.npy";

    auto loadedTensor = utils::loadNpy(*mManager, inputFile.string(), MemoryType::kCPU);

    ASSERT_EQ(loadedTensor->getSize(), 96);
    EXPECT_EQ(loadedTensor->getShape().nbDims, 2);
    EXPECT_EQ(loadedTensor->getShape().d[0], 8);
    EXPECT_EQ(loadedTensor->getShape().d[1], 12);
}

TEST_F(UtilsTest, LoadStoreNpy)
{
    auto dims = ITensor::makeShape({2, 3, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    auto tensorRange = BufferRange<float>(*tensor);
    std::iota(tensorRange.begin(), tensorRange.end(), 0);

    std::string filename{"tensor.npy"};
    utils::saveNpy(*mManager, *tensor, filename);
    auto loadedTensor = utils::loadNpy(*mManager, filename, MemoryType::kCPU);

    ASSERT_EQ(loadedTensor->getSize(), tensor->getSize());
    EXPECT_EQ(loadedTensor->getShape().nbDims, tensor->getShape().nbDims);
    EXPECT_EQ(loadedTensor->getShape().d[0], tensor->getShape().d[0]);
    EXPECT_EQ(loadedTensor->getShape().d[1], tensor->getShape().d[1]);
    EXPECT_EQ(loadedTensor->getShape().d[2], tensor->getShape().d[2]);

    auto loadedTensorRange = BufferRange<float>(*loadedTensor);
    for (size_t i = 0; i < tensor->getSize(); ++i)
    {
        EXPECT_EQ(loadedTensorRange[i], tensorRange[i]);
    }
}

TEST_F(UtilsTest, LoadStoreNpyGPU)
{
    auto dims = ITensor::makeShape({2, 3, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    auto tensorRange = BufferRange<float>(*tensor);
    std::iota(tensorRange.begin(), tensorRange.end(), 0);

    auto deviceTensor = mManager->copyFrom(*tensor, MemoryType::kGPU);

    std::string filename{"tensor.npy"};
    utils::saveNpy(*mManager, *deviceTensor, filename);
    auto loadedTensor = utils::loadNpy(*mManager, filename, MemoryType::kGPU);

    ASSERT_EQ(loadedTensor->getSize(), tensor->getSize());
    EXPECT_EQ(loadedTensor->getShape().nbDims, tensor->getShape().nbDims);
    EXPECT_EQ(loadedTensor->getShape().d[0], tensor->getShape().d[0]);
    EXPECT_EQ(loadedTensor->getShape().d[1], tensor->getShape().d[1]);
    EXPECT_EQ(loadedTensor->getShape().d[2], tensor->getShape().d[2]);

    auto hostTensor = mManager->copyFrom(*loadedTensor, MemoryType::kCPU);

    auto loadedTensorRange = BufferRange<float>(*hostTensor);
    for (size_t i = 0; i < tensor->getSize(); ++i)
    {
        EXPECT_EQ(loadedTensorRange[i], tensorRange[i]);
    }
}
