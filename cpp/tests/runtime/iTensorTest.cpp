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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace tensorrt_llm::runtime;
using namespace ::testing;

namespace
{

TEST(iTensorTest, UnsqueezeShape)
{
    auto oldShape = ITensor::makeShape({2, 3, 4, 5});
    {
        auto shape = ITensor::unsqueeze(oldShape, 0);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 1);
        EXPECT_EQ(shape.d[1], 2);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }
    {
        auto shape = ITensor::unsqueeze(oldShape, 1);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 1);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }

    {
        auto shape = ITensor::unsqueeze(oldShape, 4);

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 3);
        EXPECT_EQ(shape.d[2], 4);
        EXPECT_EQ(shape.d[3], 5);
        EXPECT_EQ(shape.d[4], 1);
    }

    std::vector<int> invalidDims{-1, 5, 10};
    for (auto invalidDim : invalidDims)
    {
        try
        {
            auto shape = ITensor::unsqueeze(oldShape, invalidDim);
            FAIL() << "Expected failure";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Invalid dim"));
        }
        catch (...)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

TEST(iTensorTest, UnsqueezeTensor)
{
    auto oldShape = ITensor::makeShape({2, 3, 4, 5});
    BufferManager manager(std::make_shared<CudaStream>());

    {
        auto tensor = manager.cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(0);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 1);
        EXPECT_EQ(shape.d[1], 2);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }
    {
        auto tensor = manager.cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(1);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 1);
        EXPECT_EQ(shape.d[2], 3);
        EXPECT_EQ(shape.d[3], 4);
        EXPECT_EQ(shape.d[4], 5);
    }

    {
        auto tensor = manager.cpu(oldShape, nvinfer1::DataType::kINT32);
        tensor->unsqueeze(4);
        auto shape = tensor->getShape();

        EXPECT_EQ(shape.nbDims, 5);
        EXPECT_EQ(shape.d[0], 2);
        EXPECT_EQ(shape.d[1], 3);
        EXPECT_EQ(shape.d[2], 4);
        EXPECT_EQ(shape.d[3], 5);
        EXPECT_EQ(shape.d[4], 1);
    }

    std::vector<int> invalidDims{-1, 5, 10};
    for (auto invalidDim : invalidDims)
    {
        try
        {
            auto tensor = manager.cpu(oldShape, nvinfer1::DataType::kINT32);
            tensor->unsqueeze(invalidDim);
            FAIL() << "Expected failure";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Invalid dim"));
        }
        catch (...)
        {
            FAIL() << "Expected TllmException";
        }
    }
}

} // namespace
