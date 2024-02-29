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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

TEST(ITensorTest, SqueezeTensor)
{
    auto dims = ITensor::makeShape({16, 1, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};

    auto squeezeDim = 0;
    EXPECT_THROW(tensor->squeeze(squeezeDim), std::runtime_error);
    squeezeDim = 1;
    auto squeezed = ITensor::view(tensor, ITensor::squeeze(dims, squeezeDim));

    EXPECT_EQ(squeezed->getSize(), tensor->getSize());
    EXPECT_EQ(squeezed->getShape().nbDims, tensor->getShape().nbDims - 1);
    EXPECT_EQ(squeezed->getShape().d[0], tensor->getShape().d[0]);
    EXPECT_EQ(squeezed->getShape().d[1], tensor->getShape().d[2]);

    EXPECT_NO_THROW(squeezed->release());
    EXPECT_EQ(squeezed->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(ITensorTest, UnsqueezeShape)
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
            ITensor::unsqueeze(oldShape, invalidDim);
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

TEST(ITensorTest, UnsqueezeTensor)
{
    auto oldShape = ITensor::makeShape({2, 3, 4, 5});

    {
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
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
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
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
        auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
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
            auto tensor = BufferManager::cpu(oldShape, nvinfer1::DataType::kINT32);
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

TEST(ITensorTest, TensorView)
{
    auto const dims = ITensor::makeShape({16, 1, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor = BufferManager::cpu(dims, dataType);

    auto const viewDims = ITensor::makeShape({16, 1, 2});

    auto view = ITensor::view(tensor, viewDims);
    EXPECT_EQ(view->getSize(), tensor->getSize() / 2);
    EXPECT_EQ(view->getShape().nbDims, tensor->getShape().nbDims);
    EXPECT_EQ(view->getShape().d[2], tensor->getShape().d[2] / 2);

    EXPECT_NO_THROW(view->release());
    EXPECT_EQ(view->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);
}

TEST(ITensorTest, TensorSlice)
{
    auto dims = ITensor::makeShape({16, 8, 4});
    auto constexpr dataType = nvinfer1::DataType::kFLOAT;
    ITensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
    auto offset = dims.d[0] / 4;
    auto slice = ITensor::slice(tensor, offset);
    auto const sizeSlice = 3 * tensor->getSize() / 4;
    EXPECT_EQ(slice->getShape().d[0], dims.d[0] - offset);
    EXPECT_EQ(slice->getSize(), sizeSlice);
    EXPECT_EQ(slice->getCapacity(), sizeSlice);
    EXPECT_EQ(static_cast<std::uint8_t*>(slice->data()) - static_cast<std::uint8_t*>(tensor->data()),
        offset * ITensor::volume(dims) / dims.d[0] * BufferDataType(dataType).getSize());

    auto dimsNew = ITensor::makeShape({12, 32});
    EXPECT_EQ(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[1], dimsNew.d[1]);
    dimsNew.d[0] = 6;
    EXPECT_LT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_NO_THROW(slice->reshape(dimsNew));
    EXPECT_EQ(slice->getShape().d[0], dimsNew.d[0]);
    dimsNew.d[0] = 16;
    EXPECT_GT(ITensor::volume(dimsNew), sizeSlice);
    EXPECT_THROW(slice->reshape(dimsNew), std::runtime_error);

    EXPECT_NO_THROW(slice->resize(sizeSlice));
    EXPECT_NO_THROW(slice->resize(sizeSlice / 2));
    EXPECT_EQ(slice->getShape().d[0], sizeSlice / 2);
    EXPECT_THROW(slice->resize(sizeSlice * 2), std::runtime_error);
    EXPECT_NO_THROW(slice->release());
    EXPECT_EQ(slice->data(), nullptr);
    EXPECT_NE(tensor->data(), nullptr);

    std::shared_ptr<ITensor const> constTensor{tensor};
    auto constSlice = ITensor::slice(constTensor, offset);
    EXPECT_EQ(constSlice->getShape().d[0], dims.d[0] - offset);
    auto uniqueSlice = ITensor::slice(std::move(constSlice), 1);
    EXPECT_EQ(uniqueSlice->getShape().d[0], dims.d[0] - offset - 1);
}
