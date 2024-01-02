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
#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/torchView.h"

#include <memory>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class TorchTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount > 0)
        {
            mStream = std::make_unique<CudaStream>();
            Torch::setCurrentStream(*mStream);
        }
        else
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}

    int mDeviceCount;
    BufferManager::CudaStreamPtr mStream;
};

namespace
{
template <nvinfer1::DataType DType>
void checkFilled(IBuffer& buffer, int fillValue)
{
    if (DType == buffer.getDataType())
    {
        EXPECT_THAT(BufferRange<typename DataTypeTraits<DType>::type>(buffer), ::testing::Each(fillValue));
    }
}
} // namespace

TEST_F(TorchTest, Aten)
{
    BufferManager manager(mStream);
    auto const shapeTllm = ITensor::makeShape({1, 2, 3, 4});
    auto const shapeAten = TorchUtils::shape(shapeTllm);

    auto const shapeSmall = ITensor::makeShape({1, 2, 3, 2});
    ASSERT_GT(ITensor::volume(shapeTllm), ITensor::volume(shapeSmall));
    auto const shapeLarge = ITensor::makeShape({1, 2, 3, 8});
    ASSERT_LT(ITensor::volume(shapeTllm), ITensor::volume(shapeLarge));

    for (int i = 0; i < shapeAten.size(); ++i)
    {
        EXPECT_EQ(shapeAten[i], shapeTllm.d[i]) << i;
    }

    auto constexpr fillValue = 1;
    auto tensorHostBase = manager.allocate(MemoryType::kPINNED, shapeTllm, nvinfer1::DataType::kINT64);

    for (auto memoryType : {MemoryType::kCPU, MemoryType::kGPU, MemoryType::kPINNED})
    {
        for (auto dtype : {nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT8,
                 nvinfer1::DataType::kUINT8, nvinfer1::DataType::kINT32, nvinfer1::DataType::kINT64,
                 nvinfer1::DataType::kBF16, nvinfer1::DataType::kFP8, nvinfer1::DataType::kBOOL})
        {
            ITensor::SharedPtr tensorTllm{manager.allocate(memoryType, shapeTllm, dtype)};

            // Conversion to ATen tensor
            auto tensorAten = Torch::tensor(tensorTllm);
            EXPECT_TRUE(
                (memoryType == MemoryType::kGPU && tensorAten.device().is_cuda()) || tensorAten.device().is_cpu());
            EXPECT_EQ(memoryType == MemoryType::kPINNED, tensorAten.is_pinned());
            EXPECT_EQ(TorchUtils::dataType(dtype), tensorAten.dtype());
            EXPECT_THAT(tensorAten.sizes(), ::testing::ElementsAreArray(shapeAten));
            EXPECT_EQ(tensorAten.data_ptr(), tensorTllm->data());

            if (dtype != nvinfer1::DataType::kFP8)
            {
                tensorAten.fill_(c10::Scalar(fillValue));
                auto tensorHost = ITensor::wrap(tensorHostBase->data(), dtype, shapeTllm);
                manager.copy(*tensorTllm, *tensorHost);
                mStream->synchronize();
                checkFilled<nvinfer1::DataType::kFLOAT>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kHALF>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kINT8>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kUINT8>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kINT32>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kINT64>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kBF16>(*tensorHost, fillValue);
                checkFilled<nvinfer1::DataType::kBOOL>(*tensorHost, fillValue);
            }

            // Conversion back to TRT-LLM tensor
            auto tensorView = TorchView::of(tensorAten);
            EXPECT_EQ(tensorView->getDataType(), dtype);
            EXPECT_EQ(tensorView->getMemoryType(), memoryType);
            EXPECT_EQ(tensorView->getSize(), tensorTllm->getSize());
            EXPECT_EQ(tensorView->getCapacity(), tensorTllm->getCapacity());
            EXPECT_EQ(TorchUtils::shape(tensorView->getShape()), shapeAten);
            EXPECT_EQ(tensorView->data(), tensorTllm->data());

            tensorView->reshape(shapeSmall);
            EXPECT_EQ(TorchUtils::shape(tensorView->getShape()), TorchUtils::shape(shapeSmall));
            EXPECT_LT(tensorView->getSize(), tensorTllm->getSize());
            EXPECT_EQ(tensorView->getSize(), tensorAten.numel());
            EXPECT_EQ(tensorView->data(), tensorTllm->data());
            EXPECT_THROW(tensorView->reshape(shapeLarge), tc::TllmException);

            tensorView->release();
            EXPECT_EQ(tensorView->data(), nullptr);
            EXPECT_EQ(tensorView->getSize(), 0);
        }
    }
}

TEST_F(TorchTest, Resize)
{
    auto devices = {at::Device{at::kCUDA, 0}, at::Device{at::kCPU}};
    for (auto device : devices)
    {
        auto tensorAten = at::randn({1, 2, 3, 4}).to(device);
        auto tensorView = TorchView::of(tensorAten);
        auto size = tensorView->getSize();
        EXPECT_NO_THROW(tensorView->resize(size / 2));
        EXPECT_EQ(tensorView->getSize(), size / 2);
        EXPECT_NO_THROW(tensorView->resize(size * 2));
        EXPECT_EQ(tensorView->getSize(), size * 2);
    }
}
