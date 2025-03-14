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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/tensor.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <functional>
#include <initializer_list>
#include <memory>

using namespace std::placeholders;

using namespace tensorrt_llm::executor;

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

class TensorTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount > 0)
        {
            mStream = std::make_unique<tr::CudaStream>();
        }
        else
        {
            GTEST_SKIP();
        }
    }

    void TearDown() override {}

    int mDeviceCount;
    std::shared_ptr<tr::CudaStream> mStream;
};

namespace
{

template <typename T>
T convertType(std::size_t val)
{
    return static_cast<T>(val);
}

template <>
half convertType(std::size_t val)
{
    return __float2half_rn(static_cast<float>(val));
}

template <typename T>
void testCreateTensor(std::shared_ptr<tr::CudaStream> const& stream)
{
    EXPECT_FALSE(Tensor{});

    std::initializer_list<std::function<Tensor(Shape)>> const constructors = {&Tensor::cpu<T>, &Tensor::pinned<T>,
        &Tensor::managed<T>, [&stream](auto shape) { return Tensor::gpu<T>(stream, shape); }};

    for (auto const& constructor : constructors)
    {
        auto empty = constructor({});
        EXPECT_EQ(empty.getSize(), 0);
        EXPECT_EQ(empty.getData(), nullptr);
        EXPECT_NE(empty.getDataType(), DataType::kUNKNOWN);
        EXPECT_NE(empty.getMemoryType(), MemoryType::kUNKNOWN);
        EXPECT_TRUE(static_cast<bool>(empty));

        auto nonEmpty = constructor({1, 2, 3, 4});
        auto const shape = nonEmpty.getShape();
        EXPECT_EQ(shape.size(), 4);
        Shape::DimType64 volume{1};
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            EXPECT_EQ(shape[i], i + 1);
            volume *= shape[i];
        }
        EXPECT_EQ(nonEmpty.getSize(), volume);
        EXPECT_NE(nonEmpty.getData(), nullptr);
        EXPECT_TRUE(static_cast<bool>(nonEmpty));

        auto iTensor = detail::toITensor(nonEmpty);
        EXPECT_EQ(iTensor->data(), nonEmpty.getData());
        EXPECT_EQ(detail::ofITensor(iTensor), nonEmpty);
    }
}

TEST_F(TensorTest, testCreateTensor)
{
    testCreateTensor<float>(mStream);
    testCreateTensor<half>(mStream);
    testCreateTensor<std::int8_t>(mStream);
    testCreateTensor<std::uint8_t>(mStream);
    testCreateTensor<std::int32_t>(mStream);
}

template <typename T>
void testRoundTrip(std::shared_ptr<tr::CudaStream> const& stream)
{
    auto constexpr size = 128;
    std::vector<T> inputCpu(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        inputCpu[i] = convertType<T>(i);
    }

    // WAR spurious failure: run twice
    bool success{false};
    for (std::size_t run = 0; run < 2; ++run)
    {
        auto inputGpu = Tensor::of(inputCpu).copyToGpu(stream);
        auto dtype = inputGpu.getDataType();
        auto outputCpu = inputGpu.copyToManaged(stream).copyToPinned().copyToCpu();
        EXPECT_EQ(inputCpu.size(), outputCpu.getSize());
        stream->synchronize();
        auto outputCpuTyped = static_cast<T*>(outputCpu.getData());

        std::size_t numMismatches{0};
        for (size_t i = 0; i < inputCpu.size(); ++i)
        {
            numMismatches += (inputCpu[i] != outputCpuTyped[i]);
        }
        if (numMismatches > 0)
        {
            TLLM_LOG_WARNING("testRoundTrip mismatches for dtype %d: %lu", static_cast<int>(dtype), numMismatches);
            continue;
        }

        inputGpu.setZero(stream);
        outputCpu.setFrom(inputGpu, stream);
        stream->synchronize();
        for (size_t i = 0; i < inputCpu.size(); ++i)
        {
            EXPECT_EQ(0, static_cast<int32_t>(outputCpuTyped[i]));
        }

        success = true;
        break;
    }
    ASSERT_TRUE(success);
}

TEST_F(TensorTest, CreateCopyRoundTrip)
{
    testRoundTrip<float>(mStream);
    testRoundTrip<half>(mStream);
    testRoundTrip<std::int8_t>(mStream);
    testRoundTrip<std::uint8_t>(mStream);
    testRoundTrip<std::int32_t>(mStream);
}

using ParamType = std::tuple<MemoryType>;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto const memoryType = std::get<0>(info.param);
    std::string name = "serializeDeserialize";
    if (memoryType == MemoryType::kCPU)
    {
        name += "Cpu";
    }
    else if (memoryType == MemoryType::kCPU_PINNED)
    {
        name += "CpuPinned";
    }
    else if (memoryType == MemoryType::kUVM)
    {
        name += "UVM";
    }
    else if (memoryType == MemoryType::kGPU)
    {
        name += "Gpu";
    }
    return name;
}

class ParamTest : public TensorTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, serializeDeserialize)
{
    MemoryType memoryType = std::get<0>(GetParam());

    DataType dataType = DataType::kFP32;
    Tensor cpuTensor = Tensor::cpu(dataType, Shape({2, 2}));
    float* data = reinterpret_cast<float*>(cpuTensor.getData());
    data[0] = 123;
    data[1] = 456;
    data[2] = 789;
    data[3] = 10;

    Tensor tensor;
    switch (memoryType)
    {
    case MemoryType::kCPU: tensor = cpuTensor; break;
    case MemoryType::kCPU_PINNED: tensor = cpuTensor.copyToPinned(); break;
    case MemoryType::kUVM: tensor = cpuTensor.copyToManaged(); break;
    case MemoryType::kGPU:
        auto stream = std::make_shared<tr::CudaStream>();
        tensor = cpuTensor.copyToGpu(stream);
        stream->synchronize();
        break;
    }

    auto serializedSize = Serialization::serializedSize(tensor);
    std::ostringstream os;
    Serialization::serialize(tensor, os);
    EXPECT_EQ(os.str().size(), serializedSize);

    std::istringstream is(os.str());
    auto newTensor = Serialization::deserializeTensor(is);

    EXPECT_EQ(newTensor.getShape().size(), tensor.getShape().size());
    EXPECT_EQ(newTensor.getDataType(), tensor.getDataType());
    EXPECT_EQ(newTensor.getMemoryType(), tensor.getMemoryType());

    auto stream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    auto newCpuTensor = newTensor.copyToCpu(stream);
    stream->synchronize();
    float* newData = reinterpret_cast<float*>(newCpuTensor.getData());
    EXPECT_EQ(data[0], newData[0]);
    EXPECT_EQ(data[1], newData[1]);
    EXPECT_EQ(data[2], newData[2]);
    EXPECT_EQ(data[3], newData[3]);
}

INSTANTIATE_TEST_SUITE_P(TensorTest, ParamTest,
    testing::Combine(testing::Values(MemoryType::kCPU, MemoryType::kCPU_PINNED, MemoryType::kUVM, MemoryType::kGPU)),
    generateTestName);

} // namespace
