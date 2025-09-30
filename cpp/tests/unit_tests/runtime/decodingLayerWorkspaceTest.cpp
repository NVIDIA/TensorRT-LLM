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

#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include <gtest/gtest.h>
#include <random>

using namespace tensorrt_llm;

namespace
{
void populateCpuBufferWithRandomBytes(uint64_t seed, runtime::IBuffer& buffer)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);
    auto* bufferPtr = reinterpret_cast<unsigned char*>(buffer.data());
    for (size_t i = 0; i < buffer.getSizeInBytes(); ++i)
    {
        *(bufferPtr + i) = static_cast<unsigned char>(distribution(generator));
    }
}

bool areMemoryRegionsEqual(void const* ptr1, void const* ptr2, size_t size)
{
    // Use std::memcmp to compare the memory regions
    return std::memcmp(ptr1, ptr2, size) == 0;
}

void testBufferEqual(runtime::IBuffer& left, runtime::IBuffer& right)
{
    auto const size = left.getSizeInBytes();
    ASSERT_EQ(size, right.getSizeInBytes());
    ASSERT_TRUE(areMemoryRegionsEqual(left.data(), right.data(), size));
}
} // namespace

auto const maxBatchSizePowersOfTwo = testing::Range(0, 14, 1);
auto const workspaceSizePowersOfTwo = testing::Range(0, 30, 2);

auto const initialBatchAndWorkspaceSizes = testing::Combine(maxBatchSizePowersOfTwo, workspaceSizePowersOfTwo);

using BasicUsageParamType = std::tuple<int32_t, int32_t>;

class BasicUsageTest : public testing::TestWithParam<BasicUsageParamType>
{
    void SetUp() override
    {
        auto const deviceCount = common::getDeviceCount();
        if (deviceCount > 0)
        {
            mBufferManager = std::make_shared<runtime::BufferManager>(std::make_unique<runtime::CudaStream>());
        }
        else
        {
            GTEST_SKIP() << "This test suite cannot run on systems with no devices.";
        }
    }

protected:
    std::shared_ptr<runtime::BufferManager> mBufferManager = nullptr;
};

TEST_P(BasicUsageTest, TestBasicUsageOfDecodingLayerWorkspace)
{
    auto const [maxBatchSizePowerOfTwo, workspaceSizePowerOfTwo] = GetParam();
    auto const maxBatchSize = static_cast<runtime::SizeType32>(std::pow(2, maxBatchSizePowerOfTwo));
    auto const workspaceSizeInBytes = static_cast<size_t>(std::pow(2, workspaceSizePowerOfTwo));
    auto const decoderDomain = tensorrt_llm::layers::DecoderDomain(maxBatchSize, 1, 1000, 1024);

    // Testing constructing the workspace.
    auto workspace = runtime::DecodingLayerWorkspace(
        mBufferManager, decoderDomain, tensorrt_llm::runtime::TRTDataType<float>::value, workspaceSizeInBytes);
    mBufferManager->getStream().synchronize();
    ASSERT_EQ(workspace.getWorkspaceDeviceBuffer()->getSizeInBytes(), workspaceSizeInBytes)
        << "The workspace size is not equal to the size we asked it to be.";
    ASSERT_EQ(workspace.getDeviceBatchSlots()->getSize(), maxBatchSize)
        << "The size of the device batch slots is not the max batch size provided to the workspace";

    // Testing enlarging the workspace.
    workspace.resize(workspaceSizeInBytes / 2);
    ASSERT_EQ(workspace.getWorkspaceDeviceBuffer()->getSizeInBytes(), workspaceSizeInBytes)
        << "The workspace size should not shrink.";
    auto const biggerWorkspaceSize = workspaceSizeInBytes * 2;
    workspace.resize(biggerWorkspaceSize);
    ASSERT_EQ(workspace.getWorkspaceDeviceBuffer()->getSizeInBytes(), biggerWorkspaceSize)
        << "The workspace was not enlarged as expected";

    // Checking that the device batch slots are actually on device
    auto const deviceBatchSlots = workspace.getDeviceBatchSlots();
    ASSERT_EQ(deviceBatchSlots->getMemoryType(), runtime::MemoryType::kGPU)
        << "The device batch slots should be on device.";

    auto const* deviceBatchSlotsPtr = workspace.getDeviceBatchSlotsPtr();
    ASSERT_EQ(tensorrt_llm::common::getPtrCudaMemoryType(deviceBatchSlotsPtr), cudaMemoryType::cudaMemoryTypeDevice)
        << "Pointer to device batch slots should have cudaMemoryType = device.";
}

INSTANTIATE_TEST_SUITE_P(BasicUsage, BasicUsageTest, initialBatchAndWorkspaceSizes);

auto const randomSeeds = testing::Values(static_cast<std::uint64_t>(1234));
auto const tensorDimensions = testing::Values(10, 100);
auto const tensorDataTypes = testing::Values(runtime::TRTDataType<bool>::value, runtime::TRTDataType<half>::value,
    runtime::TRTDataType<float>::value, runtime::TRTDataType<float*>::value);
auto const tensorDataTypesTuples = testing::Combine(tensorDataTypes, tensorDataTypes, tensorDataTypes);

auto const tensorShapeTuples = testing::Combine(tensorDimensions, tensorDimensions, tensorDimensions);
auto const mirrorInWorkspaceParams = testing::Combine(tensorDataTypesTuples, tensorShapeTuples, randomSeeds);

using MirrorInWorkspaceParamType = std::tuple<std::tuple<nvinfer1::DataType, nvinfer1::DataType, nvinfer1::DataType>,
    std::tuple<std::int32_t, std::int32_t, std::int32_t>, std::uint64_t>;

class MirrorInWorkspaceTest : public testing::TestWithParam<MirrorInWorkspaceParamType>
{
    void SetUp() override
    {
        auto const deviceCount = common::getDeviceCount();
        if (deviceCount > 0)
        {
            mBufferManager = std::make_shared<runtime::BufferManager>(std::make_unique<runtime::CudaStream>());
        }
        else
        {
            GTEST_SKIP() << "This test suite cannot run on systems with no devices.";
        }
    }

protected:
    std::shared_ptr<runtime::BufferManager> mBufferManager = nullptr;
};

TEST_P(MirrorInWorkspaceTest, TestMirrorInWorkspaceFunctionality)
{
    auto const [tensorDataTypes, tensorDimensions, randomSeed] = GetParam();
    auto const [tensorDataType1, tensorDataType2, tensorDataType3] = tensorDataTypes;
    auto const [tensorDimension1, tensorDimension2, tensorDimension3] = tensorDimensions;
    auto const decoderDomain = tensorrt_llm::layers::DecoderDomain(128, 1, 1000, 1024);

    // Testing constructing the workspace.
    auto const hostTensorShape1
        = tensorrt_llm::runtime::ITensor::makeShape({tensorDimension1, tensorDimension2, tensorDimension3});
    auto const hostTensorShape2
        = tensorrt_llm::runtime::ITensor::makeShape({tensorDimension2, tensorDimension3, tensorDimension1});
    auto const hostTensorShape3
        = tensorrt_llm::runtime::ITensor::makeShape({tensorDimension3, tensorDimension1, tensorDimension2});
    runtime::ITensor::SharedPtr const hostTensor1 = mBufferManager->cpu(hostTensorShape1, tensorDataType1);
    runtime::ITensor::SharedPtr const hostTensor2 = mBufferManager->cpu(hostTensorShape1, tensorDataType2);
    runtime::ITensor::SharedPtr const hostTensor3 = mBufferManager->cpu(hostTensorShape1, tensorDataType3);

    auto const requiredWorkspaceSize = tensorrt_llm::runtime::DecodingLayerWorkspace::calculateRequiredWorkspaceSize(
        std::make_pair(hostTensorShape1, tensorDataType1), std::make_pair(hostTensorShape2, tensorDataType2),
        std::make_pair(hostTensorShape3, tensorDataType3));
    auto workspace = runtime::DecodingLayerWorkspace(
        mBufferManager, decoderDomain, tensorrt_llm::runtime::TRTDataType<float>::value, requiredWorkspaceSize);
    mBufferManager->getStream().synchronize();

    ASSERT_LE(hostTensor1->getSizeInBytes() + hostTensor2->getSizeInBytes() + hostTensor3->getSizeInBytes(),
        requiredWorkspaceSize)
        << "The calculated workspace size cannot possibly be enough to contain all the tensors.";

    constexpr std::size_t addressAlignment = tensorrt_llm::common::kCudaMemAlign;
    constexpr std::size_t numTensors = 3;
    constexpr std::size_t maxAlignmentOverhead = numTensors * addressAlignment;
    ASSERT_GE(hostTensor1->getSizeInBytes() + hostTensor2->getSizeInBytes() + hostTensor3->getSizeInBytes()
            + maxAlignmentOverhead,
        requiredWorkspaceSize)
        << "We probably overestimate the amount of space the workspace requires.";
    populateCpuBufferWithRandomBytes(randomSeed, *hostTensor1);
    populateCpuBufferWithRandomBytes(randomSeed, *hostTensor2);
    populateCpuBufferWithRandomBytes(randomSeed, *hostTensor3);

    auto const [deviceTensor1, deviceTensor2, deviceTensor3]
        = workspace.mirrorInWorkspace(hostTensor1, hostTensor2, hostTensor3);

    runtime::ITensor::SharedPtr const hostTensorCopy1 = mBufferManager->cpu(hostTensorShape1, tensorDataType1);
    runtime::ITensor::SharedPtr const hostTensorCopy2 = mBufferManager->cpu(hostTensorShape1, tensorDataType2);
    runtime::ITensor::SharedPtr const hostTensorCopy3 = mBufferManager->cpu(hostTensorShape1, tensorDataType3);
    mBufferManager->copy(*deviceTensor1, *hostTensorCopy1);
    mBufferManager->copy(*deviceTensor2, *hostTensorCopy2);
    mBufferManager->copy(*deviceTensor3, *hostTensorCopy3);
    testBufferEqual(*hostTensor1, *hostTensorCopy1);
    testBufferEqual(*hostTensor2, *hostTensorCopy2);
    testBufferEqual(*hostTensor3, *hostTensorCopy3);
}

INSTANTIATE_TEST_SUITE_P(MirrorInWorkspace, MirrorInWorkspaceTest, mirrorInWorkspaceParams);
