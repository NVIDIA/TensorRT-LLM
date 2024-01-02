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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <limits>
#include <memory>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

class BufferManagerTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount > 0)
        {
            mStream = std::make_unique<CudaStream>();
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
void testRoundTrip(BufferManager& manager)
{
    auto constexpr size = 128;
    std::vector<T> inputCpu(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        inputCpu[i] = convertType<T>(i);
    }
    auto inputGpu = manager.copyFrom(inputCpu, MemoryType::kGPU);
    auto outputCpu = manager.copyFrom(*inputGpu, MemoryType::kPINNED);
    EXPECT_EQ(inputCpu.size(), outputCpu->getSize());
    manager.getStream().synchronize();
    auto outputCpuTyped = bufferCast<T>(*outputCpu);
    for (size_t i = 0; i < inputCpu.size(); ++i)
    {
        EXPECT_EQ(inputCpu[i], outputCpuTyped[i]);
    }

    manager.setZero(*inputGpu);
    manager.copy(*inputGpu, *outputCpu);
    manager.getStream().synchronize();
    for (size_t i = 0; i < inputCpu.size(); ++i)
    {
        EXPECT_EQ(0, static_cast<int32_t>(outputCpuTyped[i]));
    }
}
} // namespace

TEST_F(BufferManagerTest, CreateCopyRoundTrip)
{
    BufferManager manager(mStream);
    testRoundTrip<float>(manager);
    testRoundTrip<half>(manager);
    testRoundTrip<std::int8_t>(manager);
    testRoundTrip<std::uint8_t>(manager);
    testRoundTrip<std::int32_t>(manager);
}

TEST_F(BufferManagerTest, Pointers)
{
    // This could be any C++ type supported by TensorRT.
    using cppBaseType = TokenIdType;
    // We want to store pointers to the C++ base type in the buffer.
    using cppPointerType = cppBaseType*;
    // This represents the TensorRT type for the pointer.
    auto constexpr trtPointerType = TRTDataType<cppPointerType>::value;
    static_assert(std::is_same_v<decltype(trtPointerType), BufferDataType const>);
    static_assert(trtPointerType.isPointer());
    static_assert(trtPointerType.getDataType() == TRTDataType<cppBaseType>::value);
    static_assert(static_cast<nvinfer1::DataType>(trtPointerType) == BufferDataType::kTrtPointerType);
    static_assert(trtPointerType == BufferDataType::kTrtPointerType); // uses implicit type conversion
    // The C++ type corresponding to the TensorRT type for storing pointers (int64_t)
    using cppStorageType = DataTypeTraits<trtPointerType>::type;
    static_assert(sizeof(cppStorageType) == sizeof(cppPointerType));

    BufferManager manager(mStream);
    auto constexpr batchSize = 16;
    // This buffer is on the CPU for convenient testing. In real code, this would be on the GPU.
    auto pointers = manager.allocate(MemoryType::kCPU, batchSize, trtPointerType);
    // We cast to the correct C++ pointer type checking that the underlying storage type is int64_t.
    auto pointerBuf = bufferCast<cppPointerType>(*pointers);

    // Create the GPU tensors.
    std::vector<ITensor::UniquePtr> tensors(batchSize);
    auto constexpr beamWidth = 4;
    auto constexpr maxSeqLen = 10;
    auto const shape = ITensor::makeShape({beamWidth, maxSeqLen});
    for (auto i = 0u; i < batchSize; ++i)
    {
        tensors[i] = manager.allocate(MemoryType::kGPU, shape, TRTDataType<cppBaseType>::value);
        pointerBuf[i] = bufferCast<cppBaseType>(*tensors[i]);
    }

    // Test that all pointers are valid
    for (auto i = 0u; i < batchSize; ++i)
    {
        EXPECT_EQ(pointerBuf[i], tensors[i]->data());
    }
}

TEST_F(BufferManagerTest, MemPoolAttributes)
{
    BufferManager manager(mStream); // sets attributes of the default memory pool
    auto const device = mStream->getDevice();
    ::cudaMemPool_t memPool;
    TLLM_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&memPool, device));
    std::uint64_t threshold{0};
    TLLM_CUDA_CHECK(cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &threshold));
    EXPECT_EQ(threshold, std::numeric_limits<std::uint64_t>::max());

    manager.memoryPoolTrimTo(0);
    auto const reserved = manager.memoryPoolReserved();
    auto const used = manager.memoryPoolUsed();
    auto const free = manager.memoryPoolFree();
    EXPECT_EQ(free, reserved - used);
    auto constexpr kBytesToReserve = 1 << 20;
    {
        auto const mem = manager.allocate(MemoryType::kGPU, kBytesToReserve);
        EXPECT_EQ(mem->getSize(), kBytesToReserve);
        EXPECT_GE(manager.memoryPoolReserved(), reserved + kBytesToReserve);
        EXPECT_GE(manager.memoryPoolUsed(), used + kBytesToReserve);
    }
    EXPECT_GE(manager.memoryPoolFree(), free + kBytesToReserve);
    manager.memoryPoolTrimTo(0);
    EXPECT_LE(manager.memoryPoolReserved(), reserved);
    EXPECT_LE(manager.memoryPoolFree(), free);
}
