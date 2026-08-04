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
#include "tensorrt_llm/runtime/cudaMemPool.h"

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
            mBufferManager = std::make_unique<BufferManager>(std::make_unique<CudaStream>());
        }
        else
        {
            GTEST_SKIP() << "This test suite cannot run on systems with no devices.";
        }
    }

    void TearDown() override {}

    std::size_t memoryPoolReserved()
    {
        return mBufferManager->memoryPoolReserved();
    }

    std::size_t memoryPoolFree()
    {
        return mBufferManager->memoryPoolFree();
    }

    int mDeviceCount;
    std::unique_ptr<BufferManager> mBufferManager = nullptr;
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
    auto outputCpu = manager.copyFrom(*inputGpu, MemoryType::kPINNEDPOOL);
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
    testRoundTrip<float>(*mBufferManager);
    testRoundTrip<half>(*mBufferManager);
    testRoundTrip<std::int8_t>(*mBufferManager);
    testRoundTrip<std::uint8_t>(*mBufferManager);
    testRoundTrip<std::int32_t>(*mBufferManager);
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

    auto constexpr batchSize = 16;
    // This buffer is on the CPU for convenient testing. In real code, this would be on the GPU.
    auto pointers = mBufferManager->allocate(MemoryType::kCPU, batchSize, trtPointerType);
    // We cast to the correct C++ pointer type checking that the underlying storage type is int64_t.
    auto pointerBuf = bufferCast<cppPointerType>(*pointers);

    // Create the GPU tensors.
    std::vector<ITensor::UniquePtr> tensors(batchSize);
    auto constexpr beamWidth = 4;
    auto constexpr maxSeqLen = 10;
    auto const shape = ITensor::makeShape({beamWidth, maxSeqLen});
    for (auto i = 0u; i < batchSize; ++i)
    {
        tensors[i] = mBufferManager->allocate(MemoryType::kGPU, shape, TRTDataType<cppBaseType>::value);
        pointerBuf[i] = bufferCast<cppBaseType>(*tensors[i]);
    }

    // Test that all pointers are valid
    for (auto i = 0u; i < batchSize; ++i)
    {
        EXPECT_EQ(pointerBuf[i], tensors[i]->data());
    }
}

class GpuAllocateAndFreeTest : public testing::TestWithParam<std::tuple<std::int32_t, std::int32_t>>
{
    void SetUp() override
    {
        auto const deviceCount = tc::getDeviceCount();
        if (deviceCount > 0)
        {
            mBufferManager = std::make_unique<BufferManager>(std::make_unique<CudaStream>());
        }
        else
        {
            GTEST_SKIP() << "This test suite cannot run on systems with no devices.";
        }
    }

protected:
    std::unique_ptr<BufferManager> mBufferManager = nullptr;
};

TEST_P(GpuAllocateAndFreeTest, MemPoolAttributes)
{
    auto const supportsMemPools = CudaMemPool::supportsMemoryPool(mBufferManager->getStream().getDevice());
    if (!supportsMemPools)
    {
        GTEST_SKIP() << "Test not runnable when memory pools are not supported.";
    }
    auto const params = GetParam();
    auto const initialAllocationSize = 1 << std::get<0>(params);
    IBuffer::UniquePtr initialAllocation{};
    initialAllocation = mBufferManager->allocate(MemoryType::kGPU, initialAllocationSize);
    mBufferManager->getStream().synchronize();
    EXPECT_EQ(initialAllocation->getSize(), initialAllocationSize)
        << "The initial memory allocation does not have the correct size.";
    auto const reservedAfterInitial = mBufferManager->memoryPoolReserved();
    ASSERT_GE(reservedAfterInitial, initialAllocationSize)
        << "The pool has less memory reserved than the initial allocation requires.";

    auto const usedAfterInitial = mBufferManager->memoryPoolUsed();
    auto const freeAfterInitial = mBufferManager->memoryPoolFree();
    EXPECT_EQ(freeAfterInitial, reservedAfterInitial - usedAfterInitial)
        << "Relationship between free, reserved and used memory is incorrect.";
    auto const additionalAllocationSize = 1 << std::get<1>(params);
    auto const additionalMemoryRequired = additionalAllocationSize - freeAfterInitial;
    IBuffer::UniquePtr additionalAllocation{};
    additionalAllocation = mBufferManager->allocate(MemoryType::kGPU, additionalAllocationSize);
    mBufferManager->getStream().synchronize();
    EXPECT_EQ(additionalAllocation->getSize(), additionalAllocationSize)
        << "The additional memory allocation does not have the correct size.";
    auto const reservedAfterAdditional = mBufferManager->memoryPoolReserved();
    auto const usedAfterAdditional = mBufferManager->memoryPoolUsed();
    auto const freeAfterAdditional = mBufferManager->memoryPoolFree();
    EXPECT_EQ(freeAfterAdditional, reservedAfterAdditional - usedAfterAdditional)
        << "Relationship between free, reserved and used memory is incorrect.";
    EXPECT_GE(reservedAfterAdditional, reservedAfterInitial + additionalMemoryRequired)
        << "The pool does not have enough reserved memory to contain the initial and the additional allocation.";
    EXPECT_GE(usedAfterAdditional, usedAfterInitial + additionalAllocationSize)
        << "The used memory in the pool is not sufficient to contain both the initial and additional allocation";
    additionalAllocation->release();
    mBufferManager->getStream().synchronize();
    auto const reservedAfterAdditionalRelease = mBufferManager->memoryPoolReserved();
    auto const usedAfterAdditionalRelease = mBufferManager->memoryPoolUsed();
    auto const freeAfterAdditionalRelease = mBufferManager->memoryPoolFree();
    EXPECT_EQ(freeAfterAdditionalRelease, reservedAfterAdditionalRelease - usedAfterAdditionalRelease)
        << "Relationship between free, reserved and used memory is incorrect.";
    EXPECT_EQ(usedAfterAdditionalRelease, usedAfterInitial)
        << "Releasing the additional allocation did not bring us back to the initial memory usage in the pool";
    EXPECT_LE(reservedAfterAdditionalRelease, reservedAfterAdditional)
        << "Freeing memory resulted in an increased pool reservation";

    mBufferManager->memoryPoolTrimTo(0);
    auto const reservedAfterTrim = mBufferManager->memoryPoolReserved();
    auto const usedAfterTrim = mBufferManager->memoryPoolUsed();
    auto const freeAfterTrim = mBufferManager->memoryPoolFree();
    EXPECT_EQ(freeAfterTrim, reservedAfterTrim - usedAfterTrim)
        << "Relationship between free, reserved and used memory is incorrect.";
    EXPECT_LE(reservedAfterTrim, reservedAfterAdditional)
        << "Trimming the memory pool resulted in more memory reserved. Expected less.";
}

auto const powers = testing::Range(0, 30, 5);

auto const powersCombinations = testing::Combine(powers, powers);

INSTANTIATE_TEST_SUITE_P(GpuAllocations, GpuAllocateAndFreeTest, powersCombinations);

TEST_F(BufferManagerTest, MemPoolAttributes)
{
    auto const supportsMemPools = CudaMemPool::supportsMemoryPool(mBufferManager->getStream().getDevice());
    if (!supportsMemPools)
    {
        GTEST_SKIP() << "Test not runnable when memory pools are not supported.";
    }
    mBufferManager->memoryPoolTrimTo(0);
    auto const reserved = mBufferManager->memoryPoolReserved();
    auto const used = mBufferManager->memoryPoolUsed();
    auto const free = mBufferManager->memoryPoolFree();
    EXPECT_EQ(free, reserved - used);
    auto constexpr kBytesToReserve = 1 << 20;
    {
        auto const mem = mBufferManager->allocate(MemoryType::kGPU, kBytesToReserve);
        EXPECT_EQ(mem->getSize(), kBytesToReserve);
        EXPECT_GE(mBufferManager->memoryPoolReserved(), reserved + kBytesToReserve);
        EXPECT_GE(mBufferManager->memoryPoolUsed(), used + kBytesToReserve);
    }
    EXPECT_GE(mBufferManager->memoryPoolFree(), free + kBytesToReserve);
    mBufferManager->memoryPoolTrimTo(0);
    EXPECT_LE(mBufferManager->memoryPoolReserved(), reserved);
    EXPECT_LE(mBufferManager->memoryPoolFree(), free);
}

TEST_F(BufferManagerTest, TrimPoolOnDestruction)
{
    auto const supportsMemPools = CudaMemPool::supportsMemoryPool(mBufferManager->getStream().getDevice());
    if (!supportsMemPools)
    {
        GTEST_SKIP() << "Test not runnable when memory pools are not supported.";
    }
    mBufferManager->memoryPoolTrimTo(0);
    mBufferManager = std::make_unique<BufferManager>(std::make_unique<CudaStream>(), true);
    auto const reserved = mBufferManager->memoryPoolReserved();
    auto const free = mBufferManager->memoryPoolFree();
    auto constexpr kBytesToReserve = 1 << 20;
    {
        auto const mem = mBufferManager->allocate(MemoryType::kGPU, kBytesToReserve);
    }
    EXPECT_GE(mBufferManager->memoryPoolFree(), free + kBytesToReserve);
    mBufferManager = std::make_unique<BufferManager>(std::make_unique<CudaStream>());
    EXPECT_LE(memoryPoolReserved(), reserved);
    EXPECT_LE(memoryPoolFree(), free);
}
