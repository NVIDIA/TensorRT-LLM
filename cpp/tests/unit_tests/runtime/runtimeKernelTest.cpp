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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <NvInferRuntime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

using TensorPtr = std::shared_ptr<tr::ITensor>;
using BufferPtr = std::shared_ptr<tr::IBuffer>;

class RuntimeKernelTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override
    {
        mDeviceCount = tc::getDeviceCount();
        if (mDeviceCount == 0)
        {
            GTEST_SKIP() << "No devices. These tests will not run.";
        }

        mStream = std::make_unique<tr::CudaStream>();
        mManager = std::make_unique<tr::BufferManager>(mStream);
    }

    void TearDown() override {}

    int mDeviceCount;
    std::unique_ptr<tr::BufferManager> mManager;
    tr::BufferManager::CudaStreamPtr mStream;
};

namespace
{
template <typename T>
void testFill(tr::IBuffer& buffer, tr::BufferManager& manager, tr::CudaStream& stream)
{
    T constexpr value{3};
    tr::kernels::invokeFill(buffer, value, stream);
    auto bufferHost = manager.copyFrom(buffer, tr::MemoryType::kCPU);
    auto bufferPtr = tr::bufferCast<T>(*bufferHost);
    auto constexpr expected = value;

    auto anyMismatch = false;
    for (std::size_t i = 0; i < buffer.getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
        anyMismatch |= bufferPtr[i] != expected;
    }
    ASSERT_FALSE(anyMismatch);
}
} // namespace

TEST_F(RuntimeKernelTest, FillBufferInt8)
{
    for (auto size : {123LLU, 1025LLU, 1LLU << 32})
    {
        auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT8);
        testFill<std::int8_t>(*buffer, *mManager, *mStream);
    }
}

TEST_F(RuntimeKernelTest, FillTensorInt8)
{
    for (auto size : {123, 1025, std::numeric_limits<int32_t>::max()})
    {
        auto tensor = mManager->gpu(tr::ITensor::makeShape({size, 2}), nvinfer1::DataType::kINT8);
        testFill<std::int8_t>(*tensor, *mManager, *mStream);
    }
}

TEST_F(RuntimeKernelTest, ScatterHalf)
{
    tr::SizeType32 const beamWidth{3};

    std::vector<half> const input{
        28524.F, 287.F, 5093.F, 12.F, 23316.F, 4881.F, 11.F, 30022.F, 263.F, 8776.F, 355.F, 257.F};
    tr::SizeType32 const batchSize{4};
    auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
    auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);
    mManager->setZero(*outputTensor);

    tr::kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
    auto* outputPtr = tr::bufferCast<half>(*outputHost);

    for (tr::SizeType32 batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = tc::flat_index2(batchIdx, i, inputLength);
                auto const expected = beam == 0 ? input[inputIdx] : half(0.F);
                auto const outputIdx = tc::flat_index3(batchIdx, beam, i, beamWidth, inputLength);
                EXPECT_EQ(outputPtr[outputIdx], expected)
                    << "Error at index (" << batchIdx << ',' << beam << ',' << i << ')';
            }
        }
    }
}

namespace
{
template <typename T>
void verifyTiling(std::vector<T> const& input, tr::ITensor const& outputTensor, tr::BufferManager& manager)
{
    auto outputHost = manager.copyFrom(outputTensor, tr::MemoryType::kCPU);
    auto outputPtr = tr::bufferCast<T>(*outputHost);

    auto const& shape = outputTensor.getShape();
    auto batchSize = static_cast<std::size_t>(shape.d[0]);
    auto beamWidth = static_cast<std::size_t>(shape.d[1]);
    auto inputLength = outputTensor.getSize() / batchSize / beamWidth;

    for (std::size_t b = 0; b < batchSize; ++b)
    {
        for (std::size_t beam = 0; beam < beamWidth; ++beam)
        {
            for (std::size_t i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = tc::flat_index2(b, i, inputLength);
                auto const outputIdx = tc::flat_index3(b, beam, i, beamWidth, inputLength);
                EXPECT_EQ(outputPtr[outputIdx], input[inputIdx])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}
} // namespace

TEST_F(RuntimeKernelTest, TileInt32)
{
    tr::SizeType32 const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    tr::SizeType32 const batchSize{4};
    auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
    auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);

    tr::kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);

    outputTensor->reshape(tr::ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileHalf)
{
    tr::SizeType32 const beamWidth{3};

    std::vector<half> const input{
        28524.F, 287.F, 5093.F, 12.F, 23316.F, 4881.F, 11.F, 30022.F, 263.F, 8776.F, 355.F, 257.F};
    tr::SizeType32 const batchSize{4};
    auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
    auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);

    tr::kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);

    outputTensor->reshape(tr::ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInt8Large)
{
    // Force synchronize to ensure that all memory allocations and de-allocations are completed before running the test,
    // as it uses a significant portion of the total memory on smaller devices and tends to cause OOMs.
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    std::int8_t constexpr value{3};
    tr::SizeType32 constexpr batchSize{1};
    tr::SizeType32 constexpr beamWidth{2};

    tr::SizeType32 const d2{2};
    auto const d3 = std::numeric_limits<int32_t>::max();
    auto const inputShape = tr::ITensor::makeShape({batchSize, d2, d3});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, d2, d3});

    // Ensure the test is not too memory hungry for the current device.
    auto const totalBytesAllocated = tr::ITensor::volume(inputShape) + tr::ITensor::volume(outputShape);
    auto const [_, totalDeviceMemory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
    auto constexpr memoryProportionThreshold{0.9};
    if (static_cast<double>(totalBytesAllocated) > (static_cast<double>(totalDeviceMemory) * memoryProportionThreshold))
    {
        GTEST_SKIP() << "Skipping test due to large memory allocation that could make test flaky.";
    }

    // Scope the allocated tensors to ensure they are de-allocated before the test ends.
    {
        auto inputTensor = mManager->gpu(inputShape, nvinfer1::DataType::kINT8);
        tr::kernels::invokeFill(*inputTensor, value, *mStream);
        mStream->synchronize();

        auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT8);
        tr::kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
        mStream->synchronize();

        auto bufferHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
        auto* bufferPtr = tr::bufferCast<std::int8_t>(*bufferHost);
        auto constexpr expected = value;
        for (std::size_t i = 0; i < bufferHost->getSize(); ++i)
        {
            EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
        }
    }
    mStream->synchronize();
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
}

namespace
{
void testCopyBatch(tr::SizeType64 stride, tr::BufferManager& manager, tr::CudaStream& stream)
{
    tr::SizeType32 constexpr rows{8};
    tr::SizeType32 constexpr numIndices{rows / 2};

    auto const bufferShape = tr::ITensor::makeShape({rows, stride});
    auto const indicesShape = tr::ITensor::makeShape({numIndices});
    auto srcBufferHost = tr::BufferManager::cpu(bufferShape, nvinfer1::DataType::kINT32);
    auto dstBufferDevice = manager.gpu(bufferShape, nvinfer1::DataType::kINT32);
    auto srcOffsets = tr::BufferManager::pinned(indicesShape, nvinfer1::DataType::kINT64);
    auto dstOffsets = tr::BufferManager::pinned(indicesShape, nvinfer1::DataType::kINT64);
    auto sizes = tr::BufferManager::pinned(indicesShape, nvinfer1::DataType::kINT64);
    tr::kernels::invokeFill(*dstBufferDevice, 0, stream);

    auto* srcBufferHostPtr = tr::bufferCast<std::int32_t>(*srcBufferHost);
    for (tr::SizeType32 row = 0; row < rows; ++row)
    {
        for (tr::SizeType32 ci = 0; ci < stride; ++ci)
        {
            auto const idx = row * stride + ci;
            srcBufferHostPtr[idx] = idx;
        }
    }

    auto* srcOffsetsPtr = tr::bufferCast<tr::SizeType64>(*srcOffsets);
    auto* dstOffsetsPtr = tr::bufferCast<tr::SizeType64>(*dstOffsets);
    auto* sizesPtr = tr::bufferCast<tr::SizeType64>(*sizes);
    for (tr::SizeType32 idx = 0; idx < numIndices; ++idx)
    {
        // Copy rows 0, 2, 4, etc to 0, 1, 2, 3...
        srcOffsetsPtr[idx] = 2 * idx * stride;
        dstOffsetsPtr[idx] = idx * stride;
        sizesPtr[idx] = stride;
    }

    auto srcBufferDevice = manager.copyFrom(*srcBufferHost, tr::MemoryType::kGPU);

    // TODO: test different dataSizes copy
    tr::kernels::invokeCopyBatch(*srcBufferDevice, *dstBufferDevice, *srcOffsets, *dstOffsets, *sizes, stride, stream);

    auto dstBufferHost = manager.copyFrom(*dstBufferDevice, tr::MemoryType::kCPU);

    auto* dstBufferHostPtr = tr::bufferCast<std::int32_t>(*dstBufferHost);
    for (tr::SizeType32 idx = 0; idx < rows; ++idx)
    {
        for (tr::SizeType64 ci = 0; ci < stride; ++ci)
        {
            if (idx < numIndices && ci < sizesPtr[idx])
            {
                auto const refIdx = srcOffsetsPtr[idx] + ci;
                auto const ref = srcBufferHostPtr[refIdx];

                auto const outIdx = dstOffsetsPtr[idx] + ci;
                auto const out = dstBufferHostPtr[outIdx];

                EXPECT_EQ(ref, out) << "Error at index row: " << idx << " column: " << ci << " for stride " << stride;
            }
            else
            {
                auto const outIdx = idx * stride + ci;
                auto const out = dstBufferHostPtr[outIdx];

                EXPECT_EQ(0, out) << "Error at index row: " << idx << " column: " << ci << " for stride " << stride;
            }
        }
    }
}
} // namespace

TEST_F(RuntimeKernelTest, CopyBatchStride64)
{
    testCopyBatch(64, *mManager, *mStream);
}

TEST_F(RuntimeKernelTest, CopyBatchStride5)
{
    testCopyBatch(5, *mManager, *mStream);
}
