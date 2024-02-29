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
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <NvInferRuntime.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

using TensorPtr = std::shared_ptr<ITensor>;
using BufferPtr = std::shared_ptr<IBuffer>;

class RuntimeKernelTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
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

namespace
{
template <typename T>
void testFill(IBuffer& buffer, BufferManager& manager, CudaStream& stream)
{
    T constexpr value{3};
    kernels::invokeFill(buffer, value, stream);
    auto bufferHost = manager.copyFrom(buffer, MemoryType::kCPU);
    auto bufferPtr = bufferCast<T>(*bufferHost);
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
    for (auto size : {123llu, 1025llu, 1llu << 32})
    {
        auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT8);
        testFill<std::int8_t>(*buffer, *mManager, *mStream);
    }
}

TEST_F(RuntimeKernelTest, FillTensorInt8)
{
    for (auto size : {123, 1025, std::numeric_limits<SizeType>::max()})
    {
        auto tensor = mManager->gpu(ITensor::makeShape({size, 2}), nvinfer1::DataType::kINT8);
        testFill<std::int8_t>(*tensor, *mManager, *mStream);
    }
}

namespace
{
void testAdd(IBuffer& buffer, BufferManager& manager, CudaStream& stream)
{
    SizeType constexpr value{3};
    manager.setZero(buffer);
    kernels::invokeAdd(buffer, value, stream);
    kernels::invokeAdd(buffer, value, stream);
    auto bufferHost = manager.copyFrom(buffer, MemoryType::kCPU);
    auto bufferPtr = bufferCast<SizeType>(*bufferHost);
    auto constexpr expected = 2 * value;

    auto anyMismatch = false;
    for (std::size_t i = 0; i < buffer.getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
        anyMismatch |= bufferPtr[i] != expected;
    }
    ASSERT_FALSE(anyMismatch);
}
} // namespace

TEST_F(RuntimeKernelTest, AddBufferInt32)
{
    for (auto size : {123, 1025})
    {
        auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT32);
        testAdd(*buffer, *mManager, *mStream);
    }
}

TEST_F(RuntimeKernelTest, AddTensorInt32)
{
    for (auto size : {123, 1025})
    {
        auto tensor = mManager->gpu(ITensor::makeShape({size, size}), nvinfer1::DataType::kINT32);
        testAdd(*tensor, *mManager, *mStream);
    }
}

namespace
{
void testReduce(IBuffer& buffer, BufferManager& manager, CudaStream& stream)
{
    auto output = manager.gpu(1, nvinfer1::DataType::kINT32);
    manager.setZero(*output);

    SizeType constexpr value{3};
    kernels::invokeFill(buffer, value, stream);
    kernels::reduce(*output, buffer, stream);
    auto outputHost = manager.copyFrom(*output, MemoryType::kCPU);
    auto outputPtr = bufferCast<SizeType>(*outputHost);
    auto const expected = buffer.getSize() * value;

    EXPECT_EQ(*outputPtr, expected);
}
} // namespace

TEST_F(RuntimeKernelTest, ReduceBufferInt32)
{
    for (auto size : {123, 1025})
    {
        auto buffer = mManager->gpu(size, nvinfer1::DataType::kINT32);
        testReduce(*buffer, *mManager, *mStream);
    }
}

TEST_F(RuntimeKernelTest, Transpose)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    auto input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);

    kernels::invokeTranspose(*output, *input, *mStream);

    auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
    auto outputHostData = bufferCast<SizeType>(*outputHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType i = 0; i < rowSize; ++i)
        {
            auto const inputIndex = tc::flat_index2(b, i, rowSize);
            auto const outputIndex = tc::flat_index2(i, b, batchSize);
            EXPECT_EQ(outputHostData[outputIndex], inputHost[inputIndex]) << "Error at index (" << b << ',' << i << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithOutputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (SizeType sliceId = 0; sliceId < batchSize; ++sliceId)
    {
        auto inputView = ITensor::slice(input, sliceId, 1);
        kernels::invokeTransposeWithOutputOffset(*output, *inputView, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
        auto outputHostData = bufferCast<SizeType>(*outputHost);

        for (SizeType b = 0; b < batchSize; ++b)
        {
            for (SizeType i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(b, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, b, batchSize);
                auto expected = b <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithInputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    SizeType const batchSize{4};
    auto const rowSize = static_cast<SizeType>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, ITensor::makeShape({batchSize, rowSize}), MemoryType::kGPU);

    TensorPtr output = mManager->gpu(ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (SizeType sliceId = 0; sliceId < rowSize; ++sliceId)
    {
        auto outputView = ITensor::slice(output, sliceId, 1);
        kernels::invokeTransposeWithInputOffset(*outputView, *input, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, MemoryType::kCPU);
        auto outputHostData = bufferCast<SizeType>(*outputHost);

        for (SizeType b = 0; b < batchSize; ++b)
        {
            for (SizeType i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(b, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, b, batchSize);
                auto expected = i <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, BuildTokenMask)
{
    SizeType constexpr batchSize{7};
    std::vector<SizeType> inputLengthsVec(batchSize);
    std::iota(inputLengthsVec.begin(), inputLengthsVec.end(), 3);
    auto const maxInputLength = *std::max_element(inputLengthsVec.begin(), inputLengthsVec.end());

    SizeType constexpr maxNewTokens{1};
    auto const maxSeqLength = maxInputLength + maxNewTokens;

    TensorPtr inputLengths = mManager->copyFrom(inputLengthsVec, ITensor::makeShape({batchSize, 1}), MemoryType::kGPU);

    TensorPtr tokenMask = mManager->gpu(ITensor::makeShape({batchSize, maxSeqLength}), nvinfer1::DataType::kINT32);
    kernels::invokeBuildTokenMask(*tokenMask, *inputLengths, maxInputLength, *mStream);

    std::vector<SizeType> tokenMaskVec(tokenMask->getSize());
    mManager->copy(*tokenMask, tokenMaskVec.data());

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < maxSeqLength; ++j)
        {
            auto const index = i * maxSeqLength + j;
            if (j < inputLengthsVec[i])
                EXPECT_EQ(tokenMaskVec[index], 0) << "tokenMask should be 0 up to inputLengths[i]";
            else if (j < maxInputLength)
                EXPECT_EQ(tokenMaskVec[index], 1) << "tokenMask should be 1 up to maxInputLength";
            else
                EXPECT_EQ(tokenMaskVec[index], 0) << "tokenMask should be 0 after maxInputLength";
        }
    }
}

TEST_F(RuntimeKernelTest, BuildAttentionMask)
{
    SizeType constexpr batchSize{1};
    SizeType constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<SizeType>(input.size());

    TensorPtr inputIds = mManager->copyFrom(input, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, MemoryType::kGPU);
    kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    std::vector<SizeType> attentionMaskVec(attentionMask->getSize());
    mManager->copy(*attentionMask, attentionMaskVec.data());

    std::vector<std::int32_t> attentionMaskHost(input);
    std::for_each(attentionMaskHost.begin(), attentionMaskHost.end(), [padId](auto& x) { x = x != padId; });

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < maxInputLength; ++j)
        {
            auto const index = i * maxInputLength + j;
            EXPECT_EQ(attentionMaskVec[index], attentionMaskHost[index]) << "Error at index (" << i << ',' << j << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, ExtendAttentionMask)
{
    SizeType constexpr batchSize{1};
    SizeType constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<SizeType>(input.size());

    TensorPtr inputIds = mManager->copyFrom(input, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, MemoryType::kGPU);
    kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    auto attentionMaskHost = mManager->copyFrom(*attentionMask, MemoryType::kCPU);
    auto const* attentionMaskData = reinterpret_cast<SizeType const*>(attentionMaskHost->data());
    auto const shape = attentionMask->getShape();
    auto const nbInputs = shape.d[0];
    auto const oldLength = shape.d[1];
    auto const newLength = oldLength + 1;
    auto const newShape = ITensor::makeShape({nbInputs, newLength});
    std::vector<SizeType> attentionMaskVec(ITensor::volume(newShape));
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(attentionMaskData + i * oldLength, attentionMaskData + (i + 1) * oldLength,
            std::begin(attentionMaskVec) + i * newLength);
        attentionMaskVec[(i + 1) * newLength - 1] = 1;
    }

    TensorPtr newAttentionMask = mManager->gpu(newShape, nvinfer1::DataType::kINT32);
    mManager->setZero(*newAttentionMask);
    kernels::invokeExtendAttentionMask(*newAttentionMask, *attentionMask, *mStream);

    std::vector<SizeType> newAttentionMaskVec(newAttentionMask->getSize());
    mManager->copy(*newAttentionMask, newAttentionMaskVec.data());

    for (SizeType i = 0; i < batchSize; ++i)
    {
        for (SizeType j = 0; j < oldLength; ++j)
        {
            auto const oldIndex = i * oldLength + j;
            auto const newIndex = i * newLength + j;
            EXPECT_EQ(attentionMaskVec[oldIndex], newAttentionMaskVec[newIndex])
                << "Error at index (" << i << ',' << j << ')';
        }
        EXPECT_EQ(attentionMaskVec[(i + 1) * newLength - 1], 1) << "Error at index (" << i << ',' << (-1) << ')';
    }
}

TEST_F(RuntimeKernelTest, CopyInputToOutput)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    auto const beamWidth = 5;
    SizeType constexpr maxNewTokens{3};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<std::int32_t> inputsHost(batchSize * maxInputLength);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.end(), inputsHost.begin() + i * maxInputLength);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32);

    kernels::invokeCopyInputToOutput(*outputIds, *inputIds, *inputLengths, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i]) << "Error at index (" << b << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId) << "Error at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, CopyPackedInputToOutput)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    SizeType constexpr maxNewTokens{3};
    auto const beamWidth = 5;
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    std::vector<SizeType> inputOffsetsHost(batchSize + 1);
    tc::stl_utils::inclusiveScan(inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
    auto const totalInputSize = inputOffsetsHost.back();

    std::vector<std::int32_t> inputsHost(totalInputSize);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.begin() + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32);

    auto inputOffsets = std::shared_ptr(mManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32));
    mManager->setZero(*inputOffsets);
    kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, *mManager, *mStream);
    auto inputOffsetsHost2 = mManager->copyFrom(*inputOffsets, MemoryType::kCPU);

    for (std::size_t b = 0; b < inputOffsetsHost.size(); ++b)
    {
        EXPECT_EQ(inputOffsetsHost[b], inputOffsetsHost[b]) << "Error at index " << b;
    }

    kernels::invokeCopyPackedInputToOutput(*outputIds, *inputIds, *inputOffsets, maxInputLength, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId)
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, CopyInputToOutputTransposed)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    auto const beamWidth = 5;
    SizeType constexpr maxNewTokens{3};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<std::int32_t> inputsHost(batchSize * maxInputLength);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.end(), inputsHost.begin() + i * maxInputLength);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({batchSize, maxInputLength}), MemoryType::kGPU);

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    kernels::invokeCopyInputToOutputTransposed(*outputIds, *inputIds, *inputLengths, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    std::cout << *outputIdsHost;

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i]) << "Error at index (" << b << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId) << "Error at index (" << b << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, CopyPackedInputToOutputTransposed)
{
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    auto const maxInputLength = static_cast<SizeType>(input.size());
    auto const batchSize = maxInputLength;
    SizeType constexpr maxNewTokens{3};
    auto const beamWidth = 5;
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    SizeType constexpr padId{50256};

    std::vector<SizeType> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, ITensor::makeShape({batchSize}), MemoryType::kGPU);

    std::vector<SizeType> inputOffsetsHost(batchSize + 1);
    tc::stl_utils::inclusiveScan(inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
    auto const totalInputSize = inputOffsetsHost.back();

    std::vector<std::int32_t> inputsHost(totalInputSize);
    for (SizeType i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.begin() + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
    }
    auto inputIds = mManager->copyFrom(inputsHost, ITensor::makeShape({1, totalInputSize}), MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    auto inputOffsets = std::shared_ptr(mManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32));
    mManager->setZero(*inputOffsets);
    kernels::invokeInclusiveSum(*ITensor::slice(inputOffsets, 1), *inputLengths, *mManager, *mStream);
    auto inputOffsetsHost2 = mManager->copyFrom(*inputOffsets, MemoryType::kCPU);

    for (std::size_t b = 0; b < inputOffsetsHost.size(); ++b)
    {
        EXPECT_EQ(inputOffsetsHost[b], inputOffsetsHost[b]) << "Error at index " << b;
    }

    kernels::invokeCopyPackedInputToOutputTransposed(
        *outputIds, *inputIds, *inputOffsets, maxInputLength, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, MemoryType::kCPU);
    auto outputIdsHostData = bufferCast<SizeType>(*outputIdsHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
            for (SizeType i = inputLengthsHost[b]; i < maxInputLength; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], padId)
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, ScatterInt32)
{
    SizeType const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
    mManager->setZero(*outputTensor);

    kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto outputPtr = bufferCast<SizeType>(*outputHost);

    std::cout << *outputHost;

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = tc::flat_index2(b, i, inputLength);
                auto const expected = beam == 0 ? input[inputIdx] : 0;
                auto const outputIdx = tc::flat_index3(b, beam, i, beamWidth, inputLength);
                EXPECT_EQ(outputPtr[outputIdx], expected) << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, SplitTransposed)
{
    SizeType const split{2};
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    std::vector<std::int32_t> const output{28524, 5093, 23316, 11, 263, 355, 287, 12, 4881, 30022, 8776, 257};
    std::vector<std::int32_t> const output2{28524, 287, 23316, 4881, 263, 8776, 5093, 12, 11, 30022, 355, 257};

    {
        SizeType const batchSize{6};
        auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
        auto const inputShape = ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = ITensor::makeShape({split, batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
        auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensor);

        kernels::splitTransposed(*outputTensor, *inputTensor, split, *mStream);
        auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
        auto outputPtr = bufferCast<SizeType>(*outputHost);
        cudaError_t cudaerr = cudaDeviceSynchronize();

        for (SizeType i = 0; i < static_cast<SizeType>(input.size()); ++i)
        {
            EXPECT_EQ(outputPtr[i], output[i]);
        }
    }

    {
        SizeType const batchSize{3};
        auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
        auto const inputShape = ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = ITensor::makeShape({split, batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
        auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensor);

        kernels::splitTransposed(*outputTensor, *inputTensor, split, *mStream);
        auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
        auto outputPtr = bufferCast<SizeType>(*outputHost);
        cudaError_t cudaerr = cudaDeviceSynchronize();

        for (SizeType i = 0; i < static_cast<SizeType>(input.size()); ++i)
        {
            EXPECT_EQ(outputPtr[i], output2[i]);
        }
    }
}

TEST_F(RuntimeKernelTest, ScatterHalf)
{
    SizeType const beamWidth{3};

    std::vector<half> const input{
        28524.f, 287.f, 5093.f, 12.f, 23316.f, 4881.f, 11.f, 30022.f, 263.f, 8776.f, 355.f, 257.f};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);
    mManager->setZero(*outputTensor);

    kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto outputPtr = bufferCast<half>(*outputHost);

    for (SizeType b = 0; b < batchSize; ++b)
    {
        for (SizeType beam = 0; beam < beamWidth; ++beam)
        {
            for (SizeType i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = tc::flat_index2(b, i, inputLength);
                auto const expected = beam == 0 ? input[inputIdx] : half(0.f);
                auto const outputIdx = tc::flat_index3(b, beam, i, beamWidth, inputLength);
                EXPECT_EQ(outputPtr[outputIdx], expected) << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
        }
    }
}

namespace
{
template <typename T>
void verifyTiling(std::vector<T> const& input, ITensor const& outputTensor, BufferManager& manager)
{
    auto outputHost = manager.copyFrom(outputTensor, MemoryType::kCPU);
    auto outputPtr = bufferCast<T>(*outputHost);

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
    SizeType const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);

    kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);

    outputTensor->reshape(ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileHalf)
{
    SizeType const beamWidth{3};

    std::vector<half> const input{
        28524.f, 287.f, 5093.f, 12.f, 23316.f, 4881.f, 11.f, 30022.f, 263.f, 8776.f, 355.f, 257.f};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);

    kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);

    outputTensor->reshape(ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInplaceInt32)
{
    SizeType const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);

    kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    outputTensor->reshape(ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInplaceHalf)
{
    SizeType const beamWidth{3};

    std::vector<half> const input{
        28524.f, 287.f, 5093.f, 12.f, 23316.f, 4881.f, 11.f, 30022.f, 263.f, 8776.f, 355.f, 257.f};
    SizeType const batchSize{4};
    auto const inputLength = static_cast<SizeType>(input.size() / batchSize);
    auto const inputShape = ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kHALF);

    kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    outputTensor->reshape(ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInt8Large)
{
    std::int8_t constexpr value{3};
    SizeType constexpr batchSize{1};
    SizeType constexpr beamWidth{2};

    SizeType const d2{2};
    auto const d3 = std::numeric_limits<SizeType>::max();
    auto const inputShape = ITensor::makeShape({batchSize, d2, d3});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, d2, d3});

    auto inputTensor = mManager->gpu(inputShape, nvinfer1::DataType::kINT8);
    kernels::invokeFill(*inputTensor, value, *mStream);
    mStream->synchronize();

    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT8);
    kernels::tileTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    mStream->synchronize();

    auto bufferHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto bufferPtr = bufferCast<std::int8_t>(*bufferHost);
    auto constexpr expected = value;
    for (std::size_t i = 0; i < bufferHost->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
    }
}

TEST_F(RuntimeKernelTest, TileInplaceInt8Large)
{
    std::int8_t constexpr value{3};
    SizeType constexpr batchSize{1};
    SizeType constexpr beamWidth{2};

    SizeType const d2{2};
    auto const d3 = std::numeric_limits<SizeType>::max();
    auto const inputShape = ITensor::makeShape({batchSize, d2, d3});
    auto const outputShape = ITensor::makeShape({batchSize * beamWidth, d2, d3});

    auto inputTensor = mManager->gpu(inputShape, nvinfer1::DataType::kINT8);
    kernels::invokeFill(*inputTensor, value, *mStream);

    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT8);
    kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    auto bufferHost = mManager->copyFrom(*outputTensor, MemoryType::kCPU);
    auto bufferPtr = bufferCast<std::int8_t>(*bufferHost);
    auto constexpr expected = value;
    for (std::size_t i = 0; i < bufferHost->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
    }
}

namespace
{
void testCopyBatch(SizeType stride, BufferManager& manager, CudaStream& stream)
{
    SizeType constexpr rows{8};
    SizeType constexpr numIndices{rows / 2};

    auto const bufferShape = ITensor::makeShape({rows, stride});
    auto const indicesShape = ITensor::makeShape({numIndices});
    auto srcBufferHost = manager.cpu(bufferShape, nvinfer1::DataType::kINT32);
    auto dstBufferDevice = manager.gpu(bufferShape, nvinfer1::DataType::kINT32);
    auto srcOffsets = manager.pinned(indicesShape, nvinfer1::DataType::kINT32);
    auto dstOffsets = manager.pinned(indicesShape, nvinfer1::DataType::kINT32);
    auto sizes = manager.pinned(indicesShape, nvinfer1::DataType::kINT32);
    kernels::invokeFill(*dstBufferDevice, 0, stream);

    auto srcBufferHostPtr = bufferCast<std::int32_t>(*srcBufferHost);
    for (SizeType row = 0; row < rows; ++row)
    {
        for (SizeType ci = 0; ci < stride; ++ci)
        {
            const auto idx = row * stride + ci;
            srcBufferHostPtr[idx] = idx;
        }
    }

    auto srcOffsetsPtr = bufferCast<std::int32_t>(*srcOffsets);
    auto dstOffsetsPtr = bufferCast<std::int32_t>(*dstOffsets);
    auto sizesPtr = bufferCast<std::int32_t>(*sizes);
    for (SizeType idx = 0; idx < numIndices; ++idx)
    {
        // Copy rows 0, 2, 4, etc to 0, 1, 2, 3...
        srcOffsetsPtr[idx] = 2 * idx * stride;
        dstOffsetsPtr[idx] = idx * stride;
        sizesPtr[idx] = stride;
    }

    auto srcBufferDevice = manager.copyFrom(*srcBufferHost, MemoryType::kGPU);

    // TODO(nkorobov): test different dataSizes copy
    kernels::invokeCopyBatch(*srcBufferDevice, *dstBufferDevice, *srcOffsets, *dstOffsets, *sizes, stride, stream);

    auto dstBufferHost = manager.copyFrom(*dstBufferDevice, MemoryType::kCPU);

    auto dstBufferHostPtr = bufferCast<std::int32_t>(*dstBufferHost);
    for (SizeType idx = 0; idx < rows; ++idx)
    {
        for (SizeType ci = 0; ci < stride; ++ci)
        {
            if (idx < numIndices && ci < sizesPtr[idx])
            {
                const auto refIdx = srcOffsetsPtr[idx] + ci;
                const auto ref = srcBufferHostPtr[refIdx];

                const auto outIdx = dstOffsetsPtr[idx] + ci;
                const auto out = dstBufferHostPtr[outIdx];

                EXPECT_EQ(ref, out) << "Error at index row: " << idx << " column: " << ci << " for stride " << stride;
            }
            else
            {
                const auto outIdx = idx * stride + ci;
                const auto out = dstBufferHostPtr[outIdx];

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
