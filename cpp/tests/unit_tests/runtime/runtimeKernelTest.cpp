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
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include <NvInferRuntime.h>

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

namespace
{
void testAdd(tr::IBuffer& buffer, tr::BufferManager& manager, tr::CudaStream& stream)
{
    tr::SizeType32 constexpr value{3};
    manager.setZero(buffer);
    tr::kernels::invokeAdd(buffer, value, stream);
    tr::kernels::invokeAdd(buffer, value, stream);
    auto bufferHost = manager.copyFrom(buffer, tr::MemoryType::kCPU);
    auto bufferPtr = tr::bufferCast<tr::SizeType32>(*bufferHost);
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
        auto tensor = mManager->gpu(tr::ITensor::makeShape({size, size}), nvinfer1::DataType::kINT32);
        testAdd(*tensor, *mManager, *mStream);
    }
}

namespace
{
void testReduce(tr::IBuffer& buffer, tr::BufferManager& manager, tr::CudaStream& stream)
{
    auto output = manager.gpu(1, nvinfer1::DataType::kINT32);
    manager.setZero(*output);

    tr::SizeType32 constexpr value{3};
    tr::kernels::invokeFill(buffer, value, stream);
    tr::kernels::reduce(*output, buffer, stream);
    auto outputHost = manager.copyFrom(*output, tr::MemoryType::kCPU);
    auto* outputPtr = tr::bufferCast<tr::SizeType32>(*outputHost);
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

    tr::SizeType32 const batchSize{4};
    auto const rowSize = static_cast<tr::SizeType32>(inputHost.size()) / batchSize;

    auto input = mManager->copyFrom(inputHost, tr::ITensor::makeShape({batchSize, rowSize}), tr::MemoryType::kGPU);

    TensorPtr output = mManager->gpu(tr::ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);

    tr::kernels::invokeTranspose(*output, *input, *mStream);

    auto outputHost = mManager->copyFrom(*output, tr::MemoryType::kCPU);
    auto* outputHostData = tr::bufferCast<tr::SizeType32>(*outputHost);

    for (tr::SizeType32 batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (tr::SizeType32 i = 0; i < rowSize; ++i)
        {
            auto const inputIndex = tc::flat_index2(batchIdx, i, rowSize);
            auto const outputIndex = tc::flat_index2(i, batchIdx, batchSize);
            EXPECT_EQ(outputHostData[outputIndex], inputHost[inputIndex])
                << "Error at index (" << batchIdx << ',' << i << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithOutputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    tr::SizeType32 const batchSize{4};
    auto const rowSize = static_cast<tr::SizeType32>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, tr::ITensor::makeShape({batchSize, rowSize}), tr::MemoryType::kGPU);

    TensorPtr output = mManager->gpu(tr::ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (tr::SizeType32 sliceId = 0; sliceId < batchSize; ++sliceId)
    {
        auto inputView = tr::ITensor::slice(input, sliceId, 1);
        tr::kernels::invokeTransposeWithOutputOffset(*output, *inputView, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, tr::MemoryType::kCPU);
        auto* outputHostData = tr::bufferCast<tr::SizeType32>(*outputHost);

        for (tr::SizeType32 batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (tr::SizeType32 i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(batchIdx, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, batchIdx, batchSize);
                auto expected = batchIdx <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << batchIdx << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, TransposeWithInputOffset)
{
    std::vector<std::int32_t> const inputHost{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};

    tr::SizeType32 const batchSize{4};
    auto const rowSize = static_cast<tr::SizeType32>(inputHost.size()) / batchSize;

    TensorPtr input = mManager->copyFrom(inputHost, tr::ITensor::makeShape({batchSize, rowSize}), tr::MemoryType::kGPU);

    TensorPtr output = mManager->gpu(tr::ITensor::makeShape({rowSize, batchSize}), nvinfer1::DataType::kINT32);
    mManager->setZero(*output);

    for (tr::SizeType32 sliceId = 0; sliceId < rowSize; ++sliceId)
    {
        auto outputView = tr::ITensor::slice(output, sliceId, 1);
        tr::kernels::invokeTransposeWithInputOffset(*outputView, *input, sliceId, *mStream);

        auto outputHost = mManager->copyFrom(*output, tr::MemoryType::kCPU);
        auto* outputHostData = tr::bufferCast<tr::SizeType32>(*outputHost);

        for (tr::SizeType32 batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (tr::SizeType32 i = 0; i < rowSize; ++i)
            {
                auto const inputIndex = tc::flat_index2(batchIdx, i, rowSize);
                auto const outputIndex = tc::flat_index2(i, batchIdx, batchSize);
                auto expected = i <= sliceId ? inputHost[inputIndex] : 0;
                EXPECT_EQ(outputHostData[outputIndex], expected)
                    << "Error after slice " << sliceId << " at index (" << batchIdx << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, InclusiveScan)
{
    auto constexpr inputLength = 10000;
    auto constexpr expectedOutputLength = 10000;
    std::vector<tr::SizeType32> inputHost(inputLength);
    std::vector<tr::SizeType32> expectedOutput(expectedOutputLength);
    tc::stl_utils::inclusiveScan(inputHost.begin(), inputHost.end(), expectedOutput.begin());

    auto input = mManager->copyFrom(inputHost, tr::MemoryType::kGPU);
    auto output = mManager->gpu(input->getSize(), input->getDataType());

    tr::kernels::invokeInclusiveSum(*output, *input, *mManager, *mStream);

    auto outputHost = mManager->copyFrom(*output, tr::MemoryType::kCPU);
    auto* outputHostPtr = tr::bufferCast<tr::SizeType32>(*outputHost);

    for (std::size_t i = 0; i < expectedOutput.size(); ++i)
    {
        EXPECT_EQ(expectedOutput[i], outputHostPtr[i]) << "Error at index " << i;
    }
}

TEST_F(RuntimeKernelTest, InclusiveScanTmp)
{
    auto constexpr inputLength = 10000;
    auto constexpr expectedOutputLength = 10000;
    std::vector<tr::SizeType32> inputHost(inputLength);
    std::vector<tr::SizeType32> expectedOutput(expectedOutputLength);
    tc::stl_utils::inclusiveScan(inputHost.begin(), inputHost.end(), expectedOutput.begin());

    auto input = mManager->copyFrom(inputHost, tr::MemoryType::kGPU);
    auto output = mManager->gpu(input->getSize(), input->getDataType());
    auto tmp = mManager->emptyBuffer(tr::MemoryType::kGPU);

    tr::kernels::invokeInclusiveSum(*output, *tmp, *input, *mStream);

    auto outputHost = mManager->copyFrom(*output, tr::MemoryType::kCPU);
    auto* outputHostPtr = tr::bufferCast<tr::SizeType32>(*outputHost);

    for (std::size_t i = 0; i < expectedOutput.size(); ++i)
    {
        EXPECT_EQ(expectedOutput[i], outputHostPtr[i]) << "Error at index " << i;
    }
}

TEST_F(RuntimeKernelTest, BuildTokenMask)
{
    tr::SizeType32 constexpr batchSize{7};
    std::vector<tr::SizeType32> inputLengthsVec(batchSize);
    std::iota(inputLengthsVec.begin(), inputLengthsVec.end(), 3);
    auto const maxInputLength = *std::max_element(inputLengthsVec.begin(), inputLengthsVec.end());

    tr::SizeType32 constexpr maxNewTokens{1};
    auto const maxSeqLength = maxInputLength + maxNewTokens;

    TensorPtr inputLengths
        = mManager->copyFrom(inputLengthsVec, tr::ITensor::makeShape({batchSize, 1}), tr::MemoryType::kGPU);

    TensorPtr tokenMask = mManager->gpu(tr::ITensor::makeShape({batchSize, maxSeqLength}), nvinfer1::DataType::kINT32);
    tr::kernels::invokeBuildTokenMask(*tokenMask, *inputLengths, maxInputLength, *mStream);

    std::vector<tr::SizeType32> tokenMaskVec(tokenMask->getSize());
    mManager->copy(*tokenMask, tokenMaskVec.data());

    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        for (tr::SizeType32 j = 0; j < maxSeqLength; ++j)
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
    tr::SizeType32 constexpr batchSize{1};
    tr::SizeType32 constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());

    TensorPtr inputIds
        = mManager->copyFrom(input, tr::ITensor::makeShape({batchSize, maxInputLength}), tr::MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, tr::MemoryType::kGPU);
    tr::kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    std::vector<tr::SizeType32> attentionMaskVec(attentionMask->getSize());
    mManager->copy(*attentionMask, attentionMaskVec.data());

    std::vector<std::int32_t> attentionMaskHost(input);
    std::for_each(attentionMaskHost.begin(), attentionMaskHost.end(), [padId](auto& x) { x = x != padId; });

    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        for (tr::SizeType32 j = 0; j < maxInputLength; ++j)
        {
            auto const index = i * maxInputLength + j;
            EXPECT_EQ(attentionMaskVec[index], attentionMaskHost[index]) << "Error at index (" << i << ',' << j << ')';
        }
    }
}

TEST_F(RuntimeKernelTest, ExtendAttentionMask)
{
    tr::SizeType32 constexpr batchSize{1};
    tr::SizeType32 constexpr padId{50256};
    std::vector<std::int32_t> const input{padId, 287, 5093, 12, 50256, padId, 11, 30022, 263, 8776, 355, padId};
    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());

    TensorPtr inputIds
        = mManager->copyFrom(input, tr::ITensor::makeShape({batchSize, maxInputLength}), tr::MemoryType::kGPU);
    TensorPtr attentionMask = mManager->copyFrom(*inputIds, tr::MemoryType::kGPU);
    tr::kernels::invokeBuildAttentionMask(*attentionMask, padId, *mStream);

    auto attentionMaskHost = mManager->copyFrom(*attentionMask, tr::MemoryType::kCPU);
    auto const* attentionMaskData = reinterpret_cast<tr::SizeType32 const*>(attentionMaskHost->data());
    auto const shape = attentionMask->getShape();
    auto const nbInputs = shape.d[0];
    auto const oldLength = shape.d[1];
    auto const newLength = oldLength + 1;
    auto const newShape = tr::ITensor::makeShape({nbInputs, newLength});
    std::vector<tr::SizeType32> attentionMaskVec(tr::ITensor::volume(newShape));
    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        std::copy(attentionMaskData + i * oldLength, attentionMaskData + (i + 1) * oldLength,
            std::begin(attentionMaskVec) + i * newLength);
        attentionMaskVec[(i + 1) * newLength - 1] = 1;
    }

    TensorPtr newAttentionMask = mManager->gpu(newShape, nvinfer1::DataType::kINT32);
    mManager->setZero(*newAttentionMask);
    tr::kernels::invokeExtendAttentionMask(*newAttentionMask, *attentionMask, *mStream);

    std::vector<tr::SizeType32> newAttentionMaskVec(newAttentionMask->getSize());
    mManager->copy(*newAttentionMask, newAttentionMaskVec.data());

    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        for (tr::SizeType32 j = 0; j < oldLength; ++j)
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

    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());
    auto const batchSize = maxInputLength;
    auto const beamWidth = 5;
    tr::SizeType32 constexpr maxNewTokens{3};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    tr::SizeType32 constexpr padId{50256};

    std::vector<std::int32_t> inputsHost(batchSize * maxInputLength);
    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.end(), inputsHost.begin() + i * maxInputLength);
    }
    auto inputIds
        = mManager->copyFrom(inputsHost, tr::ITensor::makeShape({batchSize, maxInputLength}), tr::MemoryType::kGPU);

    std::vector<tr::SizeType32> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, tr::ITensor::makeShape({batchSize}), tr::MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(tr::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32);

    tr::kernels::invokeCopyInputToOutput(*outputIds, *inputIds, *inputLengths, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, tr::MemoryType::kCPU);
    auto* outputIdsHostData = tr::bufferCast<tr::SizeType32>(*outputIdsHost);

    for (tr::SizeType32 b = 0; b < batchSize; ++b)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i]) << "Error at index (" << b << ',' << i << ')';
            }
            for (tr::SizeType32 i = inputLengthsHost[b]; i < maxInputLength; ++i)
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

    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());
    auto const batchSize = maxInputLength;
    tr::SizeType32 constexpr maxNewTokens{3};
    auto const beamWidth = 5;
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    tr::SizeType32 constexpr padId{50256};

    std::vector<tr::SizeType32> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, tr::ITensor::makeShape({batchSize}), tr::MemoryType::kGPU);

    std::vector<tr::SizeType32> inputOffsetsHost(batchSize + 1);
    tc::stl_utils::inclusiveScan(inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
    auto const totalInputSize = inputOffsetsHost.back();

    std::vector<std::int32_t> inputsHost(totalInputSize);
    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.begin() + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
    }
    auto inputIds = mManager->copyFrom(inputsHost, tr::ITensor::makeShape({1, totalInputSize}), tr::MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(tr::ITensor::makeShape({batchSize, beamWidth, maxSeqLength}), nvinfer1::DataType::kINT32);

    auto inputOffsets
        = std::shared_ptr(mManager->gpu(tr::ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32));
    mManager->setZero(*inputOffsets);
    tr::kernels::invokeInclusiveSum(*tr::ITensor::slice(inputOffsets, 1), *inputLengths, *mManager, *mStream);
    auto inputOffsetsHost2 = mManager->copyFrom(*inputOffsets, tr::MemoryType::kCPU);
    auto* inputOffsetsHost2Ptr = tr::bufferCast<tr::SizeType32>(*inputOffsetsHost2);

    for (std::size_t b = 0; b < inputOffsetsHost.size(); ++b)
    {
        EXPECT_EQ(inputOffsetsHost[b], inputOffsetsHost2Ptr[b]) << "Error at index " << b;
    }

    tr::kernels::invokeCopyPackedInputToOutput(*outputIds, *inputIds, *inputOffsets, maxInputLength, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, tr::MemoryType::kCPU);
    auto* outputIdsHostData = tr::bufferCast<tr::SizeType32>(*outputIdsHost);

    for (tr::SizeType32 b = 0; b < batchSize; ++b)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(b, beam, i, beamWidth, maxSeqLength);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
            for (tr::SizeType32 i = inputLengthsHost[b]; i < maxInputLength; ++i)
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

    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());
    auto const batchSize = maxInputLength;
    auto const beamWidth = 5;
    tr::SizeType32 constexpr maxNewTokens{3};
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    tr::SizeType32 constexpr padId{50256};

    std::vector<std::int32_t> inputsHost(batchSize * maxInputLength);
    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.end(), inputsHost.begin() + i * maxInputLength);
    }
    auto inputIds
        = mManager->copyFrom(inputsHost, tr::ITensor::makeShape({batchSize, maxInputLength}), tr::MemoryType::kGPU);

    std::vector<tr::SizeType32> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, tr::ITensor::makeShape({batchSize}), tr::MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(tr::ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    tr::kernels::invokeCopyInputToOutputTransposed(*outputIds, *inputIds, *inputLengths, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, tr::MemoryType::kCPU);
    auto* outputIdsHostData = tr::bufferCast<tr::SizeType32>(*outputIdsHost);

    std::cout << *outputIdsHost;

    for (tr::SizeType32 b = 0; b < batchSize; ++b)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i]) << "Error at index (" << b << ',' << i << ')';
            }
            for (tr::SizeType32 i = inputLengthsHost[b]; i < maxInputLength; ++i)
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

    auto const maxInputLength = static_cast<tr::SizeType32>(input.size());
    auto const batchSize = maxInputLength;
    tr::SizeType32 constexpr maxNewTokens{3};
    auto const beamWidth = 5;
    auto const maxSeqLength = maxInputLength + maxNewTokens;
    tr::SizeType32 constexpr padId{50256};

    std::vector<tr::SizeType32> inputLengthsHost(batchSize);
    std::iota(inputLengthsHost.begin(), inputLengthsHost.end(), 1);
    auto inputLengths = mManager->copyFrom(inputLengthsHost, tr::ITensor::makeShape({batchSize}), tr::MemoryType::kGPU);

    std::vector<tr::SizeType32> inputOffsetsHost(batchSize + 1);
    tc::stl_utils::inclusiveScan(inputLengthsHost.begin(), inputLengthsHost.end(), inputOffsetsHost.begin() + 1);
    auto const totalInputSize = inputOffsetsHost.back();

    std::vector<std::int32_t> inputsHost(totalInputSize);
    for (tr::SizeType32 i = 0; i < batchSize; ++i)
    {
        std::copy(input.begin(), input.begin() + inputLengthsHost[i], inputsHost.begin() + inputOffsetsHost[i]);
    }
    auto inputIds = mManager->copyFrom(inputsHost, tr::ITensor::makeShape({1, totalInputSize}), tr::MemoryType::kGPU);

    TensorPtr outputIds
        = mManager->gpu(tr::ITensor::makeShape({maxSeqLength, batchSize, beamWidth}), nvinfer1::DataType::kINT32);

    auto inputOffsets
        = std::shared_ptr(mManager->gpu(tr::ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32));
    mManager->setZero(*inputOffsets);
    tr::kernels::invokeInclusiveSum(*tr::ITensor::slice(inputOffsets, 1), *inputLengths, *mManager, *mStream);
    auto inputOffsetsHost2 = mManager->copyFrom(*inputOffsets, tr::MemoryType::kCPU);
    auto* inputOffsetsHost2Ptr = tr::bufferCast<tr::SizeType32>(*inputOffsetsHost2);

    for (std::size_t offset = 0; offset < inputOffsetsHost.size(); ++offset)
    {
        EXPECT_EQ(inputOffsetsHost[offset], inputOffsetsHost2Ptr[offset]) << "Error at index " << offset;
    }

    tr::kernels::invokeCopyPackedInputToOutputTransposed(
        *outputIds, *inputIds, *inputOffsets, maxInputLength, padId, *mStream);

    auto outputIdsHost = mManager->copyFrom(*outputIds, tr::MemoryType::kCPU);
    auto* outputIdsHostData = tr::bufferCast<tr::SizeType32>(*outputIdsHost);

    for (tr::SizeType32 b = 0; b < batchSize; ++b)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLengthsHost[b]; ++i)
            {
                auto const outputIndex = tc::flat_index3(i, b, beam, batchSize, beamWidth);
                EXPECT_EQ(outputIdsHostData[outputIndex], input[i])
                    << "Error at index (" << b << ',' << beam << ',' << i << ')';
            }
            for (tr::SizeType32 i = inputLengthsHost[b]; i < maxInputLength; ++i)
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
    tr::SizeType32 const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    tr::SizeType32 const batchSize{4};
    auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
    auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
    mManager->setZero(*outputTensor);

    tr::kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    auto outputHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
    auto* outputPtr = tr::bufferCast<tr::SizeType32>(*outputHost);

    std::cout << *outputHost;

    for (tr::SizeType32 batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        for (tr::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            for (tr::SizeType32 i = 0; i < inputLength; ++i)
            {
                auto const inputIdx = tc::flat_index2(batchIdx, i, inputLength);
                auto const expected = beam == 0 ? input[inputIdx] : 0;
                auto const outputIdx = tc::flat_index3(batchIdx, beam, i, beamWidth, inputLength);
                EXPECT_EQ(outputPtr[outputIdx], expected)
                    << "Error at index (" << batchIdx << ',' << beam << ',' << i << ')';
            }
        }
    }
}

TEST_F(RuntimeKernelTest, SplitTransposed)
{
    tr::SizeType32 const split{2};
    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    std::vector<std::int32_t> const output{28524, 5093, 23316, 11, 263, 355, 287, 12, 4881, 30022, 8776, 257};
    std::vector<std::int32_t> const output2{28524, 287, 23316, 4881, 263, 8776, 5093, 12, 11, 30022, 355, 257};

    {
        tr::SizeType32 const batchSize{6};
        auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
        auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = tr::ITensor::makeShape({split, batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
        auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensor);

        tr::kernels::splitTransposed(*outputTensor, *inputTensor, split, *mStream);
        auto outputHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
        auto* outputPtr = tr::bufferCast<tr::SizeType32>(*outputHost);
        cudaError_t cudaerr = cudaDeviceSynchronize();

        for (tr::SizeType32 i = 0; i < static_cast<tr::SizeType32>(input.size()); ++i)
        {
            EXPECT_EQ(outputPtr[i], output[i]);
        }
    }

    {
        tr::SizeType32 const batchSize{3};
        auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
        auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
        auto const outputShape = tr::ITensor::makeShape({split, batchSize, inputLength / split});

        auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
        auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);
        mManager->setZero(*outputTensor);

        tr::kernels::splitTransposed(*outputTensor, *inputTensor, split, *mStream);
        auto outputHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
        auto* outputPtr = tr::bufferCast<tr::SizeType32>(*outputHost);
        cudaError_t cudaerr = cudaDeviceSynchronize();

        for (tr::SizeType32 i = 0; i < static_cast<tr::SizeType32>(input.size()); ++i)
        {
            EXPECT_EQ(outputPtr[i], output2[i]);
        }
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

TEST_F(RuntimeKernelTest, TileInplaceInt32)
{
    tr::SizeType32 const beamWidth{3};

    std::vector<std::int32_t> const input{28524, 287, 5093, 12, 23316, 4881, 11, 30022, 263, 8776, 355, 257};
    tr::SizeType32 const batchSize{4};
    auto const inputLength = static_cast<tr::SizeType32>(input.size() / batchSize);
    auto const inputShape = tr::ITensor::makeShape({batchSize, inputLength});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, inputLength});

    auto inputTensor = mManager->copyFrom(input, inputShape, tr::MemoryType::kGPU);
    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT32);

    tr::kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    tr::kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    outputTensor->reshape(tr::ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInplaceHalf)
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

    tr::kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    tr::kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    outputTensor->reshape(tr::ITensor::makeShape({batchSize, beamWidth, inputLength}));
    verifyTiling(input, *outputTensor, *mManager);
}

TEST_F(RuntimeKernelTest, TileInt8Large)
{
    std::int8_t constexpr value{3};
    tr::SizeType32 constexpr batchSize{1};
    tr::SizeType32 constexpr beamWidth{2};

    tr::SizeType32 const d2{2};
    auto const d3 = std::numeric_limits<int32_t>::max();
    auto const inputShape = tr::ITensor::makeShape({batchSize, d2, d3});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, d2, d3});

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

TEST_F(RuntimeKernelTest, TileInplaceInt8Large)
{
    std::int8_t constexpr value{3};
    tr::SizeType32 constexpr batchSize{1};
    tr::SizeType32 constexpr beamWidth{2};

    tr::SizeType32 const d2{2};
    auto const d3 = std::numeric_limits<int32_t>::max();
    auto const inputShape = tr::ITensor::makeShape({batchSize, d2, d3});
    auto const outputShape = tr::ITensor::makeShape({batchSize * beamWidth, d2, d3});

    auto const inputBytes = tr::ITensor::volume(inputShape);
    auto const outputBytes = tr::ITensor::volume(outputShape);
    TLLM_LOG_INFO("Allocating %lu bytes for input, and %lu bytes for output.", inputBytes, outputBytes);
    auto inputTensor = mManager->gpu(inputShape, nvinfer1::DataType::kINT8);
    tr::kernels::invokeFill(*inputTensor, value, *mStream);

    auto outputTensor = mManager->gpu(outputShape, nvinfer1::DataType::kINT8);
    tr::kernels::scatterTensor(*outputTensor, *inputTensor, beamWidth, *mStream);
    tr::kernels::tileTensorInplace(*outputTensor, beamWidth, *mStream);

    auto bufferHost = mManager->copyFrom(*outputTensor, tr::MemoryType::kCPU);
    auto* bufferPtr = tr::bufferCast<std::int8_t>(*bufferHost);
    auto constexpr expected = value;
    for (std::size_t i = 0; i < bufferHost->getSize(); ++i)
    {
        EXPECT_EQ(bufferPtr[i], expected) << "Error at index " << i;
    }
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
