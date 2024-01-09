/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::kernels
{

namespace
{

template <typename T>
__global__ void fill(T* data, std::size_t size, T const value)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        data[idx] = value;
    }
}
} // namespace

template <typename T>
void invokeFill(IBuffer& buffer, T const value, CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const size = buffer.getSize();
    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(size, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax))};

    fill<<<gridSize, blockSize, 0, stream.get()>>>(data, size, value);
}

// template instantiation
template void invokeFill(IBuffer&, std::int32_t, CudaStream const&);
template void invokeFill(IBuffer&, std::int8_t, CudaStream const&);
template void invokeFill(IBuffer&, std::uint8_t, CudaStream const&);
template void invokeFill(IBuffer&, bool, CudaStream const&);
template void invokeFill(IBuffer&, half, CudaStream const&);
template void invokeFill(IBuffer&, float, CudaStream const&);

namespace
{
template <typename T>
__global__ void fillBatch(T* data, std::int32_t const* indices, std::size_t size, T const* values)
{
    auto const batchIdx = indices[blockIdx.y];
    const T value = values[blockIdx.y];
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const startIdx = batchIdx * size;
    auto const endIdx = startIdx + size;

    for (auto idx = startIdx + tidx; idx < endIdx; idx += stride)
    {
        data[idx] = value;
    }
}
} // namespace

template <typename T>
void invokeFillBatch(IBuffer& buffer, IBuffer const& slotIndices, std::size_t slotStride, IBuffer const& values,
    CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto indices = bufferCast<std::int32_t>(slotIndices);
    auto fillValues = bufferCast<T>(values);
    auto numSlots = slotIndices.getSize();
    auto const size = slotStride;
    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(size, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), static_cast<std::uint32_t>(numSlots)};

    fillBatch<<<gridSize, blockSize, 0, stream.get()>>>(data, indices, size, fillValues);
}

// template instantiation
template void invokeFillBatch<float>(IBuffer&, IBuffer const&, std::size_t, IBuffer const&, CudaStream const&);
template void invokeFillBatch<std::int8_t>(IBuffer&, IBuffer const&, std::size_t, IBuffer const&, CudaStream const&);
template void invokeFillBatch<std::int32_t>(IBuffer&, IBuffer const&, std::size_t, IBuffer const&, CudaStream const&);

namespace
{
template <typename VecT>
__global__ void copyBatch(const uint8_t* srcData, uint8_t* dstData, std::int32_t const* srcOffsets,
    std::int32_t const* dstOffsets, std::int32_t const* sizes, std::int32_t const dataTypeSize)
{
    constexpr auto VEC_ELTS = static_cast<int32_t>(sizeof(VecT));
    auto const srcStartIdx = srcOffsets[blockIdx.y] * dataTypeSize;
    auto const dstStartIdx = dstOffsets[blockIdx.y] * dataTypeSize;
    auto const size = sizes[blockIdx.y] * dataTypeSize;
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC_ELTS;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x * VEC_ELTS;
    auto const srcEndIdx = srcStartIdx + size;

    auto srcIdx = srcStartIdx + tidx;
    auto dstIdx = dstStartIdx + tidx;

    for (; srcIdx < srcEndIdx; srcIdx += stride, dstIdx += stride)
    {
        *reinterpret_cast<VecT*>(&dstData[dstIdx]) = *reinterpret_cast<const VecT*>(&srcData[srcIdx]);
    }
}
} // namespace

void invokeCopyBatch(IBuffer const& srcBuffer, IBuffer& dstBuffer, IBuffer const& srcOffsets, IBuffer const& dstOffsets,
    IBuffer const& sizes, std::size_t maxStride, CudaStream const& stream)
{
    auto srcDataPtr = reinterpret_cast<const uint8_t*>(srcBuffer.data());
    auto dstDataPtr = reinterpret_cast<uint8_t*>(dstBuffer.data());
    auto srcOffsetsPtr = bufferCast<std::int32_t>(srcOffsets);
    auto dstOffsetsPtr = bufferCast<std::int32_t>(dstOffsets);
    auto sizesPtr = bufferCast<std::int32_t>(sizes);
    auto numSlots = srcOffsets.getSize();
    auto const size = maxStride;
    auto const dataTypeSize = BufferDataType(srcBuffer.getDataType()).getSize();
    auto const copyRowSizeInBytes = size * dataTypeSize;

    auto copyBatchInvocation = copyBatch<uint8_t>;
    auto vectorSize = 1;
    if (dataTypeSize % 16 == 0)
    {
        vectorSize = 16;
        copyBatchInvocation = copyBatch<uint4>;
    }
    else if (dataTypeSize % 8 == 0)
    {
        vectorSize = 8;
        copyBatchInvocation = copyBatch<uint2>;
    }
    else if (dataTypeSize % 4 == 0)
    {
        vectorSize = 4;
        copyBatchInvocation = copyBatch<uint32_t>;
    }
    else if (dataTypeSize % 2 == 0)
    {
        vectorSize = 2;
        copyBatchInvocation = copyBatch<uint16_t>;
    }

    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(copyRowSizeInBytes / vectorSize, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), static_cast<std::uint32_t>(numSlots)};
    copyBatchInvocation<<<gridSize, blockSize, 0, stream.get()>>>(
        srcDataPtr, dstDataPtr, srcOffsetsPtr, dstOffsetsPtr, sizesPtr, static_cast<int32_t>(dataTypeSize));
}

namespace
{
template <typename T>
__global__ void add(T* data, std::size_t size, T const value)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        data[idx] += value;
    }
}
} // namespace

template <typename T>
void invokeAdd(IBuffer& buffer, T const value, CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const size = buffer.getSize();
    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(size, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax))};

    add<<<gridSize, blockSize, 0, stream.get()>>>(data, size, value);
}

template void invokeAdd(IBuffer&, std::int32_t, CudaStream const&);
template void invokeAdd(IBuffer&, std::int8_t, CudaStream const&);
template void invokeAdd(IBuffer&, float, CudaStream const&);

namespace
{
template <typename T>
__global__ void reduceSum(T* output, T const* input, std::size_t size)
{
    T threadSum = 0;
    for (auto index = threadIdx.x; index < size; index += blockDim.x)
    {
        threadSum += input[index];
    }

    T blockSum = 0;
    if (blockDim.x <= 32)
    {
        blockSum = tc::warpReduceSum(threadSum);
    }
    else
    {
        blockSum = tc::blockReduceSum(threadSum);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        *output = blockSum;
    }
}
} // namespace

template <typename T>
void invokeReduce(IBuffer& output, IBuffer const& input, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");
    TLLM_CHECK_WITH_INFO(output.getSize() == 1, common::fmtstr("Output size (%ld) has to be 1", output.getSize()));

    auto outputPtr = bufferCast<T>(output);
    auto inputPtr = bufferCast<T>(input);
    auto const size = input.getSize();

    dim3 blockSize{std::min(512u, static_cast<std::uint32_t>(size))};
    dim3 gridSize{1};

    reduceSum<<<gridSize, blockSize, 0, stream.get()>>>(outputPtr, inputPtr, size);
}

void reduce(IBuffer& output, IBuffer const& input, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeReduce<SizeType>(output, input, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeReduce<float>(output, input, stream); break;
    case nvinfer1::DataType::kHALF: invokeReduce<half>(output, input, stream); break;
    case nvinfer1::DataType::kINT8: invokeReduce<int8_t>(output, input, stream); break;
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

namespace
{
__global__ void transpose(SizeType* output, SizeType const* input, SizeType const batchSize, SizeType const rowSize)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < rowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * rowSize + tokenIdx;
            auto const outputIdx = tokenIdx * batchSize + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");
    TLLM_CHECK_WITH_INFO(input.getSize() == output.getSize(),
        common::fmtstr("Input size (%ld) and output size (%ld) differ", input.getSize(), output.getSize()));

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(
        inputShape.nbDims == 2, common::fmtstr("Input shape must have 2 dimensions, but has %d", inputShape.nbDims));

    SizeType const batchSize = inputShape.d[0];
    SizeType const rowSize = inputShape.d[1];

    dim3 const blockSize(256, 1);
    dim3 const gridSize((rowSize + blockSize.x - 1) / blockSize.x, batchSize);

    transpose<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<SizeType>(output), bufferCast<SizeType const>(input), batchSize, rowSize);
}

namespace
{
__global__ void transposeWithOutputOffset(SizeType* output, SizeType const* input, SizeType const nbInputRows,
    SizeType const inputRowSize, SizeType const outputRowSize, SizeType const outputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < nbInputRows; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < inputRowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + tokenIdx;
            auto const outputIdx = tokenIdx * outputRowSize + outputOffset + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType const outputOffset, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(
        inputShape.nbDims == 2, common::fmtstr("Input shape must have 2 dimensions, but has %d", inputShape.nbDims));
    SizeType const nbInputRows = inputShape.d[0];
    SizeType const inputRowSize = inputShape.d[1];

    auto const& outputShape = output.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 2, common::fmtstr("Output shape must have 2 dimensions, but has %d", outputShape.nbDims));
    SizeType const nbOutputRows = outputShape.d[0];
    SizeType const outputRowSize = outputShape.d[1];

    TLLM_CHECK_WITH_INFO(inputRowSize == nbOutputRows,
        common::fmtstr("Input dim 1 (%d) and output dim 0 (%d) differ", inputRowSize, nbOutputRows));
    TLLM_CHECK_WITH_INFO(outputOffset + nbInputRows <= outputRowSize,
        common::fmtstr("Input (%d rows) does not fit into output (%d columns, offset %d)", nbInputRows, inputRowSize,
            outputOffset));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((inputRowSize + blockSize.x - 1) / blockSize.x, nbInputRows);

    transposeWithOutputOffset<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(output),
        bufferCast<SizeType const>(input), nbInputRows, inputRowSize, outputRowSize, outputOffset);
}

namespace
{
__global__ void transposeWithInputOffset(SizeType* output, SizeType const* input, SizeType const outputRowSize,
    SizeType const nbOutputRows, SizeType const inputRowSize, SizeType const inputOffset)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < outputRowSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < nbOutputRows; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = batchIdx * inputRowSize + inputOffset + tokenIdx;
            auto const outputIdx = tokenIdx * outputRowSize + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}
} // namespace

void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType const inputOffset, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(
        inputShape.nbDims == 2, common::fmtstr("Input shape must have 2 dimensions, but has %d", inputShape.nbDims));
    SizeType const nbInputRows = inputShape.d[0];
    SizeType const inputRowSize = inputShape.d[1];

    auto const& outputShape = output.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 2, common::fmtstr("Output shape must have 2 dimensions, but has %d", outputShape.nbDims));
    SizeType const nbOutputRows = outputShape.d[0];
    SizeType const outputRowSize = outputShape.d[1];

    TLLM_CHECK_WITH_INFO(nbInputRows == outputRowSize,
        common::fmtstr("Input dim 0 (%d) and output dim 1 (%d) differ", nbInputRows, outputRowSize));
    TLLM_CHECK_WITH_INFO(inputOffset + nbOutputRows <= inputRowSize,
        common::fmtstr("Cannot extract output (%d rows) from input (%d columns, offset %d)", nbOutputRows, inputRowSize,
            inputOffset));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((nbOutputRows + blockSize.x - 1) / blockSize.x, outputRowSize);

    transposeWithInputOffset<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(output),
        bufferCast<SizeType const>(input), outputRowSize, nbOutputRows, inputRowSize, inputOffset);
}

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream)
{
    auto const size = input.getSize();
    auto const* inputData = bufferCast<SizeType>(input);
    auto* outputData = bufferCast<SizeType>(output);

    std::size_t tempStorageBytes{0};
    cub::DeviceScan::InclusiveSum(nullptr, tempStorageBytes, inputData, outputData, size, stream.get());
    auto tempStorage = manager.gpu(tempStorageBytes, nvinfer1::DataType::kUINT8);
    auto* tempStorageData = bufferCast<std::uint8_t>(*tempStorage);
    cub::DeviceScan::InclusiveSum(tempStorageData, tempStorageBytes, inputData, outputData, size, stream.get());
}

namespace
{
__global__ void buildTokenMask(SizeType* tokenMask, SizeType const* inputLengths, SizeType const batchSize,
    SizeType const maxInputLength, SizeType const maxSeqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType tokenIdx = tidx; tokenIdx < maxSeqLength; tokenIdx += blockDim.x * gridDim.x)
        {
            tokenMask[batchIdx * maxSeqLength + tokenIdx]
                = (tokenIdx >= inputLength && tokenIdx < maxInputLength) ? 1 : 0;
        }
    }
}
} // namespace

void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType const maxInputLength, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == tokenMask.getDataType(), "tokenMask has wrong data type");
    TLLM_CHECK_WITH_INFO(
        TRTDataType<SizeType>::value == inputLengths.getDataType(), "inputLengths has wrong data type");

    auto const& shape = tokenMask.getShape();
    SizeType const batchSize = shape.d[0];
    SizeType const maxSeqLength = shape.d[1];

    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "TtokenMask dimension 1 (%d) is smaller than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxSeqLength + blockSize.x - 1) / blockSize.x, batchSize);

    buildTokenMask<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(tokenMask),
        bufferCast<SizeType const>(inputLengths), batchSize, maxInputLength, maxSeqLength);
}

namespace
{
__global__ void buildAttentionMask(SizeType* attentionMask, SizeType const size, SizeType const padId)
{
    SizeType const tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (SizeType i = tid; i < size; i += blockDim.x * gridDim.x)
    {
        auto const x = attentionMask[i];
        attentionMask[i] = (x != padId);
    }
}
} // namespace

void invokeBuildAttentionMask(ITensor& attentionMask, SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        TRTDataType<SizeType>::value == attentionMask.getDataType(), "attentionMask has wrong data type");

    auto const size = attentionMask.getSize();
    dim3 const blockSize(256);
    dim3 const gridSize((size + blockSize.x - 1) / blockSize.x);

    buildAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(attentionMask), size, padId);
}

namespace
{
__global__ void extendAttentionMask(
    SizeType* newMask, SizeType const* oldMask, SizeType const batchSize, SizeType const seqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType tokenIdx = tidx; tokenIdx < seqLength + 1; tokenIdx += blockDim.x * gridDim.x)
        {
            SizeType oldIndex = batchIdx * seqLength + tokenIdx;
            SizeType newIndex = batchIdx * (seqLength + 1) + tokenIdx;
            newMask[newIndex] = (tokenIdx < seqLength) ? oldMask[oldIndex] : 1;
        }
    }
}
} // namespace

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == newMask.getDataType(), "attentionMask has wrong data type");
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType>::value == oldMask.getDataType(), "attentionMask has wrong data type");

    auto const& shape = oldMask.getShape();
    SizeType const batchSize = shape.d[0];
    SizeType const seqLength = shape.d[1];

    dim3 const blockSize(256, 1);
    dim3 const gridSize((seqLength + blockSize.x - 1) / blockSize.x, batchSize);

    extendAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<SizeType>(newMask), bufferCast<SizeType>(oldMask), batchSize, seqLength);
}

namespace
{
__global__ void copyInputToOutputTransposed(SizeType* outputIds, SizeType const* inputIds, SizeType const* inputLengths,
    SizeType const padId, SizeType const batchSize, SizeType const beamWidth, SizeType const maxInputLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[batchIdx * maxInputLength + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(tokenIdx, batchIdx, beamIdx, batchSize, beamWidth);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyInputToOutputTransposed(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const batchSize = static_cast<SizeType>(inputLengths.getSize());
    auto const& inputShape = inputIds.getShape();
    SizeType const maxInputLength = inputShape.d[inputShape.nbDims - 1];
    auto const& outputShape = outputIds.getShape();
    SizeType const maxSeqLength = outputShape.d[0];
    SizeType const beamWidth = outputShape.d[2];

    auto const inputBatchSize = inputIds.getSize() / maxInputLength;
    TLLM_CHECK_WITH_INFO(std::size_t(batchSize) == inputBatchSize,
        common::fmtstr("Input ids batch size (%ld) does not match inputLengths size (%ld)", inputBatchSize,
            std::size_t(batchSize)));
    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[1],
        common::fmtstr(
            "Output ids batch size (%d) does not match inputLengths size (%d)", outputShape.d[1], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyInputToOutputTransposed<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputLengths), padId, batchSize, beamWidth,
        maxInputLength);
}

namespace
{
__global__ void copyPackedInputToOutputTransposed(SizeType* outputIds, SizeType const* inputIds,
    SizeType const* inputOffsets, SizeType const padId, SizeType const batchSize, SizeType const beamWidth,
    SizeType const maxInputLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const tokenBegin = inputOffsets[batchIdx];
        auto const tokenEnd = inputOffsets[batchIdx + 1];
        auto const inputLength = tokenEnd - tokenBegin;

        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[tokenBegin + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(tokenIdx, batchIdx, beamIdx, batchSize, beamWidth);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyPackedInputToOutputTransposed(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType const maxInputLength, SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const batchSize = static_cast<SizeType>(inputOffsets.getSize()) - 1;
    auto const& outputShape = outputIds.getShape();
    SizeType const maxSeqLength = outputShape.d[0];
    SizeType const beamWidth = outputShape.d[2];

    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[1],
        common::fmtstr(
            "Output ids batch size (%d) does not match inputOffsets batch size (%d)", outputShape.d[1], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyPackedInputToOutputTransposed<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputOffsets), padId, batchSize, beamWidth,
        maxInputLength);
}

namespace
{
__global__ void copyInputToOutput(SizeType* outputIds, SizeType const* inputIds, SizeType const* inputLengths,
    SizeType const padId, SizeType const batchSize, SizeType const beamWidth, SizeType const maxInputLength,
    SizeType const maxSeqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[batchIdx * maxInputLength + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(batchIdx, beamIdx, tokenIdx, beamWidth, maxSeqLength);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const& inputShape = inputIds.getShape();
    auto const& outputShape = outputIds.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 3, common::fmtstr("Output shape must have 3 dimensions, but has %d", outputShape.nbDims));

    auto const batchSize = static_cast<SizeType>(inputLengths.getSize());
    SizeType const maxInputLength = inputShape.d[inputShape.nbDims - 1];
    SizeType const beamWidth = outputShape.d[1];
    SizeType const maxSeqLength = outputShape.d[2];

    auto const inputBatchSize = inputIds.getSize() / maxInputLength;
    TLLM_CHECK_WITH_INFO(std::size_t(batchSize) == inputBatchSize,
        common::fmtstr("Input ids batch size (%ld) does not match inputLengths size (%ld)", inputBatchSize,
            std::size_t(batchSize)));
    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[0],
        common::fmtstr(
            "Output ids batch size (%d) does not match inputLengths size (%d)", outputShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputLengths), padId, batchSize, beamWidth,
        maxInputLength, maxSeqLength);
}

namespace
{
__global__ void copyPackedInputToOutput(SizeType* outputIds, SizeType const* inputIds, SizeType const* inputOffsets,
    SizeType const padId, SizeType const batchSize, SizeType const beamWidth, SizeType const maxInputLength,
    SizeType const maxSeqLength)
{
    SizeType const tidx = blockIdx.x * blockDim.x + threadIdx.x;
    SizeType const tidy = blockIdx.y * blockDim.y + threadIdx.y;

    for (SizeType batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const tokenBegin = inputOffsets[batchIdx];
        auto const tokenEnd = inputOffsets[batchIdx + 1];
        auto const inputLength = tokenEnd - tokenBegin;

        for (SizeType tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[tokenBegin + tokenIdx] : padId;
            for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(batchIdx, beamIdx, tokenIdx, beamWidth, maxSeqLength);
                outputIds[outputIdx] = value;
            }
        }
    }
}
} // namespace

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType const maxInputLength, SizeType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const& outputShape = outputIds.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 3, common::fmtstr("Output shape must have 3 dimensions, but has %d", outputShape.nbDims));

    auto const batchSize = static_cast<SizeType>(inputOffsets.getSize()) - 1;
    SizeType const beamWidth = outputShape.d[1];
    SizeType const maxSeqLength = outputShape.d[2];

    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[0],
        common::fmtstr(
            "Output ids batch size (%d) does not match inputOffsets batch size (%d)", outputShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyPackedInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType>(outputIds),
        bufferCast<SizeType const>(inputIds), bufferCast<SizeType const>(inputOffsets), padId, batchSize, beamWidth,
        maxInputLength, maxSeqLength);
}

void initOutputIds(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    ITensor const& inputOffsets, TokenIdType const padId, TokenIdType const endId, SizeType const maxInputLength,
    bool const inputPacked, CudaStream const& stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    kernels::invokeFill(outputIds, endId, stream);

    if (inputPacked)
    {
        kernels::invokeCopyPackedInputToOutput(outputIds, inputIds, inputOffsets, maxInputLength, padId, stream);
    }
    else
    {
        kernels::invokeCopyInputToOutput(outputIds, inputIds, inputLengths, padId, stream);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
template <typename T>
__global__ void scatterTensor(T* output, T const* input, std::uint32_t const batchSize,
    std::uint32_t const inputRowSize, std::size_t const outputRowSize, std::uint32_t const beamWidth)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const tidy = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    for (auto batchIdx = tidy; batchIdx < batchSize; batchIdx += stridey)
    {
        for (auto columnIdx = tidx; columnIdx < inputRowSize; columnIdx += stridex)
        {
            auto const inputIdx = batchIdx * inputRowSize + columnIdx;
            auto const value = input[inputIdx];
            std::size_t constexpr beamIdx{0};
            auto const outputIdx = (batchIdx * beamWidth + beamIdx) * outputRowSize + columnIdx;
            output[outputIdx] = value;
        }
    }
}

template <typename T>
__global__ void splitTransposed(T* output, T const* input, std::uint32_t const batchSize,
    std::uint32_t const inputRowSize, std::uint32_t const split)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const tidy = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto const tidz = static_cast<std::size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;
    auto const stridez = static_cast<std::size_t>(blockDim.z) * gridDim.z;

    auto const splitRowSize = static_cast<std::size_t>(inputRowSize / split);
    for (auto pIdx = tidz; pIdx < split; pIdx += stridez)
    {
        for (auto bid = tidx; bid < batchSize; bid += stridex)
        {
            for (auto colIdx = tidy; colIdx < splitRowSize; colIdx += stridey)
            {
                auto outputIdx
                    = common::flat_index3(pIdx, bid, colIdx, static_cast<std::size_t>(batchSize), splitRowSize);
                auto inputIdx
                    = common::flat_index2(bid, colIdx + pIdx * splitRowSize, static_cast<std::size_t>(inputRowSize));
                output[outputIdx] = input[inputIdx];
            }
        }
    }
}

template <typename T>
__global__ void tileTensor(T* output, T const* input, std::uint32_t const batchSize, std::size_t const inputRowSize,
    std::size_t const outputRowSize, std::uint32_t const beamWidth)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const tidy = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    for (auto batchIdx = tidy; batchIdx < batchSize; batchIdx += stridey)
    {
        for (auto columnIdx = tidx; columnIdx < inputRowSize; columnIdx += stridex)
        {
            auto const inputIdx = batchIdx * inputRowSize + columnIdx;
            auto const value = input[inputIdx];
            for (std::size_t beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = (batchIdx * beamWidth + beamIdx) * outputRowSize + columnIdx;
                output[outputIdx] = value;
            }
        }
    }
}

template <typename T>
__global__ void tileTensorInPlace(
    T* inputOutput, std::uint32_t const batchSize, std::size_t const inputOutputRowSize, std::uint32_t const beamWidth)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const tidy = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    for (auto batchIdx = tidy; batchIdx < batchSize; batchIdx += stridey)
    {
        for (auto columnIdx = tidx; columnIdx < inputOutputRowSize; columnIdx += stridex)
        {
            auto const inputIdx = (batchIdx * beamWidth + 0) * inputOutputRowSize + columnIdx;
            auto const value = inputOutput[inputIdx];
            for (std::size_t beamIdx = 1; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = (batchIdx * beamWidth + beamIdx) * inputOutputRowSize + columnIdx;
                inputOutput[outputIdx] = value;
            }
        }
    }
}

} // namespace

template <typename T>
void invokeScatterTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream)
{
    auto const& inputShape = input.getShape();
    auto const nbInputRows = static_cast<std::uint32_t>(inputShape.d[0]);
    auto const inputRowSize = input.getSize() / static_cast<std::size_t>(nbInputRows);
    auto const& outputShape = output.getShape();
    auto const nbOutputRows = static_cast<std::uint32_t>(outputShape.d[0]);
    auto const outputRowSize = output.getSize() / static_cast<std::size_t>(nbOutputRows);

    TLLM_CHECK_WITH_INFO(nbOutputRows == beamWidth * nbInputRows,
        common::fmtstr(
            "nbOutputRows (%d) must be beamWidth (%d) times nbInputRows (%d)", nbOutputRows, beamWidth, nbInputRows));
    TLLM_CHECK_WITH_INFO(outputRowSize >= inputRowSize,
        common::fmtstr("output row size (%ld) must be at least input row size (%ld)", outputRowSize, inputRowSize));

    dim3 const blockSize{256, 1};
    std::size_t const gridx{tc::ceilDiv(inputRowSize, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), nbInputRows};
    scatterTensor<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<T>(output), bufferCast<T const>(input),
        nbInputRows, inputRowSize, outputRowSize, static_cast<uint32_t>(beamWidth));
}

void scatterTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeScatterTensor<SizeType>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeScatterTensor<float>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kHALF: invokeScatterTensor<half>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kINT8: invokeScatterTensor<int8_t>(output, input, beamWidth, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeScatterTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
#endif // ENABLE_FP8
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

template <typename T>
void invokeSplitTransposed(ITensor& output, ITensor const& input, SizeType split, CudaStream const& stream)
{
    auto const& inputShape = input.getShape();
    auto const nbInputRows = static_cast<std::uint32_t>(inputShape.d[0]);
    auto const inputRowSize = input.getSize() / static_cast<std::size_t>(nbInputRows);
    auto const& outputShape = output.getShape();
    auto const nbOutputRows = static_cast<std::uint32_t>(outputShape.d[0]);
    auto const outputRowSize = output.getSize() / static_cast<std::size_t>(nbOutputRows);
    auto const inputNbElems = input.getSize();
    auto const outputNbElems = output.getSize();

    TLLM_CHECK_WITH_INFO(
        nbOutputRows == split, common::fmtstr("nbOutputRows (%d) must be split (%d)", nbOutputRows, split));
    TLLM_CHECK_WITH_INFO(
        inputNbElems == outputNbElems, common::fmtstr("input and output must have the same number of elements"));

    dim3 const blockSize{256, 1, 1};
    std::size_t const gridx{tc::ceilDiv(nbInputRows, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{
        static_cast<std::uint32_t>(std::min(gridx, gridMax)), static_cast<std::uint32_t>(inputRowSize), 1};
    splitTransposed<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<T>(output), bufferCast<T const>(input), nbInputRows, inputRowSize, static_cast<uint32_t>(split));
}

void splitTransposed(ITensor& output, ITensor const& input, SizeType split, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeSplitTransposed<SizeType>(output, input, split, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeSplitTransposed<float>(output, input, split, stream); break;
    case nvinfer1::DataType::kHALF: invokeSplitTransposed<half>(output, input, split, stream); break;
    case nvinfer1::DataType::kINT8: invokeSplitTransposed<int8_t>(output, input, split, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeSplitTransposed<__nv_fp8_e4m3>(output, input, split, stream); break;
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: invokeSplitTransposed<__nv_bfloat16>(output, input, split, stream); break;
#endif // ENABLE_BF16
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

template <typename T>
void invokeTileTensor(ITensor& output, ITensor const& input, SizeType const beamWidth, CudaStream const& stream)
{
    auto const& inputShape = input.getShape();
    auto const nbInputRows = static_cast<std::uint32_t>(inputShape.d[0]);
    auto const inputRowSize = input.getSize() / static_cast<std::size_t>(nbInputRows);
    auto const& outputShape = output.getShape();
    auto const nbOutputRows = static_cast<std::uint32_t>(outputShape.d[0]);
    auto const outputRowSize = output.getSize() / static_cast<std::size_t>(nbOutputRows);

    TLLM_CHECK_WITH_INFO(nbOutputRows == beamWidth * nbInputRows,
        common::fmtstr(
            "nbOutputRows (%d) must be beamWidth (%d) times nbInputRows (%d)", nbOutputRows, beamWidth, nbInputRows));
    TLLM_CHECK_WITH_INFO(outputRowSize >= inputRowSize,
        common::fmtstr("output row size (%ld) must be at least input row size (%ld)", outputRowSize, inputRowSize));

    dim3 const blockSize{256, 1};
    std::size_t const gridx{tc::ceilDiv(inputRowSize, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), nbInputRows};
    tileTensor<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<T>(output), bufferCast<T const>(input), nbInputRows,
        inputRowSize, outputRowSize, static_cast<uint32_t>(beamWidth));
}

void tileTensor(ITensor& output, ITensor const& input, SizeType beamWidth, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeTileTensor<SizeType>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeTileTensor<float>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kHALF: invokeTileTensor<half>(output, input, beamWidth, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: invokeTileTensor<__nv_bfloat16>(output, input, beamWidth, stream); break;
#endif // ENABLE_BF16
    case nvinfer1::DataType::kINT8: invokeTileTensor<int8_t>(output, input, beamWidth, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeTileTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
#endif // ENABLE_FP8
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

template <typename T>
void invokeTileTensorInPlace(ITensor& inputOutput, SizeType const beamWidth, CudaStream const& stream)
{
    auto const& inputOutputShape = inputOutput.getShape();
    auto const nbOutputRows = static_cast<std::uint32_t>(inputOutputShape.d[0]);
    auto const nbInputRows = nbOutputRows / static_cast<std::uint32_t>(beamWidth);
    auto const inputOutputRowSize = inputOutput.getSize() / static_cast<std::size_t>(nbOutputRows);

    dim3 const blockSize{256, 1};
    std::size_t const gridx{tc::ceilDiv(inputOutputRowSize, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), nbInputRows};
    tileTensorInPlace<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<T>(inputOutput), nbInputRows, inputOutputRowSize, static_cast<std::uint32_t>(beamWidth));
}

void tileTensorInplace(ITensor& tensor, SizeType beamWidth, CudaStream const& stream)
{
    switch (tensor.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeTileTensorInPlace<SizeType>(tensor, beamWidth, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeTileTensorInPlace<float>(tensor, beamWidth, stream); break;
    case nvinfer1::DataType::kHALF: invokeTileTensorInPlace<half>(tensor, beamWidth, stream); break;
    case nvinfer1::DataType::kINT8: invokeTileTensorInPlace<int8_t>(tensor, beamWidth, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeTileTensorInPlace<__nv_fp8_e4m3>(tensor, beamWidth, stream); break;
#endif // ENABLE_FP8
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

// In the following kernel, we launch a grid with batchSize blocks of threads. Each thread block
// copies the logits from the "logits" tensor to the "lastTokenLogits" tensor for the last token
// of each sequence.
//
// TODO: Enable vector copies for higher BW utilization.

template <typename T>
__global__ void gatherLastTokenLogitsKernel(T* lastTokenLogits, T const* logits, int const* lastTokenIds,
    int maxInputLength, int beamWidth, int vocabSizePadded)
{
    // This sequence.
    int seqIdx = blockIdx.x;
    // Find the index of the last token in that sequence.
    // Since lastTokenIds is the accumulated length instead of real ids, so we need to minus 1.
    // For length [11, 23], we hope to get the results of id 10 and 22, in fact.
    int lastTokenIdx = lastTokenIds[seqIdx] - 1;

    // The output pointer.
    T* lastTokenLogitsPtr = &lastTokenLogits[seqIdx * beamWidth * vocabSizePadded];
    // The input pointer.
    T const* logitsPtr = &logits[lastTokenIdx * vocabSizePadded];

    // The threads in the block collaborate to copy the logits.
    for (int idx = threadIdx.x; idx < vocabSizePadded; idx += blockDim.x)
    {
        T value = logitsPtr[idx];
        for (int beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            lastTokenLogitsPtr[beamIdx * vocabSizePadded + idx] = value;
        }
    }
}

template <typename T>
void invokeGatherLastTokenLogits(
    ITensor& output, ITensor const& input, ITensor const& lastTokenIds, CudaStream const& stream)
{
    auto const& outputShape = output.getShape();
    auto const batchSize = static_cast<std::uint32_t>(outputShape.d[0]);
    auto const beamWidth = static_cast<std::uint32_t>(outputShape.d[1]);
    auto const vocabSizePadded = static_cast<std::uint32_t>(outputShape.d[2]);

    auto const& inputShape = input.getShape();
    auto const maxInputLength = static_cast<std::uint32_t>(inputShape.d[1]);

    TLLM_CHECK_WITH_INFO(inputShape.d[0] == batchSize, "Invalid input shape: dim[0]");
    TLLM_CHECK_WITH_INFO(inputShape.d[2] == vocabSizePadded, "Invalid input shape: dim[2]");

    dim3 const blockSize{256, 1};
    dim3 const gridSize{static_cast<std::uint32_t>(batchSize), 1};
    gatherLastTokenLogitsKernel<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<T>(output), bufferCast<T>(input),
        bufferCast<int32_t>(lastTokenIds), static_cast<std::uint32_t>(maxInputLength),
        static_cast<std::uint32_t>(beamWidth), vocabSizePadded);
}

void gatherLastTokenLogits(ITensor& output, ITensor const& input, ITensor const& lastTokenIds, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: invokeGatherLastTokenLogits<float>(output, input, lastTokenIds, stream); break;
    case nvinfer1::DataType::kHALF: invokeGatherLastTokenLogits<half>(output, input, lastTokenIds, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        invokeGatherLastTokenLogits<__nv_bfloat16>(output, input, lastTokenIds, stream);
        break;
#endif // ENABLE_BF16
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8:
        invokeGatherLastTokenLogits<__nv_fp8_e4m3>(output, input, lastTokenIds, stream);
        break;
#endif // ENABLE_FP8
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

// In the following kernel, we launch a grid with (microBatchSize * beamWidth, outputLen) blocks of threads. Each thread
// block copies a `vocabSizePadded` length logits tensor from the "inputLogits (microBatchSize, beamWidth,
// vocabSizePadded)" to the "outputGenerationLogits (batchSize, beamWidth, outputLen, vocabSizePadded)"
template <typename T>
__global__ void mergeLogitsFragmentsKernel(T* output, T** fragmentsVector, const int outputLen, int firstBatchSlotIdx,
    int microBatchSize, int beamWidth, int vocabSizePadded, int stepOffset)
{
    // output: shape: [batchSize, beamWidth, outputLen, vocabSize]
    // inputVecor.at(i): shape: [microBatchSize, beamWidth, vocabSize]

    // Current step
    int curStep = blockIdx.y;

    // The relatively batch slot index that this thread block in microBatchSize.
    int relativeBatchSlotIdx = blockIdx.x / beamWidth;

    // The Absolute batch slot index in batchSize.
    int absoluteBatchSlotIdx = firstBatchSlotIdx + relativeBatchSlotIdx;

    // The beam index that this thread block process
    int mbeamIdx = blockIdx.x % beamWidth;

    // The output pointer
    const unsigned int outputOffset
        = (absoluteBatchSlotIdx * beamWidth * outputLen + mbeamIdx * outputLen + curStep + stepOffset)
        * vocabSizePadded;

    T* outputPtr = &output[outputOffset];

    const unsigned int inputOffset = (relativeBatchSlotIdx * beamWidth + mbeamIdx) * vocabSizePadded;
    // The input pointer.
    T const* inputPtr = &fragmentsVector[curStep][inputOffset];

    // The threads in the block collaborate to copy the logits.
    for (int idx = threadIdx.x; idx < vocabSizePadded; idx += blockDim.x)
    {
        outputPtr[idx] = inputPtr[idx];
    }
}

template <typename T>
void invokeMergeLogitsFragments(BufferManager const& bufferManager, ITensor& output,
    std::vector<TensorPtr> fragmentsVector, ITensor& cachePointerDevice, ITensor& cachePointerHost,
    SizeType firstBatchSlotIdx, SizeType const microBatchSize, SizeType const beamWidth, CudaStream const& stream,
    int stepOffset)
{
    size_t fragmentsVectorSize = fragmentsVector.size();

    auto cachePointerHostPtr = bufferCast<T*>(cachePointerHost);

    for (int i = 0; i < fragmentsVectorSize; i++)
    {
        cachePointerHostPtr[i] = static_cast<T*>(fragmentsVector.at(i)->data());
    }
    bufferManager.copy(cachePointerHost, cachePointerDevice);

    dim3 blockSize(256);
    dim3 gridSize{(unsigned int) (microBatchSize * beamWidth), (unsigned int) (fragmentsVectorSize)};

    auto const& outputShape = output.getShape();
    auto const vocabSizePadded = static_cast<SizeType>(outputShape.d[outputShape.nbDims - 1]);
    auto const outputLen = static_cast<SizeType>(outputShape.d[outputShape.nbDims - 2]);

    TLLM_CHECK_WITH_INFO(outputLen >= fragmentsVectorSize, "Fragments size does not match outputLen size");

    mergeLogitsFragmentsKernel<T><<<gridSize, blockSize, 0, stream.get()>>>(static_cast<T*>(output.data()),
        static_cast<T**>(cachePointerDevice.data()), outputLen, firstBatchSlotIdx, microBatchSize, beamWidth,
        vocabSizePadded, stepOffset);
}

void mergeLogitsFragments(BufferManager const& bufferManager, ITensor& output, std::vector<TensorPtr> fragmentsVector,
    ITensor& cachePointerDevice, ITensor& cachePointerHost, SizeType firstBatchSlotIdx, SizeType const microBatchSize,
    SizeType const beamWidth, CudaStream const& stream, int stepOffset)
{
    switch (output.getDataType())
    {
    case nvinfer1::DataType::kFLOAT:
        invokeMergeLogitsFragments<float>(bufferManager, output, fragmentsVector, cachePointerDevice, cachePointerHost,
            firstBatchSlotIdx, microBatchSize, beamWidth, stream, stepOffset);
        break;
    case nvinfer1::DataType::kHALF:
        invokeMergeLogitsFragments<half>(bufferManager, output, fragmentsVector, cachePointerDevice, cachePointerHost,
            firstBatchSlotIdx, microBatchSize, beamWidth, stream, stepOffset);
        break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16:
        invokeMergeLogitsFragments<__nv_bfloat16>(bufferManager, output, fragmentsVector, cachePointerDevice,
            cachePointerHost, firstBatchSlotIdx, microBatchSize, beamWidth, stream, stepOffset);
        break;
#endif // ENABLE_BF16
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8:
        invokeMergeLogitsFragments<__nv_fp8_e4m3>(bufferManager, output, fragmentsVector, cachePointerDevice,
            cachePointerHost, firstBatchSlotIdx, microBatchSize, beamWidth, stream, stepOffset);
        break;
#endif // ENABLE_FP8
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

} // namespace tensorrt_llm::runtime::kernels
