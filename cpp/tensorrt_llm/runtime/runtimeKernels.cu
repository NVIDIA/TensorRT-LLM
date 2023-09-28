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
template void invokeFill(IBuffer&, float, CudaStream const&);

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
    case nvinfer1::DataType::kFP8: invokeScatterTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
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
    case nvinfer1::DataType::kINT8: invokeTileTensor<int8_t>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kFP8: invokeTileTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
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
    case nvinfer1::DataType::kFP8: invokeTileTensorInPlace<__nv_fp8_e4m3>(tensor, beamWidth, stream); break;
    default: TLLM_CHECK_WITH_INFO(false, "data type not supported");
    }
}

} // namespace tensorrt_llm::runtime::kernels
