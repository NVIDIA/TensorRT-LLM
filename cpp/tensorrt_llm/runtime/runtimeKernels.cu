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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <NvInferRuntimeBase.h>
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
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        data[idx] = value;
    }
}

//! @param data    expected shape [indicesRange, size]
//! @param indices expected shape [gridDim.y]
//! @param size
//! @param values  expected shape [gridDim.y]
template <typename T>
__global__ void fillBatch(T* data, std::int32_t const* indices, std::size_t size, T const* values)
{
    auto const batchIdx = indices[blockIdx.y];
    T const value = values[blockIdx.y];
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const startIdx = batchIdx * size;
    auto const endIdx = startIdx + size;

    for (auto idx = startIdx + tidx; idx < endIdx; idx += stride)
    {
        data[idx] = value;
    }
}

template <typename T>
void invokeFillBatch(IBuffer& buffer, IBuffer const& slotIndices, std::size_t slotStride, IBuffer const& values,
    CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const* const indices = bufferCast<std::int32_t>(slotIndices);
    auto fillValues = bufferCast<T>(values);
    auto numSlots = slotIndices.getSize();
    auto const size = slotStride;
    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(size, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), static_cast<std::uint32_t>(numSlots)};

    fillBatch<<<gridSize, blockSize, 0, stream.get()>>>(data, indices, size, fillValues);
}

//! @param data    expected shape [gridDim.y, size]
//! @param indices expected shape [gridDim.y]
//! @param size
//! @param values  expected shape [indicesRange, size]
template <typename T>
__global__ void gatherBatch(T* data, T const* values, std::int32_t const* indices, std::size_t size)
{
    auto const tidx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        auto const batchIdx = blockIdx.y;
        auto const slotIdx = indices[blockIdx.y];
        data[batchIdx + idx] = values[slotIdx + idx];
    }
}

template <typename T>
void invokeGatherBatch(IBuffer& buffer, IBuffer const& values, IBuffer const& slotIndices, std::size_t slotStride,
    CudaStream const& stream)
{
    auto data = bufferCast<T>(buffer);
    auto const* const indices = bufferCast<std::int32_t>(slotIndices);
    auto sparseValues = bufferCast<T>(values);
    auto numSlots = slotIndices.getSize();
    auto const size = slotStride;
    dim3 const blockSize{256};
    std::size_t const gridx{tc::ceilDiv(size, blockSize.x)};
    std::size_t const gridMax{std::numeric_limits<std::uint32_t>::max()};
    dim3 const gridSize{static_cast<std::uint32_t>(std::min(gridx, gridMax)), static_cast<std::uint32_t>(numSlots)};

    gatherBatch<<<gridSize, blockSize, 0, stream.get()>>>(data, sparseValues, indices, size);
}

template <typename VecT>
__global__ void copyBatch(uint8_t const* srcData, uint8_t* dstData, SizeType64 const* srcOffsets,
    SizeType64 const* dstOffsets, SizeType64 const* sizes, SizeType64 const dataTypeSize)
{
    constexpr auto VEC_ELTS = static_cast<int32_t>(sizeof(VecT));
    SizeType64 const srcStartIdx = srcOffsets[blockIdx.y] * dataTypeSize;
    SizeType64 const dstStartIdx = dstOffsets[blockIdx.y] * dataTypeSize;
    SizeType64 const size = sizes[blockIdx.y] * dataTypeSize;
    SizeType64 const tidx = (static_cast<SizeType64>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC_ELTS;
    SizeType64 const stride = static_cast<SizeType64>(blockDim.x) * gridDim.x * VEC_ELTS;
    SizeType64 const srcEndIdx = srcStartIdx + size;

    SizeType64 srcIdx = srcStartIdx + tidx;
    SizeType64 dstIdx = dstStartIdx + tidx;

    for (; srcIdx < srcEndIdx; srcIdx += stride, dstIdx += stride)
    {
        *reinterpret_cast<VecT*>(&dstData[dstIdx]) = *reinterpret_cast<VecT const*>(&srcData[srcIdx]);
    }
}

template <typename T>
__global__ void add(T* data, std::size_t size, T const value)
{
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (auto idx = tidx; idx < size; idx += stride)
    {
        data[idx] += value;
    }
}

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

template <typename T>
void invokeReduce(IBuffer& output, IBuffer const& input, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");
    TLLM_CHECK_WITH_INFO(output.getSize() == 1, common::fmtstr("Output size (%ld) has to be 1", output.getSize()));

    auto outputPtr = bufferCast<T>(output);
    auto inputPtr = bufferCast<T>(input);
    auto const size = input.getSize();

    dim3 const blockSize{std::min(512U, static_cast<std::uint32_t>(size))};
    dim3 const gridSize{1};

    reduceSum<<<gridSize, blockSize, 0, stream.get()>>>(outputPtr, inputPtr, size);
}

__global__ void transposeWithOutputOffset(SizeType32* output, SizeType32 const* input, SizeType32 const nbInputRows,
    SizeType32 const inputRowSize, SizeType32 const outputRowSize, SizeType32 const outputOffset)
{
    SizeType32 const tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    SizeType32 const tidy = (blockIdx.y * blockDim.y) + threadIdx.y;

    for (SizeType32 batchIdx = tidy; batchIdx < nbInputRows; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType32 tokenIdx = tidx; tokenIdx < inputRowSize; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const inputIdx = (batchIdx * inputRowSize) + tokenIdx;
            auto const outputIdx = (tokenIdx * outputRowSize) + outputOffset + batchIdx;
            output[outputIdx] = input[inputIdx];
        }
    }
}

__global__ void buildAttentionMask(SizeType32* attentionMask, SizeType32 const size, SizeType32 const padId)
{
    SizeType32 const tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    for (SizeType32 i = tid; i < size; i += blockDim.x * gridDim.x)
    {
        auto const x = attentionMask[i];
        attentionMask[i] = (x != padId);
    }
}

__global__ void extendAttentionMask(
    SizeType32* newMask, SizeType32 const* oldMask, SizeType32 const batchSize, SizeType32 const seqLength)
{
    SizeType32 const tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    SizeType32 const tidy = (blockIdx.y * blockDim.y) + threadIdx.y;

    for (SizeType32 batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        for (SizeType32 tokenIdx = tidx; tokenIdx < seqLength + 1; tokenIdx += blockDim.x * gridDim.x)
        {
            SizeType32 const oldIndex = (batchIdx * seqLength) + tokenIdx;
            SizeType32 const newIndex = (batchIdx * (seqLength + 1)) + tokenIdx;
            newMask[newIndex] = (tokenIdx < seqLength) ? oldMask[oldIndex] : 1;
        }
    }
}

__global__ void copyInputToOutput(TokenIdType* outputIds, TokenIdType const* inputIds, SizeType32 const* inputLengths,
    TokenIdType const padId, SizeType32 const batchSize, SizeType32 const beamWidth, SizeType32 const maxInputLength,
    SizeType32 const maxSeqLength)
{
    SizeType32 const tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    SizeType32 const tidy = (blockIdx.y * blockDim.y) + threadIdx.y;

    for (SizeType32 batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const inputLength = inputLengths[batchIdx];
        for (SizeType32 tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[(batchIdx * maxInputLength) + tokenIdx] : padId;
            for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(batchIdx, beamIdx, tokenIdx, beamWidth, maxSeqLength);
                outputIds[outputIdx] = value;
            }
        }
    }
}

// In the following kernel, we launch a grid with batchSize blocks of threads. Each thread block
// copies the logits from the "logits" tensor to the "lastTokenLogits" tensor for the last token
// of each sequence.
//
// TODO: Enable vector copies for higher BW utilization.

template <typename T>
__global__ void gatherLastTokenLogitsKernel(
    T* lastTokenLogits, T const* logits, int const* lastTokenIds, int beamWidth, int vocabSizePadded)
{
    // This sequence.
    int const seqIdx = blockIdx.x;
    // Find the index of the last token in that sequence.
    // Since lastTokenIds is the accumulated length instead of real ids, so we need to minus 1.
    // For length [11, 23], we hope to get the results of id 10 and 22, in fact.
    int const lastTokenIdx = lastTokenIds[seqIdx] - 1;

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
            lastTokenLogitsPtr[(beamIdx * vocabSizePadded) + idx] = value;
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

    TLLM_CHECK_WITH_INFO(inputShape.d[0] == batchSize, "Invalid input shape: dim[0]");
    TLLM_CHECK_WITH_INFO(inputShape.d[2] == vocabSizePadded, "Invalid input shape: dim[2]");

    dim3 const blockSize{256, 1};
    dim3 const gridSize{static_cast<std::uint32_t>(batchSize), 1};
    gatherLastTokenLogitsKernel<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<T>(output), bufferCast<T>(input),
        bufferCast<int32_t>(lastTokenIds), static_cast<std::uint32_t>(beamWidth), vocabSizePadded);
}

__global__ void copyPackedInputToOutput(TokenIdType* outputIds, TokenIdType const* inputIds,
    SizeType32 const* inputOffsets, TokenIdType const padId, SizeType32 const batchSize, SizeType32 const beamWidth,
    SizeType32 const maxInputLength, SizeType32 const maxSeqLength)
{
    SizeType32 const tidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    SizeType32 const tidy = (blockIdx.y * blockDim.y) + threadIdx.y;

    for (SizeType32 batchIdx = tidy; batchIdx < batchSize; batchIdx += blockDim.y * gridDim.y)
    {
        auto const tokenBegin = inputOffsets[batchIdx];
        auto const tokenEnd = inputOffsets[batchIdx + 1];
        auto const inputLength = tokenEnd - tokenBegin;

        for (SizeType32 tokenIdx = tidx; tokenIdx < maxInputLength; tokenIdx += blockDim.x * gridDim.x)
        {
            auto const value = (tokenIdx < inputLength) ? inputIds[tokenBegin + tokenIdx] : padId;
            for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = tc::flat_index3(batchIdx, beamIdx, tokenIdx, beamWidth, maxSeqLength);
                outputIds[outputIdx] = value;
            }
        }
    }
}

template <typename T>
__global__ void scatterTensor(T* output, T const* input, std::uint32_t const batchSize,
    std::uint32_t const inputRowSize, std::size_t const outputRowSize, std::uint32_t const beamWidth)
{
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const tidy = (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    for (auto batchIdx = tidy; batchIdx < batchSize; batchIdx += stridey)
    {
        for (auto columnIdx = tidx; columnIdx < inputRowSize; columnIdx += stridex)
        {
            auto const inputIdx = (batchIdx * inputRowSize) + columnIdx;
            auto const value = input[inputIdx];
            std::size_t constexpr beamIdx{0};
            auto const outputIdx = ((batchIdx * beamWidth + beamIdx) * outputRowSize) + columnIdx;
            output[outputIdx] = value;
        }
    }
}

template <typename T>
__global__ void tileTensor(T* output, T const* input, std::uint32_t const batchSize, std::size_t const inputRowSize,
    std::size_t const outputRowSize, std::uint32_t const beamWidth)
{
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    auto const tidy = (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
    auto const stridex = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    auto const stridey = static_cast<std::size_t>(blockDim.y) * gridDim.y;

    for (auto batchIdx = tidy; batchIdx < batchSize; batchIdx += stridey)
    {
        for (auto columnIdx = tidx; columnIdx < inputRowSize; columnIdx += stridex)
        {
            auto const inputIdx = (batchIdx * inputRowSize) + columnIdx;
            auto const value = input[inputIdx];
            for (std::size_t beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const outputIdx = ((batchIdx * beamWidth + beamIdx) * outputRowSize) + columnIdx;
                output[outputIdx] = value;
            }
        }
    }
}

template <typename T>
void invokeScatterTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream)
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

template <typename T>
void invokeTileTensor(ITensor& output, ITensor const& input, SizeType32 const beamWidth, CudaStream const& stream)
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

// In the following kernel, we launch a grid with (microBatchSize * beamWidth, outputLen) blocks of threads. Each thread
// block copies a `vocabSizePadded` length logits tensor from the "inputLogits (microBatchSize, beamWidth,
// vocabSizePadded)" to the "outputGenerationLogits (batchSize, beamWidth, outputLen, vocabSizePadded)"
template <typename T>
__global__ void mergeLogitsFragmentsKernel(T* output, T** fragmentsVector, int const outputLen, int firstBatchSlotIdx,
    int beamWidth, int vocabSizePadded, int stepOffset)
{
    // output: shape: [batchSize, beamWidth, outputLen, vocabSize]
    // inputVecor.at(i): shape: [microBatchSize, beamWidth, vocabSize]

    // Current step
    int const curStep = blockIdx.y;

    // The relatively batch slot index that this thread block in microBatchSize.
    int const relativeBatchSlotIdx = blockIdx.x / beamWidth;

    // The Absolute batch slot index in batchSize.
    int const absoluteBatchSlotIdx = firstBatchSlotIdx + relativeBatchSlotIdx;

    // The beam index that this thread block process
    int const mbeamIdx = blockIdx.x % beamWidth;

    // The output pointer
    unsigned int const outputOffset
        = (absoluteBatchSlotIdx * beamWidth * outputLen + mbeamIdx * outputLen + curStep + stepOffset)
        * vocabSizePadded;

    T* outputPtr = &output[outputOffset];

    unsigned int const inputOffset = (relativeBatchSlotIdx * beamWidth + mbeamIdx) * vocabSizePadded;
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
    std::vector<TensorPtr> const& fragmentsVector, ITensor& cachePointerDevice, ITensor& cachePointerHost,
    SizeType32 firstBatchSlotIdx, SizeType32 microBatchSize, SizeType32 beamWidth, CudaStream const& stream,
    int stepOffset)
{
    size_t const fragmentsVectorSize = fragmentsVector.size();

    auto cachePointerHostPtr = bufferCast<T*>(cachePointerHost);

    for (int i = 0; i < fragmentsVectorSize; i++)
    {
        cachePointerHostPtr[i] = bufferCast<T>(*fragmentsVector.at(i));
    }
    bufferManager.copy(cachePointerHost, cachePointerDevice);

    dim3 const blockSize(256);
    dim3 const gridSize{(unsigned int) (microBatchSize * beamWidth), (unsigned int) (fragmentsVectorSize)};

    auto const& outputShape = output.getShape();
    auto const vocabSizePadded = static_cast<SizeType32>(outputShape.d[outputShape.nbDims - 1]);
    auto const outputLen = static_cast<SizeType32>(outputShape.d[outputShape.nbDims - 2]);

    TLLM_CHECK_WITH_INFO(outputLen >= fragmentsVectorSize, "Fragments size does not match outputLen size");

    mergeLogitsFragmentsKernel<T><<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<T>(output),
        bufferCast<T*>(cachePointerDevice), outputLen, firstBatchSlotIdx, beamWidth, vocabSizePadded, stepOffset);
}

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType32 const maxInputLength, TokenIdType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const& outputShape = outputIds.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 3, common::fmtstr("Output shape must have 3 dimensions, but has %d", outputShape.nbDims));

    auto const batchSize = static_cast<SizeType32>(inputOffsets.getSize()) - 1;
    SizeType32 const beamWidth = outputShape.d[1];
    SizeType32 const maxSeqLength = outputShape.d[2];

    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[0],
        common::fmtstr("Output ids batch size (" FMT_DIM ") does not match inputOffsets batch size (%d)",
            outputShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyPackedInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<TokenIdType>(outputIds),
        bufferCast<TokenIdType const>(inputIds), bufferCast<SizeType32 const>(inputOffsets), padId, batchSize,
        beamWidth, maxInputLength, maxSeqLength);
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
template void invokeFill(IBuffer&, SizeType64, CudaStream const&);
template void invokeFill(IBuffer&, std::int32_t, CudaStream const&);
template void invokeFill(IBuffer&, std::int8_t, CudaStream const&);
template void invokeFill(IBuffer&, std::uint8_t, CudaStream const&);
template void invokeFill(IBuffer&, bool, CudaStream const&);
template void invokeFill(IBuffer&, half, CudaStream const&);
template void invokeFill(IBuffer&, float, CudaStream const&);
#ifdef ENABLE_BF16
template void invokeFill(IBuffer&, __nv_bfloat16, CudaStream const&);
#endif // ENABLE_BF16

void invokeFillBatch(IBuffer& buffer, IBuffer const& slotIndices, std::size_t slotStride, IBuffer const& values,
    CudaStream const& stream)
{
    switch (buffer.getDataType())
    {
    case nvinfer1::DataType::kINT32:
        invokeFillBatch<std::int32_t>(buffer, slotIndices, slotStride, values, stream);
        break;
    case nvinfer1::DataType::kINT8:
        invokeFillBatch<std::int8_t>(buffer, slotIndices, slotStride, values, stream);
        break;
    case nvinfer1::DataType::kFLOAT: invokeFillBatch<float>(buffer, slotIndices, slotStride, values, stream); break;
    default: TLLM_THROW("data type not supported");
    }
}

void invokeGatherBatch(IBuffer& buffer, IBuffer const& values, IBuffer const& slotIndices, std::size_t slotStride,
    CudaStream const& stream)
{
    switch (buffer.getDataType())
    {
    case nvinfer1::DataType::kINT32:
        invokeGatherBatch<std::int32_t>(buffer, values, slotIndices, slotStride, stream);
        break;
    case nvinfer1::DataType::kINT8:
        invokeGatherBatch<std::int8_t>(buffer, values, slotIndices, slotStride, stream);
        break;
    case nvinfer1::DataType::kFLOAT: invokeGatherBatch<float>(buffer, values, slotIndices, slotStride, stream); break;
    default: TLLM_THROW("data type not supported");
    }
}

void invokeCopyBatch(IBuffer const& srcBuffer, IBuffer& dstBuffer, IBuffer const& srcOffsets, IBuffer const& dstOffsets,
    IBuffer const& sizes, std::size_t maxStride, CudaStream const& stream)
{
    auto const* srcDataPtr = reinterpret_cast<uint8_t const*>(srcBuffer.data());
    auto* dstDataPtr = reinterpret_cast<uint8_t*>(dstBuffer.data());
    auto const* srcOffsetsPtr = bufferCast<SizeType64>(srcOffsets);
    auto const* dstOffsetsPtr = bufferCast<SizeType64>(dstOffsets);
    auto const* sizesPtr = bufferCast<SizeType64>(sizes);
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
        srcDataPtr, dstDataPtr, srcOffsetsPtr, dstOffsetsPtr, sizesPtr, static_cast<SizeType64>(dataTypeSize));
}

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

void reduce(IBuffer& output, IBuffer const& input, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeReduce<SizeType32>(output, input, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeReduce<float>(output, input, stream); break;
    case nvinfer1::DataType::kHALF: invokeReduce<half>(output, input, stream); break;
    case nvinfer1::DataType::kINT8: invokeReduce<int8_t>(output, input, stream); break;
    default: TLLM_THROW("data type not supported");
    }
}

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType32 const outputOffset, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(input.getDataType() == output.getDataType(), "Input and output have different data types");

    auto const& inputShape = input.getShape();
    TLLM_CHECK_WITH_INFO(
        inputShape.nbDims == 2, common::fmtstr("Input shape must have 2 dimensions, but has %d", inputShape.nbDims));
    SizeType32 const nbInputRows = inputShape.d[0];
    SizeType32 const inputRowSize = inputShape.d[1];

    auto const& outputShape = output.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 2, common::fmtstr("Output shape must have 2 dimensions, but has %d", outputShape.nbDims));
    SizeType32 const nbOutputRows = outputShape.d[0];
    SizeType32 const outputRowSize = outputShape.d[1];

    TLLM_CHECK_WITH_INFO(inputRowSize == nbOutputRows,
        common::fmtstr("Input dim 1 (%d) and output dim 0 (%d) differ", inputRowSize, nbOutputRows));
    TLLM_CHECK_WITH_INFO(outputOffset + nbInputRows <= outputRowSize,
        common::fmtstr("Input (%d rows) does not fit into output (%d columns, offset %d)", nbInputRows, inputRowSize,
            outputOffset));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((inputRowSize + blockSize.x - 1) / blockSize.x, nbInputRows);

    transposeWithOutputOffset<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType32>(output),
        bufferCast<SizeType32 const>(input), nbInputRows, inputRowSize, outputRowSize, outputOffset);
}

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream)
{
    auto const size = input.getSize();
    auto const* inputData = bufferCast<SizeType32>(input);
    auto* outputData = bufferCast<SizeType32>(output);

    std::size_t tempStorageBytes{0};
    cub::DeviceScan::InclusiveSum(nullptr, tempStorageBytes, inputData, outputData, size, stream.get());
    auto tempStorage = manager.gpu(tempStorageBytes, nvinfer1::DataType::kUINT8);
    auto* tempStorageData = bufferCast<std::uint8_t>(*tempStorage);
    cub::DeviceScan::InclusiveSum(tempStorageData, tempStorageBytes, inputData, outputData, size, stream.get());
}

void invokeBuildAttentionMask(ITensor& attentionMask, SizeType32 const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        TRTDataType<SizeType32>::value == attentionMask.getDataType(), "attentionMask has wrong data type");

    auto const size = attentionMask.getSize();
    dim3 const blockSize(256);
    dim3 const gridSize((size + blockSize.x - 1) / blockSize.x);

    buildAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<SizeType32>(attentionMask), size, padId);
}

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType32>::value == newMask.getDataType(), "attentionMask has wrong data type");
    TLLM_CHECK_WITH_INFO(TRTDataType<SizeType32>::value == oldMask.getDataType(), "attentionMask has wrong data type");

    auto const& shape = oldMask.getShape();
    SizeType32 const batchSize = shape.d[0];
    SizeType32 const seqLength = shape.d[1];

    dim3 const blockSize(256, 1);
    dim3 const gridSize((seqLength + blockSize.x - 1) / blockSize.x, batchSize);

    extendAttentionMask<<<gridSize, blockSize, 0, stream.get()>>>(
        bufferCast<SizeType32>(newMask), bufferCast<SizeType32>(oldMask), batchSize, seqLength);
}

void invokeCopyInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    TokenIdType const padId, CudaStream const& stream)
{
    TLLM_CHECK_WITH_INFO(
        inputIds.getDataType() == outputIds.getDataType(), "Input and output have different data types");

    auto const& inputShape = inputIds.getShape();
    auto const& outputShape = outputIds.getShape();
    TLLM_CHECK_WITH_INFO(
        outputShape.nbDims == 3, common::fmtstr("Output shape must have 3 dimensions, but has %d", outputShape.nbDims));

    auto const batchSize = static_cast<SizeType32>(inputLengths.getSize());
    SizeType32 const maxInputLength = inputShape.d[inputShape.nbDims - 1];
    SizeType32 const beamWidth = outputShape.d[1];
    SizeType32 const maxSeqLength = outputShape.d[2];

    auto const inputBatchSize = inputIds.getSize() / maxInputLength;
    TLLM_CHECK_WITH_INFO(std::size_t(batchSize) == inputBatchSize,
        common::fmtstr("Input ids batch size (%ld) does not match inputLengths size (%ld)", inputBatchSize,
            std::size_t(batchSize)));
    TLLM_CHECK_WITH_INFO(batchSize == outputShape.d[0],
        common::fmtstr(
            "Output ids batch size (" FMT_DIM ") does not match inputLengths size (%d)", outputShape.d[0], batchSize));
    TLLM_CHECK_WITH_INFO(maxInputLength < maxSeqLength,
        common::fmtstr(
            "Output sequence length (%d) has to be larger than max input length (%d)", maxSeqLength, maxInputLength));

    dim3 const blockSize(256, 1);
    dim3 const gridSize((maxInputLength + blockSize.x - 1) / blockSize.x, batchSize);

    copyInputToOutput<<<gridSize, blockSize, 0, stream.get()>>>(bufferCast<TokenIdType>(outputIds),
        bufferCast<TokenIdType const>(inputIds), bufferCast<SizeType32 const>(inputLengths), padId, batchSize,
        beamWidth, maxInputLength, maxSeqLength);
}

void initOutputIds(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    ITensor const& inputOffsets, TokenIdType const padId, TokenIdType const endId, SizeType32 const maxInputLength,
    bool const inputPacked, CudaStream const& stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    kernels::invokeFill(outputIds, endId, stream);

    if (inputPacked)
    {
        invokeCopyPackedInputToOutput(outputIds, inputIds, inputOffsets, maxInputLength, padId, stream);
    }
    else
    {
        kernels::invokeCopyInputToOutput(outputIds, inputIds, inputLengths, padId, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void scatterTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeScatterTensor<SizeType32>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeScatterTensor<float>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kHALF: invokeScatterTensor<half>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kINT8: invokeScatterTensor<int8_t>(output, input, beamWidth, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeScatterTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
#endif // ENABLE_FP8
    default: TLLM_THROW("data type not supported");
    }
}

void tileTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream)
{
    switch (input.getDataType())
    {
    case nvinfer1::DataType::kINT32: invokeTileTensor<SizeType32>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kFLOAT: invokeTileTensor<float>(output, input, beamWidth, stream); break;
    case nvinfer1::DataType::kHALF: invokeTileTensor<half>(output, input, beamWidth, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: invokeTileTensor<__nv_bfloat16>(output, input, beamWidth, stream); break;
#endif // ENABLE_BF16
    case nvinfer1::DataType::kINT8: invokeTileTensor<int8_t>(output, input, beamWidth, stream); break;
#ifdef ENABLE_FP8
    case nvinfer1::DataType::kFP8: invokeTileTensor<__nv_fp8_e4m3>(output, input, beamWidth, stream); break;
#endif // ENABLE_FP8
    default: TLLM_THROW("data type not supported");
    }
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
    default: TLLM_THROW("data type not supported");
    }
}

void mergeLogitsFragments(BufferManager const& bufferManager, ITensor& output,
    std::vector<TensorPtr> const& fragmentsVector, ITensor& cachePointerDevice, ITensor& cachePointerHost,
    SizeType32 firstBatchSlotIdx, SizeType32 const microBatchSize, SizeType32 const beamWidth, CudaStream const& stream,
    int stepOffset)
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
    default: TLLM_THROW("data type not supported");
    }
}

void invokeUpdateKVBlockArrayDraftTokenLocation(ITensor const& seqAcceptedDraftTokenOffsets,
    ITensor const& packedAcceptedDraftTokensIndices, ITensor const& pastKeyValueLengths, void* const* pointerArray,
    ::tensorrt_llm::kernels::KVCacheIndex const* offsetArray, SizeType32 layerCount, SizeType32 seqCount,
    SizeType32 numKVHeads, SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, ITensor const& seqSlotRemapping, ITensor const& batchSlots,
    SizeType32 maxKVCacheLen, SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, bool canUseOneMoreBlock,
    cudaStream_t stream)
{
    tensorrt_llm::kernels::speculative_decoding::updateKVBlockArrayDraftTokenLocation(
        bufferCast<SizeType32>(seqAcceptedDraftTokenOffsets), bufferCast<SizeType32>(packedAcceptedDraftTokensIndices),
        bufferCast<SizeType32>(pastKeyValueLengths), pointerArray, offsetArray, layerCount, seqCount, numKVHeads,
        sizeInBytesPerKVHead, rewindDraftTokenCommonCount, rewindDraftTokenSeparateAdjustments,
        bufferCast<SizeType32>(seqSlotRemapping), bufferCast<SizeType32>(batchSlots), maxKVCacheLen, maxBlocksPerSeq,
        tokensPerBlock, canUseOneMoreBlock, stream);
}

} // namespace tensorrt_llm::runtime::kernels
