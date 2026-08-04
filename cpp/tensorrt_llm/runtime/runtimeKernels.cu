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
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <NvInferRuntimeBase.h>
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
    SizeType32 const* rewindDraftTokenSeparateAdjustments, ITensor const& batchSlots, SizeType32 maxKVCacheLen,
    SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, bool canUseOneMoreBlock, cudaStream_t stream)
{
    tensorrt_llm::kernels::speculative_decoding::updateKVBlockArrayDraftTokenLocation(
        bufferCast<SizeType32>(seqAcceptedDraftTokenOffsets), bufferCast<SizeType32>(packedAcceptedDraftTokensIndices),
        bufferCast<SizeType32>(pastKeyValueLengths), pointerArray, offsetArray, layerCount, seqCount, numKVHeads,
        sizeInBytesPerKVHead, rewindDraftTokenCommonCount, rewindDraftTokenSeparateAdjustments, nullptr,
        bufferCast<SizeType32>(batchSlots), maxKVCacheLen, maxBlocksPerSeq, tokensPerBlock, canUseOneMoreBlock, stream);
}

} // namespace tensorrt_llm::runtime::kernels
