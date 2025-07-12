/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::kernels
{
class KVCacheIndex;
} // namespace tensorrt_llm::kernels

namespace tensorrt_llm::runtime::kernels
{

using TensorPtr = runtime::ITensor::SharedPtr;

template <typename T>
void invokeFill(IBuffer& buffer, T value, CudaStream const& stream);

void invokeFillBatch(
    IBuffer& buffer, IBuffer const& indices, std::size_t stride, IBuffer const& values, CudaStream const& stream);

void invokeGatherBatch(IBuffer& buffer, IBuffer const& values, IBuffer const& slotIndices, std::size_t slotStride,
    CudaStream const& stream);

void invokeCopyBatch(IBuffer const& srcBuffer, IBuffer& dstBuffer, IBuffer const& srcOffsets, IBuffer const& dstOffsets,
    IBuffer const& sizes, std::size_t maxStride, CudaStream const& stream);

void scatterTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream);

void tileTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream);

void mergeLogitsFragments(BufferManager const& bufferManager, ITensor& output,
    std::vector<TensorPtr> const& fragmentsVector, ITensor& cachePointerDevice, ITensor& cachePointerHost,
    SizeType32 firstBatchSlotIdx, SizeType32 microBatchSize, SizeType32 beamWidth, CudaStream const& stream,
    int stepOffset);

void invokeUpdateKVBlockArrayDraftTokenLocation(ITensor const& seqAcceptedDraftTokenOffsets,
    ITensor const& packedAcceptedDraftTokensIndices, ITensor const& pastKeyValueLengths, void* const* pointerArray,
    ::tensorrt_llm::kernels::KVCacheIndex const* offsetArray, SizeType32 layerCount, SizeType32 seqCount,
    SizeType32 numKVHeads, SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, ITensor const& batchSlots, SizeType32 maxKVCacheLen,
    SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, bool canUseOneMoreBlock, cudaStream_t stream);

} // namespace tensorrt_llm::runtime::kernels
