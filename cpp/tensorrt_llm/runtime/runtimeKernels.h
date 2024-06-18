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

#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime::kernels
{

using TensorPtr = runtime::ITensor::SharedPtr;

template <typename T>
void invokeFill(IBuffer& buffer, T value, CudaStream const& stream);

template <typename T>
void invokeFillBatch(
    IBuffer& buffer, IBuffer const& indices, std::size_t stride, IBuffer const& values, CudaStream const& stream);

void invokeCopyBatch(IBuffer const& srcBuffer, IBuffer& dstBuffer, IBuffer const& srcOffsets, IBuffer const& dstOffsets,
    IBuffer const& sizes, std::size_t maxStride, CudaStream const& stream);

template <typename T>
void invokeAdd(IBuffer& buffer, T value, CudaStream const& stream);

void reduce(IBuffer& output, IBuffer const& input, CudaStream const& stream);

void invokeTranspose(ITensor& output, ITensor const& input, CudaStream const& stream);

void invokeTransposeWithOutputOffset(
    ITensor& output, ITensor const& input, SizeType32 outputOffset, CudaStream const& stream);

void invokeTransposeWithInputOffset(
    ITensor& output, ITensor const& input, SizeType32 inputOffset, CudaStream const& stream);

void invokeInclusiveSum(IBuffer& output, IBuffer const& input, BufferManager const& manager, CudaStream const& stream);

void invokeInclusiveSum(IBuffer& output, IBuffer& tmpBuffer, IBuffer const& input, CudaStream const& stream);

void invokeBuildTokenMask(
    ITensor& tokenMask, ITensor const& inputLengths, SizeType32 maxInputLength, CudaStream const& stream);

void invokeBuildAttentionMask(ITensor& attentionMask, TokenIdType padId, CudaStream const& stream);

void invokeExtendAttentionMask(ITensor& newMask, ITensor const& oldMask, CudaStream const& stream);

void invokeCopyInputToOutputTransposed(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    TokenIdType padId, CudaStream const& stream);

void invokeCopyPackedInputToOutputTransposed(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType32 maxInputLength, TokenIdType padId, CudaStream const& stream);

void invokeCopyInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths, SizeType32 padId,
    CudaStream const& stream);

void invokeCopyPackedInputToOutput(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputOffsets,
    SizeType32 maxInputLength, TokenIdType padId, CudaStream const& stream);

void initOutputIds(ITensor& outputIds, ITensor const& inputIds, ITensor const& inputLengths,
    ITensor const& inputOffsets, TokenIdType padId, TokenIdType endId, SizeType32 maxInputLength, bool inputPacked,
    CudaStream const& stream);

void scatterTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream);

void splitTransposed(ITensor& output, ITensor const& input, SizeType32 split, CudaStream const& stream);

void tileTensor(ITensor& output, ITensor const& input, SizeType32 beamWidth, CudaStream const& stream);

void tileTensorInplace(ITensor& tensor, SizeType32 beamWidth, CudaStream const& stream);

void gatherLastTokenLogits(
    ITensor& output, ITensor const& input, ITensor const& lastTokenIds, CudaStream const& stream);

void copyLatestTokenLogitsInGeneration(ITensor& output, ITensor const& input, SizeType32 step,
    SizeType32 firstBatchSlotIdx, SizeType32 microBatchSize, SizeType32 beamWidth, CudaStream const& stream);

void mergeLogitsFragments(BufferManager const& bufferManager, ITensor& output, std::vector<TensorPtr> inputVector,
    ITensor& cachePointerDevice, ITensor& cachePointerHost, SizeType32 firstBatchSlotIdx,
    SizeType32 const microBatchSize, SizeType32 const beamWidth, CudaStream const& stream, int stepOffset);

void invokeUpdateKVBlockArrayDraftTokenLocation(ITensor const& seqAcceptedDraftTokenOffsets,
    ITensor const& packedAcceptedDraftTokensIndices, ITensor const& pastKeyValueLengths, void* const* pointerArray,
    ::tensorrt_llm::kernels::KVCacheIndex const* offsetArray, SizeType32 layerCount, SizeType32 seqCount,
    SizeType32 numKVHeads, SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, ITensor const& seqSlotRemapping, ITensor const& batchSlots,
    SizeType32 maxKVCacheLen, SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, cudaStream_t stream);
} // namespace tensorrt_llm::runtime::kernels
