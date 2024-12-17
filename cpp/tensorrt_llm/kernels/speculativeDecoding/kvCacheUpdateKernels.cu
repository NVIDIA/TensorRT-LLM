/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "kvCacheUpdateKernels.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <array>
#include <vector>

namespace tensorrt_llm::kernels::speculative_decoding
{

using namespace tensorrt_llm::runtime;
static constexpr SizeType32 kUpdateKVCacheKernelShmSize = 16384;

namespace
{
template <typename KVCacheBuffer, SizeType32 MaxLayerCount, typename MoveEltType>
__global__ void updateKVCacheDraftTokenLocationBatchedKernel(std::array<KVCacheBuffer, MaxLayerCount> kvCacheBuffers,
    SizeType32 const* seqAcceptedDraftTokenOffsets, IndexType const* packedAcceptedDraftTokensIndices,
    SizeType32 const* pastKeyValueLengths, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, SizeType32 const* seqSlotRemapping,
    SizeType32 const* batchSlots, SizeType32 eltCountPerHead)
{
    auto const seqIdx = static_cast<SizeType32>(blockIdx.x);
    auto const headIdx = static_cast<SizeType32>(blockIdx.y);
    auto const layerIdx = static_cast<SizeType32>(blockIdx.z);
    auto const warpIdx = static_cast<SizeType32>(threadIdx.x / 32);
    auto const warpCount = static_cast<SizeType32>(blockDim.x / 32);
    auto const laneIdx = static_cast<SizeType32>(threadIdx.x & 0x1f);
    auto const seqDraftTokenStart = seqAcceptedDraftTokenOffsets[seqIdx];
    auto const seqDraftTokenEnd = seqAcceptedDraftTokenOffsets[seqIdx + 1];
    auto const seqSlot = seqSlotRemapping == nullptr ? seqIdx : seqSlotRemapping[seqIdx];
    auto const seqDraftCount = seqDraftTokenEnd - seqDraftTokenStart;
    auto const maxEltCountPerMove
        = static_cast<SizeType32>(kUpdateKVCacheKernelShmSize / sizeof(MoveEltType) / seqDraftCount);
    auto const eltCountPerMove = min(maxEltCountPerMove, eltCountPerHead);
    if (seqDraftCount == 0 || eltCountPerMove == 0)
    {
        return;
    }
    KVCacheBuffer& kvCacheBuffer = kvCacheBuffers[layerIdx];
    auto tokenStartIdx = pastKeyValueLengths[seqSlot] - rewindDraftTokenCommonCount;
    if (rewindDraftTokenSeparateAdjustments != nullptr)
    {
        auto const batchSlot = batchSlots == nullptr ? seqIdx : batchSlots[seqIdx];
        tokenStartIdx -= rewindDraftTokenSeparateAdjustments[batchSlot];
    }
    __shared__ char loadSmemBuffer[kUpdateKVCacheKernelShmSize];
    auto* eltLoadSmemBuffer = reinterpret_cast<MoveEltType*>(&loadSmemBuffer[0]);
    for (SizeType32 startChannelOffset = 0; startChannelOffset < eltCountPerHead; startChannelOffset += eltCountPerMove)
    {
        SizeType32 eltCountCurrentMove = min(eltCountPerMove, eltCountPerHead - startChannelOffset);
        // load K
        for (SizeType32 tokenIdx = warpIdx; tokenIdx < seqDraftCount; tokenIdx += warpCount)
        {
            auto const tokenPos = packedAcceptedDraftTokensIndices[seqDraftTokenStart + tokenIdx];
            auto* tokenSmemBuffer = eltLoadSmemBuffer + tokenIdx * eltCountCurrentMove;
            auto const tokenKVPosition = tokenStartIdx + tokenPos;
            auto* kPtr = reinterpret_cast<MoveEltType*>(kvCacheBuffer.getKBlockPtr(seqSlot, tokenKVPosition));
            for (SizeType32 loadChannelIdx = laneIdx; loadChannelIdx < eltCountCurrentMove; loadChannelIdx += 32)
            {
                auto const channelIdx = loadChannelIdx + startChannelOffset;
                auto const kvLocationIdx
                    = kvCacheBuffer.getKVLocalIdx(tokenKVPosition, headIdx, eltCountPerHead, channelIdx);
                tokenSmemBuffer[loadChannelIdx] = kPtr[kvLocationIdx];
            }
        }
        __syncthreads();
        // store K
        for (SizeType32 tokenIdx = warpIdx; tokenIdx < seqDraftCount; tokenIdx += warpCount)
        {
            auto const tokenPos = tokenIdx;
            auto* tokenSmemBuffer = eltLoadSmemBuffer + tokenIdx * eltCountCurrentMove;
            auto const tokenKVPosition = tokenStartIdx + tokenPos;
            auto* kPtr = reinterpret_cast<MoveEltType*>(kvCacheBuffer.getKBlockPtr(seqSlot, tokenKVPosition));
            for (SizeType32 loadChannelIdx = laneIdx; loadChannelIdx < eltCountCurrentMove; loadChannelIdx += 32)
            {
                auto const channelIdx = loadChannelIdx + startChannelOffset;
                auto const kvLocationIdx
                    = kvCacheBuffer.getKVLocalIdx(tokenKVPosition, headIdx, eltCountPerHead, channelIdx);
                kPtr[kvLocationIdx] = tokenSmemBuffer[loadChannelIdx];
            }
        }
        __syncthreads();
        // load V
        for (SizeType32 tokenIdx = warpIdx; tokenIdx < seqDraftCount; tokenIdx += warpCount)
        {
            auto const tokenPos = packedAcceptedDraftTokensIndices[seqDraftTokenStart + tokenIdx];
            auto* tokenSmemBuffer = eltLoadSmemBuffer + tokenIdx * eltCountCurrentMove;
            auto const tokenKVPosition = tokenStartIdx + tokenPos;
            auto* vPtr = reinterpret_cast<MoveEltType*>(kvCacheBuffer.getVBlockPtr(seqSlot, tokenKVPosition));
            for (SizeType32 loadChannelIdx = laneIdx; loadChannelIdx < eltCountCurrentMove; loadChannelIdx += 32)
            {
                auto const channelIdx = loadChannelIdx + startChannelOffset;
                auto const kvLocationIdx
                    = kvCacheBuffer.getKVLocalIdx(tokenKVPosition, headIdx, eltCountPerHead, channelIdx);
                tokenSmemBuffer[loadChannelIdx] = vPtr[kvLocationIdx];
            }
        }
        __syncthreads();
        // store V
        for (SizeType32 tokenIdx = warpIdx; tokenIdx < seqDraftCount; tokenIdx += warpCount)
        {
            auto const tokenPos = tokenIdx;
            auto* tokenSmemBuffer = eltLoadSmemBuffer + tokenPos * eltCountCurrentMove;
            auto const tokenKVPosition = tokenStartIdx + tokenPos;
            auto* vPtr = reinterpret_cast<MoveEltType*>(kvCacheBuffer.getVBlockPtr(seqSlot, tokenKVPosition));
            for (SizeType32 loadChannelIdx = laneIdx; loadChannelIdx < eltCountCurrentMove; loadChannelIdx += 32)
            {
                auto const channelIdx = loadChannelIdx + startChannelOffset;
                auto const kvLocationIdx
                    = kvCacheBuffer.getKVLocalIdx(tokenKVPosition, headIdx, eltCountPerHead, channelIdx);
                vPtr[kvLocationIdx] = tokenSmemBuffer[loadChannelIdx];
            }
        }
        __syncthreads();
    }
}
} // namespace

template <typename KVCacheBuffer, SizeType32 MaxLayerCount>
void updateKVCacheDraftTokenLocationBatched(KVCacheBuffer const* kvCacheBuffers,
    SizeType32 const* seqAcceptedDraftTokenOffsets, IndexType const* packedAcceptedDraftTokensIndices,
    SizeType32 const* pastKeyValueLengths, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, SizeType32 const* seqSlotRemapping,
    SizeType32 const* batchSlots, cudaStream_t stream)
{
    // make sure launch buffer is enough
    static_assert(MaxLayerCount * sizeof(KVCacheBuffer) <= 3072);
    if (seqCount == 0 || layerCount == 0)
    {
        return;
    }
    SizeType32 alignedBytes = 16;
    while (alignedBytes > 0 && (sizeInBytesPerKVHead % alignedBytes != 0))
    {
        alignedBytes >>= 1;
    }
    TLLM_CHECK_WITH_INFO(alignedBytes > 0, "alignedByte should be positive");
    SizeType32 eltCountPerHead = sizeInBytesPerKVHead / alignedBytes;
    dim3 grid(seqCount, numKVHeads, layerCount);
    dim3 block(128, 1, 1);
    std::array<KVCacheBuffer, MaxLayerCount> kvCacheBufferArray;
    for (SizeType32 i = 0; i < layerCount; i++)
    {
        kvCacheBufferArray[i] = kvCacheBuffers[i];
    }
    void (*pKernelFunc)(std::array<KVCacheBuffer, MaxLayerCount>, SizeType32 const*, IndexType const*,
        SizeType32 const*, SizeType32, SizeType32 const*, SizeType32 const*, SizeType32 const*, SizeType32)
        = nullptr;
    switch (alignedBytes)
    {
    case 16:
    {
        pKernelFunc = &updateKVCacheDraftTokenLocationBatchedKernel<KVCacheBuffer, MaxLayerCount, int4>;
        break;
    }
    case 8:
    {
        pKernelFunc = &updateKVCacheDraftTokenLocationBatchedKernel<KVCacheBuffer, MaxLayerCount, int64_t>;
        break;
    }
    case 4:
    {
        pKernelFunc = &updateKVCacheDraftTokenLocationBatchedKernel<KVCacheBuffer, MaxLayerCount, SizeType32>;
        break;
    }
    case 2:
    {
        pKernelFunc = &updateKVCacheDraftTokenLocationBatchedKernel<KVCacheBuffer, MaxLayerCount, int16_t>;
        break;
    }
    default:
    {
        TLLM_CHECK_WITH_INFO(alignedBytes == 1, "Strange alignedBytes");
        pKernelFunc = &updateKVCacheDraftTokenLocationBatchedKernel<KVCacheBuffer, MaxLayerCount, int8_t>;
        break;
    }
    }
    pKernelFunc<<<grid, block, 0, stream>>>(kvCacheBufferArray, seqAcceptedDraftTokenOffsets,
        packedAcceptedDraftTokensIndices, pastKeyValueLengths, rewindDraftTokenCommonCount,
        rewindDraftTokenSeparateAdjustments, seqSlotRemapping, batchSlots, eltCountPerHead);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

/*!
 * Update KV cache for parallel decoding algorithms.
 * In following examples, we assume we have 2 sequences, accepted count is [3, 2]
 * @tparam KVCacheBuffer : Type of KV cache, should be LinearKVCache or KVBlockArray
 * @param kvCacheBuffers : list of KVCacheBuffer object
 * @param seqAcceptedDraftTokenOffsets : Array of length seqCount + 1, like [0, 3, 5]
 * @param packedAcceptedDraftTokensIndices : Array of length seqAcceptedDraftTokenOffsets[seqCount], each value is in
 * range [0, maxDraftTokenCount - 1]
 * @param pastKeyValueLengths : Array of length seqCount, meaning how many tokens are already in KV cache
 * @param seqCount : Count of sequence
 * @param numKVHeads : Number of KV heads
 * @param sizeInBytesPerKVHead : Size of each KV head
 * @param rewindDraftTokenCommonCount : Common count to rewind
 * @param rewindDraftTokenSeparateAdjustments : Separate adjustment to rewind for each sequence, if nullptr, just use
 * rewindDraftTokenCommonCount, else use rewindDraftTokenSeparateAdjustments[i] + rewindDraftTokenCommonCount
 * @param stream : CUDA stream to use.
 */
template <typename KVCacheBuffer>
void updateKVCacheDraftTokenLocation(KVCacheBuffer const* kvCacheBuffers,
    SizeType32 const* seqAcceptedDraftTokenOffsets, IndexType const* packedAcceptedDraftTokensIndices,
    SizeType32 const* pastKeyValueLengths, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, SizeType32 const* seqSlotRemapping,
    SizeType32 const* batchSlots, cudaStream_t stream)
{
    SizeType32 startLayer = 0;
    static constexpr SizeType32 kMaxLayersPerIter = 32;
    while (startLayer < layerCount)
    {
        SizeType32 microBatchLayerCount = std::min(layerCount - startLayer, kMaxLayersPerIter);
        updateKVCacheDraftTokenLocationBatched<KVCacheBuffer, kMaxLayersPerIter>(kvCacheBuffers + startLayer,
            seqAcceptedDraftTokenOffsets, packedAcceptedDraftTokensIndices, pastKeyValueLengths, microBatchLayerCount,
            seqCount, numKVHeads, sizeInBytesPerKVHead, rewindDraftTokenCommonCount,
            rewindDraftTokenSeparateAdjustments, seqSlotRemapping, batchSlots, stream);
        startLayer += microBatchLayerCount;
    }
}

void updateLinearKVCacheDraftTokenLocation(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, SizeType32 const* seqSlotRemapping, SizeType32 maxKVCacheLen,
    cudaStream_t stream)
{
    std::vector<KVLinearBuffer> kvLinearBuffers;
    kvLinearBuffers.reserve(layerCount);
    auto const sizePerToken = numKVHeads * sizeInBytesPerKVHead;
    for (SizeType32 i = 0; i < layerCount; i++)
    {
        kvLinearBuffers.emplace_back(
            seqCount, maxKVCacheLen, sizePerToken, maxKVCacheLen, 0, false, pastKeyValueList[i]);
    }
    updateKVCacheDraftTokenLocation(kvLinearBuffers.data(), seqAcceptedDraftTokenOffsets,
        packedAcceptedDraftTokensIndices, pastKeyValueLengths, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead,
        rewindDraftTokenCommonCount, rewindDraftTokenSeparateAdjustments, seqSlotRemapping, nullptr, stream);
}

void updateKVBlockArrayDraftTokenLocation(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths, void* const* pointerArray,
    KVBlockArray::DataType* offsetArray, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCommonCount,
    SizeType32 const* rewindDraftTokenSeparateAdjustments, SizeType32 const* seqSlotRemapping,
    SizeType32 const* batchSlots, SizeType32 maxKVCacheLen, SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock,
    bool canUseOneMoreBlock, cudaStream_t stream)
{
    std::vector<KVBlockArray> kvBlockArrays;
    kvBlockArrays.reserve(layerCount);
    auto const bytesPerToken = numKVHeads * sizeInBytesPerKVHead;
    auto const bytesPerBlock = tokensPerBlock * bytesPerToken;
    for (SizeType32 layerIdx = 0; layerIdx < layerCount; layerIdx++)
    {
        auto const layerOffset = layerIdx * 2 * bytesPerBlock;
        auto* const primaryPoolPointer
            = reinterpret_cast<void*>(reinterpret_cast<char*>(pointerArray[0]) + layerOffset);
        auto* const secondaryPoolPointer
            = reinterpret_cast<void*>(reinterpret_cast<char*>(pointerArray[1]) + layerOffset);

        kvBlockArrays.emplace_back(seqCount, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, maxKVCacheLen,
            maxKVCacheLen, 0, canUseOneMoreBlock, primaryPoolPointer, secondaryPoolPointer, offsetArray);
    }
    updateKVCacheDraftTokenLocation(kvBlockArrays.data(), seqAcceptedDraftTokenOffsets,
        packedAcceptedDraftTokensIndices, pastKeyValueLengths, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead,
        rewindDraftTokenCommonCount, rewindDraftTokenSeparateAdjustments, seqSlotRemapping, batchSlots, stream);
}

void updateLinearKVCacheDraftTokenLocationCommonRewind(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCount, SizeType32 const* seqSlotRemapping,
    SizeType32 maxKVCacheLen, cudaStream_t stream)
{
    updateLinearKVCacheDraftTokenLocation(seqAcceptedDraftTokenOffsets, packedAcceptedDraftTokensIndices,
        pastKeyValueLengths, pastKeyValueList, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead,
        rewindDraftTokenCount, nullptr, seqSlotRemapping, maxKVCacheLen, stream);
}

void updateKVBlockArrayDraftTokenLocationCommonRewind(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths, void* const* pointerArray,
    KVBlockArray::DataType* offsetArray, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 rewindDraftTokenCount, SizeType32 const* seqSlotRemapping,
    SizeType32 maxKVCacheLen, SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, bool canUseOneMoreBlock,
    cudaStream_t stream)
{
    updateKVBlockArrayDraftTokenLocation(seqAcceptedDraftTokenOffsets, packedAcceptedDraftTokensIndices,
        pastKeyValueLengths, pointerArray, offsetArray, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead,
        rewindDraftTokenCount, nullptr, seqSlotRemapping, nullptr, maxKVCacheLen, maxBlocksPerSeq, tokensPerBlock,
        canUseOneMoreBlock, stream);
}

void updateLinearKVCacheDraftTokenLocationSeparateRewind(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths,
    int8_t* const* pastKeyValueList, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32* rewindDraftTokenCounts, SizeType32 const* seqSlotRemapping,
    SizeType32 maxKVCacheLen, cudaStream_t stream)
{
    updateLinearKVCacheDraftTokenLocation(seqAcceptedDraftTokenOffsets, packedAcceptedDraftTokensIndices,
        pastKeyValueLengths, pastKeyValueList, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead, 0,
        rewindDraftTokenCounts, seqSlotRemapping, maxKVCacheLen, stream);
}

void updateKVBlockArrayDraftTokenLocationSeparateRewind(SizeType32 const* seqAcceptedDraftTokenOffsets,
    IndexType const* packedAcceptedDraftTokensIndices, SizeType32 const* pastKeyValueLengths, void* const* pointerArray,
    KVBlockArray::DataType* offsetArray, SizeType32 layerCount, SizeType32 seqCount, SizeType32 numKVHeads,
    SizeType32 sizeInBytesPerKVHead, SizeType32 const* rewindDraftTokenCounts, SizeType32 const* seqSlotRemapping,
    SizeType32 maxKVCacheLen, SizeType32 maxBlocksPerSeq, SizeType32 tokensPerBlock, bool canUseOneMoreBlock,
    cudaStream_t stream)
{
    updateKVBlockArrayDraftTokenLocation(seqAcceptedDraftTokenOffsets, packedAcceptedDraftTokensIndices,
        pastKeyValueLengths, pointerArray, offsetArray, layerCount, seqCount, numKVHeads, sizeInBytesPerKVHead, 0,
        rewindDraftTokenCounts, seqSlotRemapping, nullptr, maxKVCacheLen, maxBlocksPerSeq, tokensPerBlock,
        canUseOneMoreBlock, stream);
}

} // namespace tensorrt_llm::kernels::speculative_decoding
