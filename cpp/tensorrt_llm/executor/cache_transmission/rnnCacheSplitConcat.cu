/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "cacheSplitConcat.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace tensorrt_llm::executor::rnn_cache
{

using namespace tensorrt_llm::runtime;

__device__ __forceinline__ void getLayerIdInDomainPPandRankInDomainPP(int layerId, int domainPPSize,
    uint64_t* prefixLayerNumDevPtr, int& layerIdInDomainPP, int& rankInDomainPP, int& layerNumInSpecPP)
{
    __shared__ int sharedLayerIdInDomainPP;
    __shared__ int sharedRankInDomainPP;
    __shared__ int sharedLayerNumInSpecPP;

#pragma unroll 1
    for (int ppRank = threadIdx.x; ppRank < domainPPSize; ppRank += blockDim.x)
    {
        if (layerId >= prefixLayerNumDevPtr[ppRank] && layerId < prefixLayerNumDevPtr[ppRank + 1])
        {
            sharedLayerIdInDomainPP = layerId - prefixLayerNumDevPtr[ppRank];
            sharedRankInDomainPP = ppRank;
            sharedLayerNumInSpecPP = prefixLayerNumDevPtr[ppRank + 1] - prefixLayerNumDevPtr[ppRank];
            break;
        }
    }

    __syncthreads();
    layerIdInDomainPP = sharedLayerIdInDomainPP;
    rankInDomainPP = sharedRankInDomainPP;
    layerNumInSpecPP = sharedLayerNumInSpecPP;
}

/**
 * @brief Kernel to split RNN conv state from source TP to destination TP
 *
 * Input:  [numLayers, maxBatchSize, convDim, dConv-1] per block
 * Output: Split across different TP ranks based on destCacheState
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitRnnConvStateKernel(T const** __restrict__ inputConvBlocks, T** __restrict__ outputCaches,
    int const slotIdx, int const maxBatchSize, int convDimLocal, int dConvMinus1, int numLayers, int inputBlockNum,
    int domainPPSize, int convDimDomainTP, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                T const* inputBlockPtr = inputConvBlocks[blockId];
                int64_t batchStride = static_cast<int64_t>(convDimLocal) * dConvMinus1;
                int64_t layerStride = static_cast<int64_t>(batchStride) * maxBatchSize;
                T const* inputPtr
                    = inputBlockPtr + layerId * layerStride + slotIdx * batchStride + convDimId * dConvMinus1;

                int outputCacheIdx = convDimId / convDimDomainTP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                // Calculate position within output cache
                int convDimIdInDomainTP = convDimId % convDimDomainTP;
                T* outputPtr = outputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * convDimDomainTP * dConvMinus1)
                    + static_cast<int64_t>(layerIdInDomainPP) * convDimDomainTP * dConvMinus1
                    + static_cast<int64_t>(convDimIdInDomainTP) * dConvMinus1;

#pragma unroll 1
                for (int tempId = laneId * numElePerThread; tempId < dConvMinus1;
                     tempId += (subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + tempId, outputPtr + tempId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to split RNN SSM state from source TP to destination TP
 *
 * Input:  [numLayers, maxBatchSize, numHeads, headDim, dState] per block
 * Output: Split across different TP ranks based on destCacheState
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitRnnSsmStateKernel(T const** __restrict__ inputSsmBlocks, T** __restrict__ outputCaches,
    int const slotIdx, int const maxBatchSize, int numHeadsLocal, int headDim, int dState, int numLayers,
    int inputBlockNum, int domainPPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    int const headDimTimesDState = headDim * dState;

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                T const* inputBlockPtr = inputSsmBlocks[blockId];
                int64_t batchStride = static_cast<int64_t>(numHeadsLocal) * headDimTimesDState;
                int64_t layerStride = static_cast<int64_t>(batchStride) * maxBatchSize;
                T const* inputPtr
                    = inputBlockPtr + layerId * layerStride + slotIdx * batchStride + headId * headDimTimesDState;

                int outputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                // Calculate position within output cache
                int headIdInDomainTP = headId % headNumDomainTP;
                T* outputPtr = outputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * headNumDomainTP * headDimTimesDState)
                    + static_cast<int64_t>(layerIdInDomainPP) * headNumDomainTP * headDimTimesDState
                    + static_cast<int64_t>(headIdInDomainTP) * headDimTimesDState;

#pragma unroll 1
                // Copy [headDim, dState] data with vectorization
                for (int elemId = subWarpIdInGroup * subWarpSize * numElePerThread + laneId * numElePerThread;
                     elemId < headDimTimesDState; elemId += (subWarpNumInGroup * subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + elemId, outputPtr + elemId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to concat RNN conv state from multiple sources to destination TP
 *
 * Input:  Split across different source TP ranks
 * Output: [numLayers, maxBatchSize, convDimLocal, dConv-1] per block
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatRnnConvStateKernel(T const** __restrict__ inputCaches, T** __restrict__ outputConvBlocks,
    int const slotIdx, int const maxBatchSize, int convDimLocal, int dConvMinus1, int numLayers, int outputBlockNum,
    int domainPPSize, int convDimDomainTP, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                T* outputBlockPtr = outputConvBlocks[blockId];
                int64_t batchStride = static_cast<int64_t>(convDimLocal) * dConvMinus1;
                int64_t layerStride = static_cast<int64_t>(batchStride) * maxBatchSize;
                T* outputPtr = outputBlockPtr + layerId * layerStride + slotIdx * batchStride + convDimId * dConvMinus1;

                int inputCacheIdx = convDimId / convDimDomainTP * domainPPSize + rankInDomainPP;
                int convDimIdInDomainTP = convDimId % convDimDomainTP;

                T const* inputCachePtr = inputCaches[inputCacheIdx];
                T const* inputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * convDimDomainTP * dConvMinus1)
                    + static_cast<int64_t>(layerIdInDomainPP) * convDimDomainTP * dConvMinus1
                    + static_cast<int64_t>(convDimIdInDomainTP) * dConvMinus1;

#pragma unroll 1
                for (int tempId = laneId * numElePerThread; tempId < dConvMinus1;
                     tempId += (subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + tempId, outputPtr + tempId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to concat RNN SSM state from multiple sources to destination TP
 *
 * Input:  Split across different source TP ranks
 * Output: [numLayers, maxBatchSize, numHeadsLocal, headDim, dState] per block
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatRnnSsmStateKernel(T const** __restrict__ inputCaches, T** __restrict__ outputSsmBlocks,
    int const slotIdx, int const maxBatchSize, int numHeadsLocal, int headDim, int dState, int numLayers,
    int outputBlockNum, int domainPPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    int const headDimTimesDState = headDim * dState;

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                T* outputBlockPtr = outputSsmBlocks[blockId];
                int64_t batchStride = static_cast<int64_t>(numHeadsLocal) * headDimTimesDState;
                int64_t layerStride = static_cast<int64_t>(batchStride) * maxBatchSize;
                T* outputPtr
                    = outputBlockPtr + layerId * layerStride + slotIdx * batchStride + headId * headDimTimesDState;

                int inputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                int headIdInDomainTP = headId % headNumDomainTP;

                T const* inputCachePtr = inputCaches[inputCacheIdx];
                T const* inputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * headNumDomainTP * headDimTimesDState)
                    + static_cast<int64_t>(layerIdInDomainPP) * headNumDomainTP * headDimTimesDState
                    + static_cast<int64_t>(headIdInDomainTP) * headDimTimesDState;

#pragma unroll 1
                for (int elemId = subWarpIdInGroup * numElePerThread; elemId < headDimTimesDState;
                     elemId += (subWarpNumInGroup * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + elemId, outputPtr + elemId);
                }
            }
        }
    }
}

template <typename T>
void splitRnnConvState(std::vector<runtime::ITensor::SharedPtr> const& inputConvBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, SizeType32 const slotIdx,
    SizeType32 const maxBatchSize, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputConvBlocks.empty());

    size_t inputBlockNum = inputConvBlocks.size();

    // Get target rank information using the existing targetIRanks from kv_cache namespace
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);

    // Verify domain sizes
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    // Calculate output cache count (accounting for peer duplication)
    auto outputCacheNum = targetRankInfo.mIRanks.size();
    outputCacheNum = outputCacheNum / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);

    // Prepare pointer arrays for device
    std::vector<uint64_t> cachePtrs;
    auto cacheDataType = inputConvBlocks.front()->getDataType();

    // Collect input block pointers
    for (auto const& convBlock : inputConvBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(convBlock->data()));
    }

    // Collect output block pointers
    for (auto const& outputBlock : outputSplitBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    // Build prefix layer number array for PP distribution
    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    // Allocate and copy pointer array to device
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    // Extract model and parallel configurations
    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);

    // Calculate conv dimensions
    int const convDimLocal = selfModelConfig.mConvDimSize / selfTPSizePerDPGroup;
    int const dConvMinus1 = selfModelConfig.mDConv - 1;

    // Calculate domain parameters
    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;

    // Calculate conv dimensions per domain TP rank
    int const convDimDomainTP = convDimLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "splitRnnConvState - numLayers: %d, convDimLocal: %d, dConvMinus1: %d, "
        "domainPPSize: %d, domainTPSize: %d, convDimDomainTP: %d, "
        "outputCacheNum: %zu, inputBlockNum: %zu, peerDupHeadFactor: %d",
        numLayers, convDimLocal, dConvMinus1, domainPPSize, domainTPSize, convDimDomainTP, outputSplitBlocks.size(),
        inputBlockNum, targetRankInfo.mPeerDupHeadFactor);

    // Set up device pointers
    T const** inputConvPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNum + outputSplitBlocks.size();

    constexpr int subWarpSize = 32;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, inputBlockNum);
    dim3 blockDim(blockDimx);

    int const rowStrideBytes = dConvMinus1 * sizeof(T);
    int vecSizeByte = 16;
    while (vecSizeByte > static_cast<int>(sizeof(T)) && (rowStrideBytes % vecSizeByte) != 0)
    {
        vecSizeByte /= 2;
    }

    if (vecSizeByte == 16)
    {
        splitRnnConvStateKernel<T, subWarpSize, 16><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputConvPtrsDev, outputCachePtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
            inputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
    }
    else if (vecSizeByte == 8)
    {
        splitRnnConvStateKernel<T, subWarpSize, 8><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputConvPtrsDev, outputCachePtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
            inputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
    }
    else if constexpr (sizeof(T) <= 4)
    {
        if (vecSizeByte == 4)
        {
            splitRnnConvStateKernel<T, subWarpSize, 4><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
                inputConvPtrsDev, outputCachePtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
                inputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
        }
        else if constexpr (sizeof(T) <= 2)
        {
            // vecSizeByte == 2, only valid for 1 or 2 byte types
            splitRnnConvStateKernel<T, subWarpSize, 2><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
                inputConvPtrsDev, outputCachePtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
                inputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
        }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void splitRnnSsmState(std::vector<runtime::ITensor::SharedPtr> const& inputSsmBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, SizeType32 const slotIdx,
    SizeType32 const maxBatchSize, size_t convBytesPerLayer, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSsmBlocks.empty());

    size_t inputBlockNum = inputSsmBlocks.size();
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto outputCacheNum = targetRankInfo.mIRanks.size();
    outputCacheNum = outputCacheNum / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);

    std::vector<uint64_t> cachePtrs;
    auto cacheDataType = inputSsmBlocks.front()->getDataType();

    for (auto const& ssmBlock : inputSsmBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(ssmBlock->data()));
    }

    for (size_t i = 0; i < outputSplitBlocks.size(); i++)
    {
        // Calculate offset for this specific target
        SizeType32 layersForThisTarget = targetRankInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(i));
        size_t ssmOffset = layersForThisTarget * convBytesPerLayer / targetRankInfo.mDomainTPSize;

        // Apply offset to UINT8 buffer
        uint8_t* basePtr = static_cast<uint8_t*>(outputSplitBlocks[i]->data());
        T* typedPtr = reinterpret_cast<T*>(basePtr + ssmOffset);

        cachePtrs.push_back(reinterpret_cast<uint64_t>(typedPtr));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);

    // Note: mNumHeads is GLOBAL, need to divide by TP
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPSizePerDPGroup;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "splitRnnSsmState - numLayers: %d, numHeadsLocal: %d, headDim: %d, dState: %d, "
        "domainPPSize: %d, domainTPSize: %d, headNumDomainTP: %d, "
        "outputCacheNum: %zu, peerDupHeadFactor: %d",
        numLayers, numHeadsLocal, headDim, dState, domainPPSize, domainTPSize, headNumDomainTP,
        outputSplitBlocks.size(), targetRankInfo.mPeerDupHeadFactor);

    T const** inputSsmPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNum + outputSplitBlocks.size();

    constexpr int subWarpSize = 32;
    constexpr int subWarpNumInGroup = 4;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, inputBlockNum);
    dim3 blockDim(blockDimx);
    int const headDimTimesDState = headDim * dState;
    int const remainder = headDimTimesDState * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
    {
        splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev,
                    slotIdx, maxBatchSize, numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
            break;
        }
    }
    default:
    {
        // For 8-byte types or unhandled remainders, use 8-byte vectorization
        splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatRnnConvState(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputConvBlocks, SizeType32 const slotIdx, SizeType32 const maxBatchSize,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    TLLM_CHECK(!outputConvBlocks.empty());

    size_t outputBlockNum = outputConvBlocks.size();

    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto inputCacheNum = targetRankInfo.mIRanks.size();
    inputCacheNum = inputCacheNum / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);

    std::vector<uint64_t> cachePtrs;
    auto cacheDataType = inputSplitBlocks.front()->getDataType();

    for (auto const& outputBlock : outputConvBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    for (auto const& inputBlock : inputSplitBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(inputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const convDimLocal = selfModelConfig.mConvDimSize / selfTPSizePerDPGroup;
    int const dConvMinus1 = selfModelConfig.mDConv - 1;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const convDimDomainTP = convDimLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "concatRnnConvState - numLayers: %d, convDimLocal: %d, dConvMinus1: %d, "
        "domainPPSize: %d, domainTPSize: %d, convDimDomainTP: %d",
        numLayers, convDimLocal, dConvMinus1, domainPPSize, domainTPSize, convDimDomainTP);

    T** outputConvPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputCachePtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNum + inputSplitBlocks.size();

    constexpr int subWarpSize = 32;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, outputBlockNum);
    dim3 blockDim(blockDimx);

    int const rowStrideBytes = dConvMinus1 * sizeof(T);
    int vecSizeByte = 16;
    while (vecSizeByte > static_cast<int>(sizeof(T)) && (rowStrideBytes % vecSizeByte) != 0)
    {
        vecSizeByte /= 2;
    }

    if (vecSizeByte == 16)
    {
        concatRnnConvStateKernel<T, subWarpSize, 16><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputCachePtrsDev, outputConvPtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
            outputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
    }
    else if (vecSizeByte == 8)
    {
        concatRnnConvStateKernel<T, subWarpSize, 8><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputCachePtrsDev, outputConvPtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
            outputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
    }
    else if constexpr (sizeof(T) <= 4)
    {
        if (vecSizeByte == 4)
        {
            concatRnnConvStateKernel<T, subWarpSize, 4><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
                inputCachePtrsDev, outputConvPtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
                outputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
        }
        else if constexpr (sizeof(T) <= 2)
        {
            // vecSizeByte == 2, only valid for 1 or 2 byte types
            concatRnnConvStateKernel<T, subWarpSize, 2><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
                inputCachePtrsDev, outputConvPtrsDev, slotIdx, maxBatchSize, convDimLocal, dConvMinus1, numLayers,
                outputBlockNum, domainPPSize, convDimDomainTP, prefixLayerNumDevPtr);
        }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatRnnSsmState(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSsmBlocks, SizeType32 const slotIdx, SizeType32 const maxBatchSize,
    size_t convBytesPerLayer, kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState,
    int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    TLLM_CHECK(!outputSsmBlocks.empty());

    size_t outputBlockNum = outputSsmBlocks.size();

    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto inputCacheNum = targetRankInfo.mIRanks.size();
    inputCacheNum = inputCacheNum / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);

    std::vector<uint64_t> cachePtrs;
    auto cacheDataType = inputSplitBlocks.front()->getDataType();

    for (auto const& outputBlock : outputSsmBlocks)
    {
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    for (size_t i = 0; i < inputSplitBlocks.size(); i++)
    {
        SizeType32 layersFromThisSource = targetRankInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(i));
        size_t ssmOffset = layersFromThisSource * convBytesPerLayer;

        uint8_t* basePtr = static_cast<uint8_t*>(inputSplitBlocks[i]->data());
        T* typedPtr = reinterpret_cast<T*>(basePtr + ssmOffset);

        cachePtrs.push_back(reinterpret_cast<uint64_t>(typedPtr));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPSizePerDPGroup;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "concatRnnSsmState - numLayers: %d, numHeadsLocal: %d, headDim: %d, dState: %d, "
        "domainPPSize: %d, domainTPSize: %d, headNumDomainTP: %d, "
        "inputCacheNum: %zu, peerDupHeadFactor: %d",
        numLayers, numHeadsLocal, headDim, dState, domainPPSize, domainTPSize, headNumDomainTP, inputSplitBlocks.size(),
        targetRankInfo.mPeerDupHeadFactor);

    T** outputSsmPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputCachePtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNum + inputSplitBlocks.size();

    constexpr int subWarpSize = 32;
    constexpr int subWarpNumInGroup = 4;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, outputBlockNum);
    dim3 blockDim(blockDimx);
    int const remainder = (headDim * dState) * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
    {
        concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev,
                    slotIdx, maxBatchSize, numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
            break;
        }
    }
    default:
    {
        // For 8-byte types or unhandled remainders, use 8-byte vectorization
        concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev, slotIdx,
                maxBatchSize, numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

void splitRnnConvStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputConvBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, SizeType32 const slotIdx,
    SizeType32 const maxBatchSize, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputConvBlocks.empty());
    auto dataType = inputConvBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        splitRnnConvState<int64_t>(inputConvBlocks, outputSplitBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        splitRnnConvState<int32_t>(inputConvBlocks, outputSplitBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        splitRnnConvState<int16_t>(inputConvBlocks, outputSplitBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        splitRnnConvState<int8_t>(inputConvBlocks, outputSplitBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("splitRnnConvStateDispatch: unsupported data type");
    }
}

void splitRnnSsmStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSsmBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, SizeType32 const slotIdx,
    SizeType32 const maxBatchSize, size_t convBytesPerLayer, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSsmBlocks.empty());
    auto dataType = inputSsmBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        splitRnnSsmState<int64_t>(inputSsmBlocks, outputSplitBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        splitRnnSsmState<int32_t>(inputSsmBlocks, outputSplitBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        splitRnnSsmState<int16_t>(inputSsmBlocks, outputSplitBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        splitRnnSsmState<int8_t>(inputSsmBlocks, outputSplitBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("splitRnnSsmStateDispatch: unsupported data type");
    }
}

void concatRnnConvStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputConvBlocks, SizeType32 const slotIdx, SizeType32 const maxBatchSize,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    auto dataType = outputConvBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        concatRnnConvState<int64_t>(inputSplitBlocks, outputConvBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        concatRnnConvState<int32_t>(inputSplitBlocks, outputConvBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        concatRnnConvState<int16_t>(inputSplitBlocks, outputConvBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        concatRnnConvState<int8_t>(inputSplitBlocks, outputConvBlocks, slotIdx, maxBatchSize, destCacheState,
            selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("concatRnnConvStateDispatch: unsupported data type");
    }
}

void concatRnnSsmStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSsmBlocks, SizeType32 const slotIdx, SizeType32 const maxBatchSize,
    size_t convBytesPerLayer, kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState,
    int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    auto dataType = outputSsmBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        concatRnnSsmState<int64_t>(inputSplitBlocks, outputSsmBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        concatRnnSsmState<int32_t>(inputSplitBlocks, outputSsmBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        concatRnnSsmState<int16_t>(inputSplitBlocks, outputSsmBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        concatRnnSsmState<int8_t>(inputSplitBlocks, outputSsmBlocks, slotIdx, maxBatchSize, convBytesPerLayer,
            destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("concatRnnSsmStateDispatch: unsupported data type");
    }
}

// ===========================================================================
// Unified Pool Split/Concat Kernels
// ===========================================================================
// These kernels operate on the unified KV/recurrent pool layout used by
// CppMambaHybridCacheManager:
//   pool shape: {numLocalLayers, numBlocks, blockSize_bytes}
//   block layout: [SSM_bytes | Conv_bytes]
//     SSM portion: [numHeads, headDim, dState] — split by heads
//     Conv portion: [section0_dim, section1_dim, ...] x [dConv-1] — section-aware split
//
// The input/output pointer arrays follow the pattern: input pointers → output pointers → prefixLayerNum.

/**
 * @brief Kernel to split SSM state from unified pool blocks to per-target buffers.
 *
 * Input:  Pool block pointers, each pointing to {blockSize_bytes} of data.
 *         SSM data starts at offset 0, shaped [numHeads_local, headDim, dState].
 * Output: Per-target SSM buffers, each [numBlocks, layersInPP, headsPerDomainTP, headDim, dState].
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitUnifiedPoolSsmKernel(T const** __restrict__ inputPoolBlocks, T** __restrict__ outputCaches,
    int numHeadsLocal, int headDim, int dState, int numLayers, int inputBlockNum, int domainPPSize, int headNumDomainTP,
    int blockStrideElements, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    int const headDimTimesDState = headDim * dState;

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                // Input: pool[layerId][blockIdx] at SSM offset, then index by head
                T const* inputBlockPtr = inputPoolBlocks[blockId];
                // Each layer's data is at layerId * blockStrideElements
                T const* inputPtr
                    = inputBlockPtr + static_cast<int64_t>(layerId) * blockStrideElements + headId * headDimTimesDState;

                int outputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                int headIdInDomainTP = headId % headNumDomainTP;
                T* outputPtr = outputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * headNumDomainTP * headDimTimesDState)
                    + static_cast<int64_t>(layerIdInDomainPP) * headNumDomainTP * headDimTimesDState
                    + static_cast<int64_t>(headIdInDomainTP) * headDimTimesDState;

#pragma unroll 1
                for (int elemId = subWarpIdInGroup * subWarpSize * numElePerThread + laneId * numElePerThread;
                     elemId < headDimTimesDState; elemId += (subWarpNumInGroup * subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + elemId, outputPtr + elemId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to split conv state from unified pool blocks with section-aware splitting.
 *
 * Input:  Pool block pointers. Conv data starts at ssmBytesElements offset within each block.
 *         Conv layout: [section0_dim_local | section1_dim_local | ...] x [dConv-1].
 * Output: Per-target Conv buffers, each section independently split by TP.
 *
 * numConvSections: number of sections (e.g. 3 for [x|B|C] or [Q|K|V])
 * convSectionDimsLocal: per-section local element counts (already TP-divided)
 * convSectionDimsDomainTP: per-section element counts per domain TP rank
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitUnifiedPoolConvKernel(T const** __restrict__ inputPoolBlocks, T** __restrict__ outputCaches,
    int dConvMinus1, int numLayers, int inputBlockNum, int domainPPSize, int blockStrideElements, int ssmOffsetElements,
    int numConvSections, int const* convSectionDimsLocal, int const* convSectionDimsDomainTP,
    int const* convSectionOffsetsLocal, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    // Total conv dim across all sections (local)
    int convDimLocal = 0;
    for (int s = 0; s < numConvSections; ++s)
    {
        convDimLocal += convSectionDimsLocal[s];
    }

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

            // Input base for this layer+block's conv data
            T const* inputBlockPtr = inputPoolBlocks[blockId];
            T const* layerConvBase
                = inputBlockPtr + static_cast<int64_t>(layerId) * blockStrideElements + ssmOffsetElements;

            // Iterate over all conv dimension rows (across all sections)
#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                // Determine which section this convDimId belongs to, and position within section
                int sectionIdx = 0;
                int posInSection = convDimId;
                for (int s = 0; s < numConvSections; ++s)
                {
                    if (posInSection < convSectionDimsLocal[s])
                    {
                        sectionIdx = s;
                        break;
                    }
                    posInSection -= convSectionDimsLocal[s];
                }

                T const* inputPtr = layerConvBase + static_cast<int64_t>(convDimId) * dConvMinus1;

                // Map to output target based on section-specific TP split
                int domainTPDim = convSectionDimsDomainTP[sectionIdx];
                int tpIdx = posInSection / domainTPDim;
                int posInDomainTP = posInSection % domainTPDim;

                // Output target index: tpIdx * domainPPSize + ppRank
                int outputCacheIdx = tpIdx * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                // Calculate output conv dim for this section+TP rank
                // Total output conv dim per domain TP rank = sum of all domainTP section dims
                int outputConvDimTotal = 0;
                for (int s = 0; s < numConvSections; ++s)
                {
                    outputConvDimTotal += convSectionDimsDomainTP[s];
                }

                // Output section offset = sum of preceding section domainTP dims
                int outputSectionOffset = 0;
                for (int s = 0; s < sectionIdx; ++s)
                {
                    outputSectionOffset += convSectionDimsDomainTP[s];
                }

                T* outputPtr = outputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * outputConvDimTotal * dConvMinus1)
                    + static_cast<int64_t>(layerIdInDomainPP) * outputConvDimTotal * dConvMinus1
                    + static_cast<int64_t>(outputSectionOffset + posInDomainTP) * dConvMinus1;

#pragma unroll 1
                for (int tempId = laneId * numElePerThread; tempId < dConvMinus1;
                     tempId += (subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + tempId, outputPtr + tempId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to concat SSM state from per-source buffers into unified pool blocks.
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatUnifiedPoolSsmKernel(T const** __restrict__ inputCaches, T** __restrict__ outputPoolBlocks,
    int numHeadsLocal, int headDim, int dState, int numLayers, int outputBlockNum, int domainPPSize,
    int headNumDomainTP, int blockStrideElements, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    int const headDimTimesDState = headDim * dState;

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                // Output: pool[layerId][blockIdx] at SSM offset
                T* outputBlockPtr = outputPoolBlocks[blockId];
                T* outputPtr = outputBlockPtr + static_cast<int64_t>(layerId) * blockStrideElements
                    + headId * headDimTimesDState;

                int inputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                int headIdInDomainTP = headId % headNumDomainTP;

                T const* inputCachePtr = inputCaches[inputCacheIdx];
                T const* inputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * headNumDomainTP * headDimTimesDState)
                    + static_cast<int64_t>(layerIdInDomainPP) * headNumDomainTP * headDimTimesDState
                    + static_cast<int64_t>(headIdInDomainTP) * headDimTimesDState;

#pragma unroll 1
                for (int elemId = subWarpIdInGroup * subWarpSize * numElePerThread + laneId * numElePerThread;
                     elemId < headDimTimesDState; elemId += (subWarpNumInGroup * subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + elemId, outputPtr + elemId);
                }
            }
        }
    }
}

/**
 * @brief Kernel to concat conv state from per-source buffers into unified pool blocks (section-aware).
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatUnifiedPoolConvKernel(T const** __restrict__ inputCaches, T** __restrict__ outputPoolBlocks,
    int dConvMinus1, int numLayers, int outputBlockNum, int domainPPSize, int blockStrideElements,
    int ssmOffsetElements, int numConvSections, int const* convSectionDimsLocal, int const* convSectionDimsDomainTP,
    int const* convSectionOffsetsLocal, uint64_t* prefixLayerNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);

    int convDimLocal = 0;
    for (int s = 0; s < numConvSections; ++s)
    {
        convDimLocal += convSectionDimsLocal[s];
    }

#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

            // Output base for this layer+block's conv data
            T* outputBlockPtr = outputPoolBlocks[blockId];
            T* layerConvBase = outputBlockPtr + static_cast<int64_t>(layerId) * blockStrideElements + ssmOffsetElements;

#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                int sectionIdx = 0;
                int posInSection = convDimId;
                for (int s = 0; s < numConvSections; ++s)
                {
                    if (posInSection < convSectionDimsLocal[s])
                    {
                        sectionIdx = s;
                        break;
                    }
                    posInSection -= convSectionDimsLocal[s];
                }

                T* outputPtr = layerConvBase + static_cast<int64_t>(convDimId) * dConvMinus1;

                int domainTPDim = convSectionDimsDomainTP[sectionIdx];
                int tpIdx = posInSection / domainTPDim;
                int posInDomainTP = posInSection % domainTPDim;

                int inputCacheIdx = tpIdx * domainPPSize + rankInDomainPP;

                int outputConvDimTotal = 0;
                for (int s = 0; s < numConvSections; ++s)
                {
                    outputConvDimTotal += convSectionDimsDomainTP[s];
                }

                int inputSectionOffset = 0;
                for (int s = 0; s < sectionIdx; ++s)
                {
                    inputSectionOffset += convSectionDimsDomainTP[s];
                }

                T const* inputCachePtr = inputCaches[inputCacheIdx];
                T const* inputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId) * (layerNumInSpecPP * outputConvDimTotal * dConvMinus1)
                    + static_cast<int64_t>(layerIdInDomainPP) * outputConvDimTotal * dConvMinus1
                    + static_cast<int64_t>(inputSectionOffset + posInDomainTP) * dConvMinus1;

#pragma unroll 1
                for (int tempId = laneId * numElePerThread; tempId < dConvMinus1;
                     tempId += (subWarpSize * numElePerThread))
                {
                    common::copy<vecSizeByte>(inputPtr + tempId, outputPtr + tempId);
                }
            }
        }
    }
}

// ===========================================================================
// Host dispatch functions for unified pool split/concat
// ===========================================================================

template <typename T>
void splitUnifiedPoolSsm(runtime::ITensor::SharedPtr const& pool, std::vector<SizeType32> const& realBlockIndices,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes, size_t blockSizeBytes,
    runtime::BufferManager const& bufferManager)
{
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);
    auto outputCacheNum = targetRankInfo.mIRanks.size() / targetRankInfo.mPeerDupHeadFactor;
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());

    size_t inputBlockNum = realBlockIndices.size();
    if (inputBlockNum == 0)
        return;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPSizePerDPGroup;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    // Pool shape: {numLayers, numBlocks, blockSizeBytes}
    // blockStrideElements = blockSizeBytes / sizeof(T) for the stride across the pool's block dimension
    int const blockStrideElements = static_cast<int>(blockSizeBytes / sizeof(T));

    // Build pointer arrays: input pool block pointers (pointing to SSM start), output buffers, prefixLayerNum
    std::vector<uint64_t> allPtrs;
    SizeType32 const numBlocks = pool->getShape().d[1];
    uint8_t* poolBase = static_cast<uint8_t*>(pool->data());

    for (auto blockIdx : realBlockIndices)
    {
        // Point to {layer=0, blockIdx, offset=0} — SSM starts at offset 0
        allPtrs.push_back(reinterpret_cast<uint64_t>(poolBase + static_cast<size_t>(blockIdx) * blockSizeBytes));
    }
    for (auto const& outputBlock : outputSplitBlocks)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(domainPPSize + 1, 0);
    for (int i = 0; i < domainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    allPtrs.insert(allPtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    auto PtrsDeviceBuffer = bufferManager.gpu(allPtrs.size(), nvinfer1::DataType::kINT64);
    bufferManager.copy(allPtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    T const** inputPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    uint64_t* prefixLayerNumDevPtr = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNum + outputCacheNum;

    constexpr int subWarpSize = 32;
    constexpr int subWarpNumInGroup = 4;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, inputBlockNum);
    dim3 blockDim(blockDimx);
    int const remainder = (headDim * dState) * sizeof(T) % 16;

    // Note: blockStrideElements is the stride in elements between layers for the same block
    // In the pool layout {numLayers, numBlocks, blockSize}, the stride between layers for
    // the same block = numBlocks * blockSizeElements
    int const layerStrideElements = static_cast<int>(numBlocks) * blockStrideElements;

    switch (remainder)
    {
    case 0:
        splitUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    case 8:
        splitUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    case 4:
    case 12:
        if constexpr (sizeof(T) <= 4)
        {
            splitUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                    headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                    prefixLayerNumDevPtr);
            break;
        }
        [[fallthrough]];
    default:
        splitUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void splitUnifiedPoolConv(runtime::ITensor::SharedPtr const& pool, std::vector<SizeType32> const& realBlockIndices,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes, size_t blockSizeBytes,
    runtime::BufferManager const& bufferManager)
{
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(destCacheState, selfCacheState, selfIdx);
    auto outputCacheNum = targetRankInfo.mIRanks.size() / targetRankInfo.mPeerDupHeadFactor;
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());

    size_t inputBlockNum = realBlockIndices.size();
    if (inputBlockNum == 0)
        return;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const dConvMinus1 = selfModelConfig.mDConv - 1;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const effectiveDomainTP = domainTPSize / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK_WITH_INFO(
        selfModelConfig.hasConvSections(), "Failed to get conv state info, please double check the model type");

    static constexpr int numConvSections = kv_cache::CacheState::RnnModelConfig::kNumConvSections;
    auto const globalSectionDims = selfModelConfig.getConvSectionDims();

    // Compute section dims: global → local (TP-divided) and domainTP dims
    std::array<int, numConvSections> sectionDimsLocal;
    std::array<int, numConvSections> sectionDimsDomainTP;
    std::array<int, numConvSections> sectionOffsetsLocal;
    int localOffset = 0;
    for (int s = 0; s < numConvSections; ++s)
    {
        int globalDim = globalSectionDims[s];
        sectionDimsLocal[s] = globalDim / selfTPSizePerDPGroup;
        sectionDimsDomainTP[s] = sectionDimsLocal[s] / effectiveDomainTP;
        sectionOffsetsLocal[s] = localOffset;
        localOffset += sectionDimsLocal[s];
    }

    SizeType32 const numBlocks = pool->getShape().d[1];
    int const blockStrideElements = static_cast<int>(blockSizeBytes / sizeof(T));
    int const layerStrideElements = static_cast<int>(numBlocks) * blockStrideElements;
    int const ssmOffsetElements = static_cast<int>(ssmBytes / sizeof(T));

    // Build pointer arrays
    std::vector<uint64_t> allPtrs;
    uint8_t* poolBase = static_cast<uint8_t*>(pool->data());

    for (auto blockIdx : realBlockIndices)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(poolBase + static_cast<size_t>(blockIdx) * blockSizeBytes));
    }
    for (auto const& outputBlock : outputSplitBlocks)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(domainPPSize + 1, 0);
    for (int i = 0; i < domainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    allPtrs.insert(allPtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    // Also copy section dim arrays to device
    std::vector<int> sectionInfo;
    sectionInfo.insert(sectionInfo.end(), sectionDimsLocal.begin(), sectionDimsLocal.end());
    sectionInfo.insert(sectionInfo.end(), sectionDimsDomainTP.begin(), sectionDimsDomainTP.end());
    sectionInfo.insert(sectionInfo.end(), sectionOffsetsLocal.begin(), sectionOffsetsLocal.end());

    auto PtrsDeviceBuffer = bufferManager.gpu(allPtrs.size(), nvinfer1::DataType::kINT64);
    bufferManager.copy(allPtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto sectionInfoBuffer = bufferManager.gpu(sectionInfo.size(), nvinfer1::DataType::kINT32);
    bufferManager.copy(sectionInfo.data(), *sectionInfoBuffer, runtime::MemoryType::kCPU);

    T const** inputPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    uint64_t* prefixLayerNumDevPtr = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNum + outputCacheNum;

    int const* sectionDimsLocalDev = static_cast<int const*>(sectionInfoBuffer->data());
    int const* sectionDimsDomainTPDev = sectionDimsLocalDev + numConvSections;
    int const* sectionOffsetsLocalDev = sectionDimsDomainTPDev + numConvSections;

    constexpr int subWarpSize = 32;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, inputBlockNum);
    dim3 blockDim(blockDimx);

    int const rowStrideBytes = dConvMinus1 * sizeof(T);
    int vecSizeByte = 16;
    while (vecSizeByte > static_cast<int>(sizeof(T)) && (rowStrideBytes % vecSizeByte) != 0)
    {
        vecSizeByte /= 2;
    }

    if (vecSizeByte == 16)
    {
        splitUnifiedPoolConvKernel<T, subWarpSize, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                numLayers, inputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
    }
    else if (vecSizeByte == 8)
    {
        splitUnifiedPoolConvKernel<T, subWarpSize, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                numLayers, inputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
    }
    else if constexpr (sizeof(T) <= 4)
    {
        if (vecSizeByte == 4)
        {
            splitUnifiedPoolConvKernel<T, subWarpSize, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                    numLayers, inputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                    sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
        }
        else if constexpr (sizeof(T) <= 2)
        {
            if (vecSizeByte >= 2)
            {
                splitUnifiedPoolConvKernel<T, subWarpSize, 2>
                    <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev,
                        dConvMinus1, numLayers, inputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements,
                        numConvSections, sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev,
                        prefixLayerNumDevPtr);
            }
            else if constexpr (sizeof(T) == 1)
            {
                splitUnifiedPoolConvKernel<T, subWarpSize, 1>
                    <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev,
                        dConvMinus1, numLayers, inputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements,
                        numConvSections, sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev,
                        prefixLayerNumDevPtr);
            }
        }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatUnifiedPoolSsm(runtime::ITensor::SharedPtr const& pool, std::vector<SizeType32> const& realBlockIndices,
    std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks, kv_cache::CacheState const& srcCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes, size_t blockSizeBytes,
    runtime::BufferManager const& bufferManager)
{
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(srcCacheState, selfCacheState, selfIdx);
    auto inputCacheNum = targetRankInfo.mIRanks.size() / targetRankInfo.mPeerDupHeadFactor;
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());

    size_t outputBlockNum = realBlockIndices.size();
    if (outputBlockNum == 0)
        return;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPSizePerDPGroup;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    SizeType32 const numBlocks = pool->getShape().d[1];
    int const blockStrideElements = static_cast<int>(blockSizeBytes / sizeof(T));
    int const layerStrideElements = static_cast<int>(numBlocks) * blockStrideElements;

    std::vector<uint64_t> allPtrs;
    uint8_t* poolBase = static_cast<uint8_t*>(pool->data());

    for (auto blockIdx : realBlockIndices)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(poolBase + static_cast<size_t>(blockIdx) * blockSizeBytes));
    }
    for (auto const& inputBlock : inputSplitBlocks)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(inputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(domainPPSize + 1, 0);
    for (int i = 0; i < domainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    allPtrs.insert(allPtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    auto PtrsDeviceBuffer = bufferManager.gpu(allPtrs.size(), nvinfer1::DataType::kINT64);
    bufferManager.copy(allPtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    T** outputPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    uint64_t* prefixLayerNumDevPtr = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNum + inputCacheNum;

    constexpr int subWarpSize = 32;
    constexpr int subWarpNumInGroup = 4;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, outputBlockNum);
    dim3 blockDim(blockDimx);
    int const remainder = (headDim * dState) * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
        concatUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    case 8:
        concatUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    case 4:
    case 12:
        if constexpr (sizeof(T) <= 4)
        {
            concatUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                    headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                    prefixLayerNumDevPtr);
            break;
        }
        [[fallthrough]];
    default:
        concatUnifiedPoolSsmKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, numHeadsLocal,
                headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP, layerStrideElements,
                prefixLayerNumDevPtr);
        break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatUnifiedPoolConv(runtime::ITensor::SharedPtr const& pool, std::vector<SizeType32> const& realBlockIndices,
    std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks, kv_cache::CacheState const& srcCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes, size_t blockSizeBytes,
    runtime::BufferManager const& bufferManager)
{
    auto targetRankInfo = executor::kv_cache::targetIRanksForRnn(srcCacheState, selfCacheState, selfIdx);
    auto inputCacheNum = targetRankInfo.mIRanks.size() / targetRankInfo.mPeerDupHeadFactor;
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());

    size_t outputBlockNum = realBlockIndices.size();
    if (outputBlockNum == 0)
        return;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getRnnModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfDPSize = selfParallelConfig.mDPsize;
    int const selfTPSizePerDPGroup = selfParallelConfig.mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfCacheState.getRnnCacheState().mLayerNumPerPP.at(selfPPRank);
    int const dConvMinus1 = selfModelConfig.mDConv - 1;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const effectiveDomainTP = domainTPSize / targetRankInfo.mPeerDupHeadFactor;

    TLLM_CHECK_WITH_INFO(
        selfModelConfig.hasConvSections(), "Failed to get conv state info, please double check the model type");

    static constexpr int numConvSections = kv_cache::CacheState::RnnModelConfig::kNumConvSections;
    auto const globalSectionDims = selfModelConfig.getConvSectionDims();

    std::array<int, numConvSections> sectionDimsLocal;
    std::array<int, numConvSections> sectionDimsDomainTP;
    std::array<int, numConvSections> sectionOffsetsLocal;
    int localOffset = 0;
    for (int s = 0; s < numConvSections; ++s)
    {
        int globalDim = globalSectionDims[s];
        sectionDimsLocal[s] = globalDim / selfTPSizePerDPGroup;
        sectionDimsDomainTP[s] = sectionDimsLocal[s] / effectiveDomainTP;
        sectionOffsetsLocal[s] = localOffset;
        localOffset += sectionDimsLocal[s];
    }

    SizeType32 const numBlocks = pool->getShape().d[1];
    int const blockStrideElements = static_cast<int>(blockSizeBytes / sizeof(T));
    int const layerStrideElements = static_cast<int>(numBlocks) * blockStrideElements;
    int const ssmOffsetElements = static_cast<int>(ssmBytes / sizeof(T));

    std::vector<uint64_t> allPtrs;
    uint8_t* poolBase = static_cast<uint8_t*>(pool->data());

    for (auto blockIdx : realBlockIndices)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(poolBase + static_cast<size_t>(blockIdx) * blockSizeBytes));
    }
    for (auto const& inputBlock : inputSplitBlocks)
    {
        allPtrs.push_back(reinterpret_cast<uint64_t>(inputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(domainPPSize + 1, 0);
    for (int i = 0; i < domainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerLayerNumInDomainPP[i];
    }
    allPtrs.insert(allPtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    std::vector<int> sectionInfo;
    sectionInfo.insert(sectionInfo.end(), sectionDimsLocal.begin(), sectionDimsLocal.end());
    sectionInfo.insert(sectionInfo.end(), sectionDimsDomainTP.begin(), sectionDimsDomainTP.end());
    sectionInfo.insert(sectionInfo.end(), sectionOffsetsLocal.begin(), sectionOffsetsLocal.end());

    auto PtrsDeviceBuffer = bufferManager.gpu(allPtrs.size(), nvinfer1::DataType::kINT64);
    bufferManager.copy(allPtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto sectionInfoBuffer = bufferManager.gpu(sectionInfo.size(), nvinfer1::DataType::kINT32);
    bufferManager.copy(sectionInfo.data(), *sectionInfoBuffer, runtime::MemoryType::kCPU);

    T** outputPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    uint64_t* prefixLayerNumDevPtr = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNum + inputCacheNum;

    int const* sectionDimsLocalDev = static_cast<int const*>(sectionInfoBuffer->data());
    int const* sectionDimsDomainTPDev = sectionDimsLocalDev + numConvSections;
    int const* sectionOffsetsLocalDev = sectionDimsDomainTPDev + numConvSections;

    constexpr int subWarpSize = 32;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, outputBlockNum);
    dim3 blockDim(blockDimx);

    int const rowStrideBytes = dConvMinus1 * sizeof(T);
    int vecSizeByte = 16;
    while (vecSizeByte > static_cast<int>(sizeof(T)) && (rowStrideBytes % vecSizeByte) != 0)
    {
        vecSizeByte /= 2;
    }

    if (vecSizeByte == 16)
    {
        concatUnifiedPoolConvKernel<T, subWarpSize, 16>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                numLayers, outputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
    }
    else if (vecSizeByte == 8)
    {
        concatUnifiedPoolConvKernel<T, subWarpSize, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                numLayers, outputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
    }
    else if constexpr (sizeof(T) <= 4)
    {
        if (vecSizeByte == 4)
        {
            concatUnifiedPoolConvKernel<T, subWarpSize, 4>
                <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev, dConvMinus1,
                    numLayers, outputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements, numConvSections,
                    sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev, prefixLayerNumDevPtr);
        }
        else if constexpr (sizeof(T) <= 2)
        {
            if (vecSizeByte >= 2)
            {
                concatUnifiedPoolConvKernel<T, subWarpSize, 2>
                    <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev,
                        dConvMinus1, numLayers, outputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements,
                        numConvSections, sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev,
                        prefixLayerNumDevPtr);
            }
            else if constexpr (sizeof(T) == 1)
            {
                concatUnifiedPoolConvKernel<T, subWarpSize, 1>
                    <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputPtrsDev, outputPtrsDev,
                        dConvMinus1, numLayers, outputBlockNum, domainPPSize, layerStrideElements, ssmOffsetElements,
                        numConvSections, sectionDimsLocalDev, sectionDimsDomainTPDev, sectionOffsetsLocalDev,
                        prefixLayerNumDevPtr);
            }
        }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

// ===========================================================================
// Public dispatch functions for unified pool split/concat
// ===========================================================================

void splitUnifiedPoolSsmDispatch(runtime::ITensor::SharedPtr const& pool,
    std::vector<SizeType32> const& realBlockIndices, std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    size_t ssmBytes, size_t blockSizeBytes, nvinfer1::DataType ssmDataType, runtime::BufferManager const& bufferManager)
{
    auto dataSize = tensorrt_llm::common::getDTypeSize(ssmDataType);
    switch (dataSize)
    {
    case 4:
        splitUnifiedPoolSsm<int32_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 2:
        splitUnifiedPoolSsm<int16_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 1:
        splitUnifiedPoolSsm<int8_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    default: TLLM_THROW("splitUnifiedPoolSsmDispatch: unsupported SSM data type size %zu", dataSize);
    }
}

void splitUnifiedPoolConvDispatch(runtime::ITensor::SharedPtr const& pool,
    std::vector<SizeType32> const& realBlockIndices, std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    size_t ssmBytes, size_t blockSizeBytes, nvinfer1::DataType convDataType,
    runtime::BufferManager const& bufferManager)
{
    auto dataSize = tensorrt_llm::common::getDTypeSize(convDataType);
    switch (dataSize)
    {
    case 4:
        splitUnifiedPoolConv<int32_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState,
            selfIdx, ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 2:
        splitUnifiedPoolConv<int16_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState,
            selfIdx, ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 1:
        splitUnifiedPoolConv<int8_t>(pool, realBlockIndices, outputSplitBlocks, destCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    default: TLLM_THROW("splitUnifiedPoolConvDispatch: unsupported conv data type size %zu", dataSize);
    }
}

void concatUnifiedPoolSsmDispatch(runtime::ITensor::SharedPtr const& pool,
    std::vector<SizeType32> const& realBlockIndices, std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    kv_cache::CacheState const& srcCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes,
    size_t blockSizeBytes, nvinfer1::DataType ssmDataType, runtime::BufferManager const& bufferManager)
{
    auto dataSize = tensorrt_llm::common::getDTypeSize(ssmDataType);
    switch (dataSize)
    {
    case 4:
        concatUnifiedPoolSsm<int32_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 2:
        concatUnifiedPoolSsm<int16_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 1:
        concatUnifiedPoolSsm<int8_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    default: TLLM_THROW("concatUnifiedPoolSsmDispatch: unsupported SSM data type size %zu", dataSize);
    }
}

void concatUnifiedPoolConvDispatch(runtime::ITensor::SharedPtr const& pool,
    std::vector<SizeType32> const& realBlockIndices, std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    kv_cache::CacheState const& srcCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx, size_t ssmBytes,
    size_t blockSizeBytes, nvinfer1::DataType convDataType, runtime::BufferManager const& bufferManager)
{
    auto dataSize = tensorrt_llm::common::getDTypeSize(convDataType);
    switch (dataSize)
    {
    case 4:
        concatUnifiedPoolConv<int32_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 2:
        concatUnifiedPoolConv<int16_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    case 1:
        concatUnifiedPoolConv<int8_t>(pool, realBlockIndices, inputSplitBlocks, srcCacheState, selfCacheState, selfIdx,
            ssmBytes, blockSizeBytes, bufferManager);
        break;
    default: TLLM_THROW("concatUnifiedPoolConvDispatch: unsupported conv data type size %zu", dataSize);
    }
}

} // namespace tensorrt_llm::executor::rnn_cache
