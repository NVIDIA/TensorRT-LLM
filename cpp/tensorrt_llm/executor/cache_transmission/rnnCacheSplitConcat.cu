/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::executor::rnn_cache
{

using namespace tensorrt_llm::runtime;

/**
 * @brief Kernel to split RNN conv state from source TP to destination TP
 *
 * Input:  [numLayers, convDimLocal, dConv-1] per block
 * Output: Split across different TP ranks based on destCacheState
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitRnnConvStateKernel(T const** __restrict__ inputConvBlocks, T** __restrict__ outputCaches,
    int convDimLocal, int dConvMinus1, int numLayers, int inputBlockNum, int domainPPSize, int convDimDomainTP,
    uint64_t* prefixLayerNumDevPtr)
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
            tensorrt_llm::executor::kv_cache::getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                T const* inputBlockPtr = inputConvBlocks[blockId];
                T const* inputPtr = inputBlockPtr + layerId * convDimLocal * dConvMinus1 + convDimId * dConvMinus1;

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
 * Input:  [numLayers, numHeadsLocal, headDim, dState] per block
 * Output: Split across different TP ranks based on destCacheState
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitRnnSsmStateKernel(T const** __restrict__ inputSsmBlocks, T** __restrict__ outputCaches,
    int numHeadsLocal, int headDim, int dState, int numLayers, int inputBlockNum, int domainPPSize, int headNumDomainTP,
    uint64_t* prefixLayerNumDevPtr)
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
            tensorrt_llm::executor::kv_cache::getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                T const* inputBlockPtr = inputSsmBlocks[blockId];
                T const* inputPtr
                    = inputBlockPtr + layerId * numHeadsLocal * headDimTimesDState + headId * headDimTimesDState;

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
 * Output: [numLayers, convDimLocal, dConv-1] per block
 */
template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatRnnConvStateKernel(T const** __restrict__ inputCaches, T** __restrict__ outputConvBlocks,
    int convDimLocal, int dConvMinus1, int numLayers, int outputBlockNum, int domainPPSize, int convDimDomainTP,
    uint64_t* prefixLayerNumDevPtr)
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
            tensorrt_llm::executor::kv_cache::getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int convDimId = subWarpId; convDimId < convDimLocal; convDimId += subWarpNum)
            {
                T* outputBlockPtr = outputConvBlocks[blockId];
                T* outputPtr = outputBlockPtr + layerId * convDimLocal * dConvMinus1 + convDimId * dConvMinus1;

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
 * Output: [numLayers, numHeadsLocal, headDim, dState] per block
 */
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatRnnSsmStateKernel(T const** __restrict__ inputCaches, T** __restrict__ outputSsmBlocks,
    int numHeadsLocal, int headDim, int dState, int numLayers, int outputBlockNum, int domainPPSize,
    int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
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
            tensorrt_llm::executor::kv_cache::getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < numHeadsLocal; headId += subWarpGroupNum)
            {
                T* outputBlockPtr = outputSsmBlocks[blockId];
                T* outputPtr
                    = outputBlockPtr + layerId * numHeadsLocal * headDimTimesDState + headId * headDimTimesDState;

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
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, rnn_cache::RnnCacheState const& destCacheState,
    rnn_cache::RnnCacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputConvBlocks.empty());

    size_t inputBlockNum = inputConvBlocks.size();

    // Get target rank information using the existing targetIRanks from kv_cache namespace
    auto targetRankInfo = executor::kv_cache::targetIRanks(destCacheState, selfCacheState, selfIdx);

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
        TLLM_CHECK(convBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(convBlock->data()));
    }

    // Collect output block pointers
    for (auto const& outputBlock : outputSplitBlocks)
    {
        TLLM_CHECK_WITH_INFO(outputBlock->getDataType() == cacheDataType, "outputSplitBlock data type mismatch");
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    // Build prefix layer number array for PP distribution
    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    // Allocate and copy pointer array to device
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    // Extract model and parallel configurations
    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfParallelConfig.mRnnLayerNumPerPP.at(selfPPRank);

    // Calculate conv dimensions
    int const convDimLocal = selfModelConfig.mConvDimSize / selfTPNum;
    int const dConvMinus1 = selfModelConfig.mDConv - 1;

    // Calculate domain parameters
    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;

    // Calculate conv dimensions per domain TP rank
    int const convDimDomainTP = convDimLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "splitRnnConvState - numLayers: %d, convDimLocal: %d, dConvMinus1: %d, "
        "domainPPSize: %d, domainTPSize: %d, convDimDomainTP: %d",
        numLayers, convDimLocal, dConvMinus1, domainPPSize, domainTPSize, convDimDomainTP);

    // Set up device pointers
    T const** inputConvPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNum + outputSplitBlocks.size();

    constexpr int subWarpSize = 32;
    constexpr int blockDimx = 128;
    dim3 gridDim(numLayers, inputBlockNum);
    dim3 blockDim(blockDimx);
    int const remainder = dConvMinus1 * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
    {
        splitRnnConvStateKernel<T, subWarpSize, 16><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputConvPtrsDev, outputCachePtrsDev, convDimLocal, dConvMinus1, numLayers, inputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        splitRnnConvStateKernel<T, subWarpSize, 8><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputConvPtrsDev, outputCachePtrsDev, convDimLocal, dConvMinus1, numLayers, inputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    default:
    {
        // Fallback to 4-byte vectorization
        splitRnnConvStateKernel<T, subWarpSize, 4><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputConvPtrsDev, outputCachePtrsDev, convDimLocal, dConvMinus1, numLayers, inputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void splitRnnSsmState(std::vector<runtime::ITensor::SharedPtr> const& inputSsmBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, size_t convBytesPerLayer,
    rnn_cache::RnnCacheState const& destCacheState, rnn_cache::RnnCacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSsmBlocks.empty());

    size_t inputBlockNum = inputSsmBlocks.size();
    auto targetRankInfo = executor::kv_cache::targetIRanks(destCacheState, selfCacheState, selfIdx);
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
        TLLM_CHECK(ssmBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(ssmBlock->data()));
    }

    for (size_t i = 0; i < outputSplitBlocks.size(); i++)
    {
        // Calculate offset for this specific target
        SizeType32 layersForThisTarget = targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
        size_t ssmOffset = layersForThisTarget * convBytesPerLayer;

        // Apply offset to UINT8 buffer
        uint8_t* basePtr = static_cast<uint8_t*>(outputSplitBlocks[i]->data());
        T* typedPtr = reinterpret_cast<T*>(basePtr + ssmOffset);

        TLLM_CHECK_WITH_INFO(outputSplitBlocks[i]->getDataType() == cacheDataType
                || outputSplitBlocks[i]->getDataType() == nvinfer1::DataType::kUINT8,
            "outputSplitBlock data type mismatch");
        cachePtrs.push_back(reinterpret_cast<uint64_t>(typedPtr));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfParallelConfig.mRnnLayerNumPerPP.at(selfPPRank);

    // Note: mNumHeads is GLOBAL, need to divide by TP
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPNum;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "splitRnnSsmState - numLayers: %d, numHeadsLocal: %d, headDim: %d, dState: %d, "
        "domainPPSize: %d, domainTPSize: %d, headNumDomainTP: %d",
        numLayers, numHeadsLocal, headDim, dState, domainPPSize, domainTPSize, headNumDomainTP);

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
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev,
                numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev,
                numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    default:
    {
        // Fallback to 4-byte vectorization
        splitRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 4>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputSsmPtrsDev, outputCachePtrsDev,
                numHeadsLocal, headDim, dState, numLayers, inputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatRnnConvState(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputConvBlocks, rnn_cache::RnnCacheState const& destCacheState,
    rnn_cache::RnnCacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    TLLM_CHECK(!outputConvBlocks.empty());

    size_t outputBlockNum = outputConvBlocks.size();

    auto targetRankInfo = executor::kv_cache::targetIRanks(destCacheState, selfCacheState, selfIdx);
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
        TLLM_CHECK(outputBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    for (auto const& inputBlock : inputSplitBlocks)
    {
        TLLM_CHECK_WITH_INFO(inputBlock->getDataType() == cacheDataType, "inputSplitBlock data type mismatch");
        cachePtrs.push_back(reinterpret_cast<uint64_t>(inputBlock->data()));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfParallelConfig.mRnnLayerNumPerPP.at(selfPPRank);
    int const convDimLocal = selfModelConfig.mConvDimSize / selfTPNum;
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
    int const remainder = dConvMinus1 * sizeof(T) % 16;

    switch (remainder)
    {
    case 0:
    {
        concatRnnConvStateKernel<T, subWarpSize, 16><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputCachePtrsDev, outputConvPtrsDev, convDimLocal, dConvMinus1, numLayers, outputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        concatRnnConvStateKernel<T, subWarpSize, 8><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputCachePtrsDev, outputConvPtrsDev, convDimLocal, dConvMinus1, numLayers, outputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    default:
    {
        // Fallback to 4-byte vectorization
        concatRnnConvStateKernel<T, subWarpSize, 4><<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(
            inputCachePtrsDev, outputConvPtrsDev, convDimLocal, dConvMinus1, numLayers, outputBlockNum, domainPPSize,
            convDimDomainTP, prefixLayerNumDevPtr);
        break;
    }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void concatRnnSsmState(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSsmBlocks, size_t convBytesPerLayer,
    rnn_cache::RnnCacheState const& destCacheState, rnn_cache::RnnCacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    TLLM_CHECK(!outputSsmBlocks.empty());

    size_t outputBlockNum = outputSsmBlocks.size();

    auto targetRankInfo = executor::kv_cache::targetIRanks(destCacheState, selfCacheState, selfIdx);
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
        TLLM_CHECK(outputBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputBlock->data()));
    }

    for (size_t i = 0; i < inputSplitBlocks.size(); i++)
    {
        // Calculate offset for this specific source
        SizeType32 layersFromThisSource = targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
        size_t ssmOffset = layersFromThisSource * convBytesPerLayer;

        // Apply offset to UINT8 buffer
        uint8_t* basePtr = static_cast<uint8_t*>(inputSplitBlocks[i]->data());
        T* typedPtr = reinterpret_cast<T*>(basePtr + ssmOffset);

        TLLM_CHECK_WITH_INFO(inputSplitBlocks[i]->getDataType() == cacheDataType
                || inputSplitBlocks[i]->getDataType() == nvinfer1::DataType::kUINT8,
            "inputSplitBlock data type mismatch");
        cachePtrs.push_back(reinterpret_cast<uint64_t>(typedPtr));
    }

    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(uint64_t));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();

    int const selfTPNum = selfParallelConfig.mTensorParallelism;
    int const selfPPRank = selfIdx / selfTPNum;
    int const numLayers = selfParallelConfig.mRnnLayerNumPerPP.at(selfPPRank);
    int const numHeadsLocal = selfModelConfig.mNumHeads / selfTPNum;
    int const headDim = selfModelConfig.mHeadDim;
    int const dState = selfModelConfig.mDState;

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const headNumDomainTP = numHeadsLocal / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);

    TLLM_LOG_DEBUG(
        "concatRnnSsmState - numLayers: %d, numHeadsLocal: %d, headDim: %d, dState: %d, "
        "domainPPSize: %d, domainTPSize: %d, headNumDomainTP: %d",
        numLayers, numHeadsLocal, headDim, dState, domainPPSize, domainTPSize, headNumDomainTP);

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
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev,
                numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    case 8:
    {
        concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev,
                numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    default:
    {
        // Fallback to 4-byte vectorization
        concatRnnSsmStateKernel<T, subWarpSize, subWarpNumInGroup, 4>
            <<<gridDim, blockDim, 0, bufferManager.getStream().get()>>>(inputCachePtrsDev, outputSsmPtrsDev,
                numHeadsLocal, headDim, dState, numLayers, outputBlockNum, domainPPSize, headNumDomainTP,
                prefixLayerNumDevPtr);
        break;
    }
    }

    TLLM_CUDA_CHECK(cudaGetLastError());
}

void splitRnnConvStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputConvBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, rnn_cache::RnnCacheState const& destCacheState,
    rnn_cache::RnnCacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputConvBlocks.empty());
    auto dataType = inputConvBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        splitRnnConvState<int64_t>(
            inputConvBlocks, outputSplitBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        splitRnnConvState<int32_t>(
            inputConvBlocks, outputSplitBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        splitRnnConvState<int16_t>(
            inputConvBlocks, outputSplitBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        splitRnnConvState<int8_t>(
            inputConvBlocks, outputSplitBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("splitRnnConvStateDispatch: unsupported data type");
    }
}

void splitRnnSsmStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSsmBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, size_t convBytesPerLayer,
    rnn_cache::RnnCacheState const& destCacheState, rnn_cache::RnnCacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSsmBlocks.empty());
    auto dataType = inputSsmBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        splitRnnSsmState<int64_t>(inputSsmBlocks, outputSplitBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 4:
        splitRnnSsmState<int32_t>(inputSsmBlocks, outputSplitBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 2:
        splitRnnSsmState<int16_t>(inputSsmBlocks, outputSplitBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 1:
        splitRnnSsmState<int8_t>(inputSsmBlocks, outputSplitBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    default: TLLM_THROW("splitRnnSsmStateDispatch: unsupported data type");
    }
}

void concatRnnConvStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputConvBlocks, rnn_cache::RnnCacheState const& destCacheState,
    rnn_cache::RnnCacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    auto dataType = inputSplitBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        concatRnnConvState<int64_t>(
            inputSplitBlocks, outputConvBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 4:
        concatRnnConvState<int32_t>(
            inputSplitBlocks, outputConvBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 2:
        concatRnnConvState<int16_t>(
            inputSplitBlocks, outputConvBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    case 1:
        concatRnnConvState<int8_t>(
            inputSplitBlocks, outputConvBlocks, destCacheState, selfCacheState, selfIdx, bufferManager);
        break;
    default: TLLM_THROW("concatRnnConvStateDispatch: unsupported data type");
    }
}

void concatRnnSsmStateDispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSsmBlocks, size_t convBytesPerLayer,
    rnn_cache::RnnCacheState const& destCacheState, rnn_cache::RnnCacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{
    TLLM_CHECK(!inputSplitBlocks.empty());
    auto dataType = inputSplitBlocks.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
        concatRnnSsmState<int64_t>(inputSplitBlocks, outputSsmBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 4:
        concatRnnSsmState<int32_t>(inputSplitBlocks, outputSsmBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 2:
        concatRnnSsmState<int16_t>(inputSplitBlocks, outputSsmBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    case 1:
        concatRnnSsmState<int8_t>(inputSplitBlocks, outputSsmBlocks, convBytesPerLayer, destCacheState, selfCacheState,
            selfIdx, bufferManager);
        break;
    default: TLLM_THROW("concatRnnSsmStateDispatch: unsupported data type");
    }
}

} // namespace tensorrt_llm::executor::rnn_cache
