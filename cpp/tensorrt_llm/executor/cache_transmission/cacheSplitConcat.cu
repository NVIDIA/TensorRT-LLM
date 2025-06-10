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

#include "cacheSplitConcat.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
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

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
inline bool isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}
} // namespace

// inputBlockNums: [outputBlockNum, inputRanks.size]
// [PP, TP]
TargetRanksInfo TargetRanksInfoForDP(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto const& peerParConfig = peerCacheState.getParallelConfig();
    auto const& selfParConfig = selfCacheState.getParallelConfig();

    auto const peerPPNum = peerParConfig.mPipelineParallelism;
    auto const selfPPNum = selfParConfig.mPipelineParallelism;
    auto const peerTPNum = peerParConfig.mTensorParallelism;
    auto const selfTPNum = selfParConfig.mTensorParallelism;

    for (auto val : {peerPPNum, selfPPNum, peerTPNum, selfTPNum})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }

    auto const selfTPRank = selfRank % selfParConfig.mTensorParallelism;
    auto const selfPPRank = selfRank / selfParConfig.mTensorParallelism;

    int peerPPRankStart = 0;
    int mDomainPPSize = 1;
    int peerPPRankEnd = 0;

    if (selfPPNum <= peerPPNum)
    {
        mDomainPPSize = peerPPNum / selfPPNum;
        peerPPRankStart = selfPPRank * mDomainPPSize;
        peerPPRankEnd = (selfPPRank + 1) * mDomainPPSize;
    }
    else
    {
        peerPPRankStart = selfPPRank / (selfPPNum / peerPPNum);
        peerPPRankEnd = peerPPRankStart + mDomainPPSize;
    }

    int peerTPRankStart = 0;
    int mDomainTPSize = 1;
    int peerTPRankEnd = 0;

    int const peerDpRank = peerParConfig.mEnableAttentionDP ? peerParConfig.mDPrank : 0;
    int const selfTPSizePerDPGroup = selfParConfig.mEnableAttentionDP ? selfTPNum / selfParConfig.mDPsize : selfTPNum;
    int const peerTPSizePerDPGroup = peerParConfig.mEnableAttentionDP ? peerTPNum / peerParConfig.mDPsize : peerTPNum;

    int const selfNbHeadsPerLayer = selfCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const peerNbHeadsPerLayer = peerCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const selfTPrankInDPGroup = selfTPRank % selfTPSizePerDPGroup;

    if (selfTPSizePerDPGroup <= peerTPSizePerDPGroup)
    {
        mDomainTPSize = peerTPSizePerDPGroup / selfTPSizePerDPGroup;
        peerTPRankStart = selfTPrankInDPGroup * mDomainTPSize + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }
    else
    {
        peerTPRankStart
            = selfTPrankInDPGroup / (selfTPSizePerDPGroup / peerTPSizePerDPGroup) + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }

    std::vector<int> retRanks;
    for (int i = peerTPRankStart; i < peerTPRankEnd; i++)
    {
        for (int j = peerPPRankStart; j < peerPPRankEnd; j++)
        {
            int irank = j * peerTPNum + i;
            retRanks.push_back(irank);
        }
    }

    int mDupHeadFactor = 1;
    int mPeerDupHeadFactor = 1;

    if (selfNbHeadsPerLayer * selfTPSizePerDPGroup > peerNbHeadsPerLayer * peerTPSizePerDPGroup)
    {
        mDupHeadFactor = (selfNbHeadsPerLayer * selfTPSizePerDPGroup) / (peerNbHeadsPerLayer * peerTPSizePerDPGroup);
    }
    if (peerNbHeadsPerLayer * peerTPSizePerDPGroup > selfNbHeadsPerLayer * selfTPSizePerDPGroup)
    {
        mPeerDupHeadFactor
            = (peerNbHeadsPerLayer * peerTPSizePerDPGroup) / (selfNbHeadsPerLayer * selfTPSizePerDPGroup);
    }

    return {mDomainPPSize, mDomainTPSize, std::move(retRanks), mDupHeadFactor, mPeerDupHeadFactor};
}

TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    return TargetRanksInfoForDP(peerCacheState, selfCacheState, selfRank);
}

template <typename T>
struct BlockInfo
{
    T* data;
    int startTokenId;
    int tokensPerBlock;
    int startHeadId;
    int headsPerBlock;
    int startLayerId;
    int layersPerBlock;
    int dimsPerHead;
    size_t offset; // (data-offset)[idx]

    __forceinline__ __device__ __host__ T* getKblockPtr(int layerid) const
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVblockPtr(int layerid) const
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getKDimsPtr(int layerid, int headid, int tokenid)
    {
        return getKblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getKDimsPtr(int layerid, int headid, int tokenid) const
    {
        return getKblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVDimsPtr(int layerid, int headid, int tokenid)
    {
        return getVblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getVDimsPtr(int layerid, int headid, int tokenid) const
    {
        return getVblockPtr(layerid) + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "{data ptr: " << data << ", startTokenId: " << startTokenId << ", tokensPerBlock: " << tokensPerBlock
           << ", startHeadId: " << startHeadId << ", headsPerBlock: " << headsPerBlock
           << ", startLayerId:" << startLayerId << ", layersPerBlock: " << layersPerBlock
           << ", dimsPerHead: " << dimsPerHead << ", offset: " << offset << "}";
        return ss.str();
    }
};

// Reference to blockPtr

// Block shape: [numHeads, numTokens, dimsPerHead]
// CacheBlock shape: [numLayers, 2, mBlockSize]
// Note: mBlockSize refers to the size of each block

// Handling key and value copying
// Note: k and v are not stored contiguously in memory

__forceinline__ __device__ int getInputBlockId(int outputBlockId, int headId, int layerId, int inputBlockNumEachOutput,
    int headNumPerBlock, int layerNumPerBlock, int headNumInputModel, int layerNumInputModel)
{
    int const offset = outputBlockId * inputBlockNumEachOutput;
    int const layerOffset = layerId / layerNumPerBlock;
    int const headOffset = headId / headNumPerBlock;
    int const headBlockNum = headNumInputModel / headNumPerBlock;
    return offset + layerOffset * headBlockNum + headOffset;
}

// subWarpSize * subWarpGroupSize
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitAndConcatBlocksKernel(BlockInfo<T> const* iBlockInfo, BlockInfo<T>* oBlockInfo, int iBlockNum,
    int iNumBlockEachO, int oBlockNum, int headNumInputModel, int layerNumInputModel, int iHeadsPerBlock,
    int iLayersPerBlock)
{
    // blockDim.y corresponds to the number of output blocks
    // blockDim.x corresponds to the number of layers

    // Warp-level parallelism spans heads * tokens
    // Thread-level parallelism spans dimsPerHead

    // input_id can be derived from output_id, layer_id, and head_id
    // Total number of CUDA blocks = numLayers * outputBlockNum

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

#pragma unroll 1
    for (int oBlockId = blockIdx.y; oBlockId < oBlockNum; oBlockId += gridDim.y)
    {
        int oLayerNum = oBlockInfo[oBlockId].layersPerBlock;
        int headNum = oBlockInfo[oBlockId].headsPerBlock;
        int tokenNum = oBlockInfo[oBlockId].tokensPerBlock;
        int dimsPerHead = oBlockInfo[oBlockId].dimsPerHead;
#pragma unroll 1

        for (int layerid = blockIdx.x; layerid < oLayerNum; layerid += gridDim.x)
        {
#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {
                int const targetHeadId = oBlockInfo[oBlockId].startHeadId + headId;
                int const targetLayerId = oBlockInfo[oBlockId].startLayerId + layerid;

                int const iBlockId = getInputBlockId(oBlockId, targetHeadId, targetLayerId, iNumBlockEachO,
                    iHeadsPerBlock, iLayersPerBlock, headNumInputModel, layerNumInputModel);
                int const iLayerId = targetLayerId % iLayersPerBlock;
                int const iHeadId = targetHeadId % iHeadsPerBlock;
#pragma unroll 1
                for (int tokenId = subWarpIdInGroup; tokenId < tokenNum; tokenId += subWarpNumInGroup)
                {
                    T* oKPtr = oBlockInfo[oBlockId].getKDimsPtr(layerid, headId, tokenId);
                    T const* iKPtr = iBlockInfo[iBlockId].getKDimsPtr(iLayerId, iHeadId, tokenId);
                    T* oVPtr = oBlockInfo[oBlockId].getVDimsPtr(layerid, headId, tokenId);
                    T const* iVPtr = iBlockInfo[iBlockId].getVDimsPtr(iLayerId, iHeadId, tokenId);
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        common::copy<vecSizeByte>(iKPtr + channelId, oKPtr + channelId);
                        common::copy<vecSizeByte>(iVPtr + channelId, oVPtr + channelId);
                    }
                }
            }
        }
    }
}

template <typename T>
void concatKVCache(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum, std::vector<int> const& inputRanks,
    kv_cache::CacheState const& iCacheState, runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRank,
    kv_cache::CacheState const& oCacheState, runtime::BufferManager const& bufferManager)

{
    TLLM_CHECK_WITH_INFO(!inputRanks.empty(), "inputRanks should not be empty.");
    TLLM_CHECK_WITH_INFO(inputBlockNum == outputBlockNum * inputRanks.size(),
        "inputBlockNum must equal outputBlockNum multiplied by the size of inputRanks.");
    TLLM_CHECK(inputRanks == targetIRanks(iCacheState, oCacheState, oRank).mIRanks);

    auto const& iParallelConfig = iCacheState.getParallelConfig();
    auto const& oParallelConfig = oCacheState.getParallelConfig();
    auto const& iModelConfig = iCacheState.getModelConfig();
    auto const& oModelConfig = oCacheState.getModelConfig();

    int const inputAllRankNum = iParallelConfig.mPipelineParallelism * iParallelConfig.mTensorParallelism;
    std::vector<BlockInfo<T>> blockInfos(outputBlockNum * inputAllRankNum + outputBlockNum);

    auto fillBlockInfo = [](kv_cache::CacheState const& cacheState, runtime::ITensor::SharedPtr buffer, int rank)
    {
        const auto& parallelConfig = cacheState.getParallelConfig();
        const auto& modelConfig = cacheState.getModelConfig();

        const int tpRank = rank % parallelConfig.mTensorParallelism;
        const int ppRank = rank / parallelConfig.mTensorParallelism;
        const int ppNum = parallelConfig.mPipelineParallelism;
        const int headsPerBlock = modelConfig.mNbKvHeadsPerLayer[0];
        const int layersPerBlock = modelConfig.mNbKvHeadsPerLayer.size() / ppNum;

        const int tokensPerBlock = modelConfig.mTokensPerBlock;
        const int dimsPerBlock = modelConfig.mSizePerHead;
        const int startHead = tpRank * headsPerBlock;
        const int startLayer = ppRank * layersPerBlock;

        constexpr int startTokenId = 0;
        auto* data = static_cast<T*>(buffer->data());
        return BlockInfo<T>{
            data, startTokenId, tokensPerBlock, startHead, headsPerBlock, startLayer, layersPerBlock, dimsPerBlock, 0};
    };
    // fill blcokInfo from CacheState and inputBlocks
    for (int oi = 0; oi < outputBlockNum; oi++)
    {
        int iRankNum = inputRanks.size();
        for (int i = 0; i < iRankNum; i++)
        {
            int iRank = inputRanks[i];
            blockInfos[oi * inputAllRankNum + iRank]
                = fillBlockInfo(iCacheState, inputBlocks[oi * iRankNum + i], iRank);
        }

        blockInfos[outputBlockNum * inputAllRankNum + oi] = fillBlockInfo(oCacheState, outputBlocks[oi], oRank);
    }
    runtime::BufferManager::IBufferPtr blockInfosDeviceBuffer
        = bufferManager.gpu(sizeof(BlockInfo<T>) * (blockInfos.size()), nvinfer1::DataType::kUINT8);
    bufferManager.copy((blockInfos.data()), *blockInfosDeviceBuffer, runtime::MemoryType::kCPU);

    BlockInfo<T>* iBlockInfoDevice = static_cast<BlockInfo<T>*>(blockInfosDeviceBuffer->data());
    BlockInfo<T>* oBlockInfoDevice = iBlockInfoDevice + outputBlockNum * inputAllRankNum;

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    constexpr int blockDimx = 128;

    int oPPNum = oParallelConfig.mPipelineParallelism;
    int iPPNum = iParallelConfig.mPipelineParallelism;
    unsigned int gridDimx = oModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const headsInputModel = iModelConfig.mNbKvHeadsPerLayer[0] * iParallelConfig.mTensorParallelism;
    int const layersInputModel = iModelConfig.mNbKvHeadsPerLayer.size();
    int const iHeadsPerBlock = iModelConfig.mNbKvHeadsPerLayer[0];
    int const iLayersPerBlock = iModelConfig.mNbKvHeadsPerLayer.size() / iPPNum;
    int const sizePerHead = oModelConfig.mSizePerHead;
    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 8:
    {
        splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 8>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 4>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {

            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 2>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            splitAndConcatBlocksKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
        }
        else
        {
            TLLM_THROW("concatKVCacheDispatch encountered an unsupported data type error.");
        }
    }
    }
}

void concatKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
    std::vector<int> const& inputRanks, kv_cache::CacheState const& iCacheState,
    runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRanks, kv_cache::CacheState const& oCacheState,
    runtime::BufferManager const& bufferManager)
{
    auto dataType = outputBlocks[0]->getDataType();
    int dataSize = tensorrt_llm::common::getDTypeSize(dataType);

    switch (dataSize)
    {
    case 8:
    {
        concatKVCache<int64_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 4:
    {
        concatKVCache<int32_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 2:
    {
        concatKVCache<int16_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    case 1:
    {
        concatKVCache<int8_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum, oRanks,
            oCacheState, bufferManager);
        break;
    }

    default:
    {
        TLLM_THROW("concatKVCacheDispatch encountered an unsupported data type error.");
    }
    }
}

nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState)
{

    int64_t blockSize = static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer[0]
        * cacheState.getModelConfig().mTokensPerBlock * cacheState.getModelConfig().mSizePerHead);
    int PPNum = cacheState.getParallelConfig().mPipelineParallelism;
    return runtime::ITensor::makeShape(
        {static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / PPNum),
            cacheState.getAttentionConfig().mKvFactor, blockSize});
}

// MLA Head 1: One thread block per [(2), tokens, dimsPerHead]

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitKVCacheForMLAKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int kvFactor)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1

    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1

        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
#pragma unroll 1
            for (int headId = 0; headId < headNum; headId++)
            {
                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                int const outputCacheIdx = layerId / layerNumDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];
                int const layerIdInDomainPP = layerId % layerNumDomainPP;
                int const headIdInDomainTP = headId;

                T* kOutputPtr = outputCachePtr
                    + blockId * (layerNumDomainPP * kvFactor * headNum * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;
                int const kvOffset = headNum * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpId; tokenId < tokensPerBlock; tokenId += subWarpNum)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += subWarpSize * numElePerThread)
                    {
#pragma unroll 1
                        for (int kvId = 0; kvId < kvFactor; kvId++)
                        {
                            common::copy<vecSizeByte>(
                                iKPtr + kvId * kvOffset + channelId, oKPtr + kvId * kvOffset + channelId);
                        }
                    }
                }
            }
        }
    }
}

// Block shape: [head, tokens, dimsPerHead]
// CacheBlock shape: [numLayers, 2, mBlockSize]
// Output split caches shape: [outputSplitCaches, numLayers, 2, head, tokensPerBlock, dimsPerHead]
// Note: The number of tokens can be large
// subWarpSize * subWarpGroupSize

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitKVCacheKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int headNumDomainTP)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1

    for (int blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1

        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
#pragma unroll 1

            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T const* vInputPtr = inputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int outputCacheIdx = headId / headNumDomainTP * DomainPPSize + layerId / layerNumDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];
                int layerIdInDomainPP = layerId % layerNumDomainPP;

                int headIdInDomainTP = headId % headNumDomainTP;
                T* kOutputPtr = outputCachePtr
                    + blockId * (layerNumDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;

                T* vOutputPtr = kOutputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1

                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        auto offset = baseOffset + channelId;
                        common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                        common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                    }
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatKVCacheForMLAKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int kvFactor)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {

#pragma unroll 1

            for (int headId = 0; headId < headNum; headId++)
            {
                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                int inputCacheIdx = layerId / layerNumDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];
                int layerIdInDomainPP = layerId % layerNumDomainPP;
                int headIdInDomainTP = headId;

                T const* kInputPtr = inputCachePtr
                    + blockId * (layerNumDomainPP * kvFactor * headNum * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;
                int const kvOffset = headNum * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpId; tokenId < tokensPerBlock; tokenId += subWarpNum)
                {
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += subWarpSize * numElePerThread)
                    {
#pragma unroll 1
                        for (int kvId = 0; kvId < kvFactor; kvId++)
                        {
                            common::copy<vecSizeByte>(
                                iKPtr + kvId * kvOffset + channelId, oKPtr + kvId * kvOffset + channelId);
                        }
                    }
                }
            }
        }
    }
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void concatKVCacheKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int DomainPPSize,
    int DomainTPSize, int layerNumDomainPP, int headNumDomainTP)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1
    for (int blockId = blockIdx.y; blockId < outputBlockNum; blockId += gridDim.y)
    {
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T* vOutputPtr = outputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int inputCacheIdx = headId / headNumDomainTP * DomainPPSize + layerId / layerNumDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];
                int layerIdInDomainPP = layerId % layerNumDomainPP;

                int headIdInDomainTP = headId % headNumDomainTP;
                T const* kInputPtr = inputCachePtr
                    + blockId * (layerNumDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead)
                    + layerIdInDomainPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + headIdInDomainTP * tokensPerBlock * dimsPerHead;

                T const* vInputPtr = kInputPtr + headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1
                for (int tokenId = subWarpIdInGroup; tokenId < tokensPerBlock; tokenId += subWarpNumInGroup)
                {
                    auto baseOffset = tokenId * dimsPerHead;
#pragma unroll 1
                    for (int channelId = laneId * numElePerThread; channelId < dimsPerHead;
                         channelId += (subWarpSize * numElePerThread))
                    {
                        auto offset = baseOffset + channelId;
                        common::copy<vecSizeByte>(kInputPtr + offset, kOutputPtr + offset);
                        common::copy<vecSizeByte>(vInputPtr + offset, vOutputPtr + offset);
                    }
                }
            }
        }
    }
}

template <typename T>
void splitKVCache(std::vector<runtime::ITensor::SharedPtr> const& kVCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{

    auto inputBlockNum = kVCacheBlocks.size();
    auto targetRankInfo = targetIRanks(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));
    auto outputCacheNum = targetRankInfo.mIRanks.size();
    if (selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        outputCacheNum = targetRankInfo.mDomainPPSize;
    }
    else
    {
        outputCacheNum = outputCacheNum / targetRankInfo.mPeerDupHeadFactor;
    }
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);
    auto cacheBlockSize = kVCacheBlocks.at(0)->getSize();
    auto cacheDataType = kVCacheBlocks.at(0)->getDataType();
    std::vector<T*> cachePtrs;

    for (auto&& kvCacheBlock : kVCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        cachePtrs.push_back(static_cast<T*>(kvCacheBlock->data()));
    }

    for (auto&& outputSplitBlock : outputSplitBlocks)
    {
        TLLM_CHECK(outputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(outputSplitBlock->getSize() == cacheBlockSize * inputBlockNum / outputCacheNum);
        cachePtrs.push_back(static_cast<T*>(outputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(T*));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    constexpr int blockDimx = 128;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();
    auto const& destParallelConfig = destCacheState.getParallelConfig();
    auto const& selfAttentionConfig = selfCacheState.getAttentionConfig();

    int oPPNum = selfParallelConfig.mPipelineParallelism;

    // layers
    unsigned int gridDimx = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    // blockNum
    unsigned int gridDimy = inputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};

    int const sizePerHead = selfModelConfig.mSizePerHead;
    T const** inputBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const numLayers = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];
    int const dimsPerHead = selfModelConfig.mSizePerHead;
    int const DomainPPSize = targetRankInfo.mDomainPPSize;
    int const DomainTPSize = targetRankInfo.mDomainTPSize;
    int const layerNumDomainPP = numLayers / DomainPPSize;
    int const headNumDomainTP
        = headNum / (DomainTPSize / targetRankInfo.mPeerDupHeadFactor); // TODO: duplicate head factor
    int const kvFactor = selfAttentionConfig.mKvFactor;
    bool const isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    constexpr int mlaSubWarpSize = 16;

    TLLM_LOG_DEBUG(
        "splitKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "layersPerDomainPP: %d, headsPerDomainTP: %d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);

    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 16><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputBlockPtrsDev, outputCachePtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum,
                DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 8><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputBlockPtrsDev, outputCachePtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum,
                DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
        }
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, kvFactor);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, headNumDomainTP);
            }
            break;
        }
    }

    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, kvFactor);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, headNumDomainTP);
            }
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            if (isMLA)
            {
                splitKVCacheForMLAKernel<T, mlaSubWarpSize, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, kvFactor);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                        layerNumDomainPP, headNumDomainTP);
            }
            break;
        }
        else
        {
            TLLM_THROW("splitKVCacheDispatch encountered an unsupported data type error.");
        }
    }
    }
}

void splitKVCacheDispatch(std::vector<runtime::ITensor::SharedPtr> const& kVCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr>& ouputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{
    auto dataType = kVCacheBlocks.at(0)->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        splitKVCache<int64_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        splitKVCache<int32_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        splitKVCache<int16_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        splitKVCache<int8_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW("splitKVCacheDispatch encountered an unsupported data type error.");
    }
    }
}

template <typename T>
void concatKVCache(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks,

    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager)
{

    auto outputBlockNum = outputKvCacheBlocks.size();
    auto targetRankInfo = targetIRanks(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize)));

    auto inputCacheNum = targetRankInfo.mIRanks.size();
    if (selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        inputCacheNum = targetRankInfo.mDomainPPSize;
    }
    else
    {
        inputCacheNum = inputCacheNum / targetRankInfo.mPeerDupHeadFactor;
    }
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);
    auto cacheBlockSize = outputKvCacheBlocks.at(0)->getSize();
    auto cacheDataType = outputKvCacheBlocks.at(0)->getDataType();
    std::vector<T*> cachePtrs;
    for (auto&& kvCacheBlock : outputKvCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        cachePtrs.push_back(static_cast<T*>(kvCacheBlock->data()));
    }
    for (auto&& inputSplitBlock : inputSplitBlocks)
    {
        TLLM_CHECK(inputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(inputSplitBlock->getSize() == cacheBlockSize * outputBlockNum / inputCacheNum);
        cachePtrs.push_back(static_cast<T*>(inputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(T*));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();
    auto const& destParallelConfig = destCacheState.getParallelConfig();
    auto const& selfAttentionConfig = selfCacheState.getAttentionConfig();

    int oPPNum = selfParallelConfig.mPipelineParallelism;
    // layers
    unsigned int gridDimx = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    // blockNum
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const sizePerHead = selfModelConfig.mSizePerHead;
    int const endLayerId = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    T** ouptutBlockPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputSplitBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const numLayers = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];
    int const dimsPerHead = selfModelConfig.mSizePerHead;
    int const DomainPPSize = targetRankInfo.mDomainPPSize;
    int const DomainTPSize = targetRankInfo.mDomainTPSize;

    int const layerNumDomainPP = numLayers / DomainPPSize;
    int const headNumDomainTP
        = headNum / (DomainTPSize / targetRankInfo.mPeerDupHeadFactor); // TODO: duplicate head factor
    int const kvFactor = selfAttentionConfig.mKvFactor;

    bool isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    TLLM_LOG_DEBUG(
        "concatKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "layersPerDomainPP: %d, headsPerDomainTP: %d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);

    int const remainder = sizePerHead * sizeof(T) % 16;

    int const mlaSubWarpSize = 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            concatKVCacheForMLAKernel<T, mlaSubWarpSize, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, kvFactor);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            concatKVCacheForMLAKernel<T, mlaSubWarpSize, 8><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputSplitBlockPtrsDev, ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead,
                outputBlockNum, DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, headNumDomainTP);
        }
        break;
    }
    case 4:
    case 12:
    {
        if constexpr (sizeof(T) <= 4)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
            }

            break;
        }
    }
    case 2:
    case 6:
    case 10:
    case 14:
    {
        if constexpr (sizeof(T) <= 2)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
            }
            break;
        }
    }
    default:
    {
        if constexpr (sizeof(T) <= 1)
        {
            if (isMLA)
            {
                concatKVCacheForMLAKernel<T, mlaSubWarpSize, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
            }
            break;
        }
        else
        {
            TLLM_THROW("concatKVCache encountered an unsupported data type error.");
        }
    }
    }
}

void concatKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{

    auto dataType = outputKvCacheBlocks.at(0)->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        concatKVCache<int64_t>(inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        concatKVCache<int32_t>(inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        concatKVCache<int16_t>(inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        concatKVCache<int8_t>(inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW("concatKVCache encountered an unsupported data type error.");
    }
    }
}

} // namespace tensorrt_llm::executor::kv_cache
