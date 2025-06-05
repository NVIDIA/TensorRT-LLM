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

#include "cacheConcatenate.h"
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

bool isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// inputBlockNums:[ outputBlockNum , inputRanks.size]
// [PP,TP]
TargetRanksInfo TargetRanksInfoForDP(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{

    TLLM_CHECK(isPowerOfTwo(peerCacheState.getParallelConfig().mTensorParallelism));
    TLLM_CHECK(isPowerOfTwo(selfCacheState.getParallelConfig().mTensorParallelism));
    TLLM_CHECK(isPowerOfTwo(peerCacheState.getParallelConfig().mPipelineParallelism));
    TLLM_CHECK(isPowerOfTwo(selfCacheState.getParallelConfig().mPipelineParallelism));
    int selfTpRank = selfRank % selfCacheState.getParallelConfig().mTensorParallelism;
    int selfPpRank = selfRank / selfCacheState.getParallelConfig().mTensorParallelism;
    int peerPPNum = peerCacheState.getParallelConfig().mPipelineParallelism;
    int selfPPNum = selfCacheState.getParallelConfig().mPipelineParallelism;
    int peerTPNum = peerCacheState.getParallelConfig().mTensorParallelism;
    int selfTPNum = selfCacheState.getParallelConfig().mTensorParallelism;
    int peerPpRankStart = 0;
    int mDomainPPSize = 1;
    int peerPpRankEnd = 0;
    if (selfPPNum <= peerPPNum)
    {
        mDomainPPSize = peerPPNum / selfPPNum;
        peerPpRankStart = selfPpRank * mDomainPPSize;
        peerPpRankEnd = (selfPpRank + 1) * mDomainPPSize;
    }
    else
    {
        peerPpRankStart = selfPpRank / (selfPPNum / peerPPNum);
        peerPpRankEnd = peerPpRankStart + mDomainPPSize;
    }
    int peerTpRankStart = 0;
    int mDomainTPSize = 1;
    int peerTpRankEnd = 0;
    int peerDpRank
        = peerCacheState.getParallelConfig().mEnableAttentionDP ? peerCacheState.getParallelConfig().mDPrank : 0;
    int selfTPSizeOneDPGroup = selfCacheState.getParallelConfig().mEnableAttentionDP
        ? selfCacheState.getParallelConfig().mTensorParallelism / selfCacheState.getParallelConfig().mDPsize
        : selfTPNum;

    int peerTPSizeOneDPGroup = peerCacheState.getParallelConfig().mEnableAttentionDP
        ? peerCacheState.getParallelConfig().mTensorParallelism / peerCacheState.getParallelConfig().mDPsize
        : peerTPNum;

    int selfNbHeadsPerLayer = selfCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int peerNbHeadsPerLayer = peerCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int selfTPrankInDPGroup = selfTpRank % selfTPSizeOneDPGroup;

    {
        if (selfTPSizeOneDPGroup <= peerTPSizeOneDPGroup)
        {
            mDomainTPSize = peerTPSizeOneDPGroup / selfTPSizeOneDPGroup;
            peerTpRankStart = selfTPrankInDPGroup * mDomainTPSize + peerDpRank * peerTPSizeOneDPGroup;
            peerTpRankEnd = peerTpRankStart + mDomainTPSize;
        }
        else
        {
            peerTpRankStart = selfTPrankInDPGroup / (selfTPSizeOneDPGroup / peerTPSizeOneDPGroup)
                + peerDpRank * peerTPSizeOneDPGroup;
            peerTpRankEnd = peerTpRankStart + mDomainTPSize;
        }
    }

    std::vector<int> retRanks;

    for (int i = peerTpRankStart; i < peerTpRankEnd; i++)
    {
        for (int j = peerPpRankStart; j < peerPpRankEnd; j++)
        {
            int irank = j * peerTPNum + i;
            retRanks.push_back(irank);
        }
    }
    int mDuplicateHeadFactor = 1;
    int mPeerDuplicateHeadFactor = 1;
    if (selfNbHeadsPerLayer * selfTPSizeOneDPGroup > peerNbHeadsPerLayer * peerTPSizeOneDPGroup)
    {
        mDuplicateHeadFactor
            = (selfNbHeadsPerLayer * selfTPSizeOneDPGroup) / (peerNbHeadsPerLayer * peerTPSizeOneDPGroup);
    }
    if (peerNbHeadsPerLayer * peerTPSizeOneDPGroup > selfNbHeadsPerLayer * selfTPSizeOneDPGroup)
    {
        mPeerDuplicateHeadFactor
            = (peerNbHeadsPerLayer * peerTPSizeOneDPGroup) / (selfNbHeadsPerLayer * selfTPSizeOneDPGroup);
    }

    return {mDomainPPSize, mDomainTPSize, std::move(retRanks), mDuplicateHeadFactor, mPeerDuplicateHeadFactor};
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

    __forceinline__ __device__ __host__ T* getKblockPtr(int layerid)
    {
        // return layerid- startLayerId
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVblockPtr(int layerid)
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getKDimsPtr(int layerid, int headid, int tokenid)
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getKDimsPtr(int layerid, int headid, int tokenid) const
    {
        return data + (layerid * 2) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T* getVDimsPtr(int layerid, int headid, int tokenid)
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    __forceinline__ __device__ __host__ T const* getVDimsPtr(int layerid, int headid, int tokenid) const
    {
        return data + (layerid * 2 + 1) * headsPerBlock * tokensPerBlock * dimsPerHead
            + headid * tokensPerBlock * dimsPerHead + tokenid * dimsPerHead;
    }

    std::string to_string()
    {
        std::stringstream ss;
        ss << "{data ptr: " << data << "startTokenId: " << startTokenId << "tokensPerBlock:  " << tokensPerBlock
           << " startHeadId: " << startHeadId << "headsPerBlock: " << headsPerBlock << "startLayerId:" << startLayerId
           << "layersPerBlock: " << layersPerBlock << "dimsPerHead: " << dimsPerHead << " offset: " << offset << "}";
        return ss.str();
    }
};

// refer blockPtr

// Block shape [ head,tokens,dimsPerHead]
//  CacheBlock [numLayers,2,mBlockSize] . BlockSize[

// kV  and copy

// note k and v not continuous

__forceinline__ __device__ int getInputBlockId(int outputBlockId, int headId, int layerId, int inputBlockNumEachOutput,
    int headNumPerBlock, int layerNumPerBlock, int headNumInputModel, int layerNumInputModel)
{

    int offset = outputBlockId * inputBlockNumEachOutput;

    int layerOffset = layerId / layerNumPerBlock;

    int headOffset = headId / headNumPerBlock;

    int headBlockNum = headNumInputModel / headNumPerBlock;
    return offset + layerOffset * headBlockNum + headOffset;
}

// subWarpSize*subWarpGroupSize
template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitAndConcatenateBlocksKernel(BlockInfo<T> const* iBlockInfo, BlockInfo<T>* oBlockInfo, int iBlockNum,
    int iNumBlockEachO, int oBlockNum, int headNumInputModel, int layerNumInputModel, int iHeadsPerBlock,
    int iLayersPerBlock)
{

    // for blockDim.y for output_blockNum
    // blockDim.x for layer

    // wraps for heads*tokens
    // threads for dimsPerHead

    // input_id can be decided by outputid,layerid,headid
    // cuda blockNum layers*oBlockNum

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup; //
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    // using VecType = typename common::packed_as<T,numElePerThread>::type;
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
void concatenateKVCache(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum, std::vector<int> const& inputRanks,
    kv_cache::CacheState const& iCacheState, runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int oRank,
    kv_cache::CacheState const& oCacheState, runtime::BufferManager const& bufferManager)

{

    TLLM_CHECK_WITH_INFO(!inputRanks.empty(), "input should not be empty!");
    TLLM_CHECK_WITH_INFO(
        inputBlockNum == outputBlockNum * inputRanks.size(), "inputBlockNum==outputBlockNum*inputRanks.size()");

    TLLM_CHECK(inputRanks == targetIRanks(iCacheState, oCacheState, oRank).mIRanks);
    int const inputAllRankNum
        = iCacheState.getParallelConfig().mPipelineParallelism * iCacheState.getParallelConfig().mTensorParallelism;
    std::vector<BlockInfo<T>> blockInfos(outputBlockNum * inputAllRankNum + outputBlockNum);

    auto fillBlockInfo = [](kv_cache::CacheState const& cacheState, runtime::ITensor::SharedPtr buffer, int rank)
    {
        int tpRank = rank % cacheState.getParallelConfig().mTensorParallelism;
        int ppRank = rank / cacheState.getParallelConfig().mTensorParallelism;
        int ppNum = cacheState.getParallelConfig().mPipelineParallelism;
        int headsPerBlock = cacheState.getModelConfig().mNbKvHeadsPerLayer[0];
        int layersPerBlock = cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / ppNum; //  TODO:need  / PPSize?

        int tokensPerBlock = cacheState.getModelConfig().mTokensPerBlock;
        int dimsPerBlock = cacheState.getModelConfig().mSizePerHead;
        int startHead = tpRank * headsPerBlock;
        int startLayer = ppRank * layersPerBlock;
        // TODO:just ignore start Tokenid
        int startTokenId = 0;
        T* data = static_cast<T*>(buffer->data());
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
    int blockDimx = 128;
    int oPpNum = oCacheState.getParallelConfig().mPipelineParallelism;
    int iPpNum = iCacheState.getParallelConfig().mPipelineParallelism;
    unsigned int gridDimx = oCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const headsInputModel
        = iCacheState.getModelConfig().mNbKvHeadsPerLayer[0] * iCacheState.getParallelConfig().mTensorParallelism;
    int const layersInputModel = iCacheState.getModelConfig().mNbKvHeadsPerLayer.size();
    int const iHeadsPerBlock = iCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const iLayersPerBlock = iCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / iPpNum;
    int const sizePerHead = oCacheState.getModelConfig().mSizePerHead;
    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 16>
            <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel, layersInputModel,
                iHeadsPerBlock, iLayersPerBlock);
        break;
    }
    case 8:
    {
        splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 8>
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
            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 4>
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

            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 2>
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
            splitAndConcatenateBlocksKernel<T, subWarpSize, subWarpNumInGroup, 1>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(iBlockInfoDevice, oBlockInfoDevice,
                    outputBlockNum * inputAllRankNum, inputAllRankNum, outputBlockNum, headsInputModel,
                    layersInputModel, iHeadsPerBlock, iLayersPerBlock);
        }
        else
        {
            TLLM_THROW(" concatenateKVCacheDispatch no support data type error");
        }
    }
    }
}

void concatenateKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
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
        concatenateKVCache<double>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 4:
    {
        concatenateKVCache<float>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }
    case 2:
    {

        concatenateKVCache<half>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    case 1:
    {

        concatenateKVCache<uint8_t>(inputBlocks, inputBlockNum, inputRanks, iCacheState, outputBlocks, outputBlockNum,
            oRanks, oCacheState, bufferManager);
        break;
    }

    default:
    {
        TLLM_THROW(" concatenateKVCacheDispatch no support");
    }
    }
}

nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState)
{

    int64_t blockSize = static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer[0]
        * cacheState.getModelConfig().mTokensPerBlock * cacheState.getModelConfig().mSizePerHead);
    int PpNum = cacheState.getParallelConfig().mPipelineParallelism;
    return runtime::ITensor::makeShape(
        {static_cast<int64_t>(cacheState.getModelConfig().mNbKvHeadsPerLayer.size() / PpNum),
            cacheState.getAttentionConfig().mKvFactor, blockSize});
}

// MLA head 1 , one threadBlock for [(2), tokens,dimsPerHead]

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitKVCacheForMLAKernelForMLA(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
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
                int outputCacheIdx = layerId / layerNumDomainPP;

                T* outputCachePtr = outputCaches[outputCacheIdx];

                int layerIdInDomainPP = layerId % layerNumDomainPP;

                int headIdInDomainTP = headId;

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

// Block shape [ head,tokens,dimsPerHead]
//  CacheBlock [numLayers,2,mBlockSize] .

//[outputSplitCaches,numLayers,2,head,tokens_per_block,dimsPerHead]
// tokens maybe large, so We

// subWarpSize*subWarpGroupSize
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
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T const* iVPtr = vInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
                    T* oVPtr = vOutputPtr + tokenId * dimsPerHead;
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

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatenateKVCacheForMLAKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
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
__global__ void concatenateKVCacheKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
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
                    T const* iKPtr = kInputPtr + tokenId * dimsPerHead;
                    T const* iVPtr = vInputPtr + tokenId * dimsPerHead;
                    T* oKPtr = kOutputPtr + tokenId * dimsPerHead;
                    T* oVPtr = vOutputPtr + tokenId * dimsPerHead;

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
        outputCacheNum = outputCacheNum / targetRankInfo.mPeerDuplicateHeadFactor;
    }
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);
    auto cacheBlockSize = kVCacheBlocks.at(0)->getSize();
    auto cacheDataType = kVCacheBlocks.at(0)->getDataType();
    std::vector<T*> CachePtrs;

    for (auto&& kvCacheBlock : kVCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        CachePtrs.push_back(reinterpret_cast<T*>(kvCacheBlock->data()));
    }

    for (auto&& outputSplitBlock : outputSplitBlocks)
    {
        TLLM_CHECK(outputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(outputSplitBlock->getSize() == cacheBlockSize * inputBlockNum / outputCacheNum);
        CachePtrs.push_back(reinterpret_cast<T*>(outputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(CachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == CachePtrs.size() * sizeof(T*));
    bufferManager.copy(CachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    int oPpNum = selfCacheState.getParallelConfig().mPipelineParallelism;
    // layers
    unsigned int gridDimx = selfCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    // blockNum
    unsigned int gridDimy = inputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};

    int const sizePerHead = selfCacheState.getModelConfig().mSizePerHead;
    T const** inputBlockPtrsDev = reinterpret_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = reinterpret_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNum;
    int tokensPerBlock = selfCacheState.getModelConfig().mTokensPerBlock;
    int numLayers = selfCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    int headNum = selfCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int dimsPerHead = selfCacheState.getModelConfig().mSizePerHead;
    int DomainPPSize = targetRankInfo.mDomainPPSize;
    int DomainTPSize = targetRankInfo.mDomainTPSize;
    int iPPNum = destCacheState.getParallelConfig().mPipelineParallelism;
    int iTPNum = destCacheState.getParallelConfig().mTensorParallelism;
    int oTPNum = selfCacheState.getParallelConfig().mTensorParallelism;
    int layerNumDomainPP = numLayers / DomainPPSize;
    int headNumDomainTP
        = headNum / (DomainTPSize / targetRankInfo.mPeerDuplicateHeadFactor); // TODO: duplicate head factor
    int kvFactor = selfCacheState.getAttentionConfig().mKvFactor;
    bool isMLA = selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA;
    constexpr int mlaSubWarpSize = 16;

    TLLM_LOG_DEBUG(
        "splitKVCache: numLayers: %d, headNum: %d, DomainPPSize:%d, DomainTPSize:%d, layerNumDomainPP:%d, "
        "headNumDomainTP:%d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);

    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernelForMLA<T, mlaSubWarpSize, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, kvFactor);
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
            splitKVCacheForMLAKernelForMLA<T, mlaSubWarpSize, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, kvFactor);
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
                splitKVCacheForMLAKernelForMLA<T, mlaSubWarpSize, 4>
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
                splitKVCacheForMLAKernelForMLA<T, mlaSubWarpSize, 2>
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
                splitKVCacheForMLAKernelForMLA<T, mlaSubWarpSize, 1>
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
            TLLM_THROW(" splitKVCacheDispatch no support data type error");
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
        splitKVCache<double>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        splitKVCache<float>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        splitKVCache<half>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        splitKVCache<uint8_t>(kVCacheBlocks, ouputSplitBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW(" splitKVCacheDispatch no support data type error");
    }
    }
}

template <typename T>
void concatenateKVCache(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
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
        inputCacheNum = inputCacheNum / targetRankInfo.mPeerDuplicateHeadFactor;
    }
    TLLM_CHECK(inputCacheNum == inputSplitBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);
    auto cacheBlockSize = outputKvCacheBlocks.at(0)->getSize();
    auto cacheDataType = outputKvCacheBlocks.at(0)->getDataType();
    std::vector<T*> CachePtrs;
    for (auto&& kvCacheBlock : outputKvCacheBlocks)
    {
        TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
        TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
        CachePtrs.push_back(reinterpret_cast<T*>(kvCacheBlock->data()));
    }
    for (auto&& inputSplitBlock : inputSplitBlocks)
    {
        TLLM_CHECK(inputSplitBlock->getDataType() == cacheDataType);
        TLLM_CHECK(inputSplitBlock->getSize() == cacheBlockSize * outputBlockNum / inputCacheNum);
        CachePtrs.push_back(reinterpret_cast<T*>(inputSplitBlock->data()));
    }
    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(CachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == CachePtrs.size() * sizeof(T*));
    bufferManager.copy(CachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    constexpr int subWarpSize = 8;
    constexpr int subWarpNumInGroup = 8;
    int blockDimx = 128;

    int oPpNum = selfCacheState.getParallelConfig().mPipelineParallelism;
    // layers
    unsigned int gridDimx = selfCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    // blockNum
    unsigned int gridDimy = outputBlockNum;

    dim3 gridDim{gridDimx, gridDimy};
    int const sizePerHead = selfCacheState.getModelConfig().mSizePerHead;
    int endLayerId = selfCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    T** ouptutBlockPtrsDev = reinterpret_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputSplitBlockPtrsDev = reinterpret_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNum;
    int tokensPerBlock = selfCacheState.getModelConfig().mTokensPerBlock;
    int numLayers = selfCacheState.getModelConfig().mNbKvHeadsPerLayer.size() / oPpNum;
    int headNum = selfCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int dimsPerHead = selfCacheState.getModelConfig().mSizePerHead;
    int DomainPPSize = targetRankInfo.mDomainPPSize;
    int DomainTPSize = targetRankInfo.mDomainTPSize;
    int iPPNum = destCacheState.getParallelConfig().mPipelineParallelism;
    int iTPNum = destCacheState.getParallelConfig().mTensorParallelism;
    int oTPNum = selfCacheState.getParallelConfig().mTensorParallelism;
    int layerNumDomainPP = numLayers / DomainPPSize;
    int headNumDomainTP
        = headNum / (DomainTPSize / targetRankInfo.mPeerDuplicateHeadFactor); // TODO: duplicate head factor
    int kvFactor = selfCacheState.getAttentionConfig().mKvFactor;

    bool isMLA = selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA;
    TLLM_LOG_DEBUG(
        "concatenateKVCache: numLayers: %d, headNum: %d, DomainPPSize:%d, DomainTPSize:%d, layerNumDomainPP:%d, "
        "headNumDomainTP:%d",
        numLayers, headNum, DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
    int const remainder = sizePerHead * sizeof(T) % 16;

    int const mlaSubWarpSize = 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            concatenateKVCacheForMLAKernel<T, mlaSubWarpSize, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, kvFactor);
        }
        else
        {
            concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
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
            concatenateKVCacheForMLAKernel<T, mlaSubWarpSize, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum, DomainPPSize, DomainTPSize,
                    layerNumDomainPP, kvFactor);
        }
        else
        {
            concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
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
                concatenateKVCacheForMLAKernel<T, mlaSubWarpSize, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
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
                concatenateKVCacheForMLAKernel<T, mlaSubWarpSize, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
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
                concatenateKVCacheForMLAKernel<T, mlaSubWarpSize, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, kvFactor);
            }
            else
            {
                concatenateKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNum,
                        DomainPPSize, DomainTPSize, layerNumDomainPP, headNumDomainTP);
            }

            break;
        }
        else
        {
            TLLM_THROW(" concatenateKVCache no support data type error");
        }
    }
    }
}

void concatenateKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocks,
    std::vector<runtime::ITensor::SharedPtr>& outputKvCacheBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager)
{

    auto dataType = outputKvCacheBlocks.at(0)->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        concatenateKVCache<double>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 4:
    {
        concatenateKVCache<float>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 2:
    {
        concatenateKVCache<half>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    case 1:
    {
        concatenateKVCache<uint8_t>(
            inputSplitBlocks, outputKvCacheBlocks, iCacheState, oCacheState, selfIdx, bufferManager);
        break;
    }
    default:
    {
        TLLM_THROW(" concatenateKVCache no support data type error");
    }
    }
}

} // namespace tensorrt_llm::executor::kv_cache
