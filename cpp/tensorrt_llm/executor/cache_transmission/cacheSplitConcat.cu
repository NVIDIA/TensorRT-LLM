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
#include "tensorrt_llm/common/envUtils.h"
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

int getBlockNumAccountingForCP(int cpRank, int cpSize, int numTotalBlocks)
{
    TLLM_CHECK(cpRank >= 0 && cpRank < cpSize);
    if (cpSize == 1)
    {
        return numTotalBlocks;
    }
    // Blocks are distributed among CP ranks as evenly as possible. When the number of blocks is not
    // divisible by cpSize, the remainder shall be distributed evenly among lowest-indexed CP ranks
    // (let's call them overflow ranks).
    int numBlocksCurrRank = numTotalBlocks / cpSize;
    if (numTotalBlocks % cpSize > cpRank)
    {
        numBlocksCurrRank++;
    }
    return numBlocksCurrRank;
}

int getGlobalBlockIdAccountingForCP(int localBlockIdx, int cpSize, int cpRank, int numTotalBlocks)
{
    if (tensorrt_llm::common::getEnvUseRoundRobinBlockDistForCP())
    {
        return localBlockIdx * cpSize + cpRank;
    }
    else
    {
        int const minBlocksPerCPRank = numTotalBlocks / cpSize;
        int const minBlocksOnPrevCPRanks = minBlocksPerCPRank * cpRank;
        int const numOverflowRanks = numTotalBlocks % cpSize;
        // Each previous overflow rank has one more block than minBlocksPerCPRank.
        int const overflowBlocksOnPrevCPRanks = std::min(cpRank, numOverflowRanks);
        return minBlocksOnPrevCPRanks + overflowBlocksOnPrevCPRanks + localBlockIdx;
    }
}

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
    auto const peerCPNum = peerParConfig.mContextParallelism;
    auto const selfCPNum = selfParConfig.mContextParallelism;

    auto const selfTPRank = selfRank % selfTPNum;
    auto const selfPPRank = selfRank / (selfTPNum * selfCPNum);
    auto const selfCPRank = (selfRank % (selfTPNum * selfCPNum)) / selfTPNum;

    int peerPPRankStart = 0;
    int mDomainPPSize = 1;
    int peerPPRankEnd = 0;
    std::vector<SizeType32> peerNumLayerPerPP = peerParConfig.mAttentionLayerNumPerPP;
    std::vector<SizeType32> selfNumLayerPerPP = selfParConfig.mAttentionLayerNumPerPP;
    TLLM_CHECK(peerNumLayerPerPP.size() == peerPPNum);
    TLLM_CHECK(selfNumLayerPerPP.size() == selfPPNum);
    int selfStartLayerId = 0;
    // global start layer id for selfPPrank, which is the sum of the layer num of the previous PP ranks.
    // compute the target PP ranks and layer num need to be fetched from each target PP rank, according to [global start
    // layer id, global end layer id)

    for (int ppRank = 0; ppRank < selfPPRank; ppRank++)
    {
        selfStartLayerId += selfNumLayerPerPP[ppRank];
    }
    int selfEndLayerId = selfStartLayerId + selfNumLayerPerPP[selfPPRank];
    int prePeerPPLayerId = 0;
    std::vector<int> targetPeerPPRanks;
    std::vector<int> targetPeerPPLayerNum;
    for (int ppRank = 0; ppRank < peerPPNum; ppRank++)
    {
        int peerPPStartLayerId = prePeerPPLayerId;
        int peerPPEndLayerId = peerPPStartLayerId + peerNumLayerPerPP[ppRank];

        prePeerPPLayerId += peerNumLayerPerPP[ppRank];

        if (selfStartLayerId < peerPPEndLayerId && selfEndLayerId > peerPPStartLayerId)
        {
            targetPeerPPRanks.push_back(ppRank);
            int layerNumInDomainPP
                = std::min(peerPPEndLayerId, selfEndLayerId) - std::max(peerPPStartLayerId, selfStartLayerId);
            targetPeerPPLayerNum.push_back(layerNumInDomainPP);
        }
    }
    mDomainPPSize = static_cast<int>(targetPeerPPRanks.size());
    peerPPRankStart = targetPeerPPRanks.front();
    peerPPRankEnd = peerPPRankStart + mDomainPPSize;
    TLLM_CHECK(targetPeerPPLayerNum.size() == mDomainPPSize);

    int targetPeerPpLayerNumSum = std::accumulate(targetPeerPPLayerNum.begin(), targetPeerPPLayerNum.end(), 0);
    TLLM_CHECK(targetPeerPpLayerNumSum == selfNumLayerPerPP[selfPPRank]);

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "selfPPRank:%d,selfPPNum:%d,peerPPNum:%d,selfTPNum:%d,peerTPNum:%d,peerPPRankStart:%d,peerPPRankEnd:%d",
        selfPPRank, selfPPNum, peerPPNum, selfTPNum, peerTPNum, peerPPRankStart, peerPPRankEnd);
    int peerTPRankStart = 0;
    int mDomainTPSize = 1;
    int peerTPRankEnd = 0;

    int const peerDpRank = peerParConfig.mEnableAttentionDP ? peerParConfig.mDPrank : 0;
    int const selfTPSizePerDPGroup = selfParConfig.mEnableAttentionDP ? selfTPNum / selfParConfig.mDPsize : selfTPNum;
    int const peerTPSizePerDPGroup = peerParConfig.mEnableAttentionDP ? peerTPNum / peerParConfig.mDPsize : peerTPNum;

    int const selfNbHeadsPerLayer = selfCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const peerNbHeadsPerLayer = peerCacheState.getModelConfig().mNbKvHeadsPerLayer[0];
    int const selfTPrankInDPGroup = selfTPRank % selfTPSizePerDPGroup;
    for (auto val : {peerTPSizePerDPGroup, selfTPSizePerDPGroup})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
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

    int mDomainCPSize = 1;
    int peerCPRankStart = 0;
    int peerCPRankEnd = 0;
    for (auto val : {peerCPNum, selfCPNum})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
    if (selfCPNum <= peerCPNum)
    {
        mDomainCPSize = peerCPNum / selfCPNum;
        peerCPRankStart = selfCPRank * mDomainCPSize;
        peerCPRankEnd = (selfCPRank + 1) * mDomainCPSize;
    }
    else
    {
        peerCPRankStart = selfCPRank / (selfCPNum / peerCPNum);
        peerCPRankEnd = peerCPRankStart + mDomainCPSize;
    }

    std::vector<int> retRanks;
    for (int i = peerTPRankStart; i < peerTPRankEnd; i++)
    {
        for (int j = peerCPRankStart; j < peerCPRankEnd; j++)
        {
            for (int k = peerPPRankStart; k < peerPPRankEnd; k++)
            {
                int irank = (k * peerTPNum * peerCPNum) + (j * peerTPNum) + i;
                retRanks.push_back(irank);
            }
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

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "mDomainPPSize:%d, mDomainTPSize:%d, mDupHeadFactor:%d, mPeerDupHeadFactor:%d, selfPPRank:%d, selfPPNum:%d, "
        "peerPPNum:%d, selfTPNum:%d, peerTPNum:%d, selfTPSizePerDPGroup:%d, peerTPSizePerDPGroup:%d, "
        "selfNbHeadsPerLayer:%d, peerNbHeadsPerLayer:%d, selfTPrankInDPGroup:%d, peerDpRank:%d, selfRank:%d",
        mDomainPPSize, mDomainTPSize, mDupHeadFactor, mPeerDupHeadFactor, selfPPRank, selfPPNum, peerPPNum, selfTPNum,
        peerTPNum, selfTPSizePerDPGroup, peerTPSizePerDPGroup, selfNbHeadsPerLayer, peerNbHeadsPerLayer,
        selfTPrankInDPGroup, peerDpRank, selfRank);

    auto vector_to_string = [](std::vector<int> const& vec)
    {
        std::stringstream ss;
        for (auto val : vec)
        {
            ss << val << ",";
        }
        return ss.str();
    };
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "retRanks:%s , targetPeerPPLayerNum:%s",
        vector_to_string(retRanks).c_str(), vector_to_string(targetPeerPPLayerNum).c_str());
    return {mDomainPPSize, mDomainTPSize, mDomainCPSize, std::move(retRanks), mDupHeadFactor, mPeerDupHeadFactor,
        std::move(targetPeerPPLayerNum)};
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
        auto const& parallelConfig = cacheState.getParallelConfig();
        auto const& modelConfig = cacheState.getModelConfig();

        int const tpRank = rank % parallelConfig.mTensorParallelism;
        int const ppRank = rank / parallelConfig.mTensorParallelism;
        int const ppNum = parallelConfig.mPipelineParallelism;
        int const headsPerBlock = modelConfig.mNbKvHeadsPerLayer[0];
        int const layersPerBlock = modelConfig.mNbKvHeadsPerLayer.size() / ppNum;

        int const tokensPerBlock = modelConfig.mTokensPerBlock;
        int const dimsPerBlock = modelConfig.mSizePerHead;
        int const startHead = tpRank * headsPerBlock;
        int const startLayer = ppRank * layersPerBlock;

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

__device__ __forceinline__ void getBlockIdInDomainCPandRankInDomainCP(
    int blockId, int domainCPSize, uint64_t* prefixBlockNumDevPtr, int& blockIdInDomainCP, int& rankInDomainCP)
{
    __shared__ int sharedBlockIdInDomainCP;
    __shared__ int sharedRankInDomainCP;

#pragma unroll 1
    for (int cpRank = threadIdx.x; cpRank < domainCPSize; cpRank += blockDim.x)
    {
        if (blockId >= prefixBlockNumDevPtr[cpRank] && blockId < prefixBlockNumDevPtr[cpRank + 1])
        {
            sharedBlockIdInDomainCP = blockId - prefixBlockNumDevPtr[cpRank];
            sharedRankInDomainCP = cpRank;
            break;
        }
    }

    __syncthreads();
    blockIdInDomainCP = sharedBlockIdInDomainCP;
    rankInDomainCP = sharedRankInDomainCP;
}

// MLA Head 1: One thread block per [(2), tokens, dimsPerHead]

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void splitKVCacheForMLAKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize,
    int domainTPSize, int domainCPSize, int kvFactor, uint64_t* prefixLayerNumDevPtr, bool isCPRoundRobin,
    uint64_t* prefixBlockNumDevPtr)
{
    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;

    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;
#pragma unroll 1

    for (int64_t blockId = blockIdx.y; blockId < inputBlockNum; blockId += gridDim.y)
    {
        // Default to CP round-robin.
        int rankInDomainCP = blockId % domainCPSize;    // genCPRank to which this block belongs.
        int blockIdInDomainCP = blockId / domainCPSize; // localBlockId on genCPRank.
        if (domainCPSize > 1 && !isCPRoundRobin)
        {
            // NOTE: domainCPSize is the same as genCPSize as contextCP is always 1 currently. So,
            // - rankInDomainCP is the same as genCPRank to which this block belongs.
            // - blockIdInDomainCP is the localBlockId on this genCPRank.
            getBlockIdInDomainCPandRankInDomainCP(
                blockId, domainCPSize, prefixBlockNumDevPtr, blockIdInDomainCP, rankInDomainCP);
        }
#pragma unroll 1
        for (int layerId = blockIdx.x; layerId < numLayers; layerId += gridDim.x)
        {
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = 0; headId < headNum; headId++)
            {
                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                // Example to understand the indexing: Say PP=2 and CP=2 for generation while PP=1 and CP=1 for context:
                // genRank 0 is ppRank0, cpRank0
                // genRank 1 is ppRank0, cpRank1
                // genRank 2 is ppRank1, cpRank0
                // genRank 3 is ppRank1, cpRank1
                // However, the order of cache transmission (targetRankInfo.mIRanks) is {0, 2, 1, 3}. i.e. {{pp0cp0},
                // {pp1cp0}, {pp0cp1}, {pp1cp1}}. So, outputCaches of all ppRanks corresponding to a given cpRank are
                // grouped together. We do blockId % domainCPSize because blocks are distributed among cpRanks in a
                // round-robin fashion.
                int outputCacheIdx = rankInDomainCP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                int const headIdInDomainTP = headId;

                T* kOutputPtr = outputCachePtr
                    + blockIdInDomainCP
                        * (static_cast<int64_t>(layerNumInSpecPP * kvFactor * headNum * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;
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
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize,
    int domainTPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
{

    // layerNumDomainPP =  numLayers/domainPPSize

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

            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1

            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T const* inputBlockPtr = inputBlocks[blockId];
                T const* kInputPtr = inputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T const* vInputPtr = inputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int outputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T* outputCachePtr = outputCaches[outputCacheIdx];

                int headIdInDomainTP = headId % headNumDomainTP;
                T* kOutputPtr = outputCachePtr
                    + static_cast<int64_t>(blockId)
                        * (static_cast<int64_t>(layerNumInSpecPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;

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

__device__ __forceinline__ void getPosId(int const posLayerId, int const* blockNumPerWindow, int const* layersPerWindow,
    int windowNum, int domainPPSize, int& inputBlockId, int& inputLayerId, int& outputAllLayerOffset,
    int& outputPPOffset)
{

    int offset = 0;
    int inputTensorIdx = 0;
    int layerInInputTensor = 0;
    int tensorIdxOffset = 0;
    int prevOutputLayerOffset = 0;
    __shared__ int sharedInputBlockId;
    __shared__ int sharedInputLayerId;
    __shared__ int sharedOutputAllLayerOffset;
    __shared__ int sharedOutputPPOffset;

    if (threadIdx.x == 0)
    {
#pragma unroll 1
        for (int i = 0; i < windowNum; i++)
        {
            int blockNumInWindow = blockNumPerWindow[i];
            int layersInWindow = layersPerWindow[i];

            // tensorIdxOffset
            if (posLayerId < offset + (blockNumInWindow * layersInWindow))
            {
                int remaindLayerInWindow = posLayerId - offset;

                int layerInWindowQuotient = remaindLayerInWindow / layersInWindow;
                int layerInWindowRemainder = remaindLayerInWindow % layersInWindow;

                inputTensorIdx = layerInWindowQuotient + tensorIdxOffset;
                layerInInputTensor = layerInWindowRemainder;

                int layersPerDomainPP = layersInWindow / domainPPSize;
                sharedOutputPPOffset = layerInWindowRemainder / layersPerDomainPP;

                sharedInputBlockId = inputTensorIdx;
                sharedInputLayerId = layerInInputTensor;

                sharedOutputAllLayerOffset = prevOutputLayerOffset + layerInWindowQuotient * layersPerDomainPP
                    + layerInWindowRemainder % layersPerDomainPP;

                break;
            }

            offset += blockNumInWindow * layersInWindow;
            tensorIdxOffset += blockNumInWindow;
            prevOutputLayerOffset += blockNumInWindow * (layersInWindow / domainPPSize);
        }
    }
    __syncthreads();
    inputBlockId = sharedInputBlockId;
    inputLayerId = sharedInputLayerId;
    outputAllLayerOffset = sharedOutputAllLayerOffset;
    outputPPOffset = sharedOutputPPOffset;
}

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
__global__ void splitKVCacheForWindowKernel(T const** __restrict__ inputBlocks, T** __restrict__ outputCaches,
    int const* blockNumPerWindow, int const* layersPerWindow, int windowNum, int inputAllLayerNum, int tokensPerBlock,
    int headNum, int dimsPerHead, int inputBlockNum, int domainPPSize, int domainTPSize, int headNumDomainTP)
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

    // one thread block per layer

#pragma unroll 1
    for (int threadBlockIdx = blockIdx.x; threadBlockIdx < inputAllLayerNum; threadBlockIdx += gridDim.x)
    {
        int inputBlockId, inputLayerId, outputAllLayerOffset;
        int outputPPOffset;
        getPosId(threadBlockIdx, blockNumPerWindow, layersPerWindow, windowNum, domainPPSize, inputBlockId,
            inputLayerId, outputAllLayerOffset, outputPPOffset);

        T const* inputBlockPtr = inputBlocks[inputBlockId];
        T const* inputLayerPtr = inputBlockPtr + inputLayerId * 2 * headNum * tokensPerBlock * dimsPerHead;
        size_t outputLayerEleOffset = outputAllLayerOffset * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead;
#pragma unroll 1
        for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
        {
            T const* kInputPtr = inputLayerPtr + headId * tokensPerBlock * dimsPerHead;
            T const* vInputPtr = kInputPtr + headNum * tokensPerBlock * dimsPerHead;
            int outputCacheIdx = headId / headNumDomainTP * domainPPSize + outputPPOffset;
            int headIdInDomainTP = headId % headNumDomainTP;
            T* outputCachePtr = outputCaches[outputCacheIdx];
            T* kOutputPtr = outputCachePtr + outputLayerEleOffset + headIdInDomainTP * tokensPerBlock * dimsPerHead;
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

template <typename T, int subWarpSize, int vecSizeByte>
__global__ void concatKVCacheForMLAKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize,
    int domainTPSize, int kvFactor, uint64_t* prefixLayerNumDevPtr)
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
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);
#pragma unroll 1

            for (int headId = 0; headId < headNum; headId++)
            {
                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                int inputCacheIdx = rankInDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];
                int headIdInDomainTP = headId;

                T const* kInputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId)
                        * (static_cast<int64_t>(layerNumInSpecPP * kvFactor * headNum * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * kvFactor * headNum * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;
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
    int tokensPerBlock, int numLayers, int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize,
    int DomainTPSize, int headNumDomainTP, uint64_t* prefixLayerNumDevPtr)
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
            int layerIdInDomainPP{};
            int rankInDomainPP{};
            int layerNumInSpecPP{};
            getLayerIdInDomainPPandRankInDomainPP(
                layerId, domainPPSize, prefixLayerNumDevPtr, layerIdInDomainPP, rankInDomainPP, layerNumInSpecPP);

#pragma unroll 1
            for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
            {

                T* outputBlockPtr = outputBlocks[blockId];
                T* kOutputPtr = outputBlockPtr + layerId * 2 * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;
                T* vOutputPtr = outputBlockPtr + (layerId * 2 + 1) * headNum * tokensPerBlock * dimsPerHead
                    + headId * tokensPerBlock * dimsPerHead;

                int inputCacheIdx = headId / headNumDomainTP * domainPPSize + rankInDomainPP;
                T const* inputCachePtr = inputCaches[inputCacheIdx];

                int headIdInDomainTP = headId % headNumDomainTP;
                T const* kInputPtr = inputCachePtr
                    + static_cast<int64_t>(blockId)
                        * (static_cast<int64_t>(layerNumInSpecPP * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead))
                    + static_cast<int64_t>(layerIdInDomainPP) * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead
                    + static_cast<int64_t>(headIdInDomainTP) * tokensPerBlock * dimsPerHead;

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

template <typename T, int subWarpSize, int subWarpNumInGroup, int vecSizeByte>
// __maxnreg__(32)
__global__ void concatKVCacheForWindowKernel(T const** __restrict__ inputCaches, T** __restrict__ outputBlocks,
    int const* blockNumPerWindow, int const* layersPerWindow, int windowNum, int outputAllLayerNum, int tokensPerBlock,
    int headNum, int dimsPerHead, int outputBlockNum, int domainPPSize, int domainTPSize, int headNumDomainTP)
{

    int const subWarpId = threadIdx.x / subWarpSize;
    int const laneId = threadIdx.x % subWarpSize;
    int const subWarpNum = blockDim.x / subWarpSize;
    int const subWarpGroupId = subWarpId / subWarpNumInGroup;
    int const subWarpGroupNum = subWarpNum / subWarpNumInGroup;
    int const subWarpIdInGroup = subWarpId % subWarpNumInGroup;
    static_assert(vecSizeByte >= sizeof(T));
    int constexpr numElePerThread = vecSizeByte / sizeof(T);
    using VecType = typename common::BytesToType<vecSizeByte>::type;

#pragma unroll 1
    for (int threadBlockIdx = blockIdx.x; threadBlockIdx < outputAllLayerNum; threadBlockIdx += gridDim.x)
    {
        int outputBlockId, outputLayerId, inputAllLayerOffset;
        int inputPPOffset;
        getPosId(threadBlockIdx, blockNumPerWindow, layersPerWindow, windowNum, domainPPSize, outputBlockId,
            outputLayerId, inputAllLayerOffset, inputPPOffset);
        T* outputBlockPtr = outputBlocks[outputBlockId];
        T* outputLayerPtr = outputBlockPtr + outputLayerId * 2 * headNum * tokensPerBlock * dimsPerHead;
        size_t inputLayerEleOffset = inputAllLayerOffset * 2 * headNumDomainTP * tokensPerBlock * dimsPerHead;

#pragma unroll 1
        for (int headId = subWarpGroupId; headId < headNum; headId += subWarpGroupNum)
        {
            T* kOutputPtr = outputLayerPtr + headId * tokensPerBlock * dimsPerHead;
            T* vOutputPtr = kOutputPtr + headNum * tokensPerBlock * dimsPerHead;
            int inputCacheIdx = headId / headNumDomainTP * domainPPSize + inputPPOffset;
            int headIdInDomainTP = headId % headNumDomainTP;
            T const* inputCachePtr = inputCaches[inputCacheIdx];
            T const* kInputPtr = inputCachePtr + inputLayerEleOffset + headIdInDomainTP * tokensPerBlock * dimsPerHead;
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

template <typename T>
void splitKVCache(std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> const& kVCacheBlocksPerWindow,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& destCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager,
    bool isIndexerKCache = false)
{

    size_t inputBlockNumSum = 0;
    for (auto const& [window, blocks] : kVCacheBlocksPerWindow)
    {
        inputBlockNumSum += blocks.size();
    }
    auto targetRankInfo = targetIRanks(destCacheState, selfCacheState, selfIdx);
    TLLM_CHECK(targetRankInfo.mIRanks.size()
        == (static_cast<size_t>(
            targetRankInfo.mDomainPPSize * targetRankInfo.mDomainTPSize * targetRankInfo.mDomainCPSize)));
    auto outputCacheNum = targetRankInfo.mIRanks.size();
    if (selfCacheState.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        outputCacheNum = targetRankInfo.mDomainPPSize * targetRankInfo.mDomainCPSize;
    }
    else
    {
        outputCacheNum = outputCacheNum / targetRankInfo.mPeerDupHeadFactor;
    }
    TLLM_CHECK(outputCacheNum == outputSplitBlocks.size());
    TLLM_CHECK(inputBlockNumSum > 0);
    // we want to reduce the call of `cudaMemcpyAsync H2D` , cachePtrs is used to store the pointers of the cache blocks
    // and the values of the prefix layer num.
    std::vector<uint64_t> cachePtrs;
    std::vector<SizeType32> windowSizes;
    std::vector<SizeType32> blockNumInwindow;
    std::vector<SizeType32> layersInWindow;
    size_t cacheBlockSizeSum = 0;
    size_t inputBlockLayerNumSum = 0;
    nvinfer1::DataType cacheDataType = nvinfer1::DataType::kINT64;
    if (isIndexerKCache)
    {
        cacheDataType = nvinfer1::DataType::kUINT8;
    }
    else
    {
        cacheDataType = kVCacheBlocksPerWindow.begin()->second.front()->getDataType();
    }

    for (auto const& [window, blocks] : kVCacheBlocksPerWindow)
    {
        auto cacheBlockSize = blocks.front()->getSize();
        auto cacheDataType = blocks.front()->getDataType();
        windowSizes.push_back(window);
        blockNumInwindow.push_back(blocks.size());
        TLLM_LOG_DEBUG("window: %d, blockNum: %d  blockshape:[%d,%d]", window, blocks.size(),
            blocks.front()->getShape().d[0], blocks.front()->getShape().d[1]);
        auto layersNum = blocks.front()->getDimension<1>();
        layersInWindow.push_back(layersNum);
        for (auto&& kvCacheBlock : blocks)
        {
            TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
            TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
            cacheBlockSizeSum += kvCacheBlock->getSize();
            cachePtrs.push_back(reinterpret_cast<uint64_t>((kvCacheBlock->data())));
            inputBlockLayerNumSum += layersNum;
        }
    }

    for (auto&& outputSplitBlock : outputSplitBlocks)
    {
        TLLM_CHECK_WITH_INFO(outputSplitBlock->getDataType() == cacheDataType,
            "outputSplitBlock data type mismatch expected: %d, actual: %d", static_cast<int>(cacheDataType),
            static_cast<int>(outputSplitBlock->getDataType()));
        cachePtrs.push_back(reinterpret_cast<uint64_t>(outputSplitBlock->data()));
    }
    std::vector<uint64_t> prefixLayerNum(targetRankInfo.mDomainPPSize + 1, 0);
    prefixLayerNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainPPSize; i++)
    {
        prefixLayerNum[i + 1] = prefixLayerNum[i] + targetRankInfo.mPeerAttentionLayerNumInDomainPP[i];
    }
    cachePtrs.insert(cachePtrs.end(), prefixLayerNum.begin(), prefixLayerNum.end());
    bool const isWindow = windowSizes.size() > 1;

    std::vector<uint64_t> prefixBlockNum(targetRankInfo.mDomainCPSize + 1, 0);
    prefixBlockNum[0] = 0;
    for (int i = 0; i < targetRankInfo.mDomainCPSize; i++)
    {
        prefixBlockNum[i + 1]
            = prefixBlockNum[i] + getBlockNumAccountingForCP(i, targetRankInfo.mDomainCPSize, inputBlockNumSum);
    }
    cachePtrs.insert(cachePtrs.end(), prefixBlockNum.begin(), prefixBlockNum.end());

    runtime::BufferManager::IBufferPtr PtrsDeviceBuffer
        = bufferManager.gpu(cachePtrs.size(), nvinfer1::DataType::kINT64);
    TLLM_CHECK(PtrsDeviceBuffer->getSizeInBytes() == cachePtrs.size() * sizeof(T*));
    bufferManager.copy(cachePtrs.data(), *PtrsDeviceBuffer, runtime::MemoryType::kCPU);

    runtime::BufferManager::IBufferPtr windowInfoDeviceBuffer;
    std::vector<SizeType32> windowInfoHostBuffer;
    if (isWindow)
    {
        windowInfoHostBuffer.reserve(windowSizes.size() * 2);

        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), blockNumInwindow.begin(), blockNumInwindow.end());
        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), layersInWindow.begin(), layersInWindow.end());
        windowInfoDeviceBuffer = bufferManager.gpu(windowInfoHostBuffer.size(), nvinfer1::DataType::kINT32);
        bufferManager.copy(windowInfoHostBuffer.data(), *windowInfoDeviceBuffer, runtime::MemoryType::kCPU);

        for (auto layerNum : layersInWindow)
        {

            TLLM_CHECK_WITH_INFO(
                layerNum % targetRankInfo.mDomainPPSize == 0, "layerNum in Window must be divisible by domainPPSize");
        }
    }
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
    unsigned int gridDimy = inputBlockNumSum;

    int const* blockNumPerWindowDevPtr = nullptr;
    int const* layersPerWindowDevPtr = nullptr;
    int windowNum = windowSizes.size();

    if (isWindow)
    {
        gridDimx = inputBlockLayerNumSum;
        gridDimy = 1;
        blockNumPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data());
        layersPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data()) + windowSizes.size();

        TLLM_LOG_DEBUG("windowNum:%d, inputBlockLayerNumSum:%d, ", windowNum, inputBlockLayerNumSum);
    }

    dim3 gridDim{gridDimx, gridDimy};

    int sizePerHead = selfModelConfig.mSizePerHead;
    T const** inputBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data());
    T** outputCachePtrsDev = static_cast<T**>(PtrsDeviceBuffer->data()) + inputBlockNumSum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNumSum + outputSplitBlocks.size();
    uint64_t* prefixBlockNumDevPtr = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + inputBlockNumSum
        + outputSplitBlocks.size() + prefixLayerNum.size();

    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const selfPPRank = selfIdx / (selfParallelConfig.mTensorParallelism * selfParallelConfig.mContextParallelism);
    int const numLayers = selfParallelConfig.mAttentionLayerNumPerPP.at(selfPPRank);
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];

    int dimsPerHead = selfModelConfig.mSizePerHead;
    if (isIndexerKCache)
    {
        dimsPerHead = selfCacheState.getIndexerDimPerHead()
            + selfCacheState.getIndexerDimPerHead() / selfCacheState.getIndexerKCacheQuantBlockSize() * 4;
        sizePerHead = dimsPerHead;
    }
    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;
    int const domainCPSize = targetRankInfo.mDomainCPSize;
    int const headNumDomainTP = headNum / (domainTPSize / targetRankInfo.mPeerDupHeadFactor);
    int const kvFactor = selfAttentionConfig.mKvFactor;
    bool const isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    bool const isCPRoundRobin = tensorrt_llm::common::getEnvUseRoundRobinBlockDistForCP();
    constexpr int mlaSubWarpSize = 16;

    TLLM_LOG_DEBUG(
        "splitKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "headsPerDomainTP: %d",
        numLayers, headNum, domainPPSize, domainTPSize, headNumDomainTP);

    int const remainder = sizePerHead * sizeof(T) % 16;
    switch (remainder)
    {
    case 0:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    domainCPSize, kvFactor, prefixLayerNumDevPtr, isCPRoundRobin, prefixBlockNumDevPtr);
        }
        else if (isWindow)
        {
            splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            splitKVCacheForMLAKernel<T, mlaSubWarpSize, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    domainCPSize, kvFactor, prefixLayerNumDevPtr, isCPRoundRobin, prefixBlockNumDevPtr);
        }
        else if (isWindow)
        {
            splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
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
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr, isCPRoundRobin, prefixBlockNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr);
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
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr, isCPRoundRobin, prefixBlockNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr);
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
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        domainCPSize, kvFactor, prefixLayerNumDevPtr, isCPRoundRobin, prefixBlockNumDevPtr);
            }
            else if (isWindow)
            {
                splitKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, inputBlockLayerNumSum,
                        tokensPerBlock, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP);
            }
            else
            {
                splitKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputBlockPtrsDev, outputCachePtrsDev,
                        tokensPerBlock, numLayers, headNum, dimsPerHead, inputBlockNumSum, domainPPSize, domainTPSize,
                        headNumDomainTP, prefixLayerNumDevPtr);
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

void splitKVCacheDispatch(std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> const& kVCacheBlocksPerWindow,
    std::vector<runtime::ITensor::SharedPtr>& outputSplitBlocks, kv_cache::CacheState const& iCacheState,
    kv_cache::CacheState const& oCacheState, int selfIdx, runtime::BufferManager const& bufferManager,
    bool isIndexerKCache)
{
    auto dataType = nvinfer1::DataType::kINT64;
    if (isIndexerKCache)
    {
        dataType = nvinfer1::DataType::kUINT8;
    }
    else
    {
        dataType = kVCacheBlocksPerWindow.begin()->second.front()->getDataType();
    }
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        splitKVCache<int64_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 4:
    {
        splitKVCache<int32_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 2:
    {
        splitKVCache<int16_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 1:
    {
        splitKVCache<int8_t>(kVCacheBlocksPerWindow, outputSplitBlocks, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
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
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>>& outputKvCacheBlocksPerWindow,
    kv_cache::CacheState const& destCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{

    size_t outputBlockNumSum = 0;
    for (auto const& [window, blocks] : outputKvCacheBlocksPerWindow)
    {
        outputBlockNumSum += blocks.size();
    }

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
    TLLM_CHECK(outputBlockNumSum > 0);

    std::vector<uint64_t> cachePtrs;
    std::vector<SizeType32> windowSizes;
    std::vector<SizeType32> blockNumInwindow;
    std::vector<SizeType32> layersInWindow;
    size_t cacheBlockSizeSum = 0;
    size_t outputBlockLayerNumSum = 0;
    auto cacheDataType = inputSplitBlocks.front()->getDataType();

    for (auto const& [window, blocks] : outputKvCacheBlocksPerWindow)
    {
        auto cacheBlockSize = blocks.front()->getSize();
        auto cacheDataType = blocks.front()->getDataType();
        windowSizes.push_back(window);
        blockNumInwindow.push_back(blocks.size());
        auto layersNum = blocks.front()->getDimension<1>();
        layersInWindow.push_back(layersNum);
        for (auto&& kvCacheBlock : blocks)
        {
            TLLM_CHECK(kvCacheBlock->getDataType() == cacheDataType);
            TLLM_CHECK(kvCacheBlock->getSize() == cacheBlockSize);
            cachePtrs.push_back(reinterpret_cast<uint64_t>(kvCacheBlock->data()));
            cacheBlockSizeSum += kvCacheBlock->getSize();
        }
        outputBlockLayerNumSum += layersNum * blocks.size();
    }
    for (auto&& inputSplitBlock : inputSplitBlocks)
    {
        TLLM_CHECK(inputSplitBlock->getDataType() == cacheDataType);
        cachePtrs.push_back(reinterpret_cast<uint64_t>(inputSplitBlock->data()));
    }

    // the prefix layer num is used to store the layer num of the previous PP ranks.
    // which is helpful for the kernel to get layer num info.  refer to the function
    // `getLayerIdInDomainPPandRankInDomainPP`.

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
    bool const isWindow = windowSizes.size() > 1;
    runtime::BufferManager::IBufferPtr windowInfoDeviceBuffer;
    std::vector<SizeType32> windowInfoHostBuffer;
    if (isWindow)
    {
        windowInfoHostBuffer.reserve(windowSizes.size() * 2);

        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), blockNumInwindow.begin(), blockNumInwindow.end());
        windowInfoHostBuffer.insert(windowInfoHostBuffer.end(), layersInWindow.begin(), layersInWindow.end());
        windowInfoDeviceBuffer = bufferManager.gpu(windowInfoHostBuffer.size(), nvinfer1::DataType::kINT32);
        bufferManager.copy(windowInfoHostBuffer.data(), *windowInfoDeviceBuffer, runtime::MemoryType::kCPU);
    }
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
    int const* blockNumPerWindowDevPtr = nullptr;
    int const* layersPerWindowDevPtr = nullptr;
    int windowNum = windowSizes.size();
    unsigned int gridDimy = outputBlockNumSum;
    if (isWindow)
    {
        gridDimx = outputBlockLayerNumSum;
        gridDimy = 1;
        blockNumPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data());
        layersPerWindowDevPtr = static_cast<int const*>(windowInfoDeviceBuffer->data()) + windowSizes.size();
    }
    dim3 gridDim{gridDimx, gridDimy};
    int sizePerHead = selfModelConfig.mSizePerHead;
    int const endLayerId = selfModelConfig.mNbKvHeadsPerLayer.size() / oPPNum;
    T** ouptutBlockPtrsDev = static_cast<T**>(PtrsDeviceBuffer->data());
    T const** inputSplitBlockPtrsDev = static_cast<T const**>(PtrsDeviceBuffer->data()) + outputBlockNumSum;
    uint64_t* prefixLayerNumDevPtr
        = static_cast<uint64_t*>(PtrsDeviceBuffer->data()) + outputBlockNumSum + inputSplitBlocks.size();
    int const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    int const selfPPRank = selfIdx / (selfParallelConfig.mTensorParallelism * selfParallelConfig.mContextParallelism);
    int const numLayers = selfParallelConfig.mAttentionLayerNumPerPP.at(selfPPRank);
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "concatKVCache numLayers:%d", numLayers);
    int const headNum = selfModelConfig.mNbKvHeadsPerLayer[0];
    int dimsPerHead = selfModelConfig.mSizePerHead;
    if (isIndexerKCache)
    {
        dimsPerHead = selfCacheState.getIndexerDimPerHead()
            + selfCacheState.getIndexerDimPerHead() / selfCacheState.getIndexerKCacheQuantBlockSize() * 4;
        sizePerHead = dimsPerHead;
    }

    int const domainPPSize = targetRankInfo.mDomainPPSize;
    int const domainTPSize = targetRankInfo.mDomainTPSize;

    int const headNumDomainTP
        = headNum / (domainTPSize / targetRankInfo.mPeerDupHeadFactor); // TODO: duplicate head factor
    int const kvFactor = selfAttentionConfig.mKvFactor;

    bool isMLA = selfAttentionConfig.mAttentionType == CacheState::AttentionType::kMLA;
    TLLM_LOG_DEBUG(
        "concatKVCache - numLayers: %d, headNum: %d, domainPPSize: %d, domainTPSize: %d, "
        "headsPerDomainTP: %d",
        numLayers, headNum, domainPPSize, domainTPSize, headNumDomainTP);

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
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, outputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 16>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
        }
        break;
    }
    case 8:
    {
        if (isMLA)
        {
            concatKVCacheForMLAKernel<T, mlaSubWarpSize, 8><<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(
                inputSplitBlockPtrsDev, ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead,
                outputBlockNumSum, domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
        }
        else if (isWindow)
        {
            concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum, outputBlockLayerNumSum, tokensPerBlock,
                    headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize, headNumDomainTP);
        }
        else
        {
            concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 8>
                <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev, ouptutBlockPtrsDev,
                    tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum, domainPPSize, domainTPSize,
                    headNumDomainTP, prefixLayerNumDevPtr);
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
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 4>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
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
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 2>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
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
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, kvFactor, prefixLayerNumDevPtr);
            }
            else if (isWindow)
            {
                concatKVCacheForWindowKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, blockNumPerWindowDevPtr, layersPerWindowDevPtr, windowNum,
                        outputBlockLayerNumSum, tokensPerBlock, headNum, dimsPerHead, outputBlockNumSum, domainPPSize,
                        domainTPSize, headNumDomainTP);
            }
            else
            {
                concatKVCacheKernel<T, subWarpSize, subWarpNumInGroup, 1>
                    <<<gridDim, blockDimx, 0, bufferManager.getStream().get()>>>(inputSplitBlockPtrsDev,
                        ouptutBlockPtrsDev, tokensPerBlock, numLayers, headNum, dimsPerHead, outputBlockNumSum,
                        domainPPSize, domainTPSize, headNumDomainTP, prefixLayerNumDevPtr);
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
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>>& outputKvCacheBlocksPerWindow,
    kv_cache::CacheState const& iCacheState, kv_cache::CacheState const& oCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{

    auto dataType = outputKvCacheBlocksPerWindow.begin()->second.front()->getDataType();
    auto dataSize = tensorrt_llm::common::getDTypeSize(dataType);
    switch (dataSize)
    {
    case 8:
    {
        concatKVCache<int64_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 4:
    {
        concatKVCache<int32_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 2:
    {
        concatKVCache<int16_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    case 1:
    {
        concatKVCache<int8_t>(inputSplitBlocks, outputKvCacheBlocksPerWindow, iCacheState, oCacheState, selfIdx,
            bufferManager, isIndexerKCache);
        break;
    }
    default:
    {
        TLLM_THROW("concatKVCache encountered an unsupported data type error.");
    }
    }
}

} // namespace tensorrt_llm::executor::kv_cache
