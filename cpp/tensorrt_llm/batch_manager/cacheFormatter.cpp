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

#include "cacheFormatter.h"
#include "mlaCacheFormatter.h"

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

BlockRange getBlockRangeForSending(
    BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest, BlockKey const& lastBlockKey, int32_t indexFromEnd)
{
    auto poolNum = cacheManager->getBlockManager().getNumPools();
    if (poolNum > 1 || !cacheManager->isEnableBlockReuse() || lastBlockKey.uniqueTokens.size() == 0)
    {
        // disable reuse path, and vwsa don't support reuse.
        bool needSendAllForWindow = common::getEnvKVCacheTransferAllBlocksForWindow();

        auto blockRange = BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId);
        // auto inputLen = llmRequest.getPromptLen();

        auto const& windowsMetadata = cacheManager->getBlockManager().getWindowSizesMetadata();

        if ((windowsMetadata.size() == 1 || needSendAllForWindow))
        {
            return blockRange;
        }
        auto const& blockIdsPerWindow = blockRange.getBlockIdsPerWindow();

        for (auto const& [windowSize, metadata] : windowsMetadata)
        {
            auto windowStartBlockIdx = needSendAllForWindow
                ? 0
                : static_cast<SizeType32>(blockIdsPerWindow.at(windowSize).size())
                    - (windowSize / cacheManager->getBlockManager().getTokensPerBlock() + 1);
            // TODO: promptLen to get the startBlockIdx
            SizeType32 startBlockIdx = std::max(0, windowStartBlockIdx);
            TLLM_LOG_DEBUG("getBlockRangeForSending windowSize: %d, startBlockIdx: %d  windowStartBlockIdx: %d",
                windowSize, startBlockIdx, windowStartBlockIdx);
            blockRange.setBlockIdsForWindow(windowSize,
                std::vector<SizeType32>(
                    blockIdsPerWindow.at(windowSize).begin() + startBlockIdx, blockIdsPerWindow.at(windowSize).end()));
        }

        return blockRange;
    }

    TLLM_CHECK_WITH_INFO(lastBlockKey.uniqueTokens.size() > 0, "lastBlockKey must be non-empty when reuse is enabled");
    return BlockRange::fromReuseTree(*cacheManager, lastBlockKey, indexFromEnd);
}

BlockRange getBlockRangeForReceiving(
    BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest, bool srcEnableBlockReuse)
{
    auto poolNum = cacheManager->getBlockManager().getNumPools();
    if (poolNum == 1 && srcEnableBlockReuse)
    {
        // Build from all block ids, then slice off the reused blocks so we only transfer newly allocated ones.
        auto windowSize = cacheManager->getBlockManager().getWindowSizesMetadata().begin()->first;
        auto range = BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId);
        auto const& allBlockIds = range.getBlockIdsPerWindow().at(windowSize);
        auto const totalBlocks = static_cast<SizeType32>(allBlockIds.size());
        // Derive reused blocks count from number of unique prepopulated tokens
        auto const tokensPerBlock = cacheManager->getBlockManager().getTokensPerBlock();
        auto const prepopulatedTokens = llmRequest.getPrepopulatedPromptLen();
        auto const totalUniqueTokens = llmRequest.getPromptLen();
        auto const usedBlocks = std::min<SizeType32>(
            static_cast<SizeType32>((totalUniqueTokens + tokensPerBlock - 1) / tokensPerBlock), totalBlocks);
        auto const reusedBlocks
            = std::min<SizeType32>(static_cast<SizeType32>((prepopulatedTokens / tokensPerBlock)), usedBlocks);

        std::vector<SizeType32> newBlockIds;
        if (reusedBlocks < usedBlocks)
        {
            newBlockIds.assign(allBlockIds.begin() + reusedBlocks, allBlockIds.begin() + usedBlocks);
        }
        else
        {
            if (usedBlocks > 0 && usedBlocks <= totalBlocks)
            {
                newBlockIds.push_back(allBlockIds[usedBlocks - 1]);
            }
        }
        range.setBlockIdsForWindow(windowSize, std::move(newBlockIds));
        return range;
    }

    auto const& windowsMetadata = cacheManager->getBlockManager().getWindowSizesMetadata();
    if (windowsMetadata.size() == 1 || common::getEnvKVCacheTransferAllBlocksForWindow())
    {

        return BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId);
    }
    auto blockRange = BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId);

    for (auto const& [windowSize, metadata] : windowsMetadata)
    {
        auto const& blockIdsPerWindow = blockRange.getBlockIdsPerWindow();
        auto windowStartBlockIdx = static_cast<SizeType32>(blockIdsPerWindow.at(windowSize).size())
            - (windowSize / cacheManager->getBlockManager().getTokensPerBlock() + 1);
        SizeType32 startBlockIdx = std::max(0, windowStartBlockIdx);
        blockRange.setBlockIdsForWindow(windowSize,
            std::vector<SizeType32>(
                blockIdsPerWindow.at(windowSize).begin() + startBlockIdx, blockIdsPerWindow.at(windowSize).end()));
    }
    return blockRange;
}

bool CacheFormatter::needSendCache(
    CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (targetInfo.mDupHeadFactor <= 1)
    {
        return true;
    }

    int selfTpRank = selfIdx % selfConfig.getParallelConfig().mTensorParallelism;
    int selfTpRankInDpGroup = selfTpRank;
    if (selfConfig.getParallelConfig().mEnableAttentionDP)
    {
        int selfTPNumInDPGroup
            = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
        selfTpRankInDpGroup = selfTpRank % selfTPNumInDPGroup;
    }
    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;

    return (destDPRank % targetInfo.mDupHeadFactor) == (selfTpRankInDpGroup % targetInfo.mDupHeadFactor);
}

void checkAlternateWindow(BaseKVCacheManager* cacheManager, BaseCacheFormatter::CacheState const& selfConfig,
    BaseCacheFormatter::CacheState const& destConfig)
{
    // TODO: VSWA do not support uneven layer per PP.
    // if gen PP and context PP are different, cache formatter only support alternative window like gpt-oss.
    // which is one layer is WSA, and another layer is Full attention.

    auto numPools = cacheManager->getBlockManager().getNumPools();
    auto layerNum = cacheManager->getBlockManager().getNumLayers();

    auto selfPPNum = selfConfig.getParallelConfig().mPipelineParallelism;
    auto selfAllLayerNum = selfConfig.getModelConfig().mNbKvHeadsPerLayer.size();
    auto destPPNum = destConfig.getParallelConfig().mPipelineParallelism;
    auto destAllLayerNum = destConfig.getModelConfig().mNbKvHeadsPerLayer.size();
    TLLM_CHECK_WITH_INFO(selfAllLayerNum % selfPPNum == 0, " For VWSA selfAllLayerNum must be divisible by selfPPNum");
    TLLM_CHECK_WITH_INFO(destAllLayerNum % destPPNum == 0, "For VWSA destAllLayerNum must be divisible by destPPNum");

    std::vector<SizeType32> poolIdxs(numPools);
    TLLM_CHECK(layerNum >= numPools);
    for (int i = 0; i < numPools; i++)
    {
        poolIdxs[i] = cacheManager->getBlockManager().getLayerPoolIdx(i);
        TLLM_LOG_DEBUG("poolIdxs[%d] = %d layerNum:%d", i, poolIdxs[i], layerNum);
    }

    std::unordered_set<SizeType32> uniquePoolIdxs(poolIdxs.begin(), poolIdxs.end());
    TLLM_CHECK_WITH_INFO(uniquePoolIdxs.size() == poolIdxs.size(), "poolIdxs must contain unique elements");
    for (int i = numPools; i < layerNum; i++)
    {
        TLLM_CHECK_WITH_INFO(poolIdxs[i % numPools] == cacheManager->getBlockManager().getLayerPoolIdx(i),
            "only support Alternate Window");
    }
}

std::vector<size_t> CacheFormatter::pickRecvConnections(
    size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (targetInfo.mPeerDupHeadFactor <= 1)
    {
        std::vector<size_t> ret(numConnections);
        std::iota(ret.begin(), ret.end(), 0);
        return ret;
    }
    TLLM_CHECK(numConnections == targetInfo.mIRanks.size());
    int selfDPRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;

    std::vector<size_t> ret;
    for (int i = 0; i < targetInfo.mDomainTPSize; i++)
    {
        if ((i % targetInfo.mPeerDupHeadFactor) == (selfDPRank % targetInfo.mPeerDupHeadFactor))
        {
            for (int j = 0; j < targetInfo.mDomainPPSize; j++)
            {
                ret.push_back((i * targetInfo.mDomainPPSize) + j);
            }
        }
    }
    return ret;
}

void CacheFormatter::format(tensorrt_llm::batch_manager::TransferSession& session)
{
    NVTX3_SCOPED_RANGE(CacheFormatter_format);
    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending KV cache for request ID: %ld.", llmRequest.mRequestId);

    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");
    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto indexFromEnd = session.getIndexFromEnd();
    auto& bufferManager = session.getBufferManager();
    // Some TP rank don't need to send cache since duplicate header is not needed.
    if (!needSendCache(selfConfig, destConfig, selfIdx))
    {
        return;
    }
    auto& blockManager = mCacheManager->getBlockManager();
    auto const& lastBlockKey = session.getLastBlockKey();
    auto blockRange = getBlockRangeForSending(mCacheManager, llmRequest, lastBlockKey, indexFromEnd);
    auto const numPools = blockManager.getNumPools();
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...

    auto lastTokenTime = llmRequest.getPerfMetrics().timingMetrics.lastTokenTime;
    bool recordDelay = lastTokenTime != std::chrono::steady_clock::time_point();

    bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
    if (layerWise)
    {
        auto& progress = llmRequest.getContextProgress();
        SizeType32 const numLayers = blockManager.getNumLayers();
        runtime::ITensor::Shape offset = runtime::ITensor::makeShape({0, 0});
        for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
        {
            auto const poolIdx = blockManager.getLayerPoolIdx(layerIdx);
            auto const layerIdxInPool = blockManager.getPoolLayerIdx(layerIdx);
            offset.d[1] = layerIdxInPool;
            if (progress != nullptr)
            {
                progress->wait(layerIdx);
            }
            auto const& windowSizes = blockRange.getWindowSizes();
            for (auto const& windowSize : windowSizes)
            {
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    // Block dim: [1, numLayersInPool, ...], offset = {0, layerIndexInPool}
                    auto layer = runtime::ITensor::slice(it, offset, 1);
                    if (offset.d[1] == 0)
                    {
                        TLLM_LOG_DEBUG("Block %p of pool %d shape = %s", it->data(), poolIdx,
                            runtime::ITensor::toString(it->getShape()).c_str());
                    }
                    for (size_t i = 0; i < connections.size(); i++)
                    {
                        TLLM_LOG_DEBUG("Send layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                        session.send(i, layer->data(), layer->getSizeInBytes());
                    }
                }
            }
        }
    }
    else
    {
        int blockNum = 0;

        size_t allCacheBlockSize = 0;
        auto const& windowSizes = blockRange.getWindowSizes();
        TLLM_LOG_DEBUG(
            mpi::MpiComm::world().getRank(), " blockRange.getWindowSizes(); windowSizes size: %d", windowSizes.size());
        TLLM_CHECK_WITH_INFO(
            static_cast<int>(windowSizes.size()) == numPools, "window sizes should be the same as numPools");

        std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> inputKvCacheBlocksPerWindow;

        for (auto const& windowSize : windowSizes)
        {
            auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " format  windowSize: %d blockRangeForWindow size: %d",
                windowSize, blockRangeForWindow.size());
            inputKvCacheBlocksPerWindow.emplace(windowSize, std::vector<runtime::ITensor::SharedPtr>());
            for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
            {
                inputKvCacheBlocksPerWindow.at(windowSize).push_back(it);
                allCacheBlockSize += it->getSize();
                blockNum++;
            }
        }
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "inputKvCacheBlocks size: %ld,blockNum: %d , windowSizes: %ld",
            inputKvCacheBlocksPerWindow.size(), blockNum, windowSizes.size());

        if (inputKvCacheBlocksPerWindow.size() > 1)
        {
            if (selfConfig.getParallelConfig().mPipelineParallelism
                != destConfig.getParallelConfig().mPipelineParallelism)
            {
                checkAlternateWindow(mCacheManager, selfConfig, destConfig);
            }
        }
        TLLM_CHECK(!inputKvCacheBlocksPerWindow.empty());
        TLLM_CHECK(blockNum > 0);
        int deviceId = mCacheManager->getBlockManager().getStreamDevice();
        auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && (destConfig.getParallelConfig().mPipelineParallelism
                == selfConfig.getParallelConfig().mPipelineParallelism)
            && (destConfig.getParallelConfig().mTensorParallelism == selfConfig.getParallelConfig().mTensorParallelism))
        {
            TLLM_LOG_DEBUG("Try using zero-copy for the KV cache.");
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CHECK(connections.size() == 1);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (size_t i = 0; i < connections.size(); i++)
            {
                for (auto const& [window, blocks] : inputKvCacheBlocksPerWindow)
                {
                    for (auto const& block : blocks)
                    {
                        session.send(i, block->data(), block->getSizeInBytes());
                    }
                }
            }
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.",
                llmRequest.mRequestId);

            return;
        }

        // formatter flow
        // 1. collect cache blocks of the request.
        // 2. compute the buffer size for each target.
        // 3. prepare the pre-allocated buffer for each target according to the buffer size.
        // 4. call splitKVCacheDispatch to split the cache blocks according to the different parallelis and gather the
        // cache blocks to the corresponding buffer.
        // 5. send the buffer to the corresponding target. Ideally, we send only once (one buffer) for each target.

        auto cacheBufferId = mCacheTransBufferManager->assignBufferIndexForSend();
        int peerDuplicateHeadFactor = targetInfo.mPeerDupHeadFactor;
        auto targetNum = connections.size();
        auto bufferTargetNum = targetNum / peerDuplicateHeadFactor;
        auto ppRank = selfIdx
            / (selfConfig.getParallelConfig().mTensorParallelism * selfConfig.getParallelConfig().mContextParallelism);
        int selfAttentionLayerNum = selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);

        auto getBufferSizeForTarget = [&]()
        {
            std::vector<size_t> bufferSizeForTarget(targetNum, 0);
            // only first bufferTargetNum is used.
            if (inputKvCacheBlocksPerWindow.size() > 1)
            {
                // for VWSA
                for (size_t i = 0; i < targetNum; i++)
                {
                    bufferSizeForTarget[i] = allCacheBlockSize * peerDuplicateHeadFactor / targetNum;
                }
                return bufferSizeForTarget;
            }

            for (size_t i = 0; i < targetNum; i++)
            {
                bufferSizeForTarget[i] = allCacheBlockSize * peerDuplicateHeadFactor / targetInfo.mDomainTPSize
                    / selfAttentionLayerNum * targetInfo.getPeerPPDomainLayerNum(i);
            }

            return bufferSizeForTarget;
        };
        auto bufferEleSizes = getBufferSizeForTarget();
        auto result = mCacheTransBufferManager->getOrAllocateSendBuffers(
            cacheBufferId, static_cast<int>(bufferTargetNum), bufferEleSizes, bufferManager);

        auto& outputSplitCaches = std::get<0>(result);
        auto& bufferCoverTargetNum = std::get<1>(result);
        auto& onlyUseDynamicBuffer = std::get<2>(result);

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            " format bufferTargetNum: %d, targetNum: %d, peerDuplicateHeadFactor: %d duplicate:%d "
            "bufferCoverTargetNum:%d connections.size():%ld",
            bufferTargetNum, targetNum, peerDuplicateHeadFactor, targetInfo.mDupHeadFactor, bufferCoverTargetNum,
            connections.size());
        auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
        if (agentConnnecion != nullptr)
        {
            TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == bufferTargetNum, "Agent need all buffer pre-allocated");
            TLLM_CHECK(onlyUseDynamicBuffer == false);
        }
        // TODO: add parameters for layerNumForEachOutput
        tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(
            inputKvCacheBlocksPerWindow, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager);

        bufferManager.getStream().synchronize();

        auto preAllocSendBuffer = mCacheTransBufferManager->getSendBuffer(cacheBufferId);
        if (preAllocSendBuffer != nullptr)
        {
            TLLM_CHECK(preAllocSendBuffer->getDataType()
                == inputKvCacheBlocksPerWindow.begin()->second.front()->getDataType());
        }
        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " send processIdx: %ld", processIdx);
            NVTX3_SCOPED_RANGE(sendBufferFun);
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            TLLM_CHECK(connections.size() > (processIdx / peerDuplicateHeadFactor));
            TLLM_CHECK(outputSplitCaches.size() > (processIdx / peerDuplicateHeadFactor));
            auto startTime = llmRequest.getSteadyClockNow();

            size_t ppDomainSize = targetInfo.mDomainPPSize;
            size_t bufferTpRank = (processIdx / ppDomainSize) / peerDuplicateHeadFactor;
            size_t bufferIdx = (bufferTpRank * ppDomainSize) + (processIdx % ppDomainSize);
            size_t size = outputSplitCaches[bufferIdx]->getSizeInBytes();

            if (bufferIdx < bufferCoverTargetNum)
            {
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " send processIdx: %d bufferIdx: %d size:%ld",
                    processIdx, bufferIdx, outputSplitCaches[bufferIdx]->getSizeInBytes());
                session.send(
                    processIdx, outputSplitCaches[bufferIdx]->data(), outputSplitCaches[bufferIdx]->getSizeInBytes());
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " end send processIdx: %d bufferIdx: %d size:%ld",
                    processIdx, bufferIdx, outputSplitCaches[bufferIdx]->getSizeInBytes());
            }
            else
            {
                // If cacheIdx< bufferCoverTargetNum, the ouputSplitCaches.at(cacheIdx) is allocated by cudaMallocAsync,
                // which is unable to be transferred by UCX GPU-direct RDMA. We need copy the data to pre-allocated
                // cudaMalloc buffer,and then start send.
                // bufferCoverTargetNum == 0, mSendBuffer size < one outputSlice
                // send multiple times

                size_t remainSendSize = outputSplitCaches[processIdx]->getSize();
                size_t needSendSize = outputSplitCaches[processIdx]->getSize();
                auto sendBufferIdx = bufferCoverTargetNum == 0 ? 0 : bufferIdx % bufferCoverTargetNum;

                auto sendUseAllocBuffer
                    = bufferCoverTargetNum == 0 ? preAllocSendBuffer : outputSplitCaches[sendBufferIdx];

                while (remainSendSize > 0)
                {
                    TLLM_CHECK(sendUseAllocBuffer != nullptr);
                    auto sendBufferEleSize = sendUseAllocBuffer->getSize();

                    auto sendSize = std::min(remainSendSize, sendBufferEleSize);
                    auto copySlice = runtime::ITensor::slice(
                        outputSplitCaches[bufferIdx], needSendSize - remainSendSize, sendSize);
                    auto copyTargetSlice = runtime::ITensor::slice(sendUseAllocBuffer, 0, sendSize);
                    bufferManager.copy(*copySlice, *copyTargetSlice);
                    bufferManager.getStream().synchronize();
                    session.send(processIdx, copyTargetSlice->data(), copyTargetSlice->getSizeInBytes());
                    remainSendSize -= sendSize;
                }
            }

            auto endTime = llmRequest.getSteadyClockNow();
            double delay = 0.0;
            if (recordDelay)
            {
                delay = std::chrono::duration<double, std::milli>(startTime - lastTokenTime).count();
            }
            double cacheTransferTime
                = std::max(0.0, std::chrono::duration<double, std::milli>(endTime - startTime).count());
            session.appendMeasure(delay, cacheTransferTime, size);
        };

        if (connections.size() > 1)
        {
            if (!common::getEnvEnableReceiveKVCacheParallel())
            {
                TLLM_LOG_DEBUG("Disable parallel receiving of the KV cache.");

                for (size_t i = 0; i < connections.size(); i++)
                {
                    sendBufferFun(deviceId, i);
                }
            }
            else
            {
                // concurrency num should <=bufferCoverTargetNum to avoid data-race.
                auto concurrencyNum
                    = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), connections.size());

                auto remainSendNum = connections.size();

                while (remainSendNum > 0)
                {
                    auto sendConcurrencyNum = std::min(remainSendNum, concurrencyNum);
                    std::vector<std::future<void>> futures;
                    futures.reserve(sendConcurrencyNum);
                    for (size_t i = 0; i < sendConcurrencyNum; i++)
                    {
                        TLLM_CHECK((i + (connections.size() - remainSendNum)) < connections.size());
                        futures.push_back(std::async(
                            std::launch::async, sendBufferFun, deviceId, i + (connections.size() - remainSendNum)));
                    }
                    for (auto& future : futures)
                    {
                        future.get();
                    }
                    remainSendNum -= sendConcurrencyNum;
                }
            }
        }
        else
        {
            sendBufferFun(deviceId, 0);
        }

        mCacheTransBufferManager->freeBufferIndexForSend(cacheBufferId);
    }
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID:%ld ", llmRequest.mRequestId);
}

void CacheFormatter::unformat(tensorrt_llm::batch_manager::TransferSession& session)
{
    NVTX3_SCOPED_RANGE(CacheFormatter_unformat);
    auto const& llmRequest = session.getLlmRequest();
    auto const ctxReqId = llmRequest.getContextPhaseParams().value().getReqId();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "Start receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId, ctxReqId);
    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();
    auto blockRange = getBlockRangeForReceiving(mCacheManager, llmRequest, destConfig.getEnableBlockReuse());

    auto arrivalTime = llmRequest.getPerfMetrics().timingMetrics.arrivalTime;
    bool recordDelay = arrivalTime != std::chrono::steady_clock::time_point();

    auto pickUpConnections = pickRecvConnections(connections.size(), selfConfig, selfIdx, destConfig);

    TLLM_LOG_DEBUG("pickUpConnections size: %d connections size: %d", pickUpConnections.size(), connections.size());
    std::vector<runtime::ITensor::SharedPtr> recvBufferTmps;
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> outputBuffersPerWindow;
    auto const numPools = mCacheManager->getBlockManager().getNumPools();
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
    size_t blockNum = 0;
    size_t cacheBlockSizeSum = 0;

    auto windowSizes = blockRange.getWindowSizes();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " unformat windowSizes size: %d", windowSizes.size());
    for (auto const& windowSize : windowSizes)
    {
        auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "  unformat windowSize: %d blockRangeForWindow size: %d",
            windowSize, blockRangeForWindow.size());
        outputBuffersPerWindow.emplace(windowSize, std::vector<runtime::ITensor::SharedPtr>());

        for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
        {
            outputBuffersPerWindow.at(windowSize).push_back(it);
            cacheBlockSizeSum += it->getSize();
            blockNum++;
        }
    }
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "outputBuffersPerWindow size: %ld,blockNum: %d , windowSizes: %ld",
        outputBuffersPerWindow.size(), blockNum, windowSizes.size());
    TLLM_CHECK(!outputBuffersPerWindow.empty());
    if (outputBuffersPerWindow.size() > 1)
    {
        // We only support limited case for VSWA.
        if (selfConfig.getParallelConfig().mPipelineParallelism != destConfig.getParallelConfig().mPipelineParallelism)
        {
            checkAlternateWindow(mCacheManager, selfConfig, destConfig);
        }
    }
    {
        NVTX3_SCOPED_RANGE(formatInputRecvBuffer);

        auto dataType = mCacheManager->getPrimaryPool(0)->getDataType();
        bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
        if (layerWise)
        {
            // [numLayersInPool, ...]
            auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
            auto cacheVolume = runtime::ITensor::volume(cacheShape);
            size_t bufferNum = blockNum * pickUpConnections.size();
            runtime::ITensor::SharedPtr recvBufferTemp;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                recvBufferTemp = bufferManager.gpu(
                    runtime::ITensor::makeShape({static_cast<int64_t>(cacheVolume * bufferNum)}), dataType);
                recvBufferTmps.resize(bufferNum);
                for (size_t i = 0; i < bufferNum; i++)
                {
                    recvBufferTmps[i] = runtime::ITensor::slice(recvBufferTemp, i * cacheVolume, cacheVolume);
                }
                // sync to alloc buffer
                bufferManager.getStream().synchronize();
            }
            SizeType32 const numLocalLayers = mCacheManager->getBlockManager().getNumLayers();
            SizeType32 const numLayers = cacheShape.d[0];
            TLLM_CHECK(numLayers % numLocalLayers == 0 || numLocalLayers % numLayers == 0);
            auto layerVolume = cacheVolume / cacheShape.d[0];
            for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
            {
                // TODO: only send/recv required layers for ctxPP < genPP (numLayers > numLocalLayers)
                auto const poolIdx = 0;
                auto const layerIdxInPool = layerIdx;
                int idx = 0;
                // blockRange.updatePoolIdx(poolIdx);
                auto const window = mCacheManager->getBlockManager().getPoolLayerIdx(layerIdx);
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(window);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    if (layerIdxInPool == 0)
                    {
                        TLLM_LOG_DEBUG("Buffer %d of pool %d shape = %s", idx, poolIdx,
                            runtime::ITensor::toString(recvBufferTmps[idx]->getShape()).c_str());
                    }
                    for (size_t i = 0; i < pickUpConnections.size(); i++)
                    {
                        TLLM_LOG_DEBUG("Receive layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                        // Buffer dim: [numLayersInPool * layerVolume]
                        auto layer
                            = runtime::ITensor::slice(recvBufferTmps[idx], layerIdxInPool * layerVolume, layerVolume);
                        llmRequest.updateKvCacheSize((*layer).getSizeInBytes());
                        session.recv(pickUpConnections[i], layer->data(), layer->getSizeInBytes());
                        idx++;
                    }
                }
            }
            {
                NVTX3_SCOPED_RANGE(formatInputConcatenate);
                executor::kv_cache::concatKVCacheDispatch(recvBufferTmps.data(), recvBufferTmps.size(),
                    getCounterparts(selfConfig, selfIdx, destConfig), destConfig,
                    outputBuffersPerWindow.begin()->second.data(), outputBuffersPerWindow.begin()->second.size(),
                    selfIdx, selfConfig, bufferManager);
                bufferManager.getStream().synchronize();
            }
        }
        else
        {
            // non-layer-wise
            int deviceId = bufferManager.getStream().getDevice();

            if (common::getEnvTryZCopyForKVCacheTransfer() && destConfig == selfConfig)
            {
                TLLM_LOG_DEBUG("try zcopy for KV cache");
                NVTX3_SCOPED_RANGE(recvBufferFun);

                TLLM_CHECK(pickUpConnections.size() == 1);

                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                for (size_t i = 0; i < pickUpConnections.size(); i++)
                {
                    for (auto const& [window, blocks] : outputBuffersPerWindow)
                    {
                        for (auto const& block : blocks)
                        {
                            llmRequest.updateKvCacheSize((*block).getSizeInBytes());
                            session.recv(pickUpConnections[i], block->data(), block->getSizeInBytes());
                        }
                    }
                }
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
                    ctxReqId);
                return;
            }
            // unformatted flow
            // 1. collect cache blocks of the request.
            // 2. compute the buffer size for each target.
            // 3. prepare the pre-allocated buffer for each target according to the buffer size.
            // 4. receive the buffer from the corresponding target. Ideally, we receive only once (one buffer) for each
            // target.
            // 5. call concatKvCacheV2Dispatch to  concatenate the cache blocks according to the different parallelis

            runtime::ITensor::SharedPtr recvBufferTemp;
            std::vector<runtime::ITensor::SharedPtr> recvSplitCaches;

            auto dataType = outputBuffersPerWindow.begin()->second.front()->getDataType();
            auto targetNum = pickUpConnections.size();
            TLLM_CHECK(cacheBlockSizeSum % targetNum == 0);
            auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
            auto ppRank = selfIdx
                / (selfConfig.getParallelConfig().mTensorParallelism
                    * selfConfig.getParallelConfig().mContextParallelism);
            int selfAttentionLayerNum = selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);
            auto getTargetBufferEleSize = [&]()
            {
                if (outputBuffersPerWindow.size() > 1)
                {
                    std::vector<size_t> bufferSizeForTarget(targetNum, 0);
                    for (size_t i = 0; i < targetNum; i++)
                    {
                        bufferSizeForTarget[i] = cacheBlockSizeSum / targetNum;
                    }
                    return bufferSizeForTarget;
                }
                // for duplicate header, gen will not recv from TP which has duplicate header, and will not prepare
                // buffer for it.
                size_t validTpSize = pickUpConnections.size() / targetInfo.mDomainPPSize;
                TLLM_CHECK_WITH_INFO(cacheBlockSizeSum % validTpSize == 0,
                    "cacheBlockSizeSum must be divisible by validTpSize %ld", validTpSize);
                TLLM_CHECK_WITH_INFO((cacheBlockSizeSum % (selfAttentionLayerNum * validTpSize)) == 0,
                    "cacheBlockSizeSum must be divisible by validTpSize %ld * selfAttentionLayerNum %d", validTpSize,
                    selfAttentionLayerNum);
                TLLM_CHECK(targetNum == pickUpConnections.size());
                // the sum of buffer size is cacheBlockSizeSum.
                size_t cacheBlockSizePerLayer = cacheBlockSizeSum / (validTpSize * selfAttentionLayerNum);

                std::vector<size_t> bufferEleSizes(targetNum, 0);

                for (size_t i = 0; i < targetNum; i++)
                {
                    auto layerNum = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(pickUpConnections[i]));
                    bufferEleSizes[i] = cacheBlockSizePerLayer * layerNum;
                }
                return bufferEleSizes;
            };
            auto bufferEleSizes = getTargetBufferEleSize();

            size_t remainNoCoverTargetNum = 0;
            size_t bufferCoverTargetNum = 0;
            std::optional<int> cacheBufferId = std::nullopt;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                TLLM_CHECK(blockNum > 0);

                auto* agentConnnecion
                    = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[pickUpConnections[0]]);
                if (agentConnnecion != nullptr)
                {
                    cacheBufferId = agentConnnecion->getCacheBufferId();
                    TLLM_CHECK(cacheBufferId.has_value());
                }
                else
                {
                    cacheBufferId = mCacheTransBufferManager->assignBufferIndexForRecv();
                }
                TLLM_CHECK(cacheBufferId.has_value());
                auto [recvSplitCachestmp, bufferCoverTargetNumtmp, onlyUseDynamicBuffer]
                    = mCacheTransBufferManager->getOrAllocateRecvBuffers(
                        cacheBufferId, static_cast<int>(targetNum), bufferEleSizes, bufferManager);
                bufferCoverTargetNum = bufferCoverTargetNumtmp;
                remainNoCoverTargetNum = targetNum > bufferCoverTargetNum ? targetNum - bufferCoverTargetNum : 0;

                if (agentConnnecion != nullptr)
                {
                    TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == targetNum, "Agent need buffer pre-allocated");
                    TLLM_CHECK(onlyUseDynamicBuffer == false);
                }
                recvSplitCaches = std::move(recvSplitCachestmp);

                // sync to alloc buffer
                bufferManager.getStream().synchronize();
            }

            runtime::ITensor::SharedPtr preAllocRecvBuffer = nullptr;
            if (cacheBufferId.has_value())
            {
                preAllocRecvBuffer = mCacheTransBufferManager->getRecvBuffer(cacheBufferId);
                TLLM_CHECK(preAllocRecvBuffer != nullptr);
                TLLM_CHECK(preAllocRecvBuffer->getDataType() == dataType);
            }

            auto recvBufferFun = [&](int deviceId, size_t processIdx)
            {
                NVTX3_SCOPED_RANGE(recvBufferFun);
                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                TLLM_CHECK(pickUpConnections.size() > processIdx);
                TLLM_CHECK(recvSplitCaches.size() > processIdx);
                auto startTime = llmRequest.getSteadyClockNow();
                size_t size = 0;

                if (processIdx >= remainNoCoverTargetNum)
                {
                    llmRequest.updateKvCacheSize((*recvSplitCaches.at(processIdx)).getSizeInBytes());
                    auto& buffer = recvSplitCaches[processIdx];
                    size = buffer->getSizeInBytes();
                    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " start recv bufferIdx: %d size:%ld", processIdx,
                        buffer->getSizeInBytes());
                    session.recv(pickUpConnections[processIdx], buffer->data(), buffer->getSizeInBytes());
                    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), " recv bufferIdx: %d size:%ld", processIdx,
                        buffer->getSizeInBytes());
                }
                else
                {
                    auto recvBufferIdx
                        = bufferCoverTargetNum == 0 ? 0 : processIdx % bufferCoverTargetNum + remainNoCoverTargetNum;
                    // bufferCoverTargetNum == 0
                    auto recvBufferUsed
                        = bufferCoverTargetNum == 0 ? preAllocRecvBuffer : recvSplitCaches[recvBufferIdx];

                    size_t remainRecvSize = recvSplitCaches[processIdx]->getSize();
                    size_t needRecvSize = recvSplitCaches[processIdx]->getSize();
                    while (remainRecvSize > 0)
                    {
                        TLLM_CHECK(recvBufferUsed != nullptr);
                        auto recvBufferEleSize = recvBufferUsed->getSize();
                        auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                        auto recvSlice = runtime::ITensor::slice(recvBufferUsed, 0, recvSize);
                        auto copySlice = runtime::ITensor::slice(
                            recvSplitCaches[processIdx], needRecvSize - remainRecvSize, recvSize);
                        size += recvSlice->getSizeInBytes();
                        llmRequest.updateKvCacheSize((*recvSlice).getSizeInBytes());
                        session.recv(pickUpConnections[processIdx], recvSlice->data(), recvSlice->getSizeInBytes());
                        bufferManager.copy(*recvSlice, *copySlice);
                        bufferManager.getStream().synchronize();
                        remainRecvSize -= recvSize;
                    }
                }

                auto endTime = llmRequest.getSteadyClockNow();
                double delay = 0.0;
                if (recordDelay)
                {
                    delay = std::chrono::duration<double, std::milli>(startTime - arrivalTime).count();
                }
                double cacheTransferTime
                    = std::max(0.0, std::chrono::duration<double, std::milli>(endTime - startTime).count());
                session.appendMeasure(delay, cacheTransferTime, size);
            };
            if (pickUpConnections.size() > 1)
            {
                if (!common::getEnvEnableReceiveKVCacheParallel())
                {

                    for (size_t i = 0; i < pickUpConnections.size(); i++)
                    {
                        recvBufferFun(deviceId, i);
                    }
                }
                else
                {
                    // concurrency num
                    auto concurrencyNum
                        = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), pickUpConnections.size());
                    auto remainRecvNum = pickUpConnections.size();

                    while (remainRecvNum > 0)
                    {

                        auto recvConcurrencyNum = std::min(remainRecvNum, concurrencyNum);

                        if (remainRecvNum > concurrencyNum && remainRecvNum < (2 * concurrencyNum))
                        {
                            recvConcurrencyNum = remainRecvNum - concurrencyNum;
                        }
                        std::vector<std::future<void>> futures;
                        futures.reserve(recvConcurrencyNum);
                        for (size_t i = 0; i < recvConcurrencyNum; i++)
                        {
                            TLLM_CHECK((i + (pickUpConnections.size() - remainRecvNum)) < pickUpConnections.size());
                            futures.push_back(std::async(std::launch::async, recvBufferFun, deviceId,
                                i + (pickUpConnections.size() - remainRecvNum)));
                        }
                        for (auto& future : futures)
                        {
                            future.get();
                        }
                        remainRecvNum -= recvConcurrencyNum;
                    }
                }
            }
            else
            {
                recvBufferFun(deviceId, 0);
            }

            {
                NVTX3_SCOPED_RANGE(formatInputConcatenate);

                executor::kv_cache::concatKvCacheV2Dispatch(
                    recvSplitCaches, outputBuffersPerWindow, destConfig, selfConfig, selfIdx, bufferManager);

                bufferManager.getStream().synchronize();
                if (cacheBufferId.has_value())
                {
                    mCacheTransBufferManager->freeBufferIndexForRecv(cacheBufferId);
                }
            }
        }
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
}

[[nodiscard]] bool CacheFormatter::inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const
{
    if (selfConfig.getDataType() != destConfig.getDataType())
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: selfConfig.getDataType() != destConfig.getDataType()");
        return false;
    }

    std::unordered_set<SizeType32> setVecSelf{
        selfConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), selfConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecSelf.size() != 1)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support equal number of heads per layer");
        return false;
    }
    if (selfConfig.getAttentionConfig().mAttentionType != destConfig.getAttentionConfig().mAttentionType)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support same attention type");
        return false;
    }
    if (selfConfig.getAttentionConfig().mKvFactor != destConfig.getAttentionConfig().mKvFactor)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support same kv factor");
        return false;
    }
    if (selfConfig.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support non-MLA");
        return false;
    }
    if (selfConfig.getParallelConfig().mContextParallelism != 1
        || destConfig.getParallelConfig().mContextParallelism != 1)
    {
        TLLM_LOG_WARNING(
            "CacheFormatter::inquireSupport: context parallelism is not currently supported (selfCP=%d, destCP=%d).",
            selfConfig.getParallelConfig().mContextParallelism, destConfig.getParallelConfig().mContextParallelism);
        return false;
    }

    std::unordered_set<int> setVecDest{
        destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecDest.size() != 1)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support same number of heads per layer");
        return false;
    }
    if (selfConfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
        || selfConfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support same tokens per block and size per head");
        return false;
    }
    if (selfConfig.getModelConfig().mNbKvHeadsPerLayer.size() != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: only support same number of layers");
        TLLM_LOG_WARNING("self: %zu dest %zu", selfConfig.getModelConfig().mNbKvHeadsPerLayer.size(),
            destConfig.getModelConfig().mNbKvHeadsPerLayer.size());
        return false;
    }
    return true;
}

std::unique_ptr<BaseCacheFormatter> createCacheFormatter(
    BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager, bool isMLA)
{
    if (isMLA)
    {
        return std::make_unique<MLACacheFormatter>(cacheManager, cacheTransBufferManager);
    }
    return std::make_unique<CacheFormatter>(cacheManager, cacheTransBufferManager);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
