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
#include <cstddef>
#include <cstdint>
#include <future>
#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

BlockRange getBlockRangeForSending(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest)
{
    size_t requestBlockNum = llmRequest.getRequestedBlockHashes().size();
    constexpr SizeType32 beam{0};
    auto blockRange = BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId, beam);
    auto poolNum = cacheManager->getBlockManager().getNumPools();
    if (poolNum > 1 || common::getEnvDisableSelectiveCacheTransfer())
    {
        // disable selective cache transfer for poolNum > 1
        return blockRange;
    }
    if (requestBlockNum < blockRange.size() && requestBlockNum > 0)
    {
        // handle block reuse, the prefix blocks are reused
        // TODO(zhengd): pass the hashes directly instead of from llmRequest; use hash instead of block num
        auto const& ids = blockRange.getBlockIds();
        blockRange.setBlockIds({ids.end() - requestBlockNum, ids.end()});
    }
    return blockRange;
}

BlockRange getBlockRangeForReceiving(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest)
{

    auto poolNum = cacheManager->getBlockManager().getNumPools();
    if (poolNum > 1 || common::getEnvDisableSelectiveCacheTransfer())
    {
        constexpr SizeType32 beam{0};
        return BlockRange::fromAllBlockIds(*cacheManager, llmRequest.mRequestId, beam);
    }
    return BlockRange::fromNewlyAllocatedBlockIds(*cacheManager, llmRequest.mRequestId);
}

bool CacheFormatter::needSendCache(
    CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    // int selfTpRank = selfIdx % selfConfig.getParallelConfig().mTensorParallelism;
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

    return selfTpRankInDpGroup % targetInfo.mDupHeadFactor == 0;
}

void checkAlternateWindow(BaseKVCacheManager* cacheManager, BaseCacheFormatter::CacheState const& selfConfig,
    BaseCacheFormatter::CacheState const& destConfig)
{
    auto numPools = cacheManager->getBlockManager().getNumPools();
    auto layerNum = cacheManager->getBlockManager().getNumLayers();

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

    std::vector<size_t> ret;
    for (int i = 0; i < targetInfo.mDomainTPSize; i++)
    {
        if (i % targetInfo.mPeerDupHeadFactor == 0)
        {
            for (int j = 0; j < targetInfo.mDomainPPSize; j++)
            {
                ret.push_back((i * targetInfo.mDomainPPSize) + j);
            }
        }
    }
    return ret;
}

void CacheFormatter::format(TransferSession& session)
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
    auto& bufferManager = session.getBufferManager();
    if (!needSendCache(selfConfig, destConfig, selfIdx))
    {
        return;
    }
    auto& blockManager = mCacheManager->getBlockManager();
    auto blockRange = getBlockRangeForSending(mCacheManager, llmRequest);

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
            blockRange.updatePoolIdx(poolIdx);
            for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
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
    else
    {
        int blockNum = 0;

        size_t allCacheBlockSize = 0;

        std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> inputKvCacheBlocks;
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            blockRange.updatePoolIdx(poolIdx);
            SizeType32 window = mCacheManager->getBlockManager().getPoolWindowSize(poolIdx);
            TLLM_CHECK_WITH_INFO(inputKvCacheBlocks.find(window) == inputKvCacheBlocks.end(),
                "window size already exists, which is not supported");
            inputKvCacheBlocks.emplace(window, std::vector<runtime::ITensor::SharedPtr>());
            auto maxBlockThisWindow = window / selfConfig.getModelConfig().mTokensPerBlock;
            SizeType32 blockNumThisWindow = 0;
            for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
            {
                blockNum++;
                inputKvCacheBlocks.at(window).push_back(it);
                allCacheBlockSize += it->getSize();
                blockNumThisWindow++;
                if (blockNumThisWindow >= maxBlockThisWindow)
                {
                    break;
                }
            }
        }

        if (inputKvCacheBlocks.size() > 1)
        {
            if (selfConfig.getParallelConfig().mPipelineParallelism
                != destConfig.getParallelConfig().mPipelineParallelism)
            {
                checkAlternateWindow(mCacheManager, selfConfig, destConfig);
            }
        }
        TLLM_CHECK(!inputKvCacheBlocks.empty());
        TLLM_CHECK(blockNum > 0);
        int deviceId = mCacheManager->getBlockManager().getStreamDevice();
        auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && (destConfig.getParallelConfig().mPipelineParallelism
                <= selfConfig.getParallelConfig().mPipelineParallelism)
            && (destConfig.getParallelConfig().mTensorParallelism <= selfConfig.getParallelConfig().mTensorParallelism))
        {
            TLLM_LOG_DEBUG("Try using zero-copy for the KV cache.");
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CHECK(connections.size() == 1);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (size_t i = 0; i < connections.size(); i++)
            {
                for (auto const& [window, blocks] : inputKvCacheBlocks)
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

        auto cacheBufferId = mCacheTransBufferManager->assignBufferIndexForSend();
        int peerDuplicateHeadFactor = targetInfo.mPeerDupHeadFactor;
        auto targetNum = connections.size();
        auto const targetBufferSize = allCacheBlockSize / targetNum * peerDuplicateHeadFactor;
        auto bufferTargetNum = targetNum / peerDuplicateHeadFactor;
        TLLM_LOG_DEBUG(" formatOutput bufferTargetNum: %d, targetNum: %d, peerDuplicateHeadFactor: %d dupliacete:%d ",
            bufferTargetNum, targetNum, peerDuplicateHeadFactor, targetInfo.mDupHeadFactor);

        auto result = mCacheTransBufferManager->getOrAllocateSendBuffers(
            cacheBufferId, bufferTargetNum, targetBufferSize, bufferManager);
        auto& outputSplitCaches = std::get<0>(result);
        auto& bufferCoverTargetNum = std::get<1>(result);
        auto& onlyUseDynamicBuffer = std::get<2>(result);
        auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
        if (agentConnnecion != nullptr)
        {
            TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == bufferTargetNum, "Agent need all buffer pre-allocated");
            TLLM_CHECK(onlyUseDynamicBuffer == false);
        }

        tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(
            inputKvCacheBlocks, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager);

        bufferManager.getStream().synchronize();

        auto preAllocSendBuffer = mCacheTransBufferManager->getSendBuffer(cacheBufferId);
        if (preAllocSendBuffer != nullptr)
        {
            TLLM_CHECK(preAllocSendBuffer->getDataType() == inputKvCacheBlocks.begin()->second.front()->getDataType());
        }
        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            TLLM_CHECK(connections.size() > (processIdx / peerDuplicateHeadFactor));
            TLLM_CHECK(outputSplitCaches.size() > (processIdx / peerDuplicateHeadFactor));
            auto startTime = std::chrono::steady_clock::now();
            size_t size;

            size_t ppDomainSize = targetInfo.mDomainPPSize;
            size_t bufferTpRank = (processIdx / ppDomainSize) / peerDuplicateHeadFactor;
            size_t bufferIdx = (bufferTpRank * ppDomainSize) + (processIdx % ppDomainSize);
            if (bufferIdx < bufferCoverTargetNum)
            {

                size = outputSplitCaches[bufferIdx]->getSizeInBytes();
                session.send(processIdx, outputSplitCaches[bufferIdx]->data(), size);
            }
            else if (bufferCoverTargetNum > 0)
            {
                // copy buffer allocated by cudaMallocAsync to buffer allocated by cudaMalloc before sending
                auto sendBufferIdx = bufferIdx % bufferCoverTargetNum;
                bufferManager.copy(*outputSplitCaches[processIdx], *outputSplitCaches.at(sendBufferIdx));
                bufferManager.getStream().synchronize();
                size = outputSplitCaches.at(sendBufferIdx)->getSizeInBytes();
                session.send(processIdx, outputSplitCaches.at(sendBufferIdx)->data(), size);
            }
            else
            {
                // bufferCoverTargetNum == 0, mSendBuffer size < one outputSlice
                // send multiple times
                size = targetBufferSize;
                size_t remainSendSize = targetBufferSize;

                while (remainSendSize > 0)
                {
                    TLLM_CHECK(preAllocSendBuffer != nullptr);
                    auto sendBufferEleSize = preAllocSendBuffer->getSize();

                    auto sendSize = std::min(remainSendSize, sendBufferEleSize);
                    auto copySlice = runtime::ITensor::slice(
                        outputSplitCaches[bufferIdx], targetBufferSize - remainSendSize, sendSize);

                    auto copyTargetSlice = runtime::ITensor::slice(preAllocSendBuffer, 0, sendSize);
                    bufferManager.copy(*copySlice, *copyTargetSlice);
                    bufferManager.getStream().synchronize();
                    session.send(processIdx, copyTargetSlice->data(), copyTargetSlice->getSizeInBytes());
                    remainSendSize -= sendSize;
                }
            }

            auto endTime = std::chrono::steady_clock::now();
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
                // concurrency num
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

void CacheFormatter::unformat(TransferSession& session)
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
    auto blockRange = getBlockRangeForReceiving(mCacheManager, llmRequest);

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
    for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        blockRange.updatePoolIdx(poolIdx);
        SizeType32 window = mCacheManager->getBlockManager().getPoolWindowSize(poolIdx);
        TLLM_CHECK_WITH_INFO(outputBuffersPerWindow.find(window) == outputBuffersPerWindow.end(),
            "window size already exists, which is not supported");
        outputBuffersPerWindow.emplace(window, std::vector<runtime::ITensor::SharedPtr>());
        auto maxBlockThisWindow = window / selfConfig.getModelConfig().mTokensPerBlock;
        SizeType32 blockNumThisWindow = 0;
        for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
        {
            blockNum++;
            blockNumThisWindow++;
            outputBuffersPerWindow.at(window).push_back(it);
            cacheBlockSizeSum += it->getSize();
            if (blockNumThisWindow >= maxBlockThisWindow)
            {
                break;
            }
        }
    }
    TLLM_CHECK(!outputBuffersPerWindow.empty());
    if (outputBuffersPerWindow.size() > 1)
    {
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
                blockRange.updatePoolIdx(poolIdx);
                for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
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
            // legacyPath: context executor rank only send data to one gen executor rank. it sends multiple cache
            // blocks.
            auto legacyPath = common::getEnvTryZCopyForKVCacheTransfer()
                && (destConfig.getParallelConfig().mPipelineParallelism
                    >= selfConfig.getParallelConfig().mPipelineParallelism)
                && (destConfig.getParallelConfig().mTensorParallelism
                    >= selfConfig.getParallelConfig().mTensorParallelism);

            runtime::ITensor::SharedPtr recvBufferTemp;
            std::vector<runtime::ITensor::SharedPtr> recvSplitCaches;

            auto dataType = outputBuffersPerWindow.begin()->second.front()->getDataType();
            auto targetNum = pickUpConnections.size();
            TLLM_CHECK(cacheBlockSizeSum % targetNum == 0);
            auto targetBufferSize = cacheBlockSizeSum / targetNum;

            size_t remainNoCoverTargetNum = 0;
            size_t bufferCoverTargetNum = 0;
            std::optional<int> cacheBufferId = std::nullopt;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                TLLM_CHECK(blockNum > 0);
                if (legacyPath)
                {

                    TLLM_LOG_DEBUG("formatOutput using legacy path");
                    auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
                    auto cacheVolume = runtime::ITensor::volume(cacheShape);

                    size_t bufferNum = blockNum * pickUpConnections.size();
                    recvBufferTemp = bufferManager.gpu(
                        runtime::ITensor::makeShape({static_cast<int64_t>(cacheVolume * bufferNum)}), dataType);
                    recvSplitCaches.resize(bufferNum);
                    for (size_t i = 0; i < bufferNum; i++)
                    {
                        recvSplitCaches[i] = runtime::ITensor::slice(recvBufferTemp, i * cacheVolume, cacheVolume);
                    }
                }
                else
                {
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
                            cacheBufferId, targetNum, targetBufferSize, bufferManager);
                    bufferCoverTargetNum = bufferCoverTargetNumtmp;
                    remainNoCoverTargetNum = targetNum > bufferCoverTargetNum ? targetNum - bufferCoverTargetNum : 0;

                    if (agentConnnecion != nullptr)
                    {
                        TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == targetNum, "Agent need buffer pre-allocated");
                        TLLM_CHECK(onlyUseDynamicBuffer == false);
                    }
                    recvSplitCaches = std::move(recvSplitCachestmp);
                }

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
                auto startTime = std::chrono::steady_clock::now();
                size_t size = 0;
                if (legacyPath)
                {
                    size_t idx = processIdx * blockNum;

                    for (size_t i = 0; i < blockNum; i++)
                    {
                        size_t commIdx = idx / (blockNum);
                        size_t blockIdx = idx % (blockNum);
                        size_t recvBufferIdx = blockIdx * pickUpConnections.size() + commIdx;
                        llmRequest.updateKvCacheSize((*recvSplitCaches[recvBufferIdx]).getSizeInBytes());
                        auto& buffer = recvSplitCaches.at(recvBufferIdx);
                        size += buffer->getSizeInBytes();
                        session.recv(pickUpConnections[processIdx], buffer->data(), buffer->getSizeInBytes());
                        idx++;
                    }
                }
                else
                {
                    if (processIdx >= remainNoCoverTargetNum)
                    {
                        llmRequest.updateKvCacheSize((*recvSplitCaches.at(processIdx)).getSizeInBytes());
                        auto& buffer = recvSplitCaches[processIdx];
                        size = buffer->getSizeInBytes();
                        session.recv(pickUpConnections[processIdx], buffer->data(), buffer->getSizeInBytes());
                    }
                    else if (bufferCoverTargetNum > 0)
                    {
                        auto recvBufferIdx = processIdx % bufferCoverTargetNum
                            + remainNoCoverTargetNum; // caches.at(recvBufferIdx) is allocated by cudaMalloc
                        llmRequest.updateKvCacheSize((*recvSplitCaches.at(recvBufferIdx)).getSizeInBytes());
                        auto& buffer = recvSplitCaches.at(recvBufferIdx);
                        size = buffer->getSizeInBytes();
                        session.recv(pickUpConnections[processIdx], buffer->data(), buffer->getSizeInBytes());
                        bufferManager.copy(*recvSplitCaches.at(recvBufferIdx), *recvSplitCaches[processIdx]);
                        bufferManager.getStream().synchronize();
                    }
                    else
                    {
                        // bufferCoverTargetNum == 0
                        size_t remainRecvSize = targetBufferSize;
                        while (remainRecvSize > 0)
                        {
                            TLLM_CHECK(preAllocRecvBuffer != nullptr);
                            auto recvBufferEleSize = preAllocRecvBuffer->getSize();
                            auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                            auto recvSlice = runtime::ITensor::slice(preAllocRecvBuffer, 0, recvSize);
                            auto copySlice = runtime::ITensor::slice(
                                recvSplitCaches[processIdx], targetBufferSize - remainRecvSize, recvSize);
                            size += recvSlice->getSizeInBytes();
                            llmRequest.updateKvCacheSize((*recvSlice).getSizeInBytes());
                            session.recv(pickUpConnections[processIdx], recvSlice->data(), recvSlice->getSizeInBytes());
                            bufferManager.copy(*recvSlice, *copySlice);
                            bufferManager.getStream().synchronize();
                            remainRecvSize -= recvSize;
                        }
                    }
                }
                auto endTime = std::chrono::steady_clock::now();
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

                if (legacyPath)
                {
                    TLLM_CHECK(outputBuffersPerWindow.size() == 1);
                    executor::kv_cache::concatKVCacheDispatch(recvSplitCaches.data(), recvSplitCaches.size(),
                        getCounterparts(selfConfig, selfIdx, destConfig), destConfig,
                        outputBuffersPerWindow.begin()->second.data(), outputBuffersPerWindow.begin()->second.size(),
                        selfIdx, selfConfig, bufferManager);
                }
                else
                {
                    executor::kv_cache::concatKvCacheV2Dispatch(
                        recvSplitCaches, outputBuffersPerWindow, destConfig, selfConfig, selfIdx, bufferManager);
                }
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
    int selfNumLayers = selfConfig.getModelConfig().mNbKvHeadsPerLayer.size();
    int selfPPSize = selfConfig.getParallelConfig().mPipelineParallelism;
    int destPPSize = destConfig.getParallelConfig().mPipelineParallelism;
    int destNumLayers = destConfig.getModelConfig().mNbKvHeadsPerLayer.size();

    if (selfPPSize == destPPSize)
    {
        return true;
    }
    if (selfNumLayers % selfPPSize != 0)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: layers %d must be divisible by pipeline parallelism :%d",
            selfNumLayers, selfPPSize);
        return false;
    }
    if (destNumLayers % destPPSize != 0)
    {
        TLLM_LOG_WARNING("CacheFormatter::inquireSupport: layers %d must be divisible by pipeline parallelism :%d ",
            destNumLayers, destPPSize);
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
