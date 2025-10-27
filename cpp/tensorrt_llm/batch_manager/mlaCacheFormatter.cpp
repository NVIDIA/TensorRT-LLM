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

#include "mlaCacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstddef>
#include <cstdint>
#include <future>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

// some context rank in connection
std::vector<size_t> MLACacheFormatter::pickRecvConnections(
    size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
{

    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    // This function is called only by gen side and we only support CPSize=1 on context size.
    TLLM_CHECK(targetInfo.mDomainCPSize == 1);
    TLLM_CHECK(numConnections == targetInfo.mIRanks.size());
    std::vector<size_t> ret;
    // targetInfo , mRanks [tpranks, ppranks]
    int dpRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;

    for (int i = 0; i < targetInfo.mDomainPPSize; i++)
    {
        ret.push_back(i + (dpRank % (targetInfo.mDomainTPSize)) * targetInfo.mDomainPPSize);
    }
    return ret;
}

bool MLACacheFormatter::needSendCache(
    CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    int selfTpRank = selfIdx % selfConfig.getParallelConfig().mTensorParallelism;

    int destTPNumInDPGroup = destConfig.getParallelConfig().mEnableAttentionDP
        ? destConfig.getParallelConfig().mTensorParallelism / destConfig.getParallelConfig().mDPsize
        : destConfig.getParallelConfig().mTensorParallelism;
    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;

    if (selfConfig.getParallelConfig().mEnableAttentionDP)
    {
        int selfTPNumInDPGroup
            = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;

        int selfTPrankINDPGroup = selfTpRank % selfTPNumInDPGroup;
        if (selfTPNumInDPGroup <= destTPNumInDPGroup)
        {
            return true;
        }

        int dupHeadFactor = selfTPNumInDPGroup / destTPNumInDPGroup;
        return selfTPrankINDPGroup % dupHeadFactor == destDPRank % dupHeadFactor;
    }

    int destTPNum = destConfig.getParallelConfig().mEnableAttentionDP
        ? destConfig.getParallelConfig().mTensorParallelism / destConfig.getParallelConfig().mDPsize
        : destConfig.getParallelConfig().mTensorParallelism;
    int selfTPNum = selfConfig.getParallelConfig().mTensorParallelism;
    if (selfTPNum <= destTPNum)
    {
        return true;
    }
    int dupHeadFactor = selfTPNum / destTPNum;
    return selfTpRank % dupHeadFactor == destDPRank % dupHeadFactor;
}

void MLACacheFormatter::format(tensorrt_llm::batch_manager::TransferSession& session)
{
    NVTX3_SCOPED_RANGE(MLACacheFormatter_format);
    session.setTime(TransferSession::kTimeFormatter);
    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending KV cache for request ID: %ld.", llmRequest.mRequestId);
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto indexFromEnd = session.getIndexFromEnd();
    auto const& lastBlockKey = session.getLastBlockKey();
    auto const& connections = session.getConnections();
    auto& bufferManager = session.getBufferManager();
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    TLLM_CHECK(!connections.empty());
    if (!needSendCache(selfConfig, destConfig, selfIdx))
    {
        return;
    }
    bool hasIndexerKCache = mCacheManager->getIndexerKCachePool() != nullptr;
    std::vector<bool> transferringIndexerKCache;
    transferringIndexerKCache.push_back(false);
    if (hasIndexerKCache)
    {
        transferringIndexerKCache.push_back(true);
    }
    auto const numPools = mCacheManager->getBlockManager().getNumPools(
        /*includeBlockScalePools=*/false, /*includeIndexerKCachePools=*/false);
    bool const recvSideHasCP = destConfig.getParallelConfig().mContextParallelism > 1;
    auto blockRange = getBlockRangeForSending(mCacheManager, llmRequest, lastBlockKey, indexFromEnd, recvSideHasCP);
    auto const& windowSizes = blockRange.getWindowSizes();
    TLLM_CHECK_WITH_INFO(
        static_cast<int>(windowSizes.size()) == numPools, "window sizes should be the same as numPools");

    for (auto transferIndexerKCache : transferringIndexerKCache)
    {
        int blockNum = 0;
        std::vector<runtime::ITensor::SharedPtr> inputKvCacheBlocks;
        if (!transferIndexerKCache)
        {
            for (auto const& windowSize : windowSizes)
            {
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    inputKvCacheBlocks.push_back(it);
                    blockNum++;
                }
            }
        }
        else
        {
            auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSizes.at(0), true);
            for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
            {
                inputKvCacheBlocks.push_back(it);
                blockNum++;
            }
        }

        TLLM_CHECK(blockNum > 0);
        int deviceId = mCacheManager->getBlockManager().getStreamDevice();

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && destConfig.getParallelConfig().mPipelineParallelism
                == selfConfig.getParallelConfig().mPipelineParallelism)
        {

            TLLM_LOG_DEBUG("Try using zero-copy for the KV cache.");
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (size_t i = 0; i < connections.size(); i++)
            {
                for (auto const& block : inputKvCacheBlocks)
                {
                    session.send(i, block->data(), block->getSizeInBytes());
                }
            }

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.",
                llmRequest.mRequestId);

            return;
        }

        auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
        size_t pPDomainSize = targetInfo.mDomainPPSize;
        size_t cPDomainSize = targetInfo.mDomainCPSize;

        auto getBufferSizeForTarget = [&]()
        {
            auto const ppRank = selfIdx
                / (selfConfig.getParallelConfig().mTensorParallelism
                    * selfConfig.getParallelConfig().mContextParallelism);
            auto const selfAttentionLayerNum = selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);
            auto const cacheBlockSize = inputKvCacheBlocks.at(0)->getSize();
            auto const blockSizePerLayer = cacheBlockSize / selfAttentionLayerNum;
            std::vector<size_t> bufferSizeForTarget(pPDomainSize * cPDomainSize, 0);
            for (size_t ppDomainIdx = 0; ppDomainIdx < pPDomainSize; ppDomainIdx++)
            {
                auto const peerAttentionLayerNum = targetInfo.getPeerPPDomainLayerNum(ppDomainIdx);
                for (size_t cpDomainIdx = 0; cpDomainIdx < cPDomainSize; cpDomainIdx++)
                {
                    auto const idx = cpDomainIdx * pPDomainSize + ppDomainIdx;
                    // Note: contextCP is always 1. So, cpDomainSize == genCPSize and cpDomainIdx == genCPRank.
                    auto const peerBlockNum
                        = executor::kv_cache::getBlockNumAccountingForCP(cpDomainIdx, cPDomainSize, blockNum);
                    bufferSizeForTarget[idx] = blockSizePerLayer * peerAttentionLayerNum * peerBlockNum;
                }
            }
            return bufferSizeForTarget;
        };
        auto bufferEleSizes = getBufferSizeForTarget();
        auto cacheBufferId = mCacheTransBufferManagers[transferIndexerKCache]->assignBufferIndexForSend();
        auto result = mCacheTransBufferManagers[transferIndexerKCache]->getOrAllocateSendBuffers(
            cacheBufferId, static_cast<int>(pPDomainSize * cPDomainSize), bufferEleSizes, bufferManager);
        auto& outputSplitCaches = std::get<0>(result);
        auto& bufferCoverTargetNum = std::get<1>(result);
        auto& onlyUseDynamicBuffer = std::get<2>(result);
        auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
        if (agentConnnecion != nullptr)
        {
            TLLM_CHECK_WITH_INFO(
                bufferCoverTargetNum == pPDomainSize * cPDomainSize, "Agent need all buffer pre-allocated");
            TLLM_CHECK(onlyUseDynamicBuffer == false);
        }

        // The size of outputSplitCaches should be equal to pPDomainSize * cPDomainSize.
        SizeType32 window = mCacheManager->getBlockManager().getPoolWindowSize(0);
        std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> inputKvCacheBlocksPerWindow;
        inputKvCacheBlocksPerWindow.emplace(window, inputKvCacheBlocks);
        tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(inputKvCacheBlocksPerWindow, outputSplitCaches,
            destConfig, selfConfig, selfIdx, bufferManager, transferIndexerKCache);

        bufferManager.getStream().synchronize();
        session.setTime(TransferSession::kTimePreprocess);

        auto preAllocSendBuffer = mCacheTransBufferManagers[transferIndexerKCache]->getSendBuffer(cacheBufferId);
        if (preAllocSendBuffer != nullptr)
        {
            TLLM_CHECK(preAllocSendBuffer->getDataType() == inputKvCacheBlocks.at(0)->getDataType());
        }
        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            auto startTime = LlmRequest::getSteadyClockNow();
            auto cacheIdx = processIdx % (pPDomainSize * cPDomainSize);
            if (cacheIdx < bufferCoverTargetNum)
            {
                size_t size = outputSplitCaches.at(cacheIdx)->getSizeInBytes();
                session.send(processIdx, outputSplitCaches.at(cacheIdx)->data(), size);
            }
            else
            {

                // If cacheIdx< bufferCoverTargetNum, the ouputSplitCaches.at(cacheIdx) is allocated by cudaMallocAsync,
                // which is unable to be transferred by UCX GPU-direct RDMA. We need copy the data to pre-allocated
                // cudaMalloc buffer,and then start send.
                // bufferCoverTargetNum=0, mSendBuffer size < one outputSlice send multiple times
                size_t remainSendSize = outputSplitCaches.at(cacheIdx)->getSize();
                size_t needSendSize = outputSplitCaches.at(cacheIdx)->getSize();
                auto sendBufferIdx = bufferCoverTargetNum == 0 ? 0 : cacheIdx % bufferCoverTargetNum;
                auto sendBufferUsed
                    = bufferCoverTargetNum == 0 ? preAllocSendBuffer : outputSplitCaches.at(sendBufferIdx);
                while (remainSendSize > 0)
                {
                    TLLM_CHECK(sendBufferUsed != nullptr);
                    auto sendBufferEleSize = sendBufferUsed->getSize();
                    auto sendSize = std::min(remainSendSize, sendBufferEleSize);
                    auto copySlice = runtime::ITensor::slice(
                        outputSplitCaches.at(cacheIdx), needSendSize - remainSendSize, sendSize);
                    auto copyTargetSlice = runtime::ITensor::slice(sendBufferUsed, 0, sendSize);
                    bufferManager.copy(*copySlice, *copyTargetSlice);
                    bufferManager.getStream().synchronize();
                    session.send(processIdx, copyTargetSlice->data(), copyTargetSlice->getSizeInBytes());

                    remainSendSize -= sendSize;
                }
            }
            auto endTime = LlmRequest::getSteadyClockNow();
            session.appendMeasure(startTime, endTime, outputSplitCaches.at(cacheIdx)->getSizeInBytes());
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
                    = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), pPDomainSize * cPDomainSize);

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
        mCacheTransBufferManagers[transferIndexerKCache]->freeBufferIndexForSend(cacheBufferId);
    }
    session.setTime(TransferSession::kTimeTransmissions);
    session.setTime(TransferSession::kTimePostprocess);

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.", llmRequest.mRequestId);
}

void MLACacheFormatter::unformat(tensorrt_llm::batch_manager::TransferSession& session)
{
    NVTX3_SCOPED_RANGE(MLACacheFormatter_unformat);
    session.setTime(TransferSession::kTimeFormatter);
    auto const& llmRequest = session.getLlmRequest();
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    auto const ctxReqId = llmRequest.getContextPhaseParams().value().getReqId();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "Start receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId, ctxReqId);
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto const& connections = session.getConnections();
    auto& bufferManager = session.getBufferManager();
    auto pickUpConnections = pickRecvConnections(connections.size(), selfConfig, selfIdx, destConfig);
    bool const recvSideHasCP = selfConfig.getParallelConfig().mContextParallelism > 1;
    auto blockRange
        = getBlockRangeForReceiving(mCacheManager, llmRequest, destConfig.getEnableBlockReuse(), recvSideHasCP);
    auto const numPools = mCacheManager->getBlockManager().getNumPools(
        /*includeBlockScalePools=*/false, /*includeIndexerKCachePools=*/false);
    auto const& windowSizes = blockRange.getWindowSizes();
    TLLM_CHECK_WITH_INFO(
        static_cast<int>(windowSizes.size()) == numPools, "window sizes should be the same as numPools");
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
    bool hasIndexerKCache = mCacheManager->getIndexerKCachePool() != nullptr;
    std::vector<bool> transferringIndexerKCache;
    transferringIndexerKCache.push_back(false);
    if (hasIndexerKCache)
    {
        transferringIndexerKCache.push_back(true);
    }
    for (auto transferIndexerKCache : transferringIndexerKCache)
    {
        std::vector<runtime::ITensor::SharedPtr> recvBufferTmps;
        std::vector<runtime::ITensor::SharedPtr> outputBuffers;
        size_t blockNum = 0;
        if (!transferIndexerKCache)
        {
            for (auto const& windowSize : windowSizes)
            {
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    outputBuffers.push_back(it);
                    blockNum++;
                }
            }
        }
        else
        {
            auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSizes.at(0), true);
            for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
            {
                outputBuffers.push_back(it);
                blockNum++;
            }
        }

        int deviceId = bufferManager.getStream().getDevice();

        std::optional<int> cacheBufferId = std::nullopt;

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && destConfig.getParallelConfig().mPipelineParallelism
                == selfConfig.getParallelConfig().mPipelineParallelism)
        {
            // recv
            TLLM_LOG_DEBUG("Try zcopy for KV cache");
            NVTX3_SCOPED_RANGE(recvBufferFun);
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            TLLM_CHECK(pickUpConnections.size() == 1);
            for (size_t i = 0; i < pickUpConnections.size(); i++)
            {
                for (auto const& block : outputBuffers)
                {
                    llmRequest.updateKvCacheSize(block->getSizeInBytes());
                    session.recv(pickUpConnections[i], block->data(), block->getSizeInBytes());
                }
            }
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
                llmRequest.getContextPhaseParams().value().getReqId());
            return;
        }
        else
        {
            auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
            if (agentConnnecion != nullptr)
            {
                cacheBufferId = agentConnnecion->getCacheBufferId();
                TLLM_CHECK(cacheBufferId.has_value());
            }
            else
            {
                cacheBufferId = mCacheTransBufferManagers[transferIndexerKCache]->assignBufferIndexForRecv();
            }

            auto targetNum = pickUpConnections.size();

            auto getBufferSizeForTarget = [&]()
            {
                auto const targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
                auto const cacheBlockSize = outputBuffers.at(0)->getSize();
                auto const ppRank = selfIdx
                    / (selfConfig.getParallelConfig().mTensorParallelism
                        * selfConfig.getParallelConfig().mContextParallelism);
                auto const selfAttentionLayerNum = selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);
                TLLM_CHECK_WITH_INFO(selfAttentionLayerNum != 0, "selfAttentionLayerNum should not be 0");
                std::vector<size_t> bufferEleSizes(targetNum, 0);
                auto const cacheSizePerLayer = cacheBlockSize * blockNum / selfAttentionLayerNum;
                for (size_t i = 0; i < targetNum; i++)
                {
                    auto const peerAttentionLayerNum
                        = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(pickUpConnections[i]));
                    bufferEleSizes[i] = cacheSizePerLayer * peerAttentionLayerNum;
                }
                return bufferEleSizes;
            };
            auto bufferEleSizes = getBufferSizeForTarget();

            auto result = mCacheTransBufferManagers[transferIndexerKCache]->getOrAllocateRecvBuffers(
                cacheBufferId, static_cast<int>(targetNum), bufferEleSizes, bufferManager);
            auto& recvSplitCaches = std::get<0>(result);
            auto& bufferCoverTargetNum = std::get<1>(result);
            size_t remainNoCoverTargetNum = targetNum > bufferCoverTargetNum ? targetNum - bufferCoverTargetNum : 0;
            auto& onlyUseDynamicBuffer = std::get<2>(result);
            if (agentConnnecion != nullptr)
            {
                TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == targetNum, "Agent need buffer pre-allocated");
                TLLM_CHECK(onlyUseDynamicBuffer == false);
            }
            bufferManager.getStream().synchronize();
            session.setTime(TransferSession::kTimePreprocess);

            auto preAllocRecvBuffer = mCacheTransBufferManagers[transferIndexerKCache]->getRecvBuffer(cacheBufferId);
            if (preAllocRecvBuffer != nullptr)
            {
                TLLM_CHECK(preAllocRecvBuffer->getDataType() == outputBuffers.at(0)->getDataType());
            }

            auto recvBufferFun = [&](int deviceId, size_t processIdx)
            {
                NVTX3_SCOPED_RANGE(recvBufferFun);
                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                auto startTime = LlmRequest::getSteadyClockNow();
                size_t size = 0;
                if (processIdx >= remainNoCoverTargetNum)
                {
                    auto& buffer = recvSplitCaches.at(processIdx);
                    llmRequest.updateKvCacheSize(buffer->getSizeInBytes());
                    size = buffer->getSizeInBytes();
                    session.recv(pickUpConnections.at(processIdx), buffer->data(), buffer->getSizeInBytes());
                }
                else
                {
                    auto recvBufferIdx
                        = bufferCoverTargetNum == 0 ? 0 : processIdx % bufferCoverTargetNum + remainNoCoverTargetNum;
                    auto recvBufferUsed
                        = bufferCoverTargetNum == 0 ? preAllocRecvBuffer : recvSplitCaches[recvBufferIdx];
                    // bufferCoverTargetNum==0
                    size_t remainRecvSize = recvBufferUsed->getSize();
                    size_t needRecvSize = recvSplitCaches.at(processIdx)->getSize();
                    while (remainRecvSize > 0)
                    {
                        TLLM_CHECK(recvBufferUsed != nullptr);
                        auto recvBufferEleSize = recvBufferUsed->getSize();
                        auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                        auto recvSlice = runtime::ITensor::slice(recvBufferUsed, 0, recvSize);
                        auto copySlice = runtime::ITensor::slice(
                            recvSplitCaches.at(processIdx), needRecvSize - remainRecvSize, recvSize);
                        llmRequest.updateKvCacheSize(recvSlice->getSizeInBytes());
                        size += recvSlice->getSizeInBytes();
                        session.recv(pickUpConnections.at(processIdx), recvSlice->data(), recvSlice->getSizeInBytes());
                        bufferManager.copy(*recvSlice, *copySlice);
                        bufferManager.getStream().synchronize();
                        remainRecvSize -= recvSize;
                    }
                }
                auto endTime = LlmRequest::getSteadyClockNow();
                session.appendMeasure(startTime, endTime, size);
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
            session.setTime(TransferSession::kTimeTransmissions);

            {
                std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> outputCachesPerWindow;
                SizeType32 window = mCacheManager->getBlockManager().getPoolWindowSize(0);
                outputCachesPerWindow.emplace(window, outputBuffers);
                NVTX3_SCOPED_RANGE(formatInputConcatenate);

                // recvSplitCaches size == ppdomainsize * cpdomainsize.
                executor::kv_cache::concatKvCacheV2Dispatch(recvSplitCaches, outputCachesPerWindow, destConfig,
                    selfConfig, selfIdx, bufferManager, transferIndexerKCache);
            }
            bufferManager.getStream().synchronize();
        }

        if (cacheBufferId.has_value())
        {
            mCacheTransBufferManagers[transferIndexerKCache]->freeBufferIndexForRecv(cacheBufferId);
        }
    }
    session.setTime(TransferSession::kTimePostprocess);

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
}

[[nodiscard]] bool MLACacheFormatter::inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const
{
    if (selfConfig.getDataType() != destConfig.getDataType())
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support same data type");
        return false;
    }
    if (selfConfig.getAttentionConfig().mAttentionType != CacheState::AttentionType::kMLA
        || destConfig.getAttentionConfig().mAttentionType != CacheState::AttentionType::kMLA)
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support MLA");
        return false;
    }
    if (selfConfig.getAttentionConfig().mKvFactor != destConfig.getAttentionConfig().mKvFactor)
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support same kv factor");
        return false;
    }

    std::unordered_set<SizeType32> setVecSelf{
        selfConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), selfConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecSelf.size() != 1)
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support equal number of heads per layer");
        return false;
    }
    std::unordered_set<int> setVecDest{
        destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecDest.size() != 1)
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support equal number of heads per layer");
        return false;
    }
    if (selfConfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
        || selfConfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support same tokens per block and size per head");
        return false;
    }
    if (selfConfig.getModelConfig().mNbKvHeadsPerLayer.size() != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support same number of layers");
        return false;
    }
    if ((selfConfig.getModelConfig().mNbKvHeadsPerLayer.at(0) != 1)
        || (selfConfig.getModelConfig().mNbKvHeadsPerLayer.at(0) != 1))
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: only support MLA");
        return false;
    }
    if (selfConfig.getParallelConfig().mEnableAttentionDP
        && (selfConfig.getParallelConfig().mTensorParallelism % selfConfig.getParallelConfig().mDPsize != 0))
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: TP size must be divisible by DP size");
        return false;
    }
    if (destConfig.getParallelConfig().mEnableAttentionDP
        && (destConfig.getParallelConfig().mTensorParallelism % destConfig.getParallelConfig().mDPsize != 0))
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: TP size must be divisible by DP size");
        return false;
    }
    if ((destConfig.getParallelConfig().mEnableAttentionDP)
        && (destConfig.getParallelConfig().mTensorParallelism != destConfig.getParallelConfig().mDPsize))
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: TP size must be equal to DP size");
        return false;
    }

    return true;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
