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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

std::pair<std::vector<size_t>, std::vector<size_t>> MLACacheFormatter::pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks) const
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (targetInfo.mIRanks.empty())
    {
        return {{}, {}};
    }

    TLLM_CHECK(targetInfo.mDomainCPSize == 1);
    TLLM_CHECK(numConnections == counterPartRanks.size());
    std::vector<size_t> pickUpConnections;
    std::vector<size_t> localRankIndices;
    int dpRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;

    for (int i = 0; i < targetInfo.mDomainPPSize; i++)
    {
        size_t rankIdx = i + (dpRank % (targetInfo.mDomainTPSize)) * targetInfo.mDomainPPSize;
        auto rank = targetInfo.mIRanks.at(rankIdx);
        auto it = std::find(counterPartRanks.begin(), counterPartRanks.end(), rank);
        TLLM_CHECK_WITH_INFO(it != counterPartRanks.end(), "Required rank %d not found in counterPartRanks", rank);
        pickUpConnections.push_back(std::distance(counterPartRanks.begin(), it));
        localRankIndices.push_back(rankIdx);
    }
    return {pickUpConnections, localRankIndices};
}

std::vector<size_t> MLACacheFormatter::pickSendConnections(size_t numConnections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig, std::vector<SizeType32> const& counterPartRanks) const
{
    TLLM_CHECK(numConnections == counterPartRanks.size());
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);

    // Map needed ranks to indices in counterPartRanks
    std::vector<size_t> indices;
    for (auto rank : targetInfo.mIRanks)
    {
        auto it = std::find(counterPartRanks.begin(), counterPartRanks.end(), rank);
        TLLM_CHECK_WITH_INFO(it != counterPartRanks.end(), "Required rank %d not found in counterPartRanks", rank);
        indices.push_back(std::distance(counterPartRanks.begin(), it));
    }
    return indices;
}

bool MLACacheFormatter::needSendCache(
    CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    int selfCpSize = selfConfig.getParallelConfig().mContextParallelism;
    int selfTpRank = (selfIdx % (selfConfig.getParallelConfig().mTensorParallelism * selfCpSize)) / selfCpSize;

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
    auto pickUpConnections
        = pickSendConnections(connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks());
    auto targetNum = pickUpConnections.size();
    if (targetNum == 0)
    {
        TLLM_LOG_DEBUG("No targets to send KV cache to for request ID: %ld", llmRequest.mRequestId);
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
    auto const ppSize = selfConfig.getParallelConfig().mPipelineParallelism;
    auto blockRange
        = getBlockRangeForSending(mCacheManager, llmRequest, lastBlockKey, indexFromEnd, recvSideHasCP, ppSize);
    auto const& windowSizes = blockRange.getWindowSizes();
    TLLM_CHECK_WITH_INFO(
        static_cast<int>(windowSizes.size()) == numPools, "window sizes should be the same as numPools");

    for (auto transferIndexerKCache : transferringIndexerKCache)
    {
        auto bufferKind = transferIndexerKCache ? static_cast<uint8_t>(BufferKind::kKV_INDEXER)
                                                : static_cast<uint8_t>(BufferKind::kKV);
        for (size_t i = 0; i < pickUpConnections.size(); i++)
        {
            auto const* connection = connections.at(pickUpConnections[i]);
            connection->activateBuffer(bufferKind);
        }
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
            for (size_t i = 0; i < pickUpConnections.size(); i++)
            {
                for (auto const& block : inputKvCacheBlocks)
                {
                    session.send(pickUpConnections[i], block->data(), block->getSizeInBytes());
                }
            }

            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.",
                llmRequest.mRequestId);

            return;
        }

        // The indexer K cache pass shares the attention pass's connections/topology but runs in
        // "indexer layer space": with a masked indexer pool only full-indexer layers own a row,
        // so both the self layer count and the per-peer layer counts come from the indexer
        // layer counts (dense models fall back to the attention counts).
        auto targetInfo = transferIndexerKCache
            ? executor::kv_cache::targetIRanksForIndexerKCache(destConfig, selfConfig, selfIdx)
            : executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
        size_t pPDomainSize = targetInfo.mDomainPPSize;
        size_t cPDomainSize = targetInfo.mDomainCPSize;

        auto getBufferSizeForTarget = [&]()
        {
            auto const ppRank = selfIdx
                / (selfConfig.getParallelConfig().mTensorParallelism
                    * selfConfig.getParallelConfig().mContextParallelism);
            auto const selfLayerNum = transferIndexerKCache
                ? selfConfig.getIndexerLayerNumPerPP().at(ppRank)
                : selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);
            auto const cacheBlockSize = inputKvCacheBlocks.at(0)->getSize();
            auto const blockSizePerLayer = cacheBlockSize / selfLayerNum;
            std::vector<size_t> bufferSizeForTarget(pPDomainSize * cPDomainSize, 0);
            for (size_t ppDomainIdx = 0; ppDomainIdx < pPDomainSize; ppDomainIdx++)
            {
                auto const peerLayerNum = targetInfo.getPeerPPDomainLayerNum(ppDomainIdx);
                for (size_t cpDomainIdx = 0; cpDomainIdx < cPDomainSize; cpDomainIdx++)
                {
                    auto const idx = cpDomainIdx * pPDomainSize + ppDomainIdx;
                    // Note: contextCP is always 1. So, cpDomainSize == genCPSize and cpDomainIdx == genCPRank.
                    auto const peerBlockNum
                        = executor::kv_cache::getBlockNumAccountingForCP(cpDomainIdx, cPDomainSize, blockNum);
                    bufferSizeForTarget[idx] = blockSizePerLayer * peerLayerNum * peerBlockNum;
                }
            }
            return bufferSizeForTarget;
        };
        auto bufferEleSizes = getBufferSizeForTarget();
        auto const* sendCancelFlag
            = common::getEnvDisaggEnableInflightCancel() ? &session.getDataContext().getTransferTerminate() : nullptr;
        auto cacheBufferId = mCacheTransBufferManagers[transferIndexerKCache]->assignBufferIndexForSend(sendCancelFlag);
        BufferIndexHolder sendHolder(
            *mCacheTransBufferManagers[transferIndexerKCache], cacheBufferId, /*isRecv=*/false);
        auto result = mCacheTransBufferManagers[transferIndexerKCache]->getOrAllocateSendBuffers(
            cacheBufferId, static_cast<int>(pPDomainSize * cPDomainSize), bufferEleSizes, bufferManager);
        auto& outputSplitCaches = std::get<0>(result);
        auto& bufferCoverTargetNum = std::get<1>(result);
        auto& onlyUseDynamicBuffer = std::get<2>(result);
        auto const* agentConnection
            = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[pickUpConnections[0]]);
        if (agentConnection != nullptr)
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
        // Connections are ordered CP-major (all connections for CP=0, then all for CP=1, etc.)
        // from targetIRanks(). Within each CP domain, connections are ordered by (TP, PP).
        // The bufferSizeForTarget is indexed as: cpDomainIdx * pPDomainSize + ppDomainIdx.
        // So we need to compute cacheIdx based on the CP-major connection ordering.
        auto connectionsPerCPDomain = connections.size() / cPDomainSize;
        TLLM_CHECK_WITH_INFO(connectionsPerCPDomain > 0, "connectionsPerCPDomain must be > 0");

        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            auto startTime = LlmRequest::getSteadyClockNow();
            // Compute cacheIdx based on CP-major connection ordering:
            // - cpDomainIdx = which CP domain this connection belongs to.
            // - ppDomainIdx = which PP domain within the CP domain (for PP > 1).
            auto cpDomainIdx = processIdx / connectionsPerCPDomain;
            auto ppDomainIdx = (processIdx % connectionsPerCPDomain) % pPDomainSize;
            auto cacheIdx = cpDomainIdx * pPDomainSize + ppDomainIdx;
            // Helix: skip CP ranks that own no blocks for this sequence (num_total_blocks < cp_size).
            // The matching gen rank skips its receive, so no 0-byte transfer is posted on either side.
            auto const& splitCache = outputSplitCaches.at(cacheIdx);
            if (splitCache == nullptr || splitCache->getSizeInBytes() == 0)
            {
                return;
            }
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

        if (sendCancelFlag != nullptr && sendCancelFlag->load(std::memory_order_relaxed))
        {
            TLLM_THROW("MLA cache transfer cancelled before NIXL submission");
        }

        try
        {
            if (pickUpConnections.size() > 1)
            {
                if (!common::getEnvEnableReceiveKVCacheParallel())
                {
                    TLLM_LOG_DEBUG("Disable parallel receiving of the KV cache.");
                    for (size_t i = 0; i < pickUpConnections.size(); i++)
                    {
                        sendBufferFun(deviceId, pickUpConnections[i]);
                    }
                }
                else
                {
                    // concurrency num
                    auto concurrencyNum
                        = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), pPDomainSize * cPDomainSize);

                    auto remainSendNum = pickUpConnections.size();

                    while (remainSendNum > 0)
                    {
                        auto sendConcurrencyNum = std::min(remainSendNum, concurrencyNum);
                        std::vector<std::future<void>> futures;
                        futures.reserve(sendConcurrencyNum);
                        for (size_t i = 0; i < sendConcurrencyNum; i++)
                        {
                            size_t idx = i + (pickUpConnections.size() - remainSendNum);
                            size_t connIdx = pickUpConnections[idx];
                            TLLM_CHECK(idx < pickUpConnections.size());
                            TLLM_CHECK(connIdx < session.getConnections().size());
                            futures.push_back(std::async(std::launch::async, sendBufferFun, deviceId, connIdx));
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
                sendBufferFun(deviceId, pickUpConnections[0]);
            }
        }
        catch (...)
        {
            if (agentConnection != nullptr && common::getEnvDisaggEnableInflightCancel())
            {
                sendHolder.poison();
            }
            throw;
        }
        sendHolder.release();
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
    auto pickRecvConnResult
        = pickRecvConnections(connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks());
    auto pickUpConnections = std::get<0>(pickRecvConnResult);
    auto localRankIndices = std::get<1>(pickRecvConnResult);
    if (pickUpConnections.empty())
    {
        TLLM_LOG_DEBUG("No targets to receive KV cache for request ID: %ld", llmRequest.mRequestId);
        return;
    }
    bool const recvSideHasCP = selfConfig.getParallelConfig().mContextParallelism > 1;
    auto const srcPpSize = destConfig.getParallelConfig().mPipelineParallelism;
    auto blockRange = getBlockRangeForReceiving(mCacheManager, llmRequest, destConfig.getEnableBlockReuse(),
        destConfig.getEnablePartialReuse(), recvSideHasCP, srcPpSize);
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

        // Helix: an "empty" CP rank owns no KV blocks for this sequence (num_total_blocks < cp_size).
        // There is nothing to receive; the sender (context, CP=1) skips the matching 0-byte transfer.
        if (blockNum == 0)
        {
            continue;
        }

        int deviceId = bufferManager.getStream().getDevice();

        std::optional<int> cacheBufferId = std::nullopt;
        BufferIndexHolder recvHolder;

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
            auto bufferKind = transferIndexerKCache ? static_cast<uint8_t>(BufferKind::kKV_INDEXER)
                                                    : static_cast<uint8_t>(BufferKind::kKV);
            auto preAssignedId = connections[pickUpConnections[0]]->getPreAssignedBufferId(bufferKind);
            if (preAssignedId.has_value())
            {
                cacheBufferId = static_cast<int>(*preAssignedId);
                if (!session.hasReservedRecvBuffer(*mCacheTransBufferManagers[transferIndexerKCache]))
                {
                    recvHolder = BufferIndexHolder(
                        *mCacheTransBufferManagers[transferIndexerKCache], cacheBufferId, /*isRecv=*/true);
                }
            }
            else
            {
                cacheBufferId = mCacheTransBufferManagers[transferIndexerKCache]->assignBufferIndexForRecv();
                recvHolder = BufferIndexHolder(
                    *mCacheTransBufferManagers[transferIndexerKCache], cacheBufferId, /*isRecv=*/true);
            }

            auto targetNum = pickUpConnections.size();

            auto getBufferSizeForTarget = [&]()
            {
                // Indexer K cache pass: sizes are computed in "indexer layer space" (masked
                // pool: only full-indexer layers own a row); dense models fall back to the
                // attention counts. Peers whose overlap holds no full-indexer layer get a
                // zero-sized buffer and the receive is skipped, mirroring the sender.
                auto const targetInfo = transferIndexerKCache
                    ? executor::kv_cache::targetIRanksForIndexerKCache(destConfig, selfConfig, selfIdx)
                    : executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
                auto const cacheBlockSize = outputBuffers.at(0)->getSize();
                auto const ppRank = selfIdx
                    / (selfConfig.getParallelConfig().mTensorParallelism
                        * selfConfig.getParallelConfig().mContextParallelism);
                auto const selfLayerNum = transferIndexerKCache
                    ? selfConfig.getIndexerLayerNumPerPP().at(ppRank)
                    : selfConfig.getParallelConfig().mAttentionLayerNumPerPP.at(ppRank);
                TLLM_CHECK_WITH_INFO(selfLayerNum != 0, "selfLayerNum should not be 0");
                std::vector<size_t> bufferEleSizes(targetNum, 0);
                auto const cacheSizePerLayer = cacheBlockSize * blockNum / selfLayerNum;
                for (size_t i = 0; i < targetNum; i++)
                {
                    auto const peerLayerNum
                        = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(localRankIndices[i]));
                    bufferEleSizes[i] = cacheSizePerLayer * peerLayerNum;
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
            if (preAssignedId.has_value())
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
                // Zero-sized target (e.g. an indexer K cache peer whose layer overlap holds no
                // full-indexer layer): the sender skips the matching send, so skip the receive.
                if (recvSplitCaches.at(processIdx) != nullptr && recvSplitCaches.at(processIdx)->getSize() == 0)
                {
                    return;
                }
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

        (void) session.releaseReservedRecvBuffer(*mCacheTransBufferManagers[transferIndexerKCache]);
        recvHolder.release();
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
    if (selfConfig.getHasIndexerKCache() != destConfig.getHasIndexerKCache())
    {
        TLLM_LOG_WARNING("MLACacheFormatter::inquireSupport: both sides must agree on the indexer K cache");
        return false;
    }
    if (selfConfig.getHasIndexerKCache())
    {
        if (selfConfig.getIndexerDimPerHead() != destConfig.getIndexerDimPerHead()
            || selfConfig.getIndexerKCacheQuantBlockSize() != destConfig.getIndexerKCacheQuantBlockSize()
            || selfConfig.getIndexerKCacheUseFp4() != destConfig.getIndexerKCacheUseFp4())
        {
            TLLM_LOG_WARNING(
                "MLACacheFormatter::inquireSupport: indexer K cache layout (dim per head, quant block size, fp4) "
                "must match on both sides");
            return false;
        }
        // With a masked indexer pool (per-layer indexer mask) both sides must derive the same
        // global full-indexer-layer schedule: the totals must agree even when the per-PP splits
        // differ (PP resharding).
        auto const& selfIndexerPerPP = selfConfig.getIndexerLayerNumPerPP();
        auto const& destIndexerPerPP = destConfig.getIndexerLayerNumPerPP();
        auto const selfIndexerTotal = std::accumulate(selfIndexerPerPP.begin(), selfIndexerPerPP.end(), SizeType32{0});
        auto const destIndexerTotal = std::accumulate(destIndexerPerPP.begin(), destIndexerPerPP.end(), SizeType32{0});
        if (selfIndexerTotal != destIndexerTotal)
        {
            TLLM_LOG_WARNING(
                "MLACacheFormatter::inquireSupport: both sides must own the same total number of indexer K cache "
                "layers (self: %d, dest: %d)",
                selfIndexerTotal, destIndexerTotal);
            return false;
        }
        // Zero-layer indexer exchanges (a rank or a PP-domain peer whose attention overlap holds
        // no full-indexer layer) are not supported yet: the agent-backend handshake and the
        // partial-coverage bounce-buffer accounting cannot represent zero-sized targets. Require
        // every PP rank on both sides to own at least one full-indexer layer, and every
        // attention-domain overlap to contain at least one full-indexer layer.
        auto hasZeroEntry = [](std::vector<SizeType32> const& perPP)
        { return std::any_of(perPP.begin(), perPP.end(), [](SizeType32 layerNum) { return layerNum == 0; }); };
        if (hasZeroEntry(selfIndexerPerPP) || hasZeroEntry(destIndexerPerPP))
        {
            TLLM_LOG_WARNING(
                "MLACacheFormatter::inquireSupport: a PP rank without indexer K cache layers (all local layers "
                "use cross-layer indexer sharing) is not supported");
            return false;
        }
        // Every attention-domain overlap must contain at least one full-indexer layer; a zero
        // overlap would produce a zero-sized transfer target.
        auto overlapsHaveIndexerLayers = [](CacheState const& a, CacheState const& b)
        {
            auto const& aAttention = a.getParallelConfig().mAttentionLayerNumPerPP;
            auto const& bAttention = b.getParallelConfig().mAttentionLayerNumPerPP;
            auto const& aIndexer = a.getIndexerLayerNumPerPP();
            auto const& bIndexer = b.getIndexerLayerNumPerPP();
            int64_t aAttentionStart = 0;
            int64_t aIndexerStart = 0;
            for (size_t aRank = 0; aRank < aAttention.size(); aRank++)
            {
                int64_t const aAttentionEnd = aAttentionStart + aAttention[aRank];
                int64_t const aIndexerEnd = aIndexerStart + aIndexer[aRank];
                int64_t bAttentionStart = 0;
                int64_t bIndexerStart = 0;
                for (size_t bRank = 0; bRank < bAttention.size(); bRank++)
                {
                    int64_t const bAttentionEnd = bAttentionStart + bAttention[bRank];
                    int64_t const bIndexerEnd = bIndexerStart + bIndexer[bRank];
                    auto const attentionOverlap
                        = std::min(aAttentionEnd, bAttentionEnd) - std::max(aAttentionStart, bAttentionStart);
                    auto const indexerOverlap
                        = std::min(aIndexerEnd, bIndexerEnd) - std::max(aIndexerStart, bIndexerStart);
                    if (attentionOverlap > 0 && indexerOverlap <= 0)
                    {
                        return false;
                    }
                    bAttentionStart = bAttentionEnd;
                    bIndexerStart = bIndexerEnd;
                }
                aAttentionStart = aAttentionEnd;
                aIndexerStart = aIndexerEnd;
            }
            return true;
        };
        if (!overlapsHaveIndexerLayers(selfConfig, destConfig))
        {
            TLLM_LOG_WARNING(
                "MLACacheFormatter::inquireSupport: a pipeline-parallel layer overlap between the two sides "
                "holds no full-indexer layer (zero-sized indexer K cache transfer target); this PP partition "
                "combination is not supported");
            return false;
        }
    }

    return true;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
