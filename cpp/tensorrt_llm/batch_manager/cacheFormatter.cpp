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

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cstddef>
#include <cstdint>
#include <future>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

void CacheFormatter::formatOutput(LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(formatOutput);
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending KV cache for request ID: %ld.", llmRequest.mRequestId);

    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");
    TLLM_CHECK(!connections.empty());
    constexpr SizeType32 beam{0};
    auto& blockManager = mCacheManager->getBlockManager();
    size_t requestBlockNum = llmRequest.getRequestedBlockHashes().size();
    auto blockRange = BlockRange::fromOldAllocatedBlockIds(*mCacheManager, llmRequest.mRequestId, beam);
    if (requestBlockNum < blockRange.size() && requestBlockNum > 0)
    {
        // handle block reuse, the prefix blocks are reused
        // TODO(zhengd): pass the hashes directly instead of from llmRequest; use hash instead of block num
        auto const& ids = blockRange.getBlockIds();
        blockRange.setBlockIds({ids.end() - requestBlockNum, ids.end()});
    }

    auto const numPools = blockManager.getNumPools();
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...

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
                for (auto const& connection : connections)
                {
                    TLLM_LOG_DEBUG("Send layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                    TransferHelper::sendBuffer(*connection, *layer, llmRequest.mRequestId);
                }
            }
        }
    }
    else
    {
        int blockNum = 0;
        std::vector<runtime::ITensor::SharedPtr> inputKvCacheBlocks;
        for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            blockRange.updatePoolIdx(poolIdx);
            for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
            {
                blockNum++;
                inputKvCacheBlocks.push_back(it);
            }
        }
        TLLM_CHECK(!inputKvCacheBlocks.empty());
        TLLM_CHECK(blockNum > 0);
        int deviceId = mCacheManager->getBlockManager().getStreamDevice();

        if (common::getEnvTryZCopyForKVCacheTransfer()
            && (destConfig.getParallelConfig().mPipelineParallelism
                <= selfConfig.getParallelConfig().mPipelineParallelism)
            && (destConfig.getParallelConfig().mTensorParallelism <= selfConfig.getParallelConfig().mTensorParallelism))
        {
            TLLM_LOG_DEBUG("Try using zero-copy for the KV cache.");
            NVTX3_SCOPED_RANGE(sendBufferFun);

            TLLM_CHECK(connections.size() == 1);

            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            for (auto const& connection : connections)
            {
                for (auto const& block : inputKvCacheBlocks)
                {
                    TransferHelper::sendBuffer(*connection, *block, llmRequest.mRequestId);
                }
            }
            TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.",
                llmRequest.mRequestId);

            return;
        }

        auto cacheBlockSize = inputKvCacheBlocks.front()->getSize();

        auto cacheBufferId = mCacheTransBufferManager->assignBufferIndexForSend();

        auto targetNum = connections.size();
        TLLM_CHECK((cacheBlockSize * blockNum) % targetNum == 0);
        auto const targetBufferSize = (cacheBlockSize * blockNum) / targetNum;

        auto result = mCacheTransBufferManager->getOrAllocateSendBuffers(
            cacheBufferId, targetNum, targetBufferSize, bufferManager);
        auto& outputSplitCaches = std::get<0>(result);
        auto& bufferCoverTargetNum = std::get<1>(result);
        auto& onlyUseDynamicBuffer = std::get<2>(result);
        auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
        if (agentConnnecion != nullptr)
        {
            TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == targetNum, "Agent need all buffer pre-allocated");
            TLLM_CHECK(onlyUseDynamicBuffer == false);
        }

        tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(
            inputKvCacheBlocks, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager);

        bufferManager.getStream().synchronize();

        auto preAllocSendBuffer = mCacheTransBufferManager->getSendBuffer(cacheBufferId);
        if (preAllocSendBuffer != nullptr)
        {
            TLLM_CHECK(preAllocSendBuffer->getDataType() == inputKvCacheBlocks.front()->getDataType());
        }
        auto sendBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(sendBufferFun);
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
            TLLM_CHECK(connections.size() > processIdx);
            TLLM_CHECK(outputSplitCaches.size() > processIdx);
            auto startTime = std::chrono::steady_clock::now();
            size_t size;
            if (processIdx < bufferCoverTargetNum)
            {
                size = (*outputSplitCaches[processIdx]).getSizeInBytes();
                TransferHelper::sendBuffer(
                    *connections[processIdx], *outputSplitCaches[processIdx], llmRequest.mRequestId);
            }
            else if (bufferCoverTargetNum > 0)
            {
                // copy buffer allocated by cudaMallocAsync to buffer allocated by cudaMalloc before sending
                auto sendBufferIdx = processIdx % bufferCoverTargetNum;
                bufferManager.copy(*outputSplitCaches[processIdx], *outputSplitCaches.at(sendBufferIdx));
                bufferManager.getStream().synchronize();
                size = (*outputSplitCaches.at(sendBufferIdx)).getSizeInBytes();
                TransferHelper::sendBuffer(
                    *connections[processIdx], *outputSplitCaches.at(sendBufferIdx), llmRequest.mRequestId);
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
                        outputSplitCaches[processIdx], targetBufferSize - remainSendSize, sendSize);

                    auto copyTargetSlice = runtime::ITensor::slice(preAllocSendBuffer, 0, sendSize);
                    bufferManager.copy(*copySlice, *copyTargetSlice);
                    bufferManager.getStream().synchronize();
                    TransferHelper::sendBuffer(*connections[processIdx], *copyTargetSlice, llmRequest.mRequestId);
                    remainSendSize -= sendSize;
                }
            }

            auto endTime = std::chrono::steady_clock::now();
            double cacheTransferTime
                = std::max(0.0, std::chrono::duration<double, std::milli>(endTime - startTime).count());
            kvCacheMeasureHelper.appendKVCacheTransfer(llmRequest.mRequestId, cacheTransferTime, size);
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

void CacheFormatter::formatInput(LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(formatInput);
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "Start receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
    TLLM_CHECK(!connections.empty());
    auto blockRange = BlockRange::fromNewlyAllocatedBlockIds(*mCacheManager, llmRequest.mRequestId);
    std::vector<runtime::ITensor::SharedPtr> recvBufferTmps;
    std::vector<runtime::ITensor::SharedPtr> outputBuffers;
    auto const numPools = mCacheManager->getBlockManager().getNumPools();
    // TODO(oargov): are we sure the other side has the same number of pools? this might not hold for pp_size>1...
    size_t blockNum = 0;
    for (auto poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        blockRange.updatePoolIdx(poolIdx);
        for (auto it = blockRange.begin(); it != blockRange.end(); ++it)
        {
            blockNum++;
            outputBuffers.push_back(it);
        }
    }
    TLLM_CHECK(!outputBuffers.empty());
    {
        NVTX3_SCOPED_RANGE(formatInputRecvBuffer);

        auto reqId = llmRequest.getContextPhaseParams().value().getReqId();
        auto dataType = mCacheManager->getPrimaryPool(0)->getDataType();
        bool layerWise = common::getEnvDisaggLayerwise() && numPools == 1;
        if (layerWise)
        {
            // [numLayersInPool, ...]
            auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
            auto cacheVolume = runtime::ITensor::volume(cacheShape);
            size_t bufferNum = blockNum * connections.size();
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
                    for (auto const& connection : connections)
                    {
                        TLLM_LOG_DEBUG("Receive layer %d(%d-%d)", layerIdx, poolIdx, layerIdxInPool);
                        // Buffer dim: [numLayersInPool * layerVolume]
                        auto layer
                            = runtime::ITensor::slice(recvBufferTmps[idx], layerIdxInPool * layerVolume, layerVolume);
                        llmRequest.updateKvCacheSize((*layer).getSizeInBytes());
                        TransferHelper::recvBuffer(*connection, *layer, reqId);
                        idx++;
                    }
                }
            }
            {
                NVTX3_SCOPED_RANGE(formatInputConcatenate);
                executor::kv_cache::concatenateKVCacheDispatch(recvBufferTmps.data(), recvBufferTmps.size(),
                    getCounterparts(selfConfig, selfIdx, destConfig), destConfig, outputBuffers.data(),
                    outputBuffers.size(), selfIdx, selfConfig, bufferManager);
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

                TLLM_CHECK(connections.size() == 1);

                TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                for (auto const& connection : connections)
                {
                    for (auto const& block : outputBuffers)
                    {
                        llmRequest.updateKvCacheSize((*block).getSizeInBytes());
                        TransferHelper::recvBuffer(*connection, *block, reqId);
                    }
                }
                TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
                    "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
                    llmRequest.getContextPhaseParams().value().getReqId());
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

            auto cacheBlockSize = outputBuffers.front()->getSize();

            auto dataType = outputBuffers.front()->getDataType();
            auto targetNum = connections.size();
            TLLM_CHECK((cacheBlockSize * blockNum) % targetNum == 0);
            auto targetBufferSize = (cacheBlockSize * blockNum) / targetNum;

            size_t remainNoCoverTargetNum = 0;
            size_t bufferCoverTargetNum = 0;
            std::optional<int> cacheBufferId = std::nullopt;
            {
                NVTX3_SCOPED_RANGE(formatInputAllocBuffer);

                TLLM_CHECK(blockNum > 0);
                TLLM_CHECK(outputBuffers.size() == blockNum);
                if (legacyPath)
                {

                    TLLM_LOG_DEBUG("formatOutput using legacy path");
                    auto cacheShape = executor::kv_cache::makeShapeFromCacheState(destConfig);
                    auto cacheVolume = runtime::ITensor::volume(cacheShape);

                    size_t bufferNum = blockNum * connections.size();
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
                    auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
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
                TLLM_CHECK(connections.size() > processIdx);
                TLLM_CHECK(recvSplitCaches.size() > processIdx);
                if (legacyPath)
                {
                    size_t idx = processIdx * blockNum;

                    for (size_t i = 0; i < blockNum; i++)
                    {
                        size_t commIdx = idx / (blockNum);
                        size_t blockIdx = idx % (blockNum);
                        size_t recvBufferIdx = blockIdx * connections.size() + commIdx;
                        llmRequest.updateKvCacheSize((*recvSplitCaches[recvBufferIdx]).getSizeInBytes());
                        TransferHelper::recvBuffer(*connections[processIdx], *recvSplitCaches.at(recvBufferIdx), reqId);
                        idx++;
                    }
                }
                else
                {
                    if (processIdx >= remainNoCoverTargetNum)
                    {
                        llmRequest.updateKvCacheSize((*recvSplitCaches.at(processIdx)).getSizeInBytes());
                        TransferHelper::recvBuffer(*connections[processIdx], *recvSplitCaches[processIdx], reqId);
                    }
                    else if (bufferCoverTargetNum > 0)
                    {
                        auto recvBufferIdx = processIdx % bufferCoverTargetNum
                            + remainNoCoverTargetNum; // caches.at(recvBufferIdx) is allocated by cudaMalloc
                        llmRequest.updateKvCacheSize((*recvSplitCaches.at(recvBufferIdx)).getSizeInBytes());
                        TransferHelper::recvBuffer(*connections[processIdx], *recvSplitCaches.at(recvBufferIdx), reqId);
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
                            llmRequest.updateKvCacheSize((*recvSlice).getSizeInBytes());
                            TransferHelper::recvBuffer(*connections[processIdx], *recvSlice, reqId);
                            bufferManager.copy(*recvSlice, *copySlice);
                            bufferManager.getStream().synchronize();
                            remainRecvSize -= recvSize;
                        }
                    }
                }
            };
            if (connections.size() > 1)
            {
                if (!common::getEnvEnableReceiveKVCacheParallel())
                {

                    for (size_t i = 0; i < connections.size(); i++)
                    {
                        recvBufferFun(deviceId, i);
                    }
                }
                else
                {
                    // concurrency num
                    auto concurrencyNum
                        = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), connections.size());
                    auto remainRecvNum = connections.size();

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
                            TLLM_CHECK((i + (connections.size() - remainRecvNum)) < connections.size());
                            futures.push_back(std::async(
                                std::launch::async, recvBufferFun, deviceId, i + (connections.size() - remainRecvNum)));
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
                    executor::kv_cache::concatenateKVCacheDispatch(recvSplitCaches.data(), recvSplitCaches.size(),
                        getCounterparts(selfConfig, selfIdx, destConfig), destConfig, outputBuffers.data(),
                        outputBuffers.size(), selfIdx, selfConfig, bufferManager);
                }
                else
                {
                    executor::kv_cache::concatenateKvCacheV2Dispatch(
                        recvSplitCaches, outputBuffers, destConfig, selfConfig, selfIdx, bufferManager);
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
    std::unordered_set<SizeType32> setVecSelf{
        selfConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), selfConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecSelf.size() != 1)
    {
        return false;
    }
    if (selfConfig.getAttentionConfig().mAttentionType != destConfig.getAttentionConfig().mAttentionType)
    {
        return false;
    }
    if (selfConfig.getAttentionConfig().mKvFactor != destConfig.getAttentionConfig().mKvFactor)
    {
        return false;
    }
    if (selfConfig.getAttentionConfig().mAttentionType == CacheState::AttentionType::kMLA)
    {
        return false;
    }
    std::unordered_set<int> setVecDest{
        destConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), destConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecDest.size() != 1)
    {
        return false;
    }
    if (selfConfig.getModelConfig().mTokensPerBlock != destConfig.getModelConfig().mTokensPerBlock
        || selfConfig.getModelConfig().mSizePerHead != destConfig.getModelConfig().mSizePerHead)
    {
        return false;
    }
    if (selfConfig.getModelConfig().mNbKvHeadsPerLayer.size() != destConfig.getModelConfig().mNbKvHeadsPerLayer.size())
    {
        return false;
    }
    int selfTPInDP = selfConfig.getParallelConfig().mEnableAttentionDP
        ? selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize
        : selfConfig.getParallelConfig().mTensorParallelism;
    int destTPInDP = destConfig.getParallelConfig().mEnableAttentionDP
        ? destConfig.getParallelConfig().mTensorParallelism / destConfig.getParallelConfig().mDPsize
        : destConfig.getParallelConfig().mTensorParallelism;
    int selfNumHeads = selfConfig.getModelConfig().mNbKvHeadsPerLayer[0] * selfTPInDP;
    int destNumHeads = destConfig.getModelConfig().mNbKvHeadsPerLayer[0] * destTPInDP;
    return selfNumHeads == destNumHeads;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
