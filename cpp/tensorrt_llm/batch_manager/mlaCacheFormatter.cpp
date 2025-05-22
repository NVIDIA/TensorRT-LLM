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

#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
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
std::vector<executor::kv_cache::Connection const*> MLACacheFormatter::pickRecvConnections(
    std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig)
{

    TLLM_CHECK(!connections.empty());

    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    TLLM_CHECK(targetInfo.mIRanks.size() == connections.size());
    std::vector<executor::kv_cache::Connection const*> ret;
    // targetInfo , mRanks [tpranks, dpranks]
    for (int i = 0; i < targetInfo.mDomainPPSize; i++)
    {
        ret.push_back(connections.at(i));
    }
    return ret;
}

bool MLACacheFormatter::needSendCache(
    CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx)
{
    int selfTpRank = selfIdx % selfConfig.getParallelConfig().mTensorParallelism;

    if (selfConfig.getParallelConfig().mEnableAttentionDP)
    {
        int selfTPNumInDPGroup
            = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
        int destTPNumInDPGroup = destConfig.getParallelConfig().mEnableAttentionDP
            ? destConfig.getParallelConfig().mTensorParallelism / destConfig.getParallelConfig().mDPsize
            : destConfig.getParallelConfig().mTensorParallelism;
        int selfTPrankINDPGroup = selfTpRank % selfTPNumInDPGroup;
        if (selfTPNumInDPGroup <= destTPNumInDPGroup)
        {
            return true;
        }
        return selfTPrankINDPGroup % (selfTPNumInDPGroup / destTPNumInDPGroup) == 0;
    }

    int destTPNum = destConfig.getParallelConfig().mEnableAttentionDP
        ? destConfig.getParallelConfig().mTensorParallelism / destConfig.getParallelConfig().mDPsize
        : destConfig.getParallelConfig().mTensorParallelism;
    int selfTPNum = selfConfig.getParallelConfig().mTensorParallelism;
    if (selfTPNum <= destTPNum)
    {
        return true;
    }
    return selfTpRank % (selfTPNum / destTPNum) == 0;
}

void MLACacheFormatter::formatOutput(LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(formatOutput);
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending KV cache for request ID: %ld.", llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    TLLM_CHECK(!connections.empty());
    // diff start
    if (!needSendCache(selfConfig, destConfig, selfIdx))
    {
        return;
    }

    // diff end
    auto reqId = llmRequest.mRequestId;

    constexpr SizeType32 beam{0};
    auto const numPools = mCacheManager->getBlockManager().getNumPools();
    auto blockRange = BlockRange::fromOldAllocatedBlockIds(*mCacheManager, llmRequest.mRequestId, beam);

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
    TLLM_CHECK(blockNum > 0);
    int deviceId = mCacheManager->getBlockManager().getStreamDevice();

    if (common::getEnvTryZCopyForKVCacheTransfer()
        && destConfig.getParallelConfig().mPipelineParallelism == selfConfig.getParallelConfig().mPipelineParallelism)
    {

        TLLM_LOG_DEBUG("Try using zero-copy for the KV cache.");
        NVTX3_SCOPED_RANGE(sendBufferFun);

        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        for (auto const& connection : connections)
        {
            for (auto const& block : inputKvCacheBlocks)
            {
                TransferHelper::sendBuffer(*connection, *block, reqId);
            }
        }

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.",
            llmRequest.mRequestId);

        return;
    }

    auto cacheBlockSize = inputKvCacheBlocks.at(0)->getSize();

    auto cacheBufferId = mCacheTransBufferManager->assignBufferIndexForSend();
    // diff start

    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);

    size_t const pPDomainSize = targetInfo.mDomainPPSize;
    TLLM_CHECK((cacheBlockSize * blockNum) % pPDomainSize == 0);
    auto const targetBufferSize = (cacheBlockSize * blockNum) / pPDomainSize;
    auto result = mCacheTransBufferManager->getOrAllocateSendBuffers(
        cacheBufferId, pPDomainSize, targetBufferSize, bufferManager);
    auto& outputSplitCaches = std::get<0>(result);
    auto& bufferCoverTargetNum = std::get<1>(result);
    auto& onlyUseDynamicBuffer = std::get<2>(result);
    auto* agentConnnecion = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[0]);
    if (agentConnnecion != nullptr)
    {
        TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == pPDomainSize, "Agent need all buffer pre-allocated");
        TLLM_CHECK(onlyUseDynamicBuffer == false);
    }
    // diff end

    // The size of outputSplitCaches should be equal to pPDomainSize

    tensorrt_llm::executor::kv_cache::splitKVCacheDispatch(
        inputKvCacheBlocks, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager);

    bufferManager.getStream().synchronize();

    auto preAllocSendBuffer = mCacheTransBufferManager->getSendBuffer(cacheBufferId);
    if (preAllocSendBuffer != nullptr)
    {
        TLLM_CHECK(preAllocSendBuffer->getDataType() == inputKvCacheBlocks.at(0)->getDataType());
    }
    auto sendBufferFun = [&](int deviceId, size_t processIdx)
    {
        NVTX3_SCOPED_RANGE(sendBufferFun);

        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        auto cacheIdx = processIdx % pPDomainSize;
        if (cacheIdx < bufferCoverTargetNum)
        {

            TransferHelper::sendBuffer(*connections.at(processIdx), *outputSplitCaches.at(cacheIdx), reqId);
        }
        else if (bufferCoverTargetNum > 0)
        {
            // copy buffer allocated by cudaMallocAsync to buffer allocated by cudaMalloc before sending
            auto sendBufferIdx = cacheIdx % bufferCoverTargetNum;
            bufferManager.copy(*outputSplitCaches.at(cacheIdx), *outputSplitCaches.at(sendBufferIdx));
            bufferManager.getStream().synchronize();
            TransferHelper::sendBuffer(*connections.at(processIdx), *outputSplitCaches.at(sendBufferIdx), reqId);
        }
        else
        {

            // bufferCoverTargetNum=0, mSendBuffer size < one outputSlice
            // send multiple times
            size_t remainSendSize = targetBufferSize;
            while (remainSendSize > 0)
            {
                TLLM_CHECK(preAllocSendBuffer != nullptr);
                auto sendBufferEleSize = preAllocSendBuffer->getSize();
                auto sendSize = std::min(remainSendSize, sendBufferEleSize);
                auto copySlice = runtime::ITensor::slice(
                    outputSplitCaches.at(cacheIdx), targetBufferSize - remainSendSize, sendSize);
                auto copyTargetSlice = runtime::ITensor::slice(preAllocSendBuffer, 0, sendSize);
                bufferManager.copy(*copySlice, *copyTargetSlice);
                bufferManager.getStream().synchronize();
                TransferHelper::sendBuffer(*connections.at(processIdx), *copyTargetSlice, reqId);

                remainSendSize -= sendSize;
            }
        }
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
            auto concurrencyNum = std::min(std::max(static_cast<size_t>(1), bufferCoverTargetNum), pPDomainSize);

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

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "End the sending of KV cache for the request ID: %ld.", llmRequest.mRequestId);
}

void MLACacheFormatter::formatInput(LlmRequest const& llmRequest,
    std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
    SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
{
    NVTX3_SCOPED_RANGE(formatInput);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently only supports beam width 1.");
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "Start receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
    auto reqId = llmRequest.getContextPhaseParams().value().getReqId();
    TLLM_CHECK(!connections.empty());
    // diff start
    auto pickUpConnections = pickRecvConnections(connections, selfConfig, selfIdx, destConfig);
    // diff end
    constexpr SizeType32 beam{0};
    auto blockRange = BlockRange::fromOldAllocatedBlockIds(*mCacheManager, llmRequest.mRequestId, beam);
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

    int deviceId = bufferManager.getStream().getDevice();

    std::optional<int> cacheBufferId = std::nullopt;

    if (common::getEnvTryZCopyForKVCacheTransfer()
        && destConfig.getParallelConfig().mPipelineParallelism == selfConfig.getParallelConfig().mPipelineParallelism)
    {
        // recv
        TLLM_LOG_DEBUG("Try zcopy for KV cache");
        NVTX3_SCOPED_RANGE(recvBufferFun);
        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        TLLM_CHECK(pickUpConnections.size() == 1);
        for (auto const& connection : pickUpConnections)
        {
            for (auto const& block : outputBuffers)
            {
                TransferHelper::recvBuffer(*connection, *block, reqId);
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
            cacheBufferId = mCacheTransBufferManager->assignBufferIndexForRecv();
        }

        auto cacheBlockSize = outputBuffers.at(0)->getSize();

        auto targetNum = pickUpConnections.size();
        TLLM_CHECK((cacheBlockSize * blockNum) % targetNum == 0);
        auto targetBufferSize = (cacheBlockSize * blockNum) / targetNum;
        auto result = mCacheTransBufferManager->getOrAllocateRecvBuffers(
            cacheBufferId, targetNum, targetBufferSize, bufferManager);
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

        auto preAllocRecvBuffer = mCacheTransBufferManager->getRecvBuffer(cacheBufferId);
        if (preAllocRecvBuffer != nullptr)
        {
            TLLM_CHECK(preAllocRecvBuffer->getDataType() == outputBuffers.at(0)->getDataType());
        }

        auto recvBufferFun = [&](int deviceId, size_t processIdx)
        {
            NVTX3_SCOPED_RANGE(recvBufferFun);
            TLLM_CUDA_CHECK(cudaSetDevice(deviceId));

            if (processIdx >= remainNoCoverTargetNum)
            {

                TransferHelper::recvBuffer(*pickUpConnections.at(processIdx), *recvSplitCaches.at(processIdx), reqId);
            }
            else if (bufferCoverTargetNum > 0)
            {
                auto recvBufferIdx = processIdx % bufferCoverTargetNum
                    + remainNoCoverTargetNum; // caches.at(recvBufferIdx) is allocated by cudaMalloc
                TransferHelper::recvBuffer(
                    *pickUpConnections.at(processIdx), *recvSplitCaches.at(recvBufferIdx), reqId);
                bufferManager.copy(*recvSplitCaches.at(recvBufferIdx), *recvSplitCaches.at(processIdx));
                bufferManager.getStream().synchronize();
            }
            else
            {
                // bufferCoverTargetNum==0
                size_t remainRecvSize = targetBufferSize;
                while (remainRecvSize > 0)
                {
                    TLLM_CHECK(preAllocRecvBuffer != nullptr);
                    auto recvBufferEleSize = preAllocRecvBuffer->getSize();
                    auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                    auto recvSlice = runtime::ITensor::slice(preAllocRecvBuffer, 0, recvSize);
                    auto copySlice = runtime::ITensor::slice(
                        recvSplitCaches.at(processIdx), targetBufferSize - remainRecvSize, recvSize);
                    TransferHelper::recvBuffer(*pickUpConnections.at(processIdx), *recvSlice, reqId);
                    bufferManager.copy(*recvSlice, *copySlice);
                    bufferManager.getStream().synchronize();
                    remainRecvSize -= recvSize;
                }
            }
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

            // recvSplitCaches size == ppdomainsize
            executor::kv_cache::concatenateKvCacheV2Dispatch(
                recvSplitCaches, outputBuffers, destConfig, selfConfig, selfIdx, bufferManager);
        }
        bufferManager.getStream().synchronize();
    }

    if (cacheBufferId.has_value())
    {
        mCacheTransBufferManager->freeBufferIndexForRecv(cacheBufferId);
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "End receiving KV cache for request ID: %ld, context request ID: %ld.", llmRequest.mRequestId,
        llmRequest.getContextPhaseParams().value().getReqId());
}

[[nodiscard]] bool MLACacheFormatter::inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const
{
    if (selfConfig.getAttentionConfig().mAttentionType != CacheState::AttentionType::kMLA
        || destConfig.getAttentionConfig().mAttentionType != CacheState::AttentionType::kMLA)
    {

        return false;
    }
    if (selfConfig.getAttentionConfig().mKvFactor != destConfig.getAttentionConfig().mKvFactor)
    {
        return false;
    }

    std::unordered_set<SizeType32> setVecSelf{
        selfConfig.getModelConfig().mNbKvHeadsPerLayer.begin(), selfConfig.getModelConfig().mNbKvHeadsPerLayer.end()};

    if (setVecSelf.size() != 1)
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
    if ((selfConfig.getModelConfig().mNbKvHeadsPerLayer.at(0) != 1)
        || (selfConfig.getModelConfig().mNbKvHeadsPerLayer.at(0) != 1))
    {
        return false;
    }

    if (selfConfig.getAttentionConfig().mKvFactor != destConfig.getAttentionConfig().mKvFactor)
    {
        return false;
    }
    if (selfConfig.getParallelConfig().mEnableAttentionDP
        && (selfConfig.getParallelConfig().mTensorParallelism % selfConfig.getParallelConfig().mDPsize != 0))
    {

        return false;
    }
    if (destConfig.getParallelConfig().mEnableAttentionDP
        && (destConfig.getParallelConfig().mTensorParallelism % destConfig.getParallelConfig().mDPsize != 0))
    {
        return false;
    }
    if ((destConfig.getParallelConfig().mEnableAttentionDP)
        && (destConfig.getParallelConfig().mTensorParallelism != destConfig.getParallelConfig().mDPsize))
    {
        return false;
    }

    return true;
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
