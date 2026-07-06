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

#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include <algorithm>

namespace tensorrt_llm::batch_manager
{
using CacheState = executor::kv_cache::CacheState;

RnnCacheFormatter::RnnCacheFormatter(rnn_state_manager::RnnStateManager* rnnStateManager,
    rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager)
    : mRnnStateManager{rnnStateManager}
    , mRnnCacheTransBufferManager{rnnCacheTransBufferManager}
{
    TLLM_CHECK(mRnnStateManager != nullptr);
    TLLM_CHECK(mRnnCacheTransBufferManager != nullptr);
}

RnnCacheFormatter::RnnCacheFormatter(kv_cache_manager::BaseKVCacheManager* kvCacheManager,
    rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager)
    : mRnnCacheTransBufferManager{rnnCacheTransBufferManager}
    , mKvCacheManager{kvCacheManager}
{
    TLLM_CHECK(mKvCacheManager != nullptr);
    TLLM_CHECK(mRnnCacheTransBufferManager != nullptr);
}

void RnnCacheFormatter::format(TransferSession& session)
{
    if (isUnifiedPoolMode())
    {
        formatUnifiedPoolMode(session);
    }
    else
    {
        formatSlotMode(session);
    }
}

void RnnCacheFormatter::unformat(TransferSession& session)
{
    if (isUnifiedPoolMode())
    {
        unformatUnifiedPoolMode(session);
    }
    else
    {
        unformatSlotMode(session);
    }
}

void RnnCacheFormatter::formatSlotMode(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_format);
    session.setTime(TransferSession::kTimeFormatter);

    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending RNN state for request ID: %ld.", llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    if (!cache_formatter_utils::needSendCache(selfConfig, destConfig, selfIdx, targetInfo))
    {
        return;
    }

    auto pickUpConnections = cache_formatter_utils::pickSendConnections(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks(), targetInfo);
    auto const targetNum = pickUpConnections.size();
    if (targetNum == 0)
    {
        TLLM_LOG_DEBUG("No targets to send RNN state to for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    auto const slotIdx = mRnnStateManager->getCacheIndex(llmRequest.mRequestId);
    int deviceId;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const selfTPNum = selfParallel.mTensorParallelism;
    auto const selfPPRank = selfIdx / selfTPNum;
    auto const& selfLayersPerPP = selfConfig.getRnnCacheState().mLayerNumPerPP;
    SizeType32 const numLocalLayers = selfLayersPerPP[selfPPRank];

    if (common::getEnvTryZCopyForKVCacheTransfer() && destConfig == selfConfig)
    {
        TLLM_LOG_DEBUG("Try using zero-copy for the RNN cache.");
        NVTX3_SCOPED_RANGE(RnnZeroCopySend);

        TLLM_CHECK(pickUpConnections.size() == 1);

        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        for (size_t i = 0; i < pickUpConnections.size(); i++)
        {
            for (SizeType32 layer = 0; layer < numLocalLayers; layer++)
            {

                // Get conv state for this layer: shape is [maxBatchSize, convDim, dConv-1]
                auto convState = mRnnStateManager->getConvStates(mRnnStateManager->getGlobalLayerNum(layer));
                // Slice out the specific slot: shape becomes [convDim, dConv-1]
                auto slotConv = runtime::ITensor::slice(convState, slotIdx, 1);
                slotConv->squeeze(0);

                // Receive conv state
                // llmRequest.updateKvCacheSize(slotConv->getSizeInBytes());
                session.send(pickUpConnections[i], slotConv->data(), slotConv->getSizeInBytes());

                // Get SSM state for this layer: shape is [maxBatchSize, numHeads, headDim, dState]
                auto ssmState = mRnnStateManager->getSsmStates(mRnnStateManager->getGlobalLayerNum(layer));
                // Slice out the specific slot: shape becomes [numHeads, headDim, dState]
                auto slotSsm = runtime::ITensor::slice(ssmState, slotIdx, 1);
                slotSsm->squeeze(0);

                // Receive SSM state
                // llmRequest.updateKvCacheSize(slotSsm->getSizeInBytes());
                session.send(pickUpConnections[i], slotSsm->data(), slotSsm->getSizeInBytes());
            }
        }
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End the sending of RNN cache for the request ID: %ld.",
            llmRequest.mRequestId);

        return;
    }

    // Calculate buffer sizes for each target
    //    Each target gets: conv states + ssm states for overlapping layers
    auto const& modelConfig = selfConfig.getRnnModelConfig();
    auto const maxBatchSize = mRnnStateManager->getMaxBatchSize();
    int const selfTPSizePerDPGroup = selfConfig.getParallelConfig().mEnableAttentionDP
        ? selfTPNum / selfConfig.getParallelConfig().mDPsize
        : selfTPNum;
    SizeType32 convDimLocal = modelConfig.mConvDimSize / selfTPSizePerDPGroup;
    SizeType32 numHeadsLocal = modelConfig.mNumHeads / selfTPSizePerDPGroup;

    size_t convBytesPerLayer
        = convDimLocal * (modelConfig.mDConv - 1) * common::getDTypeSize(selfConfig.getConvStateDataType());
    convBytesPerLayer = (convBytesPerLayer + 15) & ~static_cast<size_t>(15);
    size_t ssmBytesPerLayer = numHeadsLocal * modelConfig.mHeadDim * modelConfig.mDState
        * common::getDTypeSize(selfConfig.getSsmStateDataType());

    int peerDuplicateHeadFactor = targetInfo.mPeerDupHeadFactor;
    auto bufferTargetNum = targetNum / peerDuplicateHeadFactor;

    std::vector<size_t> bufferSizesPerTarget(targetNum, 0);

    for (size_t i = 0; i < targetNum; i++)
    {
        SizeType32 layersForTarget = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(i));
        bufferSizesPerTarget[i] = layersForTarget * (convBytesPerLayer + ssmBytesPerLayer) * peerDuplicateHeadFactor
            / targetInfo.mDomainTPSize;
    }

    auto cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForSend();
    auto allocationResult = mRnnCacheTransBufferManager->getOrAllocateSendBuffers(
        cacheBufferId, static_cast<int>(bufferTargetNum), bufferSizesPerTarget, bufferManager);
    auto& outputBuffers = std::get<0>(allocationResult);
    auto& bufferCoverTargetNum = std::get<1>(allocationResult);
    auto& onlyUseDynamicBuffer = std::get<2>(allocationResult);

    TLLM_CHECK(cacheBufferId.has_value() || onlyUseDynamicBuffer);

    auto const* agentConnection
        = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections[pickUpConnections[0]]);
    if (agentConnection != nullptr)
    {
        TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == bufferTargetNum, "Agent needs all RNN send buffers pre-allocated");
        TLLM_CHECK(onlyUseDynamicBuffer == false);
    }

    std::vector<runtime::ITensor::SharedPtr> inputConvBlocks;
    std::vector<runtime::ITensor::SharedPtr> inputSsmBlocks;

    auto convStates = mRnnStateManager->getConvStates(); // [numLocalLayers, maxBatchSize, convDim, dConv-1]
    auto ssmStates = mRnnStateManager->getSsmStates();   // [numLocalLayers, maxBatchSize, numHeads, headDim, dState]

    inputConvBlocks.push_back(convStates);
    inputSsmBlocks.push_back(ssmStates);

    tensorrt_llm::executor::rnn_cache::splitRnnConvStateDispatch(
        inputConvBlocks, outputBuffers, slotIdx, maxBatchSize, destConfig, selfConfig, selfIdx, bufferManager);

    // Conv and SSM use same output buffer. So need to track convBytesPerLayer to compute offset.
    tensorrt_llm::executor::rnn_cache::splitRnnSsmStateDispatch(inputSsmBlocks, outputBuffers, slotIdx, maxBatchSize,
        convBytesPerLayer, destConfig, selfConfig, selfIdx, bufferManager);

    bufferManager.getStream().synchronize();
    session.setTime(TransferSession::kTimePreprocess);

    auto preAllocSendBuffer = mRnnCacheTransBufferManager->getSendBuffer(cacheBufferId);

    sendAllBuffers(session, deviceId, outputBuffers, bufferCoverTargetNum, preAllocSendBuffer, bufferManager,
        targetInfo, pickUpConnections);

    session.setTime(TransferSession::kTimeTransmissions);

    mRnnCacheTransBufferManager->freeBufferIndexForSend(cacheBufferId);
    session.setTime(TransferSession::kTimePostprocess);

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "End sending RNN state for request ID: %ld.", llmRequest.mRequestId);
}

void RnnCacheFormatter::unformatSlotMode(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_unformat);
    session.setTime(TransferSession::kTimeFormatter);

    auto& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start receiving RNN state for request ID: %ld.", llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    auto sourceInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    int deviceId;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

    auto pickRecvConnResult = cache_formatter_utils::pickRecvConnections(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks(), sourceInfo);
    auto pickUpConnections = std::get<0>(pickRecvConnResult);
    auto localRankIndices = std::get<1>(pickRecvConnResult);
    auto const sourceNum = pickUpConnections.size();

    if (sourceNum == 0)
    {
        TLLM_LOG_DEBUG("No sources to receive RNN state from for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    if (common::getEnvDisaggLayerwise())
    {
        TLLM_LOG_ERROR("Layer-wise RNN cache transfer is not supported yet");
        return;
    }

    // Since allocation happens earlier
    auto const slotIdx = mRnnStateManager->getCacheIndex(llmRequest.mRequestId);

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const selfTPNum = selfParallel.mTensorParallelism;
    auto const selfPPRank = selfIdx / selfTPNum;
    auto const& selfLayersPerPP = selfConfig.getRnnCacheState().mLayerNumPerPP;
    SizeType32 const numLocalLayers = selfLayersPerPP[selfPPRank];

    if (common::getEnvTryZCopyForKVCacheTransfer() && destConfig == selfConfig)
    {
        TLLM_LOG_DEBUG("try zcopy for RNN cache");
        NVTX3_SCOPED_RANGE(RnnZeroCopyRecv);

        TLLM_CHECK(sourceNum == 1);

        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        for (size_t i = 0; i < sourceNum; i++)
        {
            for (SizeType32 layer = 0; layer < numLocalLayers; layer++)
            {

                // Get conv state for this layer: shape is [maxBatchSize, convDim, dConv-1]
                auto convState = mRnnStateManager->getConvStates(mRnnStateManager->getGlobalLayerNum(layer));
                // Slice out the specific slot: shape becomes [convDim, dConv-1]
                auto slotConv = runtime::ITensor::slice(convState, slotIdx, 1);
                slotConv->squeeze(0);

                // Send conv state
                // llmRequest.updateKvCacheSize(slotConv->getSizeInBytes());
                session.recv(pickUpConnections[i], slotConv->data(), slotConv->getSizeInBytes());

                // Get SSM state for this layer: shape is [maxBatchSize, numHeads, headDim, dState]
                auto ssmState = mRnnStateManager->getSsmStates(mRnnStateManager->getGlobalLayerNum(layer));
                // Slice out the specific slot: shape becomes [numHeads, headDim, dState]
                auto slotSsm = runtime::ITensor::slice(ssmState, slotIdx, 1);
                slotSsm->squeeze(0);

                // Send SSM state
                // llmRequest.updateKvCacheSize(slotSsm->getSizeInBytes());
                session.recv(pickUpConnections[i], slotSsm->data(), slotSsm->getSizeInBytes());
            }
        }
        TLLM_LOG_DEBUG(
            mpi::MpiComm::world().getRank(), "End receiving RNN cache for request ID: %ld.", llmRequest.mRequestId);
        return;
    }

    // Calculate buffer sizes
    auto const& modelConfig = selfConfig.getRnnModelConfig();
    int const selfTPSizePerDPGroup = selfParallel.mEnableAttentionDP ? selfTPNum / selfParallel.mDPsize : selfTPNum;
    SizeType32 selfConvDimLocal = modelConfig.mConvDimSize / selfTPSizePerDPGroup;
    int const selfNumHeadsLocal = modelConfig.mNumHeads / selfTPSizePerDPGroup;

    size_t convBytesPerLayer
        = selfConvDimLocal * (modelConfig.mDConv - 1) * common::getDTypeSize(selfConfig.getConvStateDataType());
    convBytesPerLayer = (convBytesPerLayer + 15) & ~static_cast<size_t>(15);
    size_t ssmBytesPerLayer = selfNumHeadsLocal * modelConfig.mHeadDim * modelConfig.mDState
        * common::getDTypeSize(selfConfig.getSsmStateDataType());

    std::vector<size_t> bufferSizesPerSource(sourceNum, 0);
    size_t validTpSources = sourceNum / sourceInfo.mDomainPPSize;

    // Compute source conv bytes for SSM offset
    size_t sourceConvBytesPerLayer = convBytesPerLayer / validTpSources;

    for (size_t i = 0; i < sourceNum; i++)
    {
        SizeType32 layersFromSource = sourceInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(localRankIndices[i]));
        bufferSizesPerSource[i] = layersFromSource * (convBytesPerLayer + ssmBytesPerLayer) / validTpSources;
    }

    // Allocate receive buffers
    size_t remainNoCoverSourceNum = 0;
    size_t bufferCoverSourceNum = 0;
    std::optional<int> cacheBufferId = std::nullopt;

    auto preAssignedRnnId = session.getPreAssignedBufferId(static_cast<uint8_t>(BufferKind::kRNN));
    if (preAssignedRnnId.has_value())
    {
        cacheBufferId = static_cast<int>(*preAssignedRnnId);
    }
    else
    {
        cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForRecv();
    }

    auto allocationResult = mRnnCacheTransBufferManager->getOrAllocateRecvBuffers(
        cacheBufferId, static_cast<int>(sourceNum), bufferSizesPerSource, bufferManager);
    auto& recvBuffers = std::get<0>(allocationResult);
    auto& bufferCoverSourceNumTmp = std::get<1>(allocationResult);
    auto& onlyUseDynamicBuffer = std::get<2>(allocationResult);

    TLLM_CHECK(cacheBufferId.has_value() || onlyUseDynamicBuffer);

    if (preAssignedRnnId.has_value())
    {
        TLLM_CHECK_WITH_INFO(bufferCoverSourceNumTmp == sourceNum, "Agent needs all RNN recv buffers pre-allocated");
        TLLM_CHECK(onlyUseDynamicBuffer == false);
    }

    bufferCoverSourceNum = bufferCoverSourceNumTmp;
    remainNoCoverSourceNum = sourceNum > bufferCoverSourceNum ? sourceNum - bufferCoverSourceNum : 0;

    bufferManager.getStream().synchronize();
    session.setTime(TransferSession::kTimePreprocess);

    // Get pre-allocated buffer for chunked receive
    runtime::ITensor::SharedPtr preAllocRecvBuffer = nullptr;
    if (cacheBufferId.has_value())
    {
        preAllocRecvBuffer = mRnnCacheTransBufferManager->getRecvBuffer(cacheBufferId);
        TLLM_CHECK(preAllocRecvBuffer != nullptr);
    }

    auto recvBufferFun = [&](int devId, size_t srcIdx)
    {
        NVTX3_SCOPED_RANGE(recvBufferFun);
        TLLM_CUDA_CHECK(cudaSetDevice(devId));
        TLLM_CHECK(recvBuffers.size() > srcIdx);
        auto startTime = LlmRequest::getSteadyClockNow();
        size_t size = 0;

        if (srcIdx >= remainNoCoverSourceNum)
        {
            // Fast path: buffer is pre-allocated, receive directly
            auto& buffer = recvBuffers[srcIdx];
            size = buffer->getSizeInBytes();
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), " start recv srcIdx: %lu size:%lu", srcIdx, buffer->getSizeInBytes());
            session.recv(pickUpConnections[srcIdx], buffer->data(), buffer->getSizeInBytes());
            TLLM_LOG_DEBUG(
                mpi::MpiComm::world().getRank(), " recv srcIdx: %lu size:%lu", srcIdx, buffer->getSizeInBytes());
        }
        else
        {
            // Slow path: chunked receive for buffers that couldn't be pre-allocated
            auto recvBufferIdx = bufferCoverSourceNum == 0 ? 0 : srcIdx % bufferCoverSourceNum + remainNoCoverSourceNum;
            auto recvBufferUsed = bufferCoverSourceNum == 0 ? preAllocRecvBuffer : recvBuffers[recvBufferIdx];

            size_t remainRecvSize = recvBuffers[srcIdx]->getSize();
            size_t needRecvSize = recvBuffers[srcIdx]->getSize();

            while (remainRecvSize > 0)
            {
                TLLM_CHECK(recvBufferUsed != nullptr);
                auto recvBufferEleSize = recvBufferUsed->getSize();
                auto recvSize = std::min(remainRecvSize, recvBufferEleSize);
                auto recvSlice = runtime::ITensor::slice(recvBufferUsed, 0, recvSize);
                auto copySlice = runtime::ITensor::slice(recvBuffers[srcIdx], needRecvSize - remainRecvSize, recvSize);
                size += recvSlice->getSizeInBytes();
                session.recv(pickUpConnections[srcIdx], recvSlice->data(), recvSlice->getSizeInBytes());
                // Use cudaMemcpyAsync since we're copying bytes
                TLLM_CUDA_CHECK(cudaMemcpyAsync(copySlice->data(), recvSlice->data(), recvSlice->getSizeInBytes(),
                    cudaMemcpyDeviceToDevice, bufferManager.getStream().get()));
                bufferManager.getStream().synchronize();
                remainRecvSize -= recvSize;
            }
        }

        auto endTime = LlmRequest::getSteadyClockNow();
        session.appendMeasure(startTime, endTime, size);
    };

    // Dispatch receives (sequential or parallel based on env var)
    if (sourceNum > 1)
    {
        if (!common::getEnvEnableReceiveKVCacheParallel())
        {
            TLLM_LOG_DEBUG("Sequential receive for RNN cache.");
            for (size_t i = 0; i < sourceNum; i++)
            {
                recvBufferFun(deviceId, i);
            }
        }
        else
        {
            // Parallel receive with controlled concurrency
            auto concurrencyNum = std::min(std::max(static_cast<size_t>(1), bufferCoverSourceNum), sourceNum);
            auto remainRecvNum = sourceNum;

            while (remainRecvNum > 0)
            {
                auto recvConcurrencyNum = std::min(remainRecvNum, concurrencyNum);

                // Avoid leaving a tiny remainder
                if (remainRecvNum > concurrencyNum && remainRecvNum < (2 * concurrencyNum))
                {
                    recvConcurrencyNum = remainRecvNum - concurrencyNum;
                }

                std::vector<std::future<void>> futures;
                futures.reserve(recvConcurrencyNum);
                for (size_t i = 0; i < recvConcurrencyNum; i++)
                {
                    size_t idx = i + (sourceNum - remainRecvNum);
                    TLLM_CHECK(idx < sourceNum);
                    futures.push_back(std::async(std::launch::async, recvBufferFun, deviceId, idx));
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

    // Unpack received buffers into RNN states
    std::vector<runtime::ITensor::SharedPtr> outputConvBlocks;
    std::vector<runtime::ITensor::SharedPtr> outputSsmBlocks;

    auto const maxBatchSize = mRnnStateManager->getMaxBatchSize();
    auto convStates = mRnnStateManager->getConvStates(); // [numLocalLayers, maxBatchSize, convDim, dConv-1]
    auto ssmStates = mRnnStateManager->getSsmStates();   // [numLocalLayers, maxBatchSize, numHeads, headDim, dState]

    outputConvBlocks.push_back(convStates);
    outputSsmBlocks.push_back(ssmStates);

    tensorrt_llm::executor::rnn_cache::concatRnnConvStateDispatch(
        recvBuffers, outputConvBlocks, slotIdx, maxBatchSize, destConfig, selfConfig, selfIdx, bufferManager);

    tensorrt_llm::executor::rnn_cache::concatRnnSsmStateDispatch(recvBuffers, outputSsmBlocks, slotIdx, maxBatchSize,
        sourceConvBytesPerLayer, destConfig, selfConfig, selfIdx, bufferManager);

    bufferManager.getStream().synchronize();

    if (cacheBufferId.has_value())
    {
        mRnnCacheTransBufferManager->freeBufferIndexForRecv(cacheBufferId);
    }
    session.setTime(TransferSession::kTimePostprocess);

    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "End receiving RNN state for request ID: %ld.", llmRequest.mRequestId);
}

void RnnCacheFormatter::formatUnifiedPoolMode(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_formatUnifiedPool);
    session.setTime(TransferSession::kTimeFormatter);

    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "Start sending unified pool RNN state for request ID: %ld.",
        llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    if (!cache_formatter_utils::needSendCache(selfConfig, destConfig, selfIdx, targetInfo))
    {
        return;
    }

    auto rnnSendConns = cache_formatter_utils::pickSendConnections(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks(), targetInfo);
    if (rnnSendConns.empty())
    {
        TLLM_LOG_DEBUG("No targets to send unified pool RNN state to for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    // Get block range for sending (same as KV formatter would).
    auto& blockManager = mKvCacheManager->getBlockManager();
    auto const& lastBlockKey = session.getLastBlockKey();
    auto const ppSize = selfConfig.getParallelConfig().mPipelineParallelism;
    bool const recvSideHasCP = destConfig.getParallelConfig().mContextParallelism > 1;
    auto const indexFromEnd = session.getIndexFromEnd();
    auto blockRange = kv_cache_manager::getBlockRangeForSending(
        mKvCacheManager, llmRequest, lastBlockKey, indexFromEnd, recvSideHasCP, ppSize);

    auto const& blockIdsPerWindow = blockRange.getBlockIdsPerWindow();
    auto const allWindowSizes = blockRange.getWindowSizes();

    for (auto const& ws : allWindowSizes)
    {
        if (!kv_cache_manager::LinearAttentionMetadata::hasRecurrentStatesCache(ws))
        {
            continue;
        }
        auto it = blockIdsPerWindow.find(ws);
        if (it == blockIdsPerWindow.end() || it->second.empty())
        {
            continue;
        }

        // Find pool for this window size.
        runtime::ITensor::SharedPtr pool;
        auto const totalPools = blockManager.getNumPools(false, false);
        for (SizeType32 poolIdx = 0; poolIdx < totalPools; ++poolIdx)
        {
            if (blockManager.getPoolWindowSize(poolIdx) == ws)
            {
                pool = blockManager.getPrimaryPool(poolIdx);
                break;
            }
        }
        TLLM_CHECK_WITH_INFO(pool != nullptr, "Could not find pool for recurrent state window");

        {
            // Unified path: always use split kernels (works for both TP-match and TP-mismatch).
            auto const& rnnState = selfConfig.getRnnCacheState();
            auto const& rnnModel = rnnState.mModelConfig;
            auto const selfTPNum = selfConfig.getParallelConfig().mTensorParallelism;
            auto const selfDPSize = selfConfig.getParallelConfig().mDPsize;
            auto const selfTPPerDP
                = selfConfig.getParallelConfig().mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;

            // Derive blockSizeBytes from the pool tensor itself (ground truth).
            // Pool shape: {numLayers, numBlocks, kvFactor, blockSize} with kvFactor=1 for recurrent states.
            // Per-block byte size = kvFactor * blockSize * elementSize.
            size_t const blockSizeBytes
                = pool->getSizeInBytes() / (static_cast<size_t>(pool->getShape().d[0]) * pool->getShape().d[1]);
            auto const& linearMeta = blockManager.getLinearAttentionMetadata();
            TLLM_CHECK(linearMeta.has_value());
            size_t const ssmBytes = linearMeta->rnnSsmBytes;

            int const numHeadsLocal = rnnModel.mNumHeads / selfTPPerDP;
            auto const globalSectionDims = rnnModel.getConvSectionDims();

            // Collect real block indices
            std::vector<SizeType32> realBlockIndices;
            for (auto const& blockId : it->second)
            {
                auto const& block = blockManager.getBlockById(blockId, ws);
                if (!block->isPlaceholder())
                {
                    realBlockIndices.push_back(static_cast<SizeType32>(block->getMemoryPoolBlockIndex()));
                }
            }

            if (realBlockIndices.empty())
            {
                continue;
            }

            auto const numTargets = targetInfo.mIRanks.size() / targetInfo.mPeerDupHeadFactor;

            // Allocate output buffers via mRnnCacheTransBufferManager (pre-allocated, no conflict with model forward).
            int const headNumDomainTP = numHeadsLocal / (targetInfo.mDomainTPSize / targetInfo.mPeerDupHeadFactor);
            int convDimDomainTPTotal = 0;
            for (int s = 0; s < executor::kv_cache::CacheState::RnnModelConfig::kNumConvSections; ++s)
            {
                int sectionLocal = globalSectionDims[s] / selfTPPerDP;
                convDimDomainTPTotal += sectionLocal / (targetInfo.mDomainTPSize / targetInfo.mPeerDupHeadFactor);
            }

            // Compute per-target buffer sizes (SSM + conv combined per target).
            size_t const bufferTargetNum = numTargets;
            std::vector<size_t> bufferSizesPerTarget(bufferTargetNum);
            for (size_t t = 0; t < bufferTargetNum; ++t)
            {
                SizeType32 layersForTarget = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(t));
                size_t ssmBufBytes = realBlockIndices.size() * layersForTarget * headNumDomainTP * rnnModel.mHeadDim
                    * rnnModel.mDState * common::getDTypeSize(rnnState.mSsmStateDataType);
                size_t convBufBytes = realBlockIndices.size() * layersForTarget * convDimDomainTPTotal
                    * (rnnModel.mDConv - 1) * common::getDTypeSize(rnnState.mConvStateDataType);
                bufferSizesPerTarget[t] = ssmBufBytes + convBufBytes;
            }

            auto cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForSend();
            auto allocationResult = mRnnCacheTransBufferManager->getOrAllocateSendBuffers(
                cacheBufferId, static_cast<int>(bufferTargetNum), bufferSizesPerTarget, bufferManager);
            auto& outputBuffers = std::get<0>(allocationResult);
            auto& bufferCoverTargetNum = std::get<1>(allocationResult);

            // Split each outputBuffer into SSM and conv portions.
            std::vector<runtime::ITensor::SharedPtr> ssmOutputBuffers(numTargets);
            std::vector<runtime::ITensor::SharedPtr> convOutputBuffers(numTargets);
            for (size_t t = 0; t < numTargets; ++t)
            {
                SizeType32 layersForTarget = targetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(t));
                size_t ssmBufBytes = realBlockIndices.size() * layersForTarget * headNumDomainTP * rnnModel.mHeadDim
                    * rnnModel.mDState * common::getDTypeSize(rnnState.mSsmStateDataType);
                size_t convBufBytes = realBlockIndices.size() * layersForTarget * convDimDomainTPTotal
                    * (rnnModel.mDConv - 1) * common::getDTypeSize(rnnState.mConvStateDataType);
                ssmOutputBuffers[t] = runtime::ITensor::slice(outputBuffers[t], 0, ssmBufBytes);
                convOutputBuffers[t] = runtime::ITensor::slice(
                    outputBuffers[t], static_cast<runtime::ITensor::DimType64>(ssmBufBytes), convBufBytes);
            }

            // Run split kernels
            executor::rnn_cache::splitUnifiedPoolSsmDispatch(pool, realBlockIndices, ssmOutputBuffers, destConfig,
                selfConfig, selfIdx, ssmBytes, blockSizeBytes, rnnState.mSsmStateDataType, bufferManager);
            executor::rnn_cache::splitUnifiedPoolConvDispatch(pool, realBlockIndices, convOutputBuffers, destConfig,
                selfConfig, selfIdx, ssmBytes, blockSizeBytes, rnnState.mConvStateDataType, bufferManager);

            bufferManager.getStream().synchronize();
            session.setTime(TransferSession::kTimePreprocess);

            // Send (same protocol as slot mode).
            int deviceId;
            TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
            auto preAllocSendBuffer = mRnnCacheTransBufferManager->getSendBuffer(cacheBufferId);
            sendAllBuffers(session, deviceId, outputBuffers, bufferCoverTargetNum, preAllocSendBuffer, bufferManager,
                targetInfo, rnnSendConns);

            session.setTime(TransferSession::kTimeTransmissions);
            mRnnCacheTransBufferManager->freeBufferIndexForSend(cacheBufferId);
        }
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End sending unified pool RNN state for request ID: %ld.",
        llmRequest.mRequestId);
}

void RnnCacheFormatter::unformatUnifiedPoolMode(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_unformatUnifiedPool);
    session.setTime(TransferSession::kTimeFormatter);

    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "Start receiving unified pool RNN state for request ID: %ld.",
        llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getCacheState().value();
    auto const& destConfig = session.getOtherState().getCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    if (!selfConfig.hasRnnConfig() || !destConfig.hasRnnConfig())
    {
        return;
    }

    // Compute RNN-specific recv connections.
    auto rnnTargetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    auto rnnRecvResult = cache_formatter_utils::pickRecvConnections(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks(), rnnTargetInfo);
    auto rnnRecvConns = std::get<0>(rnnRecvResult);

    if (rnnRecvConns.empty())
    {
        TLLM_LOG_DEBUG("No sources to receive unified pool RNN state from for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    // Get block range for receiving.
    auto& blockManager = mKvCacheManager->getBlockManager();
    auto const srcPpSize = destConfig.getParallelConfig().mPipelineParallelism;
    bool const recvSideHasCP = selfConfig.getParallelConfig().mContextParallelism > 1;
    auto blockRange = kv_cache_manager::getBlockRangeForReceiving(mKvCacheManager, llmRequest,
        destConfig.getEnableBlockReuse(), destConfig.getEnablePartialReuse(), recvSideHasCP, srcPpSize);

    auto const& blockIdsPerWindow = blockRange.getBlockIdsPerWindow();
    auto const allWindowSizes = blockRange.getWindowSizes();

    for (auto const& ws : allWindowSizes)
    {
        if (!kv_cache_manager::LinearAttentionMetadata::hasRecurrentStatesCache(ws))
        {
            continue;
        }
        auto it = blockIdsPerWindow.find(ws);
        if (it == blockIdsPerWindow.end() || it->second.empty())
        {
            continue;
        }

        // Find pool for this window size.
        runtime::ITensor::SharedPtr pool;
        auto const totalPools = blockManager.getNumPools(false, false);
        for (SizeType32 poolIdx = 0; poolIdx < totalPools; ++poolIdx)
        {
            if (blockManager.getPoolWindowSize(poolIdx) == ws)
            {
                pool = blockManager.getPrimaryPool(poolIdx);
                break;
            }
        }
        TLLM_CHECK_WITH_INFO(pool != nullptr, "Could not find pool for recurrent state window");

        {
            // Unified path: always use concat kernels (works for both TP-match and TP-mismatch).
            auto const& rnnState = selfConfig.getRnnCacheState();
            auto const& rnnModel = rnnState.mModelConfig;
            auto const selfTPNum = selfConfig.getParallelConfig().mTensorParallelism;
            auto const selfDPSize = selfConfig.getParallelConfig().mDPsize;
            auto const selfTPPerDP
                = selfConfig.getParallelConfig().mEnableAttentionDP ? selfTPNum / selfDPSize : selfTPNum;

            // Derive blockSizeBytes from the pool tensor itself (ground truth).
            // Pool shape: {numLayers, numBlocks, kvFactor, blockSize} with kvFactor=1 for recurrent states.
            // Per-block byte size = totalPoolBytes / (numLayers * numBlocks).
            size_t const blockSizeBytes
                = pool->getSizeInBytes() / (static_cast<size_t>(pool->getShape().d[0]) * pool->getShape().d[1]);
            auto const& linearMeta = blockManager.getLinearAttentionMetadata();
            TLLM_CHECK(linearMeta.has_value());
            size_t const ssmBytes = linearMeta->rnnSsmBytes;

            int const numHeadsLocal = rnnModel.mNumHeads / selfTPPerDP;
            auto const globalSectionDims = rnnModel.getConvSectionDims();

            // Collect real block indices
            std::vector<SizeType32> realBlockIndices;
            for (auto const& blockId : it->second)
            {
                auto const& block = blockManager.getBlockById(blockId, ws);
                if (!block->isPlaceholder())
                {
                    realBlockIndices.push_back(static_cast<SizeType32>(block->getMemoryPoolBlockIndex()));
                }
            }

            if (realBlockIndices.empty())
            {
                continue;
            }

            auto const numSources = rnnTargetInfo.mIRanks.size() / rnnTargetInfo.mPeerDupHeadFactor;

            // Compute per-source buffer sizes
            int const headNumDomainTP
                = numHeadsLocal / (rnnTargetInfo.mDomainTPSize / rnnTargetInfo.mPeerDupHeadFactor);
            int convDimDomainTPTotal = 0;
            for (int s = 0; s < executor::kv_cache::CacheState::RnnModelConfig::kNumConvSections; ++s)
            {
                int sectionLocal = globalSectionDims[s] / selfTPPerDP;
                convDimDomainTPTotal += sectionLocal / (rnnTargetInfo.mDomainTPSize / rnnTargetInfo.mPeerDupHeadFactor);
            }

            // Allocate receive buffers via mRnnCacheTransBufferManager (pre-allocated).
            size_t const sourceNum = numSources;
            std::vector<size_t> bufferSizesPerSource(sourceNum);
            for (size_t t = 0; t < sourceNum; ++t)
            {
                SizeType32 layersFromSource = rnnTargetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(t));
                size_t ssmBufBytes = realBlockIndices.size() * layersFromSource * headNumDomainTP * rnnModel.mHeadDim
                    * rnnModel.mDState * common::getDTypeSize(rnnState.mSsmStateDataType);
                size_t convBufBytes = realBlockIndices.size() * layersFromSource * convDimDomainTPTotal
                    * (rnnModel.mDConv - 1) * common::getDTypeSize(rnnState.mConvStateDataType);
                bufferSizesPerSource[t] = ssmBufBytes + convBufBytes;
            }

            // Use pre-assigned buffer ID from NIXL connection if available (same as slot mode).
            std::optional<int> cacheBufferId = std::nullopt;
            auto preAssignedRnnId = session.getPreAssignedBufferId(static_cast<uint8_t>(BufferKind::kRNN));
            if (preAssignedRnnId.has_value())
            {
                cacheBufferId = static_cast<int>(*preAssignedRnnId);
            }
            else
            {
                cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForRecv();
            }

            auto allocationResult = mRnnCacheTransBufferManager->getOrAllocateRecvBuffers(
                cacheBufferId, static_cast<int>(sourceNum), bufferSizesPerSource, bufferManager);
            auto& recvBuffers = std::get<0>(allocationResult);

            // Receive from pre-allocated buffers via session.recv() (same as opaque blob protocol).
            size_t totalBytesRecv = 0;
            for (size_t t = 0; t < sourceNum; ++t)
            {
                TLLM_CHECK_WITH_INFO(t < rnnRecvConns.size(),
                    "unformatUnifiedPool: source index %zu >= rnnRecvConns size %zu", t, rnnRecvConns.size());
                size_t connIdx = rnnRecvConns[t];
                // recvBuffers are from pre-allocated pool (cudaMalloc, NIXL-registered).
                totalBytesRecv += recvBuffers[t]->getSizeInBytes();
                session.recv(connIdx, recvBuffers[t]->data(), recvBuffers[t]->getSizeInBytes());
            }

            // Split each recvBuffer into SSM and conv portions for concat kernels.
            std::vector<runtime::ITensor::SharedPtr> ssmInputBuffers(numSources);
            std::vector<runtime::ITensor::SharedPtr> convInputBuffers(numSources);
            for (size_t t = 0; t < numSources; ++t)
            {
                SizeType32 layersFromSource = rnnTargetInfo.getPeerPPDomainLayerNum(static_cast<SizeType32>(t));
                size_t ssmBufBytes = realBlockIndices.size() * layersFromSource * headNumDomainTP * rnnModel.mHeadDim
                    * rnnModel.mDState * common::getDTypeSize(rnnState.mSsmStateDataType);
                size_t convBufBytes = realBlockIndices.size() * layersFromSource * convDimDomainTPTotal
                    * (rnnModel.mDConv - 1) * common::getDTypeSize(rnnState.mConvStateDataType);
                ssmInputBuffers[t] = runtime::ITensor::slice(recvBuffers[t], 0, ssmBufBytes);
                convInputBuffers[t] = runtime::ITensor::slice(
                    recvBuffers[t], static_cast<runtime::ITensor::DimType64>(ssmBufBytes), convBufBytes);
            }

            TLLM_LOG_DEBUG(
                "RnnCacheFormatter::unformatUnifiedPool: blockSizeBytes=%zu, ssmBytes=%zu, numSources=%zu, "
                "realBlocks=%zu, totalBytesRecv=%zu, requestId=%lu",
                blockSizeBytes, ssmBytes, numSources, realBlockIndices.size(), totalBytesRecv, llmRequest.mRequestId);

            // Concat received buffers into the pool
            executor::rnn_cache::concatUnifiedPoolSsmDispatch(pool, realBlockIndices, ssmInputBuffers, destConfig,
                selfConfig, selfIdx, ssmBytes, blockSizeBytes, rnnState.mSsmStateDataType, bufferManager);
            executor::rnn_cache::concatUnifiedPoolConvDispatch(pool, realBlockIndices, convInputBuffers, destConfig,
                selfConfig, selfIdx, ssmBytes, blockSizeBytes, rnnState.mConvStateDataType, bufferManager);

            bufferManager.getStream().synchronize();

            mRnnCacheTransBufferManager->freeBufferIndexForRecv(cacheBufferId);
        }
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "End receiving unified pool RNN state for request ID: %ld.",
        llmRequest.mRequestId);
}

bool RnnCacheFormatter::inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const
{
    if (!selfConfig.hasRnnConfig() || !destConfig.hasRnnConfig())
    {
        return false;
    }

    if (selfConfig.getConvStateDataType() != destConfig.getConvStateDataType())
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: conv state data type mismatch (self=%d, dest=%d)",
            static_cast<int>(selfConfig.getConvStateDataType()), static_cast<int>(destConfig.getConvStateDataType()));
        return false;
    }

    if (selfConfig.getSsmStateDataType() != destConfig.getSsmStateDataType())
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: SSM state data type mismatch (self=%d, dest=%d)",
            static_cast<int>(selfConfig.getSsmStateDataType()), static_cast<int>(destConfig.getSsmStateDataType()));
        return false;
    }

    auto const& selfModel = selfConfig.getRnnModelConfig();
    auto const& destModel = destConfig.getRnnModelConfig();

    if (selfModel.mDState != destModel.mDState || selfModel.mHeadDim != destModel.mHeadDim
        || selfModel.mDConv != destModel.mDConv || selfModel.mNGroups != destModel.mNGroups
        || selfModel.mNumLayers != destModel.mNumLayers)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: model config mismatch");
        return false;
    }

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const& destParallel = destConfig.getParallelConfig();

    if (selfParallel.mContextParallelism != 1 || destParallel.mContextParallelism != 1)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: RNN only supports CP=1 (selfCP=%d, destCP=%d)",
            selfParallel.mContextParallelism, destParallel.mContextParallelism);
        return false;
    }

    return true;
}

std::vector<RnnCacheFormatter::SizeType32> RnnCacheFormatter::getCounterparts(
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
{
    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    return targetInfo.mIRanks;
}

std::pair<std::vector<size_t>, std::vector<size_t>> RnnCacheFormatter::pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks) const
{
    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    return cache_formatter_utils::pickRecvConnections(
        numConnections, selfConfig, selfIdx, destConfig, counterPartRanks, targetInfo);
}

} // namespace tensorrt_llm::batch_manager
