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

#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include <algorithm>

namespace tensorrt_llm::batch_manager
{
using RnnCacheState = executor::rnn_cache::RnnCacheState;

RnnCacheFormatter::RnnCacheFormatter(rnn_state_manager::RnnStateManager* rnnStateManager,
    rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager)
    : mRnnStateManager{rnnStateManager}
    , mRnnCacheTransBufferManager{rnnCacheTransBufferManager}
{
    TLLM_CHECK(mRnnStateManager != nullptr);
    TLLM_CHECK(mRnnCacheTransBufferManager != nullptr);
}

void RnnCacheFormatter::format(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_format);
    session.setTime(TransferSession::kTimeFormatter);

    auto const& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start sending RNN state for request ID: %ld.", llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getRnnCacheState().value();
    auto const& destConfig = session.getOtherState().getRnnCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (!cache_formatter_utils::needSendCache<RnnCacheState>(selfConfig, destConfig, selfIdx))
    {
        return;
    }

    auto pickUpConnections = cache_formatter_utils::pickSendConnections<RnnCacheState>(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks());
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
    auto const& selfLayersPerPP = selfParallel.mRnnLayerNumPerPP;
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
    auto const& modelConfig = selfConfig.getModelConfig();
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

void RnnCacheFormatter::unformat(TransferSession& session)
{
    NVTX3_SCOPED_RANGE(RnnCacheFormatter_unformat);
    session.setTime(TransferSession::kTimeFormatter);

    auto& llmRequest = session.getLlmRequest();
    TLLM_LOG_DEBUG(
        mpi::MpiComm::world().getRank(), "Start receiving RNN state for request ID: %ld.", llmRequest.mRequestId);
    TLLM_CHECK_WITH_INFO(llmRequest.mSamplingConfig.beamWidth == 1, "Currently, only beam width 1 is supported.");

    auto const& connections = session.getConnections();
    auto const& selfConfig = session.getSelfState().getRnnCacheState().value();
    auto const& destConfig = session.getOtherState().getRnnCacheState().value();
    auto const selfIdx = session.getSelfState().getCommState().value().getSelfIdx();
    auto& bufferManager = session.getBufferManager();

    auto sourceInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    int deviceId;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

    auto pickRecvConnResult = cache_formatter_utils::pickRecvConnections<RnnCacheState>(
        connections.size(), selfConfig, selfIdx, destConfig, session.getCounterPartRanks());
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
    auto const& selfLayersPerPP = selfParallel.mRnnLayerNumPerPP;
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
    auto const& modelConfig = selfConfig.getModelConfig();
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
    auto cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForRecv();
    auto allocationResult = mRnnCacheTransBufferManager->getOrAllocateRecvBuffers(
        cacheBufferId, static_cast<int>(sourceNum), bufferSizesPerSource, bufferManager);
    auto& recvBuffers = std::get<0>(allocationResult);
    auto& bufferCoverSourceNumTmp = std::get<1>(allocationResult);
    auto& onlyUseDynamicBuffer = std::get<2>(allocationResult);

    TLLM_CHECK(cacheBufferId.has_value() || onlyUseDynamicBuffer);
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

bool RnnCacheFormatter::inquireSupport(RnnCacheState const& selfConfig, RnnCacheState const& destConfig) const
{
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

    auto const& selfModel = selfConfig.getModelConfig();
    auto const& destModel = destConfig.getModelConfig();

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
    RnnCacheState const& selfConfig, SizeType32 selfIdx, RnnCacheState const& destConfig) const
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    return targetInfo.mIRanks;
}

} // namespace tensorrt_llm::batch_manager
