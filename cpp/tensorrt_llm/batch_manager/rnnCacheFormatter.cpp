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
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager
{

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

    auto targetInfo = targetIRanks(destConfig, selfConfig, selfIdx);
    auto const targetNum = connections.size();
    if (targetNum == 0)
    {
        TLLM_LOG_WARNING("No targets to send RNN state to for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    auto const slotIdx = mRnnStateManager->getCacheIndex(llmRequest.mRequestId);

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const selfTPNum = selfParallel.mTensorParallelism;
    auto const selfPPRank = selfIdx / selfTPNum;
    auto const& selfLayersPerPP = selfParallel.mRnnLayerNumPerPP;

    // TODO: add zero data copy path

    SizeType32 selfStartLayer = 0; // global layer id
    for (SizeType32 pp = 0; pp < selfPPRank; pp++)
    {
        selfStartLayer += selfLayersPerPP[pp];
    }
    SizeType32 const selfNumLayers = selfLayersPerPP[selfPPRank];

    // Calculate buffer sizes for each target
    //    Each target gets: conv states + ssm states for overlapping layers
    auto const& modelConfig = selfConfig.getModelConfig();
    SizeType32 convDimLocal = modelConfig.mConvDimSize / selfTPNum;
    SizeType32 numHeadsLocal = modelConfig.mNumHeads / selfTPNum;

    size_t convBytesPerLayer
        = convDimLocal * (modelConfig.mDConv - 1) * common::getDTypeSize(selfConfig.getConvStateDataType());
    size_t ssmBytesPerLayer = numHeadsLocal * modelConfig.mHeadDim * modelConfig.mDState
        * common::getDTypeSize(selfConfig.getSsmStateDataType());

    std::vector<size_t> bufferSizesPerTarget(targetNum, 0);

    for (size_t i = 0; i < targetNum; i++)
    {
        SizeType32 layersForTarget = targetInfo.mPeerAttentionLayerNumInDomainPP[i];
        bufferSizesPerTarget[i] = layersForTarget * (convBytesPerLayer + ssmBytesPerLayer);
    }

    auto cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForSend();
    auto [outputBuffers, bufferCoverTargetNum, onlyUseDynamicBuffer]
        = mRnnCacheTransBufferManager->getOrAllocateSendBuffers(
            cacheBufferId, static_cast<int>(targetNum), bufferSizesPerTarget, bufferManager);

    TLLM_CHECK(outputBuffers.size() == targetNum);
    TLLM_CHECK(cacheBufferId.has_value() || onlyUseDynamicBuffer);

    // TODO: This should go in a kernel if perf is slow
    auto stream = bufferManager.getStream().get();
    SizeType32 localLayerOffset = 0;
    for (size_t targetIdx = 0; targetIdx < targetNum; targetIdx++)
    {
        SizeType32 layersToSend = targetInfo.mPeerAttentionLayerNumInDomainPP[targetIdx];

        int8_t* outPtr = static_cast<int8_t*>(outputBuffers[targetIdx]->data());
        size_t byteOffset = 0;

        for (SizeType32 layer = 0; layer < layersToSend; layer++)
        {
            auto convState = mRnnStateManager->getConvStates(selfStartLayer + localLayerOffset + layer);
            auto slotConv = runtime::ITensor::slice(convState, slotIdx, 1);
            size_t numBytes = slotConv->getSizeInBytes();

            TLLM_CUDA_CHECK(
                cudaMemcpyAsync(outPtr + byteOffset, slotConv->data(), numBytes, cudaMemcpyDeviceToDevice, stream));
            byteOffset += numBytes;
        }

        for (SizeType32 layer = 0; layer < layersToSend; layer++)
        {
            auto ssmState = mRnnStateManager->getSsmStates(selfStartLayer + localLayerOffset + layer);
            auto slotSsm = runtime::ITensor::slice(ssmState, slotIdx, 1);
            size_t numBytes = slotSsm->getSizeInBytes();

            TLLM_CUDA_CHECK(
                cudaMemcpyAsync(outPtr + byteOffset, slotSsm->data(), numBytes, cudaMemcpyDeviceToDevice, stream));
            byteOffset += numBytes;
        }
        localLayerOffset += layersToSend;
    }

    bufferManager.getStream().synchronize();
    session.setTime(TransferSession::kTimePreprocess);

    auto preAllocSendBuffer = mRnnCacheTransBufferManager->getSendBuffer(cacheBufferId);
    int deviceId;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

    sendAllBuffers(
        session, deviceId, outputBuffers, bufferCoverTargetNum, preAllocSendBuffer, bufferManager, targetInfo);

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

    auto sourceInfo = targetIRanks(destConfig, selfConfig, selfIdx);
    auto const sourceNum = connections.size();

    if (sourceNum == 0)
    {
        TLLM_LOG_WARNING("No sources to receive RNN state from for request ID: %ld", llmRequest.mRequestId);
        return;
    }

    auto const slotIdx = mRnnStateManager->addRequest(llmRequest.mRequestId);

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const selfTPNum = selfParallel.mTensorParallelism;
    auto const selfPPRank = selfIdx / selfTPNum;
    auto const& selfLayersPerPP = selfParallel.mRnnLayerNumPerPP;

    SizeType32 selfStartLayer = 0;
    for (SizeType32 pp = 0; pp < selfPPRank; pp++)
    {
        selfStartLayer += selfLayersPerPP[pp];
    }

    // Calculate buffer sizes
    auto const& modelConfig = selfConfig.getModelConfig();
    SizeType32 convDimLocal = modelConfig.mConvDimSize / selfTPNum;
    SizeType32 numHeadsLocal = modelConfig.mNumHeads / selfTPNum;

    size_t convBytesPerLayer
        = convDimLocal * (modelConfig.mDConv - 1) * common::getDTypeSize(selfConfig.getConvStateDataType());
    size_t ssmBytesPerLayer = numHeadsLocal * modelConfig.mHeadDim * modelConfig.mDState
        * common::getDTypeSize(selfConfig.getSsmStateDataType());

    std::vector<size_t> bufferSizesPerSource(sourceNum, 0);
    for (size_t i = 0; i < sourceNum; i++)
    {
        SizeType32 layersFromSource = sourceInfo.mPeerAttentionLayerNumInDomainPP[i];
        bufferSizesPerSource[i] = layersFromSource * (convBytesPerLayer + ssmBytesPerLayer);
    }

    // Allocate receive buffers
    size_t remainNoCoverSourceNum = 0;
    size_t bufferCoverSourceNum = 0;
    auto cacheBufferId = mRnnCacheTransBufferManager->assignBufferIndexForRecv();
    auto [recvBuffers, bufferCoverSourceNumTmp, onlyUseDynamicBuffer]
        = mRnnCacheTransBufferManager->getOrAllocateRecvBuffers(
            cacheBufferId, static_cast<int>(sourceNum), bufferSizesPerSource, bufferManager);

    TLLM_CHECK(cacheBufferId.has_value() || onlyUseDynamicBuffer);
    bufferCoverSourceNum = bufferCoverSourceNumTmp;
    remainNoCoverSourceNum = sourceNum > bufferCoverSourceNum ? sourceNum - bufferCoverSourceNum : 0;

    TLLM_CHECK(recvBuffers.size() == sourceNum);
    bufferManager.getStream().synchronize();
    session.setTime(TransferSession::kTimePreprocess);

    // Get pre-allocated buffer for chunked receive
    runtime::ITensor::SharedPtr preAllocRecvBuffer = nullptr;
    if (cacheBufferId.has_value())
    {
        preAllocRecvBuffer = mRnnCacheTransBufferManager->getRecvBuffer(cacheBufferId);
        TLLM_CHECK(preAllocRecvBuffer != nullptr);
    }

    int deviceId;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));

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
            session.recv(srcIdx, buffer->data(), buffer->getSizeInBytes());
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
                session.recv(srcIdx, recvSlice->data(), recvSlice->getSizeInBytes());
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
    {
        NVTX3_SCOPED_RANGE(unpackRnnStates);
        auto stream = bufferManager.getStream().get();
        SizeType32 localLayerOffset = 0;

        for (size_t srcIdx = 0; srcIdx < sourceNum; srcIdx++)
        {
            SizeType32 layersFromSource = sourceInfo.mPeerAttentionLayerNumInDomainPP[srcIdx];

            int8_t const* inPtr = static_cast<int8_t const*>(recvBuffers[srcIdx]->data());
            size_t byteOffset = 0;

            // Unpack conv states first
            for (SizeType32 layer = 0; layer < layersFromSource; layer++)
            {
                auto convState = mRnnStateManager->getConvStates(selfStartLayer + localLayerOffset + layer);
                auto slotConv = runtime::ITensor::slice(convState, slotIdx, 1);
                size_t numBytes = slotConv->getSizeInBytes();

                TLLM_CUDA_CHECK(
                    cudaMemcpyAsync(slotConv->data(), inPtr + byteOffset, numBytes, cudaMemcpyDeviceToDevice, stream));
                byteOffset += numBytes;
            }

            // Unpack SSM states
            for (SizeType32 layer = 0; layer < layersFromSource; layer++)
            {
                auto ssmState = mRnnStateManager->getSsmStates(selfStartLayer + localLayerOffset + layer);
                auto slotSsm = runtime::ITensor::slice(ssmState, slotIdx, 1);
                size_t numBytes = slotSsm->getSizeInBytes();

                TLLM_CUDA_CHECK(
                    cudaMemcpyAsync(slotSsm->data(), inPtr + byteOffset, numBytes, cudaMemcpyDeviceToDevice, stream));
                byteOffset += numBytes;
            }
            localLayerOffset += layersFromSource;
        }

        bufferManager.getStream().synchronize();
    }

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
    // TODO: check for same dimensions across all layers
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

    // Require same TP for now (different TP would require state resharding)
    // TODO
    if (selfParallel.mTensorParallelism != destParallel.mTensorParallelism)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: different TP not yet supported (self=%d, dest=%d)",
            selfParallel.mTensorParallelism, destParallel.mTensorParallelism);
        return false;
    }

    if (common::getEnvDisaggLayerwise())
    {
        TLLM_LOG_WARNING(
            "RnnCacheFormatter::inquireSupport: layerwise cache transfer not yet supported for RNN. \
                            Will default to non-layerwise transfer.");
    }

    return true;
}

std::vector<RnnCacheFormatter::SizeType32> RnnCacheFormatter::getCounterparts(
    RnnCacheState const& selfConfig, SizeType32 selfIdx, RnnCacheState const& destConfig) const
{
    auto targetInfo = targetIRanks(destConfig, selfConfig, selfIdx);
    return targetInfo.mIRanks;
}

std::vector<size_t> RnnCacheFormatter::pickRecvConnections(
    size_t numConnections, RnnCacheState const& selfConfig, SizeType32 selfIdx, RnnCacheState const& destConfig) const
{
    // no duplication for RNN so all ranks are valid
    auto targetInfo = targetIRanks(destConfig, selfConfig, selfIdx);
    std::vector<size_t> ret(numConnections);
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
}

} // namespace tensorrt_llm::batch_manager
