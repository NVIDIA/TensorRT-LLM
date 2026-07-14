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
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
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

            // Send buffers to targets.
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

void RnnCacheFormatter::unformat(TransferSession& session)
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

            // Use pre-assigned buffer ID from NIXL connection if available.
            std::optional<int> cacheBufferId = std::nullopt;
            auto preAssignedRnnId
                = connections[rnnRecvConns[0]]->getPreAssignedBufferId(static_cast<uint8_t>(BufferKind::kRNN));
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
