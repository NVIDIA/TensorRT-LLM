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

#include "rnnCacheTransBuffer.h"
#include "cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

size_t RnnCacheTransBufferManager::computeTransferBufferSize(
    RnnStateManager* rnnStateManager, std::optional<size_t> maxNumTokens)
{
    SizeType32 numLocalLayers = rnnStateManager->getNumLocalLayers();

    // Get the tensor for one layer to determine per-slot dimensions
    // Conv state shape per layer: [maxBatchSize, convDim_local, dConv-1]
    // SSM state shape per layer:  [maxBatchSize, numHeads_local, headDim, dState]
    // The tensors are shaped [maxBatchSize, ...], so one slot = total_size / maxBatchSize
    auto convState = rnnStateManager->getConvStates(rnnStateManager->getGlobalLayerNum(0)); // Get first layer's tensor
    auto ssmState = rnnStateManager->getSsmStates(rnnStateManager->getGlobalLayerNum(0));

    auto convShape = convState->getShape();
    auto ssmShape = ssmState->getShape();

    // Compute elements per slot per layer (divide total volume by batch size)
    size_t convElemsPerSlotPerLayer = runtime::ITensor::volume(convShape) / convShape.d[0];
    size_t ssmElemsPerSlotPerLayer = runtime::ITensor::volume(ssmShape) / ssmShape.d[0];

    size_t convDtypeSize = common::getDTypeSize(rnnStateManager->getConvStateDataType());
    size_t ssmDtypeSize = common::getDTypeSize(rnnStateManager->getSsmStateDataType());

    size_t convBytesPerSlotPerLayer = convElemsPerSlotPerLayer * convDtypeSize;
    size_t ssmBytesPerSlotPerLayer = ssmElemsPerSlotPerLayer * ssmDtypeSize;

    size_t bufferSizePerSlot = numLocalLayers * (convBytesPerSlotPerLayer + ssmBytesPerSlotPerLayer);

    TLLM_LOG_DEBUG(
        "RNN computeTransferBufferSize: numLocalLayers=%d, convBytesPerLayer=%lu, ssmBytesPerLayer=%lu, "
        "totalPerSlot=%lu",
        numLocalLayers, convBytesPerSlotPerLayer, ssmBytesPerSlotPerLayer, bufferSizePerSlot);

    return bufferSizePerSlot > 0 ? bufferSizePerSlot : common::getEnvMemSizeForKVCacheTransferBuffer();
}

RnnCacheTransBufferManager::RnnCacheTransBufferManager(
    RnnStateManager* rnnStateManager, std::optional<size_t> maxNumTokens)
    : BaseTransBufferManager(computeTransferBufferSize(rnnStateManager, maxNumTokens),
        nvinfer1::DataType::kUINT8, // Use byte buffer for mixed dtypes
        maxNumTokens)
    , mRnnStateManager{rnnStateManager}
{
    TLLM_CHECK(mRnnStateManager != nullptr);
    TLLM_LOG_INFO("RnnCacheTransBufferManager created for RNN cache");
}

size_t RnnCacheTransBufferManager::computeTransferBufferSizeFromPool(
    kv_cache_manager::BaseKVCacheManager* kvCacheManager, executor::kv_cache::CacheState const& cacheState,
    std::optional<size_t> maxNumTokens)
{
    TLLM_CHECK(cacheState.hasRnnConfig());
    auto const& rnnCacheState = cacheState.getRnnCacheState();
    auto const& rnnModel = rnnCacheState.mModelConfig;

    int const tpNum = cacheState.getParallelConfig().mTensorParallelism;
    int const dpSize = cacheState.getParallelConfig().mDPsize;
    int const tpPerDP = cacheState.getParallelConfig().mEnableAttentionDP ? tpNum / dpSize : tpNum;

    int const numHeadsLocal = rnnModel.mNumHeads / tpPerDP;

    // Compute conv dim local (accounting for conv section layout)
    int convDimLocal = 0;
    auto const globalSectionDims = rnnModel.getConvSectionDims();
    for (int s = 0; s < executor::kv_cache::CacheState::RnnModelConfig::kNumConvSections; ++s)
    {
        convDimLocal += globalSectionDims[s] / tpPerDP;
    }

    size_t const convDtypeSize = common::getDTypeSize(rnnCacheState.mConvStateDataType);
    size_t const ssmDtypeSize = common::getDTypeSize(rnnCacheState.mSsmStateDataType);

    size_t const convBytesPerLayer = static_cast<size_t>(convDimLocal) * (rnnModel.mDConv - 1) * convDtypeSize;
    size_t const ssmBytesPerLayer
        = static_cast<size_t>(numHeadsLocal) * rnnModel.mHeadDim * rnnModel.mDState * ssmDtypeSize;

    // Use max layer count across PP ranks for buffer sizing (conservative).
    // This ensures the buffer is large enough regardless of which PP rank we are.
    SizeType32 numLocalLayers = 0;
    for (auto layerCount : rnnCacheState.mLayerNumPerPP)
    {
        numLocalLayers = std::max(numLocalLayers, layerCount);
    }

    size_t bufferSizePerBlock = numLocalLayers * (convBytesPerLayer + ssmBytesPerLayer);

    // Compute max real blocks per request for buffer sizing.
    // Real blocks are allocated at: every statesSnapshotInterval tokens + end-of-prompt + saveLastSnapshot.
    auto const& blockManager = kvCacheManager->getBlockManager();
    auto const& linearMeta = blockManager.getLinearAttentionMetadata();
    SizeType32 maxRealBlocksPerSeq = 1; // Default: at least 1 real block (end-of-prompt)
    if (linearMeta.has_value() && linearMeta->statesSnapshotInterval > 0)
    {
        auto const recurrentWs
            = static_cast<SizeType32>(kv_cache_manager::LinearAttentionMetadata::LinearCacheType::kRecurrentStates);
        auto const wsMeta = blockManager.getWindowSizeMetadata(recurrentWs);
        SizeType32 maxTokenNum = wsMeta.maxTokenNum;
        // Number of interval snapshots + end-of-prompt block + saveLastSnapshot.
        maxRealBlocksPerSeq
            = maxTokenNum / linearMeta->statesSnapshotInterval + 1 + (linearMeta->saveLastSnapshot ? 1 : 0);
    }

    size_t bufferSize = static_cast<size_t>(maxRealBlocksPerSeq) * bufferSizePerBlock;

    TLLM_LOG_DEBUG(
        "RNN computeTransferBufferSizeFromPool: numLocalLayers=%d, convBytesPerLayer=%lu, ssmBytesPerLayer=%lu, "
        "bufferSizePerBlock=%lu, maxRealBlocksPerSeq=%d, totalBufferSize=%lu",
        numLocalLayers, convBytesPerLayer, ssmBytesPerLayer, bufferSizePerBlock, maxRealBlocksPerSeq, bufferSize);

    return bufferSize > 0 ? bufferSize : common::getEnvMemSizeForKVCacheTransferBuffer();
}

RnnCacheTransBufferManager::RnnCacheTransBufferManager(kv_cache_manager::BaseKVCacheManager* kvCacheManager,
    executor::kv_cache::CacheState const& cacheState, std::optional<size_t> maxNumTokens)
    : BaseTransBufferManager(computeTransferBufferSizeFromPool(kvCacheManager, cacheState, maxNumTokens),
        nvinfer1::DataType::kUINT8, maxNumTokens)
    , mRnnStateManager{nullptr}
{
    TLLM_CHECK(kvCacheManager != nullptr);
    TLLM_LOG_INFO("RnnCacheTransBufferManager created for unified pool RNN cache");
}

size_t RnnCacheTransBufferManager::preAllocBufferSize(
    size_t rnnStateSizeBytes, std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig)
{
    if (!cacheTransceiverConfig.has_value())
    {
        return 0;
    }
    if (!cacheTransceiverConfig->getBackendType().has_value())
    {
        return 0;
    }

    size_t transferBufferSize
        = rnnStateSizeBytes > 0 ? rnnStateSizeBytes : common::getEnvMemSizeForKVCacheTransferBuffer();

    bool useFabricMemory = kv_cache_manager::FabricMemory::supportFbaricMemory()
        && (!(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer()));

    if (useFabricMemory)
    {
        transferBufferSize = kv_cache_manager::FabricMemory::getAlignedSize(transferBufferSize);
    }

    size_t recvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    size_t sendBufferCount = common::getEnvKVCacheSendMaxConcurrenceNum();

    return transferBufferSize * (recvBufferCount + sendBufferCount);
}

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
