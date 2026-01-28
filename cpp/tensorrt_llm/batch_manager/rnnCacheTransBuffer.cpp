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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

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
