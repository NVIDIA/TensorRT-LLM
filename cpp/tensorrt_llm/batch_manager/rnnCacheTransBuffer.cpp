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
#include "cacheTransBuffer.h" // For FabricMemory
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

size_t RnnCacheTransBufferManager::computeTransferBufferSize(
    RnnStateManager* rnnStateManager, std::optional<size_t> maxNumTokens)
{
    // TODO: Compute based on RNN state sizes when needed
    // For now, use the environment variable default
    return common::getEnvMemSizeForKVCacheTransferBuffer();
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
