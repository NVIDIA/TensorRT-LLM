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

#pragma once

#include "cacheTransBuffer.h"
#include "dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/cache_transmission/cacheConcatenate.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntimeBase.h>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class TransferHelper
{
public:
    static void sendBuffer(
        executor::kv_cache::Connection const& connection, runtime::IBuffer const& buf, uint64_t requestId)
    {
        int const tag = ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
        connection.send(executor::kv_cache::DataContext{tag}, buf.data(), buf.getSizeInBytes());
    }

    static void recvBuffer(executor::kv_cache::Connection const& connection, runtime::IBuffer& buf, uint64_t requestId)
    {
        int const tag = ((requestId & 0xFFF) << 8) | (kDATA_TAG & 0xFF);
        connection.recv(executor::kv_cache::DataContext{tag}, buf.data(), buf.getSizeInBytes());
    }

private:
    static constexpr int32_t kDATA_TAG{43};
};

BlockRange getBlockRangeForSending(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest);

BlockRange getBlockRangeForReceiving(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest);

// Used to support the cache transmission with different layouts and different protocols.
class BaseCacheFormatter
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheState = executor::kv_cache::CacheState;

    virtual void formatOutput(LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
        = 0;

    virtual void formatInput(LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager)
        = 0;

    /// @brief Determine whether the sender is applicable to the source and target.
    /// @param selfConfig Source data arrangement.
    /// @param destConfig Target data arrangement.
    /// @return Whether the sender is applicable to the source and target.
    [[nodiscard]] virtual bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const = 0;

    /// @brief Obtain the indies of the counterparts that need to be actually communicated with.
    /// @param selfConfig Source data arrangement.
    /// @param selfIdx The sequential index of the current executor process within the entire parallel group.
    /// @param destConfig Target data arrangement.
    /// @return The indies of the counterparts.
    [[nodiscard]] virtual std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
        = 0;

    [[nodiscard]] virtual BaseKVCacheManager* getCacheManager() const noexcept = 0;

    [[nodiscard]] virtual std::vector<executor::kv_cache::Connection const*> pickRecvConnections(
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig) const
        = 0;

    /// @brief Destructor.
    virtual ~BaseCacheFormatter() = default;
};

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class CacheFormatter final : public BaseCacheFormatter
{
public:
    CacheFormatter(BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager)
        : mCacheManager{cacheManager}
        , mCacheTransBufferManager{cacheTransBufferManager}
    {
        TLLM_CHECK(mCacheManager);
        TLLM_CHECK(mCacheTransBufferManager);
    }

    void formatOutput(LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager) override;

    void formatInput(LlmRequest const& llmRequest,
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, runtime::BufferManager const& bufferManager) override;

    [[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const override;

    [[nodiscard]] std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx).mIRanks;
    }

    [[nodiscard]] BaseKVCacheManager* getCacheManager() const noexcept override
    {
        return mCacheManager;
    }

    static bool needSendCache(CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx);
    std::vector<executor::kv_cache::Connection const*> pickRecvConnections(
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig) const override;

private:
    BaseKVCacheManager* mCacheManager;
    CacheTransBufferManager* mCacheTransBufferManager;
    KvCacheMeasureHelper kvCacheMeasureHelper{common::getEnvKVCacheTransferOutputPath()};
};

std::unique_ptr<BaseCacheFormatter> createCacheFormatter(
    BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager, bool isMLA = false);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
