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

#include "dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/logger.h"
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

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class MLACacheFormatter final : public IOFormatter
{
public:
    using CacheState = executor::kv_cache::CacheState;

    MLACacheFormatter(BaseKVCacheManager* cacheManager)
        : mCacheManager{cacheManager}
    {
        TLLM_CHECK(mCacheManager);
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

    static bool needSendCache(CacheState const& selfConfig, CacheState const& destConfig, runtime::SizeType32 selfIdx);
    static std::vector<executor::kv_cache::Connection const*> pickRecvConnections(
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig);

private:
    BaseKVCacheManager* mCacheManager{};

    struct ConcurrenceSendResource
    {
        std::unordered_map<int, runtime::ITensor::SharedPtr> mSendbuffers;
        std::mutex mSendbuffersMutex;
        std::condition_variable mSendbuffersCV;
        std::atomic_int mConcurrence = 0;
    };

    ConcurrenceSendResource mConcurrenceSendResource;

    std::unordered_map<std::string, runtime::ITensor::SharedPtr> mProcessToRecvBuffer;
    std::mutex mProcessToRecvBufferMutex;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
