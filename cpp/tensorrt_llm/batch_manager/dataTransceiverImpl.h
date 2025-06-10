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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

namespace tensorrt_llm::batch_manager
{
struct TransceiverTag
{
    enum class Id : uint64_t
    {
        REQUEST_SEND = 1,
        TERMINATION = 2
    };

    static constexpr int32_t kID_TAG{19};
    static constexpr int32_t kINFO_SIZE_TAG{22};
    static constexpr int32_t kINFO_TAG{32};
};

class DataSenderImpl : public DataSender, public TransceiverTag
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using RequestMapInfo
        = std::vector<std::pair<executor::kv_cache::Connection const*, executor::DataTransceiverState>>;

    DataSenderImpl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter);

    [[nodiscard]] RequestInfo recvRequestInfo() override;

    void sendSync(LlmRequest const& llmRequest) override;

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const override;

    void setCommState(executor::kv_cache::CommState commState) override;

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const override;

    void release(LlmRequest::RequestIdType requestId) override;

private:
    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, RequestMapInfo> mRequestToComms;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<IOFormatter> mFormatter;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
};

class DataReceiverImpl : public DataReceiver, public TransceiverTag
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DataReceiverImpl(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<IOFormatter> formatter);

    void sendRequestInfo(LlmRequest const& llmRequest) override;

    void receiveSync(LlmRequest const& llmRequest) override;

private:
    struct ReceiveCacheResource
    {
        runtime::BufferManager mBufferManager;
        runtime::CudaEvent mCudaEvent;

        ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent&& cudaEvent)
            : mBufferManager(bufferManager)
            , mCudaEvent(std::move(cudaEvent))
        {
        }
    };

    static void sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info);

    [[nodiscard]] std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest);

    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<IOFormatter> mFormatter;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
};

} // namespace tensorrt_llm::batch_manager
