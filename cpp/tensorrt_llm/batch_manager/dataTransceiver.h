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
#include <fstream>
#include <future>
#include <map>
#include <string>
#include <vector>

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class BaseCacheFormatter;
}

using BaseCacheFormatter = kv_cache_manager::BaseCacheFormatter;

// TODO: unify the following class into a namespace like tensorrt_llm::transmission
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using Connection = tensorrt_llm::executor::kv_cache::Connection;
using ConnectionManager = tensorrt_llm::executor::kv_cache::ConnectionManager;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

class TransferSession
{
public:
    struct Measure
    {
        double delay;     // from last token (ctx) or arrival time (gen), in ms
        double duration;  // in ms
        double bandwidth; // in Gbps
    };

    TransferSession(std::vector<Connection const*> connections, DataContext dataContext,
        executor::DataTransceiverState const& selfState, executor::DataTransceiverState otherState,
        runtime::BufferManager const& bufferManager, LlmRequest const* llmRequest = nullptr, bool recordMeasure = false)
        : mConnections(std::move(connections))
        , mDataContext(dataContext)
        , mSelfState(&selfState)
        , mOtherState(std::move(otherState))
        , mBufferManager(&bufferManager)
        , mRequest(llmRequest)
        , mRecordMeasure(recordMeasure)
    {
        TLLM_CHECK(!mConnections.empty());
    }

    [[nodiscard]] std::vector<Connection const*> const& getConnections() const;

    // should be called only during the initialization of the TransferSession
    void setConnection(size_t idx, Connection const* conn);

    [[nodiscard]] DataContext const& getDataContext() const;

    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const;

    [[nodiscard]] executor::DataTransceiverState const& getOtherState() const;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const;

    void send(size_t idx, void const* data, size_t size);

    void recv(size_t idx, void* data, size_t size);

    [[nodiscard]] LlmRequest const& getLlmRequest() const;

    // in CacheSender, the LlmRequest is not available until the sendSync is called
    void setLlmRequest(LlmRequest const& llmRequest);

    void appendMeasure(double delay, double duration, size_t size);

    // TODO: 1. use global id instead of context request id; 2. export to llm metrics instead of file
    void exportMeasure(std::ofstream& outFile, bool isContext) const;

private:
    std::vector<Connection const*> mConnections;
    DataContext mDataContext;
    executor::DataTransceiverState const* mSelfState; // stored in CacheReceiver/CacheSender
    executor::DataTransceiverState mOtherState;
    runtime::BufferManager const* mBufferManager;
    LlmRequest const* mRequest;
    std::vector<Measure> mMeasures;
    bool mRecordMeasure{false};
};

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

// Used to store the information that needs to be sent to the context executor to ensure the generation
// executor smoothly receives the data.
class RequestInfo
{
public:
    /// @brief Constructor.
    /// @param requestId The ID used in the context phase of the current request.
    /// @param transState The state of the data transceiver.
    RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState);

    RequestInfo(LlmRequest::RequestIdType requestId, std::vector<size_t> blockHashes,
        executor::DataTransceiverState transState);
    RequestInfo() = default;

    /// @brief Equality comparison operator.
    /// @param rhs The right operand of the operator.
    [[nodiscard]] bool operator==(RequestInfo const& rhs) const;

    /// @brief Return the ID used in the context phase of the current request.
    /// @return The request ID.
    [[nodiscard]] LlmRequest::RequestIdType getRequestId() const noexcept;

    [[nodiscard]] std::vector<size_t> const& getBlockHashes() const noexcept;

    /// @brief Return the state of the data transceiver.
    /// @return The state of the data transceiver.
    [[nodiscard]] executor::DataTransceiverState const& getTransState() const noexcept;

    /// @brief Serialization.
    /// @param requestInfo Request information to be serialized.
    /// @param os The output stream to which the serialization result points.
    static void serialize(RequestInfo const& requestInfo, std::ostream& os);

    /// @brief Deserialization.
    /// @return The request information obtained from deserialization.
    [[nodiscard]] static RequestInfo deserialize(std::istream& is);

    /// @brief The number of bytes occupied by the serialized data structure.
    /// @param requestInfo Request information to be serialized.
    /// @return The number of bytes.
    [[nodiscard]] static std::size_t serializedSize(RequestInfo const& requestInfo);

private:
    // The ID used in the context phase of the current request.
    LlmRequest::RequestIdType mRequestId;

    std::vector<size_t> mBlockHashes;

    // The state of the data transceiver.
    executor::DataTransceiverState mTransState;
};

class CacheSender
{
public:
    /// @brief Constructor.
    CacheSender(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Asynchronously respond to the request and send data.
    /// @param llmRequest Request object. Its data should be ready when called, and the data for this request
    /// should remain valid until future synchronization.
    /// @return Once the data is fully sent, the future object will become valid.
    [[nodiscard]] std::future<void> sendAsync(LlmRequest& llmRequest) const;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    /// @brief Reset the internal communicator status.
    /// @param commState The communicator status.
    void setCommState(executor::kv_cache::CommState commState);

    /// @brief Receive the request information.
    /// @return The request information.
    [[nodiscard]] RequestInfo recvRequestInfo();

    /// @brief Synchronously send data.
    /// @param llmRequest The request object to which the data belongs.
    void sendSync(LlmRequest const& llmRequest);

    /// @brief Send the request information.
    /// @param llmRequest The request object to which the information belongs.
    [[nodiscard]] TransferSession sendRequestInfo(LlmRequest const& llmRequest);

    /// @brief Synchronously receive data.
    /// @param session The transfer session.
    void receiveSync(TransferSession& session);

    /// @brief Destructor.
    ~CacheSender();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class CacheReceiver
{
public:
    /// @brief Constructor.
    CacheReceiver(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Asynchronously send a request to receive data.
    /// @param llmRequest Request object. Its data should be in an allocated but unwritten state when called, and the
    /// data for this request should remain intact only after future synchronization.
    /// @return Once the data is fully received, the future object will become valid.
    [[nodiscard]] std::future<void> receiveAsync(LlmRequest& llmRequest) const;

    TransferSession sendRequestInfo(LlmRequest const& llmRequest);

    void receiveSync(TransferSession& session);
    /// @brief Destructor.
    ~CacheReceiver();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace tensorrt_llm::batch_manager
