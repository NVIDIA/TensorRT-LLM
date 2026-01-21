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
#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
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
using BlockKey = kv_cache_manager::BlockKey;

// TODO: unify the following class into a namespace like tensorrt_llm::transmission
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using Connection = tensorrt_llm::executor::kv_cache::Connection;
using ConnectionManager = tensorrt_llm::executor::kv_cache::ConnectionManager;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using BlockKey = tensorrt_llm::batch_manager::kv_cache_manager::BlockKey;
using UniqueToken = tensorrt_llm::runtime::UniqueToken;

class TransferSession
{
public:
    // measures for each single transmission
    struct Measure
    {
        LlmRequest::TimePoint start;
        LlmRequest::TimePoint end;
        size_t size = 0;
    };

    enum TimeNames : uint8_t
    {
        kTimeRequestInfo = 0,
        kTimeFormatter,
        kTimePreprocess,
        kTimeTransmissions,
        kTimePostprocess,
        kTimeCounts
    };

    struct KVCacheTimes
    {
        std::array<LlmRequest::TimePoint, kTimeCounts> times;
        std::vector<Measure> measures;
    };

    TransferSession(std::vector<Connection const*> connections, DataContext dataContext,
        executor::DataTransceiverState const& selfState, executor::DataTransceiverState otherState,
        runtime::BufferManager const& bufferManager, int32_t indexFromEnd, BlockKey const& lastBlockKey,
        LlmRequest const* llmRequest = nullptr, bool recordTiming = false)
        : mConnections(std::move(connections))
        , mDataContext(std::move(dataContext))
        , mSelfState(&selfState)
        , mOtherState(std::move(otherState))
        , mBufferManager(&bufferManager)
        , mRequest(llmRequest)
        , mIndexFromEnd(indexFromEnd)
        , mLastBlockKey(lastBlockKey)
    {
        TLLM_CHECK(!mConnections.empty());
        if (recordTiming)
        {
            mTimes = std::make_unique<KVCacheTimes>();
        }
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

    void setTime(TimeNames name);

    void appendMeasure(LlmRequest::TimePoint start, LlmRequest::TimePoint end, size_t size);

    // TODO: 1. use global id instead of context request id; 2. export to llm metrics instead of file
    void exportMeasure(std::ofstream& outFile, bool isContext) const;

    [[nodiscard]] int32_t getIndexFromEnd() const
    {
        return mIndexFromEnd;
    }

    [[nodiscard]] BlockKey const& getLastBlockKey() const
    {
        return mLastBlockKey;
    }

private:
    std::vector<Connection const*> mConnections;
    DataContext mDataContext;
    executor::DataTransceiverState const* mSelfState; // stored in CacheReceiver/CacheSender
    executor::DataTransceiverState mOtherState;
    runtime::BufferManager const* mBufferManager;
    LlmRequest const* mRequest;
    std::unique_ptr<KVCacheTimes> mTimes;
    int32_t mIndexFromEnd{0};
    BlockKey mLastBlockKey{};
};

using UniqueToken = tensorrt_llm::runtime::UniqueToken;
using BlockKey = tensorrt_llm::batch_manager::kv_cache_manager::BlockKey;

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
    static constexpr int32_t kREADY_SIGNAL_TAG{42};
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

    RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState, int32_t indexFromEnd,
        BlockKey const& lastBlockKey);
    RequestInfo() = default;

    /// @brief Equality comparison operator.
    /// @param rhs The right operand of the operator.
    [[nodiscard]] bool operator==(RequestInfo const& rhs) const;

    /// @brief Return the ID used in the context phase of the current request.
    /// @return The request ID.
    [[nodiscard]] LlmRequest::RequestIdType getRequestId() const noexcept;

    [[nodiscard]] int32_t getIndexFromEnd() const noexcept
    {
        return mIndexFromEnd;
    }

    /// @brief Return the state of the data transceiver.
    /// @return The state of the data transceiver.
    [[nodiscard]] executor::DataTransceiverState const& getTransState() const noexcept;

    [[nodiscard]] BlockKey const& getLastBlockKey() const noexcept
    {
        return mLastBlockKey;
    }

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
    // Index from end indicating how many trailing blocks to transfer (index+1)
    int32_t mIndexFromEnd{0};

    // Last block key, used to derive other block keys on receiver
    BlockKey mLastBlockKey{};

    // The state of the data transceiver.
    executor::DataTransceiverState mTransState;
};

/// @brief Base implementation class for cache senders (KV and RNN).
/// Contains common threading infrastructure and queue management.
class BaseCacheSenderImpl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    /// @brief Constructor with common initialization.
    BaseCacheSenderImpl(executor::kv_cache::ConnectionManager* manager, std::unique_ptr<BaseCacheFormatter> formatter,
        SizeType32 selfIndex);

    virtual ~BaseCacheSenderImpl();

    // Non-copyable, non-movable
    BaseCacheSenderImpl(BaseCacheSenderImpl const&) = delete;
    BaseCacheSenderImpl& operator=(BaseCacheSenderImpl const&) = delete;
    BaseCacheSenderImpl(BaseCacheSenderImpl&&) = delete;
    BaseCacheSenderImpl& operator=(BaseCacheSenderImpl&&) = delete;

    /// @brief Asynchronously respond to the request and send data.
    [[nodiscard]] std::future<void> sendAsync(LlmRequest& llmRequest);

    /// @brief Return the internal communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    /// @brief Reset the internal communicator status.
    void setCommState(executor::kv_cache::CommState commState);

    /// @brief Synchronously send data.
    void sendSync(LlmRequest const& llmRequest);

    /// @brief Cancel the request.
    bool cancelRequest(LlmRequest const& llmRequest);

    /// @brief Send ready signal.
    void sendReadySignal(RequestIdType requestId, bool isReady);

    /// @brief Get count of counterpart connections for a request.
    [[nodiscard]] size_t getCounterpartsCount(RequestIdType requestId);

    /// @brief Release resources for a completed request.
    void release(RequestIdType requestId);

    /// @brief Receive request information - pure virtual, implemented by derived classes.
    [[nodiscard]] virtual RequestInfo recvRequestInfo() = 0;

protected:
    /// @brief Accessor for the self state - derived classes own their state.
    [[nodiscard]] virtual executor::DataTransceiverState& getSelfState() = 0;
    [[nodiscard]] virtual executor::DataTransceiverState const& getSelfState() const = 0;

    struct Response
    {
        LlmRequest* mRequest;
        std::promise<void> mPromise;
    };

    struct AsyncSendResource
    {
        std::deque<Response> mSendQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    // Common members accessible to derived classes
    executor::kv_cache::ConnectionManager* mManager;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    runtime::BufferManager mBufferManager;
    std::map<RequestIdType, TransferSession> mRequestToSession;
    std::mutex mMtxForMap;
    std::atomic<bool> mTerminate{false};
    int mDeviceId{-1};

private:
    // Thread management
    void response() noexcept;
    void handleAsyncSend(AsyncSendResource& resource);
    void sendAndRemoveResponse(RequestIdType id, Response resp) noexcept;
    void asyncSendAndRemoveResponse(RequestIdType id, Response resp) noexcept;
    void sendResponse(std::map<RequestIdType, Response>::iterator it);
    void terminate();
    void removeResponse(std::map<RequestIdType, Response>::iterator it);
    [[nodiscard]] RequestIdType getCurrentRequestId() const;
    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse();

    // Thread-related members
    std::optional<RequestIdType> mCurrentRequest;
    std::set<RequestIdType> mCancelledRequests;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false};
    std::condition_variable mSenderCv, mResponderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<RequestIdType, int> mRemainSendCount;
    AsyncSendResource mAsyncSendResource;
    std::vector<std::future<void>> mAsyncSendFutures;
    std::ofstream mMeasuresFile;
};

/// @brief KV cache sender - inherits from BaseCacheSenderImpl with KV-specific logic.
class CacheSender : public BaseCacheSenderImpl
{
public:
    /// @brief Constructor.
    CacheSender(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Receive request information - KV-specific implementation.
    [[nodiscard]] RequestInfo recvRequestInfo() override;

protected:
    [[nodiscard]] executor::DataTransceiverState& getSelfState() override;
    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const override;

private:
    executor::DataTransceiverState mSelfState;
};

/// @brief Base implementation class for cache receivers (KV and RNN).
/// Contains common threading infrastructure and queue management.
class BaseCacheReceiverImpl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;

    /// @brief Constructor with common initialization.
    BaseCacheReceiverImpl(executor::kv_cache::ConnectionManager* manager, std::unique_ptr<BaseCacheFormatter> formatter,
        SizeType32 selfIndex);

    virtual ~BaseCacheReceiverImpl();

    // Non-copyable, non-movable
    BaseCacheReceiverImpl(BaseCacheReceiverImpl const&) = delete;
    BaseCacheReceiverImpl& operator=(BaseCacheReceiverImpl const&) = delete;
    BaseCacheReceiverImpl(BaseCacheReceiverImpl&&) = delete;
    BaseCacheReceiverImpl& operator=(BaseCacheReceiverImpl&&) = delete;

    /// @brief Asynchronously send a request to receive data.
    [[nodiscard]] std::future<void> receiveAsync(LlmRequest& llmRequest);

    /// @brief Synchronously receive data using formatter.
    void receiveSync(TransferSession& session);

    /// @brief Cancel the request.
    bool cancelRequest(LlmRequest const& llmRequest);

    /// @brief Receive ready signal from sender.
    bool receiveReadySignal(TransferSession& session);

    /// @brief Send request information - pure virtual, implemented by derived classes.
    [[nodiscard]] virtual TransferSession sendRequestInfo(LlmRequest const& llmRequest) = 0;

protected:
    /// @brief Accessor for the self state - derived classes own their state.
    [[nodiscard]] virtual executor::DataTransceiverState& getSelfState() = 0;
    [[nodiscard]] virtual executor::DataTransceiverState const& getSelfState() const = 0;

    struct RequestAndPromise
    {
        LlmRequest* mRequest;
        std::unique_ptr<std::promise<void>> mPromise;

        RequestAndPromise();
        RequestAndPromise(LlmRequest* request, std::unique_ptr<std::promise<void>>&& promise);
        RequestAndPromise(RequestAndPromise const&) = delete;
        RequestAndPromise(RequestAndPromise&& other) noexcept;
        RequestAndPromise& operator=(RequestAndPromise&& other) noexcept;
    };

    struct AsyncResource
    {
        std::deque<RequestAndPromise> mRequestsQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    struct ReceiveCacheResource
    {
        runtime::BufferManager mBufferManager;
        runtime::CudaEvent mCudaEvent;

        ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent cudaEvent);
    };

    /// @brief Get or create receive cache resource for a request.
    [[nodiscard]] std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest);

    /// @brief Send request info to a connection (non-agent path).
    void sendRequestInfoToConnection(executor::kv_cache::Connection const* connection, RequestInfo const& info);

    // Common members accessible to derived classes
    executor::kv_cache::ConnectionManager* mManager;
    std::unique_ptr<BaseCacheFormatter> mFormatter;
    runtime::BufferManager mBufferManager;
    std::atomic<bool> mTerminate{false};
    int mDeviceId{-1};

private:
    // Thread management
    void requestSync(LlmRequest& llmRequest);
    [[nodiscard]] std::future<void> requestAndReceiveAsyncMultiThreads(LlmRequest& llmRequest);
    void request(AsyncResource& resource);

    // Thread-related members
    static constexpr char const* kDefaultProcessInfo = "default";
    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
    std::ofstream mMeasuresFile;
    std::mutex mMeasuresFileMutex;
};

/// @brief KV cache receiver - inherits from BaseCacheReceiverImpl with KV-specific logic.
class CacheReceiver : public BaseCacheReceiverImpl
{
public:
    /// @brief Constructor.
    CacheReceiver(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        SizeType32 selfIndex, std::unique_ptr<BaseCacheFormatter> formatter);

    /// @brief Send request information - KV-specific implementation.
    [[nodiscard]] TransferSession sendRequestInfo(LlmRequest const& llmRequest) override;

protected:
    [[nodiscard]] executor::DataTransceiverState& getSelfState() override;
    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const override;

private:
    executor::DataTransceiverState mSelfState;
};

} // namespace tensorrt_llm::batch_manager
