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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <future>
#include <map>
#include <string>

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class BaseCacheFormatter;
} // namespace kv_cache_manager

struct TransceiverTag
{
    static constexpr int32_t kID_TAG{19};
    static constexpr int32_t kDATA_TAG{43};
};

// TODO: unify the following class into namespace tensorrt_llm::transmission
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using Connection = tensorrt_llm::executor::kv_cache::Connection;
using ConnectionManager = tensorrt_llm::executor::kv_cache::ConnectionManager;

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

    [[nodiscard]] std::vector<size_t> const& getBlockHashes() const noexcept
    {
        return mBlockHashes;
    }

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

class TransferSession
{
public:
    TransferSession(std::vector<Connection const*> connections, DataContext dataContext,
        executor::DataTransceiverState const& selfState, executor::DataTransceiverState otherState,
        runtime::BufferManager& bufferManager)
        : mConnections(std::move(connections))
        , mDataContext(dataContext)
        , mSelfState(&selfState)
        , mOtherState(std::move(otherState))
        , mBufferManager(&bufferManager)
    {
        TLLM_CHECK(!mConnections.empty());
    }

    [[nodiscard]] std::vector<Connection const*> const& getConnections() const noexcept
    {
        return mConnections;
    }

    // TODO: set up connections at session creation
    [[nodiscard]] std::vector<Connection const*>& getConnectionsMutable() noexcept
    {
        return mConnections;
    }

    [[nodiscard]] DataContext const& getDataContext() const noexcept
    {
        return mDataContext;
    }

    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const noexcept
    {
        return *mSelfState;
    }

    [[nodiscard]] executor::DataTransceiverState const& getOtherState() const noexcept
    {
        return mOtherState;
    }

    [[nodiscard]] runtime::BufferManager& getBufferManager() noexcept
    {
        return *mBufferManager;
    }

    void send(size_t connIdx, void const* data, size_t size)
    {
        mConnections[connIdx]->send(mDataContext, data, size);
    }

    void recv(size_t connIdx, void* data, size_t size)
    {
        mConnections[connIdx]->recv(mDataContext, data, size);
    }

private:
    std::vector<Connection const*> mConnections;
    DataContext mDataContext;
    // self state is stored in DataSender/Receiver and is always valid
    executor::DataTransceiverState const* mSelfState;
    // the other state is temporary and should be copied from the ContextPhaseParams/RequestInfo
    executor::DataTransceiverState mOtherState;
    runtime::BufferManager* mBufferManager;
};

// Operators required for data transmission in specific communication protocols.
class DataSender
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DataSender(ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        std::unique_ptr<kv_cache_manager::BaseCacheFormatter> formatter);

    /// @brief Receive the request information.
    /// @return The request information.
    [[nodiscard]] RequestInfo recvRequestInfo();

    /// @brief Synchronously send data.
    /// @param llmRequest The request object to which the data belongs.
    void sendSync(LlmRequest const& llmRequest);

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const;

    void release(LlmRequest::RequestIdType requestId);

    ~DataSender(); // = default;

private:
    ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<kv_cache_manager::BaseCacheFormatter> mFormatter;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
};

// Operators required for data transmission in specific communication protocols.
class DataReceiver
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DataReceiver(executor::kv_cache::ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        std::unique_ptr<kv_cache_manager::BaseCacheFormatter> formatter);

    /// @brief Send the request information.
    /// @param llmRequest The request object to which the information belongs.
    void sendRequestInfo(LlmRequest const& llmRequest);

    /// @brief Synchronously receive data.
    /// @param llmRequest The request object to which the data belongs.
    void receiveSync(LlmRequest const& llmRequest);

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

    [[nodiscard]] std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest);

    executor::kv_cache::ConnectionManager* mManager;
    executor::DataTransceiverState mSelfState;
    std::unique_ptr<kv_cache_manager::BaseCacheFormatter> mFormatter;
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
};

class DataResponder
{
public:
    /// @brief Constructor.
    /// @param sender The sender used at the underlying level.
    explicit DataResponder(ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        std::unique_ptr<kv_cache_manager::BaseCacheFormatter> formatter);

    /// @brief Asynchronously respond to the request and send data.
    /// @param llmRequest Request object. Its data should be ready when called, and the data for this request
    /// should remain valid until future synchronization.
    /// @return Once the data is fully sent, the future object will become valid.
    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest& llmRequest) const;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    /// @brief Destructor.
    ~DataResponder();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class DataRequester
{
public:
    /// @brief Constructor.
    /// @param receiver The receiver used at the underlying level.
    explicit DataRequester(ConnectionManager* manager, executor::kv_cache::CacheState selfCacheState,
        std::unique_ptr<kv_cache_manager::BaseCacheFormatter> formatter);

    /// @brief Asynchronously send a request to receive data.
    /// @param llmRequest Request object. Its data should be in an allocated but unwritten state when called, and the
    /// data for this request should remain intact only after future synchronization.
    /// @return Once the data is fully received, the future object will become valid.
    [[nodiscard]] std::future<void> requestAndReceiveAsync(LlmRequest& llmRequest) const;

    /// @brief Destructor.
    ~DataRequester();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

class KvCacheMeasureHelper
{
public:
    KvCacheMeasureHelper(std::string output_path)
        : mOutputPath(std::move(output_path))
    {
    }

    void appendKVCacheTransfer(LlmRequest::RequestIdType requestId, double duration, size_t size)
    {
        auto bandwidth = size * 8 / (duration / 1000) / 1e9;
        if (mOutputPath.empty())
        {
            return;
        }

        std::lock_guard<std::mutex> lock(mMutex);
        mRequestKVCacheTranfserMeasure[requestId].emplace_back(duration, bandwidth);
    }

    ~KvCacheMeasureHelper()
    {
        if (!mRequestKVCacheTranfserMeasure.empty() && !mOutputPath.empty())
        {
            auto rank = mpi::MpiComm::world().getRank();
            std::string outFilePath = mOutputPath + "rank_" + std::to_string(rank) + ".txt";
            std::ofstream outFile(outFilePath);

            TLLM_CHECK_WITH_INFO(outFile.is_open(), "Cannot write to file " + outFilePath);

            size_t numTransferMeasure = mRequestKVCacheTranfserMeasure.begin()->second.size();

            outFile << "RequestID";
            for (size_t i = 0; i < numTransferMeasure; i++)
            {
                outFile << ",TimeDuration,Bandwidth";
            }
            outFile << '\n';

            for (auto const& [requestID, measures] : mRequestKVCacheTranfserMeasure)
            {
                outFile << requestID;

                for (auto const& [time, bandwidth] : measures)
                {
                    outFile << "," << time << "," << bandwidth;
                }
                outFile << '\n';
            }

            outFile.close();
        }
    }

private:
    std::map<LlmRequest::RequestIdType, std::vector<std::pair<double, double>>> mRequestKVCacheTranfserMeasure;
    std::string mOutputPath;
    std::mutex mMutex;
};

} // namespace tensorrt_llm::batch_manager
