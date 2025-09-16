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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

namespace tensorrt_llm::batch_manager
{

// TODO: unify the following class into a namespace like tensorrt_llm::transmission
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

    [[nodiscard]] std::vector<Connection const*> const& getConnections() const
    {
        return mConnections;
    }

    // should be called only during the initialization of the TransferSession
    void setConnection(size_t idx, Connection const* conn)
    {
        mConnections.at(idx) = conn;
    }

    [[nodiscard]] DataContext const& getDataContext() const
    {
        return mDataContext;
    }

    [[nodiscard]] executor::DataTransceiverState const& getSelfState() const
    {
        return *mSelfState;
    }

    [[nodiscard]] executor::DataTransceiverState const& getOtherState() const
    {
        return mOtherState;
    }

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const
    {
        return *mBufferManager;
    }

    void send(size_t idx, void const* data, size_t size)
    {
        try
        {
            mConnections.at(idx)->send(mDataContext, data, size);
        }
        catch (std::exception const& e)
        {
            throw common::RequestSpecificException(
                __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
        }
    }

    void recv(size_t idx, void* data, size_t size)
    {
        try
        {
            mConnections.at(idx)->recv(mDataContext, data, size);
        }
        catch (std::exception const& e)
        {
            throw common::RequestSpecificException(
                __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
        }
    }

    [[nodiscard]] LlmRequest const& getLlmRequest() const
    {
        TLLM_CHECK(mRequest != nullptr);
        return *mRequest;
    }

    // in DataSender, the LlmRequest is not available until the sendSync is called
    void setLlmRequest(LlmRequest const& llmRequest)
    {
        mRequest = &llmRequest;
    }

    void appendMeasure(double delay, double duration, size_t size);

    // TODO: 1. use global id instead of context request id; 2. export to llm metrics instead of file
    void exportMeasure(std::ofstream& outFile, bool isContext) const;

private:
    std::vector<Connection const*> mConnections;
    DataContext mDataContext;
    executor::DataTransceiverState const* mSelfState; // stored in DataRequester/DataResponder
    executor::DataTransceiverState mOtherState;
    runtime::BufferManager const* mBufferManager;
    LlmRequest const* mRequest;
    bool mRecordMeasure;
    std::vector<Measure> mMeasures;
};

// Operators required for data transmission in specific communication protocols.
class DataSender
{
public:
    /// @brief Receive the request information.
    /// @return The request information.
    [[nodiscard]] virtual RequestInfo recvRequestInfo() = 0;

    /// @brief Synchronously send data.
    /// @param llmRequest The request object to which the data belongs.
    virtual void sendSync(LlmRequest const& llmRequest) = 0;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] virtual executor::kv_cache::CommState const& getCommState() const = 0;

    /// @brief Reset the internal communicator status.
    /// @param commState The communicator status.
    virtual void setCommState(executor::kv_cache::CommState commState) = 0;

    [[nodiscard]] virtual size_t getCounterpartsCount(LlmRequest::RequestIdType requestId) const = 0;

    virtual void release(LlmRequest::RequestIdType requestId) = 0;

    /// @brief Destructor.
    virtual ~DataSender() = default;
};

// Operators required for data transmission in specific communication protocols.
class DataReceiver
{
public:
    /// @brief Send the request information.
    /// @param llmRequest The request object to which the information belongs.
    virtual TransferSession sendRequestInfo(LlmRequest const& llmRequest) = 0;

    /// @brief Synchronously receive data.
    /// @param session The transfer session.
    virtual void receiveSync(TransferSession& session) = 0;

    /// @brief Destructor.
    virtual ~DataReceiver() = default;
};

class DataResponder
{
public:
    /// @brief Constructor.
    /// @param sender The sender used at the underlying level.
    explicit DataResponder(std::unique_ptr<DataSender> sender);

    /// @brief Asynchronously respond to the request and send data.
    /// @param llmRequest Request object. Its data should be ready when called, and the data for this request
    /// should remain valid until future synchronization.
    /// @return Once the data is fully sent, the future object will become valid.
    [[nodiscard]] std::future<void> respondAndSendAsync(LlmRequest& llmRequest) const;

    /// @brief Return the internal communicator status.
    /// @return The communicator status.
    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const;

    /// @brief Reset the internal communicator status.
    /// @param commState The communicator status.
    void setCommState(executor::kv_cache::CommState commState);

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
    explicit DataRequester(std::unique_ptr<DataReceiver> receiver);

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

} // namespace tensorrt_llm::batch_manager
