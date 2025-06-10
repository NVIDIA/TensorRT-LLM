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

// Used to support the data transmission with different layouts and different protocols.
class IOFormatter
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

    [[nodiscard]] virtual std::vector<executor::kv_cache::Connection const*> pickRecvConnections(
        std::vector<executor::kv_cache::Connection const*> const& connections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig) const
        = 0;

    /// @brief Destructor.
    virtual ~IOFormatter() = default;
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
    virtual void sendRequestInfo(LlmRequest const& llmRequest) = 0;

    /// @brief Synchronously receive data.
    /// @param llmRequest The request object to which the data belongs.
    virtual void receiveSync(LlmRequest const& llmRequest) = 0;

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
