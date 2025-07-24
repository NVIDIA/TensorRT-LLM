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
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

BlockRange getBlockRangeForSending(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest);

BlockRange getBlockRangeForReceiving(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest);

using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using Connection = tensorrt_llm::executor::kv_cache::Connection;
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
        mConnections.at(idx)->send(mDataContext, data, size);
    }

    void recv(size_t idx, void* data, size_t size)
    {
        mConnections.at(idx)->recv(mDataContext, data, size);
    }

    [[nodiscard]] LlmRequest const& getLlmRequest() const
    {
        TLLM_CHECK(mRequest != nullptr);
        return *mRequest;
    }

    // in CacheSender, the LlmRequest is not available until the sendSync is called
    void setLlmRequest(LlmRequest const& llmRequest)
    {
        mRequest = &llmRequest;
    }

    void appendMeasure(double delay, double duration, size_t size)
    {
        if (!mRecordMeasure)
        {
            return;
        }
        auto bandwidth = size * 8 / (duration / 1000) / 1e9; // byte, ms => Gbps
        mMeasures.emplace_back(Measure{delay, duration, bandwidth});
    }

    // TODO: 1. use global id instead of context request id; 2. export to llm metrics instead of file
    void exportMeasure(std::ofstream& outFile, bool isContext) const
    {
        if (mMeasures.empty())
        {
            return;
        }
        // write header if not exist
        if (outFile.tellp() == 0)
        {
            outFile << "RequestID";
            for (size_t i = 0; i < mMeasures.size(); i++)
            {
                outFile << ",Delay(ms),Duration(ms),Bandwidth(Gbps)";
            }
            outFile << '\n';
        }
        // write measures
        TLLM_CHECK(isContext || mRequest->getContextPhaseParams().has_value());
        auto reqId = isContext ? mRequest->mRequestId : mRequest->getContextPhaseParams().value().getReqId();
        outFile << reqId;
        for (auto const& measure : mMeasures)
        {
            outFile << "," << measure.delay << "," << measure.duration << "," << measure.bandwidth;
        }
        outFile << '\n' << std::flush;
    }

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

// Used to support the cache transmission with different layouts and different protocols.
class BaseCacheFormatter
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheState = executor::kv_cache::CacheState;

    /// @brief Format the cache data into bytes for sending.
    /// @param session The transfer session.
    virtual void format(TransferSession& session) = 0;

    /// @brief Unformat the cache data from received bytes.
    /// @param session The transfer session.
    virtual void unformat(TransferSession& session) = 0;

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

    [[nodiscard]] virtual std::vector<size_t> pickRecvConnections(
        size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
        = 0;

    /// @brief Destructor.
    virtual ~BaseCacheFormatter() = default;
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

    void format(TransferSession& session) override;

    void unformat(TransferSession& session) override;

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
    std::vector<size_t> pickRecvConnections(size_t numConnections, CacheState const& selfConfig, SizeType32 selfIdx,
        CacheState const& destConfig) const override;

private:
    BaseKVCacheManager* mCacheManager;
    CacheTransBufferManager* mCacheTransBufferManager;
};

std::unique_ptr<BaseCacheFormatter> createCacheFormatter(
    BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager, bool isMLA = false);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
