/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/batch_manager/namedTensor.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace texec = tensorrt_llm::executor;
namespace mpi = tensorrt_llm::mpi;
namespace trt = nvinfer1;

namespace
{

using TensorPtr = ITensor::SharedPtr;

class LoraLib
{
public:
    LoraLib(std::string const& loraDir)
        : mLoraDir(loraDir)
        , mBufferManager(std::make_shared<CudaStream>())
        , mTaskPaths(parseDirPaths(mLoraDir))
        , mLoras(readLoras(mTaskPaths))
    {
    }

    TensorPtr getLoraWeights(uint64_t taskId) const
    {
        return mLoras.at(taskId).first;
    }

    TensorPtr getLoraConfig(uint64_t taskId) const
    {
        return mLoras.at(taskId).second;
    }

    void clear()
    {
        mLoras.clear();
    }

    std::map<uint64_t, std::pair<TensorPtr, TensorPtr>> const& getLoras()
    {
        return mLoras;
    }

private:
    std::string const mLoraDir;
    BufferManager mBufferManager;
    std::map<uint64_t, fs::path> mTaskPaths;
    std::map<uint64_t, std::pair<TensorPtr, TensorPtr>> mLoras;

    std::map<uint64_t, std::pair<TensorPtr, TensorPtr>> readLoras(std::map<uint64_t, fs::path> taskPaths)
    {
        std::map<uint64_t, std::pair<TensorPtr, TensorPtr>> loras;
        for (auto const& [id, p] : taskPaths)
        {
            TensorPtr loraWeights
                = utils::loadNpy(mBufferManager, (p / "model.lora_weights.npy").string(), MemoryType::kCPU);
            TensorPtr loraConfig
                = utils::loadNpy(mBufferManager, (p / "model.lora_config.npy").string(), MemoryType::kCPU);
            loras.insert_or_assign(id, std::make_pair(loraWeights, loraConfig));
        }
        return loras;
    }

    std::map<uint64_t, fs::path> parseDirPaths(std::string const& loraDir)
    {
        std::map<uint64_t, fs::path> taskPaths;
        if (loraDir == "")
        {
            return taskPaths;
        }
        for (auto const& entry : fs::recursive_directory_iterator(loraDir))
        {
            if (entry.is_directory())
            {
                auto taskId = parseId(entry.path());
                taskPaths.insert_or_assign(taskId, entry.path());
            }
        }
        return taskPaths;
    }

    uint64_t parseId(fs::path p)
    {
        auto fn = p.filename().string();
        auto dashPos = fn.find_first_of("-");
        std::string idStr = fn;
        if (dashPos != std::string::npos)
        {
            auto idStr = fn.substr(0, dashPos);
        }
        uint64_t id = static_cast<uint64_t>(std::stoi(idStr));
        return id;
    }
};

struct BenchmarkParams
{
    std::optional<SizeType32> maxTokensInPagedKvCache{std::nullopt};
    std::optional<float> freeGpuMemoryFraction{std::nullopt};
    bool enableTrtOverlap{false};
    bool enableBlockReuse{false};
    bool enableChunkedContext{false};
    bool streaming{false};
    bool enableExpDelays{false};
    std::optional<float> requestRate{std::nullopt};
    std::optional<SizeType32> maxBatchSize{std::nullopt};
    std::optional<SizeType32> maxNumTokens{std::nullopt};
    int randomSeed = 430;
    std::optional<int> maxAttentionWindow{std::nullopt};

    // lora / peft params
    std::optional<std::string> loraDir{std::nullopt};
    SizeType32 loraDeviceNumModLayers{0};
    size_t loraHostCacheSize{1024 * 2024 * 1024};

    // KV cache block offloading
    size_t kvHostCacheSize{0};
    bool kvOnboardBlocks{true};

    // Weights offloading
    float gpuWeightsPercent{1.0};

    // Decoding params
    std::optional<std::vector<std::vector<SizeType32>>> medusaChoices;
};

class InferenceRequestsSyncSend
{
public:
    InferenceRequestsSyncSend(std::shared_ptr<tensorrt_llm::mpi::MpiComm> comm,
        std::list<std::shared_ptr<InferenceRequest>> const& inferenceRequests, int const peer)
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        TLLM_LOG_DEBUG("start send requests to rank %d", peer);
        mNumNewWorkItems = static_cast<int64_t>(inferenceRequests.size());
        comm->send(&mNumNewWorkItems, 1, mpi::MpiType::kINT64, peer, 0);
        if (mNumNewWorkItems > 0)
        {
            for (auto const& infReq : inferenceRequests)
            {
                auto vpacked = infReq->serialize();
                mPacked.push_back(static_cast<int64_t>(vpacked.size()));
                mPacked.insert(mPacked.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
            }
            mVecSize = static_cast<int64_t>(mPacked.size());
            comm->send(&mVecSize, 1, mpi::MpiType::kINT64, peer, 1);
            comm->send(mPacked.data(), mPacked.size(), mpi::MpiType::kINT64, peer, 2);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

private:
    int64_t mNumNewWorkItems;
    int64_t mVecSize;
    std::vector<int64_t> mPacked;
};
} // namespace

void inferenceRequestsRecv(std::shared_ptr<tensorrt_llm::mpi::MpiComm> comm,
    std::list<std::shared_ptr<InferenceRequest>>& inferenceRequests, int const peer)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("start recv requests from rank %d", peer);
    int64_t numNewWorkItems = 0;
    comm->recv(&numNewWorkItems, 1, mpi::MpiType::kINT64, peer, 0);
    if (numNewWorkItems > 0)
    {
        std::vector<int64_t> packed;
        int64_t vecSize;
        comm->recv(&vecSize, 1, mpi::MpiType::kINT64, peer, 1);
        packed.resize(vecSize);
        comm->recv(packed.data(), packed.size(), mpi::MpiType::kINT64, peer, 2);
        int64_t* packed_ptr = packed.data();
        for (int64_t count = 0; count < numNewWorkItems; ++count)
        {
            int64_t n = *(packed_ptr++);
            auto infReq = InferenceRequest::deserialize(packed_ptr);
            packed_ptr += n;
            inferenceRequests.emplace_back(infReq);
        }
    }
    TLLM_LOG_DEBUG("end recv requests from rank %d", peer);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
public:
    WorkItem(std::shared_ptr<InferenceRequest> inferenceRequest, uint64_t requestId)
        : mInferenceRequest(std::move(inferenceRequest))
        , mRequestId(requestId)
    {
    }

    [[nodiscard]] uint64_t requestId() const
    {
        return mRequestId;
    }

    [[nodiscard]] std::shared_ptr<InferenceRequest> getInferenceRequest() const
    {
        return mInferenceRequest;
    }

private:
    std::shared_ptr<InferenceRequest> mInferenceRequest;
    uint64_t mRequestId;
};

/// @brief Thread-safe queue of work items
class WorkItemsQueue
{
public:
    void clear()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mPendingWorkItems.clear();
        mPendingWorkItemsReqIds.clear();
        mInProgressWorkItems.clear();
    }

    // Note: this function only be called under a lock
    bool hasInProgressReqId(uint64_t const reqId) const
    {
        return (mInProgressWorkItems.find(reqId) != mInProgressWorkItems.end());
    }

    // Note: this function only be called under a lock
    bool hasPendingReqId(uint64_t const reqId) const
    {
        return (mPendingWorkItemsReqIds.find(reqId) != mPendingWorkItemsReqIds.end());
    }

    bool empty() const
    {
        return mPendingWorkItems.empty() && mInProgressWorkItems.empty() && mPendingWorkItemsReqIds.empty();
    }

    /// @brief Add a new work item to the queue
    /// Throws an error if requestId already exists

    void push(std::shared_ptr<InferenceRequest> request, uint64_t requestId)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        TLLM_CHECK_WITH_INFO(!hasInProgressReqId(requestId) && !hasPendingReqId(requestId),
            "requestId %lu is already in progress, request is ignored.", requestId);

        auto workItem = std::make_shared<WorkItem>(request, requestId);
        mPendingWorkItems.push_back(workItem);
        mPendingWorkItemsReqIds.insert(workItem->requestId());
    }

    /// @brief Get a new work item from the queue, and move it to the list of
    /// in progress work items if it hasn't been stopped
    /// @return A tuple of the workItem and a boolean flag indicating if the work item
    /// has been marked in progress
    std::tuple<std::shared_ptr<WorkItem>, bool> pop()
    {
        std::lock_guard<std::mutex> lock(mMutex);

        auto workItem = mPendingWorkItems.front();
        mPendingWorkItems.pop_front();
        mPendingWorkItemsReqIds.erase(workItem->requestId());

        bool markedInProgress = false;
        mInProgressWorkItems.emplace(workItem->requestId(), workItem);
        markedInProgress = true;

        return {workItem, markedInProgress};
    }

    size_t numPendingWorkItems() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mPendingWorkItems.size();
    }

    size_t numInProgressWorkItems() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mInProgressWorkItems.size();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mPendingWorkItems.size() + mInProgressWorkItems.size();
    }

    /// @brief  Mark a request as being finished
    /// @param requestId
    void markFinished(uint64_t const requestId)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (hasInProgressReqId(requestId))
        {
            mInProgressWorkItems.erase(requestId);
        }
    }

private:
    /// Queue of work items
    std::list<std::shared_ptr<WorkItem>> mPendingWorkItems;
    /// requestIds of work items in the queue
    std::set<uint64_t> mPendingWorkItemsReqIds;

    /// work items currently in progress
    std::unordered_map<uint64_t, std::shared_ptr<WorkItem>> mInProgressWorkItems;

    mutable std::mutex mMutex;
};

struct BenchInfo
{
    BenchInfo() = default;

    BenchInfo(int inputLength, std::chrono::time_point<std::chrono::steady_clock> start)
        : inputLength(inputLength)
        , start(start)
    {
    }

    int inputLength;
    int outputLength{0};
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    std::chrono::time_point<std::chrono::steady_clock> firstTokenTs;
    float latency{}; // millisecond
    bool hasError{false};
    float firstTokenLatency{};
    std::optional<float> avgGenT2TLatency{};
    bool firstTokenSeen{false};
};

class Recorder
{
    using TensorPtr = ITensor::SharedPtr;

public:
    explicit Recorder(std::string opCsvFile, bool streaming = false, int beamWidth = 1,
        std::string responsesJsonFile = "", bool excludeInputInOutput = false)
        : mOpCsvFile(std::move(opCsvFile))
        , mStreaming(streaming)
        , mBeamWidth(beamWidth)
        , mRespJsonFile(std::move(responsesJsonFile))
        , mOutputHasInput(!excludeInputInOutput)
    {
    }

    void initialize()
    {
        mStart = std::chrono::steady_clock::now();
    }

    void finalize()
    {
        mEnd = std::chrono::steady_clock::now();
    }

    void recordStart(std::shared_ptr<InferenceRequest> request, uint64_t requestId)
    {
        auto const inputLength = request->getInputIds()->getSize();
        auto const maxNewTokens = request->getMaxNewTokensNamed();
        auto const& outputLengthTensor = maxNewTokens.tensor;
        TLLM_CHECK_WITH_INFO(outputLengthTensor != nullptr && outputLengthTensor->getSize() > 0,
            "Undefined scalar vector for %s", maxNewTokens.name.c_str());
        auto const outputLength = *bufferCast<SizeType32>(*outputLengthTensor);
        auto const start = std::chrono::steady_clock::now();
        mRequestBenchInfos[requestId] = BenchInfo(inputLength, start);
    }

    // number of output tokens not calculated from output sequence here, instead set to max_output_len
    //   - if eos_id == -1 (default behavior), this is correct since output seq will have max permissible length.
    //   - However, if eos_id != -1, the token size of output sequence may be less than max_output_len, and token
    //   throughput may be inaccurate
    void recordStart(SizeType32 inputLength, SizeType32 maxNewTokens, uint64_t requestId,
        std::chrono::time_point<std::chrono::steady_clock> const& start)
    {
        mRequestBenchInfos[requestId] = BenchInfo(inputLength, start);
    }

    void recordEnd(uint64_t requestId, bool hasError)
    {
        mRequestBenchInfos[requestId].end = std::chrono::steady_clock::now();
        mRequestBenchInfos[requestId].hasError = hasError;
    }

    void recordToken(uint64_t requestId)
    {
        TLLM_CHECK(mStreaming);
        TLLM_CHECK_WITH_INFO(mBeamWidth == 1, "gptManagerBenchmark streaming mode does not support beam > 1");

        if (!mRequestBenchInfos[requestId].firstTokenSeen)
        {
            mRequestBenchInfos[requestId].firstTokenTs = std::chrono::steady_clock::now();
            mRequestBenchInfos[requestId].firstTokenSeen = true;
        }

        mRequestBenchInfos[requestId].outputLength += 1;
    }

    void recordEnd(uint64_t requestId, std::list<NamedTensor> const& responseTensors, bool hasError)
    {
        this->recordEnd(requestId, hasError);

        if (!mStreaming)
        {
            for (auto& tensor : responseTensors)
            {
                if (tensor.name == inference_request::kOutputIdsTensorName)
                {
                    mResponseTensors[requestId] = tensor.tensor;
                }
                else if (tensor.name == inference_request::kSequenceLengthTensorName)
                {
                    // Tensor of shape nBeams, and we only need the first one
                    int32_t outputSeqLen = *(bufferCast<int32_t>(*(tensor.tensor)));
                    if (mOutputHasInput)
                    {
                        int inputSeqLen = mRequestBenchInfos[requestId].inputLength;
                        outputSeqLen -= inputSeqLen;
                    }
                    mRequestBenchInfos[requestId].outputLength = outputSeqLen;
                }
            }
        }
        else
        {
            this->recordToken(requestId);
        }
    }

    void recordEnd(uint64_t requestId, texec::Response const& response)
    {

        this->recordEnd(requestId, response.hasError());

        // Get the actual output length
        if (!response.hasError())
        {
            if (!mStreaming)
            {
                auto outputTokenIds = response.getResult().outputTokenIds;

                int32_t outSeqLen = 0;
                for (auto const& beam : outputTokenIds)
                {
                    outSeqLen = std::max(static_cast<int32_t>(beam.size()), outSeqLen);
                }
                if (mOutputHasInput)
                {
                    int inputSeqLen = mRequestBenchInfos[requestId].inputLength;
                    outSeqLen -= inputSeqLen;
                }
                mRequestBenchInfos[requestId].outputLength = outSeqLen;
            }
            else
            {
                this->recordToken(requestId);
            }
        }
    }

    float calcPercentile(std::vector<float> const& latencies, int percentile)
    {
        int const index = static_cast<int>(std::ceil((percentile / 100.0) * latencies.size())) - 1;
        return latencies[index];
    }

    void calculateLatencies()
    {
        for (auto& reqInfo : mRequestBenchInfos)
        {
            reqInfo.second.latency
                = std::chrono::duration<float, std::milli>(reqInfo.second.end - reqInfo.second.start).count();
            if (mStreaming)
            {
                reqInfo.second.firstTokenLatency
                    = std::chrono::duration<float, std::milli>(reqInfo.second.firstTokenTs - reqInfo.second.start)
                          .count();
                if (reqInfo.second.outputLength > 1)
                {
                    reqInfo.second.avgGenT2TLatency
                        = std::chrono::duration<float, std::milli>(reqInfo.second.end - reqInfo.second.firstTokenTs)
                              .count()
                        / static_cast<float>(reqInfo.second.outputLength - 1);
                }
            }
        }
    }

    void calculateMetrics()
    {
        calculateLatencies();

        std::vector<float> reqLatencies;
        std::vector<float> ftLatencies;
        std::vector<float> genT2TLatencies;

        int totalOutputTokens{0};
        mNumErrorSamples = 0;
        mNumSamples = 0;
        for (auto reqInfo : mRequestBenchInfos)
        {
            if (!reqInfo.second.hasError)
            {
                reqLatencies.push_back(reqInfo.second.latency);
                totalOutputTokens += reqInfo.second.outputLength;

                if (mStreaming)
                {
                    ftLatencies.push_back(reqInfo.second.firstTokenLatency);

                    if (reqInfo.second.avgGenT2TLatency)
                    {
                        genT2TLatencies.push_back(reqInfo.second.avgGenT2TLatency.value());
                    }
                }
                ++mNumSamples;
            }
            else
            {
                ++mNumErrorSamples;
            }
        }

        mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
        mSeqThroughput = mNumSamples / (mTotalLatency / 1000);
        mTokenThroughput = totalOutputTokens / (mTotalLatency / 1000);

        mAvgSeqLatency = std::accumulate(reqLatencies.begin(), reqLatencies.end(), 0.F) / reqLatencies.size();

        std::sort(reqLatencies.begin(), reqLatencies.end());

        mP99SeqLatency = calcPercentile(reqLatencies, 99);
        mP90SeqLatency = calcPercentile(reqLatencies, 90);
        mP50SeqLatency = calcPercentile(reqLatencies, 50);
        mMaxSeqLatency = reqLatencies.back();
        mMinSeqLatency = reqLatencies.front();

        if (mStreaming)
        {
            mAvgFtLatency = std::accumulate(ftLatencies.begin(), ftLatencies.end(), 0.F) / ftLatencies.size();

            std::sort(ftLatencies.begin(), ftLatencies.end());

            mP99FtLatency = calcPercentile(ftLatencies, 99);
            mP90FtLatency = calcPercentile(ftLatencies, 90);
            mP50FtLatency = calcPercentile(ftLatencies, 50);
            mMaxFtLatency = ftLatencies.back();
            mMinFtLatency = ftLatencies.front();

            if (!genT2TLatencies.empty())
            {
                mAvgGenT2TLatency
                    = std::accumulate(genT2TLatencies.begin(), genT2TLatencies.end(), 0.F) / genT2TLatencies.size();

                std::sort(genT2TLatencies.begin(), genT2TLatencies.end());

                mP99GenT2TLatency = calcPercentile(genT2TLatencies, 99);
                mP90GenT2TLatency = calcPercentile(genT2TLatencies, 90);
                mP50GenT2TLatency = calcPercentile(genT2TLatencies, 50);
                mMaxGenT2TLatency = genT2TLatencies.back();
                mMinGenT2TLatency = genT2TLatencies.front();
            }
        }
    }

    void report()
    {

        printf("[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] num_error_samples %d\n", mNumErrorSamples);
        printf("\n[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[BENCHMARK] token_throughput(token/sec) %.2f\n\n", mTokenThroughput);

        printf("[BENCHMARK] avg_sequence_latency(ms) %.2f\n", mAvgSeqLatency);
        printf("[BENCHMARK] max_sequence_latency(ms) %.2f\n", mMaxSeqLatency);
        printf("[BENCHMARK] min_sequence_latency(ms) %.2f\n", mMinSeqLatency);
        printf("[BENCHMARK] p99_sequence_latency(ms) %.2f\n", mP99SeqLatency);
        printf("[BENCHMARK] p90_sequence_latency(ms) %.2f\n", mP90SeqLatency);
        printf("[BENCHMARK] p50_sequence_latency(ms) %.2f\n\n", mP50SeqLatency);

        if (mStreaming)
        {
            printf("[BENCHMARK] avg_time_to_first_token(ms) %.2f\n", mAvgFtLatency);
            printf("[BENCHMARK] max_time_to_first_token(ms) %.2f\n", mMaxFtLatency);
            printf("[BENCHMARK] min_time_to_first_token(ms) %.2f\n", mMinFtLatency);
            printf("[BENCHMARK] p99_time_to_first_token(ms) %.2f\n", mP99FtLatency);
            printf("[BENCHMARK] p90_time_to_first_token(ms) %.2f\n", mP90FtLatency);
            printf("[BENCHMARK] p50_time_to_first_token(ms) %.2f\n\n", mP50FtLatency);

            printf("[BENCHMARK] avg_inter_token_latency(ms) %.2f\n", mAvgGenT2TLatency);
            printf("[BENCHMARK] max_inter_token_latency(ms) %.2f\n", mMaxGenT2TLatency);
            printf("[BENCHMARK] min_inter_token_latency(ms) %.2f\n", mMinGenT2TLatency);
            printf("[BENCHMARK] p99_inter_token_latency(ms) %.2f\n", mP99GenT2TLatency);
            printf("[BENCHMARK] p90_inter_token_latency(ms) %.2f\n", mP90GenT2TLatency);
            printf("[BENCHMARK] p50_inter_token_latency(ms) %.2f\n\n", mP50GenT2TLatency);
        }
    }

    void writeOpMetricsToCsv()
    {
        if (!mOpCsvFile.empty())
        {
            std::vector<std::string> headers = {"num_samples", "num_error_samples", "total_latency(ms)",
                "seq_throughput(seq/sec)", "token_throughput(token/sec)", "avg_sequence_latency(ms)",
                "max_sequence_latency(ms)", "min_sequence_latency(ms)", "p99_sequence_latency(ms)",
                "p90_sequence_latency(ms)", "p50_sequence_latency(ms)"};

            if (mStreaming)
            {
                std::vector<std::string> streamingHeaders
                    = {"avg_time_to_first_token(ms)", "max_time_to_first_token(ms)", "min_time_to_first_token(ms)",
                        "p99_time_to_first_token(ms)", "p90_time_to_first_token(ms)", "p50_time_to_first_token(ms)",
                        "avg_inter_token_latency(ms)", "max_inter_token_latency(ms)", "min_inter_token_latency(ms)",
                        "p99_inter_token_latency(ms)", "p90_inter_token_latency(ms)", "p50_inter_token_latency(ms)"};

                headers.insert(headers.end(), streamingHeaders.begin(), streamingHeaders.end());
            }

            std::ofstream outputFile(mOpCsvFile);

            if (outputFile.is_open())
            {
                for (auto const& header : headers)
                {
                    outputFile << header << ",";
                }
                outputFile << "\n";
                outputFile << mNumSamples << "," << mNumErrorSamples << "," << mTotalLatency << "," << mSeqThroughput
                           << "," << mTokenThroughput << "," << mAvgSeqLatency << "," << mMaxSeqLatency << ","
                           << mMinSeqLatency << "," << mP99SeqLatency << "," << mP90SeqLatency << "," << mP50SeqLatency;
                if (mStreaming)
                {
                    outputFile << "," << mAvgFtLatency << "," << mMaxFtLatency << "," << mMinFtLatency << ","
                               << mP99FtLatency << "," << mP90FtLatency << "," << mP50FtLatency << ","
                               << mAvgGenT2TLatency << "," << mMaxGenT2TLatency << "," << mMinGenT2TLatency << ","
                               << mP99GenT2TLatency << "," << mP90GenT2TLatency << "," << mP50GenT2TLatency;
                }

                outputFile << "\n";
            }
            else
            {
                std::cerr << "Error opening file '" << mOpCsvFile << "' for writing.\n";
            }
        }
    }

    void dumpResponseSeqs()
    {
        if (mRespJsonFile.empty())
            return;
        nlohmann::json jsonResponses = nlohmann::json::array();
        for (auto const& [respId, respTokensTensor] : mResponseTensors)
        {
            int inputLength = mRequestBenchInfos[respId].inputLength;
            int outputLength = mRequestBenchInfos[respId].outputLength;
            std::vector<int32_t> outputTokens(outputLength);

            int32_t* outputToksBufferPtr = bufferCast<int32_t>(*respTokensTensor);
            if (mOutputHasInput)
                outputToksBufferPtr += inputLength;
            std::copy(outputToksBufferPtr, outputToksBufferPtr + outputLength, outputTokens.begin());

            nlohmann::json currResp;
            currResp["response_id"] = respId;
            currResp["response_tokens"] = outputTokens;
            jsonResponses.push_back(currResp);
        }
        std::ofstream outFile(mRespJsonFile);
        outFile << jsonResponses;
        outFile.close();
    }

private:
    std::unordered_map<uint64_t, BenchInfo> mRequestBenchInfos;

    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mEnd;
    int mNumSamples{};
    int mNumErrorSamples{};
    float mTotalLatency{};
    float mSeqThroughput{};
    float mAvgSeqLatency{};
    float mAvgGenT2TLatency{};
    float mAvgFtLatency{};
    float mTokenThroughput{};
    float mP99SeqLatency{};
    float mP90SeqLatency{};
    float mP50SeqLatency{};
    float mMaxSeqLatency{};
    float mMinSeqLatency{};
    float mP99FtLatency{};
    float mP90FtLatency{};
    float mP50FtLatency{};
    float mMaxFtLatency{};
    float mMinFtLatency{};
    float mP99GenT2TLatency{};
    float mP90GenT2TLatency{};
    float mP50GenT2TLatency{};
    float mMaxGenT2TLatency{};
    float mMinGenT2TLatency{};

    std::string mOpCsvFile;
    bool mStreaming;
    int mBeamWidth;
    std::string mRespJsonFile;
    std::unordered_map<uint64_t, TensorPtr> mResponseTensors;
    bool mOutputHasInput;

}; // class Recorder

class ExecutorServer
{
public:
    ExecutorServer(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, int32_t maxBeamWidth,
        texec::CapacitySchedulerPolicy capacitySchedulerPolicy, BenchmarkParams const& benchmarkParams,
        std::shared_ptr<Recorder> recorder, std::chrono::milliseconds waitSleep,
        std::optional<uint64_t> const staticEmulatedBatchSize, bool logIterationData)
        : mRecorder(std::move(recorder))
        , mWaitSleep(waitSleep)
        , mStaticEmulatedBatchSize(staticEmulatedBatchSize)
        , mActiveCount(0)
        , mShutdown(false)
    {

        texec::SchedulerConfig schedulerConfig(capacitySchedulerPolicy);
        texec::KvCacheConfig kvCacheConfig(benchmarkParams.enableBlockReuse, benchmarkParams.maxTokensInPagedKvCache,
            benchmarkParams.maxAttentionWindow, std::nullopt, benchmarkParams.freeGpuMemoryFraction,
            benchmarkParams.kvHostCacheSize, benchmarkParams.kvOnboardBlocks);
        texec::PeftCacheConfig peftCacheConfig(0, benchmarkParams.loraDeviceNumModLayers, 8, 64, 4, 4, 4, 24, 8,
            std::nullopt, benchmarkParams.loraHostCacheSize);
        texec::ExecutorConfig executorConfig(
            maxBeamWidth, schedulerConfig, kvCacheConfig, benchmarkParams.enableChunkedContext, true);
        executorConfig.setGpuWeightsPercent(benchmarkParams.gpuWeightsPercent);
        executorConfig.setPeftCacheConfig(peftCacheConfig);
        executorConfig.setBatchingType(
            modelType == TrtGptModelType::V1 ? texec::BatchingType::kSTATIC : texec::BatchingType::kINFLIGHT);
        if (benchmarkParams.maxBatchSize)
        {
            executorConfig.setMaxBatchSize(benchmarkParams.maxBatchSize.value());
        }
        if (benchmarkParams.maxNumTokens)
        {
            executorConfig.setMaxNumTokens(benchmarkParams.maxNumTokens.value());
        }

        executorConfig.setDecodingConfig(texec::DecodingConfig(
            benchmarkParams.medusaChoices.has_value() ? texec::DecodingMode::Medusa() : texec::DecodingMode::Auto(),
            std::nullopt, benchmarkParams.medusaChoices));

        mExecutor = std::make_unique<texec::Executor>(trtEnginePath, texec::ModelType::kDECODER_ONLY, executorConfig);

        if (logIterationData)
        {
            mCollectStatsThread = std::thread(&ExecutorServer::collectStats, this);
        }
    }

    ~ExecutorServer()
    {
        mShutdown = true;
        if (mCollectStatsThread.joinable())
        {
            mCollectStatsThread.join();
        }
    }

    void enqueue(std::vector<texec::Request> requests, bool warmup = false)
    {
        try
        {
            std::vector<SizeType32> inputLengths;
            std::vector<SizeType32> maxNewTokens;
            for (auto const& request : requests)
            {
                inputLengths.push_back(request.getInputTokenIds().size());
                maxNewTokens.push_back(request.getMaxNewTokens());
            }
            auto const start = std::chrono::steady_clock::now();
            auto reqIds = mExecutor->enqueueRequests(std::move(requests));
            for (int req = 0; req < reqIds.size(); ++req)
            {
                if (!warmup)
                {
                    mRecorder->recordStart(inputLengths.at(req), maxNewTokens.at(req), reqIds.at(req), start);
                }
                mActiveCount++;
            }
        }
        catch (std::exception const& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void waitForResponses(SizeType32 numRequests, bool warmup = false)
    {
        SizeType32 numFinished = 0;
        while (mActiveCount || (numFinished < numRequests))
        {
            auto responses = mExecutor->awaitResponses(mWaitSleep);
            for (auto const& response : responses)
            {
                auto const reqId = response.getRequestId();

                if (response.getResult().isFinal)
                {
                    mActiveCount--;
                    numFinished++;
                    if (!warmup)
                    {
                        mRecorder->recordEnd(reqId, response);
                    }
                }
                else
                {
                    if (!warmup && !response.hasError())
                    {
                        mRecorder->recordToken(reqId);
                    }
                }
            }
        }
    }

    void collectStats()
    {
        while (!mShutdown)
        {
            auto iterStats = mExecutor->getLatestIterationStats();
            for (auto const& iterStat : iterStats)
            {
                TLLM_LOG_INFO(texec::JsonSerialization::toJsonStr(iterStat));
            }
            auto const waitSleep = std::chrono::milliseconds(50);
            std::this_thread::sleep_for(waitSleep);
        }
    }

private:
    std::unique_ptr<texec::Executor> mExecutor;
    std::thread mCollectStatsThread;
    std::shared_ptr<Recorder> mRecorder;
    std::chrono::milliseconds mWaitSleep;
    std::optional<int> mStaticEmulatedBatchSize;
    std::atomic<uint64_t> mActiveCount;
    std::atomic<bool> mShutdown;
}; // class ExecutorServer

class GptServer
{
public:
    GptServer(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        TrtGptModelOptionalParams const& optionalParams, std::shared_ptr<Recorder> recorder,
        std::optional<uint64_t> terminateReqId, std::chrono::milliseconds waitSleep,
        std::optional<SizeType32> const staticEmulatedBatchSize,
        std::optional<std::chrono::milliseconds> const batchTimeout, bool logIterationData, bool excludeInputInOutput)
        : mRecorder(std::move(recorder))
        , mTerminateReqId(terminateReqId)
        , mWaitSleep(waitSleep)
        , mStaticEmulatedBatchSize(staticEmulatedBatchSize)
        , mBatchTimeout(batchTimeout.value_or(std::chrono::milliseconds{0}))
        , mActiveCount(0)
        , mInferReqSyncSndHdl(nullptr)
    {
        auto const jsonConfig = GptJsonConfig::parse(trtEnginePath / "config.json");
        mWorldConfig = WorldConfig::mpi(jsonConfig.getGpusPerNode(), jsonConfig.getTensorParallelism(),
            jsonConfig.getPipelineParallelism(), optionalParams.deviceIds);
        auto& comm = COMM_SESSION;
        mCommTensorParallel = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            comm.split(mWorldConfig.getPipelineParallelRank(), mWorldConfig.getTensorParallelRank()));
        mCommPipelineParallel = std::make_shared<tensorrt_llm::mpi::MpiComm>(
            comm.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));

        ReturnBatchManagerStatsCallback iterationDataCallback = [this, logIterationData](std::string const& log)
        {
            if (logIterationData)
            {
                TLLM_LOG_INFO(log);
            }

            if (mStaticEmulatedBatchSize)
            {
                auto const json = nlohmann::json::parse(log);
                auto const activeRequests = json["Active Request Count"];
                TLLM_CHECK(activeRequests <= mStaticEmulatedBatchSize.value());
            }
        };

        mBatchManager = std::make_shared<GptManager>(
            trtEnginePath, modelType, [this](int max_num_requests) { return getInferenceRequests(max_num_requests); },
            [this](uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response,
                std::string const& errMsg)
            { return sendResponse(requestId, response_tensors, final_response, errMsg); },
            nullptr, iterationDataCallback, optionalParams, terminateReqId, excludeInputInOutput);
    }

    ~GptServer()
    {
        mWorkItemsQueue.clear();
    }

    std::string getLayerProfileInfo()
    {
        return mBatchManager->getLayerProfileInfo();
    }

    void setLayerProfiler()
    {
        return mBatchManager->setLayerProfiler();
    }

    void enqueue(std::shared_ptr<InferenceRequest> const& request)
    {
        TLLM_CHECK(request != nullptr);
        auto const requestId = request->getRequestId();
        if (requestId == mTerminateReqId)
        {
            mWorkItemsQueue.push(request, requestId);
            return;
        }

        // Enqueue
        try
        {
            mRecorder->recordStart(request, requestId);
            mWorkItemsQueue.push(request, requestId);
        }
        catch (tc::TllmException const& e)
        {
            throw;
        }
        catch (std::exception const& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void resetBatchDeadline()
    {
        mBatchDeadline = (std::chrono::steady_clock::now() + mBatchTimeout).time_since_epoch();
    }

    void waitForEmpty() const
    {
        while (!mWorkItemsQueue.empty())
        {
            std::this_thread::sleep_for(mWaitSleep);
        }
    }

    void waitBatchManager() const
    {
        mBatchManager->waitUntilTerminate();
    }

    void shutdown() const
    {
        mBatchManager->shutdown();
    }

    // Return up to max_num_requests inference requests.
    std::list<std::shared_ptr<InferenceRequest>> getInferenceRequests(int const max_num_requests)
    {
        mInferReqSyncSndHdl = nullptr;
        std::list<std::shared_ptr<InferenceRequest>> inferenceRequests;
        auto& comm = COMM_SESSION;
        if (max_num_requests > 0)
        {
            auto rank = comm.getRank();
            if (rank == 0)
            {
                auto const numNewWorkItems = std::min(static_cast<int64_t>(mWorkItemsQueue.numPendingWorkItems()),
                    static_cast<int64_t>(max_num_requests));

                bool const timeout = std::chrono::steady_clock::now().time_since_epoch() > mBatchDeadline.load();
                bool readyForNextBatch = numNewWorkItems > 0 && timeout;
                if (mStaticEmulatedBatchSize)
                {
                    if (numNewWorkItems > 0)
                    {
                        bool const previousBatchFinished = mActiveCount == 0;
                        bool const haveEnoughForNextBatch = numNewWorkItems >= mStaticEmulatedBatchSize.value();
                        readyForNextBatch = previousBatchFinished && (timeout || haveEnoughForNextBatch);
                    }
                    if (numNewWorkItems == 0 || readyForNextBatch)
                    {
                        // Timeout should only begin once we have at least 1 pending request.
                        // Reset timeout when no requests are pending or we submit a new batch.
                        resetBatchDeadline();
                    }
                }

                if (readyForNextBatch)
                {
                    // Only add a single batch at a time when emulating static batching
                    auto const numItemsToAdd = std::min(
                        numNewWorkItems, static_cast<int64_t>(mStaticEmulatedBatchSize.value_or(numNewWorkItems)));
                    mActiveCount += numItemsToAdd;
                    while (inferenceRequests.size() < numItemsToAdd)
                    {
                        auto [workItem, markedInProgress] = mWorkItemsQueue.pop();

                        if (markedInProgress)
                        {
                            inferenceRequests.emplace_back(workItem->getInferenceRequest());
                        }
                        else
                        {
                            auto warnStr = tc::fmtstr(
                                "request Id %lu has been stopped. Request is ignored.", workItem->requestId());
                            TLLM_LOG_WARNING(warnStr);
                            sendResponse(workItem->requestId(), {}, true, warnStr);
                        }
                    }
                }
                if (mWorldConfig.isTensorParallel())
                {
                    auto numNewWorkItems = static_cast<int64_t>(inferenceRequests.size());
                    if (numNewWorkItems > 0 || mBatchManager->getNumActiveRequests() > 0)
                    {
                        mCommTensorParallel->bcast(&numNewWorkItems, 1, mpi::MpiType::kINT64, 0);
                    }
                    if (numNewWorkItems > 0)
                    {
                        std::vector<int64_t> packed;
                        for (auto const& infReq : inferenceRequests)
                        {
                            auto vpacked = infReq->serialize();
                            packed.push_back(static_cast<int64_t>(vpacked.size()));
                            packed.insert(
                                packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                        }
                        mCommTensorParallel->bcast(packed, 0);
                    }
                }
            }
            else
            {
                // subordinate ranks hang until master rank sends work
                if (mWorldConfig.isFirstPipelineParallelRank())
                {
                    int64_t numNewWorkItems = 0;
                    mCommTensorParallel->bcast(&numNewWorkItems, 1, mpi::MpiType::kINT64, 0);
                    if (numNewWorkItems > 0)
                    {
                        std::vector<int64_t> packed;
                        mCommTensorParallel->bcast(packed, 0);
                        int64_t* packed_ptr = packed.data();
                        for (int64_t count = 0; count < numNewWorkItems; ++count)
                        {
                            int64_t n = *(packed_ptr++);
                            auto infReq = InferenceRequest::deserialize(packed_ptr);
                            packed_ptr += n;
                            inferenceRequests.emplace_back(infReq);
                        }
                    }
                }
                else
                {
                    auto const peer = mWorldConfig.getPipelineParallelRank() - 1;
                    inferenceRequestsRecv(mCommPipelineParallel, inferenceRequests, peer);
                }
            }
            if (!mWorldConfig.isLastPipelineParallelRank())
            {
                auto const peer = mWorldConfig.getPipelineParallelRank() + 1;
                mInferReqSyncSndHdl
                    = std::make_shared<InferenceRequestsSyncSend>(mCommPipelineParallel, inferenceRequests, peer);
            }
        }
        return inferenceRequests;
    }

    void sendResponse(uint64_t requestId, [[maybe_unused]] std::list<NamedTensor> const& response_tensors,
        bool final_response, [[maybe_unused]] std::string const& errMsg)
    {
        // `response_tensors` contains `outputIds, sequenceLength, [contextLogits, generationLogits], logProbs,
        // cumLogProbs`. `contextLogits, generationLogits` are optional, only contained when `gather_context_logits` and
        // `gather_generation_logits` are enabled respectively. Or enable 'gather_all_token_logits' to enable both of
        // them.
        try
        {

            if (final_response)
            {
                mWorkItemsQueue.markFinished(requestId);
                mRecorder->recordEnd(requestId, response_tensors, !errMsg.empty());
                mActiveCount--;
            }
            else
            {
                if (errMsg.empty())
                {
                    mRecorder->recordToken(requestId);
                }
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("Failed to send response for requestId %lu\n%s", requestId, e.what());
        }
    }

private:
    std::shared_ptr<GptManager> mBatchManager;
    std::shared_ptr<Recorder> mRecorder;
    WorkItemsQueue mWorkItemsQueue;
    std::optional<uint64_t> mTerminateReqId;
    std::chrono::milliseconds mWaitSleep;
    std::optional<SizeType32> mStaticEmulatedBatchSize;
    std::chrono::milliseconds mBatchTimeout;
    std::atomic<std::chrono::steady_clock::time_point::duration> mBatchDeadline;
    std::atomic<uint64_t> mActiveCount;
    WorldConfig mWorldConfig;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommTensorParallel;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommPipelineParallel;
    std::shared_ptr<InferenceRequestsSyncSend> mInferReqSyncSndHdl;

}; // class GptServer

namespace
{

struct Sample
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t taskId;
};

using Samples = std::vector<Sample>;

Samples parseWorkloadJson(
    std::filesystem::path const& datasetPath, int maxNumSamples, std::optional<SizeType32> const maxPromptLen)
{
    auto constexpr allowExceptions = true;
    auto constexpr ignoreComments = true;
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(datasetPath), "File does not exist: %s", datasetPath.c_str());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ignoreComments);

    Samples samples;

    for (auto const& sample : json["samples"])
    {
        if (samples.size() >= maxNumSamples)
            break;
        int32_t taskId = sample.count("task_id") ? sample["task_id"].template get<int32_t>() : -1;
        auto input_ids(sample["input_ids"].template get<std::vector<int32_t>>());
        if (maxPromptLen && (input_ids.size() > maxPromptLen.value()))
        {
            input_ids.resize(maxPromptLen.value());
        }
        samples.emplace_back(Sample{std::move(input_ids), sample["output_len"], taskId});
    }
    return samples;
}

std::vector<double> generateRandomExponentialValues(int count, float lambda, int seed)
{
    // Set a constant seed for reproducibility
    std::mt19937 gen(seed);

    // Create an exponential distribution object
    std::exponential_distribution<double> distribution(lambda);

    // Generate random numbers from the exponential distribution
    std::vector<double> randomValues;
    for (int i = 0; i < count; ++i)
    {
        double randomValue = distribution(gen);
        randomValues.push_back(randomValue);
    }

    return randomValues;
}

std::vector<double> computeTimeDelays(BenchmarkParams const& benchmarkParams, int numDelays)
{
    std::vector<double> timeDelays;
    if (benchmarkParams.requestRate.has_value() && benchmarkParams.requestRate.value() > 0.0)
    {
        if (benchmarkParams.enableExpDelays)
        {
            timeDelays = generateRandomExponentialValues(
                numDelays, benchmarkParams.requestRate.value(), benchmarkParams.randomSeed);
        }
        else
        {
            timeDelays.assign(numDelays, 1.0 / benchmarkParams.requestRate.value());
        }
    }
    else
    {
        timeDelays.assign(numDelays, 0.0);
    }

    return timeDelays;
}

std::shared_ptr<InferenceRequest> makeRequest(std::uint64_t reqId, Sample const& sample, bool streaming,
    ITensor::SharedPtr const& beamWidthTensor, ITensor::SharedPtr const& eosId, ITensor::SharedPtr const& padId,
    BufferManager const& bufferManager, ITensor::SharedPtr const& returnContextLogits = nullptr,
    ITensor::SharedPtr const& returnGenerationLogits = nullptr, ITensor::SharedPtr const& loraWeights = nullptr,
    ITensor::SharedPtr const& loraConfig = nullptr)
{
    auto request = std::make_shared<InferenceRequest>(reqId);
    auto const& inputIds = sample.inputIds;
    request->setInputIds(bufferManager.copyFrom(
        inputIds, ITensor::makeShape({static_cast<SizeType32>(inputIds.size())}), MemoryType::kCPU));
    auto const requestOutputLen = sample.outputLen;
    request->setMaxNewTokens(bufferManager.copyFrom(&requestOutputLen, ITensor::makeShape({1, 1}), MemoryType::kCPU));
    request->setBeamWidth(beamWidthTensor);
    if (eosId != nullptr)
    {
        request->setEndId(eosId);
    }
    if (padId != nullptr)
    {
        request->setPadId(padId);
    }
    if (returnContextLogits)
    {
        request->setReturnContextLogits(returnContextLogits);
    }
    if (returnGenerationLogits)
    {
        request->setReturnGenerationLogits(returnGenerationLogits);
    }
    if (sample.taskId >= 0)
    {
        uint64_t taskId = static_cast<uint64_t>(sample.taskId);
        request->setLoraTaskId(bufferManager.copyFrom(&taskId, ITensor::makeShape({1}), MemoryType::kPINNED));
    }
    if (loraWeights)
    {
        request->setLoraWeights(loraWeights);
    }
    if (loraConfig)
    {
        request->setLoraConfig(loraConfig);
    }
    if (streaming)
    {
        request->setIsStreaming(true);
    }
    return request;
}

texec::Request makeExecutorRequest(Sample const& sample, SizeType32 const& beamWidth,
    std::optional<SizeType32> const& eosId, std::optional<SizeType32> const& padId, bool streaming = false,
    bool const& returnContextLogits = false, bool const& returnGenerationLogits = false,
    std::optional<texec::LoraConfig> const& loraConfig = std::nullopt)
{
    auto samplingConfig = texec::SamplingConfig{beamWidth};
    auto outputConfig = texec::OutputConfig{false, returnContextLogits, returnGenerationLogits, false};
    return texec::Request(sample.inputIds, sample.outputLen, streaming, samplingConfig, outputConfig, eosId, padId,
        std::nullopt, // badWords
        std::nullopt, // stopWords
        std::nullopt, // embeddingBias
        std::nullopt, // speculativeDecoding
        std::nullopt, // pTuning
        loraConfig);
}

void benchmarkGptManager(std::filesystem::path const& engineDir, TrtGptModelType modelType,
    std::string const& datasetPath, std::string const& opCsvFile, int maxNumSamples, int beamWidth, int warmUp,
    std::optional<TokenIdType> const& eosId, std::optional<TokenIdType> const& padId,
    BenchmarkParams const& benchmarkParams, texec::CapacitySchedulerPolicy capacitySchedulerPolicy,
    std::chrono::milliseconds waitSleep, bool returnContextLogits, bool returnGenerationLogits,
    std::optional<SizeType32> const staticEmulatedBatchSize,
    std::optional<std::chrono::milliseconds> const batchTimeout, bool logIterationData, bool excludeInputInOutput,
    std::string const& responsesJsonFile, std::optional<SizeType32> const maxPromptLen, bool dumpProfile)
{
    TrtGptModelOptionalParams optionalParams;

    if (benchmarkParams.maxTokensInPagedKvCache)
    {
        optionalParams.kvCacheConfig.maxTokens = benchmarkParams.maxTokensInPagedKvCache;
    }
    if (benchmarkParams.freeGpuMemoryFraction)
    {
        optionalParams.kvCacheConfig.freeGpuMemoryFraction = benchmarkParams.freeGpuMemoryFraction;
    }
    if (benchmarkParams.maxAttentionWindow)
    {
        optionalParams.kvCacheConfig.maxAttentionWindow = benchmarkParams.maxAttentionWindow;
    }
    optionalParams.kvCacheConfig.enableBlockReuse = benchmarkParams.enableBlockReuse;
    optionalParams.enableChunkedContext = benchmarkParams.enableChunkedContext;
    optionalParams.enableTrtOverlap = benchmarkParams.enableTrtOverlap;
    optionalParams.peftCacheManagerConfig.hostCacheSize = benchmarkParams.loraHostCacheSize;
    optionalParams.peftCacheManagerConfig.numDeviceModuleLayer = benchmarkParams.loraDeviceNumModLayers;
    optionalParams.peftCacheManagerConfig.numPutWorkers = 4;
    optionalParams.peftCacheManagerConfig.numEnsureWorkers = 4;
    optionalParams.peftCacheManagerConfig.numCopyStreams = 4;
    optionalParams.kvCacheConfig.hostCacheSize = benchmarkParams.kvHostCacheSize;
    optionalParams.kvCacheConfig.onboardBlocks = benchmarkParams.kvOnboardBlocks;
    optionalParams.gpuWeightsPercent = benchmarkParams.gpuWeightsPercent;
    optionalParams.maxBeamWidth = beamWidth;
    optionalParams.maxBatchSize = benchmarkParams.maxBatchSize;
    optionalParams.maxNumTokens = benchmarkParams.maxNumTokens;
    optionalParams.schedulerConfig = texec::SchedulerConfig{capacitySchedulerPolicy};
    optionalParams.decodingConfig = texec::DecodingConfig(
        benchmarkParams.medusaChoices.has_value() ? texec::DecodingMode::Medusa() : texec::DecodingMode::Auto(),
        std::nullopt, benchmarkParams.medusaChoices);

    auto const jsonConfig = GptJsonConfig::parse(engineDir / "config.json");
    auto const worldConfig = WorldConfig::mpi(jsonConfig.getGpusPerNode(), jsonConfig.getTensorParallelism(),
        jsonConfig.getPipelineParallelism(), optionalParams.deviceIds);

    BufferManager bufferManager{std::make_shared<CudaStream>()}; // the stream is not used

    ITensor::SharedPtr beamWidthTensor{
        bufferManager.copyFrom(&beamWidth, ITensor::makeShape({1}), MemoryType::kPINNED)};

    // Load dataset
    auto const samples = parseWorkloadJson(datasetPath, maxNumSamples, maxPromptLen);
    auto const numSamples = samples.size();

    auto recorder = std::make_shared<Recorder>(
        opCsvFile, benchmarkParams.streaming, beamWidth, responsesJsonFile, excludeInputInOutput);
    uint64_t terminateReqId = numSamples + 1;
    auto gptServer = std::make_shared<GptServer>(engineDir, modelType, optionalParams, recorder, terminateReqId,
        waitSleep, staticEmulatedBatchSize, batchTimeout, logIterationData, excludeInputInOutput);

    ITensor::SharedPtr eosIdTensor{
        eosId ? bufferManager.copyFrom(&eosId.value(), ITensor::makeShape({1}), MemoryType::kPINNED) : nullptr};
    ITensor::SharedPtr padIdTensor{
        padId ? bufferManager.copyFrom(&padId.value(), ITensor::makeShape({1}), MemoryType::kPINNED) : nullptr};

    ITensor::SharedPtr returnContextLogitsFlagTensor{returnContextLogits
            ? bufferManager.copyFrom(&returnContextLogits, ITensor::makeShape({1}), MemoryType::kPINNED)
            : nullptr};

    ITensor::SharedPtr returnGenerationLogitsFlagTensor{returnGenerationLogits
            ? bufferManager.copyFrom(&returnGenerationLogits, ITensor::makeShape({1}), MemoryType::kPINNED)
            : nullptr};

    if (worldConfig.getRank() == 0)
    {
        if (benchmarkParams.loraDir)
        {
            auto startLoraLoad = std::chrono::steady_clock::now();
            LoraLib loras(benchmarkParams.loraDir.value());
            SizeType32 reqId = 0;
            gptServer->resetBatchDeadline();
            for (auto const& [taskId, p] : loras.getLoras())
            {
                reqId++;
                if (reqId == terminateReqId)
                {
                    reqId++;
                }
                Sample s{std::vector<int32_t>{1, 2, 3, 4, 5}, 1, static_cast<int32_t>(taskId)};
                auto r = makeRequest(reqId, s, benchmarkParams.streaming, beamWidthTensor, eosIdTensor, padIdTensor,
                    bufferManager, nullptr, nullptr, p.first, p.second);
                gptServer->enqueue(r);
            }
            gptServer->waitForEmpty();
            auto endLoraLoad = std::chrono::steady_clock::now();
            printf("[BENCHMARK] time to preload LoRAs(ms) %.2f\n",
                std::chrono::duration<float, std::milli>(endLoraLoad - startLoraLoad).count());
        }

        // Warm up
        gptServer->resetBatchDeadline();
        SizeType32 reqId = 0;
        for (auto i = 0; i < warmUp; ++i)
        {
            ++reqId;
            if (i == terminateReqId)
                ++reqId;
            auto request = makeRequest(
                reqId, samples[0], benchmarkParams.streaming, beamWidthTensor, eosIdTensor, padIdTensor, bufferManager);
            gptServer->enqueue(request);
        }
        gptServer->waitForEmpty();

        // Time delay
        auto timeDelays = computeTimeDelays(benchmarkParams, numSamples - 1);

        // Benchmark
        recorder->initialize();
        gptServer->resetBatchDeadline();

        for (std::size_t i = 0; i < numSamples; ++i)
        {
            auto request = makeRequest(i + 1, samples[i], benchmarkParams.streaming, beamWidthTensor, eosIdTensor,
                padIdTensor, bufferManager, returnContextLogitsFlagTensor, returnGenerationLogitsFlagTensor);
            gptServer->enqueue(request);

            if (i < numSamples - 1)
            {
                auto delayInMs = static_cast<int>(timeDelays.at(i) * 1000);
                std::chrono::milliseconds delay(delayInMs);
                std::this_thread::sleep_for(delay);
            }
        }
        gptServer->waitForEmpty();
        recorder->finalize();
        recorder->calculateMetrics();
        recorder->report();
        recorder->writeOpMetricsToCsv();
        recorder->dumpResponseSeqs();
        if (dumpProfile)
        {
            // Do per-layer profiling after normal benchmarking to avoid introducing perf overhead.
            gptServer->resetBatchDeadline();
            gptServer->setLayerProfiler();
            for (std::size_t i = 0; i < numSamples; ++i)
            {
                auto request = makeRequest(i + 1, samples[i], benchmarkParams.streaming, beamWidthTensor, eosIdTensor,
                    padIdTensor, bufferManager, returnContextLogitsFlagTensor, returnGenerationLogitsFlagTensor);
                gptServer->enqueue(request);
            }
            gptServer->waitForEmpty();
            if (worldConfig.getRank() == 0)
            {
                printf("[BENCHMARK] Per layer performance profile\n%s\n", gptServer->getLayerProfileInfo().c_str());
            }
        }
        // Send terminateReqId to terminate servers on all ranks
        // Server on rank 0 will broadcast the terminate signal to other servers on multi-GPU cases
        gptServer->enqueue(std::make_shared<InferenceRequest>(terminateReqId));
    }
    // Wait until benchmarking is done and batch manager is terminated
    gptServer->waitBatchManager();
}

void benchmarkExecutor(std::filesystem::path const& engineDir, TrtGptModelType modelType,
    std::string const& datasetPath, std::string const& opCsvFile, int maxNumSamples, int beamWidth, int warmUp,
    std::optional<int32_t> const& eosId, std::optional<int32_t> const& padId, BenchmarkParams const& benchmarkParams,
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy, std::chrono::milliseconds waitSleep,
    bool returnContextLogits, bool returnGenerationLogits, std::optional<int> const staticEmulatedBatchSize,
    bool logIterationData, std::optional<SizeType32> const maxPromptLen)
{
    auto const& world = tensorrt_llm::mpi::MpiComm::world();
    auto worldRank = world.getRank();

    // Load dataset
    auto const samples = parseWorkloadJson(datasetPath, maxNumSamples, maxPromptLen);
    auto const numSamples = samples.size();

    auto recorder = std::make_shared<Recorder>(opCsvFile, benchmarkParams.streaming, beamWidth);

    auto executorServer = std::make_shared<ExecutorServer>(engineDir, modelType, beamWidth, capacitySchedulerPolicy,
        benchmarkParams, recorder, waitSleep, staticEmulatedBatchSize, logIterationData);

    if (worldRank == 0)
    {
        if (benchmarkParams.loraDir)
        {
            auto startLoraLoad = std::chrono::steady_clock::now();
            LoraLib loras(benchmarkParams.loraDir.value());
            std::vector<texec::Request> requests;
            for (auto& [taskId, p] : loras.getLoras())
            {
                // squeeze lora configs and weights since LoraConfig requires them to be 2D tensors
                p.first->squeeze(0);
                p.second->squeeze(0);
                texec::LoraConfig loraConfig(
                    taskId, texec::detail::ofITensor(p.first), texec::detail::ofITensor(p.second));
                Sample s{std::vector<int32_t>{1, 2, 3, 4, 5}, 1, static_cast<int32_t>(taskId)};
                requests.emplace_back(makeExecutorRequest(s, beamWidth, eosId, padId, false, false, false, loraConfig));
            }
            executorServer->enqueue(std::move(requests), true);
            executorServer->waitForResponses(loras.getLoras().size(), true);
            auto endLoraLoad = std::chrono::steady_clock::now();
            printf("[BENCHMARK] time to preload LoRAs(ms) %.2f\n",
                std::chrono::duration<float, std::milli>(endLoraLoad - startLoraLoad).count());
        }
        // Warm up
        {
            std::vector<texec::Request> requests;
            for (auto i = 0; i < warmUp; ++i)
            {
                requests.emplace_back(makeExecutorRequest(samples[0], beamWidth, eosId, padId,
                    benchmarkParams.streaming, returnContextLogits, returnGenerationLogits));
            }
            executorServer->enqueue(std::move(requests), true);
            executorServer->waitForResponses(warmUp, true);
        }

        // Benchmark
        {
            auto timeDelays = computeTimeDelays(benchmarkParams, numSamples - 1);

            // Create requests
            recorder->initialize();
            std::vector<texec::Request> requests;

            for (std::size_t i = 0; i < numSamples; ++i)
            {
                std::optional<texec::LoraConfig> loraConfig;
                if (samples[i].taskId >= 0)
                {
                    loraConfig = texec::LoraConfig(samples[i].taskId);
                }
                requests.emplace_back(makeExecutorRequest(samples[i], beamWidth, eosId, padId,
                    benchmarkParams.streaming, returnContextLogits, returnGenerationLogits, loraConfig));
            }

            bool hasDelay
                = std::any_of(timeDelays.begin(), timeDelays.end(), [](auto const& delay) { return delay > 0.0; });
            if (hasDelay && staticEmulatedBatchSize)
            {
                TLLM_THROW("Executor benchmark doesn't support delays with emulated static batch sizes");
            }

            if (!hasDelay)
            {
                if (!staticEmulatedBatchSize)
                {
                    executorServer->enqueue(std::move(requests));
                    executorServer->waitForResponses(numSamples);
                }
                else
                {
                    SizeType32 numRequests = requests.size();
                    SizeType32 maxBatchSize = staticEmulatedBatchSize.value();
                    for (SizeType32 req = 0; req < numRequests; req += maxBatchSize)
                    {
                        auto batchSize = std::min(maxBatchSize, numRequests - req);

                        std::vector<texec::Request> requestsBatch(std::make_move_iterator(requests.begin() + req),
                            std::make_move_iterator(requests.begin() + req + batchSize));
                        // Enqueue in batches
                        executorServer->enqueue(std::move(requestsBatch));
                        // Wait for current batch to be done
                        executorServer->waitForResponses(batchSize);
                    }
                }
            }
            else
            {
                // Launch a thread that will wait for responses
                std::thread waitThread(
                    [numSamples, executorServer]() { executorServer->waitForResponses(numSamples); });
                // Enqueue requests one by one
                for (std::size_t i = 0; i < numSamples; ++i)
                {
                    executorServer->enqueue({std::move(requests.at(i))});
                    if (i < numSamples - 1)
                    {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(static_cast<int>(timeDelays.at(i) * 1000)));
                    }
                }
                waitThread.join();
            }
        }
        recorder->finalize();
        recorder->calculateMetrics();
        recorder->report();
        recorder->writeOpMetricsToCsv();
        // Send terminateReqId to terminate servers on all ranks
        // Sever on rank 0 will broadcast the terminate signal to other servers on multi-GPU cases
        // gptServer->enqueue(std::make_shared<InferenceRequest>(terminateReqId));
    }
}

std::vector<std::vector<SizeType32>> parseVectorOfVectors(std::string const& input)
{
    std::vector<std::vector<SizeType32>> result;
    std::regex outer_regex(R"(\[(.*?)\])");
    std::regex inner_regex(R"(\d+)");
    auto outer_begin = std::sregex_iterator(input.begin(), input.end(), outer_regex);
    auto outer_end = std::sregex_iterator();

    for (std::sregex_iterator i = outer_begin; i != outer_end; ++i)
    {
        std::smatch match = *i;
        std::string inner_str = match.str(1);
        std::vector<int> inner_vec;
        auto inner_begin = std::sregex_iterator(inner_str.begin(), inner_str.end(), inner_regex);
        auto inner_end = std::sregex_iterator();

        for (std::sregex_iterator j = inner_begin; j != inner_end; ++j)
        {
            std::smatch inner_match = *j;
            inner_vec.push_back(std::stoi(inner_match.str()));
        }
        result.push_back(inner_vec);
    }
    return result;
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT-LLM BatchManager Benchmark", "TensorRT-LLM BatchManager Benchmark for GPT and GPT-like models.");
    options.add_options()("h,help", "Print usage");
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()(
        "api", "API type: gptManager or executor.", cxxopts::value<std::string>()->default_value("executor"));
    options.add_options()("type", "Batching type: IFB, UIFB (unfused IFB) or V1 (non-IFB) batching.",
        cxxopts::value<std::string>()->default_value("IFB"));
    options.add_options()("dataset", "Dataset that is used for benchmarking BatchManager.",
        cxxopts::value<std::string>()->default_value(""));
    options.add_options()(
        "output_csv", "Write output metrics to CSV", cxxopts::value<std::string>()->default_value(""));
    options.add_options()("max_num_samples", "maximum number of samples to use from dataset/generate",
        cxxopts::value<int>()->default_value("100000"));
    options.add_options()(
        "beam_width", "Specify beam width you want to benchmark.", cxxopts::value<int>()->default_value("1"));
    options.add_options()(
        "warm_up", "Specify warm up iterations before benchmark starts.", cxxopts::value<int>()->default_value("2"));
    options.add_options()(
        "eos_id", "Specify the end-of-sequence token id.", cxxopts::value<TokenIdType>()->default_value("-1"));
    options.add_options()("pad_id", "Specify the padding token id.", cxxopts::value<TokenIdType>());
    options.add_options()("max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>());
    options.add_options()("max_attention_window", "Max KV cache length per sequence", cxxopts::value<int>());
    options.add_options()(
        "random_seed", "integer random seed for exponential time delays.", cxxopts::value<int>()->default_value("420"));
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());
    options.add_options()("request_rate",
        "request rate in reqs/sec. Skipping this arg or negative value will trigger offline/0-delay.",
        cxxopts::value<float>());
    options.add_options()("max_batch_size", "The max runtime batch size when benchmarking", cxxopts::value<int>());
    options.add_options()(
        "max_num_tokens", "The max runtime number of tokens per batch when benchmarking", cxxopts::value<int>());
    options.add_options()("enable_trt_overlap", "Overlap TRT context preparation and execution",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_exp_delays", "Enables exponential delay distr to mimic real world request arrival",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("streaming", "Operate in streaming mode", cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "enable_kv_cache_reuse", "Enables the KV cache reuse.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_chunked_context", "Whether to enable context chunking.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "return_context_logits", "Whether to return context logits.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("return_generation_logits", "Whether to return generation logits.",
        cxxopts::value<bool>()->default_value("false"));

    options.add_options()("scheduler_policy", "Choose scheduler policy between max_utilization/guaranteed_no_evict.",
        cxxopts::value<std::string>()->default_value("guaranteed_no_evict"));

    options.add_options()("first_batch_delay",
        "Delay before submitting the first batch of requests. This can be used to increase the size of the first "
        "batch.",
        cxxopts::value<int32_t>());
    options.add_options()("static_emulated_batch_size",
        "Emulate static batching performance with the provided batch size.", cxxopts::value<SizeType32>());
    options.add_options()("static_emulated_timeout",
        "Timeout (ms) before launching a partial batch in emulated static batching mode",
        cxxopts::value<int32_t>()->default_value("500"));
    options.add_options()("log_level", "Choose log level between verbose/info/warning/error/internal_error.",
        cxxopts::value<std::string>()->default_value("error"));
    options.add_options()("log_iteration_data", "On each decoder iteration, print batch state metadata.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("wait_sleep", "Specify how many milliseconds to sleep each iteration of waitForEmpty loop.",
        cxxopts::value<int>()->default_value("25"));
    options.add_options()("lora_dir", "Directory containing LoRAs", cxxopts::value<std::string>()->default_value(""));
    options.add_options()("lora_host_cache_bytes", "LoRA host cache memory in bytes", cxxopts::value<size_t>());
    options.add_options()("lora_num_device_mod_layers", "LoRA number 1d cache rows", cxxopts::value<int>());
    options.add_options()("kv_host_cache_bytes",
        "Size of secondary memory pool used for offloading kv cache blocks (in bytes).",
        cxxopts::value<size_t>()->default_value("0"));
    options.add_options()("kv_dont_onboard_blocks",
        "If offloaded blocks should be onboarded to primary memory before reuse",
        cxxopts::value<bool>()->default_value("false"));

    options.add_options()("exclude_input_in_output_seq",
        "When enabled, GptManager will exclude the input sequence from output. (Only works if --api is gptManager)",
        cxxopts::value<bool>());

    options.add_options()("responses_json_file",
        "When specified, dumps the responses to JSON file. (only works if --api is gptManager)",
        cxxopts::value<std::string>()->default_value(""));

    options.add_options()(
        "max_prompt_len", "Truncate all prompts from dataset to the length specified.", cxxopts::value<SizeType32>());

    options.add_options()("dump_profile", "Print profile information per layer.", cxxopts::value<bool>());
    options.add_options()("gpu_weights_percent",
        "Specify the percentage of weights that reside on GPU (from 0.0 to 1.0).",
        cxxopts::value<float>()->default_value("1.0"));
    options.add_options()(
        "medusa_choices", "Medusa choices in the format of [[0], [0, 1], [0, 0, 1]]", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Argument: Engine directory
    if (!result.count("engine_dir"))
    {
        std::cout << options.help() << std::endl;
        TLLM_LOG_ERROR("Please specify engine directory.");
        return 1;
    }

    // Argument: API
    auto const api = result["api"].as<std::string>();

    // Argument: Batching Type
    auto const type = result["type"].as<std::string>();
    TrtGptModelType modelType{TrtGptModelType::V1};
    if (type == "V1")
    {
        modelType = TrtGptModelType::V1;
    }
    else if (type == "UIFB")
    {
        modelType = TrtGptModelType::InflightBatching;
    }
    else if (type == "IFB")
    {
        modelType = TrtGptModelType::InflightFusedBatching;
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected batching type: %s", type.c_str());
        return 1;
    }

    // Argument: Dataset
    auto const datasetPath = result["dataset"].as<std::string>();
    auto const maxNumSamples = result["max_num_samples"].as<int>();

    // Argument: Output metrics CSV
    auto const opCsvFile = result["output_csv"].as<std::string>();

    // Argument: beam width
    auto const beamWidth = result["beam_width"].as<int>();

    // Argument: wait_sleep
    auto const waitSleep = std::chrono::milliseconds(result["wait_sleep"].as<int>());
    BenchmarkParams benchmarkParams;

    // Argument: Max tokens in paged K-V Cache
    if (result.count("max_tokens_in_paged_kvcache"))
    {
        benchmarkParams.maxTokensInPagedKvCache = result["max_tokens_in_paged_kvcache"].as<int>();
    }

    // Argument: Max KV cache length
    if (result.count("max_attention_window"))
    {
        benchmarkParams.maxAttentionWindow = result["max_attention_window"].as<int>();
    }

    if (result.count("random_seed"))
    {
        benchmarkParams.randomSeed = result["random_seed"].as<int>();
    }

    // Argument: K-V Cache Free Gpu Mem Fraction
    if (result.count("kv_cache_free_gpu_mem_fraction"))
    {
        benchmarkParams.freeGpuMemoryFraction = result["kv_cache_free_gpu_mem_fraction"].as<float>();
    }
    // Argument: Enable TRT overlap
    benchmarkParams.enableTrtOverlap = result["enable_trt_overlap"].as<bool>();

    // Argument: Enable KV cache reuse
    benchmarkParams.enableBlockReuse = result["enable_kv_cache_reuse"].as<bool>();

    // Argument: streaming
    benchmarkParams.streaming = result["streaming"].as<bool>();

    // Argument: request rate
    if (result.count("request_rate"))
    {
        benchmarkParams.requestRate = result["request_rate"].as<float>();
    }

    // Argument: request rate
    if (result.count("max_batch_size"))
    {
        benchmarkParams.maxBatchSize = result["max_batch_size"].as<int>();
    }

    // Argument: request rate
    if (result.count("max_num_tokens"))
    {
        benchmarkParams.maxNumTokens = result["max_num_tokens"].as<int>();
    }

    benchmarkParams.enableExpDelays = result["enable_exp_delays"].as<bool>();

    // Argument: Enable batch stats output
    bool logIterationData = result["log_iteration_data"].as<bool>();

    // Argument: Enable chunked context
    benchmarkParams.enableChunkedContext = result["enable_chunked_context"].as<bool>();

    // Argument: Enable return context logits
    bool returnContextLogits = result["return_context_logits"].as<bool>();

    // Argument: Enable return context logits
    bool returnGenerationLogits = result["return_generation_logits"].as<bool>();

    if (result.count("lora_dir"))
    {
        benchmarkParams.loraDir = result["lora_dir"].as<std::string>();
    }
    if (result.count("lora_host_cache_bytes"))
    {
        benchmarkParams.loraHostCacheSize = result["lora_host_cache_bytes"].as<size_t>();
    }
    if (result.count("lora_num_device_mod_layers"))
    {
        benchmarkParams.loraDeviceNumModLayers = result["lora_num_device_mod_layers"].as<SizeType32>();
    }

    // Argument: How many KV cache blocks (as fraction of number of GPU kv cache blocks).
    benchmarkParams.kvHostCacheSize = result["kv_host_cache_bytes"].as<size_t>();

    // Argument: If offloaded blocks should be onboarded to primary memory before they are reused.
    benchmarkParams.kvOnboardBlocks = !result["kv_dont_onboard_blocks"].as<bool>();

    // Argument: Medusa choices for the Medusa speculative decoding.
    if (result.count("medusa_choices"))
    {
        benchmarkParams.medusaChoices = parseVectorOfVectors(result["medusa_choices"].as<std::string>());
    }

    std::optional<TokenIdType> padId;
    // Argument: Padding token id
    if (result.count("pad_id"))
    {
        padId = result["pad_id"].as<TokenIdType>();
    }

    // Argument: End-of-sentence token id
    std::optional<TokenIdType> eosId = result["eos_id"].as<TokenIdType>();

    std::optional<std::chrono::milliseconds> batchTimeout;
    // Argument: first_batch_delay
    if (result.count("first_batch_delay"))
    {
        batchTimeout = std::chrono::milliseconds{result["first_batch_delay"].as<int32_t>()};
    }

    std::optional<SizeType32> staticEmulatedBatchSize;
    // Argument: Static emulated batch size
    if (result.count("static_emulated_batch_size"))
    {
        staticEmulatedBatchSize = result["static_emulated_batch_size"].as<SizeType32>();

        batchTimeout = std::chrono::milliseconds{result["static_emulated_timeout"].as<int32_t>()};
    }

    // Argument: Scheduler policy
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy;
    auto const capacitySchedulerPolicyArg = result["scheduler_policy"].as<std::string>();
    if (capacitySchedulerPolicyArg == "max_utilization")
    {
        capacitySchedulerPolicy = texec::CapacitySchedulerPolicy::kMAX_UTILIZATION;
    }
    else if (capacitySchedulerPolicyArg == "guaranteed_no_evict")
    {
        capacitySchedulerPolicy = texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected scheduler policy: " + capacitySchedulerPolicyArg);
        return 1;
    }

    // Argument: max_prompt_len
    std::optional<SizeType32> maxPromptLen;
    if (result.count("max_prompt_len"))
    {
        maxPromptLen = result["max_prompt_len"].as<SizeType32>();
    }

    // Argument: GPU weights percentage
    std::istringstream ssGpuPercentArg;
    auto gpuWeightsPercent = result["gpu_weights_percent"].as<float>();
    if (gpuWeightsPercent < 0 || gpuWeightsPercent > 1)
    {
        TLLM_LOG_ERROR("--gpu_weights_percent must be between 0.0 and 1.0 but got: %f", gpuWeightsPercent);
        return 1;
    }
    benchmarkParams.gpuWeightsPercent = gpuWeightsPercent;

    // Argument: Log level
    auto logger = std::make_shared<TllmLogger>();
    auto const logLevel = result["log_level"].as<std::string>();
    if (logLevel == "verbose")
    {
        logger->setLevel(trt::ILogger::Severity::kVERBOSE);
    }
    else if (logLevel == "info")
    {
        logger->setLevel(trt::ILogger::Severity::kINFO);
    }
    else if (logLevel == "warning")
    {
        logger->setLevel(trt::ILogger::Severity::kWARNING);
    }
    else if (logLevel == "error")
    {
        logger->setLevel(trt::ILogger::Severity::kERROR);
    }
    else if (logLevel == "internal_error")
    {
        logger->setLevel(trt::ILogger::Severity::kINTERNAL_ERROR);
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected log level: " + logLevel);
        return 1;
    }

    // Argument: dump profile
    bool dumpProfile = result["dump_profile"].as<bool>();

    initTrtLlmPlugins(logger.get());

    if (api == "gptManager")
    {
        try
        {
            benchmarkGptManager(result["engine_dir"].as<std::string>(), modelType, datasetPath, opCsvFile,
                maxNumSamples, beamWidth, result["warm_up"].as<int>(), eosId, padId, benchmarkParams,
                capacitySchedulerPolicy, waitSleep, returnContextLogits, returnGenerationLogits,
                staticEmulatedBatchSize, batchTimeout, logIterationData,
                result["exclude_input_in_output_seq"].as<bool>(), result["responses_json_file"].as<std::string>(),
                maxPromptLen, dumpProfile);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(e.what());
            return 1;
        }
    }
    else if (api == "executor")
    {
        try
        {
            benchmarkExecutor(result["engine_dir"].as<std::string>(), modelType, datasetPath, opCsvFile, maxNumSamples,
                beamWidth, result["warm_up"].as<int>(), eosId, padId, benchmarkParams, capacitySchedulerPolicy,
                waitSleep, returnContextLogits, returnGenerationLogits, staticEmulatedBatchSize, logIterationData,
                maxPromptLen);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(e.what());
            return 1;
        }
    }
    else
    {
        TLLM_LOG_ERROR("api parameter must be gptManager or executor");
        return 1;
    }

    return 0;
}
