/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/batch_manager/NamedTensor.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/gptSession.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <chrono>
#include <cxxopts.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::mpi;

namespace tc = tensorrt_llm::common;
namespace trt = nvinfer1;

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
public:
    WorkItem(std::shared_ptr<InferenceRequest> ir, uint64_t RequestId)
        : mInferenceRequest(ir)
        , mRequestId(RequestId)
    {
    }

    ~WorkItem() {}

    uint64_t requestId() const
    {
        return mRequestId;
    }

    std::shared_ptr<InferenceRequest> getInferenceRequest() const
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
        std::lock_guard<std::mutex> lk(mMutex);
        mPendingWorkItems.clear();
        mPendingWorkItemsReqIds.clear();
        mInProgressWorkItems.clear();
    }

    // Note: this function only be called under a lock
    bool hasInProgressReqId(const uint64_t reqId) const
    {
        return (mInProgressWorkItems.find(reqId) != mInProgressWorkItems.end());
    }

    // Note: this function only be called under a lock
    bool hasPendingReqId(const uint64_t reqId) const
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
        std::lock_guard<std::mutex> lk(mMutex);
        if (hasInProgressReqId(requestId) || hasPendingReqId(requestId))
        {
            std::string errStr
                = "requestId " + std::to_string(requestId) + " is already in progress, request is ignored.";
            throw std::runtime_error(errStr);
        }
        else
        {
            auto workItem = std::make_shared<WorkItem>(request, requestId);
            mPendingWorkItems.push_back(workItem);
            mPendingWorkItemsReqIds.insert(workItem->requestId());
        }
    }

    /// @brief Get a new work item from the queue, and move it to the list of
    /// in progress work items if it hasn't been stopped
    /// @return A tuple of the workItem and a boolean flag indicating if the work item
    /// has been marked in progress
    std::tuple<std::shared_ptr<WorkItem>, bool> pop()
    {
        std::lock_guard<std::mutex> lk(mMutex);

        auto workItem = mPendingWorkItems.front();
        mPendingWorkItems.pop_front();
        mPendingWorkItemsReqIds.erase(workItem->requestId());

        bool markedInProgress;
        mInProgressWorkItems.emplace(std::make_pair(workItem->requestId(), workItem));
        markedInProgress = true;

        return {workItem, markedInProgress};
    }

    size_t numPendingWorkItems() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mPendingWorkItems.size();
    }

    size_t numInProgressWorkItems() const
    {
        std::lock_guard<std::mutex> lk(mMutex);
        return mInProgressWorkItems.size();
    }

    size_t size() const
    {
        return numPendingWorkItems() + numInProgressWorkItems();
    }

    /// @brief  Mark a request as being finished
    /// @param requestId
    void markFinished(const uint64_t requestId)
    {
        std::lock_guard<std::mutex> lk(mMutex);
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
    BenchInfo() {}

    BenchInfo(int _inputLength, int _outputLength, std::chrono::time_point<std::chrono::steady_clock> _start)
        : inputLength(_inputLength)
        , outputLength(_outputLength)
        , start(_start)
    {
    }

    int inputLength;
    int outputLength;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;
    float latency; // millisecond
};

class Recorder
{
public:
    Recorder() {}

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
        const auto& input_ids_tensor = request->getInputTensor("input_ids");
        std::vector<int64_t> tensorShape(input_ids_tensor->getShape().nbDims);
        auto const inputLength = tensorShape[1];
        auto const [specified, outputLength]
            = request->getScalarValueFromTensor<int>("request_output_len", {1, 1}, false);
        assert(specified);
        auto const start = std::chrono::steady_clock::now();
        mRequestBenchInfos[requestId] = BenchInfo(inputLength, outputLength, start);
    }

    void recordEnd(uint64_t requestId)
    {
        mRequestBenchInfos[requestId].end = std::chrono::steady_clock::now();
        mRequestBenchInfos[requestId].latency = std::chrono::duration<float, std::milli>(
            mRequestBenchInfos[requestId].end - mRequestBenchInfos[requestId].start)
                                                    .count();
    }

    void calculateMetrics()
    {
        mNumSamples = mRequestBenchInfos.size();
        mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
        mSeqThroughput = mNumSamples / (mTotalLatency / 1000);
        mAvgSeqLatency = 0;
        int totalOutputTokens = 0;
        for (auto reqInfo : mRequestBenchInfos)
        {
            mAvgSeqLatency += reqInfo.second.latency;
            totalOutputTokens += reqInfo.second.outputLength;
        }
        mAvgSeqLatency /= mNumSamples;
        mTokenThroughput = totalOutputTokens / (mTotalLatency / 1000);
    }

    void report()
    {
        printf("[BENCHMARK] num_samples(ms) %d\n", mNumSamples);
        printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[BENCHMARK] avg_sequence_latency(ms) %.2f\n", mAvgSeqLatency);
        printf("[BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);
    }

private:
    std::unordered_map<uint64_t, BenchInfo> mRequestBenchInfos;

    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mEnd;
    int mNumSamples;
    float mTotalLatency;
    float mSeqThroughput;
    float mAvgSeqLatency;
    float mTokenThroughput;
}; // class Recorder

class GptServer
{
public:
    GptServer(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, int32_t maxBeamWidth,
        batch_scheduler::SchedulerPolicy schedulerPolicy, std::optional<int32_t> maxNumSequences,
        std::optional<int32_t> maxTokensInPagedKvCache, std::optional<float> kvCacheFreeGpuMemFraction,
        std::optional<bool> enableTrtOverlap, std::shared_ptr<Recorder> recorder,
        std::optional<uint64_t> terminateReqId)
    {
        const TrtGptModelOptionalParams& optionalParams = TrtGptModelOptionalParams(
            maxNumSequences, maxTokensInPagedKvCache, kvCacheFreeGpuMemFraction, enableTrtOverlap);
        mBatchManager = std::make_shared<GptManager>(
            trtEnginePath, modelType, maxBeamWidth, schedulerPolicy,
            [this](int max_num_requests) { return getInferenceRequests(max_num_requests); },
            [this](uint64_t requestId, std::list<NamedTensor> response_tensors, bool final_response,
                const std::string& errMsg)
            { return sendResponse(requestId, response_tensors, final_response, errMsg); },
            nullptr, nullptr, optionalParams, terminateReqId);
        mRecorder = recorder;
        mTerminateReqId = terminateReqId;
    }

    ~GptServer()
    {
        mWorkItemsQueue.clear();
    }

    void enqueue(std::vector<NamedTensor> tensors, uint64_t requestId, bool streaming)
    {
        // Create InferenceRequest from a set of tensors
        auto request = std::make_shared<InferenceRequest>(requestId);
        if (requestId == mTerminateReqId)
        {
            mWorkItemsQueue.push(request, requestId);
            return;
        }
        for (auto t : tensors)
        {
            request->emplaceInputTensor(t.name, std::move(t.tensor));
        }
        request->setIsStreaming(streaming);

        // Enqueue
        try
        {
            mRecorder->recordStart(request, requestId);
            mWorkItemsQueue.push(request, requestId);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error(e.what());
        }
        return;
    }

    void waitForEmpty() const
    {
        while (mWorkItemsQueue.size() > 0)
        {
        }
    }

    void waitBatchManager() const
    {
        mBatchManager->waitUntilTerminate();
    }

    // Return up to max_num_requests inference requests.
    std::list<std::shared_ptr<InferenceRequest>> getInferenceRequests(const int max_num_requests)
    {
        std::list<std::shared_ptr<InferenceRequest>> rval;
        if (max_num_requests > 0)
        {
            auto world_size = getCommWorldSize();
            auto rank = getCommWorldRank();
            if (rank == 0)
            {
                int64_t num_new_work_items = std::min(static_cast<int64_t>(mWorkItemsQueue.numPendingWorkItems()),
                    static_cast<int64_t>(max_num_requests));
                if (world_size > 1)
                {
                    bcast(&num_new_work_items, 1, MPI_TYPE_INT64_T, 0, COMM_WORLD);
                }

                if (num_new_work_items > 0)
                {
                    int count = 0;
                    while (count < num_new_work_items)
                    {
                        auto [workItem, markedInProgress] = mWorkItemsQueue.pop();

                        if (markedInProgress)
                        {
                            rval.emplace_back(workItem->getInferenceRequest());
                            count++;
                        }
                        else
                        {
                            std::string warnStr = std::string("request Id ") + std::to_string(workItem->requestId())
                                + std::string(" has been stopped. Request is ignored.");
                            TLLM_LOG_WARNING(warnStr);
                            sendResponse(workItem->requestId(), {}, true, warnStr);
                        }
                    }
                    if (world_size > 1)
                    {
                        std::vector<int64_t> packed;
                        for (auto ir : rval)
                        {
                            auto vpacked = ir->serialize();
                            packed.push_back(static_cast<int64_t>(vpacked.size()));
                            packed.insert(
                                packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                        }
                        bcast(packed, 0, COMM_WORLD);
                    }
                }
            }
            else
            {
                // subordinate ranks hang until master rank sends work
                int64_t num_new_work_items;
                bcast(&num_new_work_items, 1, MPI_TYPE_INT64_T, 0, COMM_WORLD);
                if (num_new_work_items > 0)
                {
                    std::vector<int64_t> packed;
                    bcast(packed, 0, COMM_WORLD);
                    int64_t* packed_ptr = packed.data();
                    for (int64_t count = 0; count < num_new_work_items; ++count)
                    {
                        int64_t n = *(packed_ptr++);
                        auto ir = InferenceRequest::deserialize(packed_ptr);
                        packed_ptr += n;
                        rval.emplace_back(ir);
                    }
                }
            }
        }
        return rval;
    }

    void sendResponse(uint64_t requestId, std::list<NamedTensor> const& response_tensors, bool final_response,
        const std::string& errMsg)
    {
        std::string errStr = std::string("Failed to send response for requestId: ") + std::to_string(requestId);
        try
        {
            if (final_response)
            {
                mWorkItemsQueue.markFinished(requestId);
                mRecorder->recordEnd(requestId);
            }
        }
        catch (const std::exception& e)
        {
            TLLM_LOG_ERROR(errStr);
        }
    }

private:
    std::shared_ptr<GptManager> mBatchManager;
    std::shared_ptr<Recorder> mRecorder;
    WorkItemsQueue mWorkItemsQueue;
    std::optional<uint64_t> mTerminateReqId;

}; // class GptServer

namespace
{

std::pair<std::vector<std::vector<int32_t>>, std::vector<int32_t>> parseDataset(
    std::filesystem::path const& datasetPath)
{
    auto constexpr allowExceptions = true;
    auto constexpr ingoreComments = true;
    TLLM_CHECK_WITH_INFO(
        std::filesystem::exists(datasetPath), std::string("File does not exist: ") + datasetPath.string());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);

    std::vector<std::vector<int32_t>> input_ids_list;
    std::vector<int32_t> output_ids_list;
    for (auto& sample : json)
    {
        input_ids_list.push_back(sample["input_ids"]);
        output_ids_list.push_back(sample["output_len"]);
    }
    return std::make_pair(input_ids_list, output_ids_list);
}

void benchmarkGptManager(std::string const& modelName, std::filesystem::path const& engineDir, std::string const& type,
    std::string const& datasetPath, std::shared_ptr<nvinfer1::ILogger> const& logger,
    std::optional<int32_t> maxNumSequences, std::optional<int32_t> maxTokensInPagedKvCache,
    std::optional<float> kvCacheFreeGpuMemFraction, std::optional<bool> enableTrtOverlap,
    batch_scheduler::SchedulerPolicy schedulerPolicy)
{
    auto const worldConfig = WorldConfig::mpi(*logger);

    TrtGptModelType modelType;
    if (type == "V1")
    {
        modelType = TrtGptModelType::V1;
    }
    else if (type == "IFB")
    {
        modelType = TrtGptModelType::InflightFusedBatching;
    }
    else
    {
        const std::string errStr = std::string("Unexpected batching type: ") + type;
        TLLM_LOG_ERROR(errStr);
    }

    // Load dataset
    auto dataset = parseDataset(datasetPath);
    std::vector<std::vector<NamedTensor>> tensors_list;
    const auto num_samples = dataset.first.size();
    for (int i = 0; i < num_samples; ++i)
    {
        const auto input_ids = dataset.first[i];
        const auto request_output_len = dataset.second[i];
        std::vector<int64_t> input_ids_shape = {1, static_cast<int64_t>(input_ids.size())};
        auto input_ids_tensor = NamedTensor(nvinfer1::DataType::kINT32, input_ids_shape, "input_ids", input_ids.data());
        auto request_output_len_tensor
            = NamedTensor(nvinfer1::DataType::kINT32, {1, 1}, "request_output_len", &request_output_len);
        std::vector<NamedTensor> tensors = {input_ids_tensor, request_output_len_tensor};
        tensors_list.push_back(tensors);
    }

    const int maxBeamWidth = 1;
    auto recorder = std::make_shared<Recorder>();
    uint64_t terminateReqId = num_samples + 1;
    auto gptServer = std::make_shared<GptServer>(engineDir, modelType, maxBeamWidth, schedulerPolicy, maxNumSequences,
        maxTokensInPagedKvCache, kvCacheFreeGpuMemFraction, enableTrtOverlap, recorder, terminateReqId);

    if (worldConfig.getRank() == 0)
    {
        recorder->initialize();
        for (int i = 0; i < tensors_list.size(); ++i)
        {
            gptServer->enqueue(tensors_list[i], 1 + i, false);
        }
        gptServer->waitForEmpty();
        recorder->finalize();
        recorder->calculateMetrics();
        recorder->report();
        // Send terminateReqId to terminate servers on all ranks
        // Sever on rank 0 will broadcast the terminate signal to other servers on multi-GPU cases
        gptServer->enqueue({}, terminateReqId, false);
    }
    // Wait until benchmarking is done and batch manager is terminated
    gptServer->waitBatchManager();
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT-LLM BatchManager Benchmark", "TensorRT-LLM BatchManager Benchmark for GPT and GPT-like models.");
    options.add_options()("h,help", "Print usage");
    options.add_options()(
        "m,model", "Model name specified for engines.", cxxopts::value<std::string>()->default_value("gpt_350m"));
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()(
        "type", "Batching type: IFB or V1(non-IFB) batching.", cxxopts::value<std::string>()->default_value("IFB"));
    options.add_options()("dataset", "Dataset that is used for benchmarking BatchManager.",
        cxxopts::value<std::string>()->default_value(""));

    options.add_options()("max_num_sequences", "Max number of Sequences.", cxxopts::value<int>()->default_value("-1"));
    options.add_options()(
        "max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>()->default_value("-1"));
    options.add_options()("kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.",
        cxxopts::value<float>()->default_value("-1"));
    options.add_options()("scheduler_policy", "Choose scheduler policy between max_utilization/guaranteed_no_evict.",
        cxxopts::value<std::string>()->default_value("guaranteed_no_evict"));
    options.add_options()("enable_trt_overlap", "Overlap TRT context preparation and execution",
        cxxopts::value<bool>()->default_value("false"));

    options.add_options()("log_level", "Choose log level between verbose/info/warning/error/internal_error.",
        cxxopts::value<std::string>()->default_value("error"));

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

    // Argument: Batching Type
    auto const type = result["type"].as<std::string>();

    // Argument: Dataset
    auto const datasetPath = result["dataset"].as<std::string>();

    // Argument: Max Num Sequences
    std::optional<int32_t> maxNumSequences = std::nullopt;
    if (result["max_num_sequences"].as<int>() != -1)
    {
        maxNumSequences = result["max_num_sequences"].as<int>();
    }

    // Argument: Max tokens in paged K-V Cache
    std::optional<int32_t> maxTokensInPagedKvCache = std::nullopt;
    if (result["max_tokens_in_paged_kvcache"].as<int>() != -1)
    {
        maxTokensInPagedKvCache = result["max_tokens_in_paged_kvcache"].as<int>();
    }

    // Argument: K-V Cache Free Gpu Mem Fraction
    std::optional<float> kvCacheFreeGpuMemFraction = std::nullopt;
    if (result["kv_cache_free_gpu_mem_fraction"].as<float>() != -1)
    {
        kvCacheFreeGpuMemFraction = result["kv_cache_free_gpu_mem_fraction"].as<float>();
    }

    // Argument: Enable TRT overlap
    std::optional<bool> enableTrtOverlap = std::nullopt;
    if (result["enable_trt_overlap"].as<bool>() != -1)
    {
        enableTrtOverlap = result["enable_trt_overlap"].as<bool>();
    }

    // Argument: Scheduler policy
    batch_scheduler::SchedulerPolicy schedulerPolicy;
    auto const schedulerPolicyArg = result["scheduler_policy"].as<std::string>();
    if (schedulerPolicyArg == "max_utilization")
    {
        schedulerPolicy = batch_scheduler::SchedulerPolicy::MAX_UTILIZATION;
    }
    else if (schedulerPolicyArg == "guaranteed_no_evict")
    {
        schedulerPolicy = batch_scheduler::SchedulerPolicy::GUARANTEED_NO_EVICT;
    }
    else
    {
        TLLM_LOG_ERROR("Unexpected scheduler policy: " + schedulerPolicyArg);
        return 1;
    }

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

    initTrtLlmPlugins(logger.get());

    try
    {
        benchmarkGptManager(result["model"].as<std::string>(), result["engine_dir"].as<std::string>(), type,
            datasetPath, logger, maxNumSequences, maxTokensInPagedKvCache, kvCacheFreeGpuMemFraction, enableTrtOverlap,
            schedulerPolicy);
    }
    catch (const std::exception& e)
    {
        TLLM_LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
