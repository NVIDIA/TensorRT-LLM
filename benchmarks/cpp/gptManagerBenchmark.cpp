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
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <chrono>
#include <cxxopts.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace mpi = tensorrt_llm::mpi;
namespace trt = nvinfer1;

// Class holding all infos regarding a single work item.
// This includes the original request, associated response factor
// and state.
class WorkItem
{
public:
    WorkItem(std::shared_ptr<InferenceRequest> ir, uint64_t requestId)
        : mInferenceRequest(ir)
        , mRequestId(requestId)
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
        auto const inputLength = request->getInputIds()->getSize();
        auto const maxNewTokens = request->getMaxNewTokensNamed();
        auto const& outputLengthTensor = maxNewTokens.tensor;
        TLLM_CHECK_WITH_INFO(outputLengthTensor != nullptr && outputLengthTensor->getSize() > 0,
            "Undefined scalar vector for %s", maxNewTokens.name.c_str());
        auto const outputLength = *bufferCast<SizeType>(*outputLengthTensor);
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
        batch_scheduler::SchedulerPolicy schedulerPolicy, TrtGptModelOptionalParams const& optionalParams,
        std::shared_ptr<Recorder> recorder, std::optional<uint64_t> terminateReqId)
    {
        ReturnBatchManagerStatsCallback iterationDataCallback{nullptr};
        if (optionalParams.logIterationData)
        {
            iterationDataCallback = [this](const std::string& s) { return TLLM_LOG_INFO(s); };
        }

        mBatchManager = std::make_shared<GptManager>(
            trtEnginePath, modelType, maxBeamWidth, schedulerPolicy,
            [this](int max_num_requests) { return getInferenceRequests(max_num_requests); },
            [this](uint64_t requestId, std::list<NamedTensor> response_tensors, bool final_response,
                const std::string& errMsg)
            { return sendResponse(requestId, response_tensors, final_response, errMsg); },
            nullptr, iterationDataCallback, optionalParams, terminateReqId);
        mRecorder = recorder;
        mTerminateReqId = terminateReqId;
    }

    ~GptServer()
    {
        mWorkItemsQueue.clear();
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
        catch (const tc::TllmException& e)
        {
            throw;
        }
        catch (const std::exception& e)
        {
            TLLM_THROW("%s", e.what());
        }
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
        auto& comm = COMM_SESSION;
        if (max_num_requests > 0)
        {
            auto world_size = comm.getSize();
            auto rank = comm.getRank();
            if (rank == 0)
            {
                auto num_new_work_items = std::min(static_cast<int64_t>(mWorkItemsQueue.numPendingWorkItems()),
                    static_cast<int64_t>(max_num_requests));
                if (world_size > 1)
                {
                    comm.bcast(&num_new_work_items, 1, mpi::MpiType::kINT64, 0);
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
                            auto warnStr = tc::fmtstr(
                                "request Id %lu has been stopped. Request is ignored.", workItem->requestId());
                            TLLM_LOG_WARNING(warnStr);
                            sendResponse(workItem->requestId(), {}, true, warnStr);
                        }
                    }
                    if (world_size > 1)
                    {
                        std::vector<int64_t> packed;
                        for (auto const& ir : rval)
                        {
                            auto vpacked = ir->serialize();
                            packed.push_back(static_cast<int64_t>(vpacked.size()));
                            packed.insert(
                                packed.end(), std::move_iterator(vpacked.begin()), std::move_iterator(vpacked.end()));
                        }
                        comm.bcast(packed, 0);
                    }
                }
            }
            else
            {
                // subordinate ranks hang until master rank sends work
                int64_t num_new_work_items;
                comm.bcast(&num_new_work_items, 1, mpi::MpiType::kINT64, 0);
                if (num_new_work_items > 0)
                {
                    std::vector<int64_t> packed;
                    comm.bcast(packed, 0);
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

    void sendResponse(uint64_t requestId, [[maybe_unused]] std::list<NamedTensor> const& response_tensors,
        bool final_response, [[maybe_unused]] const std::string& errMsg)
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
                mRecorder->recordEnd(requestId);
            }
        }
        catch (const std::exception& e)
        {
            TLLM_LOG_ERROR("Failed to send response for requestId: %ul\n%s", requestId, e.what());
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
    TLLM_CHECK_WITH_INFO(std::filesystem::exists(datasetPath), "File does not exist: %s", datasetPath.string().c_str());
    std::ifstream jsonStream(datasetPath);
    auto json = nlohmann::json::parse(jsonStream, nullptr, allowExceptions, ingoreComments);

    std::vector<std::vector<int32_t>> inputIds;
    std::vector<int32_t> outputIds;
    for (auto& sample : json)
    {
        inputIds.push_back(sample["input_ids"]);
        outputIds.push_back(sample["output_len"]);
    }
    return std::make_pair(inputIds, outputIds);
}

std::shared_ptr<InferenceRequest> makeRequest(std::uint64_t reqId,
    std::pair<std::vector<std::vector<int32_t>>, std::vector<int32_t>> const& dataset, std::size_t sample_idx,
    ITensor::SharedPtr const& beamWidthTensor, ITensor::SharedPtr const& eosId, ITensor::SharedPtr const& padId,
    BufferManager const& bufferManager)
{
    auto request = std::make_shared<InferenceRequest>(reqId);
    auto const& inputIds = dataset.first[sample_idx];
    request->setInputIds(bufferManager.copyFrom(
        inputIds, ITensor::makeShape({static_cast<SizeType>(inputIds.size())}), MemoryType::kPINNED));
    auto const request_output_len = dataset.second[sample_idx];
    request->setMaxNewTokens(
        bufferManager.copyFrom(&request_output_len, ITensor::makeShape({1, 1}), MemoryType::kPINNED));
    request->setBeamWidth(beamWidthTensor);
    if (eosId != nullptr)
    {
        request->setEndId(eosId);
    }
    if (padId != nullptr)
    {
        request->setPadId(padId);
    }
    return request;
}

void benchmarkGptManager([[maybe_unused]] std::string const& modelName, std::filesystem::path const& engineDir,
    std::string const& type, std::string const& datasetPath, int beamWidth, int warmUp,
    const std::optional<int32_t>& eosId, const std::optional<int32_t>& padId,
    std::shared_ptr<nvinfer1::ILogger> const& logger, TrtGptModelOptionalParams const& optionalParams,
    batch_scheduler::SchedulerPolicy schedulerPolicy)
{
    auto const worldConfig = WorldConfig::mpi();

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
        TLLM_LOG_ERROR("Unexpected batching type: %s", type.c_str());
    }

    BufferManager bufferManager{std::make_shared<CudaStream>()}; // the stream is not used

    ITensor::SharedPtr beamWidthTensor{
        bufferManager.copyFrom(&beamWidth, ITensor::makeShape({1}), MemoryType::kPINNED)};

    // Load dataset
    auto dataset = parseDataset(datasetPath);
    const auto numSamples = dataset.first.size();

    const int maxBeamWidth = beamWidth;
    auto recorder = std::make_shared<Recorder>();
    uint64_t terminateReqId = numSamples + 1;
    auto gptServer = std::make_shared<GptServer>(
        engineDir, modelType, maxBeamWidth, schedulerPolicy, optionalParams, recorder, terminateReqId);

    ITensor::SharedPtr eosIdTensor{
        eosId ? bufferManager.copyFrom(&eosId.value(), ITensor::makeShape({1}), MemoryType::kPINNED) : nullptr};
    ITensor::SharedPtr padIdTensor{
        padId ? bufferManager.copyFrom(&padId.value(), ITensor::makeShape({1}), MemoryType::kPINNED) : nullptr};

    if (worldConfig.getRank() == 0)
    {
        // Warm up
        SizeType reqId = 0;
        for (auto i = 0; i < warmUp; ++i)
        {
            ++reqId;
            if (i == terminateReqId)
                ++reqId;
            auto request = makeRequest(reqId, dataset, 0, beamWidthTensor, eosIdTensor, padIdTensor, bufferManager);
            gptServer->enqueue(request);
        }
        gptServer->waitForEmpty();

        // Benchmark
        recorder->initialize();
        for (std::size_t i = 0; i < numSamples; ++i)
        {
            auto request = makeRequest(i + 1, dataset, i, beamWidthTensor, eosIdTensor, padIdTensor, bufferManager);
            gptServer->enqueue(request);
        }
        gptServer->waitForEmpty();
        recorder->finalize();
        recorder->calculateMetrics();
        recorder->report();
        // Send terminateReqId to terminate servers on all ranks
        // Sever on rank 0 will broadcast the terminate signal to other servers on multi-GPU cases
        gptServer->enqueue(std::make_shared<InferenceRequest>(terminateReqId));
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
    options.add_options()(
        "beam_width", "Specify beam width you want to benchmark.", cxxopts::value<int>()->default_value("1"));
    options.add_options()(
        "warm_up", "Specify warm up iterations before benchmark starts.", cxxopts::value<int>()->default_value("2"));
    options.add_options()("eos_id", "Specify the end-of-sequence token id.", cxxopts::value<int>());
    options.add_options()("pad_id", "Specify the padding token id.", cxxopts::value<int>());
    options.add_options()("max_tokens_in_paged_kvcache", "Max tokens in paged K-V Cache.", cxxopts::value<int>());
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());
    options.add_options()(
        "enable_trt_overlap", "Overlap TRT context preparation and execution", cxxopts::value<bool>());
    options.add_options()("enable_kv_cache_reuse", "Enables the KV cache reuse.", cxxopts::value<bool>());

    options.add_options()("scheduler_policy", "Choose scheduler policy between max_utilization/guaranteed_no_evict.",
        cxxopts::value<std::string>()->default_value("guaranteed_no_evict"));

    options.add_options()("log_level", "Choose log level between verbose/info/warning/error/internal_error.",
        cxxopts::value<std::string>()->default_value("error"));
    options.add_options()(
        "log_iteration_data", "On each decoder iteration, print batch state metadata.", cxxopts::value<bool>());

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

    // Argument: beam width
    auto const beamWidth = result["beam_width"].as<int>();

    TrtGptModelOptionalParams optionalParams;
    // Argument: Max tokens in paged K-V Cache
    if (result.count("max_tokens_in_paged_kvcache"))
    {
        optionalParams.kvCacheConfig.maxTokens = result["max_tokens_in_paged_kvcache"].as<int>();
    }
    // Argument: K-V Cache Free Gpu Mem Fraction
    if (result.count("kv_cache_free_gpu_mem_fraction"))
    {
        optionalParams.kvCacheConfig.freeGpuMemoryFraction = result["kv_cache_free_gpu_mem_fraction"].as<float>();
    }
    // Argument: Enable TRT overlap
    if (result.count("enable_trt_overlap"))
    {
        optionalParams.enableTrtOverlap = result["enable_trt_overlap"].as<bool>();
    }
    // Argument: Enable KV cache reuse
    if (result.count("enable_kv_cache_reuse"))
    {
        optionalParams.kvCacheConfig.enableBlockReuse = result["enable_kv_cache_reuse"].as<bool>();
    }
    // Argument: Enable batch stats output
    if (result.count("log_iteration_data"))
    {
        optionalParams.logIterationData = result["log_iteration_data"].as<bool>();
    }

    std::optional<int32_t> padId;
    // Argument: Padding token id
    if (result.count("pad_id"))
    {
        padId = result["pad_id"].as<int>();
    }

    std::optional<int32_t> eosId;
    // Argument: End-of-sentence token id
    if (result.count("eos_id"))
    {
        eosId = result["eos_id"].as<int>();
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
            datasetPath, beamWidth, result["warm_up"].as<int>(), eosId, padId, logger, optionalParams, schedulerPolicy);
    }
    catch (const std::exception& e)
    {
        TLLM_LOG_ERROR(e.what());
        return 1;
    }
    return 0;
}
