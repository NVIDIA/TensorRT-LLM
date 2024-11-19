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
#include "tensorrt_llm/executor/types.h"
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
    std::optional<float> crossKvCacheFraction{std::nullopt};
    bool enableTrtOverlap{false};
    bool enableBatchSizeTuning{false};
    bool enableBlockReuse{false};
    bool enableChunkedContext{false};
    bool streaming{false};
    bool enableExpDelays{false};
    std::optional<float> requestRate{std::nullopt};
    std::optional<int> concurrency{std::nullopt};
    std::optional<SizeType32> maxBatchSize{std::nullopt};
    std::optional<SizeType32> maxNumTokens{std::nullopt};
    int randomSeed = 430;
    std::optional<std::vector<int>> maxAttentionWindowVec{std::nullopt};
    std::optional<int> sinkTokenLength{std::nullopt};
    bool multiBlockMode{true};
    bool enableContextFMHAFP32Acc{false};
    bool cudaGraphMode{false};
    SizeType32 cudaGraphCacheSize{0};

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

    std::optional<texec::LookaheadDecodingConfig> executorLookaheadConfig;
    std::optional<texec::LookaheadDecodingConfig> requestLookaheadConfig;
};
} // namespace

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
    SizeType32 decodingIter{0};
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
        mRequestsQueueingLatencies.clear();
    }

    void finalize()
    {
        mEnd = std::chrono::steady_clock::now();
    }

    void recordQueueLatency(std::vector<float> const& latencies)
    {
        mRequestsQueueingLatencies.insert(mRequestsQueueingLatencies.end(), latencies.begin(), latencies.end());
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

        mRequestBenchInfos[requestId].decodingIter += 1;
    }

    void recordToken(uint64_t requestId, std::list<NamedTensor> const& responseTensors)
    {
        int32_t outputLength = 1;
        for (auto& tensor : responseTensors)
        {
            if (tensor.name == inference_request::kSequenceLengthTensorName)
            {
                // Tensor of shape nBeams, and we only need the first one
                outputLength = *(bufferCast<int32_t>(*(tensor.tensor)));
                break;
            }
        }

        mRequestBenchInfos[requestId].outputLength += outputLength;
        this->recordToken(requestId);
    }

    void recordToken(uint64_t requestId, texec::Response const& response)
    {
        auto outputTokenIds = response.getResult().outputTokenIds;

        int32_t outputLength = 1;
        for (auto const& beam : outputTokenIds)
        {
            outputLength = std::max(static_cast<int32_t>(beam.size()), outputLength);
        }

        mRequestBenchInfos[requestId].outputLength += outputLength;
        this->recordToken(requestId);
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
            this->recordToken(requestId, responseTensors);
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
                TLLM_LOG_DEBUG("response.getResult().outputTokenIds");
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
                mRequestBenchInfos[requestId].decodingIter = response.getResult().decodingIter;
            }
            else
            {
                this->recordToken(requestId, response);
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
        int totalDecodingIter{0};
        mNumErrorSamples = 0;
        mNumSamples = 0;
        for (auto reqInfo : mRequestBenchInfos)
        {
            if (!reqInfo.second.hasError)
            {
                reqLatencies.push_back(reqInfo.second.latency);
                totalOutputTokens += reqInfo.second.outputLength;
                totalDecodingIter += reqInfo.second.decodingIter;

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
        mAcceptanceRate = totalDecodingIter
            ? (static_cast<float>(totalOutputTokens) / static_cast<float>(totalDecodingIter))
            : 0.0f;

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

            mAvgReqQueueingLatency
                = std::accumulate(mRequestsQueueingLatencies.begin(), mRequestsQueueingLatencies.end(), 0.F)
                / mRequestsQueueingLatencies.size();
            std::sort(mRequestsQueueingLatencies.begin(), mRequestsQueueingLatencies.end());
            mP99ReqQueueingLatency = calcPercentile(mRequestsQueueingLatencies, 99);
            mP90ReqQueueingLatency = calcPercentile(mRequestsQueueingLatencies, 90);
            mP50ReqQueueingLatency = calcPercentile(mRequestsQueueingLatencies, 50);
            mMaxReqQueueingLatency = mRequestsQueueingLatencies.back();
            mMinReqQueueingLatency = mRequestsQueueingLatencies.front();
        }
    }

    void report()
    {

        printf("[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] num_error_samples %d\n", mNumErrorSamples);
        printf("\n[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);
        printf("[BENCHMARK] avg_acceptance_rate(tokens/decoding steps) %.2f\n\n", mAcceptanceRate);

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

            printf("[BENCHMARK] avg_request_queueing_latency(ms) %.2f\n", mAvgReqQueueingLatency);
            printf("[BENCHMARK] max_request_queueing_latency(ms) %.2f\n", mMaxReqQueueingLatency);
            printf("[BENCHMARK] min_request_queueing_latency(ms) %.2f\n", mMinReqQueueingLatency);
            printf("[BENCHMARK] p99_request_queueing_latency(ms) %.2f\n", mP99ReqQueueingLatency);
            printf("[BENCHMARK] p90_request_queueing_latency(ms) %.2f\n", mP90ReqQueueingLatency);
            printf("[BENCHMARK] p50_request_queueing_latency(ms) %.2f\n\n", mP50ReqQueueingLatency);
        }
    }

    void writeOpMetricsToCsv()
    {
        if (!mOpCsvFile.empty())
        {
            std::vector<std::string> headers = {"num_samples", "num_error_samples", "total_latency(ms)",
                "seq_throughput(seq/sec)", "token_throughput(token/sec)", "avg_sequence_latency(ms)",
                "max_sequence_latency(ms)", "min_sequence_latency(ms)", "p99_sequence_latency(ms)",
                "p90_sequence_latency(ms)", "p50_sequence_latency(ms)", "avg_acceptance_rate(tokens/decoding steps)"};

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
                           << mMinSeqLatency << "," << mP99SeqLatency << "," << mP90SeqLatency << "," << mP50SeqLatency
                           << "," << mAcceptanceRate;
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
    float mAcceptanceRate{};
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
    float mAvgReqQueueingLatency{};
    float mP99ReqQueueingLatency{};
    float mP90ReqQueueingLatency{};
    float mP50ReqQueueingLatency{};
    float mMaxReqQueueingLatency{};
    float mMinReqQueueingLatency{};
    std::vector<float> mRequestsQueueingLatencies{};

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
    ExecutorServer(std::optional<std::filesystem::path> const& decoderTrtEnginePath,
        std::optional<std::filesystem::path> const& encoderTrtEnginePath, texec::BatchingType batchingType,
        int32_t maxBeamWidth, texec::CapacitySchedulerPolicy capacitySchedulerPolicy,
        BenchmarkParams const& benchmarkParams, std::shared_ptr<Recorder> recorder, std::chrono::milliseconds waitSleep,
        bool logIterationData, texec::ModelType executorModelType)
        : mRecorder(std::move(recorder))
        , mWaitSleep(waitSleep)
        , mConcurrency(benchmarkParams.concurrency)
        , mActiveCount(0)
        , mNumFinished(0)
        , mShutdown(false)
        , mLogIterationData(logIterationData)
    {
        texec::DynamicBatchConfig dynamicBatchConfig(benchmarkParams.enableBatchSizeTuning);
        texec::SchedulerConfig schedulerConfig(capacitySchedulerPolicy, std::nullopt, dynamicBatchConfig);

        texec::KvCacheConfig kvCacheConfig(benchmarkParams.enableBlockReuse, benchmarkParams.maxTokensInPagedKvCache,
            benchmarkParams.maxAttentionWindowVec, benchmarkParams.sinkTokenLength,
            benchmarkParams.freeGpuMemoryFraction, benchmarkParams.kvHostCacheSize, benchmarkParams.kvOnboardBlocks,
            benchmarkParams.crossKvCacheFraction);
        texec::PeftCacheConfig peftCacheConfig(0, benchmarkParams.loraDeviceNumModLayers, 8, 64, 4, 4, 4, 24, 8,
            std::nullopt, benchmarkParams.loraHostCacheSize);
        texec::ExtendedRuntimePerfKnobConfig extendedRuntimePerfKnobConfig(benchmarkParams.multiBlockMode,
            benchmarkParams.enableContextFMHAFP32Acc, benchmarkParams.cudaGraphMode,
            benchmarkParams.cudaGraphCacheSize);
        texec::ExecutorConfig executorConfig(
            maxBeamWidth, schedulerConfig, kvCacheConfig, benchmarkParams.enableChunkedContext, true);
        executorConfig.setGpuWeightsPercent(benchmarkParams.gpuWeightsPercent);
        executorConfig.setPeftCacheConfig(peftCacheConfig);
        executorConfig.setBatchingType(batchingType);
        if (benchmarkParams.maxBatchSize)
        {
            executorConfig.setMaxBatchSize(benchmarkParams.maxBatchSize.value());
        }
        if (benchmarkParams.maxNumTokens)
        {
            executorConfig.setMaxNumTokens(benchmarkParams.maxNumTokens.value());
        }

        executorConfig.setDecodingConfig(
            texec::DecodingConfig(benchmarkParams.medusaChoices.has_value() ? texec::DecodingMode::Medusa()
                    : benchmarkParams.executorLookaheadConfig.has_value()   ? texec::DecodingMode::Lookahead()
                                                                            : texec::DecodingMode::Auto(),
                benchmarkParams.executorLookaheadConfig, benchmarkParams.medusaChoices));
        executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);

        if (executorModelType == texec::ModelType::kDECODER_ONLY)
        {
            mExecutor
                = std::make_unique<texec::Executor>(decoderTrtEnginePath.value(), executorModelType, executorConfig);
        }
        else if (executorModelType == texec::ModelType::kENCODER_DECODER)
        {
            mExecutor = std::make_unique<texec::Executor>(
                encoderTrtEnginePath.value(), decoderTrtEnginePath.value(), executorModelType, executorConfig);
        }
        else if (executorModelType == texec::ModelType::kENCODER_ONLY)
        {
            mExecutor
                = std::make_unique<texec::Executor>(encoderTrtEnginePath.value(), executorModelType, executorConfig);
        }
        else
        {
            TLLM_LOG_ERROR("not a supported executor model type in executor server.");
        }

        auto const& world = tensorrt_llm::mpi::MpiComm::world();
        auto worldRank = world.getRank();
        if (worldRank == 0)
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
                maxNewTokens.push_back(request.getMaxTokens());
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

    void resetNumFinished()
    {
        mNumFinished = 0;
    }

    bool canEnqueue(int numSentRequests) const
    {
        return !mConcurrency || (numSentRequests - mNumFinished < mConcurrency);
    }

    void waitForResponses(SizeType32 numRequests, bool warmup = false)
    {
        while (mActiveCount || (mNumFinished < numRequests))
        {
            auto responses = mExecutor->awaitResponses(mWaitSleep);
            for (auto const& response : responses)
            {
                auto const reqId = response.getRequestId();
                TLLM_LOG_DEBUG("response.getResult().isFinal");
                if (response.getResult().isFinal)
                {
                    mActiveCount--;
                    mNumFinished++;
                    if (!warmup)
                    {
                        mRecorder->recordEnd(reqId, response);
                    }
                }
                else
                {
                    if (!warmup && !response.hasError())
                    {
                        mRecorder->recordToken(reqId, response);
                    }
                }
            }
        }
    }

    void collectStats() const
    {
        while (!mShutdown)
        {
            auto iterStats = mExecutor->getLatestIterationStats();
            for (auto const& iterStat : iterStats)
            {
                SizeType32 numNewActiveRequests = iterStat.numNewActiveRequests;
                if (numNewActiveRequests > 0)
                {
                    float avgQueueingTime
                        = static_cast<float>(iterStat.newActiveRequestsQueueLatencyMS / numNewActiveRequests);
                    std::vector<float> requestsQueueLatencyMS(numNewActiveRequests, avgQueueingTime);
                    mRecorder->recordQueueLatency(requestsQueueLatencyMS);
                }
                if (mLogIterationData)
                {
                    TLLM_LOG_INFO(texec::JsonSerialization::toJsonStr(iterStat));
                }
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
    std::optional<int> mConcurrency;
    std::atomic<uint64_t> mActiveCount;
    std::atomic<uint64_t> mNumFinished;
    std::atomic<bool> mShutdown;
    bool mLogIterationData;
}; // class ExecutorServer

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

texec::Request makeExecutorRequest(Sample const& sample, SizeType32 const& beamWidth,
    std::optional<SizeType32> const& eosId, std::optional<SizeType32> const& padId, bool streaming = false,
    bool const& returnContextLogits = false, bool const& returnGenerationLogits = false,
    std::optional<texec::LoraConfig> const& loraConfig = std::nullopt,
    std::optional<texec::LookaheadDecodingConfig> const& lookaheadConfig = std::nullopt,
    std::optional<texec::VecTokens> encoderInputTokenIds = std::nullopt)
{
    auto samplingConfig = texec::SamplingConfig{beamWidth};
    auto outputConfig = texec::OutputConfig{false, returnContextLogits, returnGenerationLogits, false};
    return texec::Request(sample.inputIds, sample.outputLen, streaming, samplingConfig, outputConfig, eosId, padId,
        std::nullopt,    // positionIds
        std::nullopt,    // badWords
        std::nullopt,    // stopWords
        std::nullopt,    // embeddingBias
        std::nullopt,    // speculativeDecoding
        std::nullopt,    // pTuning
        std::nullopt,    // mRopeConfig
        loraConfig,      // loraConfig
        lookaheadConfig, // lookaheadConfig
        std::nullopt,    // kvCacheRetentionConfig
        std::nullopt,    // logitsPostProcessorName
        encoderInputTokenIds.has_value() ? encoderInputTokenIds : std::nullopt);
}

void benchmarkExecutor(std::optional<std::filesystem::path> const& decoderEngineDir,
    std::optional<std::filesystem::path> const& encoderEngineDir, texec::BatchingType batchingType,
    std::string const& datasetPath, std::string const& opCsvFile, int maxNumSamples, int beamWidth, int warmUp,
    std::optional<int32_t> const& eosId, std::optional<int32_t> const& padId, BenchmarkParams const& benchmarkParams,
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy, std::chrono::milliseconds waitSleep,
    bool returnContextLogits, bool returnGenerationLogits, std::optional<int> const staticEmulatedBatchSize,
    bool logIterationData, std::optional<SizeType32> const maxPromptLen, texec::ModelType executorModelType)
{
    auto const& world = tensorrt_llm::mpi::MpiComm::world();
    auto worldRank = world.getRank();

    // Load dataset
    auto const samples = parseWorkloadJson(datasetPath, maxNumSamples, maxPromptLen);
    auto const numSamples = samples.size();

    auto recorder = std::make_shared<Recorder>(opCsvFile, benchmarkParams.streaming, beamWidth);
    int32_t decoderStartTokenId = 0;
    std::shared_ptr<ExecutorServer> executorServer;

    if (executorModelType == texec::ModelType::kDECODER_ONLY)
    {
        TLLM_CHECK_WITH_INFO(
            decoderEngineDir.has_value(), "decoder models require a path to decoder engine in executor benchmark.");
        executorServer
            = std::make_shared<ExecutorServer>(decoderEngineDir.value(), std::nullopt, batchingType, beamWidth,
                capacitySchedulerPolicy, benchmarkParams, recorder, waitSleep, logIterationData, executorModelType);
    }
    else if (executorModelType == texec::ModelType::kENCODER_DECODER)
    {
        TLLM_CHECK_WITH_INFO(encoderEngineDir.has_value(),
            "encoder-decoder models require a path to encoder engine in executor benchmark.");
        executorServer = std::make_shared<ExecutorServer>(decoderEngineDir.value(), encoderEngineDir.value(),
            batchingType, beamWidth, capacitySchedulerPolicy, benchmarkParams, recorder, waitSleep, logIterationData,
            executorModelType);
        try
        {
            std::ifstream decoderJsonConfigPath(decoderEngineDir.value() / "config.json");
            auto const decoderPretrainedConfig
                = nlohmann::json::parse(decoderJsonConfigPath, nullptr, true, true).at("pretrained_config");
            decoderStartTokenId = decoderPretrainedConfig.at("decoder_start_token_id").template get<int32_t>();
        }
        catch (nlohmann::json::out_of_range& e)
        {
            TLLM_LOG_ERROR(
                "Parameter %s cannot be read from decoder config.json in pretrained_config. Using default id %d.",
                std::string("decoder_start_token_id").c_str(), decoderStartTokenId);
        }
        catch (nlohmann::json::type_error const& e)
        {
            TLLM_LOG_ERROR(
                "Parameter %s has error type in decoder config.json in pretrained_config. Using default id %d.",
                std::string("decoder_start_token_id").c_str(), decoderStartTokenId);
        }
    }
    else if (executorModelType == texec::ModelType::kENCODER_ONLY)
    {
        TLLM_CHECK_WITH_INFO(
            encoderEngineDir.has_value(), "encoder models require a path to encoder engine in executor benchmark.");
        executorServer
            = std::make_shared<ExecutorServer>(std::nullopt, encoderEngineDir.value(), batchingType, beamWidth,
                capacitySchedulerPolicy, benchmarkParams, recorder, waitSleep, logIterationData, executorModelType);
    }
    else
    {
        TLLM_LOG_ERROR("not a supported executor model type in executor benchmark.");
        return;
    }

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
                if (executorModelType == texec::ModelType::kENCODER_DECODER)
                {
                    Sample s{std::vector<int32_t>{decoderStartTokenId}, 1, static_cast<int32_t>(taskId)};
                    requests.emplace_back(makeExecutorRequest(s, beamWidth, eosId, padId, false, false, false,
                        loraConfig, std::nullopt, std::vector<int32_t>{1, 2, 3, 4, 5}));
                }
                else
                {
                    Sample s{std::vector<int32_t>{1, 2, 3, 4, 5}, 1, static_cast<int32_t>(taskId)};
                    requests.emplace_back(
                        makeExecutorRequest(s, beamWidth, eosId, padId, false, false, false, loraConfig, std::nullopt));
                }
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
                if (executorModelType == texec::ModelType::kENCODER_DECODER)
                {
                    Sample s{std::vector<int32_t>{decoderStartTokenId}, samples[0].outputLen, samples[0].taskId};
                    requests.emplace_back(makeExecutorRequest(s, beamWidth, eosId, padId, benchmarkParams.streaming,
                        returnContextLogits, returnGenerationLogits, std::nullopt,
                        benchmarkParams.requestLookaheadConfig, samples[0].inputIds));
                }
                else
                {
                    requests.emplace_back(makeExecutorRequest(samples[0], beamWidth, eosId, padId,
                        benchmarkParams.streaming, returnContextLogits, returnGenerationLogits, std::nullopt,
                        benchmarkParams.requestLookaheadConfig));
                }
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
                if (executorModelType == texec::ModelType::kENCODER_DECODER)
                {
                    Sample s{std::vector<int32_t>{decoderStartTokenId}, samples[i].outputLen, samples[i].taskId};
                    requests.emplace_back(makeExecutorRequest(s, beamWidth, eosId, padId, benchmarkParams.streaming,
                        returnContextLogits, returnGenerationLogits, loraConfig, benchmarkParams.requestLookaheadConfig,
                        samples[i].inputIds));
                }
                else
                {
                    requests.emplace_back(makeExecutorRequest(samples[i], beamWidth, eosId, padId,
                        benchmarkParams.streaming, returnContextLogits, returnGenerationLogits, loraConfig,
                        benchmarkParams.requestLookaheadConfig));
                }
            }

            bool const hasDelay
                = std::any_of(timeDelays.begin(), timeDelays.end(), [](auto const& delay) { return delay > 0.0; });
            executorServer->resetNumFinished();
            if (!staticEmulatedBatchSize)
            {
                // Launch a thread that will wait for responses
                std::thread waitThread(
                    [numSamples, executorServer]() { executorServer->waitForResponses(numSamples); });

                // Enqueue requests one by one
                int numSentRequests = 0;
                while (numSentRequests < numSamples)
                {
                    if (executorServer->canEnqueue(numSentRequests))
                    {
                        executorServer->enqueue({requests.at(numSentRequests)});
                        if (hasDelay && numSentRequests < numSamples - 1)
                        {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(static_cast<int>(timeDelays.at(numSentRequests) * 1000)));
                        }
                        numSentRequests += 1;
                    }
                }
                waitThread.join();
            }
            else
            {
                TLLM_CHECK_WITH_INFO(
                    !hasDelay, "Executor benchmark doesn't support delays with emulated static batch sizes");
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

texec::LookaheadDecodingConfig parseLookaheadConfig(std::string const& input)
{
    std::regex regex("\\[ *(\\d+) *, *(\\d+) *, *(\\d+) *\\]");
    std::smatch match;
    if (std::regex_match(input, match, regex))
    {
        TLLM_CHECK(match.size() == 4);
        auto w = std::stoi(match[1]);
        auto n = std::stoi(match[2]);
        auto g = std::stoi(match[3]);
        return texec::LookaheadDecodingConfig(w, n, g);
    }
    else
    {
        TLLM_LOG_WARNING("cannot parse lookahead config from '%s'", input.c_str());
        return texec::LookaheadDecodingConfig();
    }
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT-LLM BatchManager Benchmark", "TensorRT-LLM BatchManager Benchmark for GPT and GPT-like models.");
    options.add_options()("h,help", "Print usage");
    options.add_options()("engine_dir, decoder_engine_dir", "Directory that store the engines of decoder models.",
        cxxopts::value<std::string>());
    options.add_options()(
        "encoder_engine_dir", "Directory that store the engines of the encoder models.", cxxopts::value<std::string>());
    options.add_options()(
        "api", "API type: gptManager or executor.", cxxopts::value<std::string>()->default_value("executor"));
    options.add_options()("type",
        "Batching type: choose between inflight/static. (IFB/V1 options are going to be deprecated)",
        cxxopts::value<std::string>()->default_value("inflight"));
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
    options.add_options()(
        "max_attention_window", "Max KV cache length per sequence", cxxopts::value<std::vector<int>>());
    options.add_options()("sink_token_len", "Sink token length in kv cache per sequence.", cxxopts::value<int>());
    options.add_options()(
        "random_seed", "integer random seed for exponential time delays.", cxxopts::value<int>()->default_value("420"));
    options.add_options()(
        "kv_cache_free_gpu_mem_fraction", "K-V Cache Free Gpu Mem Fraction.", cxxopts::value<float>());
    options.add_options()(
        "cross_kv_cache_fraction", "Cross K-V Cache Fraction (from 0.0 to 1.0).", cxxopts::value<float>());
    options.add_options()("request_rate",
        "request rate in reqs/sec. Skipping this arg or negative value will trigger offline/0-delay.",
        cxxopts::value<float>());
    options.add_options()("concurrency", "Concurrent number of connections with the server.", cxxopts::value<int>());
    options.add_options()("max_batch_size", "The max runtime batch size when benchmarking", cxxopts::value<int>());
    options.add_options()(
        "max_num_tokens", "The max runtime number of tokens per batch when benchmarking", cxxopts::value<int>());
    options.add_options()(
        "enable_batch_size_tuning", "Dynamic tuning of batch size", cxxopts::value<bool>()->default_value("false"));
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

    options.add_options()("scheduler_policy",
        "Choose scheduler policy between max_utilization/guaranteed_no_evict/static_batch.",
        cxxopts::value<std::string>()->default_value("guaranteed_no_evict"));

    options.add_options()("static_emulated_batch_size",
        "Emulate static batching performance with the provided batch size.", cxxopts::value<SizeType32>());
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
    options.add_options()("kv_onboard_blocks", "If offloaded blocks should be onboarded to primary memory before reuse",
        cxxopts::value<bool>()->default_value("true"));
    options.add_options()(
        "max_prompt_len", "Truncate all prompts from dataset to the length specified.", cxxopts::value<SizeType32>());

    options.add_options()("gpu_weights_percent",
        "Specify the percentage of weights that reside on GPU (from 0.0 to 1.0).",
        cxxopts::value<float>()->default_value("1.0"));
    options.add_options()(
        "medusa_choices", "Medusa choices in the format of [[0], [0, 1], [0, 0, 1]]", cxxopts::value<std::string>());

    options.add_options()("multi_block_mode",
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel",
        cxxopts::value<bool>()->default_value("true"));
    options.add_options()("cuda_graph_mode", "When enabled, inference is executed with cuda graph.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("cuda_graph_cache_size",
        "Specify how many cuda graphs are cached in the runtime. Larger cache gives better perf, but consumes more GPU "
        "memory.",
        cxxopts::value<SizeType32>()->default_value("0"));

    options.add_options()("enable_context_fmha_fp32_acc", "Enable FMHA runner FP32 accumulation",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("executor_lookahead_config",
        "lookahead config in the format of [max_window_size, max_ngram_size, max_verification_set_size]",
        cxxopts::value<std::string>());
    options.add_options()("request_lookahead_config",
        "lookahead config in the format of [max_window_size, max_ngram_size, max_verification_set_size], and each <= "
        "executor lookahead config",
        cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Argument: Engine directory
    if (!result.count("engine_dir") && !result.count("encoder_engine_dir"))
    {
        std::cout << options.help() << std::endl;
        TLLM_LOG_ERROR("Please specify engine directory.");
        return 1;
    }

    // Argument: Batching Type
    auto const type = result["type"].as<std::string>();
    texec::BatchingType batchingType{texec::BatchingType::kINFLIGHT};
    if (type == "V1" || type == "static")
    {
        if (type == "V1")
        {
            TLLM_LOG_WARNING("type option \"V1\" is going to be renamed to \"static\".");
        }
        batchingType = texec::BatchingType::kSTATIC;
    }
    else if (type == "IFB" || type == "inflight")
    {
        if (type == "IFB")
        {
            TLLM_LOG_WARNING("type option \"IFB\" is going to be renamed to \"inflight\".");
        }
        batchingType = texec::BatchingType::kINFLIGHT;
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
        benchmarkParams.maxAttentionWindowVec = result["max_attention_window"].as<std::vector<int>>();
    }

    // Argument: Sink token length
    if (result.count("sink_token_len"))
    {
        benchmarkParams.sinkTokenLength = result["sink_token_len"].as<int>();
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
    // Argument: K-V Cache Cross Attention Fraction. Only applicable to enc-dec models.
    if (result.count("encoder_engine_dir") && result.count("decoder_engine_dir"))
    {
        if (result.count("cross_kv_cache_fraction"))
        {
            benchmarkParams.crossKvCacheFraction = result["cross_kv_cache_fraction"].as<float>();
        }
        else
        {
            benchmarkParams.crossKvCacheFraction
                = 0.5f; // default value if not set. but non enc-dec should not even have this param set
        }
    }

    // Argument: Enable dynamic tuning of batch size
    benchmarkParams.enableBatchSizeTuning = result["enable_batch_size_tuning"].as<bool>();

    // Argument: Enable KV cache reuse
    benchmarkParams.enableBlockReuse = result["enable_kv_cache_reuse"].as<bool>();

    // Argument: streaming
    benchmarkParams.streaming = result["streaming"].as<bool>();

    TLLM_CHECK_WITH_INFO(!(result.count("request_rate") && result.count("concurrency")),
        "request_rate and concurrency cannot be specified at the same time.");

    // Argument: request rate
    if (result.count("request_rate"))
    {
        benchmarkParams.requestRate = result["request_rate"].as<float>();
    }

    // Argument: concurrency
    if (result.count("concurrency"))
    {
        benchmarkParams.concurrency = result["concurrency"].as<int>();
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

    if (logIterationData)
    {
        TLLM_LOG_WARNING("Setting log_iteration_data to true adds overheads and may result in lower perf");
    }

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
    benchmarkParams.kvOnboardBlocks = result["kv_onboard_blocks"].as<bool>();

    // Argument: Medusa choices for the Medusa speculative decoding.
    if (result.count("medusa_choices"))
    {
        benchmarkParams.medusaChoices = parseVectorOfVectors(result["medusa_choices"].as<std::string>());
    }
    if (result.count("executor_lookahead_config"))
    {
        benchmarkParams.executorLookaheadConfig
            = parseLookaheadConfig(result["executor_lookahead_config"].as<std::string>());
    }
    if (result.count("request_lookahead_config"))
    {
        benchmarkParams.requestLookaheadConfig
            = parseLookaheadConfig(result["request_lookahead_config"].as<std::string>());
    }

    // Argument: multi_block_mode
    benchmarkParams.multiBlockMode = result["multi_block_mode"].as<bool>();

    // Argument: enable_context_fmha_fp32_acc
    benchmarkParams.enableContextFMHAFP32Acc = result["enable_context_fmha_fp32_acc"].as<bool>();

    // Argument: cuda_graph_mode
    benchmarkParams.cudaGraphMode = result["cuda_graph_mode"].as<bool>();

    // Argument: cuda_graph_cache_size
    benchmarkParams.cudaGraphCacheSize = result["cuda_graph_cache_size"].as<SizeType32>();

    std::optional<TokenIdType> padId;
    // Argument: Padding token id
    if (result.count("pad_id"))
    {
        padId = result["pad_id"].as<TokenIdType>();
    }

    // Argument: End-of-sentence token id
    std::optional<TokenIdType> eosId = result["eos_id"].as<TokenIdType>();

    std::optional<SizeType32> staticEmulatedBatchSize;
    // Argument: Static emulated batch size
    if (result.count("static_emulated_batch_size"))
    {
        staticEmulatedBatchSize = result["static_emulated_batch_size"].as<SizeType32>();
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
    else if (capacitySchedulerPolicyArg == "static_batch")
    {
        capacitySchedulerPolicy = texec::CapacitySchedulerPolicy::kSTATIC_BATCH;
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

    initTrtLlmPlugins(logger.get());

    // Argument: API
    auto const api = result["api"].as<std::string>();
    if (api == "executor")
    {
        texec::ModelType executorModelType;
        std::optional<std::string> decoderEngineDir = std::nullopt, encoderEngineDir = std::nullopt;
        if (result.count("encoder_engine_dir") && result.count("decoder_engine_dir"))
        {
            TLLM_CHECK_WITH_INFO(api == "executor", "encoder-decoder only support executor api.");
            TLLM_CHECK_WITH_INFO(
                batchingType == texec::BatchingType::kINFLIGHT, "encoder-decoder only support inflight batching.");
            executorModelType = texec::ModelType::kENCODER_DECODER;
            encoderEngineDir = result["encoder_engine_dir"].as<std::string>();
            decoderEngineDir = result["decoder_engine_dir"].as<std::string>();
        }
        else if (result.count("engine_dir"))
        {
            executorModelType = texec::ModelType::kDECODER_ONLY;
            decoderEngineDir = result["engine_dir"].as<std::string>();
        }
        else
        {
            executorModelType = texec::ModelType::kENCODER_ONLY;
            encoderEngineDir = result["encoder_engine_dir"].as<std::string>();
        }
        try
        {
            benchmarkExecutor(decoderEngineDir, encoderEngineDir, batchingType, datasetPath, opCsvFile, maxNumSamples,
                beamWidth, result["warm_up"].as<int>(), eosId, padId, benchmarkParams, capacitySchedulerPolicy,
                waitSleep, returnContextLogits, returnGenerationLogits, staticEmulatedBatchSize, logIterationData,
                maxPromptLen, executorModelType);
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(e.what());
            return 1;
        }
    }
    else if (api == "gptManager")
    {
        TLLM_LOG_ERROR("gptManager is deprecated, please use the executor API.");
        return 1;
    }
    else
    {
        TLLM_LOG_ERROR("api parameter must be gptManager or executor");
        return 1;
    }

    return 0;
}
