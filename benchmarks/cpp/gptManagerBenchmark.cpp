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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "utils/utils.h"

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::benchmark;
namespace texec = tensorrt_llm::executor;
namespace trt = nvinfer1;
namespace fs = std::filesystem;

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

    // number of output tokens not calculated from output sequence here, instead set to max_output_len
    //   - if eos_id == -1 (default behavior), this is correct since output seq will have max permissible length.
    //   - However, if eos_id != -1, the token size of output sequence may be less than max_output_len, and token
    //   throughput may be inaccurate
    void recordStart(
        SizeType32 inputLength, uint64_t requestId, std::chrono::time_point<std::chrono::steady_clock> const& start)
    {
        TLLM_CHECK_WITH_INFO(mRequestBenchInfos.find(requestId) == mRequestBenchInfos.end(),
            "Request %lu already exists in record before start, please report a bug to developers.", requestId);
        std::lock_guard<std::mutex> const lock(mRequestBenchInfosMutex);
        mRequestBenchInfos[requestId] = BenchInfo(inputLength, start);
    }

    void recordToken(
        texec::Response const& response, std::chrono::time_point<std::chrono::steady_clock> const& tokenTime)
    {
        auto const requestId = response.getRequestId();
        auto outputTokenIds = response.getResult().outputTokenIds;

        int32_t outputLength = 1;
        for (auto const& beam : outputTokenIds)
        {
            outputLength = std::max(static_cast<int32_t>(beam.size()), outputLength);
        }

        std::lock_guard<std::mutex> const lock(mRequestBenchInfosMutex);
        mRequestBenchInfos[requestId].outputLength += outputLength;

        if (!mRequestBenchInfos[requestId].firstTokenSeen)
        {
            mRequestBenchInfos[requestId].firstTokenTs = tokenTime;
            mRequestBenchInfos[requestId].firstTokenSeen = true;
        }

        mRequestBenchInfos[requestId].decodingIter += 1;
    }

    void recordEnd(texec::Response const& response, std::chrono::time_point<std::chrono::steady_clock> const& end)
    {
        auto const requestId = response.getRequestId();
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
                std::lock_guard<std::mutex> const lock(mRequestBenchInfosMutex);
                mRequestBenchInfos[requestId].outputLength = outSeqLen;
                mRequestBenchInfos[requestId].decodingIter = response.getResult().decodingIter;

                // We record the first beam for the response file
                mResponseTensors[requestId] = outputTokenIds[0];
            }
            else
            {
                TLLM_CHECK_WITH_INFO(mBeamWidth == 1, "gptManagerBenchmark streaming mode does not support beam > 1");
                this->recordToken(response, end);
            }
        }

        std::lock_guard<std::mutex> const lock(mRequestBenchInfosMutex);
        mRequestBenchInfos[requestId].end = end;
        mRequestBenchInfos[requestId].hasError = response.hasError();
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
        std::vector<float> userTokensPerSecond;

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
                    if (reqInfo.second.avgGenT2TLatency.value() > 0)
                    {
                        userTokensPerSecond.push_back(1000.F / reqInfo.second.avgGenT2TLatency.value());
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

            if (!userTokensPerSecond.empty())
            {
                mAvgUserTokensPerSecond = std::accumulate(userTokensPerSecond.begin(), userTokensPerSecond.end(), 0.F)
                    / userTokensPerSecond.size();
                std::sort(userTokensPerSecond.begin(), userTokensPerSecond.end());
                mP99UserTokensPerSecond = calcPercentile(userTokensPerSecond, 99);
                mP90UserTokensPerSecond = calcPercentile(userTokensPerSecond, 90);
                mP50UserTokensPerSecond = calcPercentile(userTokensPerSecond, 50);
                mMaxUserTokensPerSecond = userTokensPerSecond.back();
                mMinUserTokensPerSecond = userTokensPerSecond.front();
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

            printf("[BENCHMARK] avg_user_tokens_per_second(tokens/sec/user) %.2f\n", mAvgUserTokensPerSecond);
            printf("[BENCHMARK] max_user_tokens_per_second(tokens/sec/user) %.2f\n", mMaxUserTokensPerSecond);
            printf("[BENCHMARK] min_user_tokens_per_second(tokens/sec/user) %.2f\n", mMinUserTokensPerSecond);
            printf("[BENCHMARK] p99_user_tokens_per_second(tokens/sec/user) %.2f\n", mP99UserTokensPerSecond);
            printf("[BENCHMARK] p90_user_tokens_per_second(tokens/sec/user) %.2f\n", mP90UserTokensPerSecond);
            printf("[BENCHMARK] p50_user_tokens_per_second(tokens/sec/user) %.2f\n\n", mP50UserTokensPerSecond);

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
                std::vector<std::string> streamingHeaders = {
                    "avg_time_to_first_token(ms)",
                    "max_time_to_first_token(ms)",
                    "min_time_to_first_token(ms)",
                    "p99_time_to_first_token(ms)",
                    "p90_time_to_first_token(ms)",
                    "p50_time_to_first_token(ms)",
                    "avg_inter_token_latency(ms)",
                    "max_inter_token_latency(ms)",
                    "min_inter_token_latency(ms)",
                    "p99_inter_token_latency(ms)",
                    "p90_inter_token_latency(ms)",
                    "p50_inter_token_latency(ms)",
                    "avg_user_tokens_per_second(tokens/sec/user)",
                    "max_user_tokens_per_second(tokens/sec/user)",
                    "min_user_tokens_per_second(tokens/sec/user)",
                    "p99_user_tokens_per_second(tokens/sec/user)",
                    "p90_user_tokens_per_second(tokens/sec/user)",
                    "p50_user_tokens_per_second(tokens/sec/user)",
                };

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
                               << mP99GenT2TLatency << "," << mP90GenT2TLatency << "," << mP50GenT2TLatency << ","
                               << mAvgUserTokensPerSecond << "," << mMaxUserTokensPerSecond << ","
                               << mMinUserTokensPerSecond << "," << mP99UserTokensPerSecond << ","
                               << mP90UserTokensPerSecond << "," << mP50UserTokensPerSecond << ",";
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
            auto respTokens = mResponseTensors[respId];
            int respLength = respTokens.size();
            int* respBufferPtr = respTokens.data();

            if (mOutputHasInput)
            {
                int inputSeqLen = mRequestBenchInfos[respId].inputLength;
                respBufferPtr += inputSeqLen;
                respLength -= inputSeqLen;
            }

            std::vector<int32_t> outputTokens(respLength);
            std::copy(respBufferPtr, respBufferPtr + respLength, outputTokens.begin());

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
    float mAvgUserTokensPerSecond{};
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
    float mP99UserTokensPerSecond{};
    float mP90UserTokensPerSecond{};
    float mP50UserTokensPerSecond{};
    float mMaxUserTokensPerSecond{};
    float mMinUserTokensPerSecond{};
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
    std::unordered_map<uint64_t, texec::VecTokens> mResponseTensors;
    bool mOutputHasInput;
    std::mutex mRequestBenchInfosMutex;

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
        texec::DynamicBatchConfig dynamicBatchConfig(
            benchmarkParams.enableBatchSizeTuning, benchmarkParams.enableMaxNumTokensTuning);
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
        executorConfig.setEnableTrtOverlap(benchmarkParams.enableTrtOverlap);
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

        auto decodingMode = texec::DecodingMode::Auto();
        if (benchmarkParams.medusaChoices.has_value())
        {
            decodingMode = texec::DecodingMode::Medusa();
        }
        else if (benchmarkParams.executorLookaheadConfig.has_value())
        {
            decodingMode = texec::DecodingMode::Lookahead();
        }
        else if (benchmarkParams.eagleConfig.has_value())
        {
            decodingMode = texec::DecodingMode::Eagle();
        }

        executorConfig.setDecodingConfig(texec::DecodingConfig(decodingMode, benchmarkParams.executorLookaheadConfig,
            benchmarkParams.medusaChoices, benchmarkParams.eagleConfig));
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
            for (auto const& request : requests)
            {
                inputLengths.push_back(request.getInputTokenIds().size());
            }
            auto const start = std::chrono::steady_clock::now();
            auto reqIds = mExecutor->enqueueRequests(std::move(requests));
            for (int req = 0; req < reqIds.size(); ++req)
            {
                if (!warmup)
                {
                    mRecorder->recordStart(inputLengths.at(req), reqIds.at(req), start);
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
            auto const tokenTime = std::chrono::steady_clock::now();
            for (auto const& response : responses)
            {
                if (response.getResult().isFinal)
                {
                    mActiveCount--;
                    mNumFinished++;
                    if (!warmup)
                    {
                        mRecorder->recordEnd(response, tokenTime);
                    }
                }
                else
                {
                    if (!warmup && !response.hasError())
                    {
                        mRecorder->recordToken(response, tokenTime);
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

texec::Request makeExecutorRequest(Sample const& sample, SizeType32 const& beamWidth,
    std::optional<SizeType32> const& eosId, std::optional<SizeType32> const& padId, bool streaming = false,
    bool const& returnContextLogits = false, bool const& returnGenerationLogits = false,
    std::optional<texec::LoraConfig> const& loraConfig = std::nullopt,
    std::optional<texec::LookaheadDecodingConfig> const& lookaheadConfig = std::nullopt,
    std::optional<texec::VecTokens> encoderInputTokenIds = std::nullopt,
    std::optional<float> temperature = std::nullopt)
{
    auto samplingConfig = texec::SamplingConfig{beamWidth};
    samplingConfig.setTemperature(temperature);
    auto outputConfig = texec::OutputConfig{false, returnContextLogits, returnGenerationLogits, false};
    return texec::Request(sample.inputIds, sample.outputLen, streaming, samplingConfig, outputConfig, eosId, padId,
        std::nullopt,    // positionIds
        std::nullopt,    // badWords
        std::nullopt,    // stopWords
        std::nullopt,    // embeddingBias
        std::nullopt,    // speculativeDecoding
        std::nullopt,    // pTuning
        std::nullopt,    // multimodalInput
        std::nullopt,    // multimodalEmbedding
        std::nullopt,    // mRopeConfig
        loraConfig,      // loraConfig
        lookaheadConfig, // lookaheadConfig
        std::nullopt,    // kvCacheRetentionConfig
        std::nullopt,    // logitsPostProcessorName
        std::nullopt,    // logitsPostProcessor
        encoderInputTokenIds.has_value() ? encoderInputTokenIds : std::nullopt,
        std::nullopt);   // cacheSaltID
}

void benchmarkExecutor(std::optional<std::filesystem::path> const& decoderEngineDir,
    std::optional<std::filesystem::path> const& encoderEngineDir, texec::BatchingType batchingType,
    std::string const& datasetPath, std::string const& opCsvFile, int maxNumSamples, int beamWidth, int warmUp,
    std::optional<int32_t> const& eosId, std::optional<int32_t> const& padId, BenchmarkParams const& benchmarkParams,
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy, std::chrono::milliseconds waitSleep,
    bool returnContextLogits, bool returnGenerationLogits, std::optional<int> const staticEmulatedBatchSize,
    bool logIterationData, std::optional<SizeType32> const maxPromptLen, texec::ModelType executorModelType,
    std::string const& responsesJsonFile)
{
    auto const& world = tensorrt_llm::mpi::MpiComm::world();
    auto worldRank = world.getRank();

    // Load dataset
    auto const samples = parseWorkloadJson(datasetPath, maxNumSamples, maxPromptLen);
    auto const numSamples = samples.size();

    auto recorder = std::make_shared<Recorder>(opCsvFile, benchmarkParams.streaming, beamWidth, responsesJsonFile);
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
                        benchmarkParams.requestLookaheadConfig, std::nullopt, benchmarkParams.temperature));
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
                        benchmarkParams.requestLookaheadConfig, std::nullopt, benchmarkParams.temperature));
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
        recorder->dumpResponseSeqs();
        // Send terminateReqId to terminate servers on all ranks
        // Sever on rank 0 will broadcast the terminate signal to other servers on multi-GPU cases
    }
}

} // namespace

int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "TensorRT LLM BatchManager Benchmark", "TensorRT LLM BatchManager Benchmark for GPT and GPT-like models.");
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
    options.add_options()("enable_max_num_tokens_tuning", "Dynamic tuning of max num tokens",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_exp_delays", "Enables exponential delay distr to mimic real world request arrival",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("streaming",
        "Operate in streaming mode. Note: it reflects time-to-first-token and inter-token-latency",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "enable_kv_cache_reuse", "Enables the KV cache reuse.", cxxopts::value<bool>()->default_value("true"));
    options.add_options()(
        "enable_chunked_context", "Whether to enable context chunking.", cxxopts::value<bool>()->default_value("true"));
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
        cxxopts::value<std::string>()->default_value("warning"));
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
    options.add_options()(
        "eagle_choices", "Eagle choices in the format of [[0], [0, 1], [0, 0, 1]]", cxxopts::value<std::string>());
    options.add_options()("eagle_posterior_threshold",
        "Minimum token probability threshold for typical acceptance. Enables typical acceptance in Eagle",
        cxxopts::value<float>());
    options.add_options()("temperature", "Sampling temperature for each request", cxxopts::value<float>());
    options.add_options()(
        "eagle_use_dynamic_tree", "Whether to use Eagle-2", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("eagle_dynamic_tree_max_top_k",
        "The max topK for dynamic tree, also the number of draft tokens that will expand for each node",
        cxxopts::value<SizeType32>());

    options.add_options()("multi_block_mode",
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel",
        cxxopts::value<bool>()->default_value("true"));
    options.add_options()("cuda_graph_mode", "When enabled, inference is executed with cuda graph.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("cuda_graph_cache_size",
        "Specify how many cuda graphs are cached in the runtime. Larger cache gives better perf, but consumes more GPU "
        "memory.",
        cxxopts::value<SizeType32>()->default_value("0"));
    options.add_options()("enable_trt_overlap", "Enable TRT Overlap", cxxopts::value<bool>()->default_value("false"));

    options.add_options()("enable_context_fmha_fp32_acc", "Enable FMHA runner FP32 accumulation",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("executor_lookahead_config",
        "lookahead config in the format of [max_window_size, max_ngram_size, max_verification_set_size]",
        cxxopts::value<std::string>());
    options.add_options()("request_lookahead_config",
        "lookahead config in the format of [max_window_size, max_ngram_size, max_verification_set_size], and each <= "
        "executor lookahead config",
        cxxopts::value<std::string>());
    options.add_options()("responses_json", "Write output response sequences to a json file",
        cxxopts::value<std::string>()->default_value(""));

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
        bool streaming = result["streaming"].as<bool>();
        if (streaming)
        {
            TLLM_LOG_ERROR("Streaming is not supported in static batching.\n");
            return 1;
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

    // Argument: Enable dynamic tuning of max num tokens
    benchmarkParams.enableMaxNumTokensTuning = result["enable_max_num_tokens_tuning"].as<bool>();

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
    // Argument: Eagle choices for the Eagle speculative decoding.
    if (result.count("eagle_choices") || result.count("eagle_posterior_threshold")
        || result.count("eagle_use_dynamic_tree") || result.count("eagle_dynamic_tree_max_top_k"))
    {
        std::optional<float> posteriorThreshold;
        if (result.count("eagle_posterior_threshold"))
        {
            posteriorThreshold = result["eagle_posterior_threshold"].as<float>();
        }
        std::optional<texec::EagleChoices> choices;
        if (result.count("eagle_choices"))
        {
            choices = parseVectorOfVectors(result["eagle_choices"].as<std::string>());
        }
        bool eagleUseDynamicTree = false;
        if (result.count("eagle_use_dynamic_tree"))
        {
            eagleUseDynamicTree = result["eagle_use_dynamic_tree"].as<bool>();
        }
        std::optional<SizeType32> eagleDynamicTreeMaxTopK;
        if (result.count("eagle_dynamic_tree_max_top_k"))
        {
            eagleDynamicTreeMaxTopK = result["eagle_dynamic_tree_max_top_k"].as<SizeType32>();
        }
        benchmarkParams.eagleConfig = texec::EagleConfig(
            choices, !posteriorThreshold.has_value(), posteriorThreshold, eagleUseDynamicTree, eagleDynamicTreeMaxTopK);
    }
    if (result.count("temperature"))
    {
        benchmarkParams.temperature = result["temperature"].as<float>();
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

    // Argument: enable_trt_overlap
    benchmarkParams.enableTrtOverlap = result["enable_trt_overlap"].as<bool>();

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

    // Argument: output sequences JSON
    auto const responsesJsonFile = result["responses_json"].as<std::string>();

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
                maxPromptLen, executorModelType, responsesJsonFile);
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
