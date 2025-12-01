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
#include "tensorrt_llm/executor/disaggServerUtil.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "utils/utils.h"

#include "cxxopts.hpp"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::benchmark;
using namespace tensorrt_llm::executor::disagg_executor;
namespace texec = tensorrt_llm::executor;
namespace trt = nvinfer1;

namespace
{

class Recorder
{

public:
    explicit Recorder(std::string opCsvFile, bool streaming = false, int beamWidth = 1,
        bool calculateKvCacheTransferTime = true, bool calculateQueueTime = true, std::string responsesJsonFile = "",
        bool excludeInputInOutput = false)
        : mOpCsvFile(std::move(opCsvFile))
        , mStreaming(streaming)
        , mBeamWidth(beamWidth)
        , mRespJsonFile(std::move(responsesJsonFile))
        , mOutputHasInput(!excludeInputInOutput)
        , mCalculateKVCacheTransferTime(calculateKvCacheTransferTime)
        , mCalculateQueueTime(calculateQueueTime)
    {
    }

    void initialize()
    {
        mStart = std::chrono::steady_clock::now();
        mSeqLatency.mDataTimes.clear();
        mFtLatency.mDataTimes.clear();
        mGenLatency.mDataTimes.clear();
        mGenFirstTokenLatency.mDataTimes.clear();
        mGenT2TLatency.mDataTimes.clear();
        mGenExcludeFirstIterT2TLatency.mDataTimes.clear();
        mContextReqQueuingLatency.mDataTimes.clear();
        mGenReqQueuingLatency.mDataTimes.clear();
        mGenReqKvCacheTransferLatency.mDataTimes.clear();
        mKvCacheThroughput.mDataTps.clear();
    }

    void finalize()
    {
        mEnd = std::chrono::steady_clock::now();
    }

    void recordContextQueueLatency(std::vector<float> const& latencies)
    {
        mContextReqQueuingLatency.mDataTimes.insert(
            mContextReqQueuingLatency.mDataTimes.end(), latencies.begin(), latencies.end());
    }

    void recordGenQueueLatency(std::vector<float> const& latencies)
    {
        mGenReqQueuingLatency.mDataTimes.insert(
            mGenReqQueuingLatency.mDataTimes.end(), latencies.begin(), latencies.end());
    }

    void recordKvCacheTransferLatency(std::vector<float> const& latencies)
    {
        mGenReqKvCacheTransferLatency.mDataTimes.insert(
            mGenReqKvCacheTransferLatency.mDataTimes.end(), latencies.begin(), latencies.end());
    }

    void recordKvCacheThroughput(std::vector<float> const& throughputs)
    {
        mKvCacheThroughput.mDataTps.insert(mKvCacheThroughput.mDataTps.end(), throughputs.begin(), throughputs.end());
    }

    void recordContextStart(SizeType32 inputLength, SizeType32 maxNewTokens, uint64_t requestId,
        std::chrono::time_point<std::chrono::steady_clock> const& start)
    {
        mRequestBenchInfos[requestId] = BenchInfo(inputLength, start);
    }

    void recordContextEnd(tensorrt_llm::executor::IdType requestId, bool hasError)
    {
        TLLM_CHECK(mRequestBenchInfos.find(requestId) != mRequestBenchInfos.end());
        mRequestBenchInfos.at(requestId).contextEnd = std::chrono::steady_clock::now();
        mRequestBenchInfos.at(requestId).contextHasError = hasError;
        mRequestBenchInfos.at(requestId).decodingIter += 1;
    }

    void recordToken(tensorrt_llm::executor::IdType requestId)
    {
        TLLM_CHECK(mStreaming);
        TLLM_CHECK_WITH_INFO(mBeamWidth == 1, "gptManagerBenchmark streaming mode does not support beam > 1");
        TLLM_CHECK(mRequestBenchInfos.find(requestId) != mRequestBenchInfos.end());

        if (!mRequestBenchInfos.at(requestId).genFirstTokenSeen)
        {
            mRequestBenchInfos.at(requestId).genFirstTokenTs = std::chrono::steady_clock::now();
            mRequestBenchInfos.at(requestId).genFirstTokenSeen = true;
        }
        mRequestBenchInfos.at(requestId).decodingIter += 1;
    }

    void recordToken(tensorrt_llm::executor::IdType requestId, texec::Response const& response)
    {

        TLLM_CHECK(mRequestBenchInfos.find(requestId) != mRequestBenchInfos.end());

        auto outputTokenIds = response.getResult().outputTokenIds;

        int32_t outputLength = 1;
        for (auto const& beam : outputTokenIds)
        {
            outputLength = std::max(static_cast<int32_t>(beam.size()), outputLength);
        }

        mRequestBenchInfos[requestId].outputLength += outputLength;
        this->recordToken(requestId);
    }

    void recordGenStart(
        tensorrt_llm::executor::IdType requestId, std::chrono::time_point<std::chrono::steady_clock> const& start)
    {

        TLLM_CHECK(mRequestBenchInfos.find(requestId) != mRequestBenchInfos.end());
        mRequestBenchInfos.at(requestId).genStart = start;
    }

    void recordGenEnd(tensorrt_llm::executor::IdType requestId, bool hasError)
    {
        TLLM_CHECK(mRequestBenchInfos.find(requestId) != mRequestBenchInfos.end());
        mRequestBenchInfos.at(requestId).genEnd = std::chrono::steady_clock::now();
        mRequestBenchInfos.at(requestId).genHasError = hasError;
    }

    void recordGenEnd(tensorrt_llm::executor::IdType requestId, texec::Response const& response)
    {
        recordGenEnd(requestId, response.hasError());
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
                recordToken(requestId, response);
            }
        }
    }

    void reserve(size_t size)
    {
        mRequestBenchInfos.reserve(size);
    }

    void calculateLatencies()
    {
        for (auto& reqInfo : mRequestBenchInfos)
        {

            reqInfo.second.latency
                = std::chrono::duration<float, std::milli>(reqInfo.second.genEnd - reqInfo.second.contextStart).count();
            reqInfo.second.firstTokenLatency
                = std::chrono::duration<float, std::milli>(reqInfo.second.contextEnd - reqInfo.second.contextStart)
                      .count();
            reqInfo.second.genLatency
                = std::chrono::duration<float, std::milli>(reqInfo.second.genEnd - reqInfo.second.genStart).count();
            if (mStreaming)
            {
                reqInfo.second.genFirstTokenLatency
                    = std::chrono::duration<float, std::milli>(reqInfo.second.genFirstTokenTs - reqInfo.second.genStart)
                          .count();
                // include the latency of the second token+ kv Cache transfer latency

                if (reqInfo.second.outputLength > 1)
                {
                    reqInfo.second.avgGenT2TLatency
                        = std::chrono::duration<float, std::milli>(reqInfo.second.genEnd - reqInfo.second.genStart)
                              .count()
                        / static_cast<float>(reqInfo.second.outputLength - 1);
                }
                if (reqInfo.second.outputLength > 2)
                {
                    reqInfo.second.avgGenExcludeFirstIterT2TLatency
                        = std::chrono::duration<float, std::milli>(
                              reqInfo.second.genEnd - reqInfo.second.genFirstTokenTs)
                              .count()
                        / static_cast<float>(reqInfo.second.outputLength - 2);
                }
            }
        }
    }

    void calculateMetrics()
    {

        calculateLatencies();

        int totalOutputTokens{0};
        int totalDecodingIter{0};
        mNumContextErrorSamples = 0;
        mNumGenErrorSamples = 0;
        mNumSamples = 0;
        for (auto const& reqInfo : mRequestBenchInfos)
        {

            if (!reqInfo.second.contextHasError && !reqInfo.second.genHasError)
            {
                mSeqLatency.mDataTimes.push_back(reqInfo.second.latency);
                mNumSamples++;
            }
            if (!reqInfo.second.contextHasError)
            {
                mFtLatency.mDataTimes.push_back(reqInfo.second.firstTokenLatency);
            }
            else
            {
                mNumContextErrorSamples++;
            }
            if (!reqInfo.second.genHasError)
            {
                mGenLatency.mDataTimes.push_back(reqInfo.second.genLatency);
                totalOutputTokens += reqInfo.second.outputLength;
                totalDecodingIter += reqInfo.second.decodingIter;
                if (mStreaming)
                {
                    mGenFirstTokenLatency.mDataTimes.push_back(reqInfo.second.genFirstTokenLatency);

                    if (reqInfo.second.avgGenT2TLatency.has_value())
                    {
                        mGenT2TLatency.mDataTimes.push_back(reqInfo.second.avgGenT2TLatency.value());
                    }
                    if (reqInfo.second.avgGenExcludeFirstIterT2TLatency.has_value())
                    {
                        mGenExcludeFirstIterT2TLatency.mDataTimes.push_back(
                            reqInfo.second.avgGenExcludeFirstIterT2TLatency.value());
                    }
                }
            }
            else
            {
                mNumGenErrorSamples++;
            }
        }
        mTotalLatency = std::chrono::duration<float, std::milli>(mEnd - mStart).count();
        mSeqThroughput = mNumSamples / (mTotalLatency / 1000);
        mTokenThroughput = totalOutputTokens / (mTotalLatency / 1000);
        mAcceptanceRate = totalDecodingIter
            ? (static_cast<float>(totalOutputTokens) / static_cast<float>(totalDecodingIter))
            : 0.0F;

        mSeqLatency.calculate();
        mFtLatency.calculate();
        mGenLatency.calculate();
        if (mStreaming)
        {

            mGenFirstTokenLatency.calculate();

            if (!mGenT2TLatency.mDataTimes.empty())
            {
                mGenT2TLatency.calculate();
                std::vector<float> userTokensPerSecond;
                userTokensPerSecond.reserve(mGenT2TLatency.mDataTimes.size());
                for (auto const& latency : mGenT2TLatency.mDataTimes)
                {
                    userTokensPerSecond.push_back(1000.F / latency);
                }
                mAvgUserTokensPerSecond = std::accumulate(userTokensPerSecond.begin(), userTokensPerSecond.end(), 0.F)
                    / userTokensPerSecond.size();
            }
            if (!mGenExcludeFirstIterT2TLatency.mDataTimes.empty())
            {

                mGenExcludeFirstIterT2TLatency.calculate();
            }
        }
        if (mCalculateQueueTime)
        {

            mContextReqQueuingLatency.calculate();
            mGenReqQueuingLatency.calculate();
        }
        if (mCalculateKVCacheTransferTime)
        {
            mGenReqKvCacheTransferLatency.calculate();
            mKvCacheThroughput.calculate();
        }
    }

    void report()
    {
        printf("[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] num_context_error_samples %d\n", mNumContextErrorSamples);
        printf("[BENCHMARK] num_gen_error_samples %d\n", mNumGenErrorSamples);
        printf("\n[BENCHMARK] num_samples %d\n", mNumSamples);
        printf("[BENCHMARK] total_latency(ms) %.2f\n", mTotalLatency);
        printf("[BENCHMARK] seq_throughput(seq/sec) %.2f\n", mSeqThroughput);
        printf("[BENCHMARK] token_throughput(token/sec) %.2f\n", mTokenThroughput);
        if (mStreaming)
        {
            printf("[BENCHMARK] user_tokens_per_second(tokens/sec/user) %.2f\n", mAvgUserTokensPerSecond);
        }
        printf("[BENCHMARK] avg_acceptance_rate(tokens/decoding steps) %.2f\n\n", mAcceptanceRate);

        mSeqLatency.report();
        mFtLatency.report();
        mGenLatency.report();
        if (mStreaming)
        {
            mGenFirstTokenLatency.report();
            mGenT2TLatency.report();
            mGenExcludeFirstIterT2TLatency.report();
        }
        if (mCalculateQueueTime)
        {
            mContextReqQueuingLatency.report();
            mGenReqQueuingLatency.report();
        }
        if (mCalculateKVCacheTransferTime)
        {
            mGenReqKvCacheTransferLatency.report();
            mKvCacheThroughput.report();
        }
    }

    void writeOpMetricsToCsv()
    {
        if (!mOpCsvFile.empty())
        {
            std::vector<std::string> headers{"num_samples", "num_context_error_samples", "num_gen_error_samples",
                "total_latency(ms)", "seq_throughput(seq/sec)", "token_throughput(token/sec)"};
            auto seqLatencyHeader = mSeqLatency.genHeaders();
            headers.insert(headers.end(), std::make_move_iterator(seqLatencyHeader.begin()),
                std::make_move_iterator(seqLatencyHeader.end()));
            auto contextLatencyHeader = mFtLatency.genHeaders();
            headers.insert(headers.end(), std::make_move_iterator(contextLatencyHeader.begin()),
                std::make_move_iterator(contextLatencyHeader.end()));
            auto genLatencyHeader = mGenLatency.genHeaders();
            headers.insert(headers.end(), std::make_move_iterator(genLatencyHeader.begin()),
                std::make_move_iterator(genLatencyHeader.end()));
            if (mStreaming)
            {
                auto genFirstTokenHeader = mGenFirstTokenLatency.genHeaders();
                headers.insert(headers.end(), std::make_move_iterator(genFirstTokenHeader.begin()),
                    std::make_move_iterator(genFirstTokenHeader.end()));
                auto genIngterHeader = mGenT2TLatency.genHeaders();
                headers.insert(headers.end(), std::make_move_iterator(genIngterHeader.begin()),
                    std::make_move_iterator(genIngterHeader.end()));
                auto excludeFirstIterIngterHeader = mGenExcludeFirstIterT2TLatency.genHeaders();
                headers.insert(headers.end(), std::make_move_iterator(excludeFirstIterIngterHeader.begin()),
                    std::make_move_iterator(excludeFirstIterIngterHeader.end()));
                headers.push_back("avg_user_tokens_per_second(tokens/sec/user)");
            }
            if (mCalculateKVCacheTransferTime)
            {
                auto genReqKVCacheTransferHeader = mGenReqKvCacheTransferLatency.genHeaders();
                headers.insert(headers.end(), std::make_move_iterator(genReqKVCacheTransferHeader.begin()),
                    std::make_move_iterator(genReqKVCacheTransferHeader.end()));
                auto kvCacheTpHeader = mKvCacheThroughput.genHeaders();
                headers.insert(headers.end(), std::make_move_iterator(kvCacheTpHeader.begin()),
                    std::make_move_iterator(kvCacheTpHeader.end()));
            }

            std::ofstream outputFile(mOpCsvFile);

            if (outputFile.is_open())
            {
                for (auto const& header : headers)
                {
                    outputFile << header << ",";
                }
                outputFile << "\n";

                outputFile << mNumSamples << "," << mNumContextErrorSamples << "," << mNumGenErrorSamples << ","
                           << mTotalLatency << "," << mSeqThroughput << "," << mTokenThroughput << "," << mSeqLatency
                           << "," << mFtLatency << "," << mGenLatency;
                if (mStreaming)
                {

                    outputFile << "," << mGenFirstTokenLatency << "," << mGenT2TLatency << ","
                               << mGenExcludeFirstIterT2TLatency << "," << mAvgUserTokensPerSecond;
                }
                if (mCalculateKVCacheTransferTime)
                {
                    outputFile << "," << mGenReqKvCacheTransferLatency << "," << mKvCacheThroughput;
                }

                outputFile << "\n";
            }
            else
            {
                std::cerr << "Error opening file '" << mOpCsvFile << "' for writing.\n";
            }
        }
    }

private:
    struct BenchInfo
    {
        BenchInfo() = default;

        BenchInfo(int inputLength, std::chrono::time_point<std::chrono::steady_clock> start)
            : inputLength(inputLength)
            , contextStart(start)
        {
        }

        int inputLength{};
        int outputLength{};
        std::chrono::time_point<std::chrono::steady_clock> contextStart;
        std::chrono::time_point<std::chrono::steady_clock> contextEnd;
        std::chrono::time_point<std::chrono::steady_clock> genFirstTokenTs;
        std::chrono::time_point<std::chrono::steady_clock> genStart;
        std::chrono::time_point<std::chrono::steady_clock> genEnd;
        float latency{}; // millisecond
        float genLatency{};
        bool contextHasError{false};
        bool genHasError{false};
        float firstTokenLatency{};
        float genFirstTokenLatency{};
        std::optional<float> avgGenT2TLatency;
        std::optional<float> avgGenExcludeFirstIterT2TLatency;
        bool genFirstTokenSeen{false};
        SizeType32 decodingIter{0};
    };

    std::unordered_map<uint64_t, BenchInfo> mRequestBenchInfos;

    std::chrono::time_point<std::chrono::steady_clock> mStart;
    std::chrono::time_point<std::chrono::steady_clock> mEnd;
    int mNumSamples{};
    int mNumContextErrorSamples{};
    int mNumGenErrorSamples{};
    float mTotalLatency{};
    float mSeqThroughput{};
    RecordTimeMetric mSeqLatency{"sequence_latency"};
    RecordTimeMetric mFtLatency{"context_latency"};
    RecordTimeMetric mGenLatency{"gen_latency"};

    RecordTimeMetric mGenFirstTokenLatency{"time_to_gen_first_token"};
    RecordTimeMetric mGenT2TLatency{"inter_token_latency"};
    RecordTimeMetric mGenExcludeFirstIterT2TLatency{"exclude_first_iter_inter_token_latency"};
    RecordTimeMetric mContextReqQueuingLatency{"context_req_queueing_latency"};

    RecordTimeMetric mGenReqQueuingLatency{"gen_req_queueing_latency"};
    RecordTimeMetric mGenReqKvCacheTransferLatency{"gen_req_kv_cache_transfer_latency"};

    RecordBwMetric mKvCacheThroughput{"gen_req_kv_cache_transfer_throughput"};

    float mTokenThroughput{};
    float mAcceptanceRate{};

    std::string mOpCsvFile;
    bool mStreaming;
    int mBeamWidth;
    std::string mRespJsonFile;
    std::unordered_map<uint64_t, tensorrt_llm::executor::TensorPtr> mResponseTensors;
    bool mOutputHasInput;
    bool mCalculateKVCacheTransferTime;
    bool mCalculateQueueTime;
    float mAvgUserTokensPerSecond{};
};

texec::Request makeExecutorContextRequest(Sample const& sample, SizeType32 const& beamWidth,
    std::optional<SizeType32> const& eosId, std::optional<SizeType32> const& padId, bool streaming = false,
    bool const& returnContextLogits = false, bool const& returnGenerationLogits = false,
    std::optional<texec::LoraConfig> const& loraConfig = std::nullopt,
    std::optional<texec::LookaheadDecodingConfig> const& lookaheadConfig = std::nullopt,
    std::optional<texec::VecTokens> const& encoderInputTokenIds = std::nullopt)
{
    auto samplingConfig = texec::SamplingConfig{beamWidth};
    auto outputConfig = texec::OutputConfig{false, returnContextLogits, returnGenerationLogits, false};
    auto request
        = texec::Request(sample.inputIds, sample.outputLen, streaming, samplingConfig, outputConfig, eosId, padId,
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
    request.setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY);
    return request;
}

class DisaggExecutorServer
{

public:
    DisaggExecutorServer(std::vector<std::filesystem::path> const& contextEnginePaths,
        std::vector<std::filesystem::path> const& genEnginePaths,
        std::optional<std::vector<std::vector<SizeType32>>> const& deviceIdsForInstance, int32_t maxBeamWidth,
        texec::CapacitySchedulerPolicy capacitySchedulerPolicy, BenchmarkParams const& benchmarkParams,
        std::shared_ptr<Recorder> recorder, std::chrono::milliseconds waitSleep, bool logIterationData,
        bool hasContextAwaitThreads, bool hasGenAwaitThreads)
        : mRecorder(std::move(recorder))
        , mWaitSleep(waitSleep)
        , mConcurrency(benchmarkParams.concurrency)
        , mShutdown(false)
        , mLogIterationData(logIterationData)
        , mEnableCollectKvCacheTransferTime(benchmarkParams.enableCollectkvCacheTransferTime)
        , mEnableCollectIterStats(benchmarkParams.enableCollectIterStats)
    {

        int worldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        int worldSize = tensorrt_llm::mpi::MpiComm::world().getSize();
        mIsOrchestrator = (worldRank == 0);
        auto contextNum = contextEnginePaths.size();
        auto genNum = genEnginePaths.size();
        int deviceCount = -1;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        std::vector<std::unique_ptr<tensorrt_llm::executor::Executor>> instances;
        auto instanceNum = genNum + contextNum;
        if (worldRank == 0)
        {
            TLLM_LOG_INFO("context enigne num :%d gen enigne num:%d", contextNum, genNum);
        }

        int startRank = 0;
        std::vector<texec::ExecutorConfig> ctxExecutorConfigs;
        std::vector<texec::ExecutorConfig> genExecutorConfigs;
        for (auto in = 0; in < instanceNum; in++)
        {
            auto&& enginePath = in < contextNum ? contextEnginePaths.at(in) : genEnginePaths.at(in - contextNum);
            auto decoderJsonConfig = tensorrt_llm::runtime::GptJsonConfig::parse(enginePath / "config.json");
            size_t instanceRanks = decoderJsonConfig.getWorldSize();
            std::vector<SizeType32> participateRank(instanceRanks);
            std::vector<SizeType32> deviceIds;
            if (deviceIdsForInstance.has_value())
            {
                deviceIds = deviceIdsForInstance.value().at(in);
            }
            for (int i = 0; i < instanceRanks; i++)
            {
                startRank++;
                participateRank.at(i) = startRank;
                if (!deviceIdsForInstance.has_value())
                {
                    deviceIds.push_back((startRank - 1) % deviceCount);
                }
            }
            texec::DynamicBatchConfig dynamicBatchConfig(benchmarkParams.enableBatchSizeTuning);
            texec::SchedulerConfig schedulerConfig(capacitySchedulerPolicy, std::nullopt, dynamicBatchConfig);
            texec::KvCacheConfig kvCacheConfig(benchmarkParams.enableBlockReuse,
                benchmarkParams.maxTokensInPagedKvCache, benchmarkParams.maxAttentionWindowVec,
                benchmarkParams.sinkTokenLength, benchmarkParams.freeGpuMemoryFractions.at(in),
                benchmarkParams.kvHostCacheSize, benchmarkParams.kvOnboardBlocks);
            texec::ExtendedRuntimePerfKnobConfig extendedRuntimePerfKnobConfig(benchmarkParams.multiBlockMode,
                benchmarkParams.enableContextFMHAFP32Acc, benchmarkParams.cudaGraphMode,
                benchmarkParams.cudaGraphCacheSize);
            texec::ExecutorConfig executorConfig(maxBeamWidth, schedulerConfig, kvCacheConfig,
                benchmarkParams.enableChunekedContextVec.at(in).value_or(false));
            executorConfig.setGpuWeightsPercent(benchmarkParams.gpuWeightsPercent);
            texec::OrchestratorConfig orchestratorConfig{mIsOrchestrator, "", nullptr, false};
            texec::ParallelConfig parallelConfig{tensorrt_llm::executor::CommunicationType::kMPI,
                tensorrt_llm::executor::CommunicationMode::kORCHESTRATOR, deviceIds, participateRank,
                orchestratorConfig};
            executorConfig.setParallelConfig(parallelConfig);
            if (benchmarkParams.maxBatchSizes.at(in))
            {
                executorConfig.setMaxBatchSize(benchmarkParams.maxBatchSizes.at(in).value());
            }
            if (benchmarkParams.maxNumTokensVec.at(in))
            {
                executorConfig.setMaxNumTokens(benchmarkParams.maxNumTokensVec.at(in).value());
            }

            executorConfig.setDecodingConfig(
                texec::DecodingConfig(benchmarkParams.medusaChoices.has_value() ? texec::DecodingMode::Medusa()
                        : benchmarkParams.executorLookaheadConfig.has_value()   ? texec::DecodingMode::Lookahead()
                                                                                : texec::DecodingMode::Auto(),
                    benchmarkParams.executorLookaheadConfig, benchmarkParams.medusaChoices));
            executorConfig.setExtendedRuntimePerfKnobConfig(extendedRuntimePerfKnobConfig);
            executorConfig.setCacheTransceiverConfig(
                texec::CacheTransceiverConfig(texec::CacheTransceiverConfig::BackendType::DEFAULT));
            constexpr int maxIterationsForRequestStats = 1000;
            if (mEnableCollectKvCacheTransferTime)
            {
                executorConfig.setRequestStatsMaxIterations(maxIterationsForRequestStats);
            }
            if (!benchmarkParams.enableCollectIterStats)
            {
                executorConfig.setIterStatsMaxIterations(0);
            }

            if (in < contextNum)
            {
                ctxExecutorConfigs.push_back(executorConfig);
            }
            else
            {
                genExecutorConfigs.push_back(executorConfig);
            }
        }

        mDisaggExecutor = std::make_unique<DisaggExecutorOrchestrator>(contextEnginePaths, genEnginePaths,
            ctxExecutorConfigs, genExecutorConfigs, hasContextAwaitThreads, hasGenAwaitThreads);

        if (mIsOrchestrator)
        {

            if (mEnableCollectIterStats || mEnableCollectKvCacheTransferTime)
            {
                mCollectStatsThread = std::thread(&DisaggExecutorServer::collectStats, this);
            }
        }
        tensorrt_llm::mpi::MpiComm::world().barrier();
    }

    std::vector<tensorrt_llm::executor::IdType> enqueueContext(std::vector<texec::Request> const& requests,
        std::optional<int> selectContextId = std::nullopt, bool warmup = false, bool batch = false)
    {
        std::vector<SizeType32> inputLengths;
        std::vector<SizeType32> maxNewTokens;
        if (!warmup)
        {
            for (auto const& request : requests)
            {
                inputLengths.push_back(static_cast<SizeType32>(request.getInputTokenIds().size()));
                maxNewTokens.push_back(request.getMaxTokens());
            }
        }
        auto const start = std::chrono::steady_clock::now();
        std::vector<tensorrt_llm::executor::IdType> globalReqIds
            = mDisaggExecutor->enqueueContext(requests, selectContextId, batch);
        if (!warmup)
        {
            for (size_t i = 0; i < requests.size(); ++i)
            {
                mRecorder->recordContextStart(inputLengths.at(i), maxNewTokens.at(i), globalReqIds.at(i), start);
            }
        }
        mNumContextActive += requests.size();
        return globalReqIds;
    }

    void enqueueGeneration(std::vector<texec::Request> const& requests,
        std::vector<tensorrt_llm::executor::IdType> const& globalRequestIds,
        std::optional<int> selectGenIdx = std::nullopt, bool warmup = false, bool batch = false)
    {
        TLLM_CHECK(globalRequestIds.size() == requests.size());
        auto const start = std::chrono::steady_clock::now();
        mDisaggExecutor->enqueueGeneration(requests, globalRequestIds, selectGenIdx, batch);
        if (!warmup)
        {
            for (int i = 0; i < requests.size(); i++)
            {

                mRecorder->recordGenStart(globalRequestIds.at(i), start);
            }
        }
        mNumGenActive += requests.size();
    }

    std::vector<ResponseWithId> waitForContextResponse(SizeType32 numRequests, bool warmup = false)
    {
        std::vector<ResponseWithId> ret;
        ret.reserve(numRequests);
        while ((mNumContextActive != 0) || (mNumContextFinished < numRequests))
        {
            auto responses = mDisaggExecutor->awaitContextResponses(mWaitSleep);
            for (auto&& response : responses)
            {
                TLLM_CHECK(response.response.getResult().isFinal);
                if (response.response.getResult().isFinal)
                {
                    mNumContextActive--;
                    mNumContextFinished++;
                }
                if (!warmup)
                {
                    mRecorder->recordContextEnd(response.gid, response.response.hasError());
                }
                ret.emplace_back(std::move(response));
            }
        }
        return ret;
    }

    void waitForGenResponse(SizeType32 numRequests, bool warmup = false)
    {
        while (mNumGenActive > 0 || (mNumGenFinished < numRequests))
        {
            auto responses = mDisaggExecutor->awaitGenerationResponses(mWaitSleep);
            for (auto&& response : responses)
            {
                if (response.response.getResult().isFinal)
                {
                    mNumGenActive--;
                    mNumGenFinished++;

                    if (!warmup)
                    {
                        mRecorder->recordGenEnd(response.gid, response.response);
                    }
                }
                else
                {
                    // streaming
                    if (!warmup && !response.response.hasError())
                    {
                        mRecorder->recordToken(response.gid, response.response);
                    }
                }
            }
        }
    }

    bool canEnqueue(int numSentRequests) const
    {
        return mIsOrchestrator && (!mConcurrency || (numSentRequests - mNumGenFinished < mConcurrency));
    }

    ~DisaggExecutorServer()
    {
        mShutdown = true;
        if (mCollectStatsThread.joinable())
        {
            mCollectStatsThread.join();
        }
    }

    void resetNumFinished()
    {
        mNumContextFinished = 0;
        mNumGenFinished = 0;
    }

    void resetNumActive()
    {
        mNumContextActive = 0;
        mNumGenActive = 0;
    }

    void collectStats() const
    {
        while (!mShutdown)
        {
            std::vector<std::deque<tensorrt_llm::executor::IterationStats>> contextStats;
            std::vector<std::deque<tensorrt_llm::executor::IterationStats>> generationStats;
            std::vector<std::deque<tensorrt_llm::executor::RequestStatsPerIteration>>
                generationRequestStatsPerIteration;
            contextStats.reserve(mDisaggExecutor->getContextExecutors().size());
            for (auto&& executor : mDisaggExecutor->getContextExecutors())
            {
                if (executor->canEnqueueRequests())
                {
                    contextStats.emplace_back(executor->getLatestIterationStats());
                }
            }
            generationStats.reserve(mDisaggExecutor->getGenExecutors().size());
            for (auto&& executor : mDisaggExecutor->getGenExecutors())
            {
                if (executor->canEnqueueRequests())
                {
                    if (mEnableCollectIterStats)
                    {
                        generationStats.emplace_back(executor->getLatestIterationStats());
                    }
                    if (mEnableCollectKvCacheTransferTime)
                    {

                        generationRequestStatsPerIteration.emplace_back(executor->getLatestRequestStats());
                    }
                }
            }
            if (mEnableCollectIterStats)
            {
                for (std::size_t i = 0; i < contextStats.size(); i++)
                {
                    auto const& iterStats = contextStats.at(i);
                    for (auto const& stat : iterStats)
                    {
                        SizeType32 numNewActiveRequests = stat.numNewActiveRequests;
                        if (numNewActiveRequests > 0)
                        {
                            auto avgQueueingTime
                                = static_cast<float>(stat.newActiveRequestsQueueLatencyMS / numNewActiveRequests);
                            std::vector<float> requestsQueueLatencyMS(numNewActiveRequests, avgQueueingTime);
                            mRecorder->recordContextQueueLatency(requestsQueueLatencyMS);
                        }
                        if (mLogIterationData)
                        {
                            TLLM_LOG_INFO(
                                "ctx_id %d, ctx_stat: %s", i, texec::JsonSerialization::toJsonStr(stat).c_str());
                        }
                    }
                }

                for (std::size_t i = 0; i < generationStats.size(); i++)
                {
                    auto const& iterStats = generationStats.at(i);
                    for (auto const& stat : iterStats)
                    {
                        SizeType32 numNewActiveRequests = stat.numNewActiveRequests;
                        if (numNewActiveRequests > 0)
                        {
                            float avgQueueingTime
                                = static_cast<float>(stat.newActiveRequestsQueueLatencyMS / numNewActiveRequests);
                            std::vector<float> requestsQueueLatencyMS(numNewActiveRequests, avgQueueingTime);
                            mRecorder->recordGenQueueLatency(requestsQueueLatencyMS);
                        }
                        if (mLogIterationData)
                        {
                            TLLM_LOG_INFO(
                                "gen_id %d, gen_stat: %s", i, texec::JsonSerialization::toJsonStr(stat).c_str());
                        }
                    }
                }
            }

            if (mEnableCollectKvCacheTransferTime)
            {
                for (std::size_t i = 0; i < generationRequestStatsPerIteration.size(); i++)
                {
                    auto const& stats = generationRequestStatsPerIteration.at(i);
                    for (auto const& stat : stats)
                    {
                        std::vector<float> kvCacheTransferMs;
                        std::vector<float> kvCacheThroughput;
                        for (auto const& requestStat : stat.requestStats)
                        {
                            if (requestStat.stage == tensorrt_llm::executor::RequestStage::kGENERATION_COMPLETE)
                            {
                                kvCacheTransferMs.push_back(
                                    static_cast<float>(requestStat.disServingStats->kvCacheTransferMS));
                                kvCacheThroughput.push_back(static_cast<float>(requestStat.disServingStats->kvCacheSize)
                                    * 8 / (static_cast<float>(requestStat.disServingStats->kvCacheTransferMS) / 1000)
                                    / 1e9f);
                            }
                        }
                        if (kvCacheTransferMs.size() > 0)
                        {
                            mRecorder->recordKvCacheTransferLatency(kvCacheTransferMs);
                        }
                        if (kvCacheThroughput.size() > 0)
                        {
                            mRecorder->recordKvCacheThroughput(kvCacheThroughput);
                        }
                        if (mLogIterationData)
                        {
                            TLLM_LOG_INFO(
                                "gen_id %d, gen_req_stat: %s", i, texec::JsonSerialization::toJsonStr(stat).c_str());
                        }
                    }
                }
            }
            auto const waitSleep = std::chrono::milliseconds(50);
            std::this_thread::sleep_for(waitSleep);
        }
    }

    std::unique_ptr<DisaggExecutorOrchestrator> const& getDisaggExecutor() const noexcept
    {
        return mDisaggExecutor;
    }

private:
    std::unique_ptr<DisaggExecutorOrchestrator> mDisaggExecutor;

    std::atomic<bool> mShutdown{false};
    bool mIsOrchestrator{false};

    std::shared_ptr<Recorder> mRecorder;
    std::chrono::milliseconds mWaitSleep;
    std::optional<int> mConcurrency;
    bool mLogIterationData{false};
    bool const mEnableCollectKvCacheTransferTime;
    bool const mEnableCollectIterStats;
    std::thread mCollectStatsThread;
    std::atomic<uint64_t> mNumGenFinished{0};
    std::atomic<uint64_t> mNumContextFinished{0};
    std::atomic<uint64_t> mNumGenActive{0};
    std::atomic<uint64_t> mNumContextActive{0};
};

} // namespace

void benchmark(std::vector<std::filesystem::path> const& contextEngineDirs,
    std::vector<std::filesystem::path> const& generationEngineDirs,
    std::optional<std::vector<std::vector<int>>> const& deviceIdsForInstances, std::string const& datasetPath,
    std::string const& opCsvFile, int maxNumSamples, int beamWidth, int warmUp, std::optional<int32_t> const& eosId,
    std::optional<int32_t> const& padId, BenchmarkParams const& benchmarkParams,
    texec::CapacitySchedulerPolicy capacitySchedulerPolicy, std::chrono::milliseconds waitSleep,
    bool returnContextLogits, bool returnGenerationLogits, std::optional<int> const staticEmulatedBatchSize,
    bool logIterationData, std::optional<SizeType32> const maxPromptLen, bool hasContextAwait, bool hasGenAwait)
{

    auto const& world = tensorrt_llm::mpi::MpiComm::world();
    auto worldRank = world.getRank();

    // Load dataset
    auto const samples = parseWorkloadJson(datasetPath, maxNumSamples, maxPromptLen);
    auto const numSamples = samples.size();
    auto recorder = std::make_shared<Recorder>(opCsvFile, benchmarkParams.streaming, beamWidth,
        benchmarkParams.enableCollectkvCacheTransferTime, benchmarkParams.enableCollectIterStats);
    auto disaggExecutor = std::make_shared<DisaggExecutorServer>(contextEngineDirs, generationEngineDirs,
        deviceIdsForInstances, beamWidth, capacitySchedulerPolicy, benchmarkParams, recorder, waitSleep,
        logIterationData, hasContextAwait, hasGenAwait);
    constexpr size_t numMap = 8;
    std::vector<std::unordered_map<tensorrt_llm::executor::IdType, tensorrt_llm::executor::Request>> gidToRequestMaps(
        numMap);
    std::vector<std::mutex> mtxForMaps(numMap);

    auto fillRequestMap = [&](std::vector<tensorrt_llm::executor::IdType> const& reqIds,
                              std::vector<tensorrt_llm::executor::Request>&& requests)
    {
        TLLM_CHECK(reqIds.size() == requests.size());
        for (size_t i = 0; i < reqIds.size(); i++)
        {

            size_t mapIdx = reqIds[i] % numMap;
            std::scoped_lock<std::mutex> lock(mtxForMaps[mapIdx]);
            gidToRequestMaps.at(mapIdx).emplace(reqIds[i], std::move(requests[i]));
        }
    };

    auto makeGenRequest = [&](std::vector<ResponseWithId>&& contextResponse)
    {
        std::vector<tensorrt_llm::executor::IdType> gids;
        gids.reserve(contextResponse.size());
        std::vector<tensorrt_llm::executor::Request> genRequest;
        genRequest.reserve(contextResponse.size());
        for (auto&& ctxResponse : contextResponse)
        {
            gids.emplace_back(ctxResponse.gid);
            size_t mapIdx = ctxResponse.gid % numMap;

            std::unique_lock<std::mutex> lock(mtxForMaps[mapIdx]);
            TLLM_CHECK(gidToRequestMaps.at(mapIdx).find(ctxResponse.gid) != gidToRequestMaps.at(mapIdx).end());
            auto ctxRequest = std::move(gidToRequestMaps.at(mapIdx).at(ctxResponse.gid));
            gidToRequestMaps.at(mapIdx).erase(ctxResponse.gid);
            lock.unlock();
            ctxRequest.setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_GENERATION_ONLY);
            ctxRequest.setContextPhaseParams(ctxResponse.response.getResult().contextPhaseParams.value());
            genRequest.emplace_back(std::move(ctxRequest));
        }
        return std::make_pair(genRequest, gids);
    };
    if (worldRank == 0)
    {
        { // warmup
            TLLM_LOG_INFO("Warmup start");

            size_t contextNum = contextEngineDirs.size();
            size_t generationNum = generationEngineDirs.size();
            for (auto con = 0; con < contextNum; con++)
            {
                for (auto gen = 0; gen < generationNum; gen++)
                {
                    std::vector<tensorrt_llm::executor::Request> contextRequests;
                    contextRequests.reserve(warmUp);
                    for (int i = 0; i < warmUp; ++i)
                    {
                        contextRequests.emplace_back(makeExecutorContextRequest(samples[0], beamWidth, eosId, padId,
                            benchmarkParams.streaming, returnContextLogits, returnGenerationLogits, std::nullopt,
                            benchmarkParams.requestLookaheadConfig));
                    }
                    auto reqIds = disaggExecutor->enqueueContext(contextRequests, con, true);
                    fillRequestMap(reqIds, std::move(contextRequests));
                    auto contextResponse = disaggExecutor->waitForContextResponse(warmUp, true);
                    auto&& [genRequests, gids] = makeGenRequest(std::move(contextResponse));
                    disaggExecutor->enqueueGeneration(genRequests, gids, gen, true);
                    disaggExecutor->waitForGenResponse(warmUp, true);
                    disaggExecutor->resetNumFinished();
                    disaggExecutor->resetNumActive();
                }
            }

            auto const warmUpWaitSleep = std::chrono::milliseconds(50);
            std::this_thread::sleep_for(warmUpWaitSleep);
            TLLM_LOG_INFO("Warmup done");
        }

        {

            auto timeDelays = computeTimeDelays(benchmarkParams, numSamples - 1);

            std::vector<texec::Request> contextRequests;

            for (std::size_t i = 0; i < numSamples; ++i)
            {
                std::optional<texec::LoraConfig> loraConfig = std::nullopt;
                contextRequests.emplace_back(makeExecutorContextRequest(samples[i], beamWidth, eosId, padId,
                    benchmarkParams.streaming, returnContextLogits, returnGenerationLogits, loraConfig,
                    benchmarkParams.requestLookaheadConfig));
            }

            bool const hasDelay
                = std::any_of(timeDelays.begin(), timeDelays.end(), [](auto const& delay) { return delay > 0.0; });
            disaggExecutor->resetNumFinished();
            disaggExecutor->resetNumActive();

            recorder->reserve(numSamples);
            recorder->initialize();
            if (!staticEmulatedBatchSize)
            {

                std::thread waitContextResponseAndEnqueGenThread{[&]()
                    {
                        auto numRequest = numSamples;
                        while (numRequest > 0)
                        {
                            auto contextResponseWithIds
                                = disaggExecutor->getDisaggExecutor()->awaitContextResponses(waitSleep);
                            if (contextResponseWithIds.empty())
                            {
                                continue;
                            }
                            for (auto&& contextResponseWithId : contextResponseWithIds)
                            {
                                recorder->recordContextEnd(
                                    contextResponseWithId.gid, contextResponseWithId.response.hasError());
                            }
                            numRequest -= contextResponseWithIds.size();
                            auto&& [genReqeust, genGids] = makeGenRequest(std::move(contextResponseWithIds));
                            disaggExecutor->enqueueGeneration(genReqeust, genGids);
                        }
                    }};

                std::thread waitGenResponseThread{[&]() { disaggExecutor->waitForGenResponse(numSamples); }};
                int numSentRequests = 0;
                while (numSentRequests < numSamples)
                {

                    if (disaggExecutor->canEnqueue(numSentRequests))
                    {
                        auto gids = disaggExecutor->enqueueContext({contextRequests.at(numSentRequests)});
                        fillRequestMap(gids, {contextRequests.at(numSentRequests)});

                        if (hasDelay && numSentRequests < numSamples - 1)
                        {
                            std::this_thread::sleep_for(
                                std::chrono::milliseconds(static_cast<int>(timeDelays.at(numSentRequests) * 1000)));
                        }
                        numSentRequests += 1;
                    }
                }
                waitContextResponseAndEnqueGenThread.join();
                waitGenResponseThread.join();
            }
            else
            {
                TLLM_CHECK_WITH_INFO(
                    !hasDelay, "Executor benchmark doesn't support delays with emulated static batch sizes");
                auto numRequests = contextRequests.size();
                int maxBatchSize = staticEmulatedBatchSize.value();
                for (int req = 0; req < numRequests; req += maxBatchSize)
                {
                    auto batchSize = std::min(static_cast<size_t>(maxBatchSize), numRequests - req);

                    std::vector<texec::Request> requestsBatch(std::make_move_iterator(contextRequests.begin() + req),
                        std::make_move_iterator(contextRequests.begin() + req + static_cast<int64_t>(batchSize)));
                    // Enqueue in batches

                    auto reqIds = disaggExecutor->enqueueContext(requestsBatch);
                    fillRequestMap(reqIds, std::move(requestsBatch));
                    auto contextResponse = disaggExecutor->waitForContextResponse(static_cast<SizeType32>(batchSize));
                    auto&& [genRequests, genReqIds] = makeGenRequest(std::move(contextResponse));
                    disaggExecutor->enqueueGeneration(genRequests, genReqIds);
                    disaggExecutor->waitForGenResponse(static_cast<SizeType32>(batchSize));

                    // Wait for current batch to be done
                }
            }
        }
        recorder->finalize();
        // sleep for collect stats
        if (benchmarkParams.enableCollectIterStats || benchmarkParams.enableCollectkvCacheTransferTime)
        {
            auto const collectWaitSleep = std::chrono::milliseconds(50);
            std::this_thread::sleep_for(collectWaitSleep);
        }
        recorder->calculateMetrics();
        recorder->report();
        recorder->writeOpMetricsToCsv();
    }
}

int main(int argc, char* argv[])

{
    cxxopts::Options options("TensorRT LLM DisaggServer Benchmark");
    options.add_options()("h,help", "Print usage");
    options.add_options()("context_engine_dirs", "Directories that store context engines,separator is a ,",
        cxxopts::value<std::vector<std::string>>());
    options.add_options()("generation_engine_dirs", "Directories that store generation engines,separator is a , ",
        cxxopts::value<std::vector<std::string>>());
    options.add_options()("device_ids_for_instances",
        "device ids  for each instances , example: \"[[0,1],[2,3],[4,5,6,7]]\" ", cxxopts::value<std::string>());
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
    options.add_options()("kv_cache_free_gpu_mem_fractions", "K-V Cache Free Gpu Mem Fraction,each for per instance",
        cxxopts::value<std::vector<float>>());
    options.add_options()("request_rate",
        "request rate in reqs/sec. Skipping this arg or negative value will trigger offline/0-delay.",
        cxxopts::value<float>());
    options.add_options()("concurrency", "Concurrent number of connections with the server.", cxxopts::value<int>());
    options.add_options()("max_batch_sizes", "The max runtime batch size when benchmarking, each for per instance",
        cxxopts::value<std::vector<int>>());
    options.add_options()("max_num_tokens_per_instance",
        "The max runtime number of tokens per batch when benchmarking, each for per instance",
        cxxopts::value<std::vector<int>>());
    options.add_options()(
        "enable_batch_size_tuning", "Dynamic tuning of batch size", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_exp_delays", "Enables exponential delay distr to mimic real world request arrival",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("streaming", "Operate in streaming mode", cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "enable_kv_cache_reuse", "Enables the KV cache reuse.", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_chunked_context_per_instance", "Whether to enable context chunking for per instance",
        cxxopts::value<std::vector<bool>>()->default_value("false"));
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
    options.add_options()("context_await", "When enabled, will has a thread to await context response.",
        cxxopts::value<bool>()->default_value("true"));
    options.add_options()("gen_await", "When enabled,will has a thread to await gen response.",
        cxxopts::value<bool>()->default_value("true"));
    options.add_options()("enable_collect_kvcache_transfer_time", "When enabled, will collect kvcache transfer time.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("enable_collect_iter_stats", "When enabled, will collect iteration stats.",
        cxxopts::value<bool>()->default_value("false"));

    auto result = options.parse(argc, argv);

    if ((result.count("context_engine_dirs") == 0) || (result.count("generation_engine_dirs") == 0))
    {
        std::cout << options.help() << std::endl;
        TLLM_LOG_ERROR("Please specify context engine and generation engine directory.");
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

    // Argument: Dataset
    auto const datasetPath = result["dataset"].as<std::string>();
    auto const maxNumSamples = result["max_num_samples"].as<int>();

    // Argument: Output metrics CSV
    auto const opCsvFile = result["output_csv"].as<std::string>();

    // Argument: beam width
    auto const beamWidth = result["beam_width"].as<int>();
    TLLM_CHECK_WITH_INFO(beamWidth == 1, "Currently only support beamWidth=1");
    // Argument: wait_sleep
    auto const waitSleep = std::chrono::milliseconds(result["wait_sleep"].as<int>());
    auto const hasContextAwait = result["context_await"].as<bool>();
    auto const hasGenAwait = result["gen_await"].as<bool>();
    BenchmarkParams benchmarkParams;
    benchmarkParams.enableCollectkvCacheTransferTime = result["enable_collect_kvcache_transfer_time"].as<bool>();
    benchmarkParams.enableCollectIterStats = result["enable_collect_iter_stats"].as<bool>();

    std::vector<std::string> contextEngineDirs = result["context_engine_dirs"].as<std::vector<std::string>>();
    std::vector<std::string> generationEngineDirs = result["generation_engine_dirs"].as<std::vector<std::string>>();
    if (tensorrt_llm::mpi::MpiComm::world().getRank() == 0)
    {
        std::string contextEngineStrings;
        for (auto&& contextEngineDir : contextEngineDirs)
        {
            contextEngineStrings += contextEngineDir + ",";
        }
        std::string generationEnginesStrings;
        for (auto&& genEngineDir : generationEngineDirs)
        {
            generationEnginesStrings += genEngineDir + ",";
        }
        TLLM_LOG_INFO(
            "Will Launch benchmark with %d context engines and %d generation engines. Context Engines:%s ; Generation "
            "Engines:%s ;",
            contextEngineDirs.size(), generationEngineDirs.size(), contextEngineStrings.c_str(),
            generationEnginesStrings.c_str());
    }
    std::vector<std::filesystem::path> contextEnigePaths;
    std::vector<std::filesystem::path> generationEnginePaths;

    contextEnigePaths.reserve(contextEngineDirs.size());

    for (auto& contextEngineDir : contextEngineDirs)
    {

        contextEnigePaths.emplace_back(contextEngineDir);
    }
    generationEnginePaths.reserve(generationEngineDirs.size());
    for (auto& genEngineDir : generationEngineDirs)
    {

        generationEnginePaths.emplace_back(genEngineDir);
    }

    int const instanceNum = contextEngineDirs.size() + generationEngineDirs.size();
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
    benchmarkParams.freeGpuMemoryFractions.resize(instanceNum);
    if (result.count("kv_cache_free_gpu_mem_fractions"))
    {
        auto fractions = result["kv_cache_free_gpu_mem_fractions"].as<std::vector<float>>();
        TLLM_CHECK_WITH_INFO(fractions.size() == instanceNum || fractions.size() == 1,
            "the number of fraction should be equal to the number of instances or equal to 1");
        for (int i = 0; i < instanceNum; i++)
        {
            benchmarkParams.freeGpuMemoryFractions.at(i) = fractions.size() == 1 ? fractions[0] : fractions[i];
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

    // Argument: max_batch_sizes
    benchmarkParams.maxBatchSizes.resize(instanceNum);
    if (result.count("max_batch_sizes"))
    {
        auto batchSizes = result["max_batch_sizes"].as<std::vector<int>>();
        TLLM_CHECK_WITH_INFO(batchSizes.size() == instanceNum || batchSizes.size() == 1,
            "the number of batch size should be equal to the number of instances or equal to 1");
        for (int i = 0; i < instanceNum; i++)
        {
            benchmarkParams.maxBatchSizes.at(i) = batchSizes.size() == 1 ? batchSizes[0] : batchSizes[i];
        }
    }

    // Argument: max_num_tokens_per_instance
    benchmarkParams.maxNumTokensVec.resize(instanceNum);
    if (result.count("max_num_tokens_per_instance"))
    {
        auto maxNumTokensVec = result["max_num_tokens_per_instance"].as<std::vector<int>>();
        TLLM_CHECK_WITH_INFO(maxNumTokensVec.size() == instanceNum || maxNumTokensVec.size() == 1,
            "the number of max_num_tokens should be equal to the number of instances or equal to 1");
        for (int i = 0; i < instanceNum; i++)
        {
            benchmarkParams.maxNumTokensVec.at(i)
                = maxNumTokensVec.size() == 1 ? maxNumTokensVec[0] : maxNumTokensVec[i];
        }
    }

    benchmarkParams.enableExpDelays = result["enable_exp_delays"].as<bool>();

    // Argument: Enable batch stats output
    bool logIterationData = result["log_iteration_data"].as<bool>();

    // Argument: Enable chunked context
    benchmarkParams.enableChunekedContextVec.resize(instanceNum);
    if (result.count("enable_chunked_context_per_instance"))
    {
        auto enableChunkedContextVec = result["enable_chunked_context_per_instance"].as<std::vector<bool>>();

        TLLM_CHECK_WITH_INFO(enableChunkedContextVec.size() == instanceNum || enableChunkedContextVec.size() == 1,
            "the number of enable_chunked_context_per_instance should be equal to the number of instances or equal to "
            "1");
        for (int i = 0; i < instanceNum; i++)
        {
            benchmarkParams.enableChunekedContextVec.at(i)
                = enableChunkedContextVec.size() == 1 ? enableChunkedContextVec[0] : enableChunkedContextVec[i];
        }
    }
    // Argument: Enable return context logits
    bool returnContextLogits = result["return_context_logits"].as<bool>();
    TLLM_CHECK_WITH_INFO(returnContextLogits == false, "Currently disaggServer don't support returnContextLogits!");
    // Argument: Enable return context logits
    bool returnGenerationLogits = result["return_generation_logits"].as<bool>();
    TLLM_CHECK_WITH_INFO(
        returnGenerationLogits == false, "Currently disaggServer don't support returnGenerationLogits!");

    if (result.count("lora_dir"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support lora!");
        benchmarkParams.loraDir = result["lora_dir"].as<std::string>();
    }
    if (result.count("lora_host_cache_bytes"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support lora!");

        benchmarkParams.loraHostCacheSize = result["lora_host_cache_bytes"].as<size_t>();
    }
    if (result.count("lora_num_device_mod_layers"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support lora!");

        benchmarkParams.loraDeviceNumModLayers = result["lora_num_device_mod_layers"].as<SizeType32>();
    }

    // Argument: How many KV cache blocks (as fraction of number of GPU kv cache blocks).
    benchmarkParams.kvHostCacheSize = result["kv_host_cache_bytes"].as<size_t>();
    TLLM_CHECK_WITH_INFO(
        benchmarkParams.kvHostCacheSize == false, "Currently disaggServer don't support kv_host_cache!");

    // Argument: If offloaded blocks should be onboarded to primary memory before they are reused.
    benchmarkParams.kvOnboardBlocks = result["kv_onboard_blocks"].as<bool>();
    TLLM_CHECK_WITH_INFO(
        benchmarkParams.kvOnboardBlocks == true, "Currently disaggServer don't support kv_onboard_blocks =false!");
    // Argument: Medusa choices for the Medusa speculative decoding.
    if (result.count("medusa_choices"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support medusa!");

        benchmarkParams.medusaChoices = parseVectorOfVectors(result["medusa_choices"].as<std::string>());
    }
    if (result.count("executor_lookahead_config"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support lookhead!");

        benchmarkParams.executorLookaheadConfig
            = parseLookaheadConfig(result["executor_lookahead_config"].as<std::string>());
    }
    if (result.count("request_lookahead_config"))
    {
        TLLM_CHECK_WITH_INFO(false, "Currently disaggServer don't support lookhead!");

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

    std::optional<std::chrono::milliseconds> batchTimeout;

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

    std::optional<std::vector<std::vector<int>>> deviceIdsForInstance = std::nullopt;
    if (result.count("device_ids_for_instances"))
    {
        deviceIdsForInstance = parseVectorOfVectors(result["device_ids_for_instances"].as<std::string>());
    }
    benchmark(contextEnigePaths, generationEnginePaths, deviceIdsForInstance, datasetPath, opCsvFile, maxNumSamples,
        beamWidth, result["warm_up"].as<int>(), eosId, padId, benchmarkParams, capacitySchedulerPolicy, waitSleep,
        returnContextLogits, returnContextLogits, staticEmulatedBatchSize, logIterationData, maxPromptLen,
        hasContextAwait, hasGenAwait);
}
