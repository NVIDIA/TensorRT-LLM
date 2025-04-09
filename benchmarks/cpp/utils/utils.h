
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

#include "tensorrt_llm/executor/executor.h"

#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#pragma once

namespace tensorrt_llm::benchmark
{

// using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::runtime;

namespace texec = tensorrt_llm::executor;

std::vector<std::vector<SizeType32>> parseVectorOfVectors(std::string const& input);

texec::LookaheadDecodingConfig parseLookaheadConfig(std::string const& input);

struct BenchmarkParams
{
    std::optional<SizeType32> maxTokensInPagedKvCache{std::nullopt};
    std::optional<float> freeGpuMemoryFraction{std::nullopt};
    std::vector<std::optional<float>> freeGpuMemoryFractions{std::nullopt};

    std::optional<float> crossKvCacheFraction{std::nullopt};
    bool enableTrtOverlap{false};
    bool enableBatchSizeTuning{false};
    bool enableMaxNumTokensTuning{false};
    bool enableBlockReuse{false};
    bool enableChunkedContext{true};
    bool streaming{false};
    bool enableExpDelays{false};
    std::vector<std::optional<bool>> enableChunekedContextVec{std::nullopt};
    std::optional<float> requestRate{std::nullopt};
    std::optional<int> concurrency{std::nullopt};
    std::optional<SizeType32> maxBatchSize{std::nullopt};
    std::vector<std::optional<SizeType32>> maxBatchSizes{std::nullopt};
    std::optional<SizeType32> maxNumTokens{std::nullopt};
    std::vector<std::optional<SizeType32>> maxNumTokensVec{std::nullopt};
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

    std::optional<texec::EagleConfig> eagleConfig;
    std::optional<float> temperature;

    std::optional<texec::LookaheadDecodingConfig> executorLookaheadConfig;
    std::optional<texec::LookaheadDecodingConfig> requestLookaheadConfig;

    bool enableCollectkvCacheTransferTime = false;
    bool enableCollectIterStats = false;
};

struct RecordTimeMetric
{

    RecordTimeMetric(std::string tag)
        : mTag(std::move(tag))
    {
    }

    std::string mTag;

    std::vector<float> mDataTimes;

    float mAvg;
    float mP99;
    float mP95;
    float mP90;
    float mP50;
    float mMax;
    float mMin;

    static float calcPercentile(std::vector<float> const& latencies, int percentile)
    {
        int const index = static_cast<int>(std::ceil((percentile / 100.0) * latencies.size())) - 1;
        return latencies[index];
    }

    void calculate()
    {
        TLLM_CHECK_WITH_INFO(mDataTimes.size() > 0, "No data to calculate for tag:%s", mTag.c_str());
        mAvg = std::accumulate(mDataTimes.begin(), mDataTimes.end(), 0.F) / mDataTimes.size();

        std::sort(mDataTimes.begin(), mDataTimes.end());

        mP99 = calcPercentile(mDataTimes, 99);
        mP90 = calcPercentile(mDataTimes, 90);
        mP50 = calcPercentile(mDataTimes, 50);
        mMax = mDataTimes.back();
        mMin = mDataTimes.front();
    }

    void report() const
    {

        printf("[BENCHMARK] avg_%s(ms) %.2f\n", mTag.c_str(), mAvg);
        printf("[BENCHMARK] max_%s(ms) %.2f\n", mTag.c_str(), mMax);
        printf("[BENCHMARK] min_%s(ms) %.2f\n", mTag.c_str(), mMin);

        printf("[BENCHMARK] p99_%s(ms) %.2f\n", mTag.c_str(), mP99);

        printf("[BENCHMARK] p90_%s(ms) %.2f\n", mTag.c_str(), mP90);

        printf("[BENCHMARK] p50_%s(ms) %.2f\n\n", mTag.c_str(), mP50);
    }

    std::vector<std::string> genHeaders() const
    {
        std::string timeTag = mTag + "(ms)";
        return {
            "avg_" + timeTag, "max_" + timeTag, "min_" + timeTag, "p99" + timeTag, "p90" + timeTag, "p50" + timeTag};
    }
};

struct RecordBwMetric
{

    RecordBwMetric(std::string tag)
        : mTag(std::move(tag))
    {
    }

    std::string mTag;

    std::vector<float> mDataTps;

    float mAvg;
    float mP99;
    float mP95;
    float mP90;
    float mP50;
    float mMax;
    float mMin;

    static float calcPercentile(std::vector<float> const& throughputs, int percentile)
    {
        int const index = static_cast<int>(std::ceil((percentile / 100.0) * throughputs.size())) - 1;
        return throughputs[index];
    }

    void calculate()
    {
        TLLM_CHECK_WITH_INFO(mDataTps.size() > 0, "No data to calculate for tag:%s", mTag.c_str());
        mAvg = std::accumulate(mDataTps.begin(), mDataTps.end(), 0.F) / mDataTps.size();

        std::sort(mDataTps.begin(), mDataTps.end(), std::greater<float>());

        mP99 = calcPercentile(mDataTps, 99);
        mP90 = calcPercentile(mDataTps, 90);
        mP50 = calcPercentile(mDataTps, 50);
        mMax = mDataTps.front();
        mMin = mDataTps.back();
    }

    void report() const
    {

        printf("[BENCHMARK] avg_%s(Gb/sec) %.8f\n", mTag.c_str(), mAvg);
        printf("[BENCHMARK] max_%s(Gb/sec) %.8f\n", mTag.c_str(), mMax);
        printf("[BENCHMARK] min_%s(Gb/sec) %.8f\n", mTag.c_str(), mMin);

        printf("[BENCHMARK] p99_%s(Gb/sec) %.8f\n", mTag.c_str(), mP99);

        printf("[BENCHMARK] p90_%s(Gb/sec) %.8f\n", mTag.c_str(), mP90);

        printf("[BENCHMARK] p50_%s(Gb/sec) %.8f\n\n", mTag.c_str(), mP50);
    }

    std::vector<std::string> genHeaders() const
    {
        std::string tpTag = mTag + "(Gb/sec)";
        return {"avg_" + tpTag, "max_" + tpTag, "min_" + tpTag, "p99" + tpTag, "p90" + tpTag, "p50" + tpTag};
    }
};

std::ostream& operator<<(std::ostream& os, RecordTimeMetric const& metric);
std::ostream& operator<<(std::ostream& os, RecordBwMetric const& metric);

struct Sample
{
    std::vector<int32_t> inputIds;
    int32_t outputLen;
    int32_t taskId;
};

using Samples = std::vector<Sample>;

Samples parseWorkloadJson(
    std::filesystem::path const& datasetPath, int maxNumSamples, std::optional<SizeType32> const maxPromptLen);

std::vector<double> generateRandomExponentialValues(int count, float lambda, int seed);

std::vector<double> computeTimeDelays(BenchmarkParams const& benchmarkParams, int numDelays);

} // namespace tensorrt_llm::benchmark
