
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

#include "utils.h"
#include "tensorrt_llm/common/logger.h"
#include <random>

#include <filesystem>
#include <fstream>

namespace tensorrt_llm::benchmark
{

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

    if (samples.size() < maxNumSamples)
    {
        TLLM_LOG_WARNING(
            "Dataset size %zu is smaller than given max_num_samples %d, max_num_samples will be ignored.\n",
            samples.size(), maxNumSamples);
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

std::ostream& operator<<(std::ostream& os, RecordTimeMetric const& metric)
{
    os << metric.mAvg << "," << metric.mMax << "," << metric.mMin << "," << metric.mP99 << "," << metric.mP90 << ","
       << metric.mP50;
    return os;
}

std::ostream& operator<<(std::ostream& os, RecordBwMetric const& metric)
{
    os << metric.mAvg << "," << metric.mMax << "," << metric.mMin << "," << metric.mP99 << "," << metric.mP90 << ","
       << metric.mP50;
    return os;
}

} // namespace tensorrt_llm::benchmark
