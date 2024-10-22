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

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include <cxxopts.hpp>

namespace tle = tensorrt_llm::executor;

namespace fs = std::filesystem;

struct RuntimeOptions
{
    std::string trtEnginePath;
    std::string inputTokensCsvFile;
    std::string outputTokensCsvFile;

    bool streaming;
    bool excludeInputFromOutput;
    tle::SizeType32 maxNewTokens;
    tle::SizeType32 beamWidth;
    std::optional<tle::SizeType32> numReturnSequences;
    tle::SizeType32 timeoutMs;

    bool useOrchestratorMode;
    std::string workerExecutablePath;
};

// Utility function to parse input arguments
RuntimeOptions parseArgs(int argc, char* argv[]);

// Function that enqueues requests
std::vector<tle::IdType> enqueueRequests(RuntimeOptions const& runtimeOpts, tle::Executor& executor);

// Function that waits for responses and stores output tokens
std::unordered_map<tle::IdType, tle::BeamTokens> waitForResponses(
    RuntimeOptions const& runtimeOpts, std::vector<tle::IdType> const& requestIds, tle::Executor& executor);

// Utility function to read input tokens from csv file
std::vector<tle::VecTokens> readInputTokens(std::string const& path);

// Utility function to write output tokens from csv file
void writeOutputTokens(std::string const& path, std::vector<tle::IdType>& requestIds,
    std::unordered_map<tle::IdType, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth);

tle::SizeType32 getNumSequencesPerRequest(RuntimeOptions const& runtimeOpts);

// Main
int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    auto runtimeOpts = parseArgs(argc, argv);

    // Create the executor for this engine
    auto executorConfig = tle::ExecutorConfig(runtimeOpts.beamWidth);

    if (runtimeOpts.useOrchestratorMode)
    {
        auto orchestratorConfig = tle::OrchestratorConfig(true, runtimeOpts.workerExecutablePath);
        auto parallelConfig = tle::ParallelConfig(tle::CommunicationType::kMPI, tle::CommunicationMode::kORCHESTRATOR,
            std::nullopt, std::nullopt, orchestratorConfig);
        executorConfig.setParallelConfig(parallelConfig);
    }

    auto executor = tle::Executor(runtimeOpts.trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    if (executor.canEnqueueRequests())
    {
        // Create the requests
        auto requestIds = enqueueRequests(runtimeOpts, executor);

        // Wait for responses and store output tokens
        auto outputTokens = waitForResponses(runtimeOpts, requestIds, executor);

        // Write output tokens csv file
        TLLM_LOG_INFO("Writing output tokens to %s", runtimeOpts.outputTokensCsvFile.c_str());
        auto numSequences = getNumSequencesPerRequest(runtimeOpts);
        writeOutputTokens(runtimeOpts.outputTokensCsvFile, requestIds, outputTokens, numSequences);
    }
    TLLM_LOG_INFO("Exiting.");
    return 0;
}

RuntimeOptions parseArgs(int argc, char* argv[])
{
    RuntimeOptions runtimeOpts;

    cxxopts::Options options(argv[0], "Example that demonstrates how to use the Executor API");
    options.add_options()("h,help", "Print usage");
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()("beam_width", "The beam width", cxxopts::value<int>()->default_value("1"));
    options.add_options()(
        "num_return_sequences", "The number of return sequences per request.", cxxopts::value<std::optional<int>>());
    options.add_options()("streaming", "Operate in streaming mode", cxxopts::value<bool>()->default_value("false"));
    options.add_options()("exclude_input_from_output",
        "Exclude input tokens when writing output tokens. Only has effect for streaming = false. For streaming = true, "
        "output tokens are not included.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()(
        "max_new_tokens", "The maximum number of tokens to generate", cxxopts::value<int>()->default_value("10"));
    options.add_options()(
        "input_tokens_csv_file", "Path to a csv file that contains input tokens", cxxopts::value<std::string>());
    options.add_options()("output_tokens_csv_file", "Path to a csv file that will contain the output tokens",
        cxxopts::value<std::string>()->default_value("outputTokens.csv"));
    options.add_options()("timeout_ms", "The maximum time to wait for all responses, in milliseconds.",
        cxxopts::value<int>()->default_value("10000"));
    options.add_options()("use_orchestrator_mode", "Use orchestrator communication mode.",
        cxxopts::value<bool>()->default_value("false"));
    options.add_options()("worker_executable_path", "The location of the worker executable.",
        cxxopts::value<std::string>()->default_value(""));

    auto parsedOptions = options.parse(argc, argv);

    // Argument: help
    if (parsedOptions.count("help"))
    {
        TLLM_LOG_ERROR(options.help());
        exit(0);
    }

    // Argument: Engine directory
    if (!parsedOptions.count("engine_dir"))
    {
        TLLM_LOG_ERROR(options.help());
        TLLM_LOG_ERROR("Please specify engine directory.");
        exit(1);
    }
    runtimeOpts.trtEnginePath = parsedOptions["engine_dir"].as<std::string>();
    if (!fs::exists(runtimeOpts.trtEnginePath) || !fs::is_directory(runtimeOpts.trtEnginePath))
    {
        TLLM_LOG_ERROR("Engine directory doesn't exist.");
        exit(1);
    }

    // Argument: Input tokens csv file
    if (!parsedOptions.count("input_tokens_csv_file"))
    {
        TLLM_LOG_ERROR(options.help());
        TLLM_LOG_ERROR("Please specify input_tokens_csv_file");
        exit(1);
    }
    runtimeOpts.inputTokensCsvFile = parsedOptions["input_tokens_csv_file"].as<std::string>();
    runtimeOpts.streaming = parsedOptions["streaming"].as<bool>();
    runtimeOpts.excludeInputFromOutput = parsedOptions["exclude_input_from_output"].as<bool>();
    runtimeOpts.maxNewTokens = parsedOptions["max_new_tokens"].as<int>();
    runtimeOpts.beamWidth = parsedOptions["beam_width"].as<int>();
    if (parsedOptions.count("num_return_sequences") > 0)
    {
        runtimeOpts.numReturnSequences = parsedOptions["num_return_sequences"].as<std::optional<int>>();
    }
    runtimeOpts.timeoutMs = parsedOptions["timeout_ms"].as<int>();
    runtimeOpts.outputTokensCsvFile = parsedOptions["output_tokens_csv_file"].as<std::string>();

    runtimeOpts.useOrchestratorMode = parsedOptions["use_orchestrator_mode"].as<bool>();
    runtimeOpts.workerExecutablePath = parsedOptions["worker_executable_path"].as<std::string>();

    return runtimeOpts;
}

std::vector<tle::IdType> enqueueRequests(RuntimeOptions const& runtimeOpts, tle::Executor& executor)
{
    tle::OutputConfig outputConfig;
    outputConfig.excludeInputFromOutput = runtimeOpts.excludeInputFromOutput;
    tle::SamplingConfig samplingConfig(runtimeOpts.beamWidth);
    if (runtimeOpts.numReturnSequences && runtimeOpts.beamWidth == 1)
    {
        samplingConfig.setTopP(0.9);
    }
    samplingConfig.setNumReturnSequences(runtimeOpts.numReturnSequences);

    TLLM_LOG_INFO("Reading input tokens from %s", runtimeOpts.inputTokensCsvFile.c_str());
    auto inputTokens = readInputTokens(runtimeOpts.inputTokensCsvFile);
    TLLM_LOG_INFO("Number of requests: %d", inputTokens.size());

    std::vector<tle::Request> requests;
    for (auto& tokens : inputTokens)
    {
        TLLM_LOG_INFO("Creating request with %d input tokens", tokens.size());
        requests.emplace_back(
            std::move(tokens), runtimeOpts.maxNewTokens, runtimeOpts.streaming, samplingConfig, outputConfig);
    }

    // Enqueue the requests
    auto requestIds = executor.enqueueRequests(std::move(requests));

    return requestIds;
}

std::unordered_map<tle::IdType, tle::BeamTokens> waitForResponses(
    RuntimeOptions const& runtimeOpts, std::vector<tle::IdType> const& requestIds, tle::Executor& executor)
{
    // Map that will be used to store output tokens for requests
    std::unordered_map<tle::IdType, tle::BeamTokens> outputTokens;
    auto numSequences = getNumSequencesPerRequest(runtimeOpts);
    for (auto requestId : requestIds)
    {
        outputTokens[requestId] = tle::BeamTokens(numSequences);
    }

    tle::SizeType32 numFinished{0};
    tle::SizeType32 iter{0};

    // Get the new tokens for each request
    while (numFinished < static_cast<tle::SizeType32>(requestIds.size()) && iter < runtimeOpts.timeoutMs)
    {
        std::chrono::milliseconds waitTime(1);
        // Wait for any response
        auto responses = executor.awaitResponses(waitTime);

        auto insertResponseTokens
            = [&outputTokens](tle::IdType requestId, tle::SizeType32 seqIdx, tle::VecTokens const& respTokens)
        {
            TLLM_LOG_INFO("Got %d tokens for seqIdx %d for requestId %d", respTokens.size(), seqIdx, requestId);

            // Store the output tokens for that request id
            auto& outTokens = outputTokens.at(requestId).at(seqIdx);
            outTokens.insert(outTokens.end(), std::make_move_iterator(respTokens.begin()),
                std::make_move_iterator(respTokens.end()));
        };

        // Loop over the responses
        for (auto const& response : responses)
        {
            auto requestId = response.getRequestId();
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                if (runtimeOpts.beamWidth > 1)
                {
                    for (tle::SizeType32 beam = 0; beam < numSequences; ++beam)
                    {
                        insertResponseTokens(requestId, beam, result.outputTokenIds.at(beam));
                    }
                }
                else
                {
                    insertResponseTokens(requestId, result.sequenceIndex, result.outputTokenIds.at(0));
                }
                if (result.isFinal)
                {
                    TLLM_LOG_INFO("Request id %lu is completed.", requestId);
                }
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                if (response.getErrorMsg() != err)
                {
                    TLLM_THROW("Request id %lu encountered error: %s", requestId, response.getErrorMsg().c_str());
                }
            }
        }
        ++iter;
    }
    if (iter == runtimeOpts.timeoutMs)
    {
        TLLM_THROW("Timeout exceeded.");
    }

    return outputTokens;
}

std::vector<tle::VecTokens> readInputTokens(std::string const& path)
{
    std::vector<tle::VecTokens> data;
    std::ifstream file(path);

    if (!file.is_open())
    {
        auto const err = std::string{"Failed to open file: "} + path;
        TLLM_LOG_ERROR(err);
        TLLM_THROW(err);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<tle::TokenIdType> row;
        std::stringstream ss(line);
        std::string token;

        while (std::getline(ss, token, ','))
        {
            try
            {
                row.push_back(std::stoi(token));
            }
            catch (std::invalid_argument const& e)
            {
                TLLM_LOG_ERROR("Invalid argument: %s", e.what());
            }
            catch (std::out_of_range const& e)
            {
                TLLM_LOG_ERROR("Out of range: %s", e.what());
            }
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

void writeOutputTokens(std::string const& path, std::vector<tle::IdType>& requestIds,
    std::unordered_map<tle::IdType, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth)
{
    std::ofstream file(path);

    if (!file.is_open())
    {
        TLLM_LOG_ERROR("Failed to open file %s", path.c_str());
        return;
    }

    for (auto requestId : requestIds)
    {
        auto const& outTokens = outputTokens.at(requestId);
        for (tle::SizeType32 beam = 0; beam < beamWidth; ++beam)
        {
            auto const& beamTokens = outTokens.at(beam);
            for (size_t i = 0; i < beamTokens.size(); ++i)
            {
                file << beamTokens[i];
                if (i < beamTokens.size() - 1)
                {
                    file << ", ";
                }
            }
            file << "\n";
        }
    }

    file.close();
}

tle::SizeType32 getNumSequencesPerRequest(RuntimeOptions const& runtimeOpts)
{
    auto numReturnSequences = runtimeOpts.numReturnSequences.value_or(runtimeOpts.beamWidth);
    return runtimeOpts.beamWidth > 1 ? std::min(numReturnSequences, runtimeOpts.beamWidth) : numReturnSequences;
}
