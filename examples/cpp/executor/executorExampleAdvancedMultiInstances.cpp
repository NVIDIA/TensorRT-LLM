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
#include <map>
#include <stdexcept>
#include <string>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
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
    tle::SizeType32 timeoutMs;

    bool useOrchestratorMode;
    std::string workerExecutablePath;
    bool spawnProcesses;
};

// Utility function to parse input arguments
RuntimeOptions parseArgs(int argc, char* argv[]);

// Function that enqueues requests
std::vector<std::pair<int32_t, tle::IdType>> enqueueRequests(
    RuntimeOptions const& runtimeOpts, std::deque<tle::Executor>& executors);

// Function that waits for responses and stores output tokens
std::map<std::pair<int32_t, tle::IdType>, tle::BeamTokens> waitForResponses(RuntimeOptions const& runtimeOpts,
    std::vector<std::pair<int32_t, tle::IdType>> const& instanceRequestIds, std::deque<tle::Executor>& executors);

// Utility function to read input tokens from csv file
std::vector<tle::VecTokens> readInputTokens(std::string const& path);

// Utility function to write output tokens from csv file
void writeOutputTokens(std::string const& path, std::vector<std::pair<int32_t, tle::IdType>>& requestIds,
    std::map<std::pair<int32_t, tle::IdType>, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth);

// Main
int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    auto runtimeOpts = parseArgs(argc, argv);

    // Create the executor for this engine
    auto executorConfig = tle::ExecutorConfig(runtimeOpts.beamWidth);

    tle::KvCacheConfig kvCacheConfig{false, 10000};
    executorConfig.setKvCacheConfig(kvCacheConfig);

    bool isOrchestrator = true;
    if (!runtimeOpts.spawnProcesses)
    {
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
        int myRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        isOrchestrator = (myRank == 0);
    }

    auto orchestratorConfig = tle::OrchestratorConfig(
        isOrchestrator, runtimeOpts.workerExecutablePath, nullptr, runtimeOpts.spawnProcesses);
    auto parallelConfig = tle::ParallelConfig(tle::CommunicationType::kMPI, tle::CommunicationMode::kORCHESTRATOR,
        std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    int numInstances = 3;
    if (!runtimeOpts.spawnProcesses)
    {
        // Keep one rank for orchestrator
        numInstances = tensorrt_llm::mpi::MpiComm::world().getSize() - 1;
    }
    std::deque<tle::Executor> executors;
    for (int instanceId = 0; instanceId < numInstances; ++instanceId)
    {
        auto executorConfigTmp = executorConfig;
        // Set the rank id participating in each model instance
        if (!runtimeOpts.spawnProcesses)
        {
            parallelConfig.setParticipantIds({instanceId + 1});
        }
        executorConfigTmp.setParallelConfig(parallelConfig);
        executors.emplace_back(runtimeOpts.trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfigTmp);
    }

    // Only orchestrator rank (rank 0) will enter
    if (isOrchestrator)
    {
        // Create the requests
        auto instanceRequestIds = enqueueRequests(runtimeOpts, executors);

        // Wait for responses and store output tokens
        auto outputTokens = waitForResponses(runtimeOpts, instanceRequestIds, executors);

        // Write output tokens csv file
        TLLM_LOG_INFO("Writing output tokens to %s", runtimeOpts.outputTokensCsvFile.c_str());
        writeOutputTokens(runtimeOpts.outputTokensCsvFile, instanceRequestIds, outputTokens, runtimeOpts.beamWidth);
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
    options.add_options()("worker_executable_path", "The location of the worker executable.",
        cxxopts::value<std::string>()->default_value(""));
    options.add_options()("spawn_processes",
        "Flag that controls if MPI_Comm_spawn should be used to spawn worker processes, or if they have been launched "
        "with mpi already.",
        cxxopts::value<bool>()->default_value("true"));

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
    runtimeOpts.timeoutMs = parsedOptions["timeout_ms"].as<int>();
    runtimeOpts.outputTokensCsvFile = parsedOptions["output_tokens_csv_file"].as<std::string>();

    runtimeOpts.workerExecutablePath = parsedOptions["worker_executable_path"].as<std::string>();
    runtimeOpts.spawnProcesses = parsedOptions["spawn_processes"].as<bool>();

    return runtimeOpts;
}

std::vector<std::pair<int32_t, tle::IdType>> enqueueRequests(
    RuntimeOptions const& runtimeOpts, std::deque<tle::Executor>& executors)
{
    tle::OutputConfig outputConfig;
    outputConfig.excludeInputFromOutput = runtimeOpts.excludeInputFromOutput;
    tle::SamplingConfig samplingConfig(runtimeOpts.beamWidth);

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
    // Round robin over instances
    std::vector<std::pair<int32_t, tle::IdType>> instanceRequestIds;
    for (size_t req = 0; req < requests.size(); ++req)
    {
        auto instanceId = req % executors.size();
        TLLM_LOG_INFO("Enqueuing request %d for instance %d", req, instanceId);
        auto requestId = executors.at(instanceId).enqueueRequest(requests[req]);
        instanceRequestIds.emplace_back(instanceId, requestId);
    }
    TLLM_LOG_INFO("Enqueued %d requests", instanceRequestIds.size());
    return instanceRequestIds;
}

std::map<std::pair<int32_t, tle::IdType>, tle::BeamTokens> waitForResponses(RuntimeOptions const& runtimeOpts,
    std::vector<std::pair<int32_t, tle::IdType>> const& instanceRequestIds, std::deque<tle::Executor>& executors)
{
    // Map that will be used to store output tokens for requests
    int numRequests = 0;
    std::map<std::pair<int32_t, tle::IdType>, tle::BeamTokens> outputTokens;
    for (auto instanceRequestId : instanceRequestIds)
    {
        outputTokens[instanceRequestId] = tle::BeamTokens(runtimeOpts.beamWidth);
        numRequests++;
    }

    tle::SizeType32 numFinished{0};
    tle::SizeType32 iter{0};

    // Get the new tokens for each request
    while (numFinished < numRequests && iter < runtimeOpts.timeoutMs)
    {
        std::chrono::milliseconds waitTime(1);
        for (size_t instanceId = 0; instanceId < executors.size(); ++instanceId)
        {
            // Wait for any response for given instance
            auto responses = executors.at(instanceId).awaitResponses(waitTime);
            // Loop over the responses
            for (auto const& response : responses)
            {
                auto requestId = response.getRequestId();
                if (!response.hasError())
                {
                    auto result = response.getResult();
                    numFinished += result.isFinal;
                    TLLM_LOG_INFO("Number of finished requests: %d", numFinished);

                    for (tle::SizeType32 beam = 0; beam < runtimeOpts.beamWidth; ++beam)
                    {
                        auto& respTokens = result.outputTokenIds.at(beam);

                        TLLM_LOG_INFO("Got %d tokens for beam %d for requestId %d", respTokens.size(), beam, requestId);

                        // Store the output tokens for that request id
                        auto& outTokens = outputTokens.at(std::make_pair(instanceId, requestId)).at(beam);
                        outTokens.insert(outTokens.end(), std::make_move_iterator(respTokens.begin()),
                            std::make_move_iterator(respTokens.end()));
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

void writeOutputTokens(std::string const& path, std::vector<std::pair<int32_t, tle::IdType>>& instanceRequestIds,
    std::map<std::pair<int32_t, tle::IdType>, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth)
{
    std::ofstream file(path);

    if (!file.is_open())
    {
        TLLM_LOG_ERROR("Failed to open file %s", path.c_str());
        return;
    }

    for (auto instanceRequestId : instanceRequestIds)
    {
        auto const& outTokens = outputTokens.at(instanceRequestId);
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
