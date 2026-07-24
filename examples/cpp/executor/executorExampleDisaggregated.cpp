/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <cxxopts.hpp>

namespace tle = tensorrt_llm::executor;

namespace fs = std::filesystem;

struct RuntimeOptions
{
    std::string trtContextEnginePath;
    std::string trtGenerationEnginePath;
    std::string inputTokensCsvFile;
    std::string outputTokensCsvFile;

    bool streaming;
    bool excludeInputFromOutput;
    int contextRankSize;
    int generationRankSize;
    tle::SizeType32 maxNewTokens;
    tle::SizeType32 beamWidth;
    std::optional<tle::SizeType32> numReturnSequences;
    tle::SizeType32 timeoutMs;
};

RuntimeOptions parseArgs(int argc, char* argv[]);

// Function that enqueues requests into context executor and generation executor
std::unordered_map<tle::IdType, tle::IdType> enqueueRequests(
    RuntimeOptions const& runtimeOpts, tle::Executor& contextExecutor, tle::Executor& generationExecutor);

// Function that waits for gen responses and stores output tokens
std::unordered_map<tle::IdType, tle::BeamTokens> waitForGenResponses(RuntimeOptions const& runtimeOpts,
    std::unordered_map<tle::IdType, tle::IdType> const& genRequestIdToContextRequestId,
    tle::Executor& generationExecutor);

// Utility function to read input tokens from csv file
std::vector<tle::VecTokens> readInputTokens(std::string const& path);

// Utility function to write output tokens from csv file
void writeOutputTokens(std::string const& path,
    std::unordered_map<tle::IdType, tle::IdType>& genRequestIdToContextRequestId,
    std::unordered_map<tle::IdType, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth);

int main(int argc, char* argv[])
{

    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    auto runtimeOpts = parseArgs(argc, argv);
    TLLM_CHECK_WITH_INFO(runtimeOpts.beamWidth == 1, "Only support beamWidth =1");
    TLLM_CHECK_WITH_INFO(
        runtimeOpts.numReturnSequences.has_value() == false || runtimeOpts.numReturnSequences.value() == 1,
        "Only support numReturnSequences =1");
    // Create the executor for this engine
    auto contextExecutorConfig = tle::ExecutorConfig(runtimeOpts.beamWidth);
    auto generationExecutorConfig = tle::ExecutorConfig(runtimeOpts.beamWidth);
    bool isOrchestrator = (tensorrt_llm::mpi::MpiComm::world().getRank() == 0);
    auto orchestratorConfig = tle::OrchestratorConfig(isOrchestrator, "", nullptr, false);
    int contextRankSize = runtimeOpts.contextRankSize;
    int generationRankSize = runtimeOpts.generationRankSize;
    TLLM_CHECK_WITH_INFO(tensorrt_llm::mpi::MpiComm::world().getSize() >= contextRankSize + generationRankSize + 1,
        " MPI should launch at least [contextRankSize+generationRankSize+1]: %d processes",
        contextRankSize + generationRankSize + 1);
    int deviceCount = -1;
    TLLM_CHECK(cudaGetDeviceCount(&deviceCount) == cudaSuccess);

    std::vector<int32_t> contextRankIds(contextRankSize);
    std::vector<int32_t> contextDeviceIds(contextRankSize);
    std::vector<int32_t> generationRankIds(generationRankSize);
    std::vector<int32_t> generationDeviceIds(generationRankSize);
    for (int i = 0; i < contextRankSize; i++)
    {
        contextRankIds[i] = i + 1;
        contextDeviceIds[i] = i % deviceCount;
        TLLM_LOG_INFO("context Rank %d on device %d", contextRankIds[i], contextDeviceIds[i]);
    }
    tle::ParallelConfig contextParallelConfig{tensorrt_llm::executor::CommunicationType::kMPI,
        tensorrt_llm::executor::CommunicationMode::kORCHESTRATOR, contextDeviceIds, contextRankIds, orchestratorConfig};

    for (int i = 0; i < generationRankSize; i++)
    {
        generationRankIds[i] = i + 1 + contextRankSize;
        generationDeviceIds[i] = (i + contextRankSize) % deviceCount;
        TLLM_LOG_INFO("generation Rank %d on device %d", generationRankIds[i], generationDeviceIds[i]);
    }
    tle::ParallelConfig generationParallelConfig{tensorrt_llm::executor::CommunicationType::kMPI,
        tensorrt_llm::executor::CommunicationMode::kORCHESTRATOR, generationDeviceIds, generationRankIds,
        orchestratorConfig};

    contextExecutorConfig.setParallelConfig(contextParallelConfig);
    generationExecutorConfig.setParallelConfig(generationParallelConfig);

    auto contextExecutor
        = tle::Executor(runtimeOpts.trtContextEnginePath, tle::ModelType::kDECODER_ONLY, contextExecutorConfig);
    auto generationExecutor
        = tle::Executor(runtimeOpts.trtGenerationEnginePath, tle::ModelType::kDECODER_ONLY, generationExecutorConfig);
    tensorrt_llm::mpi::MpiComm::world().barrier();

    if (tensorrt_llm::mpi::MpiComm::world().getRank() == 0)
    {

        TLLM_CHECK_WITH_INFO(contextExecutor.canEnqueueRequests(), "contextExecutor can't enqueue requests");
        TLLM_CHECK_WITH_INFO(generationExecutor.canEnqueueRequests(), "generationExecutor can't enqueue requests");
        auto genRequestIdsToContextRequestIds = enqueueRequests(runtimeOpts, contextExecutor, generationExecutor);
        auto outputTokens = waitForGenResponses(runtimeOpts, genRequestIdsToContextRequestIds, generationExecutor);
        TLLM_LOG_INFO("Writing output tokens to %s", runtimeOpts.outputTokensCsvFile.c_str());
        writeOutputTokens(
            runtimeOpts.outputTokensCsvFile, genRequestIdsToContextRequestIds, outputTokens, runtimeOpts.beamWidth);
    }
    tensorrt_llm::mpi::MpiComm::world().barrier();

    TLLM_LOG_INFO("Exiting.");
    return 0;
}

RuntimeOptions parseArgs(int argc, char* argv[])
{

    RuntimeOptions runtimeOpts;

    cxxopts::Options options(argv[0], "Example that demonstrates how to use the Executor Disaggregated API");
    options.add_options()("h,help", "Print usage");
    options.add_options()(
        "context_engine_dir", "Directory that store the context engine.", cxxopts::value<std::string>());
    options.add_options()(
        "generation_engine_dir", "Directory that store the  generation engine.", cxxopts::value<std::string>());
    options.add_options()(
        "context_rank_size", "The number of ranks for the context engine", cxxopts::value<int>()->default_value("1"));
    options.add_options()("generation_rank_size", "The number of ranks for the generation engine",
        cxxopts::value<int>()->default_value("1"));
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

    auto parsedOptions = options.parse(argc, argv);

    // Argument: help
    if (parsedOptions.count("help"))
    {
        TLLM_LOG_ERROR(options.help());
        exit(0);
    }

    runtimeOpts.trtContextEnginePath = parsedOptions["context_engine_dir"].as<std::string>();
    if (!fs::exists(runtimeOpts.trtContextEnginePath) || !fs::is_directory(runtimeOpts.trtContextEnginePath))
    {
        TLLM_LOG_ERROR("Context engine directory doesn't exist.");
        exit(1);
    }

    runtimeOpts.trtGenerationEnginePath = parsedOptions["generation_engine_dir"].as<std::string>();
    if (!fs::exists(runtimeOpts.trtGenerationEnginePath) || !fs::is_directory(runtimeOpts.trtGenerationEnginePath))
    {
        TLLM_LOG_ERROR("Generation engine directory doesn't exist.");
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
    runtimeOpts.contextRankSize = parsedOptions["context_rank_size"].as<int>();
    runtimeOpts.generationRankSize = parsedOptions["generation_rank_size"].as<int>();
    if (parsedOptions.count("num_return_sequences") > 0)
    {
        runtimeOpts.numReturnSequences = parsedOptions["num_return_sequences"].as<std::optional<int>>();
    }
    runtimeOpts.timeoutMs = parsedOptions["timeout_ms"].as<int>();
    runtimeOpts.outputTokensCsvFile = parsedOptions["output_tokens_csv_file"].as<std::string>();

    return runtimeOpts;
}

std::unordered_map<tle::IdType, tle::IdType> enqueueRequests(
    RuntimeOptions const& runtimeOpts, tle::Executor& contextExecutor, tle::Executor& generationExecutor)
{

    tle::OutputConfig outputConfig;
    outputConfig.excludeInputFromOutput = runtimeOpts.excludeInputFromOutput;
    tle::SamplingConfig samplingConfig(runtimeOpts.beamWidth);
    std::unordered_map<tle::IdType, tle::IdType> genRequestIdToContextRequestId;
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
        requests.back().setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY);
    }

    auto contextRequestIds = contextExecutor.enqueueRequests(requests);

    for (size_t i = 0; i < requests.size(); i++)
    {

        TLLM_LOG_INFO("waiting response for Context request id: %lu,", contextRequestIds[i]);
        auto response = contextExecutor.awaitResponses(contextRequestIds[i]);
        TLLM_LOG_INFO("response received for Context request id: %lu", contextRequestIds[i]);
        TLLM_CHECK(response.size() == 1);
        TLLM_CHECK(response.back().getResult().contextPhaseParams.has_value());
        requests.at(i).setContextPhaseParams(response.back().getResult().contextPhaseParams.value());
        requests.at(i).setRequestType(tensorrt_llm::executor::RequestType::REQUEST_TYPE_GENERATION_ONLY);
        auto genRequestId = generationExecutor.enqueueRequest(requests.at(i));
        genRequestIdToContextRequestId[genRequestId] = contextRequestIds[i];

        TLLM_LOG_INFO("enqueuing generation request for Context request id: %lu, generation request id: %lu",
            contextRequestIds[i], genRequestId);
    }

    return genRequestIdToContextRequestId;
}

std::unordered_map<tle::IdType, tle::BeamTokens> waitForGenResponses(RuntimeOptions const& runtimeOpts,
    std::unordered_map<tle::IdType, tle::IdType> const& genRequestIdToContextRequestId,
    tle::Executor& generationExecutor)
{

    // Map that will be used to store output tokens for requests
    std::unordered_map<tle::IdType, tle::BeamTokens> outputTokens;
    std::vector<tle::IdType> contextRequestIds{};
    std::vector<tle::IdType> genRequestIds{};
    for (auto const& [key, value] : genRequestIdToContextRequestId)
    {
        genRequestIds.push_back(key);
        contextRequestIds.push_back(value);
    }
    for (auto contextRequestId : contextRequestIds)
    {
        outputTokens[contextRequestId] = tle::BeamTokens(runtimeOpts.beamWidth);
    }

    tle::SizeType32 numFinished{0};
    tle::SizeType32 iter{0};

    // Get the new tokens for each request
    while (numFinished < static_cast<tle::SizeType32>(genRequestIds.size()) && iter < runtimeOpts.timeoutMs)
    {
        std::chrono::milliseconds waitTime(1);
        // Wait for any response
        auto responses = generationExecutor.awaitResponses(waitTime);

        auto insertResponseTokens = [&outputTokens, &genRequestIdToContextRequestId](tle::IdType genRequestId,
                                        tle::SizeType32 seqIdx, tle::VecTokens const& respTokens)
        {
            TLLM_LOG_INFO("Got %d tokens for seqIdx %d for genRequestId %d,contextRequestId %d", respTokens.size(),
                seqIdx, genRequestId, genRequestIdToContextRequestId.at(genRequestId));

            // Store the output tokens for that request id
            auto& outTokens = outputTokens.at(genRequestIdToContextRequestId.at(genRequestId)).at(seqIdx);
            outTokens.insert(outTokens.end(), std::make_move_iterator(respTokens.begin()),
                std::make_move_iterator(respTokens.end()));
        };

        // Loop over the responses
        for (auto const& response : responses)
        {
            auto genRequestId = response.getRequestId();
            if (!response.hasError())
            {
                auto result = response.getResult();
                numFinished += result.isFinal;
                if (runtimeOpts.beamWidth > 1)
                {
                    for (tle::SizeType32 beam = 0; beam < runtimeOpts.beamWidth; ++beam)
                    {
                        insertResponseTokens(genRequestId, beam, result.outputTokenIds.at(beam));
                    }
                }
                else
                {
                    insertResponseTokens(genRequestId, result.sequenceIndex, result.outputTokenIds.at(0));
                }
                if (result.isFinal)
                {
                    TLLM_LOG_INFO("genRequest id %lu ,contextRequestId %lu is completed.", genRequestId,
                        genRequestIdToContextRequestId.at(genRequestId));
                }
            }
            else
            {
                // Allow response with error only if awaitResponse processed a terminated request id
                std::string err = "genReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                if (response.getErrorMsg() != err)
                {
                    TLLM_THROW("GenRequest id %lu encountered error: %s", genRequestId, response.getErrorMsg().c_str());
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

void writeOutputTokens(std::string const& path,
    std::unordered_map<tle::IdType, tle::IdType>& genRequestIdToContextRequestId,
    std::unordered_map<tle::IdType, tle::BeamTokens> const& outputTokens, tle::SizeType32 beamWidth)
{
    std::ofstream file(path);

    if (!file.is_open())
    {
        TLLM_LOG_ERROR("Failed to open file %s", path.c_str());
        return;
    }
    std::vector<tle::IdType> requestIds;
    for (auto const& [key, value] : genRequestIdToContextRequestId)
    {
        requestIds.push_back(value);
    }
    std::sort(requestIds.begin(), requestIds.end());

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
