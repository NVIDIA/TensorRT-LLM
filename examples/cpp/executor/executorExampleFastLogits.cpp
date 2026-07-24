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

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

namespace fs = std::filesystem;

struct RuntimeOptions
{
    std::string trtDraftEnginePath;
    std::string trtEnginePath;
    bool fastLogits;
    tle::SizeType32 numDraftTokens;
};

// Utility function to parse input arguments
RuntimeOptions parseArgs(int argc, char* argv[]);

// Runs a draft request
tle::Result executeDraftRequest(tle::Executor& executor, RuntimeOptions const& runtimeOpts);

// Runs a target request
tle::Result executeTargetRequest(
    tle::Executor& executor, tle::Result const& draftResult, RuntimeOptions const& runtimeOpts);

// Main
int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    auto runtimeOpts = parseArgs(argc, argv);

    // Create the executor for this engine
    auto executorConfig = tle::ExecutorConfig();

    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
    int const myRank = tensorrt_llm::mpi::MpiComm::world().getRank();
    bool const isOrchestrator = (myRank == 0);

    auto kvCacheConfig = tle::KvCacheConfig(true /* enableBlockReuse */);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    auto orchestratorConfig
        = tle::OrchestratorConfig(isOrchestrator, "" /* workerExecutablePath */, nullptr, false /* spawnPrcesses */);
    auto parallelConfig = tle::ParallelConfig(tle::CommunicationType::kMPI, tle::CommunicationMode::kORCHESTRATOR,
        std::nullopt, std::nullopt, orchestratorConfig);
    executorConfig.setParallelConfig(parallelConfig);

    auto specDecConfig = tle::SpeculativeDecodingConfig(runtimeOpts.fastLogits);
    executorConfig.setSpecDecConfig(specDecConfig);

    std::unique_ptr<tle::Executor> draftExecutor;
    std::unique_ptr<tle::Executor> targetExecutor;

    if (isOrchestrator)
    {
        auto executorConfigDraft = executorConfig;
        parallelConfig.setParticipantIds({1});
        executorConfigDraft.setParallelConfig(parallelConfig);

        draftExecutor = std::make_unique<tle::Executor>(
            runtimeOpts.trtDraftEnginePath, tle::ModelType::kDECODER_ONLY, executorConfigDraft);

        parallelConfig.setParticipantIds({2});
        executorConfig.setParallelConfig(parallelConfig);

        targetExecutor
            = std::make_unique<tle::Executor>(runtimeOpts.trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);
    }
    else if (myRank == 1) // draft model process
    {
        parallelConfig.setParticipantIds({1});
        parallelConfig.setDeviceIds({0});
        executorConfig.setParallelConfig(parallelConfig);
        draftExecutor = std::make_unique<tle::Executor>(
            runtimeOpts.trtDraftEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);
    }
    else if (myRank == 2) // target model process
    {
        parallelConfig.setParticipantIds({2});
        parallelConfig.setDeviceIds({1});
        executorConfig.setParallelConfig(parallelConfig);
        targetExecutor
            = std::make_unique<tle::Executor>(runtimeOpts.trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);
        ;
    }

    // Only orchestrator rank (rank 0) will enter
    if (isOrchestrator)
    {
        auto draftResult = executeDraftRequest(*draftExecutor, runtimeOpts);

        executeTargetRequest(*targetExecutor, draftResult, runtimeOpts);
    }
    TLLM_LOG_INFO("Exiting.");
    return 0;
}

RuntimeOptions parseArgs(int argc, char* argv[])
{
    RuntimeOptions runtimeOpts;

    cxxopts::Options options(argv[0], "Example that demonstrates how to use the Executor API");
    options.add_options()("h,help", "Print usage");
    options.add_options()("engine_dir", "Directory that store the engine.", cxxopts::value<std::string>());
    options.add_options()("draft_engine_dir", "Directory that store the draft engine.", cxxopts::value<std::string>());
    options.add_options()(
        "fast_logits", "Use speculative decoding fast logits feature", cxxopts::value<bool>()->default_value("true"));
    options.add_options()(
        "num_draft_tokens", "Number of draft tokens to use", cxxopts::value<int>()->default_value("5"));

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

    // Argument: Draft engine directory
    if (!parsedOptions.count("draft_engine_dir"))
    {
        TLLM_LOG_ERROR(options.help());
        TLLM_LOG_ERROR("Please specify draft engine directory.");
        exit(1);
    }
    runtimeOpts.trtDraftEnginePath = parsedOptions["draft_engine_dir"].as<std::string>();
    if (!fs::exists(runtimeOpts.trtDraftEnginePath) || !fs::is_directory(runtimeOpts.trtDraftEnginePath))
    {
        TLLM_LOG_ERROR("Draft engine directory doesn't exist.");
        exit(1);
    }

    runtimeOpts.fastLogits = parsedOptions["fast_logits"].as<bool>();
    runtimeOpts.numDraftTokens = parsedOptions["num_draft_tokens"].as<int>();

    return runtimeOpts;
}

tle::Result executeDraftRequest(tle::Executor& executor, RuntimeOptions const& runtimeOpts)
{
    tle::OutputConfig outputConfig;
    outputConfig.returnGenerationLogits = true;

    // Create the request
    tle::SizeType32 maxNewTokens = runtimeOpts.numDraftTokens;
    tle::VecTokens inputTokens{1, 2, 3, 4};

    tle::Request request{std::move(inputTokens), maxNewTokens};
    request.setOutputConfig(outputConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    if (responses.at(0).hasError())
    {
        TLLM_LOG_ERROR(responses.at(0).getErrorMsg());
        exit(1);
    }

    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(0);

    TLLM_LOG_INFO("[DRAFT] Output tokens: %s", tlc::vec2str(outputTokens).c_str());

    return responses.at(0).getResult();
}

tle::Result executeTargetRequest(
    tle::Executor& executor, tle::Result const& draftResult, RuntimeOptions const& runtimeOpts)
{
    // Create the request
    tle::SizeType32 maxNewTokens = runtimeOpts.numDraftTokens + 1;
    tle::VecTokens inputTokens{1, 2, 3, 4};

    tle::Request request{std::move(inputTokens), maxNewTokens};

    tle::VecTokens const& outputTokenIds = draftResult.outputTokenIds.at(0);
    tle::VecTokens draftTokens(outputTokenIds.end() - runtimeOpts.numDraftTokens, outputTokenIds.end());
    TLLM_LOG_INFO("[DRAFT] Draft tokens: %s", tlc::vec2str(draftTokens).c_str());

    tle::Tensor logitsTensor;

    if (runtimeOpts.fastLogits)
    {
        auto const& logitsInfo = draftResult.specDecFastLogitsInfo.value();
        logitsTensor = logitsInfo.toTensor();
    }
    else
    {
        auto generationLogits = draftResult.generationLogits.value();
        auto logitsShape = generationLogits.getShape();
        TLLM_CHECK(logitsShape[0] == 1);
        logitsTensor = tle::Tensor::cpu(generationLogits.getDataType(), {logitsShape[1], logitsShape[2]});
        std::memcpy(logitsTensor.getData(), generationLogits.getData(), generationLogits.getSizeInBytes());
    }

    tle::ExternalDraftTokensConfig draftTokensConfig(
        std::move(draftTokens), logitsTensor, std::nullopt /* acceptance threshold */, runtimeOpts.fastLogits);
    request.setExternalDraftTokensConfig(draftTokensConfig);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    if (responses.at(0).hasError())
    {
        TLLM_LOG_ERROR(responses.at(0).getErrorMsg());
        exit(1);
    }

    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(0);

    TLLM_LOG_INFO("[TARGET] Output tokens: %s", tlc::vec2str(outputTokens).c_str());

    return responses.at(0).getResult();
}
