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
#include <iostream>
#include <stdexcept>
#include <string>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

namespace tle = tensorrt_llm::executor;

int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins and the logger
    auto logger = std::make_shared<tensorrt_llm::runtime::TllmLogger>();
    initTrtLlmPlugins(logger.get());

    if (argc != 2)
    {
        logger->log(nvinfer1::ILogger::Severity::kERROR, "Usage: ./tensorrt_llm_executor <dir_with_engine_files>");
        return 1;
    }

    // Create the executor for this engine
    tle::SizeType beamWidth = 1;
    auto executorConfig = tle::ExecutorConfig(beamWidth);
    auto trtEnginePath = argv[1];
    auto executor = tle::Executor(trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    tle::SizeType maxNewTokens = 5;
    tle::VecTokens inputTokens{1, 2, 3, 4};
    auto request = tle::Request(inputTokens, maxNewTokens);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(std::move(request));

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    // Get outputTokens
    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(beamWidth - 1);

    logger->log(nvinfer1::ILogger::Severity::kINFO, "Output tokens: ");
    for (auto& outputToken : outputTokens)
    {
        logger->log(nvinfer1::ILogger::Severity::kINFO, std::to_string(outputToken).c_str());
    }

    return 0;
}
