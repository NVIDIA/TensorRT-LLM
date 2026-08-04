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
#include <string>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    if (argc != 2)
    {
        TLLM_LOG_ERROR("Usage: %s <dir_with_engine_files>", argv[0]);
        return 1;
    }

    // Create the executor for this engine
    tle::SizeType32 beamWidth = 1;
    auto executorConfig = tle::ExecutorConfig(beamWidth);
    auto trtEnginePath = argv[1];
    auto executor = tle::Executor(trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    // Create the request
    tle::SizeType32 maxNewTokens = 5;
    tle::VecTokens inputTokens{1, 2, 3, 4};
    auto request = tle::Request(inputTokens, maxNewTokens);

    // Enqueue the request
    auto requestId = executor.enqueueRequest(request);

    // Wait for the response
    auto responses = executor.awaitResponses(requestId);

    // Get outputTokens
    auto outputTokens = responses.at(0).getResult().outputTokenIds.at(beamWidth - 1);

    TLLM_LOG_INFO("Output tokens: %s", tlc::vec2str(outputTokens).c_str());

    return 0;
}
