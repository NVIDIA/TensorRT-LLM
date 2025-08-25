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

#pragma once

#include "common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
}

namespace tensorrt_llm::batch_manager
{

class LogitsPostProcessor : Algorithm
{
public:
    using LogitsPostProcessorBatched = std::function<void(std::vector<batch_manager::LlmRequest::RequestIdType> const&,
        std::vector<batch_manager::LlmRequest::TensorPtr>&,
        std::vector<std::reference_wrapper<batch_manager::LlmRequest::BeamTokens const>> const&,
        runtime::BufferManager::CudaStreamPtr const&,
        std::vector<std::optional<batch_manager::LlmRequest::RequestIdType>> const&)>;

    constexpr static auto name{"LogitsPostProcessor"};

    LogitsPostProcessor() = default;

    bool operator()(RequestVector const& contextRequests, RequestVector const& generationRequests,
        bool replicateLogitsPostProcessor, std::vector<batch_manager::LlmRequest::TensorPtr>& seqSlotLogits,
        runtime::WorldConfig const& worldConfig, runtime::TllmRuntime& runtime,
        std::optional<LogitsPostProcessorBatched> const& logitsPostProcessorBatched = std::nullopt) const;
};

} // namespace tensorrt_llm::batch_manager
