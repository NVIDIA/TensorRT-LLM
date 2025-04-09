/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::runtime
{

class PromptLookupBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorMap = StringPtrMap<ITensor>;
    using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
    using RequestVector = std::vector<std::shared_ptr<LlmRequest>>;
    using RequestIdType = LlmRequest::RequestIdType;

    PromptLookupBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, BufferManager const& manager,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
        TllmRuntime const& runtime);

    void updateDraftTokens(std::shared_ptr<LlmRequest> request);

    void removePool(RequestIdType requestID);

    void printPool() const; // For debug

    // PLD pool (mPool) maps from request ID to the Prompt-Map
    // Prompt-Map maps from a list of tokens (pattern to match) to a list of candidate-draft-token-list
    using PromptMap = std::map<std::vector<TokenIdType>, std::vector<std::vector<TokenIdType>>>;
    std::map<LlmRequest::RequestIdType, PromptMap> mPool;

    // mStartIndices notes the index of the prompt to update the pool in the next step
    std::map<LlmRequest::RequestIdType, SizeType32> mStartIndices;
};

} // namespace tensorrt_llm::runtime
