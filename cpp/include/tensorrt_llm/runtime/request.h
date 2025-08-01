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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>

namespace tensorrt_llm::runtime::decoder_batch
{

class Request
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    SizeType32 generatedTokensPerEngineStep{1};

    BufferPtr draftTokens;                // [generatedTokensPerEngineStep - 1] on gpu
    std::optional<TensorPtr> draftLogits; // [generatedTokensPerEngineStep - 1, vocabSize] on gpu
    TensorPtr medusaPaths;                // [maxDecodingTokens, maxPathLen], on gpu
    TensorPtr medusaTreeIds;              // [maxDecodingTokens], on gpu
    std::optional<executor::EagleConfig> eagleConfig;
};

} // namespace tensorrt_llm::runtime::decoder_batch
