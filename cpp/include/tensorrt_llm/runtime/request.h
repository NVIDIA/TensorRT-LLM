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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include <optional>

namespace tensorrt_llm::runtime::decoder_batch
{

class Request
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(TensorConstPtr ids, SizeType32 inputLen, std::optional<SizeType32> maxNewTokens = std::nullopt,
        std::optional<SizeType32> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , generatedTokensPerEngineStep(1)
    {
    }

    //! Mandatory parameters
    TensorConstPtr ids;  // The input sequence of token ids, [inputSeqLen], on gpu
    SizeType32 inputLen; // Input length without draft tokens, increasing with generation steps

    // optional parameters
    std::optional<SizeType32> maxNewTokens;  // maximum number of tokens to generate for this request
    std::optional<SizeType32> endId;         // end token id
    SizeType32 generatedTokensPerEngineStep; //
    TensorPtr embeddingBias;                 // [vocabSizePadded], on gpu
    TensorPtr badWordsList;                  // [2, badWordsLength] on gpu
    TensorPtr stopWordsList;                 // [2, stopWordsLength] on gpu

    //! Optional parameters for speculative decoding
    BufferPtr draftTokens;                // [generatedTokensPerEngineStep - 1] on gpu
    std::optional<TensorPtr> draftLogits; // [generatedTokensPerEngineStep - 1, vocabSize] on gpu
    TensorPtr medusaPaths;                // [maxDecodingTokens, maxPathLen], on gpu
    TensorPtr medusaTreeIds;              // [maxDecodingTokens], on gpu
    nvinfer1::DataType dtype;             // Request data type, only used by explicit draft tokens.
    std::optional<executor::LookaheadDecodingConfig> lookaheadRuntimeConfig;
    std::optional<executor::EagleConfig> eagleConfig;
};

} // namespace tensorrt_llm::runtime::decoder_batch
