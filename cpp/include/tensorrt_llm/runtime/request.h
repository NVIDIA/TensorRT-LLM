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
    using ConstTensorPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(ConstTensorPtr ids, SizeType32 inputLen, std::optional<SizeType32> maxNewTokens = std::nullopt,
        std::optional<SizeType32> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , generatedTokensPerEngineStep(1)
    {
    }

    // mandatory parameters
    ConstTensorPtr ids;  // [inputSeqLen], the input sequence of token ids, on gpu
    SizeType32 inputLen; // the input length without draft tokens

    // optional parameters
    std::optional<SizeType32> maxNewTokens; // maximum number of tokens to generate for this request
    std::optional<SizeType32> endId;        // end token id
    BufferPtr draftTokens;   // [generatedTokensPerStep - 1], on gpu, draft tokens from speculative decoding
    std::optional<TensorPtr>
        draftLogits;         // [generatedTokensPerStep - 1, vocabSize], on gpu, draft tokens from speculative decoding
    TensorPtr embeddingBias; // [vocabSizePadded], on gpu
    TensorPtr badWordsList;  // [2, badWordsLength], on gpu
    TensorPtr stopWordsList; // [2, stopWordsLength], on gpu

    SizeType32 generatedTokensPerEngineStep;
    TensorPtr medusaPaths;   // [maxDraftTokens + 1, maxAcceptedDraftTokensPerStep + 1], on gpu
    TensorPtr medusaTreeIds; // [maxDraftTokens + 1], on gpu
    std::optional<executor::LookaheadDecodingConfig> lookaheadRuntimeConfig;
};

} // namespace tensorrt_llm::runtime::decoder_batch
