/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <functional>
#include <utility>

namespace tensorrt_llm::runtime
{

template <typename TTensor>
class GenericGenerationOutput
{
public:
    using TensorPtr = TTensor;
    using Callback = std::function<void(TensorPtr const& ids, SizeType step, bool finished)>;

    explicit GenericGenerationOutput(TensorPtr ids, TensorPtr lengths)
        : ids{std::move(ids)}
        , lengths{std::move(lengths)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    // mandatory parameters
    TensorPtr ids;     // [batchSize, beamWidth, maxInputLength + maxNewTokens]
    TensorPtr lengths; // [batchSize, beamWidth]

    // optional parameters
    TensorPtr cumLogProbs;      // [batchSize, beamWidth], must be float*, on gpu
    TensorPtr logProbs;         // [batchSize, beamWidth, maxInputLength + maxNewTokens], must be float*, on gpu
    TensorPtr contextLogits;    // [batch_size, max_input_length, vocab_size_padded], if packed, the shape will be
                                // [packed_size, vocab_size_padded]
    TensorPtr generationLogits; // [batch_size, beam_width, max_output_length, vocab_size_padded]

    // callbacks
    Callback onTokenGenerated;
};

class GenerationOutput : public GenericGenerationOutput<ITensor::SharedPtr>
{
public:
    using Base = GenericGenerationOutput<ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationOutput(TensorPtr ids, TensorPtr lengths)
        : GenericGenerationOutput(std::move(ids), std::move(lengths))
    {
    }
};

} // namespace tensorrt_llm::runtime
