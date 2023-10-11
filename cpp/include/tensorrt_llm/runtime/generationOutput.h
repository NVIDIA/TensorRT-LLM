/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

class GenerationOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    using Callback = std::function<void(TensorPtr const& ids, SizeType step, bool finished)>;

    explicit GenerationOutput(TensorPtr ids)
        : ids{std::move(ids)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
    }

    // mandatory parameters
    TensorPtr ids; // [batchSize, beamWidth, maxInputLength + maxNewTokens]

    // optional parameters
    TensorPtr logProbs;      // [request_output_length, batch_size * beam_width], must be float*, on gpu
    TensorPtr contextLogits; // [batch_size, max_input_length, vocab_size_padded]

    // callbacks
    Callback onTokenGenerated;
};

} // namespace tensorrt_llm::runtime
