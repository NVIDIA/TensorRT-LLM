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

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime
{

class GenerationConfig
{
public:
    GenerationConfig() = default;

    explicit GenerationConfig(SizeType batchSize, SizeType beamWidth, SizeType maxInputLength,
        SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSeqLength,
        SizeType inputLengthSum = SizeType(0))
        : batchSize{batchSize}
        , beamWidth{beamWidth}
        , maxInputLength{maxInputLength}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , maxSeqLength{maxSeqLength}
        , inputLengthSum{inputLengthSum}
    {
    }

    SizeType batchSize{};
    SizeType beamWidth{};
    SizeType maxInputLength{};
    SizeType maxAttentionWindow{};
    SizeType sinkTokenLength{};
    SizeType maxSeqLength{};
    SizeType inputLengthSum{}; // Initialized only if inputPacked is set to true in fromInput.

    static GenerationConfig fromInput(ITensor const& inputIds, ITensor& inputLengths, bool inputPacked,
        SizeType beamWidth, SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSequenceLength);
};

} // namespace tensorrt_llm::runtime
