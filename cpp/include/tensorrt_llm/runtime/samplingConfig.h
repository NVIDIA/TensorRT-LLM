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

#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{

class SamplingConfig
{
    using FloatType = float;

    template <typename T>
    using OptVec = std::optional<std::vector<T>>;

public:
    explicit SamplingConfig(SizeType beamWidth = 1)
        : beamWidth{beamWidth}
    {
    }

    SizeType beamWidth;

    OptVec<FloatType> temperature;       // [1] or [batch_size] on cpu
    OptVec<SizeType> minLength;          // [1] or [batch_size] on cpu
    OptVec<FloatType> repetitionPenalty; // [1] or [batch_size] on cpu
    OptVec<FloatType> presencePenalty;   // [1] or [batch_size] on cpu
    OptVec<FloatType> frequencyPenalty;  // [1] or [batch_size] on cpu

    // sampling layers
    OptVec<SizeType> topK;         // [1] or [batch_size] on cpu
    OptVec<FloatType> topP;        // [1] or [batch_size] on cpu
    OptVec<uint64_t> randomSeed;   // [1] or [batch_size] on cpu
    OptVec<FloatType> topPDecay;   // [batch_size], must between [0, 1]
    OptVec<FloatType> topPMin;     // [batch_size], must between [0, 1]
    OptVec<SizeType> topPResetIds; // [batch_size]

    // beam search layer
    OptVec<FloatType> beamSearchDiversityRate;
    OptVec<FloatType> lengthPenalty;

    // speculative decoding
    OptVec<FloatType> draftAcceptanceThreshold;

    std::optional<bool> normalizeLogProbs;
};

} // namespace tensorrt_llm::runtime
