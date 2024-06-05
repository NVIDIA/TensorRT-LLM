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

#include "tensorrt_llm/common/assert.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace tensorrt_llm
{
namespace kernels
{

enum class DecodingPenaltyType
{
    Temperature, // the temperature penalty
    Repetition,  // the repetition penalty
    Presence,    // the presence penalty
    Frequency,   // the frequency penalty
    MinLength,   // the min length penalty
};

inline std::pair<float, float> getLimitsPenalty(DecodingPenaltyType penaltyType)
{
    auto constexpr fltMax = std::numeric_limits<float>::max();
    auto constexpr fltMin = std::numeric_limits<float>::lowest();
    auto constexpr fltEpsilon = std::numeric_limits<float>::epsilon();

    switch (penaltyType)
    {
    case DecodingPenaltyType::Temperature: return std::make_pair(0.f, fltMax);
    case DecodingPenaltyType::Repetition: return std::make_pair(0.f, fltMax);
    case DecodingPenaltyType::Presence: return std::make_pair(fltMin, fltMax);
    case DecodingPenaltyType::Frequency: return std::make_pair(fltMin, fltMax);
    case DecodingPenaltyType::MinLength: return std::make_pair(-fltEpsilon, fltMax);
    }
    TLLM_CHECK_WITH_INFO(false, "Unknown penalty type %d", static_cast<int32_t>(penaltyType));
    return std::make_pair(fltMin, fltMax);
}
} // namespace kernels
} // namespace tensorrt_llm
