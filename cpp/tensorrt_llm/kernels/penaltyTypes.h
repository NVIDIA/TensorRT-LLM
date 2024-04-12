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

inline float getDefaultPenaltyValue(DecodingPenaltyType penalty_type)
{
    switch (penalty_type)
    {
    case DecodingPenaltyType::Temperature: return 1.0f;
    case DecodingPenaltyType::Repetition: return 1.0f;
    case DecodingPenaltyType::Presence: return 0.0f;
    case DecodingPenaltyType::Frequency: return 0.0f;
    case DecodingPenaltyType::MinLength: return 1.0f;
    default: break;
    }
    return 0.0f;
}

} // namespace kernels
} // namespace tensorrt_llm
