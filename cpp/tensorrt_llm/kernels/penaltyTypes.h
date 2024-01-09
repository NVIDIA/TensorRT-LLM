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

enum class RepetitionPenaltyType
{
    Temperature, // the temperature penalty
    Repetition,  // the repetition penalty
    Presence,    // the presence penalty
    Frequency,   // the frequency penalty
    MinLength,   // the min length penalty
};

inline float getDefaultPenaltyValue(RepetitionPenaltyType penalty_type)
{
    switch (penalty_type)
    {
    case RepetitionPenaltyType::Temperature: return 1.0f;
    case RepetitionPenaltyType::Repetition: return 1.0f;
    case RepetitionPenaltyType::Presence: return 0.0f;
    case RepetitionPenaltyType::Frequency: return 0.0f;
    case RepetitionPenaltyType::MinLength: return 1.0f;
    default: break;
    }
    return 0.0f;
}

} // namespace kernels
} // namespace tensorrt_llm
