/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/executor.h"

#include <utility>

namespace tensorrt_llm::executor
{

DebugConfig::DebugConfig(
    bool debugInputTensors, bool debugOutputTensors, StringVec debugTensorNames, SizeType32 debugTensorsMaxIterations)
    : mDebugInputTensors{debugInputTensors}
    , mDebugOutputTensors{debugOutputTensors}
    , mDebugTensorNames{std::move(debugTensorNames)}
    , mDebugTensorsMaxIterations{debugTensorsMaxIterations}
{
}

bool DebugConfig::operator==(DebugConfig const& other) const
{
    return mDebugInputTensors == other.mDebugInputTensors && mDebugOutputTensors == other.mDebugOutputTensors
        && mDebugTensorNames == other.mDebugTensorNames
        && mDebugTensorsMaxIterations == other.mDebugTensorsMaxIterations;
}

[[nodiscard]] bool DebugConfig::getDebugInputTensors() const
{
    return mDebugInputTensors;
}

[[nodiscard]] bool DebugConfig::getDebugOutputTensors() const
{
    return mDebugOutputTensors;
}

[[nodiscard]] DebugConfig::StringVec const& DebugConfig::getDebugTensorNames() const
{
    return mDebugTensorNames;
}

[[nodiscard]] SizeType32 DebugConfig::getDebugTensorsMaxIterations() const
{
    return mDebugTensorsMaxIterations;
}

void DebugConfig::setDebugInputTensors(bool debugInputTensors)
{
    mDebugInputTensors = debugInputTensors;
}

void DebugConfig::setDebugOutputTensors(bool debugOutputTensors)
{
    mDebugOutputTensors = debugOutputTensors;
}

void DebugConfig::setDebugTensorNames(StringVec const& debugTensorNames)
{
    mDebugTensorNames = debugTensorNames;
}

void DebugConfig::setDebugTensorsMaxIterations(SizeType32 debugTensorsMaxIterations)
{
    mDebugTensorsMaxIterations = debugTensorsMaxIterations;
}

} // namespace tensorrt_llm::executor
