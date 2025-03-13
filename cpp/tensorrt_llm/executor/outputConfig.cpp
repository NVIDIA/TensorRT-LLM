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

namespace tensorrt_llm::executor
{

OutputConfig::OutputConfig(bool inReturnLogProbs, bool inReturnContextLogits, bool inReturnGenerationLogits,
    bool inExcludeInputFromOutput, bool inReturnEncoderOutput, bool inReturnPerfMetrics,
    std::optional<std::vector<AdditionalModelOutput>> additionalModelOutputs)
    : returnLogProbs(inReturnLogProbs)
    , returnContextLogits(inReturnContextLogits)
    , returnGenerationLogits(inReturnGenerationLogits)
    , excludeInputFromOutput(inExcludeInputFromOutput)
    , returnEncoderOutput(inReturnEncoderOutput)
    , returnPerfMetrics(inReturnPerfMetrics)
    , additionalModelOutputs(std::move(additionalModelOutputs))
{
}

OutputConfig::AdditionalModelOutput::AdditionalModelOutput(std::string name, bool gatherContext)
    : name(std::move(name))
    , gatherContext(gatherContext)
{
}

} // namespace tensorrt_llm::executor
