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

#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"

#include <vector>

namespace tensorrt_llm::runtime::utils
{
struct TreeNode
{
    SizeType32 nodeId;
    SizeType32 depth;
    SizeType32 parentLinearIdx;
    SizeType32 linearIdx;
    std::vector<SizeType32> childLinearIndices;
};

SizeType32 initTensorsFromChoices(SpeculativeDecodingModule const& speculativeDecodingModule,
    std::vector<std::vector<SizeType32>> const& choices, std::vector<SizeType32>& topKs,
    ITensor::SharedPtr generationInputLengths, ITensor::SharedPtr positionOffsets, ITensor::SharedPtr treeIds,
    ITensor::SharedPtr paths, ITensor::SharedPtr packedMask,
    std::optional<SizeType32> maxNonLeafNodesPerLayer = std::nullopt);

} // namespace tensorrt_llm::runtime::utils
