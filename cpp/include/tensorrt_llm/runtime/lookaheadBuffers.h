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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{

class LookaheadDecodingBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    LookaheadDecodingBuffers(
        SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, BufferManager const& bufferManager);
    TensorPtr generationLengths; // [mMaxNumRequests]
    TensorPtr positionOffsets;   // [mMaxNumRequests, maxTokensPerStep]
    TensorPtr packedMasks;       // [mMaxNumRequests, maxTokensPerStep, divUp(maxTokensPerStep, 32)]
    TensorPtr positionIds;
};

} // namespace tensorrt_llm::runtime
