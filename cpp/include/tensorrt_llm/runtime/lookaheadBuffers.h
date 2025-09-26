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
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
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

class LookaheadRuntimeBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using TensorMap = StringPtrMap<ITensor>;

    LookaheadRuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, BufferManager const& manager,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
        TllmRuntime const& runtime);

    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, ITensor const& requestTypes,
        ITensor const& seqSlots, LookaheadDecodingBuffers const& decoderLookaheadBuffers, TllmRuntime const& runtime,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 tokensPerStep);

    void insertInputTensors(TensorMap& inputBuffers, TensorMap& outputBuffers, WorldConfig const& worldConfig) const;

    void enableLookaheadDecoding(SizeType32 maxBatchSize, SizeType32 tokensPerStep);

    void disableLookaheadDecoding();

public:
    TensorPtr cumSumLength;            // [1] the cumulative sum of generation length, on pinned
    TensorPtr packedMasksDevice;       // [forwardBatchSize, tokensPerStep, numPackedMasks], on gpu
    TensorPtr generationLengthsDevice; // [forwardBatchSize], on gpu
    TensorPtr positionOffsetsDevice;   // [forwardBatchSize, tokensPerStep], on gpu
    TensorPtr positionIdsDevice;       // [forwardBatchSize, tokensPerStep], on gpu

    TensorPtr packedMaskHost;
    TensorPtr generationLengthsHost;
    TensorPtr positionOffsetsHost;
    TensorPtr positionIdsHost;

    TensorPtr packedMaskHostCopy;
    TensorPtr generationLengthsHostCopy;
    TensorPtr positionOffsetsHostCopy;
    TensorPtr positionIdsHostCopy;
    TensorPtr useSpecDecoding;

    TensorPtr batchSlotsHostCopy;
};

} // namespace tensorrt_llm::runtime
