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
#include "tensorrt_llm/runtime/explicitDraftTokensModule.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cstddef>

namespace tensorrt_llm::runtime
{

class ExplicitDraftTokensBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using ITensor = runtime::ITensor;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    class Inputs
    {
    public:
        //! [maxBatchSize]
        TensorPtr temperatures;
        //! [maxBatchSize]
        TensorPtr positionIdsBase;
        //! [maxBatchSize] or [numGenSequences]
        TensorPtr generationLengths;
        //! [maxBatchSize]
        TensorPtr randomDataSample;
        //! [maxBatchSize, maxNumPaths, maxPathDraftLen] or [numGenSequences, maxNumPaths, maxPathDraftLen]
        TensorPtr randomDataValidation;
        //! [maxBatchSize, maxNumPaths, maxPathLen] or [numGenSequences, maxNumPaths, maxPathLen]
        TensorPtr draftTokens;
        //! [maxBatchSize, maxNumPaths, maxPathLen] or [numGenSequences, maxNumPaths, maxPathLen]
        TensorPtr draftIndices;
        //! [maxBatchSize, maxNumPaths, maxPathDraftLen, vocabSize]
        //! or [numGenSequences, maxNumPaths, maxPathDraftLen, vocabSize]
        TensorPtr draftProbs;
        //! [maxBatchSize, maxDecodingTokens, ceil(maxDecodingTokens / 32)]
        //! or [numGenSequences, maxDecodingTokens, ceil(maxDecodingTokens / 32)]
        TensorPtr packedMasks;
        //! [maxBatchSize] or [numGenSequences]
        TensorPtr positionIds;
        // [1], on pinned
        TensorPtr maxGenLengthHost;
        // [maxBatchSize]
        TensorPtr generationLengthsHost;
        // [1], on cpu
        TensorPtr useSpecDecoding;

        void create(SizeType32 maxNumSequences, runtime::BufferManager const& manager,
            runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
    };

    class EngineInputs : public Inputs
    {
    public:
        //! [numSequences], on gpu
        TensorPtr requestTypesDevice;
        //! [numGenSequences]
        TensorPtr positionOffsets;
    } engineInputs;

    class EngineOutputs
    {
    public:
        //! [batchSize]
        TensorPtr nextGenerationLengths;
        //! [batchSize]
        TensorPtr nextPositionOffsets;
        //! [batchSize, maxDecodingTokens, maxDecodingTokens], bool
        TensorPtr masks;

        //! [batchSize, maxNumPaths, maxPathLen]
        TensorPtr nextDraftTokens;
        //! [batchSize, maxNumPaths, maxPathLen]
        TensorPtr nextDraftIndices;
        //! [batchSize, maxNumPaths, maxDraftPathLen, vocabSize]
        TensorPtr nextDraftProbs;

        //! [batchSize * maxDecodingTokens]
        TensorPtr nextFlatTokens;
        //! [batchSize]
        TensorPtr bestPathLengths;
        //! [batchSize]
        TensorPtr bestPathIndices;
        //! [1]
        TensorPtr maxGenToken;
        //! [1]
        TensorPtr totalGenToken;
        //! [batchSize * maxDecodingTokens]
        TensorPtr packedPositionIds;
    } engineOutputs;

public:
    ExplicitDraftTokensBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig);

    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ITensor const& requestTypes,
        ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& decoderBuffers,
        ITensor const& contextPositionIds, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::BufferManager const& manager,
        runtime::CudaStream const& stream) const;

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

private:
    template <typename T>
    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 vocabSizePadded,
        ITensor const& seqSlots, ExplicitDraftTokensBuffers::Inputs const& draftBuffers,
        ITensor const& contextPositionIds, runtime::ExplicitDraftTokensModule const& explicitDraftTokensModule,
        runtime::CudaStream const& stream) const;

public:
    // helper tensors
    std::size_t scanTempStorageBytes{0};
    BufferPtr scanTempStorage;
    TensorPtr cumSumGenerationLengths;
};

} // namespace tensorrt_llm::runtime
