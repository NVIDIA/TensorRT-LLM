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
#include "tensorrt_llm/runtime/eagleModule.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cstddef>

namespace tensorrt_llm::runtime
{

class EagleBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using ITensor = runtime::ITensor;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    // The datastruct is used for runtime buffer that is holding runtime state per request (shape starts with
    // maxBatchSize) and for engine inputs (shape starts with numSequences).
    class Inputs
    {
    public:
        //! [maxBatchSize] or [numSequences]
        TensorPtr temperatures;
        //! [maxBatchSize] or [numSequences]
        TensorPtr randomDataSample;
        //! [maxBatchSize, maxNumPaths, maxPathDraftLen] or [numSequences, maxNumPaths, maxPathDraftLen]
        TensorPtr randomDataValidation;
        //! [maxBatchSize, maxDecodingDraftTokens] or [numSequences, maxDecodingDraftTokens]
        TensorPtr draftTokens;
        //! [maxBatchSize] or [numSequences]
        TensorPtr draftLens;
        //! [maxBatchSize, maxNumPaths, maxPathLen]
        //! or [numSequences, maxNumPaths, maxPathLen]
        TensorPtr draftPaths;
        //! [maxBatchSize] or [numGenSequences]
        TensorPtr specDecodingGenerationLengths;
        //! [maxBatchSize] or [numGenSequences]
        TensorPtr specDecodingGenerationLengthsHost;
        //! [maxBatchSize, maxDecodingTokens, ceil(maxDecodingTokens / 32)]
        //! or [numGenSequences, maxDecodingTokens, ceil(maxDecodingTokens / 32)]
        TensorPtr specDecodingPackedMasks;
        //! [maxBatchSize] or [numGenSequences]
        TensorPtr specDecodingPositionOffsets;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetCtxRequestTypesHost;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetCtxContextLengthsHost;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetCtxPastKeyValueLengthsHost;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetGenRequestTypesHost;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetGenContextLengthsHost;
        //! [maxBatchSize] or [numSequences]
        TensorPtr eagleNetGenPastKeyValueLengthsHost;
        //! [maxBatchSize * maxDecodingTokens] or [numSequences * maxDecodingTokens]
        TensorPtr inputGenTokensHost;

        void create(SizeType32 maxNumSequences, runtime::TllmRuntime const& runtime,
            runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
    };

    Inputs engineInputs;

    class EngineOutputs
    {
    public:
        //! [batchSize, maxDecodingDraftTokens]
        TensorPtr nextDraftTokens;
        //! [batchSize]
        TensorPtr nextDraftLens;
        //! [batchSize, maxNumPaths, maxPathLen]
        TensorPtr nextDraftPaths;

        //! [batchSize, maxPathLen]
        TensorPtr acceptedTokens;
        //! [batchSize]
        TensorPtr acceptedLens;
        //! [batchSize]
        TensorPtr acceptedPaths;

    } engineOutputs;

public:
    EagleBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, runtime::TllmRuntime const& runtime);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig);

    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ITensor const& requestTypes,
        ITensor const& seqSlots, EagleBuffers::Inputs const& decoderBuffers, ITensor const& contextPositionIds,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

private:
    template <typename T>
    void setFromInputs(SizeType32 numCtxSequences, SizeType32 numGenSequences, SizeType32 vocabSizePadded,
        ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers, ITensor const& contextPositionIds,
        runtime::EagleModule const& eagleModule, runtime::CudaStream const& stream) const;

private:
    // helper tensors
    std::size_t scanTempStorageBytes{0};
    std::size_t reduceTempStorageBytes{0};
    BufferPtr scanReduceTempStorage;
    TensorPtr cumSumGenerationLengths;
    TensorPtr maxGenerationLength;
};

} // namespace tensorrt_llm::runtime
