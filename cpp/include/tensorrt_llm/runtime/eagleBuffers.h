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
#include "tensorrt_llm/runtime/eagleModule.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cstddef>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
}

namespace tensorrt_llm::runtime
{

class EagleBuffers
{
public:
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
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
        TensorPtr posteriorAlpha;
        //! [maxBatchSize] or [numSequences]
        TensorPtr posteriorThreshold;
        //! [maxBatchSize] or [numSequences]
        TensorPtr randomDataSample;
        //! [maxBatchSize, maxDecodingTokens] or [numSequences, maxDecodingTokens]
        TensorPtr randomDataValidation;
        //! [maxBatchSize, maxDecodingDraftTokens] or [numSequences, maxDecodingDraftTokens]
        TensorPtr draftTokens;
        //! [maxBatchSize] or [numSequences]
        TensorPtr draftLens;
        //! [maxBatchSize, maxNumPaths, maxPathLen]
        //! or [numSequences, maxNumPaths, maxPathLen]
        TensorPtr draftPaths;
        //! [maxBatchSize, maxNumPaths, maxPathLen]
        //! or [numSequences, maxNumPaths, maxPathLen]
        TensorPtr draftPathsHost;
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
        //! [maxBatchSize] or [numSequences]
        TensorPtr chunkedContextNextTokens;
        //! [1]
        TensorPtr useSpecDecoding;

        // For Eagle-2
        //! [1]
        TensorPtr useDynamicTreeHost;
        //! [1]
        TensorPtr dynamicTreeMaxTopKHost;
        //! [maxBatchSize, maxDecodingDraftTokens] or [numSequences, maxDecodingDraftTokens]
        TensorPtr prevScores;
        //! [maxBatchSize, maxDecodingDraftTokens] or [numSequences, maxDecodingDraftTokens]
        TensorPtr currentExpandIndices;
        //! [maxBatchSize, numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens] or [numSequences,
        //! numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens]
        TensorPtr allLayersScores;
        //! [maxBatchSize, numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens] or [numSequences,
        //! numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens]
        TensorPtr allLayersDraftTokenIds;
        //! [maxBatchSize, numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens] or [numSequences,
        //! numEagleLayers, maxDecodingDraftTokens * maxDecodingDraftTokens]
        TensorPtr allLayersDraftTokenIdsPredecessor;

        void create(SizeType32 maxNumSequences, BufferManager const& manager, ModelConfig const& modelConfig,
            WorldConfig const& worldConfig);
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
        //! [batchSize]
        TensorPtr chunkedContextNextTokens;

    } engineOutputs;

public:
    EagleBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::BufferManager const& manager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig);

    void reshape(SizeType32 numCtxSequences, SizeType32 numGenSequences, runtime::ModelConfig const& modelConfig);

    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
        runtime::ITensor const& requestTypes, ITensor const& seqSlots, EagleBuffers::Inputs const& decoderBuffers,
        runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig) const;

    void insertInputTensors(
        TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::WorldConfig const& worldConfig) const;

private:
    template <typename T>
    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
        SizeType32 vocabSizePadded, ITensor const& seqSlots, EagleBuffers::Inputs const& draftBuffers,
        runtime::EagleModule const& eagleModule, runtime::BufferManager const& manager) const;

private:
    // helper tensors
    std::size_t scanReduceTempStorageBytes{0};
    float mDefaultPosteriorThreshold{0.09f};
    bool mDoGreedySampling{true};
    BufferPtr scanReduceTempStorage;
    TensorPtr cumSumGenerationLengths;
    TensorPtr maxGenerationLength;
    TensorPtr chunkedContextNextTokensHost;
    TensorPtr greedySamplingHost;
    TensorPtr posteriorAlphaHost;
    TensorPtr posteriorThresholdHost;
};

} // namespace tensorrt_llm::runtime
