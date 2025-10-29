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

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>
#include <utility>

namespace tensorrt_llm::runtime
{

/// @brief Represents the inputs to the decoder.
/// @details This input type is assumed immutable. It represents whatever the decoder received initially, and can always
/// be referred to as such.
class DecodingInput
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    DecodingInput() = default;

    //! Mandatory parameters
    //! The index of the decoding step we are on. Only used in Python runtime
    SizeType32 step{};
    //! The maximum number of tokens to decode
    SizeType32 maxLength{};
    //! The maximum length of the attention window to consider while decoding
    SizeType32 maxAttentionWindow{};
    //! The number of tokens to use as attention sinks, https://arxiv.org/html/2309.17453v3
    SizeType32 sinkTokenLength{};
    //! The number of samples in the batch
    SizeType32 batchSize{};
    //! The beam widths of each request, [batchSize]
    std::vector<SizeType32> beamWidths;
    //! The maximum value in the `stopWordsLens` tensor
    SizeType32 maxStopWordsLen{};
    //! The maximum value in the `badWordsLens` tensor
    SizeType32 maxBadWordsLen{};
    //! The output of the model forward computation, a probability distribution over the vocabulary
    //! [batchSize][numGenTokens, beamWidth, vocabSizePadded] on gpu
    std::vector<TensorConstPtr> logitsVec;
    //! The end ids, [batchSize * beamWidth] on gpu
    TensorConstPtr endIds;
    //! Address map of the linear batch id to to the seq slots, [batchSize] on pinned, int32_t
    TensorConstPtr batchSlots;

    //! Optional parameters
    //! Finished states at current iteration (skip decoding step of a request if true), [batchSize, beamWidth] on gpu
    TensorConstPtr finishReasons;
    //! The maximum sequence length for each sequence in the batch, [batchSize] on gpu
    TensorConstPtr sequenceLimitLength;
    TensorConstPtr embeddingBias;          // [batchSize, vocabSizePadded] on gpu
    TensorConstPtr lengths;                // [batchSize, beamWidth] on gpu
    std::vector<TensorPtr> badWordsLists;  // [batchSize][2, badWordsLength] on gpu
    TensorConstPtr badWordsPtrs;           // [batchSize][2, badWordsLength] on pinned
    TensorConstPtr badWordsLens;           // [batchSize] on gpu
    std::vector<TensorPtr> stopWordsLists; // [batchSize][2, stopWordsLength] on gpu
    TensorConstPtr stopWordsPtrs;          // [batchSize][2, stopWordsLength] on pinned
    TensorConstPtr stopWordsLens;          // [batchSize] on pinned
    TensorConstPtr noRepeatNgramSize;      // [batchSize] on gpu

    //! Parameters for beam search
    //! KV cache index for beam search, [batchSize, beamWidth, maxSeqLen] on gpu
    TensorPtr cacheIndirection;
    //! Steps of each request, for Variable-Beam-Width-Search, [batchSize]
    std::optional<std::vector<SizeType32>> generationSteps;

    // Medusa
    class MedusaInputs
    {
    public:
        //! [batchSize, maxTokensPerStep, maxMedusaHeads + 1], on gpu
        TensorConstPtr medusaPaths;
        //! [batchSize, maxTokensPerStep], on gpu
        TensorConstPtr medusaTreeIds;
        //! [batchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded], on gpu
        std::vector<std::vector<TensorPtr>> medusaLogits;
        //! [batchSize], on gpu
        TensorPtr medusaCurTokensPerStep;
        //! [batchSize], on gpu
        TensorConstPtr medusaTargetTokensPerStep;
    };

    class ExternalDraftTokensInputs
    {
    public:
        TensorPtr draftLogits;
        TensorPtr draftLogitsHost;
        TensorPtr draftProbs;
        TensorPtr targetProbs;
        TensorPtr numDraftTokens;
        TensorPtr numDraftTokensHost;
        TensorPtr draftTokenIds;
        TensorPtr draftTokenIdsHost;
        TensorPtr useDraftLogits;
        TensorPtr useDraftLogitsHost;

        SizeType32 step;
        float constantThreshold;
        bool useRandomAcceptanceThreshold;
    };

    class ExplicitDraftTokensInputs
    {
    public:
        TensorConstPtr nextDraftTokens;       // [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr nextFlatTokens;        // [batchSize * maxDecodingTokens]
        TensorConstPtr nextDraftIndices;      // [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr nextDraftProbs;        // [batchSize, maxNumPaths, maxDraftPathLen, vocabSize]
        TensorConstPtr lastDraftTokens;       // [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr lastDraftIndices;      // [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr masks;                 // [batchSize, maxDecodingTokens, maxDecodingTokens], bool
        TensorConstPtr packedPositionIds;     // [batchSize * maxDecodingTokens]
        TensorConstPtr bestPathLengths;       // [batchSize]
        TensorConstPtr bestPathIndices;       // [batchSize]
        TensorConstPtr nextGenerationLengths; // [batchSize]
        TensorConstPtr lastPositionIdsBase;   // [batchSize]
        TensorConstPtr lastGenerationLengths; // [batchSize]
        TensorConstPtr maxGenLengthDevice;    // [1]
        TensorConstPtr seqSlots;              // [batchSize]
    };

    struct LookaheadInputs
    {
        TensorPtr tokensPerStep;
    };

    struct EagleInputs
    {
        TensorConstPtr nextDraftTokens;          // [batchSize, maxDecodingDraftTokens]
        TensorConstPtr nextDraftLens;            // [batchSize]
        TensorConstPtr nextDraftPaths;           // [batchSize, maxDecodingTokens, maxPathLen]
        TensorConstPtr lastDraftTokens;          // [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr lastDraftLens;            // [batchSize]
        TensorConstPtr lastDraftPaths;           // [batchSize, maxDecodingTokens, maxPathLen]
        TensorConstPtr acceptedTokens;           // [batchSize, maxPathLen]
        TensorConstPtr acceptedLens;             // [batchSize]
        TensorConstPtr acceptedPathIds;          // [batchSize]
        TensorConstPtr chunkedContextNextTokens; // [batchSize]
        TensorConstPtr seqSlots;                 // [batchSize]
    };

    std::optional<MedusaInputs> medusaInputs;

    std::optional<ExplicitDraftTokensInputs> explicitDraftTokensInputs;

    std::optional<LookaheadInputs> lookaheadInputs;

    std::optional<ExternalDraftTokensInputs> externalDraftTokensInputs;

    std::optional<EagleInputs> eagleInputs;
};

} // namespace tensorrt_llm::runtime
