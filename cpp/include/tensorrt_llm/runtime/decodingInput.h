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

#include <memory>
#include <optional>

namespace tensorrt_llm::runtime
{
class DecodingInput
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    DecodingInput(SizeType32 maxLength, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 batchSize,
        TensorPtr logits, TensorPtr endIds)
        : step{maxLength}
        , maxLength{maxLength}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , batchSize{batchSize}
        , maxStopWordsLen{0}
        , maxBadWordsLen{0}
        , logits{std::move(logits)}
        , endIds{std::move(endIds)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
    }

    // mandatory parameters
    SizeType32 step;
    SizeType32 maxLength;
    SizeType32 maxAttentionWindow;
    SizeType32 sinkTokenLength;
    SizeType32 batchSize;
    SizeType32 maxStopWordsLen; // The maximum value in the `stopWordsLens` tensor
    SizeType32 maxBadWordsLen;  // The maximum value in the `badWordsLens` tensor
    TensorPtr logits;           // [batchSize, beamWidth, vocabSizePadded], on gpu
    std::optional<std::vector<TensorPtr>>
        logitsVec;    // vector of size [batchSize] contains logits of size [beamWidth, vocabSizePadded], on gpu
    TensorPtr endIds; // [batchSize * beamWidth], on gpu

    // optional parameters
    TensorPtr finished;            // [batchSize, beamWidth], finished states at current iteration.
                                   // If true for some request, the decoding step of it is skipped, on gpu
    TensorPtr sequenceLimitLength; // [batchSize], on gpu
    TensorPtr embeddingBias;       // [batchSize, vocabSizePadded], on gpu
    TensorPtr lengths;             // [batchSize, beamWidth], on gpu
    TensorPtr badWordsList;        // [2, badWordsLength] or [batchSize, 2, badWordsLength], on gpu
    TensorPtr badWordsPtrs;        // [batchSize][2, badWordsLength], on gpu
    TensorPtr badWordsLens;        // [batchSize], on gpu
    TensorPtr stopWordsList;       // [batchSize, 2, stopWordsLength], on gpu
    TensorPtr stopWordsPtrs;       // [batchSize][2, stopWordsLength], on gpu
    TensorPtr stopWordsLens;       // [batchSize], on gpu
    TensorPtr noRepeatNgramSize;   // [batchSize], on gpu
    TensorPtr
        batchSlots; // [batchSize], optional, address map of the linear batch id to to the seq slots, int32_t, pinned

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, beamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu

    // Medusa
    class MedusaInputs
    {
    public:
        TensorPtr medusaPaths;   // [batchSize, maxTokensPerStep, maxMedusaHeads + 1], on gpu
        TensorPtr medusaTreeIds; // [batchSize, maxTokensPerStep], on gpu
        std::vector<std::vector<TensorPtr>>
            medusaLogits; // [batchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded], on gpu
        TensorPtr medusaCurTokensPerStep;    // [batchSize], on gpu
        TensorPtr medusaTargetTokensPerStep; // [batchSize], on gpu
    };

    class ExplicitDraftTokensInputs
    {
    public:
        TensorPtr nextDraftTokens;       // [batchSize, maxNumPaths, maxPathLen]
        TensorPtr nextFlatTokens;        // [batchSize * maxDecodingTokens]
        TensorPtr nextDraftIndices;      // [batchSize, maxNumPaths, maxPathLen]
        TensorPtr nextDraftProbs;        // [batchSize, maxNumPaths, maxDraftPathLen, vocabSize]
        TensorPtr lastDraftTokens;       // [batchSize, maxNumPaths, maxPathLen]
        TensorPtr lastDraftIndices;      // [batchSize, maxNumPaths, maxPathLen]
        TensorPtr masks;                 // [batchSize, maxDecodingTokens, maxDecodingTokens], bool
        TensorPtr packedPositionIds;     // [batchSize * maxDecodingTokens]
        TensorPtr bestPathLengths;       // [batchSize]
        TensorPtr bestPathIndices;       // [batchSize]
        TensorPtr nextGenerationLengths; // [batchSize]
        TensorPtr lastPositionIdsBase;   // [batchSize]
        TensorPtr lastGenerationLengths; // [batchSize]
        TensorPtr maxGenLengthDevice;    // [1]
        TensorPtr seqSlots;              // [batchSize]
    };

    std::optional<MedusaInputs> medusaInputs;

    std::optional<ExplicitDraftTokensInputs> explicitDraftTokensInputs;
};

} // namespace tensorrt_llm::runtime
