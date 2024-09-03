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

    DecodingInput(SizeType32 maxLength, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 batchSize,
        TensorConstPtr logits, TensorPtr endIds, TensorConstPtr batchSlots)
        : step{maxLength}
        , maxLength{maxLength}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , batchSize{batchSize}
        , maxStopWordsLen{0}
        , maxBadWordsLen{0}
        , logits{std::move(logits)}
        , endIds{std::move(endIds)}
        , batchSlots{std::move(batchSlots)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
    }

    // mandatory parameters

    SizeType32 step;               //!< The index of the decoding step we are on. Only used in Python runtime.

    SizeType32 maxLength;          //!< The maximum number of tokens to decode.

    SizeType32 maxAttentionWindow; //!< The maximum length of the attention window to consider while decoding.

    SizeType32 sinkTokenLength;    //!< the number of tokens to use as attention sinks, as described there: @link
                                   //!< https://arxiv.org/html/2309.17453v3

    SizeType32 batchSize;          //!< The number of samples in the batch.

    SizeType32 maxStopWordsLen;    //!<  The maximum value in the `stopWordsLens` tensor.

    SizeType32 maxBadWordsLen;     //!<  The maximum value in the `badWordsLens` tensor.

    TensorConstPtr logits;         //!<  [batchSize, beamWidth, vocabSizePadded], on gpu. Logits are are a probability
                                   //!<  distribution over the vocabulary, the output of the model.
    std::optional<std::vector<TensorConstPtr>>
        logitsVec; //!< Vector of size [batchSize] contains logits of size [beamWidth, vocabSizePadded], on gpu. This is
                   //!< another view on the @property logits

    TensorConstPtr endIds; //!<  [batchSize * beamWidth], on gpu

    TensorConstPtr
        batchSlots; //!<  [batchSize], address map of the linear batch id to to the seq slots, int32_t, pinned

    // optional parameters
    TensorConstPtr finishReasons; //!<  [batchSize, beamWidth], finished states at current iteration.
                                  //!<  If true for some request, the decoding step of it is skipped, on gpu
    TensorConstPtr
        sequenceLimitLength;      //!<  [batchSize], on gpu. The maximum sequence length for each sequence in the batch.
    TensorConstPtr embeddingBias; //!<  [batchSize, vocabSizePadded], on gpu
    TensorConstPtr lengths;       //!<  [batchSize, beamWidth], on gpu
    std::vector<TensorPtr> badWordsLists;  // vector with batchSize elements of size [2, badWordsLength], on gpu
    TensorConstPtr badWordsPtrs;           //!<  [batchSize][2, badWordsLength], on gpu
    TensorConstPtr badWordsLens;           //!<  [batchSize], on gpu
    std::vector<TensorPtr> stopWordsLists; // vector with batchSize elements of size [2, stopWordsLength], on gpu
    TensorConstPtr stopWordsPtrs;          //!<  [batchSize][2, stopWordsLength], pinned
    TensorConstPtr stopWordsLens;          //!<  [batchSize], pinned
    TensorConstPtr noRepeatNgramSize;      //!<  [batchSize], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection; //!<  [batchSize, beamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu

    // Medusa
    class MedusaInputs
    {
    public:
        TensorConstPtr medusaPaths;   //!<  [batchSize, maxTokensPerStep, maxMedusaHeads + 1], on gpu
        TensorConstPtr medusaTreeIds; //!<  [batchSize, maxTokensPerStep], on gpu
        std::vector<std::vector<TensorPtr>>
            medusaLogits; //!<  [batchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded], on gpu
        TensorPtr medusaCurTokensPerStep;         //!<  [batchSize], on gpu
        TensorConstPtr medusaTargetTokensPerStep; //!<  [batchSize], on gpu
    };

    class ExplicitDraftTokensInputs
    {
    public:
        TensorConstPtr nextDraftTokens;       //!<  [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr nextFlatTokens;        //!<  [batchSize * maxDecodingTokens]
        TensorConstPtr nextDraftIndices;      //!<  [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr nextDraftProbs;        //!<  [batchSize, maxNumPaths, maxDraftPathLen, vocabSize]
        TensorConstPtr lastDraftTokens;       //!<  [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr lastDraftIndices;      //!<  [batchSize, maxNumPaths, maxPathLen]
        TensorConstPtr masks;                 //!<  [batchSize, maxDecodingTokens, maxDecodingTokens], bool
        TensorConstPtr packedPositionIds;     //!<  [batchSize * maxDecodingTokens]
        TensorConstPtr bestPathLengths;       //!<  [batchSize]
        TensorConstPtr bestPathIndices;       //!<  [batchSize]
        TensorConstPtr nextGenerationLengths; //!<  [batchSize]
        TensorConstPtr lastPositionIdsBase;   //!<  [batchSize]
        TensorConstPtr lastGenerationLengths; //!<  [batchSize]
        TensorConstPtr maxGenLengthDevice;    //!<  [1]
        TensorConstPtr seqSlots;              //!<  [batchSize]
    };

    struct LookaheadInputs
    {
        TensorPtr tokensPerStep;
    };

    std::optional<MedusaInputs> medusaInputs;

    std::optional<ExplicitDraftTokensInputs> explicitDraftTokensInputs;

    std::optional<LookaheadInputs> lookaheadInputs;
};

} // namespace tensorrt_llm::runtime
