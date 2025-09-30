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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include <optional>
#include <utility>

namespace tensorrt_llm::batch_manager
{
class LookaheadDecodingBuffers;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::runtime
{
class DecodingOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    // BS: batch_size, BM: beam_width, MSL: max_seq_length
    // All TensorPtr without special comments are on gpu

    class BeamHypotheses
    {
    public:
        // Keep same as cpp/tensorrt_llm/kernels/beamSearchKernels.h
        TensorPtr outputIdsCBA;       // [BS, BM*2, MSL]
        TensorPtr logProbsCBA;        // [BS, BM*2, MSL]
        TensorPtr sequenceLengthsCBA; // [BS, BM*2]
        TensorPtr cumLogProbsCBA;     // [BS, BM*2]
        TensorPtr normedScoresCBA;    // [BS, BM*2]
        TensorPtr numBeamsCBA;        // [BS]
        TensorPtr minNormedScoresCBA; // [BS]
        TensorPtr batchDones;         // [BS]

        void empty(BufferManager const& manager);

        void reshape(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength);

        void release();

        void init(BufferManager const& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType32 batchIndex, SizeType32 size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    DecodingOutput() = default;

    //! Mandatory parameters
    //! Previously generated token ids for all steps before DecodingInput.step, [BS, BM, MSL]
    TensorPtr ids;
    //! The tokens computed during the gatherTree step, [BS, BM, MSL]
    //! Necessary for "Streaming + Beam Search" mode since beam search kernels store ungathered tokens in `ids`.
    TensorPtr gatheredIds;
    //! New tokens at each generated token of maxTokensPerStep, [maxTokensPerStep, BS, BM]
    TensorPtr newTokensSteps;
    //! A view of newTokensSteps for the current token, [BS, BM]
    TensorPtr newTokens;
    //! A Vector of views on newTokensSteps for each token [BS, BM].
    std::vector<TensorPtr> newTokensVec;

    //! Optional parameters
    //! FinishedState by decoding if any of the stop conditions are met or if DecodingInput.finished is true, [BS, BM]
    TensorPtr finishReasons;
    //! The sum of finished sequences per request, in pinned memory, [BS]
    TensorPtr finishedSum;

    //! Mandatory parameters for Beam Search
    //! log-probility of generated tokens, [BS, BM, MSL], float
    TensorPtr logProbs;
    //! Sum log-probility of all generated tokens, [BS, BM]
    TensorPtr cumLogProbs;
    //! Index of the beam where the previous token is, [BS, BM, MSL]
    TensorPtr parentIds;
    //! Total sequence lengths including padding, [BS, BM]
    TensorPtr lengths;
    //! K/V indirection for next generation step, [BS, BM, MSL]
    TensorPtr cacheIndirection;
    //! Buffer used to store the transpose of the logProbs, [MSL, BS, BM]
    TensorPtr logProbsTiled;

    BeamHypotheses beamHypotheses;

    // Speculative decoding
    class SpeculativeDecodingOutputs
    {
    public:
        TensorPtr nextDraftTokens;       // [maxBatchSize, maxDraftTokens]
        TensorPtr nextDraftTokensLen;    // [maxBatchSize]
        TensorPtr prevDraftTokensLen;    // [maxBatchSize]
        TensorPtr acceptedTokensLen;     // [maxBatchSize]
        TensorPtr acceptedLengthsCumSum; // [maxBatchSize + 1]
        TensorPtr pathsOffsets;          // [maxBatchSize, maxAcceptedDraftTokensPerStep]
    };

    std::optional<SpeculativeDecodingOutputs> speculativeDecodingOutputs;

    std::optional<ExplicitDraftTokensBuffers::Inputs> explicitDraftTokensBuffers;

    std::optional<LookaheadDecodingBuffers> lookaheadOutputs;

    std::optional<EagleBuffers::Inputs> eagleBuffers;
};

} // namespace tensorrt_llm::runtime
