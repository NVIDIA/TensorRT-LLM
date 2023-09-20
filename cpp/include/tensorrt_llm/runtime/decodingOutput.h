/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <utility>

namespace tensorrt_llm::runtime
{
class DecodingOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    class BeamHypotheses
    {
    public:
        TensorPtr outputIdsTgt;       // [batchSize, 2 * beamWidth, maxSeqLen]
        TensorPtr sequenceLengthsTgt; // [batchSize, 2 * beamWidth]
        TensorPtr cumLogProbs;        // [batchSize, 2 * beamWidth]
        TensorPtr normedScores;       // [batchSize, 2 * beamWidth]
        TensorPtr logProbs;           // [batchSize, 2 * beamWidth, maxSeqLen]
        TensorPtr minNormedScores;    // [batchSize]
        TensorPtr numBeams;           // [batchSize]
        TensorPtr isDone;             // [batchSize]

        void empty(BufferManager& manager);

        void reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

        void release();

        void init(BufferManager& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType batchIndex, SizeType size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    explicit DecodingOutput(TensorPtr ids)
        : ids{std::move(ids)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
    }

    // mandatory parameters
    TensorPtr ids; // [batchSize, beamWidth, maxSeqLen], on gpu, must contain previously generated token ids for all
                   // steps before DecodingInput.step
    TensorPtr newTokens; // [batchSize, beamWidth] on gpu.

    // optional parameters
    TensorPtr finished;    // [batchSize, beamWidth], mandatory in beam search and to determine whether to stop
                           // according to DecodingInput.sequenceLimitLength, on gpu
    TensorPtr finishedSum; // [1], the sum of finished sequences, in pinned memory
    TensorPtr logProbs;    // [maxNewTokens, batchSize, beamWidth], must be float*, on gpu

    // mandatory parameters for beam search
    TensorPtr cumLogProbs;      // [batchSize, beamWidth], optional for sampling, on gpu
    TensorPtr parentIds;        // [batchSize, beamWidth, maxSeqLen], on gpu
    TensorPtr lengths;          // [batchSize, beamWidth], total sequence lengths including padding, on gpu
    TensorPtr cacheIndirection; // [batchSize, beamWidth, maxSeqLen], k/v indirection for next generation step, on gpu

    BeamHypotheses beamHypotheses;
};

} // namespace tensorrt_llm::runtime
