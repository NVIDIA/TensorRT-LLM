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

#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

namespace decoder
{

class Input
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Input(TensorPtr logits)
        : logits{std::move(logits)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
    }

    // mandatory parameters
    TensorPtr logits; // [batchSize, maxBeamWidth, vocabSizePadded], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen], mandatory in beam search, on gpu
    TensorPtr sequenceLengths;  // [batchSize, maxBeamWidth], mandatory, on gpu
};
} // namespace decoder

//! GPT decoder class with support for in-flight batching
class IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    //! Setup the decoder before calling `forward()`, also calls reshapeBuffers
    virtual void setup(
        SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype)
        = 0;

    //! @brief Initialize the decoder with new batch of inputs.
    virtual void newBatch(GenerationInput const& inputs, SamplingConfig const& samplingConfig) = 0;

    //! @brief Run one step for all requests without blocking the host thread.
    virtual void forwardAsync(decoder::Output& output, decoder::Input const& input) = 0;

    //! @brief Wait for the last call to `forwardAsync` to complete and return whether all sequences have finished.
    virtual bool isFinishedSync() = 0;

    //! @brief Run one step for all requests.
    virtual bool forward(decoder::Output& output, decoder::Input const& input)
    {
        forwardAsync(output, input);
        return isFinishedSync();
    }

    //! @brief Gather final results for all requests.
    virtual TensorPtr getFinalOutputIds() const = 0;

    //! @returns [batchSize, beamWidth, maxSequenceLength], all token ids, on gpu
    virtual TensorPtr getOutputIds() const = 0;

    //! @returns [batchSize, beamWidth], latests generated tokens (per beam), on gpu
    virtual TensorPtr getNewTokens() const = 0;

    //! @returns [1], number of finished sequences, in pinned host memory
    virtual TensorPtr getNbFinished() const = 0;

protected:
    IStatefulGptDecoder() = default;
};

} // namespace tensorrt_llm::runtime
