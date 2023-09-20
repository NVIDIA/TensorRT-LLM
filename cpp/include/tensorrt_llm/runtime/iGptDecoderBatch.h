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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

namespace decoder_batch
{
class Request
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Request(TensorPtr ids, std::optional<SizeType> maxNewTokens = std::nullopt,
        std::optional<SizeType> endId = std::nullopt, std::optional<SizeType> padId = std::nullopt)
        : ids{std::move(ids)}
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , padId{padId}
    {
    }

    // mandatory parameters
    TensorPtr ids; // [inputSeqLen], the input sequence of token ids, on gpu

    // optional parameters
    std::optional<SizeType> maxNewTokens; // maximum number of tokens to generate for this request
    std::optional<SizeType> endId;        // end token id
    std::optional<SizeType> padId;        // pad token id
    TensorPtr embeddingBias;              // [vocabSizePadded], on gpu
    TensorPtr badWordsList;               // [2, badWordsLength], on gpu
    TensorPtr stopWordsList;              // [2, stopWordsLength], on gpu
};

class Input : public decoder::Input
{
public:
    using Base = decoder::Input;

    explicit Input(TensorPtr logits)
        : Base{std::move(logits)}
    {
        auto const batchSize = this->logits->getShape().d[0];
        active.resize(batchSize, true);
    }

    explicit Input(TensorPtr logits, std::vector<bool> const& active)
        : Base{std::move(logits)}
        , active{active}
    {
        auto const batchSize = static_cast<std::size_t>(this->logits->getShape().d[0]);
        TLLM_CHECK_WITH_INFO(this->active.size() == batchSize, "'active' vector size does not match logits batchSize");
    }

    // control activity of decoder slots in batch
    std::vector<bool> active; // [batchSize]
};

using Output = decoder::Output;
} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class IGptDecoderBatch : public virtual IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    //! @brief Initialize the decoder at `batchIdx` with a new `request`.
    virtual void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
        = 0;

    //! @brief Run one step for all requests.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Gather final results for request `batchIdx`.
    virtual void postProcessRequest(SizeType batchIdx) const = 0;

    //! @returns [batchSize, beamWidth], marks finished requests (per beam), on gpu
    virtual TensorPtr getFinishedBeams() const = 0;

    //! @returns [batchSize, beamWidth], total sequence lengths (per beam), on gpu
    virtual TensorPtr getOutputLengths() const = 0;

    //! @returns [batchSize, beamWidth], cumulative log probabilities (per beam), on gpu
    virtual TensorPtr getCumLogProbs() const = 0;

    virtual TensorPtr getParentIds() const = 0;

    virtual std::vector<SizeType> getNbSteps() const = 0;

protected:
    IGptDecoderBatch() = default;
};

} // namespace tensorrt_llm::runtime
