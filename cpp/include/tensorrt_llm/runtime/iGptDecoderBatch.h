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
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

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
    using ConstTensorPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(ConstTensorPtr ids, SizeType inputLen, std::optional<SizeType> maxNewTokens = std::nullopt,
        std::optional<SizeType> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , computeCumLogProbs(false)
        , computeLogProbs(false)
    {
    }

    // the number of tokens generated per step
    SizeType generatedTokensPerStep() const
    {
        return draftTokens ? draftTokens->getSize() + 1 : 1;
    }

    // mandatory parameters
    ConstTensorPtr ids; // [inputSeqLen], the input sequence of token ids, on gpu
    SizeType inputLen;  // the input length without draft tokens

    // optional parameters
    std::optional<SizeType> maxNewTokens; // maximum number of tokens to generate for this request
    std::optional<SizeType> endId;        // end token id
    BufferPtr draftTokens;   // [generatedTokensPerStep - 1], on gpu, draft tokens from speculative decoding
    std::optional<TensorPtr>
        draftLogits;         // [generatedTokensPerStep - 1, vocabSize], on gpu, draft tokens from speculative decoding
    TensorPtr embeddingBias; // [vocabSizePadded], on gpu
    TensorPtr badWordsList;  // [2, badWordsLength], on gpu
    TensorPtr stopWordsList; // [2, stopWordsLength], on gpu

    bool computeCumLogProbs; // boolean that controls if cumLogProbs should be computed for that request
    bool computeLogProbs;    // boolean that controls if cumLogProbs should be computed for that request
};

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<TensorConstPtr> const& logits, std::vector<bool> const& active)
        : logits{logits}
        , active{active}
    {
        TLLM_CHECK_WITH_INFO(
            this->active.size() == logits.size(), "'active' vector size does not match logits vector size");
    }

    explicit Input(std::vector<TensorConstPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits, std::vector<bool> const& active)
        : Input{
            utils::transformVector(logits, [](auto& x) { return std::const_pointer_cast<ITensor const>(x); }), active}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    // mandatory parameters
    std::vector<TensorConstPtr>
        logits; // batchSize * [1, beamWidth, vocabSizePadded] or [generatedTokensPerStep, 1, vocabSizePadded], on gpu

    // control activity of decoder slots in batch
    std::vector<bool> active; // [batchSize]

    // parameters for beam search
    TensorConstPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen] - indices into KV cache of different rays
                                     // within one beam for beam search, on gpu
};

using Output = decoder::Output;

class Token
{
public:
    explicit Token(CudaEvent&& event, std::vector<bool> const& active)
        : event(std::move(event))
        , active(active)
    {
    }

    CudaEvent event;
    std::vector<bool> active;
};
} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class IGptDecoderBatch : public virtual IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using TokenPtr = std::unique_ptr<decoder_batch::Token const>;

    //! @brief Initialize the decoder at `batchIdx` with a new `request`.
    virtual void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig)
        = 0;

    //! @brief Run one step for all requests without blocking the host process and return the token for synchronization.
    virtual TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Wait for the call to `forwardAsync` associated with a token to complete.
    virtual void forwardSync(decoder_batch::Token const& token) = 0;

    //! @brief Run one step for all requests and wait for completion on the host.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input)
    {
        forwardSync(*forwardAsync(output, input));
    }

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding for request `batchIdx`, on gpu
    virtual TensorPtr getOutputIds(SizeType batchIdx) const = 0;

    //! @brief Gather final beam search results for request `batchIdx`.
    //! Result will only be available after event returned
    virtual CudaEvent finalize(SizeType batchIdx) const = 0;

    //! @returns [batchSize (actual)], marks finished requests (per batch)
    virtual std::vector<bool> getFinished() const = 0;

    //! @returns [batchSize, beamWidth], cumulative log probabilities (per beam), on gpu
    virtual TensorPtr getCumLogProbs() const = 0;

    //! @returns [beamWidth], cumulative log probabilities (per beam) for request batchIdx, on gpu
    virtual TensorPtr getCumLogProbs(SizeType batchIdx) const = 0;

    //! @returns [batchSize, beamWidth, maxSeqLen], log probabilities (per beam), on gpu
    virtual TensorPtr getLogProbs() const = 0;

    //! @returns [beamWidth, maxSeqLen], cumulative log probabilities (per beam) for request batchIdx, on gpu
    virtual TensorPtr getLogProbs(SizeType batchIdx) const = 0;

    virtual TensorPtr getParentIds() const = 0;

    virtual std::vector<SizeType> getNbSteps() const = 0;

protected:
    IGptDecoderBatch() = default;
};

} // namespace tensorrt_llm::runtime
