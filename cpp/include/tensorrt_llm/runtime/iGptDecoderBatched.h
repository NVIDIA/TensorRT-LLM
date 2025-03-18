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

#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"

#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
}

namespace tensorrt_llm::runtime
{

namespace decoder_batch
{

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<TensorPtr> const& logits, std::vector<bool> const& active)
        : logits{logits}
        , active{active}
    {
        TLLM_CHECK_WITH_INFO(
            this->active.size() == logits.size(), "'active' vector size does not match logits vector size");
    }

    explicit Input(std::vector<TensorPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    // mandatory parameters
    std::vector<TensorPtr>
        logits; // batchSize * [1, beamWidth, vocabSizePadded] or [generatedTokensPerStep, 1, vocabSizePadded], on gpu

    // control activity of decoder slots in batch
    std::vector<bool> active; // [batchSize]
    TensorPtr
        batchSlots; // [maxTokensPerEngineStep, batchSize], empty buffer filled in GptDecoderBatched, sorted by slots
    TensorPtr batchSlotsRequestOrder; // [batchSize], filled with slots in request order

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen] - indices into KV cache of different rays
                                // within one beam for beam search, on gpu
    std::vector<std::vector<TensorPtr>>
        predictedDraftLogits;   // [maxBatchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded]

    // explicit draft tokens data.
    std::optional<ExplicitDraftTokensBuffers::EngineOutputs> explicitDraftTokensInputs;
    std::optional<ExplicitDraftTokensBuffers::EngineInputs> explicitDraftTokensLastInputs;

    // eagle data
    std::optional<EagleBuffers::EngineOutputs> eagleInputs;
    std::optional<EagleBuffers::Inputs> eagleLastInputs;
};

using Output = decoder::Output;

// used just as a container for easy returning / passing to function
class DecoderFinishedEvent
{
public:
    explicit DecoderFinishedEvent(CudaEvent&& event, std::vector<bool> const& active)
        : event(std::move(event))
        , active(active)
    {
    }

    CudaEvent event;
    std::vector<bool> active;
};
} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class IGptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using DecoderFinishedEventPtr = std::unique_ptr<decoder_batch::DecoderFinishedEvent const>;

    //! @brief Setup the decoder before calling `forward()`
    virtual void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig)
        = 0;

    //! @brief Setup buffers for ExplicitDraftTokens decoding.
    virtual void setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) = 0;

    //! @brief Setup buffers for Eagle decoding.
    virtual void setupEagle(EagleBuffers::Inputs eagleBuffers) = 0;

    //! @brief Setup buffers for Lookahead decoding.
    virtual void setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) = 0;

    //! @brief Disable Lookahead decoding.
    virtual void disableLookahead(
        SizeType32 maxBatchSize, RequestVector const& genRequests, TensorPtr const& batchSlots)
        = 0;

    //! @brief Run one step for all requests without blocking the host process and return the token for synchronization.
    virtual DecoderFinishedEventPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Run one step for all requests and wait for completion on the host.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding for request `batchIdx`, on gpu
    [[nodiscard]] virtual TensorPtr getIds(SizeType32 batchIdx) const = 0;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], only used for beam search in
    //! GptDecoderBatched It contains gathered token ids without padding, on gpu
    [[nodiscard]] virtual TensorPtr getGatheredIds(SizeType32 batchIdx) const = 0;

    //! @brief Get maxTokensPerStep tokens generated in the last forward pass
    //! @returns [maxTokensPerStep, batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] virtual TensorPtr getAllNewTokens() const = 0;

    //! @brief Gather final beam search results for request `batchIdx`.
    //! Result will only be available after event returned
    [[nodiscard]] virtual CudaEvent finalize(
        SizeType32 batchIdx, SamplingConfig const& samplingConfig, bool streaming) const
        = 0;

    //! @returns [batchSize], number of finished sequences per request, on gpu
    [[nodiscard]] virtual TensorPtr getFinishedSum() const = 0;

    //! @returns [batchSize, beamWidth], FinishedState value, on gpu
    [[nodiscard]] virtual TensorPtr getFinishReasons() const = 0;

    //! @returns [batchSize, beamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getCumLogProbs() const = 0;

    //! @returns [beamWidth], cumulative log probabilities (per beam) for request batchIdx, on gpu
    [[nodiscard]] virtual TensorPtr getCumLogProbs(SizeType32 batchIdx) const = 0;

    //! @returns [batchSize, beamWidth, maxSeqLen], log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getLogProbs() const = 0;

    //! @returns [beamWidth, maxSeqLen], cumulative log probabilities (per beam) for request batchIdx, on gpu
    [[nodiscard]] virtual TensorPtr getLogProbs(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual TensorPtr getParentIds() const = 0;

    [[nodiscard]] virtual executor::DecodingMode getDecodingMode() const = 0;

    //! @returns [batchSize, maxTokensPerStep-1], predicted draft tokens for next step, on gpu
    virtual TensorPtr getNextDraftTokens() const = 0;

    //! @returns [batchSize], predicted draft tokens lengths for previous step, on gpu
    virtual TensorPtr getPrevDraftTokensLengths() const = 0;

    //! @returns [batchSize], predicted draft tokens lengths for next step, on gpu
    virtual TensorPtr getNextDraftTokensLengths() const = 0;

    //! @returns [batchSize + 1], exclusive sum of accepted draft token lengths, on gpu
    virtual TensorPtr getAcceptedLengthsCumSum() const = 0;

    //! @returns [batchSize, maxAcceptedDraftTokensPerStep], accepted paths packed into continuous tensor, on gpu
    virtual TensorPtr getAcceptedPackedPaths() const = 0;

protected:
    IGptDecoderBatched() = default;
    virtual ~IGptDecoderBatched() = default;
};

} // namespace tensorrt_llm::runtime
