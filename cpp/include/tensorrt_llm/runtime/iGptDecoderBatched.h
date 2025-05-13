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
#include "tensorrt_llm/runtime/iTensor.h"

#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
}

namespace tensorrt_llm::runtime
{
class SamplingConfig;

namespace decoder
{
class DecoderState;
}

namespace decoder_batch
{

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<std::vector<TensorConstPtr>> const& logits, SizeType32 maxDecoderSteps)
        : logits{logits}
        , maxDecoderSteps{maxDecoderSteps}
    {
        TLLM_CHECK_WITH_INFO(
            logits.size() == static_cast<size_t>(maxDecoderSteps), "logits vector size does not match maxDecoderSteps");
    }

    explicit Input(std::vector<TensorConstPtr> const& logits)
        : Input{{logits}, 1}
    {
    }

    //! Mandatory parameters
    //! Logits
    // FIXME: remove first dimension of tensors
    //! [maxDecoderSteps][batchSize][1, beamWidth, vocabSizePadded], on gpu
    std::vector<std::vector<TensorConstPtr>> logits;

    //! Maximum number of decoding tokens of active slots
    SizeType32 maxDecoderSteps;

    //! Batch of active decoder slots, sorted by slots, [maxDecoderSteps][batchSize]
    std::vector<TensorPtr> batchSlots;
    //! Filled with slots in request order, [batchSize]
    TensorPtr batchSlotsRequestOrder;

    //! For Beam Search
    //! Indices into KV cache of different rays within one beam, [maxBatchSize, maxBeamWidth, maxSeqLen], on gpu
    TensorPtr cacheIndirection;
    //! The generation step of each request (for Variable-Beam-Width-Search), [batchSize]
    std::vector<SizeType32> generationSteps;

    //! For speculative decoding
    //! Logits of draft
    //! [maxBatchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded]
    std::vector<std::vector<TensorPtr>> predictedDraftLogits;

    //! Explicit draft tokens data
    std::optional<ExplicitDraftTokensBuffers::EngineOutputs> explicitDraftTokensInputs;
    std::optional<ExplicitDraftTokensBuffers::EngineInputs> explicitDraftTokensLastInputs;

    //! Eagle data
    std::optional<EagleBuffers::EngineOutputs> eagleInputs;
    std::optional<EagleBuffers::Inputs> eagleLastInputs;
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    //! parameters for beam search, [batchSize, maxBeamWidth, maxSeqLen], on gpu
    TensorPtr cacheIndirection;
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

    //! @brief Setup the decoder before calling `forward()`
    virtual void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig)
        = 0;

    //! @brief Disable Lookahead decoding.
    virtual void disableLookahead(RequestVector const& genRequests, TensorPtr const& batchSlots) = 0;

    //! @brief Run one step for all requests without blocking the host process and return the token for synchronization.
    virtual CudaEvent forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Run one step for all requests and wait for completion on the host.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Gather final beam search results for request `batchIdx`.
    //! Result will only be available after event returned
    [[nodiscard]] virtual CudaEvent finalize(decoder::DecoderState const& decoderState, SizeType32 batchSlot,
        SamplingConfig const& samplingConfig, bool streaming) const
        = 0;

protected:
    IGptDecoderBatched() = default;
    virtual ~IGptDecoderBatched() = default;
};

} // namespace tensorrt_llm::runtime
