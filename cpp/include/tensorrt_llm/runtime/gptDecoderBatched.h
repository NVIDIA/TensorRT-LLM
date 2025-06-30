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
#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::runtime
{
class SamplingConfig;
class IGptDecoder;

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

} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class GptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = ITensor::SharedPtr;

    explicit GptDecoderBatched(CudaStreamPtr stream);

    //! @brief Setup the decoder before calling `forward()`
    void setup(executor::DecodingMode const& mode, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    //! @brief Disable Lookahead decoding.
    void disableLookahead(RequestVector const& genRequests, TensorPtr const& batchSlots);

    //! @brief Run one step for all requests without blocking the host process and return the token for synchronization.
    CudaEvent forwardAsync(decoder::DecoderState const& decoderState, decoder_batch::Input const& input);
    //! @brief Run one step for all requests and wait for completion on the host.
    void forward(decoder::DecoderState const& decoderState, decoder_batch::Input const& input);

    //! @brief Gather final beam search results for request `batchSlot`.
    //! Result will only be available after event returned.
    [[nodiscard]] CudaEvent finalize(decoder::DecoderState const& decoderState, SizeType32 batchSlot,
        SamplingConfig const& samplingConfig, bool streaming) const;

    [[nodiscard]] CudaStreamPtr getDecoderStream() const
    {
        return mDecoderStream;
    }

    [[nodiscard]] IGptDecoder& getUnderlyingDecoder() const
    {
        return *mDecoder.get();
    }

    [[nodiscard]] BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

private:
    //! @brief Calls decoders for tokens per engine step
    void forwardDispatch(decoder::DecoderState const& decoderState, decoder_batch::Input const& input);

private:
    CudaStreamPtr mRuntimeStream;
    CudaStreamPtr mDecoderStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;
};

} // namespace tensorrt_llm::runtime
