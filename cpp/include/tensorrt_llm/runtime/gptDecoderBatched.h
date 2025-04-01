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
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class GptDecoderBatched : public IGptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = ITensor::SharedPtr;
    using SharedConstPtr = ITensor::SharedConstPtr;

    GptDecoderBatched(
        CudaStreamPtr stream, SpeculativeDecodingMode const& speculativeDecodingMode, nvinfer1::DataType dtype);

    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) override;

    void disableLookahead(
        SizeType32 maxBatchSize, RequestVector const& genRequests, TensorPtr const& batchSlots) override;

    DecoderFinishedEventPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;
    void forward(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    //! @brief Gather final beam search results for request `batchSlot`.
    //! Result will only be available after event returned.
    [[nodiscard]] CudaEvent finalize(decoder::DecoderState const& decoderState, SizeType32 batchSlot,
        SamplingConfig const& samplingConfig, bool streaming) const override;

    decoder::DecoderState& getDecoderState() const
    {
        return *mDecoderState;
    }

    CudaStreamPtr getDecoderStream() const
    {
        return mDecoderStream;
    }

    IGptDecoder& getUnderlyingDecoder() const
    {
        return *mDecoder.get();
    }

    [[nodiscard]] BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

private:
    //! @brief Sets inputs for explicit draft tokens.
    void setExplicitDraftTokensInputs(decoder_batch::Input const& input);

    //! @brief Sets inputs for eagle decoding.
    void setEagleInputs(decoder_batch::Input const& input);

    //! @brief Calls decoders for tokens per engine step
    void forwardDispatch(decoder_batch::Output& output, decoder_batch::Input const& input);

    //! @brief Prepare Input and Output for decoder step
    void prepareForward(SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input);

private:
    CudaStreamPtr mRuntimeStream;
    CudaStreamPtr mDecoderStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;

    std::shared_ptr<decoder::DecoderState> mDecoderState;
};
} // namespace tensorrt_llm::runtime
