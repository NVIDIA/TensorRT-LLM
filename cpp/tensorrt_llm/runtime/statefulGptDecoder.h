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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <memory>

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class StatefulGptDecoder : public IStatefulGptDecoder
{
public:
    StatefulGptDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    //! Setup the decoder before calling `forward()`
    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) override;

    //! @brief Initialize the decoder with new batch of inputs.
    void newBatch(GenerationInput const& input, GenerationOutput const& output, SamplingConfig const& samplingConfig,
        ModelConfig const& modelConfig) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    //! @brief Gather final results for all requests.
    void finalize(SamplingConfig const& samplingConfig) const override;

    //! @param step index within tokens generated in one step
    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu
    [[nodiscard]] TensorPtr getIds() const override
    {
        return mDecodingOutput->ids;
    }

    // This implementation is here to satisfy the interface requirement. Returns ids instead
    [[nodiscard]] TensorPtr getGatheredIds() const override
    {
        return mDecodingOutput->ids;
    }

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return mDecodingOutput->cumLogProbs;
    }

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return mDecodingOutput->logProbs;
    }

    //! @brief Get tokens generated in one step of last forward pass
    //! @param iter The iteration within [0; maxTokensPerStep) for which to get the tokens
    //! @returns [batchSize, beamWidth], tokens generated in `iter` (per beam), on gpu
    [[nodiscard]] TensorPtr getNewTokens(SizeType32 iter = 0) const override
    {
        TLLM_CHECK(iter == 0);
        return mDecodingOutput->newTokens;
    }

    //! @brief Get tokens generated in the last forward pass
    //! @returns [batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        TensorPtr newTokens = ITensor::view(mDecodingOutput->newTokensSteps);
        newTokens->unsqueeze(0);
        return newTokens;
    }

    //! @returns [1], number of finished sequences, in pinned host memory
    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

private:
    void reshapeBuffers(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 mMaxAttentionWindow,
        SizeType32 mSinkTokenLength, SizeType32 maxSequenceLength);

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    DecodingInputPtr mDecodingInput;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingOutputPtr mDecodingOutput;
    CudaEvent mDecodedEvent{};

    TensorPtr mFinishedSum;
    TensorPtr mSetupBatchSlots;

    SizeType32 mNbSteps;
    SizeType32 mMaxSequenceLength{};
    SizeType32 mMaxAttentionWindow{};
    SizeType32 mSinkTokenLength{};
};
} // namespace tensorrt_llm::runtime
