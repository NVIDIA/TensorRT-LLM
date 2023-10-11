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
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class StatefulGptDecoder : public IStatefulGptDecoder
{
public:
    StatefulGptDecoder(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    //! Setup the decoder before calling `forward()`
    void setup(
        SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype) override;

    //! @brief Initialize the decoder with new batch of inputs.
    void newBatch(GenerationInput const& input, SamplingConfig const& samplingConfig) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    bool isFinishedSync() override;

    //! @brief Gather final results for all requests.
    [[nodiscard]] TensorPtr getFinalOutputIds() const override;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu
    [[nodiscard]] TensorPtr getOutputIds() const override
    {
        return mDecodingOutput->ids;
    }

    //! @returns [batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr getNewTokens() const override
    {
        return mDecodingOutput->newTokens;
    }

    //! @returns [1], number of finished sequences, in pinned host memory
    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mDecodingOutput->finishedSum;
    }

private:
    void reshapeBuffers(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

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

    SizeType mNbSteps;
    SizeType mMaxSequenceLength{};
    SizeType mMaxNewTokens;
};
} // namespace tensorrt_llm::runtime
