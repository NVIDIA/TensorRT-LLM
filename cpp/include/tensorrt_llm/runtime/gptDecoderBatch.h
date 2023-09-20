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
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iGptDecoderBatch.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class GptDecoderBatch : public IGptDecoderBatch
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoderBatch(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream);

    //! Setup the decoder before calling `forward()`
    void setup(
        SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype) override;

    //! @brief Initialize the decoder at `batchIdx` with a new `request`.
    void newRequest(
        SizeType batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig) override;

    void newBatch(GenerationInput const& inputs, SamplingConfig const& samplingConfig) override;

    //! @brief Run one step for all requests.
    //! Note that this method will synchronize with the stream associated with the decoder.
    void forward(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    bool forward(decoder::Output& output, decoder::Input const& input) override;

    //! @brief Gather final results for request `batchIdx`.
    void postProcessRequest(SizeType batchIdx) const override;

    //! @return [batchSize], indicators of finished requests
    [[nodiscard]] std::vector<bool> getFinished() const override
    {
        return std::vector<bool>(mFinished.begin(), mFinished.begin() + mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu
    [[nodiscard]] TensorPtr getOutputIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
    }

    [[nodiscard]] TensorPtr getFinalOutputIds() const override;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains parent ids collected during beam
    //! search without padding, on gpu
    [[nodiscard]] TensorPtr getParentIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth], marks finished requests (per beam), on gpu
    [[nodiscard]] TensorPtr getFinishedBeams() const override
    {
        return ITensor::slice(mJointDecodingOutput->finished, 0, mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth], total sequence lengths (per beam), on gpu
    [[nodiscard]] TensorPtr getOutputLengths() const override
    {
        return ITensor::slice(mJointDecodingOutput->lengths, 0, mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr getNewTokens() const override
    {
        return ITensor::slice(mJointDecodingOutput->newTokens, 0, mActualBatchSize);
    }

    //! @returns [batchSize], the number of generation steps executed on each request
    [[nodiscard]] std::vector<SizeType> getNbSteps() const override
    {
        return std::vector<SizeType>(mNbSteps.begin(), mNbSteps.begin() + mActualBatchSize);
    }

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;
    tensorrt_llm::common::EventPtr mEventStart, mEventStop;

    std::vector<CudaStreamPtr> mStreams;
    std::vector<tensorrt_llm::common::EventPtr> mEvents;
    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    std::vector<GptDecoderPtr> mDecoders;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    std::vector<DecodingInputPtr> mDecodingInputs;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    std::vector<DecodingOutputPtr> mDecodingOutputs;

    DecodingInputPtr mJointDecodingInput;
    DecodingOutputPtr mJointDecodingOutput;

    std::vector<SizeType> mNbSteps;
    std::vector<bool> mFinished;
    std::vector<SizeType> mMaxNewTokens;
    std::vector<SizeType> mBeamWidths;
    SizeType mMaxSequenceLength{};
    SizeType mActualBatchSize{};
};
} // namespace tensorrt_llm::runtime
