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
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iGptDecoderBatch.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
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

    TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    void forwardSync(decoder_batch::Token const& e) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    bool isFinishedSync() override;

    //! @return [batchSize], indicators of finished requests
    [[nodiscard]] std::vector<bool> getFinished() const override
    {
        return {mFinished.begin(), mFinished.begin() + mActualBatchSize};
    }

    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token ids without
    //! padding for request `batchIdx`, on gpu
    [[nodiscard]] TensorPtr getOutputIds(SizeType batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu
    [[nodiscard]] TensorPtr getOutputIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
    }

    //! Execute postProcessRequest  and returns OutputIds for request `batchIdx`.
    //! Result will only be available after event returned
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token ids without
    //! padding for request `batchIdx`, on gpu
    [[nodiscard]] std::tuple<CudaEvent, TensorPtr> getFinalOutputIds(SizeType batchIdx) const override;

    //! Execute postProcessRequest and returns OutputIds.
    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu
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

    //! @returns [1], number of finished sequences, in pinned host memory
    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

private:
    //! @brief Gather final results for request `batchIdx`
    CudaEvent postProcessRequest(SizeType batchIdx) const;

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mStream;
    BufferManager mBufferManager;
    TokenPtr mForwardToken;
    CudaEvent mForwardEvent;

    std::vector<CudaStreamPtr> mStreams;
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
    TensorPtr mFinishedSum;
    std::vector<SizeType> mMaxNewTokens;
    std::vector<SizeType> mBeamWidths;
    SizeType mMaxSequenceLength{};
    SizeType mActualBatchSize{};
};
} // namespace tensorrt_llm::runtime
