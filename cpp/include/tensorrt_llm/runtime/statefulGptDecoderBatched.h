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

#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"

namespace tensorrt_llm::runtime
{

class GptDecoderBatched;

//! GPT decoder class with support for in-flight batching and stateful interface
class StatefulGptDecoderBatched : public IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = ITensor::SharedPtr;

    StatefulGptDecoderBatched(CudaStreamPtr stream, nvinfer1::DataType dtype);

    ~StatefulGptDecoderBatched() override;

    // IStatefulGptDecoder implementation
    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig) override;

    void newBatch(GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig,
        ModelConfig const& modelConfig) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;
    void forwardSync() override;

    [[nodiscard]] TensorPtr getIds() const override;
    [[nodiscard]] TensorPtr getGatheredIds() const override;
    void finalize(SamplingConfig const& samplingConfig) const override;
    [[nodiscard]] TensorPtr getCumLogProbs() const override;
    [[nodiscard]] TensorPtr getLogProbs() const override;
    [[nodiscard]] TensorPtr getNewTokens(SizeType32 iter = 0) const override;
    [[nodiscard]] TensorPtr getNbFinished() const override;

private:
    std::unique_ptr<GptDecoderBatched> mDecoder;

    // only used for IStatefulGptDecoder
    CudaEvent mDecoderFinishEvent;
    CudaEvent mForwardEvent;
    TensorPtr mFinishedSum;
    TensorPtr mBatchSlotsSetup;   // [maxBatchSize], int32_t, address map, pinned
    TensorPtr mBatchSlotsDecoder; // [maxTokensPerEngineStep, maxBatchSize], int32_t, address map, pinned
};

} // namespace tensorrt_llm::runtime
