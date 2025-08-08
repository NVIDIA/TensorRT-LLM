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

    explicit GptDecoderBatched(CudaStreamPtr stream);

    void setup(executor::DecodingMode const& mode, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig) override;

    void disableLookahead(RequestVector const& genRequests, TensorPtr const& batchSlots) override;

    CudaEvent forwardAsync(decoder::DecoderState const& decoderState, decoder_batch::Input const& input) override;
    void forward(decoder::DecoderState const& decoderState, decoder_batch::Input const& input) override;

    //! @brief Gather final beam search results for request `batchSlot`.
    //! Result will only be available after event returned.
    [[nodiscard]] CudaEvent finalize(decoder::DecoderState const& decoderState, SizeType32 batchSlot,
        SamplingConfig const& samplingConfig, bool streaming) const override;

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
