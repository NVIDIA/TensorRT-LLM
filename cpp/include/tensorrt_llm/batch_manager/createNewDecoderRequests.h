/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/request.h"

namespace tensorrt_llm::runtime
{
class DecodingInput;
class DecodingOutput;
class SpeculativeDecodingMode;
// class GptDecoderBatched;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

class CreateNewDecoderRequests : Algorithm
{
public:
    constexpr static auto name{"CreateNewDecoderRequests"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SamplingConfig = tensorrt_llm::runtime::SamplingConfig;
    using CudaStreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SharedConstPtr = runtime::ITensor::SharedConstPtr;
    using DecodingInput = runtime::DecodingInput;
    using DecodingOutput = runtime::DecodingOutput;
    using SpeculativeDecodingMode = runtime::SpeculativeDecodingMode;
    using GptDecoderBatched = runtime::GptDecoderBatched;

    CreateNewDecoderRequests() = default;

    void operator()(std::vector<SizeType32> const& seqSlots,
        std::vector<runtime::decoder_batch::Request> const& requests,
        std::vector<SamplingConfig> const& samplingConfigs, runtime::ModelConfig const& modelConfig,
        GptDecoderBatched& decoder, CudaStreamPtr runtimeStream, SizeType32 maxSequenceLength) const;

    //! @brief Initialize the decoder at `batchSlot` with a new `request`. Exposed only for static batching via
    //! GptDecoderBatched::newBatch()
    void newRequest(SizeType32 batchSlot, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, runtime::ModelConfig const& modelConfig, GptDecoderBatched& decoder,
        CudaStreamPtr runtimeStream, SizeType32 maxSequenceLength) const;

private:
    //! @brief Setups decoder internal tensors for new speculative decoding request
    void newRequestSpeculativeDecoding(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, runtime::ModelConfig const& modelConfig,
        DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStreamPtr runtimeStream,
        CudaStreamPtr decoderStream, SpeculativeDecodingMode const& speculativeDecodingMode,
        SizeType32 maxDecodingEngineTokens) const;

    //! @brief Setups decoder internal tensors for new request in Draft model Sps mode
    void newRequestDraftTokensExternal(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, DecodingInput& jointDecodingInput, CudaStreamPtr decoderStream) const;

    //! @brief Setups decoder internal tensors for new Medusa request
    void newRequestMedusa(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingInput& jointDecodingInput, CudaStreamPtr decoderStream, SizeType32 maxDecodingEngineTokens) const;

    //! @brief Setups decoder internal tensors for new Lookahead request
    void newRequestLookahead(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStreamPtr runtimeStream) const;

    //! @brief Setups decoder internal tensors for new Explicit draft tokens request
    void newRequestExplicitDraftTokens(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingOutput& jointDecodingOutput, CudaStreamPtr runtimeStream) const;

    //! @brief Setups decoder internal tensors for new Eagle request
    void newRequestEagle(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        runtime::ModelConfig const& modelConfig, DecodingOutput& jointDecodingOutput,
        CudaStreamPtr runtimeStream) const;
};

} // namespace tensorrt_llm::batch_manager
