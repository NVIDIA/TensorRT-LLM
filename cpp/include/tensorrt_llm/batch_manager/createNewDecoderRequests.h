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

#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/request.h"

namespace tensorrt_llm::runtime
{
class DecodingInput;
class DecodingOutput;
class GptDecoderBatched;
class SamplingConfig;
class SpeculativeDecodingMode;

namespace decoder
{
class DecoderState;
} // namespace decoder

} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

class CreateNewDecoderRequests : Algorithm
{
public:
    constexpr static auto name{"CreateNewDecoderRequests"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SamplingConfig = tensorrt_llm::runtime::SamplingConfig;
    using CudaStream = tensorrt_llm::runtime::CudaStream;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SharedConstPtr = runtime::ITensor::SharedConstPtr;
    using DecodingInput = runtime::DecodingInput;
    using DecodingOutput = runtime::DecodingOutput;
    using SpeculativeDecodingMode = runtime::SpeculativeDecodingMode;
    using GptDecoderBatched = runtime::GptDecoderBatched;

    CreateNewDecoderRequests() = default;

    void operator()(TensorPtr const& batchSlots, std::vector<runtime::decoder_batch::Request> const& requests,
        std::vector<SamplingConfig> const& samplingConfigs, runtime::ModelConfig const& modelConfig,
        GptDecoderBatched& decoder, CudaStream const& runtimeStream, SizeType32 maxSequenceLength) const;

    //! @brief Initialize the decoder at `batchSlot` with a new `request`. Exposed only for static batching via
    //! GptDecoderBatched::newBatch()
    static void newRequest(SizeType32 batchSlot, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, runtime::ModelConfig const& modelConfig,
        runtime::decoder::DecoderState& decoderState, CudaStream const& runtimeStream, CudaStream const& decoderStream,
        SizeType32 maxSequenceLength);

private:
    //! @brief Setups decoder internal tensors for new speculative decoding request
    static void newRequestSpeculativeDecoding(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, runtime::ModelConfig const& modelConfig,
        DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream,
        CudaStream const& decoderStream, SpeculativeDecodingMode const& speculativeDecodingMode,
        SizeType32 maxDecodingEngineTokens);

    //! @brief Setups decoder internal tensors for new request in Draft model Sps mode
    static void newRequestDraftTokensExternal(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        SamplingConfig const& samplingConfig, DecodingInput& jointDecodingInput, CudaStream const& decoderStream);

    //! @brief Setups decoder internal tensors for new Medusa request
    static void newRequestMedusa(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingInput& jointDecodingInput, CudaStream const& decoderStream, SizeType32 maxDecodingEngineTokens);

    //! @brief Setups decoder internal tensors for new Lookahead request
    static void newRequestLookahead(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream);

    //! @brief Setups decoder internal tensors for new Explicit draft tokens request
    static void newRequestExplicitDraftTokens(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream);

    //! @brief Setups decoder internal tensors for new Eagle request
    static void newRequestEagle(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
        runtime::ModelConfig const& modelConfig, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream);
};

} // namespace tensorrt_llm::batch_manager
