/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{

class EncoderBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using ITensor = tensorrt_llm::runtime::ITensor;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using ModelConfig = runtime::ModelConfig;
    using WorldConfig = runtime::WorldConfig;
    using TllmRuntime = runtime::TllmRuntime;

    TensorPtr inputIds;
    TensorPtr positionIds = nullptr;
    TensorPtr tokenTypeIds = nullptr;

    TensorPtr inputLengths;   // [numEncoderRequests]
    TensorPtr maxInputLength; // [maxInputLengthInBatch]

    // intermediate states in pipeline parallelism
    TensorPtr hiddenStates; // [numTokens, hiddenSize]

    // features for multimodal encoders (audio, image, etc.)
    TensorPtr
        inputFeatures; // [totalNumOfFeatures, featureDim] if remove_padding else [batchSize, featureDim, featureLength]

    // language adapter routing information for encoders if language adapter is presented.
    TensorPtr languageAdapterRoutings; // [numTokens, numLanguages]

    // encoder output
    TensorPtr encoderOutput; // [numEncoderTokens, hiddenSize]

    // output buffer owned by llmRequest, such that it's per-request output buffer
    // encoderBuffers class can init and reshape each buffer, without maintaining a list/set of inflight buffers
    // TODO in progress: to support BS>1 encoder, need (1) internal scratch space tensors to save the contiguous
    // batched output (2) copy from CONTIGUOUS scratch tensor to individual request's DISCRETE output tensor after
    // execution To standardize the implementation, for both BS=1 and BS>1, we use internal buffer to store BS=1/BS>1
    // results, and copy to request's external buffers. For BS=1, this introduces a redundancy copy, but ok for now.

    EncoderBuffers() = default;
    EncoderBuffers(SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        TllmRuntime const& runtime);

    std::pair<EncoderBuffers::TensorMap const&, EncoderBuffers::TensorMap&> prepareIO(RequestVector const& requests,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig, TllmRuntime const& runtime);

    void rearrangeOutputs(RequestVector const& requests, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        TllmRuntime const& runtime);

    //! @brief set shape of individual request's encoder output (Ptuning embedding table if multimodal)
    void updateReqOutputShape(RequestVector const& requests, TllmRuntime const& runtime, WorldConfig const& worldConfig,
        ModelConfig const& modelConfig);

private:
    SizeType32 numRequests{};
    SizeType32 encoderInputLen{};
    SizeType32 encoderOutputLen{};
    SizeType32 maxInputLengthInBatch{}; // max input length in a batch

    // prefilled with deterministic values to avoid runtime creation
    std::vector<SizeType32> positionIdsReserved;
    std::vector<SizeType32> tokenTypeIdsReserved;

    // engine I/O
    TensorMap inputMap;
    TensorMap outputMap;

    void init(SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        TllmRuntime const& runtime);

    //! @brief pre-allocate max buffer sizes during init
    void initBufferSizes(SizeType32 maxBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        TllmRuntime const& runtime);

    //! @brief update actual buffer usage of requests during runtime
    void updateBufferSizes(RequestVector const& requests, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig, TllmRuntime const& runtime);

    void reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void setFromInputs(RequestVector const& requests, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        TllmRuntime const& runtime);

    void fillIOMaps(ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    // additional members that are Encoder-Decoder specific
private:
    TensorPtr encoderOutputReserved; // [1, hiddenSize], dummy tensor for gen phase
    TensorPtr crossKvCacheGen;       // [1]
    SizeType32 hiddenSize;           // full hidden size (after multiplying tensor parallelism)

public:
    void create(SizeType32 maxBatchSize, ModelConfig const& modelConfig, TllmRuntime const& runtime);

    SizeType32 getMaxInputLengthInBatch() const
    {
        return maxInputLengthInBatch;
    };

    void setMaxBufferSizes(SizeType32 maxBatchSize, runtime::ModelConfig const& modelConfig);

    void setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests);

    void reshape();

    void fill(
        RequestVector const& ctxRequests, RequestVector const& genRequests, runtime::BufferManager const& manager);

    void insertInputTensors(TensorMap& inputMap);
};

} // namespace tensorrt_llm::batch_manager
