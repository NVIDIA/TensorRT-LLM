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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/common/algorithm.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/request.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{
class RuntimeBuffers;
class DecoderInputBuffers;

namespace tr = tensorrt_llm::runtime;

class GenerateRequestOptions : Algorithm
{
public:
    constexpr static auto name{"GenerateRequestOptions"};

    using SizeType32 = tr::SizeType32;
    using ITensor = tr::ITensor;
    using TensorPtr = tr::ITensor::SharedPtr;
    using BufferManager = tr::BufferManager;
    template <typename T>
    using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

    GenerateRequestOptions(bool speculativeDecodingFastLogits, bool isLeaderInOrchMode, bool isNormalizeLogProbs)
        : mSpeculativeDecodingFastLogits(speculativeDecodingFastLogits)
        , mIsLeaderInOrchMode(isLeaderInOrchMode)
        , mIsNormalizeLogProbs(isNormalizeLogProbs)
    {
    }

    /**
     * @brief Generate decoding requests for the given context requests.
     *        The logic of this function is replicated in `generate_request_options.py`.
     *
     * @param modelConfig The model configuration.
     * @param worldConfig The world configuration.
     * @param decodingConfig The decoding configuration.
     * @param contextRequests The context requests.
     * @param bufferManager The buffer manager.
     * @param logitsType The logits type.
     * @param inputBuffers The input buffers.
     * @param buffers The runtime buffers.
     * @return A tuple containing the logits, requests, and sampling configurations.
     */
    std::tuple<ITensor::SharedPtr, std::vector<tr::decoder_batch::Request>, std::vector<tr::SamplingConfig>> operator()(
        tr::ModelConfig const& modelConfig, tr::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests,
        BufferManager const& bufferManager, nvinfer1::DataType logitsType, DecoderInputBuffers& inputBuffers,
        OptionalRef<RuntimeBuffers const> buffers = std::nullopt) const;

private:
    [[nodiscard]] std::shared_ptr<runtime::ITensor> retrieveDraftLogits(tr::ModelConfig const& modelConfig,
        tr::WorldConfig const& worldConfig, std::shared_ptr<runtime::ITensor> const& tensor,
        BufferManager const& bufferManager) const;

    /// @brief Retrieve the embedding bias from the request. This potentially makes a copy of the tensor
    /// to the appropriate type if the input tensor does not match it.
    [[nodiscard]] TensorPtr getEmbeddingBias(nvinfer1::DataType logitsType, TensorPtr const& tensor) const;

    bool mSpeculativeDecodingFastLogits;
    bool mIsLeaderInOrchMode;
    bool mIsNormalizeLogProbs;
};

} // namespace tensorrt_llm::batch_manager
