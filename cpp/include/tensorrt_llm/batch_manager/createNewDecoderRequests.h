/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class SamplingConfig;

namespace decoder
{
class DecoderState;
} // namespace decoder

} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{
class MedusaBuffers;
class DecoderInputBuffers;

class CreateNewDecoderRequests : Algorithm
{
public:
    constexpr static auto name{"CreateNewDecoderRequests"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SamplingConfig = tensorrt_llm::runtime::SamplingConfig;
    using CudaStream = tensorrt_llm::runtime::CudaStream;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SharedConstPtr = runtime::ITensor::SharedConstPtr;
    template <typename T>
    using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

    CreateNewDecoderRequests(bool speculativeDecodingFastLogits, bool isLeaderInOrchMode, bool isNormalizeLogProbs)
        : mSpeculativeDecodingFastLogits(speculativeDecodingFastLogits)
        , mIsLeaderInOrchMode(isLeaderInOrchMode)
        , mIsNormalizeLogProbs(isNormalizeLogProbs)
    {
    }

    [[nodiscard]] std::tuple<TensorPtr, std::vector<SamplingConfig>, std::vector<SharedConstPtr>,
        std::vector<executor::LookaheadDecodingConfig>>
    operator()(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests,
        nvinfer1::DataType logitsType, DecoderInputBuffers& inputBuffers, runtime::decoder::DecoderState& decoderState,
        CudaStream const& runtimeStream, CudaStream const& decoderStream, SizeType32 maxSequenceLength,
        SizeType32 beamWidth, OptionalRef<MedusaBuffers const> medusaBuffers) const;

    [[nodiscard]] std::tuple<std::vector<SharedConstPtr>, std::vector<executor::LookaheadDecodingConfig>>
    createDecoderRequests(RequestVector const& finishedContextRequests, TensorPtr const& inputIds,
        executor::DecodingConfig const& decodingConfig, runtime::decoder::DecoderState& decoderState,
        nvinfer1::DataType logitsType, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        runtime::CudaStream const& runtimeStream, runtime::CudaStream const& decoderStream,
        SizeType32 maxSequenceLength, OptionalRef<MedusaBuffers const> medusaBuffers) const;

private:
    bool mSpeculativeDecodingFastLogits;
    bool mIsLeaderInOrchMode;
    bool mIsNormalizeLogProbs;
};

} // namespace tensorrt_llm::batch_manager
