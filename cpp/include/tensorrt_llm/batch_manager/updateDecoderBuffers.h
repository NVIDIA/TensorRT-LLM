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
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/modelConfig.h"

namespace tensorrt_llm::runtime
{
class BufferManager;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

class DecoderBuffers;
class DecoderOutputBuffers;

class UpdateDecoderBuffers : Algorithm
{
public:
    constexpr static auto name{"UpdateDecoderBuffers"};

    UpdateDecoderBuffers() = default;

    runtime::CudaEvent operator()(runtime::ModelConfig const& modelConfig, DecoderBuffers& decoderBuffers,
        DecoderOutputBuffers& decoderOutputBuffers, runtime::BufferManager const& copyBufferManager,
        runtime::GptDecoderBatched const& decoder, bool returnLogProbs,
        runtime::CudaEvent const& decoderFinishEvent) const;
};

} // namespace tensorrt_llm::batch_manager
