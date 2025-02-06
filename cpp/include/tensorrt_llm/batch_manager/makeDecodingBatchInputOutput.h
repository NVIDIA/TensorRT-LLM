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
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"

namespace tensorrt_llm::batch_manager
{
class DecoderBuffers;
class RuntimeBuffers;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager
{

namespace tr = tensorrt_llm::runtime;

class MakeDecodingBatchInputOutput : Algorithm
{
public:
    constexpr static auto name{"MakeDecodingBatchInputOutput"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    MakeDecodingBatchInputOutput() = default;

    std::tuple<std::unique_ptr<tr::decoder_batch::Input>, std::unique_ptr<tr::decoder_batch::Output>> operator()(
        RequestVector const& contextRequests, RequestVector const& generationRequests, DecoderBuffers& decoderBuffers,
        RuntimeBuffers const& genRuntimeBuffers, executor::DecodingMode const& decodingMode,
        runtime::ModelConfig const& modelConfig, SizeType32 maxNumSequences) const;

private:
    std::vector<bool> computeActiveVec(RequestVector const& contextRequests, RequestVector const& generationRequests,
        SizeType32 maxNumSequences) const;
};

} // namespace tensorrt_llm::batch_manager
