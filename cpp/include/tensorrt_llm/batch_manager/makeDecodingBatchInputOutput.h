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
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/modelConfig.h"

namespace tensorrt_llm::runtime::decoder
{
class DecoderState;
} // namespace tensorrt_llm::runtime::decoder

namespace tensorrt_llm::batch_manager
{
class DecoderInputBuffers;
class RuntimeBuffers;

class MakeDecodingBatchInputOutput : Algorithm
{
public:
    constexpr static auto name{"MakeDecodingBatchInputOutput"};

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorPtr = runtime::decoder_batch::Input::TensorPtr;
    template <typename T>
    using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

    MakeDecodingBatchInputOutput() = default;

    std::unique_ptr<runtime::decoder_batch::Input> operator()(DecoderInputBuffers& inputBuffers,
        runtime::decoder::DecoderState& decoderState, runtime::ModelConfig const& modelConfig,
        SizeType32 maxNumSequences, OptionalRef<RuntimeBuffers> fusedRuntimeBuffers) const;

    [[nodiscard]] static std::unique_ptr<runtime::decoder_batch::Input> createDecoderBatchInputs(
        std::vector<SizeType32> const& activeSlots, runtime::decoder::DecoderState const& decoderState,
        std::vector<TensorPtr> const& logits, SizeType32 maxNumSequences, std::vector<TensorPtr> const& batchSlots);
};

} // namespace tensorrt_llm::batch_manager
