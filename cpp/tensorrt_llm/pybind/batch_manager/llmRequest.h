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

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <ATen/ATen.h>
#include <ATen/ops/tensor.h>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>

namespace tensorrt_llm::pybind::batch_manager
{

class LlmRequest : public tensorrt_llm::batch_manager::GenericLlmRequest<at::Tensor>
{
public:
    using Base = GenericLlmRequest<at::Tensor>;
    using TensorPtr = Base::TensorPtr;
    using SizeType = Base::SizeType;
    using TokenIdType = Base::TokenIdType;
    using RequestIdType = Base::RequestIdType;
    using VecLogProbs = Base::VecLogProbs;
    using BeamTokens = Base::BeamTokens;
    using VecTokens = Base::VecTokens;

    LlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::vector<TokenIdType> inputTokens,
        runtime::SamplingConfig samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
        std::optional<SizeType> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType> promptVocabSize = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt, bool returnLogProbs = false,
        std::optional<VecTokens> draftTokens = std::nullopt, std::optional<TensorPtr> draftLogits = std::nullopt)
        : Base(requestId, maxNewTokens, std::make_shared<std::vector<TokenIdType>>(std::move(inputTokens)),
            samplingConfig, isStreaming, endId, padId, embeddingBias, badWordsList, stopWordsList, promptEmbeddingTable,
            promptVocabSize, loraWeights, loraConfig, returnLogProbs,
            draftTokens.has_value() ? std::make_shared<VecTokens>(std::move(draftTokens.value()))
                                    : std::make_shared<VecTokens>(),
            draftLogits)
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest> toTrtLlm() const;
    static void initBindings(pybind11::module_& m);
};

} // namespace tensorrt_llm::pybind::batch_manager
