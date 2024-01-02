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

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/generationInput.h"

#include <ATen/ATen.h>
#include <ATen/ops/tensor.h>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>

namespace tensorrt_llm::pybind::runtime
{

using SizeType = tensorrt_llm::runtime::SizeType;

class PromptTuningParams : public tensorrt_llm::runtime::GenericPromptTuningParams<std::optional<at::Tensor>>
{
public:
    using Base = tensorrt_llm::runtime::GenericPromptTuningParams<std::optional<at::Tensor>>;
    using TensorPtr = Base::TensorPtr;
    using SizeType = Base::SizeType;

    explicit PromptTuningParams(
        TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
        : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::runtime::PromptTuningParams> toTrtLlm() const;
    static void initBindings(pybind11::module_& m);
};

class GenerationInput
    : public tensorrt_llm::runtime::GenericGenerationInput<std::optional<at::Tensor>, PromptTuningParams>
{
public:
    using Base = tensorrt_llm::runtime::GenericGenerationInput<std::optional<at::Tensor>, PromptTuningParams>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationInput(
        SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : GenericGenerationInput(endId, padId, std::move(ids), std::move(lengths), packed)
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::runtime::GenerationInput> toTrtLlm() const;
    static void initBindings(pybind11::module_& m);
};
} // namespace tensorrt_llm::pybind::runtime
