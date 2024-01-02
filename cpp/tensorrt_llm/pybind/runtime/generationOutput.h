/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/runtime/generationOutput.h"

#include <ATen/ATen.h>
#include <optional>
#include <pybind11/pybind11.h>

namespace tensorrt_llm::pybind::runtime
{

class GenerationOutput : public tensorrt_llm::runtime::GenericGenerationOutput<std::optional<at::Tensor>>
{
public:
    using Base = tensorrt_llm::runtime::GenericGenerationOutput<std::optional<at::Tensor>>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationOutput(TensorPtr ids, TensorPtr lengths)
        : GenericGenerationOutput(std::move(ids), std::move(lengths))
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::runtime::GenerationOutput> toTrtLlm() const;
    static void initBindings(pybind11::module_& m);
};

} // namespace tensorrt_llm::pybind::runtime
