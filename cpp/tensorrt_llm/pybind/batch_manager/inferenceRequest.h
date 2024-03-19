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

#include "tensorrt_llm/batch_manager/inferenceRequest.h"
#include "tensorrt_llm/pybind/batch_manager/llmRequest.h"
#include "tensorrt_llm/pybind/batch_manager/namedTensor.h"

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <list>
#include <memory>
#include <optional>

namespace tensorrt_llm::pybind::batch_manager
{

class InferenceRequest
    : public tensorrt_llm::batch_manager::GenericInferenceRequest<at::Tensor, NamedTensor, c10::Stream>
{
public:
    using Base = tensorrt_llm::batch_manager::GenericInferenceRequest<at::Tensor, NamedTensor, c10::Stream>;
    using TensorPtr = Base::TensorPtr;
    using TensorMap = Base::TensorMap;
    using LogitsProcessorCallback = Base::LogitsPostProcessor;

    InferenceRequest(uint64_t requestId, std::optional<LogitsProcessorCallback> logitsCb = std::nullopt)
        : Base(requestId, logitsCb)
    {
    }

    InferenceRequest(uint64_t requestId, TensorMap const& inputTensors,
        std::optional<LogitsProcessorCallback> logitsCb = std::nullopt)
        : Base{requestId, inputTensors, logitsCb}
    {
    }

    InferenceRequest(
        uint64_t requestId, TensorMap&& inputTensors, std::optional<LogitsProcessorCallback> logitsCb = std::nullopt)
        : Base{requestId, std::move(inputTensors), logitsCb}
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::batch_manager::InferenceRequest> toTrtLlm() const;

    std::string serialize() const;

    static std::shared_ptr<InferenceRequest> deserialize(std::string const& serialized);

    static void initBindings(pybind11::module_& m);
};

} // namespace tensorrt_llm::pybind::batch_manager
