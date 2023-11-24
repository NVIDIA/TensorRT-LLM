/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/common/assert.h"

#include <ATen/ATen.h>

#include <ATen/ops/tensor.h>
#include <memory>
#include <optional>

namespace tensorrt_llm::pybind::batch_manager
{

class InferenceRequest : public tensorrt_llm::batch_manager::GenericInferenceRequest<at::Tensor,
                             std::unordered_map<std::string, at::Tensor>>
{
public:
    using Base
        = tensorrt_llm::batch_manager::GenericInferenceRequest<at::Tensor, std::unordered_map<std::string, at::Tensor>>;
    using TensorPtr = Base::TensorPtr;
    using TensorMap = Base::TensorMap;

    InferenceRequest(uint64_t requestId)
        : Base(requestId)
    {
    }

    InferenceRequest(TensorMap const& inputTensors, uint64_t requestId)
        : Base(inputTensors, requestId)
    {
    }

    InferenceRequest(TensorMap&& inputTensors, uint64_t requestId)
        : Base(inputTensors, requestId)
    {
    }

    [[nodiscard]] std::shared_ptr<tensorrt_llm::batch_manager::InferenceRequest> toTrtLlm() const;
};

} // namespace tensorrt_llm::pybind::batch_manager
