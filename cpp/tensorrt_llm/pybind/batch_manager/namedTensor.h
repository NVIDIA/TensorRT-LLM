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

#include "tensorrt_llm/batch_manager/namedTensor.h"

#include <ATen/ATen.h>

#include <optional>
#include <pybind11/pybind11.h>

namespace tensorrt_llm::pybind::batch_manager
{

class NamedTensor : public tensorrt_llm::batch_manager::GenericNamedTensor<std::optional<at::Tensor>>
{
public:
    using Base = tensorrt_llm::batch_manager::GenericNamedTensor<std::optional<at::Tensor>>;
    using TensorPtr = Base::TensorPtr;

    NamedTensor(TensorPtr _tensor, std::string _name)
        : Base(std::move(_tensor), std::move(_name)){};

    explicit NamedTensor(std::string _name)
        : Base(std::move(_name)){};

    explicit NamedTensor(const tensorrt_llm::batch_manager::NamedTensor& cppNamedTensor);
    static void initBindings(pybind11::module_& m);
};

} // namespace tensorrt_llm::pybind::batch_manager
