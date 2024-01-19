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
#include "namedTensor.h"

#include "tensorrt_llm/runtime/torch.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

NamedTensor::NamedTensor(const tb::NamedTensor& cppNamedTensor)
    : Base(runtime::Torch::tensor(cppNamedTensor.tensor), cppNamedTensor.name)
{
}

void NamedTensor::initBindings(py::module_& m)
{
    py::class_<NamedTensor>(m, "NamedTensor")
        .def(py::init<NamedTensor::TensorPtr, std::string>(), py::arg("tensor"), py::arg("name"))
        .def_readwrite("tensor", &NamedTensor::tensor)
        .def_readonly("name", &NamedTensor::name);
}

} // namespace tensorrt_llm::pybind::batch_manager
