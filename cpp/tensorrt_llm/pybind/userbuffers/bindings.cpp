/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bindings.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"

namespace py = pybind11;
namespace tub = tensorrt_llm::runtime::ub;

namespace tensorrt_llm::kernels::userbuffers
{

void UserBufferBindings::initBindings(pybind11::module_& m)
{
    py::class_<tub::UBBuffer>(m, "UBBuffer")
        .def_readonly("size", &tub::UBBuffer::size)
        .def_property_readonly("addr", [](tub::UBBuffer& self) { return reinterpret_cast<intptr_t>(self.addr); })
        .def_readonly("handle", &tub::UBBuffer::handle)
        .def("invalid", &tub::UBBuffer::invalid);

    m.def("ub_initialize", [](int tp_size) { tub::ub_initialize(tp_size); });
    m.def("ub_is_initialized", &tub::ub_is_initialized);
    m.def(
        "ub_allocate", [](int idx, size_t bytes) { return reinterpret_cast<intptr_t>(tub::ub_allocate(idx, bytes)); });
    m.def("ub_deallocate", [](intptr_t addr) { return tub::ub_deallocate(reinterpret_cast<void*>(addr)); });
    m.def("ub_get", &tub::ub_get);
    m.def("ub_supported", &tub::ub_supported);
}
} // namespace tensorrt_llm::kernels::userbuffers
