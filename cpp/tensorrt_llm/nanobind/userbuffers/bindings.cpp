/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/kernels/userbuffers/userbuffersManager.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace tub = tensorrt_llm::runtime::ub;

namespace tensorrt_llm::kernels::userbuffers
{

void UserBufferBindings::initBindings(nb::module_& m)
{
    nb::class_<tub::UBBuffer>(m, "UBBuffer")
        .def_ro("size", &tub::UBBuffer::size)
        .def_prop_ro("addr", [](tub::UBBuffer& self) { return reinterpret_cast<intptr_t>(self.addr); })
        .def_ro("handle", &tub::UBBuffer::handle)
        .def("invalid", &tub::UBBuffer::invalid, nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "ub_initialize", [](int tp_size) { tub::ub_initialize(tp_size); }, nb::call_guard<nb::gil_scoped_release>());
    m.def("ub_is_initialized", &tub::ub_is_initialized, nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "ub_allocate", [](size_t bytes) { return tub::ub_allocate(bytes); }, nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "ub_deallocate", [](intptr_t addr) { return tub::ub_deallocate(reinterpret_cast<void*>(addr)); },
        nb::call_guard<nb::gil_scoped_release>());
    m.def("ub_get", &tub::ub_get, nb::call_guard<nb::gil_scoped_release>());
    m.def("ub_supported", &tub::ub_supported, nb::call_guard<nb::gil_scoped_release>());

    m.def("initialize_userbuffers_manager", &tub::initialize_userbuffers_manager,
        nb::call_guard<nb::gil_scoped_release>());
}
} // namespace tensorrt_llm::kernels::userbuffers
