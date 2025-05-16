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

#include "moeBindings.h"
#include "tensorrt_llm/runtime/moeLoadBalancer.h"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::pybind::runtime
{

void initMoeBindings(pybind11::module_& m)
{
    // Bind MoeWeight struct
    py::class_<tr::MoeWeight>(m, "MoeWeight")
        .def(py::init<>())
        .def_property("weight_ptr", &tr::MoeWeight::getWeightPtr, &tr::MoeWeight::setWeightPtr)
        .def_readwrite("height", &tr::MoeWeight::mHeight)
        .def_readwrite("width", &tr::MoeWeight::mWidth)
        .def_readwrite("pitch", &tr::MoeWeight::mPitch)
        .def("__repr__",
            [](tr::MoeWeight const& self)
            {
                return "<MoeWeight ptr=" + std::to_string(self.getWeightPtr())
                    + " height=" + std::to_string(self.mHeight) + " width=" + std::to_string(self.mWidth)
                    + " pitch=" + std::to_string(self.mPitch) + ">";
            });

    // Bind SingleLayerMoeLoadBalancer class
    py::class_<tr::SingleLayerMoeLoadBalancer, std::shared_ptr<tr::SingleLayerMoeLoadBalancer>>(
        m, "SingleLayerMoeLoadBalancer")
        .def("add_single_weight_slot", &tr::SingleLayerMoeLoadBalancer::addSingleWeightSlot, py::arg("slot_id"),
            py::arg("name"), py::arg("weight_slot"), "Add a single weight slot for a specific slot ID")
        .def("add_single_host_weight", &tr::SingleLayerMoeLoadBalancer::addSingleHostWeight, py::arg("expert_id"),
            py::arg("name"), py::arg("host_weight"), "Add a single host weight for a specific expert ID")
        .def("set_initial_weight_assignments", &tr::SingleLayerMoeLoadBalancer::setInitialWeightAssignments,
            py::arg("initial_weight_assignments"), "Set initial weight assignments for each slot")
        .def("get_pointer", &tr::SingleLayerMoeLoadBalancer::getSelfPtr,
            "Get the pointer of the SingleLayerMoeLoadBalancer")
        .def("get_layer_id", &tr::SingleLayerMoeLoadBalancer::getLayerId,
            "Get the layer id of the SingleLayerMoeLoadBalancer");

    // Bind MoeLoadBalancer class
    py::class_<tr::MoeLoadBalancer>(m, "MoeLoadBalancer")
        .def(py::init<int, int, int>(), py::arg("ep_rank"), py::arg("ep_size"), py::arg("layer_updates_per_iter"),
            "Initialize the MoeLoadBalancer with the specified expert parallel rank, size, and update frequency")
        .def("add_layer", &tr::MoeLoadBalancer::AddLayer, py::arg("expert_count"), py::arg("top_k"),
            py::arg("slot_count_per_rank"), "Add a new MOE layer to the load balancer")
        .def("finalize_model", &tr::MoeLoadBalancer::finalizeModel,
            "Finalize the model structure, must be called after all layers are added")
        .def("set_warm_up_iter_count", &tr::MoeLoadBalancer::setWarmUpIterCount, py::arg("iter_count"),
            "Set the number of warm-up iterations")
        .def("start_iter", &tr::MoeLoadBalancer::startIter, py::arg("iter_id"), py::arg("enable_statistic"),
            py::arg("enable_update_weights"), "Start a new iteration with the given ID and settings")
        .def("end_iter", &tr::MoeLoadBalancer::endIter, py::arg("iter_id"), "End the iteration with the given ID")
        .def("shutdown", &tr::MoeLoadBalancer::shutdown, "Shutdown the load balancer and clean up resources");
}

} // namespace tensorrt_llm::pybind::runtime
