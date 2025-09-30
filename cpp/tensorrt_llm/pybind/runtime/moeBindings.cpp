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
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;
namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::pybind::runtime
{

void pyDoReplication(tk::MoeLoadBalanceMetaInfo const& metaInfo, std::vector<float>& expertLoadFactor,
    tr::MoePlacementCpuInfo* cpuPlacement)
{
    TLLM_CHECK_WITH_INFO(
        metaInfo.expertCount == expertLoadFactor.size(), "expert_count and expert_load_factor size mismatch");
    tr::doReplication(metaInfo, expertLoadFactor.data(), cpuPlacement);
};

void pyDoPlacement(tk::MoeLoadBalanceMetaInfo const& metaInfo, std::vector<float>& expertLoadFactor,
    tr::MoePlacementCpuInfo* cpuPlacement)
{
    TLLM_CHECK_WITH_INFO(
        metaInfo.expertCount == expertLoadFactor.size(), "expert_count and expert_load_factor size mismatch");
    tr::doPlacement(metaInfo, expertLoadFactor.data(), cpuPlacement);
};

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

    // Bind MoeLoadBalanceMetaInfo struct
    py::class_<tk::MoeLoadBalanceMetaInfo>(m, "MoeLoadBalanceMetaInfo")
        .def(py::init<int, int, int, int, int>(), py::arg("expert_count"), py::arg("top_k"), py::arg("ep_rank"),
            py::arg("ep_size"), py::arg("slot_count_per_rank"))
        .def_readwrite("expert_count", &tk::MoeLoadBalanceMetaInfo::expertCount)
        .def_readwrite("top_k", &tk::MoeLoadBalanceMetaInfo::topK)
        .def_readwrite("ep_rank", &tk::MoeLoadBalanceMetaInfo::epRank)
        .def_readwrite("ep_size", &tk::MoeLoadBalanceMetaInfo::epSize)
        .def_readwrite("slot_count_per_rank", &tk::MoeLoadBalanceMetaInfo::slotCountPerRank);

    // Bind MoePlacementCpuInfo struct
    py::class_<tr::MoePlacementCpuInfo>(m, "MoePlacementCpuInfo")
        .def(py::init<>())
        .def_readwrite("expert_replica_count", &tr::MoePlacementCpuInfo::expertReplicaCount)
        .def_readwrite("rank_expert_ids", &tr::MoePlacementCpuInfo::rankExpertIds);

    // Bind SingleLayerMoeLoadBalancer class
    py::class_<tr::SingleLayerMoeLoadBalancer, std::shared_ptr<tr::SingleLayerMoeLoadBalancer>>(
        m, "SingleLayerMoeLoadBalancer")
        .def("add_single_weight_slot", &tr::SingleLayerMoeLoadBalancer::addSingleWeightSlot, py::arg("slot_id"),
            py::arg("name"), py::arg("weight_slot"), "Add a single weight slot for a specific slot ID",
            py::call_guard<py::gil_scoped_release>())
        .def("add_single_host_weight", &tr::SingleLayerMoeLoadBalancer::addSingleHostWeight, py::arg("expert_id"),
            py::arg("name"), py::arg("host_weight"), "Add a single host weight for a specific expert ID",
            py::call_guard<py::gil_scoped_release>())
        .def("set_initial_weight_assignments", &tr::SingleLayerMoeLoadBalancer::setInitialWeightAssignments,
            py::arg("initial_weight_assignments"), "Set initial weight assignments for each slot",
            py::call_guard<py::gil_scoped_release>())
        .def("get_pointer", &tr::SingleLayerMoeLoadBalancer::getSelfPtr,
            "Get the pointer of the SingleLayerMoeLoadBalancer", py::call_guard<py::gil_scoped_release>())
        .def("get_layer_id", &tr::SingleLayerMoeLoadBalancer::getLayerId,
            "Get the layer id of the SingleLayerMoeLoadBalancer", py::call_guard<py::gil_scoped_release>());

    // Bind MoeLoadBalancer class
    py::class_<tr::MoeLoadBalancer>(m, "MoeLoadBalancer")
        .def(py::init<int, int, int>(), py::arg("ep_rank"), py::arg("ep_size"), py::arg("layer_updates_per_iter"),
            "Initialize the MoeLoadBalancer with the specified expert parallel rank, size, and update frequency",
            py::call_guard<py::gil_scoped_release>())
        .def("set_use_gpu_memcpy", &tr::MoeLoadBalancer::setUseGpuMemcpy, py::arg("use_gpu_memcpy"),
            "Set whether to use GPU memcpy for weight updates", py::call_guard<py::gil_scoped_release>())
        .def("add_layer", &tr::MoeLoadBalancer::AddLayer, py::arg("expert_count"), py::arg("top_k"),
            py::arg("slot_count_per_rank"), "Add a new MOE layer to the load balancer",
            py::call_guard<py::gil_scoped_release>())
        .def("finalize_model", &tr::MoeLoadBalancer::finalizeModel,
            "Finalize the model structure, must be called after all layers are added",
            py::call_guard<py::gil_scoped_release>())
        .def("set_warm_up_iter_count", &tr::MoeLoadBalancer::setWarmUpIterCount, py::arg("iter_count"),
            "Set the number of warm-up iterations", py::call_guard<py::gil_scoped_release>())
        .def("start_iter", &tr::MoeLoadBalancer::startIter, py::arg("iter_id"), py::arg("enable_statistic"),
            py::arg("enable_update_weights"), "Start a new iteration with the given ID and settings",
            py::call_guard<py::gil_scoped_release>())
        .def("end_iter", &tr::MoeLoadBalancer::endIter, py::arg("iter_id"), "End the iteration with the given ID",
            py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &tr::MoeLoadBalancer::shutdown, "Shutdown the load balancer and clean up resources",
            py::call_guard<py::gil_scoped_release>());

    m.def("is_host_accessible_device_memory_supported", &tr::HostAccessibleDeviceAllocator::isSupported,
        "If current system support host accessible device memory");

    // Bind do_replication function for testing
    m.def("do_replication", &pyDoReplication, py::arg("meta_info"), py::arg("expert_load_factor"),
        py::arg("cpu_placement"), "Do replication");

    // Bind do_placement function for testing
    m.def("do_placement", &pyDoPlacement, py::arg("meta_info"), py::arg("expert_load_factor"), py::arg("cpu_placement"),
        "Do placement");
}

} // namespace tensorrt_llm::pybind::runtime
