/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/moeLoadBalancer.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <vector>

namespace nb = nanobind;
namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::nanobind::runtime
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

void initMoeBindings(nb::module_& m)
{
    // Bind MoeWeight struct
    nb::class_<tr::MoeWeight>(m, "MoeWeight")
        .def(nb::init<>())
        .def_prop_rw("weight_ptr", &tr::MoeWeight::getWeightPtr, &tr::MoeWeight::setWeightPtr)
        .def_rw("height", &tr::MoeWeight::mHeight)
        .def_rw("width", &tr::MoeWeight::mWidth)
        .def_rw("pitch", &tr::MoeWeight::mPitch)
        .def("__repr__",
            [](tr::MoeWeight const& self)
            {
                return "<MoeWeight ptr=" + std::to_string(self.getWeightPtr())
                    + " height=" + std::to_string(self.mHeight) + " width=" + std::to_string(self.mWidth)
                    + " pitch=" + std::to_string(self.mPitch) + ">";
            });

    // Bind MoeLoadBalanceMetaInfo struct
    nb::class_<tk::MoeLoadBalanceMetaInfo>(m, "MoeLoadBalanceMetaInfo")
        .def(nb::init<int, int, int, int, int>(), nb::arg("expert_count"), nb::arg("top_k"), nb::arg("ep_rank"),
            nb::arg("ep_size"), nb::arg("slot_count_per_rank"))
        .def_rw("expert_count", &tk::MoeLoadBalanceMetaInfo::expertCount)
        .def_rw("top_k", &tk::MoeLoadBalanceMetaInfo::topK)
        .def_rw("ep_rank", &tk::MoeLoadBalanceMetaInfo::epRank)
        .def_rw("ep_size", &tk::MoeLoadBalanceMetaInfo::epSize)
        .def_rw("slot_count_per_rank", &tk::MoeLoadBalanceMetaInfo::slotCountPerRank);

    // Bind MoePlacementCpuInfo struct
    nb::class_<tr::MoePlacementCpuInfo>(m, "MoePlacementCpuInfo")
        .def(nb::init<>())
        .def_rw("expert_replica_count", &tr::MoePlacementCpuInfo::expertReplicaCount)
        .def_rw("rank_expert_ids", &tr::MoePlacementCpuInfo::rankExpertIds);

    // Bind SingleLayerMoeLoadBalancer class
    nb::class_<tr::SingleLayerMoeLoadBalancer>(m, "SingleLayerMoeLoadBalancer")
        .def("add_single_weight_slot", &tr::SingleLayerMoeLoadBalancer::addSingleWeightSlot, nb::arg("slot_id"),
            nb::arg("name"), nb::arg("weight_slot"), "Add a single weight slot for a specific slot ID",
            nb::call_guard<nb::gil_scoped_release>())
        .def("add_single_host_weight", &tr::SingleLayerMoeLoadBalancer::addSingleHostWeight, nb::arg("expert_id"),
            nb::arg("name"), nb::arg("host_weight"), "Add a single host weight for a specific expert ID",
            nb::call_guard<nb::gil_scoped_release>())
        .def("set_initial_weight_assignments", &tr::SingleLayerMoeLoadBalancer::setInitialWeightAssignments,
            nb::arg("initial_weight_assignments"), "Set initial weight assignments for each slot",
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_pointer", &tr::SingleLayerMoeLoadBalancer::getSelfPtr,
            "Get the pointer of the SingleLayerMoeLoadBalancer", nb::call_guard<nb::gil_scoped_release>())
        .def("get_layer_id", &tr::SingleLayerMoeLoadBalancer::getLayerId,
            "Get the layer id of the SingleLayerMoeLoadBalancer", nb::call_guard<nb::gil_scoped_release>());

    // Bind MoeLoadBalancer class
    nb::class_<tr::MoeLoadBalancer>(m, "MoeLoadBalancer")
        .def(nb::init<int, int, int>(), nb::arg("ep_rank"), nb::arg("ep_size"), nb::arg("layer_updates_per_iter"),
            "Initialize the MoeLoadBalancer with the specified expert parallel rank, size, and update frequency",
            nb::call_guard<nb::gil_scoped_release>())
        .def("set_use_gpu_memcpy", &tr::MoeLoadBalancer::setUseGpuMemcpy, nb::arg("use_gpu_memcpy"),
            "Set whether to use GPU memcpy for weight updates", nb::call_guard<nb::gil_scoped_release>())
        .def("add_layer", &tr::MoeLoadBalancer::AddLayer, nb::arg("expert_count"), nb::arg("top_k"),
            nb::arg("slot_count_per_rank"), "Add a new MOE layer to the load balancer",
            nb::call_guard<nb::gil_scoped_release>())
        .def("finalize_model", &tr::MoeLoadBalancer::finalizeModel,
            "Finalize the model structure, must be called after all layers are added",
            nb::call_guard<nb::gil_scoped_release>())
        .def("set_warm_up_iter_count", &tr::MoeLoadBalancer::setWarmUpIterCount, nb::arg("iter_count"),
            "Set the number of warm-up iterations", nb::call_guard<nb::gil_scoped_release>())
        .def("start_iter", &tr::MoeLoadBalancer::startIter, nb::arg("iter_id"), nb::arg("enable_statistic"),
            nb::arg("enable_update_weights"), "Start a new iteration with the given ID and settings",
            nb::call_guard<nb::gil_scoped_release>())
        .def("end_iter", &tr::MoeLoadBalancer::endIter, nb::arg("iter_id"), "End the iteration with the given ID",
            nb::call_guard<nb::gil_scoped_release>())
        .def("shutdown", &tr::MoeLoadBalancer::shutdown, "Shutdown the load balancer and clean up resources",
            nb::call_guard<nb::gil_scoped_release>());

    m.def("is_host_accessible_device_memory_supported", &tr::HostAccessibleDeviceAllocator::isSupported,
        "If current system support host accessible device memory");

    // Bind do_replication function for testing
    m.def("do_replication", &pyDoReplication, nb::arg("meta_info"), nb::arg("expert_load_factor"),
        nb::arg("cpu_placement"), "Do replication");

    // Bind do_placement function for testing
    m.def("do_placement", &pyDoPlacement, nb::arg("meta_info"), nb::arg("expert_load_factor"), nb::arg("cpu_placement"),
        "Do placement");
}

} // namespace tensorrt_llm::nanobind::runtime
