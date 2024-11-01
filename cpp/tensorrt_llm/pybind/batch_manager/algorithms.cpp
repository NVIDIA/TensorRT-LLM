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

#include "algorithms.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/pybind/common/algorithmBindings.h"

namespace py = pybind11;

using namespace tensorrt_llm::batch_manager;
using namespace PybindUtils;

void tensorrt_llm::pybind::batch_manager::algorithms::initBindings(pybind11::module_& m)
{
    // Algorithms with custom bindings
    py::class_<CapacityScheduler>(m, CapacityScheduler::name)
        .def_static("make", &CapacityScheduler::make, py::arg("max_num_requests"), py::arg("kv_cache_manager"),
            py::arg("cross_kv_cache_manager"), py::arg("peft_cache_manager"), py::arg("capacity_scheduler_policy"),
            py::arg("many_micro_batches") = false,
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def(py::init())
        .def("__call__", &CapacityScheduler::operator())
        .def("name", [](CapacityScheduler const&) { return CapacityScheduler::name; });

    py::class_<MicroBatchScheduler>(m, MicroBatchScheduler::name)
        .def_static("make", &MicroBatchScheduler::make, py::arg("max_batch_size"),
            py::arg_v("max_num_tokens", std::nullopt, "None"), py::arg_v("ctx_chunk_config", std::nullopt, "None"),
            py::arg_v("max_context_length", std::nullopt, "None"),
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def(py::init())
        .def("__call__", &MicroBatchScheduler::operator())
        .def("name", [](MicroBatchScheduler const&) { return MicroBatchScheduler::name; });
}
