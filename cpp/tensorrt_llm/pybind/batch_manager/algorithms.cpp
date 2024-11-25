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
#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace tensorrt_llm::batch_manager;

void tensorrt_llm::pybind::batch_manager::algorithms::initBindings(pybind11::module_& m)
{
    py::class_<CapacityScheduler>(m, CapacityScheduler::name)
        .def(py::init<SizeType32, executor::CapacitySchedulerPolicy, bool, bool, LlmRequestState, LlmRequestState>(),
            py::arg("max_num_requests"), py::arg("capacity_scheduler_policy"), py::arg("has_kv_cache_manager"),
            py::arg("many_micro_batches") = false,
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def("__call__", &CapacityScheduler::operator(), py::arg("active_requests"),
            py::arg("kv_cache_manager") = nullptr, py::arg("peft_cache_manager") = nullptr,
            py::arg("cross_kv_cache_manager") = nullptr)
        .def("name", [](CapacityScheduler const&) { return CapacityScheduler::name; });

    py::class_<MicroBatchScheduler>(m, MicroBatchScheduler::name)
        .def(py::init<std::optional<batch_scheduler::ContextChunkingConfig>, std::optional<SizeType32>, LlmRequestState,
                 LlmRequestState>(),
            py::arg("ctx_chunk_config") = std::nullopt, py::arg("max_context_length") = std::nullopt,
            py::arg_v("no_schedule_until_state", LlmRequestState::kCONTEXT_INIT, "LlmRequestState.CONTEXT_INIT"),
            py::arg_v("no_schedule_after_state", LlmRequestState::kGENERATION_COMPLETE,
                "LlmRequestState.GENERATION_COMPLETE"))
        .def("__call__", &MicroBatchScheduler::operator(), py::arg("active_requests"), py::arg("inflight_req_ids"),
            py::arg("max_batch_size_runtime"), py::arg("max_num_tokens_runtime"))
        .def("name", [](MicroBatchScheduler const&) { return MicroBatchScheduler::name; });

    py::class_<PauseRequests>(m, PauseRequests::name)
        .def(py::init<SizeType32>(), py::arg("max_input_len"))
        .def("__call__", &PauseRequests::operator(), py::arg("requests_to_pause"), py::arg("inflight_req_ids"),
            py::arg("req_ids_to_pause"), py::arg("pause_flagged"), py::arg("seq_slot_manager"),
            py::arg("kv_cache_manager") = std::nullopt, py::arg("cross_kv_cache_manager") = std::nullopt,
            py::arg("peft_cache_manager") = std::nullopt)
        .def("name", [](PauseRequests const&) { return PauseRequests::name; });

    py::class_<AssignReqSeqSlots>(m, AssignReqSeqSlots::name)
        .def(py::init())
        .def("__call__", &AssignReqSeqSlots::operator(), py::arg("seq_slot_manager"), py::arg("context_requests"),
            py::arg("generation_requests"))
        .def("name", [](AssignReqSeqSlots const&) { return AssignReqSeqSlots::name; });

    py::class_<AllocateKvCache>(m, AllocateKvCache::name)
        .def(py::init())
        .def("__call__", &AllocateKvCache::operator(), py::arg("kv_cache_manager"), py::arg("context_requests"),
            py::arg("generation_requests"), py::arg("model_config"), py::arg("cross_kv_cache_manager") = std::nullopt)
        .def("name", [](AllocateKvCache const&) { return AllocateKvCache::name; });
}
