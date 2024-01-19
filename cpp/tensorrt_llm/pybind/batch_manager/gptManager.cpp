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
#include "gptManager.h"
#include "inferenceRequest.h"
#include "namedTensor.h"
#include "tensorrt_llm/batch_manager/GptManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/pybind/utils/pathCaster.h"

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <ATen/ATen.h>

#include <ATen/ops/tensor.h>
#include <memory>
#include <optional>

namespace tb = tensorrt_llm::batch_manager;

namespace tensorrt_llm::pybind::batch_manager
{

GptManager::GptManager(std::filesystem::path const& trtEnginePath, tb::TrtGptModelType modelType, int32_t maxBeamWidth,
    tb::batch_scheduler::SchedulerPolicy schedulerPolicy, GetInferenceRequestsCallback getInferenceRequestsCb,
    SendResponseCallback sendResponseCb, tb::PollStopSignalCallback pollStopSignalCb,
    tb::ReturnBatchManagerStatsCallback returnBatchManagerStatsCb, const tb::TrtGptModelOptionalParams& optionalParams,
    std::optional<uint64_t> terminateReqId)
    : tb::GptManager(trtEnginePath, modelType, maxBeamWidth, schedulerPolicy, callbackAdapter(getInferenceRequestsCb),
        callbackAdapter(sendResponseCb), pollStopSignalCb, returnBatchManagerStatsCb, optionalParams, terminateReqId)
{
}

py::object GptManager::enter()
{
    return py::cast(this);
}

void GptManager::exit(py::handle type, py::handle value, py::handle traceback)
{
    // NOTE: we must release the GIL here. GptManager has spawned a thread for the execution loop. That thread must be
    // able to do forward progress for the shutdown process to succeed. It takes the GIL during its callbacks, so
    // we release it now. Note that we shouldn't do anything related to python objects after that.
    py::gil_scoped_release release;
    shutdown();
}

tb::GetInferenceRequestsCallback callbackAdapter(GetInferenceRequestsCallback callback)
{
    return [callback](int32_t max_sequences)
    {
        std::list<InferenceRequest> pythonResults = callback(max_sequences);
        std::list<std::shared_ptr<tb::InferenceRequest>> cppResults{};

        for (const auto& ir : pythonResults)
        {
            cppResults.push_back(ir.toTrtLlm());
        }

        return cppResults;
    };
}

tb::SendResponseCallback callbackAdapter(SendResponseCallback callback)
{
    return [callback](uint64_t id, std::list<tb::NamedTensor> const& cppTensors, bool isOk, const std::string& errMsg)
    {
        std::list<NamedTensor> pythonList{};
        for (const auto& cppNamedTensor : cppTensors)
        {
            pythonList.push_back(NamedTensor{cppNamedTensor});
        }
        callback(id, pythonList, isOk, errMsg);
    };
}

void GptManager::initBindings(py::module_& m)
{
    py::class_<GptManager>(m, "GptManager")
        .def(py::init<std::filesystem::path const&, tb::TrtGptModelType, int32_t, tb::batch_scheduler::SchedulerPolicy,
                 GetInferenceRequestsCallback, SendResponseCallback, tb::PollStopSignalCallback,
                 tb::ReturnBatchManagerStatsCallback, const tb::TrtGptModelOptionalParams&, std::optional<uint64_t>>(),
            py::arg("trt_engine_path"), py::arg("model_type"), py::arg("max_beam_width"), py::arg("scheduler_policy"),
            py::arg("get_inference_requests_cb"), py::arg("send_response_cb"), py::arg("poll_stop_signal_cb") = nullptr,
            py::arg("return_batch_manager_stats_cb") = nullptr,
            py::arg_v("optional_params", tb::TrtGptModelOptionalParams(), "TrtGptModelOptionalParams"),
            py::arg("terminate_req_id") = std::nullopt)

        // Note: attempting to bind &GptManager::shutdown() will result in a compiler error:
        //
        //  pybind11.h:1482:56: error: static assertion failed: Cannot bind an inaccessible base class method; use a
        //  lambda definition instead 1482 |         detail::is_accessible_base_of<Class, Derived>::value,
        //      |                                                        ^~~~~
        //
        // The issue is that the parent class has no bindings and is therefore not visible to pybind11.
        // To resolve, we can add something like:
        //
        //  py::class_<tensorrt_llm::batch_manager::GptManager>(m, "_GptManagerBase");
        .def("shutdown", [](GptManager& self) { self.shutdown(); })
        .def("__enter__", &GptManager::enter)
        .def("__exit__", &GptManager::exit);
}

} // namespace tensorrt_llm::pybind::batch_manager
