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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/pybind/utils/pathCaster.h"

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <optional>

namespace texec = tensorrt_llm::executor;

namespace tensorrt_llm::pybind::batch_manager
{

GptManager::GptManager(std::filesystem::path const& trtEnginePath, tb::TrtGptModelType modelType,
    GetInferenceRequestsCallback const& getInferenceRequestsCb, SendResponseCallback const& sendResponseCb,
    tb::PollStopSignalCallback const& pollStopSignalCb,
    tb::ReturnBatchManagerStatsCallback const& returnBatchManagerStatsCb,
    tb::TrtGptModelOptionalParams const& optionalParams, std::optional<uint64_t> terminateReqId)
{
    mManager = std::make_unique<tb::GptManager>(trtEnginePath, modelType, callbackAdapter(getInferenceRequestsCb),
        callbackAdapter(sendResponseCb), pollStopSignalCb, returnBatchManagerStatsCb, optionalParams, terminateReqId);
}

py::object GptManager::enter()
{
    TLLM_CHECK(static_cast<bool>(mManager));
    return py::cast(this);
}

void GptManager::exit(py::handle type, py::handle value, py::handle traceback)
{
    shutdown();
}

void GptManager::shutdown()
{
    // NOTE: we must release the GIL here. GptManager has spawned a thread for the execution loop. That thread must be
    // able to do forward progress for the shutdown process to succeed. It takes the GIL during its callbacks, so
    // we release it now. Note that we shouldn't do anything related to python objects after that.
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    py::gil_scoped_release release;
    mManager->shutdown();
    mManager = nullptr;
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
}

tb::GetInferenceRequestsCallback callbackAdapter(GetInferenceRequestsCallback const& callback)
{
    return [callback](int32_t max_sequences)
    {
        std::list<InferenceRequest> pythonResults = callback(max_sequences);
        std::list<std::shared_ptr<tb::InferenceRequest>> cppResults{};

        for (auto const& ir : pythonResults)
        {
            cppResults.push_back(ir.toTrtLlm());
        }

        return cppResults;
    };
}

tb::SendResponseCallback callbackAdapter(SendResponseCallback const& callback)
{
    return [callback](uint64_t id, std::list<tb::NamedTensor> const& cppTensors, bool isOk, std::string const& errMsg)
    {
        std::list<NamedTensor> pythonList{};
        for (auto const& cppNamedTensor : cppTensors)
        {
            pythonList.emplace_back(cppNamedTensor);
        }
        callback(id, pythonList, isOk, errMsg);
    };
}

void GptManager::initBindings(py::module_& m)
{
    py::class_<GptManager>(m, "GptManager")
        .def(py::init(
                 [](std::filesystem::path const& trtEnginePath, tb::TrtGptModelType modelType,
                     GetInferenceRequestsCallback const& getInferenceRequestsCb,
                     SendResponseCallback const& sendResponseCb, tb::PollStopSignalCallback const& pollStopSignalCb,
                     tb::ReturnBatchManagerStatsCallback const& returnBatchManagerStatsCb,
                     tb::TrtGptModelOptionalParams const& optionalParams, std::optional<uint64_t> terminateReqId)
                 {
                     PyErr_WarnEx(
                         PyExc_DeprecationWarning, "GptManager is deprecated use the executor API instead.", 1);

                     return GptManager(trtEnginePath, modelType, getInferenceRequestsCb, sendResponseCb,
                         pollStopSignalCb, returnBatchManagerStatsCb, optionalParams, terminateReqId);
                 }),
            py::arg("trt_engine_path"), py::arg("model_type"), py::arg("get_inference_requests_cb"),
            py::arg("send_response_cb"), py::arg("poll_stop_signal_cb") = nullptr,
            py::arg("return_batch_manager_stats_cb") = nullptr,
            py::arg_v("optional_params", tb::TrtGptModelOptionalParams(), "TrtGptModelOptionalParams"),
            py::arg("terminate_req_id") = std::nullopt)

        .def("shutdown", &GptManager::shutdown)
        .def("__enter__", &GptManager::enter)
        .def("__exit__", &GptManager::exit);
}

} // namespace tensorrt_llm::pybind::batch_manager
