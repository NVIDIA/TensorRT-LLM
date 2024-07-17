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

#include "executor.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/pybind/utils/pathCaster.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::pybind::executor
{

Executor::Executor(
    std::filesystem::path const& modelPath, tle::ModelType modelType, tle::ExecutorConfig const& executorConfig)
{
    mExecutor = std::make_unique<tle::Executor>(modelPath, modelType, executorConfig);
}

Executor::Executor(std::filesystem::path const& encoderModelPath, std::filesystem::path const& decoderModelPath,
    tle::ModelType modelType, tle::ExecutorConfig const& executorConfig)
{
    mExecutor = std::make_unique<tle::Executor>(encoderModelPath, decoderModelPath, modelType, executorConfig);
}

Executor::Executor(std::string const& engineBuffer, std::string const& jsonConfigStr, tle::ModelType modelType,
    tle::ExecutorConfig const& executorConfig)
{
    mExecutor = std::make_unique<tle::Executor>(
        std::vector<uint8_t>(engineBuffer.begin(), engineBuffer.end()), jsonConfigStr, modelType, executorConfig);
}

Executor::Executor(std::string const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
    std::string const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, tle::ModelType modelType,
    tle::ExecutorConfig const& executorConfig)
{
    mExecutor
        = std::make_unique<tle::Executor>(std::vector<uint8_t>(encoderEngineBuffer.begin(), encoderEngineBuffer.end()),
            encoderJsonConfigStr, std::vector<uint8_t>(decoderEngineBuffer.begin(), decoderEngineBuffer.end()),
            decoderJsonConfigStr, modelType, executorConfig);
}

py::object Executor::enter()
{
    TLLM_CHECK(static_cast<bool>(mExecutor));
    return py::cast(this);
}

void Executor::exit(
    [[maybe_unused]] py::handle type, [[maybe_unused]] py::handle value, [[maybe_unused]] py::handle traceback)
{
    shutdown();
    mExecutor = nullptr;
}

void Executor::shutdown()
{
    // NOTE: we must release the GIL here. Executor has spawned a thread for the execution loop. That thread must be
    // able to do forward progress for the shutdown process to succeed. It takes the GIL during its callbacks, so
    // we release it now. Note that we shouldn't do anything related to python objects after that.
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    py::gil_scoped_release release;
    mExecutor->shutdown();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void Executor::initBindings(py::module_& m)
{
    py::class_<Executor>(m, "Executor")
        .def(py::init<std::filesystem::path const&, tle::ModelType, tle::ExecutorConfig const&>(),
            py::arg("model_path"), py::arg("model_type"), py::arg("executor_config"))
        .def(py::init<std::filesystem::path const&, std::filesystem::path const&, tle::ModelType,
                 tle::ExecutorConfig const&>(),
            py::arg("encoder_model_path"), py::arg("decoder_model_path"), py::arg("model_type"),
            py::arg("executor_config"))
        .def(py::init<std::string const&, std::string const&, tle::ModelType, tle::ExecutorConfig const&>(),
            py::arg("engine_buffer"), py::arg("json_config_str"), py::arg("model_type"), py::arg("executor_config"))
        .def(py::init<std::string const&, std::string const&, std::string const&, std::string const&, tle::ModelType,
                 tle::ExecutorConfig const&>(),
            py::arg("encoder_engine_buffer"), py::arg("encoder_json_config_str"), py::arg("decoder_engine_buffer"),
            py::arg("decoder_json_config_str"), py::arg("model_type"), py::arg("executor_config"))
        .def("shutdown", &Executor::shutdown)
        .def("__enter__", &Executor::enter)
        .def("__exit__", &Executor::exit)
        .def("enqueue_request", &Executor::enqueueRequest, py::arg("request"))
        .def("enqueue_requests", &Executor::enqueueRequests, py::arg("requests"))
        .def("await_responses",
            py::overload_cast<std::optional<std::chrono::milliseconds> const&>(&Executor::awaitResponses),
            py::arg("timeout") = py::none())
        .def("await_responses",
            py::overload_cast<tle::IdType const&, std::optional<std::chrono::milliseconds> const&>(
                &Executor::awaitResponses),
            py::arg("id"), py::arg("timeout") = py::none())
        .def("await_responses",
            py::overload_cast<std::vector<tle::IdType> const&, std::optional<std::chrono::milliseconds> const&>(
                &Executor::awaitResponses),
            py::arg("ids"), py::arg("timeout") = py::none())
        .def("get_num_responses_ready", &Executor::getNumResponsesReady, py::arg("id") = py::none())
        .def("cancel_request", &Executor::cancelRequest, py::arg("id") = py::none())
        .def("get_latest_iteration_stats", &Executor::getLatestIterationStats)
        .def("get_latest_request_stats", &Executor::getLatestRequestStats)
        .def("can_enqueue_requests", &Executor::canEnqueueRequests);
}

} // namespace tensorrt_llm::pybind::executor
