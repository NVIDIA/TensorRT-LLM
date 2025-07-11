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
#include "tensorrt_llm/executor/tensor.h"

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace tle = tensorrt_llm::executor;

namespace
{
tle::Tensor numpyToTensor(py::array const& array)
{
    auto npDtype = array.dtype();
    tle::DataType dtype;
    if (npDtype.is(py::dtype("float16")))
    {
        dtype = tle::DataType::kFP16;
    }
    else if (npDtype.is(py::dtype("float32")))
    {
        dtype = tle::DataType::kFP32;
    }
    else if (npDtype.is(py::dtype("int8")))
    {
        dtype = tle::DataType::kINT8;
    }
    else if (npDtype.is(py::dtype("int32")))
    {
        dtype = tle::DataType::kINT32;
    }
    else if (npDtype.is(py::dtype("int64")))
    {
        dtype = tle::DataType::kINT64;
    }
    else if (npDtype.attr("kind").cast<std::string>() == "V" && npDtype.attr("itemsize").cast<int>() == 1
        && npDtype.attr("metadata")["dtype"].cast<std::string>() == "float8")
    {
        dtype = tle::DataType::kFP8;
    }
    else if (npDtype.attr("kind").cast<std::string>() == "V" && npDtype.attr("itemsize").cast<int>() == 2
        && npDtype.attr("metadata")["dtype"].cast<std::string>() == "bfloat16")
    {
        dtype = tle::DataType::kBF16;
    }
    else
    {
        TLLM_THROW("Unsupported numpy dtype: " + npDtype.attr("name").cast<std::string>());
    }

    tle::Shape shape(array.shape(), array.ndim());

    return tle::Tensor::of(dtype, const_cast<void*>(array.data()), shape);
}
} // namespace

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

Executor::Executor(pybind11::buffer engineBuffer, std::string const& jsonConfigStr, tle::ModelType modelType,
    tle::ExecutorConfig const& executorConfig, std::optional<pybind11::dict> managedWeights)
{
    py::buffer_info info = engineBuffer.request();
    uint8_t const* data = reinterpret_cast<uint8_t const*>(info.ptr);
    size_t size = info.size;
    std::optional<std::map<std::string, tle::Tensor>> managedWeightsMap = std::nullopt;
    if (managedWeights.has_value() && !managedWeights.value().empty())
    {
        managedWeightsMap = std::map<std::string, tle::Tensor>();
        for (auto const& item : managedWeights.value())
        {
            std::string name = item.first.cast<std::string>();
            py::array array = item.second.cast<py::array>();
            managedWeightsMap->emplace(name, numpyToTensor(array));
        }
    }
    mExecutor = std::make_unique<tle::Executor>(
        tle::BufferView(data, size), jsonConfigStr, modelType, executorConfig, managedWeightsMap);
}

Executor::Executor(std::string const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
    std::string const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, tle::ModelType modelType,
    tle::ExecutorConfig const& executorConfig)
{
    uint8_t const* encoderData = reinterpret_cast<uint8_t const*>(encoderEngineBuffer.data());
    size_t encoderSize = encoderEngineBuffer.size();
    uint8_t const* decoderData = reinterpret_cast<uint8_t const*>(decoderEngineBuffer.data());
    size_t decoderSize = decoderEngineBuffer.size();
    mExecutor = std::make_unique<tle::Executor>(tle::BufferView(encoderData, encoderSize), encoderJsonConfigStr,
        tle::BufferView(decoderData, decoderSize), decoderJsonConfigStr, modelType, executorConfig);
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
        .def(py::init<py::buffer, std::string const&, tle::ModelType, tle::ExecutorConfig const&, py::dict>(),
            py::arg("engine_buffer"), py::arg("json_config_str"), py::arg("model_type"), py::arg("executor_config"),
            py::arg("managed_weights") = py::dict())
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
        .def("get_latest_debug_tensors", &Executor::getLatestDebugTensors)
        .def("can_enqueue_requests", &Executor::canEnqueueRequests)
        .def("get_kv_cache_event_manager", &Executor::getKVCacheEventManager);
}

} // namespace tensorrt_llm::pybind::executor
