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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

namespace nb = nanobind;
namespace tle = tensorrt_llm::executor;

namespace nanobind::detail
{ // todo

template <>
struct dtype_traits<half>
{
    static constexpr dlpack::dtype value{
        (uint8_t) dlpack::dtype_code::Float, // type code
        16,                                  // size in bits
        1                                    // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("float16");
};
} // namespace nanobind::detail

namespace
{
// todo: test this
tle::Tensor numpyToTensor(nb::ndarray<nb::numpy> const& array)
{
    auto npDtype = array.dtype();
    char kind = '\0';
    switch (npDtype.code)
    {
    case static_cast<uint8_t>(nb::dlpack::dtype_code::Int):
        kind = 'i'; // signed integer
        break;
    case static_cast<uint8_t>(nb::dlpack::dtype_code::UInt):
        kind = 'u'; // unsigned integer
        break;
    case static_cast<uint8_t>(nb::dlpack::dtype_code::Float):
        kind = 'f'; // floating point
        break;
    case static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat):
        kind = 'f'; // brain floating point (treat as float kind)
        break;
    case static_cast<uint8_t>(nb::dlpack::dtype_code::Complex):
        kind = 'c'; // complex
        break;
    default:
        kind = 'V'; // void/other
        break;
    }
    tle::DataType dtype;
    if (npDtype == nb::dtype<half>())
    {
        dtype = tle::DataType::kFP16;
    }
    else if (npDtype == nb::dtype<float>())
    {
        dtype = tle::DataType::kFP32;
    }
    else if (npDtype == nb::dtype<int8_t>())
    {
        dtype = tle::DataType::kINT8;
    }
    else if (npDtype == nb::dtype<int32_t>())
    {
        dtype = tle::DataType::kINT32;
    }
    else if (npDtype == nb::dtype<int64_t>())
    {
        dtype = tle::DataType::kINT64;
    }
    // kind == Void
    else if (kind == 'V' && array.itemsize() == 1)
    {
        dtype = tle::DataType::kFP8;
    }
    else if (kind == 'V' && array.itemsize() == 2)
    {
        dtype = tle::DataType::kBF16;
    }
    else
    {
        TLLM_THROW("Unsupported numpy dtype.");
    }

    std::vector<int64_t> dims;
    for (size_t i = 0; i < array.ndim(); ++i)
    {
        dims.push_back(static_cast<int64_t>(array.shape(i)));
    }
    tle::Shape shape(dims.data(), dims.size());

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

Executor::Executor(nb::ndarray<nb::numpy, uint8_t> const& engineBuffer, std::string const& jsonConfigStr,
    tle::ModelType modelType, tle::ExecutorConfig const& executorConfig, std::optional<nb::dict> managedWeights)
{
    uint8_t const* data = static_cast<uint8_t const*>(engineBuffer.data());
    size_t size = engineBuffer.size();
    std::optional<std::map<std::string, tle::Tensor>> managedWeightsMap = std::nullopt;
    if (managedWeights.has_value() && !managedWeights.value().empty())
    {
        managedWeightsMap = std::map<std::string, tle::Tensor>();
        for (auto const& item : managedWeights.value())
        {
            std::string name = nb::cast<std::string>(item.first);
            nb::ndarray<nb::numpy> array = nb::cast<nb::ndarray<nb::numpy>>(item.second);
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

nb::object Executor::enter()
{
    TLLM_CHECK(static_cast<bool>(mExecutor));
    return nb::cast(this);
}

void Executor::exit(
    [[maybe_unused]] nb::handle type, [[maybe_unused]] nb::handle value, [[maybe_unused]] nb::handle traceback)
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
    nb::gil_scoped_release release;
    mExecutor->shutdown();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void Executor::initBindings(nb::module_& m)
{
    nb::class_<Executor>(m, "Executor")
        .def(nb::init<std::filesystem::path const&, tle::ModelType, tle::ExecutorConfig const&>(),
            nb::arg("model_path"), nb::arg("model_type"), nb::arg("executor_config"))
        .def(nb::init<std::filesystem::path const&, std::filesystem::path const&, tle::ModelType,
                 tle::ExecutorConfig const&>(),
            nb::arg("encoder_model_path"), nb::arg("decoder_model_path"), nb::arg("model_type"),
            nb::arg("executor_config"))
        .def(nb::init<nb::ndarray<nb::numpy, uint8_t>, std::string const&, tle::ModelType, tle::ExecutorConfig const&,
                 nb::dict>(),
            nb::arg("engine_buffer"), nb::arg("json_config_str"), nb::arg("model_type"), nb::arg("executor_config"),
            nb::arg("managed_weights") = nb::dict())
        .def(nb::init<std::string const&, std::string const&, std::string const&, std::string const&, tle::ModelType,
                 tle::ExecutorConfig const&>(),
            nb::arg("encoder_engine_buffer"), nb::arg("encoder_json_config_str"), nb::arg("decoder_engine_buffer"),
            nb::arg("decoder_json_config_str"), nb::arg("model_type"), nb::arg("executor_config"))
        .def("shutdown", &Executor::shutdown)
        .def("__enter__", &Executor::enter)
        .def("__exit__", &Executor::exit)
        .def("enqueue_request", &Executor::enqueueRequest, nb::arg("request"))
        .def("enqueue_requests", &Executor::enqueueRequests, nb::arg("requests"))
        .def("await_responses",
            nb::overload_cast<std::optional<std::chrono::milliseconds> const&>(&Executor::awaitResponses),
            nb::arg("timeout") = nb::none())
        .def("await_responses",
            nb::overload_cast<tle::IdType const&, std::optional<std::chrono::milliseconds> const&>(
                &Executor::awaitResponses),
            nb::arg("id"), nb::arg("timeout") = nb::none())
        .def("await_responses",
            nb::overload_cast<std::vector<tle::IdType> const&, std::optional<std::chrono::milliseconds> const&>(
                &Executor::awaitResponses),
            nb::arg("ids"), nb::arg("timeout") = nb::none())
        .def("get_num_responses_ready", &Executor::getNumResponsesReady, nb::arg("id") = nb::none())
        .def("cancel_request", &Executor::cancelRequest, nb::arg("id") = nb::none())
        .def("get_latest_iteration_stats", &Executor::getLatestIterationStats)
        .def("get_latest_request_stats", &Executor::getLatestRequestStats)
        .def("get_latest_debug_tensors", &Executor::getLatestDebugTensors)
        .def("can_enqueue_requests", &Executor::canEnqueueRequests)
        .def("get_kv_cache_event_manager", &Executor::getKVCacheEventManager);
}

} // namespace tensorrt_llm::pybind::executor
