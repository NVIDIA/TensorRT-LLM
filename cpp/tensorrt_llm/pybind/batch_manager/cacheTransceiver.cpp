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

#include "cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tb = tensorrt_llm::batch_manager;

namespace
{

class PyCacheTransceiver : public tb::BaseCacheTransceiver
{
public:
    // using BaseCacheTransceiver::BaseCacheTransceiver; // Inherit constructors

    void respondAndSendAsync(tb::LlmRequest* llmRequest) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, respondAndSendAsync, llmRequest);
    }

    void requestAndReceiveSync(tb::LlmRequest* llmRequest) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, requestAndReceiveSync, llmRequest);
    }

    void requestAndReceiveAsync(tb::LlmRequest* llmRequest) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, requestAndReceiveAsync, llmRequest);
    }

    void checkContextTransferStatus(bool blocking = false) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, checkContextTransferStatus, blocking);
    }

    void checkGenTransferStatus(int atLeastRequestNum = 0) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, checkGenTransferStatus, atLeastRequestNum);
    }

    bool checkGenTransferComplete() const override
    {
        PYBIND11_OVERLOAD_PURE(bool, tb::BaseCacheTransceiver, checkGenTransferComplete);
    }
};
} // namespace

void tb::CacheTransceiverBindings::initBindings(py::module_& m)
{
    py::classh<tb::BaseCacheTransceiver, PyCacheTransceiver>(m, "BaseCacheTransceiver")
        .def("respond_and_send_async", &BaseCacheTransceiver::respondAndSendAsync)
        .def("request_and_receive_sync", &BaseCacheTransceiver::requestAndReceiveSync)
        .def("request_and_receive_async", &BaseCacheTransceiver::requestAndReceiveAsync)
        .def("check_context_transfer_status", &BaseCacheTransceiver::checkContextTransferStatus)
        .def("check_gen_transfer_status", &BaseCacheTransceiver::checkGenTransferStatus)
        .def("check_gen_transfer_complete", &BaseCacheTransceiver::checkGenTransferComplete);

    py::enum_<tb::CacheTransceiver::CommType>(m, "CommType")
        .value("UNKNOWN", tb::CacheTransceiver::CommType::UNKNOWN)
        .value("MPI", tb::CacheTransceiver::CommType::MPI)
        .value("UCX", tb::CacheTransceiver::CommType::UCX);

    py::enum_<executor::kv_cache::CacheState::AttentionType>(m, "AttentionType")
        .value("DEFAULT", executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        .value("MLA", executor::kv_cache::CacheState::AttentionType::kMLA);

    py::classh<tb::CacheTransceiver, tb::BaseCacheTransceiver>(m, "CacheTransceiver")
        .def(py::init<tb::kv_cache_manager::BaseKVCacheManager*, tb::CacheTransceiver::CommType,
                 std::vector<SizeType32>, SizeType32, SizeType32, runtime::WorldConfig, nvinfer1::DataType,
                 executor::kv_cache::CacheState::AttentionType>(),
            py::arg("cache_manager"), py::arg("comm_type"), py::arg("num_kv_heads_per_layer"), py::arg("size_per_head"),
            py::arg("tokens_per_block"), py::arg("world_config"), py::arg("dtype"), py::arg("attention_type"));
}
