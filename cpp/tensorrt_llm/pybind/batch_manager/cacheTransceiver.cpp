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
#include "tensorrt_llm/executor/executor.h"
#include <ATen/ATen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <torch/csrc/jit/python/pybind_utils.h>

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

    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override
    {
        PYBIND11_OVERLOAD_PURE(void, tb::BaseCacheTransceiver, checkContextTransferStatus, atLeastRequestNum);
    }

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override
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
    // To be removed, temporary for testing
    py::classh<tb::CacheTransceiverComm>(m, "CacheTransceiverComm")
        .def(py::init([](py::object pg_obj) {
                 auto pg = torch::jit::toCustomClass<c10d::ProcessGroup>(pg_obj);
                 return tb::CacheTransceiverComm(pg);
             }),
             py::arg("process_group"))
        .def("is_mpi", &tb::CacheTransceiverComm::isMpi)
        .def("get_rank", &tb::CacheTransceiverComm::getRank)
        .def("get_size", &tb::CacheTransceiverComm::getSize)
        .def("split", &tb::CacheTransceiverComm::split, py::arg("color"), py::arg("key"))
        // allgather for torch tensors
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, at::Tensor input, at::Tensor output) {
                c10d::AllgatherOptions options;
                return self.allgather<at::Tensor, at::Tensor>(input, output, options);
            },
            py::arg("input"), py::arg("output"))
        // allgather: scalar input -> tuple(ok, vector[int64] output)
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, int64_t input) {
                std::vector<int64_t> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return py::make_tuple(ok, out);
            },
            py::arg("input"))
        // allgather: scalar(float) input -> tuple(ok, vector[float] output)
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, double input) {
                std::vector<double> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather<double, std::vector<double>&>(input, std::ref(out), options);
                return py::make_tuple(ok, out);
            },
            py::arg("input"))
        // If advanced options are needed, expose via a separate helper later
        // allgatherv for variable-sized lists (int64)
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<int64_t> input, std::vector<int> const& sizes) {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<int64_t> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return py::make_tuple(ok, output);
            },
            py::arg("input"), py::arg("sizes"))
        // allgatherv: float64
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<double> input, std::vector<int> const& sizes) {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<double> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return py::make_tuple(ok, output);
            },
            py::arg("input"), py::arg("sizes"))
        // allgatherv: char
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<char> input, std::vector<int> const& sizes) {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<char> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return py::make_tuple(ok, output);
            },
            py::arg("input"), py::arg("sizes"))
        ;

    py::classh<tb::BaseCacheTransceiver, PyCacheTransceiver>(m, "BaseCacheTransceiver")
        .def("respond_and_send_async", &BaseCacheTransceiver::respondAndSendAsync)
        .def("request_and_receive_sync", &BaseCacheTransceiver::requestAndReceiveSync)
        .def("request_and_receive_async", &BaseCacheTransceiver::requestAndReceiveAsync)
        .def("check_context_transfer_status", &BaseCacheTransceiver::checkContextTransferStatus)
        .def("check_gen_transfer_status", &BaseCacheTransceiver::checkGenTransferStatus)
        .def("check_gen_transfer_complete", &BaseCacheTransceiver::checkGenTransferComplete);

    py::enum_<executor::kv_cache::CacheState::AttentionType>(m, "AttentionType")
        .value("DEFAULT", executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        .value("MLA", executor::kv_cache::CacheState::AttentionType::kMLA);

    py::classh<tb::CacheTransceiver, tb::BaseCacheTransceiver>(m, "CacheTransceiver")
        .def(py::init<tb::kv_cache_manager::BaseKVCacheManager*, std::vector<SizeType32>, SizeType32, SizeType32,
                 runtime::WorldConfig, nvinfer1::DataType, executor::kv_cache::CacheState::AttentionType,
                 std::optional<executor::CacheTransceiverConfig>>(),
            py::arg("cache_manager"), py::arg("num_kv_heads_per_layer"), py::arg("size_per_head"),
            py::arg("tokens_per_block"), py::arg("world_config"), py::arg("dtype"), py::arg("attention_type"),
            py::arg("cache_transceiver_config") = std::nullopt);

    py::class_<tb::kv_cache_manager::CacheTransBufferManager>(m, "CacheTransBufferManager")
        .def(py::init<tb::kv_cache_manager::BaseKVCacheManager*, std::optional<size_t>>(), py::arg("cache_manager"),
            py::arg("max_num_tokens") = std::nullopt)
        .def_static("pre_alloc_buffer_size", &tb::kv_cache_manager::CacheTransBufferManager::preAllocBufferSize,
            py::arg("cache_size_bytes_per_token_per_window"), py::arg("cache_transceiver_config") = py::none());
}
