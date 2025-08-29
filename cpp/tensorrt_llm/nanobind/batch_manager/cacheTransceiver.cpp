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

#include "cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tb = tensorrt_llm::batch_manager;
namespace nb = nanobind;

namespace
{

class PyCacheTransceiver : public tb::BaseCacheTransceiver
{
public:
    // using BaseCacheTransceiver::BaseCacheTransceiver; // Inherit constructors
    NB_TRAMPOLINE(tb::BaseCacheTransceiver, 6);

    void respondAndSendAsync(tb::LlmRequest* llmRequest) override
    {
        NB_OVERRIDE_PURE(respondAndSendAsync, llmRequest);
    }

    void requestAndReceiveSync(tb::LlmRequest* llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveSync, llmRequest);
    }

    void requestAndReceiveAsync(tb::LlmRequest* llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveAsync, llmRequest);
    }

    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override
    {
        NB_OVERRIDE_PURE(checkContextTransferStatus, atLeastRequestNum);
    }

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override
    {
        NB_OVERRIDE_PURE(checkGenTransferStatus, atLeastRequestNum);
    }

    bool checkGenTransferComplete() const override
    {
        NB_OVERRIDE_PURE(checkGenTransferComplete);
    }
};
} // namespace

void tb::CacheTransceiverBindings::initBindings(nb::module_& m)
{
    nb::class_<tb::BaseCacheTransceiver, PyCacheTransceiver>(m, "BaseCacheTransceiver")
        .def("respond_and_send_async", &BaseCacheTransceiver::respondAndSendAsync,
            nb::call_guard<nb::gil_scoped_release>())
        .def("request_and_receive_sync", &BaseCacheTransceiver::requestAndReceiveSync,
            nb::call_guard<nb::gil_scoped_release>())
        .def("request_and_receive_async", &BaseCacheTransceiver::requestAndReceiveAsync,
            nb::call_guard<nb::gil_scoped_release>())
        .def("check_context_transfer_status", &BaseCacheTransceiver::checkContextTransferStatus,
            nb::call_guard<nb::gil_scoped_release>())
        .def("check_gen_transfer_status", &BaseCacheTransceiver::checkGenTransferStatus,
            nb::call_guard<nb::gil_scoped_release>())
        .def("check_gen_transfer_complete", &BaseCacheTransceiver::checkGenTransferComplete,
            nb::call_guard<nb::gil_scoped_release>());

    nb::enum_<executor::kv_cache::CacheState::AttentionType>(m, "AttentionType")
        .value("DEFAULT", executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        .value("MLA", executor::kv_cache::CacheState::AttentionType::kMLA);

    nb::class_<tb::CacheTransceiver, tb::BaseCacheTransceiver>(m, "CacheTransceiver")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::vector<SizeType32>, SizeType32, SizeType32,
                 runtime::WorldConfig, nvinfer1::DataType, executor::kv_cache::CacheState::AttentionType,
                 std::optional<executor::CacheTransceiverConfig>>(),
            nb::arg("cache_manager"), nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"),
            nb::arg("tokens_per_block"), nb::arg("world_config"), nb::arg("dtype"), nb::arg("attention_type"),
            nb::arg("cache_transceiver_config") = std::nullopt, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tb::kv_cache_manager::CacheTransBufferManager>(m, "CacheTransBufferManager")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::optional<size_t>>(), nb::arg("cache_manager"),
            nb::arg("max_num_tokens") = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
        .def_static("pre_alloc_buffer_size", &tb::kv_cache_manager::CacheTransBufferManager::preAllocBufferSize,
            nb::arg("cache_size_bytes_per_token_per_window"), nb::arg("cache_transceiver_config") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>());
}
