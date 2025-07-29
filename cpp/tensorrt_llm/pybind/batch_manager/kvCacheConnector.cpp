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

#include "tensorrt_llm/pybind/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/runtime/torch.h"

#include <torch/extension.h>

namespace
{

using KvCacheConnectorScheduler = tensorrt_llm::batch_manager::kv_connector::KvCacheConnectorScheduler;
using KvCacheConnectorWorker = tensorrt_llm::batch_manager::kv_connector::KvCacheConnectorWorker;
using KvCacheConnectorManager = tensorrt_llm::batch_manager::kv_connector::KvCacheConnectorManager;

using NumNewMatchedTokens = std::tuple<SizeType32, bool>;

namespace tb = tensorrt_llm::batch_manager;

class PyKvCacheConnectorScheduler : public KvCacheConnectorScheduler, py::trampoline_self_life_support
{
public:
    using KvCacheConnectorScheduler::KvCacheConnectorScheduler;

    NumNewMatchedTokens getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        PYBIND11_OVERRIDE_PURE(
            NumNewMatchedTokens, KvCacheConnectorScheduler, getNumNewMatchedTokens, request, numComputedTokens);
    }

    void updateStateAfterAlloc() override
    {
        PYBIND11_OVERRIDE(void, KvCacheConnectorScheduler, updateStateAfterAlloc);
    }

    bool requestFinished(LlmRequest const& request) override
    {
        PYBIND11_OVERRIDE(bool, KvCacheConnectorScheduler, requestFinished, request);
    }
};

class PyKvCacheConnectorWorker : public KvCacheConnectorWorker, py::trampoline_self_life_support
{
public:
    using KvCacheConnectorWorker::KvCacheConnectorWorker;

    void registerKvCaches(kv_connector::KvCacheConnectorPoolsData const& kvCacheConnectorPoolsData) override
    {
        PYBIND11_OVERRIDE(void, KvCacheConnectorWorker, registerKvCaches, kvCacheConnectorPoolsData);
    }

    void startLoadKv() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnectorWorker, startLoadKv);
    }

    void waitForLayerLoad(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnectorWorker, waitForLayerLoad, layer_idx);
    }

    void saveKvLayer(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnectorWorker, saveKvLayer, layer_idx);
    }

    void waitForSave() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnectorWorker, waitForSave);
    }

    using FinishedReqs = std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>>;

    FinishedReqs getFinished(std::vector<RequestIdType> const& finishedReqIds) override
    {
        PYBIND11_OVERRIDE(FinishedReqs, KvCacheConnectorWorker, getFinished, finishedReqIds);
    }
};

class PyKvCacheConnectorManager : public KvCacheConnectorManager, py::trampoline_self_life_support
{
public:
    using KvCacheConnectorManager::KvCacheConnectorManager;

    SizeType32 getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        PYBIND11_OVERRIDE_PURE_NAME(SizeType32, KvCacheConnectorManager, "get_num_new_matched_tokens",
            getNumNewMatchedTokens, request, numComputedTokens);
    }
};

} // namespace

void tensorrt_llm::batch_manager::kv_cache_manager::KVCacheManagerConnectorBindings::initBindings(py::module_& m)
{
    py::class_<tb::kv_connector::KvCacheConnectorPoolData>(m, "KvCacheConnectorPoolData")
        .def_property_readonly("tensor",
            [](tb::kv_connector::KvCacheConnectorPoolData& self)
            {
                auto const& poolTensor = self.getPoolTensor();

                return tensorrt_llm::runtime::Torch::tensor(poolTensor);
            })
        .def_property_readonly("num_blocks", &tb::kv_connector::KvCacheConnectorPoolData::getNumBlocks);

    py::class_<tb::kv_connector::KvCacheConnectorPoolsData>(m, "KvCacheConnectorPoolsData")
        .def_property_readonly("pools", &tb::kv_connector::KvCacheConnectorPoolsData::getPoolsData)
        .def_property_readonly(
            "layer_to_pool_mapping", &tb::kv_connector::KvCacheConnectorPoolsData::getLayerToPoolMapping);

    py::class_<tb::kv_connector::KvCacheConnectorWorker, PyKvCacheConnectorWorker, py::smart_holder>(
        m, "KvCacheConnectorWorker")
        .def(py::init<>())
        .def(
            "register_kv_caches", &tb::kv_connector::KvCacheConnectorWorker::registerKvCaches, py::arg("kv_cache_data"))
        .def("start_load_kv", &tb::kv_connector::KvCacheConnectorWorker::startLoadKv)
        .def("wait_for_layer_load", &tb::kv_connector::KvCacheConnectorWorker::waitForLayerLoad, py::arg("layer_idx"))
        .def("save_kv_layer", &tb::kv_connector::KvCacheConnectorWorker::saveKvLayer, py::arg("layer_idx"))
        .def("wait_for_save", &tb::kv_connector::KvCacheConnectorWorker::waitForSave)
        .def("get_finished", &tb::kv_connector::KvCacheConnectorWorker::getFinished, py::arg("finished_req_ids"));

    py::class_<tb::kv_connector::KvCacheConnectorScheduler, PyKvCacheConnectorScheduler, py::smart_holder>(
        m, "KvCacheConnectorScheduler")
        .def(py::init<>())
        .def("get_num_new_matched_tokens", &tb::kv_connector::KvCacheConnectorScheduler::getNumNewMatchedTokens,
            py::arg("request"), py::arg("num_computed_tokens"))
        .def("update_state_after_alloc", &tb::kv_connector::KvCacheConnectorScheduler::updateStateAfterAlloc)
        .def("request_finished", &tb::kv_connector::KvCacheConnectorScheduler::requestFinished, py::arg("request"));

    py::class_<tb::kv_connector::KvCacheConnectorManager, PyKvCacheConnectorManager, py::smart_holder>(
        m, "KvCacheConnectorManager")
        .def(py::init<>())
        .def("get_num_new_matched_tokens", &tb::kv_connector::KvCacheConnectorManager::getNumNewMatchedTokens,
            py::arg("request"), py::arg("num_computed_tokens"));

    py::class_<tb::kv_connector::NewRequestData>(m, "NewRequestData")
        .def_readonly("request_id", &tb::kv_connector::NewRequestData::requestId)
        .def_readonly("new_tokens", &tb::kv_connector::NewRequestData::newTokens)
        .def_readonly("block_ids", &tb::kv_connector::NewRequestData::blockIds)
        .def_readonly("num_computed_tokens", &tb::kv_connector::NewRequestData::numComputedTokens);

    py::class_<tb::kv_connector::CachedRequestData>(m, "CachedRequestData")
        .def_readonly("request_id", &tb::kv_connector::CachedRequestData::requestId)
        .def_readonly("new_tokens", &tb::kv_connector::CachedRequestData::newTokens)
        .def_readonly("new_block_ids", &tb::kv_connector::CachedRequestData::newBlockIds)
        .def_readonly("num_computed_tokens", &tb::kv_connector::CachedRequestData::numComputedTokens);

    py::class_<tb::kv_connector::SchedulerOutput>(m, "SchedulerOutput")
        .def_readonly("new_requests", &tb::kv_connector::SchedulerOutput::newRequests)
        .def_readonly("cached_requests", &tb::kv_connector::SchedulerOutput::cachedRequests);
}
