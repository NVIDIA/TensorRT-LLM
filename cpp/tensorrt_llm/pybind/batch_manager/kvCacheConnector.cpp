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

using KvCacheConnector = tensorrt_llm::batch_manager::kv_connector::KvCacheConnector;
namespace tb = tensorrt_llm::batch_manager;

class PyKvCacheConnector : public KvCacheConnector
{
public:
    using KvCacheConnector::KvCacheConnector;

    //
    // WORKER SIDE METHODS
    //

    void registerKvCaches() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, registerKvCaches);
    }

    void startLoadKv() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, startLoadKv);
    }

    void waitForLayerLoad(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, waitForLayerLoad, layer_idx);
    }

    void saveKvLayer(SizeType32 layer_idx) override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, saveKvLayer, layer_idx);
    }

    void waitForSave() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, waitForSave);
    }

    using FinishedReqs = std::tuple<std::vector<RequestIdType>, std::vector<RequestIdType>>;

    FinishedReqs getFinished(std::vector<RequestIdType> const& finishedReqIds) override
    {
        PYBIND11_OVERRIDE_PURE(FinishedReqs, KvCacheConnector, getFinished, finishedReqIds);
    }

    //
    // SCHEDULER SIDE METHODS
    //

    using NumNewMatchedTokens = std::tuple<SizeType32, bool>;

    NumNewMatchedTokens getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        PYBIND11_OVERRIDE_PURE(
            NumNewMatchedTokens, KvCacheConnector, getNumNewMatchedTokens, request, numComputedTokens);
    }

    void updateStateAfterAlloc() override
    {
        PYBIND11_OVERRIDE_PURE(void, KvCacheConnector, updateStateAfterAlloc);
    }

    bool requestFinished(LlmRequest const& request) override
    {
        PYBIND11_OVERRIDE_PURE(bool, KvCacheConnector, requestFinished, request);
    }
};

} // namespace

void tensorrt_llm::batch_manager::kv_cache_manager::KVCacheManagerConnectorBindings::initBindings(py::module_& m)
{
    py::enum_<tb::kv_connector::KvCacheConnectorRole>(m, "KvCacheConnectorRole")
        .value("Scheduler", tb::kv_connector::KvCacheConnectorRole::Scheduler)
        .value("Worker", tb::kv_connector::KvCacheConnectorRole::Worker);

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

    py::class_<tb::kv_connector::KvCacheConnector, PyKvCacheConnector, py::smart_holder>(m, "KvCacheConnector")
        .def(py::init<tb::kv_connector::KvCacheConnectorRole>(), py::arg("role"))
        .def("register_kv_caches", &tb::kv_connector::KvCacheConnector::registerKvCaches)
        .def("start_load_kv", &tb::kv_connector::KvCacheConnector::startLoadKv)
        .def("wait_for_layer_load", &tb::kv_connector::KvCacheConnector::waitForLayerLoad, py::arg("layer_idx"))
        .def("save_kv_layer", &tb::kv_connector::KvCacheConnector::saveKvLayer, py::arg("layer_idx"))
        .def("wait_for_save", &tb::kv_connector::KvCacheConnector::waitForSave)
        .def("get_finished", &tb::kv_connector::KvCacheConnector::getFinished, py::arg("finished_req_ids"))
        .def("get_num_new_matched_tokens", &tb::kv_connector::KvCacheConnector::getNumNewMatchedTokens,
            py::arg("request"), py::arg("num_computed_tokens"))
        .def("update_state_after_alloc", &tb::kv_connector::KvCacheConnector::updateStateAfterAlloc)
        .def("request_finished", &tb::kv_connector::KvCacheConnector::requestFinished, py::arg("request"))
        .def_property_readonly("role", &tb::kv_connector::KvCacheConnector::role);
}
