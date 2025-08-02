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

namespace
{
using KvCacheConnectorManager = tensorrt_llm::batch_manager::kv_connector::KvCacheConnectorManager;

namespace tb = tensorrt_llm::batch_manager;

class PyKvCacheConnectorManager : public KvCacheConnectorManager, py::trampoline_self_life_support
{
public:
    using KvCacheConnectorManager::KvCacheConnectorManager;

    SizeType32 getNumNewMatchedTokens(tb::LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        PYBIND11_OVERRIDE_PURE_NAME(SizeType32, KvCacheConnectorManager, "get_num_new_matched_tokens",
            getNumNewMatchedTokens, request, numComputedTokens);
    }
};

} // namespace

void tensorrt_llm::batch_manager::kv_cache_manager::KVCacheManagerConnectorBindings::initBindings(py::module_& m)
{
    py::class_<tb::kv_connector::KvCacheConnectorManager, PyKvCacheConnectorManager, py::smart_holder>(
        m, "KvCacheConnectorManager")
        .def(py::init<>())
        .def("get_num_new_matched_tokens", &tb::kv_connector::KvCacheConnectorManager::getNumNewMatchedTokens,
            py::arg("request"), py::arg("num_computed_tokens"));
}
