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

#include "tensorrt_llm/nanobind/batch_manager/kvCacheConnector.h"

#include <nanobind/trampoline.h>
#include <torch/extension.h>

namespace
{
using KvCacheConnectorManager = tensorrt_llm::batch_manager::kv_connector::KvCacheConnectorManager;

namespace tb = tensorrt_llm::batch_manager;

class PyKvCacheConnectorManager : KvCacheConnectorManager
{
public:
    NB_TRAMPOLINE(KvCacheConnectorManager, 1);

    SizeType32 getNumNewMatchedTokens(tb::LlmRequest const& request, SizeType32 numComputedTokens) override
    {
        NB_OVERRIDE_PURE_NAME("get_num_new_matched_tokens", getNumNewMatchedTokens, request, numComputedTokens);
    }
};

} // namespace

void tensorrt_llm::batch_manager::kv_cache_manager::KVCacheManagerConnectorBindings::initBindings(nb::module_& m)
{
    nb::class_<tb::kv_connector::KvCacheConnectorManager, PyKvCacheConnectorManager>(m, "KvCacheConnectorManager")
        .def(nb::init<>())
        .def("get_num_new_matched_tokens", &tb::kv_connector::KvCacheConnectorManager::getNumNewMatchedTokens,
            nb::arg("request"), nb::arg("num_computed_tokens"));
}
