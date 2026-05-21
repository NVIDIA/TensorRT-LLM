/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kvCacheManagerTestUtilBinding.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/testing/kvCacheManagerTestUtil.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::testing
{

void initKvCacheTestUtilBindings(nb::module_& m)
{
    m.def("simulate_prefill_completion_only_use_for_testing",
        &tensorrt_llm::testing::KvCacheManagerTestUtil::simulatePrefillCompletion, nb::arg("llm_request"),
        nb::call_guard<nb::gil_scoped_release>(),
        "NEVER USE IN PRODUCTION. Simulates prefill completion on an LlmRequest for test purposes.");
}

} // namespace tensorrt_llm::nanobind::testing
