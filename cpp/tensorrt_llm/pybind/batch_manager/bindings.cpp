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

#include "bindings.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/pybind/utils/bindTypes.h"

namespace py = pybind11;
namespace tb = tensorrt_llm::batch_manager;
namespace tle = tensorrt_llm::executor;

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::pybind::batch_manager
{

void initBindings(pybind11::module_& m)
{
    py::class_<tb::batch_scheduler::ContextChunkingConfig>(m, "ContextChunkingConfig")
        .def(py::init<tle::ContextChunkingPolicy, tensorrt_llm::runtime::SizeType32>(), py::arg("chunking_policy"),
            py::arg("chunk_unit_size"))
        .def_readwrite("chunking_policy", &tb::batch_scheduler::ContextChunkingConfig::chunkingPolicy)
        .def_readwrite("chunk_unit_size", &tb::batch_scheduler::ContextChunkingConfig::chunkUnitSize);
}

} // namespace tensorrt_llm::pybind::batch_manager
