/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/python.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

// Must in sync with cpp/include/tensorrt_llm/runtime/utils/pgUtils.h
namespace tensorrt_llm::pg_broker
{

void init_pg(c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_world,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_local);

void init_store(c10::intrusive_ptr<c10d::Store> const& default_store);

} // namespace tensorrt_llm::pg_broker

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init_pg", &tensorrt_llm::pg_broker::init_pg);
    m.def("init_store", &tensorrt_llm::pg_broker::init_store);
}
