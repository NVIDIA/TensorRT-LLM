/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <iostream>

namespace tensorrt_llm::pg_utils
{

c10::intrusive_ptr<c10d::ProcessGroup> pg_world;
c10::intrusive_ptr<c10d::ProcessGroup> pg_local;

c10::intrusive_ptr<c10d::ProcessGroup> get_world_pg()
{
    return pg_world;
}

c10::intrusive_ptr<c10d::ProcessGroup> get_local_pg()
{
    return pg_local;
}

} // namespace tensorrt_llm::pg_utils

namespace tensorrt_llm::pg_broker
{

using namespace tensorrt_llm::pg_utils;

void init_pg(c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_world,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_local)
{
    std::cout << "init_pg on rank " << process_group_world->getRank() << std::endl;
    pg_world = process_group_world;
    pg_local = process_group_local;
}

void init_store(c10::intrusive_ptr<c10d::Store> const& default_store)
{
    (void) default_store; // TODO
    std::cout << "init_store on rank " << pg_world->getRank() << std::endl;
}

} // namespace tensorrt_llm::pg_broker
