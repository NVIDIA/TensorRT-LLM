/*
 * Copyright (c) 2021-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

namespace tensorrt_llm::mpi
{

enum class MpiTag : int
{
    kDefault = 0,

    // Values 1-23, 127-129, and 1023-1025 are retired; do not reuse.

    // KvCacheEventManager
    kKvCacheEventSize = 1026,
    kKvCacheEvent = 1027,

    // Asynchronous context-transfer coordination.
    kContextTransferEvent = 1028,
    kContextTransferUpdate = 1029
};

} // namespace tensorrt_llm::mpi
