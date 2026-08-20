/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct Block;

// Boundary between the cache implementation and the native event manager.
class EventSink
{
public:
    virtual ~EventSink() = default;

    virtual void addStoredBlock(Block const& block) = 0;
    virtual void addStoredLifeCycle(Block const& block, LifeCycleId lifeCycle) = 0;
    virtual void addRemovedBlock(Digest const& blockKey) = 0;
    virtual void addRemovedLifeCycle(Digest const& blockKey, LifeCycleId lifeCycle) = 0;
    virtual void addCacheLevelUpdated(
        Digest const& blockKey, CacheLevel oldLevel, CacheLevel newLevel, LifeCycleId lifeCycle)
        = 0;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
