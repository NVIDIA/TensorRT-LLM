/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

KVCacheEventManager::KVCacheEventManager(
    std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager> kvCacheManager)
    : kvCacheManager{std::move(kvCacheManager)}
{
}

std::deque<KVCacheEvent> KVCacheEventManager::getLatestEvents(std::optional<std::chrono::milliseconds> timeout)
{
    return kvCacheManager->getLatestEvents(timeout);
}

} // namespace tensorrt_llm::executor
