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

#include "tensorrt_llm/batch_manager/peftCacheManager.h"

#include <limits>

namespace tensorrt_llm::batch_manager
{

void NoOpPeftCacheManager::addRequestPeft(std::shared_ptr<LlmRequest> llmRequest, bool tryGpuCache) {}

PeftCacheManager::PeftTable NoOpPeftCacheManager::ensureBatch(
    RequestVector const& contextRequests, RequestVector const& generationRequests, bool resetGpuCache)
{
    return PeftTable{};
}

void NoOpPeftCacheManager::resetDeviceCache() {}

void NoOpPeftCacheManager::markRequestDone(LlmRequest const& llmReq, bool pause) {}

SizeType32 NoOpPeftCacheManager::getMaxDevicePages() const
{
    return std::numeric_limits<SizeType32>::max();
}

SizeType32 NoOpPeftCacheManager::getMaxHostPages() const
{
    return std::numeric_limits<SizeType32>::max();
}

SizeType32 NoOpPeftCacheManager::determineNumPages(std::shared_ptr<LlmRequest> llmReqeust) const
{
    return 0;
}
} // namespace tensorrt_llm::batch_manager
