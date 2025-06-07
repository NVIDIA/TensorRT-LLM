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

#pragma once

#include "tensorrt_llm/runtime/common.h"

#include <map>
#include <optional>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class BaseKVCacheManager;

class NoEvictScheduledBlocksManager
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

public:
    explicit NoEvictScheduledBlocksManager(BaseKVCacheManager const& kvCacheManager);

    void decrementReservedBlocks(LlmRequest const& req);
    bool enoughAvailableBlocks(LlmRequest const& req);

private:
    BaseKVCacheManager const& mKvCacheManager;
    std::map<SizeType32, SizeType32> mAvailableBlocks;
};

class MaxUtilizationScheduledBlocksManager
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

public:
    MaxUtilizationScheduledBlocksManager(BaseKVCacheManager const& kvCacheManager, bool twoStepsLookAhead);

    std::optional<std::map<SizeType32, SizeType32>> prepareNewNumberOfBlocksIfWeEndUpScheduling(LlmRequest const& req);
    void updateScheduledBlocks(std::map<SizeType32, SizeType32> const& numBlocksIfScheduled);

private:
    BaseKVCacheManager const& mKvCacheManager;
    std::map<SizeType32, SizeType32> mNumScheduledBlocks;
    bool const mTwoStepsLookAhead;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
