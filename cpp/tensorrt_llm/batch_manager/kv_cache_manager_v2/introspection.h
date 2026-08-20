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
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCacheIntrospection
{
public:
    using ActivePageStats = std::tuple<TypedVec<CacheLevel, int>, TypedVec<CacheLevel, int>>;

    static ActivePageStats activePageStats(KvCache const& kvCache);

    // Whether the sequence's page at (ordinal, lcId) still points at a tree block;
    // nullopt when the slot is empty or holds an uncommitted page. Test hook for the
    // back-pointer invariant Block::replacePage() maintains.
    static std::optional<bool> committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId);
    static bool allTreePagesDroppable(KvCacheManager& manager);
    static TypedVec<PoolGroupIndex, StorageStatistics> storageStatistics(KvCacheManager& manager, CacheLevel level);

    // White-box hook: minimum per-pool-group slot counts to support a BatchDesc.
    // Reaches StorageManager::computeSlotsForBatch() (private) via friendship.
    static TypedVec<PoolGroupIndex, SlotCount> computeSlotsForBatch(KvCacheManager& manager, BatchDesc const& batch,
        int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse);

    // White-box test hooks: mutate auto-tuner state so accuracy tests can force a
    // pool rebalance. Reach KvCacheManager's private members via friendship.
    static void setNumSampledKvCaches(KvCacheManager& manager, int value);
    static void setLastAdjustmentTime(KvCacheManager& manager, double value);
    static void setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value);
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
