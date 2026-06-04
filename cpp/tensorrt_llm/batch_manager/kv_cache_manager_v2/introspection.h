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
#include <tuple>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCacheIntrospection
{
public:
    using ActivePageStats = std::tuple<std::vector<int>, std::vector<int>>;

    static ActivePageStats activePageStats(KvCache const& kvCache);
    static bool allTreePagesDroppable(KvCacheManager& manager);
    static bool isCommitAllowed(KvCache const& kvCache);
    static std::vector<float> currentGpuRatio(KvCacheManager& manager);
    static std::vector<StorageStatistics> storageStatistics(KvCacheManager& manager, CacheLevel level);
    static std::vector<float> storageUtilization(KvCacheManager& manager, CacheLevel level);
    static int64_t grainsForSlots(int numSlots, std::vector<int> const& slotSizeList, int granularity);
    static std::tuple<int, int64_t> grainsToSlots(
        int64_t pgGrains, std::vector<int> const& slotSizeList, int granularity);
    static std::vector<int> ratioToSlotCountList(size_t quota, std::vector<std::vector<int>> const& slotSizeLists,
        std::vector<float> const& ratioList, int granularity, std::vector<int> const& minSlots);
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
