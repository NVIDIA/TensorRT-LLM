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

#include "kv_cache_manager_v2/introspection.h"

#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/hostMemBacking.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"

#include <cuda_runtime_api.h>

#include <cctype>
#include <cstring>
#include <fstream>
#include <numa.h>
#include <numaif.h>
#include <sstream>
#include <thread>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

bool allBlockPagesDroppable(Block const& block)
{
    for (auto const* page : block.storage)
    {
        if (page != nullptr && page->status() != PageStatus::DROPPABLE)
        {
            return false;
        }
    }

    for (auto const& [_, child] : block.next)
    {
        if (!allBlockPagesDroppable(*child))
        {
            return false;
        }
    }
    return true;
}

} // namespace

KvCacheIntrospection::ActivePageStats KvCacheIntrospection::activePageStats(KvCache const& kvCache)
{
    auto& storageMgr = kvCache.manager().storage();
    CacheLevel const numTiers = storageMgr.numCacheLevels();
    TypedVec<CacheLevel, int> counts(numTiers, 0);
    TypedVec<CacheLevel, int> unscheduledEvictable(numTiers, 0);

    for (auto const& activePage : kvCache._activePages())
    {
        auto page = kvCache._page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }

        CacheLevel const level = page->cacheLevel;
        counts.at(level) += 1;
        if (storageMgr.isEvictable(*page) && !page->scheduledForEviction())
        {
            unscheduledEvictable.at(level) += 1;
        }
    }

    return {std::move(counts), std::move(unscheduledEvictable)};
}

std::optional<bool> KvCacheIntrospection::committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId)
{
    auto page = kvCache._page(BlockOrdinal{ordinal}, kDefaultBeamIndex, LifeCycleId{lcId});
    if (!page)
    {
        return std::nullopt;
    }
    auto committed = dynamicPointerCast<CommittedPage>(page);
    if (!committed)
    {
        return std::nullopt;
    }
    return committed->block != nullptr;
}

TypedVec<PoolGroupIndex, SlotCount> KvCacheIntrospection::computeSlotsForBatch(KvCacheManager& manager,
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse)
{
    return manager.storage().computePoolGroupSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
}

bool KvCacheIntrospection::allTreePagesDroppable(KvCacheManager& manager)
{
    for (auto const& [_, root] : manager.radixTree().roots())
    {
        for (auto const& [__, block] : root->next)
        {
            if (!allBlockPagesDroppable(*block))
            {
                return false;
            }
        }
    }
    return true;
}

void KvCacheIntrospection::setNumSampledKvCaches(KvCacheManager& manager, int value)
{
    manager.mNumSampledKvCaches = value;
}

void KvCacheIntrospection::setLastAdjustmentTime(KvCacheManager& manager, double value)
{
    manager.mLastAdjustmentTime = value;
}

void KvCacheIntrospection::setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value)
{
    manager.mTargetRatioListHot = std::move(value);
}

namespace
{

//! Resident pages per NUMA node for the VMA containing `base`.
//!
//! Read from /proc/self/numa_maps rather than queried with get_mempolicy or
//! move_pages: those are gated behind CAP_SYS_NICE in a container's default
//! seccomp profile, while this is an ordinary file read. The line for a mapping
//! carries "N<node>=<pages>" fields, which is what this picks out.
std::vector<std::pair<int, size_t>> residentPagesPerNode(MemAddress base)
{
    std::ifstream maps("/proc/self/numa_maps");
    if (!maps)
    {
        return {};
    }
    std::ostringstream prefix;
    prefix << std::hex << base;
    std::string const wanted = prefix.str();

    std::string line;
    while (std::getline(maps, line))
    {
        if (line.rfind(wanted, 0) != 0)
        {
            continue;
        }
        std::vector<std::pair<int, size_t>> counts;
        std::istringstream fields(line);
        std::string field;
        while (fields >> field)
        {
            if (field.size() < 2 || field[0] != 'N' || std::isdigit(field[1]) == 0)
            {
                continue;
            }
            auto const eq = field.find('=');
            if (eq == std::string::npos)
            {
                continue;
            }
            counts.emplace_back(std::stoi(field.substr(1, eq - 1)), std::stoull(field.substr(eq + 1)));
        }
        return counts;
    }
    return {};
}

} // namespace

std::optional<HostMemPlacement> KvCacheIntrospection::probeHostMemPlacement(size_t size, bool allowRemoteNumaFallback)
{
    TLLM_CHECK_WITH_INFO(size > 0, "probeHostMemPlacement needs a non-zero size");
    if (numa_available() < 0 || numa_max_node() < 1)
    {
        return std::nullopt;
    }

    HostMemBackingOptions options;
    options.allowRemoteNumaFallback = allowRemoteNumaFallback;
    auto backing = createHostMemBacking(options);
    // The VMM backing names its node to the driver, so there is no range policy
    // and no first-touch behaviour to observe.
    if (std::strcmp(backing->name(), "mmap") != 0)
    {
        return std::nullopt;
    }

    HostMemPlacement placement{};
    placement.gpuNumaNode = runtime::TopologyDetector::getInstance().getCurrentGpuNumaId();

    size_t const span = roundUp(size, backing->commitGranularity());
    MemAddress const base = backing->reserve(span);

    int mode = -1;
    if (::get_mempolicy(&mode, nullptr, 0, reinterpret_cast<void*>(base), MPOL_F_ADDR) == 0)
    {
        placement.strictBinding = mode == MPOL_BIND;
    }

    // Commit and fault on a worker bound to the GPU's node, which is what
    // HostMem's fill does. Placement then holds through first touch even where
    // mbind was refused, so this reports the production arrangement rather than
    // mbind alone.
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    std::exception_ptr failure;
    std::thread worker(
        [&]
        {
            try
            {
                // A new thread starts on device 0 whatever the caller selected,
                // so the device has to be reselected before binding by GPU.
                TLLM_CUDA_CHECK(cudaSetDevice(device));
                runtime::TopologyDetector::getInstance().bindThreadByCurrentGpu();
                backing->commit(0, span);
                std::memset(reinterpret_cast<void*>(base), 0, span);
            }
            catch (...)
            {
                failure = std::current_exception();
            }
        });
    worker.join();
    if (failure)
    {
        std::rethrow_exception(failure);
    }

    placement.nodePageCounts = residentPagesPerNode(base);
    return placement;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
