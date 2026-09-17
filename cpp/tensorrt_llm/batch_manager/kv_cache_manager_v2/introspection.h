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
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Where a host-tier allocation's pages ended up, and under which policy.
struct HostMemPlacement
{
    //! NUMA node of the GPU current at the time of the probe.
    int gpuNumaNode;
    //! True when the range carries MPOL_BIND rather than MPOL_PREFERRED, read
    //! back from the kernel so it reports what took effect rather than what was
    //! asked. Absent where the policy cannot be queried: get_mempolicy sits
    //! behind CAP_SYS_NICE in a container's default seccomp profile, which also
    //! means mbind did not apply and the placement below came from first touch.
    std::optional<bool> strictBinding;
    //! Resident pages per NUMA node, as reported by /proc/self/numa_maps. That
    //! is an ordinary proc read rather than a gated syscall, so placement stays
    //! observable even where the policy is not.
    std::vector<std::pair<int, size_t>> nodePageCounts;
};

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

    // White-box hook: minimum per-pool-group slot counts to support a BatchDesc.
    // Reaches StorageManager::computePoolGroupSlotsForBatch() (private) via friendship.
    static TypedVec<PoolGroupIndex, SlotCount> computeSlotsForBatch(KvCacheManager& manager, BatchDesc const& batch,
        int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse);

    // White-box test hooks: mutate auto-tuner state so accuracy tests can force a
    // pool rebalance. Reach KvCacheManager's private members via friendship.
    static void setNumSampledKvCaches(KvCacheManager& manager, int value);
    static void setLastAdjustmentTime(KvCacheManager& manager, double value);
    static void setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value);

    //! Allocates `size` bytes of host-tier memory the way the host tier would
    //! and reports where its pages landed.
    //!
    //! The commit runs on a worker bound to the GPU's node, as HostMem's fill
    //! does, so the result reflects the arrangement in production rather than
    //! mbind in isolation: placement holds through first touch when mbind is
    //! unavailable, and through the range policy when it is.
    //!
    //! Returns nullopt only where the question does not apply -- no libnuma, a
    //! single NUMA node, or a platform whose selected backing is not mmap. The
    //! VMM backing names its node to the driver, so there is no range policy to
    //! inspect there.
    //!
    //! Uses the current device, so the caller selects which GPU the placement is
    //! relative to. A CUDA context must already exist on it: the backing
    //! selection queries device attributes and will not create one.
    [[nodiscard]] static std::optional<HostMemPlacement> probeHostMemPlacement(
        size_t size, bool allowRemoteNumaFallback);
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
