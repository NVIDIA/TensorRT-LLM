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

#include "kv_cache_manager_v2/copyEngine.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"

// Reuse existing copy implementations (no Python round-trip).
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"

#include "tensorrt_llm/common/assert.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// dispatchCopy<DstTier, SrcTier> — template that maps CacheTier pair to
// address types and the underlying copy function via if-constexpr.
// Replaces 7 near-identical static dispatcher functions.
// ---------------------------------------------------------------------------

template <CacheTier Tier>
using TierAddr = std::conditional_t<Tier == CacheTier::DISK, DiskAddress, MemAddress>;

template <CacheTier DstTier, CacheTier SrcTier>
static void dispatchCopy(std::vector<CopyTask> const& tasks, size_t numBytes, CUstream stream)
{
    using DstAddr = TierAddr<DstTier>;
    using SrcAddr = TierAddr<SrcTier>;

    std::vector<Task<DstAddr, SrcAddr>> t;
    t.reserve(tasks.size());
    for (auto const& task : tasks)
        t.push_back({std::get<DstAddr>(task.dst), std::get<SrcAddr>(task.src)});

    constexpr auto G = CacheTier::GPU_MEM;
    constexpr auto H = CacheTier::HOST_MEM;
    constexpr auto D = CacheTier::DISK;

    // clang-format off
    if constexpr      (DstTier == G && SrcTier == G) cuCheck(copyDeviceToDevice(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == G && SrcTier == H) cuCheck(copyHostToDevice(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == G) cuCheck(copyDeviceToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == H) cuCheck(copyHostToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == D && SrcTier == D) cuCheck(copyDiskToDisk(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == H && SrcTier == D) cuCheck(copyDiskToHost(std::move(t), static_cast<ssize_t>(numBytes), stream));
    else if constexpr (DstTier == D && SrcTier == H) cuCheck(copyHostToDisk(std::move(t), static_cast<ssize_t>(numBytes), stream));
    // clang-format on
}

// ---------------------------------------------------------------------------
// CopyEngine
// ---------------------------------------------------------------------------

CopyEngine::CopyEngine(StagingBufferManager* pageStagingManager)
    : mPageStagingManager(pageStagingManager)
{
}

StagingBufferManager& CopyEngine::pageStagingManager() const
{
    TLLM_CHECK_WITH_INFO(mPageStagingManager != nullptr, "CopyEngine requires page staging for this transfer");
    return *mPageStagingManager;
}

// Two-hop transfer via host staging buffer (e.g., GPU→Disk or Disk→GPU).
// SrcTier → MidTier (staging) → DstTier.  MidTier is the staging tier (HOST_MEM).
template <CacheTier DstTier, CacheTier MidTier, CacheTier SrcTier>
static void twoHopTransfer(
    StagingBufferManager& manager, size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream)
{
    size_t remaining = tasks.size();
    size_t offset = 0;

    while (remaining > 0)
    {
        size_t const maxSize = remaining > std::numeric_limits<size_t>::max() / numBytes
            ? std::numeric_limits<size_t>::max()
            : numBytes * remaining;
        StagingBuffer buf = manager.acquire(numBytes, maxSize, numBytes, 1, stream);
        MemAddress addr = buf.address();
        size_t n = buf.size() / numBytes;
        TLLM_CHECK_DEBUG(n > 0 && n <= remaining);

        // First hop: src → staging
        {
            std::vector<CopyTask> hop1;
            hop1.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop1.push_back({Address{addr + numBytes * i}, tasks[offset + i].src});
            dispatchCopy<MidTier, SrcTier>(hop1, numBytes, *buf.stream());
        }

        // Second hop: staging → dst
        {
            std::vector<CopyTask> hop2;
            hop2.reserve(n);
            for (size_t i = 0; i < n; ++i)
                hop2.push_back({tasks[offset + i].dst, Address{addr + numBytes * i}});
            dispatchCopy<DstTier, MidTier>(hop2, numBytes, *buf.stream());
        }

        offset += n;
        remaining -= n;
    }
}

void CopyEngine::transfer(
    CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream)
{
    // Nothing to copy. Also guards the two-hop path against a div-by-zero on
    // `buf.size() / numBytes` when numBytes == 0.
    if (tasks.empty() || numBytes == 0)
    {
        return;
    }

    constexpr auto G = CacheTier::GPU_MEM;
    constexpr auto H = CacheTier::HOST_MEM;
    constexpr auto D = CacheTier::DISK;

    if (dstTier == G && srcTier == G)
        dispatchCopy<G, G>(tasks, numBytes, stream);
    else if (dstTier == G && srcTier == H)
        dispatchCopy<G, H>(tasks, numBytes, stream);
    else if (dstTier == G && srcTier == D)
        twoHopTransfer<G, H, D>(pageStagingManager(), numBytes, tasks, stream);
    else if (dstTier == H && srcTier == G)
        dispatchCopy<H, G>(tasks, numBytes, stream);
    else if (dstTier == H && srcTier == H)
        dispatchCopy<H, H>(tasks, numBytes, stream);
    else if (dstTier == H && srcTier == D)
        dispatchCopy<H, D>(tasks, numBytes, stream);
    else if (dstTier == D && srcTier == G)
        twoHopTransfer<D, H, G>(pageStagingManager(), numBytes, tasks, stream);
    else if (dstTier == D && srcTier == H)
        dispatchCopy<D, H>(tasks, numBytes, stream);
    else if (dstTier == D && srcTier == D)
        dispatchCopy<D, D>(tasks, numBytes, stream);
    else
        throw std::invalid_argument("CopyEngine::transfer: unsupported tier combination");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
