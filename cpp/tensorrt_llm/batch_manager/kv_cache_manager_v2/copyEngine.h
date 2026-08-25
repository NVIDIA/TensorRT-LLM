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
#include "kv_cache_manager_v2/stagingBuffer.h"

#include <cstddef>
#include <cuda.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

struct CopyTask
{
    Address dst;
    Address src;
};

// ---------------------------------------------------------------------------
// CopyEngine — dispatches bulk transfers between cache tiers.
//
// Single-hop pairs call the appropriate copy function from kvCacheManagerV2Utils.
// Two-hop pairs (GPU↔Disk) route through a pinned-host page ring borrowed from StorageManager.
// ---------------------------------------------------------------------------
class CopyEngine
{
public:
    explicit CopyEngine(StagingBufferManager* pageStagingManager);
    ~CopyEngine() = default;

    CopyEngine(CopyEngine const&) = delete;
    CopyEngine& operator=(CopyEngine const&) = delete;

    // Transfer numBytes per task. Tasks must all share the same source and destination tiers.
    void transfer(
        CacheTier dstTier, CacheTier srcTier, size_t numBytes, std::vector<CopyTask> const& tasks, CUstream stream);

private:
    [[nodiscard]] StagingBufferManager& pageStagingManager() const;

    // Non-owning; null when every cold tier is GPU or host memory.
    StagingBufferManager* mPageStagingManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
