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
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! Stable request rows and raw device metadata, shared across all layer groups.
//!
//! Driven by one owning thread, like its member requests. Membership is non-owning:
//! closing/destroying a request detaches it; closing/destroying a batch detaches its
//! live members without closing them. Mutable state uses the manager's API lock.
//! Publish outside graph capture, then waitReady on a consumer stream. Submit all
//! reads and call recordRead before mutating requests, rows, or tables.
class Batch : public std::enable_shared_from_this<Batch>
{
public:
    Batch(std::shared_ptr<KvCacheManager> manager, int maxRows, int maxBlocks, int maxBeamWidth = 1);
    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Attach a request at a chosen row or the first free row. Membership is exclusive.
    int add(KvCache& cache, std::optional<int> row = std::nullopt);
    //! Detach a member, leaving its row dirty so the next publication clears it.
    void remove(KvCache& cache);
    //! Detach all members. Exported arrays retain their allocation until the last owner dies.
    void close();

    //! Retry deferred sparse offloads, then upload final dirty rows and counts; return uploaded rows.
    //! Offload failures propagate and retain pending work for retry.
    //! Staging buffers are retained until their asynchronous copies complete.
    std::vector<int> publish(CudaStream stream);
    //! Wait for publication and KV readiness. Reject unpublished changes.
    void waitReady(CudaStream stream) const;
    //! Fence submitted reads before table overwrite and request storage release.
    void recordRead(CudaStream stream);
    //! Resize by stable row slot, then publish once. Empty rows return nullopt.
    std::vector<std::optional<bool>> resize(std::vector<std::optional<int>> const& capacities,
        std::vector<std::optional<int>> const& historyLengths, CudaStream stream);

    std::vector<int> dirtyRows() const;

    int maxRows() const noexcept
    {
        return mMaxRows;
    }

    int maxBlocks() const noexcept
    {
        return mMaxBlocks;
    }

    int maxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    int numLayerGroups() const noexcept
    {
        return mNumLayerGroups;
    }

    int deviceId() const noexcept
    {
        return mDeviceId;
    }

    //! Internal device views: [row, beam, block] and [row, beam], respectively.
    //! Addresses stay fixed for the batch lifetime; callers must obey the read contract.
    MemAddress pageTableAddress(LayerGroupId group) const;
    MemAddress numBlocksAddress(LayerGroupId group) const;

private:
    friend class KvCache;
    void markDirty(int row) noexcept;
    void checkOpen() const;
    void checkPublished() const;
    size_t rowOffset(int group, int row) const noexcept;
    size_t countOffset(int group, int row) const noexcept;

    struct Upload
    {
        std::unique_ptr<HostMem> staging;
        CachedCudaEvent completion = CachedCudaEvent::makeNull();
    };

    std::shared_ptr<KvCacheManager> mManager;
    int mMaxRows;
    int mMaxBlocks;
    int mMaxBeamWidth;
    int mNumLayerGroups = 0;
    int mDeviceId = 0;
    size_t mTableElements = 0;
    size_t mTotalBytes = 0;
    CudaUniqPtr mDeviceMemory;
    std::vector<KvCache*> mRows;
    std::vector<uint8_t> mDirty;
    std::list<Upload> mUploads;
    CachedCudaEvent mReady = CachedCudaEvent::makeNull();
    std::vector<CachedCudaEvent> mReaders;
    bool mPublished = false;
    bool mClosed = false;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
