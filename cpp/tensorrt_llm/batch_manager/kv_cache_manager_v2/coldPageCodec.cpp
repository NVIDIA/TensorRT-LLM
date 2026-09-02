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

#include "kv_cache_manager_v2/coldPageCodec.h"
#include "kv_cache_manager_v2/batchedPageCopy.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

struct PoolCopyPlan
{
    std::byte* hotBase;
    size_t hotPageBytes;
    size_t coldPageOffset;
};

class ConcatKvCacheColdPageCodec final : public IKvCacheColdPageCodec
{
public:
    // The concat layout depends only on hot-pool geometry. Therefore, all lifecycle variants in one hot pool group
    // form one codec batching equivalence class. Codecs with lifecycle-specific algorithms or sizes need finer-grained
    // configuration.
    struct GroupConfig
    {
        size_t coldPageBytes;
        LayerGroupId batchingLayerGroupId;
        TypedVec<PoolIndex, PoolCopyPlan> copyPlans;
    };

    bool configure(PoolGroupDesc const* gpuDescs, PoolGroupIndex numGpuDescs) noexcept override
    {
        try
        {
            configureImpl(gpuDescs, numGpuDescs);
            return true;
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Failed to configure the default cold-page codec: %s", error.what());
            return false;
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Failed to configure the default cold-page codec: unknown error");
            return false;
        }
    }

    [[nodiscard]] size_t queryColdPageBytes(LayerGroupId layerGroupId) const noexcept override
    {
        GroupConfig const* const group = findGroup(layerGroupId);
        return group == nullptr ? 0 : group->coldPageBytes;
    }

    [[nodiscard]] LayerGroupId getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept override
    {
        GroupConfig const* const group = findGroup(layerGroupId);
        return group == nullptr ? LayerGroupId{-1} : group->batchingLayerGroupId;
    }

    [[nodiscard]] PageIndexLocation queryPageIndexLocation(LayerGroupId layerGroupId) const noexcept override
    {
        return findGroup(layerGroupId) == nullptr ? PageIndexLocation::kBadLocation : mCopier.pageIndexLocation();
    }

    bool encode(LayerGroupId layerGroupId, void* dstBasePtr, PageIndexPair const* pageIndices, size_t numBasePages,
        cudaStream_t stream) noexcept override
    {
        try
        {
            return dispatch<true>(findGroup(layerGroupId), dstBasePtr, nullptr, pageIndices, numBasePages, stream);
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Default cold-page encoding failed: %s", error.what());
            return false;
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Default cold-page encoding failed: unknown error");
            return false;
        }
    }

    bool decode(LayerGroupId layerGroupId, void const* srcBasePtr, PageIndexPair const* pageIndices,
        size_t numBasePages, cudaStream_t stream) noexcept override
    {
        try
        {
            return dispatch<false>(findGroup(layerGroupId), nullptr, srcBasePtr, pageIndices, numBasePages, stream);
        }
        catch (std::exception const& error)
        {
            TLLM_LOG_ERROR("Default cold-page decoding failed: %s", error.what());
            return false;
        }
        catch (...)
        {
            TLLM_LOG_ERROR("Default cold-page decoding failed: unknown error");
            return false;
        }
    }

private:
    void configureImpl(PoolGroupDesc const* gpuDescs, PoolGroupIndex numGpuDescs)
    {
        TLLM_CHECK(gpuDescs != nullptr && numGpuDescs.value() > 0 && mGroups.empty() && mLifeCycleToGroup.empty());

        LifeCycleId numLifeCycles{0};
        for (PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            PoolGroupDesc const& gpuDesc = gpuDescs[toSizeT(poolGroupIndex)];
            TLLM_CHECK(gpuDesc.poolGroupIndex == poolGroupIndex && !gpuDesc.pools.empty()
                && !gpuDesc.slotDesc.variants.empty());
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                TLLM_CHECK(variant.lifeCycleId.value() >= 0
                    && variant.lifeCycleId.value() < std::numeric_limits<LifeCycleId::ValueType>::max()
                    && variant.coalescedBuffers.stdSize() == gpuDesc.pools.stdSize());
                numLifeCycles = std::max(numLifeCycles, LifeCycleId{variant.lifeCycleId.value() + 1});
            }
        }

        TypedVec<PoolGroupIndex, std::unique_ptr<GroupConfig>> groups(numGpuDescs);
        TypedVec<LifeCycleId, PoolGroupIndex> lifeCycleToGroup(numLifeCycles, PoolGroupIndex{-1});
        for (PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            PoolGroupDesc const& gpuDesc = gpuDescs[toSizeT(poolGroupIndex)];
            TypedVec<PoolIndex, PoolCopyPlan> copyPlans;
            copyPlans.reserve(gpuDesc.pools.size());

            size_t coldOffset = 0;
            for (PoolIndex poolIndex{0}; poolIndex < gpuDesc.pools.size(); ++poolIndex)
            {
                PoolDesc const& pool = gpuDesc.pools.at(poolIndex);
                TLLM_CHECK(pool.poolIndex == poolIndex && pool.baseAddress != 0 && pool.slotBytes > 0
                    && pool.slotBytes <= std::numeric_limits<size_t>::max() - coldOffset);
                copyPlans.push_back(
                    PoolCopyPlan{reinterpret_cast<std::byte*>(pool.baseAddress), pool.slotBytes, coldOffset});
                coldOffset += pool.slotBytes;
            }

            LayerGroupId batchingLayerGroupId = gpuDesc.slotDesc.variants.front().lifeCycleId;
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                TLLM_CHECK(lifeCycleToGroup.at(variant.lifeCycleId).value() < 0);
                for (PoolIndex poolIndex{0}; poolIndex < gpuDesc.pools.size(); ++poolIndex)
                {
                    TLLM_CHECK(variant.coalescedBuffers.at(poolIndex).size() == gpuDesc.pools.at(poolIndex).slotBytes);
                }
                lifeCycleToGroup.at(variant.lifeCycleId) = poolGroupIndex;
                batchingLayerGroupId = std::min(batchingLayerGroupId, variant.lifeCycleId);
            }

            groups.at(poolGroupIndex)
                = std::make_unique<GroupConfig>(GroupConfig{coldOffset, batchingLayerGroupId, std::move(copyPlans)});
        }

        mGroups = std::move(groups);
        mLifeCycleToGroup = std::move(lifeCycleToGroup);
    }

    [[nodiscard]] GroupConfig const* findGroup(LayerGroupId layerGroupId) const
    {
        if (layerGroupId.value() < 0 || layerGroupId >= mLifeCycleToGroup.size())
        {
            return nullptr;
        }
        PoolGroupIndex const groupIndex = mLifeCycleToGroup.at(layerGroupId);
        if (groupIndex.value() < 0 || groupIndex >= mGroups.size())
        {
            return nullptr;
        }
        return mGroups.at(groupIndex).get();
    }

    template <bool isEncode>
    bool dispatch(GroupConfig const* group, void* dstBasePtr, void const* srcBasePtr, PageIndexPair const* pageIndices,
        size_t numBasePages, cudaStream_t stream) const
    {
        if (numBasePages == 0)
        {
            return group != nullptr;
        }
        // Null stream is rejected, not defaulted: cuMemcpyBatchAsync documents "must not be legacy
        // NULL stream" and returns CUDA_ERROR_INVALID_VALUE for both 0 and CU_STREAM_LEGACY.
        if (group == nullptr || pageIndices == nullptr || (isEncode ? dstBasePtr == nullptr : srcBasePtr == nullptr)
            || stream == nullptr || numBasePages > std::numeric_limits<size_t>::max() / group->copyPlans.stdSize())
        {
            return false;
        }
        // KVCM never submits a negative index, so this is a debug-only sanity check rather than a
        // validated precondition -- an O(numBasePages) host scan does not belong on the eviction
        // critical path. The copy path carries matching assertions.
        if (mCopier.pageIndexLocation() == PageIndexLocation::kHost)
        {
            TLLM_CHECK_DEBUG(std::all_of(pageIndices, pageIndices + numBasePages,
                [](PageIndexPair const& pair) { return pair.dst >= 0 && pair.src >= 0; }));
        }

        // One dispatcher launch per pool, all on the same stream. Stream ordering serialises them,
        // so each may use the full grid. The dispatcher picks the copy kernel or cuMemcpyBatchAsync
        // and, when the kernel is selected, tolerates pages that straddle a HostMem registration
        // chunk -- so the per-page splitting this used to do is no longer needed.
        auto* const coldDstBase = static_cast<std::byte*>(dstBasePtr);
        auto const* const coldSrcBase = static_cast<std::byte const*>(srcBasePtr);
        {
            // PoolCopyArgs is a 32-bit interface; numBasePages in particular is unbounded on the
            // HOST tier path, so narrow explicitly rather than silently.
            TLLM_CHECK(numBasePages <= std::numeric_limits<uint32_t>::max());
            for (PoolCopyPlan const& plan : group->copyPlans)
            {
                TLLM_CHECK(plan.hotPageBytes <= std::numeric_limits<uint32_t>::max());
                PoolCopyArgs args{};
                args.bytesPerPage = static_cast<uint32_t>(plan.hotPageBytes);
                args.pairs = pageIndices;
                args.numPairs = static_cast<uint32_t>(numBasePages);
                if constexpr (isEncode)
                {
                    args.dstBase = reinterpret_cast<CUdeviceptr>(coldDstBase + plan.coldPageOffset);
                    args.dstStride = group->coldPageBytes;
                    args.srcBase = reinterpret_cast<CUdeviceptr>(plan.hotBase);
                    args.srcStride = plan.hotPageBytes;
                }
                else
                {
                    args.dstBase = reinterpret_cast<CUdeviceptr>(plan.hotBase);
                    args.dstStride = plan.hotPageBytes;
                    args.srcBase = reinterpret_cast<CUdeviceptr>(coldSrcBase + plan.coldPageOffset);
                    args.srcStride = group->coldPageBytes;
                }
                mCopier.launch(
                    args, isEncode ? CopyDirection::kD2H : CopyDirection::kH2D, reinterpret_cast<CUstream>(stream));
            }
        }
        return true;
    }

    //! Mutable: launch() reuses per-dispatcher descriptor scratch, and dispatch() is const.
    mutable BatchedPageCopier mCopier;
    TypedVec<PoolGroupIndex, std::unique_ptr<GroupConfig>> mGroups;
    TypedVec<LifeCycleId, PoolGroupIndex> mLifeCycleToGroup;
};

} // namespace

IKvCacheColdPageCodec::IKvCacheColdPageCodec() = default;
IKvCacheColdPageCodec::~IKvCacheColdPageCodec() = default;

LayerGroupId IKvCacheColdPageCodec::getBatchingLayerGroupId(LayerGroupId layerGroupId) const noexcept
{
    return layerGroupId;
}

std::unique_ptr<IKvCacheColdPageCodec> createDefaultKvCacheColdPageCodec()
{
    return std::make_unique<ConcatKvCacheColdPageCodec>();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
