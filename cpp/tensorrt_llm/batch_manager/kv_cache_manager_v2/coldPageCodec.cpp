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
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

//! One byte range of a hot slot kept in the cold page.
struct CopySegment
{
    size_t hotOffset;
    size_t coldOffset;
    size_t bytes;
};

struct PoolCopyPlan
{
    std::byte* hotBase;
    size_t hotPageBytes;
    //! Ranges of the hot slot stored in the cold page, at coldPageOffset + segment.coldOffset.
    //! One segment covering the whole slot unless a keep-segment spec matches the pool.
    std::vector<CopySegment> segments;
    size_t coldPageOffset;
};

//! A pool holding exactly one buffer of `role` and `bufferBytes` a slot keeps only `segments`
//! (hot offset, bytes) in the cold tier; the rest of the slot is not stored and is undefined
//! after a decode. Set by the owner of that buffer's layout, which knows which bytes are dead
//! once a page leaves the hot tier (TRTLLM_KVCM2_COLD_KEEP_SEGMENTS="role:bufferBytes:off+len,...").
struct KeepSegmentSpec
{
    std::string role;
    size_t bufferBytes = 0;
    std::vector<std::pair<size_t, size_t>> segments;
};

std::optional<KeepSegmentSpec> parseKeepSegmentSpec()
{
    char const* const env = std::getenv("TRTLLM_KVCM2_COLD_KEEP_SEGMENTS");
    if (env == nullptr || *env == '\0')
    {
        return std::nullopt;
    }
    std::string const text(env);
    auto const first = text.find(':');
    auto const second = first == std::string::npos ? std::string::npos : text.find(':', first + 1);
    TLLM_CHECK_WITH_INFO(
        second != std::string::npos, "TRTLLM_KVCM2_COLD_KEEP_SEGMENTS must be role:bufferBytes:off+len,...: %s", env);
    KeepSegmentSpec spec;
    spec.role = text.substr(0, first);
    spec.bufferBytes = std::stoull(text.substr(first + 1, second - first - 1));
    size_t pos = second + 1;
    size_t end = 0;
    while (pos < text.size())
    {
        auto const comma = text.find(',', pos);
        std::string const item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        auto const plus = item.find('+');
        TLLM_CHECK_WITH_INFO(plus != std::string::npos, "Bad keep segment '%s' in %s", item.c_str(), env);
        size_t const offset = std::stoull(item.substr(0, plus));
        size_t const bytes = std::stoull(item.substr(plus + 1));
        // The copy paths move 16-byte vectors; segments must be ordered and disjoint.
        TLLM_CHECK_WITH_INFO(
            offset % 16 == 0 && bytes % 16 == 0 && bytes > 0 && offset >= end && offset + bytes <= spec.bufferBytes,
            "Keep segment '%s' must be 16-byte aligned, ordered and inside the buffer: %s", item.c_str(), env);
        spec.segments.emplace_back(offset, bytes);
        end = offset + bytes;
        if (comma == std::string::npos)
        {
            break;
        }
        pos = comma + 1;
    }
    TLLM_CHECK_WITH_INFO(!spec.segments.empty(), "TRTLLM_KVCM2_COLD_KEEP_SEGMENTS has no segments: %s", env);
    return spec;
}

//! Whether every life cycle of the group stores exactly one `spec` buffer in `poolIndex`.
bool poolMatchesSpec(PoolGroupDesc const& gpuDesc, PoolIndex poolIndex, KeepSegmentSpec const& spec)
{
    for (auto const& variant : gpuDesc.slotDesc.variants)
    {
        CoalescedBuffer const& buffers = variant.coalescedBuffers.at(poolIndex);
        if (buffers.numBuffers() != 1 || buffers.singleBufferSize != spec.bufferBytes
            || buffers.bufferIds.front().role != spec.role)
        {
            return false;
        }
    }
    return true;
}

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

        std::optional<KeepSegmentSpec> const keepSpec = parseKeepSegmentSpec();
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
                std::vector<CopySegment> segments;
                size_t coldBytes = 0;
                if (keepSpec && pool.slotBytes == keepSpec->bufferBytes
                    && poolMatchesSpec(gpuDesc, poolIndex, *keepSpec))
                {
                    for (auto const& [hotOffset, bytes] : keepSpec->segments)
                    {
                        segments.push_back(CopySegment{hotOffset, coldBytes, bytes});
                        coldBytes += bytes;
                    }
                    TLLM_LOG_INFO(
                        "KVCM2 cold-page codec: pool group %d pool %d (%s) keeps %zu of %zu bytes a slot "
                        "in %zu segment(s)",
                        poolGroupIndex.value(), poolIndex.value(), keepSpec->role.c_str(), coldBytes, pool.slotBytes,
                        segments.size());
                }
                else
                {
                    segments.push_back(CopySegment{0, 0, pool.slotBytes});
                    coldBytes = pool.slotBytes;
                }
                copyPlans.push_back(PoolCopyPlan{
                    reinterpret_cast<std::byte*>(pool.baseAddress), pool.slotBytes, std::move(segments), coldOffset});
                coldOffset += coldBytes;
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
        // Valid indices are a caller contract (see PageIndexPair), not something this method
        // validates: an O(numBasePages) host scan does not belong on the eviction critical path,
        // and it is impossible anyway once the array lives in device memory. This is a debug-only
        // sanity check on the host-array path, where the scan happens to be cheap and possible.
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
                for (CopySegment const& segment : plan.segments)
                {
                    TLLM_CHECK(segment.bytes <= std::numeric_limits<uint32_t>::max());
                    PoolCopyArgs args{};
                    args.bytesPerPage = static_cast<uint32_t>(segment.bytes);
                    args.pairs = pageIndices;
                    args.numPairs = static_cast<uint32_t>(numBasePages);
                    std::byte* const hot = plan.hotBase + segment.hotOffset;
                    size_t const coldOffset = plan.coldPageOffset + segment.coldOffset;
                    if constexpr (isEncode)
                    {
                        args.dstBase = reinterpret_cast<CUdeviceptr>(coldDstBase + coldOffset);
                        args.dstStride = group->coldPageBytes;
                        args.srcBase = reinterpret_cast<CUdeviceptr>(hot);
                        args.srcStride = plan.hotPageBytes;
                    }
                    else
                    {
                        args.dstBase = reinterpret_cast<CUdeviceptr>(hot);
                        args.dstStride = plan.hotPageBytes;
                        args.srcBase = reinterpret_cast<CUdeviceptr>(coldSrcBase + coldOffset);
                        args.srcStride = group->coldPageBytes;
                    }
                    mCopier.launch(
                        args, isEncode ? CopyDirection::kD2H : CopyDirection::kH2D, reinterpret_cast<CUstream>(stream));
                }
            }
        }
        return true;
    }

    //! Mutable: launch() reuses per-dispatcher descriptor scratch, and dispatch() is const.
    BatchedPageCopier mCopier;
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
