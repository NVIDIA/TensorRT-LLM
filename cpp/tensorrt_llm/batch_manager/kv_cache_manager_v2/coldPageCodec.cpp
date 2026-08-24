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
#include "kv_cache_manager_v2/coldPageCopy.h"
#include "kv_cache_manager_v2/utils/funcGuard.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

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

    void registerHostMem(HostMem const* memory)
    {
        TLLM_CHECK(
            memory != nullptr && std::find(mHostMemories.begin(), mHostMemories.end(), memory) == mHostMemories.end());
        mHostMemories.push_back(memory);
    }

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
        return findGroup(layerGroupId) == nullptr ? PageIndexLocation::kBadLocation : PageIndexLocation::kHost;
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

    [[nodiscard]] HostMem const* findHostMem(MemAddress address) const noexcept
    {
        auto const memory = std::find_if(mHostMemories.begin(), mHostMemories.end(),
            [address](HostMem const* candidate)
            {
                MemAddress const begin = candidate->address();
                return begin <= address && address - begin < candidate->size();
            });
        return memory == mHostMemories.end() ? nullptr : *memory;
    }

    template <bool isEncode>
    bool dispatch(GroupConfig const* group, void* dstBasePtr, void const* srcBasePtr, PageIndexPair const* pageIndices,
        size_t numBasePages, cudaStream_t stream) const
    {
        if (numBasePages == 0)
        {
            return group != nullptr;
        }
        if (group == nullptr || pageIndices == nullptr || (isEncode ? dstBasePtr == nullptr : srcBasePtr == nullptr)
            || stream == nullptr || numBasePages > std::numeric_limits<size_t>::max() / group->copyPlans.stdSize())
        {
            return false;
        }

        thread_local std::vector<CUdeviceptr> dsts;
        thread_local std::vector<CUdeviceptr> srcs;
        thread_local std::vector<size_t> sizes;
        dsts.clear();
        srcs.clear();
        sizes.clear();
        size_t const numCopies = group->copyPlans.stdSize() * numBasePages;
        dsts.reserve(numCopies);
        srcs.reserve(numCopies);
        sizes.reserve(numCopies);
        auto const releaseCopyVectors = FuncGuard(
            [&]() noexcept
            {
                constexpr size_t kMaxRetainedCopyEntries = 128U << 10U;
                if (numCopies > kMaxRetainedCopyEntries)
                {
                    std::vector<CUdeviceptr>().swap(dsts);
                    std::vector<CUdeviceptr>().swap(srcs);
                    std::vector<size_t>().swap(sizes);
                }
            });

        // Work around the interaction of two independent bugs. Linux kernels 6.11 through 6.13 cannot reliably pin
        // more than 2 GiB in one call, so HostMem registers a large allocation as adjacent 2 GiB regions. Separately,
        // cuMemcpyBatchAsync cannot handle one copy entry that spans two such registrations. The cold pointer can be a
        // subrange of a larger staging allocation, so calculate boundaries relative to its owning HostMem span.
        auto const* const coldBase
            = isEncode ? static_cast<std::byte const*>(dstBasePtr) : static_cast<std::byte const*>(srcBasePtr);
        MemAddress const coldBaseAddress = reinterpret_cast<MemAddress>(coldBase);
        HostMem const* const hostMem = findHostMem(coldBaseAddress);
        bool const splitRegistrationChunks = hostMem != nullptr && hostMem->size() > HostMem::kChunkSize;
        size_t const coldBaseOffset = splitRegistrationChunks ? coldBaseAddress - hostMem->address() : 0;
        auto appendCopy = [&](std::byte* dst, std::byte const* src, size_t pinnedOffset, size_t numBytes)
        {
            if (splitRegistrationChunks)
            {
                TLLM_CHECK(pinnedOffset <= hostMem->size() && numBytes <= hostMem->size() - pinnedOffset);
            }
            do
            {
                size_t copyBytes = numBytes;
                if (splitRegistrationChunks)
                {
                    size_t const bytesUntilBoundary = HostMem::kChunkSize - pinnedOffset % HostMem::kChunkSize;
                    copyBytes = std::min(copyBytes, bytesUntilBoundary);
                }
                dsts.push_back(reinterpret_cast<CUdeviceptr>(dst));
                srcs.push_back(reinterpret_cast<CUdeviceptr>(src));
                sizes.push_back(copyBytes);
                dst += copyBytes;
                src += copyBytes;
                pinnedOffset += copyBytes;
                numBytes -= copyBytes;
            } while (numBytes != 0);
        };

        auto* const coldDstBase = static_cast<std::byte*>(dstBasePtr);
        auto const* const coldSrcBase = static_cast<std::byte const*>(srcBasePtr);
        for (size_t page = 0; page < numBasePages; ++page)
        {
            PageIndexPair const pageIndex = pageIndices[page];
            if (pageIndex.dst < 0 || pageIndex.src < 0)
            {
                return false;
            }
            for (PoolCopyPlan const& plan : group->copyPlans)
            {
                auto* const hotBase
                    = plan.hotBase + static_cast<size_t>(isEncode ? pageIndex.src : pageIndex.dst) * plan.hotPageBytes;
                size_t const coldOffset
                    = static_cast<size_t>(isEncode ? pageIndex.dst : pageIndex.src) * group->coldPageBytes
                    + plan.coldPageOffset;
                size_t const pinnedOffset = coldBaseOffset + coldOffset;
                if constexpr (isEncode)
                {
                    appendCopy(coldDstBase + coldOffset, hotBase, pinnedOffset, plan.hotPageBytes);
                }
                else
                {
                    appendCopy(hotBase, coldSrcBase + coldOffset, pinnedOffset, plan.hotPageBytes);
                }
            }
        }

        detail::copyColdPageDataBatch(
            dsts.data(), srcs.data(), sizes.data(), dsts.size(), reinterpret_cast<CUstream>(stream));
        return true;
    }

    TypedVec<PoolGroupIndex, std::unique_ptr<GroupConfig>> mGroups;
    TypedVec<LifeCycleId, PoolGroupIndex> mLifeCycleToGroup;
    std::vector<HostMem const*> mHostMemories;
};

} // namespace

namespace detail
{

bool needsHostMemRegistration(IKvCacheColdPageCodec const& codec) noexcept
{
    return HostMem::shouldUseChunkedRegistration()
        && dynamic_cast<ConcatKvCacheColdPageCodec const*>(&codec) != nullptr;
}

void registerHostMem(IKvCacheColdPageCodec& codec, HostMem const* memory)
{
    auto* concatCodec = dynamic_cast<ConcatKvCacheColdPageCodec*>(&codec);
    TLLM_CHECK(concatCodec != nullptr);
    concatCodec->registerHostMem(memory);
}

} // namespace detail

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
