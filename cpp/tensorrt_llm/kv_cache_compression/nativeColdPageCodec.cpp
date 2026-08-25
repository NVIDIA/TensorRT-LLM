/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw std::overflow_error("GPU Slot buffer offsets overflow size_t");
    }
    return lhs + rhs;
}

std::uintptr_t checkedAddress(std::uintptr_t base, std::size_t offset)
{
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base)
    {
        throw std::overflow_error("GPU buffer address overflows uintptr_t");
    }
    return base + offset;
}

ResolvedColdPageLifecycle resolveLifecycle(kv::PoolGroupDesc const& gpuDesc, kv::SlotDescVariant const& variant)
{
    ResolvedColdPageLifecycle result;
    for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
    {
        auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
        auto const& pool = gpuDesc.pools.at(poolIndex);
        std::size_t offset = 0;
        for (auto const& bufferId : coalesced.bufferIds)
        {
            auto& layer = result[bufferId.layerId];
            if (!layer
                     .emplace(bufferId.role,
                         ResolvedColdPageBuffer{
                             checkedAddress(pool.baseAddress, offset), pool.slotBytes, coalesced.singleBufferSize})
                     .second)
            {
                throw std::invalid_argument("GPU lifecycle contains a duplicate buffer role");
            }
            offset = checkedAdd(offset, coalesced.singleBufferSize);
        }
    }
    return result;
}

} // namespace

NativeColdPageCodec::NativeColdPageCodec(std::unique_ptr<IColdPageCodecBackend> backend)
    : mBackend(std::move(backend))
{
    if (!mBackend)
    {
        throw std::invalid_argument("NativeColdPageCodec requires a backend");
    }
}

bool NativeColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        auto const& backendLayerIds = mBackend->getLayerIds();
        std::map<kv::LayerGroupId, LayerGroupState> pendingGroups;
        std::vector<ResolvedColdPageLifecycle> backendLifecycles;
        std::vector<kv::LayerGroupId> backendLayerGroups;
        std::set<kv::LayerId> consumedLayers;

        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto resolved = resolveLifecycle(gpuDesc, variant);
                auto const backendLayerCount = std::count_if(resolved.begin(), resolved.end(),
                    [&backendLayerIds](auto const& layer) { return backendLayerIds.count(layer.first) != 0U; });

                LayerGroupState state;
                if (backendLayerCount == 0U)
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                    state.pageIndexLocation = losslessCodec->queryPageIndexLocation(variant.lifeCycleId);
                }
                else
                {
                    if (backendLayerCount != resolved.size())
                    {
                        throw std::invalid_argument("A lifecycle cannot mix backend-owned and fallback layers");
                    }
                    for (auto const& [layerId, buffers] : resolved)
                    {
                        static_cast<void>(buffers);
                        if (!consumedLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("A backend layer appears in multiple lifecycles");
                        }
                    }
                    state.backendIndex = backendLifecycles.size();
                    backendLayerGroups.push_back(variant.lifeCycleId);
                    backendLifecycles.push_back(std::move(resolved));
                }

                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (consumedLayers != backendLayerIds)
        {
            throw std::invalid_argument("A backend layer is absent from all GPU descriptors");
        }

        auto const backendConfigs = mBackend->configure(backendLifecycles);
        if (backendConfigs.size() != backendLifecycles.size())
        {
            throw std::invalid_argument("Cold-page backend returned an unexpected lifecycle count");
        }
        for (std::size_t index = 0; index < backendConfigs.size(); ++index)
        {
            auto const& config = backendConfigs[index];
            if (config.coldPageBytes == 0U || config.pageIndexLocation == kv::PageIndexLocation::kBadLocation)
            {
                throw std::invalid_argument("Cold-page backend returned invalid storage properties");
            }
            auto& state = pendingGroups.at(backendLayerGroups[index]);
            state.coldPageBytes = config.coldPageBytes;
            state.pageIndexLocation = config.pageIndexLocation;
        }

        mLayerGroups = std::move(pendingGroups);
        mLosslessCodec = std::move(losslessCodec);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::configure rejected GPU layouts: unknown error");
        return false;
    }
}

NativeColdPageCodec::LayerGroupState const* NativeColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
}

std::size_t NativeColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? 0U : state->coldPageBytes;
}

kv::LayerGroupId NativeColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::LayerGroupId{-1} : layerGroupId;
}

kv::PageIndexLocation NativeColdPageCodec::queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? kv::PageIndexLocation::kBadLocation : state->pageIndexLocation;
}

bool NativeColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && (dstBasePtr == nullptr || pageIndices == nullptr)))
        {
            throw std::invalid_argument("encode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (!state->backendIndex)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }
        mBackend->encode(*state->backendIndex, dstBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && (srcBasePtr == nullptr || pageIndices == nullptr)))
        {
            throw std::invalid_argument("decode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (!state->backendIndex)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }
        mBackend->decode(*state->backendIndex, srcBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: unknown error");
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
