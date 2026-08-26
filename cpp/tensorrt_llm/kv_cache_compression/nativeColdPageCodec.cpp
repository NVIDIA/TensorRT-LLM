/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <exception>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

ResolvedHotLifecycle resolveLifecycle(kv::PoolGroupDesc const& gpuDesc, kv::SlotDescVariant const& variant)
{
    ResolvedHotLifecycle result{variant.lifeCycleId, {}};
    for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
    {
        auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
        auto const& pool = gpuDesc.pools.at(poolIndex);
        std::size_t offset = 0;
        for (auto const& bufferId : coalesced.bufferIds)
        {
            auto& layer = result.layers[bufferId.layerId];
            if (!layer
                     .emplace(bufferId.role,
                         ResolvedHotBuffer{pool.baseAddress + offset, pool.slotBytes, coalesced.singleBufferSize})
                     .second)
            {
                throw std::invalid_argument("GPU lifecycle contains a duplicate buffer role");
            }
            offset += coalesced.singleBufferSize;
        }
    }
    return result;
}

void drainAfterPolicyFailure(cudaStream_t stream) noexcept
{
    auto const status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        TLLM_LOG_ERROR("Cold-page policy rollback drain failed: %s", cudaGetErrorString(status));
        std::terminate();
    }
}

} // namespace

NativeColdPageCodec::NativeColdPageCodec(std::set<kv::LayerId> layerIds)
    : mLayerIds(std::move(layerIds))
{
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

        std::map<kv::LayerGroupId, LayerGroupState> pendingGroups;
        std::vector<ResolvedHotLifecycle> policyLifecycles;
        std::set<kv::LayerId> consumedLayers;

        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto resolved = resolveLifecycle(gpuDesc, variant);
                auto const policyLayerCount = std::count_if(resolved.layers.begin(), resolved.layers.end(),
                    [this](auto const& layer) { return mLayerIds.count(layer.first) != 0U; });

                LayerGroupState state;
                if (policyLayerCount == 0U)
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                    state.pageIndexLocation = losslessCodec->queryPageIndexLocation(variant.lifeCycleId);
                }
                else
                {
                    if (policyLayerCount != resolved.layers.size())
                    {
                        throw std::invalid_argument("A lifecycle cannot mix policy-owned and fallback layers");
                    }
                    for (auto const& [layerId, buffers] : resolved.layers)
                    {
                        static_cast<void>(buffers);
                        if (!consumedLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("A policy layer appears in multiple lifecycles");
                        }
                    }
                    state.lifecycleIndex = policyLifecycles.size();
                    policyLifecycles.push_back(std::move(resolved));
                }

                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (consumedLayers != mLayerIds)
        {
            throw std::invalid_argument("A policy layer is absent from all GPU descriptors");
        }

        auto const properties = configurePolicy(policyLifecycles);
        if (properties.size() != policyLifecycles.size())
        {
            throw std::invalid_argument("Cold-page policy returned an unexpected lifecycle count");
        }
        for (std::size_t index = 0; index < properties.size(); ++index)
        {
            auto const& lifecycle = properties[index];
            if (lifecycle.coldPageBytes == 0U || lifecycle.pageIndexLocation == kv::PageIndexLocation::kBadLocation)
            {
                throw std::invalid_argument("Cold-page policy returned invalid storage properties");
            }
            auto& state = pendingGroups.at(policyLifecycles[index].lifeCycleId);
            state.coldPageBytes = lifecycle.coldPageBytes;
            state.pageIndexLocation = lifecycle.pageIndexLocation;
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
    bool policyStarted = false;
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
        if (!state->lifecycleIndex)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }
        policyStarted = true;
        encodePolicy(*state->lifecycleIndex, dstBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (policyStarted)
        {
            drainAfterPolicyFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (policyStarted)
        {
            drainAfterPolicyFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    bool policyStarted = false;
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
        if (!state->lifecycleIndex)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }
        policyStarted = true;
        decodePolicy(*state->lifecycleIndex, srcBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (policyStarted)
        {
            drainAfterPolicyFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (policyStarted)
        {
            drainAfterPolicyFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: unknown error");
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
