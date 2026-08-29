/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/batch_manager/kv_cache_compression/nativeColdPageCodec.h"

#include "tensorrt_llm/common/assert.h"
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

void drainAfterProviderFailure(cudaStream_t stream) noexcept
{
    auto const status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess)
    {
        TLLM_LOG_ERROR("Cold-page provider rollback drain failed: %s", cudaGetErrorString(status));
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
        std::vector<ResolvedHotLifecycle> providerLifecycles;
        std::set<kv::LayerId> consumedLayers;

        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto resolved = resolveLifecycle(gpuDesc, variant);
                auto const providerLayerCount = std::count_if(resolved.layers.begin(), resolved.layers.end(),
                    [this](auto const& layer) { return mLayerIds.count(layer.first) != 0U; });

                LayerGroupState state;
                if (providerLayerCount == 0U)
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                    state.pageIndexLocation = losslessCodec->queryPageIndexLocation(variant.lifeCycleId);
                }
                else
                {
                    if (providerLayerCount != resolved.layers.size())
                    {
                        throw std::invalid_argument("A lifecycle cannot mix provider-owned and fallback layers");
                    }
                    for (auto const& [layerId, buffers] : resolved.layers)
                    {
                        static_cast<void>(buffers);
                        if (!consumedLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("A provider layer appears in multiple lifecycles");
                        }
                    }
                    state.lifecycleIndex = providerLifecycles.size();
                    providerLifecycles.push_back(std::move(resolved));
                }

                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (consumedLayers != mLayerIds)
        {
            throw std::invalid_argument("A provider layer is absent from all GPU descriptors");
        }

        auto const properties = configureProvider(providerLifecycles);
        if (properties.size() != providerLifecycles.size())
        {
            throw std::invalid_argument("Cold-page provider returned an unexpected lifecycle count");
        }
        for (std::size_t index = 0; index < properties.size(); ++index)
        {
            auto const& lifecycle = properties[index];
            if (lifecycle.coldPageBytes == 0U || lifecycle.pageIndexLocation == kv::PageIndexLocation::kBadLocation)
            {
                throw std::invalid_argument("Cold-page provider returned invalid storage properties");
            }
            auto& state = pendingGroups.at(providerLifecycles[index].lifeCycleId);
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

bool NativeColdPageCodec::needsHostMemRegistration() const noexcept
{
    return mLosslessCodec != nullptr && mLosslessCodec->needsHostMemRegistration();
}

void NativeColdPageCodec::registerHostMem(kv::HostMem const* memory)
{
    TLLM_CHECK(mLosslessCodec != nullptr);
    mLosslessCodec->registerHostMem(memory);
}

bool NativeColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    bool providerStarted = false;
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
        providerStarted = true;
        encodeProvider(*state->lifecycleIndex, dstBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::encode failed before completion fencing: unknown error");
        return false;
    }
}

bool NativeColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    bool providerStarted = false;
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
        providerStarted = true;
        decodeProvider(*state->lifecycleIndex, srcBasePtr, pageIndices, numBasePages, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        if (providerStarted)
        {
            drainAfterProviderFailure(stream);
        }
        TLLM_LOG_ERROR("NativeColdPageCodec::decode failed before completion fencing: unknown error");
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
