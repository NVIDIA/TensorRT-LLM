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

#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodec.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

constexpr std::size_t kColdRecordAlignment = 16U;
constexpr std::size_t kElementsPerScaleGroup = 16U;
constexpr std::size_t kPackedElementsPerByte = 2U;
constexpr char const* kKeyRole = "key";
constexpr char const* kValueRole = "value";

// Descriptor-derived byte arithmetic must not wrap into an undersized cold record.
std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs * rhs;
}

std::uintptr_t checkedAddress(std::uintptr_t base, std::size_t offset)
{
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base)
    {
        throw std::overflow_error("GPU buffer address overflows uintptr_t");
    }
    return base + offset;
}

std::size_t alignColdRecord(std::size_t bytes)
{
    return checkedAdd(bytes, kColdRecordAlignment - 1U, "Cold Page size overflows size_t") / kColdRecordAlignment
        * kColdRecordAlignment;
}

std::size_t checkedElementCount(Nvfp4ColdPageLayerConfig const& config)
{
    if (config.numKvHeads <= 0 || config.tokensPerPage <= 0 || config.headDim <= 0)
    {
        throw std::invalid_argument("NVFP4 cold Page geometry must be positive");
    }
    if (config.headDim % static_cast<std::int32_t>(kElementsPerScaleGroup) != 0)
    {
        throw std::invalid_argument("NVFP4 cold Pages require headDim divisible by 16");
    }
    auto const headsTimesTokens = checkedMul(static_cast<std::size_t>(config.numKvHeads),
        static_cast<std::size_t>(config.tokensPerPage), "NVFP4 Page geometry overflows size_t");
    return checkedMul(
        headsTimesTokens, static_cast<std::size_t>(config.headDim), "NVFP4 Page geometry overflows size_t");
}

kernels::Nvfp4BoundaryKernelParams makeKernelParams(Nvfp4ColdPageLayerConfig const& config, std::size_t scaleIndex)
{
    kernels::Nvfp4BoundaryKernelParams params{};
    params.numKvHeads = config.numKvHeads;
    params.tokensPerPage = config.tokensPerPage;
    params.headDim = config.headDim;
    params.nvfp4ScaleOrigQuant = config.nvfp4ScaleOrigQuant[scaleIndex];
    params.nvfp4ScaleQuantOrig = config.nvfp4ScaleQuantOrig[scaleIndex];
    params.fp8ScaleOrigQuant = config.fp8ScaleOrigQuant[scaleIndex];
    params.fp8ScaleQuantOrig = config.fp8ScaleQuantOrig[scaleIndex];
    return params;
}

struct BufferLocation
{
    kv::PoolIndex poolIndex{0};
    std::size_t offset = 0;
    std::size_t bytes = 0;
    bool found = false;
};

struct AttentionLayerBuffers
{
    BufferLocation key;
    BufferLocation value;
    // Roles such as MLA index_key remain byte-exact.
    std::vector<BufferLocation> sideBuffers;
};

using LayerConfigs = std::map<kv::LayerId, Nvfp4ColdPageLayerConfig>;
using AttentionBufferMap = std::map<kv::LayerId, AttentionLayerBuffers>;

struct LifecycleBuffers
{
    AttentionBufferMap attention;
    bool hasNonAttentionLayer = false;
};

LifecycleBuffers discoverLifecycleBuffers(kv::SlotDescVariant const& variant, LayerConfigs const& layerConfigs)
{
    LifecycleBuffers result;
    for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
    {
        auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
        std::size_t offset = 0U;
        for (auto const& bufferId : coalesced.bufferIds)
        {
            auto const config = layerConfigs.find(bufferId.layerId);
            if (config == layerConfigs.end())
            {
                result.hasNonAttentionLayer = true;
            }
            else
            {
                auto& buffers = result.attention[bufferId.layerId];
                if (bufferId.role == kKeyRole || bufferId.role == kValueRole)
                {
                    auto& location = bufferId.role == kKeyRole ? buffers.key : buffers.value;
                    if (location.found)
                    {
                        throw std::invalid_argument("GPU lifecycle contains a duplicate K/V buffer");
                    }
                    location = BufferLocation{poolIndex, offset, coalesced.singleBufferSize, true};
                }
                else
                {
                    buffers.sideBuffers.push_back(BufferLocation{poolIndex, offset, coalesced.singleBufferSize, true});
                }
            }
            offset = checkedAdd(offset, coalesced.singleBufferSize, "GPU Slot buffer offsets overflow size_t");
        }
    }
    return result;
}

struct AttentionPlan
{
    kernels::Nvfp4BoundaryPreparedPlan kernelPlan;
    std::size_t coldPageBytes = 0;
};

AttentionPlan buildAttentionPlan(
    kv::PoolGroupDesc const& gpuDesc, AttentionBufferMap const& layerBuffers, LayerConfigs const& layerConfigs)
{
    AttentionPlan result;
    std::vector<kernels::Nvfp4BoundaryBufferPlan> bufferPlans;
    auto const runtimeType = layerConfigs.at(layerBuffers.begin()->first).runtimeType;

    for (auto const& [layerId, buffers] : layerBuffers)
    {
        // MHA/GQA has K and V; MLA exposes its latent KV as key only.
        if (!buffers.key.found)
        {
            throw std::invalid_argument("Configured Attention layer is missing key");
        }
        auto const& config = layerConfigs.at(layerId);
        if (runtimeType != config.runtimeType)
        {
            throw std::invalid_argument("Attention lifecycle must use one runtime dtype");
        }

        auto const elements = checkedElementCount(config);
        auto const layerOffset = result.coldPageBytes;
        auto const packedBytes = elements / kPackedElementsPerByte;
        auto const scaleBytes = elements / kElementsPerScaleGroup;
        auto const compressedBufferCount = buffers.value.found ? 2U : 1U;
        auto const packedRegionBytes = packedBytes * compressedBufferCount;
        auto const scaleRegionBytes = scaleBytes * compressedBufferCount;

        // Per layer: [K packed | V packed? | K scale | V scale? | side buffers | padding].
        auto const scaleOffset = checkedAdd(layerOffset, packedRegionBytes, "NVFP4 cold Page size overflows size_t");
        auto coldOffset = checkedAdd(scaleOffset, scaleRegionBytes, "NVFP4 cold Page size overflows size_t");

        auto appendNvfp4 = [&](BufferLocation const& location, std::size_t scaleIndex, std::size_t coldDataOffset,
                               std::size_t coldScaleOffset)
        {
            auto const& pool = gpuDesc.pools.at(location.poolIndex);
            bufferPlans.push_back(kernels::Nvfp4BoundaryBufferPlan{checkedAddress(pool.baseAddress, location.offset),
                pool.slotBytes, location.bytes, coldDataOffset, coldScaleOffset, 0U, 0U,
                kernels::Nvfp4BoundaryTransform::kNvfp4, makeKernelParams(config, scaleIndex)});
        };

        appendNvfp4(buffers.key, 0U, layerOffset, scaleOffset);
        if (buffers.value.found)
        {
            appendNvfp4(buffers.value, 1U, layerOffset + packedBytes, scaleOffset + scaleBytes);
        }
        for (auto const& location : buffers.sideBuffers)
        {
            auto const& pool = gpuDesc.pools.at(location.poolIndex);
            bufferPlans.push_back(
                kernels::Nvfp4BoundaryBufferPlan{checkedAddress(pool.baseAddress, location.offset), pool.slotBytes,
                    location.bytes, coldOffset, 0U, 0U, 0U, kernels::Nvfp4BoundaryTransform::kLosslessCopy, {}});
            coldOffset = checkedAdd(coldOffset, location.bytes, "Lossless Attention side-buffer size overflows size_t");
        }

        auto const alignedEnd = alignColdRecord(coldOffset);
        bufferPlans.back().coldPaddingOffset = coldOffset;
        bufferPlans.back().coldPaddingBytes = static_cast<std::uint32_t>(alignedEnd - coldOffset);
        result.coldPageBytes = alignedEnd;
    }

    result.kernelPlan = kernels::prepareNvfp4BoundaryPlan(bufferPlans, result.coldPageBytes, runtimeType);
    return result;
}

} // namespace

// Validate and retain algorithm-owned metadata before KVCM creates its physical pools.
Nvfp4ColdPageCodec::Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs)
{
    auto const validScales = [](auto const& scales) {
        return std::all_of(scales.begin(), scales.end(), [](float value) { return std::isfinite(value) && value > 0; });
    };
    for (auto& config : layerConfigs)
    {
        if (config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kBfloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec received an unsupported runtime type");
        }
        static_cast<void>(checkedElementCount(config));
        if (!validScales(config.nvfp4ScaleOrigQuant) || !validScales(config.nvfp4ScaleQuantOrig)
            || (config.runtimeType == kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3
                && (!validScales(config.fp8ScaleOrigQuant) || !validScales(config.fp8ScaleQuantOrig))))
        {
            throw std::invalid_argument("NVFP4 runtime scales must be finite and positive");
        }
        auto const layerId = config.layerId;
        if (!mLayerConfigs.emplace(layerId, std::move(config)).second)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec layer IDs must be unique");
        }
    }
}

// Bind the immutable metadata to KVCM's authoritative hot-pool layout exactly once.
bool Nvfp4ColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        // Use KVCM's default codec for non-Attention lifecycles.
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        std::map<kv::LayerGroupId, LayerGroupState> pendingGroups;
        std::set<kv::LayerId> discoveredAttentionLayers;
        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto const buffers = discoverLifecycleBuffers(variant, mLayerConfigs);

                LayerGroupState state;
                if (!buffers.attention.empty())
                {
                    if (buffers.hasNonAttentionLayer)
                    {
                        throw std::invalid_argument("A lifecycle cannot mix Attention and non-Attention layers");
                    }

                    auto plan = buildAttentionPlan(gpuDesc, buffers.attention, mLayerConfigs);
                    state.format = ColdPageFormat::kNvfp4Kv;
                    state.preparedPlan = std::move(plan.kernelPlan);
                    state.coldPageBytes = plan.coldPageBytes;
                    for (auto const& layer : buffers.attention)
                    {
                        discoveredAttentionLayers.insert(layer.first);
                    }
                }
                else
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                }
                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (discoveredAttentionLayers.size() != mLayerConfigs.size())
        {
            throw std::invalid_argument("A configured Attention layer is absent from all GPU descriptors");
        }

        mLayerGroups = std::move(pendingGroups);
        mLosslessCodec = std::move(losslessCodec);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
}

Nvfp4ColdPageCodec::LayerGroupState const* Nvfp4ColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
}

std::size_t Nvfp4ColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? 0U : state->coldPageBytes;
}

kv::LayerGroupId Nvfp4ColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::LayerGroupId{-1} : layerGroupId;
}

kv::PageIndexLocation Nvfp4ColdPageCodec::queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::PageIndexLocation::kBadLocation : kv::PageIndexLocation::kHost;
}

bool Nvfp4ColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
    std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && pageIndices == nullptr))
        {
            throw std::invalid_argument("encode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (state->format == ColdPageFormat::kLossless)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }

        thread_local std::vector<kernels::Nvfp4BoundaryOffloadPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].src, pageIndices[page].dst});
        }
        kernels::invokeNvfp4BoundaryOffloadCompress(pages, state->preparedPlan, dstBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
}

bool Nvfp4ColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
    kv::PageIndexPair const* pageIndices, std::size_t numBasePages, cudaStream_t stream) noexcept
{
    try
    {
        auto const* state = findLayerGroup(layerGroupId);
        if (state == nullptr || (numBasePages != 0U && pageIndices == nullptr))
        {
            throw std::invalid_argument("decode received an invalid lifecycle or Page batch");
        }
        if (numBasePages == 0U)
        {
            return true;
        }
        if (state->format == ColdPageFormat::kLossless)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }

        thread_local std::vector<kernels::Nvfp4BoundaryOnboardPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].dst, pageIndices[page].src});
        }
        kernels::invokeNvfp4BoundaryOnboardDecompress(pages, state->preparedPlan, srcBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
