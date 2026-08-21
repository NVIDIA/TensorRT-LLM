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
#include "tensorrt_llm/common/nvtxUtils.h"

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

constexpr std::size_t kCompactAlignment = 16U;
constexpr std::size_t kElementsPerBlockScale = 16U;
constexpr std::size_t kPackedElementsPerByte = 2U;
constexpr char const* kKeyRole = "key";
constexpr char const* kValueRole = "value";

std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1U))
    {
        throw std::overflow_error("Cold Page size overflows size_t");
    }
    return (value + alignment - 1U) / alignment * alignment;
}

std::size_t checkedMul(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (rhs != 0U && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs * rhs;
}

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw std::overflow_error(label);
    }
    return lhs + rhs;
}

std::size_t scalarCount(Nvfp4ColdPageLayerConfig const& config)
{
    if (config.numKvHeads <= 0 || config.tokensPerPage <= 0 || config.headDim <= 0)
    {
        throw std::invalid_argument("NVFP4 cold Page geometry must be positive");
    }
    if (config.headDim % static_cast<std::int32_t>(kElementsPerBlockScale) != 0)
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

std::uintptr_t checkedAddress(std::uintptr_t base, std::size_t offset)
{
    if (offset > std::numeric_limits<std::uintptr_t>::max() - base)
    {
        throw std::overflow_error("GPU buffer address overflows uintptr_t");
    }
    return base + offset;
}

} // namespace

Nvfp4ColdPageCodec::Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs)
{
    for (auto& config : layerConfigs)
    {
        static_cast<void>(scalarCount(config));
        if (config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kBfloat16
            && config.runtimeType != kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3)
        {
            throw std::invalid_argument("Nvfp4ColdPageCodec received an unsupported runtime type");
        }
        auto const validScales = [](auto const& scales) {
            return std::all_of(
                scales.begin(), scales.end(), [](float value) { return std::isfinite(value) && value > 0; });
        };
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

bool Nvfp4ColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        struct BufferLocation
        {
            kv::PoolIndex poolIndex{0};
            std::size_t offset = 0;
            std::size_t bytes = 0;
            bool found = false;
        };

        struct AttentionBuffers
        {
            BufferLocation key;
            BufferLocation value;
            std::vector<BufferLocation> losslessBuffers;
        };

        // Use KVCM's default codec for non-Attention lifecycles.
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        std::map<kv::LayerGroupId, LayerGroupState> pending;
        std::set<kv::LayerId> configuredAttentionLayers;
        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                std::map<kv::LayerId, AttentionBuffers> attentionBuffers;
                bool hasForeignLayerBuffer = false;
                for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
                {
                    auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
                    std::size_t offset = 0U;
                    for (auto const& bufferId : coalesced.bufferIds)
                    {
                        auto const config = mLayerConfigs.find(bufferId.layerId);
                        if (config == mLayerConfigs.end())
                        {
                            hasForeignLayerBuffer = true;
                        }
                        else
                        {
                            configuredAttentionLayers.insert(bufferId.layerId);
                            auto& buffers = attentionBuffers[bufferId.layerId];
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
                                buffers.losslessBuffers.push_back(
                                    BufferLocation{poolIndex, offset, coalesced.singleBufferSize, true});
                            }
                        }
                        offset
                            = checkedAdd(offset, coalesced.singleBufferSize, "GPU Slot buffer offsets overflow size_t");
                    }
                }

                LayerGroupState state;
                if (!attentionBuffers.empty())
                {
                    if (hasForeignLayerBuffer)
                    {
                        throw std::invalid_argument("Attention lifecycle mixes configured and unconfigured layers");
                    }
                    state.transform = Transform::kNvfp4Attention;
                    std::vector<kernels::Nvfp4BoundaryBufferPlan> plans;
                    auto const runtimeType = mLayerConfigs.at(attentionBuffers.begin()->first).runtimeType;
                    for (auto const& [layerId, buffers] : attentionBuffers)
                    {
                        if (!buffers.key.found)
                        {
                            throw std::invalid_argument("Configured Attention layer is missing key");
                        }
                        auto const& config = mLayerConfigs.at(layerId);
                        if (runtimeType != config.runtimeType)
                        {
                            throw std::invalid_argument("Attention lifecycle must use one runtime dtype");
                        }

                        auto const elements = scalarCount(config);
                        auto const rawElementBytes
                            = config.runtimeType == kernels::Nvfp4BoundaryRuntimeType::kFp8E4m3 ? 1U : 2U;
                        auto const rawBytes
                            = checkedMul(elements, rawElementBytes, "Runtime KV Page size overflows size_t");
                        if (buffers.key.bytes != rawBytes || (buffers.value.found && buffers.value.bytes != rawBytes))
                        {
                            throw std::invalid_argument("GPU Attention buffer size does not match its geometry");
                        }

                        auto const layerOffset = state.coldPageBytes;
                        auto const packedBytesPerBuffer = elements / kPackedElementsPerByte;
                        auto const scaleBytesPerBuffer = elements / kElementsPerBlockScale;
                        auto const compressedBufferCount = buffers.value.found ? 2U : 1U;
                        auto const scaleOffset = checkedAdd(layerOffset,
                            checkedMul(
                                packedBytesPerBuffer, compressedBufferCount, "NVFP4 packed Page size overflows size_t"),
                            "NVFP4 cold Page size overflows size_t");

                        auto appendNvfp4 = [&](BufferLocation const& location, std::size_t scaleIndex,
                                               std::size_t coldDataOffset, std::size_t coldScaleOffset)
                        {
                            auto const& pool = gpuDesc.pools.at(location.poolIndex);
                            plans.push_back(
                                kernels::Nvfp4BoundaryBufferPlan{checkedAddress(pool.baseAddress, location.offset),
                                    pool.slotBytes, location.bytes, coldDataOffset, coldScaleOffset, 0U, 0U,
                                    kernels::Nvfp4BoundaryTransform::kNvfp4, makeKernelParams(config, scaleIndex)});
                        };

                        appendNvfp4(buffers.key, 0U, layerOffset, scaleOffset);
                        if (buffers.value.found)
                        {
                            appendNvfp4(buffers.value, 1U,
                                checkedAdd(layerOffset, packedBytesPerBuffer, "NVFP4 cold Page size overflows size_t"),
                                checkedAdd(scaleOffset, scaleBytesPerBuffer, "NVFP4 cold Page size overflows size_t"));
                        }

                        auto coldOffset = checkedAdd(scaleOffset,
                            checkedMul(
                                scaleBytesPerBuffer, compressedBufferCount, "NVFP4 scale Page size overflows size_t"),
                            "NVFP4 cold Page size overflows size_t");
                        for (auto const& location : buffers.losslessBuffers)
                        {
                            auto const& pool = gpuDesc.pools.at(location.poolIndex);
                            plans.push_back(kernels::Nvfp4BoundaryBufferPlan{
                                checkedAddress(pool.baseAddress, location.offset), pool.slotBytes, location.bytes,
                                coldOffset, 0U, 0U, 0U, kernels::Nvfp4BoundaryTransform::kLossless, {}});
                            coldOffset = checkedAdd(
                                coldOffset, location.bytes, "Lossless Attention side-buffer size overflows size_t");
                        }

                        auto const alignedEnd = alignUp(coldOffset, kCompactAlignment);
                        auto const paddingBytes = alignedEnd - coldOffset;
                        plans.back().coldPaddingOffset = coldOffset;
                        plans.back().coldPaddingBytes = static_cast<std::uint32_t>(paddingBytes);
                        state.coldPageBytes = alignedEnd;
                    }
                    state.preparedPlan = kernels::prepareNvfp4BoundaryPlan(plans, state.coldPageBytes, runtimeType);
                }
                else
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                }
                pending.emplace(variant.lifeCycleId, std::move(state));
            }
        }
        if (configuredAttentionLayers.size() != mLayerConfigs.size())
        {
            throw std::invalid_argument("A configured Attention layer is absent from all GPU descriptors");
        }

        mLayerGroups = std::move(pending);
        mLosslessCodec = std::move(losslessCodec);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("Nvfp4ColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
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

Nvfp4ColdPageCodec::LayerGroupState const* Nvfp4ColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
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
        if (state->transform == Transform::kLosslessConcat)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }

        NVTX3_SCOPED_RANGE(KVCC_OFFLOAD_COMPRESS_D2H);
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
        if (state->transform == Transform::kLosslessConcat)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }

        NVTX3_SCOPED_RANGE(KVCC_ONBOARD_H2D_DECOMPRESS);
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
