/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/coldPageCodec.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace tensorrt_llm::kv_cache_compression
{
namespace
{

std::size_t checkedAdd(std::size_t lhs, std::size_t rhs, char const* label)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        throw std::overflow_error(label);
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

struct BufferLocation
{
    kv::PoolIndex poolIndex{0};
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

using LayerBuffers = std::map<kv::DataRole, BufferLocation>;
using LifecycleBuffers = std::map<kv::LayerId, LayerBuffers>;
using LayerPlans = std::map<kv::LayerId, ColdPageLayerPlan>;

LifecycleBuffers indexLifecycleBuffers(kv::SlotDescVariant const& variant)
{
    LifecycleBuffers result;
    for (kv::PoolIndex poolIndex{0}; poolIndex < variant.coalescedBuffers.size(); ++poolIndex)
    {
        auto const& coalesced = variant.coalescedBuffers.at(poolIndex);
        std::size_t offset = 0;
        for (auto const& bufferId : coalesced.bufferIds)
        {
            auto& buffers = result[bufferId.layerId];
            if (!buffers.emplace(bufferId.role, BufferLocation{poolIndex, offset, coalesced.singleBufferSize}).second)
            {
                throw std::invalid_argument("GPU lifecycle contains a duplicate buffer role");
            }
            offset = checkedAdd(offset, coalesced.singleBufferSize, "GPU Slot buffer offsets overflow size_t");
        }
    }
    return result;
}

kernels::Nvfp4ColdPageKernelParams toKernelParams(Nvfp4ColdPageParams const& params)
{
    return kernels::Nvfp4ColdPageKernelParams{params.numKvHeads, params.tokensPerPage, params.headDim,
        params.nvfp4ScaleOrigQuant, params.nvfp4ScaleQuantOrig, params.fp8ScaleOrigQuant, params.fp8ScaleQuantOrig};
}

struct CompiledLayerGroup
{
    kernels::Nvfp4ColdPagePreparedPlan preparedPlan;
    std::size_t coldPageBytes = 0;
    std::set<kv::LayerId> consumedLayers;
};

CompiledLayerGroup compileLayerGroup(
    kv::PoolGroupDesc const& gpuDesc, LifecycleBuffers const& physicalLayers, LayerPlans const& layerPlans)
{
    CompiledLayerGroup result;
    std::vector<kernels::Nvfp4ColdPageBufferPlan> groupBuffers;
    std::optional<kernels::Nvfp4ColdPageRuntimeType> groupRuntimeType;

    for (auto const& [layerId, physicalBuffers] : physicalLayers)
    {
        auto const planIt = layerPlans.find(layerId);
        if (planIt == layerPlans.end())
        {
            throw std::invalid_argument("A planned lifecycle contains a layer without a cold-page plan");
        }
        auto const& layerPlan = planIt->second;
        std::vector<kernels::Nvfp4ColdPageBufferPlan> layerBuffers;
        layerBuffers.reserve(layerPlan.buffers.size());
        std::optional<kernels::Nvfp4ColdPageRuntimeType> layerRuntimeType;

        for (auto const& bufferPlan : layerPlan.buffers)
        {
            auto const locationIt = physicalBuffers.find(bufferPlan.role);
            if (locationIt == physicalBuffers.end())
            {
                throw std::invalid_argument(
                    "Cold-page plan references a buffer "
                    "absent from the GPU lifecycle");
            }
            auto const& location = locationIt->second;
            if (bufferPlan.rawBytes != location.bytes)
            {
                throw std::invalid_argument("Cold-page plan raw size does not match the GPU buffer size");
            }

            auto const& pool = gpuDesc.pools.at(location.poolIndex);
            kernels::Nvfp4ColdPageBufferPlan nativePlan{checkedAddress(pool.baseAddress, location.offset),
                pool.slotBytes, bufferPlan.rawBytes, bufferPlan.coldDataOffset, bufferPlan.coldScaleOffset, 0U, 0U,
                kernels::Nvfp4ColdPageTransform::kLosslessCopy, {}};
            switch (bufferPlan.transform)
            {
            case ColdPageTransformKind::kLosslessCopy: break;
            case ColdPageTransformKind::kNvfp4:
                if (!bufferPlan.nvfp4Params)
                {
                    throw std::invalid_argument("NVFP4 cold-page transform requires NVFP4 parameters");
                }
                nativePlan.transform = kernels::Nvfp4ColdPageTransform::kNvfp4;
                nativePlan.params = toKernelParams(*bufferPlan.nvfp4Params);
                if (layerRuntimeType && *layerRuntimeType != bufferPlan.nvfp4Params->runtimeType)
                {
                    throw std::invalid_argument("One layer cold-page plan must use one runtime type");
                }
                layerRuntimeType = bufferPlan.nvfp4Params->runtimeType;
                break;
            default: throw std::invalid_argument("Unsupported cold-page transform kind");
            }
            layerBuffers.push_back(nativePlan);
        }
        if (layerBuffers.size() != physicalBuffers.size())
        {
            throw std::invalid_argument("Cold-page layer plan must cover every GPU buffer role");
        }

        layerBuffers.back().coldPaddingOffset = layerPlan.coldPaddingOffset;
        layerBuffers.back().coldPaddingBytes = static_cast<std::uint32_t>(layerPlan.coldPaddingBytes);
        auto const runtimeType = layerRuntimeType.value_or(kernels::Nvfp4ColdPageRuntimeType::kFloat16);
        auto const validatedLayer
            = kernels::prepareNvfp4ColdPagePlan(layerBuffers, layerPlan.coldPageBytes, runtimeType);

        if (layerRuntimeType)
        {
            if (groupRuntimeType && *groupRuntimeType != *layerRuntimeType)
            {
                throw std::invalid_argument("One lifecycle cold-page plan must use one runtime type");
            }
            groupRuntimeType = *layerRuntimeType;
        }
        for (std::uint32_t index = 0; index < validatedLayer.numBuffers; ++index)
        {
            auto nativePlan = validatedLayer.buffers[index];
            nativePlan.coldDataOffset
                = checkedAdd(result.coldPageBytes, nativePlan.coldDataOffset, "Cold-page data offset overflows size_t");
            if (nativePlan.transform == kernels::Nvfp4ColdPageTransform::kNvfp4)
            {
                nativePlan.coldScaleOffset = checkedAdd(
                    result.coldPageBytes, nativePlan.coldScaleOffset, "Cold-page scale offset overflows size_t");
            }
            if (nativePlan.coldPaddingBytes != 0U)
            {
                nativePlan.coldPaddingOffset = checkedAdd(
                    result.coldPageBytes, nativePlan.coldPaddingOffset, "Cold-page padding offset overflows size_t");
            }
            groupBuffers.push_back(nativePlan);
        }
        result.coldPageBytes
            = checkedAdd(result.coldPageBytes, layerPlan.coldPageBytes, "Lifecycle cold-page size overflows size_t");
        result.consumedLayers.insert(layerId);
    }

    result.preparedPlan = kernels::prepareNvfp4ColdPagePlan(
        groupBuffers, result.coldPageBytes, groupRuntimeType.value_or(kernels::Nvfp4ColdPageRuntimeType::kFloat16));
    return result;
}

} // namespace

PlannedColdPageCodec::PlannedColdPageCodec(std::vector<ColdPageLayerPlan> layerPlans)
{
    for (auto& layerPlan : layerPlans)
    {
        if (layerPlan.coldPageBytes == 0U || layerPlan.buffers.empty())
        {
            throw std::invalid_argument("A cold-page layer plan must contain a non-empty record");
        }
        if (layerPlan.coldPaddingOffset > layerPlan.coldPageBytes
            || layerPlan.coldPaddingBytes != layerPlan.coldPageBytes - layerPlan.coldPaddingOffset
            || layerPlan.coldPaddingBytes > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::invalid_argument("Cold-page layer padding exceeds its record");
        }

        std::set<kv::DataRole> roles;
        for (auto const& buffer : layerPlan.buffers)
        {
            if (buffer.role.empty() || buffer.rawBytes == 0U || !roles.emplace(buffer.role).second)
            {
                throw std::invalid_argument(
                    "Cold-page layer plans require unique "
                    "non-empty buffer roles and sizes");
            }
            if (buffer.transform != ColdPageTransformKind::kLosslessCopy
                && buffer.transform != ColdPageTransformKind::kNvfp4)
            {
                throw std::invalid_argument("Unsupported cold-page transform kind");
            }
        }
        auto const layerId = layerPlan.layerId;
        if (!mLayerPlans.emplace(layerId, std::move(layerPlan)).second)
        {
            throw std::invalid_argument("Cold-page layer plan IDs must be unique");
        }
    }
}

bool PlannedColdPageCodec::configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept
{
    try
    {
        if (mLosslessCodec || !mLayerGroups.empty())
        {
            throw std::invalid_argument("PlannedColdPageCodec can be configured only once");
        }
        auto losslessCodec = kv::createDefaultKvCacheColdPageCodec();
        if (!losslessCodec->configure(gpuDescs, numGpuDescs))
        {
            throw std::invalid_argument("Default lossless codec rejected GPU layouts");
        }

        std::map<kv::LayerGroupId, LayerGroupState> pendingGroups;
        std::set<kv::LayerId> consumedLayers;
        for (kv::PoolGroupIndex poolGroupIndex{0}; poolGroupIndex < numGpuDescs; ++poolGroupIndex)
        {
            auto const& gpuDesc = gpuDescs[kv::toSizeT(poolGroupIndex)];
            for (auto const& variant : gpuDesc.slotDesc.variants)
            {
                auto const physicalLayers = indexLifecycleBuffers(variant);
                auto const plannedLayerCount = std::count_if(physicalLayers.begin(), physicalLayers.end(),
                    [this](auto const& layer) { return mLayerPlans.count(layer.first) != 0U; });

                LayerGroupState state;
                if (plannedLayerCount == 0U)
                {
                    state.coldPageBytes = losslessCodec->queryColdPageBytes(variant.lifeCycleId);
                }
                else
                {
                    if (plannedLayerCount != physicalLayers.size())
                    {
                        throw std::invalid_argument("A lifecycle cannot mix planned and unplanned layers");
                    }
                    auto compiled = compileLayerGroup(gpuDesc, physicalLayers, mLayerPlans);
                    state.execution = ExecutionKind::kPlanned;
                    state.preparedPlan = std::move(compiled.preparedPlan);
                    state.coldPageBytes = compiled.coldPageBytes;
                    for (auto const layerId : compiled.consumedLayers)
                    {
                        if (!consumedLayers.emplace(layerId).second)
                        {
                            throw std::invalid_argument("A cold-page layer plan appears in multiple lifecycles");
                        }
                    }
                }
                if (!pendingGroups.emplace(variant.lifeCycleId, std::move(state)).second)
                {
                    throw std::invalid_argument("GPU lifecycle ID appears in multiple pool groups");
                }
            }
        }
        if (consumedLayers.size() != mLayerPlans.size())
        {
            throw std::invalid_argument("A cold-page layer plan is absent from all GPU descriptors");
        }

        mLayerGroups = std::move(pendingGroups);
        mLosslessCodec = std::move(losslessCodec);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("PlannedColdPageCodec::configure rejected GPU layouts: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR(
            "PlannedColdPageCodec::configure rejected GPU layouts: "
            "unknown error");
        return false;
    }
}

PlannedColdPageCodec::LayerGroupState const* PlannedColdPageCodec::findLayerGroup(
    kv::LayerGroupId layerGroupId) const noexcept
{
    auto const found = mLayerGroups.find(layerGroupId);
    return found == mLayerGroups.end() ? nullptr : &found->second;
}

std::size_t PlannedColdPageCodec::queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept
{
    auto const* state = findLayerGroup(layerGroupId);
    return state == nullptr ? 0U : state->coldPageBytes;
}

kv::LayerGroupId PlannedColdPageCodec::getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::LayerGroupId{-1} : layerGroupId;
}

kv::PageIndexLocation PlannedColdPageCodec::queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept
{
    return findLayerGroup(layerGroupId) == nullptr ? kv::PageIndexLocation::kBadLocation : kv::PageIndexLocation::kHost;
}

bool PlannedColdPageCodec::encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
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
        if (state->execution == ExecutionKind::kLossless)
        {
            return mLosslessCodec->encode(layerGroupId, dstBasePtr, pageIndices, numBasePages, stream);
        }

        thread_local std::vector<kernels::Nvfp4ColdPageOffloadPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].src, pageIndices[page].dst});
        }
        kernels::invokeNvfp4ColdPageEncode(pages, state->preparedPlan, dstBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("PlannedColdPageCodec::encode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR(
            "PlannedColdPageCodec::encode failed before completion "
            "fencing: unknown error");
        return false;
    }
}

bool PlannedColdPageCodec::decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr,
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
        if (state->execution == ExecutionKind::kLossless)
        {
            return mLosslessCodec->decode(layerGroupId, srcBasePtr, pageIndices, numBasePages, stream);
        }

        thread_local std::vector<kernels::Nvfp4ColdPageOnboardPageTask> pages;
        pages.clear();
        pages.reserve(numBasePages);
        for (std::size_t page = 0; page < numBasePages; ++page)
        {
            pages.push_back({pageIndices[page].dst, pageIndices[page].src});
        }
        kernels::invokeNvfp4ColdPageDecode(pages, state->preparedPlan, srcBasePtr, stream);
        return true;
    }
    catch (std::exception const& error)
    {
        TLLM_LOG_ERROR("PlannedColdPageCodec::decode failed before completion fencing: %s", error.what());
        return false;
    }
    catch (...)
    {
        TLLM_LOG_ERROR(
            "PlannedColdPageCodec::decode failed before completion "
            "fencing: unknown error");
        return false;
    }
}

} // namespace tensorrt_llm::kv_cache_compression
