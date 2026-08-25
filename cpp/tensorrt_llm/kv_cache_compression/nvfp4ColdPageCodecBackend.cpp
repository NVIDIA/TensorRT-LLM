/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kv_cache_compression/nvfp4ColdPageCodecBackend.h"

#include "tensorrt_llm/kv_cache_compression/nativeColdPageCodec.h"

#include <limits>
#include <map>
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

kernels::Nvfp4ColdPageKernelParams makeKernelParams(
    Nvfp4ColdPageLayerLayout const& layout, Nvfp4ColdPageScales const& scales)
{
    return kernels::Nvfp4ColdPageKernelParams{layout.numKvHeads, layout.tokensPerPage, layout.headDim,
        scales.nvfp4ScaleOrigQuant, scales.nvfp4ScaleQuantOrig, scales.fp8ScaleOrigQuant, scales.fp8ScaleQuantOrig};
}

class Nvfp4ColdPageCodecBackend final : public IColdPageCodecBackend
{
public:
    explicit Nvfp4ColdPageCodecBackend(std::vector<Nvfp4ColdPageLayerLayout> layerLayouts)
    {
        for (auto& layout : layerLayouts)
        {
            if (layout.coldPageBytes == 0U || layout.buffers.empty() || layout.coldPaddingOffset > layout.coldPageBytes)
            {
                throw std::invalid_argument("An NVFP4 layer layout must contain a valid cold-page record");
            }
            std::set<kv::DataRole> roles;
            for (auto const& buffer : layout.buffers)
            {
                if (buffer.role.empty() || !roles.emplace(buffer.role).second)
                {
                    throw std::invalid_argument("NVFP4 layer layouts require unique non-empty buffer roles");
                }
            }
            auto const layerId = layout.layerId;
            if (!mLayerLayouts.emplace(layerId, std::move(layout)).second)
            {
                throw std::invalid_argument("NVFP4 layer layout IDs must be unique");
            }
            mLayerIds.emplace(layerId);
        }
    }

    [[nodiscard]] std::set<kv::LayerId> const& getLayerIds() const noexcept override
    {
        return mLayerIds;
    }

    std::vector<ColdPageLifecycleConfig> configure(std::vector<ResolvedColdPageLifecycle> const& lifecycles) override
    {
        std::vector<kernels::Nvfp4ColdPagePreparedPlan> preparedPlans;
        std::vector<ColdPageLifecycleConfig> configs;
        preparedPlans.reserve(lifecycles.size());
        configs.reserve(lifecycles.size());

        for (auto const& lifecycle : lifecycles)
        {
            std::vector<kernels::Nvfp4ColdPageBufferPlan> buffers;
            std::optional<kernels::Nvfp4ColdPageRuntimeType> runtimeType;
            std::size_t coldPageBytes = 0;

            for (auto const& [layerId, resolvedBuffers] : lifecycle)
            {
                auto const& layout = mLayerLayouts.at(layerId);
                if (layout.buffers.size() != resolvedBuffers.size())
                {
                    throw std::invalid_argument("NVFP4 layer layout must cover every GPU buffer role");
                }
                if (runtimeType && *runtimeType != layout.runtimeType)
                {
                    throw std::invalid_argument("One lifecycle must use one NVFP4 runtime type");
                }
                runtimeType = layout.runtimeType;

                for (auto const& bufferLayout : layout.buffers)
                {
                    auto const resolved = resolvedBuffers.find(bufferLayout.role);
                    if (resolved == resolvedBuffers.end())
                    {
                        throw std::invalid_argument("NVFP4 layer layout references an absent GPU buffer role");
                    }
                    auto const& raw = resolved->second;
                    auto const transform = bufferLayout.scales ? kernels::Nvfp4ColdPageTransform::kNvfp4
                                                               : kernels::Nvfp4ColdPageTransform::kLosslessCopy;
                    buffers.push_back(kernels::Nvfp4ColdPageBufferPlan{raw.rawBase, raw.rawSlotBytes, raw.rawBytes,
                        checkedAdd(
                            coldPageBytes, bufferLayout.coldDataOffset, "Cold-page data offset overflows size_t"),
                        bufferLayout.scales ? checkedAdd(
                            coldPageBytes, bufferLayout.coldScaleOffset, "Cold-page scale offset overflows size_t")
                                            : 0U,
                        0U, 0U, transform,
                        bufferLayout.scales ? makeKernelParams(layout, *bufferLayout.scales)
                                            : kernels::Nvfp4ColdPageKernelParams{}});
                }

                auto const paddingBytes = layout.coldPageBytes - layout.coldPaddingOffset;
                if (paddingBytes > std::numeric_limits<std::uint32_t>::max())
                {
                    throw std::overflow_error("Cold-page padding exceeds the kernel ABI");
                }
                buffers.back().coldPaddingOffset
                    = checkedAdd(coldPageBytes, layout.coldPaddingOffset, "Cold-page padding offset overflows size_t");
                buffers.back().coldPaddingBytes = static_cast<std::uint32_t>(paddingBytes);
                coldPageBytes
                    = checkedAdd(coldPageBytes, layout.coldPageBytes, "Lifecycle cold-page size overflows size_t");
            }

            if (!runtimeType)
            {
                throw std::invalid_argument("NVFP4 backend received an empty lifecycle");
            }
            preparedPlans.push_back(kernels::prepareNvfp4ColdPagePlan(buffers, coldPageBytes, *runtimeType));
            configs.push_back({coldPageBytes, kv::PageIndexLocation::kHost});
        }

        mPreparedPlans = std::move(preparedPlans);
        return configs;
    }

    void encode(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices, std::size_t numPages,
        cudaStream_t stream) override
    {
        thread_local std::vector<kernels::Nvfp4ColdPageOffloadPageTask> pages;
        pages.clear();
        pages.reserve(numPages);
        for (std::size_t page = 0; page < numPages; ++page)
        {
            pages.push_back({pageIndices[page].src, pageIndices[page].dst});
        }
        kernels::invokeNvfp4ColdPageEncode(pages, mPreparedPlans.at(lifecycleIndex), coldBase, stream);
    }

    void decode(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream) override
    {
        thread_local std::vector<kernels::Nvfp4ColdPageOnboardPageTask> pages;
        pages.clear();
        pages.reserve(numPages);
        for (std::size_t page = 0; page < numPages; ++page)
        {
            pages.push_back({pageIndices[page].dst, pageIndices[page].src});
        }
        kernels::invokeNvfp4ColdPageDecode(pages, mPreparedPlans.at(lifecycleIndex), coldBase, stream);
    }

private:
    std::map<kv::LayerId, Nvfp4ColdPageLayerLayout> mLayerLayouts;
    std::set<kv::LayerId> mLayerIds;
    std::vector<kernels::Nvfp4ColdPagePreparedPlan> mPreparedPlans;
};

} // namespace

std::unique_ptr<kv::IKvCacheColdPageCodec> createNvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerLayout> layerLayouts)
{
    return std::make_unique<NativeColdPageCodec>(std::make_unique<Nvfp4ColdPageCodecBackend>(std::move(layerLayouts)));
}

} // namespace tensorrt_llm::kv_cache_compression
