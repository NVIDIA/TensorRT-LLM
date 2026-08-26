/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/nvfp4ColdPageKernels.h"

#include <cstdint>
#include <torch/extension.h>

namespace
{

using namespace tensorrt_llm::kernels;

void checkMetadata(at::Tensor const& wide, at::Tensor const& integers, at::Tensor const& scales)
{
    TORCH_CHECK(wide.device().is_cpu() && wide.scalar_type() == at::kLong && wide.is_contiguous()
            && wide.sizes() == at::IntArrayRef({kNvfp4ColdPageMaxBuffersPerLaunch, kNvfp4ColdPageWideFields}),
        "NVFP4 wide metadata must be a contiguous CPU int64 [256, 6] tensor");
    TORCH_CHECK(integers.device().is_cpu() && integers.scalar_type() == at::kInt && integers.is_contiguous()
            && integers.sizes() == at::IntArrayRef({kNvfp4ColdPageMaxBuffersPerLaunch, kNvfp4ColdPageIntegerFields}),
        "NVFP4 integer metadata must be a contiguous CPU int32 [256, 5] tensor");
    TORCH_CHECK(scales.device().is_cpu() && scales.scalar_type() == at::kFloat && scales.is_contiguous()
            && scales.sizes() == at::IntArrayRef({kNvfp4ColdPageMaxBuffersPerLaunch, kNvfp4ColdPageScaleFields}),
        "NVFP4 scale metadata must be a contiguous CPU float32 [256, 4] tensor");
}

template <bool Encode>
void runNvfp4ColdPage(at::Tensor const& wide, at::Tensor const& integers, at::Tensor const& scales,
    std::int64_t numBuffers, std::int64_t maxHalfGroupsPerTile, std::int64_t coldPageBytes, std::int64_t runtimeType,
    std::int64_t coldBase, std::int64_t pagePairs, std::int64_t pageCount, std::int64_t stream)
{
    // This internal op consumes metadata prepared and validated by its Python policy.
    checkMetadata(wide, integers, scales);
    TORCH_CHECK(numBuffers > 0 && numBuffers <= kNvfp4ColdPageMaxBuffersPerLaunch, "Invalid NVFP4 buffer count");
    TORCH_CHECK(maxHalfGroupsPerTile > 0 && maxHalfGroupsPerTile <= 2048, "Invalid NVFP4 tile geometry");
    TORCH_CHECK(coldPageBytes > 0 && pageCount >= 0, "Invalid NVFP4 cold-page size or Page count");
    TORCH_CHECK(runtimeType >= 0 && runtimeType <= 2, "Invalid NVFP4 runtime type");
    TORCH_CHECK(coldBase >= 0 && pagePairs >= 0 && stream >= 0, "NVFP4 pointer arguments must be non-negative");

    auto const* wideData = wide.const_data_ptr<std::int64_t>();
    auto const* integerData = integers.const_data_ptr<std::int32_t>();
    auto const* scaleData = scales.const_data_ptr<float>();
    auto const type = static_cast<Nvfp4ColdPageRuntimeType>(runtimeType);
    auto const* pages = reinterpret_cast<void const*>(static_cast<std::uintptr_t>(pagePairs));
    auto const cudaStream = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(stream));

    if constexpr (Encode)
    {
        invokeNvfp4ColdPageEncode(pages, static_cast<std::size_t>(pageCount), wideData, integerData, scaleData,
            static_cast<std::uint32_t>(numBuffers), static_cast<std::uint32_t>(maxHalfGroupsPerTile),
            static_cast<std::size_t>(coldPageBytes), type,
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(coldBase)), cudaStream);
    }
    else
    {
        invokeNvfp4ColdPageDecode(pages, static_cast<std::size_t>(pageCount), wideData, integerData, scaleData,
            static_cast<std::uint32_t>(numBuffers), static_cast<std::uint32_t>(maxHalfGroupsPerTile),
            static_cast<std::size_t>(coldPageBytes), type,
            reinterpret_cast<void const*>(static_cast<std::uintptr_t>(coldBase)), cudaStream);
    }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "nvfp4_cold_page_encode(Tensor wide, Tensor integers, Tensor scales, int num_buffers, "
        "int max_half_groups_per_tile, int cold_page_bytes, int runtime_type, int cold_base, int page_pairs, "
        "int page_count, int stream) -> ()");
    m.def(
        "nvfp4_cold_page_decode(Tensor wide, Tensor integers, Tensor scales, int num_buffers, "
        "int max_half_groups_per_tile, int cold_page_bytes, int runtime_type, int cold_base, int page_pairs, "
        "int page_count, int stream) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("nvfp4_cold_page_encode", &runNvfp4ColdPage<true>);
    m.impl("nvfp4_cold_page_decode", &runNvfp4ColdPage<false>);
}
