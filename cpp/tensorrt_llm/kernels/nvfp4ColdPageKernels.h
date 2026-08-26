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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kv_cache_compression/coldPageCallbackAbi.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Active GPU representation encoded into cold Pages.
enum class Nvfp4ColdPageRuntimeType : std::uint8_t
{
    kFloat16 = 0,
    kBfloat16 = 1,
    kFp8E4m3 = 2,
};

using ColdPageIndexPair = ::tensorrt_llm::kv_cache_compression::ColdPageIndexPair;

inline constexpr std::uint32_t kNvfp4ColdPageMaxBuffersPerLaunch = 256;
inline constexpr std::uint32_t kNvfp4ColdPageWideFields = 6;
inline constexpr std::uint32_t kNvfp4ColdPageIntegerFields = 5;
inline constexpr std::uint32_t kNvfp4ColdPageScaleFields = 4;

using Nvfp4ColdPageWideTable
    = std::array<std::array<std::int64_t, kNvfp4ColdPageWideFields>, kNvfp4ColdPageMaxBuffersPerLaunch>;
using Nvfp4ColdPageIntegerTable
    = std::array<std::array<std::int32_t, kNvfp4ColdPageIntegerFields>, kNvfp4ColdPageMaxBuffersPerLaunch>;
using Nvfp4ColdPageScaleTable
    = std::array<std::array<float, kNvfp4ColdPageScaleFields>, kNvfp4ColdPageMaxBuffersPerLaunch>;

//! Compress one whole KVCM Page-index batch; the launcher performs 256-Page chunking internally.
void invokeNvfp4ColdPageEncode(void const* pages, std::size_t numPages, std::int64_t const* wide,
    std::int32_t const* integers, float const* scales, std::uint32_t numBuffers, std::uint32_t maxHalfGroupsPerTile,
    std::size_t coldPageBytes, Nvfp4ColdPageRuntimeType runtimeType, void* coldBase, cudaStream_t stream);

//! Restore one whole KVCM Page-index batch; the launcher performs 256-Page chunking internally.
void invokeNvfp4ColdPageDecode(void const* pages, std::size_t numPages, std::int64_t const* wide,
    std::int32_t const* integers, float const* scales, std::uint32_t numBuffers, std::uint32_t maxHalfGroupsPerTile,
    std::size_t coldPageBytes, Nvfp4ColdPageRuntimeType runtimeType, void const* coldBase, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
