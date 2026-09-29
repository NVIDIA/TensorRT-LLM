/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "tensorrt_llm/common/config.h"

#include <cuda_runtime_api.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::fp8GroupedGemmConfig
{

inline constexpr int kSm90 = 90;
inline constexpr int kSm100 = 100;

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
struct Sm90Config
{
    using ArchTag = cutlass::arch::Sm90;
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

    static constexpr bool kUsesDynamicClusterShape = false;
};
#endif

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
struct Sm100Config
{
    using ArchTag = cutlass::arch::Sm100;
    using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
    using ClusterShape = cute::Shape<int32_t, int32_t, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

    static constexpr bool kUsesDynamicClusterShape = true;

    static dim3 clusterShape()
    {
        return {4, 2, 1};
    }

    static dim3 clusterShapeFallback()
    {
        return {2, 1, 1};
    }
};
#endif

} // namespace kernels::fp8GroupedGemmConfig

TRTLLM_NAMESPACE_END
