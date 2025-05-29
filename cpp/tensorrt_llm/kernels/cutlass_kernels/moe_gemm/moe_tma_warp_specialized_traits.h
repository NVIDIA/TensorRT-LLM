/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "cutlass/arch/mma_sm90.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "../include/moe_gemm_kernels.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif

namespace tensorrt_llm::kernels::cutlass_kernels
{

// Blackwell arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion
    = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidSM120MOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) // TODO Is there a better choice
    return cutlass::platform::is_same<T, __nv_fp4_e2m1>::value
        && cutlass::platform::is_same<T, WeightType>::value
        && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value
        && Fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
#else
    return false; // CUTLASS_ARCH_MMA_SM100_SUPPORTED is set when Blackwell kernels are enabled
#endif
}

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion
    = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidBlackwellMOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) // TODO Is there a better choice
    return cutlass::platform::is_same<T, WeightType>::value
        && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value
        && Fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
#else
    return false; // CUTLASS_ARCH_MMA_SM100_SUPPORTED is set when Blackwell kernels are enabled
#endif
}

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion
    = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidHopperMOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    return (cutlass::platform::is_same<T, WeightType>::value 
        || (cutlass::platform::is_same<cutlass::uint4b_t, WeightType>::value
        && cutlass::platform::is_same<T, __nv_fp8_e4m3>::value))
#ifdef ENABLE_FP4
        && !cutlass::platform::is_same<T, __nv_fp4_e2m1>::value
#endif
        && cutlass::platform::is_same<EpilogueTag, cutlass_extensions::EpilogueOpDefault>::value;
#else
    return false; // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED is set when Hopper kernels are enabled
#endif
}

template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion
    = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidTmaWarpSpecializedMOESpecialisation()
{
    // Check at least one of the implementations are valid
    return isValidBlackwellMOESpecialisation<T, WeightType, EpilogueTag, Fusion>()
        || isValidHopperMOESpecialisation<T, WeightType, EpilogueTag, Fusion>();
}

// Hopper arch
template <typename T, typename WeightType, typename EpilogueTag = cutlass_extensions::EpilogueOpDefault,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion Fusion
    = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE>
constexpr bool isValidAmpereMOESpecialisation()
{
#ifdef ENABLE_FP4
    return !std::is_same_v<T, __nv_fp4_e2m1>;
#else
    return true;  // Default to true
#endif
}

} // namespace tensorrt_llm::kernels::cutlass_kernels
