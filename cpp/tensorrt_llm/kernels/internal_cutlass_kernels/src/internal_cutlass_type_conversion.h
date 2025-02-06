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
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
// We forward declare so we don't have to pull in a million cutlass includes
namespace cutlass
{
// FP4 and FP6 types
struct float_e2m1_t;
struct float_e3m2_t;
} // namespace cutlass

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
#if defined(ENABLE_FP4)
template <>
struct TllmToCutlassTypeAdapter<__nv_fp4_e2m1>
{
    using type = cutlass::float_e2m1_t;
};
#endif

#if defined(ENABLE_FP4)
template <>
struct CutlassToTllmTypeAdapter<cutlass::float_e2m1_t>
{
    using type = __nv_fp4_e2m1;
};
#endif

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
