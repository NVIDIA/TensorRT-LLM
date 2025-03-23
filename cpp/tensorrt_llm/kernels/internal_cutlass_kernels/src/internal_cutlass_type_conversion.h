/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
struct float_ue4m3_t;
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

#if defined(ENABLE_FP4)
template <>
struct CutlassType<nvinfer1::DataType::kFP4>
{
    using type = cutlass::float_e2m1_t;
};
#endif

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
