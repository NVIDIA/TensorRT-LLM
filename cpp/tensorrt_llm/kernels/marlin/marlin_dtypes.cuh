/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _data_types_cuh
#define _data_types_cuh
#include "marlin.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME
{

// Marlin type traits, specialized per CUDA compute type.
// Matrix fragment layouts documented at:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
//   #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
template <typename T>
struct MarlinType
{
};

template <>
struct MarlinType<nv_bfloat16>
{
    using scalar_t = nv_bfloat16;
    using scalar_t2 = nv_bfloat162;
    using scalar_t4 = nv_bfloat162;
    using scalar_32bit_t = nv_bfloat162;

    using FragA = Vec<nv_bfloat162, 4>;
    using FragB = Vec<nv_bfloat162, 2>;
    using FragC = Vec<float, 4>;
    using FragS = Vec<nv_bfloat162, 1>;
    using FragS0 = Vec<__nv_fp8x2_e4m3, 1>;
    using FragZP = Vec<nv_bfloat162, 4>;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static __device__ float inline num2float(const nv_bfloat16 x)
    {
        return __bfloat162float(x);
    }

    static __device__ nv_bfloat162 inline num2num2(const nv_bfloat16 x)
    {
        return __bfloat162bfloat162(x);
    }

    static __device__ nv_bfloat162 inline nums2num2(const nv_bfloat16 x1, const nv_bfloat16 x2)
    {
        return __halves2bfloat162(x1, x2);
    }

    static __host__ __device__ nv_bfloat16 inline float2num(float const x)
    {
        return __float2bfloat16(x);
    }

    static __host__ __device__ float2 inline num22float2(const nv_bfloat162 x)
    {
        return __bfloat1622float2(x);
    }
#endif
};

} // namespace MARLIN_NAMESPACE_NAME

#endif
