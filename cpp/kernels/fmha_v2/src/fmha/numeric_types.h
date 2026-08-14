/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <cuda_runtime_api.h>

#pragma once

#if CUDART_VERSION >= 11080
// TODO Better way?
#define FMHA_CUDA_SUPPORTS_FP8 true
#endif
#include <cuda_bf16.h>
#if FMHA_CUDA_SUPPORTS_FP8
#include <cuda_fp8.h>
#endif
namespace fmha
{

using fp16_t = uint16_t;
using fp32_t = float;
using tf32_t = uint32_t;
using bf16_t = nv_bfloat16;
#if FMHA_CUDA_SUPPORTS_FP8
using e4m3_t = __nv_fp8_e4m3;
using e5m2_t = __nv_fp8_e5m2;
#else
using e4m3_t = char;
using e5m2_t = char;
#endif

static constexpr float MAX_E4M3 = 448.f;   // 0x7E 2^8  * 1.75
static constexpr float MAX_E5M2 = 57344.f; // 0x7B 2^15 * 1.75

template <typename T>
__host__ __device__ constexpr inline float Softmax_fp_quant_scale();

template <>
__host__ __device__ constexpr inline float Softmax_fp_quant_scale<e4m3_t>()
{
    // Softmax has max output of 1.0, therefore we choose fp32-to-fp8 quantization scale as the
    // largest power-of-2 below the e4m3 limit:
    // 2^(floor(log2(E4M3_MAX / amax_exp_p))) = 2^(floor(log2(448 / 1))) = 2 ^ 8
    return 256.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
